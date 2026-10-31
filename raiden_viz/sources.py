"""Dataset source adapters.

The viewer browses more than one raw dataset layout. Each layout is handled by a
Source subclass exposing the same methods, so the API routes and the frontend are
format-agnostic:

    list_tasks()                    -> [task, ...]
    list_episodes(task)             -> [episode, ...]   (oldest first)
    episode_detail(task, episode)   -> {instruction, status, cameras[], robot, ...}
    episode_facts(task)             -> {episode: {timestamp, status}}  (cheap)
    video_path(task, episode, cam)  -> local Path to a decoded MP4
    episode_stat(task, episode)     -> compact record for analytics
    overview() / stats()            -> dataset-wide aggregates

RaidenSource: <prefix>/<task>/<episode>/{metadata.json, cameras/*.svo2, robot_data.npz}
YamMcapSource: <prefix>/<task>/episode_<uuid>/output.mcap  (one Foxglove-protobuf MCAP)
"""

import hashlib
import json
import os
import random
import re
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import logging

from botocore.exceptions import ClientError

from . import cache, calib_overlay, config, fk, lerobot, previews, robot_data, s3, svo, webbed, yam

# In-flight/finished background stat scans, keyed by a per-source scan id. Held in
# memory (a scan is cheap to restart and its per-episode records are disk-cached).
_SCANS: dict[str, dict] = {}
_SCANS_GUARD = threading.Lock()


def _ee_traces(npz_path, calibration) -> dict | None:
    """End-effector trajectories (via FK) + per-scene-camera projection params, so
    the frontend can draw a future-EE trace on the video, synced to playback.

    EE points are computed in the ``left_arm_base`` frame (the calibration frame):
    the left arm's FK is already there; the right arm's FK is mapped through
    ``bimanual_transform.right_base_to_left_base``. Sampled at ~video rate to keep
    the payload small; the frontend indexes by playback time.
    """
    import numpy as np

    data = np.load(npz_path, allow_pickle=True)
    keys = set(data.files)
    if "timestamps" not in keys:
        return None
    ts = data["timestamps"].astype(np.float64)
    dur = (ts[-1] - ts[0]) / 1e9
    if dur <= 0:
        return None
    n = len(ts)

    # Sample ~30 Hz (video rate) rather than the ~98 Hz robot rate.
    target_hz = 30.0
    stride = max(1, int(round(n / dur / target_hz)))
    idx = np.arange(0, n, stride)
    times = ((ts[idx] - ts[0]) / 1e9).round(3).tolist()

    # Only the LEFT arm is emitted. FK is verified correct (matches raiden's own
    # i2rt.Kinematics grasp_site to ~8mm), but the RIGHT arm — whose FK is in its
    # own base frame and must be mapped via bimanual_transform then projected
    # through the scene-camera extrinsic — does not land on the gripper. Raiden's
    # own visualizer only plots the right EE in a 3D world view, never projected
    # onto the scene image, so there's no reference for that projection. Rather
    # than draw a wrong overlay, the right arm is gated off until validated.
    # (In practice YAM episodes are largely single-active-arm anyway.)
    arms = []
    for side, jkey in (("left", "follower_l_joint_pos"),):
        if jkey not in keys:
            continue
        ee = fk.ee_trajectory(data[jkey][idx])  # (M,3) in the arm's base frame
        arms.append({"side": side, "xyz": np.round(ee, 4).tolist()})

    if not arms:
        return None

    # Per-scene-camera projection: scaled K + extrinsic (R,t). Frontend applies
    # X_cam = R^T (X_base - t); uv = K X_cam. K is scaled to the decoded frame
    # size (left eye = full width here, so image_size already matches).
    cams = {}
    for cname, c in (calibration.get("cameras") or {}).items() if calibration else []:
        ext = c.get("extrinsics")
        intr = c.get("intrinsics", {})
        if not ext or "camera_matrix" not in intr:
            continue
        cams[cname] = {
            "K": intr["camera_matrix"],
            "R": ext["rotation_matrix"],
            "t": ext["translation_vector"],
            "image_size": intr.get("image_size"),
        }

    return {"time": times, "duration_s": round(dur, 3), "arms": arms, "cameras": cams}


def poster_name(clip_name: str) -> str:
    return clip_name[:-4] + ".jpg" if clip_name.endswith(".mp4") else clip_name + ".jpg"


# Blob key of the sampled stats pass. Bump when the pass gains fields (v2: "summary", v3: "embodiments"): old blobs are then
# ignored by _swr/_swr_peek's persisted read and recomputed by the next warm / background build.
STATS_KEY = "stats_v3"


def _median(xs):
    xs = sorted(xs)
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2


# LeRobot info.json robot_type → readable embodiment label (unknown values show as is)
EMBODIMENT_LABEL = {"yam": "YAM", "bi_yam_follower": "YAM", "yam_bimanual": "YAM", "arx": "ARX", "franka": "Franka",
                    "hand": "Hand"}


def embodiments(robot_types) -> list:
    return sorted({EMBODIMENT_LABEL.get(str(r).lower(), str(r)) for r in robot_types if r})


def summarize(episodes: list, total: int, tasks: list) -> dict:
    """Dataset-page tiles from the stats pass + the overview's exact per-task counts. Hours are the sampled mean
    scaled to every episode when the pass sampled (and flagged estimated). annotated_episodes is None when the
    adapter does not know (has_annotations None everywhere)."""
    durs = [e["duration_s"] for e in episodes if e.get("duration_s")]
    hours = None
    estimated = False
    if durs:
        mean_s = sum(durs) / len(durs)
        estimated = len(episodes) < total
        hours = round((mean_s * total if estimated else sum(durs)) / 3600.0, 1)
    ann = [e.get("has_annotations") for e in episodes]
    return {
        "hours": hours, "estimated": estimated,
        "median_duration_s": round(_median(durs), 1) if durs else None,
        "episodes_per_task_median": _median([t.get("episodes") or 0 for t in tasks]) if tasks else None,
        "unique_tasks": len(tasks) if tasks else None,
        "annotated_episodes": sum(1 for a in ann if a) if any(a is not None for a in ann) else None,
        "embodiments": embodiments(e.get("robot_type") for e in episodes),   # [] when the records carry none
    }


class Source:
    def __init__(self, spec: dict):
        self.id = spec["id"]
        self.label = spec["label"]
        self.bucket = spec["bucket"]
        self.prefix = spec["prefix"].strip("/")
        self.spec = spec
        # Memoised overview(): (built_at, value). Sources live in the module-level
        # _SOURCES cache, so this survives for the life of the process.
        self._overview_lock = threading.Lock()
        # per-source JSON memos (overview, stats): {key: {"value", "thread", "skip_persisted"}}, see _swr
        self._swr_state: dict[str, dict] = {}

    # ---- browsing (shared) ----
    def list_tasks(self) -> list[str]:
        return s3.list_dirs(self.prefix, bucket=self.bucket)

    def list_episodes(self, task: str) -> list[str]:
        # Oldest first, matching how the raiden recorder numbers episodes on disk
        # (0000, 0001, ... in capture order) so viewer index == recording index.
        # Both naming schemes sort chronologically as strings: zero-padded counters
        # (0000/0001) and station_<ISO-timestamp> dirs.
        eps = s3.list_dirs(f"{self.prefix}/{task}", bucket=self.bucket)
        return sorted(eps)

    def featured_episodes(self, task: str) -> list[str]:
        """The task's FEATURED episodes: a stable random sample of config.FEATURED_N (the pick is kept in a small
        JSON so it never changes between restarts and can be pre-rendered), in list order. The whole list when
        the task is small enough."""
        eps = self.list_episodes(task)
        if len(eps) <= config.FEATURED_N:
            return eps
        key = f"featured_{self.id}_{re.sub(r'[^A-Za-z0-9_.-]', '_', task)}.json"
        hit = cache.get_json(key, remote=True)
        # keep the stored pick while its episodes exist and the task has not grown/shrunk by more than 20 %
        # (a re-pick throws away pre-rendered clips, so small additions do not trigger one)
        if hit and set(hit.get("episodes") or []) <= set(eps) and abs(len(eps) - hit.get("of", len(eps))) <= 0.2 * hit.get("of", len(eps)):
            return hit["episodes"][:config.FEATURED_N]   # a smaller N keeps the head of the old pick (its clips are rendered)
        pick = random.Random(f"{self.id}/{task}").sample(sorted(eps), config.FEATURED_N)
        pick.sort(key=eps.index)
        cache.put_json(key, {"episodes": pick, "of": len(eps)}, remote=True)
        return pick

    def task_facts(self, task: str) -> dict:
        """Extra per-task overview fields (prompt, tags, ...); sources without them return {}."""
        return {}

    # ---- per-source ----
    def episode_detail(self, task: str, episode: str) -> dict:
        raise NotImplementedError

    def video_cache_name(self, task: str, episode: str, camera: str, eye: str) -> str | None:
        """The cache name video_path() would produce for this clip, computed WITHOUT
        producing it (cheap: metadata + at most one HEAD), or None if the adapter
        cannot know it up front. Lets /video/status answer "ready" on the first poll
        for a clip the derived tier already holds. Raises like video_path for a
        missing camera."""
        return None

    def poster_path(self, task: str, episode: str, camera: str, eye: str) -> Path:
        """First-frame JPEG of a clip that is ALREADY rendered (local or derived
        tier). Never decodes: a poster is a nicety, a decode is minutes. Adapters
        that can render cheaply (LeRobot) override this."""
        name = self.video_cache_name(task, episode, camera, eye)
        if not name:
            raise FileNotFoundError("no posters for this source")
        pname = poster_name(name)

        def _produce(dst: Path):
            clip = cache.path_for(name)
            if not (clip.exists() and clip.stat().st_size > 0) and not cache.exists(name):
                raise FileNotFoundError("clip is not rendered yet")
            lerobot.poster(clip, dst)

        return cache.get_or_create(pname, _produce, fetch=False)

    def episode_facts(self, task: str) -> dict:
        """Cheap per-episode facts for the browse list: {episode: {timestamp, status}}.

        Must stay listing-cheap (no per-episode GETs unless they're small): this is
        called on every task switch to label the sidebar. Sources with nothing cheap
        to offer return {} and the sidebar just shows indices."""
        return {}

    def episode_time(self, task: str, episode: str) -> str | None:
        """Capture wallclock (ISO8601) of ONE episode, as cheaply as the format
        allows — overview() calls it twice per task (first + last) to date the task.
        None where the format carries no timestamp."""
        return None

    def video_path(self, task: str, episode: str, camera: str, eye: str) -> Path:
        raise NotImplementedError

    def episode_stat(self, task: str, episode: str) -> dict | None:
        raise NotImplementedError

    # ---- aggregates (shared, built on the above) ----
    def overview(self) -> dict:
        """Cheap per-source facts: in-process memo → derived-tier blob → inline build (see _swr).

        _build_overview walks every task listing every episode — about 130 paginated
        LIST calls cross-region on the largest source — and this is called on every
        dataset click as well as by every catalog card build. The blob survives a
        container, so a fresh deploy answers the first dataset click instantly.

        Returns a SHALLOW COPY: the /overview route adds a "region" key to what it
        gets back, and that must not accumulate on the cached object.
        """
        return self._swr("overview_v2", self._build_overview_with_previews)

    def _build_overview_with_previews(self) -> dict:
        built = self._build_overview()
        # Small precomputed previews (poster + 5 s clip names) per task, when the
        # preview warm has run for this source; one JSON read, memoized.
        pv = self.previews()
        for row in built.get("tasks", []):
            if row.get("task") in pv:
                row["preview"] = pv[row["task"]]
        return built

    # ---- stale-while-revalidate for per-source JSON (overview, stats) -----------------

    def _swr_blob(self, key: str) -> str:
        return f"{key}_{self.id}.json"

    def _swr(self, key: str, build) -> dict:
        """Serve the memoised value; on a cold container restore the derived-tier blob so
        the first dataset click is instant; a value older than OVERVIEW_TTL_S is served
        as is while ONE background thread rebuilds it; only a source with nothing cached
        anywhere builds inline. Returns a shallow copy."""
        now = time.time()
        with self._overview_lock:
            st = self._swr_state.setdefault(key, {})
            cur = st.get("value")
            if cur is None and not st.get("skip_persisted"):
                saved = cache.get_json(self._swr_blob(key), remote=True)
                if isinstance(saved, dict) and isinstance(saved.get("built_at"), (int, float)):
                    cur = st["value"] = saved
            fresh = cur is not None and now - float(cur["built_at"]) < config.OVERVIEW_TTL_S
            th = st.get("thread")
            running = th is not None and th.is_alive()
            if cur is not None and not fresh and not running:
                th = threading.Thread(target=self._swr_build, args=(key, build), daemon=True,
                                      name=f"swr-{key}-{self.id}")
                st["thread"] = th
                th.start()
        if cur is not None:
            return dict(cur)
        return dict(self._swr_build(key, build))

    def _swr_peek(self, key: str, build) -> dict | None:
        """Like _swr but never blocks a request: the memo / persisted blob or None, and when there is nothing
        (or it is stale) ONE background thread builds + persists it for the next call."""
        now = time.time()
        with self._overview_lock:
            st = self._swr_state.setdefault(key, {})
            cur = st.get("value")
            if cur is None and not st.get("skip_persisted"):
                saved = cache.get_json(self._swr_blob(key), remote=True)
                if isinstance(saved, dict) and isinstance(saved.get("built_at"), (int, float)):
                    cur = st["value"] = saved
            fresh = cur is not None and now - float(cur["built_at"]) < config.OVERVIEW_TTL_S
            th = st.get("thread")
            if not fresh and not (th is not None and th.is_alive()):
                th = threading.Thread(target=self._swr_build, args=(key, build), daemon=True,
                                      name=f"swr-{key}-{self.id}")
                st["thread"] = th
                th.start()
        return dict(cur) if cur is not None else None

    def _swr_build(self, key: str, build) -> dict:
        built = build()
        built["built_at"] = time.time()
        cache.put_json(self._swr_blob(key), built, remote=True)
        with self._overview_lock:
            st = self._swr_state.setdefault(key, {})
            st["value"] = built
            st["skip_persisted"] = False
        return built

    def _swr_invalidate(self, key: str) -> None:
        """Forget the memo AND skip the persisted copy on the next call (the data changed)."""
        with self._overview_lock:
            st = self._swr_state.setdefault(key, {})
            st["value"] = None
            st["skip_persisted"] = True

    # ---- small per-task previews (see previews.py) ----------------------------------

    def previews(self) -> dict:
        """{task: preview entry} from the persisted previews blob; {} when the preview
        warm has never run for this source. Memoized per container."""
        with self._overview_lock:
            pv = getattr(self, "_previews_cached", None)
        if pv is None:
            blob = cache.get_json(previews.blob_name(self.id), remote=True) or {}
            pv = blob.get("tasks", {}) if blob.get("v") == previews.PREVIEWS_V else {}
            with self._overview_lock:
                self._previews_cached = pv
        return pv

    def prune_previews(self, live_tasks) -> int:
        """Drop preview entries for tasks that no longer exist (a dataset removed from
        S3); rewrite the blob only when something was dropped. Returns the count."""
        live = set(live_tasks)
        current = self.previews()
        gone = [t for t in current if t not in live]
        if not gone:
            return 0
        kept = {t: e for t, e in current.items() if t in live}
        cache.put_json(previews.blob_name(self.id),
                       {"v": previews.PREVIEWS_V, "built_at": time.time(), "tasks": kept}, remote=True)
        with self._overview_lock:
            self._previews_cached = kept
        self._swr_invalidate("overview_v2")
        return len(gone)

    def preview_warm(self, tasks=None, progress=None, only_missing: bool = True) -> dict:
        """Build the small preview assets (poster + 5 s clip per camera) for the first
        episode of every task and persist the blob. Sequential: one task at a time,
        one S3 range read per raw camera file — cheap enough to run beside traffic.
        Resumable: the blob is rewritten after every task."""
        import tempfile
        live = self.list_tasks()
        pruned = self.prune_previews(live)          # removed datasets leave no ghosts
        current = dict(self.previews())
        todo = [t for t in (tasks or live) if not (only_missing and t in current)]
        failed = []
        for i, task in enumerate(todo):
            try:
                eps = self.featured_episodes(task)   # a featured episode: its clips are what warm pre-renders anyway
                if not eps:
                    continue
                entry, err = None, "no camera frames"
                for ep in eps:                         # a stray/broken first pick must not cost the task its preview
                    try:
                        with tempfile.TemporaryDirectory(dir=str(config.CACHE_DIR)) as d:
                            assets = previews.build_for(self, task, ep, Path(d))
                        entry = previews.entry_from(ep, assets)
                    except Exception as ex:  # noqa: BLE001
                        log.warning("preview for %s/%s/%s failed: %s", self.id, task, ep, ex)
                        err = f"{type(ex).__name__}: {ex}"
                    if entry is not None:
                        break
                if entry is None:
                    failed.append((task, err))
                else:
                    current[task] = entry
                    cache.put_json(previews.blob_name(self.id),
                                   {"v": previews.PREVIEWS_V, "built_at": time.time(), "tasks": current},
                                   remote=True)
                    with self._overview_lock:
                        self._previews_cached = current
                    self._swr_invalidate("overview_v2")      # overview rows pick the entry up
            except Exception as ex:  # noqa: BLE001
                log.exception("preview warm failed for %s/%s", self.id, task)
                failed.append((task, f"{type(ex).__name__}: {ex}"))
            if progress:
                progress(i + 1, len(todo), failed[-1] if failed and failed[-1][0] == task else None)
        return {"tasks": len(todo), "ok": len(todo) - len(failed), "failed": failed, "pruned": pruned}

    def _build_overview(self) -> dict:
        tasks = self.list_tasks()
        per_task, total, stations = [], 0, set()
        for task in tasks:
            eps = self.list_episodes(task)
            total += len(eps)
            latest = eps[-1] if eps else None  # lists are oldest-first
            for ep in eps:
                m = re.match(r"^([A-Za-z][\w-]*)_\d{4}-\d{2}-\d{2}T", ep)
                if m:
                    stations.add(m.group(1))
            per_task.append({"task": task, "episodes": len(eps), "latest": latest,
                             "featured": min(len(eps), config.FEATURED_N), "_eps": eps, **self.task_facts(task)})
        per_task.sort(key=lambda t: t["episodes"], reverse=True)
        self._add_collection_span(per_task)
        return {
            "source": self.id, "bucket": self.bucket, "prefix": self.prefix,
            "num_tasks": len(tasks), "num_episodes": total,
            "stations": sorted(stations), "tasks": per_task,
        }

    # How many episodes to try inward from each end of a task before giving up on
    # dating it. Episode lists are chronological, so the first/last episode normally
    # answers it in one read — but an episode can be missing its metadata (an aborted
    # recording leaves the directory behind), which would otherwise blank the date.
    SPAN_PROBE_DEPTH = 3

    def _add_collection_span(self, per_task: list[dict]) -> None:
        """Stamp each task row with when it was collected: ``collected_start`` /
        ``collected_end`` (ISO8601, None where the format has no timestamps).

        Episode lists are chronological, so a task's span is its first and last
        episode — a couple of cheap reads per task instead of one per episode. Reads
        run in parallel across tasks; a source without timestamps leaves every value
        None (episode_time returns None) and the frontend hides the sort.
        """
        jobs = []
        for row in per_task:
            eps = row.pop("_eps")
            d = self.SPAN_PROBE_DEPTH
            for key, candidates in (("collected_start", eps[:d]),
                                    ("collected_end", eps[::-1][:d])):
                row[key] = None
                if candidates:
                    jobs.append((row, key, candidates))
        if not jobs:
            return
        with ThreadPoolExecutor(max_workers=32) as pool:
            times = pool.map(lambda j: self._first_time(j[2], j[0]["task"]), jobs)
        for (row, key, _), ts in zip(jobs, times):
            row[key] = ts

    def _first_time(self, episodes: list[str], task: str) -> str | None:
        """The first readable timestamp among ``episodes``, in order (they're ordered
        outermost-first, so this walks inward from one end of the task)."""
        for ep in episodes:
            ts = self._safe_episode_time(task, ep)
            if ts:
                return ts
        return None

    def _safe_episode_time(self, task, episode) -> str | None:
        """episode_time that never breaks the overview: one unreadable episode just
        leaves that end of the span unknown."""
        try:
            return self.episode_time(task, episode)
        except Exception:
            return None

    # Per-source cap on how many episodes a *quick* (non-full) stats pass samples.
    # Reading one stat is cheap, but datasets can have tens of thousands of
    # episodes, so the default charts pass samples (evenly per task) and reports
    # coverage. A filter needs every episode, so it requests a full scan instead.
    STATS_MAX = 1200

    def _stat_pairs(self, full: bool) -> tuple[list[tuple[str, str]], int]:
        """(task, episode) pairs to read for stats, plus the true total count.
        ``full`` reads every episode; otherwise sample down to STATS_MAX."""
        by_task = {t: self.list_episodes(t) for t in self.list_tasks()}
        total = sum(len(v) for v in by_task.values())
        if full or total <= self.STATS_MAX:
            return [(t, e) for t, eps in by_task.items() for e in eps], total
        # Evenly subsample within each task, proportional to its size.
        pairs = []
        for t, eps in by_task.items():
            share = max(1, round(self.STATS_MAX * len(eps) / total))
            step = max(1, len(eps) // share)
            pairs.extend((t, eps[i]) for i in range(0, len(eps), step))
        return pairs, total

    def stats(self, full: bool = False, wait: bool = True) -> dict | None:
        """Stats pass for the charts + the dataset page's summary tiles. The default (sampled) pass is served
        through _swr: memo → derived-tier blob → inline, refreshed in the background when stale. ``wait=False``
        (the HTTP route) never computes on the request: it returns the persisted pass or None and leaves the
        build to a background thread / the warm — "precompute and just fetch" (user, 2026-10-31).
        ``full=True`` (small sources) is always computed now."""
        if full:
            return self._stats_now(True)
        if not wait:
            return self._swr_peek(STATS_KEY, lambda: self._stats_now(False))
        return self._swr(STATS_KEY, lambda: self._stats_now(False))

    def _stats_now(self, full: bool) -> dict:
        pairs, total = self._stat_pairs(full)
        episodes = []
        with ThreadPoolExecutor(max_workers=32) as pool:
            for fut in [pool.submit(self._safe_stat, t, e) for t, e in pairs]:
                rec = fut.result()
                if rec:
                    episodes.append(rec)
        episodes.sort(key=lambda e: e.get("timestamp") or "")
        return {
            "num_episodes": len(episodes),
            "total_episodes": total,
            "scanned": len(episodes),
            "sampled": len(pairs) < total,
            "episodes": episodes,
            "summary": summarize(episodes, total, self.overview().get("tasks", [])),
        }

    # ---- cached per-episode stats + background full scan (for filtering) ----

    def _stat_cache_key(self, task, episode) -> str | None:
        """Cache filename for one episode's stat record, keyed by the cheap head's
        etag so a re-uploaded episode invalidates. None if the object is missing."""
        obj = self._stat_head(task, episode)
        if obj is None:
            return None
        return f"stat_{self.id}_{obj.etag}.json"

    def _stat_head(self, task, episode):
        """The small object whose etag identifies this episode's content (the
        metadata.json for raiden, the MCAP for yam). Overridden per source."""
        raise NotImplementedError

    def _safe_stat(self, task, episode):
        """episode_stat with disk memoization: stat records are deterministic per
        content etag, so a full scan is paid once and every later filter is instant."""
        key = self._stat_cache_key(task, episode)
        if key:
            hit = cache.get_json(key)
            if hit is not None:
                return hit
        try:
            rec = self.episode_stat(task, episode)
        except Exception:
            return None
        if rec and key:
            cache.put_json(key, rec)
        return rec

    def _scan_id(self) -> str:
        return hashlib.sha1(f"{self.id}:{self.bucket}:{self.prefix}".encode()).hexdigest()[:12]

    def _scan_blob(self) -> str:
        return f"scan_{self._scan_id()}.json"

    def scan_start(self, force: bool = False, workers: int | None = None,
                   pause: float = 0.0) -> dict:
        """Begin (or resume) a background full scan of every episode's stats.
        Returns an immediate snapshot; poll scan_snapshot() for progress. Idempotent:
        a scan already running/finished for this source is reused — unless ``force``,
        which discards a FINISHED scan and starts over (a scan that ran before a
        dataset was uploaded is complete and wrong; a running one is left alone).

        A finished scan is persisted to the derived tier and, while younger than
        SCAN_PERSIST_TTL_S, restored instead of rescanned: CACHE_DIR dies with the
        container, so before this every deploy paid the full scan again. ``workers``
        / ``pause`` throttle a real rescan (the boot warmup uses a few threads with a
        pause so clip requests are not starved on a 2-vCPU task)."""
        sid = self._scan_id()
        with _SCANS_GUARD:
            st = _SCANS.get(sid)
            if st and (st["running"] or (st["done"] and not force)):
                return self._snapshot(st)
            if not force:
                saved = self._restore_scan()
                if saved is not None:
                    _SCANS[sid] = saved
                    return self._snapshot(saved)
            pairs, total = self._stat_pairs(full=True)
            st = {"running": True, "done": False, "total": total,
                  "episodes": [], "error": None, "lock": threading.Lock()}
            _SCANS[sid] = st
        t = threading.Thread(target=self._run_scan,
                             args=(sid, pairs, st, workers or config.SCAN_WORKERS, pause),
                             daemon=True)
        t.start()
        return self._snapshot(st)

    def _restore_scan(self) -> dict | None:
        try:
            saved = cache.get_json(self._scan_blob(), remote=True)
        except Exception:
            return None
        if not saved or time.time() - float(saved.get("saved_at", 0)) > config.SCAN_PERSIST_TTL_S:
            return None
        return {"running": False, "done": True, "total": saved.get("total", 0),
                "episodes": list(saved.get("episodes", [])), "error": None,
                "lock": threading.Lock()}

    def _run_scan(self, sid, pairs, st, workers=32, pause=0.0):
        err = None

        def _one(t, e):
            rec = self._safe_stat(t, e)
            if pause:
                time.sleep(pause)      # yield the CPU to requests during a boot rescan
            return rec

        try:
            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                for fut in [pool.submit(_one, t, e) for t, e in pairs]:
                    rec = fut.result()
                    if rec:
                        with st["lock"]:
                            st["episodes"].append(rec)
        except Exception as e:  # never leave a scan stuck "running"
            err = str(e)
        finally:
            # Publish the terminal flags UNDER the same lock that guards episodes,
            # so a poll that observes done=True is guaranteed to also see the final
            # (complete) episode list — never a torn read that drops the last records.
            with st["lock"]:
                st["error"] = err
                st["running"] = False
                st["done"] = True
                eps = list(st["episodes"])
            if err is None:
                cache.put_json(self._scan_blob(),
                               {"saved_at": time.time(), "total": st["total"], "episodes": eps},
                               remote=True)

    def scan_snapshot(self) -> dict | None:
        """Current progress of an in-flight/finished scan, or None if none started."""
        with _SCANS_GUARD:
            st = _SCANS.get(self._scan_id())
        return self._snapshot(st) if st else None

    def _snapshot(self, st) -> dict:
        # Read the flags and the episode list together under the lock so they can't
        # disagree (see _run_scan's terminal publish).
        with st["lock"]:
            eps = list(st["episodes"])
            running, done, error = st["running"], st["done"], st["error"]
        return {
            "running": running, "done": done,
            "total_episodes": st["total"], "scanned": len(eps),
            "error": error, "episodes": eps,
        }


class RaidenSource(Source):
    def _ep_prefix(self, task, episode):
        return f"{self.prefix}/{task}/{episode}"

    def episode_time(self, task: str, episode: str) -> str | None:
        # The recorder's own capture stamp, from the ~1 KB metadata.json (via the
        # etag-keyed stat cache, so the overview's reads are paid once).
        rec = self._safe_stat(task, episode)
        return rec.get("timestamp") if rec else None

    def episode_facts(self, task: str) -> dict:
        # metadata.json is ~1 KB, so fetching one per episode in parallel is cheap
        # enough to label the whole browse list (largest raiden task is ~270 eps).
        # Reuses the etag-keyed stat cache, so repeat visits are served from disk.
        eps = self.list_episodes(task)
        out = {}
        with ThreadPoolExecutor(max_workers=32) as pool:
            recs = pool.map(lambda e: (e, self._safe_stat(task, e)), eps)
        for ep, rec in recs:
            if rec:
                out[ep] = {"timestamp": rec.get("timestamp"), "status": rec.get("status")}
        return out

    def episode_detail(self, task, episode):
        prefix = self._ep_prefix(task, episode)
        # stray recordings (aborted dagger/test runs) have cameras but no metadata.json: show them without it
        metadata = (s3.get_json(f"{prefix}/metadata.json", bucket=self.bucket)
                    if s3.try_head(f"{prefix}/metadata.json", bucket=self.bucket) else {})

        calibration = None
        if s3.try_head(f"{prefix}/calibration_results.json", bucket=self.bucket):
            calibration = s3.get_json(f"{prefix}/calibration_results.json", bucket=self.bucket)

        cameras = []
        for obj in s3.list_files(f"{prefix}/cameras", bucket=self.bucket):
            if not obj.key.endswith(".svo2"):
                continue
            name = obj.key.rsplit("/", 1)[-1][: -len(".svo2")]
            cameras.append({
                "name": name,
                "size_mb": round(obj.size / 1024 / 1024, 1),
                "has_video": obj.size > 100_000,  # stub header files are ~1.5 KB
                "eyes": ["left", "right"],  # side-by-side stereo
            })
        cameras.sort(key=lambda c: c["name"])

        robot = None
        ee_traces = None
        robot_obj = s3.try_head(f"{prefix}/robot_data.npz", bucket=self.bucket)
        if robot_obj:
            npz = cache.get_or_create(
                f"{robot_obj.etag}_robot.npz",
                lambda dst: s3.download(robot_obj.key, dst, bucket=self.bucket),
            )
            robot = robot_data.summarize(npz)
            ee_traces = _ee_traces(npz, calibration)

        return {
            "source": self.id, "task": task, "episode": episode,
            "instruction": metadata.get("task_instruction") or metadata.get("task_name"),
            "status": metadata.get("status"),
            "metadata": metadata, "calibration": calibration,
            "cameras": cameras, "robot": robot, "annotations": [],
            "ee_traces": ee_traces,
        }

    def video_cache_name(self, task, episode, camera, eye):
        key = f"{self._ep_prefix(task, episode)}/cameras/{camera}.svo2"
        obj = s3.try_head(key, bucket=self.bucket)
        if obj is None:
            raise FileNotFoundError(f"camera not found: {camera}")
        return f"{obj.etag}_{camera}_{eye}.mp4"

    def video_path(self, task, episode, camera, eye):
        key = f"{self._ep_prefix(task, episode)}/cameras/{camera}.svo2"
        obj = s3.try_head(key, bucket=self.bucket)
        if obj is None:
            raise FileNotFoundError(f"camera not found: {camera}")
        if obj.size < 100_000:
            raise ValueError(f"camera '{camera}' has no recorded video (stub file)")
        # LOCAL ONLY. This is the transcode's input, not its output: a
        # byte-for-byte copy of an object that is already durable in the datasets
        # bucket. Publishing it duplicated source data into the derived tier (~2.8 GB
        # there, about a third of it) to buy back a re-download in the one case where
        # the decoded MP4 expired but its source had not. The MP4 below IS published:
        # that one costs minutes of ffmpeg to reproduce. Same split the yam adapter
        # already makes with its MCAP, which it keeps as a temp file.
        svo_local = cache.get_or_create(
            f"{obj.etag}_{camera}.svo2",
            lambda dst: s3.download(key, dst, bucket=self.bucket),
            remote=False,
        )
        return cache.get_or_create(
            f"{obj.etag}_{camera}_{eye}.mp4",
            lambda dst: svo.decode_to_mp4(svo_local, dst, eye=eye),
        )

    def calib_overlay_path(self, task, episode, camera):
        """Render a calibration-check overlay (arm-base axis triads projected onto
        a still frame of `camera`). Only works for scene-type cameras that carry
        extrinsics in the base frame. Returns a cached PNG path, or raises."""
        prefix = self._ep_prefix(task, episode)
        calib_key = f"{prefix}/calibration_results.json"
        if s3.try_head(calib_key, bucket=self.bucket) is None:
            raise FileNotFoundError("no calibration for this episode")
        calib = s3.get_json(calib_key, bucket=self.bucket)
        cam_calib = (calib.get("cameras") or {}).get(camera)
        if not cam_calib or not cam_calib.get("extrinsics"):
            raise ValueError(f"camera '{camera}' has no scene extrinsics to visualize")

        mp4 = self.video_path(task, episode, camera, "left")
        obj = s3.head(f"{prefix}/cameras/{camera}.svo2", bucket=self.bucket)
        bt = (calib.get("bimanual_transform") or {}).get("right_base_to_left_base")

        def _produce(dst: Path):
            frame = dst.with_suffix(".frame.png")
            calib_overlay.extract_frame(mp4, frame, frame_index=0)
            ok = calib_overlay.draw_overlay(frame, cam_calib, bt, dst)
            frame.unlink(missing_ok=True)
            if not ok:
                raise ValueError("could not render calibration overlay")

        return cache.get_or_create(f"{obj.etag}_{camera}_calib.png", _produce)

    def _stat_head(self, task, episode):
        return s3.try_head(f"{self._ep_prefix(task, episode)}/metadata.json", bucket=self.bucket)

    def episode_stat(self, task, episode):
        md = s3.get_json(f"{self._ep_prefix(task, episode)}/metadata.json", bucket=self.bucket)
        return {
            "task": task, "episode": episode,
            "duration_s": md.get("duration_s"), "robot_frames": md.get("robot_frames"),
            "robot_hz": md.get("robot_hz"), "num_cameras": len(md.get("cameras", [])),
            "status": md.get("status"), "station": md.get("station_name"),
            "teacher": md.get("teacher_name"), "control": md.get("control"),
            "timestamp": md.get("timestamp"),
            "has_annotations": None,  # raiden has no subtask annotations
        }


class RolloutSource(RaidenSource):
    """yam-eval policy rollouts: the raiden recording layout (metadata.json + cameras/*.svo2 + robot_data.npz)
    nested under `<run>_<step>/<task>/rollout_NNNN` (Vibe, depth 1) or `<formal_id>/<run>_<step>/<task>/rollout_NNNN`
    (ranked evals, depth 2). One task spans every model / eval, so the episode id carries the lead dirs:
    `<lead>::<lead>::rollout_NNNN` ("::" never appears in run or eval ids and survives the URL path)."""
    SEP = "::"

    def __init__(self, spec):
        super().__init__(spec)
        self.depth = int(spec.get("depth", 1))

    def _leads(self) -> list[str]:
        leads = [""]
        for _ in range(self.depth):
            leads = [f"{lead}/{d}" if lead else d
                     for lead in leads for d in s3.list_dirs(f"{self.prefix}/{lead}".rstrip("/"), bucket=self.bucket)]
        return leads

    def list_tasks(self) -> list[str]:
        return sorted({t for lead in self._leads() for t in s3.list_dirs(f"{self.prefix}/{lead}", bucket=self.bucket)})

    def list_episodes(self, task: str) -> list[str]:
        return sorted(self.SEP.join(lead.split("/") + [r])
                      for lead in self._leads() for r in s3.list_dirs(f"{self.prefix}/{lead}/{task}", bucket=self.bucket))

    def _ep_prefix(self, task, episode):
        *lead, rollout = episode.split(self.SEP)
        return "/".join([self.prefix, *lead, task, rollout])


class YamMcapSource(Source):
    """One MCAP per episode. Download the big MCAP once, extract all small
    artifacts (per-camera mp4 + robot/instruction json), cache them, drop the MCAP.

    The per-episode MCAP basename varies by dataset (``output.mcap`` for the
    russet/yam_raw layout, ``episode.mcap`` for ABC-130k), set via spec['mcap_name']."""

    def _mcap_key(self, task, episode):
        name = self.spec.get("mcap_name", "output.mcap")
        return f"{self.prefix}/{task}/{episode}/{name}"

    # Skip the browse-list timestamp listing for tasks bigger than this: the labels
    # are a browsing nicety, not worth paging tens of thousands of keys (ABC-130k's
    # largest task has ~11k episodes) on every task switch.
    FACTS_MAX_EPISODES = 2000

    def episode_time(self, task: str, episode: str) -> str | None:
        # A HEAD on the MCAP: no download, and its LastModified is the closest
        # available wallclock stamp (uuid episode ids carry no time).
        obj = s3.try_head(self._mcap_key(task, episode), bucket=self.bucket)
        return obj.last_modified if obj else None

    def episode_facts(self, task: str) -> dict:
        # uuid-named episodes carry no timestamp, and the MCAP is far too big to
        # open here — but a single recursive listing yields every episode's MCAP
        # LastModified, the closest available wallclock stamp. No status: the MCAP
        # format has no success/failure field.
        # Size-check on the cheaper delimiter listing first, so an oversized task
        # bails without paying for the recursive walk.
        if len(self.list_episodes(task)) > self.FACTS_MAX_EPISODES:
            return {}
        name = self.spec.get("mcap_name", "output.mcap")
        out = {}
        for obj in s3.list_keys(f"{self.prefix}/{task}", bucket=self.bucket, suffix=name):
            ep = obj.key[len(f"{self.prefix}/{task}/"):].rsplit("/", 1)[0]
            out[ep] = {"timestamp": obj.last_modified, "status": None}
        return out

    def _mine(self, obj, task=None, episode=None) -> dict:
        """Download the raw MCAP to a TEMP file, extract everything (all camera
        MP4s + robot/instruction JSON), cache those small artifacts keyed by ETag,
        then delete the big MCAP. Idempotent: skips work already cached.

        The raw MCAP is 200-880 MB and must never linger in the cache, so it's a
        temp file (not a cache.get_or_create artifact) that's removed in finally.

        Subtask annotations may live in a sibling ``annotation.mcap`` (ABC-130k,
        annotated episodes only); when present it's merged into ``annotations``."""
        meta_json = cache.path_for(f"yam_{obj.etag}_meta.json")
        if meta_json.exists():
            ex = json.loads(meta_json.read_text())
            # Ensure the per-camera MP4s exist too (a partial prior run may have
            # written meta but not videos).
            if all(cache.exists(f"yam_{obj.etag}_{c}.mp4") for c in ex["cameras"]):
                return ex

        # Make room for the big MCAP + its extracted MP4s before downloading, so
        # a near-full cache can't wedge mid-download.
        cache.evict(headroom_gb=min(2.0, obj.size / 1024**3 * 1.5))
        tmp = cache.path_for(f"yam_{obj.etag}.mcap.tmp{os.getpid()}")
        try:
            s3.download(obj.key, tmp, bucket=self.bucket)
            probe = yam.probe(tmp)
            # Extract every camera to a cached MP4 in this single pass.
            for cam in probe["cameras"]:
                if not cache.exists(f"yam_{obj.etag}_{cam}.mp4"):
                    cache.get_or_create(
                        f"yam_{obj.etag}_{cam}.mp4",
                        lambda dst, _c=cam: yam.extract_camera_mp4(tmp, _c, dst),
                    )
            mr = yam.extract_meta_and_robot(tmp)
            # If a sibling annotation.mcap exists and the episode.mcap had no inline
            # subtask labels, pull them from there (relative to the episode start).
            if not mr.get("annotations") and task is not None:
                sib = self._annotation_key(task, episode)
                sib_obj = s3.try_head(sib, bucket=self.bucket)
                if sib_obj is not None:
                    atmp = cache.path_for(f"yam_{obj.etag}_ann.mcap.tmp{os.getpid()}")
                    try:
                        s3.download(sib, atmp, bucket=self.bucket)
                        mr["annotations"] = yam.read_annotation_mcap(atmp, mr.get("start_abs", 0.0))
                    finally:
                        atmp.unlink(missing_ok=True)

            ex = {"etag": obj.etag, "cameras": probe["cameras"], **mr}
            meta_json.write_text(json.dumps(ex))
            return ex
        finally:
            tmp.unlink(missing_ok=True)

    def _annotation_key(self, task, episode):
        """Sibling annotation.mcap path next to the episode's MCAP."""
        ep_key = self._mcap_key(task, episode)
        return ep_key.rsplit("/", 1)[0] + "/annotation.mcap"

    def _head(self, task, episode):
        obj = s3.try_head(self._mcap_key(task, episode), bucket=self.bucket)
        if obj is None:
            raise FileNotFoundError(f"no output.mcap for {task}/{episode}")
        return obj

    def _sidecar(self, episode) -> dict:
        """Fetch + split the per-episode metadata sidecar, if this source has one.
        Keyed by the episode uuid (``episode_<uuid>`` -> ``<prefix>/<uuid>.json``).
        Small (~3 KB) and read live per view, so it stays out of the MCAP cache."""
        prefix = self.spec.get("metadata_prefix")
        if not prefix:
            return {}
        uuid = episode[len("episode_"):] if episode.startswith("episode_") else episode
        raw = s3.try_get_json(f"{prefix}/{uuid}.json", bucket=self.bucket)
        return yam.parse_sidecar(raw) if raw else {}

    def episode_detail(self, task, episode):
        ex = self._mine(self._head(task, episode), task, episode)
        cameras = [{"name": c, "has_video": True, "eyes": ["left"]} for c in ex["cameras"]]
        sidecar = self._sidecar(episode)
        sm = sidecar.get("sidecar_meta") or {}
        metadata = {"task_instruction": ex.get("instruction"),
                    "num_annotations": len(ex.get("annotations") or [])}
        if sm.get("duration_s") is not None:
            metadata["duration_s"] = round(sm["duration_s"], 2)
        if sm.get("env_loop_frequency") is not None:
            metadata["control_hz"] = sm["env_loop_frequency"]
        if sm.get("arm_type") is not None:
            metadata["arm_type"] = sm["arm_type"]
        return {
            "source": self.id, "task": task, "episode": episode,
            "instruction": ex.get("instruction"), "status": None,
            "metadata": metadata,
            "calibration": sidecar.get("calibration"), "cameras": cameras,
            "robot": ex.get("robot"), "annotations": ex.get("annotations") or [],
        }

    def video_cache_name(self, task, episode, camera, eye):
        return f"yam_{self._head(task, episode).etag}_{camera}.mp4"

    def video_path(self, task, episode, camera, eye):
        obj = self._head(task, episode)
        mp4 = cache.path_for(f"yam_{obj.etag}_{camera}.mp4")
        # cache.exists() consults the derived S3 tier and pulls the clip down if it
        # is there. A bare local probe would report a miss for something already
        # decoded, and _mine answers a miss by re-downloading the whole 200-880 MB
        # MCAP — the single most expensive thing this app can do.
        if not cache.exists(f"yam_{obj.etag}_{camera}.mp4"):
            # Mining extracts all cameras at once (single MCAP download), then
            # drops the MCAP — so this only downloads on a true cold miss.
            self._mine(obj, task, episode)
        if not cache.exists(f"yam_{obj.etag}_{camera}.mp4"):
            raise FileNotFoundError(f"camera {camera!r} not found in {task}/{episode}")
        return mp4

    def _stat_head(self, task, episode):
        return s3.try_head(self._mcap_key(task, episode), bucket=self.bucket)

    def episode_stat(self, task, episode):
        # Analytics/filtering must not trigger 200-880 MB downloads per episode. The
        # summary section at the END of the MCAP holds duration + per-channel message
        # counts, so a small tail range-read yields every cheap facet (duration,
        # cameras, annotations, robot frames). The MCAP last-modified time is the
        # closest available wallclock stamp (uuid ids carry no timestamp).
        obj = s3.try_head(self._mcap_key(task, episode), bucket=self.bucket)
        if obj is None:
            return None
        base = {
            "task": task, "episode": episode, "num_cameras": None,
            "duration_s": None, "robot_frames": None, "robot_hz": None,
            "status": None, "station": None, "teacher": None, "control": None,
            "timestamp": obj.last_modified,
            "has_annotations": None, "n_annotations": None,
        }
        # Read a tail window; widen once if the summary didn't fit. stats_from_tail
        # returns {} only when the window didn't reach the summary, so break (and
        # merge every parsed facet) as soon as it returns anything — not solely on
        # duration, which would discard cameras/annotations from a summary that
        # happens to lack a Statistics record.
        for window in (4_000_000, 16_000_000):
            start = max(0, obj.size - window)
            tail = s3.get_range(obj.key, start, obj.size - 1, bucket=self.bucket)
            st = yam.stats_from_tail(tail, obj.size)
            if st:
                base.update(st)
                break
        return base


class LeRobotSource(Source):
    """LeRobot v3.0 datasets: ``<prefix>/<task>[/<subdir>]/{meta,data,videos}``.

    Each task folder is a self-contained LeRobot dataset. An optional ``subdir`` in the
    spec puts the dataset one level below the task folder, for sources whose task
    folders also carry other artefacts (yam_sim: ``sim_datasets/<name>/`` holds the
    source h5 + dataset card, and the LeRobot copy under ``lerobot/``). Task folders
    without ``<subdir>/meta/info.json`` are not tasks and are skipped. Unlike the raiden/yam
    layouts (one folder or one MCAP per episode), many episodes may be PACKED into
    shared parquet/mp4 files; ``meta/episodes`` maps each episode_index to its data
    file, its per-camera video file, and the ``[from_ts, to_ts]`` slice within them.

    Timeseries (observation.state/action, subtask labels) live in the packed data
    parquet. Video is transcoded to browser-safe H.264 on demand, trimmed to the
    episode's window — unless the source file is ALREADY H.264 4:2:0 and holds exactly
    this one episode, in which case it is stream-copied (no decode; see video_path).
    Per-task metadata (info/tasks/episodes) is small and read once, then memoized."""

    def __init__(self, spec: dict):
        super().__init__(spec)
        self.subdir = (spec.get("subdir") or "").strip("/")
        self._meta_cache: dict[str, dict] = {}
        self._clips_cache: dict[str, dict] = {}      # task -> {"ep|cam": cache name} (warmed clips)
        self._source_index: dict | None = None
        self._index_lock = threading.Lock()
        self._meta_lock = threading.Lock()

    # LeRobot indexes episodes by integer; expose them as zero-padded names so the
    # existing string-keyed API routes/frontend work unchanged.
    def _ep_name(self, idx: int) -> str:
        return f"episode_{int(idx):06d}"

    def _ep_index(self, name: str) -> int:
        return int(name.rsplit("_", 1)[-1])

    def _task_root(self, task: str) -> str:
        root = f"{self.prefix}/{task}"
        return f"{root}/{self.subdir}" if self.subdir else root

    def _list_tasks_raw(self) -> list[str]:
        names = super().list_tasks()
        if not self.subdir:
            return names
        # With a subdir the prefix also holds folders that are not datasets (cards
        # only); one HEAD per folder keeps those out of the task list.
        return [t for t in names
                if s3.try_head(f"{self._task_root(t)}/meta/info.json", bucket=self.bucket) is not None]

    def list_tasks(self) -> list[str]:
        # From the source index (one S3 GET per container) rather than a listing
        # plus one HEAD per folder on every call.
        return [t["task"] for t in self.source_index()["tasks"]]

    # ---- source index: everything the overview / stats / browse pages need ------

    SOURCE_INDEX_V = 4   # v3: per-task preview (episode, camera, artifact names) + facts; v4: robot_type on the stat records

    def _source_blob(self) -> str:
        return f"lerobot_source_{self.id}_v{self.SOURCE_INDEX_V}.json"

    def source_index(self) -> dict:
        """{built_at, tasks: [{task, ikey, episodes, latest, cameras, fps}], stats: [...]}.
        Memoized per container; read from the derived tier; built (once) and
        published when absent. Refreshed in the background once older than
        SOURCE_INDEX_TTL_S, so a dataset uploaded without an explicit rebuild still
        appears — the stale index is served meanwhile."""
        with self._index_lock:
            idx = self._source_index
            if idx is None:
                idx = cache.get_json(self._source_blob(), remote=True)
                if idx and idx.get("v") != self.SOURCE_INDEX_V:
                    idx = None
                self._source_index = idx
        if idx is None:
            return self.rebuild_source_index()
        if time.time() - float(idx.get("built_at", 0)) > config.SOURCE_INDEX_TTL_S \
                and not idx.get("_refreshing"):
            idx["_refreshing"] = True
            threading.Thread(target=self._safe_rebuild_index, daemon=True,
                             name=f"index-{self.id}").start()
        return idx

    def _safe_rebuild_index(self):
        try:
            self.rebuild_source_index()
        except Exception:
            log.exception("source index rebuild failed for %s", self.id)

    def rebuild_source_index(self) -> dict:
        """List the datasets, load (or restore) each task's meta, and publish one
        JSON with per-task facts plus every episode's stat record."""
        tasks, stats = [], []
        for t in self._list_tasks_raw():
            try:
                meta = self._meta(t)
                names = self.list_episodes(t)
            except Exception:
                log.exception("source index: skipping %s", t)
                continue
            info = meta["info"]
            cams = list(info.get("cameras", []))
            recs = [r for r in (self.episode_stat(t, n) for n in names) if r]
            stats.extend(recs)
            durs = sorted(r["duration_s"] for r in recs if r.get("duration_s") is not None)
            status_counts: dict[str, int] = {}
            for r in recs:
                status_counts[str(r.get("status") or "unknown")] = \
                    status_counts.get(str(r.get("status") or "unknown"), 0) + 1
            # Deterministic preview: the first episode, the scene camera if there is
            # one, else the first camera. Recorded here so cards need no compute.
            preview = None
            if names and cams:
                cam = next((c for c in cams if "scene" in c), cams[0])
                preview = {"episode": names[0], "camera": cam, "cameras": cams}
                # Artifact names straight from the clip manifest, so a card can 302 to
                # the poster/clip with no meta load at all (GET /api/artifact/{name}).
                try:
                    manifest = self._clip_manifest(t)
                except Exception:
                    manifest = {}
                clip = manifest.get(f"{names[0]}|{cam}")
                if clip:
                    preview["clip_name"] = clip
                    preview["poster_name"] = poster_name(clip)
                    preview["poster_names"] = {c: poster_name(manifest[f"{names[0]}|{c}"])
                                               for c in cams if f"{names[0]}|{c}" in manifest}
            tasks.append({"task": t, "ikey": meta.get("ikey"), "episodes": len(names),
                          "latest": names[-1] if names else None,
                          "cameras": cams, "fps": info.get("fps"),
                          "preview": preview,
                          "facts": {"episodes": len(names), "cameras": cams, "fps": info.get("fps"),
                                    "duration_median_s": durs[len(durs) // 2] if durs else None,
                                    "status_counts": status_counts}})
        idx = {"v": self.SOURCE_INDEX_V, "built_at": time.time(), "source": self.id,
               "tasks": tasks, "stats": stats}
        cache.put_json(self._source_blob(), idx, remote=True)
        with self._index_lock:
            self._source_index = idx
        try:
            self.prune_previews([t["task"] for t in tasks])
        except Exception:
            log.exception("preview prune failed for %s", self.id)
        return idx

    def _build_overview(self) -> dict:
        idx = self.source_index()
        per_task = [{"task": t["task"], "episodes": t["episodes"], "latest": t["latest"],
                     "featured": min(t["episodes"], config.FEATURED_N),
                     "collected_start": None, "collected_end": None,
                     "preview": t.get("preview"), "facts": t.get("facts")}
                    for t in idx["tasks"]]
        per_task.sort(key=lambda t: t["episodes"], reverse=True)
        return {
            "source": self.id, "bucket": self.bucket, "prefix": self.prefix,
            "num_tasks": len(per_task), "num_episodes": sum(t["episodes"] for t in per_task),
            "stations": [], "tasks": per_task,
        }

    def stats(self, full: bool = False, wait: bool = True) -> dict:
        # Every episode's record is in the source index already: no per-episode work,
        # no sampling, nothing to wait for.
        eps = list(self.source_index()["stats"])   # episode_stat records, robot_type included since index v4
        total = len(eps)
        if not full and total > self.STATS_MAX:          # keep the chart payload bounded
            step = max(1, total // self.STATS_MAX)
            eps = eps[::step]
        return {"num_episodes": len(eps), "total_episodes": total, "scanned": len(eps),
                "sampled": len(eps) < total, "episodes": eps,
                "summary": summarize(eps, total, self.overview().get("tasks", []))}

    # ---- warmed-clip manifest: (episode, camera) -> cache name, no HEADs ----------

    def _clip_manifest(self, task: str) -> dict:
        m = self._clips_cache.get(task)
        if m is None:
            ikey = self._meta(task).get("ikey")
            m = (cache.get_json(self._clips_blob(ikey), remote=True) or {}).get("clips", {}) if ikey else {}
            cache.note_remote_many(m.values())     # every listed clip IS in the tier
            self._clips_cache[task] = m
        return m

    INDEX_V = 1

    def _index_key(self, task: str) -> str:
        """Content key of a task's meta: ONE listing of <root>/meta/ (4-5 objects),
        hashed with their etags — so re-uploaded meta (e.g. a new column) invalidates
        every artifact derived from it, and nothing has to be parsed to know that."""
        root = self._task_root(task)
        objs = sorted((o.key, o.etag) for o in s3.list_keys(f"{root}/meta/", bucket=self.bucket))
        if not objs:
            raise FileNotFoundError(f"no LeRobot meta under {root}")
        return hashlib.sha1(("|".join(f"{k}:{e}" for k, e in objs)).encode()).hexdigest()[:16]

    def _meta_blob(self, ikey: str) -> str:
        return f"lerobot_meta_{ikey}_v{self.INDEX_V}.json"

    def _detail_blob(self, ikey: str, idx: int) -> str:
        return f"lerobot_detail_{ikey}_{idx}_v{self.INDEX_V}.json"

    def _clips_blob(self, ikey: str) -> str:
        return f"lerobot_clips_{ikey}_v{self.INDEX_V}.json"

    def _load_meta(self, task: str) -> dict:
        """Task meta = {info, tasks, episodes, ikey}. Read as ONE JSON blob from the
        derived tier when it exists (bytes from S3, no parquet parsing); otherwise
        parsed from meta/*.parquet once and published for every later container."""
        root = self._task_root(task)
        ikey = self._index_key(task)
        blob = cache.get_json(self._meta_blob(ikey), remote=True)
        if blob and blob.get("v") == self.INDEX_V:
            return {"info": blob["info"],
                    "tasks": {int(k) if str(k).lstrip("-").isdigit() else k: v
                              for k, v in blob["tasks"].items()},
                    "episodes": {int(k): v for k, v in blob["episodes"].items()},
                    "ikey": ikey}
        info = lerobot.parse_info(s3.get_json(f"{root}/meta/info.json", bucket=self.bucket))
        tasks_tbl = lerobot.read_table(s3.get_bytes(f"{root}/meta/tasks.parquet", bucket=self.bucket))
        task_map = lerobot.parse_tasks(tasks_tbl)
        # meta/episodes may itself be chunked across several parquet files.
        episodes: dict[int, dict] = {}
        for obj in sorted(s3.list_keys(f"{root}/meta/episodes", bucket=self.bucket, suffix=".parquet"),
                          key=lambda o: o.key):
            tbl = lerobot.read_table(s3.get_bytes(obj.key, bucket=self.bucket))
            episodes.update(lerobot.parse_episodes(tbl, info["video_keys"]))
        cache.put_json(self._meta_blob(ikey),
                       {"v": self.INDEX_V, "info": info,
                        "tasks": {str(k): v for k, v in task_map.items()},
                        "episodes": {str(k): v for k, v in episodes.items()}},
                       remote=True)
        return {"info": info, "tasks": task_map, "episodes": episodes, "ikey": ikey}

    def _meta(self, task: str) -> dict:
        # Lock across the load so 32 concurrent scan workers don't all fetch the
        # same task's meta; datasets are static, so a plain per-task memo is enough.
        with self._meta_lock:
            m = self._meta_cache.get(task)
            if m is None:
                m = self._load_meta(task)
                self._meta_cache[task] = m
            return m

    def list_episodes(self, task: str) -> list[str]:
        return [self._ep_name(i) for i in sorted(self._meta(task)["episodes"])]

    def _row(self, task: str, episode: str) -> tuple[dict, dict]:
        meta = self._meta(task)
        row = meta["episodes"].get(self._ep_index(episode))
        if row is None:
            raise FileNotFoundError(f"no such episode: {task}/{episode}")
        return meta, row

    def _data_table(self, task: str, meta: dict, row: dict):
        """Load the (possibly multi-episode) data parquet for an episode and filter
        it down to just that episode's rows."""
        key = meta["info"]["data_path"].format(
            chunk_index=row["data_chunk"], file_index=row["data_file"])
        try:
            raw = s3.get_bytes(f"{self._task_root(task)}/{key}", bucket=self.bucket)
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") not in ("NoSuchKey", "404"):
                raise
            # Some exports (WorldEngine) point episodes at data files that were never uploaded while
            # their videos exist: show the episode without traces instead of failing it.
            log.warning("%s/%s: data parquet missing (%s); detail without traces", task, row["episode_index"], key)
            return lerobot.empty_table()
        tbl = lerobot.read_table(raw)
        return lerobot.filter_episode(tbl, row["episode_index"])

    def episode_detail(self, task: str, episode: str) -> dict:
        """Per-episode detail (robot traces, annotations, instruction) as ONE small
        JSON from the derived tier; computed from the packed data parquet only the
        first time (or ahead of time by warm_task) and published."""
        meta, row = self._row(task, episode)
        blob = self._detail_blob(meta.get("ikey", "nokey"), row["episode_index"])
        hit = cache.get_json(blob, remote=True) if meta.get("ikey") else None
        if hit is not None:
            return hit
        tbl = self._data_table(task, meta, row)
        detail = self._build_detail(task, episode, meta, row, tbl)
        if meta.get("ikey"):
            cache.put_json(blob, detail, remote=True)
        return detail

    def _build_detail(self, task: str, episode: str, meta: dict, row: dict, tbl) -> dict:
        info = meta["info"]
        robot = lerobot.build_robot(tbl, info)
        annotations = lerobot.subtasks_to_annotations(tbl)
        instruction = lerobot.instruction_for(tbl, meta["tasks"], row)
        cameras = [{"name": c, "has_video": True, "eyes": ["left"]} for c in info["cameras"]]
        metadata = {"num_annotations": len(annotations)}
        if info.get("robot_type"):
            metadata["arm_type"] = info["robot_type"]
        if info.get("fps"):
            metadata["control_hz"] = info["fps"]
        return {
            "source": self.id, "task": task, "episode": episode,
            "instruction": instruction, "status": row.get("status"),
            "metadata": metadata, "calibration": None,
            "cameras": cameras, "robot": robot,
            "annotations": annotations,
        }

    def _clip_source(self, task: str, episode: str, camera: str):
        """(source key, S3Object, from_ts, to_ts, cache name) for one clip; one HEAD."""
        meta, row = self._row(task, episode)
        info = meta["info"]
        vid = row.get("videos", {}).get(camera)
        full_key = info["video_keys"].get(camera)
        if vid is None or full_key is None:
            raise FileNotFoundError(f"camera {camera!r} not found in {task}/{episode}")
        vkey = info["video_path"].format(
            video_key=full_key, chunk_index=vid["chunk"], file_index=vid["file"])
        s3key = f"{self._task_root(task)}/{vkey}"
        obj = s3.try_head(s3key, bucket=self.bucket)
        if obj is None:
            raise FileNotFoundError(f"video not found: {s3key}")
        from_ts, to_ts = vid["from_ts"], vid["to_ts"]
        win = f"{from_ts:.3f}-{'end' if to_ts is None else f'{to_ts:.3f}'}"
        return s3key, obj, from_ts, to_ts, f"lerobot_{obj.etag}_{camera}_{win}.mp4"

    def video_cache_name(self, task: str, episode: str, camera: str, eye: str) -> str | None:
        name = self._clip_manifest(task).get(f"{episode}|{camera}")
        if name:
            return name                      # warmed: zero network
        return self._clip_source(task, episode, camera)[4]

    def video_path(self, task: str, episode: str, camera: str, eye: str) -> Path:
        meta, _row = self._row(task, episode)
        info = meta["info"]
        s3key, obj, from_ts, to_ts, cache_name = self._clip_source(task, episode, camera)

        def _produce(dst: Path):
            # Pull the source mp4 to a temp file and drop it afterwards — only the
            # per-episode clip is cached. Two ways to make that clip:
            #  - FAST PATH: the file is already H.264 4:2:0 and holds exactly this
            #    episode (one-episode-per-file exports such as yam_sim). Stream-copy
            #    it: no decode, milliseconds, bit-identical frames.
            #  - otherwise (packed AV1, or a packed H.264 file where the window is a
            #    slice): transcode the window to H.264. Cutting a slice with -c copy
            #    would snap to keyframes, so a slice is never stream-copied.
            cache.evict(headroom_gb=min(2.0, obj.size / 1024**3 * 1.5))
            # per pid AND thread: the warm runs several episodes of one packed chunk in parallel threads, and a
            # shared temp name let one thread truncate/unlink the file another thread's ffmpeg was reading
            tmp = cache.path_for(f"lerobot_{obj.etag}_{camera}.src.mp4.tmp{os.getpid()}_{threading.get_ident()}")
            try:
                s3.download(s3key, tmp, bucket=self.bucket)
                if self._can_stream_copy(tmp, from_ts, to_ts, info.get("fps")):
                    lerobot.remux(tmp, dst)
                else:
                    lerobot.transcode(tmp, dst, from_ts, to_ts, info.get("fps"))
            finally:
                tmp.unlink(missing_ok=True)

        # Derived-first: a clip the tier already holds is never pulled through the
        # container — /video presigns it and the browser fetches it from the bucket.
        return cache.get_or_create(cache_name, _produce, fetch=False)

    @staticmethod
    def _can_stream_copy(src: Path, from_ts: float, to_ts, fps) -> bool:
        """The fast-path decision, isolated so it can be tested without ffmpeg."""
        try:
            probe = lerobot.probe(src)
        except Exception:
            return False  # unreadable/odd container: let the transcode decide (and fail loudly)
        return lerobot.browser_playable(probe) and lerobot.covers_whole_file(
            from_ts, to_ts, probe.get("duration"), fps)

    def poster_path(self, task: str, episode: str, camera: str, eye: str) -> Path:
        """First-frame JPEG of a clip, derived-first like the clip itself."""
        name = self.video_cache_name(task, episode, camera, eye)
        pname = poster_name(name)

        def _produce(dst: Path):
            clip = cache.path_for(name)
            if not (clip.exists() and clip.stat().st_size > 0) and not cache.exists(name):
                clip = self.video_path(task, episode, camera, eye)   # decode, then fetch
                if not clip.exists():
                    cache.exists(name)
            lerobot.poster(clip, dst)

        return cache.get_or_create(pname, _produce, fetch=False)

    def warm_task(self, task: str, workers: int = 3, progress=None, posters: bool = True,
                  details: bool = True) -> dict:
        """Leave EVERYTHING the viewer needs for a task in the derived tier: the meta
        index, one detail JSON per episode, every clip, its poster, and the clip
        manifest; then refresh the source index. Idempotent and resumable — every
        artifact is checked before it is produced. Returns counts + failures."""
        meta = self._meta(task)
        info, ikey = meta["info"], meta.get("ikey")
        eps = sorted(meta["episodes"])
        cams = list(info["cameras"])
        failed: list[tuple[str, str, str]] = []
        n_total = (len(eps) if details else 0) + len(eps) * len(cams)
        done = [0]

        def _tick(res=None):
            done[0] += 1
            if res is not None:
                failed.append(res)
            if progress:
                progress(done[0], n_total, res)

        # 1) details, grouped by packed data file so each chunk is downloaded once
        if details and ikey:
            by_file: dict[tuple, list[int]] = {}
            for i in eps:
                if cache.remote_ready(self._detail_blob(ikey, i)):
                    _tick()
                    continue
                row = meta["episodes"][i]
                by_file.setdefault((row["data_chunk"], row["data_file"]), []).append(i)
            for (chunk, file), idxs in by_file.items():
                try:
                    key = info["data_path"].format(chunk_index=chunk, file_index=file)
                    tbl = lerobot.read_table(s3.get_bytes(f"{self._task_root(task)}/{key}", bucket=self.bucket))
                except Exception as ex:  # noqa: BLE001
                    for i in idxs:
                        _tick((self._ep_name(i), "detail", f"{type(ex).__name__}: {ex}"))
                    continue
                for i in idxs:
                    ep = self._ep_name(i)
                    try:
                        sub = lerobot.filter_episode(tbl, i)
                        cache.put_json(self._detail_blob(ikey, i),
                                       self._build_detail(task, ep, meta, meta["episodes"][i], sub),
                                       remote=True)
                        _tick()
                    except Exception as ex:  # noqa: BLE001
                        _tick((ep, "detail", f"{type(ex).__name__}: {ex}"))

        # 2) clips (+ posters), collecting cache names for the manifest
        manifest = dict(self._clip_manifest(task)) if ikey else {}

        def _one(job):
            ep, cam = job
            try:
                path = self.video_path(task, ep, cam, "left")
                if posters:
                    self.poster_path(task, ep, cam, "left")
                return (ep, cam, path.name, None)
            except Exception as ex:  # noqa: BLE001
                return (ep, cam, None, f"{type(ex).__name__}: {ex}")

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for ep, cam, name, err in pool.map(_one, [(self._ep_name(i), c) for i in eps for c in cams]):
                if err is None:
                    manifest[f"{ep}|{cam}"] = name
                    _tick()
                else:
                    _tick((ep, cam, err))

        # 3) manifest + source index
        if ikey:
            cache.put_json(self._clips_blob(ikey), {"v": self.INDEX_V, "clips": manifest}, remote=True)
            cache.note_remote_many(manifest.values())
            self._clips_cache[task] = manifest
        self.rebuild_source_index()
        return {"task": task, "episodes": len(eps), "clips": len(eps) * len(cams),
                "ok": n_total - len(failed), "failed": failed}

    def warm(self, task: str, cameras=None, episodes=None, workers: int = 4,
             progress=None) -> dict:
        """Produce (and, with the derived tier on, publish) every clip of a task so
        first views are cache hits. Returns {"clips", "ok", "failed": [...]}.

        Idempotent: video_path() returns immediately for clips already local or in
        the derived tier. Skips failures rather than stopping — one bad file must not
        leave the other 1499 clips cold."""
        info = self._meta(task)["info"]
        cams = list(cameras or info["cameras"])
        eps = list(episodes) if episodes is not None else self.list_episodes(task)
        jobs = [(e, c) for e in eps for c in cams]
        failed: list[tuple[str, str, str]] = []
        done = 0

        def _one(job):
            e, c = job
            try:
                self.video_path(task, e, c, "left")
                return None
            except Exception as ex:  # noqa: BLE001 - reported, not raised
                return (e, c, f"{type(ex).__name__}: {ex}")

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for res in pool.map(_one, jobs):
                done += 1
                if res is not None:
                    failed.append(res)
                if progress:
                    progress(done, len(jobs), res)
        return {"clips": len(jobs), "ok": len(jobs) - len(failed), "failed": failed}

    def episode_facts(self, task: str) -> dict:
        # meta/episodes is already memoized, so labelling the browse list costs
        # nothing extra. No timestamps in the format; status only where the
        # dataset wrote one (our sim exports do).
        return {self._ep_name(i): {"timestamp": None, "status": row.get("status")}
                for i, row in self._meta(task)["episodes"].items()}

    def _safe_stat(self, task, episode):
        # LeRobot per-episode stats come entirely from the in-memory-cached task
        # meta, so skip the base class's etag-keyed disk cache (one etag per task
        # would collide across episodes) — recomputing is already essentially free.
        try:
            return self.episode_stat(task, episode)
        except Exception:
            return None

    def episode_stat(self, task: str, episode: str) -> dict | None:
        try:
            meta, row = self._row(task, episode)
        except FileNotFoundError:
            return None
        info = meta["info"]
        # Duration from any camera's video window; else from frame count / fps.
        dur = next((round(v["to_ts"] - v.get("from_ts", 0.0), 3)
                    for v in row.get("videos", {}).values() if v.get("to_ts") is not None), None)
        if dur is None and row.get("length") and info.get("fps"):
            dur = round(row["length"] / info["fps"], 3)
        return {
            "task": task, "episode": episode,
            "duration_s": dur, "robot_frames": row.get("length"),
            "robot_hz": info.get("fps"), "num_cameras": len(info["cameras"]),
            "status": row.get("status"), "station": None, "teacher": None, "control": None,
            "timestamp": None,
            "has_annotations": None,  # would require reading each episode's data parquet
            "robot_type": info.get("robot_type"),
        }


class LeRobotSingleRootSource(LeRobotSource):
    """A LeRobot v3.0 dataset that is a SINGLE dataset at the prefix root
    (``<prefix>/{meta,data,videos}``), rather than one dataset per ``<prefix>/<task>``
    subfolder (which the base LeRobotSource assumes).

    Here the dataset's own ``task_index`` / per-episode ``tasks`` label is what we
    surface as the viewer's "tasks": all meta is loaded ONCE from the root, and its
    tens of thousands of episodes are grouped by task label so the UI browses
    task -> episode as usual. All packing/slicing/transcode logic is inherited —
    only task discovery and path rooting change (every path ignores the task arg
    and points at the single root)."""

    # ---- everything roots at the prefix, regardless of the task arg ----
    def _task_root(self, task: str) -> str:
        return self.prefix

    def _load_all(self) -> dict:
        # Reuse the base loader against the root (its _task_root("") == prefix),
        # then index episodes by task label. Cached under a fixed key.
        meta = super()._load_meta("")
        by_task: dict[str, list[int]] = {}
        for idx, row in meta["episodes"].items():
            tasks = row.get("tasks")
            label = tasks[0] if isinstance(tasks, (list, tuple)) and tasks else (tasks or "unlabeled")
            by_task.setdefault(str(label), []).append(idx)
        for v in by_task.values():
            v.sort()
        meta["by_task"] = by_task
        return meta

    def _meta(self, task: str = "") -> dict:
        # Single shared meta for the whole dataset (task arg is irrelevant to the
        # load); memoized under one key so the ~large episodes parquet is read once.
        with self._meta_lock:
            m = self._meta_cache.get("__root__")
            if m is None:
                m = self._load_all()
                self._meta_cache["__root__"] = m
            return m

    def list_tasks(self) -> list[str]:
        return sorted(self._meta()["by_task"])

    def _list_tasks_raw(self) -> list[str]:
        return self.list_tasks()

    def list_episodes(self, task: str) -> list[str]:
        idxs = self._meta()["by_task"].get(task, [])
        # oldest-first (ascending episode_index) to match the other sources' ordering
        return [self._ep_name(i) for i in sorted(idxs)]

    def episode_facts(self, task: str) -> dict:
        meta = self._meta()
        return {self._ep_name(i): {"timestamp": None, "status": meta["episodes"][i].get("status")}
                for i in meta["by_task"].get(task, [])}

    def _row(self, task: str, episode: str) -> tuple[dict, dict]:
        meta = self._meta()
        row = meta["episodes"].get(self._ep_index(episode))
        if row is None:
            raise FileNotFoundError(f"no such episode: {episode}")
        return meta, row


log = logging.getLogger("raiden_viz")

class WebbedSource(Source):
    """Ge's AnyData "webbed" export: one dataset folder per task, a split_all.json manifest per dataset and
    per-camera tar shards (see webbed.py). An episode is mined ONCE (every camera's rgb mp4, a turbo depth mp4,
    the robot summary from cam_high's lowdim) into the cache tiers under names keyed by the manifest etag, so
    the derived-first video route and the warm CLI work exactly as for the other sources. The export carries no
    capture timestamps (only the manifest's upload time), so collected_start / collected_end stay None here."""

    def __init__(self, spec):
        super().__init__(spec)
        self.datasets = list(spec["datasets"])
        self.split = spec.get("split", "split_all__IKDLA_48")
        self._manifests: dict[str, tuple[str, dict, dict]] = {}   # task -> (etag, parsed manifest, shared meta)
        self._lock = threading.Lock()

    def _split_prefix(self, task):
        return f"{self.prefix}/{task}/{self.split}"

    def _manifest(self, task):
        """(etag, parsed manifest, shared meta), re-read when the manifest object changes (one HEAD per call)."""
        key = f"{self._split_prefix(task)}/split_all.json"
        obj = s3.try_head(key, bucket=self.bucket)
        if obj is None:
            raise FileNotFoundError(f"no manifest for {task}")
        with self._lock:
            hit = self._manifests.get(task)
            if hit and hit[0] == obj.etag:
                return hit
        parsed = webbed.parse_manifest(s3.get_json(key, bucket=self.bucket))
        shared = webbed.shared_meta(s3.try_get_json(f"{self._split_prefix(task)}/metadata_shared.json",
                                                    bucket=self.bucket) or {})
        with self._lock:
            self._manifests[task] = (obj.etag, parsed, shared)
        return self._manifests[task]

    def list_tasks(self):
        return self.datasets

    def list_episodes(self, task):
        return sorted(self._manifest(task)[1]["episodes"])

    def task_facts(self, task):
        try:
            _, parsed, shared = self._manifest(task)
        except Exception:
            return {}
        return {"prompt": shared.get("prompt"), "tags": shared.get("tags"), "shards": parsed.get("n_files")}

    def _shards(self, task, episode):
        etag, parsed, shared = self._manifest(task)
        shards = parsed["episodes"].get(episode)
        if shards is None:
            raise FileNotFoundError(f"no episode {episode} in {task}")
        return etag, parsed, shared, shards

    def _cameras(self, parsed, shared):
        return list(parsed["cameras"] or shared.get("cameras") or [])

    @staticmethod
    def _base(etag, episode):
        return f"webbed_{etag}_{episode}"

    def video_cache_name(self, task, episode, camera, eye):
        etag, parsed, shared, _ = self._shards(task, episode)
        cams = self._cameras(parsed, shared)
        if camera not in cams and camera not in [f"{c}_depth" for c in cams]:
            raise FileNotFoundError(f"camera {camera!r} not found in {task}/{episode}")
        return f"{self._base(etag, episode)}_{camera}.mp4"

    def _mine(self, task, episode) -> dict:
        """Download every camera's shards for the episode, build one rgb mp4 + one depth mp4 per camera and the
        robot summary from the lowdim; publish them through cache.get_or_create (local + derived tier) and keep
        a small meta JSON beside them. Idempotent: a mined episode is answered from the meta JSON."""
        etag, parsed, shared, shards = self._shards(task, episode)
        base = self._base(etag, episode)
        meta_name = f"{base}_meta.json"
        meta = cache.get_json(meta_name, remote=True)
        cams = self._cameras(parsed, shared)
        # only the cameras that had video are checked: a camera without rgb would otherwise re-mine the episode on every view
        if meta and all(cache.exists(f"{base}_{c['name']}.mp4") for c in meta.get("cameras", []) if c.get("has_video")):
            return meta
        cache.evict(headroom_gb=1.0)
        tmp = Path(tempfile.mkdtemp(prefix="webbed_", dir=config.CACHE_DIR))
        try:
            jobs = [(c, s) for c in cams for s in shards]
            key_prefix = f"{self._split_prefix(task)}/{webbed.seq_key(episode)}"
            with ThreadPoolExecutor(max_workers=8) as pool:
                paths = dict(zip(jobs, pool.map(lambda cs: self._fetch_shard(key_prefix, cs[0], cs[1], tmp), jobs)))
            robot, prompt, cameras = None, None, []
            for c in cams:
                # one camera's shards in memory at a time: depth is float32, >1 GB per camera for a 1300-frame
                # episode, and holding every camera plus a concatenated copy OOM-killed the 8 GB hosted task
                shs = [webbed.read_shard(paths[(c, s)], tmp / c / s[:-4]) for s in shards]
                rgb = [sh["rgb"] for sh in shs if sh["rgb"] is not None]
                if rgb:
                    cache.get_or_create(f"{base}_{c}.mp4", lambda dst, _r=rgb: webbed.concat_rgb(_r, dst))
                cameras.append({"name": c, "has_video": bool(rgb), "eyes": ["left"]})
                depth = [sh["depth"] for sh in shs if sh["depth"] is not None]
                if depth:
                    cache.get_or_create(f"{base}_{c}_depth.mp4", lambda dst, _d=depth: webbed.depth_to_mp4(_d, dst))
                    cameras.append({"name": f"{c}_depth", "has_video": True, "eyes": ["left"]})
                if robot is None:
                    lows = [sh["lowdim"] for sh in shs if sh["lowdim"] is not None]
                    robot = webbed.robot_summary(lows)
                    prompt = next((lo["language"] for lo in lows if lo.get("language")), None)
                del shs, depth
            # rgb cameras first, depth tiles after them
            cameras.sort(key=lambda c: c["name"].endswith("_depth"))
            n = webbed.episode_frames(shards)
            meta = {"etag": etag, "cameras": cameras, "robot": robot, "instruction": prompt or shared.get("prompt"),
                    "shard_bounds_s": webbed.shard_bounds_s(shards),
                    "metadata": {"task_instruction": prompt or shared.get("prompt"), "dataset": task, "frames": n,
                                 "shards": len(shards), "duration_s": round(n / webbed.FPS, 2),
                                 "framerate": webbed.FPS, "resolution": shared.get("resolution"),
                                 "tags": shared.get("tags"), "action_dim": shared.get("action_dim")}}
            cache.put_json(meta_name, meta, remote=True)
            return meta
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _fetch_shard(self, key_prefix, cam, shard, tmp):
        dst = tmp / cam / shard
        dst.parent.mkdir(parents=True, exist_ok=True)
        s3.download(f"{key_prefix}/{cam}/{shard}", dst, bucket=self.bucket)
        return dst

    def episode_detail(self, task, episode):
        meta = self._mine(task, episode)
        return {"source": self.id, "task": task, "episode": episode, "instruction": meta["instruction"],
                "status": None, "metadata": meta["metadata"], "calibration": None, "cameras": meta["cameras"],
                "robot": meta["robot"], "annotations": [], "shard_bounds_s": meta["shard_bounds_s"]}

    def video_path(self, task, episode, camera, eye):
        name = self.video_cache_name(task, episode, camera, eye)
        if not cache.exists(name):
            self._mine(task, episode)
        if not cache.exists(name):
            raise FileNotFoundError(f"camera {camera!r} has no video in {task}/{episode}")
        return cache.path_for(name)

    def _stat_head(self, task, episode):
        return s3.try_head(f"{self._split_prefix(task)}/split_all.json", bucket=self.bucket)

    def _stat_cache_key(self, task, episode):
        key = super()._stat_cache_key(task, episode)   # the manifest etag is per dataset, so add the episode
        return key and key.replace(".json", f"_{episode}.json")

    def episode_stat(self, task, episode):
        obj = self._stat_head(task, episode)
        _, parsed, shared, shards = self._shards(task, episode)
        n = webbed.episode_frames(shards)
        return {"task": task, "episode": episode, "duration_s": round(n / webbed.FPS, 2), "robot_frames": n,
                "robot_hz": webbed.FPS, "num_cameras": len(self._cameras(parsed, shared)),
                "status": "success" if "success" in (shared.get("tags") or []) else None, "station": None,
                "timestamp": obj.last_modified if obj else None, "has_annotations": None}


_KINDS = {"raiden": RaidenSource, "raiden_rollouts": RolloutSource, "yam": YamMcapSource, "lerobot": LeRobotSource,
          "lerobot_single": LeRobotSingleRootSource, "webbed": WebbedSource}
_SOURCES: dict[str, Source] = {}


def _accessible(src: Source) -> bool:
    """Can this host actually read the source? Used to auto-hide sources whose
    bucket needs creds this host lacks (e.g. the vendor bucket on the EC2 box)."""
    try:
        s3.list_dirs(src.prefix, bucket=src.bucket)
        return True
    except Exception:
        return False


def get_sources(specs) -> dict[str, Source]:
    global _SOURCES
    if not _SOURCES:
        out = {}
        for s in specs:
            src = _KINDS[s["kind"]](s)
            # Sources flagged requires_access only register where readable.
            if s.get("requires_access") and not _accessible(src):
                continue
            out[s["id"]] = src
        _SOURCES = out
    return _SOURCES


def get_source(specs, sid: str) -> Source:
    src = get_sources(specs).get(sid)
    if src is None:
        raise KeyError(sid)
    return src
