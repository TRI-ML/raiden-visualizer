"""Dataset source adapters.

The viewer browses more than one raw dataset layout. Each layout is handled by a
Source subclass exposing the same methods, so the API routes and the frontend are
format-agnostic:

    list_tasks()                    -> [task, ...]
    list_episodes(task)             -> [episode, ...]   (newest first)
    episode_detail(task, episode)   -> {instruction, status, cameras[], robot, ...}
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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import cache, calib_overlay, config, fk, robot_data, s3, svo, webbed, yam

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


class Source:
    def __init__(self, spec: dict):
        self.id = spec["id"]
        self.label = spec["label"]
        self.bucket = spec["bucket"]
        self.prefix = spec["prefix"].strip("/")
        self.spec = spec

    # ---- browsing (shared) ----
    def list_tasks(self) -> list[str]:
        return s3.list_dirs(self.prefix, bucket=self.bucket)

    def all_episodes(self, task: str) -> list[str]:
        eps = s3.list_dirs(f"{self.prefix}/{task}", bucket=self.bucket)
        return sorted(eps, reverse=True)

    def list_episodes(self, task: str, all: bool = False) -> list[str]:
        """The task's FEATURED episodes (config.FEATURED_N, a stable random sample kept in a small JSON per task
        so the pick never changes and can be pre-decoded on the host); ``all`` returns every episode."""
        eps = self.all_episodes(task)
        if all or len(eps) <= config.FEATURED_N:
            return eps
        key = f"featured_{self.id}_{re.sub(r'[^A-Za-z0-9_.-]', '_', task)}.json"
        hit = cache.get_json(key)
        if hit and set(hit.get("episodes") or []) <= set(eps):
            return hit["episodes"]
        pick = random.Random(f"{self.id}/{task}").sample(sorted(eps), config.FEATURED_N)
        pick.sort(key=eps.index)
        cache.put_json(key, {"episodes": pick, "of": len(eps)})
        return pick

    # ---- per-source ----
    def task_facts(self, task: str) -> dict:
        """Extra per-task overview fields (prompt, tags, shards); sources without them return {}."""
        return {}

    def episode_detail(self, task: str, episode: str) -> dict:
        raise NotImplementedError

    def video_path(self, task: str, episode: str, camera: str, eye: str) -> Path:
        raise NotImplementedError

    def episode_stat(self, task: str, episode: str) -> dict | None:
        raise NotImplementedError

    # ---- aggregates (shared, built on the above) ----
    def overview(self) -> dict:
        tasks = self.list_tasks()
        per_task, total, stations = [], 0, set()
        for task in tasks:
            eps = self.all_episodes(task)
            total += len(eps)
            latest = eps[0] if eps else None
            for ep in eps:
                m = re.match(r"^([A-Za-z][\w-]*)_\d{4}-\d{2}-\d{2}T", ep)
                if m:
                    stations.add(m.group(1))
            per_task.append({"task": task, "episodes": len(eps), "featured": min(len(eps), config.FEATURED_N),
                             "latest": latest, **self.task_facts(task)})
        per_task.sort(key=lambda t: t["episodes"], reverse=True)
        return {
            "source": self.id, "bucket": self.bucket, "prefix": self.prefix,
            "num_tasks": len(tasks), "num_episodes": total,
            "stations": sorted(stations), "tasks": per_task,
        }

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

    def stats(self, full: bool = False) -> dict:
        """Synchronous stats pass (used for the charts). For a full scan of a huge
        source prefer scan_start()/scan_snapshot(), which stream progress."""
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

    def scan_start(self) -> dict:
        """Begin (or resume) a background full scan of every episode's stats.
        Returns an immediate snapshot; poll scan_snapshot() for progress. Idempotent:
        a scan already running/finished for this source is reused."""
        sid = self._scan_id()
        with _SCANS_GUARD:
            st = _SCANS.get(sid)
            if st and (st["running"] or st["done"]):
                return self._snapshot(st)
            pairs, total = self._stat_pairs(full=True)
            st = {"running": True, "done": False, "total": total,
                  "episodes": [], "error": None, "lock": threading.Lock()}
            _SCANS[sid] = st
        t = threading.Thread(target=self._run_scan, args=(sid, pairs, st), daemon=True)
        t.start()
        return self._snapshot(st)

    def _run_scan(self, sid, pairs, st):
        err = None
        try:
            with ThreadPoolExecutor(max_workers=32) as pool:
                for fut in [pool.submit(self._safe_stat, t, e) for t, e in pairs]:
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

    def episode_detail(self, task, episode):
        prefix = self._ep_prefix(task, episode)
        metadata = s3.get_json(f"{prefix}/metadata.json", bucket=self.bucket)

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

    def video_path(self, task, episode, camera, eye):
        key = f"{self._ep_prefix(task, episode)}/cameras/{camera}.svo2"
        obj = s3.try_head(key, bucket=self.bucket)
        if obj is None:
            raise FileNotFoundError(f"camera not found: {camera}")
        if obj.size < 100_000:
            raise ValueError(f"camera '{camera}' has no recorded video (stub file)")
        svo_local = cache.get_or_create(
            f"{obj.etag}_{camera}.svo2",
            lambda dst: s3.download(key, dst, bucket=self.bucket),
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
            "timestamp": md.get("timestamp"),
            "has_annotations": None,  # raiden has no subtask annotations
        }


class YamMcapSource(Source):
    """One MCAP per episode. Download the big MCAP once, extract all small
    artifacts (per-camera mp4 + robot/instruction json), cache them, drop the MCAP.

    The per-episode MCAP basename varies by dataset (``output.mcap`` for the
    russet/yam_raw layout, ``episode.mcap`` for ABC-130k), set via spec['mcap_name']."""

    def _mcap_key(self, task, episode):
        name = self.spec.get("mcap_name", "output.mcap")
        return f"{self.prefix}/{task}/{episode}/{name}"

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
            if all(cache.path_for(f"yam_{obj.etag}_{c}.mp4").exists() for c in ex["cameras"]):
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
                mp4 = cache.path_for(f"yam_{obj.etag}_{cam}.mp4")
                if not mp4.exists():
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

    def video_path(self, task, episode, camera, eye):
        obj = self._head(task, episode)
        mp4 = cache.path_for(f"yam_{obj.etag}_{camera}.mp4")
        if not mp4.exists():
            # Mining extracts all cameras at once (single MCAP download), then
            # drops the MCAP — so this only downloads on a true cold miss.
            self._mine(obj, task, episode)
        if not mp4.exists():
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
            "status": None, "station": None, "timestamp": obj.last_modified,
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


class WebbedSource(Source):
    """Ge's AnyData "webbed" export: one dataset folder per task, a split_all.json manifest per dataset and
    per-camera tar shards (see webbed.py). Episodes are mined once (all cameras + depth + lowdim) into the
    cache keyed by the manifest etag."""

    def __init__(self, spec):
        super().__init__(spec)
        self.datasets = list(spec["datasets"])
        self.split = spec.get("split", "split_all__IKDLA_48")
        self._manifests: dict[str, tuple[str, dict, dict]] = {}   # task -> (etag, parsed manifest, shared meta)
        self._lock = threading.Lock()

    def _split_prefix(self, task):
        return f"{self.prefix}/{task}/{self.split}"

    def _manifest(self, task):
        """(etag, parsed manifest, shared meta), re-read when the manifest object changes."""
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

    def all_episodes(self, task):
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

    def _mine(self, task, episode) -> dict:
        """Download every camera's shards for the episode, build one rgb mp4 + one depth mp4 per camera and the
        robot summary from cam_high's lowdim; cache all of it keyed by (manifest etag, episode). Idempotent."""
        etag, parsed, shared, shards = self._shards(task, episode)
        base = f"webbed_{etag}_{episode}"
        meta_path = cache.path_for(f"{base}_meta.json")
        cams = parsed["cameras"] or shared.get("cameras") or []
        if meta_path.exists() and all(cache.path_for(f"{base}_{c}.mp4").exists() for c in cams):
            return json.loads(meta_path.read_text())
        cache.evict(headroom_gb=1.0)
        tmp = Path(tempfile.mkdtemp(prefix="webbed_", dir=config.CACHE_DIR))
        try:
            per_cam = {c: [] for c in cams}
            jobs = [(c, s) for c in cams for s in shards]
            key_prefix = f"{self._split_prefix(task)}/{webbed.seq_key(episode)}"
            with ThreadPoolExecutor(max_workers=8) as pool:
                paths = list(pool.map(lambda cs: self._fetch_shard(key_prefix, cs[0], cs[1], tmp), jobs))
            for (c, s), p in zip(jobs, paths):
                per_cam[c].append(webbed.read_shard(p, tmp / c / s[:-4]))
            robot, prompt = None, None
            for c in cams:
                rgb = [sh["rgb"] for sh in per_cam[c] if sh["rgb"] is not None]
                if rgb and not cache.path_for(f"{base}_{c}.mp4").exists():
                    cache.get_or_create(f"{base}_{c}.mp4", lambda dst, _r=rgb: webbed.concat_rgb(_r, dst))
                depth = [sh["depth"] for sh in per_cam[c] if sh["depth"] is not None]
                if depth and not cache.path_for(f"{base}_{c}_depth.mp4").exists():
                    stack = __import__("numpy").concatenate(depth)
                    cache.get_or_create(f"{base}_{c}_depth.mp4", lambda dst, _d=stack: webbed.depth_to_mp4(_d, dst))
                if robot is None:
                    lows = [sh["lowdim"] for sh in per_cam[c] if sh["lowdim"] is not None]
                    robot = webbed.robot_summary(lows)
                    prompt = next((l["language"] for l in lows if l.get("language")), None)
            cameras = [{"name": c, "has_video": cache.path_for(f"{base}_{c}.mp4").exists(), "eyes": ["left"]} for c in cams]
            cameras += [{"name": f"{c}_depth", "has_video": True, "eyes": ["left"]}
                        for c in cams if cache.path_for(f"{base}_{c}_depth.mp4").exists()]
            meta = {"etag": etag, "cameras": cameras, "robot": robot,
                    "instruction": prompt or shared.get("prompt"),
                    "shard_bounds_s": webbed.shard_bounds_s(shards),
                    "metadata": {"task_instruction": prompt or shared.get("prompt"), "dataset": task,
                                 "frames": webbed.episode_frames(shards), "shards": len(shards),
                                 "duration_s": round(webbed.episode_frames(shards) / webbed.FPS, 2),
                                 "framerate": webbed.FPS, "resolution": shared.get("resolution"),
                                 "tags": shared.get("tags"), "action_dim": shared.get("action_dim")}}
            meta_path.write_text(json.dumps(meta))
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
        etag, *_ = self._shards(task, episode)
        mp4 = cache.path_for(f"webbed_{etag}_{episode}_{camera}.mp4")
        if not mp4.exists():
            self._mine(task, episode)
        if not mp4.exists():
            raise FileNotFoundError(f"camera {camera!r} not found in {task}/{episode}")
        return mp4

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
                "robot_hz": webbed.FPS, "num_cameras": len(parsed["cameras"] or shared.get("cameras") or []),
                "status": "success" if "success" in (shared.get("tags") or []) else None, "station": None,
                "timestamp": obj.last_modified if obj else None, "has_annotations": None}


_KINDS = {"raiden": RaidenSource, "yam": YamMcapSource, "webbed": WebbedSource}
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
