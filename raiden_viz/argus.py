"""Zheyuan Hu's Argus sub-task labels (GPT-6.1 Sol + Claude Opus 5.5 via Bedrock, 2026-10), mirrored at
s3://tri-ml-datasets-uw2/geyan/argus/argus_labels/<pool>/<run>/tables*/<repo>/meta/loong/annotations/
(LeRobot v3 ``meta/loong/annotations`` tables: ``info.json`` + ``chunk-000/file-NNN.parquet`` with columns
episode_index, stem, annotations = list of {level, text, start, end, weight, given, source, label_class}; start/end are
frame indices on the repo's 30 Hz clock). A source opts in with spec["argus"] = {"tables": <prefix>, "repo_prefix": ...}
(one repo per task, ``<repo_prefix>NNN_<task>``, joined by the episode uuid = ``stem``) or {"tables", "repo", "join":
"index"} (one repo, joined by the LeRobot episode index). Only the level-1 ``subtask`` rows of SOURCE are shown, as the
viewer's ``annotations: [{t, text}]`` (t = start / fps, seconds from the episode start)."""
from __future__ import annotations

import io
import re
import threading

from . import cache, config, s3

SOURCE = "argus:sol61"          # Sol's sub-tasks: boundaries within 3 s of 97 % of the human ones (campaign README §4.4)
_lock = threading.Lock()
_repos: dict[str, dict] = {}    # sid -> {task: repo_name}
_tables: dict[str, dict] = {}   # f"{sid}/{repo}" -> {key: [[t, text], ...]}


def _repo_for(spec: dict, task: str) -> str | None:
    a = spec["argus"]
    if a.get("repo"):
        return a["repo"]
    sid = spec["id"]
    with _lock:
        m = _repos.get(sid)
    if m is None:
        m = {}
        for d in s3.list_dirs(a["tables"], bucket=a.get("bucket") or config.S3_BUCKET):
            mm = re.match(re.escape(a["repo_prefix"]) + r"\d+_(.+)$", d)
            if mm:
                m[mm.group(1)] = d
        with _lock:
            _repos[sid] = m
    return m.get(task)


def _load(spec: dict, repo: str) -> dict:
    """{stem | str(episode_index): [[t_s, text], ...]} for one repo table; parsed once, persisted as JSON."""
    a = spec["argus"]
    key = f"{spec['id']}/{repo}"
    with _lock:
        hit = _tables.get(key)
    if hit is not None:
        return hit
    name = f"argus_{spec['id']}_{repo}.json"
    tbl = cache.get_json(name, remote=True)
    if tbl is None:
        import pyarrow.parquet as pq
        bucket = a.get("bucket") or config.S3_BUCKET
        base = f"{a['tables']}/{repo}/meta/loong/annotations"
        tbl = {}
        for obj in s3.list_keys(base, bucket=bucket, suffix=".parquet"):
            pt = pq.read_table(io.BytesIO(s3.get_bytes(obj.key, bucket=bucket)))
            fps = 30.0
            for row in pt.to_pylist():
                segs = sorted(((r["start"] / fps, r["text"]) for r in row["annotations"]
                               if r["source"] == SOURCE and r["label_class"] == "subtask"), key=lambda x: x[0])
                if segs:
                    tbl[str(row["episode_index"])] = segs
                    if row.get("stem"):
                        tbl[row["stem"]] = segs
        cache.put_json(name, tbl, remote=True)
    with _lock:
        _tables[key] = tbl
    return tbl


def _key(spec: dict, episode: str) -> str:
    if spec["argus"].get("join") == "index":
        m = re.search(r"(\d+)$", episode)
        return str(int(m.group(1))) if m else episode
    return episode.removeprefix("episode_")


def annotations_for(spec: dict, task: str, episode: str) -> list[dict]:
    """Viewer-shape Argus labels for one episode, [] when the dataset or the episode has none."""
    if not spec.get("argus"):
        return []
    repo = _repo_for(spec, task)
    if not repo:
        return []
    return [{"t": round(t, 3), "text": text, "source": SOURCE} for t, text in _load(spec, repo).get(_key(spec, episode), [])]


def coverage(spec: dict, task: str, episodes: list) -> int | None:
    """How many of the task's episodes have at least one Argus sub-task row (None when the dataset has no tables)."""
    if not spec.get("argus"):
        return None
    repo = _repo_for(spec, task)
    if not repo:
        return 0
    tbl = _load(spec, repo)
    return sum(1 for e in episodes if _key(spec, e) in tbl)
