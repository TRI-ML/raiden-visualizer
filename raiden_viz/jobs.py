"""Background jobs — the featured / preview / per-task warms and the stats build — run as ONE niced child
process at a time per host, never on the API process's threads. On 2026-10-31 ECS killed the hosted dev task
twice for failed ALB health checks while in-process warms (CPU-bound Python + ffmpeg) starved the event loop.

Status lives in ``CACHE_DIR/job_<kind>_<sid>[_<task>].json`` so the GET routes and external drivers read it
unchanged; a killed container only loses the in-memory queue (the persisted artifacts survive).

    python -m raiden_viz.jobs <kind> <sid> '<params json>'     # what the queue runner spawns
"""

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import deque

from . import cache, config

INLINE = False   # tests: run the job in a thread of this process (so monkeypatches reach it) instead of a child

_queue: deque = deque()
_lock = threading.Lock()
_runner: threading.Thread | None = None
_current: str | None = None


def _name(kind: str, sid: str, task: str | None = None) -> str:
    safe = lambda s: re.sub(r"[^A-Za-z0-9_.-]", "_", s)  # noqa: E731
    return f"job_{kind}_{safe(sid)}" + (f"_{safe(task)}" if task else "") + ".json"


def _active(name: str) -> bool:
    return name == _current or any(n == name for n, _ in _queue)


def status(kind: str, sid: str, task: str | None = None) -> dict:
    """The job's status file, or the idle shape. A file left 'running' by a host that died is reported as
    interrupted rather than running forever."""
    name = _name(kind, sid, task)
    st = cache.get_json(name)
    if not st:
        return {"running": False, "done": False}
    if st.get("running") and not INLINE and not _active(name):
        st = {**st, "running": False, "error": st.get("error") or "interrupted (the host restarted)"}
    return st


def start(kind: str, sid: str, task: str | None = None, src=None, **params) -> dict:
    """Queue the job (idempotent while it is queued or running) and return its status."""
    global _runner
    name = _name(kind, sid, task)
    with _lock:
        st = cache.get_json(name)
        if st and st.get("running") and _active(name):
            return dict(st)
        st = {"running": True, "done": False, "queued": True, "total": 0, "completed": 0, "failed": 0,
              "error": None, "started_at": time.time(), "result": None}
        cache.put_json(name, st)
        if INLINE:
            threading.Thread(target=run, args=(kind, sid, task, params, src), daemon=True, name=f"job-{kind}").start()
            return dict(st)
        _queue.append((name, [sys.executable, "-m", "raiden_viz.jobs", kind, sid, json.dumps({"task": task, **params})]))
        if _runner is None or not _runner.is_alive():
            _runner = threading.Thread(target=_drain, daemon=True, name="jobs")
            _runner.start()
    return dict(st)


def _drain() -> None:
    global _current
    while True:
        with _lock:
            if not _queue:
                _current = None
                return
            name, argv = _queue.popleft()
            _current = name
        if shutil.which("nice"):
            argv = ["nice", "-n", "10", *argv]
        # decoders in the child keep one core for the API process (config.FFMPEG_THREADS reads this)
        env = {**os.environ, "RAIDEN_FFMPEG_THREADS": os.environ.get("RAIDEN_FFMPEG_THREADS", "1")}
        err = None
        try:
            rc = subprocess.Popen(argv, env=env).wait()
            if rc:
                err = f"job exited {rc}"
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
        st = cache.get_json(name) or {}
        if st.get("running") or err:        # the child died before writing its end state
            st.update(running=False, done=True, error=st.get("error") or err)
            cache.put_json(name, st)


def run(kind: str, sid: str, task: str | None, params: dict, src=None) -> None:
    """The job body (child process, or a thread when INLINE). Progress is flushed to the status file."""
    from . import sources
    from . import warm as warm_mod
    name = _name(kind, sid, task)
    st = cache.get_json(name) or {"total": 0, "completed": 0, "failed": 0, "error": None, "result": None}
    st.update(running=True, done=False, queued=False, started_at=time.time())
    last = [0.0]

    def flush(force=False):
        if force or time.time() - last[0] > 1:
            last[0] = time.time()
            cache.put_json(name, st)

    def progress(done, total, res):
        st["completed"], st["total"] = done, total
        if res is not None:
            st["failed"] += 1
        flush()

    flush(True)
    try:
        from . import argus
        argus.BUILD_INLINE = True            # the child may parse label tables; the API process never does
        src = src or sources.get_source(config.SOURCES, sid)
        if kind == "index":
            # LeRobot source index (streamed per task) + every Argus label table the source joins, so the API
            # process only ever READS these blobs. Other kinds just need the source resolved.
            idx = src.rebuild_source_index()
            tasks = [r["task"] for r in idx["tasks"]]
            st["total"] = len(tasks)
            for i, t in enumerate(tasks, 1):
                if src.spec.get("argus"):
                    for repo in argus._repos_for(src.spec, t):
                        argus._load(src.spec, repo)
                st["completed"] = i
                flush()
            st["result"] = {"tasks": len(tasks), "episodes": idx.get("total")}
        elif kind == "featured":
            tasks = src.list_tasks()
            st["total"] = len(tasks)
            for i, t in enumerate(tasks, 1):
                if warm_mod.warm_featured(src, [t], workers=max(1, min(int(params.get("workers", 2)), 4))):
                    st["failed"] += 1
                st["completed"] = i
                flush()
            warm_mod.warm_meta(src)
            st["result"] = {"tasks": len(tasks), "featured_n": config.FEATURED_N}
        elif kind == "previews":
            tf = params.get("task_filter")
            st["result"] = src.preview_warm(tasks=[tf] if tf else None, progress=progress,
                                            only_missing=params.get("only_missing", True))
        elif kind == "task":
            st["result"] = src.warm_task(task, workers=max(1, min(int(params.get("workers", 3)), 6)), progress=progress,
                                         posters=params.get("posters", True), details=params.get("details", True))
        elif kind == "stats":
            src.stats(wait=True)          # builds inline here and persists the blob the API process then serves
            st["result"] = {"ok": True}
        else:
            raise ValueError(f"unknown job kind {kind!r}")
    except Exception as e:  # noqa: BLE001
        st["error"] = f"{type(e).__name__}: {e}"
    finally:
        st["running"], st["done"] = False, True
        flush(True)


if __name__ == "__main__":
    _kind, _sid, _params = sys.argv[1], sys.argv[2], json.loads(sys.argv[3] if len(sys.argv) > 3 else "{}")
    run(_kind, _sid, _params.pop("task", None), _params)
