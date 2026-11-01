"""jobs.py: warms run in a niced child process, one at a time, with a status file the routes read."""
import json
import time

from fastapi.testclient import TestClient

from raiden_viz import app as app_module
from raiden_viz import cache, jobs


class _Src:
    spec = {"id": "r"}

    def list_tasks(self):
        return ["a"]


def test_warm_route_spawns_one_niced_child_and_status_reads_its_file(cache_dir, monkeypatch):
    monkeypatch.setattr(jobs, "INLINE", False)
    monkeypatch.setattr(jobs, "_queue", type(jobs._queue)())
    monkeypatch.setattr(app_module, "_src", lambda sid: _Src())
    spawned = []

    class FakeProc:
        def __init__(self, argv, env):
            spawned.append((argv, env))

        def wait(self):   # the child writes its end state, like jobs.run does
            name = jobs._name("featured", "r")
            cache.put_json(name, {**cache.get_json(name), "running": False, "done": True, "queued": False,
                                  "total": 1, "completed": 1, "result": {"tasks": 1}})
            return 0

    monkeypatch.setattr(jobs.subprocess, "Popen", FakeProc)
    c = TestClient(app_module.app)
    st = c.post("/api/sources/r/warm").json()
    assert st["running"] and st["queued"]
    for _ in range(200):
        st = c.get("/api/sources/r/warm").json()
        if st["done"]:
            break
        time.sleep(0.01)
    assert st["done"] and st["completed"] == 1 and st["result"] == {"tasks": 1}
    assert len(spawned) == 1
    argv, env = spawned[0]
    assert argv[:3] == ["nice", "-n", "10"] and "raiden_viz.jobs" in argv and argv[-3:-1] == ["featured", "r"]
    assert json.loads(argv[-1]) == {"task": None, "workers": 2}
    assert env["RAIDEN_FFMPEG_THREADS"] == "1"


def test_status_reports_a_file_left_running_by_a_dead_host_as_interrupted(cache_dir, monkeypatch):
    monkeypatch.setattr(jobs, "INLINE", False)
    cache.put_json(jobs._name("previews", "r"), {"running": True, "done": False, "completed": 3, "total": 9})
    st = jobs.status("previews", "r")
    assert st["running"] is False and st["completed"] == 3 and "interrupted" in st["error"]
    assert jobs.status("previews", "nothing") == {"running": False, "done": False}


def test_child_entry_runs_the_job_and_persists_the_end_state(cache_dir, monkeypatch):
    from raiden_viz import sources
    calls = []

    class S:
        def preview_warm(self, tasks=None, progress=None, only_missing=True):
            calls.append((tasks, only_missing))
            progress(1, 1, None)
            return {"ok": 1, "failed": []}

    monkeypatch.setattr(sources, "get_source", lambda specs, sid: S())
    jobs.run("previews", "r", None, {"task_filter": "t1", "only_missing": False})
    st = cache.get_json(jobs._name("previews", "r"))
    assert st["done"] and not st["running"] and st["result"]["ok"] == 1 and st["completed"] == 1
    assert calls == [(["t1"], False)]
