"""The LeRobot source index and the Argus label tables are built by the ``index`` job, never on a request
(2026-11-01: RealOmni's 589k-episode index was built and held in the API process)."""
import json
import threading
import time

import pytest
from starlette.testclient import TestClient

from raiden_viz import app as app_module
from raiden_viz import argus, cache, clips, config, jobs, sources

SPEC = {"id": "yam_sim", "label": "YAM Sim", "kind": "lerobot",
        "bucket": "tri-yam", "prefix": "sim_datasets", "subdir": "lerobot"}
INFO = {"fps": 30, "cameras": ["scene_camera"], "robot_type": "yam",
        "video_keys": {"scene_camera": "observation.images.scene_camera"},
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"}


def _row(i, status="success"):
    return {"episode_index": i, "length": 300 + i, "data_chunk": 0, "data_file": 0, "tasks": None, "status": status,
            "videos": {"scene_camera": {"chunk": 0, "file": i, "from_ts": 0.0, "to_ts": 10.0 + i}}}


@pytest.fixture
def remote(monkeypatch, fake_s3):
    monkeypatch.setattr(config, "DERIVED_BUCKET", "derived-bucket")
    monkeypatch.setattr(config, "DERIVED_PREFIX", "derived")
    monkeypatch.setattr(cache, "_derived_client", lambda: fake_s3)
    return fake_s3


def _src(monkeypatch, tasks):
    src = sources.LeRobotSource(SPEC)
    for t, n in tasks.items():
        src._meta_cache[t] = {"info": INFO, "tasks": {}, "episodes": {i: _row(i) for i in range(n)}, "ikey": f"k-{t}"}
    monkeypatch.setattr(src, "_list_tasks_raw", lambda: list(tasks))
    return src


def test_missing_index_is_not_built_on_the_request_and_queues_the_index_job(remote, monkeypatch):
    src = _src(monkeypatch, {"plate": 3})
    started = []
    monkeypatch.setattr(sources.threading, "Thread",
                        lambda target=None, **kw: started.append(target) or type("T", (), {"start": lambda self: None})())
    assert src.source_index() is None                      # nothing persisted → not built here
    assert started and started[0] == src._safe_rebuild_index   # ... but the build was handed off (thread when inline)
    assert src.list_tasks() == []
    assert src.stats(wait=False) is None
    ov = src.overview()
    assert ov["computing"] is True and ov["tasks"] == []
    assert "derived/" + src._swr_blob("overview_v2") not in remote.objects   # a computing overview is never persisted
    monkeypatch.setattr(app_module, "_src", lambda sid: src)
    c = TestClient(app_module.app)
    assert c.get("/api/sources/yam_sim/stats").json()["computing"] is True
    assert c.get("/api/sources/yam_sim/overview").json()["computing"] is True
    assert c.get("/api/sources/yam_sim/tasks").json() == {"tasks": []}


def test_index_job_builds_a_streamed_index_the_server_then_serves(remote, monkeypatch):
    src = _src(monkeypatch, {"plate": 3, "banana": 2})
    monkeypatch.setattr(sources, "get_source", lambda specs, sid: src)
    jobs.run("index", "yam_sim", None, {}, src=src)
    st = jobs.status("index", "yam_sim")
    assert st["done"] and st["error"] is None and st["result"] == {"tasks": 2, "episodes": 5}
    idx = src.source_index()
    assert idx["v"] == sources.LeRobotSource.SOURCE_INDEX_V and "stats" not in idx
    assert [t["task"] for t in idx["tasks"]] == ["plate", "banana"]
    assert idx["tasks"][0]["episode_indices"] == [0, 1, 2] and idx["total"] == 5
    assert len(idx["sample"]) == 5                                 # small source: every record is in the sample
    assert src.list_episodes("banana") == ["episode_000000", "episode_000001"]   # from the index, no meta
    assert src.stats()["total_episodes"] == 5 and src.stats()["episodes"][0]["status"] == "success"
    assert src.overview()["num_episodes"] == 5
    # the streamed build keeps at most one task's records alive: the per-task meta memo is dropped as it goes
    assert "plate" not in src._meta_cache


def test_index_sample_is_bounded_and_covers_every_task(remote, monkeypatch):
    src = _src(monkeypatch, {"a": 40, "b": 40, "c": 2})
    monkeypatch.setattr(sources.LeRobotSource, "STATS_MAX", 30)
    idx = src.rebuild_source_index()
    assert idx["total"] == 82 and len(idx["sample"]) <= 30
    assert {r["task"] for r in idx["sample"]} == {"a", "b", "c"}
    st = src.stats()
    assert st["sampled"] is True and st["total_episodes"] == 82


def test_argus_tables_are_not_parsed_on_a_server_request(remote, monkeypatch):
    spec = {"id": "abc", "argus": {"tables": "geyan/argus/x", "repo": "{task}", "join": "index"}}
    monkeypatch.setattr(argus, "BUILD_INLINE", False)
    monkeypatch.setattr(argus.s3, "list_keys", lambda *a, **k: pytest.fail("parsed parquet on the request path"))
    queued = []
    monkeypatch.setattr(argus.jobs, "start", lambda kind, sid, *a, **k: queued.append((kind, sid)) or {})
    assert argus.annotations_for(spec, "plate", "episode_000003") == []
    assert argus.coverage(spec, "plate", ["episode_000003"]) == 0
    assert queued == [("index", "abc"), ("index", "abc")]
    # once the job has persisted the blob the server reads it
    cache.put_json("argus_abc_plate.json", {"3": [[1.5, "grab"]]}, remote=True)
    argus._tables.clear()
    assert argus.annotations_for(spec, "plate", "episode_000003") == [{"t": 1.5, "text": "grab", "source": argus.SOURCE}]


def test_on_demand_decodes_are_capped_per_container(monkeypatch):
    monkeypatch.setattr(config, "MAX_DECODES", 1)
    jobs_ = clips.ClipJobs()
    running, peak, lock = [0], [0], threading.Lock()
    release = threading.Event()

    def produce():
        with lock:
            running[0] += 1; peak[0] = max(peak[0], running[0])
        release.wait(2)
        with lock:
            running[0] -= 1
        return "/tmp/x.mp4"
    for k in ("a", "b", "c"):
        assert jobs_.ensure(k, produce)["decoding"] is True
    time.sleep(0.3)
    assert peak[0] == 1                                     # two of them wait for the single slot
    release.set()
    for _ in range(50):
        if all((jobs_.state(k) or {}).get("ready") for k in ("a", "b", "c")):
            break
        time.sleep(0.05)
    assert all(jobs_.state(k)["ready"] for k in ("a", "b", "c")) and peak[0] == 1
