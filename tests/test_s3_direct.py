"""S3-direct serving for LeRobot sources: every page answers from artifacts that were
computed once and published to the derived tier — task meta index, source index (tasks,
counts, stat records), per-episode detail JSON, clip manifest, posters — instead of
re-parsing parquet / re-HEADing S3 per request or per container.
"""

import json
import time
from pathlib import Path

import pytest

from raiden_viz import cache, config, lerobot, sources

SPEC = {"id": "yam_sim", "label": "YAM Sim", "kind": "lerobot",
        "bucket": "tri-yam", "prefix": "sim_datasets", "subdir": "lerobot"}
INFO = {"fps": 30, "cameras": ["scene_camera"],
        "video_keys": {"scene_camera": "observation.images.scene_camera"},
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"}
ROW = {"episode_index": 7, "length": 1410, "data_chunk": 0, "data_file": 0, "tasks": None,
       "status": "success",
       "videos": {"scene_camera": {"chunk": 0, "file": 7, "from_ts": 0.0, "to_ts": 47.0}}}


@pytest.fixture
def remote(monkeypatch, fake_s3):
    monkeypatch.setattr(config, "DERIVED_BUCKET", "derived-bucket")
    monkeypatch.setattr(config, "DERIVED_PREFIX", "derived")
    monkeypatch.setattr(cache, "_derived_client", lambda: fake_s3)
    return fake_s3


@pytest.fixture
def meta_listing(monkeypatch):
    """<root>/meta/ holds four objects; their etags define the index key."""
    def list_keys(prefix, bucket=None, suffix=None):
        assert prefix.endswith("/meta/") or "/meta/episodes" in prefix
        objs = [sources.s3.S3Object(f"{prefix.rstrip('/')}/{n}", 10, f"e-{n}")
                for n in ("info.json", "tasks.parquet", "episodes/chunk-000/file-000.parquet")]
        return [o for o in objs if not suffix or o.key.endswith(suffix)]
    monkeypatch.setattr(sources.s3, "list_keys", list_keys)


def _blob(src, task):
    return src._meta_blob(src._index_key(task))


def test_meta_is_restored_from_the_derived_json_without_parquet(remote, meta_listing, monkeypatch):
    src = sources.LeRobotSource(SPEC)
    remote.objects["derived/" + _blob(src, "plate")] = json.dumps(
        {"v": 1, "info": INFO, "tasks": {"0": "put the plate"}, "episodes": {"7": ROW}}).encode()
    monkeypatch.setattr(sources.s3, "get_bytes", lambda *a, **k: pytest.fail("parquet was read"))
    monkeypatch.setattr(sources.s3, "get_json", lambda *a, **k: pytest.fail("info.json was read"))
    m = src._meta("plate")
    assert m["episodes"][7]["status"] == "success"        # int keys back
    assert m["tasks"][0] == "put the plate"
    assert m["ikey"] == src._index_key("plate")


def test_meta_parsed_once_is_published(remote, meta_listing, monkeypatch):
    import pyarrow as pa
    src = sources.LeRobotSource(SPEC)
    monkeypatch.setattr(sources.s3, "get_json", lambda key, bucket=None: dict(INFO))
    tasks_tbl = pa.table({"task_index": [0], "task": ["put the plate"]})
    eps_tbl = pa.table({"episode_index": [7], "length": [1410], "status": ["success"]})
    monkeypatch.setattr(sources.s3, "get_bytes",
                        lambda key, bucket=None: b"tasks" if key.endswith("tasks.parquet") else b"eps")
    monkeypatch.setattr(lerobot, "read_table", lambda raw: tasks_tbl if raw == b"tasks" else eps_tbl)
    m = src._meta("plate")
    assert m["episodes"][7]["length"] == 1410
    assert ("derived/" + _blob(src, "plate")) in remote.objects


def _indexed(src, remote, tasks=("plate",)):
    idx = {"v": 1, "built_at": time.time(), "source": src.id,
           "tasks": [{"task": t, "ikey": "k", "episodes": 1, "latest": "episode_000007",
                      "cameras": ["scene_camera"], "fps": 30} for t in tasks],
           "stats": [{"task": t, "episode": "episode_000007", "duration_s": 47.0, "status": "success"}
                     for t in tasks]}
    remote.objects["derived/" + src._source_blob()] = json.dumps(idx).encode()
    return idx


def test_source_index_answers_tasks_overview_and_stats_without_listing(remote, monkeypatch):
    src = sources.LeRobotSource(SPEC)
    _indexed(src, remote, ("plate", "banana"))
    monkeypatch.setattr(sources.s3, "list_dirs", lambda *a, **k: pytest.fail("listed S3"))
    monkeypatch.setattr(sources.s3, "try_head", lambda *a, **k: pytest.fail("HEAD"))
    assert src.list_tasks() == ["plate", "banana"]
    ov = src.overview()
    assert ov["num_tasks"] == 2 and ov["num_episodes"] == 2
    st = src.stats()
    assert st["total_episodes"] == 2 and st["sampled"] is False and st["episodes"][0]["status"] == "success"


def test_stale_source_index_is_served_and_refreshed_in_background(remote, monkeypatch):
    src = sources.LeRobotSource(SPEC)
    idx = _indexed(src, remote)
    idx["built_at"] = time.time() - config.SOURCE_INDEX_TTL_S - 1
    remote.objects["derived/" + src._source_blob()] = json.dumps(idx).encode()
    started = []
    monkeypatch.setattr(sources.threading, "Thread",
                        lambda target=None, **kw: started.append(target) or type("T", (), {"start": lambda self: None})())
    assert src.list_tasks() == ["plate"]              # stale answer, immediately
    assert started, "no background refresh was started"


def test_clip_manifest_names_the_clip_with_zero_network(remote, monkeypatch):
    src = sources.LeRobotSource(SPEC)
    src._meta_cache["plate"] = {"info": INFO, "tasks": {}, "episodes": {7: ROW}, "ikey": "k"}
    remote.objects["derived/" + src._clips_blob("k")] = json.dumps(
        {"v": 1, "clips": {"episode_000007|scene_camera": "lerobot_e_scene_camera_0.000-47.000.mp4"}}).encode()
    monkeypatch.setattr(sources.s3, "try_head", lambda *a, **k: pytest.fail("HEAD on the source"))
    heads = []
    monkeypatch.setattr(remote, "head_object", lambda **kw: heads.append(kw))
    name = src.video_cache_name("plate", "episode_000007", "scene_camera", "left")
    assert name == "lerobot_e_scene_camera_0.000-47.000.mp4"
    assert cache.remote_ready(name) and heads == []      # manifest primed the HEAD cache


def test_episode_detail_is_read_from_the_derived_json(remote, monkeypatch):
    src = sources.LeRobotSource(SPEC)
    src._meta_cache["plate"] = {"info": INFO, "tasks": {}, "episodes": {7: ROW}, "ikey": "k"}
    detail = {"source": "yam_sim", "task": "plate", "episode": "episode_000007", "status": "success",
              "robot": {"keys": ["j0"]}, "cameras": [], "annotations": [], "instruction": "x",
              "metadata": {}, "calibration": None}
    remote.objects["derived/" + src._detail_blob("k", 7)] = json.dumps(detail).encode()
    monkeypatch.setattr(src, "_data_table", lambda *a: pytest.fail("data parquet was downloaded"))
    assert src.episode_detail("plate", "episode_000007") == detail


def test_episode_detail_miss_is_computed_and_published(remote, monkeypatch):
    src = sources.LeRobotSource(SPEC)
    src._meta_cache["plate"] = {"info": INFO, "tasks": {}, "episodes": {7: ROW}, "ikey": "k"}
    monkeypatch.setattr(src, "_data_table", lambda *a: None)
    monkeypatch.setattr(lerobot, "build_robot", lambda tbl, info: {"keys": []})
    monkeypatch.setattr(lerobot, "subtasks_to_annotations", lambda tbl: [])
    monkeypatch.setattr(lerobot, "instruction_for", lambda tbl, tasks, r: "put the plate")
    d = src.episode_detail("plate", "episode_000007")
    assert d["status"] == "success" and d["instruction"] == "put the plate"
    assert ("derived/" + src._detail_blob("k", 7)) in remote.objects


def test_poster_endpoint_redirects_to_the_derived_jpeg(remote, monkeypatch):
    from fastapi.testclient import TestClient
    from raiden_viz import app as app_module

    class Src:
        spec = SPEC
        def poster_path(self, task, episode, camera, eye):
            return cache.path_for("lerobot_e_scene_camera_0.000-47.000.jpg")

    remote.objects["derived/lerobot_e_scene_camera_0.000-47.000.jpg"] = b"jpg"
    monkeypatch.setattr(app_module, "_src", lambda sid: Src())
    r = TestClient(app_module.app).get(
        "/api/sources/yam_sim/tasks/plate/episodes/episode_000007/video/poster?camera=scene_camera",
        follow_redirects=False)
    assert r.status_code == 302 and "scene_camera_0.000-47.000.jpg" in r.headers["location"]


def test_warm_endpoint_runs_warm_task_in_the_background(monkeypatch):
    from fastapi.testclient import TestClient
    from raiden_viz import app as app_module

    class Src:
        spec = SPEC
        def warm_task(self, task, workers=3, progress=None, posters=True, details=True):
            progress(1, 2, None); progress(2, 2, ("e", "c", "boom"))
            return {"task": task, "episodes": 1, "clips": 1, "ok": 1, "failed": [("e", "c", "boom")]}

    monkeypatch.setattr(app_module, "_WARMS", {})
    monkeypatch.setattr(app_module, "_src", lambda sid: Src())
    c = TestClient(app_module.app)
    assert c.post("/api/sources/yam_sim/tasks/plate/warm").status_code == 200
    for _ in range(200):
        st = c.get("/api/sources/yam_sim/tasks/plate/warm").json()
        if st["done"]:
            break
        time.sleep(0.01)
    assert st["completed"] == 2 and st["failed"] == 1 and st["result"]["failed_n"] == 1


def test_warm_task_leaves_every_artifact(remote, monkeypatch):
    """One episode, one camera: detail JSON, clip, poster, manifest and source index
    all land in the derived tier; a second run produces nothing."""
    import pyarrow as pa
    src = sources.LeRobotSource(SPEC)
    src._meta_cache["plate"] = {"info": INFO, "tasks": {}, "episodes": {7: ROW}, "ikey": "k"}
    monkeypatch.setattr(src, "_list_tasks_raw", lambda: ["plate"])
    monkeypatch.setattr(sources.s3, "get_bytes", lambda key, bucket=None: b"data")
    monkeypatch.setattr(lerobot, "read_table", lambda raw: pa.table({"episode_index": [7]}))
    monkeypatch.setattr(lerobot, "build_robot", lambda tbl, info: {"keys": []})
    monkeypatch.setattr(lerobot, "subtasks_to_annotations", lambda tbl: [])
    monkeypatch.setattr(lerobot, "instruction_for", lambda tbl, tasks, r: "x")
    produced = []
    monkeypatch.setattr(src, "video_path", lambda t, e, c, eye: produced.append("clip") or cache.path_for("lerobot_e_c.mp4"))
    monkeypatch.setattr(src, "video_cache_name", lambda t, e, c, eye: "lerobot_e_c.mp4")
    monkeypatch.setattr(lerobot, "poster", lambda s, d: produced.append("poster") or Path(d).write_bytes(b"jpg"))
    cache.path_for("lerobot_e_c.mp4").write_bytes(b"mp4")
    res = src.warm_task("plate", workers=1)
    assert res["failed"] == [] and res["ok"] == 2
    keys = set(remote.objects)
    assert "derived/" + src._detail_blob("k", 7) in keys
    assert "derived/lerobot_e_c.jpg" in keys
    assert "derived/" + src._clips_blob("k") in keys
    assert "derived/" + src._source_blob() in keys
    assert produced == ["clip", "poster"]
    res2 = src.warm_task("plate", workers=1)
    assert res2["failed"] == [] and produced == ["clip", "poster", "clip"]   # video_path is the cheap derived check
