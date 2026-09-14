"""Derived-first clips: a pre-rendered clip is never pulled through the container.

Before: /video/status started a thread whose produce() DOWNLOADED the derived mp4 to
local disk before reporting ready; then /video re-derived the clip (meta + source
HEAD + derived HEAD) and redirected. A warmed first load cost seconds and fought the
boot scan for the CPU. Now the status poll names the clip, HEADs the tier once
(memoized), answers ready, and /video reuses the resolved path.
"""

import time
from pathlib import Path

import pytest

from raiden_viz import cache, clips, config, sources


@pytest.fixture
def remote(monkeypatch, fake_s3):
    monkeypatch.setattr(config, "DERIVED_BUCKET", "derived-bucket")
    monkeypatch.setattr(config, "DERIVED_PREFIX", "derived")
    monkeypatch.setattr(cache, "_derived_client", lambda: fake_s3)
    return fake_s3


def test_fetch_false_returns_the_path_without_downloading(cache_dir, remote):
    remote.objects["derived/clip.mp4"] = b"video"
    calls = []
    out = cache.get_or_create("clip.mp4", lambda dst: calls.append(dst), fetch=False)
    assert out == cache.path_for("clip.mp4") and not out.exists()   # not pulled down
    assert calls == []                                              # not re-decoded
    assert cache.remote_url("clip.mp4")                             # /video can presign


def test_fetch_false_still_produces_on_a_true_miss(cache_dir, remote):
    out = cache.get_or_create("new.mp4", lambda dst: dst.write_bytes(b"fresh"), fetch=False)
    assert out.read_bytes() == b"fresh"
    assert "derived/new.mp4" in remote.objects            # published for the next container


def test_remote_head_is_memoized(cache_dir, remote, monkeypatch):
    remote.objects["derived/clip.mp4"] = b"video"
    n = {"heads": 0}
    real = remote.head_object

    def counting(Bucket, Key):
        n["heads"] += 1
        return real(Bucket=Bucket, Key=Key)

    monkeypatch.setattr(remote, "head_object", counting)
    assert cache.remote_ready("clip.mp4") and cache.remote_ready("clip.mp4")
    assert cache.remote_url("clip.mp4")
    assert n["heads"] == 1


def test_mark_ready_registers_a_finished_job_with_its_path():
    jobs = clips.ClipJobs()
    st = jobs.mark_ready("k", Path("/c/clip.mp4"))
    assert st["ready"] and not st["decoding"] and st["path"] == "/c/clip.mp4"
    # a known job is never overwritten
    assert jobs.mark_ready("k", Path("/other"))["path"] == "/c/clip.mp4"


class _NamedSource:
    """Adapter that can name its clip up front (like LeRobotSource)."""
    def __init__(self):
        self.spec = {"id": "s1", "label": "S", "kind": "lerobot"}
        self.produced = 0

    def video_cache_name(self, task, episode, camera, eye):
        return f"lerobot_etag_{camera}.mp4"

    def video_path(self, task, episode, camera, eye):
        self.produced += 1
        return cache.path_for(f"lerobot_etag_{camera}.mp4")


def test_status_is_ready_on_the_first_poll_for_a_derived_clip(cache_dir, remote, monkeypatch):
    from fastapi.testclient import TestClient
    from raiden_viz import app as app_module

    remote.objects["derived/lerobot_etag_cam0.mp4"] = b"video"
    src = _NamedSource()
    monkeypatch.setattr(app_module, "_CLIPS", clips.ClipJobs())
    monkeypatch.setattr(app_module, "_src", lambda sid: src)
    client = TestClient(app_module.app)
    base = "/api/sources/s1/tasks/t/episodes/e/video"
    r = client.get(f"{base}/status?camera=cam0&eye=left")
    assert r.json() == {"ready": True, "decoding": False}
    assert src.produced == 0                         # no thread, no download
    r = client.get(f"{base}?camera=cam0&eye=left", follow_redirects=False)
    assert r.status_code == 302 and "derived/lerobot_etag_cam0.mp4" in r.headers["location"]
    assert src.produced == 0                         # /video reused the resolved path


def test_status_falls_back_to_a_decode_on_a_miss(cache_dir, remote, monkeypatch):
    from fastapi.testclient import TestClient
    from raiden_viz import app as app_module
    from tests.test_clips import _ImmediateThread

    src = _NamedSource()
    monkeypatch.setattr(app_module.clips.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(app_module, "_CLIPS", clips.ClipJobs())
    monkeypatch.setattr(app_module, "_src", lambda sid: src)
    client = TestClient(app_module.app)
    r = client.get("/api/sources/s1/tasks/t/episodes/e/video/status?camera=cam0&eye=left")
    assert r.status_code == 200 and src.produced == 1


def test_lerobot_names_its_clip_without_producing(monkeypatch):
    src = sources.LeRobotSource({"id": "yam_sim", "label": "x", "kind": "lerobot",
                                 "bucket": "tri-yam", "prefix": "sim_datasets", "subdir": "lerobot"})
    info = {"fps": 30, "cameras": ["scene_camera"],
            "video_keys": {"scene_camera": "observation.images.scene_camera"},
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"}
    row = {"episode_index": 7, "length": 1410, "data_chunk": 0, "data_file": 0, "tasks": None,
           "videos": {"scene_camera": {"chunk": 0, "file": 7, "from_ts": 0.0, "to_ts": 47.0}}}
    src._meta_cache["plate"] = {"info": info, "tasks": {}, "episodes": {7: row}}
    monkeypatch.setattr(sources.s3, "try_head",
                        lambda key, bucket=None: sources.s3.S3Object(key, 1000, "etag7"))
    assert src.video_cache_name("plate", "episode_000007", "scene_camera", "left") \
        == "lerobot_etag7_scene_camera_0.000-47.000.mp4"
    with pytest.raises(FileNotFoundError):
        src.video_cache_name("plate", "episode_000007", "nope", "left")


# ---- boot scan: persisted + throttled -------------------------------------------

class _ScanSource(sources.Source):
    def __init__(self):
        super().__init__({"id": "sc", "label": "S", "kind": "raiden", "bucket": "b", "prefix": "p"})
        self.stats_calls = 0

    def _stat_pairs(self, full=False):
        return [("t", f"e{i}") for i in range(3)], 3

    def _safe_stat(self, task, episode):
        self.stats_calls += 1
        return {"task": task, "episode": episode, "duration_s": 1.0}


@pytest.fixture
def scans(monkeypatch):
    monkeypatch.setattr(sources, "_SCANS", {})


def test_finished_scan_is_persisted_and_restored(cache_dir, remote, scans, monkeypatch):
    src = _ScanSource()
    src.scan_start(workers=1)
    for _ in range(200):
        if src.scan_snapshot()["done"]:
            break
        time.sleep(0.01)
    assert src.stats_calls == 3
    assert any(k.startswith("derived/scan_") for k in remote.objects)

    # a new container: in-memory registry empty, local cache empty
    monkeypatch.setattr(sources, "_SCANS", {})
    for p in cache_dir.iterdir():
        p.unlink()
    fresh = _ScanSource()
    snap = fresh.scan_start()
    assert snap["done"] and snap["scanned"] == 3 and fresh.stats_calls == 0   # restored, not rescanned


def test_stale_persisted_scan_is_rescanned(cache_dir, remote, scans, monkeypatch):
    monkeypatch.setattr(config, "SCAN_PERSIST_TTL_S", 0.0)
    src = _ScanSource()
    blob = src._scan_blob()
    cache.put_json(blob, {"saved_at": time.time() - 10, "total": 3, "episodes": [1, 2, 3]}, remote=True)
    snap = src.scan_start(workers=1)
    assert snap["running"] or src.stats_calls > 0


def test_force_ignores_the_persisted_scan(cache_dir, remote, scans):
    src = _ScanSource()
    cache.put_json(src._scan_blob(), {"saved_at": time.time(), "total": 3, "episodes": []}, remote=True)
    src.scan_start(force=True, workers=1)
    for _ in range(200):
        if src.scan_snapshot()["done"]:
            break
        time.sleep(0.01)
    assert src.stats_calls == 3
