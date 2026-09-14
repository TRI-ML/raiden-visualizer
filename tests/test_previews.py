"""Small, fast dataset previews (raiden_viz.previews): frames from the HEAD of a raw
MCAP/svo2 file (no full download, no decode), 320 px poster + 5 s clip, a persisted
per-source blob that cards/overview/external pages render from with zero API chain.
"""

import io
import json
import struct
import subprocess
import time
from pathlib import Path

import pytest

from raiden_viz import cache, config, previews, sources, svo, yam


@pytest.fixture
def remote(monkeypatch, fake_s3):
    monkeypatch.setattr(config, "DERIVED_BUCKET", "derived-bucket")
    monkeypatch.setattr(config, "DERIVED_PREFIX", "derived")
    monkeypatch.setattr(cache, "_derived_client", lambda: fake_s3)
    return fake_s3


def _mcap_bytes(topic, schema_name, payloads, encoding="protobuf"):
    from mcap.writer import CompressionType, Writer
    buf = io.BytesIO()
    w = Writer(buf, compression=CompressionType.NONE, chunk_size=64)   # one small chunk per message
    w.start()
    sid = w.register_schema(name=schema_name, encoding="protobuf", data=b"")
    cid = w.register_channel(topic=topic, message_encoding=encoding, schema_id=sid)
    for i, p in enumerate(payloads):
        w.add_message(channel_id=cid, log_time=i * 33_000_000, data=p, publish_time=i * 33_000_000)
    w.finish()
    return buf.getvalue()


def test_head_frames_reads_a_truncated_svo2_prefix(tmp_path):
    frames = [bytes([i]) * 16 for i in range(8)]
    payloads = [svo._HDR.pack(len(f) + 8, len(f)) + f for f in frames]
    raw = _mcap_bytes("/zed/side_by_side", "zed.Frame", payloads)
    head = tmp_path / "cam.head"
    head.write_bytes(raw[: int(len(raw) * 0.8)])          # footer/summary and the last chunks are gone
    got = previews.head_frames(head, previews._svo_want("ego_camera"), previews._svo_extract, max_frames=5)
    assert list(got) == ["ego_camera"]
    assert got["ego_camera"][0] == frames[:5] and got["ego_camera"][1] == "h264"


def test_head_frames_names_yam_cameras_from_topics(tmp_path, monkeypatch):
    monkeypatch.setattr(yam, "_CAMERA_SCHEMA", "foxglove.CompressedVideo", raising=False)
    monkeypatch.setattr(yam, "_cv_fields", lambda buf: (buf, "h264"))
    raw = _mcap_bytes("/top-camera/image-raw", yam._CAMERA_SCHEMA, [b"a", b"b", b"c"])
    head = tmp_path / "o.head"
    head.write_bytes(raw)
    got = previews.head_frames(head, previews._yam_want, yam._cv_fields)
    assert got == {"top_camera": ([b"a", b"b", b"c"], "h264")}


def test_render_assets_are_small(tmp_path):
    src = tmp_path / "in.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=30",
                    "-t", "8", "-pix_fmt", "yuv420p", str(src)], check=True)
    poster, clip = tmp_path / "p.jpg", tmp_path / "c.mp4"
    previews.render_assets(src, poster, clip)
    assert 0 < poster.stat().st_size <= 40_000
    assert 0 < clip.stat().st_size <= 300_000
    probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(clip)],
                                      capture_output=True, check=True).stdout)
    v = probe["streams"][0]
    assert v["codec_name"] == "h264" and v["width"] == previews.POSTER_W
    assert 4.5 <= float(probe["format"]["duration"]) <= 5.5


def test_publish_assets_skips_what_the_tier_holds(remote):
    remote.objects["derived/pv_E_cam.jpg"] = b"j"
    remote.objects["derived/pv_E_cam.mp4"] = b"m"
    assert previews.publish_assets("E", "cam", lambda a, b: pytest.fail("rendered")) == ("pv_E_cam.jpg", "pv_E_cam.mp4")

    def render(pj, pm):
        pj.write_bytes(b"jpg"); pm.write_bytes(b"mp4")
    assert previews.publish_assets("F", "cam", render) == ("pv_F_cam.jpg", "pv_F_cam.mp4")
    assert remote.objects["derived/pv_F_cam.jpg"] == b"jpg"


def test_pick_main_camera_prefers_scene_like_names():
    assert previews.pick_main_camera(["left_wrist", "scene_camera"]) == "scene_camera"
    assert previews.pick_main_camera(["right_wrist_camera", "top_camera"]) == "top_camera"
    assert previews.pick_main_camera(["b", "a"]) == "b"
    assert previews.pick_main_camera([]) is None


class _Src(sources.Source):
    def __init__(self):
        super().__init__({"id": "r", "label": "R", "kind": "raiden", "bucket": "b", "prefix": "p"})
    def list_tasks(self): return ["MakeCoffee", "Empty"]
    def list_episodes(self, task): return ["e0", "e1"] if task == "MakeCoffee" else []
    def _build_overview(self):
        return {"source": "r", "bucket": "b", "prefix": "p", "num_tasks": 2, "num_episodes": 2,
                "stations": [], "tasks": [{"task": "MakeCoffee", "episodes": 2}, {"task": "Empty", "episodes": 0}]}


def test_preview_warm_persists_a_blob_and_overview_carries_it(remote, monkeypatch):
    src = _Src()
    monkeypatch.setattr(previews, "build_for",
                        lambda s, task, ep, wd: {"ego_camera": ("pv_E_ego_camera.jpg", "pv_E_ego_camera.mp4"),
                                                 "wrist": ("pv_E_wrist.jpg", "pv_E_wrist.mp4")})
    res = src.preview_warm()
    assert res == {"tasks": 2, "ok": 2, "failed": []}
    blob = json.loads(remote.objects["derived/" + previews.blob_name("r")])
    e = blob["tasks"]["MakeCoffee"]
    assert e["episode"] == "e0" and e["camera"] == "ego_camera" and e["poster_name"] == "pv_E_ego_camera.jpg"
    assert e["poster_names"]["wrist"] == "pv_E_wrist.jpg"
    assert "Empty" not in blob["tasks"]
    ov = src.overview()
    assert ov["tasks"][0]["preview"]["clip_name"] == "pv_E_ego_camera.mp4" and "preview" not in ov["tasks"][1]
    # a new container reads the blob, no rebuild
    fresh = _Src()
    assert fresh.previews()["MakeCoffee"]["episode"] == "e0"
    # only_missing: nothing rebuilt
    monkeypatch.setattr(previews, "build_for", lambda *a: pytest.fail("rebuilt"))
    assert fresh.preview_warm() == {"tasks": 1, "ok": 1, "failed": []}   # only the empty task is re-listed


def test_catalog_card_prefers_the_small_previews(remote, monkeypatch):
    from raiden_viz import catalog
    src = _Src()
    src._previews_cached = {"MakeCoffee": {"episode": "e0", "camera": "ego_camera", "cameras": ["ego_camera"],
                                           "poster_name": "pv_E_ego_camera.jpg", "clip_name": "pv_E_ego_camera.mp4"}}
    pv = catalog._pick_preview(src, ["ego_camera"])
    assert pv["task"] == "MakeCoffee" and pv["poster_name"] == "pv_E_ego_camera.jpg"


def test_previews_endpoints(remote, monkeypatch):
    from fastapi.testclient import TestClient
    from raiden_viz import app as app_module
    src = _Src()
    monkeypatch.setattr(previews, "build_for",
                        lambda s, task, ep, wd: {"ego_camera": ("pv_E_ego_camera.jpg", "pv_E_ego_camera.mp4")})
    monkeypatch.setattr(app_module, "_WARMS", {})
    monkeypatch.setattr(app_module, "_src", lambda sid: src)
    c = TestClient(app_module.app)
    assert c.post("/api/sources/r/previews/warm").status_code == 200
    for _ in range(300):
        st = c.get("/api/sources/r/previews/warm").json()
        if st["done"]:
            break
        time.sleep(0.01)
    assert st["result"]["ok"] == 2
    assert c.get("/api/sources/r/previews").json()["tasks"]["MakeCoffee"]["poster_name"] == "pv_E_ego_camera.jpg"


def test_frontend_uses_the_precomputed_names_and_lazy_loading():
    js = (Path(__file__).resolve().parents[1] / "static" / "app.js").read_text()
    assert "IntersectionObserver" in js and 'video.preload = "none"' in js
    assert "/api/artifact/" in js and "clip_names" in js
    for forbidden in ("/episodes`", "/video/status"):
        # the preview box must not poll status or list episodes
        box = js[js.index("function previewBox"):js.index("function previewStrip")]
        assert forbidden not in box
