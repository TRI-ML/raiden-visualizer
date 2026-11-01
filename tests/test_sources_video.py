"""Which video artifacts earn a place in the durable derived tier.

The tier is for things expensive to PRODUCE. A decoded MP4 qualifies: minutes of
ffmpeg over a large download. The .svo2 it was decoded from does not — it is a
byte-for-byte copy of an object that already lives in the datasets bucket, pulled
only as the transcode's input. Publishing it duplicated ~2.8 GB of source data into
the derived bucket, about a third of its contents, to buy back a re-download in the
one case where the MP4 expired but its source did not.

The yam adapter already treats its MCAP this way: a .tmp file removed in a finally,
never published.
"""

from pathlib import Path

import pytest

from raiden_viz import sources


@pytest.fixture
def raiden():
    return sources.RaidenSource(
        {"id": "raiden", "label": "Raiden", "kind": "raiden",
         "bucket": "tri-ml-datasets-uw2", "prefix": "raiden_datasets/raw"}
    )


@pytest.fixture
def recorded(monkeypatch):
    """Record every (cache_name, remote) the adapter asks for, producing nothing."""
    calls = []

    def fake_get_or_create(cache_name, produce, remote=True):
        calls.append((cache_name, remote))
        return Path("/tmp/unused")

    monkeypatch.setattr(sources.cache, "get_or_create", fake_get_or_create)
    monkeypatch.setattr(
        sources.s3, "try_head",
        lambda key, bucket=None: sources.s3.S3Object(key=key, size=5_000_000, etag="ETAG"),
    )
    return calls


def _by_suffix(calls, suffix):
    return [c for c in calls if c[0].endswith(suffix)]


def test_svo2_intermediate_is_kept_local(raiden, recorded):
    raiden.video_path("task", "ep", "cam0", "left")
    svo2 = _by_suffix(recorded, ".svo2")
    assert svo2, "adapter did not fetch an .svo2 at all"
    assert svo2[0][1] is False, "the .svo2 intermediate is still being published"


def test_decoded_mp4_is_still_published(raiden, recorded):
    """The expensive artifact must keep surviving redeploys — that is the whole
    point of the tier, and the reason a warmed episode stays warm for everyone."""
    raiden.video_path("task", "ep", "cam0", "left")
    mp4 = _by_suffix(recorded, ".mp4")
    assert mp4, "adapter did not produce an mp4"
    assert mp4[0][1] is True, "the decoded clip is no longer published"


def test_missing_camera_still_raises(raiden, monkeypatch):
    monkeypatch.setattr(sources.s3, "try_head", lambda key, bucket=None: None)
    with pytest.raises(FileNotFoundError):
        raiden.video_path("task", "ep", "cam0", "left")


def test_stub_file_still_raises(raiden, monkeypatch):
    monkeypatch.setattr(
        sources.s3, "try_head",
        lambda key, bucket=None: sources.s3.S3Object(key=key, size=1500, etag="E"),
    )
    with pytest.raises(ValueError):
        raiden.video_path("task", "ep", "cam0", "left")


def test_raiden_detail_tolerates_a_missing_metadata_json(monkeypatch):
    """Stray recordings (aborted dagger/test runs) have cameras but no metadata.json."""
    from raiden_viz import sources
    src = sources.RaidenSource({"id": "r", "label": "R", "kind": "raiden", "bucket": "b", "prefix": "p"})
    monkeypatch.setattr(sources.s3, "try_head", lambda key, bucket=None: None)
    monkeypatch.setattr(sources.s3, "get_json", lambda key, bucket=None: pytest.fail(f"read {key}"))
    monkeypatch.setattr(sources.s3, "list_files", lambda prefix, bucket=None: [])
    d = src.episode_detail("HangMugs_dagger", "2026-10-19")
    assert d["metadata"] == {} and d["instruction"] is None and d["cameras"] == []


def test_raiden_traces_are_built_once_and_served_from_the_blob(monkeypatch, tmp_path):
    """The npz summary + FK traces are the slow part of a raiden episode page; they are cached by npz etag."""
    from raiden_viz import robot_data, s3, sources
    src = sources.RaidenSource({"id": "r", "label": "R", "kind": "raiden", "bucket": "b", "prefix": "p"})
    npz_obj = s3.S3Object(key="p/T/e/robot_data.npz", size=10, etag="abc")
    monkeypatch.setattr(sources.s3, "try_head", lambda key, bucket=None: npz_obj if key.endswith("robot_data.npz") else None)
    monkeypatch.setattr(sources.s3, "list_files", lambda prefix, bucket=None: [])
    monkeypatch.setattr(sources.cache, "get_or_create", lambda name, produce, **kw: tmp_path / "x.npz")
    calls = []
    monkeypatch.setattr(robot_data, "summarize", lambda npz: calls.append("summarize") or {"keys": []})
    monkeypatch.setattr(sources, "_ee_traces", lambda npz, calib: None)
    store = {}
    monkeypatch.setattr(sources.cache, "get_json", lambda name, remote=False: store.get(name))
    monkeypatch.setattr(sources.cache, "put_json", lambda name, value, remote=False: store.__setitem__(name, value))
    d1 = src.episode_detail("T", "e")
    d2 = src.episode_detail("T", "e")
    assert d1["robot"] == {"keys": []} and d2["robot"] == {"keys": []}
    assert calls == ["summarize"] and list(store) == ["raiden_traces_abc_nocalib_v1.json"]


def test_corrupt_svo2_is_a_value_error_not_a_crash(tmp_path):
    """A recording cut off mid-write (BimanualTest/0000) must reach the UI as 422 'no video', not a 500."""
    from raiden_viz import svo
    bad = tmp_path / "scene_camera.svo2"
    bad.write_bytes(b"\x89MCAP0\r\n" + b"\x18" + (2**40).to_bytes(8, "little") + b"garbage" * 64)
    with pytest.raises(ValueError, match="corrupt .svo2"):
        svo._extract_h264(bad, tmp_path / "out.h264")


def test_truncated_svo2_keeps_the_frames_before_the_cut(tmp_path, monkeypatch):
    """The MCAP reader surfaces a cut-off record as struct.error: frames read so far are kept; none at all → 422."""
    import struct

    from raiden_viz import svo

    def cut_after(n):
        def inner(svo_path, out):
            if n:
                svo._FRAMES_SO_FAR[out] = n
            raise struct.error("unpack requires a buffer of 8 bytes")
        return inner
    monkeypatch.setattr(svo, "_extract_h264_inner", cut_after(640))
    assert svo._extract_h264(tmp_path / "x.svo2", tmp_path / "x.h264") == 640
    monkeypatch.setattr(svo, "_extract_h264_inner", cut_after(0))
    (tmp_path / "y.h264").write_bytes(b"")
    with pytest.raises(ValueError, match="corrupt .svo2"):
        svo._extract_h264(tmp_path / "x.svo2", tmp_path / "y.h264")
    assert not (tmp_path / "y.h264").exists()    # the empty scratch file is removed


def test_yam_detail_reads_the_mined_blob_without_touching_the_mcap(monkeypatch):
    """First open of a featured episode must not re-download the MCAP: the mined detail is a derived-tier blob."""
    from raiden_viz import s3, sources
    src = sources.YamMcapSource({"id": "yam", "label": "Y", "kind": "yam", "bucket": "b", "prefix": "p"})
    obj = s3.S3Object(key="p/T/e/output.mcap", size=500_000_000, etag="e1")
    monkeypatch.setattr(src, "_head", lambda task, ep: obj)
    monkeypatch.setattr(src, "_sidecar", lambda ep: {})
    monkeypatch.setattr(src, "_mine", lambda *a, **k: pytest.fail("mined the MCAP although the blob exists"))
    blob = {"etag": "e1", "cameras": ["cam_high"], "instruction": "fold", "annotations": [], "robot": None}
    monkeypatch.setattr(sources.cache, "get_json", lambda name, remote=False: blob if name == "yam_e1_meta.json" and remote else None)
    d = src.episode_detail("T", "e")
    assert d["cameras"] == [{"name": "cam_high", "has_video": True, "eyes": ["left"]}]
    assert d["instruction"] == "fold"


def test_featured_warm_builds_the_detail_of_every_featured_episode():
    from raiden_viz import warm

    class Src:
        def featured_episodes(self, task):
            return ["e1", "e2"]
        def __init__(self):
            self.detail_calls, self.clip_calls = [], []
        def episode_detail(self, task, ep):
            self.detail_calls.append((task, ep))
            return {"cameras": [{"name": "cam_high", "has_video": True}]}
        def video_path(self, task, ep, cam, eye):
            self.clip_calls.append((task, ep, cam, eye))
        def poster_path(self, *a):
            return None
    src = Src()
    assert warm.warm_featured(src, ["T"], workers=1) == 0
    assert sorted(src.detail_calls) == [("T", "e1"), ("T", "e2")]
    assert sorted(src.clip_calls) == [("T", "e1", "cam_high", "left"), ("T", "e2", "cam_high", "left")]
