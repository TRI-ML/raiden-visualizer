"""LeRobot adapter: per-task ``subdir`` resolution and the stream-copy fast path.

yam_sim keeps each dataset at ``sim_datasets/<name>/lerobot/`` (the task folder also
holds the source h5 and the dataset card), exported as H.264 4:2:0 with one episode per
video file. Two consequences the adapter has to get right: task folders without the
subdir are not tasks, and a clip that IS its file must be stream-copied, never decoded.
"""

from pathlib import Path

import pytest

from raiden_viz import config, lerobot, sources

SPEC = {"id": "yam_sim", "label": "YAM Sim", "kind": "lerobot",
        "bucket": "tri-yam", "prefix": "sim_datasets", "subdir": "lerobot"}
FLAT = {"id": "we", "label": "WorldEngine", "kind": "lerobot",
        "bucket": "b", "prefix": "yam_public/WorldEngine"}


# ---- subdir -------------------------------------------------------------------

def test_task_root_with_and_without_subdir():
    assert sources.LeRobotSource(SPEC)._task_root("plate") == "sim_datasets/plate/lerobot"
    assert sources.LeRobotSource(FLAT)._task_root("plate") == "yam_public/WorldEngine/plate"


def test_subdir_is_normalised():
    s = sources.LeRobotSource({**SPEC, "subdir": "/lerobot/"})
    assert s._task_root("t") == "sim_datasets/t/lerobot"
    assert sources.LeRobotSource({**SPEC, "subdir": None}).subdir == ""


def test_list_tasks_skips_folders_without_a_dataset(monkeypatch):
    """Part-file folders keep only their card; they must not show up as tasks."""
    monkeypatch.setattr(sources.s3, "list_dirs",
                        lambda prefix, bucket=None: ["front2_tbl", "front2_tbl_a", "front2_tbl_b"])
    present = {"sim_datasets/front2_tbl/lerobot/meta/info.json"}
    heads = []

    def try_head(key, bucket=None):
        heads.append((key, bucket))
        return sources.s3.S3Object(key, 1, "e") if key in present else None

    monkeypatch.setattr(sources.s3, "try_head", try_head)
    assert sources.LeRobotSource(SPEC).list_tasks() == ["front2_tbl"]
    assert all(b == "tri-yam" for _, b in heads)


def test_list_tasks_without_subdir_does_not_probe(monkeypatch):
    monkeypatch.setattr(sources.s3, "list_dirs", lambda prefix, bucket=None: ["a", "b"])
    monkeypatch.setattr(sources.s3, "try_head",
                        lambda *a, **k: pytest.fail("flat layout must not HEAD per task"))
    assert sources.LeRobotSource(FLAT).list_tasks() == ["a", "b"]


def test_yam_sim_source_reads_tri_yam():
    spec = next(s for s in config.SOURCES if s["id"] == "yam_sim")
    assert spec["kind"] == "lerobot"
    assert (spec["bucket"], spec["prefix"], spec["subdir"]) == ("tri-yam", "sim_datasets", "lerobot")


# ---- fast-path decision ---------------------------------------------------------

@pytest.mark.parametrize("info,ok", [
    ({"codec": "h264", "pix_fmt": "yuv420p"}, True),
    ({"codec": "h264", "pix_fmt": "yuvj420p"}, True),
    ({"codec": "h264", "pix_fmt": "yuv444p"}, False),   # browsers won't decode 4:4:4
    ({"codec": "av1", "pix_fmt": "yuv420p"}, False),
    ({"codec": None, "pix_fmt": None}, False),
])
def test_browser_playable(info, ok):
    assert lerobot.browser_playable(info) is ok


@pytest.mark.parametrize("from_ts,to_ts,duration,fps,ok", [
    (0.0, 47.0, 47.0, 30, True),           # exactly the file
    (0.0, 46.98, 47.0, 30, True),          # within a frame of the end
    (0.0, None, 47.0, 30, True),           # open-ended window = whole file
    (0.0, 30.0, 47.0, 30, False),          # a slice: would snap to keyframes
    (12.5, 47.0, 47.0, 30, False),         # does not start at 0
    (0.0, 47.0, None, 30, False),          # duration unknown: cannot prove coverage
    (0.0, 47.0, 47.0, None, True),         # no fps: 50 ms tolerance still passes exact
])
def test_covers_whole_file(from_ts, to_ts, duration, fps, ok):
    assert lerobot.covers_whole_file(from_ts, to_ts, duration, fps) is ok


def test_can_stream_copy_combines_both_conditions(monkeypatch):
    src = sources.LeRobotSource(SPEC)
    monkeypatch.setattr(lerobot, "probe",
                        lambda p: {"codec": "h264", "pix_fmt": "yuv420p", "duration": 47.0})
    assert src._can_stream_copy(Path("x.mp4"), 0.0, 47.0, 30) is True
    assert src._can_stream_copy(Path("x.mp4"), 0.0, 20.0, 30) is False
    monkeypatch.setattr(lerobot, "probe",
                        lambda p: {"codec": "av1", "pix_fmt": "yuv420p", "duration": 47.0})
    assert src._can_stream_copy(Path("x.mp4"), 0.0, 47.0, 30) is False


def test_unprobeable_file_falls_back_to_transcode(monkeypatch):
    def boom(p):
        raise RuntimeError("ffprobe failed")
    monkeypatch.setattr(lerobot, "probe", boom)
    assert sources.LeRobotSource(SPEC)._can_stream_copy(Path("x.mp4"), 0.0, 47.0, 30) is False


# ---- video_path end to end (S3 + ffmpeg doubled) ------------------------------------

@pytest.fixture
def wired(monkeypatch):
    """A one-episode dataset whose meta is preloaded; records which producer ran."""
    src = sources.LeRobotSource(SPEC)
    info = {"fps": 30, "cameras": ["scene_camera"],
            "video_keys": {"scene_camera": "observation.images.scene_camera"},
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"}
    row = {"episode_index": 7, "length": 1410, "data_chunk": 0, "data_file": 0, "tasks": None,
           "videos": {"scene_camera": {"chunk": 0, "file": 7, "from_ts": 0.0, "to_ts": 47.0}}}
    src._meta_cache["plate"] = {"info": info, "tasks": {}, "episodes": {7: row}}

    seen = {"head": None, "download": None, "remux": 0, "transcode": 0, "cache_name": None}
    monkeypatch.setattr(sources.s3, "try_head",
                        lambda key, bucket=None: seen.__setitem__("head", (key, bucket))
                        or sources.s3.S3Object(key, 1_000_000, "etag7"))
    monkeypatch.setattr(sources.s3, "download",
                        lambda key, dest, bucket=None: seen.__setitem__("download", key)
                        or Path(dest).write_bytes(b"mp4"))
    monkeypatch.setattr(sources.cache, "evict", lambda headroom_gb=0.0: None)
    monkeypatch.setattr(lerobot, "remux",
                        lambda s, d: seen.__setitem__("remux", seen["remux"] + 1) or Path(d).write_bytes(b"x"))
    monkeypatch.setattr(lerobot, "transcode",
                        lambda s, d, *a: seen.__setitem__("transcode", seen["transcode"] + 1)
                        or Path(d).write_bytes(b"x"))

    def get_or_create(cache_name, produce, remote=True):
        seen["cache_name"] = cache_name
        out = sources.cache.path_for(cache_name)
        produce(out)
        return out

    monkeypatch.setattr(sources.cache, "get_or_create", get_or_create)
    return src, row, seen


def test_whole_file_h264_is_stream_copied(wired, monkeypatch, tmp_path):
    src, row, seen = wired
    monkeypatch.setattr(sources.cache, "path_for", lambda name: tmp_path / name)
    monkeypatch.setattr(lerobot, "probe",
                        lambda p: {"codec": "h264", "pix_fmt": "yuv420p", "duration": 47.0})
    out = src.video_path("plate", "episode_000007", "scene_camera", "left")
    # resolved under the subdir, in the source bucket
    assert seen["head"] == (
        "sim_datasets/plate/lerobot/videos/observation.images.scene_camera/chunk-000/file-007.mp4", "tri-yam")
    assert seen["download"] == seen["head"][0]
    assert (seen["remux"], seen["transcode"]) == (1, 0)
    assert out.name == "lerobot_etag7_scene_camera_0.000-47.000.mp4"
    assert not list(tmp_path.glob("*.src.mp4.tmp*")), "source download must not be kept"


def test_av1_or_packed_windows_still_transcode(wired, monkeypatch, tmp_path):
    src, row, seen = wired
    monkeypatch.setattr(sources.cache, "path_for", lambda name: tmp_path / name)
    monkeypatch.setattr(lerobot, "probe",
                        lambda p: {"codec": "av1", "pix_fmt": "yuv420p", "duration": 47.0})
    src.video_path("plate", "episode_000007", "scene_camera", "left")
    assert (seen["remux"], seen["transcode"]) == (0, 1)
    # h264 but this episode is a slice of a packed file
    row["videos"]["scene_camera"].update({"from_ts": 10.0, "to_ts": 20.0})
    monkeypatch.setattr(lerobot, "probe",
                        lambda p: {"codec": "h264", "pix_fmt": "yuv420p", "duration": 470.0})
    src.video_path("plate", "episode_000007", "scene_camera", "left")
    assert (seen["remux"], seen["transcode"]) == (0, 2)


def test_warm_touches_every_episode_camera_and_reports_failures(wired, monkeypatch):
    src, row, seen = wired
    src._meta_cache["plate"]["episodes"][8] = {**row, "episode_index": 8}
    calls = []

    def video_path(task, ep, cam, eye):
        calls.append((ep, cam))
        if ep.endswith("8"):
            raise FileNotFoundError("video not found")
        return Path("/tmp/x")

    monkeypatch.setattr(src, "video_path", video_path)
    res = src.warm("plate", workers=2)
    assert sorted(calls) == [("episode_000007", "scene_camera"), ("episode_000008", "scene_camera")]
    assert res["clips"] == 2 and res["ok"] == 1
    assert res["failed"][0][:2] == ("episode_000008", "scene_camera")


def test_remux_and_probe_against_real_ffmpeg(tmp_path):
    """One real 1 s H.264 file: probe reads it back and remux keeps codec + duration."""
    import shutil
    import subprocess
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        pytest.skip("ffmpeg not installed")
    src = tmp_path / "src.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=red:s=64x64:r=30:d=1",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src)], check=True)
    info = lerobot.probe(src)
    assert lerobot.browser_playable(info) and abs(info["duration"] - 1.0) < 0.1
    dst = tmp_path / "dst.mp4"
    lerobot.remux(src, dst)
    again = lerobot.probe(dst)
    assert again["codec"] == "h264" and abs(again["duration"] - info["duration"]) < 0.05
    assert lerobot.covers_whole_file(0.0, 1.0, again["duration"], 30)


# ---- per-episode status -----------------------------------------------------------

def _episodes_table(with_status):
    import pyarrow as pa
    cols = {"episode_index": [0, 1], "length": [10, 20],
            "data/chunk_index": [0, 0], "data/file_index": [0, 0],
            "videos/observation.images.scene_camera/chunk_index": [0, 0],
            "videos/observation.images.scene_camera/file_index": [0, 1],
            "videos/observation.images.scene_camera/from_timestamp": [0.0, 0.0],
            "videos/observation.images.scene_camera/to_timestamp": [0.33, 0.66]}
    if with_status:
        cols["status"] = ["success", "failure"]
    return pa.table(cols)


def test_parse_episodes_reads_the_optional_status_column():
    keys = {"scene_camera": "observation.images.scene_camera"}
    rows = lerobot.parse_episodes(_episodes_table(True), keys)
    assert [rows[i]["status"] for i in (0, 1)] == ["success", "failure"]
    rows = lerobot.parse_episodes(_episodes_table(False), keys)
    assert [rows[i]["status"] for i in (0, 1)] == [None, None]


def test_status_flows_to_facts_detail_and_stat(wired, monkeypatch):
    src, row, _ = wired
    row["status"] = "success"
    assert src.episode_facts("plate") == {"episode_000007": {"timestamp": None, "status": "success"}}
    assert src.episode_stat("plate", "episode_000007")["status"] == "success"
    # episode_detail needs the data parquet; stub the table-derived pieces.
    monkeypatch.setattr(src, "_data_table", lambda task, meta, r: None)
    monkeypatch.setattr(lerobot, "build_robot", lambda tbl, info: {})
    monkeypatch.setattr(lerobot, "subtasks_to_annotations", lambda tbl: [])
    monkeypatch.setattr(lerobot, "instruction_for", lambda tbl, tasks, r: "put the plate in the rack")
    assert src.episode_detail("plate", "episode_000007")["status"] == "success"


def test_single_root_facts_only_cover_the_tasks_episodes():
    src = sources.LeRobotSingleRootSource({"id": "we", "label": "WE", "kind": "lerobot_single",
                                           "bucket": "b", "prefix": "p"})
    src._meta_cache["__root__"] = {"info": {}, "tasks": {}, "by_task": {"a": [0], "b": [1]},
                           "episodes": {0: {"status": "success"}, 1: {"status": None}}}
    assert src.episode_facts("a") == {"episode_000000": {"timestamp": None, "status": "success"}}
    assert src.episode_facts("b") == {"episode_000001": {"timestamp": None, "status": None}}


def test_overview_page_has_the_preview_block():
    root = Path(__file__).resolve().parents[1] / "static"
    html = (root / "index.html").read_text()
    js = (root / "app.js").read_text()
    assert 'id="preview-body"' in html and html.index('id="hist-canvas"') < html.index('id="preview-body"')
    assert "async function waitForClip" in js and "renderPreview(stats.episodes" in js and "id=\"preview-task\"" in html
    assert '"Loading…"' in js
