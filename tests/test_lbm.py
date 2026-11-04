"""LBM spartan source: real-only discovery, range-GET low-dim reads, first-frame inflate, adapter output.

A synthetic observations.npz (deflate zip with one tiny camera member + depth + low-dim tail) and actions.npz
served from an in-memory S3 double; the test asserts the state panel never fetches the camera bytes."""
import io
import zipfile

import numpy as np
import pytest

from raiden_viz import lbm, sources
from raiden_viz.s3 import S3Object

T, H, W = 5, 16, 16
CUTOFF = 2000   # fixture-sized stand-in for lbm.LOWDIM_MAX_BYTES: the camera member (3.8 KB) is above it
BASE = "efs/data/tasks/AssembleBaconSandwich/hersey/real/bc/teleop/2024-01-11T12-52-02-05-00"
EP = f"{BASE}/diffusion_spartan/episode_0/processed"


def _npy(arr) -> bytes:
    b = io.BytesIO(); np.save(b, arr); return b.getvalue()


def _actions() -> bytes:
    b = io.BytesIO(); np.savez(b, actions=np.arange(T * 20, dtype=np.float64).reshape(T, 20)); return b.getvalue()


def _observations() -> bytes:
    rng = np.random.default_rng(0)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("6CD146030E9B.npy", _npy(rng.integers(0, 255, (T, H, W, 3), dtype=np.uint8)))
        z.writestr("6CD146030E9B_depth.npy", _npy(rng.integers(0, 2000, (T, H, W), dtype=np.uint16)))
        for arm in ("left", "right"):
            z.writestr(f"robot__actual__poses__{arm}::panda__xyz.npy", _npy(rng.normal(size=(T, 3))))
            z.writestr(f"robot__actual__poses__{arm}::panda__rot_6d.npy", _npy(rng.normal(size=(T, 6))))
            z.writestr(f"robot__actual__grippers__{arm}::panda_hand.npy", _npy(rng.uniform(0, 0.08, (T, 1))))
            z.writestr(f"robot__actual__joint_position__{arm}::panda.npy", _npy(rng.normal(size=(T, 7))))
    return buf.getvalue()


@pytest.fixture
def bucket(monkeypatch):
    """Objects of one LBM task: a real teleop episode, a sim episode, a config-only stub session."""
    objs = {
        f"{EP}/observations.npz": _observations(),
        f"{EP}/actions.npz": _actions(),
        f"{EP}/images_6CD146030E9B/camera_info.yaml": b"image_height: 4\nimage_width: 6\n",
        f"{BASE}/demonstration_0/summary.yaml": b"last_step_reward: 0.0\nlast_t: 0.5\nseed: 100\nsuccess: true\n",
        f"{BASE}/cmd.txt": b"run",
        # sim branch: never an episode
        "efs/data/tasks/AssembleBaconSandwich/hersey/sim/bc/teleop/2024-01-12T00-00-00-05-00/diffusion_spartan/episode_0/processed/observations.npz": b"x",
        # config-only stub session (what AddBeansToSalad looks like)
        "efs/data/tasks/AssembleBaconSandwich/riverway/real/bc/teleop/2023-07-06T14-38-44-04-00/cmd.txt": b"run",
    }
    fetched = {"bytes": 0}

    def get_range(key, start, end, bucket=None):
        fetched["bytes"] += end - start + 1
        return objs[key][start:end + 1]

    def list_keys(prefix, bucket=None, suffix=None):
        prefix = prefix.rstrip("/") + "/"
        return [S3Object(key=k, size=len(v), etag="e", last_modified="2024-01-11T20:00:00+00:00")
                for k, v in objs.items() if k.startswith(prefix) and (not suffix or k.endswith(suffix))]

    for mod in (lbm, sources):
        monkeypatch.setattr(mod.s3, "get_range", get_range)
        monkeypatch.setattr(mod.s3, "list_keys", list_keys)
        monkeypatch.setattr(mod.s3, "get_bytes", lambda key, bucket=None: objs[key])
        monkeypatch.setattr(mod.s3, "list_dirs", lambda prefix, bucket=None: ["hersey", "riverway"] if prefix.endswith("AssembleBaconSandwich") else ["AssembleBaconSandwich"])
    return objs, fetched


def test_discover_keeps_real_teleop_episodes_only(bucket):
    objs, _ = bucket
    eps = lbm.discover([S3Object(key=k, size=len(v), etag="e") for k, v in objs.items()])
    assert list(eps) == ["hersey_2024-01-11T12-52-02-05-00_episode_0"]
    e = eps["hersey_2024-01-11T12-52-02-05-00_episode_0"]
    assert e["cameras"] == ["6CD146030E9B"] and e["station"] == "hersey" and e["n"] == 0
    assert lbm.session_time(e["session"]) == "2024-01-11T12:52:02-05:00"


def test_lowdim_reads_only_the_small_members(bucket):
    objs, _ = bucket
    key = f"{EP}/observations.npz"
    arrs, nbytes = lbm.lowdim(key, "b", len(objs[key]), max_bytes=CUTOFF)
    assert "robot__actual__poses__left::panda__xyz" in arrs and arrs["robot__actual__joint_position__right::panda"].shape == (T, 7)
    assert "6CD146030E9B" not in arrs and "6CD146030E9B_depth" not in arrs
    assert lbm.cameras_of(lbm.open_npz(key, "b", len(objs[key]))[0], max_bytes=CUTOFF) == ["6CD146030E9B"]
    assert nbytes < len(objs[key]) / 2      # central directory + tail members, never the camera bytes


def test_first_frame_partial_inflate(bucket):
    objs, _ = bucket
    key = f"{EP}/observations.npz"
    frame = lbm.first_frame(key, "b", len(objs[key]), "6CD146030E9B")
    assert frame.shape == (H, W, 3) and frame.dtype == np.uint8
    full = np.load(io.BytesIO(zipfile.ZipFile(io.BytesIO(objs[key])).read("6CD146030E9B.npy")))
    assert np.array_equal(frame, full[0])


def test_source_detail_and_stat(bucket, cache_dir, monkeypatch):
    monkeypatch.setattr(sources.cache, "get_json", lambda name, remote=False: None)
    monkeypatch.setattr(sources.cache, "put_json", lambda name, value, remote=False: None)
    src = sources.LbmSource({"id": "lbm_real", "label": "LBM", "kind": "lbm", "bucket": "robotics-manip-lbm",
                             "prefix": "efs/data/tasks", "tasks": ["AssembleBaconSandwich"]})
    assert src.list_tasks() == ["AssembleBaconSandwich"]
    eps = src.list_episodes("AssembleBaconSandwich")
    assert eps == ["hersey_2024-01-11T12-52-02-05-00_episode_0"]
    d = src.episode_detail("AssembleBaconSandwich", eps[0])
    assert d["status"] == "success" and d["metadata"]["station"] == "hersey" and d["metadata"]["control_hz"] == 10.0
    assert d["robot"]["signals"]["action"]["dims"] == 20 and d["robot"]["summary"]["num_steps"] == T
    assert [c["name"] for c in d["cameras"]] == ["6CD146030E9B"]
    assert d["instruction"] == "assemble bacon sandwich"
    assert src._stat_cache_key("AssembleBaconSandwich", eps[0]) == "stat_lbm_real_e.json"   # the scan keys by npz etag
    st = src.episode_stat("AssembleBaconSandwich", eps[0])
    assert st["duration_s"] == 0.5 and st["num_cameras"] == 1 and st["robot_type"] == "dual_franka_panda"
    assert src.episode_facts("AssembleBaconSandwich")[eps[0]]["timestamp"] == "2024-01-11T12:52:02-05:00"
    with pytest.raises(FileNotFoundError):
        src.video_path("AssembleBaconSandwich", eps[0], "nope", "left")


def test_first_frames_partial_inflate_and_clip(bucket, tmp_path):
    objs, _ = bucket
    key = f"{EP}/observations.npz"
    frames = lbm.first_frames(key, "b", len(objs[key]), "6CD146030E9B", 3)
    full = np.load(io.BytesIO(zipfile.ZipFile(io.BytesIO(objs[key])).read("6CD146030E9B.npy")))
    assert frames.shape == (3, H, W, 3) and np.array_equal(frames, full[:3])
    assert lbm.first_frames(key, "b", len(objs[key]), "6CD146030E9B", 99).shape[0] == T   # shorter episode: all frames
    lbm.frames_to_mp4(frames, tmp_path / "head.mp4", fps=10)
    assert (tmp_path / "head.mp4").stat().st_size > 0


def test_preview_builder_for_lbm(bucket, cache_dir, monkeypatch, tmp_path):
    from raiden_viz import previews
    monkeypatch.setattr(sources.cache, "get_json", lambda name, remote=False: None)
    monkeypatch.setattr(sources.cache, "put_json", lambda name, value, remote=False: None)
    monkeypatch.setattr(previews.cache, "remote_ready", lambda name: False)
    published = {}

    def publish(key, cam, render):
        a, b = tmp_path / f"{key}_{cam}.jpg", tmp_path / f"{key}_{cam}.mp4"
        render(a, b)
        published[cam] = (key, a.stat().st_size > 0, b.stat().st_size > 0)
        return (f"pv_{key}_{cam}.jpg", f"pv_{key}_{cam}.mp4")

    monkeypatch.setattr(previews, "publish_assets", publish)
    src = sources.LbmSource({"id": "lbm_real", "label": "LBM", "kind": "lbm", "bucket": "robotics-manip-lbm",
                             "prefix": "efs/data/tasks", "tasks": ["AssembleBaconSandwich"]})
    ep = src.list_episodes("AssembleBaconSandwich")[0]
    out = previews.build_for(src, "AssembleBaconSandwich", ep, tmp_path)
    assert out == {"6CD146030E9B": ("pv_e_6CD146030E9B.jpg", "pv_e_6CD146030E9B.mp4")}   # keyed by the npz etag
    assert published["6CD146030E9B"] == ("e", True, True)
