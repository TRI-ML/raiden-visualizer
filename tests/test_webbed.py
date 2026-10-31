"""Webbed (AnyData cv_webbed) source: manifest parsing, shard member parsing, robot summary, featured sampling."""
import io
import json
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import pytest

from raiden_viz import webbed

FIX = Path(__file__).parent / "fixtures" / "webbed_split_all.json"


def test_manifest_parse_and_frame_math():
    m = webbed.parse_manifest(json.loads(FIX.read_text()))
    assert set(m["episodes"]) == {"chunk_000__episode_000000", "chunk_000__episode_000001", "chunk_000__episode_000002"}
    shards = m["episodes"]["chunk_000__episode_000000"]
    assert shards[0] == "0000_0044.tar" and shards[-1] == "0441_0485.tar" and len(shards) == 11
    assert webbed.episode_frames(shards) == 485
    assert webbed.shard_bounds_s(shards)[:2] == [round(44 / 30, 3), round(88 / 30, 3)]
    assert webbed.seq_key("chunk_000__episode_000000") == "chunk_000/episode_000000"
    assert m["cameras"] == ["cam_high", "cam_left_wrist", "cam_right_wrist"] and m["n_files"] == 1661
    with pytest.raises(ValueError):
        webbed.shard_range("notashard.tar")


def _npz_bytes(**arrays) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def test_read_shard_parses_members_by_label(tmp_path):
    """A synthetic shard with the real member names: depth (F,1,H,W) mm, lowdim action/state (F,32) + language."""
    f, h, w = 3, 4, 6
    depth = np.zeros((f, 1, h, w), np.float32); depth[:, 0, 1, 1] = 500.0
    action = np.arange(f * 32, dtype=np.float32).reshape(f, 32)
    lowdim = _npz_bytes(
        timestep=np.arange(f)[:, None] / 30.0,
        action=np.array({"action": action, "state": action + 1}, dtype=object),
        language=np.array({"task": np.array(["pick it up"] * f), "prompt": np.array(["pick it up"] * f)}, dtype=object),
    )
    tar = tmp_path / "0000_0003.tar"
    with tarfile.open(tar, "w") as tf:
        for name, data in (("0_0.depth.(0:3,0).3.npz", _npz_bytes(depth=depth)), ("0_0.lowdim.(0:3,0).3.npz", lowdim),
                           ("0_0.metadata.3.npz", _npz_bytes(metadata=np.array({"length": 48}, dtype=object))),
                           ("0_0.rgb.(0:3,0).3.mp4", b"\x00" * 16)):
            ti = tarfile.TarInfo(name); ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    sh = webbed.read_shard(tar, tmp_path / "out")
    assert sh["rgb"].name == "0_0.rgb.(0:3,0).3.mp4" and sh["rgb"].stat().st_size == 16
    assert sh["depth"].shape == (f, h, w) and sh["depth"][0, 1, 1] == 500.0
    assert sh["lowdim"]["action"].shape == (f, 32) and sh["lowdim"]["state"][0, 0] == 1.0
    assert sh["lowdim"]["language"] == "pick it up"
    rs = webbed.robot_summary([sh["lowdim"], sh["lowdim"]])
    assert rs["summary"]["num_steps"] == 6 and rs["summary"]["hz"] == 30 and rs["signals"]["action"]["dims"] == 32
    assert rs["time"][-1] == round(5 / 30, 3)
    assert webbed.robot_summary([None, {"action": None}]) is None


@pytest.mark.skipif(subprocess.run(["which", "ffmpeg"], capture_output=True).returncode != 0, reason="no ffmpeg")
def test_depth_to_mp4_encodes(tmp_path):
    depth = np.zeros((4, 32, 48), np.float32); depth[:, 8:24, 8:40] = np.linspace(400, 1200, 4)[:, None, None]
    out = tmp_path / "d.mp4"
    webbed.depth_to_mp4(depth, out, fps=30)
    assert out.stat().st_size > 500


def test_featured_episodes_are_a_stable_sample(tmp_path, monkeypatch):
    from raiden_viz import config, sources

    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(config, "FEATURED_N", 3)
    eps = [f"ep_{i:03d}" for i in range(20)]

    class Fake(sources.Source):
        def list_episodes(self, task):
            return list(reversed(eps))     # newest first, like the real sources

    src = Fake({"id": "fake", "label": "f", "bucket": "b", "prefix": "p"})
    first = src.featured_episodes("t")
    assert len(first) == 3 and set(first) <= set(eps) and first == src.featured_episodes("t")
    assert first == sorted(first, key=list(reversed(eps)).index)          # keeps the newest-first order
    assert src.list_episodes("t") == list(reversed(eps))
    assert json.loads((tmp_path / "featured_fake_t.json").read_text())["of"] == 20
    assert Fake({"id": "fake", "label": "f", "bucket": "b", "prefix": "p"}).featured_episodes("t") == first   # from the JSON
    # a task that grows by more than 20 % gets a fresh pick; a small addition keeps it
    eps.extend(f"ep_{i:03d}" for i in range(20, 22))                   # +10 %: kept
    assert src.featured_episodes("t") == first
    eps.extend(f"ep_{i:03d}" for i in range(22, 40))                   # now 40 of 20 recorded: re-picked
    second = src.featured_episodes("t")
    assert len(second) == 3 and json.loads((tmp_path / "featured_fake_t.json").read_text())["of"] == 40


def test_depth_to_mp4_accepts_per_shard_chunks(tmp_path):
    """_mine hands depth_to_mp4 the per-shard stacks instead of one concatenated copy (memory)."""
    rng = np.random.default_rng(0)
    depth = (rng.random((6, 16, 16)) * 2000).astype(np.float32)
    out = tmp_path / "d.mp4"
    webbed.depth_to_mp4([depth[:2], depth[2:]], out, fps=30)
    assert out.stat().st_size > 0
