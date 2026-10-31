"""Ge's "webbed" YAM datasets (AnyData cv_webbed export, UW).

Layout on S3, one dataset per folder:

    <prefix>/<dataset>_seg48/<split>/split_all.json            manifest: sequences {"chunk_000/episode_000000": [shards]}
    <prefix>/<dataset>_seg48/<split>/metadata_shared.json      cameras, resolution, framerate, language prompt, tags
    <prefix>/<dataset>_seg48/<split>/chunk_000/episode_000000/<camera>/<start>_<end>.tar

A shard is one camera's ~44-48 consecutive steps ("webbed 48" = segment length): the tar holds
``<ep>_<seg>.rgb.(a:b,0).3.mp4`` (h264, one frame per step but muxed at 10 fps), ``….depth.(a:b,0).3.npz``
(float32 (F,1,H,W) millimetres), ``….lowdim.(a:b,0).3.npz`` (timestep, intrinsics, action{action,state} (F,32),
language) and ``….metadata.3.npz``. The lowdim members are identical across cameras; the viewer reads cam_high's.
"""

import json
import re
import subprocess
import tarfile
from pathlib import Path

import numpy as np

FPS = 30            # steps are 30 Hz (lowdim timestep is k/30); the shard mp4s are muxed at 10 fps and re-timed here
MAX_POINTS = 600    # plot series subsampling, as robot_data.summarize

_SHARD_RE = re.compile(r"^(\d+)_(\d+)\.tar$")


def shard_range(name: str) -> tuple[int, int]:
    m = _SHARD_RE.match(name)
    if not m:
        raise ValueError(f"not a shard name: {name!r}")
    return int(m.group(1)), int(m.group(2))


def episode_id(seq_key: str) -> str:
    """Manifest sequence key -> URL-safe episode id (``chunk_000/episode_000000`` -> ``chunk_000__episode_000000``)."""
    return seq_key.replace("/", "__")


def seq_key(ep_id: str) -> str:
    return ep_id.replace("__", "/")


def parse_manifest(manifest: dict) -> dict:
    """{episodes: {ep_id: [shard, ...]}, cameras, n_files}. Shards sorted by their frame range."""
    seqs = manifest.get("sequences") or {}
    episodes = {episode_id(k): sorted(v, key=shard_range) for k, v in seqs.items()}
    cfg = manifest.get("config") or {}
    return {"episodes": episodes, "cameras": list(cfg.get("cameras") or []),
            "n_files": (manifest.get("info") or {}).get("num_files")}


def episode_frames(shards: list[str]) -> int:
    return shard_range(shards[-1])[1] if shards else 0


def shard_bounds_s(shards: list[str], fps: int = FPS) -> list[float]:
    """Seconds at which a new shard starts (the first shard's 0 excluded)."""
    return [round(shard_range(s)[0] / fps, 3) for s in shards[1:]]


def read_shard(tar_path: Path, out_dir: Path) -> dict:
    """Extract one camera shard. Returns {rgb: Path|None, depth: ndarray (F,H,W) mm | None,
    lowdim: {action, state, language} | None}. Members are matched by their label token, not position."""
    out: dict = {"rgb": None, "depth": None, "lowdim": None}
    with tarfile.open(tar_path) as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            parts = m.name.split(".")
            label = parts[1] if len(parts) > 1 else ""
            if label == "rgb":
                tf.extract(m, out_dir, filter="data")
                out["rgb"] = out_dir / m.name
            elif label == "depth":
                d = np.load(tf.extractfile(m), allow_pickle=True)
                out["depth"] = np.asarray(d["depth"])[:, 0]          # (F,1,H,W) -> (F,H,W)
            elif label == "lowdim":
                d = np.load(tf.extractfile(m), allow_pickle=True)
                act = d["action"].item() if d["action"].dtype == object else {}
                lang = d["language"].item() if "language" in d.files and d["language"].dtype == object else {}
                prompt = lang.get("task") if isinstance(lang, dict) else None
                out["lowdim"] = {"action": np.asarray(act.get("action")), "state": np.asarray(act.get("state")),
                                 "language": str(np.asarray(prompt).ravel()[0]) if prompt is not None else None}
    return out


def concat_rgb(parts: list[Path], out_mp4: Path, fps: int = FPS) -> None:
    """Join the per-shard mp4s into one clip at the step rate (the shards are muxed at 10 fps, so a plain
    stream copy would play 3x slow)."""
    lst = out_mp4.with_suffix(".txt")
    lst.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
           "-vf", f"setpts=N/{fps}/TB", "-r", str(fps), "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-f", "mp4", str(out_mp4)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    finally:
        lst.unlink(missing_ok=True)


def depth_to_mp4(depth_mm: np.ndarray | list[np.ndarray], out_mp4: Path, fps: int = FPS) -> None:
    """Turbo-colormap a (F,H,W) millimetre depth stack (or a list of per-shard stacks, encoded in order without
    concatenating them) into an mp4. Range = [min, p99] of the valid pixels of the whole episode so colours stay
    stable across shards; zeros (no return) are black."""
    import cv2

    chunks = [depth_mm] if isinstance(depth_mm, np.ndarray) else depth_mm
    # range from every 8th frame: a full `depth[depth > 0]` copy of a float32 episode is >1 GB by itself
    sample = np.concatenate([c[::8].ravel() for c in chunks])
    valid = sample[sample > 0]
    lo = float(valid.min()) if valid.size else 0.0
    hi = float(np.percentile(valid, 99)) if valid.size else 1.0
    hi = hi if hi > lo else lo + 1.0
    _, h, w = chunks[0].shape
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", "-f", "mp4", str(out_mp4)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame in (fr for chunk in chunks for fr in chunk):
            u8 = np.clip((frame - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
            bgr = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
            bgr[frame <= 0] = 0
            proc.stdin.write(bgr.tobytes())
    finally:
        proc.stdin.close()
        err = proc.stderr.read()
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg depth encode failed: {err.decode(errors='replace')[-400:]}")


def robot_summary(lowdims: list[dict], fps: int = FPS, max_points: int = MAX_POINTS) -> dict | None:
    """Concatenate the shards' action/state into the viewer's robot dict (same shape as robot_data.summarize)."""
    acts = [d["action"] for d in lowdims if d and d.get("action") is not None and d["action"].ndim == 2]
    if not acts:
        return None
    action = np.concatenate(acts)
    states = [d["state"] for d in lowdims if d and d.get("state") is not None and d["state"].ndim == 2]
    n = action.shape[0]
    stride = max(1, n // max_points)
    idx = np.arange(0, n, stride)
    out = {"keys": ["action", "state"], "time": (idx / fps).round(3).tolist(),
           "summary": {"num_steps": int(n), "duration_s": round(n / fps, 3), "hz": fps}, "signals": {}}
    for name, arr in (("action", action), ("state", np.concatenate(states) if states else None)):
        if arr is None or arr.shape[0] != n:
            continue
        out["signals"][name] = {"dims": int(arr.shape[1]), "series": np.round(arr[idx].astype(np.float64), 5).tolist(),
                                "min": round(float(arr.min()), 5), "max": round(float(arr.max()), 5)}
    return out


def shared_meta(meta: dict) -> dict:
    """The dataset-level facts the overview shows, from metadata_shared.json."""
    lang = meta.get("language") or {}
    return {"prompt": lang.get("task"), "tags": (meta.get("info") or {}).get("tags") or meta.get("tags") or [],
            "cameras": meta.get("cameras") or [], "resolution": meta.get("resolution"),
            "framerate": meta.get("framerate"), "action_dim": (meta.get("action") or {}).get("action_dim")}


def dumps(obj) -> str:
    return json.dumps(obj)
