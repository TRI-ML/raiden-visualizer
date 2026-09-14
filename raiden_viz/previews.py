"""Small, fast dataset previews — one per task, precomputed into the derived tier.

A card must paint its poster within ~300 ms and start its hover clip within ~1 s, so
the full episode clip is the wrong asset: it is minutes of video at native size.
Each task gets, for its preview episode and every camera, a 320 px poster JPEG
(``pv_<key>_<camera>.jpg``, ~20 KB) and a 5 s 320 px H.264 faststart clip
(``pv_<key>_<camera>.mp4``, ~200 KB). ``key`` is the source object's etag (or the
LeRobot clip name), so a re-upload invalidates.

Raw sources are NOT decoded in full for this. Both the raiden ``.svo2`` files and the
yam ``output.mcap`` are MCAP containers whose camera messages carry H.264 payloads in
time order, so the first seconds of video live in the first megabytes: one S3 range
GET of the file head, a non-seeking MCAP read of that prefix, and ffmpeg on the
resulting elementary stream. No 200-880 MB download, no ZED SDK, no full transcode.

The per-source result ``previews_<sid>_v1.json`` ({task: {episode, camera, cameras,
poster_name, clip_name, poster_names, clip_names}}) is what catalog cards, overview
task rows, the task header and external pages (yam_eval's Data page) render from:
one JSON, then ONE request per visible asset (``GET /api/artifact/<name>`` -> 302).
"""
from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path

from mcap.reader import NonSeekingReader

from . import cache, s3, svo, yam

log = logging.getLogger("raiden_viz")

PREVIEWS_V = 1
POSTER_W = 320
CLIP_SECONDS = 5
FRAMES_WANTED = 30 * CLIP_SECONDS
HEAD_BYTES = 24 * 1024 ** 2          # first read of a raw file
HEAD_BYTES_RETRY = 96 * 1024 ** 2    # if the first read held no video yet
MAIN_CAMERA_HINTS = ("scene", "top", "ego", "front", "high", "head")


def blob_name(sid: str) -> str:
    return f"previews_{sid}_v{PREVIEWS_V}.json"


def asset_names(key: str, camera: str) -> tuple[str, str]:
    stem = f"pv_{key}_{camera}"
    return stem + ".jpg", stem + ".mp4"


def pick_main_camera(cams: list[str]) -> str | None:
    if not cams:
        return None
    for hint in MAIN_CAMERA_HINTS:
        for c in cams:
            if hint in c.lower():
                return c
    return cams[0]


# ---- frames from the head of an MCAP-container file ---------------------------------

def head_frames(head: Path, want, extract, max_frames: int = FRAMES_WANTED) -> dict[str, tuple[list[bytes], str | None]]:
    """Walk a (truncated) MCAP prefix; return {camera: (payloads, codec)} for every
    topic ``want(topic, schema_name)`` names, up to ``max_frames`` each. A truncated
    tail raises inside the reader — that is the expected end of the walk."""
    out: dict[str, list[bytes]] = {}
    codecs: dict[str, str | None] = {}
    try:
        with open(head, "rb") as f:
            for schema, channel, message in NonSeekingReader(f).iter_messages(log_time_order=False):
                cam = want(channel.topic, schema.name if schema else None)
                if cam is None:
                    continue
                buf = out.setdefault(cam, [])
                if len(buf) >= max_frames:
                    continue
                payload, fmt = extract(message.data)
                if payload:
                    buf.append(payload)
                    if fmt and cam not in codecs:
                        codecs[cam] = fmt.lower()
    except Exception:  # noqa: BLE001 — truncated prefix: we have what the head held
        pass
    return {cam: (frames, codecs.get(cam)) for cam, frames in out.items() if frames}


def _svo_extract(data: bytes):
    if len(data) < svo._HDR.size:
        return b"", None
    _total, n = svo._HDR.unpack_from(data, 0)
    return data[svo._HDR.size: svo._HDR.size + n], "h264"


def _svo_want(camera: str):
    return lambda topic, schema: camera if topic.endswith("side_by_side") else None


def _yam_want(topic: str, schema: str | None):
    return yam.camera_name(topic) if yam.is_camera_topic(topic, schema) else None


def fetch_head(key: str, bucket: str | None, dest: Path, nbytes: int) -> None:
    dest.write_bytes(s3.get_range(key, 0, nbytes - 1, bucket=bucket))


# ---- ffmpeg: poster + short clip ----------------------------------------------------

def render_assets(src: Path, poster: Path, clip: Path, *, demuxer: str | None = None,
                  fps: int = 30, crop: str | None = None) -> None:
    """Poster (first frame) and a CLIP_SECONDS clip, both POSTER_W wide, from an
    elementary stream (``demuxer`` = h264/hevc) or an mp4 (``demuxer`` None)."""
    vf = ",".join(([crop] if crop else []) + [f"scale={POSTER_W}:-2"])
    inp = ["-f", demuxer, "-r", str(fps), "-i", str(src)] if demuxer else ["-i", str(src)]
    base = ["ffmpeg", "-y", "-loglevel", "error", *inp]
    subprocess.run(base + ["-frames:v", "1", "-vf", vf, "-q:v", "5", "-f", "image2", str(poster)], check=True)
    subprocess.run(base + ["-t", str(CLIP_SECONDS), "-vf", vf, "-c:v", "libx264", "-preset", "veryfast",
                           "-crf", "28", "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart",
                           "-f", "mp4", str(clip)], check=True)


def publish_assets(key: str, camera: str, render) -> tuple[str, str]:
    """Render into temp files via ``render(poster_tmp, clip_tmp)`` unless both assets
    are already in the tier; publish; return (poster_name, clip_name)."""
    pj, pm = asset_names(key, camera)
    if cache.remote_ready(pj) and cache.remote_ready(pm):
        return pj, pm
    tag = f".tmp{os.getpid()}_{int(time.time() * 1000) % 100000}"
    tj, tm = cache.path_for(pj + tag), cache.path_for(pm + tag)
    try:
        render(tj, tm)
        for tmp, name in ((tj, pj), (tm, pm)):
            dst = cache.path_for(name)
            tmp.replace(dst)
            cache.note_produced(dst.stat().st_size)
            cache.push_remote(name, dst)
    finally:
        tj.unlink(missing_ok=True)
        tm.unlink(missing_ok=True)
    return pj, pm


# ---- per-kind builders: {camera: (poster_name, clip_name)} for one episode -----------

def build_lerobot(src, task: str, episode: str) -> dict[str, tuple[str, str]]:
    info = src._meta(task)["info"]
    out = {}
    for cam in info["cameras"]:
        name = src.video_cache_name(task, episode, cam, "left")
        clip = cache.path_for(name)
        if not (clip.exists() and clip.stat().st_size > 0) and not cache.exists(name):
            clip = src.video_path(task, episode, cam, "left")
            if not clip.exists():
                cache.exists(name)
        out[cam] = publish_assets(name[:-4] if name.endswith(".mp4") else name, cam,
                                  lambda pj, pm, c=clip: render_assets(c, pj, pm))
    return out


def build_raiden(src, task: str, episode: str, workdir: Path) -> dict[str, tuple[str, str]]:
    prefix = src._ep_prefix(task, episode)
    out = {}
    for obj in s3.list_files(f"{prefix}/cameras", bucket=src.bucket):
        if not obj.key.endswith(".svo2") or obj.size < 100_000:
            continue
        cam = obj.key.rsplit("/", 1)[-1][: -len(".svo2")]
        pj, pm = asset_names(obj.etag, cam)
        if cache.remote_ready(pj) and cache.remote_ready(pm):
            out[cam] = (pj, pm)
            continue
        frames = None
        for nbytes in (HEAD_BYTES, HEAD_BYTES_RETRY):
            head = workdir / f"{cam}.head"
            fetch_head(obj.key, src.bucket, head, min(nbytes, obj.size))
            got = head_frames(head, _svo_want(cam), _svo_extract)
            head.unlink(missing_ok=True)
            if got.get(cam):
                frames = got[cam][0]
                break
            if nbytes >= obj.size:
                break
        if not frames:
            log.warning("preview: no video in the first %d MB of %s", HEAD_BYTES_RETRY >> 20, obj.key)
            continue
        stream = workdir / f"{cam}.h264"
        stream.write_bytes(b"".join(frames))
        try:
            out[cam] = publish_assets(obj.etag, cam,
                                      lambda a, b, s=stream: render_assets(s, a, b, demuxer="h264",
                                                                           crop="crop=iw/2:ih:0:0"))
        finally:
            stream.unlink(missing_ok=True)
    return out


def build_yam(src, task: str, episode: str, workdir: Path) -> dict[str, tuple[str, str]]:
    obj = src._head(task, episode)
    got = {}
    for nbytes in (HEAD_BYTES, HEAD_BYTES_RETRY):
        head = workdir / "output.head"
        fetch_head(obj.key, src.bucket, head, min(nbytes, obj.size))
        got = head_frames(head, _yam_want, yam._cv_fields)
        head.unlink(missing_ok=True)
        if got or nbytes >= obj.size:
            break
    out = {}
    for cam, (frames, codec) in got.items():
        pj, pm = asset_names(obj.etag, cam)
        if cache.remote_ready(pj) and cache.remote_ready(pm):
            out[cam] = (pj, pm)
            continue
        stream = workdir / f"{cam}.bitstream"
        stream.write_bytes(b"".join(frames))
        demuxer = yam._FMT_TO_FFMPEG.get(codec or "h264", "h264")
        try:
            out[cam] = publish_assets(obj.etag, cam,
                                      lambda a, b, s=stream, d=demuxer: render_assets(s, a, b, demuxer=d))
        finally:
            stream.unlink(missing_ok=True)
    return out


def build_for(src, task: str, episode: str, workdir: Path) -> dict[str, tuple[str, str]]:
    kind = src.spec.get("kind")
    if kind in ("lerobot", "lerobot_single"):
        return build_lerobot(src, task, episode)
    if kind == "raiden":
        return build_raiden(src, task, episode, workdir)
    if kind == "yam":
        return build_yam(src, task, episode, workdir)
    raise ValueError(f"no preview builder for kind {kind!r}")


def entry_from(episode: str, assets: dict[str, tuple[str, str]]) -> dict | None:
    cams = sorted(assets)
    main = pick_main_camera(cams)
    if main is None:
        return None
    return {"episode": episode, "camera": main, "cameras": cams,
            "poster_name": assets[main][0], "clip_name": assets[main][1],
            "poster_names": {c: assets[c][0] for c in cams},
            "clip_names": {c: assets[c][1] for c in cams}}
