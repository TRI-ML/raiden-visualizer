"""TRI LBM real-robot episodes (spartan `observations.npz` / `actions.npz`), read by HTTP range.

Layout: ``<prefix>/<task>/<station>/real/bc/teleop/<session>/diffusion_spartan/episode_<n>/processed/``
holding ``observations.npz`` (1-3 GB: one ``<camera>.npy`` (T,H,W,3) uint8 + ``<camera>_depth.npy`` per camera,
then ~36 small ``robot__{actual,desired}__*`` low-dim members at the tail), ``actions.npz`` ((T,20): per arm xyz,
rot_6d, gripper width), per-camera ``images_<camera>/camera_info.yaml``, and ``../../../demonstration_<n>/summary.yaml``
(``success``, ``last_t``). The npz is a deflate zip: the central directory and the small members are fetched with
range GETs (~1 MB), so the state panel never downloads the cameras; a camera member can only be inflated from its
start, so a first frame costs ~1 MB and a full clip streams the whole member once (see ``camera_to_mp4``).
"""
import io
import re
import struct
import subprocess
import zipfile
from pathlib import Path

import numpy as np

from . import config, s3

FPS = 10.0          # spartan LBM logs at 10 Hz (summary.yaml last_t / T confirms ≈ 10)
MAX_POINTS = 600
LOWDIM_MAX_BYTES = 8_000_000   # members above this are camera streams, never read for the state panel
READ_AHEAD = 1_000_000         # bytes fetched per range GET beyond what zipfile asked for
TELEOP = "/real/bc/teleop/"
_EP_RE = re.compile(r"^(?:(?P<prefix>.+?)/)?(?P<task>[^/]+)/(?P<station>[^/]+)/real/bc/teleop/(?P<session>[^/]+)/"
                    r"diffusion_spartan/episode_(?P<n>\d+)/processed/observations\.npz$")


class S3RangeFile(io.RawIOBase):
    """Seekable read-only file over an S3 object, one range GET per read (zipfile seeks the tail, then members)."""

    def __init__(self, key: str, bucket: str, size: int):
        self.key, self.bucket, self.size, self.pos = key, bucket, size, 0
        self.bytes_read = 0
        self._buf_start, self._buf = None, b""

    def prefetch(self, start: int) -> None:
        """Pull [start, size) in one GET so the many small tail members cost one round trip, not one each."""
        self._buf_start, self._buf = start, s3.get_range(self.key, start, self.size - 1, bucket=self.bucket)
        self.bytes_read += len(self._buf)

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, off, whence=0):
        self.pos = (off, self.pos + off, self.size + off)[whence]
        return self.pos

    def tell(self):
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        if not (self._buf_start is not None and self._buf_start <= self.pos < self._buf_start + len(self._buf)):
            # zipfile reads in 4 KB steps; one range GET per step turned a 1 MB first frame into ~230 round trips
            end = min(self.size, self.pos + max(n, READ_AHEAD)) - 1
            self._buf_start, self._buf = self.pos, s3.get_range(self.key, self.pos, end, bucket=self.bucket)
            self.bytes_read += len(self._buf)
        o = self.pos - self._buf_start
        data = self._buf[o:o + n]
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def open_npz(key: str, bucket: str, size: int) -> tuple[zipfile.ZipFile, S3RangeFile]:
    f = S3RangeFile(key, bucket, size)
    return zipfile.ZipFile(f), f


def episode_id(station: str, session: str, n: int) -> str:
    # `<station>_<ISO stamp>…` is the raiden naming the overview's station regex already understands
    return f"{station}_{session}_episode_{int(n)}"


def discover(objs) -> dict[str, dict]:
    """Episodes of ONE task from its recursive listing: {episode_id: {obs_key, size, cameras, station, session, n,
    last_modified}}. Only ``real/bc/teleop`` sessions that carry observations.npz count (sim and config-only stubs
    are skipped); cameras come from the sibling ``images_<cam>/`` dirs in the same listing."""
    eps: dict[str, dict] = {}
    cams: dict[str, set] = {}
    for o in objs:
        m = _EP_RE.match(o.key)
        if m:
            eid = episode_id(m["station"], m["session"], m["n"])
            eps[eid] = {"obs_key": o.key, "size": o.size, "etag": o.etag, "station": m["station"], "session": m["session"],
                        "n": int(m["n"]), "last_modified": o.last_modified}
        elif TELEOP in o.key and "/processed/images_" in o.key:
            base, cam = o.key.split("/processed/images_", 1)
            cams.setdefault(base + "/processed/observations.npz", set()).add(cam.split("/", 1)[0])
    for e in eps.values():
        e["cameras"] = sorted(cams.get(e["obs_key"], ()))
    return eps


def session_time(session: str) -> str | None:
    """``2024-01-11T12-52-02-05-00`` -> ISO8601 with the zone (the dir name is a clock-safe ISO stamp)."""
    m = re.match(r"^(\d{4}-\d{2}-\d{2})T(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})$", session)
    return f"{m[1]}T{m[2]}:{m[3]}:{m[4]}-{m[5]}:{m[6]}" if m else None


def lowdim(obs_key: str, bucket: str, size: int, max_bytes: int = LOWDIM_MAX_BYTES) -> tuple[dict, int]:
    """All small members of observations.npz as arrays, by range GETs. Returns (arrays, bytes fetched)."""
    z, f = open_npz(obs_key, bucket, size)
    small = [i for i in z.infolist() if i.file_size <= max_bytes]
    if small:
        f.prefetch(min(i.header_offset for i in small))   # the low-dim members sit contiguously at the tail
    arrs = {i.filename.removesuffix(".npy"): np.load(io.BytesIO(z.read(i.filename))) for i in small}
    return arrs, f.bytes_read


def cameras_of(z: zipfile.ZipFile, max_bytes: int = LOWDIM_MAX_BYTES) -> list[str]:
    return sorted(i.filename.removesuffix(".npy") for i in z.infolist()
                  if i.file_size > max_bytes and not i.filename.endswith("_depth.npy"))


def _npy_header(m) -> tuple[tuple, np.dtype]:
    magic = m.read(6)
    if magic != b"\x93NUMPY":
        raise ValueError("not a .npy member")
    major = m.read(2)[0]
    hlen = struct.unpack("<H", m.read(2))[0] if major == 1 else struct.unpack("<I", m.read(4))[0]
    hdr = eval(m.read(hlen).decode("latin1"), {"__builtins__": {}}, {})   # the npy dict literal, as numpy does
    return tuple(hdr["shape"]), np.dtype(hdr["descr"])


def first_frame(obs_key: str, bucket: str, size: int, camera: str) -> np.ndarray:
    """Frame 0 of one camera by inflating only the start of its member (~1 MB fetched)."""
    z, _ = open_npz(obs_key, bucket, size)
    with z.open(camera + ".npy") as m:
        shape, dt = _npy_header(m)
        n = int(np.prod(shape[1:])) * dt.itemsize
        return np.frombuffer(m.read(n), dt).reshape(shape[1:])


def camera_to_mp4(obs_key: str, bucket: str, size: int, camera: str, dst: Path, fps: float = FPS,
                  max_bytes: int | None = None) -> None:
    """Stream-inflate one camera member into ffmpeg (rawvideo rgb24 -> H.264). The member must be read whole
    (deflate is not seekable); ``max_bytes`` refuses members above the cap instead of streaming gigabytes."""
    z, _ = open_npz(obs_key, bucket, size)
    info = z.getinfo(camera + ".npy")
    if max_bytes is not None and info.file_size > max_bytes:
        raise RuntimeError(f"camera member {camera} is {info.file_size / 1e6:.0f} MB, above the {max_bytes / 1e6:.0f} MB clip cap")
    with z.open(info) as m:
        shape, dt = _npy_header(m)
        t, h, w = shape[0], shape[1], shape[2]
        frame_bytes = int(np.prod(shape[1:])) * dt.itemsize
        cmd = ["ffmpeg", "-y", "-loglevel", "error", *config.FFMPEG_THREADS, "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-f", "mp4", str(dst)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for _ in range(t):
                proc.stdin.write(m.read(frame_bytes))
        finally:
            proc.stdin.close()
            err = proc.stderr.read()
            if proc.wait() != 0:
                raise RuntimeError(f"ffmpeg encode failed: {err.decode(errors='replace')[-400:]}")


def robot_summary(arrs: dict, actions: np.ndarray | None, fps: float = FPS, max_points: int = MAX_POINTS) -> dict:
    """The viewer's robot dict (shape of robot_data.summarize): measured low-dim signals + the commanded action."""
    sig = {k: np.asarray(v) for k, v in arrs.items() if np.asarray(v).ndim in (1, 2)}
    if actions is not None:
        sig["action"] = np.asarray(actions)
    n = max((a.shape[0] for a in sig.values()), default=0)
    stride = max(1, n // max_points)
    idx = np.arange(0, n, stride)
    out = {"keys": sorted(sig), "time": (idx / fps).round(3).tolist(),
           "summary": {"num_steps": int(n), "duration_s": round(n / fps, 3), "hz": fps}, "signals": {}}
    for k, a in sorted(sig.items()):
        a = (a[:, None] if a.ndim == 1 else a).astype(np.float64)
        if a.shape[0] != n or not np.isfinite(a).any():
            continue   # cart-only sessions store all-NaN joint streams (Zheyuan's LBM card, defect B1)
        sub = np.round(a[idx], 5)
        series = [[None if not np.isfinite(v) else v for v in row] for row in sub.tolist()]   # JSON has no NaN
        out["signals"][k] = {"dims": int(a.shape[1]), "series": series,
                             "min": round(float(np.nanmin(a)), 5), "max": round(float(np.nanmax(a)), 5)}
    return out
