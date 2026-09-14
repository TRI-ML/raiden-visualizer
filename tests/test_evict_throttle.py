"""evict() must not walk the cache directory on every call.

On the deployed task the boot scan leaves ~240k per-episode JSON cards in CACHE_DIR;
walking them cost 3-4 s and happened twice per clip, turning a 0.5 s stream copy into
an 8 s wait — and ten concurrent walkers starved the health check into 502s.
"""

import time

import pytest

from raiden_viz import cache, config


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(config, "CACHE_MAX_GB", 1e-6)  # ~1 KB cap so small files matter
    monkeypatch.setattr(cache, "_evict_state", {"last_walk": 0.0, "usage": 0})
    return tmp_path


def test_walk_is_skipped_while_estimate_is_under_cap(cache_dir, monkeypatch):
    (cache_dir / "a.mp4").write_bytes(b"x" * 100)
    cache.evict(force=True)                      # one real walk: usage = 100 B
    assert cache._evict_state["usage"] == 100
    stat_calls = []
    orig = type(cache_dir / "a.mp4").stat

    def counting_stat(self, *a, **k):
        stat_calls.append(self.name)
        return orig(self, *a, **k)

    monkeypatch.setattr(type(cache_dir / "a.mp4"), "stat", counting_stat)
    cache.evict()                                # fresh + under cap: no stat at all
    cache.evict(headroom_gb=0.0)
    assert stat_calls == []


def test_walk_happens_when_estimate_reaches_cap(cache_dir):
    (cache_dir / "old.mp4").write_bytes(b"x" * 600)
    time.sleep(0.02)
    (cache_dir / "new.mp4").write_bytes(b"x" * 600)
    cache.evict(force=True)                      # 1200 B > ~1073 B cap: evicts oldest
    assert not (cache_dir / "old.mp4").exists() and (cache_dir / "new.mp4").exists()
    assert cache._evict_state["usage"] == 600
    # produce more without walking: estimate crosses the cap -> next evict walks
    time.sleep(0.02)                             # distinct mtime (coarse kernel clock ties otherwise)
    (cache_dir / "newer.mp4").write_bytes(b"x" * 600)
    cache.note_produced(600)
    cache.evict()
    assert not (cache_dir / "new.mp4").exists() and (cache_dir / "newer.mp4").exists()


def test_stale_estimate_triggers_a_walk(cache_dir):
    cache.evict(force=True)
    cache._evict_state["last_walk"] = time.time() - cache.EVICT_WALK_INTERVAL_S - 1
    (cache_dir / "x.mp4").write_bytes(b"x" * 50)
    cache.evict()                                # stale: walks, picks up the 50 B
    assert cache._evict_state["usage"] == 50


def test_headroom_request_forces_walk_when_it_would_exceed_cap(cache_dir):
    cache.evict(force=True)
    (cache_dir / "x.mp4").write_bytes(b"x" * 500)
    cache.note_produced(500)
    cache.evict(headroom_gb=1e-6)                # cap - headroom = 0 -> must walk and evict
    assert not (cache_dir / "x.mp4").exists()


def test_get_or_create_updates_estimate(cache_dir, monkeypatch):
    monkeypatch.setattr(config, "CACHE_MAX_GB", 8.0)
    cache.evict(force=True)
    cache.get_or_create("clip.mp4", lambda p: p.write_bytes(b"y" * 321), remote=False)
    assert cache._evict_state["usage"] == 321
