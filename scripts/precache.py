#!/usr/bin/env python
"""Pre-decode every featured episode of every configured dataset into the viewer's cache, so the hosted viewer
serves them instantly. Resumable (cache hits are skipped), one line of progress per episode, per-task disk use.

    .venv/bin/python scripts/precache.py                     # every accessible source
    .venv/bin/python scripts/precache.py --sources webbed    # one source
    .venv/bin/python scripts/precache.py --sources raiden --tasks place_teabag,hang_mugs

Run under nohup on the hosting box with the same RAIDEN_* environment as the service (cache dir!), e.g.
    set -a; grep -h ^Environment= ~/.config/systemd/user/raiden-viz.service{,.d/*.conf} | cut -d= -f2- | sed 's/^/export /' > /tmp/rv.env; . /tmp/rv.env; set +a
    nohup .venv/bin/python scripts/precache.py > ~/precache.log 2>&1 &
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raiden_viz import config, sources  # noqa: E402


def _cache_bytes() -> int:
    return sum(e.stat().st_size for e in os.scandir(config.CACHE_DIR) if e.is_file())


def precache_episode(src, task, episode) -> tuple[bool, str]:
    d = src.episode_detail(task, episode)               # robot summary, metadata (and for webbed: everything)
    for cam in d.get("cameras") or []:
        if not cam.get("has_video"):
            continue
        for eye in (cam.get("eyes") or ["left"])[:1]:    # left eye only: what the viewer shows by default
            src.video_path(task, episode, cam["name"], eye)
    return True, f"{len(d.get('cameras') or [])} cams"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default="", help="comma-separated source ids (default: all accessible)")
    ap.add_argument("--tasks", default="", help="comma-separated task names (default: all)")
    args = ap.parse_args()
    all_sources = sources.get_sources(config.SOURCES)
    want = [s for s in args.sources.split(",") if s] or list(all_sources)
    only_tasks = {t for t in args.tasks.split(",") if t}
    print(f"cache dir {config.CACHE_DIR} ({_cache_bytes() / 1e9:.1f} GB, cap {config.CACHE_MAX_GB} GB); "
          f"featured {config.FEATURED_N}/task; sources {want}", flush=True)
    t_all = time.time()
    for sid in want:
        src = all_sources[sid]
        try:
            tasks = [t for t in src.list_tasks() if not only_tasks or t in only_tasks]
        except Exception as e:
            print(f"[{sid}] cannot list tasks: {e}", flush=True)
            continue
        for task in tasks:
            before, t0 = _cache_bytes(), time.time()
            try:
                eps = src.list_episodes(task)
            except Exception as e:
                print(f"[{sid}/{task}] cannot list episodes: {e}", flush=True)
                continue
            ok = 0
            for ep in eps:
                t1 = time.time()
                try:
                    _, note = precache_episode(src, task, ep)
                    ok += 1
                    print(f"[{sid}/{task}] {ep}: {note} in {time.time() - t1:.1f}s", flush=True)
                except Exception as e:
                    print(f"[{sid}/{task}] {ep}: FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
            print(f"[{sid}/{task}] done {ok}/{len(eps)} episodes, +{(_cache_bytes() - before) / 1e9:.2f} GB, "
                  f"{time.time() - t0:.0f}s", flush=True)
    print(f"all done in {(time.time() - t_all) / 60:.1f} min; cache {_cache_bytes() / 1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
