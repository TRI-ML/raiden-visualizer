"""Pre-render every clip of a LeRobot task into the cache tiers.

    python -m raiden_viz.warm --source yam_sim --task plate_rack_front2_tbl
    python -m raiden_viz.warm --source yam_sim --all
    python -m raiden_viz.warm --source webbed --all --featured      # any source: its featured episodes only

``--featured`` pre-renders just each task's FEATURED episodes (config.FEATURED_N, the stable sample the
browse list shows by default): episode detail plus every camera clip and poster, through the normal
episode_detail()/video_path() path, for any source kind. That is what makes a dataset's default view load
instantly; the rest of the task still renders on demand.

Run it wherever the app's credentials are: on the task host, or on any box holding a
role/profile that can read the source bucket AND write the derived bucket, with the
same ``RAIDEN_DERIVED_BUCKET`` (and ``RAIDEN_DERIVED_PREFIX``) the deployment sets —
otherwise the clips land only in this box's local cache and the deployed app never
sees them. With the tier on, the first /video request for every warmed clip is a
302 to a presigned URL instead of a decode.

Idempotent and safe to re-run: clips already in either tier are skipped by
``cache.get_or_create``. Exit status is non-zero if any clip failed.
"""

import argparse
import sys
import time

from . import cache, catalog, config, contrib, raiden_teachers, sources


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="source id, e.g. yam_sim")
    ap.add_argument("--task", action="append", default=[], help="task name (repeatable)")
    ap.add_argument("--all", action="store_true", help="every task of the source")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--featured", action="store_true", help="only each task's featured episodes (any source kind)")
    args = ap.parse_args(argv)
    if not args.task and not args.all:
        ap.error("give --task or --all")

    src = sources.get_source(config.SOURCES, args.source)
    if args.featured:
        rc = warm_featured(src, src.list_tasks() if args.all else args.task, args.workers)
        warm_meta(src)
        return rc
    if not isinstance(src, sources.LeRobotSource):
        ap.error(f"{args.source} is kind {src.spec.get('kind')!r}; warm handles lerobot sources "
                 f"(use --featured for the others)")
    if not cache.remote_enabled():
        print("WARNING: RAIDEN_DERIVED_BUCKET is not set — clips will only warm THIS box's local "
              "cache, not the deployed app.", file=sys.stderr)

    tasks = src.list_tasks() if args.all else args.task
    total_failed = 0
    for task in tasks:
        t0 = time.perf_counter()
        last = [0.0]

        def progress(done, n, res, _t0=t0, _task=task):
            now = time.perf_counter()
            if res is not None:
                print(f"[warm {_task}] FAILED {res[0]} {res[1]}: {res[2]}", file=sys.stderr, flush=True)
            if done == n or now - last[0] > 10:
                last[0] = now
                el = now - _t0
                print(f"[warm {_task}] {done}/{n} artifacts, {el:.0f}s, "
                      f"{el / max(done, 1):.2f} s/item, eta {el / max(done, 1) * (n - done) / 60:.1f} min",
                      flush=True)

        res = src.warm_task(task, workers=args.workers, progress=progress)
        total_failed += len(res["failed"])
        print(f"[warm {task}] done: {res['ok']} ok, {len(res['failed'])} failed "
              f"({res['episodes']} episodes, {res['clips']} clips), {time.perf_counter() - t0:.0f}s",
              flush=True)
    return 1 if total_failed else 0


def warm_featured(src, tasks, workers: int = 2) -> int:
    """Render each task's featured episodes: detail + every camera clip (+ poster). One episode at a time per
    worker; a failure is reported and skipped. Resumable: rendered artifacts are cache hits."""
    from concurrent.futures import ThreadPoolExecutor
    failed = 0
    for task in tasks:
        try:
            eps = src.featured_episodes(task)
        except Exception as ex:  # noqa: BLE001
            print(f"[warm {task}] FAILED listing: {type(ex).__name__}: {ex}", file=sys.stderr, flush=True)
            failed += 1
            continue
        t0 = time.perf_counter()

        def _one(ep, _task=task):
            try:
                d = src.episode_detail(_task, ep)
                for c in d.get("cameras") or []:
                    if not c.get("has_video", True):
                        continue
                    src.video_path(_task, ep, c["name"], (c.get("eyes") or ["left"])[0])
                    try:
                        src.poster_path(_task, ep, c["name"], (c.get("eyes") or ["left"])[0])
                    except Exception:  # noqa: BLE001 - a poster is a nicety
                        pass
                return None
            except Exception as ex:  # noqa: BLE001
                return f"{ep}: {type(ex).__name__}: {ex}"

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for i, err in enumerate(pool.map(_one, eps), 1):
                if err:
                    failed += 1
                    print(f"[warm {task}] FAILED {err}", file=sys.stderr, flush=True)
                print(f"[warm {task}] {i}/{len(eps)} featured episodes, {time.perf_counter() - t0:.0f}s", flush=True)
    return 1 if failed else 0


def warm_meta(src) -> None:
    """Pre-compute what a dataset page reads before any clip: the overview (task rows), the sampled stats pass
    (hours / charts), the upload-activity calendar and the per-day teacher rollup, plus the catalog card. All of
    them persist in the derived tier, so the hosted app's first paint after a deploy is instant instead of a
    cross-region S3 walk. Each step is independent; a failure is reported and the rest still run."""
    spec = src.spec
    steps = [("overview", src.overview), ("stats", src.stats),
             ("contrib", lambda: cache.put_json(contrib._cache_key(spec["id"]), contrib.build_days(spec, src), remote=True)),
             ("catalog card", lambda: catalog.CatalogBuilder().build_deep(spec["id"], src))]
    if raiden_teachers.supports(spec):
        steps.append(("teachers", lambda: cache.put_json(raiden_teachers._cache_key(spec["id"]),
                                                         raiden_teachers.build_teachers(spec, src), remote=True)))
    for name, fn in steps:
        t0 = time.perf_counter()
        try:
            fn()
            print(f"[warm {spec['id']}] {name} ready, {time.perf_counter() - t0:.0f}s", flush=True)
        except Exception as ex:  # noqa: BLE001 - one slow/failed rollup must not stop the others
            print(f"[warm {spec['id']}] {name} FAILED: {type(ex).__name__}: {ex}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    sys.exit(main())
