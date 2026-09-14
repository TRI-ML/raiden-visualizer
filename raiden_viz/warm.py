"""Pre-render every clip of a LeRobot task into the cache tiers.

    python -m raiden_viz.warm --source yam_sim --task plate_rack_front2_tbl
    python -m raiden_viz.warm --source yam_sim --all

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

from . import cache, config, sources


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="source id, e.g. yam_sim")
    ap.add_argument("--task", action="append", default=[], help="task name (repeatable)")
    ap.add_argument("--all", action="store_true", help="every task of the source")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    if not args.task and not args.all:
        ap.error("give --task or --all")

    src = sources.get_source(config.SOURCES, args.source)
    if not isinstance(src, sources.LeRobotSource):
        ap.error(f"{args.source} is kind {src.spec.get('kind')!r}; warm handles lerobot sources")
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


if __name__ == "__main__":
    sys.exit(main())
