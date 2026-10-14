"""Show AML job status and download finished outputs into local artifacts.

Usage:
    uv run --extra aml python scripts/aml_collect.py --tag main
    uv run --extra aml python scripts/aml_collect.py --tag pilot --dest artifacts/pilot
    uv run --extra aml python scripts/aml_collect.py --tag main --status-only
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

from edb.runners.aml import TERMINAL, collect, ml_client, read_log, status_of
from edb.settings import get_settings


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect AML job outputs")
    parser.add_argument("--tag", default="main")
    parser.add_argument("--dest", type=Path,
                        help="local artifacts root (default: ARTIFACTS_DIR; use a separate "
                             "root for pilots so subsets never mix with full runs)")
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--include-running", action="store_true",
                        help="also download partial outputs of running jobs")
    parser.add_argument("--wait", action="store_true",
                        help="block until every job of the tag is terminal, then collect")
    parser.add_argument("--poll", type=float, default=120.0)
    args = parser.parse_args()

    entries = read_log(args.tag)
    if not entries:
        print(f"no submitted jobs with tag {args.tag!r}")
        return 1
    ml = ml_client()
    if args.wait:
        latest = {f"{e['dataset']}/{e['run_id']}": e for e in entries}
        while True:
            states = {k: status_of(ml.jobs.get(e["name"])) for k, e in latest.items()}
            pending = sorted(k for k, s in states.items() if s not in TERMINAL)
            counts = Counter(states.values())
            print(f"{time.strftime('%H:%M:%S')} {dict(counts)} pending={len(pending)}",
                  flush=True)
            if not pending:
                break
            time.sleep(args.poll)
    if args.status_only:
        latest = {f"{e['dataset']}/{e['run_id']}": e for e in entries}
        for key, e in latest.items():
            print(f"{key:50s} {status_of(ml.jobs.get(e['name'])):12s} {e['name']}")
        return 0
    dest = args.dest or get_settings().artifacts_dir
    for s in collect(ml, entries, dest, only_terminal=not args.include_running):
        print(f"{s['run_id']:40s} {s['status']:12s} collected={s.get('collected', False)} "
              f"{s.get('download_error', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
