"""Build leaderboards and the markdown report for a dataset split.

Usage:
    uv run python scripts/build_report.py --dataset ace --split dev
    uv run python scripts/build_report.py --dataset ace --split test --no-intersect
"""

from __future__ import annotations

import argparse

from edb.reporting.report import build_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build benchmark report")
    parser.add_argument("--dataset", default="ace")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--no-intersect", action="store_true",
                        help="do not restrict each leaderboard to examples shared by all runs")
    args = parser.parse_args()
    path = build_report(args.dataset, args.split, intersect=not args.no_intersect)
    print(f"Report: {path}")
    print(path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
