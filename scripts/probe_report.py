"""Side-by-side report of probe results across targets -> artifacts/probes/report.md.

Usage:
    uv run python scripts/probe_report.py --target decision-1 qwen3.5-9b qwen3.5-2b
"""

from __future__ import annotations

import argparse

from edb.probing.report import build_probe_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare probe summaries across targets")
    parser.add_argument("--target", nargs="+", required=True)
    args = parser.parse_args()
    print(build_probe_report(args.target))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
