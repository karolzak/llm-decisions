"""Latency benchmark (plan defaults: 20 warmups, 200 examples, 5 reps, concurrency 1 and 8).

Usage:
    uv run python scripts/run_latency.py --experiment b1_decision1 --catalog short
    uv run python scripts/run_latency.py --experiment a1_gpt6_luna --n 50 --reps 2 --concurrency 1 4
"""

from __future__ import annotations

import argparse
import asyncio
import json

from edb.runners.latency import benchmark
from edb.runners.predict import resolve_runs


def main() -> int:
    parser = argparse.ArgumentParser(description="Latency benchmark")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--catalog", nargs="+", default=["short"])
    parser.add_argument("--split", default="dev")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 8])
    args = parser.parse_args()
    for spec in resolve_runs(args.experiment, args.dataset, args.catalog):
        summary = asyncio.run(benchmark(spec, args.split, args.n, args.warmup, args.reps,
                                        tuple(args.concurrency)))
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
