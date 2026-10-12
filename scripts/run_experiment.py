"""Run one or more experiments on a dataset split.

Usage:
    uv run python scripts/run_experiment.py --experiment a1_gpt6_luna --split dev --limit 50
    uv run python scripts/run_experiment.py --experiment b1_decision1 --catalog verbose
    uv run python scripts/run_experiment.py --experiment a1_gpt6_luna b1_decision1 \
        --dataset banking77 --split calibration
    uv run python scripts/run_experiment.py --experiment b1_decision1 --split test --allow-test

Runs are resumable: re-running the same command only processes missing examples
(add --retry-errors to also redo failed ones).
"""

from __future__ import annotations

import argparse
import asyncio
import json

from edb.config import load_experiment_config
from edb.runners.predict import resolve_runs, run_split


def main() -> int:
    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument("--experiment", nargs="+", required=True,
                        help="configs/experiments/<id>.yaml (one or more)")
    parser.add_argument("--dataset", help="override the experiment's default dataset")
    parser.add_argument("--catalog", nargs="+", help="label catalog roles (default: all)")
    parser.add_argument("--split", nargs="+", default=["dev"])
    parser.add_argument("--limit", type=int, help="deterministic subset size")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="archive outputs produced by a different config")
    parser.add_argument("--allow-test", action="store_true",
                        help="required to run on sealed splits (final evaluation only)")
    parser.add_argument("--include-disabled", action="store_true")
    args = parser.parse_args()

    summaries = []
    for name in args.experiment:
        exp = load_experiment_config(name)
        if not exp.enabled and not args.include_disabled:
            print(f"[skip] {exp.id} is disabled: {exp.notes or ''}".strip())
            continue
        for spec in resolve_runs(name, args.dataset, args.catalog):
            sealed = set(spec.dataset.splits.sealed)
            for split in args.split:
                if split in sealed and not args.allow_test:
                    parser.error(f"split {split!r} is sealed; pass --allow-test for final runs")
                summaries.append(asyncio.run(run_split(
                    spec, split, limit=args.limit, concurrency=args.concurrency,
                    retry_errors=args.retry_errors, force=args.force,
                    sample_seed=args.sample_seed,
                )))
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
