"""Compare runs example by example against a reference run (default: Decision-1).

The question for Part 2: how much of Decision-1's behaviour is already in its frozen base
model? Accuracy alone cannot tell; agreement, shared mistakes and correlated probabilities
can. A frozen base that agrees with Decision-1 far above what the two accuracies imply
(kappa, same wrong answers) shares its judgement; low agreement means post-training changed
the decisions, not just the readout.

Usage:
    uv run python scripts/compare_runs.py --dataset ace --split test \
        --runs r2_qwen35_9b_letter__verbose r3_ce_shuffle__verbose
    uv run python scripts/compare_runs.py --dataset banking77 --split test \
        --reference b1_decision1__short --runs r2_qwen35_9b_letter__short
    # compact reference files for GPU jobs (configs/reference/, read by scripts/job_summary.py)
    uv run python scripts/compare_runs.py --dataset ace --split test --export-reference
"""

from __future__ import annotations

import argparse

from edb.datasets.store import dataset_labels
from edb.probing.report import _fmt
from edb.reporting.compare import compare, export_reference
from edb.runners.predict import read_predictions, run_dir
from edb.settings import get_settings


def default_reference(dataset: str, split: str) -> str:
    verbose = run_dir(dataset, "b1_decision1__verbose", split)
    return "b1_decision1__verbose" if verbose.exists() else "b1_decision1__short"


def main() -> int:
    parser = argparse.ArgumentParser(description="Per-example comparison against a reference")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--reference", help="run id (default: b1_decision1__verbose if it "
                                            "exists for the dataset, else b1_decision1__short)")
    parser.add_argument("--runs", nargs="*", default=[])
    parser.add_argument("--export-reference", action="store_true",
                        help="write configs/reference/<dataset>__<reference>__<split>.jsonl.gz")
    args = parser.parse_args()
    labels = dataset_labels(args.dataset)
    reference = args.reference or default_reference(args.dataset, args.split)
    ref = read_predictions(run_dir(args.dataset, reference, args.split) / "predictions.jsonl")
    if args.export_reference:
        path = get_settings().configs_dir / "reference" / \
            f"{args.dataset}__{reference}__{args.split}.jsonl.gz"
        print(f"{export_reference(ref, labels, path)} rows -> {path}")
    if not args.runs:
        return 0
    rows = []
    for run in args.runs:
        other = read_predictions(run_dir(args.dataset, run, args.split) / "predictions.jsonl")
        rows.append({"run": run, **compare(ref, other, labels)})
    keys = [k for r in rows for k in r if k != "run"]
    keys = list(dict.fromkeys(keys))
    lines = [f"# `{args.dataset}` / `{args.split}`: runs compared with `{reference}`", "",
             "| metric | " + " | ".join(r["run"] for r in rows) + " |",
             "|---|" + "---|" * len(rows)]
    for k in keys:
        lines.append(f"| {k} | " + " | ".join(_fmt(r.get(k)) for r in rows) + " |")
    out = get_settings().artifacts_dir / "reports" / args.dataset / args.split / \
        f"compare_{reference}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
