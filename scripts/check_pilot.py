"""Automated sanity gate for pilot runs (used before launching full GPU queues).

For every expected run (experiment x catalog) on the given dataset/split, require: predictions
exist, zero errors, every distribution sums to 1 over the label set, and (optionally) accuracy
above chance. Prints per-run accuracy, latency and a projected full-run time. Exits 1 on failure.

Usage:
    uv run python scripts/check_pilot.py --root artifacts/pilot3 --dataset ace --split dev \
        --experiment c6_oss_letter_readout c7_oss_early_exit_head --catalog short --min-acc chance
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from edb.datasets.store import dataset_labels, load_split


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--experiment", nargs="+", required=True)
    ap.add_argument("--catalog", default="short")
    ap.add_argument("--min-acc", default="none", help="'chance', a number, or 'none'")
    ap.add_argument("--full-n", type=int, help="examples per full run, for the time projection")
    args = ap.parse_args()

    labels = dataset_labels(args.dataset)
    chance = 1.0 / len(labels)
    min_acc = {"none": None, "chance": chance}.get(args.min_acc)
    if min_acc is None and args.min_acc not in ("none",):
        min_acc = float(args.min_acc)
    full_n = args.full_n or len(load_split(args.dataset, "calibration")) + len(
        load_split(args.dataset, "test"))
    failures = []
    for exp in args.experiment:
        run = f"{exp}__{args.catalog}"
        path = args.root / "runs" / args.dataset / run / args.split / "predictions.jsonl"
        if not path.exists():
            failures.append(f"{run}: no predictions")
            print(f"FAIL {run:42s} missing {path}")
            continue
        preds = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                p = json.loads(line)
                preds[p["example_id"]] = p
        P = list(preds.values())
        errors = [p["error"] for p in P if p["error"]]
        bad = [p["example_id"] for p in P if not p["error"] and (
            not p["probabilities"] or abs(sum(p["probabilities"].values()) - 1) > 1e-4
            or set(p["probabilities"]) != set(labels))]
        acc = sum(p["predicted"] == p["gold"] for p in P) / len(P) if P else 0.0
        lat = sorted(p["latency_ms"] for p in P if p.get("latency_ms"))
        p50 = statistics.median(lat[1:] or lat) if lat else 0.0
        proj_h = p50 / 1000 * full_n / 3600
        ok = P and not errors and not bad and (min_acc is None or acc > min_acc)
        if not ok:
            failures.append(run)
        print(f"{'OK  ' if ok else 'FAIL'} {run:42s} n={len(P):3d} errors={len(errors)} "
              f"bad_probs={len(bad)} acc={acc:.2f} (chance {chance:.2f}) p50={p50 / 1000:.2f}s "
              f"-> full run ~{proj_h:.1f} h" + (f"  first error: {errors[0][:120]}" if errors
                                                  else ""))
    if failures:
        print("GATE FAILED:", failures)
        return 1
    print("GATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
