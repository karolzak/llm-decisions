"""Part 3: train one decision-readout arm, then evaluate it like Decision-1 (one GPU job).

    1. train   LoRA on configs/training/corpus.yaml with the arm's loss/shuffle setting
               -> artifacts/training/<arm>/{adapter.safetensors, summary.json, train_log.jsonl}
    2. eval    the r3_<arm> experiment on ACE and BANKING77 (never seen in training)
    3. probe   the r3-<arm> probe target (same probes as Decision-1)

`--arm frozen` skips training and evaluates the untrained base (r2_qwen35_2b_letter and the
qwen3.5-2b probe target), the control every trained arm is compared with.

Usage (GPU; on Azure ML via scripts/aml_run.py):
    python scripts/train_decider.py --arm ce_shuffle --split calibration test --allow-test
    python scripts/train_decider.py --arm frozen --split calibration test --allow-test
    python scripts/train_decider.py --arms-file arms-r9 --arm r9_qwen9b \
        --split calibration test --allow-test          # R9: decision-trained Qwen3.5-9B
    python scripts/train_decider.py --arm ce_shuffle --max-steps 2 --max-rows 64 \
        --split dev --limit 8 --items 2      # smoke test
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from edb.config import load_experiment_config
from edb.probing.clients import load_probe_target, make_client
from edb.probing.probes import PROBES
from edb.probing.runner import (
    analyze_target,
    build_cases,
    load_base_items,
    probe_dir,
    run_cases,
)
from edb.runners.predict import resolve_runs, run_split
from edb.training.lora import CONFIG
from edb.training.train import arm_dir, load_arm, train

DEFAULT_PROBES = ["order", "letter_names", "duplicate", "question", "empty_state",
                  "cardinality", "duplicate_names", "long_options", "paraphrase"]


def names_for(arm: str, arms_file: str = "arms") -> tuple[str, str]:
    if arm == "frozen":
        return "r2_qwen35_2b_letter", "qwen3.5-2b"
    cfg = load_arm(arm, arms_file)
    return (cfg.experiment or f"r3_{arm}",
            cfg.probe_target or f"r3-{arm.replace('_', '-')}")


async def probe(target_name: str, probes: list[str], items: int) -> list[dict]:
    target = load_probe_target(target_name)
    base = load_base_items(items)
    client = make_client(target)
    try:
        return [await run_cases(client, target, name,
                                build_cases(PROBES[name](), base)) for name in probes]
    finally:
        await client.aclose()


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate one Part 3 arm")
    parser.add_argument("--arm", required=True, help="arm in --arms-file, or frozen")
    parser.add_argument("--arms-file", default="arms", help="configs/training/<name>.yaml")
    parser.add_argument("--datasets", nargs="+", default=["ace", "banking77"])
    parser.add_argument("--split", nargs="+", default=["dev"])
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--limit", type=int, help="examples per dataset split")
    parser.add_argument("--probe", nargs="*", default=DEFAULT_PROBES, choices=list(PROBES))
    parser.add_argument("--items", type=int, default=16, help="probe base items per source")
    parser.add_argument("--skip-train", action="store_true", help="reuse an existing adapter")
    parser.add_argument("--retrain", action="store_true", help="train even if an adapter exists")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--r14", action="store_true",
                        help="also run scripts/r14_question_types.py local with the adapter")
    parser.add_argument("--r19", action="store_true",
                        help="also run scripts/r19_unseen_eval.py local with the adapter")
    args = parser.parse_args()

    experiment, target = names_for(args.arm, args.arms_file)
    report: dict = {"arm": args.arm, "experiment": experiment, "probe_target": target}
    trained = False
    if args.arm != "frozen" and not args.skip_train:
        if args.retrain or not (arm_dir(args.arm) / CONFIG).exists():
            cfg = load_arm(args.arm, args.arms_file)
            updates = {k: v for k, v in (("max_steps", args.max_steps),
                                         ("max_rows", args.max_rows)) if v}
            report["train"] = train(cfg.model_copy(update=updates))
            trained = True
        else:
            print(f"[skip] adapter exists in {arm_dir(args.arm)} (use --retrain)")

    exp = load_experiment_config(experiment)
    evals = []
    for dataset in args.datasets:
        for spec in resolve_runs(experiment, dataset, exp.label_catalogs):
            for split in args.split:
                if split in spec.dataset.splits.sealed and not args.allow_test:
                    parser.error(f"split {split!r} is sealed; pass --allow-test")
                # A fresh adapter invalidates predictions made with the previous one.
                evals.append(asyncio.run(run_split(spec, split, limit=args.limit,
                                                   force=trained)))
    report["eval"] = evals
    if args.probe:
        stale = probe_dir(target)
        if trained and stale.exists():  # probe records of the previous adapter
            stale.rename(stale.with_name(f"{target}.archived-{datetime.now(UTC):%Y%m%dT%H%M%S}"))
        report["probes"] = asyncio.run(probe(target, args.probe, args.items))
        analyze_target(target)
    if args.r14 and args.arm != "frozen":
        # Question-type comparison with Decision-1 (R14 items), read through the new adapter.
        cfg = load_arm(args.arm, args.arms_file)
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # the child process loads its own copy of the model
        subprocess.run([sys.executable, str(Path(__file__).with_name("r14_question_types.py")),
                        "local", "--model", cfg.base_model, "--adapter",
                        f"training/{args.arm}", "--label", args.arm], check=True)
    if args.r19 and args.arm != "frozen":
        # Unseen benchmarks (R19), options byte-sorted like Decision-1's.
        cfg = load_arm(args.arm, args.arms_file)
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        subprocess.run([sys.executable, str(Path(__file__).with_name("r19_unseen_eval.py")),
                        "local", "--model", cfg.base_model, "--adapter",
                        f"training/{args.arm}", "--label", args.arm], check=True)
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
