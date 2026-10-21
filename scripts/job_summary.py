"""Compact JSON summary of everything under ARTIFACTS_DIR (runs, probes, training).

Run at the end of every Azure ML job (infra/aml/run_job.sh), whose summary is then attached
to the job as MLflow tags, so results are readable through the workspace API even when the
workspace storage cannot be reached from the submitting machine
(scripts/aml_summaries.py reassembles them). Runs are also compared example by example with
Decision-1 when a compact reference exists in configs/reference/.

Usage:
    python scripts/job_summary.py --out summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from edb.datasets.store import dataset_labels
from edb.reporting.collect import evaluate_run, list_run_ids, runs_root
from edb.reporting.compare import compare, load_reference
from edb.runners.predict import read_predictions, run_dir
from edb.settings import get_settings

RAW = ("n", "errors", "accuracy", "macro_f1", "nll", "brier", "ece", "aurc", "acc@80")


def _r(v: Any) -> Any:
    return round(v, 4) if isinstance(v, float) else v


def run_metrics(dataset: str, run_id: str, split: str, labels: list[str]) -> dict | None:
    ev = evaluate_run(dataset, run_id, split, labels)
    if ev is None:
        return None
    out = {k: _r(ev.raw.get(k)) for k in RAW}
    out["T"] = _r(ev.temperature)
    if ev.calibrated:
        out["nll_cal"], out["ece_cal"] = _r(ev.calibrated.get("nll")), \
            _r(ev.calibrated.get("ece"))
        pol = (ev.calibrated.get("threshold_policy") or {}).get("80") or {}
        out["cov80"] = {k: _r(pol.get(k)) for k in ("coverage", "accuracy")} if pol else None
    out["p50_ms"] = _r(ev.ops.get("latency_p50_ms"))
    out["in_tok"] = _r(ev.ops.get("input_tokens_mean"))
    return out


def reference_for(dataset: str, run_id: str, split: str) -> Path | None:
    role = run_id.rsplit("__", 1)[-1]
    folder = get_settings().configs_dir / "reference"
    for cand in (f"{dataset}__b1_decision1__{role}__{split}", f"{dataset}__b1_decision1__short__"
                                                             f"{split}"):
        path = folder / f"{cand}.jsonl.gz"
        if path.exists():
            return path
    return None


def summarize_runs() -> tuple[dict, dict]:
    runs: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    root = get_settings().artifacts_dir / "runs"
    for ds_dir in sorted(p for p in root.glob("*") if p.is_dir()) if root.exists() else []:
        dataset = ds_dir.name
        labels = dataset_labels(dataset)
        for run_id in list_run_ids(dataset):
            for split_dir in sorted((runs_root(dataset) / run_id).glob("*")):
                split = split_dir.name
                if not (split_dir / "manifest.json").exists():
                    continue
                m = run_metrics(dataset, run_id, split, labels)
                if m:
                    runs.setdefault(dataset, {}).setdefault(run_id, {})[split] = m
                ref_path = reference_for(dataset, run_id, split)
                if ref_path is not None:
                    other = read_predictions(run_dir(dataset, run_id, split) / "predictions.jsonl")
                    c = compare(load_reference(ref_path), other, labels)
                    c["reference"] = ref_path.name.split("__")[1] + "__" + \
                        ref_path.name.split("__")[2]
                    comparisons.setdefault(dataset, {}).setdefault(run_id, {})[split] = c
    return runs, comparisons


def summarize_probes() -> dict:
    out = {}
    for path in sorted((get_settings().artifacts_dir / "probes").glob("*/summary.json")):
        s = json.loads(path.read_text(encoding="utf-8"))
        out[s["target"]] = s
    return out


def summarize_training() -> dict:
    out = {}
    for path in sorted((get_settings().artifacts_dir / "training").glob("*/summary.json")):
        s = json.loads(path.read_text(encoding="utf-8"))
        log = path.parent / "train_log.jsonl"
        evals = []
        if log.exists():
            for line in log.read_text(encoding="utf-8").splitlines():
                e = json.loads(line)
                if "holdout" in e:
                    evals.append({"step": e["step"], "loss": e["loss"], "elapsed_s":
                                  e["elapsed_s"], **e["holdout"]})
        out[path.parent.name] = {
            "train_s": s.get("train_s"), "examples_seen": s.get("examples_seen"),
            "dropped_too_long": s.get("dropped_too_long"),
            "trainable_params": s.get("trainable_params"),
            "loss": s["config"]["loss"], "shuffle": s["config"]["shuffle"],
            "max_steps": s["config"]["max_steps"], "holdout": evals}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarise job artifacts")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary: dict[str, Any] = {}
    for name, fn in (("probes", summarize_probes), ("training", summarize_training)):
        try:
            summary[name] = fn()
        except Exception as exc:  # noqa: BLE001 - a partial summary beats none
            summary[f"{name}_error"] = f"{type(exc).__name__}: {exc}"
    try:
        summary["runs"], summary["compare"] = summarize_runs()
    except Exception as exc:  # noqa: BLE001
        summary["runs_error"] = f"{type(exc).__name__}: {exc}"
    args.out.write_text(json.dumps(summary, separators=(",", ":"), ensure_ascii=False),
                        encoding="utf-8")
    print(f"summary: {args.out.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
