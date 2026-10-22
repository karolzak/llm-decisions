"""Replication-study report: Decision-1 vs. its frozen base and the trained arms.

Combines Decision-1 / gpt-oss runs stored locally with the R-series results collected from
Azure ML job tags (scripts/aml_summaries.py), and builds the side-by-side probe report.

Usage:
    uv run --extra aml python scripts/aml_summaries.py --tag replication
    uv run python scripts/replication_report.py --tag replication
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from edb.datasets.store import dataset_labels
from edb.probing.report import _fmt, build_probe_report
from edb.reporting.compare import compare
from edb.reporting.summary import run_metrics
from edb.runners.predict import read_predictions, run_dir
from edb.settings import get_settings

# (dataset, catalog) columns; label shown in tables
COLUMNS = [("ace", "short", "ACE short"), ("ace", "verbose", "ACE verbose"),
           ("banking77", "short", "BANKING77")]
# (experiment id, label, where the results live)
METHODS = [
    ("b1_decision1", "Decision-1 (post-trained Qwen3.5-9B)", "local"),
    ("r2_qwen35_9b_letter", "Qwen3.5-9B frozen, letter readout", "jobs"),
    ("r2_qwen35_9b_letter_markdown", "Qwen3.5-9B frozen, markdown prompt", "jobs"),
    ("r2_qwen35_9b_letter_json", "Qwen3.5-9B frozen, JSON prompt", "jobs"),
    ("r2_qwen35_9b_codes", "Qwen3.5-9B frozen, letter codes (one pass)", "jobs"),
    ("r2_qwen35_9b_numbers", "Qwen3.5-9B frozen, numbers (one pass)", "jobs"),
    ("r2_qwen35_2b_letter", "Qwen3.5-2B frozen, letter readout", "jobs"),
    ("r3_ce_shuffle", "2B + LoRA, CE, shuffled options", "jobs"),
    ("r3_ce_fixed", "2B + LoRA, CE, fixed option order", "jobs"),
    ("r3_brier_shuffle", "2B + LoRA, Brier, shuffled options", "jobs"),
    ("r9_qwen9b", "R9: 9B + LoRA decision training", "jobs"),
    ("r9b_kl", "R9b: R9 + none option, KL 1.0", "jobs"),
    ("r9b_nokl", "R9b: R9 + none option, no KL", "jobs"),
    ("r15_typed", "R15: R9b + score/noul training", "jobs"),
    ("r17_routing", "R17: R9b-kl + Bitext banking/support routing", "jobs"),
    ("c6_oss_letter_readout", "gpt-oss-20b letter readout (C6, context)", "local"),
]
DETAIL = ("accuracy", "macro_f1", "nll", "ece", "T", "nll_cal", "ece_cal", "acc@80", "p50_ms",
          "in_tok")
AGREE = ("acc_run", "top1_agreement", "chance_agreement", "kappa", "both_wrong",
         "same_answer_when_both_wrong", "error_correlation_phi", "p_gold_pearson",
         "p_gold_spearman", "mean_js_divergence", "prediction_mix_run")
PROBE_TARGETS = ["decision-1", "qwen3.5-9b", "r9-qwen9b", "r9b-kl", "r9b-nokl", "r15-typed",
                 "r17-routing", "qwen3.5-2b",
                 "r3-ce-shuffle", "r3-ce-fixed", "r3-brier-shuffle"]


def load_job_summaries(tag: str) -> list[dict[str, Any]]:
    folder = get_settings().artifacts_dir / "aml" / "summaries" / tag
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.json"))]


def collect(tag: str, split: str) -> tuple[dict, dict, dict, list, dict]:
    """metrics[(exp, dataset, catalog)], agreement[...], training[arm], job status rows,
    pairwise[(dataset, "run_a vs run_b")] (runs of one job compared with each other)."""
    metrics: dict[tuple, dict] = {}
    agreement: dict[tuple, dict] = {}
    training: dict[str, dict] = {}
    pairwise: dict[tuple, dict] = {}
    jobs = []
    for rec in load_job_summaries(tag):
        s = rec.get("summary") or {}
        jobs.append((rec["run_id"], rec["status"], rec.get("exit")))
        for ds, pairs in (s.get("pairwise") or {}).items():
            for pair, splits in pairs.items():
                if split in splits:
                    pairwise[(ds, pair)] = splits[split]
        for ds, runs in (s.get("runs") or {}).items():
            for run_id, splits in runs.items():
                exp, cat = run_id.rsplit("__", 1)
                if split in splits:
                    metrics[(exp, ds, cat)] = splits[split]
        for ds, runs in (s.get("compare") or {}).items():
            for run_id, splits in runs.items():
                exp, cat = run_id.rsplit("__", 1)
                if split in splits:
                    agreement[(exp, ds, cat)] = splits[split]
        training.update(s.get("training") or {})
    for exp, _, where in METHODS:
        if where != "local":
            continue
        for ds, cat, _ in COLUMNS:
            run_id = f"{exp}__{cat}"
            if (run_dir(ds, run_id, split) / "manifest.json").exists():
                m = run_metrics(ds, run_id, split, dataset_labels(ds))
                if m:
                    metrics[(exp, ds, cat)] = m
                if exp != "b1_decision1":
                    ref_id = f"b1_decision1__{cat}"
                    ref = read_predictions(run_dir(ds, ref_id, split) / "predictions.jsonl")
                    other = read_predictions(run_dir(ds, run_id, split) / "predictions.jsonl")
                    agreement[(exp, ds, cat)] = compare(ref, other, dataset_labels(ds))
    return metrics, agreement, training, jobs, pairwise


def table(header: list[str], rows: list[list[Any]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header),
            *("| " + " | ".join(_fmt(v) for v in r) + " |" for r in rows)]


def build(tag: str, split: str) -> Path:
    metrics, agreement, training, jobs, pairwise = collect(tag, split)
    lines = [f"# Replication study report (`{split}`)", "",
             "Generated from local runs (Decision-1, gpt-oss) and Azure ML job summaries "
             f"(tag `{tag}`). Design: [docs/replication.md](../../docs/replication.md).", ""]
    lines += ["## Jobs", "", *table(["job", "status", "exit"], [list(j) for j in jobs]), ""]
    lines += ["## Accuracy", "", *table(
        ["method", *(c[2] for c in COLUMNS)],
        [[label, *((metrics.get((exp, ds, cat)) or {}).get("accuracy") for ds, cat, _ in
                   COLUMNS)] for exp, label, _ in METHODS]), ""]
    for ds, cat, col in COLUMNS:
        rows = [[label, *(metrics[(exp, ds, cat)].get(k) for k in DETAIL)]
                for exp, label, _ in METHODS if (exp, ds, cat) in metrics]
        if rows:
            lines += [f"## {col}: accuracy, calibration, cost", "",
                      "`T`, `nll_cal`, `ece_cal`: temperature fit on the calibration split. "
                      "`p50_ms`: Decision-1 over the internet, local models GPU time.", "",
                      *table(["method", *DETAIL], rows), ""]
    for ds, cat, col in COLUMNS:
        rows = [[label, *(agreement[(exp, ds, cat)].get(k) for k in AGREE)]
                for exp, label, _ in METHODS if (exp, ds, cat) in agreement]
        if rows:
            lines += [f"## {col}: agreement with Decision-1, example by example", "",
                      "`kappa`: agreement beyond chance; `same_answer_when_both_wrong`: shared "
                      "mistakes; `p_gold_*`: correlation of the probability given to the right "
                      "answer.", "", *table(["method", *AGREE], rows), ""]
    if pairwise:
        rows = [[ds, pair, c.get("n"), c.get("acc_reference"), c.get("acc_run"),
                 c.get("top1_agreement"), c.get("kappa"), c.get("mean_js_divergence")]
                for (ds, pair), c in sorted(pairwise.items())]
        lines += ["## Same model, different runs (e.g. prompt templates)", "",
                  "Runs of one job compared example by example: how much the decisions move "
                  "when only the prompt changes.", "",
                  *table(["dataset", "pair", "n", "acc A", "acc B", "agreement", "kappa",
                          "mean JS"], rows), ""]
    if training:
        rows = []
        for arm, t in sorted(training.items()):
            last = (t.get("holdout") or [{}])[-1]
            rows.append([arm, t["loss"], t["shuffle"], t["max_steps"], t.get("train_s"),
                         last.get("accuracy"), last.get("nll"), last.get("ece"),
                         last.get("mean_confidence"), last.get("pred_first_share"),
                         last.get("gold_first_share")])
        lines += ["## Training arms (held-out corpus rows, options shuffled)", "", *table(
            ["arm", "loss", "shuffle", "steps", "train_s", "acc", "nll", "ece", "mean_conf",
             "pred_first", "gold_first"], rows), ""]
        for arm, t in sorted(training.items()):
            lines += [f"`{arm}` held-out accuracy by step: " + ", ".join(
                f"{h['step']}: {h['accuracy']}" for h in t.get("holdout") or [])]
        lines.append("")
    targets = [t for t in PROBE_TARGETS
               if (get_settings().artifacts_dir / "probes" / t / "summary.json").exists()]
    out_dir = get_settings().artifacts_dir / "reports" / "replication"
    out_dir.mkdir(parents=True, exist_ok=True)
    if targets:
        probe_md = build_probe_report(targets, out_dir / "probe-report.md")
        lines += ["## Probes", "", f"Side by side for {', '.join(targets)}: "
                  f"[{probe_md.name}]({probe_md.name}).", ""]
    out = out_dir / f"report-{split}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / f"data-{split}.json").write_text(json.dumps(
        {"metrics": {"|".join(k): v for k, v in metrics.items()},
         "agreement": {"|".join(k): v for k, v in agreement.items()},
         "training": training,
         "pairwise": {"|".join(k): v for k, v in pairwise.items()}}, indent=1),
        encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the replication-study report")
    parser.add_argument("--tag", default="replication")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    print(build(args.tag, args.split))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
