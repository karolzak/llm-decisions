"""Leaderboards, plots and the markdown report."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from edb.config import load_experiment_config  # noqa: E402
from edb.datasets.store import dataset_labels  # noqa: E402
from edb.reporting.collect import RunEvaluation, evaluate_run, list_run_ids  # noqa: E402
from edb.settings import get_settings  # noqa: E402

LEADERBOARDS = {
    "product": "Products and baselines: GPT-6 Luna, Decision-1, embeddings (A / B / S)",
    "controlled": "Open-weight model: gpt-oss-20b, same weights, different readout (C-series)",
}


def _supervision(ev: RunEvaluation) -> str:
    try:
        return load_experiment_config(ev.config["experiment"]["id"]).supervision
    except FileNotFoundError:
        return "?"


def _leaderboards(ev: RunEvaluation) -> list[str]:
    """Current experiment config wins, so regrouping runs needs no re-run (not hashed)."""
    try:
        return load_experiment_config(ev.config["experiment"]["id"]).leaderboards
    except FileNotFoundError:
        return ev.leaderboards


def _row(ev: RunEvaluation) -> dict[str, Any]:
    cfg = ev.config
    cal = ev.calibrated or {}
    model = (cfg.get("model") or {}).get("name")
    return {
        "run": ev.run_id,
        "method": cfg["experiment"]["method"]["type"],
        "model": model,
        "supervision": _supervision(ev),
        "labels": cfg["catalog_role"],
        "n": ev.raw["n"],
        "err": ev.raw["errors"],
        "acc": ev.raw["accuracy"],
        "macro_f1": ev.raw["macro_f1"],
        "nll": ev.raw["nll"],
        "brier": ev.raw["brier"],
        "ece": ev.raw["ece"],
        "aurc": ev.raw["aurc"],
        "acc@80": ev.raw["acc@80"],
        "acc@90": ev.raw["acc@90"],
        "T": ev.temperature,
        "nll_cal": cal.get("nll"),
        "ece_cal": cal.get("ece"),
        "p50_ms": ev.ops["latency_p50_ms"],
        "p95_ms": ev.ops["latency_p95_ms"],
        "ttft_p50_ms": ev.ops["ttft_p50_ms"],
        "in_tok": ev.ops["input_tokens_mean"],
        "out_tok": ev.ops["output_tokens_mean"],
        "usd_per_1k": ev.ops["cost_per_1k_usd"],
    }


def _fmt(df: pd.DataFrame) -> str:
    def f(v: Any) -> str:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "-"
        if isinstance(v, float):
            return f"{v:.0f}" if abs(v) >= 100 else f"{v:.3f}"
        return str(v)

    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def _plot_reliability(evals: list[RunEvaluation], path: Path) -> None:
    n = len(evals)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4), squeeze=False)
    for ax, ev in zip(axes[0], evals, strict=True):
        for result, style, name in ((ev.raw, "o-", "raw"), (ev.calibrated, "s--", "temp-scaled")):
            if not result:
                continue
            bins = [b for b in result["reliability"] if b["count"]]
            ax.plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], style,
                    ms=3, label=f"{name} ECE={result['ece']:.3f}")
        ax.plot([0, 1], [0, 1], color="grey", lw=0.8)
        ax.set_title(ev.run_id, fontsize=8)
        ax.set_xlabel("confidence")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=6, loc="upper left")
    axes[0][0].set_ylabel("accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_risk_coverage(evals: list[RunEvaluation], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = plt.get_cmap("tab20").colors
    for i, ev in enumerate(evals):
        result = ev.calibrated or ev.raw
        pts = result["risk_coverage"]
        ax.plot([p["coverage"] for p in pts], [p["risk"] for p in pts],
                color=colors[i % len(colors)], linestyle=("-", "--", ":")[i // 7 % 3],
                label=f"{ev.run_id} (AURC={result['aurc']:.3f})")
    ax.set_xlabel("coverage")
    ax.set_ylabel("risk (error rate)")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=6, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def build_report(dataset: str, split: str = "dev", intersect: bool = True) -> Path:
    labels = dataset_labels(dataset)
    out = get_settings().artifacts_dir / "reports" / dataset / split
    out.mkdir(parents=True, exist_ok=True)

    evals = [ev for rid in list_run_ids(dataset)
             if (ev := evaluate_run(dataset, rid, split, labels)) is not None]

    lines = [
        f"# Benchmark report: `{dataset}` / `{split}`",
        "",
        f"Generated {datetime.now(UTC).isoformat(timespec='seconds')}. Temperature `T` is fit "
        "on each run's `calibration` split (never on the evaluated split); `*_cal` columns "
        "use it. Failed requests count as errors with a uniform distribution. "
        "`supervised` runs fit on the dataset's train labels; `zero_shot` runs do not.",
        "",
    ]
    all_metrics: dict[str, Any] = {}
    for board, title in LEADERBOARDS.items():
        members = [ev for ev in evals if board in _leaderboards(ev)]
        lines += [f"## {title}", ""]
        if not members:
            lines += ["_No runs yet._", ""]
            continue
        if intersect:
            common = set.intersection(*(set(ev.example_ids) for ev in members))
            members = [ev for rid in [m.run_id for m in members]
                       if (ev := evaluate_run(dataset, rid, split, labels, common)) is not None]
            lines += [f"Evaluated on the {len(common)} examples shared by all runs.", ""]
            if not members:
                lines += ["_Runs share no examples; use --no-intersect._", ""]
                continue
        df = pd.DataFrame([_row(ev) for ev in members]).sort_values("acc", ascending=False)
        df.to_csv(out / f"leaderboard_{board}.csv", index=False)
        lines += [_fmt(df), ""]

        policy_rows = []
        for ev in members:
            for cov, p in ((ev.calibrated or {}).get("threshold_policy") or {}).items():
                policy_rows.append({"run": ev.run_id, "target_cov": int(cov) / 100,
                                    "threshold": p["threshold"], "coverage": p["coverage"],
                                    "accuracy": p["accuracy"]})
        if policy_rows:
            lines += ["**Selective prediction** (threshold fit on calibration split):", "",
                      _fmt(pd.DataFrame(policy_rows)), ""]

        _plot_reliability(members, out / f"reliability_{board}.png")
        _plot_risk_coverage(members, out / f"risk_coverage_{board}.png")
        lines += [f"![reliability](reliability_{board}.png)", "",
                  f"![risk-coverage](risk_coverage_{board}.png)", ""]
        all_metrics[board] = {ev.run_id: {"raw": ev.raw, "calibrated": ev.calibrated,
                                          "temperature": ev.temperature, "ops": ev.ops}
                              for ev in members}

    latency = latency_section(dataset)
    if latency:
        lines += ["## Latency benchmark", "", latency, ""]

    (out / "metrics.json").write_text(json.dumps(all_metrics, indent=1), encoding="utf-8")
    report = out / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def latency_section(dataset: str) -> str | None:
    root = get_settings().artifacts_dir / "latency" / dataset
    rows = []
    for summary in sorted(root.glob("*/summary.json")):
        data = json.loads(summary.read_text(encoding="utf-8"))
        for conc, s in data["by_concurrency"].items():
            rows.append({"run": data["run_id"], "concurrency": int(conc), **s})
    return _fmt(pd.DataFrame(rows)) if rows else None
