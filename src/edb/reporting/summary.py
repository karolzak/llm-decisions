"""Compact per-run metrics shared by job summaries and the replication report."""

from __future__ import annotations

from typing import Any

from edb.reporting.collect import evaluate_run

RAW = ("n", "errors", "accuracy", "macro_f1", "nll", "brier", "ece", "aurc", "acc@80")


def _r(v: Any) -> Any:
    return round(v, 4) if isinstance(v, float) else v


def run_metrics(dataset: str, run_id: str, split: str, labels: list[str]) -> dict | None:
    """Raw metrics, temperature-scaled NLL/ECE (T fit on the run's calibration split) and the
    80%-coverage selective-prediction policy of one stored run split."""
    ev = evaluate_run(dataset, run_id, split, labels)
    if ev is None:
        return None
    out = {k: _r(ev.raw.get(k)) for k in RAW}
    out["T"] = _r(ev.temperature)
    if ev.calibrated:
        out["nll_cal"] = _r(ev.calibrated.get("nll"))
        out["ece_cal"] = _r(ev.calibrated.get("ece"))
        pol = (ev.calibrated.get("threshold_policy") or {}).get("80") or {}
        out["cov80"] = {k: _r(pol.get(k)) for k in ("coverage", "accuracy")} if pol else None
    out["p50_ms"] = _r(ev.ops.get("latency_p50_ms"))
    out["in_tok"] = _r(ev.ops.get("input_tokens_mean"))
    return out
