"""Load run outputs and turn them into metrics."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from edb.calibration.metrics import COVERAGES, evaluate, threshold_for_coverage
from edb.calibration.temperature import apply_temperature, fit_temperature
from edb.runners.predict import read_predictions
from edb.schemas import Prediction
from edb.settings import get_settings


@dataclass
class RunSplit:
    run_id: str
    split: str
    manifest: dict[str, Any]
    predictions: dict[str, Prediction]

    @property
    def leaderboards(self) -> list[str]:
        return self.manifest.get("leaderboards", [])

    @property
    def config(self) -> dict[str, Any]:
        return self.manifest["config"]

    @property
    def wall_s(self) -> float:
        return sum(s.get("wall_s", 0.0) for s in self.manifest["sessions"])

    @property
    def submitted(self) -> int:
        return sum(s.get("submitted", 0) for s in self.manifest["sessions"])


def runs_root(dataset: str) -> Path:
    return get_settings().artifacts_dir / "runs" / dataset


def load_run_split(dataset: str, run_id: str, split: str) -> RunSplit | None:
    d = runs_root(dataset) / run_id / split
    if not (d / "manifest.json").exists():
        return None
    return RunSplit(
        run_id=run_id,
        split=split,
        manifest=json.loads((d / "manifest.json").read_text(encoding="utf-8")),
        predictions=read_predictions(d / "predictions.jsonl"),
    )


def list_run_ids(dataset: str) -> list[str]:
    root = runs_root(dataset)
    return sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []


def to_arrays(
    preds: list[Prediction], labels: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index = {v: i for i, v in enumerate(labels)}
    K = len(labels)
    P = np.full((len(preds), K), 1.0 / K)
    valid = np.zeros(len(preds), dtype=bool)
    for i, p in enumerate(preds):
        if p.ok and p.probabilities:
            P[i] = [p.probabilities.get(v, 0.0) for v in labels]
            valid[i] = True
    y = np.array([index[p.gold] for p in preds])
    return P, y, valid


def _pct(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def operational(preds: list[Prediction], run: RunSplit) -> dict[str, Any]:
    ok = [p for p in preds if p.ok]
    lat = [p.latency_ms for p in ok if p.latency_ms is not None]
    ttft = [p.ttft_ms for p in ok if p.ttft_ms is not None]
    usages = [p.usage for p in preds if p.usage]
    costs = [p.cost_usd for p in preds if p.cost_usd is not None]
    return {
        "latency_p50_ms": _pct(lat, 50),
        "latency_p95_ms": _pct(lat, 95),
        "ttft_p50_ms": _pct(ttft, 50),
        "ttft_p95_ms": _pct(ttft, 95),
        "retried_frac": float(np.mean([p.attempts > 1 for p in preds])) if preds else None,
        "input_tokens_mean": float(np.mean([u.input_tokens for u in usages])) if usages else None,
        "output_tokens_mean": float(np.mean([u.output_tokens for u in usages])) if usages else None,
        "reasoning_tokens_mean": (
            float(np.mean([u.reasoning_tokens for u in usages])) if usages else None
        ),
        "cost_per_1k_usd": float(np.mean(costs) * 1000) if costs else None,
        "cost_total_usd": float(np.sum(costs)) if costs else None,
        "throughput_eps": run.submitted / run.wall_s if run.wall_s else None,
    }


@dataclass
class RunEvaluation:
    run_id: str
    split: str
    leaderboards: list[str]
    config: dict[str, Any]
    raw: dict[str, Any]
    calibrated: dict[str, Any] | None
    temperature: float | None
    calibration_n: int
    ops: dict[str, Any]
    example_ids: list[str] = field(default_factory=list)


def evaluate_run(
    dataset: str,
    run_id: str,
    split: str,
    labels: list[str],
    example_ids: set[str] | None = None,
    calibration_split: str = "calibration",
    min_calibration: int = 30,
) -> RunEvaluation | None:
    run = load_run_split(dataset, run_id, split)
    if run is None or not run.predictions:
        return None
    preds = [p for eid, p in sorted(run.predictions.items())
             if example_ids is None or eid in example_ids]
    if not preds:
        return None
    P, y, valid = to_arrays(preds, labels)

    temperature, calibrated, cal_n = None, None, 0
    cal = load_run_split(dataset, run_id, calibration_split) if split != calibration_split else None
    if cal is not None:
        cal_preds = [p for p in cal.predictions.values() if p.ok and p.probabilities]
        cal_n = len(cal_preds)
        if cal_n >= min_calibration:
            Pc, yc, _ = to_arrays(cal_preds, labels)
            temperature = fit_temperature(Pc, yc)
            conf_c = apply_temperature(Pc, temperature).max(axis=1)
            thresholds = {c: threshold_for_coverage(conf_c, c) for c in COVERAGES}
            calibrated = evaluate(apply_temperature(P, temperature), y, valid, thresholds)

    return RunEvaluation(
        run_id=run_id,
        split=split,
        leaderboards=run.leaderboards,
        config=run.config,
        raw=evaluate(P, y, valid),
        calibrated=calibrated,
        temperature=temperature,
        calibration_n=cal_n,
        ops=operational(preds, run),
        example_ids=[p.example_id for p in preds],
    )
