"""Classification + calibration metrics.

All functions take `P` (n x K probabilities, rows sum to 1) and `y` (n gold indices).
Failed predictions should be passed as uniform rows with `valid=False`; they always count
as errors for accuracy-type metrics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import f1_score

EPS = 1e-12
COVERAGES = (0.8, 0.9, 0.95)


def predictions(
    P: np.ndarray, valid: np.ndarray, preferred: np.ndarray | None = None
) -> np.ndarray:
    """Argmax per row; on exact ties a `preferred` index (stored prediction) wins if tied."""
    pred = P.argmax(axis=1)
    if preferred is not None:
        rows = np.arange(len(P))
        ok = (preferred >= 0) & (P[rows, np.clip(preferred, 0, None)] == P.max(axis=1))
        pred = np.where(ok, preferred, pred)
    return np.where(valid, pred, -1)


def nll(P: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], EPS, 1.0))))


def brier(P: np.ndarray, y: np.ndarray) -> float:
    onehot = np.eye(P.shape[1])[y]
    return float(np.mean(np.sum((P - onehot) ** 2, axis=1)))


def reliability_bins(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> list[dict]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    bins = []
    for b in range(n_bins):
        m = idx == b
        bins.append({
            "lo": float(edges[b]), "hi": float(edges[b + 1]), "count": int(m.sum()),
            "confidence": float(conf[m].mean()) if m.any() else None,
            "accuracy": float(correct[m].mean()) if m.any() else None,
        })
    return bins


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    n = len(conf)
    return float(sum(
        b["count"] / n * abs(b["accuracy"] - b["confidence"])
        for b in reliability_bins(conf, correct, n_bins) if b["count"]
    ))


class SelectiveCurve:
    """Tie-aware risk/coverage: items with equal confidence are accepted pro rata."""

    def __init__(self, conf: np.ndarray, correct: np.ndarray):
        order = np.argsort(-conf, kind="stable")
        c, k = conf[order], correct[order].astype(float)
        levels, starts = np.unique(-c, return_index=True)  # descending confidence groups
        ends = np.append(starts[1:], len(c))
        self.n = len(c)
        self.levels = -levels
        self.cum_n = ends.astype(float)
        group_correct = np.array([k[s:e].sum() for s, e in zip(starts, ends, strict=True)])
        self.cum_correct = np.cumsum(group_correct)
        self.group_acc = group_correct / (ends - starts)

    def correct_at(self, k: np.ndarray | float) -> np.ndarray:
        k = np.asarray(k, dtype=float)
        g = np.searchsorted(self.cum_n, k, side="left")
        g = np.clip(g, 0, len(self.cum_n) - 1)
        prev_n = np.where(g > 0, self.cum_n[g - 1], 0.0)
        prev_c = np.where(g > 0, self.cum_correct[g - 1], 0.0)
        return prev_c + (k - prev_n) * self.group_acc[g]

    def accuracy_at_coverage(self, coverage: float) -> float:
        k = max(coverage * self.n, 1e-9)
        return float(self.correct_at(k) / k)

    def curve(self, points: int = 101) -> list[dict[str, float]]:
        covs = np.linspace(1.0 / self.n, 1.0, points)
        acc = self.correct_at(covs * self.n) / (covs * self.n)
        return [{"coverage": float(c), "risk": float(1 - a)} for c, a in zip(covs, acc,
                                                                              strict=True)]

    def aurc(self) -> float:
        k = np.arange(1, self.n + 1, dtype=float)
        return float(np.mean(1.0 - self.correct_at(k) / k))


def threshold_for_coverage(conf: np.ndarray, coverage: float) -> float:
    """Confidence threshold whose `conf >= t` policy accepts ceil(coverage * n) items.

    Ties at the threshold can push realized coverage above the target.
    """
    ordered = np.sort(conf)[::-1]
    k = max(1, int(np.ceil(coverage * len(ordered) - 1e-9)))
    return float(ordered[k - 1])


def evaluate(
    P: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    thresholds: dict[float, float] | None = None,
    n_bins: int = 15,
    preferred: np.ndarray | None = None,
) -> dict[str, Any]:
    pred = predictions(P, valid, preferred)
    correct = pred == y
    conf = np.where(valid, P.max(axis=1), 0.0)
    K = P.shape[1]
    curve = SelectiveCurve(conf, correct)
    out: dict[str, Any] = {
        "n": int(len(y)),
        "errors": int((~valid).sum()),
        "accuracy": float(correct.mean()),
        "macro_f1": float(f1_score(y, pred, labels=list(range(K)), average="macro",
                                   zero_division=0)),
        "nll": nll(P, y),
        "brier": brier(P, y),
        "ece": ece(conf, correct, n_bins),
        "mean_confidence": float(conf.mean()),
        "aurc": curve.aurc(),
        **{f"acc@{int(c * 100)}": curve.accuracy_at_coverage(c) for c in COVERAGES},
        "reliability": reliability_bins(conf, correct, n_bins),
        "risk_coverage": curve.curve(),
    }
    if thresholds:
        policy = {}
        for cov, thr in thresholds.items():
            accept = conf >= thr
            policy[f"{int(cov * 100)}"] = {
                "threshold": thr,
                "coverage": float(accept.mean()),
                "accuracy": float(correct[accept].mean()) if accept.any() else None,
            }
        out["threshold_policy"] = policy
    return out
