"""Temperature scaling on probability outputs (fit on the calibration split only)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

PROB_FLOOR = 1e-6  # floor before taking logs, so one-hot outputs become finite logits


def to_logits(P: np.ndarray, floor: float = PROB_FLOOR) -> np.ndarray:
    return np.log(np.clip(P, floor, 1.0))


def apply_temperature(P: np.ndarray, temperature: float, floor: float = PROB_FLOOR) -> np.ndarray:
    z = to_logits(P, floor) / temperature
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(P: np.ndarray, y: np.ndarray, floor: float = PROB_FLOOR) -> float:
    rows = np.arange(len(y))

    def loss(log_t: float) -> float:
        Q = apply_temperature(P, float(np.exp(log_t)), floor)
        return float(-np.mean(np.log(np.clip(Q[rows, y], 1e-12, 1.0))))

    res = minimize_scalar(loss, bounds=(np.log(1e-2), np.log(1e3)), method="bounded")
    return float(np.exp(res.x))
