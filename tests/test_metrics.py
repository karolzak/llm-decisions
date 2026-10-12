import numpy as np
import pytest

from edb.calibration.metrics import SelectiveCurve, brier, ece, evaluate, nll
from edb.calibration.temperature import apply_temperature, fit_temperature
from edb.providers.parsing import parse_label

VALUES = ["Compliant", "Non-Compliant", "Not-Applicable"]


@pytest.mark.parametrize("text,label,status", [
    ("Compliant", "Compliant", "exact"),
    ("  **non-compliant.** ", "Non-Compliant", "normalized"),
    ("The answer is Non-Compliant", "Non-Compliant", "search_unique"),
    ("Not Compliant? No: final answer Not-Applicable", "Not-Applicable", "search_last"),
    ("Compliant at first, but actually Non-Compliant", "Non-Compliant", "search_last"),
    ("I cannot tell", None, "failed"),
    ("", None, "failed"),
])
def test_parse_label(text, label, status):
    parsed = parse_label(text, VALUES)
    assert (parsed.label, parsed.status) == (label, status)


def test_basic_metrics():
    P = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    y = np.array([0, 2])
    assert nll(P, y) == pytest.approx(-(np.log(0.7) + np.log(0.1)) / 2)
    assert brier(P, y) == pytest.approx(((0.09 + 0.04 + 0.01) + (0.01 + 0.64 + 0.81)) / 2)
    conf, correct = P.max(axis=1), np.array([True, False])
    assert ece(conf, correct, n_bins=10) == pytest.approx(0.5 * 0.3 + 0.5 * 0.8)


def test_selective_curve_is_tie_aware():
    # All confidences tied: accuracy at any coverage equals overall accuracy.
    curve = SelectiveCurve(np.ones(4), np.array([1, 1, 0, 0]))
    assert curve.accuracy_at_coverage(0.25) == pytest.approx(0.5)
    assert curve.aurc() == pytest.approx(0.5)
    # Distinct confidences: most confident first.
    curve = SelectiveCurve(np.array([0.9, 0.8, 0.7, 0.6]), np.array([1, 1, 0, 0]))
    assert curve.accuracy_at_coverage(0.5) == pytest.approx(1.0)
    assert curve.accuracy_at_coverage(0.75) == pytest.approx(2 / 3)


def test_evaluate_counts_invalid_as_errors():
    P = np.array([[0.9, 0.05, 0.05], [1 / 3, 1 / 3, 1 / 3]])
    res = evaluate(P, np.array([0, 0]), np.array([True, False]))
    assert res["accuracy"] == 0.5 and res["errors"] == 1


def test_temperature_recovers_overconfidence():
    rng = np.random.default_rng(0)
    n, K, true_T = 4000, 3, 3.0
    logits = rng.normal(size=(n, K)) * 2
    calibrated = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y = np.array([rng.choice(K, p=p) for p in calibrated])
    overconfident = apply_temperature(calibrated, 1 / true_T)
    T = fit_temperature(overconfident, y)
    assert T == pytest.approx(true_T, rel=0.15)
    assert nll(apply_temperature(overconfident, T), y) < nll(overconfident, y)


def test_temperature_on_one_hot_matches_accuracy():
    P = np.eye(3)[[0, 1, 2, 0, 1, 2, 0, 1, 2, 0]]
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 1, 1])  # 8/10 correct
    Q = apply_temperature(P, fit_temperature(P, y))
    assert Q.max(axis=1).mean() == pytest.approx(0.8, abs=0.01)
