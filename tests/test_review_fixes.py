"""Regression tests for the logic-review findings."""

import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest

from edb.calibration.metrics import threshold_for_coverage
from edb.config import load_dataset_config, load_label_catalog, load_model_config
from edb.datasets.store import prepare_dataset
from edb.providers.azure_openai_ar import AzureOpenAIAutoregressive
from edb.providers.base import CallResult, DecisionMethod, MethodContext, normalize
from edb.runners.predict import read_predictions, resolve_runs, run_dir, run_lock, run_split
from edb.schemas import Example

EXAMPLE = Example(id="x", state="s", label="beta", source_split="train", source_index=0)


@pytest.fixture
def ctx(sandbox):
    ds = load_dataset_config("toy")
    return MethodContext(run_id="r", dataset=ds,
                         catalog=load_label_catalog(ds, "short", ["alpha", "beta", "gamma"]),
                         model=load_model_config("fake-model"))


def test_threshold_selects_ceil_order_statistic():
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    assert threshold_for_coverage(conf, 0.5) == 0.8
    assert threshold_for_coverage(conf, 0.75) == 0.7
    assert threshold_for_coverage(conf, 1.0) == 0.6


@pytest.mark.parametrize("probs", [{}, {"alpha": -0.2, "beta": 1.2}, {"alpha": 0, "beta": 0},
                                   {"alpha": float("nan")}, {"delta": 1.0}])
def test_normalize_rejects_malformed(probs):
    with pytest.raises(ValueError):
        normalize(probs, ["alpha", "beta", "gamma"])


class _Static(DecisionMethod):
    type = "static"

    def __init__(self, ctx, result):
        super().__init__(ctx)
        self.result = result

    async def call(self, example):
        return self.result


def test_prediction_is_argmax_and_disagreement_recorded(ctx):
    res = CallResult(predicted="beta", probabilities={"alpha": 0.6, "beta": 0.4},
                     prob_source="model")
    pred = asyncio.run(_Static(ctx, res).predict(EXAMPLE))
    assert pred.predicted == "alpha" and pred.extras["provider_choice"] == "beta"


def test_tie_keeps_provider_choice(ctx):
    res = CallResult(predicted="beta", probabilities={"alpha": 0.45, "beta": 0.45, "gamma": 0.1},
                     prob_source="model")
    pred = asyncio.run(_Static(ctx, res).predict(EXAMPLE))
    assert pred.predicted == "beta" and "provider_choice" not in pred.extras


def test_metrics_honour_stored_choice_on_ties():
    from edb.calibration.metrics import predictions
    P = np.array([[0.45, 0.45, 0.1], [0.45, 0.45, 0.1], [0.7, 0.2, 0.1]])
    valid = np.array([True, True, True])
    assert predictions(P, valid, np.array([1, -1, 1])).tolist() == [1, 0, 0]


def test_missing_distribution_is_an_error(ctx):
    pred = asyncio.run(_Static(ctx, CallResult(predicted="beta")).predict(EXAMPLE))
    assert not pred.ok


def _fake_response(status, text):
    return SimpleNamespace(status=status, output_text=text, usage=None, id="r1",
                           incomplete_details={"reason": "max_output_tokens"})


def test_incomplete_response_is_error(ctx, monkeypatch):
    m = AzureOpenAIAutoregressive(ctx)
    m.stream = False

    async def create(**kwargs):
        return _fake_response("incomplete", "alpha")

    monkeypatch.setattr(m.client.responses, "create", create)
    pred = asyncio.run(m.predict(EXAMPLE))
    assert not pred.ok and "incomplete" in pred.error


@pytest.mark.parametrize("conf,ok,top", [(0.1, True, "beta"), (85, True, 0.85),
                                         (-0.2, False, None), (1e9, False, None)])
def test_verbalized_confidence_validation(ctx, conf, ok, top):
    ctx.params = {"confidence": "verbalized"}
    m = AzureOpenAIAutoregressive(ctx)
    res = CallResult()
    m._parse_verbalized(json.dumps({"label": "beta", "confidence": conf}), res)
    assert (res.error is None) == ok
    if ok:
        assert max(res.probabilities, key=res.probabilities.get) == "beta"
        if isinstance(top, float):
            assert res.probabilities["beta"] == pytest.approx(top)


def test_lock_blocks_second_writer(sandbox, tmp_path):
    out = tmp_path / "x"
    out.mkdir()
    with run_lock(out):
        with pytest.raises(RuntimeError, match="locked"):
            with run_lock(out):
                pass
    assert not (out / ".lock").exists()


def test_predictions_without_manifest_refused(sandbox):
    spec = resolve_runs("fake_exp", catalogs=["short"])[0]
    out = run_dir("toy", spec.run_id, "dev")
    out.mkdir(parents=True)
    (out / "predictions.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="without a manifest"):
        asyncio.run(run_split(spec, "dev", limit=2))


def test_truncated_last_line_is_ignored(tmp_path):
    p = tmp_path / "predictions.jsonl"
    good = json.dumps({"run_id": "r", "example_id": "a", "gold": "x"})
    p.write_text(good + "\n" + good[:20], encoding="utf-8")
    assert list(read_predictions(p)) == ["a"]
    p.write_text(good[:20] + "\n" + good + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_predictions(p)


def test_split_policy_change_requires_resplit(sandbox):
    prepare_dataset("toy")
    cfg_path = sandbox / "configs" / "datasets" / "toy.yaml"
    cfg_path.write_text(cfg_path.read_text().replace("seed: 7", "seed: 8"), encoding="utf-8")
    with pytest.raises(RuntimeError, match="split policy changed"):
        prepare_dataset("toy")


def test_auto_labels_ignore_sealed_split(sandbox):
    test_csv = sandbox / "src" / "test.csv"
    test_csv.write_text(test_csv.read_text() + "late,t999,delta\n", encoding="utf-8")
    with pytest.raises(ValueError, match="delta"):
        prepare_dataset("toy")
