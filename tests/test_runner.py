import asyncio
import json
import random

import pytest

from edb.providers import registry
from edb.providers.base import CallResult, DecisionMethod
from edb.reporting.report import build_report
from edb.runners.predict import read_predictions, resolve_runs, run_dir, run_split
from edb.schemas import Usage


class FakeMethod(DecisionMethod):
    """Deterministic pseudo-classifier: correct with probability `accuracy`."""

    type = "fake"
    calls = 0

    async def call(self, example):
        FakeMethod.calls += 1
        rng = random.Random(example.id)
        correct = rng.random() < self.ctx.params.get("accuracy", 0.8)
        label = example.label if correct else rng.choice(
            [v for v in self.values if v != example.label])
        probs = {v: 0.1 for v in self.values}
        probs[label] = 0.8
        return CallResult(probabilities=probs, prob_source="model",
                          usage=Usage(input_tokens=100, output_tokens=1))


@pytest.fixture
def fake_registry(monkeypatch):
    original = registry._registry
    monkeypatch.setattr(registry, "_registry",
                        lambda include_local=False: {**original(), "fake": FakeMethod})


def test_run_resume_and_report(sandbox, fake_registry):
    specs = resolve_runs("fake_exp")
    assert [s.run_id for s in specs] == ["fake_exp__short", "fake_exp__verbose"]
    spec = specs[0]

    FakeMethod.calls = 0
    s1 = asyncio.run(run_split(spec, "dev", limit=5))
    assert s1["n"] == 5 and FakeMethod.calls == 5
    s2 = asyncio.run(run_split(spec, "dev"))  # resumes: only the remaining examples
    assert FakeMethod.calls == s2["n"]
    asyncio.run(run_split(spec, "dev"))
    assert FakeMethod.calls == s2["n"]  # nothing left to do

    preds = read_predictions(run_dir("toy", spec.run_id, "dev") / "predictions.jsonl")
    assert len(preds) == s2["n"]
    p = next(iter(preds.values()))
    assert p.cost_usd == pytest.approx((100 * 1.0 + 1 * 2.0) / 1e6)

    for s in specs:
        for split in ("dev", "calibration"):
            asyncio.run(run_split(s, split))
    report = build_report("toy", "dev")
    text = report.read_text(encoding="utf-8")
    assert "fake_exp__short" in text and "fake_exp__verbose" in text
    metrics = json.loads((report.parent / "metrics.json").read_text())
    ev = metrics["product"]["fake_exp__short"]
    assert ev["temperature"] is not None and ev["calibrated"]["nll"] <= ev["raw"]["nll"] + 0.05
    assert (report.parent / "reliability_product.png").exists()


def test_config_change_requires_force(sandbox, fake_registry):
    spec = resolve_runs("fake_exp", catalogs=["short"])[0]
    asyncio.run(run_split(spec, "dev", limit=3))
    spec.experiment.method.params["accuracy"] = 0.5
    with pytest.raises(RuntimeError, match="different config"):
        asyncio.run(run_split(spec, "dev", limit=3))
    out = asyncio.run(run_split(spec, "dev", limit=3, force=True))
    assert out["n"] == 3

def test_latency_benchmark_sets_up_once(sandbox, fake_registry, monkeypatch):
    from edb.runners.latency import benchmark

    setups = {"n": 0}
    original = FakeMethod.setup

    async def counting_setup(self):
        setups["n"] += 1
        await original(self)

    monkeypatch.setattr(FakeMethod, "setup", counting_setup)
    spec = resolve_runs("fake_exp", catalogs=["short"])[0]
    summary = asyncio.run(benchmark(spec, "dev", n=6, warmup=2, reps=2, concurrencies=(1, 3)))
    assert setups["n"] == 1
    for conc in ("1", "3"):
        s = summary["by_concurrency"][conc]
        assert s["calls"] == 12 and s["errors"] == 0 and s["p50_ms"] is not None
