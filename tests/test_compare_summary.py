"""Comparison with a compact reference, and job-summary transport through tags."""

import importlib.util
import json
import sys

import pytest

from edb.reporting.compare import compare, export_reference, load_reference
from edb.schemas import Prediction
from edb.settings import REPO_ROOT

LABELS = ["a", "b", "c"]


def pred(i, gold, predicted, probs):
    return Prediction(run_id="r", example_id=f"e{i}", gold=gold, predicted=predicted,
                      probabilities=dict(zip(LABELS, probs, strict=True)), prob_source="model")


def test_compare_agreement_and_reference_roundtrip(tmp_path):
    ref = {f"e{i}": pred(i, "a", p, [0.8 if p == "a" else 0.1, 0.1 if p == "a" else 0.8, 0.1])
           for i, p in enumerate(["a", "a", "b", "b"])}
    other = {f"e{i}": pred(i, "a", p, [0.7 if p == "a" else 0.2, 0.2 if p == "a" else 0.7, 0.1])
             for i, p in enumerate(["a", "b", "b", "a"])}
    c = compare(ref, other, LABELS)
    assert c["n"] == 4 and c["top1_agreement"] == 0.5
    assert c["acc_reference"] == 0.5 and c["acc_run"] == 0.5
    assert c["both_wrong"] == 0.25 and c["same_answer_when_both_wrong"] == 1.0
    assert c["mean_js_divergence"] is not None
    path = tmp_path / "ref.jsonl.gz"
    assert export_reference(ref, LABELS, path) == 4
    again = compare(load_reference(path), other, LABELS)
    assert again["kappa"] == c["kappa"] and again["p_gold_pearson"] == c["p_gold_pearson"]


def test_compact_reference_for_many_labels(tmp_path):
    labels = [f"l{i}" for i in range(20)]
    probs = {v: 0.05 for v in labels}
    ref = {"e0": Prediction(run_id="r", example_id="e0", gold="l1", predicted="l2",
                            probabilities=probs)}
    path = tmp_path / "ref.jsonl.gz"
    export_reference(ref, labels, path)
    row = load_reference(path)["e0"]
    assert set(row.probabilities) == {"l1", "l2"}
    c = compare(load_reference(path), ref, labels)
    assert c["n"] == 1 and c["mean_js_divergence"] is None  # partial distributions


def test_shipped_references_load():
    for path in (REPO_ROOT / "configs" / "reference").glob("*.jsonl.gz"):
        rows = load_reference(path)
        assert rows and all(p.predicted for p in rows.values())


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_summary_chunks_reassemble(monkeypatch):
    summary = {"runs": {"ace": {"x": {"test": {"accuracy": 0.5}}}}, "pad": "z" * 9000}
    text = json.dumps(summary)
    chunks = [text[i:i + 4000] for i in range(0, len(text), 4000)]
    tags = {"edb_summary_chunks": str(len(chunks)),
            **{f"edb_summary_{i:03d}": c for i, c in enumerate(chunks)}}
    pytest.importorskip("azure.ai.ml")
    assert _load_script("aml_summaries").assemble(tags) == summary
    assert _load_script("aml_summaries").assemble({}) is None


def test_job_summary_on_sandbox(sandbox, tmp_path, monkeypatch):
    import asyncio

    from test_runner import FakeMethod

    from edb.providers import registry
    from edb.runners.predict import resolve_runs, run_split

    original = registry._registry
    monkeypatch.setattr(registry, "_registry",
                        lambda include_local=False: {**original(), "fake": FakeMethod})
    for spec in resolve_runs("fake_exp", "toy", ["short", "verbose"]):
        asyncio.run(run_split(spec, "dev", limit=12))
    out = tmp_path / "s.json"
    job_summary = _load_script("job_summary")
    monkeypatch.setattr(sys, "argv", ["job_summary", "--out", str(out)])
    assert job_summary.main() == 0
    s = json.loads(out.read_text())
    assert s["runs"]["toy"]["fake_exp__short"]["dev"]["n"] == 12
    assert s["probes"] == {} and s["training"] == {}
    assert s["pairwise"] == {}  # runs of different catalogs are never paired
