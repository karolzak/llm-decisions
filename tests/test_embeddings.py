import asyncio
import re

import numpy as np
import pytest

from edb.config import load_dataset_config, load_label_catalog, load_model_config
from edb.datasets.store import load_split
from edb.embeddings import Embedder
from edb.providers.base import MethodContext
from edb.providers.embeddings import EmbeddingMethod, combine_views

DIM = 16


def fake_vector(text: str) -> np.ndarray:
    """Label-informative toy embedding: index i of 'message <split> <i>' encodes label i % 3."""
    rng = np.random.default_rng(abs(hash(text)) % 2**32)
    v = rng.normal(scale=0.3, size=DIM)
    m = re.search(r"message \w+ (\d+)", text)
    if m:
        v[int(m.group(1)) % 3] += 2.0
    return v / np.linalg.norm(v)


@pytest.fixture
def fake_api(monkeypatch):
    calls = {"requests": 0, "texts": 0}

    async def embed(self, texts):
        calls["requests"] += 1
        calls["texts"] += len(texts)
        return np.vstack([fake_vector(t) for t in texts]).astype(np.float32), 7 * len(texts)

    monkeypatch.setattr(Embedder, "embed", embed)
    return calls


def make_method(params) -> EmbeddingMethod:
    ds = load_dataset_config("toy")
    ctx = MethodContext(run_id="s1", dataset=ds,
                        catalog=load_label_catalog(ds, "short", ["alpha", "beta", "gamma"]),
                        model=load_model_config("fake-model"), params=params)
    return EmbeddingMethod(ctx)


def test_combine_views():
    a, b = np.ones((2, 3)), np.zeros((2, 3))
    assert combine_views([a]).shape == (2, 3)
    assert combine_views([a, b]).shape == (2, 12)
    assert combine_views([a, b, a]).shape == (2, 9)


@pytest.mark.parametrize("params,min_acc", [
    ({"head": "centroid"}, 0.9),
    ({"head": "knn", "k": 15}, 0.9),
    ({"head": "logreg", "cv_folds": 3}, 0.9),
    ({"head": "logreg", "views": "fields", "cv_folds": 3}, 0.9),
    ({"head": "label"}, 0.0),
])
def test_heads_learn_signal(sandbox, fake_api, params, min_acc):
    method = make_method(params)
    dev = load_split("toy", "dev")

    async def run():
        await method.setup()
        return [await method.predict(e) for e in dev]

    preds = asyncio.run(run())
    assert all(p.ok and abs(sum(p.probabilities.values()) - 1) < 1e-6 for p in preds)
    acc = np.mean([p.predicted == p.gold for p in preds])
    assert acc >= min_acc
    assert preds[0].usage.input_tokens > 0 and preds[0].cost_usd is not None
    if params["head"] == "logreg":
        assert "C" in method.info and method.info["fit_n"] > 0


def test_fit_embeddings_are_cached(sandbox, fake_api):
    asyncio.run(make_method({"head": "centroid"}).setup())
    first = fake_api["texts"]
    asyncio.run(make_method({"head": "knn"}).setup())
    assert fake_api["texts"] == first  # second fit served entirely from disk cache


def test_label_head_rejects_multiple_views(sandbox):
    with pytest.raises(ValueError):
        make_method({"head": "label", "views": "fields"})


def test_logreg_fit_is_cached(sandbox, fake_api):
    first = make_method({"head": "logreg", "cv_folds": 3})
    asyncio.run(first.setup())
    second = make_method({"head": "logreg", "cv_folds": 3})
    asyncio.run(second.setup())
    assert first.info["fit_cache"] == "miss" and second.info["fit_cache"] == "hit"
    assert first.info["C"] == second.info["C"]
    x = np.ones(DIM, dtype=np.float32) / np.sqrt(DIM)
    assert first.scores(x) == second.scores(x)
    changed = make_method({"head": "logreg", "cv_folds": 3, "Cs": [0.5]})
    asyncio.run(changed.setup())
    assert changed.info["fit_cache"] == "miss"
