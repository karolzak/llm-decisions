"""S1: bi-encoder baselines on Azure OpenAI embeddings.

The example is embedded live at prediction time (honest latency/cost); training data for
learned heads is embedded once and cached.

Params:
    head: "label" | "centroid" | "knn" | "logreg"
        label     zero-shot cosine(features, label text)              (single view only)
        centroid  cosine to the class mean of `fit_split` features
        knn       similarity-weighted vote of the k nearest `fit_split` examples
        logreg    multinomial logistic regression probe on `fit_split` (C chosen by
                  group-aware cross-validation within `fit_split`)
    views: "state" | "fields" | list[str]
        state   embed the rendered state
        fields  embed every state-template field separately (e.g. clauses and scenario)
        With two views the features are [a, b, a*b, |a-b|]; with more, concatenation.
    fit_split: str = "train"
    tau: float = 0.05          softmax temperature for label/centroid/knn scores
    k: int = 25                neighbours for knn
    Cs: list[float]            logreg regularisation grid (default 1e-3..1e2)
    cv_folds: int = 5
    batch_size: int = 64       embedding batch size when fitting
"""

from __future__ import annotations

import hashlib
import pickle
from typing import Any

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from edb.datasets.render import format_field, template_fields
from edb.datasets.store import load_split
from edb.embeddings import Embedder
from edb.providers.azure_auth import async_openai_client
from edb.providers.base import CallResult, DecisionMethod, MethodContext, softmax
from edb.schemas import Example, Usage

HEADS = ("label", "centroid", "knn", "logreg")


def combine_views(views: list[np.ndarray]) -> np.ndarray:
    """Per-example view embeddings (each n x d) -> feature matrix."""
    if len(views) == 1:
        return views[0]
    if len(views) == 2:
        a, b = views
        return np.hstack([a, b, a * b, np.abs(a - b)])
    return np.hstack(views)


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _fit_key(X: np.ndarray, y: np.ndarray, groups: np.ndarray, Cs: list[float],
             folds: int) -> str:
    """Identity of a logreg fit: exact training data + hyper-parameter grid + sklearn version."""
    h = hashlib.sha256()
    for arr in (np.ascontiguousarray(X, dtype=np.float32), y.astype(np.int64)):
        h.update(arr.tobytes())
    h.update("\x00".join(groups.tolist()).encode())
    h.update(repr((sorted(Cs), folds, sklearn.__version__)).encode())
    return h.hexdigest()[:20]


class EmbeddingMethod(DecisionMethod):
    type = "embedding"

    def __init__(self, ctx: MethodContext):
        super().__init__(ctx)
        p = ctx.params
        self.head = p.get("head", "label")
        if self.head not in HEADS:
            raise ValueError(f"unknown head {self.head!r}; expected one of {HEADS}")
        views = p.get("views", "state")
        if views == "fields":
            views = template_fields(ctx.dataset.state.template)
        self.views: list[str] = [views] if isinstance(views, str) else list(views)
        if self.head == "label" and len(self.views) != 1:
            raise ValueError("head=label needs a single view")
        self.fit_split = p.get("fit_split", "train")
        self.tau = float(p.get("tau", 0.05))
        self.k = int(p.get("k", 25))
        self.Cs = [float(c) for c in p.get("Cs", [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0])]
        self.cv_folds = int(p.get("cv_folds", 5))
        self.batch_size = int(p.get("batch_size", 64))
        max_chars = int((ctx.model.params if ctx.model else {}).get("max_input_chars", 24000))
        self.client = async_openai_client(ctx.settings, max_retries=10, timeout=ctx.run.timeout_s)
        self.embedder = Embedder(self.client, self.deployment, max_chars)
        self.info: dict[str, Any] = {"head": self.head, "views": self.views}
        self._protos: np.ndarray | None = None
        self._fit_X: np.ndarray | None = None
        self._fit_y: np.ndarray | None = None
        self._clf = None

    async def aclose(self) -> None:
        await self.client.close()

    def view_texts(self, example: Example) -> list[str]:
        texts = []
        for view in self.views:
            if view == "state":
                texts.append(example.state)
            else:
                if view not in example.fields:
                    raise KeyError(f"field {view!r} not kept for example {example.id}")
                fmt = self.ctx.dataset.state.formatters.get(view)
                texts.append(format_field(example.fields[view], fmt))
        return texts

    async def setup(self) -> None:
        if self.head == "label":
            texts = [f"{o.value}: {o.description}" if o.has_description else o.value
                     for o in self.ctx.catalog.labels]
            self._protos = await self.embedder.embed_cached(texts)
            return

        examples = load_split(self.ctx.dataset.name, self.fit_split)
        texts = [self.view_texts(e) for e in examples]
        per_view = [await self.embedder.embed_cached([t[i] for t in texts], self.batch_size)
                    for i in range(len(self.views))]
        X = combine_views(per_view)
        index = {v: i for i, v in enumerate(self.values)}
        y = np.array([index[e.label] for e in examples])
        self.info.update({"fit_split": self.fit_split, "fit_n": len(examples),
                          "feature_dim": int(X.shape[1])})

        if self.head in ("centroid", "knn"):
            Xu = _unit(X)
            if self.head == "centroid":
                self._protos = _unit(np.vstack([Xu[y == i].mean(axis=0)
                                                for i in range(len(self.values))]))
            else:
                self._fit_X, self._fit_y = Xu, y
            return

        groups = np.array([e.group or e.id for e in examples])
        key = _fit_key(X, y, groups, self.Cs, self.cv_folds)
        cache = self.ctx.settings.cache_dir / "heads" / f"logreg-{key}.pkl"
        if cache.exists():
            self._clf, fit_info = pickle.loads(cache.read_bytes())
            self.info.update(fit_info, fit_cache="hit")
            return

        folds = list(GroupKFold(n_splits=self.cv_folds).split(X, y, groups))

        def make(c: float):
            return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000))

        cv_loss = {c: -float(np.mean(cross_val_score(make(c), X, y, cv=folds,
                                                     scoring="neg_log_loss")))
                   for c in self.Cs}
        best = min(cv_loss, key=cv_loss.get)
        self._clf = make(best).fit(X, y)
        fit_info = {"C": best, "cv_log_loss": cv_loss}
        self.info.update(fit_info, fit_cache="miss")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(pickle.dumps((self._clf, fit_info)))

    def scores(self, features: np.ndarray) -> dict[str, float]:
        """Class probabilities for one feature vector."""
        if self.head == "logreg":
            proba = self._clf.predict_proba(features[None, :])[0]
            out = {v: 0.0 for v in self.values}
            for cls, p in zip(self._clf.classes_, proba, strict=True):
                out[self.values[int(cls)]] = float(p)
            return out
        f = _unit(features)
        if self.head in ("label", "centroid"):
            sims = self._protos @ f
            return softmax(dict(zip(self.values, sims.tolist(), strict=True)), self.tau)
        sims = self._fit_X @ f
        top = np.argsort(-sims)[: self.k]
        w = np.exp((sims[top] - sims[top].max()) / self.tau)
        votes = np.bincount(self._fit_y[top], weights=w, minlength=len(self.values)) + 1e-3
        return dict(zip(self.values, (votes / votes.sum()).tolist(), strict=True))

    async def call(self, example: Example) -> CallResult:
        if self._protos is None and self._clf is None and self._fit_X is None:
            raise RuntimeError("setup() was not called")
        vecs, tokens = await self.embedder.embed(self.view_texts(example))
        features = combine_views([v[None, :] for v in vecs])[0]
        return CallResult(
            probabilities=self.scores(features),
            prob_source="model" if self.head == "logreg" else "similarity",
            usage=Usage(input_tokens=tokens),
        )
