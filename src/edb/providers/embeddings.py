"""S1: bi-encoder similarity baseline (Azure OpenAI embeddings).

Params:
    mode: "label" | "centroid"
        label     -> cosine(state, label text) for each catalog option (zero-shot)
        centroid  -> cosine(state, mean embedding of the label's examples in `fit_split`)
    fit_split: str      split used to build centroids (default "train")
    tau: float          softmax temperature over cosine similarities (default 0.05;
                        refit later by temperature scaling on the calibration split)
    batch_size: int     embedding batch size for centroid fitting (default 64)
"""

from __future__ import annotations

import numpy as np

from edb.datasets.store import load_split
from edb.providers.azure_auth import async_openai_client
from edb.providers.base import CallResult, DecisionMethod, MethodContext, softmax
from edb.schemas import Example, Usage


class EmbeddingSimilarity(DecisionMethod):
    type = "embedding_similarity"

    def __init__(self, ctx: MethodContext):
        super().__init__(ctx)
        p = ctx.params
        self.mode = p.get("mode", "label")
        if self.mode not in ("label", "centroid"):
            raise ValueError(f"unknown mode {self.mode!r}")
        self.fit_split = p.get("fit_split", "train")
        self.tau = float(p.get("tau", 0.05))
        self.batch_size = int(p.get("batch_size", 64))
        self.max_chars = int((ctx.model.params if ctx.model else {}).get("max_input_chars", 24000))
        self.client = async_openai_client(ctx.settings, max_retries=8, timeout=ctx.run.timeout_s)
        self.prototypes: np.ndarray | None = None

    async def aclose(self) -> None:
        await self.client.close()

    async def _embed(self, texts: list[str]) -> tuple[np.ndarray, int]:
        resp = await self.client.embeddings.create(
            model=self.deployment, input=[t[: self.max_chars] for t in texts]
        )
        vecs = np.array([d.embedding for d in sorted(resp.data, key=lambda d: d.index)])
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs, resp.usage.prompt_tokens if resp.usage else 0

    async def setup(self) -> None:
        if self.mode == "label":
            texts = [
                f"{o.value}: {o.description}" if o.has_description else o.value
                for o in self.ctx.catalog.labels
            ]
            self.prototypes, _ = await self._embed(texts)
            return
        examples = load_split(self.ctx.dataset.name, self.fit_split)
        chunks = []
        for i in range(0, len(examples), self.batch_size):
            vecs, _ = await self._embed([e.state for e in examples[i : i + self.batch_size]])
            chunks.append(vecs)
        emb = np.vstack(chunks)
        labels = np.array([e.label for e in examples])
        protos = np.vstack([emb[labels == v].mean(axis=0) for v in self.values])
        self.prototypes = protos / np.linalg.norm(protos, axis=1, keepdims=True)

    async def call(self, example: Example) -> CallResult:
        if self.prototypes is None:
            raise RuntimeError("setup() was not called")
        vec, tokens = await self._embed([example.state])
        sims = (self.prototypes @ vec[0]).tolist()
        scores = dict(zip(self.values, sims, strict=True))
        return CallResult(
            probabilities=softmax(scores, self.tau),
            prob_source="similarity",
            usage=Usage(input_tokens=tokens),
            extras={"cosine": scores},
        )
