"""Embedding client with a content-addressed on-disk cache.

Cache: artifacts/cache/embeddings/<deployment>.npz (sha1(text) -> float32 unit vector).
"""

from __future__ import annotations

import asyncio
import hashlib
import threading
import time
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI

from edb.settings import get_settings

_LOCK = threading.Lock()


def _key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    def __init__(self, deployment: str):
        root = get_settings().artifacts_dir / "cache" / "embeddings"
        self.path: Path = root / f"{deployment}.npz"
        self._rows: dict[str, np.ndarray] = {}
        if self.path.exists():
            data = np.load(self.path)
            self._rows = dict(zip(data["keys"].tolist(), data["vectors"], strict=True))

    def get(self, text: str) -> np.ndarray | None:
        return self._rows.get(_key(text))

    def put_many(self, texts: list[str], vectors: np.ndarray) -> None:
        for t, v in zip(texts, vectors, strict=True):
            self._rows[_key(t)] = v.astype(np.float32)

    def save(self) -> None:
        with _LOCK:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            keys = np.array(list(self._rows))
            vectors = np.vstack(list(self._rows.values())) if self._rows else np.zeros((0, 0))
            tmp = self.path.with_suffix(".tmp.npz")
            np.savez(tmp, keys=keys, vectors=vectors)
            tmp.replace(self.path)


class Embedder:
    def __init__(self, client: AsyncOpenAI, deployment: str, max_chars: int = 24000,
                 fit_tpm: float | None = 200_000):
        self.client = client
        self.deployment = deployment
        self.max_chars = max_chars
        # Token pacing for bulk (fit-time) embedding, to stay under the deployment TPM
        # instead of hitting 429s and backing off.
        self.fit_tpm = fit_tpm
        self.cache = EmbeddingCache(deployment)

    async def embed(self, texts: list[str]) -> tuple[np.ndarray, int]:
        """One live API request (no cache). Returns unit vectors and prompt tokens."""
        resp = await self.client.embeddings.create(
            model=self.deployment, input=[t[: self.max_chars] for t in texts]
        )
        vecs = np.array([d.embedding for d in sorted(resp.data, key=lambda d: d.index)],
                        dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs, resp.usage.prompt_tokens if resp.usage else 0

    async def embed_cached(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Embeddings for many texts, computing only cache misses (in batches)."""
        missing = list(dict.fromkeys(t for t in texts if self.cache.get(t) is None))
        for i in range(0, len(missing), batch_size):
            batch = missing[i : i + batch_size]
            t0 = time.monotonic()
            vecs, tokens = await self.embed(batch)
            self.cache.put_many(batch, vecs)
            if (i // batch_size) % 10 == 9:
                self.cache.save()
            if self.fit_tpm:
                budget_s = 60.0 * tokens / self.fit_tpm
                await asyncio.sleep(max(0.0, budget_s - (time.monotonic() - t0)))
        if missing:
            self.cache.save()
        return np.vstack([self.cache.get(t) for t in texts])
