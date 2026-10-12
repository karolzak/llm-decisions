"""Decision method interface.

A method maps an `Example` to a distribution over the label catalog. Concrete methods
implement `call()`; `predict()` adds timing, retries, normalisation and cost accounting.
"""

from __future__ import annotations

import asyncio
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

from edb.config import DatasetConfig, ModelConfig, RunConfig
from edb.pricing import api_cost
from edb.providers.retry import backoff_delay, classify
from edb.schemas import Example, LabelCatalog, Prediction, ProbSource, TaskSpec, Usage
from edb.settings import Settings, get_settings


@dataclass
class MethodContext:
    run_id: str
    dataset: DatasetConfig
    catalog: LabelCatalog
    model: ModelConfig | None
    params: dict[str, Any] = field(default_factory=dict)
    run: RunConfig = field(default_factory=RunConfig)
    settings: Settings = field(default_factory=get_settings)

    @property
    def task(self) -> TaskSpec:
        return self.dataset.task


@dataclass
class CallResult:
    predicted: str | None = None
    probabilities: dict[str, float] | None = None
    prob_source: ProbSource = "none"
    raw_output: str | None = None
    usage: Usage | None = None
    ttft_ms: float | None = None
    error: str | None = None  # non-retryable outcome, e.g. unparseable answer or refusal
    extras: dict[str, Any] = field(default_factory=dict)


class RateLimiter:
    """Evenly spaces request starts to at most `per_minute` across concurrent workers."""

    def __init__(self, per_minute: float):
        self.interval = 60.0 / per_minute
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._next - now
            self._next = max(now, self._next) + self.interval
        if wait > 0:
            await asyncio.sleep(wait)


class DecisionMethod(ABC):
    type: ClassVar[str]

    def __init__(self, ctx: MethodContext):
        self.ctx = ctx
        self.limiter = RateLimiter(ctx.run.max_rpm) if ctx.run.max_rpm else None

    @property
    def values(self) -> list[str]:
        return self.ctx.catalog.values

    @property
    def deployment(self) -> str:
        if not self.ctx.model or not self.ctx.model.deployment:
            raise ValueError(f"method {self.type!r} requires a model with a `deployment`")
        return self.ctx.model.deployment

    async def setup(self) -> None:  # noqa: B027 - optional hook
        """One-off preparation before predictions (e.g. embedding label prototypes)."""

    async def aclose(self) -> None:  # noqa: B027 - optional hook
        """Release clients."""

    @abstractmethod
    async def call(self, example: Example) -> CallResult: ...

    async def predict(self, example: Example) -> Prediction:
        start = time.perf_counter()
        result: CallResult | None = None
        latency_ms: float | None = None
        error: str | None = None
        attempt = 0
        while True:
            attempt += 1
            if self.limiter:
                await self.limiter.acquire()
            t0 = time.perf_counter()
            try:
                result = await asyncio.wait_for(self.call(example), self.ctx.run.timeout_s)
                latency_ms = (time.perf_counter() - t0) * 1000
                break
            except Exception as exc:  # noqa: BLE001 - classified below
                retryable, suggested = classify(exc)
                if not retryable or attempt > self.ctx.run.max_retries:
                    error = f"{type(exc).__name__}: {exc}"
                    break
                await asyncio.sleep(backoff_delay(attempt, suggested))

        pred = Prediction(
            run_id=self.ctx.run_id,
            example_id=example.id,
            gold=example.label,
            latency_ms=latency_ms,
            wall_ms=(time.perf_counter() - start) * 1000,
            attempts=attempt,
            error=error,
        )
        if result is None:
            return pred

        probs = normalize(result.probabilities, self.values)
        predicted = result.predicted
        if predicted is None and probs:
            predicted = max(probs, key=probs.get)
        pred.predicted = predicted
        pred.probabilities = probs
        pred.prob_source = result.prob_source if probs else "none"
        pred.raw_output = result.raw_output
        pred.ttft_ms = result.ttft_ms
        pred.usage = result.usage
        pred.cost_usd = api_cost(result.usage, self.ctx.model.pricing if self.ctx.model else None)
        pred.error = result.error
        pred.extras = result.extras
        return pred


def normalize(probs: dict[str, float] | None, values: list[str]) -> dict[str, float] | None:
    if not probs:
        return None
    unknown = set(probs) - set(values)
    if unknown:
        raise ValueError(f"probabilities for unknown labels {sorted(unknown)}")
    clean = {v: max(0.0, float(probs.get(v, 0.0))) for v in values}
    total = sum(clean.values())
    if not math.isfinite(total) or total <= 0:
        return None
    return {v: p / total for v, p in clean.items()}


def one_hot(label: str, values: list[str]) -> dict[str, float]:
    return {v: 1.0 if v == label else 0.0 for v in values}


def spread_confidence(label: str, confidence: float, values: list[str]) -> dict[str, float]:
    """Verbalized confidence -> distribution with the remainder spread uniformly."""
    c = min(1.0, max(0.0, confidence))
    rest = (1.0 - c) / (len(values) - 1) if len(values) > 1 else 0.0
    return {v: c if v == label else rest for v in values}


def softmax(scores: dict[str, float], temperature: float = 1.0) -> dict[str, float]:
    m = max(scores.values())
    exps = {k: math.exp((s - m) / temperature) for k, s in scores.items()}
    z = sum(exps.values())
    return {k: e / z for k, e in exps.items()}
