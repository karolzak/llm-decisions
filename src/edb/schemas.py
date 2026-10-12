"""Common data schemas shared by datasets, methods, runners and reporting."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

ProbSource = Literal["model", "verbalized", "one_hot", "similarity", "none"]


class LabelOption(BaseModel):
    value: str
    description: str | None = None
    id: str | None = None

    @property
    def text(self) -> str:
        return self.description or self.value

    @property
    def has_description(self) -> bool:
        return bool(self.description) and self.description != self.value


class LabelCatalog(BaseModel):
    name: str
    labels: list[LabelOption]

    @property
    def values(self) -> list[str]:
        return [label.value for label in self.labels]

    @property
    def is_verbose(self) -> bool:
        return any(label.has_description for label in self.labels)

    def option(self, value: str) -> LabelOption:
        for label in self.labels:
            if label.value == value:
                return label
        raise KeyError(value)


class TaskSpec(BaseModel):
    """Dataset-level description of the decision to make."""

    name: str = "decision"
    instructions: str
    system_prompt: str | None = None


class Example(BaseModel):
    id: str
    state: str
    label: str
    group: str | None = None
    source_split: str
    source_index: int
    fields: dict[str, Any] = Field(default_factory=dict)


class Usage(BaseModel):
    input_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0


class Prediction(BaseModel):
    run_id: str
    example_id: str
    gold: str
    predicted: str | None = None
    # Distribution over catalog values; None when the method could not produce one.
    probabilities: dict[str, float] | None = None
    prob_source: ProbSource = "none"
    raw_output: str | None = None
    latency_ms: float | None = None  # final (successful) attempt only
    ttft_ms: float | None = None
    wall_ms: float | None = None  # including retries/backoff
    attempts: int = 1
    usage: Usage | None = None
    cost_usd: float | None = None
    error: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    extras: dict[str, Any] = Field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None
