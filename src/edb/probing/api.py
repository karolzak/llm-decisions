"""Vendor-neutral decision-API types and the client interface.

Every decision API studied here answers typed questions about one state in a single call:
TypeSafe Jev and Microsoft-Decision-1 (`POST .../systemone`), the OpenAI Decisions API
(`POST /v1/decisions`) and our local readouts of open-weight models. Probes are written
against these types, so the same experiment runs unchanged against every target.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

QuestionType = Literal["choice", "noul", "score"]


@dataclass(frozen=True)
class Option:
    value: str
    description: str | None = None

    @property
    def text(self) -> str:
        return self.description or self.value


@dataclass
class Question:
    """`options`: choice = unordered options; score = ordered levels, lowest first;
    noul = optional criteria as [Option("true", ...), Option("false", ...)]."""

    name: str
    instructions: str
    type: QuestionType = "choice"
    options: list[Option] = field(default_factory=list)

    @property
    def values(self) -> list[str]:
        if self.type == "noul":
            return ["true", "false"]
        return [o.value for o in self.options]

    def criterion(self, value: str) -> str | None:
        for o in self.options:
            if o.value == value:
                return o.description
        return None


@dataclass
class DecisionRequest:
    state: str | dict[str, Any] | list[str]
    questions: list[Question]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> DecisionRequest:
        return cls(state=data["state"], questions=[
            Question(name=q["name"], instructions=q["instructions"], type=q["type"],
                     options=[Option(**o) for o in q["options"]])
            for q in data["questions"]])


@dataclass
class Answer:
    """`probabilities` is keyed by option value (noul: "true"/"false"), in request order."""

    name: str
    type: str  # choice | noul | score | refusal
    probabilities: dict[str, float] = field(default_factory=dict)
    choice: str | None = None
    confidence: float | None = None
    score: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionResponse:
    answers: dict[str, Answer]
    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    model: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class DecisionClient(ABC):
    """One decision API (or local readout) behind the vendor-neutral types."""

    name: str = "client"
    # Largest option count the target accepts (None: unknown). Probes that exceed it still
    # send the request so the rejection itself is recorded.
    max_options: int | None = None

    @abstractmethod
    async def decide(self, request: DecisionRequest) -> DecisionResponse: ...

    async def aclose(self) -> None:  # noqa: B027 - optional hook
        """Release clients / models."""


def jev_confidence(probs: list[float]) -> float:
    """Normalised max-probability (`choice`/`noul` confidence in the Jev API docs)."""
    n = len(probs)
    return (max(probs) - 1 / n) / (1 - 1 / n) if n > 1 else 1.0
