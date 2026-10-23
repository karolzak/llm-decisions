"""Decision-API adapters: TypeSafe-style `systemone` (Decision-1, Jev) and OpenAI Decisions.

Both take the vendor-neutral `DecisionRequest` and return `DecisionResponse`; HTTP errors are
raised as `HTTPError` so the probe runner can retry or record them.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from edb.config import _load_yaml, _resolve
from edb.probing.api import (
    Answer,
    DecisionClient,
    DecisionRequest,
    DecisionResponse,
    Question,
)
from edb.providers.azure_auth import BearerAuth, token_provider
from edb.providers.retry import raise_for_status
from edb.settings import Settings, get_settings


class ProbeTarget(BaseModel):
    """A decision API or local readout to probe (`configs/probe_targets/<name>.yaml`)."""

    name: str
    kind: Literal["systemone", "openai_decisions", "local_readout"]
    description: str = ""
    # Literal URL, or "settings:<attribute>" (e.g. settings:decision1_url).
    url: str | None = None
    # entra (Entra ID bearer token) | env:<VAR> (bearer key from settings/.env/environment) | none
    auth: str = "none"
    model: str | None = None  # `model` field sent with each request
    max_options: int | None = None  # documented option limit (None: unknown)
    concurrency: int = 4
    max_rpm: float | None = None
    timeout_s: float = 120.0
    max_retries: int = 8
    headers: dict[str, str] = Field(default_factory=dict)
    # local_readout: model (configs/models name), adapter, temperature, ... (see chat_readout)
    params: dict[str, Any] = Field(default_factory=dict)


def load_probe_target(name: str) -> ProbeTarget:
    return ProbeTarget.model_validate(_load_yaml(_resolve("probe_targets", name)))


def resolve_url(target: ProbeTarget, settings: Settings) -> str:
    if not target.url:
        raise ValueError(f"probe target {target.name!r} has no url")
    if target.url.startswith("settings:"):
        return getattr(settings, target.url.split(":", 1)[1])
    return target.url


def resolve_auth(target: ProbeTarget, settings: Settings) -> httpx.Auth | None:
    if target.auth == "none":
        return None
    if target.auth == "entra":
        return BearerAuth(token_provider(settings))
    if target.auth.startswith("env:"):
        var = target.auth.split(":", 1)[1]
        key = getattr(settings, var.lower(), None) or os.environ.get(var)
        if not key:
            raise RuntimeError(f"{var} not set (see .env.example)")
        return BearerAuth(lambda: key)
    raise ValueError(f"unknown auth {target.auth!r} (entra | env:<VAR> | none)")


def make_client(target: ProbeTarget, settings: Settings | None = None) -> DecisionClient:
    settings = settings or get_settings()
    if target.kind == "systemone":
        return SystemOneClient(target, settings)
    if target.kind == "openai_decisions":
        return OpenAIDecisionsClient(target, settings)
    if target.kind == "local_readout":
        from edb.local.chat_readout import LocalReadoutClient  # imports torch

        return LocalReadoutClient(target)
    raise ValueError(f"unknown target kind {target.kind!r}")


class HttpDecisionClient(DecisionClient):
    def __init__(self, target: ProbeTarget, settings: Settings):
        self.target = target
        self.name = target.name
        self.max_options = target.max_options
        self.url = resolve_url(target, settings)
        self.http = httpx.AsyncClient(
            auth=resolve_auth(target, settings),
            timeout=target.timeout_s,
            headers={"Content-Type": "application/json", "Accept": "application/json",
                     **target.headers},
            limits=httpx.Limits(max_connections=max(8, target.concurrency)),
        )

    async def aclose(self) -> None:
        await self.http.aclose()

    def body(self, request: DecisionRequest) -> dict[str, Any]:
        raise NotImplementedError

    def parse(self, request: DecisionRequest, payload: dict[str, Any]) -> dict[str, Answer]:
        raise NotImplementedError

    async def decide(self, request: DecisionRequest) -> DecisionResponse:
        t0 = time.perf_counter()
        response = await self.http.post(self.url, json=self.body(request))
        latency_ms = (time.perf_counter() - t0) * 1000
        raise_for_status(response)
        payload = response.json()
        usage = payload.get("usage") or {}
        return DecisionResponse(
            answers=self.parse(request, payload),
            latency_ms=latency_ms,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            model=payload.get("model"),
            raw=payload,
        )


def _probabilities(raw: Any, q: Question) -> dict[str, float]:
    """Map the provider's probability field onto `q.values` (dict by value or criterion text,
    list of numbers by position, or list of {value|label, probability} objects)."""
    values = q.values
    if isinstance(raw, dict):
        if set(raw) <= set(values):
            return {v: float(raw[v]) for v in values if v in raw}
        by_text = {o.text: o.value for o in q.options}
        if set(raw) <= set(by_text):
            return {by_text[k]: float(p) for k, p in raw.items()}
        if all(str(k).isdigit() for k in raw):
            return {values[int(k)]: float(p) for k, p in raw.items() if int(k) < len(values)}
        raise ValueError(f"cannot map probabilities keys {sorted(raw)[:5]} to {values[:5]}")
    if isinstance(raw, list):
        if all(isinstance(p, int | float) for p in raw):
            return {v: float(p) for v, p in zip(values, raw, strict=False)}
        out = {}
        for item in raw:
            key = item.get("value", item.get("label"))
            if isinstance(key, int) and key < len(values):
                key = values[key]
            out[str(key)] = float(item["probability"])
        return out
    raise ValueError(f"unexpected probabilities {type(raw).__name__}")


class SystemOneClient(HttpDecisionClient):
    """TypeSafe `/v1/systemone` wire format, also served by Microsoft-Decision-1 on Foundry
    (`/providers/microsoft/v1/systemone`) and by the open replicas (strands-decider, cygnet,
    quyet, ...). Questions are an object keyed by name; criteria carry the options."""

    def question(self, q: Question) -> dict[str, Any]:
        body: dict[str, Any] = {"type": q.type, "instructions": q.instructions}
        if q.type == "choice":
            body["criteria"] = {o.value: o.text for o in q.options}
        elif q.type == "score":
            body["criteria"] = [o.text for o in q.options]
        elif q.options:  # noul with explicit true/false criteria
            body["criteria"] = {o.value: o.description or "" for o in q.options}
        return body

    def body(self, request: DecisionRequest) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.target.model} if self.target.model else {}
        body["state"] = request.state
        body["questions"] = {q.name: self.question(q) for q in request.questions}
        return body

    def parse(self, request: DecisionRequest, payload: dict[str, Any]) -> dict[str, Answer]:
        answers = payload.get("answers") or {}
        out = {}
        for q in request.questions:
            a = answers.get(q.name)
            if not isinstance(a, dict):
                out[q.name] = Answer(q.name, "missing", raw={"answer": a})
                continue
            kind = a.get("type", q.type)
            if kind == "noul":
                p = a.get("noul", a.get("probability"))
                probs = {"true": float(p), "false": 1.0 - float(p)} if p is not None else {}
                out[q.name] = Answer(q.name, kind, probs, choice="true" if (p or 0) >= 0.5
                                     else "false", confidence=a.get("confidence"), raw=a)
            elif kind in ("choice", "score"):
                probs = _probabilities(a.get("probabilities"), q)
                choice = a.get("choice")
                if choice is None and probs:
                    choice = max(probs, key=probs.get)
                out[q.name] = Answer(q.name, kind, probs, choice=choice,
                                     confidence=a.get("confidence"), score=a.get("score"), raw=a)
            else:
                out[q.name] = Answer(q.name, kind, raw=a)
        return out


class OpenAIDecisionsClient(HttpDecisionClient):
    """OpenAI `/v1/decisions` wire format: `input` + a list of named questions
    (`predicate` / `choice` with `choices` / `score` with `levels`)."""

    def question(self, q: Question) -> dict[str, Any]:
        if q.type == "noul":
            text = q.instructions
            criteria = {o.value: o.description for o in q.options if o.description}
            if criteria:
                text += "".join(f"\n{k.capitalize()} if: {v}" for k, v in criteria.items())
            return {"type": "predicate", "name": q.name, "instructions": text}
        key, label = ("choices", "value") if q.type == "choice" else ("levels", "label")
        items = []
        for o in q.options:
            item: dict[str, Any] = {label: o.value}
            if o.description:
                item["description"] = o.description
            items.append(item)
        return {"type": q.type, "name": q.name, "instructions": q.instructions, key: items}

    def body(self, request: DecisionRequest) -> dict[str, Any]:
        state = request.state if isinstance(request.state, str) else json.dumps(request.state)
        return {"model": self.target.model, "input": state,
                "questions": [self.question(q) for q in request.questions]}

    def parse(self, request: DecisionRequest, payload: dict[str, Any]) -> dict[str, Answer]:
        by_name = {a.get("name"): a for a in payload.get("answers") or []}
        out = {}
        for q in request.questions:
            a = by_name.get(q.name)
            if a is None:
                out[q.name] = Answer(q.name, "missing")
                continue
            kind = a.get("type")
            if kind == "predicate":
                p = float(a["probability"])
                out[q.name] = Answer(q.name, "noul", {"true": p, "false": 1 - p},
                                     choice="true" if p >= 0.5 else "false", raw=a)
            elif kind in ("choice", "score"):
                probs = _probabilities(a.get("probabilities"), q)
                choice = a.get("choice")
                if kind == "score" and probs:
                    choice = max(probs, key=probs.get)
                out[q.name] = Answer(q.name, kind, probs, choice=choice,
                                     confidence=a.get("confidence"), score=a.get("score"), raw=a)
            else:
                out[q.name] = Answer(q.name, kind or "unknown", raw=a)
        return out
