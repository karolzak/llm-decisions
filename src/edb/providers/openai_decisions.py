"""A2: OpenAI Decisions API (`POST /v1/decisions`, `choice` question).

Raw HTTP until the pinned openai SDK exposes `client.decisions` (>= 3.26.0).
The endpoint follows the model config provider:
    openai        -> https://api.openai.com/v1/decisions with OPENAI_API_KEY
    azure_openai  -> <azure openai base>/decisions with Entra ID (404 on Azure as of 2026-10)

Params:
    include_descriptions: bool  send catalog descriptions with each choice (default true)
"""

from __future__ import annotations

from typing import Any

import httpx

from edb.providers.azure_auth import BearerAuth, token_provider
from edb.providers.base import CallResult, DecisionMethod, MethodContext
from edb.providers.retry import raise_for_status
from edb.schemas import Example, Usage


class OpenAIDecisions(DecisionMethod):
    type = "openai_decisions"

    def __init__(self, ctx: MethodContext):
        super().__init__(ctx)
        s = ctx.settings
        self.include_descriptions = bool(ctx.params.get("include_descriptions", True))
        provider = ctx.model.provider if ctx.model else "openai"
        if provider == "openai":
            if not s.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY is required for the OpenAI Decisions API")
            self.url = s.openai_base_url_direct.rstrip("/") + "/decisions"
            auth: httpx.Auth = BearerAuth(lambda: s.openai_api_key)
        else:
            self.url = s.openai_base_url + "decisions"
            auth = BearerAuth(token_provider(s))
        self.http = httpx.AsyncClient(
            auth=auth,
            timeout=ctx.run.timeout_s,
            headers={"Content-Type": "application/json"},
            limits=httpx.Limits(max_connections=max(8, ctx.run.concurrency)),
        )

    async def aclose(self) -> None:
        await self.http.aclose()

    def request_body(self, example: Example) -> dict[str, Any]:
        choices = []
        for o in self.ctx.catalog.labels:
            choice: dict[str, Any] = {"value": o.value}
            if self.include_descriptions and o.has_description:
                choice["description"] = o.description
            choices.append(choice)
        return {
            "model": self.deployment,
            "input": example.state,
            "questions": [
                {
                    "type": "choice",
                    "name": self.ctx.task.name,
                    "instructions": self.ctx.task.instructions,
                    "choices": choices,
                }
            ],
        }

    def parse_response(self, payload: dict[str, Any]) -> CallResult:
        usage = payload.get("usage") or {}
        details = usage.get("input_tokens_details") or {}
        result = CallResult(
            usage=Usage(
                input_tokens=usage.get("input_tokens") or 0,
                cached_tokens=details.get("cached_tokens") or 0,
                output_tokens=usage.get("output_tokens") or 0,
            ),
            extras={"id": payload.get("id"), "model": payload.get("model")},
        )
        answers = {a.get("name"): a for a in payload.get("answers") or []}
        answer = answers.get(self.ctx.task.name)
        if answer is None:
            result.error = "no answer for question"
        elif answer.get("type") == "refusal":
            result.error = "refusal"
            result.extras["refusal"] = answer
        elif answer.get("type") != "choice":
            result.error = f"unexpected answer type {answer.get('type')!r}"
        else:
            result.predicted = answer.get("choice")
            result.probabilities = {
                p["value"]: p["probability"] for p in answer.get("probabilities") or []
            }
            result.prob_source = "model"
            result.extras["confidence"] = answer.get("confidence")
        return result

    async def call(self, example: Example) -> CallResult:
        response = await self.http.post(self.url, json=self.request_body(example))
        raise_for_status(response)
        return self.parse_response(response.json())
