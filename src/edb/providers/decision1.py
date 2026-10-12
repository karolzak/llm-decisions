"""B1: Microsoft-Decision-1 on Azure AI Foundry (`/providers/microsoft/v1/systemone`).

Params:
    include_descriptions: bool  send catalog descriptions as `criteria` values (default true;
                                short catalogs fall back to the label itself)
"""

from __future__ import annotations

from typing import Any

from edb.providers.azure_auth import foundry_async_http_client
from edb.providers.base import CallResult, DecisionMethod, MethodContext
from edb.providers.retry import raise_for_status
from edb.schemas import Example, Usage


class MicrosoftDecision1(DecisionMethod):
    type = "azure_foundry_decision"

    def __init__(self, ctx: MethodContext):
        super().__init__(ctx)
        self.include_descriptions = bool(ctx.params.get("include_descriptions", True))
        self.url = ctx.settings.decision1_url
        self.http = foundry_async_http_client(
            ctx.settings, timeout=ctx.run.timeout_s, max_connections=max(8, ctx.run.concurrency)
        )

    async def aclose(self) -> None:
        await self.http.aclose()

    def request_body(self, example: Example) -> dict[str, Any]:
        criteria = {
            o.value: (o.text if self.include_descriptions else o.value)
            for o in self.ctx.catalog.labels
        }
        return {
            "model": self.deployment,
            "state": example.state,
            "questions": {
                self.ctx.task.name: {
                    "type": "choice",
                    "instructions": self.ctx.task.instructions,
                    "criteria": criteria,
                }
            },
        }

    def parse_response(self, payload: dict[str, Any]) -> CallResult:
        answer = (payload.get("answers") or {}).get(self.ctx.task.name)
        usage = payload.get("usage") or {}
        result = CallResult(
            raw_output=None,
            usage=Usage(
                input_tokens=usage.get("input_tokens") or 0,
                output_tokens=usage.get("output_tokens") or 0,
            ),
            extras={"model": payload.get("model")},
        )
        if not isinstance(answer, dict) or answer.get("type") != "choice":
            result.error = f"unexpected answer: {answer!r}"
            return result
        result.predicted = answer.get("choice")
        result.probabilities = answer.get("probabilities")
        result.prob_source = "model"
        result.extras["confidence"] = answer.get("confidence")
        if result.predicted not in self.values:
            result.error = f"choice {result.predicted!r} not in catalog"
            result.predicted = None
        return result

    async def call(self, example: Example) -> CallResult:
        response = await self.http.post(self.url, json=self.request_body(example))
        raise_for_status(response)
        return self.parse_response(response.json())
