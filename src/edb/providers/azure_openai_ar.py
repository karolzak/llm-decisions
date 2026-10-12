"""A1: autoregressive label generation with an Azure OpenAI model (Responses API).

Params:
    reasoning_effort: str | None   e.g. "low"; omitted -> deployment default
    max_output_tokens: int         includes reasoning tokens (default 4096)
    confidence: "none" | "verbalized"
        none        -> free-text label, parsed; probabilities are one-hot
        verbalized  -> JSON {label, confidence} via structured outputs
    stream: bool                   stream to measure time-to-first-output-token (default true)
"""

from __future__ import annotations

import json
import time
from typing import Any

from edb.prompts import confidence_schema, system_prompt
from edb.providers.azure_auth import async_openai_client
from edb.providers.base import (
    CallResult,
    DecisionMethod,
    MethodContext,
    one_hot,
    spread_confidence,
)
from edb.providers.parsing import parse_label
from edb.schemas import Example, Usage


def usage_from_responses(usage: Any) -> Usage | None:
    if usage is None:
        return None
    u = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)
    inp = u.get("input_tokens_details") or {}
    out = u.get("output_tokens_details") or {}
    return Usage(
        input_tokens=u.get("input_tokens") or 0,
        cached_tokens=inp.get("cached_tokens") or 0,
        cache_write_tokens=inp.get("cache_write_tokens") or 0,
        output_tokens=u.get("output_tokens") or 0,
        reasoning_tokens=out.get("reasoning_tokens") or 0,
    )


class AzureOpenAIAutoregressive(DecisionMethod):
    type = "azure_openai_autoregressive"

    def __init__(self, ctx: MethodContext):
        super().__init__(ctx)
        p = ctx.params
        self.reasoning_effort: str | None = p.get("reasoning_effort")
        self.max_output_tokens: int = int(p.get("max_output_tokens", 4096))
        self.verbalized = p.get("confidence", "none") == "verbalized"
        self.stream = bool(p.get("stream", True))
        self.instructions = system_prompt(ctx.task, ctx.catalog, self.verbalized)
        self.client = async_openai_client(ctx.settings, max_retries=0, timeout=ctx.run.timeout_s)

    async def aclose(self) -> None:
        await self.client.close()

    def request_kwargs(self, example: Example) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.deployment,
            "instructions": self.instructions,
            "input": example.state,
            "max_output_tokens": self.max_output_tokens,
        }
        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        if self.verbalized:
            kwargs["text"] = {"format": confidence_schema(self.values)}
        return kwargs

    async def call(self, example: Example) -> CallResult:
        kwargs = self.request_kwargs(example)
        t0 = time.perf_counter()
        ttft_ms = None
        if self.stream:
            response = None
            stream = await self.client.responses.create(**kwargs, stream=True)
            async for event in stream:
                if event.type == "response.output_text.delta" and ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t0) * 1000
                elif event.type in ("response.completed", "response.incomplete",
                                    "response.failed"):
                    response = event.response
            if response is None:
                raise RuntimeError("stream ended without a final response event")
        else:
            response = await self.client.responses.create(**kwargs)

        text = response.output_text
        result = CallResult(
            raw_output=text,
            usage=usage_from_responses(response.usage),
            ttft_ms=ttft_ms,
            extras={"status": response.status, "response_id": response.id},
        )
        if response.status != "completed":
            details = getattr(response, "incomplete_details", None) or getattr(
                response, "error", None
            )
            result.extras["incomplete"] = str(details)

        if self.verbalized:
            self._parse_verbalized(text, result)
        else:
            parsed = parse_label(text, self.values)
            result.extras["parse_status"] = parsed.status
            if parsed.label is None:
                result.error = f"unparseable output (status={response.status})"
            else:
                result.predicted = parsed.label
                result.probabilities = one_hot(parsed.label, self.values)
                result.prob_source = "one_hot"
        return result

    def _parse_verbalized(self, text: str, result: CallResult) -> None:
        try:
            obj = json.loads(text)
            parsed = parse_label(str(obj["label"]), self.values)
            confidence = float(obj["confidence"])
        except (ValueError, KeyError, TypeError) as exc:
            result.error = f"invalid JSON answer: {exc}"
            return
        result.extras["parse_status"] = parsed.status
        result.extras["verbalized_confidence"] = confidence
        if parsed.label is None:
            result.error = "label not in catalog"
            return
        result.predicted = parsed.label
        result.probabilities = spread_confidence(parsed.label, confidence, self.values)
        result.prob_source = "verbalized"
