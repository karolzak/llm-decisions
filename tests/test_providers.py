import asyncio
import json

import httpx
import pytest

from edb.config import load_dataset_config, load_label_catalog, load_model_config
from edb.providers.azure_openai_ar import AzureOpenAIAutoregressive
from edb.providers.base import CallResult, DecisionMethod, MethodContext
from edb.providers.decision1 import MicrosoftDecision1
from edb.providers.openai_decisions import OpenAIDecisions
from edb.providers.retry import HTTPError
from edb.schemas import Example, Usage


@pytest.fixture
def ctx(sandbox):
    ds = load_dataset_config("toy")
    labels = ["alpha", "beta", "gamma"]
    return MethodContext(
        run_id="r", dataset=ds, catalog=load_label_catalog(ds, "verbose", labels),
        model=load_model_config("fake-model"),
    )


EXAMPLE = Example(id="x-1", state="Topic t: hello", label="beta", source_split="train",
                  source_index=0)


def test_decision1_request_and_response(ctx):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "model": "microsoft-decision-1",
            "answers": {"pick": {"type": "choice", "choice": "beta", "confidence": 0.7,
                                 "probabilities": {"alpha": 0.1, "beta": 0.8, "gamma": 0.1}}},
            "usage": {"input_tokens": 1_000_000, "output_tokens": 1},
        })

    m = MicrosoftDecision1(ctx)
    m.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    pred = asyncio.run(m.predict(EXAMPLE))
    q = seen["body"]["questions"]["pick"]
    assert seen["body"]["model"] == "fake" and seen["body"]["state"] == EXAMPLE.state
    assert q["type"] == "choice" and q["criteria"]["beta"] == "The second one"
    assert pred.ok and pred.predicted == "beta" and pred.prob_source == "model"
    assert pred.probabilities["beta"] == pytest.approx(0.8)
    assert pred.cost_usd == pytest.approx(1.0 + 2e-6)


def test_decision1_retries_429_then_succeeds(ctx, monkeypatch):
    monkeypatch.setattr("edb.providers.base.backoff_delay", lambda *a: 0)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "0"}, json={"error": "busy"})
        return httpx.Response(200, json={"answers": {"pick": {
            "type": "choice", "choice": "alpha", "probabilities": {"alpha": 1.0}}}})

    m = MicrosoftDecision1(ctx)
    m.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    pred = asyncio.run(m.predict(EXAMPLE))
    assert pred.ok and pred.attempts == 2
    assert pred.probabilities == {"alpha": 1.0, "beta": 0.0, "gamma": 0.0}


def test_decision1_non_retryable_error(ctx):
    m = MicrosoftDecision1(ctx)
    m.http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda r: httpx.Response(400, json={"error": "bad"})))
    pred = asyncio.run(m.predict(EXAMPLE))
    assert not pred.ok and "HTTP 400" in pred.error and pred.attempts == 1


def test_openai_decisions_body_and_parse(ctx, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from edb.settings import get_settings
    get_settings.cache_clear()
    ctx.settings = get_settings()
    ctx.model = ctx.model.model_copy(update={"provider": "openai"})
    m = OpenAIDecisions(ctx)
    body = m.request_body(EXAMPLE)
    q = body["questions"][0]
    assert m.url == "https://api.openai.com/v1/decisions"
    assert q["name"] == "pick" and q["choices"][0] == {"value": "alpha",
                                                       "description": "The first one"}
    res = m.parse_response({"answers": [{"type": "choice", "name": "pick", "choice": "gamma",
                                         "probabilities": [{"value": "gamma", "probability": .9},
                                                           {"value": "beta", "probability": .1}],
                                         "confidence": .85}],
                            "usage": {"input_tokens": 10}})
    assert res.predicted == "gamma" and res.probabilities["gamma"] == 0.9
    refusal = m.parse_response({"answers": [{"type": "refusal", "name": "pick"}]})
    assert refusal.error == "refusal"


def test_autoregressive_prompt_and_verbalized_parse(ctx):
    ctx.params = {"confidence": "verbalized", "reasoning_effort": "low"}
    m = AzureOpenAIAutoregressive(ctx)
    kwargs = m.request_kwargs(EXAMPLE)
    assert "- beta: The second one" in kwargs["instructions"]
    assert kwargs["reasoning"] == {"effort": "low"}
    assert kwargs["text"]["format"]["schema"]["properties"]["label"]["enum"] == [
        "alpha", "beta", "gamma"]
    res = CallResult()
    m._parse_verbalized('{"label": "beta", "confidence": 0.6}', res)
    assert res.probabilities == {"alpha": pytest.approx(0.2), "beta": 0.6,
                                 "gamma": pytest.approx(0.2)}
    res = CallResult()
    m._parse_verbalized("not json", res)
    assert res.error


class _Flaky(DecisionMethod):
    type = "flaky"

    async def call(self, example):
        raise HTTPError(503, "unavailable", httpx.Headers())


def test_retries_exhausted_records_error(ctx, monkeypatch):
    monkeypatch.setattr("edb.providers.base.backoff_delay", lambda *a: 0)
    pred = asyncio.run(_Flaky(ctx).predict(EXAMPLE))
    assert not pred.ok and pred.attempts == ctx.run.max_retries + 1
    assert pred.usage is None or isinstance(pred.usage, Usage)
