"""Probing: wire formats, probe case generation, analysis and forensics (no network/GPU)."""

import asyncio
import json
import math

import httpx
import pytest

from edb.probing import forensics
from edb.probing.api import (
    Answer,
    DecisionClient,
    DecisionRequest,
    DecisionResponse,
    Option,
    Question,
    jev_confidence,
)
from edb.probing.clients import (
    OpenAIDecisionsClient,
    ProbeTarget,
    SystemOneClient,
    resolve_auth,
)
from edb.probing.items import BaseItem, load_items, subset
from edb.probing.probes import PROBES, TOKEN_STATES, TokensProbe
from edb.probing.report import build_probe_report
from edb.probing.runner import analyze_target, build_cases, read_records, records_path, run_cases
from edb.settings import get_settings

PREFIX = "The customer wants: "


def make_items() -> dict[str, list[BaseItem]]:
    ace_opts = tuple(Option(v, f"{v} criterion") for v in ("Compliant", "Non-Compliant", "NA"))
    bank_opts = tuple(Option(f"intent_{i:02d}") for i in range(30))
    return {
        "ace": [BaseItem(f"ace:dev-{i}", "ace", PREFIX + ace_opts[i % 3].value, "Decide.",
                         ace_opts, ace_opts[i % 3].value) for i in range(6)],
        "banking77": [BaseItem(f"banking77:dev-{i}", "banking77", PREFIX + f"intent_{i:02d}",
                               "Which intent?", bank_opts, f"intent_{i:02d}") for i in range(6)],
    }


class FakeReadout(DecisionClient):
    """Scores options by whether they match the state's intended value, plus a prior that
    decreases with position (a crude letter readout); can ignore the question."""

    def __init__(self, position_bias: float = 1.0, read_question: bool = True):
        self.name = "fake"
        self.bias = position_bias
        self.read_question = read_question

    async def decide(self, request: DecisionRequest) -> DecisionResponse:
        state = request.state if isinstance(request.state, str) else ""
        wanted = state[len(PREFIX):] if state.startswith(PREFIX) else None
        answers = {}
        for q in request.questions:
            if q.type == "noul":
                answers[q.name] = Answer(q.name, "noul", {"true": 0.3, "false": 0.7})
                continue
            k = len(q.options)
            scores = []
            for j, o in enumerate(q.options):
                relevant = wanted is not None and (o.value == wanted or (o.description or "")
                                                   .startswith(wanted))
                s = 3.0 * relevant
                text = q.instructions
                if self.read_question and "NOT" in text:
                    s = -s
                if self.read_question and "listed first" in text:
                    s = 5.0 * (j == 0)
                if self.read_question and "listed last" in text:
                    s = 5.0 * (j == k - 1)
                scores.append(s - self.bias * j)
            m = max(scores)
            e = [math.exp(s - m) for s in scores]
            p = {o.value: x / sum(e) for o, x in zip(q.options, e, strict=True)}
            answers[q.name] = Answer(q.name, "choice", p, choice=max(p, key=p.get),
                                     confidence=jev_confidence(list(p.values())))
        q_tokens = sum(len(q.instructions.split()) + 2 * len(q.options)
                       for q in request.questions)
        return DecisionResponse(answers, latency_ms=10.0 + len(request.questions),
                                input_tokens=len(str(request.state).split()) + q_tokens + 7,
                                output_tokens=len(request.questions))


def run_all(client: DecisionClient, target_name: str) -> dict:
    target = ProbeTarget(name=target_name, kind="systemone", url="http://x", concurrency=2)
    items = make_items()

    async def go():
        for name, cls in PROBES.items():
            await run_cases(client, target, name, build_cases(cls(), items))

    asyncio.run(go())
    return analyze_target(target_name)


# --------------------------------------------------------------------------- wire formats
def target(kind="systemone", **kw):
    return ProbeTarget(name="t", kind=kind, url="http://api.test/v1/x", **kw)


def test_systemone_request_and_response():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "model": "microsoft-decision-1",
            "answers": {
                "team": {"type": "choice", "choice": "billing", "confidence": 0.7,
                         "probabilities": {"billing": 0.8, "tech": 0.2}},
                "urgent": {"type": "noul", "noul": 0.9},
                "severity": {"type": "score", "score": 1.2,
                             "probabilities": [0.1, 0.6, 0.3]},
            },
            "usage": {"input_tokens": 120, "output_tokens": 3},
        })

    client = SystemOneClient(target(model="Decision-1"), get_settings())
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    req = DecisionRequest("Charged twice", [
        Question("team", "Which team?", "choice", [Option("billing", "Payments"),
                                                   Option("tech")]),
        Question("urgent", "Is it urgent?", "noul"),
        Question("severity", "How bad?", "score", [Option("low"), Option("mid"),
                                                   Option("high", "Outage")]),
    ])
    resp = asyncio.run(client.decide(req))
    body = seen["body"]
    assert body["model"] == "Decision-1" and body["state"] == "Charged twice"
    assert body["questions"]["team"]["criteria"] == {"billing": "Payments", "tech": "tech"}
    assert body["questions"]["severity"]["criteria"] == ["low", "mid", "Outage"]
    assert "criteria" not in body["questions"]["urgent"]
    assert resp.answers["team"].probabilities == {"billing": 0.8, "tech": 0.2}
    assert resp.answers["urgent"].probabilities["true"] == pytest.approx(0.9)
    assert resp.answers["severity"].probabilities == {"low": 0.1, "mid": 0.6, "high": 0.3}
    assert resp.answers["severity"].choice == "mid"
    assert resp.input_tokens == 120 and resp.output_tokens == 3


def test_openai_decisions_request_and_response():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"answers": [
            {"type": "choice", "name": "team", "choice": "tech", "confidence": 0.5,
             "probabilities": [{"value": "billing", "probability": 0.3},
                               {"value": "tech", "probability": 0.7}]},
            {"type": "predicate", "name": "urgent", "probability": 0.2},
            {"type": "score", "name": "severity", "score": 0.4,
             "probabilities": [{"value": 0, "label": "low", "probability": 0.6},
                               {"value": 1, "label": "high", "probability": 0.4}]},
            {"type": "refusal", "name": "other"},
        ], "usage": {"input_tokens": 50, "output_tokens": 4}})

    client = OpenAIDecisionsClient(target("openai_decisions", model="gpt-6-luna"),
                                   get_settings())
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    req = DecisionRequest({"msg": "hi"}, [
        Question("team", "Which team?", "choice", [Option("billing", "Payments"),
                                                   Option("tech")]),
        Question("urgent", "Urgent?", "noul", [Option("true", "Outage"), Option("false")]),
        Question("severity", "How bad?", "score", [Option("low"), Option("high")]),
        Question("other", "Other?", "noul"),
    ])
    resp = asyncio.run(client.decide(req))
    qs = {q["name"]: q for q in seen["body"]["questions"]}
    assert seen["body"]["input"] == '{"msg": "hi"}'
    assert qs["team"]["choices"] == [{"value": "billing", "description": "Payments"},
                                     {"value": "tech"}]
    assert qs["urgent"]["type"] == "predicate" and "True if: Outage" in qs["urgent"]["instructions"]
    assert qs["severity"]["levels"] == [{"label": "low"}, {"label": "high"}]
    assert resp.answers["team"].probabilities == {"billing": 0.3, "tech": 0.7}
    assert resp.answers["urgent"].probabilities["true"] == pytest.approx(0.2)
    assert resp.answers["severity"].probabilities == {"low": 0.6, "high": 0.4}
    assert resp.answers["other"].type == "refusal"


def test_resolve_auth(monkeypatch):
    assert resolve_auth(target(auth="none"), get_settings()) is None
    monkeypatch.delenv("SOME_TEST_KEY", raising=False)
    with pytest.raises(RuntimeError, match="SOME_TEST_KEY"):
        resolve_auth(target(auth="env:SOME_TEST_KEY"), get_settings())
    monkeypatch.setenv("SOME_TEST_KEY", "k")
    assert resolve_auth(target(auth="env:SOME_TEST_KEY"), get_settings()) is not None


def test_shipped_probe_targets_load():
    from edb.probing.clients import load_probe_target
    from edb.settings import REPO_ROOT

    names = [p.stem for p in (REPO_ROOT / "configs" / "probe_targets").glob("*.yaml")]
    assert "decision-1" in names
    for n in names:
        t = load_probe_target(n)
        assert t.name == n
        if t.kind == "local_readout":
            from edb.config import load_model_config

            load_model_config(t.params["model"])


# --------------------------------------------------------------------------- items + cases
def test_subset_is_nested_and_keeps_gold():
    item = make_items()["banking77"][0]
    small, big = subset(item, 4), subset(item, 8)
    assert item.gold in small.values and len(small.options) == 4
    assert set(small.values) <= set(big.values)
    assert subset(item, 4).values == small.values  # deterministic


def test_load_items_from_dataset(sandbox):
    items = load_items("toy", 5)
    assert len(items) == 5
    assert items[0].options[0].description == "The first one"  # verbose catalog
    assert items[0].id.startswith("toy:") and items[0].source == "toy"


def test_every_probe_builds_unique_cases():
    items = make_items()
    for name, cls in PROBES.items():
        cases = build_cases(cls(), items)
        assert cases, name
        assert all(c.probe == name for c in cases)


# --------------------------------------------------------------------------- end to end
def test_biased_readout_is_detected(sandbox):
    s = run_all(FakeReadout(position_bias=1.0, read_question=True), "biased")
    p = s["probes"]
    assert p["order"]["result"]["banking77"]["first_vs_last_log_lift"] > 0
    dup = p["duplicate"]["result"]["ace"]
    assert dup["earlier_copy_share_dup_after"] > 0.5
    assert dup["earlier_copy_share_dup_before"] > 0.5
    q = p["question"]["result"]["banking77"]
    assert q["accuracy_not_match"] == 1.0 and q["accuracy_listed_first"] == 1.0
    mq = p["multi_question"]["result"]
    assert mq["shared_tokens_mean"] > 0 and mq["max_abs_dp_q1_alone_vs_combined"] == 0
    assert p["determinism"]["result"]["ace"]["identical_all_repeats"] == 1.0
    card = p["cardinality"]["result"]
    assert card["tokens_per_option"] == pytest.approx(2.0, abs=0.01)
    assert card["limits"] == {"k255": "accepted", "k256": "accepted"}
    assert s["forensics"]["confidence_is_jev_formula"] == 1.0
    dn = p["duplicate_names"]["result"]["ace"]
    assert dn["first_share_neutral"] > 0.5  # position bias, no name information
    assert dn["plain_name_share_swap_after"] < 0.5 < dn["plain_name_share_swap_before"]
    lo = p["long_options"]["result"]
    assert lo["tokens_per_filler_repeat"] is not None and "start_150" in lo["per_variant"]
    # one record per case, resumable: a second run sends nothing new
    n = len(read_records(records_path("biased", "order")))
    target = ProbeTarget(name="biased", kind="systemone", url="http://x")
    out = asyncio.run(run_cases(FakeReadout(), target, "order",
                                build_cases(PROBES["order"](), make_items())))
    assert out["submitted"] == 0 and len(read_records(records_path("biased", "order"))) == n


def test_unbiased_question_blind_readout_is_detected(sandbox):
    s = run_all(FakeReadout(position_bias=0.0, read_question=False), "blind")
    p = s["probes"]
    assert p["order"]["result"]["ace"]["flip_rate_vs_original"] == 0
    assert p["duplicate"]["result"]["ace"]["median_abs_log_ratio"] == 0
    assert p["question"]["result"]["banking77"]["same_as_match_not_match"] == 1.0
    report = build_probe_report(["biased", "blind"])  # biased has no summary here: skipped
    text = report.read_text(encoding="utf-8")
    assert "## order" in text and "| blind |" in text


def test_errors_are_recorded_not_raised(sandbox):
    class Rejecting(DecisionClient):
        async def decide(self, request):
            from edb.providers.retry import HTTPError

            raise HTTPError(400, "too many options", httpx.Headers())

    target = ProbeTarget(name="rej", kind="systemone", url="http://x", max_retries=1)
    cases = build_cases(PROBES["empty_state"](), make_items())
    out = asyncio.run(run_cases(Rejecting(), target, "empty_state", cases))
    assert out["errors"] == len(cases)
    rec = next(iter(read_records(records_path("rej", "empty_state")).values()))
    assert rec["response"] is None and "HTTP 400" in rec["error"] and rec["attempts"] == 1


# --------------------------------------------------------------------------- forensics
def test_logit_grid_detects_bf16_softmax():
    logits = [20.0, 17.375, 15.125]  # bf16 values in [16, 32): spacing 0.125
    z = sum(math.exp(x) for x in logits)
    assert forensics.logit_grid([math.exp(x) / z for x in logits]) == 0.125
    assert forensics.logit_grid([0.5, 0.3, 0.2]) == 0.0
    assert forensics.bf16_magnitude(0.125) == "[16, 32)"


def test_main_grid_step_needs_a_majority():
    def rec(probs):
        return {"response": {"latency_ms": 1.0, "output_tokens": 1, "answers": {"q": {
            "type": "choice", "probabilities": dict(enumerate(probs))}}}}

    on = [math.exp(x) for x in (0.0, -0.125, -0.5)]
    off = [0.5, 0.3, 0.2]
    assert forensics.analyze([rec(on), rec(off), rec(off)])["logit_grid_main_step"] is None
    assert forensics.analyze([rec(on), rec(on), rec(off)])["logit_grid_main_step"] == "0.125"


def test_tokens_probe_identifies_tokenizer():
    class WordTok:
        def encode(self, text, add_special_tokens=False):
            return text.split()

    class CharTok:
        def encode(self, text, add_special_tokens=False):
            return list(text)

    recs = [{"meta": {"state": k}, "response": {"input_tokens": len(v.split()) + 9}}
            for k, v in TOKEN_STATES.items()]
    out = TokensProbe().analyze(recs, {"words": WordTok(), "chars": CharTok()})
    assert out["best_tokenizer"] == "words"
    assert out["tokenizer_fits"]["words"] == {"hidden_prompt_tokens": 9,
                                              "exact_fit_fraction": 1.0}
    assert out["tokens_per_apple"] == pytest.approx(1.0)


def test_transient_entra_token_failures_are_retried():
    from azure.core.exceptions import ClientAuthenticationError

    from edb.providers.retry import classify

    assert classify(ClientAuthenticationError("az timed out")) == (True, None)
