"""Local (C-series) methods on a tiny random gpt-oss model with the real tokenizer."""

import asyncio
import math
import os

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from edb.config import load_dataset_config, load_label_catalog, load_model_config  # noqa: E402
from edb.local import harmony  # noqa: E402
from edb.local.model import LoadedModel, register  # noqa: E402
from edb.local.scoring import continuation_scores  # noqa: E402
from edb.providers.base import MethodContext  # noqa: E402
from edb.providers.registry import build_method  # noqa: E402

LABELS = ["alpha", "beta", "gamma"]


@pytest.fixture(scope="module")
def tokenizer():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        return transformers.AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"gpt-oss tokenizer not cached: {exc}")


@pytest.fixture(scope="module")
def tiny_model(tokenizer):
    torch.manual_seed(0)
    cfg = transformers.GptOssConfig(
        num_hidden_layers=2, hidden_size=32, intermediate_size=32, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, num_local_experts=4, num_experts_per_tok=2,
        sliding_window=16, vocab_size=len(tokenizer),
        layer_types=["sliding_attention", "full_attention"],
    )
    cfg._attn_implementation = "eager"
    return transformers.GptOssForCausalLM(cfg).eval()


@pytest.fixture
def local_ctx(sandbox, tokenizer, tiny_model):
    models = sandbox / "configs" / "models"
    (models / "tiny-oss.yaml").write_text(
        "name: tiny-oss\nprovider: local_hf\nhf_model_id: tiny-oss\ngpu_hourly_price_usd: 3.6\n"
        "params: {dtype: float32}\n", encoding="utf-8")
    register(("tiny-oss", "float32", None, None),
             LoadedModel(tiny_model, tokenizer, torch.device("cpu"), {"model_id": "tiny-oss"}))
    ds = load_dataset_config("toy")

    def make(method_type, params, role="verbose"):
        ctx = MethodContext(run_id="c", dataset=ds,
                            catalog=load_label_catalog(ds, role, LABELS),
                            model=load_model_config("tiny-oss"), params=params)
        return build_method(method_type, ctx)

    return make


def _example():
    from edb.datasets.store import load_split

    return load_split("toy", "dev")[0]


def test_harmony_matches_chat_template(tokenizer):
    msgs = [{"role": "system", "content": "INSTR"}, {"role": "user", "content": "STATE"}]
    reference = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                              reasoning_effort="low")
    ours = harmony.render("INSTR", "STATE", reasoning="low")
    import re
    norm = re.compile(r"Current date: \d{4}-\d{2}-\d{2}")
    assert norm.sub("D", ours) == norm.sub("D", reference)
    text = "<|channel|>analysis<|message|>think<|end|><|start|>assistant<|channel|>final" \
           "<|message|> beta <|return|>"
    assert harmony.final_message(text) == "beta" and harmony.analysis_text(text) == "think"


def test_fp32_head_matches_model_logits(tiny_model):
    from edb.local.scoring import _forward

    ids = torch.tensor([list(range(100, 140))])
    keep = torch.tensor([10, 20, 39])
    ours, _ = _forward(tiny_model, ids, keep)
    ref = tiny_model(input_ids=ids).logits[:, keep].float()
    assert torch.allclose(ours, ref, atol=1e-5)


def test_kv_equals_independent_on_long_prefix(tiny_model):
    prefix = list(range(100, 160))  # longer than the sliding window
    conts = [[5, 6, 7], [8], [9, 10]]
    a = continuation_scores(tiny_model, prefix, conts, "independent", query_ids=[1, 2])
    b = continuation_scores(tiny_model, prefix, conts, "kv", query_ids=[1, 2])
    for x, y in zip(a, b, strict=True):
        assert x.token_logprobs == pytest.approx(y.token_logprobs, abs=1e-4)
        assert x.query_logprobs == pytest.approx(y.query_logprobs, abs=1e-4)


@pytest.mark.parametrize("method_type", ["local_sequence_scoring", "local_cross_encoder"])
def test_kv_and_independent_methods_agree(local_ctx, method_type):
    ex = _example()
    p_ind = asyncio.run(local_ctx(method_type, {"kv_reuse": False}).predict(ex))
    p_kv = asyncio.run(local_ctx(method_type, {"kv_reuse": True}).predict(ex))
    assert p_ind.ok and p_kv.ok
    for v in LABELS:
        assert p_ind.probabilities[v] == pytest.approx(p_kv.probabilities[v], abs=1e-4)
    assert p_kv.usage.input_tokens < p_ind.usage.input_tokens  # prefix encoded once
    assert p_kv.cost_usd > 0 and p_kv.extras["gpu_ms"] > 0


def test_unbatched_sequence_scoring_agrees(local_ctx):
    ex = _example()
    batched = asyncio.run(local_ctx("local_sequence_scoring", {}).predict(ex))
    single = asyncio.run(local_ctx("local_sequence_scoring", {"batched": False}).predict(ex))
    for v in LABELS:
        assert batched.probabilities[v] == pytest.approx(single.probabilities[v], abs=1e-4)


def test_autoregressive_handles_missing_final_answer(local_ctx):
    pred = asyncio.run(local_ctx("local_autoregressive", {"max_new_tokens": 4}).predict(_example()))
    assert not pred.ok and "final-channel" in pred.error
    assert pred.usage.output_tokens <= 4 and pred.usage.input_tokens > 0


def test_listwise_positions_and_head(local_ctx, tokenizer):
    method = local_ctx("local_listwise_head",
                       {"layers": [1, 2], "features": ["option", "option_query"], "l2": [0.1],
                        "max_iter": 20})
    lm = method._load()
    ids, positions = method.build(lm, _example())
    assert positions[-1] == len(ids) - 1 and len(positions) == len(LABELS) + 1
    assert tokenizer.decode(ids[: positions[1] + 1]).endswith("[B] beta: The second one")
    asyncio.run(method.setup())
    assert method.info["chosen"]["layer"] in (1, 2) and method.info["fit_n"] > 0
    pred = asyncio.run(method.predict(_example()))
    assert pred.ok and abs(sum(pred.probabilities.values()) - 1) < 1e-6
    # features are cached on disk: a second setup must not recompute them
    again = local_ctx("local_listwise_head", {"layers": [1, 2], "l2": [0.1], "max_iter": 20})
    calls = {"n": 0}
    original = again.features

    def counting(*a, **k):
        calls["n"] += 1
        return original(*a, **k)

    again.features = counting
    asyncio.run(again.setup())
    assert calls["n"] == 0


def test_train_head_learns_separable_features():
    from edb.local.listwise import train_head

    torch.manual_seed(0)
    N, K, D = 200, 3, 8
    F = torch.randn(N, K + 1, 1, D)
    y = torch.randint(0, K, (N,))
    F[torch.arange(N), y, 0, 0] += 3.0  # the gold option has a large first feature
    head = train_head(F, y, 0, "option", l2=1e-3, max_iter=50)
    acc = (head.logits(F).argmax(1) == y).float().mean()
    assert acc > 0.9


def test_option_markers_and_rotation_shifts():
    from edb.local.readout import option_markers, rotation_shifts
    assert option_markers(3) == ["A", "B", "C"]
    assert option_markers(30)[:3] == ["1", "2", "3"] and option_markers(30)[-1] == "30"
    assert rotation_shifts(3, "all", 8) == [0, 1, 2]
    assert rotation_shifts(3, 1, 8) == [0]
    assert rotation_shifts(77, "all", 4) == [0, 19, 38, 58]


def test_layer_outputs_match_full_forward_and_restore(tiny_model):
    from edb.local.scoring import layer_outputs
    ids = list(range(100, 130))
    n = len(tiny_model.model.layers)
    full = tiny_model(input_ids=torch.tensor([ids]), output_hidden_states=True)
    ours = layer_outputs(tiny_model, ids, [1])
    assert torch.allclose(ours[0], full.hidden_states[1][0, -1].float(), atol=1e-5)
    assert len(tiny_model.model.layers) == n  # truncation restored


def test_letter_readout_rotations_and_prior(local_ctx):
    ex = _example()
    plain = local_ctx("local_letter_readout", {"rotations": 1})
    rot = local_ctx("local_letter_readout", {"rotations": "all"})
    asyncio.run(plain.setup())
    asyncio.run(rot.setup())
    lm = plain._load()
    single, _ = plain.position_logprobs(lm, ex.state)
    multi, _ = rot.position_logprobs(lm, ex.state)
    assert len(multi) == 3
    assert single[0] == pytest.approx(multi[0], abs=1e-4)  # kv branch == direct forward
    p = asyncio.run(rot.predict(ex))
    assert p.ok and abs(sum(p.probabilities.values()) - 1) < 1e-6 and p.extras["rotations"] == 3
    debiased = local_ctx("local_letter_readout", {"rotations": "all", "prior": "content_free"})
    asyncio.run(debiased.setup())
    assert debiased.prior_lp is not None and len(debiased.prior_lp) == 3
    q = asyncio.run(debiased.predict(ex))
    assert q.ok and q.probabilities != p.probabilities


def test_sequence_scoring_content_free_prior(local_ctx):
    ex = _example()
    base = local_ctx("local_sequence_scoring", {})
    corrected = local_ctx("local_sequence_scoring", {"prior": "content_free"})
    asyncio.run(corrected.setup())
    a = asyncio.run(base.predict(ex))
    b = asyncio.run(corrected.predict(ex))
    prior = corrected.prior_logprob
    raw = a.extras["sum_logprob"]
    expected = {v: raw[v] - prior[v] for v in LABELS}
    m = max(expected.values())
    z = sum(math.exp(x - m) for x in expected.values())
    for v in LABELS:
        assert b.probabilities[v] == pytest.approx(math.exp(expected[v] - m) / z, abs=1e-6)


def test_early_exit_head_fits_and_runs_truncated(local_ctx):
    method = local_ctx("local_early_exit_head", {"layers": [1, 2], "l2": [0.1], "max_iter": 20})
    asyncio.run(method.setup())
    assert method.info["chosen"]["layer"] in (1, 2) and method.info["fit_n"] > 0
    pred = asyncio.run(method.predict(_example()))
    assert pred.ok and pred.extras["exit_layer"] == method.head.layer
    assert abs(sum(pred.probabilities.values()) - 1) < 1e-6


def test_reason_then_read_forces_final_channel(local_ctx):
    pred = asyncio.run(local_ctx("local_reason_then_read", {"max_new_tokens": 3})
                       .predict(_example()))
    assert pred.ok and pred.extras["forced_final"] is True
    assert abs(sum(pred.probabilities.values()) - 1) < 1e-6
    assert pred.usage.output_tokens > 3  # generated tokens + forced final-channel tokens
