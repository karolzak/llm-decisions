"""Parts 2-3 on a tiny random gpt-oss model: chat letter readout, LoRA, training."""

import asyncio
import copy
import json
import os
import textwrap

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from edb.config import load_dataset_config, load_label_catalog, load_model_config  # noqa: E402
from edb.local.chat_readout import (  # noqa: E402
    ChatLetterReadout,
    LocalReadoutClient,
    ReadoutConfig,
)
from edb.local.model import LoadedModel, register  # noqa: E402
from edb.probing.api import DecisionRequest, Option, Question  # noqa: E402
from edb.probing.clients import ProbeTarget  # noqa: E402
from edb.providers.base import MethodContext  # noqa: E402
from edb.providers.registry import build_method  # noqa: E402

READOUT = {"chat_template_kwargs": {}, "answer_prefix": "<|channel|>final<|message|>"}


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
def lm(tokenizer, tiny_model):
    return LoadedModel(copy.deepcopy(tiny_model), tokenizer, torch.device("cpu"),
                       {"model_id": "tiny-oss"})


@pytest.fixture
def tiny_cfg(sandbox, lm):
    (sandbox / "configs" / "models" / "tiny-oss.yaml").write_text(
        "name: tiny-oss\nprovider: local_hf\nhf_model_id: tiny-oss\ngpu_hourly_price_usd: 3.6\n"
        f"params: {{dtype: float32, readout: {json.dumps(READOUT)}}}\n", encoding="utf-8")
    register(("tiny-oss", "float32", None, None), lm)
    return sandbox


def readout(lm, **kw) -> ChatLetterReadout:
    return ChatLetterReadout(lm, ReadoutConfig.from_params(READOUT, kw))


OPTS = [Option("alpha", "first"), Option("beta"), Option("gamma", "third")]


def test_prompt_lists_letters_and_ends_at_answer(lm):
    r = readout(lm)
    text = r.prompt_text({"msg": "hi"}, "Pick one.", OPTS)
    assert "A. alpha: first\nB. beta\nC. gamma: third" in text
    assert '{"msg":"hi"}' in text and text.endswith("<|channel|>final<|message|>")


def test_letter_variants_include_bare_and_spaced_tokens(lm):
    variants = readout(lm).variant_ids()
    assert len(variants) == 26
    a_ids = variants[0]
    assert lm.encode("A")[0] in a_ids and lm.encode(" A")[0] in a_ids
    assert all(lm.tokenizer.decode([i]).strip() == "A" for i in a_ids)


def test_single_pass_and_grouped_scores_are_distributions(lm):
    r = readout(lm)
    probs, diag = r.score_options("state", "Pick one.", OPTS)
    assert sum(probs) == pytest.approx(1.0) and diag["passes"] == 1
    assert 0 < diag["on_format_mass"] <= 1
    many = [Option(f"o{i}") for i in range(30)]
    probs, diag = r.score_options("state", "Pick one.", many)
    assert len(probs) == 30 and sum(probs) == pytest.approx(1.0)
    assert diag["passes"] == 3  # 2 groups of 15 + the winners' pass
    cooled, _ = readout(lm, temperature=4.0).score_options("state", "Pick one.", OPTS)
    hot, _ = r.score_options("state", "Pick one.", OPTS)
    assert max(cooled) <= max(hot) + 1e-9


def test_letter_logprobs_are_differentiable(lm):
    r = readout(lm)
    logits = torch.randn(len(lm.tokenizer), requires_grad=True)
    r.letter_logprobs_from_logits(logits, 3).sum().backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


def test_local_readout_client_answers_every_question_type(tiny_cfg):
    client = LocalReadoutClient(ProbeTarget(name="tiny", kind="local_readout",
                                            params={"model": "tiny-oss"}))
    req = DecisionRequest("Charged twice", [
        Question("team", "Which team?", "choice", OPTS),
        Question("urgent", "Is it urgent?", "noul"),
        Question("severity", "How bad?", "score", [Option("low"), Option("high")]),
    ])
    resp = asyncio.run(client.decide(req))
    team = resp.answers["team"]
    assert list(team.probabilities) == ["alpha", "beta", "gamma"]
    assert team.choice == max(team.probabilities, key=team.probabilities.get)
    assert resp.answers["urgent"].probabilities["true"] + \
        resp.answers["urgent"].probabilities["false"] == pytest.approx(1.0)
    assert 0 <= resp.answers["severity"].score <= 1
    assert resp.input_tokens > 0 and resp.output_tokens == 0


def test_chat_readout_dataset_method(tiny_cfg):
    from edb.datasets.store import load_split

    ds = load_dataset_config("toy")
    ctx = MethodContext(run_id="r", dataset=ds,
                        catalog=load_label_catalog(ds, "verbose", ["alpha", "beta", "gamma"]),
                        model=load_model_config("tiny-oss"), params={})
    method = build_method("local_chat_readout", ctx)
    pred = asyncio.run(method.predict(load_split("toy", "dev")[0]))
    assert pred.ok and set(pred.probabilities) == {"alpha", "beta", "gamma"}
    assert pred.prob_source == "model" and pred.usage.input_tokens > 0
    assert method.info["readout"]["answer_prefix"] == READOUT["answer_prefix"]


# --------------------------------------------------------------------------- LoRA
def test_lora_starts_as_identity_and_roundtrips(tmp_path, lm, tiny_model, tokenizer):
    from edb.training.lora import apply_lora, load_adapter, lora_parameters, save_adapter

    ids = torch.tensor([lm.encode("hello world")])
    before = lm.model(input_ids=ids).logits
    names = apply_lora(lm.model, r=4, alpha=8)
    assert names and all(not p.requires_grad for n, p in lm.model.named_parameters()
                         if "lora_" not in n)
    assert torch.allclose(lm.model(input_ids=ids).logits, before, atol=1e-6)  # B = 0
    for p in lora_parameters(lm.model):
        torch.nn.init.normal_(p, std=0.05)
    changed = lm.model(input_ids=ids).logits
    assert not torch.allclose(changed, before, atol=1e-4)
    save_adapter(lm.model, tmp_path / "ad", {"arm": "x"})
    fresh = copy.deepcopy(tiny_model)
    info = load_adapter(fresh, tmp_path / "ad")
    assert info["modules"] == len(names) and info["arm"] == "x"
    assert torch.allclose(fresh(input_ids=ids).logits, changed, atol=1e-5)


# --------------------------------------------------------------------------- training
def _write_corpus(root, n=12):
    from edb.training.corpus import corpus_dir

    (root / "configs" / "training").mkdir(parents=True, exist_ok=True)
    (root / "configs" / "training" / "corpus.yaml").write_text(textwrap.dedent("""
        name: tiny-mix
        sources: []
    """), encoding="utf-8")
    out = corpus_dir() / "tiny-mix"
    out.mkdir(parents=True, exist_ok=True)
    rows = [{"source": "s", "state": f"text {i}", "instructions": "Pick.",
             "options": [{"value": v, "description": None} for v in ("x", "y", "z")],
             "gold": "xyz"[i % 3]} for i in range(n)]
    for split in ("train", "holdout"):
        (out / f"{split}.jsonl").write_text("\n".join(json.dumps(r) for r in rows),
                                            encoding="utf-8")


@pytest.mark.parametrize("loss,shuffle", [("ce", True), ("brier", False)])
def test_train_arm_end_to_end(tiny_cfg, lm, tiny_model, loss, shuffle):
    from edb.runners.predict import adapter_fingerprint
    from edb.training.lora import load_adapter
    from edb.training.train import LoraConfig, TrainConfig, arm_dir, train

    _write_corpus(tiny_cfg)
    assert adapter_fingerprint("training/t") is None
    cfg = TrainConfig(arm="t", base_model="tiny-oss", loss=loss, shuffle=shuffle, max_steps=2,
                      batch_size=2, lr=1e-2, warmup_steps=1, eval_every=1, eval_rows=4,
                      lora=LoraConfig(r=4, alpha=8, dropout=0.0))
    summary = train(cfg, lm=lm, log=lambda *_: None)
    out = arm_dir("t")
    assert (out / "adapter.safetensors").exists() and (out / "summary.json").exists()
    log = [json.loads(x) for x in (out / "train_log.jsonl").read_text().splitlines()]
    assert log[-1]["step"] == 2 and log[-1]["holdout"]["n"] == 4
    assert summary["examples_seen"] == 4 and summary["trainable_params"] > 0
    assert adapter_fingerprint("training/t") is not None
    # the saved adapter reproduces the trained model's readout
    r_trained = readout(lm)
    fresh = LoadedModel(copy.deepcopy(tiny_model), lm.tokenizer, torch.device("cpu"), {})
    load_adapter(fresh.model, out)
    a, _ = r_trained.score_options("text 1", "Pick.", OPTS)
    b, _ = readout(fresh).score_options("text 1", "Pick.", OPTS)
    assert a == pytest.approx(b, abs=1e-5)


def test_load_arm_merges_defaults():
    from edb.training.train import load_arm

    cfg = load_arm("ce_fixed")
    assert cfg.arm == "ce_fixed" and cfg.loss == "ce" and cfg.shuffle is False
    assert cfg.base_model == "qwen3.5-2b" and cfg.lora.r == 16
    with pytest.raises(KeyError):
        load_arm("nope")
