"""Parts 2-3 on a tiny random gpt-oss model: chat letter readout, LoRA, training."""

import asyncio
import copy
import json
import math
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


def test_prompt_templates_differ_only_in_layout(lm):
    texts = {t: readout(lm, template=t).prompt_text("STATE", "Pick one.", OPTS)
             for t in ("default", "markdown", "json")}
    assert "(A) alpha: first\n(B) beta\n(C) gamma: third" in texts["markdown"]
    assert '"A": "alpha: first"' in texts["json"] and '"input": "STATE"' in texts["json"]
    assert len(set(texts.values())) == 3
    for t in texts:
        probs, _ = readout(lm, template=t).score_options("STATE", "Pick one.", OPTS)
        assert sum(probs) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="unknown template"):
        ReadoutConfig.from_params({"template": "nope"})


def test_letter_variants_include_bare_and_spaced_tokens(lm):
    variants = readout(lm).variant_ids()
    assert len(variants) == 26
    a_ids = variants[0]
    assert lm.encode("A")[0] in a_ids and lm.encode(" A")[0] in a_ids
    assert all(lm.tokenizer.decode([i]).strip() == "A" for i in a_ids)


def test_code_labels_extend_letters_in_one_pass(lm):
    r = readout(lm, labels="codes")
    assert r.label_strings(3) == ["A", "B", "C"]
    assert r.prompt_text("S", "Pick.", OPTS) == readout(lm).prompt_text("S", "Pick.", OPTS)
    many = [Option(f"o{i}") for i in range(40)]
    labels = r.label_strings(40)
    assert labels[25] == "Z" and labels[26] == "AA" and len(set(labels)) == 40
    assert all(len(lm.encode(lbl)) == 1 for lbl in labels)
    text = r.prompt_text("S", "Pick.", many)
    assert "\nAA. o26" in text and "letter code" in text
    probs, diag = r.score_options("S", "Pick.", many)
    assert diag["passes"] == 1 and sum(probs) == pytest.approx(1.0)


def test_number_labels_are_scored_exactly(lm):
    import torch as t

    r = readout(lm, labels="numbers")
    opts = [Option(f"o{i}") for i in range(12)]
    ids = r.prompt_ids("S", "Pick.", opts)
    assert "\n12. o11" in r.prompt_text("S", "Pick.", opts) and "number" in \
        r.prompt_text("S", "Pick.", opts)
    probs, diag = r.score_options("S", "Pick.", opts)
    assert diag["passes"] == 1 and sum(probs) == pytest.approx(1.0)

    digit = [lm.encode(str(d))[0] for d in range(10)]

    def lp(seq):
        with t.inference_mode():
            return t.log_softmax(lm.model(input_ids=t.tensor([seq])).logits[0, -1].float(), -1)

    def label_lp(n):
        seq, total = list(ids), 0.0
        for c in str(n):
            total += float(lp(seq)[digit[int(c)]])
            seq.append(digit[int(c)])
        return total + math.log(1 - float(lp(seq)[digit].exp().sum()))

    raw = [label_lp(n) for n in range(1, 13)]
    z = t.logsumexp(t.tensor(raw), 0)
    assert probs == pytest.approx([math.exp(x - float(z)) for x in raw], abs=1e-4)


def test_unknown_label_scheme_rejected():
    with pytest.raises(ValueError, match="unknown labels"):
        ReadoutConfig.from_params({"labels": "emoji"})
    with pytest.raises(ValueError, match="noul_style"):
        ReadoutConfig.from_params({"noul_style": "maybe"})
    with pytest.raises(ValueError, match="score_style"):
        ReadoutConfig.from_params({"score_style": "stars"})


LEVELS = [Option(t, t) for t in ("bad", "okay", "good")]
TYPED_QS = [Question("c", "Which?", "choice", OPTS),
            Question("s", "How good?", "score", LEVELS),
            Question("n", "Is it good?", "noul"),
            Question("nc", "Is it good?", "noul",
                     [Option("true", "It is good."), Option("false", "It is not good.")])]


@pytest.mark.parametrize("noul_style,score_style", [("letters", "letters"),
                                                    ("truefalse", "digits"),
                                                    ("yesno", "scale")])
def test_typed_templates_answer_every_type(lm, noul_style, score_style):
    r = readout(lm, noul_style=noul_style, score_style=score_style)
    for q in TYPED_QS:
        a, diag = r.answer("The food was fine.", q)
        assert sum(a.probabilities.values()) == pytest.approx(1.0)
        assert diag["passes"] == 1
        if q.type == "score":
            assert a.score == pytest.approx(sum(i * a.probabilities[o.value]
                                                for i, o in enumerate(LEVELS)))
    ids, _, values = r.plan("x", TYPED_QS[1])
    text = lm.tokenizer.decode(ids)
    if score_style == "digits":
        assert "0. bad" in text and "2. good" in text and values == ["bad", "okay", "good"]
    elif score_style == "scale":
        assert "0 = bad" in text and "from 0 to 2" in text
    else:
        assert "A. bad: bad" in text
    noul_text = lm.tokenizer.decode(r.plan("x", TYPED_QS[3])[0])
    if noul_style == "truefalse":
        assert "true: It is good." in noul_text and "true or false" in noul_text
    elif noul_style == "yesno":
        assert "yes: It is good." in noul_text and "yes or no" in noul_text


def test_letters_style_matches_previous_noul_readout(lm):
    """`plan` reproduces the old choice-based path for noul questions exactly."""
    r = readout(lm)
    q = TYPED_QS[3]
    a, _ = r.answer("The food was fine.", q)
    options = [Option(v, {o.value: o.description for o in q.options}[v]) for v in
               r.cfg.noul_order]
    probs, _ = r.score_options("The food was fine.", q.instructions, options)
    assert a.probabilities["true"] == pytest.approx(probs[1], abs=1e-6)


def test_shared_state_prefix_matches_separate_answers(lm):
    r = readout(lm, noul_style="truefalse", score_style="digits")
    state = "A long state. " * 20
    many = Question("m", "Which of many?", "choice", [Option(f"o{i}") for i in range(30)])
    qs = [*TYPED_QS, many]  # `many` needs grouped passes and is answered separately
    shared, diags, tokens = r.answer_shared(state, qs)
    separate = {q.name: r.answer(state, q) for q in qs}
    for q in qs:
        for v, p in separate[q.name][0].probabilities.items():
            assert shared[q.name].probabilities[v] == pytest.approx(p, abs=1e-5)
    assert diags["c"]["shared_prefix_tokens"] > 50
    assert "shared_prefix_tokens" not in diags["m"]
    assert tokens < sum(d["input_tokens"] for _, d in separate.values())


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


def test_adapters_disabled_restores_base_and_state_roundtrips(lm):
    from edb.training.lora import (
        adapters_disabled,
        apply_lora,
        load_lora_state,
        lora_parameters,
        lora_state,
    )

    ids = torch.tensor([lm.encode("hello world")])
    before = lm.model(input_ids=ids).logits
    apply_lora(lm.model, r=4, alpha=8)
    for p in lora_parameters(lm.model):
        torch.nn.init.normal_(p, std=0.05)
    changed = lm.model(input_ids=ids).logits
    with adapters_disabled(lm.model):
        assert torch.allclose(lm.model(input_ids=ids).logits, before, atol=1e-6)
    assert torch.allclose(lm.model(input_ids=ids).logits, changed)
    state = {k: v.clone() for k, v in lora_state(lm.model).items()}
    for p in lora_parameters(lm.model):
        torch.nn.init.zeros_(p)
    load_lora_state(lm.model, state)
    assert torch.allclose(lm.model(input_ids=ids).logits, changed)


# --------------------------------------------------------------------------- augmentations
def test_augmentations_build_the_right_targets():
    import random

    from edb.training.train import AugConfig, augment

    row = {"state": "s", "instructions": "Which?"}
    opts = [Option("a", "da"), Option("b"), Option("c")]
    seen = set()
    for kind, aug in [("empty", AugConfig(empty=1)), ("negation", AugConfig(negation=1)),
                      ("decoy", AugConfig(decoy=1)), ("paraphrase", AugConfig(paraphrase=1)),
                      ("plain", AugConfig())]:
        for seed in range(20):
            state, q, o, target, got = augment(row, opts, 1, aug, random.Random(seed))
            assert got == kind and len(o) == len(target)
            assert sum(target) == pytest.approx(1.0)
            seen.add(kind)
            if kind == "empty":
                assert target == [pytest.approx(1 / 3)] * 3 and state != "s"
            elif kind == "negation":
                assert len(o) == 2 and "NOT" in q and "b" in [x.value for x in o]
                assert o[target.index(1.0)].value != "b"  # target is the distractor
            elif kind == "decoy":
                values = [x.value for x in o]
                assert len(o) == 4 and len(set(values)) == 4
                assert o[target.index(1.0)].value == "b"  # the original name wins
            elif kind == "paraphrase":
                assert "Which?" in q and o == opts
            else:
                assert (state, q, o) == ("s", "Which?", opts) and target == [0, 1, 0]
    assert len(seen) == 5


def test_none_option_augmentations():
    import random

    from edb.training.train import AugConfig, augment, none_index

    row = {"state": "s", "instructions": "Which?", "source": "a"}
    pool = [{"state": "other task text", "source": "b"}]
    opts = [Option("a"), Option("b"), Option("c")]
    na = [Option("a"), Option("Not applicable"), Option("c")]
    for seed in range(30):
        rng = random.Random(seed)
        # empty + an existing "none" option -> target it; without one -> uniform
        _, _, o, t, kind = augment(row, na, 0, AugConfig(empty=1, none_aware=True), rng)
        assert kind == "empty" and t == [0.0, 1.0, 0.0]
        _, _, o, t, _ = augment(row, opts, 0, AugConfig(empty=1, none_aware=True), rng)
        assert t == [pytest.approx(1 / 3)] * 3
        # empty_none adds a "none" option and targets it
        _, _, o, t, _ = augment(row, opts, 0, AugConfig(empty=1, none_aware=True,
                                                        empty_none=1), rng)
        assert len(o) == 4 and t.index(1.0) == none_index(o)
        # mismatch: another task's state, "none" wins
        s, _, o, t, kind = augment(row, opts, 1, AugConfig(mismatch=1), rng, pool)
        assert kind == "mismatch" and s == "other task text"
        assert len(o) == 4 and t.index(1.0) == none_index(o)
        # none_option: a "none" distractor, the gold option keeps the target
        _, _, o, t, kind = augment(row, opts, 1, AugConfig(none_option=1), rng)
        assert kind == "none_option" and len(o) == 4 and o[t.index(1.0)].value == "b"
        assert none_index(o) is not None
    # R9 behaviour is unchanged by default: empty targets stay uniform even with "none"
    _, _, o, t, _ = augment(row, na, 0, AugConfig(empty=1), random.Random(0))
    assert t == [pytest.approx(1 / 3)] * 3
    assert none_index([Option("out_of_scope")]) == 0
    assert none_index([Option("Not-Applicable")]) == 0 and none_index([Option("nonce")]) is None


def test_typed_corpus_rows():
    from edb.training.corpus import CorpusSpec, SourceSpec, to_decisions

    spec = CorpusSpec(name="t", sources=[])
    cl = {"label": {"_type": "ClassLabel", "names": ["one", "two", "three"]}}
    rows = [{"text": f"t{i}", "label": i % 3} for i in range(30)]
    score = SourceSpec(name="s", dataset="d", state="{text}", label="label", type="score",
                       instructions=["Rate it."], levels=[["low", "mid", "high"],
                                                          ["1", "2", "3"]])
    out = to_decisions(score, spec, rows, cl)
    assert len(out) == 30 and {r["type"] for r in out} == {"score"}
    for r, row in zip(out, rows, strict=True):
        values = [o["value"] for o in r["options"]]
        assert values in (["low", "mid", "high"], ["1", "2", "3"])
        assert values.index(r["gold"]) == row["label"]
    rounded = SourceSpec(name="r", dataset="d", state="{text}", label="label", type="score",
                         instructions=["Rate."], round_label=True, levels=[["0", "1", "2"]])
    out = to_decisions(rounded, spec, [{"text": "a", "label": 1.6}, {"text": "b",
                                                                      "label": 0.2}], {})
    assert [r["gold"] for r in out] == ["2", "0"]
    about = SourceSpec(name="a", dataset="d", state="{text}", label="label", type="noul",
                       about=True, instructions=["Is it about {label}?"],
                       criteria={"true": "It is {label}.", "false": "Something else."})
    out = to_decisions(about, spec, rows, cl)
    assert {r["gold"] for r in out} == {"true", "false"}
    for r, row in zip(out, rows, strict=True):
        asked = r["instructions"][len("Is it about "):-1]
        assert (asked == ["one", "two", "three"][row["label"]]) == (r["gold"] == "true")
        assert r["options"] == [] or r["options"][0]["description"] == f"It is {asked}."
    boolean = SourceSpec(name="b", dataset="d", state="{text}", label="answer", type="noul",
                         true_labels=["true"], instructions=["Is it?"])
    out = to_decisions(boolean, spec, [{"text": "x", "answer": True},
                                       {"text": "y", "answer": False}], {})
    assert [r["gold"] for r in out] == ["true", "false"]


def test_rps_prefers_near_misses():
    from edb.training.train import rps

    target = [0.0, 0.0, 1.0, 0.0, 0.0]
    near = torch.log(torch.tensor([0.01, 0.01, 0.48, 0.49, 0.01]))
    far = torch.log(torch.tensor([0.49, 0.01, 0.48, 0.01, 0.01]))
    exact = torch.log(torch.tensor([1e-6, 1e-6, 1 - 4e-6, 1e-6, 1e-6]))
    assert float(rps(near, target)) < float(rps(far, target))
    assert float(rps(exact, target)) == pytest.approx(0.0, abs=1e-6)


def test_soft_target_losses():
    from edb.training.train import loss_fn

    logp = torch.log(torch.tensor([0.5, 0.25, 0.25]))
    assert float(loss_fn(logp, 0, "ce")) == pytest.approx(-math.log(0.5))
    assert float(loss_fn(logp, [1.0, 0, 0], "ce_brier")) == pytest.approx(
        float(loss_fn(logp, 0, "ce_brier")))
    uniform = [1 / 3] * 3
    assert float(loss_fn(logp, uniform, "brier")) == pytest.approx(
        (0.5 - 1 / 3) ** 2 + 2 * (0.25 - 1 / 3) ** 2)


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
    r9 = load_arm("r9_qwen9b", "arms-r9")
    assert r9.base_model == "qwen3.5-9b" and r9.readout == {"labels": "codes"}
    assert r9.experiment == "r9_qwen9b" and r9.probe_target == "r9-qwen9b"
    assert r9.kl_weight > 0 and r9.aug.decoy > 0 and r9.checkpoint_every
    kl, nokl = load_arm("r9b_kl", "arms-r9b"), load_arm("r9b_nokl", "arms-r9b")
    assert kl.kl_weight == 1.0 and "none_option" in kl.kl_kinds and nokl.kl_weight == 0
    assert kl.aug == nokl.aug and kl.aug.none_aware and kl.aug.mismatch > 0


def _wide_corpus(root, n=8, k=30):
    """Rows with more options than letters, so training must use code labels."""
    from edb.training.corpus import corpus_dir

    (root / "configs" / "training").mkdir(parents=True, exist_ok=True)
    (root / "configs" / "training" / "wide.yaml").write_text(
        "name: wide-mix\nsources: []\n", encoding="utf-8")
    out = corpus_dir() / "wide-mix"
    out.mkdir(parents=True, exist_ok=True)
    rows = [{"source": f"s{i % 2}", "state": f"text {i}", "instructions": "Pick.",
             "options": [{"value": f"v{j}", "description": None} for j in range(k)],
             "gold": f"v{i % k}"} for i in range(n)]
    for split in ("train", "holdout"):
        (out / f"{split}.jsonl").write_text("\n".join(json.dumps(r) for r in rows),
                                            encoding="utf-8")


def _r9_like(**kw):
    from edb.training.train import AugConfig, LoraConfig, TrainConfig

    base = dict(arm="w", base_model="tiny-oss", corpus="wide", loss="ce_brier", max_steps=3,
                batch_size=2, lr=1e-2, warmup_steps=1, eval_every=3, eval_rows=2,
                max_tokens=4096, lora=LoraConfig(r=4, alpha=8, dropout=0.0),
                readout={"labels": "codes"}, kl_weight=0.5,
                kl_kinds=["plain", "paraphrase", "none_option"],
                aug=AugConfig(decoy=0.15, negation=0.1, empty=0.15, paraphrase=0.15,
                              mismatch=0.15, none_option=0.15, empty_none=0.5,
                              none_aware=True))
    return TrainConfig(**{**base, **kw})


def test_train_with_codes_kl_and_augmentations(tiny_cfg, lm):
    from edb.training.train import arm_dir, train

    _wide_corpus(tiny_cfg)
    summary = train(_r9_like(), lm=lm, log=lambda *_: None)
    assert summary["examples_seen"] == 6 and sum(summary["examples_by_kind"].values()) == 6
    assert summary["final"]["holdout"]["n"] == 2
    assert not (arm_dir("w") / "checkpoint.pt").exists()


def test_train_on_all_question_types(tiny_cfg, lm):
    """R15: score and noul rows are read with the typed templates and trained."""
    from edb.training.corpus import corpus_dir
    from edb.training.train import train

    (tiny_cfg / "configs" / "training").mkdir(parents=True, exist_ok=True)
    (tiny_cfg / "configs" / "training" / "typed.yaml").write_text(
        "name: typed-mix\nsources: []\n", encoding="utf-8")
    out = corpus_dir() / "typed-mix"
    out.mkdir(parents=True, exist_ok=True)
    levels = [{"value": t, "description": t} for t in ("low", "mid", "high")]
    rows = []
    for i in range(6):
        rows += [{"source": "c", "state": f"c{i}", "instructions": "Pick.",
                  "options": [{"value": v, "description": None} for v in "xyz"],
                  "gold": "xyz"[i % 3]},
                 {"source": "s", "type": "score", "state": f"s{i}", "instructions": "Rate.",
                  "options": levels, "gold": ["low", "mid", "high"][i % 3]},
                 {"source": "n", "type": "noul", "state": f"n{i}", "instructions": "Is it?",
                  "options": [], "gold": ["true", "false"][i % 2]}]
    for split in ("train", "holdout"):
        (out / f"{split}.jsonl").write_text("\n".join(json.dumps(r) for r in rows),
                                            encoding="utf-8")
    cfg = _r9_like(arm="typed", corpus="typed", max_steps=3, batch_size=4, eval_rows=18,
                   eval_every=3, rps_weight=1.0,
                   kl_kinds=["plain", "paraphrase", "score", "noul"],
                   readout={"labels": "codes", "noul_style": "truefalse",
                            "score_style": "digits"})
    summary = train(cfg, lm=lm, log=lambda *_: None)
    kinds = summary["examples_by_kind"]
    assert kinds.get("score", 0) > 0 and kinds.get("noul", 0) > 0
    by_type = summary["final"]["holdout"]["accuracy_by_type"]
    assert set(by_type) == {"choice", "score", "noul"}


def test_training_resumes_from_checkpoint(tiny_cfg, tiny_model, tokenizer, monkeypatch):
    import edb.training.train as tr
    from edb.training.lora import lora_state

    class Crash(Exception):
        pass

    def crash(*_a, **_k):
        raise Crash

    def fresh_lm():
        return LoadedModel(copy.deepcopy(tiny_model), tokenizer, torch.device("cpu"),
                           {"model_id": "tiny-oss"})

    def interrupted(cfg):
        """Train until the final save, leaving the last periodic checkpoint behind."""
        with monkeypatch.context() as m:
            m.setattr(tr, "save_adapter", crash)
            with pytest.raises(Crash):
                tr.train(cfg, lm=fresh_lm(), log=lambda *_: None)
        assert (tr.arm_dir(cfg.arm) / "checkpoint.pt").exists()

    _wide_corpus(tiny_cfg)
    reference = fresh_lm()
    tr.train(_r9_like(arm="ref", checkpoint_every=1), lm=reference, log=lambda *_: None)

    interrupted(_r9_like(arm="cut", checkpoint_every=1))
    logs: list[str] = []
    resumed = fresh_lm()
    summary = tr.train(_r9_like(arm="cut", checkpoint_every=1), lm=resumed, log=logs.append)
    assert json.loads(logs[0]) == {"resumed_at_step": 2}
    assert summary["examples_seen"] == 6
    assert not (tr.arm_dir("cut") / "checkpoint.pt").exists()
    ref_state, got = lora_state(reference.model), lora_state(resumed.model)
    assert all(torch.allclose(ref_state[k], got[k], atol=1e-6) for k in ref_state)

    interrupted(_r9_like(arm="cut2", checkpoint_every=1))
    with pytest.raises(RuntimeError, match="different config"):
        tr.train(_r9_like(arm="cut2", checkpoint_every=1, lr=5e-3), lm=fresh_lm(),
                 log=lambda *_: None)
