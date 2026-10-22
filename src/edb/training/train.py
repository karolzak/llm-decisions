"""Train a LoRA adapter on the option-label readout (Part 3 arms, R9).

The model reads exactly the prompt `ChatLetterReadout` builds at inference time. The loss is
computed on the K option labels only (log P(label) summed over its token variants, then
renormalised over the K labels), against a target distribution (one-hot, or uniform for
uninformative states):

    ce        cross-entropy to the target             (what fine-tuning usually optimises)
    brier     sum_k (p_k - target_k)^2                (a proper scoring rule: rewards honest
                                                      probabilities, a differentiable stand-in
                                                      for "RL for calibrated decisions")
    ce_brier  both

`shuffle` re-orders the options of every example every time it is seen (the label follows).
`kl_weight` adds KL(frozen base || model) on the label distribution of the example kinds in
`kl_kinds` (default: ordinary and paraphrased rows), which keeps the base model's judgement
(the frozen base is the same network with LoRA disabled).
`aug` mixes in rows that teach interface behaviour (see `augment`), including R9b's "none"
option: uninformative or mismatched states should pick it when it is listed.
With `checkpoint_every`, training state is saved periodically and a restarted job resumes
from it.
"""

from __future__ import annotations

import json
import math
import random
import re
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from pydantic import BaseModel, Field

from edb.config import _load_yaml, _resolve, load_model_config
from edb.local.chat_readout import ChatLetterReadout, ReadoutConfig
from edb.local.model import LoadedModel, load
from edb.probing.api import Option, Question
from edb.settings import get_settings
from edb.training.corpus import build_corpus, read_jsonl
from edb.training.lora import (
    adapters_disabled,
    apply_lora,
    load_lora_state,
    lora_parameters,
    lora_state,
    save_adapter,
)

EVAL_SEED = 1234
CHECKPOINT = "checkpoint.pt"


class LoraConfig(BaseModel):
    r: int = 16
    alpha: float = 32.0
    dropout: float = 0.05
    pattern: str | None = None  # regex over module names in the decoder blocks (None: all)


class AugConfig(BaseModel):
    """Probability that a training row is turned into each kind of interface example."""

    decoy: float = 0.0  # add a copy of the gold option under a variant or unrelated name
    negation: float = 0.0  # two options; "which does NOT ..." -> the distractor
    empty: float = 0.0  # uninformative state -> a "none" option if listed, else uniform
    paraphrase: float = 0.0  # reword the question
    # R9b: teach a "none / not applicable" option instead of only uniform answers
    mismatch: float = 0.0  # state taken from another task + a "none" option -> "none"
    none_option: float = 0.0  # a "none" option added as a distractor -> gold unchanged
    empty_none: float = 0.0  # chance an empty example gets a "none" option added
    none_aware: bool = False  # empty examples target an existing "none" option


class TrainConfig(BaseModel):
    arm: str
    base_model: str
    corpus: str = "corpus"
    loss: Literal["ce", "brier", "ce_brier"] = "ce"
    shuffle: bool = True
    max_steps: int = 1500
    batch_size: int = 16
    lr: float = 1e-4
    warmup_steps: int = 50
    max_tokens: int = 1024
    lora: LoraConfig = LoraConfig()
    eval_every: int = 250
    eval_rows: int = 400
    seed: int = 0
    max_rows: int | None = None  # cap on training rows (smoke tests)
    readout: dict[str, Any] = Field(default_factory=dict)  # overrides the model's readout
    kl_weight: float = 0.0
    kl_kinds: list[str] = Field(default_factory=lambda: ["plain", "paraphrase"])
    rps_weight: float = 0.0  # ranked probability score on score rows (R15)
    aug: AugConfig = AugConfig()
    checkpoint_every: int = 0  # optimizer steps between checkpoints (0: none)
    # Names used by scripts/train_decider.py (default: r3_<arm> / r3-<arm>).
    experiment: str | None = None
    probe_target: str | None = None


def load_arm(arm: str, file: str = "arms") -> TrainConfig:
    data = _load_yaml(_resolve("training", file))
    arms = data.pop("arms")
    if arm not in arms:
        raise KeyError(f"unknown arm {arm!r}; known: {sorted(arms)}")
    merged = {**data, **arms[arm]}
    for key in ("aug", "lora", "readout"):  # nested blocks merge key by key
        if isinstance(data.get(key), dict) and isinstance(arms[arm].get(key), dict):
            merged[key] = {**data[key], **arms[arm][key]}
    return TrainConfig(arm=arm, **merged)


def arm_dir(arm: str) -> Path:
    return get_settings().artifacts_dir / "training" / arm


def arrange(row: dict[str, Any], shuffle: bool, rng: random.Random) -> tuple[list[Option], int]:
    options = [Option(o["value"], o.get("description")) for o in row["options"]]
    if shuffle:
        rng.shuffle(options)
    return options, [o.value for o in options].index(row["gold"])


# ------------------------------------------------------------------ augmentations
EMPTY_STATES = ["", "N/A", "(no information provided)", "Lorem ipsum dolor sit amet."]
DECOY_SUFFIXES = ["-2", " (copy)", "_v2", " (duplicate)", " old"]
UNRELATED_NAMES = ["zebra_crossing", "violin", "harbour", "quantum_leap", "teapot", "glacier"]
PARAPHRASES = ["{q}", "Question: {q}", "{q} Choose the best option.", "Decide: {q}",
               "Based on the text: {q}", "{q} Pick exactly one."]
NONE_NAMES = ["None of the above", "Not applicable", "N/A", "no_relevant_option",
              "Not enough information", "none"]
NONE_DESCRIPTIONS = ["The text gives no information that answers the question.",
                     "None of the other options applies to this text."]
NONE_PATTERN = re.compile(r"^(none( of the above)?|n/?a|not[ _-]?applicable|out[ _-]of[ _-]scope|"
                          r"no[ _-]relevant[ _-]option|not enough information)$", re.I)


def none_index(options: list[Option]) -> int | None:
    return next((i for i, o in enumerate(options) if NONE_PATTERN.match(o.value.strip())), None)


def with_none(options: list[Option], rng: random.Random) -> tuple[list[Option], int]:
    """Options with a "none" option (an existing one, or a new one at a random position)."""
    i = none_index(options)
    if i is not None:
        return options, i
    taken = {o.value for o in options}
    name = rng.choice([n for n in NONE_NAMES if n not in taken] or ["none_of_these"])
    desc = rng.choice(NONE_DESCRIPTIONS) if rng.random() < 0.5 else None
    i = rng.randint(0, len(options))
    return options[:i] + [Option(name, desc)] + options[i:], i


def augment(row: dict[str, Any], options: list[Option], gold: int, aug: AugConfig,
            rng: random.Random, pool: list[dict[str, Any]] | None = None
            ) -> tuple[str, str, list[Option], list[float], str]:
    """(state, instructions, options, target distribution, kind) for one training example."""
    state, q, k = row["state"], row["instructions"], len(options)
    onehot = [1.0 if i == gold else 0.0 for i in range(k)]
    u = rng.random()
    if u < aug.empty:
        empty_state = rng.choice(EMPTY_STATES)
        if aug.empty_none and rng.random() < aug.empty_none:
            options, _ = with_none(options, rng)
        i = none_index(options) if aug.none_aware else None
        n = len(options)
        target = [1.0 / n] * n if i is None else [float(j == i) for j in range(n)]
        return empty_state, q, options, target, "empty"
    u -= aug.empty
    if u < aug.mismatch and pool:
        other = rng.choice(pool)
        if other.get("source") != row.get("source"):
            opts, i = with_none(options, rng)
            return other["state"], q, opts, [float(j == i) for j in range(len(opts))], \
                "mismatch"
        return state, q, options, onehot, "plain"
    u -= aug.mismatch
    if u < aug.none_option and none_index(options) is None:
        opts, i = with_none(options, rng)
        g = gold + (i <= gold)
        return state, q, opts, [float(j == g) for j in range(len(opts))], "none_option"
    u -= aug.none_option
    if u < aug.negation and k >= 2:
        d = rng.choice([i for i in range(k) if i != gold])
        pair = [options[gold], options[d]]
        rng.shuffle(pair)
        target = [1.0 if o is options[d] else 0.0 for o in pair]
        return state, f"Which option does NOT answer this question: {q}", pair, target, \
            "negation"
    u -= aug.negation
    if u < aug.decoy:
        g = options[gold]
        if rng.random() < 0.7:
            name = g.value + rng.choice(DECOY_SUFFIXES)
        else:
            name = rng.choice(UNRELATED_NAMES)
        if name in {o.value for o in options}:
            return state, q, options, onehot, "plain"
        opts = list(options)
        opts.insert(rng.randint(0, k), Option(name, g.description))
        target = [1.0 if o is g else 0.0 for o in opts]
        return state, q, opts, target, "decoy"
    u -= aug.decoy
    if u < aug.paraphrase:
        return state, rng.choice(PARAPHRASES).format(q=q), options, onehot, "paraphrase"
    return state, q, options, onehot, "plain"


# ------------------------------------------------------------------ readout + loss
def last_logits(model, ids: list[int]) -> torch.Tensor:
    """Vocabulary logits at the last position, with gradients (no full-sequence LM head)."""
    out = model.model(input_ids=torch.tensor([ids], device=model.device))
    return model.lm_head(out.last_hidden_state[:, -1])[0]


def label_logp(readout: ChatLetterReadout, model, ids: list[int], k: int) -> torch.Tensor:
    """log P(option label) renormalised over the K labels (differentiable)."""
    if readout.cfg.labels == "numbers":
        raise ValueError("training supports single-token labels (letters, codes)")
    return variant_logp(model, ids, readout.label_variant_ids(readout.label_strings(k)))


def variant_logp(model, ids: list[int], variants: list[list[int]]) -> torch.Tensor:
    """log P(label) renormalised over the labels, each label summed over its token variants."""
    lp_all = torch.log_softmax(last_logits(model, ids).float(), dim=-1)
    lp = torch.stack([torch.logsumexp(lp_all[v], dim=-1) for v in variants])
    return lp - torch.logsumexp(lp, dim=-1)


def typed_question(row: dict[str, Any]) -> Question:
    """The score / noul question of a typed corpus row (R15)."""
    return Question("q", row["instructions"], row["type"],
                    [Option(o["value"], o.get("description")) for o in row["options"]])


def typed_example(readout: ChatLetterReadout, row: dict[str, Any]
                  ) -> tuple[list[int], list[list[int]], list[float]]:
    """(prompt ids, label variants, one-hot target) of a score / noul row, read with the
    readout's question-type templates."""
    plan = readout.plan(row["state"], typed_question(row))
    if plan is None:
        raise ValueError(f"cannot read a {row['type']} question in one pass")
    ids, variants, values = plan
    return ids, variants, [float(v == row["gold"]) for v in values]


def rps(logp: torch.Tensor, target: list[float] | torch.Tensor) -> torch.Tensor:
    """Ranked probability score over ordered levels (a proper scoring rule for ordinal
    answers: it charges probability placed far from the true level more)."""
    t = torch.as_tensor(target, dtype=logp.dtype, device=logp.device)
    diff = torch.cumsum(logp.exp(), 0)[:-1] - torch.cumsum(t, 0)[:-1]
    return (diff ** 2).sum() / max(1, len(t) - 1)


def letter_logp(readout: ChatLetterReadout, model, ids: list[int], k: int) -> torch.Tensor:
    return label_logp(readout, model, ids, k)


def loss_fn(logp: torch.Tensor, target: int | list[float] | torch.Tensor,
            kind: str) -> torch.Tensor:
    if isinstance(target, int):
        t = torch.zeros_like(logp)
        t[target] = 1.0
    else:
        t = torch.as_tensor(target, dtype=logp.dtype, device=logp.device)
    ce = -(t * logp).sum()
    if kind == "ce":
        return ce
    brier = ((logp.exp() - t) ** 2).sum()
    return brier if kind == "brier" else ce + brier


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)


@torch.inference_mode()
def evaluate(readout: ChatLetterReadout, rows: list[dict[str, Any]],
             max_tokens: int) -> dict[str, Any]:
    """Held-out accuracy and calibration, options shuffled with one fixed seed for every arm."""
    model = readout.lm.model
    was_training = model.training
    model.eval()
    rng = random.Random(EVAL_SEED)
    conf, correct, nll, first_pred, first_gold = [], [], [], [], []
    by_type: dict[str, list[float]] = {}
    for row in rows:
        kind = row.get("type", "choice")
        if kind != "choice":
            ids, variants, target = typed_example(readout, row)
            if len(ids) > max_tokens:
                continue
            p = variant_logp(model, ids, variants).exp().float().cpu().numpy()
            gold = target.index(1.0)
            pred = int(p.argmax())
            conf.append(p.max())
            correct.append(pred == gold)
            nll.append(-math.log(max(float(p[gold]), 1e-12)))
            by_type.setdefault(kind, []).append(float(pred == gold))
            continue
        options, gold = arrange(row, True, rng)
        ids = readout.prompt_ids(row["state"], row["instructions"], options)
        if len(ids) > max_tokens:
            continue
        p = label_logp(readout, model, ids, len(options)).exp().float().cpu().numpy()
        pred = int(p.argmax())
        conf.append(p.max())
        correct.append(pred == gold)
        nll.append(-math.log(max(float(p[gold]), 1e-12)))
        first_pred.append(pred == 0)
        first_gold.append(gold == 0)
        by_type.setdefault("choice", []).append(float(pred == gold))
    if was_training:
        model.train()
    c, ok = np.array(conf), np.array(correct, dtype=float)
    out = {"n": len(ok), "accuracy": round(float(ok.mean()), 4) if len(ok) else None,
           "nll": round(float(np.mean(nll)), 4) if nll else None,
           "ece": round(ece(c, ok), 4) if len(ok) else None,
           "mean_confidence": round(float(c.mean()), 4) if len(c) else None,
           "pred_first_share": round(float(np.mean(first_pred)), 4) if first_pred else None,
           "gold_first_share": round(float(np.mean(first_gold)), 4) if first_gold else None}
    if set(by_type) - {"choice"}:
        out["accuracy_by_type"] = {k: round(float(np.mean(v)), 4) for k, v in by_type.items()}
    return out


# ------------------------------------------------------------------ training
def train(cfg: TrainConfig, lm: LoadedModel | None = None, out: Path | None = None,
          log=print) -> dict[str, Any]:
    out = out or arm_dir(cfg.arm)
    out.mkdir(parents=True, exist_ok=True)
    model_cfg = load_model_config(cfg.base_model)
    if lm is None:
        mp = model_cfg.params
        lm = load(model_cfg.hf_model_id, dtype=mp.get("dtype", "bfloat16"),
                  attn_implementation=mp.get("attn_implementation"), cache=False)
    readout = ChatLetterReadout(lm, ReadoutConfig.from_params(model_cfg.params.get("readout"),
                                                              cfg.readout))
    torch.manual_seed(cfg.seed)
    modules = apply_lora(lm.model, cfg.lora.r, cfg.lora.alpha, cfg.lora.dropout, cfg.lora.pattern)
    params = lora_parameters(lm.model)
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)

    def lr_at(step: int) -> float:
        if step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    paths = build_corpus(cfg.corpus)
    rows = read_jsonl(paths["train"])[: cfg.max_rows]
    holdout = read_jsonl(paths["holdout"])[: cfg.eval_rows]
    rng = random.Random(cfg.seed)

    def stream(skip: int):
        """Rows in a fixed shuffled order per epoch, starting after `skip` consumed rows."""
        epoch, pos = divmod(skip, len(rows))
        while True:
            order = list(range(len(rows)))
            random.Random(f"{cfg.seed}:{epoch}").shuffle(order)
            yield from (rows[i] for i in order[pos:])
            epoch, pos = epoch + 1, 0

    log_path = out / "train_log.jsonl"
    ckpt_path = out / CHECKPOINT
    start, consumed, seen, dropped, kinds = 0, 0, 0, 0, {}
    history: list[dict[str, Any]] = []
    elapsed_before = 0.0
    if cfg.checkpoint_every and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ck["config"] != cfg.model_dump():
            raise RuntimeError(f"{ckpt_path} was written by a different config; delete it")
        load_lora_state(lm.model, ck["lora"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        rng.setstate(ck["rng"])
        start, consumed, seen, dropped = ck["step"], ck["consumed"], ck["seen"], ck["dropped"]
        kinds, history, elapsed_before = ck["kinds"], ck["history"], ck["elapsed_s"]
        log(json.dumps({"resumed_at_step": start}))
    else:
        log_path.unlink(missing_ok=True)
    data = stream(consumed)
    lm.model.train()
    t0 = time.perf_counter()
    window: dict[str, list[float]] = {"loss": [], "acc": [], "p_gold": [], "kl": []}

    def checkpoint(step: int) -> None:
        tmp = ckpt_path.with_suffix(".tmp")
        torch.save({"config": cfg.model_dump(), "lora": lora_state(lm.model),
                    "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "rng": rng.getstate(), "step": step, "consumed": consumed, "seen": seen,
                    "dropped": dropped, "kinds": kinds, "history": history,
                    "elapsed_s": elapsed_before + time.perf_counter() - t0}, tmp)
        tmp.replace(ckpt_path)

    for step in range(start, cfg.max_steps):
        n = 0
        while n < cfg.batch_size:
            row = next(data)
            consumed += 1
            if row.get("type", "choice") != "choice":
                ids, variants, target = typed_example(readout, row)
                kind = row["type"]
            else:
                options, gold = arrange(row, cfg.shuffle, rng)
                state, q, options, target, kind = augment(row, options, gold, cfg.aug, rng, rows)
                ids = readout.prompt_ids(state, q, options)
                variants = readout.label_variant_ids(readout.label_strings(len(options)))
            if len(ids) > cfg.max_tokens:
                dropped += 1
                if dropped > 10 * len(rows):
                    raise RuntimeError(f"almost every row exceeds max_tokens={cfg.max_tokens}")
                continue
            logp = variant_logp(lm.model, ids, variants)
            loss = loss_fn(logp, target, cfg.loss)
            if cfg.rps_weight and kind == "score":
                loss = loss + cfg.rps_weight * rps(logp, target)
            if cfg.kl_weight and kind in cfg.kl_kinds:
                with torch.no_grad(), adapters_disabled(lm.model):
                    base = variant_logp(lm.model, ids, variants)
                kl = (base.exp() * (base - logp)).sum()
                loss = loss + cfg.kl_weight * kl
                window["kl"].append(float(kl.detach()))
            (loss / cfg.batch_size).backward()
            logp = logp.detach()
            best = max(range(len(target)), key=target.__getitem__)
            window["loss"].append(float(loss.detach()))
            if kind != "empty":
                window["acc"].append(float(int(torch.argmax(logp)) == best))
                window["p_gold"].append(float(logp[best].exp()))
            kinds[kind] = kinds.get(kind, 0) + 1
            n += 1
            seen += 1
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        last = step + 1 == cfg.max_steps
        if (step + 1) % 25 == 0 or last:
            entry = {"step": step + 1, "seen": seen, "dropped": dropped,
                     "lr": sched.get_last_lr()[0],
                     **{k: round(float(np.mean(v)), 4) for k, v in window.items() if v},
                     "elapsed_s": round(elapsed_before + time.perf_counter() - t0, 1)}
            if (step + 1) % cfg.eval_every == 0 or last:
                entry["holdout"] = evaluate(readout, holdout, cfg.max_tokens)
            history.append(entry)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            log(json.dumps(entry))
            window = {k: [] for k in window}
        if cfg.checkpoint_every and (step + 1) % cfg.checkpoint_every == 0 and not last:
            checkpoint(step + 1)
    lm.model.eval()
    save_adapter(lm.model, out, {"arm": cfg.arm, "base_model": cfg.base_model,
                                 "hf_model_id": model_cfg.hf_model_id, "loss": cfg.loss,
                                 "shuffle": cfg.shuffle, "steps": cfg.max_steps,
                                 "readout": cfg.readout, "kl_weight": cfg.kl_weight,
                                 "rps_weight": cfg.rps_weight, "aug": cfg.aug.model_dump()})
    ckpt_path.unlink(missing_ok=True)
    summary = {"config": cfg.model_dump(), "lora_modules": len(modules),
               "trainable_params": sum(p.numel() for p in params), "examples_seen": seen,
               "examples_by_kind": kinds, "dropped_too_long": dropped, "train_rows": len(rows),
               "train_s": round(elapsed_before + time.perf_counter() - t0, 1),
               "final": history[-1]}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return summary
