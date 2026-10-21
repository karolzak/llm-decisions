"""Train a LoRA adapter on the option-letter readout (Part 3 arms).

The model reads exactly the prompt `ChatLetterReadout` builds at inference time. The loss is
computed on the K option letters only (log P(letter) summed over its token variants, then
renormalised over the K letters):

    ce     -log p(gold)                       (what fine-tuning usually optimises)
    brier  sum_k (p_k - 1[k = gold])^2        (a proper scoring rule: rewards honest
                                              probabilities, a differentiable stand-in for
                                              "RL for calibrated decisions")

`shuffle` re-orders the options of every example every time it is seen (the label follows);
without it options keep the dataset's canonical label order.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from pydantic import BaseModel

from edb.config import _load_yaml, _resolve, load_model_config
from edb.local.chat_readout import ChatLetterReadout, ReadoutConfig
from edb.local.model import LoadedModel, load
from edb.probing.api import Option
from edb.settings import get_settings
from edb.training.corpus import build_corpus, read_jsonl
from edb.training.lora import apply_lora, lora_parameters, save_adapter

EVAL_SEED = 1234


class LoraConfig(BaseModel):
    r: int = 16
    alpha: float = 32.0
    dropout: float = 0.05
    pattern: str | None = None  # regex over module names in the decoder blocks (None: all)


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


def load_arm(arm: str, file: str = "arms") -> TrainConfig:
    data = _load_yaml(_resolve("training", file))
    arms = data.pop("arms")
    if arm not in arms:
        raise KeyError(f"unknown arm {arm!r}; known: {sorted(arms)}")
    return TrainConfig(arm=arm, **{**data, **arms[arm]})


def arm_dir(arm: str) -> Path:
    return get_settings().artifacts_dir / "training" / arm


def arrange(row: dict[str, Any], shuffle: bool, rng: random.Random) -> tuple[list[Option], int]:
    options = [Option(o["value"], o.get("description")) for o in row["options"]]
    if shuffle:
        rng.shuffle(options)
    return options, [o.value for o in options].index(row["gold"])


def last_logits(model, ids: list[int]) -> torch.Tensor:
    """Vocabulary logits at the last position, with gradients (no full-sequence LM head)."""
    out = model.model(input_ids=torch.tensor([ids], device=model.device))
    return model.lm_head(out.last_hidden_state[:, -1])[0]


def letter_logp(readout: ChatLetterReadout, model, ids: list[int], k: int) -> torch.Tensor:
    lp = readout.letter_logprobs_from_logits(last_logits(model, ids), k)
    return lp - torch.logsumexp(lp, dim=-1)


def loss_fn(logp: torch.Tensor, gold: int, kind: str) -> torch.Tensor:
    ce = -logp[gold]
    if kind == "ce":
        return ce
    target = torch.zeros_like(logp)
    target[gold] = 1.0
    brier = ((logp.exp() - target) ** 2).sum()
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
    for row in rows:
        options, gold = arrange(row, True, rng)
        ids = readout.prompt_ids(row["state"], row["instructions"], options)
        if len(ids) > max_tokens:
            continue
        p = letter_logp(readout, model, ids, len(options)).exp().float().cpu().numpy()
        pred = int(p.argmax())
        conf.append(p.max())
        correct.append(pred == gold)
        nll.append(-math.log(max(float(p[gold]), 1e-12)))
        first_pred.append(pred == 0)
        first_gold.append(gold == 0)
    if was_training:
        model.train()
    c, ok = np.array(conf), np.array(correct, dtype=float)
    return {"n": len(ok), "accuracy": round(float(ok.mean()), 4) if len(ok) else None,
            "nll": round(float(np.mean(nll)), 4) if nll else None,
            "ece": round(ece(c, ok), 4) if len(ok) else None,
            "mean_confidence": round(float(c.mean()), 4) if len(c) else None,
            "pred_first_share": round(float(np.mean(first_pred)), 4) if first_pred else None,
            "gold_first_share": round(float(np.mean(first_gold)), 4) if first_gold else None}


def train(cfg: TrainConfig, lm: LoadedModel | None = None, out: Path | None = None,
          log=print) -> dict[str, Any]:
    out = out or arm_dir(cfg.arm)
    out.mkdir(parents=True, exist_ok=True)
    model_cfg = load_model_config(cfg.base_model)
    if lm is None:
        mp = model_cfg.params
        lm = load(model_cfg.hf_model_id, dtype=mp.get("dtype", "bfloat16"),
                  attn_implementation=mp.get("attn_implementation"), cache=False)
    readout = ChatLetterReadout(lm, ReadoutConfig.from_params(model_cfg.params.get("readout")))
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

    def stream():
        epoch = 0
        while True:
            order = list(range(len(rows)))
            random.Random(f"{cfg.seed}:{epoch}").shuffle(order)
            yield from (rows[i] for i in order)
            epoch += 1

    data = stream()
    log_path = out / "train_log.jsonl"
    log_path.unlink(missing_ok=True)
    lm.model.train()
    t0 = time.perf_counter()
    dropped, seen, window = 0, 0, {"loss": [], "acc": [], "p_gold": []}
    history = []
    for step in range(cfg.max_steps):
        n = 0
        while n < cfg.batch_size:
            row = next(data)
            options, gold = arrange(row, cfg.shuffle, rng)
            ids = readout.prompt_ids(row["state"], row["instructions"], options)
            if len(ids) > cfg.max_tokens:
                dropped += 1
                if dropped > 10 * len(rows):
                    raise RuntimeError(f"almost every row exceeds max_tokens={cfg.max_tokens}")
                continue
            logp = letter_logp(readout, lm.model, ids, len(options))
            loss = loss_fn(logp, gold, cfg.loss)
            (loss / cfg.batch_size).backward()
            logp = logp.detach()
            window["loss"].append(float(loss.detach()))
            window["acc"].append(float(int(torch.argmax(logp)) == gold))
            window["p_gold"].append(float(logp[gold].exp()))
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
                     **{k: round(float(np.mean(v)), 4) for k, v in window.items()},
                     "elapsed_s": round(time.perf_counter() - t0, 1)}
            if (step + 1) % cfg.eval_every == 0 or last:
                entry["holdout"] = evaluate(readout, holdout, cfg.max_tokens)
            history.append(entry)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            log(json.dumps(entry))
            window = {k: [] for k in window}
    lm.model.eval()
    save_adapter(lm.model, out, {"arm": cfg.arm, "base_model": cfg.base_model,
                                 "hf_model_id": model_cfg.hf_model_id, "loss": cfg.loss,
                                 "shuffle": cfg.shuffle, "steps": cfg.max_steps})
    summary = {"config": cfg.model_dump(), "lora_modules": len(modules),
               "trainable_params": sum(p.numel() for p in params), "examples_seen": seen,
               "dropped_too_long": dropped, "train_rows": len(rows),
               "train_s": round(time.perf_counter() - t0, 1), "final": history[-1]}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return summary
