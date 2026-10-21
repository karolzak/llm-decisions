"""Option-letter readout for any chat model (Part 2: the frozen-base comparison).

The recipe of the open "System One" replicas that read a frozen or LoRA-tuned instruct model
(Cygnet, SemIf/OpenJev, Quyet, H2O-Lightning):

    1. Render the state, the question and the options as letters (A., B., ...) in one user
       message through the model's own chat template (thinking disabled).
    2. One forward pass; at the answer position, sum the probability of every vocabulary
       token that decodes to each letter (`A`, ` A`, ...), then renormalise over the letters.
    3. More than `group_size` options: score groups of options, then the group winners
       against each other; P(option) = P(its group's winner) x P(option | its group).

`ChatLetterReadout` is shared by the dataset method (`local_chat_readout`), the probe client
(`LocalReadoutClient`) and training (`edb.training`), so all three see identical prompts.
"""

from __future__ import annotations

import asyncio
import json
import math
import string
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from edb.config import load_model_config
from edb.local.methods import LocalMethod
from edb.local.model import LoadedModel, load
from edb.local.scoring import _forward
from edb.probing.api import (
    Answer,
    DecisionClient,
    DecisionRequest,
    DecisionResponse,
    Option,
    Question,
    jev_confidence,
)
from edb.providers.base import CallResult
from edb.schemas import Example, Usage
from edb.settings import get_settings

LETTERS = string.ascii_uppercase

PROMPT = """Read the state and answer the question by choosing exactly one option.

<state>
{state}
</state>

<question>
{instructions}
</question>

<options>
{options}
</options>

Answer with the letter of the correct option only."""


@dataclass
class ReadoutConfig:
    chat_template_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"enable_thinking": False})
    answer_prefix: str = ""  # appended after the generation prompt (e.g. harmony channels)
    system_prompt: str | None = None
    group_size: int = 26
    temperature: float = 1.0  # p^(1/T), renormalised; applied once to the final distribution
    noul_order: tuple[str, str] = ("false", "true")  # as shown to the model (Cygnet: false first)

    @classmethod
    def from_params(cls, *params: dict[str, Any] | None) -> ReadoutConfig:
        merged: dict[str, Any] = {}
        for p in params:
            merged.update(p or {})
        if "noul_order" in merged:
            merged["noul_order"] = tuple(merged["noul_order"])
        return cls(**merged)


def render_state(state: Any) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, separators=(",", ":"))


def option_line(letter: str, option: Option) -> str:
    return f"{letter}. {option.value}" + (f": {option.description}" if option.description else "")


class ChatLetterReadout:
    def __init__(self, lm: LoadedModel, cfg: ReadoutConfig | None = None):
        self.lm = lm
        self.cfg = cfg or ReadoutConfig()
        self._variants: list[list[int]] | None = None

    # ------------------------------------------------------------------ prompt
    def prompt_text(self, state: Any, instructions: str, options: list[Option]) -> str:
        if len(options) > len(LETTERS):
            raise ValueError(f"at most {len(LETTERS)} options per pass, got {len(options)}")
        body = PROMPT.format(state=render_state(state), instructions=instructions,
                             options="\n".join(option_line(LETTERS[i], o)
                                               for i, o in enumerate(options)))
        msgs = ([{"role": "system", "content": self.cfg.system_prompt}]
                if self.cfg.system_prompt else []) + [{"role": "user", "content": body}]
        rendered = self.lm.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, **self.cfg.chat_template_kwargs)
        return rendered + self.cfg.answer_prefix

    def prompt_ids(self, state: Any, instructions: str, options: list[Option]) -> list[int]:
        return self.lm.encode(self.prompt_text(state, instructions, options))

    def variant_ids(self) -> list[list[int]]:
        """Per letter A..Z: every non-special token that decodes to it (whitespace stripped)."""
        if self._variants is None:
            tok = self.lm.tokenizer
            special = set(tok.all_special_ids)
            pieces = tok.convert_ids_to_tokens(list(range(len(tok))))
            short = [i for i, p in enumerate(pieces) if p is not None and len(p) <= 4
                     and i not in special]
            texts = tok.batch_decode([[i] for i in short])
            by_letter: dict[str, list[int]] = {c: [] for c in LETTERS}
            for i, t in zip(short, texts, strict=True):
                s = t.strip()
                if s in by_letter:
                    by_letter[s].append(i)
            missing = [c for c, ids in by_letter.items() if not ids]
            if missing:
                raise ValueError(f"tokenizer has no token for letters {missing}")
            self._variants = [by_letter[c] for c in LETTERS]
        return self._variants

    # ------------------------------------------------------------------ scoring
    def letter_logprobs_from_logits(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """[..., V] logits -> [..., k] log P(letter) over the full vocabulary (not renormalised).
        Differentiable; used by training too."""
        lp = torch.log_softmax(logits.float(), dim=-1)
        return torch.stack([torch.logsumexp(lp[..., ids], dim=-1)
                            for ids in self.variant_ids()[:k]], dim=-1)

    @torch.inference_mode()
    def _pass(self, state: Any, instructions: str,
              options: list[Option]) -> tuple[list[float], dict[str, Any], int]:
        ids = self.prompt_ids(state, instructions, options)
        device = self.lm.model.device
        logits, _ = _forward(self.lm.model, torch.tensor([ids], device=device),
                             torch.tensor([len(ids) - 1], device=device))
        logits = logits[0, -1]
        letter_lp = self.letter_logprobs_from_logits(logits, len(options))
        on_format = float(letter_lp.exp().sum())
        lp = (letter_lp - torch.logsumexp(letter_lp, 0)).tolist()
        best = int(torch.argmax(letter_lp))
        diag = {"on_format_mass": round(on_format, 6),
                "top_letter_logit": round(float(logits[self.variant_ids()[best]].max()), 3),
                "top_vocab_token": self.lm.tokenizer.decode([int(torch.argmax(logits))])}
        return lp, diag, len(ids)

    def score_options(self, state: Any, instructions: str,
                      options: list[Option]) -> tuple[list[float], dict[str, Any]]:
        """Probabilities over `options` (in order) + diagnostics."""
        g = self.cfg.group_size
        if len(options) <= g:
            lp, diag, n = self._pass(state, instructions, options)
            diag.update(passes=1, input_tokens=n)
        else:
            n_groups = math.ceil(len(options) / g)
            size = math.ceil(len(options) / n_groups)
            groups = [list(range(i, min(i + size, len(options))))
                      for i in range(0, len(options), size)]
            lp = [0.0] * len(options)
            winners, n, diags = [], 0, []
            for grp in groups:
                glp, d, m = self._pass(state, instructions, [options[i] for i in grp])
                n += m
                diags.append(d)
                for i, v in zip(grp, glp, strict=True):
                    lp[i] = v
                winners.append(grp[max(range(len(grp)), key=glp.__getitem__)])
            wlp, d, m = self._pass(state, instructions, [options[i] for i in winners])
            n += m
            for grp, v in zip(groups, wlp, strict=True):
                for i in grp:
                    lp[i] += v  # log P(option | group) + log P(group winner)
            diag = {"on_format_mass": min(x["on_format_mass"] for x in [*diags, d]),
                    "top_letter_logit": d["top_letter_logit"], "top_vocab_token":
                    d["top_vocab_token"], "passes": len(groups) + 1, "input_tokens": n}
        t = self.cfg.temperature
        z = [x / t for x in lp]
        m = max(z)
        e = [math.exp(x - m) for x in z]
        s = sum(e)
        return [x / s for x in e], diag

    def answer(self, state: Any, q: Question) -> tuple[Answer, dict[str, Any]]:
        if q.type == "noul":
            crit = {o.value: o.description for o in q.options}
            options = [Option(v, crit.get(v)) for v in self.cfg.noul_order]
        else:
            options = list(q.options)
        probs, diag = self.score_options(state, q.instructions, options)
        by_value = dict(zip([o.value for o in options], probs, strict=True))
        if q.type == "noul":
            p = by_value["true"]
            return Answer(q.name, "noul", {"true": p, "false": 1 - p},
                          choice="true" if p >= 0.5 else "false"), diag
        ordered = {v: by_value[v] for v in q.values}
        choice = max(ordered, key=ordered.get)
        score = sum(i * p for i, p in enumerate(ordered.values())) if q.type == "score" else None
        return Answer(q.name, q.type, ordered, choice=choice,
                      confidence=jev_confidence(list(ordered.values())), score=score), diag


# ---------------------------------------------------------------------- model loading
def resolve_adapter(adapter: str | None) -> str | None:
    if not adapter:
        return None
    path = Path(adapter)
    return str(path if path.is_absolute() else get_settings().artifacts_dir / path)


def load_readout(model_name: str, adapter: str | None = None,
                 readout: dict[str, Any] | None = None) -> ChatLetterReadout:
    cfg = load_model_config(model_name)
    mp = cfg.params
    lm = load(cfg.hf_model_id, dtype=mp.get("dtype", "bfloat16"),
              attn_implementation=mp.get("attn_implementation"), dequantize=mp.get("dequantize"),
              adapter=resolve_adapter(adapter))
    return ChatLetterReadout(lm, ReadoutConfig.from_params(mp.get("readout"), readout))


class LocalReadoutClient(DecisionClient):
    """Probe target backed by a local model (`kind: local_readout`). Usage reports the
    prompt tokens of every pass (no state sharing across questions); output tokens are 0."""

    def __init__(self, target):
        self.name = target.name
        self.max_options = target.max_options
        self.params = target.params
        self._readout: ChatLetterReadout | None = None

    def readout(self) -> ChatLetterReadout:
        if self._readout is None:
            self._readout = load_readout(self.params["model"], self.params.get("adapter"),
                                         self.params.get("readout"))
        return self._readout

    def _decide(self, request: DecisionRequest) -> DecisionResponse:
        readout = self.readout()
        t0 = time.perf_counter()
        answers, diags, tokens = {}, {}, 0
        for q in request.questions:
            answers[q.name], diags[q.name] = readout.answer(request.state, q)
            tokens += diags[q.name]["input_tokens"]
        if readout.lm.device.type == "cuda":
            torch.cuda.synchronize(readout.lm.device)
        return DecisionResponse(answers=answers, latency_ms=(time.perf_counter() - t0) * 1000,
                                input_tokens=tokens, output_tokens=0,
                                model=readout.lm.info.get("model_id"), extras=diags)

    async def decide(self, request: DecisionRequest) -> DecisionResponse:
        return await asyncio.to_thread(self._decide, request)


class LocalChatReadout(LocalMethod):
    """R2/R3 dataset method. Params: adapter (path under ARTIFACTS_DIR), readout (overrides
    of the model config's `readout` block)."""

    type = "local_chat_readout"

    def _load(self) -> LoadedModel:
        if self.lm is None:
            self.readout = load_readout(self.ctx.model.name, self.ctx.params.get("adapter"),
                                        self.ctx.params.get("readout"))
            self.lm = self.readout.lm
            self.info.update(self.lm.info)
            self.info["readout"] = vars(self.readout.cfg)
        return self.lm

    def compute(self, lm: LoadedModel, example: Example) -> CallResult:
        options = [Option(o.value, o.description if o.has_description else None)
                   for o in self.ctx.catalog.labels]
        probs, diag = self.readout.score_options(example.state, self.ctx.task.instructions,
                                                 options)
        dist = dict(zip(self.values, probs, strict=True))
        return CallResult(predicted=max(dist, key=dist.get), probabilities=dist,
                          prob_source="model", usage=Usage(input_tokens=diag["input_tokens"]),
                          extras=diag)
