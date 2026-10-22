"""Option-letter readout for any chat model (Part 2: the frozen-base comparison).

The recipe of the open "System One" replicas that read a frozen or LoRA-tuned instruct model
(Cygnet, SemIf/OpenJev, Quyet, H2O-Lightning):

    1. Render the state, the question and the labelled options in one user message through
       the model's own chat template (thinking disabled).
    2. One forward pass; at the answer position, sum the probability of every vocabulary
       token that decodes to each label (`A`, ` A`, ...), then renormalise over the labels.
    3. Labels (`labels`): letters A..Z (default); `codes` adds single-token two-letter codes
       (AA, AB, ...) so up to ~580 options fit one pass; `numbers` (1, 2, ...) reads
       multi-digit labels exactly, digit by digit.
    4. More options than the labels allow (letters: more than `group_size`): score groups of
       options, then the group winners against each other;
       P(option) = P(its group's winner) x P(option | its group).

`ChatLetterReadout` is shared by the dataset method (`local_chat_readout`), the probe client
(`LocalReadoutClient`) and training (`edb.training`), so all three see identical prompts.
"""

from __future__ import annotations

import asyncio
import copy
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

Answer with the {label_word} of the correct option only."""

# Alternative wordings/layouts of the same request (prompt-sensitivity experiment). Options are
# always lettered A, B, C... in the given order, so only the surrounding text differs.
PROMPT_MARKDOWN = """## Task
{instructions}

## Possible answers
{options}

## Input
{state}

Reply with a single {label_word}: the best answer above."""

PROMPTS = {"default": PROMPT, "markdown": PROMPT_MARKDOWN, "json": None}
OPTION_LINES = {"default": "{letter}. {text}", "markdown": "({letter}) {text}"}
# How options are labelled (and read back):
#   letters  A..Z, one token each; more than 26 options are scored in groups (`group_size`)
#   codes    A..Z, then single-token two-letter codes (AA, AB, ...): up to 255+ in one pass
#   numbers  1, 2, 3, ...; multi-digit labels are scored exactly as
#            P(d1) * P(d2 | d1) * ... * P(next token is not a digit | label)
LABEL_SCHEMES = ("letters", "codes", "numbers")

# Per-question-type templates (R14). Decision-1 answers `noul` and `score` questions from
# their own templates (probe finding 10): score levels are keyed by digit index, noul returns
# P(true). `letters` keeps the choice template for every type. The typed templates share the
# choice template's text up to the question, so questions about one state share a prefix (R16).
NOUL_STYLES = ("letters", "truefalse", "yesno")
SCORE_STYLES = ("letters", "digits", "scale")
TYPED_PROMPT = """Read the state and answer the question by choosing exactly one option.

<state>
{state}
</state>

<question>
{instructions}
</question>

{block}

{closing}"""
NOUL_WORDS = {"truefalse": ("true", "false"), "yesno": ("yes", "no")}


@dataclass
class ReadoutConfig:
    chat_template_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"enable_thinking": False})
    answer_prefix: str = ""  # appended after the generation prompt (e.g. harmony channels)
    system_prompt: str | None = None
    group_size: int = 26
    temperature: float = 1.0  # p^(1/T), renormalised; applied once to the final distribution
    noul_order: tuple[str, str] = ("false", "true")  # as shown to the model (Cygnet: false first)
    template: str = "default"  # default | markdown | json (see PROMPTS)
    labels: str = "letters"  # letters | codes | numbers (see LABEL_SCHEMES)
    noul_style: str = "letters"  # letters | truefalse | yesno (see NOUL_STYLES)
    score_style: str = "letters"  # letters | digits | scale (see SCORE_STYLES)
    # Questions about one state share their common prompt prefix: it is encoded once and
    # cached (R16). Only the probe client (several questions per request) uses it.
    share_state: bool = False

    @classmethod
    def from_params(cls, *params: dict[str, Any] | None) -> ReadoutConfig:
        merged: dict[str, Any] = {}
        for p in params:
            merged.update(p or {})
        if "noul_order" in merged:
            merged["noul_order"] = tuple(merged["noul_order"])
        if merged.get("template", "default") not in PROMPTS:
            raise ValueError(f"unknown template {merged['template']!r}; known: {list(PROMPTS)}")
        if merged.get("labels", "letters") not in LABEL_SCHEMES:
            raise ValueError(f"unknown labels {merged['labels']!r}; known: {LABEL_SCHEMES}")
        if merged.get("noul_style", "letters") not in NOUL_STYLES:
            raise ValueError(f"unknown noul_style {merged['noul_style']!r}; known: {NOUL_STYLES}")
        if merged.get("score_style", "letters") not in SCORE_STYLES:
            raise ValueError(f"unknown score_style {merged['score_style']!r}; "
                             f"known: {SCORE_STYLES}")
        return cls(**merged)


def render_state(state: Any) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, separators=(",", ":"))


def option_text(option: Option) -> str:
    return option.value + (f": {option.description}" if option.description else "")


def option_line(letter: str, option: Option, template: str = "default") -> str:
    return OPTION_LINES.get(template, OPTION_LINES["default"]).format(
        letter=letter, text=option_text(option))


def render_body(template: str, state: Any, instructions: str, options: list[Option],
                labels: list[str] | None = None, label_word: str = "letter") -> str:
    labels = labels or list(LETTERS[:len(options)])
    if template == "json":
        return json.dumps({"input": render_state(state), "question": instructions,
                           "options": {labels[i]: option_text(o) for i, o in enumerate(options)},
                           "answer_format": "the key of the best option, nothing else"},
                          ensure_ascii=False, indent=1)
    return PROMPTS[template].format(
        state=render_state(state), instructions=instructions, label_word=label_word,
        options="\n".join(option_line(labels[i], o, template) for i, o in enumerate(options)))


class ChatLetterReadout:
    def __init__(self, lm: LoadedModel, cfg: ReadoutConfig | None = None):
        self.lm = lm
        self.cfg = cfg or ReadoutConfig()
        self._variants: list[list[int]] | None = None
        self._by_text: dict[str, list[int]] | None = None
        self._codes: list[str] | None = None

    # ------------------------------------------------------------------ labels
    def _token_texts(self) -> dict[str, list[int]]:
        """Stripped text -> ids of every short non-special token that decodes to it."""
        if self._by_text is None:
            tok = self.lm.tokenizer
            special = set(tok.all_special_ids)
            pieces = tok.convert_ids_to_tokens(list(range(len(tok))))
            short = [i for i, p in enumerate(pieces) if p is not None and len(p) <= 4
                     and i not in special]
            by_text: dict[str, list[int]] = {}
            for i, t in zip(short, tok.batch_decode([[i] for i in short]), strict=True):
                by_text.setdefault(t.strip(), []).append(i)
            self._by_text = by_text
        return self._by_text

    def codes(self) -> list[str]:
        """A..Z, then the two-letter codes AA..ZZ that are single tokens (alphabetical)."""
        if self._codes is None:
            pairs = [a + b for a in LETTERS for b in LETTERS]
            self._codes = list(LETTERS) + [p for p in pairs
                                           if len(self.lm.encode(p)) == 1]
        return self._codes

    def capacity(self) -> int:
        """Most options one pass can label (more are scored in groups)."""
        if self.cfg.labels == "numbers":
            return 999
        return len(self.codes()) if self.cfg.labels == "codes" else len(LETTERS)

    def label_strings(self, k: int) -> list[str]:
        if self.cfg.labels == "numbers":
            return [str(i + 1) for i in range(k)]
        pool = self.codes() if self.cfg.labels == "codes" else list(LETTERS)
        if k > len(pool):
            raise ValueError(f"at most {len(pool)} options per pass, got {k}")
        return pool[:k]

    def label_word(self, k: int) -> str:
        if self.cfg.labels == "numbers":
            return "number"
        return "letter" if k <= len(LETTERS) else "letter code"

    # ------------------------------------------------------------------ prompt
    def prompt_text(self, state: Any, instructions: str, options: list[Option]) -> str:
        body = render_body(self.cfg.template, state, instructions, options,
                           self.label_strings(len(options)), self.label_word(len(options)))
        return self.wrap(body)

    def wrap(self, body: str) -> str:
        """Chat-template a user message (plus the system prompt and answer prefix)."""
        msgs = ([{"role": "system", "content": self.cfg.system_prompt}]
                if self.cfg.system_prompt else []) + [{"role": "user", "content": body}]
        rendered = self.lm.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, **self.cfg.chat_template_kwargs)
        return rendered + self.cfg.answer_prefix

    def word_variant_ids(self, words: list[str]) -> list[list[int]]:
        """Per word: the single tokens spelling it bare, capitalised or with a leading space."""
        out = []
        for w in words:
            ids = set()
            for form in {w, w.capitalize(), w.upper()}:
                for s in (form, " " + form):
                    enc = self.lm.encode(s)
                    if len(enc) == 1:
                        ids.add(enc[0])
            if not ids:
                raise ValueError(f"tokenizer has no single token for {w!r}")
            out.append(sorted(ids))
        return out

    def prompt_ids(self, state: Any, instructions: str, options: list[Option]) -> list[int]:
        return self.lm.encode(self.prompt_text(state, instructions, options))

    def label_variant_ids(self, labels: list[str]) -> list[list[int]]:
        by_text = self._token_texts()
        missing = [lbl for lbl in labels if lbl not in by_text]
        if missing:
            raise ValueError(f"tokenizer has no single token for labels {missing[:5]}")
        return [by_text[lbl] for lbl in labels]

    def variant_ids(self) -> list[list[int]]:
        """Per letter A..Z: every non-special token that decodes to it (whitespace stripped)."""
        if self._variants is None:
            self._variants = self.label_variant_ids(list(LETTERS))
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
        if self.cfg.labels == "numbers":
            return self._pass_numbers(state, instructions, options)
        ids = self.prompt_ids(state, instructions, options)
        variants = self.label_variant_ids(self.label_strings(len(options)))
        lp, diag = self._read(ids, variants)
        return lp, diag, len(ids)

    @torch.inference_mode()
    def _read(self, ids: list[int], variants: list[list[int]]
              ) -> tuple[list[float], dict[str, Any]]:
        device = self.lm.model.device
        logits, _ = _forward(self.lm.model, torch.tensor([ids], device=device),
                             torch.tensor([len(ids) - 1], device=device))
        return self._label_lp(logits[0, -1], variants)

    def _label_lp(self, logits: torch.Tensor, variants: list[list[int]]
                  ) -> tuple[list[float], dict[str, Any]]:
        """Renormalised log P(label) at one position, plus diagnostics."""
        lp_all = torch.log_softmax(logits.float(), dim=-1)
        label_lp = torch.stack([torch.logsumexp(lp_all[v], dim=-1) for v in variants])
        on_format = float(label_lp.exp().sum())
        lp = (label_lp - torch.logsumexp(label_lp, 0)).tolist()
        best = int(torch.argmax(label_lp))
        diag = {"on_format_mass": round(on_format, 6),
                "top_letter_logit": round(float(logits[variants[best]].max()), 3),
                "top_vocab_token": self.lm.tokenizer.decode([int(torch.argmax(logits))])}
        return lp, diag

    def _pass_numbers(self, state: Any, instructions: str,
                      options: list[Option]) -> tuple[list[float], dict[str, Any], int]:
        """Exact log P(answer = "n") for n = 1..K over multi-token numbers: the digits of n,
        then any non-digit token (the answer ends)."""
        from edb.local.scoring import continuation_scores

        ids = self.prompt_ids(state, instructions, options)
        digit_ids = [self.lm.encode(str(d)) for d in range(10)]
        if any(len(d) != 1 for d in digit_ids):
            raise ValueError("number labels need single-token digits")
        digit_ids = [d[0] for d in digit_ids]
        labels = self.label_strings(len(options))
        conts = [[digit_ids[int(c)] for c in lbl] for lbl in labels]
        scores = continuation_scores(self.lm.model, ids, conts, mode="independent",
                                     query_ids=digit_ids,
                                     pad_id=self.lm.tokenizer.pad_token_id or 0)
        raw = []
        for s in scores:
            p_more = min(1.0, sum(math.exp(q) for q in s.query_logprobs))
            raw.append(s.total + math.log(max(1e-12, 1.0 - p_more)))
        t = torch.tensor(raw)
        on_format = float(t.exp().sum())
        lp = (t - torch.logsumexp(t, 0)).tolist()
        diag = {"on_format_mass": round(on_format, 6), "top_letter_logit": None,
                "top_vocab_token": labels[int(torch.argmax(t))]}
        return lp, diag, len(ids)

    def score_options(self, state: Any, instructions: str,
                      options: list[Option]) -> tuple[list[float], dict[str, Any]]:
        """Probabilities over `options` (in order) + diagnostics."""
        g = min(self.cfg.group_size, self.capacity()) if self.cfg.labels == "letters" \
            else self.capacity()
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
        return self._temper(lp), diag

    def _temper(self, lp: list[float]) -> list[float]:
        t = self.cfg.temperature
        z = [x / t for x in lp]
        m = max(z)
        e = [math.exp(x - m) for x in z]
        s = sum(e)
        return [x / s for x in e]

    # ------------------------------------------------------------------ question types
    def plan(self, state: Any, q: Question) -> tuple[list[int], list[list[int]], list[str]] | None:
        """(prompt ids, label token variants, question values in label order) for a question
        answered in one pass; None when it needs grouped or digit-by-digit passes."""
        if q.type == "noul" and self.cfg.noul_style != "letters":
            words = NOUL_WORDS[self.cfg.noul_style]
            crit = {o.value: o.description for o in q.options if o.description}
            lines = [f"{w}: {crit[v]}" if crit.get(v) else w
                     for w, v in zip(words, ("true", "false"), strict=True)]
            body = TYPED_PROMPT.format(
                state=render_state(state), instructions=q.instructions,
                block="<options>\n" + "\n".join(lines) + "\n</options>",
                closing=f"Answer with {words[0]} or {words[1]} only.")
            return (self.lm.encode(self.wrap(body)), self.word_variant_ids(list(words)),
                    ["true", "false"])
        if q.type == "score" and self.cfg.score_style != "letters" and len(q.options) <= 10:
            texts = [o.description or o.value for o in q.options]
            digits = [str(i) for i in range(len(texts))]
            if self.cfg.score_style == "digits":
                block = "<options>\n" + "\n".join(f"{d}. {t}" for d, t in
                                                  zip(digits, texts, strict=True)) + "\n</options>"
                closing = "Answer with the number of the correct option only."
            else:
                block = "<scale>\n" + "\n".join(f"{d} = {t}" for d, t in
                                                zip(digits, texts, strict=True)) + "\n</scale>"
                closing = f"Answer with a single digit from 0 to {len(texts) - 1}."
            body = TYPED_PROMPT.format(state=render_state(state), instructions=q.instructions,
                                       block=block, closing=closing)
            return self.lm.encode(self.wrap(body)), self.label_variant_ids(digits), q.values
        if q.type == "noul":
            crit = {o.value: o.description for o in q.options}
            options = [Option(v, crit.get(v)) for v in self.cfg.noul_order]
        else:
            options = list(q.options)
        if self.cfg.labels == "numbers" or len(options) > (
                min(self.cfg.group_size, self.capacity()) if self.cfg.labels == "letters"
                else self.capacity()):
            return None
        return (self.prompt_ids(state, q.instructions, options),
                self.label_variant_ids(self.label_strings(len(options))),
                [o.value for o in options])

    def _answer_from(self, q: Question, values: list[str], probs: list[float]) -> Answer:
        by_value = dict(zip(values, probs, strict=True))
        if q.type == "noul":
            p = by_value["true"]
            return Answer(q.name, "noul", {"true": p, "false": 1 - p},
                          choice="true" if p >= 0.5 else "false")
        ordered = {v: by_value[v] for v in q.values}
        choice = max(ordered, key=ordered.get)
        score = sum(i * p for i, p in enumerate(ordered.values())) if q.type == "score" else None
        return Answer(q.name, q.type, ordered, choice=choice,
                      confidence=jev_confidence(list(ordered.values())), score=score)

    def answer(self, state: Any, q: Question) -> tuple[Answer, dict[str, Any]]:
        plan = self.plan(state, q)
        if plan is not None:
            ids, variants, values = plan
            lp, diag = self._read(ids, variants)
            diag.update(passes=1, input_tokens=len(ids))
            return self._answer_from(q, values, self._temper(lp)), diag
        if q.type == "noul":
            crit = {o.value: o.description for o in q.options}
            options = [Option(v, crit.get(v)) for v in self.cfg.noul_order]
        else:
            options = list(q.options)
        probs, diag = self.score_options(state, q.instructions, options)
        return self._answer_from(q, [o.value for o in options], probs), diag

    @torch.inference_mode()
    def answer_shared(self, state: Any, questions: list[Question], batch: int = 32
                      ) -> tuple[dict[str, Answer], dict[str, dict[str, Any]], int]:
        """Answer several questions about one state, encoding their common prompt prefix once
        and reading up to `batch` questions per forward pass from the cached prefix (R16).
        Questions that need several passes are answered separately. Returns answers,
        diagnostics and the number of prompt tokens processed."""
        plans = {q.name: self.plan(state, q) for q in questions}
        single = [q for q in questions if plans[q.name] is not None]
        answers, diags, tokens = {}, {}, 0
        for q in questions:
            if plans[q.name] is None:
                answers[q.name], diags[q.name] = self.answer(state, q)
                tokens += diags[q.name]["input_tokens"]
        if not single:
            return answers, diags, tokens
        seqs = [plans[q.name][0] for q in single]
        n = min(len(s) for s in seqs) - 1  # every question keeps at least one own token
        for s in seqs[1:]:
            n = next((i for i in range(n) if s[i] != seqs[0][i]), n)
        model, device = self.lm.model, self.lm.model.device
        cache = None
        if n > 0:
            _, cache = _forward(model, torch.tensor([seqs[0][:n]], device=device),
                                torch.tensor([n - 1], device=device), use_cache=True)
        tokens += n
        pad = self.lm.tokenizer.pad_token_id
        pad = pad if pad is not None else self.lm.tokenizer.eos_token_id or 0
        for start in range(0, len(single), batch):
            chunk = single[start:start + batch]
            suffixes = [plans[q.name][0][n:] for q in chunk]
            logits = self._batched_last_logits(cache, n, suffixes, pad)
            for row, (q, suffix) in enumerate(zip(chunk, suffixes, strict=True)):
                _, variants, values = plans[q.name]
                lp, diag = self._label_lp(logits[row], variants)
                diag.update(passes=1, input_tokens=len(suffix), shared_prefix_tokens=n)
                answers[q.name], diags[q.name] = (self._answer_from(q, values, self._temper(lp)),
                                                  diag)
                tokens += len(suffix)
        return answers, diags, tokens

    def _batched_last_logits(self, cache, n: int, suffixes: list[list[int]],
                             pad: int) -> torch.Tensor:
        """fp32 logits at the last token of each suffix, all suffixes continuing one cached
        prefix of `n` tokens in a single right-padded forward pass."""
        from edb.local.scoring import _fp32_head

        model, device = self.lm.model, self.lm.model.device
        b, width = len(suffixes), max(len(s) for s in suffixes)
        ids = torch.full((b, width), pad, dtype=torch.long, device=device)
        mask = torch.zeros((b, n + width), dtype=torch.long, device=device)
        mask[:, :n] = 1
        for i, s in enumerate(suffixes):
            ids[i, :len(s)] = torch.tensor(s, device=device)
            mask[i, n:n + len(s)] = 1
        kwargs: dict[str, Any] = {"attention_mask": mask}
        if cache is not None:
            past = copy.deepcopy(cache)
            repeat_cache_batch(past, b)
            kwargs.update(past_key_values=past, use_cache=True)
        body = model.model if hasattr(model, "model") else model
        hidden = body(input_ids=ids, **kwargs).last_hidden_state
        last = torch.tensor([len(s) - 1 for s in suffixes], device=device)
        h = hidden[torch.arange(b, device=device), last]
        w32 = _fp32_head(model)
        return h.float() @ w32.T if w32 is not None else model.lm_head(h).float()


# ---------------------------------------------------------------------- model loading
def repeat_cache_batch(cache: Any, b: int) -> None:
    """Expand a batch-1 cache to batch `b` in place. Covers every layer type, including
    linear-attention layers (conv / recurrent states, kept in lists) that lack
    `batch_repeat_interleave`."""

    def rep(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.repeat_interleave(b, dim=0) if value.dim() > 0 and \
                value.shape[0] == 1 else value
        if isinstance(value, list):
            return [rep(v) for v in value]
        if isinstance(value, tuple):
            return tuple(rep(v) for v in value)
        if isinstance(value, dict):
            return {k: rep(v) for k, v in value.items()}
        return value

    for layer in getattr(cache, "layers", []):
        for name, value in list(vars(layer).items()):
            setattr(layer, name, rep(value))


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
    prompt tokens processed (with `share_state`, a shared prefix counts once); output tokens
    are 0."""

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
        if readout.cfg.share_state and len(request.questions) > 1:
            answers, diags, tokens = readout.answer_shared(request.state, request.questions)
        else:
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
