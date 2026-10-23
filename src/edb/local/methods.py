"""C-series: decision readouts on a local open-weight model (gpt-oss-20b).

All methods share the model weights and prompt conventions; only the readout differs.

    local_autoregressive   C1  generate (harmony, reasoning effort), parse final label -> one-hot
    local_sequence_scoring C2  softmax over sum log p(label tokens + <|return|>) per candidate
                           C3  same with `kv_reuse: true` (prefix encoded once, KV branched)
    local_cross_encoder    C4  per candidate "is this the correct decision?" -> log p(Yes)-log p(No)
    local_listwise_head    C5  options listed after the state; hidden states at each option's
                               last token -> shared linear head trained on the train split
"""

from __future__ import annotations

import time

import torch

from edb.local import harmony
from edb.local.model import LoadedModel, load, peak_vram_gb, reset_peak, sync
from edb.local.scoring import continuation_scores
from edb.prompts import system_prompt
from edb.providers.base import CallResult, DecisionMethod, MethodContext, one_hot, softmax
from edb.providers.parsing import parse_label
from edb.schemas import Example, Usage

CROSS_ENCODER_INSTRUCTIONS = (
    "Task: {instructions}\n\n"
    "You will be shown the input followed by ONE candidate decision. Answer Yes if the candidate "
    "is the correct decision for the input, otherwise answer No. Answer with exactly one word: "
    "Yes or No."
)


class LocalMethod(DecisionMethod):
    """Shared model handling for local Hugging Face models."""

    def __init__(self, ctx: MethodContext):
        super().__init__(ctx)
        if not ctx.model or not ctx.model.hf_model_id:
            raise ValueError(f"{self.type} requires a model config with `hf_model_id`")
        self.reasoning = ctx.params.get("reasoning_effort", "low")
        self.lm: LoadedModel | None = None

    def _load(self) -> LoadedModel:
        if self.lm is None:
            mp = self.ctx.model.params
            self.lm = load(self.ctx.model.hf_model_id, dtype=mp.get("dtype", "bfloat16"),
                           attn_implementation=mp.get("attn_implementation"),
                           dequantize=mp.get("dequantize"))
            self.info.update(self.lm.info)
        return self.lm

    async def setup(self) -> None:
        self._load()

    def prompt(self, instructions: str, user: str) -> str:
        return harmony.render(instructions, user, reasoning=self.reasoning)

    async def call(self, example: Example) -> CallResult:
        lm = self._load()
        reset_peak(lm.device)
        sync(lm.device)
        t0 = time.perf_counter()
        result = self.compute(lm, example)
        sync(lm.device)
        result.extras["gpu_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        result.extras["peak_vram_gb"] = peak_vram_gb(lm.device)
        return result

    def compute(self, lm: LoadedModel, example: Example) -> CallResult:
        raise NotImplementedError


class _StepTimer:
    """LogitsProcessor that timestamps every decoding step (for time-to-final-token)."""

    def __init__(self):
        self.times: list[float] = []

    def __call__(self, input_ids, scores):
        self.times.append(time.perf_counter())
        return scores


class LocalAutoregressive(LocalMethod):
    """C1. Params: reasoning_effort (low), max_new_tokens (4096)."""

    type = "local_autoregressive"

    def compute(self, lm: LoadedModel, example: Example) -> CallResult:
        from transformers import LogitsProcessorList

        ids = lm.encode(self.prompt(system_prompt(self.ctx.task, self.ctx.catalog), example.state))
        timer = _StepTimer()
        t0 = time.perf_counter()
        out = lm.model.generate(
            input_ids=torch.tensor([ids], device=lm.device),
            attention_mask=torch.ones(1, len(ids), dtype=torch.long, device=lm.device),
            max_new_tokens=int(self.ctx.params.get("max_new_tokens", 4096)),
            do_sample=False,
            logits_processor=LogitsProcessorList([timer]),
            pad_token_id=lm.tokenizer.pad_token_id,
        )
        gen = out[0, len(ids):].tolist()
        text = lm.tokenizer.decode(gen, skip_special_tokens=False)
        answer = harmony.final_message(text)
        result = CallResult(
            raw_output=answer,
            usage=Usage(input_tokens=len(ids), output_tokens=len(gen),
                        reasoning_tokens=len(lm.encode(harmony.analysis_text(text)))),
            extras={"generated_tail": text[-300:]},
        )
        # Time to the first token of the final-channel message.
        final_start = _final_token_index(lm, gen)
        if final_start is not None and final_start < len(timer.times):
            result.ttft_ms = round((timer.times[final_start] - t0) * 1000, 2)
        if answer is None:
            result.error = "no final-channel answer (truncated generation?)"
            return result
        parsed = parse_label(answer, self.values)
        result.extras["parse_status"] = parsed.status
        if parsed.label is None:
            result.error = "unparseable output"
            return result
        result.predicted = parsed.label
        result.probabilities = one_hot(parsed.label, self.values)
        result.prob_source = "one_hot"
        return result


def _final_token_index(lm: LoadedModel, gen: list[int]) -> int | None:
    """Index in `gen` of the first token after `<|channel|>final<|message|>`."""
    marker = lm.encode(harmony.FINAL_PREFIX)
    for i in range(len(gen) - len(marker) + 1):
        if gen[i : i + len(marker)] == marker:
            return i + len(marker)
    return None


class LocalSequenceScoring(LocalMethod):
    """C2/C3/C2+. Params: kv_reuse (false), length_norm ("sum" | "mean"),
    batched (true; false = one forward per candidate, a numerics diagnostic for C2 vs C3),
    prior ("none" | "content_free": subtract each label's log-likelihood under a content-free
    state, i.e. contextual calibration, Zhao et al. 2021), chunk (candidates per batch, 16)."""

    type = "local_sequence_scoring"
    CONTENT_FREE_STATE = "N/A"

    def __init__(self, ctx: MethodContext):
        super().__init__(ctx)
        self.prior = ctx.params.get("prior", "none")
        if self.prior not in ("none", "content_free"):
            raise ValueError(f"unknown prior {self.prior!r}")
        self.prior_logprob: dict[str, float] | None = None

    async def setup(self) -> None:
        lm = self._load()
        if self.prior == "content_free":
            scores = self._score(lm, self.CONTENT_FREE_STATE)[1]
            self.prior_logprob = {v: s.total for v, s in zip(self.values, scores, strict=True)}
            self.info["prior_logprob"] = self.prior_logprob

    def _score(self, lm: LoadedModel, state: str):
        prefix = lm.encode(self.prompt(system_prompt(self.ctx.task, self.ctx.catalog), state)
                           + harmony.FINAL_PREFIX)
        ret = lm.token_id(harmony.RETURN)
        conts = [lm.encode(v) + [ret] for v in self.values]
        mode = "kv" if self.ctx.params.get("kv_reuse") else "independent"
        chunk = int(self.ctx.params.get("chunk", 16))
        if mode == "independent" and not self.ctx.params.get("batched", True):
            scores = [continuation_scores(lm.model, prefix, [c], mode=mode,
                                          pad_id=lm.tokenizer.pad_token_id)[0] for c in conts]
        else:
            scores = continuation_scores(lm.model, prefix, conts, mode=mode,
                                         pad_id=lm.tokenizer.pad_token_id, chunk=chunk)
        return prefix, scores, conts, mode

    def compute(self, lm: LoadedModel, example: Example) -> CallResult:
        prefix, scores, conts, mode = self._score(lm, example.state)
        norm = self.ctx.params.get("length_norm", "sum")
        logits = {v: (s.mean if norm == "mean" else s.total)
                  for v, s in zip(self.values, scores, strict=True)}
        if self.prior_logprob is not None:
            logits = {v: x - self.prior_logprob[v] for v, x in logits.items()}
        n_tokens = len(prefix) + sum(len(c) for c in conts) if mode == "kv" else \
            sum(len(prefix) + len(c) for c in conts)
        return CallResult(
            probabilities=softmax(logits),
            prob_source="model",
            usage=Usage(input_tokens=n_tokens),
            extras={"mode": mode, "prefix_tokens": len(prefix),
                    "sum_logprob": {v: s.total for v, s in zip(self.values, scores,
                                                               strict=True)},
                    "mean_logprob": {v: s.mean for v, s in zip(self.values, scores,
                                                               strict=True)}},
        )


class LocalCrossEncoder(LocalMethod):
    """C4. Params: kv_reuse (false)."""

    type = "local_cross_encoder"

    def compute(self, lm: LoadedModel, example: Example) -> CallResult:
        instructions = CROSS_ENCODER_INSTRUCTIONS.format(instructions=self.ctx.task.instructions)
        marker = "\x00CANDIDATE\x00"
        full = self.prompt(instructions, f"{example.state}\n\nCandidate decision: {marker}")
        head, tail = full.split(marker)
        prefix = lm.encode(head)
        conts = []
        for o in self.ctx.catalog.labels:
            text = f"{o.value}: {o.description}" if o.has_description else o.value
            conts.append(lm.encode(text + tail + harmony.FINAL_PREFIX))
        yes, no = lm.encode("Yes"), lm.encode("No")
        if len(yes) != 1 or len(no) != 1:
            raise ValueError("Yes/No must be single tokens for this tokenizer")
        mode = "kv" if self.ctx.params.get("kv_reuse") else "independent"
        scores = continuation_scores(lm.model, prefix, conts, mode=mode,
                                     query_ids=[yes[0], no[0]],
                                     pad_id=lm.tokenizer.pad_token_id,
                                     chunk=int(self.ctx.params.get("chunk", 16)))
        margins = {v: s.query_logprobs[0] - s.query_logprobs[1]
                   for v, s in zip(self.values, scores, strict=True)}
        n_tokens = len(prefix) + sum(len(c) for c in conts) if mode == "kv" else \
            sum(len(prefix) + len(c) for c in conts)
        return CallResult(
            probabilities=softmax(margins),
            prob_source="model",
            usage=Usage(input_tokens=n_tokens),
            extras={"mode": mode, "yes_no_margin": margins,
                    "p_yes": {v: float(torch.tensor(s.query_logprobs[0]).exp())
                              for v, s in zip(self.values, scores, strict=True)}},
        )


def local_method_types() -> dict[str, type[DecisionMethod]]:
    from edb.local.chat_readout import LocalChatReadout
    from edb.local.listwise import LocalListwiseHead
    from edb.local.readout import LocalEarlyExitHead, LocalLetterReadout, LocalReasonThenRead

    methods: list[type[DecisionMethod]] = [LocalAutoregressive, LocalSequenceScoring,
                                           LocalCrossEncoder, LocalListwiseHead,
                                           LocalLetterReadout, LocalEarlyExitHead,
                                           LocalReasonThenRead, LocalChatReadout]
    return {m.type: m for m in methods}
