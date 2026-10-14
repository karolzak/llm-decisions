"""Model-level scoring primitives on token ids (no prompt/text logic).

`continuation_scores` returns per-token log-probabilities of each continuation given a shared
prefix, plus (optionally) the next-token log-probabilities of `query_ids` after the full
continuation. Two equivalent computation modes:
    independent  one batched forward over the K full sequences (prefix re-encoded per candidate)
    kv           encode the prefix once, then branch each continuation from a copy of its KV cache
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch

from edb.local import SCORING_VERSION  # noqa: F401  (re-exported; see edb/local/__init__.py)


@dataclass
class ContinuationScore:
    token_logprobs: list[float]
    query_logprobs: list[float] = field(default_factory=list)  # after the last cont. token

    @property
    def total(self) -> float:
        return float(sum(self.token_logprobs))

    @property
    def mean(self) -> float:
        return self.total / max(1, len(self.token_logprobs))


def _log_softmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(logits.float(), dim=-1)


def _fp32_head(model) -> torch.Tensor | None:
    head = getattr(model, "lm_head", None)
    if head is None or getattr(head, "bias", None) is not None:
        return None
    cached = getattr(model, "_edb_lm_head_fp32", None)
    if cached is None:
        cached = head.weight.detach().float()
        model._edb_lm_head_fp32 = cached
    return cached


def _forward(model, input_ids, keep: torch.Tensor | None = None, **kwargs):
    """Forward pass returning (fp32 logits at `keep` positions or all, past_key_values).

    The body runs in the model dtype; only the output projection is upcast to float32, so
    log-probabilities are not quantized by bf16 logits.
    """
    w32 = _fp32_head(model)
    if w32 is None or not hasattr(model, "model"):
        out = model(input_ids=input_ids, logits_to_keep=keep if keep is not None else 0, **kwargs)
        return out.logits.float(), getattr(out, "past_key_values", None)
    out = model.model(input_ids=input_ids, **kwargs)
    hidden = out.last_hidden_state
    if keep is not None:
        hidden = hidden[:, keep]
    return hidden.float() @ w32.T, getattr(out, "past_key_values", None)


@torch.inference_mode()
def continuation_scores(
    model,
    prefix: list[int],
    conts: list[list[int]],
    mode: str = "independent",
    query_ids: list[int] | None = None,
    pad_id: int = 0,
    chunk: int = 16,
) -> list[ContinuationScore]:
    if not prefix or any(not c for c in conts):
        raise ValueError("prefix and continuations must be non-empty")
    if mode == "independent":
        return _independent(model, prefix, conts, query_ids or [], pad_id, chunk)
    if mode == "kv":
        return _kv(model, prefix, conts, query_ids or [])
    raise ValueError(f"unknown mode {mode!r}")


def _independent(model, prefix, conts, query_ids, pad_id,
                 chunk: int = 16) -> list[ContinuationScore]:
    """Batched scoring in chunks of `chunk` candidates (bounds memory for many options)."""
    out: list[ContinuationScore] = []
    for start in range(0, len(conts), chunk):
        out.extend(_independent_batch(model, prefix, conts[start:start + chunk], query_ids,
                                      pad_id))
    return out


def _independent_batch(model, prefix, conts, query_ids, pad_id) -> list[ContinuationScore]:
    device = model.device
    P = len(prefix)
    max_len = max(len(c) for c in conts)
    seqs = [prefix + c + [pad_id] * (max_len - len(c)) for c in conts]
    mask = [[1] * (P + len(c)) + [0] * (max_len - len(c)) for c in conts]
    input_ids = torch.tensor(seqs, device=device)
    attention_mask = torch.tensor(mask, device=device)
    # Logits only for positions P-1 .. P+max_len-1 (predictions of cont tokens + one after).
    keep = torch.arange(P - 1, P + max_len, device=device)
    logits, _ = _forward(model, input_ids, keep, attention_mask=attention_mask)
    out = []
    for k, c in enumerate(conts):
        lp = _log_softmax(logits[k, : len(c) + 1])
        tokens = [float(lp[j, t]) for j, t in enumerate(c)]
        queries = [float(lp[len(c), q]) for q in query_ids]
        out.append(ContinuationScore(tokens, queries))
    return out


def _kv(model, prefix, conts, query_ids) -> list[ContinuationScore]:
    device = model.device
    prefix_ids = torch.tensor([prefix], device=device)
    first_logits, cache = _forward(model, prefix_ids, torch.tensor([len(prefix) - 1],
                                                                   device=device), use_cache=True)
    first_lp = _log_softmax(first_logits[0, -1])
    out = []
    for c in conts:
        branch = copy.deepcopy(cache)
        logits, _ = _forward(model, torch.tensor([c], device=device), None,
                             past_key_values=branch, use_cache=True)
        lp = _log_softmax(logits[0])  # lp[j] predicts token j+1 of c; lp[-1] = after c
        tokens = [float(first_lp[c[0]])] + [float(lp[j, c[j + 1]]) for j in range(len(c) - 1)]
        queries = [float(lp[-1, q]) for q in query_ids]
        out.append(ContinuationScore(tokens, queries))
    return out


@torch.inference_mode()
def hidden_states_at(model, ids: list[int], positions: list[int],
                     layers: list[int]) -> torch.Tensor:
    """Hidden states [len(positions), len(layers), hidden] (layer 0 = embeddings)."""
    out = model(input_ids=torch.tensor([ids], device=model.device), output_hidden_states=True,
                logits_to_keep=1)
    hs = out.hidden_states
    pos = torch.tensor(positions, device=model.device)
    return torch.stack([hs[layer][0, pos] for layer in layers], dim=1).float().cpu()


@torch.inference_mode()
def next_token_logprobs(model, ids: list[int], query_ids: list[int]) -> list[float]:
    """log p(token | ids) for each of `query_ids` (fp32 output head)."""
    device = model.device
    logits, _ = _forward(model, torch.tensor([ids], device=device),
                         torch.tensor([len(ids) - 1], device=device))
    lp = _log_softmax(logits[0, -1])
    return [float(lp[q]) for q in query_ids]


@contextmanager
def truncated(model, n_layers: int):
    """Temporarily keep only the first `n_layers` decoder layers (early exit)."""
    layers = model.model.layers
    if not 0 < n_layers <= len(layers):
        raise ValueError(f"n_layers must be in 1..{len(layers)}")
    model.model.layers = layers[:n_layers]
    try:
        yield
    finally:
        model.model.layers = layers


@torch.inference_mode()
def layer_outputs(model, ids: list[int], layers: list[int], position: int = -1) -> torch.Tensor:
    """Output of decoder layer L-1 (= hidden state after L layers, pre final norm) at `position`
    for each L in `layers`, computing ONLY the first max(layers) layers. Returns [nL, hidden]."""
    captured: dict[int, torch.Tensor] = {}
    hooks = []
    for L in sorted(set(layers)):
        def hook(_module, _inp, output, L=L):
            h = output[0] if isinstance(output, tuple) else output
            captured[L] = h[0, position].float().cpu()
        hooks.append(model.model.layers[L - 1].register_forward_hook(hook))
    try:
        with truncated(model, max(layers)):
            model.model(input_ids=torch.tensor([ids], device=model.device), use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    return torch.stack([captured[L] for L in layers])
