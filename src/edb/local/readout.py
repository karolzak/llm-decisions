"""Tier-1/2 readouts on a local model, all built on one "listed options" prompt:

    <instructions> ... user: <state>\n\nOptions:\n[A] label: description\n[B] ... <|start|>assistant
    <|channel|>final<|message|>   -> next token = option marker

Markers are single tokens: letters A-Z for <= 26 options, numbers 1..K beyond.

    local_letter_readout  C6 / C6+  next-token probability of each marker (one forward).
                          C6+: average over cyclic option orderings (position debiasing) and
                          subtract a content-free prior (contextual calibration).
    local_early_exit_head C7        multinomial head on the hidden state at the answer position
                          after L layers; inference runs ONLY the first L layers.
    local_reason_then_read C8       generate the analysis (reasoning) channel, stop at the final
                          channel, then read marker probabilities instead of sampling.
"""

from __future__ import annotations

import hashlib
import json
import string
import time
from dataclasses import dataclass

import torch
from tqdm import tqdm

from edb.datasets.store import dataset_fingerprint, load_split
from edb.local import harmony
from edb.local.methods import LocalMethod
from edb.local.model import LoadedModel
from edb.local.scoring import continuation_scores, layer_outputs, next_token_logprobs
from edb.providers.base import CallResult, MethodContext, softmax
from edb.schemas import Example, Usage

READOUT_INSTRUCTIONS = (
    "Task: {instructions}\n\n"
    "The input is followed by a list of options. Answer with the {kind} of the single correct "
    "option only, and nothing else."
)
CONTENT_FREE_STATE = "N/A"


def option_markers(k: int) -> list[str]:
    return list(string.ascii_uppercase[:k]) if k <= 26 else [str(i + 1) for i in range(k)]


class ListedOptionsMixin:
    """Prompt construction shared by C6/C7/C8."""

    ctx: MethodContext

    def markers(self) -> list[str]:
        return option_markers(len(self.ctx.catalog.labels))

    def marker_ids(self, lm: LoadedModel) -> list[int]:
        ids = []
        for m in self.markers():
            enc = lm.encode(m)
            if len(enc) != 1:
                raise ValueError(f"option marker {m!r} is not a single token")
            ids.append(enc[0])
        return ids

    def instructions(self) -> str:
        kind = "letter" if len(self.ctx.catalog.labels) <= 26 else "number"
        return READOUT_INSTRUCTIONS.format(instructions=self.ctx.task.instructions, kind=kind)

    def options_text(self, order: list[int]) -> str:
        labels, markers = self.ctx.catalog.labels, self.markers()
        lines = []
        for pos, idx in enumerate(order):
            o = labels[idx]
            line = f"[{markers[pos]}] {o.value}"
            if o.has_description:
                line += f": {o.description}"
            lines.append("\n" + line)
        return "".join(lines)

    def split_prompt(self, lm: LoadedModel, state: str, order: list[int],
                     final: bool = True) -> tuple[list[int], list[int]]:
        """(shared head ids up to the options list, branch ids with options + answer prefix)."""
        mark = "\x00OPTIONS\x00"
        full = self.prompt(self.instructions(), f"{state}\n\nOptions:{mark}")
        head, tail = full.split(mark)
        branch = self.options_text(order) + tail + (harmony.FINAL_PREFIX if final else "")
        return lm.encode(head), lm.encode(branch)


def rotation_shifts(k: int, rotations: int | str, max_rotations: int) -> list[int]:
    r = k if rotations == "all" else int(rotations)
    r = max(1, min(r, k, max_rotations))
    return sorted({round(i * k / r) % k for i in range(r)})


class LocalLetterReadout(ListedOptionsMixin, LocalMethod):
    """C6/C6+. Params: rotations (1 | "all" | int), max_rotations (8), prior ("none" |
    "content_free")."""

    type = "local_letter_readout"

    def __init__(self, ctx: MethodContext):
        super().__init__(ctx)
        p = ctx.params
        k = len(ctx.catalog.labels)
        self.shifts = rotation_shifts(k, p.get("rotations", 1), int(p.get("max_rotations", 8)))
        self.prior = p.get("prior", "none")
        if self.prior not in ("none", "content_free"):
            raise ValueError(f"unknown prior {self.prior!r}")
        self.prior_lp: list[list[float]] | None = None  # [rotation][position]

    def orders(self) -> list[list[int]]:
        k = len(self.ctx.catalog.labels)
        return [[(s + j) % k for j in range(k)] for s in self.shifts]

    def position_logprobs(self, lm: LoadedModel, state: str) -> tuple[list[list[float]], int]:
        """Per rotation: log p(marker at position j), renormalised over the markers."""
        marker_ids = self.marker_ids(lm)
        orders = self.orders()
        head, _ = self.split_prompt(lm, state, orders[0])
        branches = [self.split_prompt(lm, state, o)[1] for o in orders]
        if len(branches) == 1:
            raw = [next_token_logprobs(lm.model, head + branches[0], marker_ids)]
            n_tokens = len(head) + len(branches[0])
        else:
            scores = continuation_scores(lm.model, head, branches, mode="kv",
                                         query_ids=marker_ids)
            raw = [s.query_logprobs for s in scores]
            n_tokens = len(head) + sum(len(b) for b in branches)
        out = []
        for lp in raw:
            t = torch.tensor(lp)
            out.append((t - torch.logsumexp(t, 0)).tolist())
        return out, n_tokens

    async def setup(self) -> None:
        lm = self._load()
        self.info.update({"shifts": self.shifts, "markers": self.markers()[:5]})
        if self.prior == "content_free":
            self.prior_lp, _ = self.position_logprobs(lm, CONTENT_FREE_STATE)
            self.info["prior_first_rotation"] = [round(x, 3) for x in self.prior_lp[0][:10]]

    def compute(self, lm: LoadedModel, example: Example) -> CallResult:
        per_rot, n_tokens = self.position_logprobs(lm, example.state)
        k = len(self.values)
        totals = [0.0] * k
        first_choice = None
        for r, (order, lp) in enumerate(zip(self.orders(), per_rot, strict=True)):
            if self.prior_lp is not None:
                lp = [x - c for x, c in zip(lp, self.prior_lp[r], strict=True)]
            for pos, idx in enumerate(order):
                totals[idx] += lp[pos] / len(per_rot)
            if r == 0:
                first_choice = self.values[order[max(range(k), key=lambda j: lp[j])]]
        probs = softmax(dict(zip(self.values, totals, strict=True)))
        top = max(probs, key=probs.get)
        return CallResult(
            probabilities=probs,
            prob_source="model",
            usage=Usage(input_tokens=n_tokens),
            extras={"rotations": len(per_rot), "first_rotation_choice": first_choice,
                    "rotation_disagrees": first_choice != top},
        )


@dataclass
class QuestionHead:
    layer: int
    mean: torch.Tensor
    std: torch.Tensor
    W: torch.Tensor
    b: torch.Tensor

    def logits(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.W.device)
        return ((X - self.mean) / self.std) @ self.W + self.b


def fit_question_head(X: torch.Tensor, y: torch.Tensor, k: int, l2: float, layer: int,
                      max_iter: int = 300) -> QuestionHead:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y = X.float().to(device), y.to(device)
    mean, std = X.mean(0), X.std(0).clamp_min(1e-6)
    Z = (X - mean) / std
    W = torch.zeros(X.shape[1], k, device=device, requires_grad=True)
    b = torch.zeros(k, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Z @ W + b, y) + l2 * (W * W).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return QuestionHead(layer, mean, std, W.detach(), b.detach())


class LocalEarlyExitHead(ListedOptionsMixin, LocalMethod):
    """C7. Params: layers (candidate exit layers, default [12, 16, 20]), l2 grid,
    fit_split / select_split (train / dev), max_iter (300)."""

    type = "local_early_exit_head"

    def __init__(self, ctx: MethodContext):
        super().__init__(ctx)
        p = ctx.params
        self.layers = sorted(int(x) for x in p.get("layers", [12, 16, 20]))
        self.l2_grid = [float(x) for x in p.get("l2", [1e-4, 1e-3, 1e-2, 1e-1])]
        self.fit_split = p.get("fit_split", "train")
        self.select_split = p.get("select_split", "dev")
        self.max_iter = int(p.get("max_iter", 300))
        self.head: QuestionHead | None = None

    def ids(self, lm: LoadedModel, state: str) -> list[int]:
        head, branch = self.split_prompt(lm, state, list(range(len(self.values))))
        return head + branch

    def _cache_path(self, split: str):
        key = {"model": self.ctx.model.hf_model_id,
               "dtype": self.ctx.model.params.get("dtype", "bfloat16"),
               "data": dataset_fingerprint(self.ctx.dataset),
               "catalog": self.ctx.catalog.model_dump(mode="json"), "layers": self.layers,
               "instructions": READOUT_INSTRUCTIONS, "system": harmony.SYSTEM_TEMPLATE,
               "date": harmony.FIXED_DATE, "reasoning": self.reasoning}
        digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:20]
        return self.ctx.settings.cache_dir / "features" / f"earlyexit-{digest}-{split}.pt"

    def split_features(self, lm: LoadedModel, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        examples = load_split(self.ctx.dataset.name, split)
        index = {v: i for i, v in enumerate(self.values)}
        y = torch.tensor([index[e.label] for e in examples])
        path = self._cache_path(split)
        if path.exists():
            data = torch.load(path)
            if data["ids"] == [e.id for e in examples]:
                return data["F"].float(), y
        feats = [layer_outputs(lm.model, self.ids(lm, e.state), self.layers)
                 for e in tqdm(examples, desc=f"features/{split}")]
        F = torch.stack(feats).half()  # [N, nL, D]
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"ids": [e.id for e in examples], "F": F}, path)
        return F.float(), y

    async def setup(self) -> None:
        lm = self._load()
        t0 = time.perf_counter()
        F_fit, y_fit = self.split_features(lm, self.fit_split)
        F_sel, y_sel = self.split_features(lm, self.select_split)
        self.info["feature_s"] = round(time.perf_counter() - t0, 1)
        k = len(self.values)
        grid, best = [], None
        for li, layer in enumerate(self.layers):
            for l2 in self.l2_grid:
                head = fit_question_head(F_fit[:, li], y_fit, k, l2, layer, self.max_iter)
                with torch.no_grad():
                    logits = head.logits(F_sel[:, li])
                    sel_nll = float(torch.nn.functional.cross_entropy(
                        logits, y_sel.to(logits.device)))
                    sel_acc = float((logits.argmax(1).cpu() == y_sel).float().mean())
                grid.append({"layer": layer, "l2": l2, "select_nll": round(sel_nll, 4),
                             "select_acc": round(sel_acc, 4)})
                if best is None or sel_nll < best[0]:
                    best = (sel_nll, head)
        self.head = best[1]
        n_layers = lm.model.config.num_hidden_layers
        self.info.update({"fit_n": len(y_fit), "select_n": len(y_sel), "grid": grid,
                          "chosen": min(grid, key=lambda g: g["select_nll"]),
                          "layers_run": f"{self.head.layer}/{n_layers}"})

    def compute(self, lm: LoadedModel, example: Example) -> CallResult:
        if self.head is None:
            raise RuntimeError("setup() was not called")
        ids = self.ids(lm, example.state)
        h = layer_outputs(lm.model, ids, [self.head.layer])[0]
        with torch.no_grad():
            logits = self.head.logits(h[None])[0].float().cpu().tolist()
        return CallResult(
            probabilities=softmax(dict(zip(self.values, logits, strict=True))),
            prob_source="model",
            usage=Usage(input_tokens=len(ids)),
            extras={"exit_layer": self.head.layer},
        )


class _StopAfter:
    """StoppingCriteria: stop once the generated tail equals `marker` token ids."""

    def __init__(self, marker: list[int], start: int):
        self.marker = marker
        self.start = start

    def __call__(self, input_ids, scores, **kwargs):
        gen = input_ids[0, self.start:]
        done = len(gen) >= len(self.marker) and gen[-len(self.marker):].tolist() == self.marker
        return torch.tensor([done], device=input_ids.device)


class LocalReasonThenRead(ListedOptionsMixin, LocalMethod):
    """C8. Params: reasoning_effort (low), max_new_tokens (4096)."""

    type = "local_reason_then_read"

    def compute(self, lm: LoadedModel, example: Example) -> CallResult:
        from transformers import StoppingCriteriaList

        head, branch = self.split_prompt(lm, example.state, list(range(len(self.values))),
                                         final=False)
        ids = head + branch
        final_ids = lm.encode(harmony.FINAL_PREFIX)
        t0 = time.perf_counter()
        out = lm.model.generate(
            input_ids=torch.tensor([ids], device=lm.device),
            attention_mask=torch.ones(1, len(ids), dtype=torch.long, device=lm.device),
            max_new_tokens=int(self.ctx.params.get("max_new_tokens", 4096)),
            do_sample=False,
            stopping_criteria=StoppingCriteriaList([_StopAfter(final_ids, len(ids))]),
            pad_token_id=lm.tokenizer.pad_token_id,
        )
        gen = out[0, len(ids):].tolist()
        reasoning_ms = (time.perf_counter() - t0) * 1000
        forced = gen[-len(final_ids):] != final_ids
        if forced:
            # Reasoning was cut (token budget) or the model ended without a final channel:
            # close the analysis message and force the final channel.
            text = lm.tokenizer.decode(gen, skip_special_tokens=False)
            if not text.rstrip().endswith("<|end|>"):
                gen += lm.encode("<|end|>")
            gen += lm.encode("<|start|>assistant" + harmony.FINAL_PREFIX)
        lp = next_token_logprobs(lm.model, ids + gen, self.marker_ids(lm))
        t = torch.tensor(lp)
        lp = (t - torch.logsumexp(t, 0)).tolist()
        reasoning = harmony.analysis_text(lm.tokenizer.decode(gen, skip_special_tokens=False))
        return CallResult(
            probabilities=softmax(dict(zip(self.values, lp, strict=True))),
            prob_source="model",
            usage=Usage(input_tokens=len(ids), output_tokens=len(gen),
                        reasoning_tokens=len(lm.encode(reasoning))),
            ttft_ms=round(reasoning_ms, 2),
            extras={"forced_final": forced, "reasoning_ms": round(reasoning_ms, 2)},
        )
