"""C5: listwise decision head on frozen hidden states.

Prompt: instructions, then the state, then all candidate options ([A] value: description, ...),
then the assistant final-channel prefix. Causal attention lets each option's last token see the
state and the preceding options. Hidden states at each option's last token (and at the final
"query" position) are read from selected layers; a single linear head shared across options maps
each option representation to a logit, and a softmax over options gives the distribution.

The base model is frozen; only the head is trained (on `fit_split`), with the layer, feature
type and L2 strength selected by NLL on `select_split`.

Params:
    layers: list[int]          layers to extract (default [8, 12, 16, 20, 24]; 0 = embeddings)
    features: list[str]        candidates among "option" (h_k) and "option_query" ([h_k, h_k*h_q])
    l2: list[float]            L2 grid (default [1e-4, 1e-3, 1e-2, 1e-1, 1])
    fit_split / select_split   default train / dev
    max_iter: int              LBFGS iterations (default 300)
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
from edb.local.scoring import hidden_states_at
from edb.providers.base import CallResult, MethodContext, softmax
from edb.schemas import Example, Usage

LISTWISE_INSTRUCTIONS = (
    "Task: {instructions}\n\n"
    "The input is followed by a list of candidate decisions. Select the single correct decision."
)


def option_marker(i: int) -> str:
    letters = string.ascii_uppercase
    return letters[i] if i < 26 else f"{letters[i // 26 - 1]}{letters[i % 26]}"


@dataclass
class Head:
    layer_idx: int
    kind: str
    mean: torch.Tensor
    std: torch.Tensor
    w: torch.Tensor

    def logits(self, F: torch.Tensor) -> torch.Tensor:
        """F: [N, K+1, nL, D] -> [N, K]."""
        X = head_inputs(F, self.layer_idx, self.kind).to(self.w.device)
        return ((X - self.mean) / self.std) @ self.w


def head_inputs(F: torch.Tensor, layer_idx: int, kind: str) -> torch.Tensor:
    K = F.shape[1] - 1
    h = F[:, :K, layer_idx].float()
    if kind == "option":
        return h
    q = F[:, K:, layer_idx].float()
    return torch.cat([h, h * q], dim=-1)


def train_head(F: torch.Tensor, y: torch.Tensor, layer_idx: int, kind: str, l2: float,
               max_iter: int = 300) -> Head:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = head_inputs(F, layer_idx, kind).to(device)
    y = y.to(device)
    flat = X.reshape(-1, X.shape[-1])
    mean, std = flat.mean(0), flat.std(0).clamp_min(1e-6)
    Z = (X - mean) / std
    del X
    w = torch.zeros(Z.shape[-1], device=device, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Z @ w, y) + l2 * (w @ w)
        loss.backward()
        return loss

    opt.step(closure)
    return Head(layer_idx, kind, mean, std, w.detach())


def nll(head: Head, F: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        logits = head.logits(F)
        return float(torch.nn.functional.cross_entropy(logits, y.to(logits.device)))


class LocalListwiseHead(LocalMethod):
    type = "local_listwise_head"

    def __init__(self, ctx: MethodContext):
        super().__init__(ctx)
        p = ctx.params
        self.layers = [int(x) for x in p.get("layers", [8, 12, 16, 20, 24])]
        self.kinds = list(p.get("features", ["option", "option_query"]))
        self.l2_grid = [float(x) for x in p.get("l2", [1e-4, 1e-3, 1e-2, 1e-1, 1.0])]
        self.fit_split = p.get("fit_split", "train")
        self.select_split = p.get("select_split", "dev")
        self.max_iter = int(p.get("max_iter", 300))
        self.head: Head | None = None

    # ---------------------------------------------------------------- prompt / features
    def build(self, lm: LoadedModel, example: Example) -> tuple[list[int], list[int]]:
        """Token ids and positions [option_1_end, ..., option_K_end, query]."""
        marker = "\x00OPTIONS\x00"
        full = self.prompt(LISTWISE_INSTRUCTIONS.format(instructions=self.ctx.task.instructions),
                           f"{example.state}\n\nCandidate decisions:{marker}")
        head, tail = full.split(marker)
        ids = lm.encode(head)
        positions = []
        for i, o in enumerate(self.ctx.catalog.labels):
            line = f"[{option_marker(i)}] {o.value}"
            if o.has_description:
                line += f": {o.description}"
            ids += lm.encode("\n" + line)
            positions.append(len(ids) - 1)
        ids += lm.encode(tail + harmony.FINAL_PREFIX)
        positions.append(len(ids) - 1)
        return ids, positions

    def features(self, lm: LoadedModel, example: Example) -> tuple[torch.Tensor, int]:
        ids, positions = self.build(lm, example)
        return hidden_states_at(lm.model, ids, positions, self.layers), len(ids)

    def _cache_path(self, split: str):
        key = {
            "model": self.ctx.model.hf_model_id,
            "dtype": self.ctx.model.params.get("dtype", "bfloat16"),
            "data": dataset_fingerprint(self.ctx.dataset),
            "catalog": self.ctx.catalog.model_dump(mode="json"),
            "layers": self.layers,
            "instructions": LISTWISE_INSTRUCTIONS,
            "system": harmony.SYSTEM_TEMPLATE, "date": harmony.FIXED_DATE,
            "reasoning": self.reasoning,
        }
        digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:20]
        return self.ctx.settings.cache_dir / "features" / f"listwise-{digest}-{split}.pt"

    def split_features(self, lm: LoadedModel, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        path = self._cache_path(split)
        examples = load_split(self.ctx.dataset.name, split)
        index = {v: i for i, v in enumerate(self.values)}
        y = torch.tensor([index[e.label] for e in examples])
        if path.exists():
            data = torch.load(path)
            if data["ids"] == [e.id for e in examples]:
                return data["F"].float(), y
        feats = [self.features(lm, e)[0] for e in tqdm(examples, desc=f"features/{split}")]
        F = torch.stack(feats).half()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"ids": [e.id for e in examples], "F": F}, path)
        return F.float(), y

    # ---------------------------------------------------------------- fit / predict
    async def setup(self) -> None:
        lm = self._load()
        t0 = time.perf_counter()
        F_fit, y_fit = self.split_features(lm, self.fit_split)
        F_sel, y_sel = self.split_features(lm, self.select_split)
        self.info["feature_s"] = round(time.perf_counter() - t0, 1)
        grid = []
        best: tuple[float, Head] | None = None
        for li, layer in enumerate(self.layers):
            for kind in self.kinds:
                for l2 in self.l2_grid:
                    head = train_head(F_fit, y_fit, li, kind, l2, self.max_iter)
                    sel = nll(head, F_sel, y_sel)
                    acc = float((head.logits(F_sel).argmax(1).cpu() == y_sel).float().mean())
                    grid.append({"layer": layer, "kind": kind, "l2": l2,
                                 "select_nll": round(sel, 4), "select_acc": round(acc, 4)})
                    if best is None or sel < best[0]:
                        best = (sel, head)
        self.head = best[1]
        chosen = min(grid, key=lambda g: g["select_nll"])
        self.info.update({"fit_split": self.fit_split, "select_split": self.select_split,
                          "fit_n": len(y_fit), "select_n": len(y_sel), "chosen": chosen,
                          "grid": grid})

    def compute(self, lm: LoadedModel, example: Example) -> CallResult:
        if self.head is None:
            raise RuntimeError("setup() was not called")
        F, n_tokens = self.features(lm, example)
        with torch.no_grad():
            logits = self.head.logits(F[None])[0].float().cpu().tolist()
        return CallResult(
            probabilities=softmax(dict(zip(self.values, logits, strict=True))),
            prob_source="model",
            usage=Usage(input_tokens=n_tokens),
            extras={"logits": dict(zip(self.values, logits, strict=True))},
        )
