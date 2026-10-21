"""Minimal LoRA for the linear layers of a decoder's blocks.

    y = W x + (alpha / r) * B A x      A: [r, in] (Kaiming init), B: [out, r] (zeros)

The base weights stay frozen in their dtype; A and B are float32. Adapters are saved as
`adapter.safetensors` + `adapter_config.json` (the exact module list), so loading needs
nothing but the base model.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from torch import nn

WEIGHTS, CONFIG = "adapter.safetensors", "adapter_config.json"


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.r, self.alpha = r, alpha
        self.scale = alpha / r
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=dev))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        z = self.dropout(x).to(self.lora_A.dtype) @ self.lora_A.T @ self.lora_B.T
        return y + (z * self.scale).to(y.dtype)


def _blocks(model: nn.Module) -> nn.Module:
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise ValueError("model has no `.model.layers` decoder blocks")
    return layers


def apply_lora(model: nn.Module, r: int = 16, alpha: float = 32.0, dropout: float = 0.0,
               pattern: str | None = None, modules: list[str] | None = None) -> list[str]:
    """Freeze `model` and wrap the decoder blocks' nn.Linear layers whose names (relative to
    `.model.layers`) match `pattern` (default: all), or exactly `modules`."""
    for p in model.parameters():
        p.requires_grad_(False)
    layers = _blocks(model)
    names = [n for n, m in layers.named_modules() if isinstance(m, nn.Linear)
             and (modules is None or n in modules)
             and (pattern is None or re.search(pattern, n))]
    if modules is not None and set(names) != set(modules):
        raise ValueError(f"adapter modules not found: {sorted(set(modules) - set(names))[:5]}")
    for name in names:
        parent_name, _, child = name.rpartition(".")
        parent = layers.get_submodule(parent_name) if parent_name else layers
        setattr(parent, child, LoRALinear(getattr(parent, child), r, alpha, dropout))
    return names


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for n, p in model.named_parameters() if "lora_" in n]


@contextmanager
def adapters_disabled(model: nn.Module):
    """Temporarily run the frozen base model (every LoRA update scaled to zero)."""
    mods = [m for m in model.modules() if isinstance(m, LoRALinear)]
    scales = [m.scale for m in mods]
    for m in mods:
        m.scale = 0.0
    try:
        yield
    finally:
        for m, s in zip(mods, scales, strict=True):
            m.scale = s


def lora_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().float().cpu().clone() for n, p in model.named_parameters()
            if "lora_" in n}


def load_lora_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    params = dict(model.named_parameters())
    missing = set(state) - set(params)
    if missing:
        raise KeyError(f"checkpoint has unknown LoRA parameters: {sorted(missing)[:3]}")
    for name, tensor in state.items():
        params[name].data.copy_(tensor.to(params[name].device, params[name].dtype))


def save_adapter(model: nn.Module, path: str | Path, config: dict[str, Any]) -> Path:
    from safetensors.torch import save_file

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    layers = _blocks(model)
    state = {n: p.detach().float().cpu().contiguous() for n, p in layers.named_parameters()
             if "lora_" in n}
    save_file(state, str(path / WEIGHTS))
    modules = sorted({n.rsplit(".", 1)[0] for n in state})
    first = layers.get_submodule(modules[0]) if modules else None
    cfg = {**config, "modules": modules,
           "r": getattr(first, "r", None), "alpha": getattr(first, "alpha", None)}
    (path / CONFIG).write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8")
    return path


def load_adapter(model: nn.Module, path: str | Path) -> dict[str, Any]:
    from safetensors.torch import load_file

    path = Path(path)
    cfg = json.loads((path / CONFIG).read_text(encoding="utf-8"))
    apply_lora(model, cfg["r"], cfg["alpha"], 0.0, modules=cfg["modules"])
    layers = _blocks(model)
    state = load_file(str(path / WEIGHTS))
    params = dict(layers.named_parameters())
    for name, tensor in state.items():
        params[name].data.copy_(tensor.to(params[name].device))
    model.eval()
    digest = hashlib.sha256((path / WEIGHTS).read_bytes()).hexdigest()[:16]
    return {"path": str(path), "sha256": digest, "r": cfg["r"], "alpha": cfg["alpha"],
            "modules": len(cfg["modules"]), "arm": cfg.get("arm")}
