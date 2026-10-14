"""Model loading (one copy per process) and GPU timing helpers."""

from __future__ import annotations

import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import torch

_CACHE: dict[tuple, LoadedModel] = {}
_LOCK = threading.Lock()


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any
    device: torch.device
    info: dict[str, Any] = field(default_factory=dict)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def token_id(self, token: str) -> int:
        tid = self.tokenizer.convert_tokens_to_ids(token)
        if tid is None or tid == self.tokenizer.unk_token_id:
            raise KeyError(f"unknown token {token!r}")
        return tid


def load(model_id: str, dtype: str = "bfloat16", attn_implementation: str | None = None,
         dequantize: bool | None = None) -> LoadedModel:
    """Load (or reuse) a causal LM + tokenizer. `dequantize=None` lets transformers decide."""
    key = (model_id, dtype, attn_implementation, dequantize)
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        kwargs: dict[str, Any] = {"dtype": getattr(torch, dtype), "device_map": str(device)}
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        if dequantize is not None:
            from transformers import Mxfp4Config

            kwargs["quantization_config"] = Mxfp4Config(dequantize=dequantize)
        t0 = time.perf_counter()
        path = _fetch(model_id)
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, **kwargs).eval()
        info = {
            "model_id": model_id,
            "load_s": round(time.perf_counter() - t0, 1),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "param_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
            "quantization": str(getattr(model.config, "quantization_config", None)),
            "attn_implementation": getattr(model.config, "_attn_implementation", None),
            "num_hidden_layers": model.config.num_hidden_layers,
            "torch": torch.__version__,
        }
        if device.type == "cuda":
            info["vram_after_load_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 2)
        loaded = LoadedModel(model, tokenizer, device, info)
        _CACHE[key] = loaded
        return loaded


def _fetch(model_id: str, attempts: int = 8) -> str:
    """Download a hub snapshot with backoff; loading from the local path avoids further hub
    requests. Parallel jobs share one egress IP, so anonymous hub requests can hit HTTP 429."""
    if os.path.isdir(model_id):
        return model_id
    from huggingface_hub import snapshot_download

    for attempt in range(attempts):
        try:
            return snapshot_download(model_id, ignore_patterns=["original/*", "metal/*"])
        except Exception as exc:
            if attempt == attempts - 1:
                raise
            delay = min(300.0, 15.0 * 2**attempt) * random.uniform(0.5, 1.5)
            print(f"snapshot_download({model_id}) failed ({type(exc).__name__}: {exc}); "
                  f"retry {attempt + 1}/{attempts - 1} in {delay:.0f}s", flush=True)
            time.sleep(delay)
    raise AssertionError("unreachable")


def register(key: tuple, loaded: LoadedModel) -> None:
    """Inject a preloaded model (tests)."""
    _CACHE[key] = loaded


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_vram_gb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return round(torch.cuda.max_memory_allocated(device) / 1024**3, 2)
