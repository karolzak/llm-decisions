"""Forensics on stored responses: what the numbers themselves reveal about the readout.

    logit grid     If every log-probability gap within an answer is an exact multiple of a
                   step s, the probabilities are a softmax over logits stored in a low-
                   precision format: bf16 spaces values in [2^e, 2^(e+1)) by 2^(e-7), so
                   s = 0.125 means logits of magnitude 16-32 (typical of an LM head) with
                   no fitted temperature applied afterwards.
    confidence     Which formula produces the `confidence` field (Jev's normalised
                   max-probability, or the max-probability itself).
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np

from edb.probing.api import jev_confidence

GRID_STEPS = [2.0 ** -e for e in range(0, 7)]  # 1 ... 1/64


def logit_grid(probs: list[float], rel_tol: float = 0.02) -> float | None:
    """Largest step s (1 ... 1/64) such that all log-prob gaps are multiples of s."""
    lps = [math.log(p) for p in probs if p > 1e-12]
    if len(lps) < 2:
        return None
    top = max(lps)
    gaps = [top - lp for lp in lps if top - lp > 1e-9]
    if not gaps:
        return None
    for s in GRID_STEPS:
        if all(abs(g / s - round(g / s)) <= rel_tol for g in gaps):
            return s
    return 0.0


def bf16_magnitude(step: float) -> str | None:
    if not step:
        return None
    e = int(round(math.log2(step))) + 7
    return f"[{2.0 ** e:g}, {2.0 ** (e + 1):g})"


def analyze(records: list[dict[str, Any]]) -> dict[str, Any]:
    grids: Counter[str] = Counter()
    conf_jev, conf_max, sums, argmax_ok = [], [], [], []
    out_tokens: Counter[Any] = Counter()
    latencies, n_answers = [], 0
    for r in records:
        resp = r.get("response")
        if not resp:
            continue
        latencies.append(resp["latency_ms"])
        out_tokens[resp.get("output_tokens")] += 1
        for a in resp["answers"].values():
            p = a.get("probabilities") or {}
            if a.get("type") not in ("choice", "score") or len(p) < 2:
                continue
            n_answers += 1
            vals = list(p.values())
            step = logit_grid(vals)
            grids["n/a" if step is None else ("none" if step == 0 else f"{step:g}")] += 1
            sums.append(abs(sum(vals) - 1))
            if a.get("choice") is not None:
                argmax_ok.append(a["choice"] == max(p, key=p.get))
            c = a.get("confidence")
            if c is not None and a.get("type") == "choice":
                conf_jev.append(abs(c - jev_confidence(vals)) < 1e-6)
                conf_max.append(abs(c - max(vals)) < 1e-6)
    measured = sum(v for k, v in grids.items() if k != "n/a")
    on_grid = {k: round(v / measured, 4) for k, v in grids.most_common() if k != "n/a"} \
        if measured else {}
    main = next((k for k in on_grid if k != "none"), None)
    return {
        "answers": n_answers,
        "logit_grid_share": on_grid,
        "logit_grid_main_step": main,
        "bf16_logit_magnitude_if_main_step": bf16_magnitude(float(main)) if main else None,
        "confidence_is_jev_formula": round(float(np.mean(conf_jev)), 4) if conf_jev else None,
        "confidence_is_max_prob": round(float(np.mean(conf_max)), 4) if conf_max else None,
        "max_abs_sum_error": float(max(sums)) if sums else None,
        "choice_is_argmax": round(float(np.mean(argmax_ok)), 4) if argmax_ok else None,
        "output_tokens": {str(k): v for k, v in out_tokens.most_common(5)},
        "latency_p50_ms": round(float(np.median(latencies)), 1) if latencies else None,
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 1) if latencies else None,
    }
