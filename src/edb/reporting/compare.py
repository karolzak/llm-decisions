"""Example-by-example comparison of two runs (Part 2: Decision-1 vs. its frozen base).

Works on full predictions and on a compact reference (`export_reference`: predicted label,
P(gold) and P(predicted) per example; full distributions only for small label sets), so the
comparison can run inside a GPU job that has no access to the reference run's files.
"""

from __future__ import annotations

import gzip
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr

from edb.schemas import Prediction

FULL_DISTRIBUTION_MAX_LABELS = 10


def _js(p: dict[str, float], q: dict[str, float]) -> float:
    m = {k: 0.5 * (p[k] + q[k]) for k in p}

    def kl(a):
        return sum(a[k] * math.log(a[k] / m[k]) for k in a if a[k] > 0)

    return 0.5 * kl(p) + 0.5 * kl(q)


def _usable(p: Prediction | None) -> bool:
    return p is not None and p.ok and bool(p.probabilities) and p.predicted is not None


def compare(ref: dict[str, Prediction], other: dict[str, Prediction],
            labels: list[str]) -> dict[str, Any]:
    ids = sorted(i for i in set(ref) & set(other) if _usable(ref[i]) and _usable(other[i]))
    if not ids:
        return {"n": 0}
    r_ok = np.array([ref[i].predicted == ref[i].gold for i in ids])
    o_ok = np.array([other[i].predicted == other[i].gold for i in ids])
    agree = np.array([ref[i].predicted == other[i].predicted for i in ids])
    pr = Counter(ref[i].predicted for i in ids)
    po = Counter(other[i].predicted for i in ids)
    chance = sum(pr[c] * po[c] for c in labels) / len(ids) ** 2
    both_wrong = ~r_ok & ~o_ok
    same_wrong = [ref[i].predicted == other[i].predicted
                  for i, w in zip(ids, both_wrong, strict=True) if w]
    pg_r = [ref[i].probabilities.get(ref[i].gold, 0.0) for i in ids]
    pg_o = [other[i].probabilities.get(other[i].gold, 0.0) for i in ids]
    varied = len(ids) >= 3 and np.std(pg_r) > 0 and np.std(pg_o) > 0
    phi = float(np.corrcoef(r_ok, o_ok)[0, 1]) if r_ok.std() and o_ok.std() else None
    full = [i for i in ids if len(ref[i].probabilities) == len(labels)
            and len(other[i].probabilities) == len(labels)]
    out = {
        "n": len(ids),
        "acc_reference": round(float(r_ok.mean()), 4),
        "acc_run": round(float(o_ok.mean()), 4),
        "top1_agreement": round(float(agree.mean()), 4),
        "chance_agreement": round(chance, 4),
        "kappa": round((agree.mean() - chance) / (1 - chance), 4) if chance < 1 else None,
        "both_correct": round(float((r_ok & o_ok).mean()), 4),
        "only_reference_correct": round(float((r_ok & ~o_ok).mean()), 4),
        "only_run_correct": round(float((~r_ok & o_ok).mean()), 4),
        "both_wrong": round(float(both_wrong.mean()), 4),
        "same_answer_when_both_wrong": round(float(np.mean(same_wrong)), 4) if same_wrong
        else None,
        "error_correlation_phi": round(phi, 4) if phi is not None else None,
        "p_gold_pearson": round(float(pearsonr(pg_r, pg_o)[0]), 4) if varied else None,
        "p_gold_spearman": round(float(spearmanr(pg_r, pg_o)[0]), 4) if varied else None,
        "mean_js_divergence": round(float(np.mean([_js(ref[i].probabilities,
                                                       other[i].probabilities)
                                                   for i in full])), 4) if full else None,
    }
    if len(labels) <= FULL_DISTRIBUTION_MAX_LABELS:
        out["prediction_mix_reference"] = {c: round(pr[c] / len(ids), 3) for c in labels}
        out["prediction_mix_run"] = {c: round(po[c] / len(ids), 3) for c in labels}
    return out


def export_reference(preds: dict[str, Prediction], labels: list[str], path: Path) -> int:
    """Write a compact gzipped JSONL reference (see module docstring)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for eid, p in sorted(preds.items()):
            if not _usable(p):
                continue
            row: dict[str, Any] = {"id": eid, "gold": p.gold, "pred": p.predicted}
            if len(labels) <= FULL_DISTRIBUTION_MAX_LABELS:
                row["p"] = {k: round(v, 6) for k, v in p.probabilities.items()}
            else:
                row["p"] = {k: round(p.probabilities.get(k, 0.0), 6)
                            for k in {p.gold, p.predicted}}
            f.write(json.dumps(row) + "\n")
            n += 1
    return n


def load_reference(path: Path, run_id: str = "reference") -> dict[str, Prediction]:
    out = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out[row["id"]] = Prediction(run_id=run_id, example_id=row["id"], gold=row["gold"],
                                        predicted=row["pred"], probabilities=row["p"],
                                        prob_source="model")
    return out
