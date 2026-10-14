"""C9: adaptive cascade built offline from two stored runs (no new model calls).

Fast method first; if its confidence is below a threshold, use the slow method's answer.
The threshold is chosen on the calibration split: the lowest escalation rate whose cascade
accuracy is within `--max-drop` of the slow method alone. It is then applied unchanged to the
evaluation split. Latency / cost per example = fast + (slow if escalated).

Writes a regular run directory (artifacts/runs/<dataset>/<id>__<catalog>/<split>/) so the
cascade appears in reports, plus the full threshold frontier in artifacts/analysis/.

Usage:
    uv run python scripts/build_cascade.py --dataset ace \
        --fast c6p_oss_letter_debiased --slow c8_oss_reason_then_read
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

import numpy as np

from edb.config import config_hash, load_dataset_config
from edb.reporting.collect import load_run_split
from edb.runners.predict import run_dir
from edb.schemas import Prediction, Usage
from edb.settings import get_settings


def conf(p: Prediction) -> float:
    return max(p.probabilities.values()) if p.ok and p.probabilities else 0.0


def evaluate(fast: dict, slow: dict, ids: list[str], t: float) -> tuple[float, float]:
    esc = [conf(fast[i]) < t for i in ids]
    correct = [(slow[i] if e else fast[i]).predicted == fast[i].gold for i, e in zip(ids, esc,
                                                                                 strict=True)]
    return float(np.mean(correct)), float(np.mean(esc))


def combine(fast: Prediction, slow: Prediction, t: float, run_id: str) -> Prediction:
    escalated = conf(fast) < t
    chosen = slow if escalated else fast

    def add(a, b):
        return None if a is None and b is None else (a or 0.0) + (b or 0.0)

    usage = Usage(
        input_tokens=(fast.usage.input_tokens if fast.usage else 0)
        + (slow.usage.input_tokens if escalated and slow.usage else 0),
        output_tokens=(fast.usage.output_tokens if fast.usage else 0)
        + (slow.usage.output_tokens if escalated and slow.usage else 0),
    )
    return Prediction(
        run_id=run_id, example_id=fast.example_id, gold=fast.gold, predicted=chosen.predicted,
        probabilities=chosen.probabilities, prob_source=chosen.prob_source,
        latency_ms=add(fast.latency_ms, slow.latency_ms if escalated else None),
        usage=usage, cost_usd=add(fast.cost_usd, slow.cost_usd if escalated else None),
        error=chosen.error,
        extras={"escalated": escalated, "threshold": t, "fast_confidence": conf(fast)},
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Build an offline fast/slow cascade")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--fast", required=True, help="experiment id of the fast method")
    ap.add_argument("--slow", required=True, help="experiment id of the slow method")
    ap.add_argument("--catalog", nargs="+", default=None, help="catalog roles (default: all)")
    ap.add_argument("--id", default="c9_oss_cascade")
    ap.add_argument("--fit-split", default="calibration")
    ap.add_argument("--eval-split", nargs="+", default=["test"])
    ap.add_argument("--max-drop", type=float, default=0.005)
    args = ap.parse_args()

    ds = load_dataset_config(args.dataset)
    roles = args.catalog or list(ds.label_catalogs)
    for role in roles:
        fast_id, slow_id = f"{args.fast}__{role}", f"{args.slow}__{role}"
        runs = {}
        for split in [args.fit_split, *args.eval_split]:
            f, s = load_run_split(ds.name, fast_id, split), load_run_split(ds.name, slow_id, split)
            if f is None or s is None:
                print(f"[skip] {role}/{split}: missing {fast_id if f is None else slow_id}")
                break
            runs[split] = (f, s)
        else:
            f_fit, s_fit = runs[args.fit_split]
            ids = sorted(set(f_fit.predictions) & set(s_fit.predictions))
            fast, slow = f_fit.predictions, s_fit.predictions
            slow_acc = float(np.mean([slow[i].predicted == slow[i].gold for i in ids]))
            thresholds = sorted({conf(fast[i]) for i in ids} | {0.0, 1.01})
            frontier = []
            for t in thresholds:
                acc, esc = evaluate(fast, slow, ids, t)
                frontier.append({"threshold": t, "accuracy": acc, "escalation": esc})
            ok = [p for p in frontier if p["accuracy"] >= slow_acc - args.max_drop]
            chosen = min(ok, key=lambda p: (p["escalation"], -p["accuracy"]))
            t = chosen["threshold"]
            run_id = f"{args.id}__{role}"
            snapshot = {"cascade": {"fast": fast_id, "slow": slow_id, "threshold": t,
                                    "fit_split": args.fit_split, "max_drop": args.max_drop,
                                    "fast_config": f_fit.manifest["config_hash"],
                                    "slow_config": s_fit.manifest["config_hash"]},
                        "experiment": {"id": args.id, "method": {"type": "offline_cascade"}},
                        "catalog_role": role, "model": f_fit.config.get("model"),
                        "dataset": ds.name}
            chash = config_hash(snapshot)
            analysis = {"fit": {"slow_accuracy": slow_acc, "chosen": chosen,
                                "frontier": frontier}}
            for split, (f, s) in runs.items():
                out = run_dir(ds.name, run_id, split)
                out.mkdir(parents=True, exist_ok=True)
                common = sorted(set(f.predictions) & set(s.predictions))
                preds = [combine(f.predictions[i], s.predictions[i], t, run_id) for i in common]
                (out / "predictions.jsonl").write_text(
                    "".join(p.model_dump_json() + "\n" for p in preds), encoding="utf-8")
                manifest = {"run_id": run_id, "dataset": ds.name, "split": split,
                            "config_hash": chash, "config": snapshot,
                            "leaderboards": ["product", "controlled"],
                            "sessions": [{"started_at": datetime.now(UTC).isoformat(
                                timespec="seconds"), "submitted": len(preds), "offline": True}]}
                (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                   encoding="utf-8")
                acc = float(np.mean([p.predicted == p.gold for p in preds]))
                esc = float(np.mean([p.extras["escalated"] for p in preds]))
                analysis[split] = {"accuracy": acc, "escalation": esc, "n": len(preds)}
                print(f"{run_id:32s} {split:12s} t={t:.4f} acc={acc:.4f} escalated={esc:.3f}")
            dest = get_settings().artifacts_dir / "analysis" / ds.name
            dest.mkdir(parents=True, exist_ok=True)
            (dest / f"cascade_{role}.json").write_text(json.dumps(analysis, indent=1),
                                                       encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
