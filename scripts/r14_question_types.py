"""R14: recover Decision-1's per-question-type templates (docs/replication.md, F15).

Decision-1 answers `noul` (yes/no) and `score` questions from templates of their own
(probe finding 10). This script asks Decision-1 and a local model the same single-question
requests, with several candidate templates for the local model, and measures which
candidate tracks Decision-1 best.

    build      fixed items: BANKING77 dev intents as yes/no questions (with and without
               criteria), SST-5 reviews as 5- and 3-level scores and a yes/no question
    decision1  query Decision-1 once per item (resumable)       -> decision1.jsonl
    local      every candidate on a local model (GPU)            -> ARTIFACTS_DIR/reports/r14/
               metrics against decision1.jsonl                    -> summary-<label>.json

Usage:
    python scripts/r14_question_types.py build
    python scripts/r14_question_types.py decision1
    python scripts/r14_question_types.py local --model qwen3.5-9b [--adapter training/r9_qwen9b]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from edb.probing.api import Answer, DecisionRequest, Option, Question
from edb.settings import get_settings

DATA = Path(__file__).resolve().parents[1] / "docs" / "results" / "2026-10-22-question-types"
ITEMS = DATA / "items.jsonl"
DECISION1 = DATA / "decision1.jsonl"
SST5 = ["very negative", "negative", "neutral", "positive", "very positive"]
SST3 = ["negative", "neutral", "positive"]
CANDIDATES: dict[str, tuple[str, dict[str, Any]]] = {
    # name: (question type it applies to, readout overrides)
    "noul_letters_ft": ("noul", {"noul_style": "letters", "noul_order": ["false", "true"]}),
    "noul_letters_tf": ("noul", {"noul_style": "letters", "noul_order": ["true", "false"]}),
    "noul_truefalse": ("noul", {"noul_style": "truefalse"}),
    "noul_yesno": ("noul", {"noul_style": "yesno"}),
    "score_letters": ("score", {"score_style": "letters"}),
    "score_digits": ("score", {"score_style": "digits"}),
    "score_scale": ("score", {"score_style": "scale"}),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def question(item: dict[str, Any]) -> Question:
    q = item["question"]
    return Question(q["name"], q["instructions"], q["type"],
                    [Option(o["value"], o.get("description")) for o in q["options"]])


# ---------------------------------------------------------------------- build
def build(n_bank: int, n_sst: int) -> None:
    from edb.datasets.store import dataset_labels, load_split
    from edb.training.corpus import SourceSpec, fetch_rows

    rng = random.Random(14)
    items: list[dict[str, Any]] = []
    intents = dataset_labels("banking77")
    dev = load_split("banking77", "dev")
    for k, e in enumerate(rng.sample(dev, n_bank)):
        asked = e.label if k % 2 == 0 else rng.choice([i for i in intents if i != e.label])
        topic = asked.replace("_", " ")
        text = f"Is the customer asking about {topic}?"
        crit = [{"value": "true", "description": f"The message is about {topic}."},
                {"value": "false", "description": "The message is about something else."}]
        for suffix, opts in (("plain", []), ("crit", crit)):
            items.append({"id": f"bank-{e.id}-{suffix}", "subset": f"banking_noul_{suffix}",
                          "state": e.state, "gold": "true" if asked == e.label else "false",
                          "question": {"name": "q", "instructions": text, "type": "noul",
                                       "options": opts}})
    src = SourceSpec(name="sst5-r14", dataset="SetFit/sst5", split="validation", state="{text}",
                     label="label", instructions=["-"])
    rows, _ = fetch_rows(src, n_sst * 3, seed=14)
    rows = rng.sample(rows, n_sst)
    for k, row in enumerate(rows):
        label, text = int(row["label"]), row["text"]
        levels5 = [{"value": t, "description": t} for t in SST5]
        levels3 = [{"value": t, "description": t} for t in SST3]
        pos = "true" if label >= 3 else "false" if label <= 1 else None
        base = {"state": text}
        items += [
            {**base, "id": f"sst-{k}-s5", "subset": "sst_score5", "gold": SST5[label],
             "question": {"name": "q", "instructions": "How positive is this movie review?",
                          "type": "score", "options": levels5}},
            {**base, "id": f"sst-{k}-s3", "subset": "sst_score3",
             "gold": SST3[(label > 1) + (label > 2)],
             "question": {"name": "q", "instructions": "How positive is this movie review?",
                          "type": "score", "options": levels3}},
            {**base, "id": f"sst-{k}-noul", "subset": "sst_noul", "gold": pos,
             "question": {"name": "q", "instructions": "Is this movie review positive?",
                          "type": "noul", "options": []}},
        ]
    DATA.mkdir(parents=True, exist_ok=True)
    ITEMS.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items) + "\n",
                     encoding="utf-8")
    print(f"{len(items)} items -> {ITEMS}")


# ---------------------------------------------------------------------- answers
def answer_record(a: Answer) -> dict[str, Any]:
    return {"type": a.type, "probabilities": a.probabilities, "score": a.score}


async def query_decision1(pause_s: float = 6.5) -> None:
    from edb.probing.clients import load_probe_target, make_client
    from edb.providers.retry import HTTPError

    done = {r["id"] for r in read_jsonl(DECISION1)}
    todo = [i for i in read_jsonl(ITEMS) if i["id"] not in done]
    client = make_client(load_probe_target("decision-1"))
    try:
        with DECISION1.open("a", encoding="utf-8") as f:
            for n, item in enumerate(todo, 1):
                for attempt in range(8):
                    try:
                        r = await client.decide(DecisionRequest(item["state"], [question(item)]))
                        break
                    except HTTPError as exc:  # the deployment allows ~9 requests/min
                        if exc.status != 429 or attempt == 7:
                            raise
                        await asyncio.sleep(20 * (attempt + 1))
                a = r.answers["q"]
                f.write(json.dumps({"id": item["id"], **answer_record(a), "raw": a.raw,
                                    "input_tokens": r.input_tokens}) + "\n")
                f.flush()
                if n % 20 == 0:
                    print(f"[decision1] {n}/{len(todo)}", flush=True)
                await asyncio.sleep(pause_s)
    finally:
        await client.aclose()


def run_local(model: str, adapter: str | None, label: str,
              limit: int | None = None) -> dict[str, Any]:
    from edb.local.chat_readout import ChatLetterReadout, ReadoutConfig, load_readout

    base = load_readout(model, adapter)
    items = read_jsonl(ITEMS)
    if limit:  # smoke tests: the first `limit` items of every subset
        seen: dict[str, int] = {}
        items = [i for i in items if (seen.__setitem__(i["subset"], seen.get(i["subset"], 0) + 1)
                                      or seen[i["subset"]] <= limit)]
    out: dict[str, dict[str, Any]] = {}
    for name, (qtype, overrides) in CANDIDATES.items():
        rd = ChatLetterReadout(base.lm, ReadoutConfig.from_params(vars(base.cfg), overrides))
        t0 = time.perf_counter()
        out[name] = {}
        for item in items:
            if item["question"]["type"] != qtype:
                continue
            a, diag = rd.answer(item["state"], question(item))
            out[name][item["id"]] = {**answer_record(a),
                                     "on_format_mass": diag["on_format_mass"]}
        print(f"[local] {name}: {len(out[name])} items in {time.perf_counter() - t0:.0f}s",
              flush=True)
    folder = get_settings().artifacts_dir / "reports" / "r14"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"answers-{label}.json").write_text(json.dumps(out), encoding="utf-8")
    summary = {"label": label, "model": model, "adapter": adapter,
               "metrics": compare(out, items, read_jsonl(DECISION1))}
    (folder / f"summary-{label}.json").write_text(json.dumps(summary, indent=1),
                                                  encoding="utf-8")
    print(json.dumps(summary["metrics"], indent=1))
    return summary


# ---------------------------------------------------------------------- metrics
def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _rank(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(x)).astype(float)


def _corr(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return round(float(np.corrcoef(a, b)[0, 1]), 4)


def compare(local: dict[str, dict[str, Any]], items: list[dict[str, Any]],
            d1_rows: list[dict[str, Any]]) -> dict[str, Any]:
    d1 = {r["id"]: r for r in d1_rows}
    by_id = {i["id"]: i for i in items}
    out: dict[str, Any] = {}
    for name, answers in local.items():
        groups: dict[str, list[str]] = {}
        for item_id in answers:
            if item_id in d1:
                groups.setdefault(by_id[item_id]["subset"], []).append(item_id)
        out[name] = {}
        for subset, ids in sorted(groups.items()):
            gold = [by_id[i]["gold"] for i in ids]
            if by_id[ids[0]]["question"]["type"] == "noul":
                x = [_logit(answers[i]["probabilities"]["true"]) for i in ids]
                y = [_logit(d1[i]["probabilities"]["true"]) for i in ids]
                slope, icpt = np.polyfit(x, y, 1) if np.std(x) > 0 else (0.0, float(np.mean(y)))
                fit = np.array(x) * slope + icpt
                labelled = [(xi, yi, g) for xi, yi, g in zip(x, y, gold, strict=True) if g]
                m = {"n": len(ids), "pearson_logodds": _corr(x, y),
                     "spearman_logodds": _corr(list(_rank(np.array(x))), list(_rank(np.array(y)))),
                     "mae_logodds": round(float(np.mean(np.abs(np.array(x) - y))), 4),
                     "affine_slope": round(float(slope), 4),
                     "affine_intercept": round(float(icpt), 4),
                     "mae_after_affine": round(float(np.mean(np.abs(fit - y))), 4),
                     "same_answer": round(float(np.mean([(a > 0) == (b > 0)
                                                        for a, b in zip(x, y, strict=True)])), 4)}
                if labelled:
                    m["accuracy"] = round(float(np.mean([(a > 0) == (g == "true")
                                                         for a, _, g in labelled])), 4)
                    m["accuracy_decision1"] = round(float(np.mean([(b > 0) == (g == "true")
                                                                   for _, b, g in labelled])), 4)
            else:
                values = [o["value"] for o in by_id[ids[0]]["question"]["options"]]
                k = len(values) - 1
                tv, ex_l, ex_d, same, acc, acc_d = [], [], [], [], [], []
                for i, g in zip(ids, gold, strict=True):
                    p, q = answers[i]["probabilities"], d1[i]["probabilities"]
                    tv.append(0.5 * sum(abs(p[v] - q[v]) for v in values))
                    ex_l.append(sum(n * p[v] for n, v in enumerate(values)) / k)
                    ex_d.append(sum(n * q[v] for n, v in enumerate(values)) / k)
                    top_l, top_d = max(values, key=p.get), max(values, key=q.get)
                    same.append(top_l == top_d)
                    acc.append(top_l == g)
                    acc_d.append(top_d == g)
                m = {"n": len(ids), "mean_tv": round(float(np.mean(tv)), 4),
                     "pearson_expected_score": _corr(ex_l, ex_d),
                     "mae_expected_score": round(float(np.mean(np.abs(np.array(ex_l) - ex_d))), 4),
                     "same_answer": round(float(np.mean(same)), 4),
                     "accuracy": round(float(np.mean(acc)), 4),
                     "accuracy_decision1": round(float(np.mean(acc_d)), 4)}
            m["mean_on_format_mass"] = round(float(np.mean([answers[i]["on_format_mass"]
                                                            for i in ids])), 4)
            out[name][subset] = m
    return out


def _d1_answer(a: dict[str, Any]) -> dict[str, Any]:
    """Decision-1 noul answers carry only P(true); keep the shape of local answers."""
    if a["type"] == "noul" and "true" not in a["probabilities"]:
        p = a["raw"]["noul"]
        a["probabilities"] = {"true": p, "false": 1 - p}
    return a


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--banking", type=int, default=40)
    b.add_argument("--sst", type=int, default=40)
    sub.add_parser("decision1")
    loc = sub.add_parser("local")
    loc.add_argument("--model", default="qwen3.5-9b")
    loc.add_argument("--adapter")
    loc.add_argument("--label", help="output name (default: model or adapter name)")
    loc.add_argument("--limit", type=int, help="items per subset (smoke tests)")
    args = parser.parse_args()
    if args.cmd == "build":
        build(args.banking, args.sst)
    elif args.cmd == "decision1":
        asyncio.run(query_decision1())
        rows = [_d1_answer(r) for r in read_jsonl(DECISION1)]
        DECISION1.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        print(f"{len(rows)} Decision-1 answers -> {DECISION1}")
    else:
        label = args.label or (Path(args.adapter).name if args.adapter else args.model)
        run_local(args.model, args.adapter, label, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
