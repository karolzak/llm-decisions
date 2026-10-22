"""R19: evaluate a frozen candidate replica on benchmarks never used for training or selection.

All earlier arms were compared on ACE and BANKING77 test, so the best arm's scores there are
selected on test data. R19 evaluates one candidate, chosen beforehand, on three benchmarks
we had not touched, and Decision-1 on the same items:

    minds14      PolyAI/minds14 en-US, 14 e-banking intents (spoken requests, transcribed)
    atis         tuetschek/atis test, airline-travel intents (at most 30 per intent, combined
                 intents like `flight+airfare` dropped)
    contractnli  kiddothe2b/contract-nli (contractnli_a test): does a contract excerpt entail,
                 contradict or not mention a statement? (a compliance-style task, like ACE)

    build      fixed items                            -> docs/results/2026-10-23-unseen/
    decision1  query Decision-1 once per item         -> .../decision1.jsonl (resumable)
    local      a local model (GPU), options sorted    -> ARTIFACTS_DIR/reports/r19/
               metrics, also for Decision-1 on the same items

Usage:
    python scripts/r19_unseen_eval.py build
    python scripts/r19_unseen_eval.py decision1
    python scripts/r19_unseen_eval.py local --model qwen3.5-9b [--adapter training/<arm>]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from edb.probing.api import DecisionRequest, Option, Question
from edb.settings import get_settings

DATA = Path(__file__).resolve().parents[1] / "docs" / "results" / "2026-10-23-unseen"
ITEMS = DATA / "items.jsonl"
DECISION1 = DATA / "decision1.jsonl"
NLI_DESCRIPTIONS = {
    "entailment": "The contract excerpt implies that the statement is true.",
    "contradiction": "The contract excerpt implies that the statement is false.",
    "neutral": "The contract excerpt does not settle whether the statement is true.",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def question(item: dict[str, Any]) -> Question:
    return Question("decision", item["instructions"], "choice",
                    [Option(o["value"], o.get("description")) for o in item["options"]])


# ---------------------------------------------------------------------- build
def _stratified(rows: list[dict], label: str, per_class: int, rng: random.Random) -> list[dict]:
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[str(r[label])].append(r)
    out = []
    for key in sorted(by):
        group = by[key]
        rng.shuffle(group)
        out += group[:per_class]
    rng.shuffle(out)
    return out


def build() -> None:
    from edb.training.corpus import SourceSpec, fetch_rows

    rng = random.Random(19)
    items: list[dict[str, Any]] = []

    src = SourceSpec(name="minds14-r19", dataset="PolyAI/minds14", config="en-US",
                     state="{transcription}", label="intent_class", instructions=["-"])
    rows, feats = fetch_rows(src, 563, seed=19)
    names = feats["intent_class"]["names"]
    for r in _stratified(rows, "intent_class", 15, rng):
        items.append({"dataset": "minds14", "state": r["transcription"],
                      "instructions": "What does the bank customer want?",
                      "options": [{"value": n} for n in names],
                      "gold": names[int(r["intent_class"])]})

    src = SourceSpec(name="atis-r19", dataset="tuetschek/atis", split="test", state="{text}",
                     label="intent", instructions=["-"])
    rows, _ = fetch_rows(src, 893, seed=19)
    counts = Counter(r["intent"] for r in rows)
    keep = sorted(k for k, n in counts.items() if "+" not in k and n >= 3)
    rows = [r for r in rows if r["intent"] in keep]
    for r in _stratified(rows, "intent", 30, rng):
        items.append({"dataset": "atis", "state": r["text"],
                      "instructions": "What is the traveller asking about?",
                      "options": [{"value": k} for k in keep], "gold": r["intent"]})

    src = SourceSpec(name="contractnli-r19", dataset="kiddothe2b/contract-nli",
                     config="contractnli_a", split="test", state="-", label="label",
                     instructions=["-"])
    rows, feats = fetch_rows(src, 1991, seed=19)
    names = feats["label"]["names"]
    for r in _stratified(rows, "label", 67, rng):
        state = f"Contract excerpt:\n{r['premise']}\n\nStatement:\n{r['hypothesis']}"
        items.append({"dataset": "contractnli", "state": state,
                      "instructions": "Does the contract excerpt entail the statement, contradict "
                                      "it, or leave it open?",
                      "options": [{"value": n, "description": NLI_DESCRIPTIONS[n]}
                                  for n in names],
                      "gold": names[int(r["label"])]})
    for i, item in enumerate(items):
        item["id"] = f"{item['dataset']}-{i:04d}"
    DATA.mkdir(parents=True, exist_ok=True)
    ITEMS.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items) + "\n",
                     encoding="utf-8")
    print(f"{len(items)} items -> {ITEMS}", dict(Counter(i["dataset"] for i in items)))


# ---------------------------------------------------------------------- Decision-1
async def query_decision1(pause_s: float = 6.5) -> None:
    from edb.probing.clients import load_probe_target, make_client
    from edb.providers.retry import HTTPError

    done = {r["id"] for r in read_jsonl(DECISION1)}
    todo = [i for i in read_jsonl(ITEMS) if i["id"] not in done]
    client = make_client(load_probe_target("decision-1"))
    try:
        with DECISION1.open("a", encoding="utf-8") as f:
            for n, item in enumerate(todo, 1):
                for attempt in range(10):
                    try:
                        r = await client.decide(DecisionRequest(item["state"], [question(item)]))
                        break
                    except HTTPError as exc:  # the deployment allows ~9 requests/min
                        if exc.status != 429 or attempt == 9:
                            raise
                        await asyncio.sleep(20 * (attempt + 1))
                    except httpx.TransportError:
                        if attempt == 9:
                            raise
                        await asyncio.sleep(10 * (attempt + 1))
                f.write(json.dumps({"id": item["id"],
                                    "probabilities": r.answers["decision"].probabilities,
                                    "input_tokens": r.input_tokens}) + "\n")
                f.flush()
                if n % 25 == 0:
                    print(f"[decision1] {n}/{len(todo)}", flush=True)
                await asyncio.sleep(pause_s)
    finally:
        await client.aclose()


# ---------------------------------------------------------------------- metrics
def _ece(conf: np.ndarray, ok: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    return float(sum(((conf > lo) & (conf <= hi)).mean() *
                     abs(conf[(conf > lo) & (conf <= hi)].mean() -
                         ok[(conf > lo) & (conf <= hi)].mean())
                     for lo, hi in zip(edges[:-1], edges[1:], strict=True)
                     if ((conf > lo) & (conf <= hi)).any()))


def _fit_t(logps: list[np.ndarray], golds: list[int]) -> float:
    """Temperature minimising NLL on these items (a diagnostic, fitted in-sample)."""
    best, best_nll = 1.0, math.inf
    for t in np.exp(np.linspace(math.log(0.25), math.log(8), 141)):
        nll = 0.0
        for lp, g in zip(logps, golds, strict=True):
            z = lp / t
            nll -= z[g] - np.logaddexp.reduce(z)
        if nll < best_nll:
            best, best_nll = float(t), nll
    return round(best, 3)


def metrics(answers: dict[str, dict[str, float]], items: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for ds in sorted({i["dataset"] for i in items}):
        its = [i for i in items if i["dataset"] == ds and i["id"] in answers]
        if not its:
            continue
        conf, ok, nll, logps, golds = [], [], [], [], []
        for i in its:
            values = [o["value"] for o in i["options"]]
            p = np.array([max(answers[i["id"]].get(v, 0.0), 1e-12) for v in values])
            p = p / p.sum()
            g = values.index(i["gold"])
            conf.append(p.max())
            ok.append(float(p.argmax() == g))
            nll.append(-math.log(p[g]))
            logps.append(np.log(p))
            golds.append(g)
        c, k = np.array(conf), np.array(ok)
        out[ds] = {"n": len(its), "accuracy": round(float(k.mean()), 4),
                   "nll": round(float(np.mean(nll)), 4), "ece": round(_ece(c, k), 4),
                   "mean_confidence": round(float(c.mean()), 4),
                   "T_in_sample": _fit_t(logps, golds)}
    return out


def agreement(a: dict[str, dict[str, float]], b: dict[str, dict[str, float]],
              items: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for ds in sorted({i["dataset"] for i in items}):
        ids = [i["id"] for i in items if i["dataset"] == ds and i["id"] in a and i["id"] in b]
        if not ids:
            continue
        top_a = [max(a[x], key=a[x].get) for x in ids]
        top_b = [max(b[x], key=b[x].get) for x in ids]
        po = float(np.mean([x == y for x, y in zip(top_a, top_b, strict=True)]))
        labels = sorted(set(top_a) | set(top_b))
        pe = sum((top_a.count(v) / len(ids)) * (top_b.count(v) / len(ids)) for v in labels)
        out[ds] = {"n": len(ids), "agreement": round(po, 4),
                   "kappa": round((po - pe) / (1 - pe), 4) if pe < 1 else None}
    return out


# ---------------------------------------------------------------------- local
def run_local(model: str, adapter: str | None, label: str, readout: dict[str, Any],
              limit: int | None = None) -> dict[str, Any]:
    from edb.local.chat_readout import load_readout

    rd = load_readout(model, adapter, readout)
    items = read_jsonl(ITEMS)
    if limit:
        seen: Counter = Counter()
        items = [i for i in items if (seen.update([i["dataset"]]) or seen[i["dataset"]] <= limit)]
    answers: dict[str, dict[str, float]] = {}
    t0 = time.perf_counter()
    for n, item in enumerate(items, 1):
        a, _ = rd.answer(item["state"], question(item))
        answers[item["id"]] = a.probabilities
        if n % 100 == 0:
            print(f"[local] {n}/{len(items)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    d1 = {r["id"]: r["probabilities"] for r in read_jsonl(DECISION1)}
    by_id = {i["id"]: i for i in items}
    summary = {"label": label, "model": model, "adapter": adapter, "readout": readout,
               "metrics": metrics(answers, items),
               "decision1": metrics({k: v for k, v in d1.items() if k in answers}, items),
               "vs_decision1": agreement(answers, d1, items),
               # Compact per-item answers, so Decision-1 comparisons can be (re)computed
               # locally from the job's tags: id -> [top option, p(top), p(gold)].
               "answers_compact": {k: [max(v, key=v.get), round(max(v.values()), 5),
                                       round(v.get(by_id[k]["gold"], 0.0), 5)]
                                   for k, v in answers.items()}}
    folder = get_settings().artifacts_dir / "reports" / "r19"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"answers-{label}.json").write_text(json.dumps(answers), encoding="utf-8")
    (folder / f"summary-{label}.json").write_text(json.dumps(summary, indent=1),
                                                  encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return summary


def compare_summary(path: Path) -> dict[str, Any]:
    """Decision-1 metrics and agreement for a local run, from a job summary (aml_summaries)
    or a summary-<label>.json, using the compact answers and the local decision1.jsonl."""
    data = json.loads(path.read_text(encoding="utf-8"))
    runs = data.get("summary", {}).get("reports", {}).get("r19") or {data["label"]: data}
    items = read_jsonl(ITEMS)
    by_id = {i["id"]: i for i in items}
    d1 = {r["id"]: r["probabilities"] for r in read_jsonl(DECISION1)}
    out = {}
    for label, s in runs.items():
        compact = s["answers_compact"]
        ids = [k for k in compact if k in d1]
        res: dict[str, Any] = {"metrics": s["metrics"], "decision1": metrics(
            {k: d1[k] for k in ids}, [by_id[k] for k in ids])}
        for ds in sorted({by_id[k]["dataset"] for k in ids}):
            sel = [k for k in ids if by_id[k]["dataset"] == ds]
            top_a = [compact[k][0] for k in sel]
            top_b = [max(d1[k], key=d1[k].get) for k in sel]
            po = float(np.mean([a == b for a, b in zip(top_a, top_b, strict=True)]))
            labels = sorted(set(top_a) | set(top_b))
            pe = sum((top_a.count(v) / len(sel)) * (top_b.count(v) / len(sel)) for v in labels)
            both_wrong = [(a, b) for k, a, b in zip(sel, top_a, top_b, strict=True)
                          if a != by_id[k]["gold"] and b != by_id[k]["gold"]]
            res.setdefault("vs_decision1", {})[ds] = {
                "n": len(sel), "agreement": round(po, 4),
                "kappa": round((po - pe) / (1 - pe), 4) if pe < 1 else None,
                "same_answer_when_both_wrong": round(float(np.mean(
                    [a == b for a, b in both_wrong])), 4) if both_wrong else None}
        out[label] = res
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    sub.add_parser("decision1")
    loc = sub.add_parser("local")
    loc.add_argument("--model", default="qwen3.5-9b")
    loc.add_argument("--adapter")
    loc.add_argument("--label")
    loc.add_argument("--readout", default='{"labels": "codes", "option_order": "sorted"}',
                     help="JSON readout overrides (default: code labels, byte-sorted options)")
    loc.add_argument("--limit", type=int, help="items per dataset (smoke tests)")
    cmp = sub.add_parser("compare", help="Decision-1 comparison from a job summary")
    cmp.add_argument("summary", type=Path, nargs="+")
    args = parser.parse_args()
    if args.cmd == "compare":
        for path in args.summary:
            print(path.name, json.dumps(compare_summary(path), indent=1))
        return 0
    if args.cmd == "build":
        build()
    elif args.cmd == "decision1":
        asyncio.run(query_decision1())
        print(json.dumps(metrics({r["id"]: r["probabilities"] for r in read_jsonl(DECISION1)},
                                 read_jsonl(ITEMS)), indent=1))
    else:
        label = args.label or (Path(args.adapter).name if args.adapter else args.model)
        run_local(args.model, args.adapter, label, json.loads(args.readout), args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
