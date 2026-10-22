"""R16: answer many questions about one state from a shared, cached prompt prefix.

Decision-1 bills the state once per request and answered 64 yes/no questions in under a
second (probe finding 1). `ChatLetterReadout.answer_shared` encodes the common prefix of the
questions' prompts once and reads each question from a copy of the cache. This script checks
that shared answers equal one-question-at-a-time answers and measures the speed-up.

Usage:
    python scripts/r16_shared_state.py --model qwen3.5-9b --states 4 --questions 64
"""

from __future__ import annotations

import argparse
import json
import random
import time

import torch

from edb.datasets.store import dataset_labels, load_split
from edb.local.chat_readout import ChatLetterReadout, ReadoutConfig, load_readout
from edb.probing.api import Option, Question
from edb.settings import get_settings


def sync(rd: ChatLetterReadout) -> None:
    if rd.lm.device.type == "cuda":
        torch.cuda.synchronize(rd.lm.device)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="qwen3.5-9b")
    parser.add_argument("--adapter")
    parser.add_argument("--states", type=int, default=4)
    parser.add_argument("--questions", type=int, default=64)
    parser.add_argument("--label")
    args = parser.parse_args()
    base = load_readout(args.model, args.adapter)
    rd = ChatLetterReadout(base.lm, ReadoutConfig.from_params(
        vars(base.cfg), {"noul_style": "truefalse", "score_style": "digits"}))
    rng = random.Random(16)
    intents = dataset_labels("banking77")
    states = rng.sample(load_split("banking77", "dev"), args.states)
    results = []
    for e in states:
        asked = rng.sample(intents, args.questions - 2)
        qs = [Question(f"n{i}", f"Is the customer asking about {a.replace('_', ' ')}?", "noul")
              for i, a in enumerate(asked)]
        qs.append(Question("choice", "Which banking support intent best matches the "
                           "customer's message?", "choice",
                           [Option(a) for a in rng.sample(intents, 10)]))
        qs.append(Question("urgency", "How urgent is the customer's request?", "score",
                           [Option(t, t) for t in ("not urgent", "somewhat urgent",
                                                   "very urgent")]))
        rd.answer(e.state, qs[0])  # warm-up
        sync(rd)
        t0 = time.perf_counter()
        separate = {q.name: rd.answer(e.state, q) for q in qs}
        sync(rd)
        t1 = time.perf_counter()
        shared, diags, tokens = rd.answer_shared(e.state, qs)
        sync(rd)
        t2 = time.perf_counter()
        diff = max(abs(separate[q.name][0].probabilities[v] - shared[q.name].probabilities[v])
                   for q in qs for v in shared[q.name].probabilities)
        same = sum(separate[q.name][0].choice == shared[q.name].choice for q in qs) / len(qs)
        results.append({"state_id": e.id, "questions": len(qs),
                        "separate_s": round(t1 - t0, 3), "shared_s": round(t2 - t1, 3),
                        "speedup": round((t1 - t0) / max(1e-9, t2 - t1), 2),
                        "separate_tokens": sum(d["input_tokens"] for _, d in separate.values()),
                        "shared_tokens": tokens,
                        "shared_prefix_tokens": diags["n0"].get("shared_prefix_tokens"),
                        "max_abs_prob_diff": diff, "same_choice": same})
        print(json.dumps(results[-1]), flush=True)
    summary = {"model": args.model, "adapter": args.adapter, "per_state": results,
               "mean_speedup": round(sum(r["speedup"] for r in results) / len(results), 2),
               "max_abs_prob_diff": max(r["max_abs_prob_diff"] for r in results),
               "min_same_choice": min(r["same_choice"] for r in results),
               "device": str(rd.lm.device)}
    label = args.label or args.model
    folder = get_settings().artifacts_dir / "reports" / "r16"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"summary-{label}.json").write_text(json.dumps(summary, indent=1),
                                                  encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_state"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
