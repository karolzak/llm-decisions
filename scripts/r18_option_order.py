"""R18: does Decision-1 put options into a canonical (sorted) order before the model sees them?

The `question` probe found that, for two BANKING77 options sent in both orders, "Which option
is listed first?" picks the alphabetically first option 100% of the time and "listed last"
the alphabetically last one (chance against the order we sent). This script tests that
directly, and separates name fit from sorted position for duplicate options (R7):

    position  5 options with mixed-case names in a random order; "Which option is listed
              first / second / last?" -> which ordering do the answers follow: the request's,
              byte-wise sorted, or case-insensitive sorted?
    names     the correct option duplicated, the copy named so that it sorts *before* the
              original (`aardvark`, `0_<name>`, `A <name>`): if the copy loses, name fit
              decides; if it wins, sorted position does

Usage:
    python scripts/r18_option_order.py run          # Decision-1, ~110 requests (~15 min)
    python scripts/r18_option_order.py analyze
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from edb.probing.api import DecisionRequest, Option, Question

OUT = Path(__file__).resolve().parents[1] / "docs" / "results" / "2026-10-22-question-types" / \
    "r18-decision1.jsonl"
NAME_SETS = [  # mixed case, so byte order (uppercase first) and case-insensitive order differ
    ["apple", "Banana", "cherry", "Date", "elderberry"],
    ["zeta", "Alpha", "omega", "Beta", "gamma"],
    ["river", "Mountain", "lake", "Forest", "desert"],
    ["violet", "Indigo", "blue", "Green", "yellow"],
]
POSITION_QS = {"first": "Ignore the text. Which option is listed first in the list of options?",
               "second": "Ignore the text. Which option is listed second in the list of options?",
               "last": "Ignore the text. Which option is listed last in the list of options?"}
COPY_NAMES = {"aardvark": "aardvark", "zero_prefix": "0_{name}", "capital_prefix": "A {name}"}
# R18b: names and descriptions sort in opposite orders, so the answers show the sort key.
DESC_SETS = [
    (["apple", "banana", "cherry", "date", "elderberry"],
     ["Zesty fruit.", "Yellow fruit.", "Xanthic fruit.", "Wrinkled fruit.", "Violet fruit."]),
    (["alpha", "bravo", "charlie", "delta", "echo"],
     ["Zone five.", "Zone four.", "Zone three.", "Zone two.", "Zone one."]),
]


def cases(n_items: int) -> list[dict[str, Any]]:
    from edb.probing.runner import load_base_items

    rng = random.Random(18)
    out = []
    for s, names in enumerate(NAME_SETS):
        for rep in range(4):
            order = names[:]
            rng.shuffle(order)
            for qk in POSITION_QS:
                out.append({"id": f"pos-{s}-{rep}-{qk}", "kind": "position", "question": qk,
                            "state": "The weather was mild today.",
                            "request_options": order})
    items = load_base_items(n_items, ("banking77",))["banking77"]
    for item in items:
        opts = [o.value for o in item.options]
        others = [o for o in opts if o != item.gold]
        chosen = rng.sample(others, 4) + [item.gold]
        rng.shuffle(chosen)
        for variant, pattern in COPY_NAMES.items():
            copy = pattern.format(name=item.gold)
            order = chosen[:]
            order.insert(rng.randint(0, len(order)), copy)
            out.append({"id": f"dup-{item.id}-{variant}", "kind": "names", "variant": variant,
                        "state": item.state, "instructions": item.instructions,
                        "gold": item.gold, "copy": copy, "request_options": order})
    rng_desc = random.Random("18b")  # separate stream: earlier cases keep their content
    for s, (names, descs) in enumerate(DESC_SETS):
        pairs = list(zip(names, descs, strict=True))
        for rep in range(4):
            order = pairs[:]
            rng_desc.shuffle(order)
            for qk in ("first", "last"):
                out.append({"id": f"desc-{s}-{rep}-{qk}", "kind": "desc_position", "question": qk,
                            "state": "The weather was mild today.",
                            "request_options": [n for n, _ in order],
                            "descriptions": dict(order)})
    return out


def request(case: dict[str, Any]) -> DecisionRequest:
    if case["kind"] == "desc_position":
        q = Question("decision", POSITION_QS[case["question"]], "choice",
                     [Option(v, case["descriptions"][v]) for v in case["request_options"]])
    elif case["kind"] == "position":
        q = Question("decision", POSITION_QS[case["question"]], "choice",
                     [Option(v) for v in case["request_options"]])
    else:  # the copy keeps no description either: BANKING77 options are names only
        q = Question("decision", case["instructions"], "choice",
                     [Option(v) for v in case["request_options"]])
    return DecisionRequest(case["state"], [q])


async def run(n_items: int) -> None:
    from edb.probing.clients import load_probe_target, make_client
    from edb.providers.retry import HTTPError

    done = set()
    if OUT.exists():
        done = {json.loads(x)["id"] for x in OUT.read_text(encoding="utf-8").splitlines() if x}
    client = make_client(load_probe_target("decision-1"))
    try:
        with OUT.open("a", encoding="utf-8") as f:
            for case in [c for c in cases(n_items) if c["id"] not in done]:
                for attempt in range(8):
                    try:
                        r = await client.decide(request(case))
                        break
                    except HTTPError as exc:
                        if exc.status != 429 or attempt == 7:
                            raise
                        await asyncio.sleep(20 * (attempt + 1))
                    except httpx.TransportError:
                        if attempt == 7:
                            raise
                        await asyncio.sleep(10 * (attempt + 1))
                f.write(json.dumps({**case, "probabilities":
                                    r.answers["decision"].probabilities}) + "\n")
                f.flush()
                await asyncio.sleep(6.5)
    finally:
        await client.aclose()


def analyze() -> dict[str, Any]:
    rows = [json.loads(x) for x in OUT.read_text(encoding="utf-8").splitlines() if x]
    res: dict[str, Any] = {}
    idx = {"first": 0, "second": 1, "last": -1}
    for qk in POSITION_QS:
        xs = [r for r in rows if r["kind"] == "position" and r["question"] == qk]
        hit = {"request_order": [], "byte_sorted": [], "case_insensitive_sorted": []}
        for r in xs:
            p = r["probabilities"]
            top = max(p, key=p.get)
            orders = {"request_order": r["request_options"],
                      "byte_sorted": sorted(r["request_options"]),
                      "case_insensitive_sorted": sorted(r["request_options"], key=str.lower)}
            for name, order in orders.items():
                hit[name].append(top == order[idx[qk]])
        res[f"position_{qk}"] = {"n": len(xs), **{k: round(float(np.mean(v)), 3)
                                                  for k, v in hit.items()}}
    for variant in COPY_NAMES:
        xs = [r for r in rows if r["kind"] == "names" and r["variant"] == variant]
        if not xs:
            continue
        share = [r["probabilities"][r["gold"]] /
                 (r["probabilities"][r["gold"]] + r["probabilities"][r["copy"]]) for r in xs]
        copy_first = [sorted([r["gold"], r["copy"]])[0] == r["copy"] for r in xs]
        res[f"names_{variant}"] = {"n": len(xs), "original_share": round(float(np.mean(share)), 3),
                                   "copy_sorts_first": round(float(np.mean(copy_first)), 3)}
    for qk in ("first", "last"):
        xs = [r for r in rows if r["kind"] == "desc_position" and r["question"] == qk]
        if not xs:
            continue
        hit = {"request_order": [], "sorted_by_name": [], "sorted_by_description": []}
        for r in xs:
            p = r["probabilities"]
            top = max(p, key=p.get)
            names = r["request_options"]
            orders = {"request_order": names, "sorted_by_name": sorted(names),
                      "sorted_by_description": sorted(names, key=r["descriptions"].get)}
            for name, order in orders.items():
                hit[name].append(top == order[idx[qk]])
        res[f"descriptions_{qk}"] = {"n": len(xs), **{k: round(float(np.mean(v)), 3)
                                                      for k, v in hit.items()}}
    return res


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("cmd", choices=["run", "analyze"])
    parser.add_argument("--items", type=int, default=16)
    args = parser.parse_args()
    if args.cmd == "run":
        asyncio.run(run(args.items))
    print(json.dumps(analyze(), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
