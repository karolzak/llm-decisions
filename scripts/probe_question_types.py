"""Decision-1: how are `score` and `noul` questions answered? (probe finding 10)

One request per state holds every variant, so all variants share one forward context
(questions of one request are evaluated together; duplicates are bit-identical).

Usage:
    python scripts/probe_question_types.py <raw-output.json>
Results: docs/results/2026-10-22-question-types/finding10-decision1-raw.json (and -tokens.json
for the one-question-per-request billing check).
"""

import asyncio
import json
import math
import sys
from pathlib import Path

from edb.probing.api import DecisionRequest, Option, Question
from edb.probing.clients import load_probe_target, make_client

STATES = [
    "The hotel room was clean but the staff were rude and check-in took an hour.",
    "Absolutely loved it. Friendly staff, great breakfast, would book again tomorrow.",
    "Order #4471 arrived two weeks late and the box was crushed. Refund requested.",
]
FIVE = ["very negative", "negative", "mixed", "positive", "very positive"]
TEN = [f"{i}/10 satisfaction" for i in range(1, 11)]
Q = "How satisfied is the customer?"


def lv(texts):
    return [Option(t, t) for t in texts]


def questions():
    return [
        Question("s5", Q, "score", lv(FIVE)),
        Question("s5_rev", Q, "score", lv(FIVE[::-1])),
        Question("s3", Q, "score", lv(["negative", "mixed", "positive"])),
        Question("s10", Q, "score", lv(TEN)),
        Question("s5_digits", Q, "score", lv(["1", "2", "3", "4", "5"])),
        Question("s2", "Is the customer satisfied?", "score", lv(["no", "yes"])),
        Question("c5", Q, "choice", lv(FIVE)),
        Question("c5_rev", Q, "choice", lv(FIVE[::-1])),
        Question("n_plain", "Is the customer satisfied?", "noul"),
        Question("n_crit", "Is the customer satisfied?", "noul",
                 [Option("true", "The customer is satisfied."),
                  Option("false", "The customer is not satisfied.")]),
        Question("c_tf", "Is the customer satisfied?", "choice",
                 [Option("true", "The customer is satisfied."),
                  Option("false", "The customer is not satisfied.")]),
        Question("c_ft", "Is the customer satisfied?", "choice",
                 [Option("false", "The customer is not satisfied."),
                  Option("true", "The customer is satisfied.")]),
        Question("c_yn", "Is the customer satisfied?", "choice",
                 [Option("yes"), Option("no")]),
        Question("n_neg", "Is the customer unsatisfied?", "noul"),
    ]


async def main(out: Path) -> None:
    client = make_client(load_probe_target("decision-1"))
    rows = []
    try:
        for state in STATES:
            r = await client.decide(DecisionRequest(state=state, questions=questions()))
            rows.append({"state": state, "raw": r.raw, "input_tokens": r.input_tokens,
                         "output_tokens": r.output_tokens})
            print(state[:50], "in", r.input_tokens, "out", r.output_tokens)
    finally:
        await client.aclose()
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    for row in rows:
        print("\n==", row["state"][:60])
        for name, a in row["raw"]["answers"].items():
            extra = ""
            if a.get("type") == "noul":
                p = a["noul"]
                extra = f"log-odds {math.log(p / (1 - p)):.4f}"
            print(f"  {name:10s}", json.dumps({k: v for k, v in a.items() if k != 'type'})[:300],
                  extra)


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1])))
