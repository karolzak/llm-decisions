"""R24: synthetic many-domain routing data, generated with GPT-6 Luna (Azure OpenAI).

R9-R21 showed that neither more steps nor ~20k real routing rows from 8 extra domains give our
replica Decision-1's general routing skill (F25, F27, F29). R24 tests far broader data:
about 200 customer-facing domains, each with its own intent taxonomy and varied messages.

Domains that match an evaluation set are excluded: retail banking / cards / payments / lending
/ investments (BANKING77, MInDS-14), air travel (ATIS) and legal contracts / compliance (ACE,
ContractNLI).

    domains     10 domains for each of 20 sectors            -> data/synthetic/r24/domains.json
    taxonomies  12-30 intents (name + description) per domain -> .../taxonomies.jsonl
    messages    10 customer messages per intent               -> .../messages.jsonl
    export      one row per message for the corpus builder    -> .../routing.jsonl

Every stage resumes from its output file. Usage:
    python scripts/synth_routing.py all [--concurrency 8]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from pathlib import Path
from typing import Any

from edb.providers.azure_auth import async_openai_client

OUT = Path(__file__).resolve().parents[1] / "data" / "synthetic" / "r24"
MODEL = "gpt-6-luna"
SECTORS = [
    "healthcare providers and clinics",
    "health insurance and pharmacies",
    "telecommunications and internet",
    "utilities: energy, water, waste",
    "e-commerce and online marketplaces",
    "retail stores and grocery",
    "food delivery, restaurants and cafes",
    "hotels, rentals and hospitality",
    "ground transport: rail, bus, taxi, car rental",
    "logistics, parcels and postal services",
    "software, SaaS and IT support",
    "consumer electronics and appliances",
    "automotive: dealers, repair, roadside help",
    "real estate, property management and housing",
    "education: schools, universities, online courses",
    "government and public services",
    "media, streaming, gaming and publishing",
    "fitness, wellness and beauty",
    "home services: cleaning, repair, moving",
    "events, ticketing, travel experiences and pets",
]
EXCLUDED = ("retail banking, bank accounts, debit or credit cards, payments, money transfers, "
            "top-ups, loans, mortgages, investments, crypto, airlines, flights, airports, legal "
            "contracts, regulatory compliance")


def schema(name: str, props: dict[str, Any]) -> dict[str, Any]:
    return {"type": "json_schema", "name": name, "strict": True,
            "schema": {"type": "object", "properties": props, "required": list(props),
                       "additionalProperties": False}}


DOMAINS_SCHEMA = schema("domains", {"domains": {"type": "array", "items": {"type": "string"}}})
TAXONOMY_SCHEMA = schema("taxonomy", {"intents": {"type": "array", "items": {
    "type": "object", "additionalProperties": False, "required": ["name", "description"],
    "properties": {"name": {"type": "string"}, "description": {"type": "string"}}}}})
MESSAGES_SCHEMA = schema("messages", {"intents": {"type": "array", "items": {
    "type": "object", "additionalProperties": False, "required": ["name", "messages"],
    "properties": {"name": {"type": "string"},
                   "messages": {"type": "array", "items": {"type": "string"}}}}}})


async def ask(client, sem: asyncio.Semaphore, instructions: str, prompt: str,
              fmt: dict[str, Any]) -> dict[str, Any]:
    async with sem:
        for attempt in range(8):
            try:
                r = await client.responses.create(model=MODEL, instructions=instructions,
                                                  input=prompt, text={"format": fmt},
                                                  max_output_tokens=16000)
                return json.loads(r.output_text)
            except Exception as exc:  # noqa: BLE001 - rate limits, transient errors, bad JSON
                wait = 5 * 2 ** attempt
                print(f"[retry {attempt + 1}] {type(exc).__name__}: {str(exc)[:120]} ({wait}s)",
                      flush=True)
                await asyncio.sleep(wait)
    raise RuntimeError("giving up after 8 attempts")


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:60]


async def domains(client, sem) -> list[str]:
    path = OUT / "domains.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    async def one(sector: str) -> list[str]:
        out = await ask(client, sem, "You design datasets for intent routing in customer service.",
                        f"List 10 distinct, specific businesses or services in the sector "
                        f"'{sector}' that receive customer messages needing routing to the right "
                        f"team (e.g. 'dental clinic chain', 'municipal parking office'). Do not "
                        f"include anything about: {EXCLUDED}.", DOMAINS_SCHEMA)
        return out["domains"][:10]

    lists = await asyncio.gather(*(one(s) for s in SECTORS))
    seen, result = set(), []
    for d in (d for ds in lists for d in ds):
        if slug(d) not in seen:
            seen.add(slug(d))
            result.append(d)
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


async def taxonomies(client, sem, doms: list[str]) -> list[dict[str, Any]]:
    path = OUT / "taxonomies.jsonl"
    done = {r["domain"] for r in read_jsonl(path)}

    async def one(domain: str) -> None:
        out = await ask(client, sem, "You design intent taxonomies for customer-service routing.",
                        f"Domain: {domain}.\nDefine 12 to 30 customer intents for routing incoming "
                        "messages. Each intent: a snake_case name (2-5 words) and a one-sentence "
                        "description of when it applies. Cover the realistic range of requests, "
                        "include a few intents that are easy to confuse with each other, and do "
                        f"not include anything about: {EXCLUDED}.", TAXONOMY_SCHEMA)
        intents = [{"name": slug(i["name"]), "description": i["description"].strip()}
                   for i in out["intents"] if slug(i["name"])]
        names = [i["name"] for i in intents]
        if len(set(names)) != len(names) or not 8 <= len(intents) <= 40:
            dup = len(names) - len(set(names))
            print(f"[skip] {domain}: {len(intents)} intents, duplicates={dup}")
            return
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"domain": domain, "intents": intents}, ensure_ascii=False) + "\n")

    await asyncio.gather(*(one(d) for d in doms if d not in done))
    return read_jsonl(path)


async def messages(client, sem, taxos: list[dict[str, Any]], per_intent: int) -> None:
    path = OUT / "messages.jsonl"
    done = {(r["domain"], r["chunk"]) for r in read_jsonl(path)}

    async def one(domain: str, chunk: int, intents: list[dict[str, str]]) -> None:
        listing = "\n".join(f"- {i['name']}: {i['description']}" for i in intents)
        out = await ask(client, sem, "You write realistic customer messages for training intent "
                        "classifiers.",
                        f"Domain: {domain}.\nFor each intent below, write {per_intent} different "
                        "messages a real customer might send. Vary length (a few words to three "
                        "sentences), tone, formality and detail; include some typos, indirect "
                        "phrasing and messages that mention other topics in passing. Never use the "
                        "intent name or its exact wording. Each message must clearly belong to its "
                        f"intent and not to the others.\n\nIntents:\n{listing}", MESSAGES_SCHEMA)
        valid = {i["name"] for i in intents}
        rows = [{"name": slug(x["name"]),
                 "messages": [m.strip() for m in x["messages"] if m.strip()]}
                for x in out["intents"] if slug(x["name"]) in valid]
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"domain": domain, "chunk": chunk, "intents": rows},
                               ensure_ascii=False) + "\n")

    jobs = []
    for t in taxos:
        for c in range(0, len(t["intents"]), 8):
            if (t["domain"], c // 8) not in done:
                jobs.append(one(t["domain"], c // 8, t["intents"][c:c + 8]))
    print(f"[messages] {len(jobs)} requests", flush=True)
    await asyncio.gather(*jobs)


def export() -> Path:
    taxos = {t["domain"]: t for t in read_jsonl(OUT / "taxonomies.jsonl")}
    rows, seen = [], set()
    for r in read_jsonl(OUT / "messages.jsonl"):
        desc = {i["name"]: i["description"] for i in taxos[r["domain"]]["intents"]}
        for x in r["intents"]:
            for m in x["messages"]:
                key = (r["domain"], m.lower())
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"domain": r["domain"], "intent": x["name"],
                             "description": desc[x["name"]], "text": m})
    random.Random(24).shuffle(rows)
    path = OUT / "routing.jsonl"
    path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n",
                    encoding="utf-8")
    print(f"{len(rows)} messages, {len(taxos)} domains, "
          f"{sum(len(t['intents']) for t in taxos.values())} intents -> {path}")
    return path


async def main_async(args) -> None:
    client = async_openai_client(max_retries=0, timeout=300)
    sem = asyncio.Semaphore(args.concurrency)
    try:
        doms = await domains(client, sem)
        print(f"{len(doms)} domains", flush=True)
        taxos = await taxonomies(client, sem, doms)
        print(f"{len(taxos)} taxonomies", flush=True)
        await messages(client, sem, taxos, args.per_intent)
    finally:
        await client.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("cmd", choices=["all", "export"])
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--per-intent", type=int, default=10)
    args = parser.parse_args()
    if args.cmd == "all":
        asyncio.run(main_async(args))
    export()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
