"""Run probes against a target and store one JSON record per request (resumable).

Layout: artifacts/probes/<target>/<probe>.jsonl, plus summary.json after analysis.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tqdm.asyncio import tqdm

from edb.probing import forensics
from edb.probing.api import DecisionClient
from edb.probing.clients import ProbeTarget
from edb.probing.items import DEFAULT_SOURCES, BaseItem, load_items
from edb.probing.probes import PROBES, Probe, ProbeCase
from edb.providers.base import RateLimiter
from edb.providers.retry import backoff_delay, classify
from edb.settings import get_settings


def probe_dir(target: str) -> Path:
    return get_settings().artifacts_dir / "probes" / target


def records_path(target: str, probe: str) -> Path:
    return probe_dir(target) / f"{probe}.jsonl"


def read_records(path: Path) -> dict[str, dict[str, Any]]:
    """Latest record per case id (a truncated last line from a crash is skipped)."""
    if not path.exists():
        return {}
    out = {}
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    for i, line in enumerate(lines):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                break
            raise
        out[rec["case_id"]] = rec
    return out


def load_base_items(n: int, sources: tuple[str, ...] = DEFAULT_SOURCES, split: str = "dev",
                    seed: int = 0) -> dict[str, list[BaseItem]]:
    return {s: load_items(s, n, split=split, seed=seed) for s in sources}


def build_cases(probe: Probe, items: dict[str, list[BaseItem]]) -> list[ProbeCase]:
    cases = probe.cases(items)
    ids = [c.case_id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"probe {probe.name!r} produced duplicate case ids")
    return cases


async def _one(client: DecisionClient, case: ProbeCase, target: ProbeTarget,
               limiter: RateLimiter | None) -> dict[str, Any]:
    attempt, response, error = 0, None, None
    while True:
        attempt += 1
        if limiter:
            await limiter.acquire()
        try:
            response = await asyncio.wait_for(client.decide(case.request), target.timeout_s)
            break
        except Exception as exc:  # noqa: BLE001 - classified below
            retryable, suggested = classify(exc)
            if not retryable or attempt > target.max_retries:
                error = f"{type(exc).__name__}: {exc}"
                break
            await asyncio.sleep(backoff_delay(attempt, suggested))
    return {
        "target": target.name,
        "probe": case.probe,
        "case_id": case.case_id,
        "item_id": case.item_id,
        "variant": case.variant,
        "meta": case.meta,
        "request": case.request.to_json(),
        "response": response.to_json() if response else None,
        "error": error,
        "attempts": attempt,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


async def run_cases(client: DecisionClient, target: ProbeTarget, probe: str,
                    cases: list[ProbeCase], retry_errors: bool = False,
                    concurrency: int | None = None) -> dict[str, Any]:
    path = records_path(target.name, probe)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = read_records(path)
    todo = [c for c in cases
            if c.case_id not in done or (retry_errors and done[c.case_id]["error"])]
    limiter = RateLimiter(target.max_rpm) if target.max_rpm else None
    sem = asyncio.Semaphore(concurrency or target.concurrency)
    t0 = time.perf_counter()
    if todo:
        with path.open("a", encoding="utf-8") as sink:
            async def one(case: ProbeCase) -> None:
                async with sem:
                    rec = await _one(client, case, target, limiter)
                sink.write(json.dumps(rec) + "\n")
                sink.flush()

            await tqdm.gather(*(one(c) for c in todo), desc=f"{target.name}/{probe}")
    final = read_records(path)
    wanted = [final[c.case_id] for c in cases if c.case_id in final]
    return {"probe": probe, "cases": len(cases), "submitted": len(todo),
            "errors": sum(r["error"] is not None for r in wanted),
            "wall_s": round(time.perf_counter() - t0, 1)}


def analyze_target(target: str, tokenizers: dict[str, Any] | None = None) -> dict[str, Any]:
    """Analyse every stored probe of `target`; writes and returns summary.json."""
    summary: dict[str, Any] = {"target": target, "probes": {}}
    every: list[dict[str, Any]] = []
    for name, cls in PROBES.items():
        recs = list(read_records(records_path(target, name)).values())
        if not recs:
            continue
        every += recs
        probe = cls()
        result = probe.analyze(recs, tokenizers) if name == "tokens" else probe.analyze(recs)
        summary["probes"][name] = {"records": len(recs),
                                   "errors": sum(r["error"] is not None for r in recs),
                                   "result": result}
    summary["forensics"] = forensics.analyze(every)
    out = probe_dir(target) / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def load_tokenizers(names: list[str]) -> dict[str, Any]:
    """Tokenizers for the token-accounting fit; unavailable ones are skipped."""
    out = {}
    for name in names:
        try:
            from transformers import AutoTokenizer

            out[name] = AutoTokenizer.from_pretrained(name)
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] tokenizer {name}: {type(exc).__name__}: {exc}")
    return out
