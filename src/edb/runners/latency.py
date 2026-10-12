"""Latency benchmark: warmups, a fixed example subset, repetitions, several concurrencies.

Output: artifacts/latency/<dataset>/<run_id>/{calls.jsonl, summary.json}
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np

from edb.datasets.store import load_split, sample_examples
from edb.providers.registry import build_method
from edb.runners.predict import RunSpec
from edb.settings import get_settings


def _pct(values: list[float], q: float) -> float | None:
    return round(float(np.percentile(values, q)), 1) if values else None


async def benchmark(
    spec: RunSpec,
    split: str = "dev",
    n: int = 200,
    warmup: int = 20,
    reps: int = 5,
    concurrencies: tuple[int, ...] = (1, 8),
    seed: int = 1,
) -> dict[str, Any]:
    pool = load_split(spec.dataset.name, split)
    subset = sample_examples(pool, n, seed=seed)
    chosen = {e.id for e in subset}
    warm = sample_examples([e for e in pool if e.id not in chosen], warmup, seed=seed)

    out = get_settings().artifacts_dir / "latency" / spec.dataset.name / spec.run_id
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "run_id": spec.run_id, "split": split, "n": len(subset), "warmup": len(warm),
        "reps": reps, "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "by_concurrency": {},
    }
    with (out / "calls.jsonl").open("w", encoding="utf-8") as sink:
        for conc in concurrencies:
            method = build_method(spec.experiment.method.type, spec.context(conc))
            sem = asyncio.Semaphore(conc)

            async def one(example, rep, record, _method=method, _sem=sem, _conc=conc):
                async with _sem:
                    pred = await _method.predict(example)
                if record:
                    sink.write(json.dumps({"concurrency": _conc, "rep": rep,
                                           **pred.model_dump(mode="json")}) + "\n")
                return pred

            try:
                await method.setup()
                await asyncio.gather(*(one(e, -1, False) for e in warm))
                preds, walls = [], []
                for rep in range(reps):
                    t0 = time.perf_counter()
                    preds += await asyncio.gather(*(one(e, rep, True) for e in subset))
                    walls.append(time.perf_counter() - t0)
            finally:
                await method.aclose()

            ok = [p for p in preds if p.ok]
            lat = [p.latency_ms for p in ok if p.latency_ms is not None]
            ttft = [p.ttft_ms for p in ok if p.ttft_ms is not None]
            summary["by_concurrency"][str(conc)] = {
                "calls": len(preds),
                "errors": len(preds) - len(ok),
                "retried": sum(p.attempts > 1 for p in preds),
                "p50_ms": _pct(lat, 50),
                "p95_ms": _pct(lat, 95),
                "ttft_p50_ms": _pct(ttft, 50),
                "ttft_p95_ms": _pct(ttft, 95),
                "examples_per_s": round(len(subset) / float(np.mean(walls)), 2),
                "usd_per_1k": (round(1000 * float(np.mean([p.cost_usd for p in preds
                                                           if p.cost_usd is not None])), 4)
                               if any(p.cost_usd is not None for p in preds) else None),
            }
    summary["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
