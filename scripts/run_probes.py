"""Probe a decision API (or a local readout) with perturbed decisions.

Usage:
    uv run python scripts/run_probes.py --target decision-1 --dry-run
    uv run python scripts/run_probes.py --target decision-1                    # all probes
    uv run python scripts/run_probes.py --target decision-1 --probe order duplicate --items 8
    uv run python scripts/run_probes.py --target decision-1 --analyze-only
    uv run python scripts/probe_report.py --target decision-1 qwen3.5-9b      # side by side

Targets live in configs/probe_targets/ (decision-1, jev, openai-decisions, qwen3.5-9b, ...).
Base items come from the `dev` split of each source dataset, so probes never touch the
sealed test split. Runs are resumable: re-running only sends missing requests.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

from edb.probing.clients import load_probe_target, make_client
from edb.probing.items import DEFAULT_SOURCES
from edb.probing.probes import PROBES
from edb.probing.runner import (
    analyze_target,
    build_cases,
    load_base_items,
    load_tokenizers,
    run_cases,
)

DEFAULT_TOKENIZERS = ["Qwen/Qwen3.5-9B", "openai/gpt-oss-20b"]


async def run(args: argparse.Namespace) -> list[dict]:
    target = load_probe_target(args.target)
    items = load_base_items(args.items, tuple(args.sources), seed=args.seed)
    plan = [(name, build_cases(PROBES[name](seed=args.seed), items)) for name in args.probe]
    total = sum(len(c) for _, c in plan)
    for name, cases in plan:
        print(f"[plan] {target.name}/{name}: {len(cases)} requests")
    if target.max_rpm:
        print(f"[plan] {total} requests at <= {target.max_rpm:g}/min: "
              f">= {total / target.max_rpm:.0f} min if none are cached")
    if args.dry_run:
        return []
    client = make_client(target)
    summaries = []
    try:
        for name, cases in plan:
            summaries.append(await run_cases(client, target, name, cases,
                                             retry_errors=args.retry_errors,
                                             concurrency=args.concurrency))
    finally:
        await client.aclose()
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description="Run decision-model probes")
    parser.add_argument("--target", required=True, help="configs/probe_targets/<name>.yaml")
    parser.add_argument("--probe", nargs="+", default=list(PROBES), choices=list(PROBES))
    parser.add_argument("--items", type=int, default=16, help="base items per source dataset")
    parser.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print the request plan only")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--tokenizers", nargs="*", default=DEFAULT_TOKENIZERS,
                        help="candidate tokenizers for the token-accounting fit")
    args = parser.parse_args()

    if not args.analyze_only:
        t0 = time.perf_counter()
        summaries = asyncio.run(run(args))
        if args.dry_run:
            return 0
        print(json.dumps(summaries, indent=2))
        print(f"probing took {time.perf_counter() - t0:.0f}s")
    summary = analyze_target(args.target, load_tokenizers(args.tokenizers or []))
    print(json.dumps(summary["forensics"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
