"""Preflight checks for the benchmark environment.

Verifies credentials, dataset preparation, every enabled experiment (one real prediction
through the same code path as full runs), API feature probes (A2/A3 feasibility) and the
local GPU stack. Results are written to `artifacts/preflight/`.

Usage:
    uv run python scripts/run_preflight.py                 # everything
    uv run python scripts/run_preflight.py --skip-gpu      # laptop
    uv run python scripts/run_preflight.py --dataset banking77 --include-disabled
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from edb.config import list_experiments, load_model_config
from edb.datasets.store import load_split, prepare_dataset
from edb.providers.azure_auth import foundry_http_client, get_credential, openai_client
from edb.providers.registry import build_method
from edb.runners.predict import resolve_runs
from edb.settings import get_settings

console = Console()


@dataclass
class CheckResult:
    name: str
    status: str  # PASS | FAIL | SKIP | INFO
    latency_ms: float | None = None
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)


def run_check(name: str, fn: Callable[[], tuple[str, str, dict[str, Any]]]) -> CheckResult:
    start = time.perf_counter()
    try:
        status, detail, data = fn()
    except Exception as exc:  # noqa: BLE001 - preflight must report, not crash
        status, detail = "FAIL", f"{type(exc).__name__}: {exc}"
        data = {"traceback": traceback.format_exc(limit=3)}
    return CheckResult(name, status, round((time.perf_counter() - start) * 1000, 1), detail, data)


def check_dataset(name: str):
    def _check():
        manifest = prepare_dataset(name)
        counts = {s: v["n"] for s, v in manifest["splits"].items()}
        return "PASS", f"{len(manifest['labels'])} labels, splits={counts}", {}

    return _check


def check_token():
    s = get_settings()
    token = get_credential().get_token(s.azure_token_scope)
    expires = datetime.fromtimestamp(token.expires_on, UTC).isoformat()
    return "PASS", f"token for {s.azure_token_scope}", {"expires_on": expires}


def check_experiment(experiment_id: str, dataset: str):
    def _check():
        spec = resolve_runs(experiment_id, dataset)[0]
        example = load_split(spec.dataset.name, "dev")[0]

        async def _go():
            method = build_method(spec.experiment.method.type, spec.context(1))
            try:
                await method.setup()
                return await method.predict(example)
            finally:
                await method.aclose()

        pred = asyncio.run(_go())
        if not pred.ok:
            return "FAIL", pred.error or "", {}
        probs = {k: round(v, 3) for k, v in (pred.probabilities or {}).items()}
        detail = (f"[{spec.catalog_role}] pred={pred.predicted} gold={pred.gold} "
                  f"{pred.latency_ms:.0f}ms tokens={pred.usage.input_tokens if pred.usage else '?'}"
                  f"/{pred.usage.output_tokens if pred.usage else '?'} p={probs}")
        return "PASS", detail, {"prediction": pred.model_dump(mode="json")}

    return _check


def probe_azure_decisions():
    """A2 feasibility: does Azure expose OpenAI's `POST /v1/decisions`?"""
    s = get_settings()
    model = load_model_config("gpt-6-luna")
    body = {
        "model": model.deployment,
        "input": "The invoice was paid 40 days late; terms require 30 days.",
        "questions": [{"type": "predicate", "name": "late", "instructions": "Was it late?"}],
    }
    url = f"{s.openai_base_url}decisions"
    with foundry_http_client(s) as http:
        r = http.post(url, json=body)
    if r.status_code != 200:
        return "INFO", f"not available on Azure: HTTP {r.status_code} {r.text[:200]}", {"url": url}
    return "PASS", "Decisions API responded on Azure", {"url": url, "response": r.json()}


def probe_logprobs():
    """A3 feasibility: are output token logprobs exposed for gpt-6-luna?"""
    model = load_model_config("gpt-6-luna")
    try:
        openai_client().responses.create(
            model=model.deployment, input="Say yes.", max_output_tokens=64,
            top_logprobs=5, include=["message.output_text.logprobs"],
        )
    except Exception as exc:  # noqa: BLE001
        return "INFO", f"not available: {type(exc).__name__}: {str(exc)[:200]}", {}
    return "PASS", "logprobs accepted", {}


def check_gpu():
    try:
        import torch
    except ImportError:
        return "SKIP", "torch not installed (`uv sync --extra local` on the GPU host)", {}
    if not torch.cuda.is_available():
        return "FAIL", f"torch {torch.__version__} installed but CUDA unavailable", {}
    devices = [
        {"name": p.name, "vram_gb": round(p.total_memory / 1024**3, 1)}
        for p in (torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count()))
    ]
    return "PASS", f"torch {torch.__version__}, CUDA {torch.version.cuda}: {devices}", {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Environment preflight")
    parser.add_argument("--dataset", default="ace")
    parser.add_argument("--skip-api", action="store_true")
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--include-disabled", action="store_true",
                        help="also smoke-test experiments marked enabled: false")
    args = parser.parse_args()

    s = get_settings()
    if not args.skip_api:
        console.print(f"[bold]Foundry:[/] {s.azure_ai_resource} / {s.azure_ai_project}")
        console.print(f"[bold]OpenAI base URL:[/] {s.openai_base_url}")
        console.print(f"[bold]Decision-1 URL:[/]  {s.decision1_url}\n")

    checks: list[tuple[str, Callable]] = [(f"dataset[{args.dataset}]", check_dataset(args.dataset))]
    if not args.skip_api:
        checks.append(("entra_token", check_token))
        for exp in list_experiments():
            if exp.method.model and load_model_config(exp.method.model).provider == "local_hf":
                continue
            if exp.enabled or args.include_disabled:
                checks.append((f"exp[{exp.id}]", check_experiment(exp.id, args.dataset)))
        checks += [("probe[A2 azure /decisions]", probe_azure_decisions),
                   ("probe[A3 logprobs]", probe_logprobs)]
    if not args.skip_gpu:
        checks.append(("local_gpu", check_gpu))

    results = [run_check(name, fn) for name, fn in checks]

    table = Table(title="Preflight")
    for col in ("check", "status", "ms", "detail"):
        table.add_column(col, overflow="fold")
    colors = {"PASS": "green", "FAIL": "red", "SKIP": "yellow", "INFO": "cyan"}
    for r in results:
        table.add_row(escape(r.name), f"[{colors[r.status]}]{r.status}[/]", str(r.latency_ms),
                      escape(r.detail))
    console.print(table)

    out_dir = s.artifacts_dir / "preflight"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"preflight_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    out.write_text(json.dumps([asdict(r) for r in results], indent=2, default=str))
    console.print(f"Saved {out}")
    return 1 if any(r.status == "FAIL" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
