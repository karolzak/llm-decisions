"""Preflight checks for the benchmark environment.

Verifies that credentials, Azure deployments, optional API features, the
dataset source and (optionally) the local GPU stack are all usable before any
experiment is launched. Results are written to `artifacts/preflight/`.

Usage:
    uv run python scripts/run_preflight.py            # all checks
    uv run python scripts/run_preflight.py --skip-gpu # API-only (e.g. on a laptop)
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx
from rich.console import Console
from rich.table import Table

from edb.providers.azure_auth import foundry_http_client, get_credential, openai_client
from edb.settings import get_settings

ACE_TEST_URL = (
    "https://raw.githubusercontent.com/FujitsuResearch/"
    "Fujitsu-Assessing-Compliance-in-Enterprise-Dataset/main/test.json"
)

PROBE_STATE = (
    "Clause 4.2: The Supplier shall deliver monthly reports within 10 business days of month end.\n"
    "Scenario: The Supplier delivered the March report 25 business days after month end."
)
PROBE_LABELS = {
    "Compliant": "The scenario adheres to the governing clauses.",
    "Non-Compliant": "The scenario violates at least one governing clause.",
    "Not-Applicable": "The clauses do not govern the scenario.",
}

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
        status, detail, data = "FAIL", f"{type(exc).__name__}: {exc}", {
            "traceback": traceback.format_exc(limit=3)
        }
    return CheckResult(name, status, round((time.perf_counter() - start) * 1000, 1), detail, data)


def check_token() -> tuple[str, str, dict[str, Any]]:
    s = get_settings()
    token = get_credential().get_token(s.azure_token_scope)
    expires = datetime.fromtimestamp(token.expires_on, UTC).isoformat()
    return "PASS", f"token acquired for {s.azure_token_scope}", {"expires_on": expires}


def check_gpt6_luna() -> tuple[str, str, dict[str, Any]]:
    s = get_settings()
    resp = openai_client(s).responses.create(
        model=s.gpt6_luna_deployment,
        input=f"{PROBE_STATE}\n\nAnswer with exactly one label: Compliant, Non-Compliant, "
        "or Not-Applicable.",
        max_output_tokens=256,
    )
    usage = resp.usage.model_dump() if resp.usage else {}
    return "PASS", f"output={resp.output_text!r}", {"model": resp.model, "usage": usage}


def check_gpt6_luna_logprobs() -> tuple[str, str, dict[str, Any]]:
    """A3 feasibility: are token logprobs exposed for gpt-6-luna?"""
    s = get_settings()
    try:
        resp = openai_client(s).responses.create(
            model=s.gpt6_luna_deployment,
            input=f"{PROBE_STATE}\n\nAnswer with exactly one label: Compliant, Non-Compliant, "
            "or Not-Applicable.",
            max_output_tokens=256,
            top_logprobs=5,
            include=["message.output_text.logprobs"],
        )
    except Exception as exc:  # noqa: BLE001
        return "INFO", f"logprobs not available: {type(exc).__name__}: {exc}", {}
    logprobs = []
    for item in resp.output:
        for part in getattr(item, "content", None) or []:
            logprobs.extend(getattr(part, "logprobs", None) or [])
    if not logprobs:
        return "INFO", "request accepted but no logprobs returned", {}
    first = logprobs[0]
    top = [(t.token, round(t.logprob, 4)) for t in (first.top_logprobs or [])]
    return "PASS", f"{len(logprobs)} tokens with logprobs; first top-k={top}", {}


def check_azure_decisions_api() -> tuple[str, str, dict[str, Any]]:
    """A2 feasibility: does Azure expose OpenAI's `POST /v1/decisions` for gpt-6-luna?"""
    s = get_settings()
    body = {
        "model": s.gpt6_luna_deployment,
        "input": PROBE_STATE,
        "questions": [
            {
                "type": "choice",
                "name": "compliance",
                "instructions": "Classify the scenario's compliance with the clauses.",
                "choices": [{"value": k, "description": v} for k, v in PROBE_LABELS.items()],
            }
        ],
    }
    url = f"{s.openai_base_url}decisions"
    with foundry_http_client(s) as http:
        r = http.post(url, json=body)
    if r.status_code != 200:
        return "INFO", f"Decisions API not available on Azure: HTTP {r.status_code}: " \
            f"{r.text[:300]}", {"url": url}
    return "PASS", "Decisions API responded", {"url": url, "response": r.json()}


def check_decision1() -> tuple[str, str, dict[str, Any]]:
    s = get_settings()
    body = {
        "model": s.decision1_deployment,
        "state": PROBE_STATE,
        "questions": {
            "compliance": {
                "type": "choice",
                "instructions": "Does the scenario comply with the contract clauses?",
                "criteria": PROBE_LABELS,
            }
        },
    }
    with foundry_http_client(s) as http:
        r = http.post(s.decision1_url, json=body)
    if r.status_code != 200:
        return "FAIL", f"HTTP {r.status_code}: {r.text[:500]}", {"url": s.decision1_url}
    payload = r.json()
    answer = (payload.get("answers") or {}).get("compliance", {})
    return "PASS", f"choice={answer.get('choice')!r}", {"response": payload}


def check_dataset_source() -> tuple[str, str, dict[str, Any]]:
    r = httpx.head(ACE_TEST_URL, follow_redirects=True, timeout=30)
    r.raise_for_status()
    return "PASS", f"ACE reachable ({r.headers.get('content-length', '?')} bytes)", {}


def check_gpu() -> tuple[str, str, dict[str, Any]]:
    try:
        import torch
    except ImportError:
        return "SKIP", "torch not installed (run `uv sync --extra local` on the GPU host)", {}
    if not torch.cuda.is_available():
        return "FAIL", f"torch {torch.__version__} installed but CUDA unavailable", {}
    devices = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        devices.append(
            {
                "name": p.name,
                "vram_gb": round(p.total_memory / 1024**3, 1),
                "compute_capability": f"{p.major}.{p.minor}",
            }
        )
    return "PASS", f"torch {torch.__version__}, CUDA {torch.version.cuda}: {devices}", {
        "devices": devices
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-api", action="store_true", help="skip Azure API checks")
    parser.add_argument("--skip-gpu", action="store_true", help="skip local GPU checks")
    args = parser.parse_args()

    s = get_settings()
    console.print(f"[bold]Foundry resource:[/] {s.azure_ai_resource}")
    console.print(f"[bold]OpenAI base URL:[/] {s.openai_base_url}")
    console.print(f"[bold]Decision-1 URL:[/]  {s.decision1_url}\n")

    checks: list[tuple[str, Callable]] = [("dataset_source", check_dataset_source)]
    if not args.skip_api:
        checks += [
            ("entra_token", check_token),
            ("gpt6_luna_responses", check_gpt6_luna),
            ("gpt6_luna_logprobs[A3]", check_gpt6_luna_logprobs),
            ("azure_decisions_api[A2]", check_azure_decisions_api),
            ("decision1_systemone", check_decision1),
        ]
    if not args.skip_gpu:
        checks.append(("local_gpu", check_gpu))

    results = [run_check(name, fn) for name, fn in checks]

    table = Table(title="Preflight")
    for col in ("check", "status", "ms", "detail"):
        table.add_column(col, overflow="fold")
    colors = {"PASS": "green", "FAIL": "red", "SKIP": "yellow", "INFO": "cyan"}
    for r in results:
        table.add_row(r.name, f"[{colors[r.status]}]{r.status}[/]", str(r.latency_ms), r.detail)
    console.print(table)

    out_dir = s.artifacts_dir / "preflight"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"preflight_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    out.write_text(json.dumps([asdict(r) for r in results], indent=2, default=str))
    console.print(f"Saved {out}")
    return 1 if any(r.status == "FAIL" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
