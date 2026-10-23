"""Side-by-side markdown report of probe summaries across targets."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from edb.probing.clients import load_probe_target
from edb.probing.probes import PROBES
from edb.probing.runner import probe_dir
from edb.settings import get_settings


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            out.update(flatten(v, f"{prefix} / {k}" if prefix else str(k)))
        return out
    return {prefix: value}


def _fmt(v: Any) -> str:
    if v is None:
        return "–"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v).replace("|", "\\|").replace("\n", " ")


def _table(results: dict[str, Any]) -> list[str]:
    flat = {t: flatten(r) for t, r in results.items()}
    keys: list[str] = []
    for f in flat.values():
        keys += [k for k in f if k not in keys]
    targets = list(results)
    lines = ["| metric | " + " | ".join(targets) + " |",
             "|---|" + "---|" * len(targets)]
    for k in keys:
        lines.append(f"| {k} | " + " | ".join(_fmt(flat[t].get(k)) for t in targets) + " |")
    return lines


def build_probe_report(targets: list[str], out: Path | None = None) -> Path:
    summaries = {}
    for t in targets:
        path = probe_dir(t) / "summary.json"
        if path.exists():
            summaries[t] = json.loads(path.read_text(encoding="utf-8"))
        else:
            print(f"[skip] no summary for {t} (run scripts/run_probes.py --analyze-only)")
    lines = ["# Decision-model probes", "",
             f"Generated {datetime.now(UTC).isoformat(timespec='seconds')}. What each probe "
             "tests and what each readout design predicts: [docs/replication.md]"
             "(../../docs/replication.md).", "", "| target | description |", "|---|---|"]
    for t in summaries:
        try:
            desc = load_probe_target(t).description
        except FileNotFoundError:
            desc = ""
        lines.append(f"| {t} | {desc} |")
    lines.append("")
    lines += ["## Forensics", "", *_table({t: s.get("forensics", {})
                                           for t, s in summaries.items()}), ""]
    for name, cls in PROBES.items():
        results = {t: s["probes"][name]["result"] for t, s in summaries.items()
                   if name in s.get("probes", {})}
        if not results:
            continue
        counts = ", ".join(f"{t}: {s['probes'][name]['records']} requests, "
                           f"{s['probes'][name]['errors']} errors"
                           for t, s in summaries.items() if name in s.get("probes", {}))
        lines += [f"## {name}", "", f"_{cls.summary}_ ({counts})", "", *_table(results), ""]
    out = out or get_settings().artifacts_dir / "probes" / "report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
