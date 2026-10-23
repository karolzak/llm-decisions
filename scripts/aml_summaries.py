"""Read job summaries from Azure ML job tags (no workspace-storage access needed).

Every job attaches its summary, exit code and log tail as MLflow tags (infra/aml/run_job.sh).
This script reassembles them into artifacts/aml/summaries/<tag>/<run_id>.json and installs
probe summaries as artifacts/probes/<target>/summary.json (unless the target's raw probe
records exist locally), so scripts/probe_report.py works on them.

Usage:
    uv run --extra aml python scripts/aml_summaries.py --tag replication
    uv run --extra aml python scripts/aml_summaries.py --tag replication --tail
"""

from __future__ import annotations

import argparse
import json

from edb.probing import forensics
from edb.runners.aml import ml_client, read_log, status_of
from edb.settings import get_settings


def assemble(tags: dict[str, str]) -> dict | None:
    n = int(tags.get("edb_summary_chunks", 0))
    if not n:
        return None
    text = "".join(tags.get(f"edb_summary_{i:03d}", "") for i in range(n))
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"_unparsed": text[:200], "_truncated": tags.get("edb_summary_truncated")}


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect job summaries from AML tags")
    parser.add_argument("--tag", default="replication")
    parser.add_argument("--tail", action="store_true", help="print each job's log tail")
    args = parser.parse_args()
    # submissions per (output group, unit), newest last
    entries: dict[str, list[dict]] = {}
    for e in read_log(args.tag):
        entries.setdefault(f"{e['dataset']}__{e['run_id']}", []).append(e)
    if not entries:
        print(f"no jobs with tag {args.tag!r}")
        return 1
    ml = ml_client()
    root = get_settings().artifacts_dir
    out_dir = root / "aml" / "summaries" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    merged: dict[str, dict] = {}  # probe target -> summary merged over this tag's jobs
    for run_id, subs in sorted(entries.items()):
        # latest submission, unless it was cancelled (e.g. a duplicate) and an earlier one exists
        for e in reversed(subs):
            job = ml.jobs.get(e["name"])
            if status_of(job) != "Canceled":
                break
        tags = dict(job.tags or {})
        summary = assemble(tags)
        record = {"run_id": run_id, "job": e["name"], "status": status_of(job),
                  "exit": tags.get("edb_exit"), "git_sha": e.get("git_sha"),
                  "tail": tags.get("edb_tail"), "summary": summary}
        (out_dir / f"{run_id}.json").write_text(json.dumps(record, indent=1, ensure_ascii=False),
                                                encoding="utf-8")
        targets = []
        for target, s in ((summary or {}).get("probes") or {}).items():
            merge_probe_summary(merged, target, s)
            targets.append(target)
        print(f"{run_id:42s} {record['status']:10s} exit={record['exit']} "
              f"summary={'yes' if summary else 'no'} probes={targets}")
        if args.tail and record["tail"]:
            print("    | " + "\n    | ".join(record["tail"].splitlines()[-15:]))
    for target, s in merged.items():
        d = root / "probes" / target
        if d.exists() and any(d.glob("*.jsonl")):
            continue  # raw local records win
        d.mkdir(parents=True, exist_ok=True)
        (d / "summary.json").write_text(json.dumps(forensics.refresh(s), indent=1,
                                                   ensure_ascii=False), encoding="utf-8")
    print(f"-> {out_dir}")
    return 0


def merge_probe_summary(merged: dict[str, dict], target: str, s: dict) -> None:
    """Several jobs may probe one target with different probes: keep every probe (a later job
    wins for the same probe) and the forensics computed over the most answers."""
    if target not in merged:
        merged[target] = json.loads(json.dumps(s))
        return
    m = merged[target]
    m.setdefault("probes", {}).update(s.get("probes") or {})
    if (s.get("forensics") or {}).get("answers", 0) > (m.get("forensics") or {}).get("answers", 0):
        m["forensics"] = s["forensics"]


if __name__ == "__main__":
    raise SystemExit(main())
