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
    # latest submission per (output group, unit)
    entries = {f"{e['dataset']}__{e['run_id']}": e for e in read_log(args.tag)}
    if not entries:
        print(f"no jobs with tag {args.tag!r}")
        return 1
    ml = ml_client()
    root = get_settings().artifacts_dir
    out_dir = root / "aml" / "summaries" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    for run_id, e in sorted(entries.items()):
        job = ml.jobs.get(e["name"])
        tags = dict(job.tags or {})
        summary = assemble(tags)
        record = {"run_id": run_id, "job": e["name"], "status": status_of(job),
                  "exit": tags.get("edb_exit"), "git_sha": e.get("git_sha"),
                  "tail": tags.get("edb_tail"), "summary": summary}
        (out_dir / f"{run_id}.json").write_text(json.dumps(record, indent=1, ensure_ascii=False),
                                                encoding="utf-8")
        installed = []
        for target, s in ((summary or {}).get("probes") or {}).items():
            d = root / "probes" / target
            if d.exists() and any(d.glob("*.jsonl")):
                continue  # raw local records win
            d.mkdir(parents=True, exist_ok=True)
            (d / "summary.json").write_text(json.dumps(forensics.refresh(s), indent=1,
                                                       ensure_ascii=False), encoding="utf-8")
            installed.append(target)
        print(f"{run_id:42s} {record['status']:10s} exit={record['exit']} "
              f"summary={'yes' if summary else 'no'} probes={installed}")
        if args.tail and record["tail"]:
            print("    | " + "\n    | ".join(record["tail"].splitlines()[-15:]))
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
