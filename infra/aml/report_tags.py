"""Attach a job's summary, exit code and log tail to its Azure ML run as MLflow tags.

Runs inside an AML job (separate venv with mlflow-skinny + azureml-mlflow; AML sets
MLFLOW_TRACKING_URI and MLFLOW_RUN_ID). Tags are readable through the workspace API
(`MLClient.jobs.get(name).tags`), which does not touch workspace storage:
scripts/aml_summaries.py reassembles them. Never fails the job.

    python report_tags.py --exit-code 0 --summary summary.json --log job.log
"""

from __future__ import annotations

import argparse
import os
import sys

CHUNK = 4000
MAX_CHUNKS = 60


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exit-code", required=True)
    parser.add_argument("--summary")
    parser.add_argument("--log")
    args = parser.parse_args()
    run_id = os.environ.get("MLFLOW_RUN_ID")
    if not run_id:
        print("[report] MLFLOW_RUN_ID not set; skipping tags")
        return 0
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        tags = {"edb_exit": str(args.exit_code)}
        if args.log and os.path.exists(args.log):
            with open(args.log, encoding="utf-8", errors="replace") as f:
                text = f.read()
            # progress bars rewrite one line with \r; keep the final state of each line
            lines = [ln.split("\r")[-1] for ln in text.splitlines()]
            tags["edb_tail"] = "\n".join(lines)[-CHUNK:]
        if args.summary and os.path.exists(args.summary):
            with open(args.summary, encoding="utf-8") as f:
                data = f.read()
            chunks = [data[i:i + CHUNK] for i in range(0, len(data), CHUNK)]
            if len(chunks) > MAX_CHUNKS:
                chunks = chunks[:MAX_CHUNKS]
                tags["edb_summary_truncated"] = "true"
            tags["edb_summary_chunks"] = str(len(chunks))
            for i, c in enumerate(chunks):
                tags[f"edb_summary_{i:03d}"] = c
        for k, v in tags.items():
            client.set_tag(run_id, k, v)
        print(f"[report] set {len(tags)} tags on {run_id}")
    except Exception as exc:  # noqa: BLE001 - reporting must never fail the job
        print(f"[report] failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
