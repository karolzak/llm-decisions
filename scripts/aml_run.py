"""Submit any repo script as an Azure ML GPU job (probes on local targets, training arms).

The job runs `python <script> <args>` through infra/aml/run_job.sh; its ARTIFACTS_DIR is
synced to edb/<tag>/<group>/<key>/ (resumable), and scripts/aml_collect.py --tag <tag>
downloads runs/, probes/ and training/ into local artifacts.

Usage:
    uv run --extra aml python scripts/aml_run.py --key probes-qwen3.5-9b --tag replication \
        -- scripts/run_probes.py --target qwen3.5-9b
    uv run --extra aml python scripts/aml_run.py --key r3-ce_shuffle --tag replication \
        -- scripts/train_decider.py --arm ce_shuffle --split calibration test --allow-test
    uv run --extra aml python scripts/aml_collect.py --tag replication --wait
"""

from __future__ import annotations

import argparse

from edb.runners.aml import ensure_environment, ml_client, submit_script, wait_for


def main() -> int:
    parser = argparse.ArgumentParser(description="Submit a script as an AML job")
    parser.add_argument("--key", required=True, help="unit name (fixed output folder)")
    parser.add_argument("--group", default="replication", help="output folder group")
    parser.add_argument("--tag", default="replication")
    parser.add_argument("--compute", help="AML cluster (default: AML_COMPUTE setting)")
    parser.add_argument("--git-ref", help="run this pushed commit/branch from GitHub instead of "
                                          "uploading the working tree")
    parser.add_argument("--wait", action="store_true", help="block until the job finishes")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("script", nargs=argparse.REMAINDER,
                        help="-- <script.py> <args...>")
    args = parser.parse_args()
    script = args.script[1:] if args.script[:1] == ["--"] else args.script
    if not script or not script[0].endswith(".py"):
        parser.error("give the script after `--`, e.g. -- scripts/run_probes.py --target X")
    print(f"[plan] {args.tag}/{args.group}/{args.key}: python {' '.join(script)}")
    if args.dry_run:
        return 0
    ml = ml_client()
    name = submit_script(ml, environment=ensure_environment(ml), group=args.group,
                         key=args.key, args=script, tag=args.tag, compute=args.compute,
                         git_ref=args.git_ref)
    print(f"[submitted] {args.key} -> {name}", flush=True)
    if args.wait:
        status = wait_for(ml, name, log=lambda m: print(m, flush=True))
        return 0 if status == "Completed" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
