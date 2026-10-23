"""Submit benchmark runs to the Azure ML GPU cluster: one job per experiment x catalog unit.

Default (--sequential): submit one job, wait until it reaches a terminal state, then submit the
next. Only one node is requested at a time, and a warm node (within the cluster's idle timeout)
picks up the next job. Units that already completed under the same tag are skipped, so
re-running the command resumes the queue. A failed unit is reported and the queue continues.

--parallel submits all units at once (one node each).

Usage:
    uv run --extra aml python scripts/aml_submit.py --experiment c2_oss_sequence_scoring \
        c3_oss_kv_scoring --catalog short --split dev --limit 20 --tag pilot
    uv run --extra aml python scripts/aml_submit.py --experiment c1_oss_autoregressive \
        --split calibration test --allow-test --tag main
    uv run --extra aml python scripts/aml_submit.py ... --dry-run

Outputs go to workspaceblobstore:edb/<tag>/<dataset>/<experiment>__<catalog>/ (fixed per unit,
so resubmitting resumes). Collect with scripts/aml_collect.py.
"""

from __future__ import annotations

import argparse

from edb.runners.aml import (
    ensure_environment,
    job_key,
    ml_client,
    read_log,
    status_of,
    submit,
    wait_for,
)
from edb.runners.predict import resolve_runs


def main() -> int:
    parser = argparse.ArgumentParser(description="Submit AML jobs")
    parser.add_argument("--experiment", nargs="+", required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--catalog", nargs="+")
    parser.add_argument("--split", nargs="+", default=["calibration", "test"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--tag", default="main", help="output namespace (e.g. pilot, main)")
    parser.add_argument("--compute", help="AML cluster (default: AML_COMPUTE setting)")
    parser.add_argument("--git-ref", help="run this pushed commit/branch from GitHub instead of "
                                          "uploading the working tree")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--sequential", action="store_true", default=True,
                      help="one job at a time (default)")
    mode.add_argument("--parallel", action="store_true", help="submit all jobs at once")
    parser.add_argument("--poll", type=float, default=30.0, help="status poll interval (s)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    extra: list[str] = []
    if args.limit:
        extra += ["--limit", str(args.limit)]
    if args.allow_test:
        extra.append("--allow-test")
    if args.retry_errors:
        extra.append("--retry-errors")

    units: list[tuple[str, str]] = []
    dataset = None
    for name in args.experiment:
        for spec in resolve_runs(name, args.dataset, args.catalog):
            sealed = set(spec.dataset.splits.sealed) & set(args.split)
            if sealed and not args.allow_test:
                parser.error(f"splits {sorted(sealed)} are sealed; pass --allow-test")
            if dataset not in (None, spec.dataset.name):
                parser.error("all experiments in one submission must use the same dataset")
            dataset = spec.dataset.name
            units.append((name, spec.catalog_role))
    if not units:
        parser.error("nothing to run")

    mode_name = "parallel" if args.parallel else "sequential"
    for exp, cat in units:
        print(f"[plan] {job_key(exp, cat):40s} splits={args.split} {' '.join(extra)} "
              f"({mode_name})")
    if args.dry_run:
        return 0

    ml = ml_client()
    environment = ensure_environment(ml)
    print(f"environment: {environment}", flush=True)

    # Resume: skip units whose latest job under this tag (same dataset/splits) completed.
    latest = {}
    for e in read_log(args.tag):
        if e.get("dataset") == dataset:
            latest[e["run_id"]] = e
    results = {}
    for exp, cat in units:
        key = job_key(exp, cat)
        prev = latest.get(key)
        if prev and prev.get("splits") == args.split and \
                status_of(ml.jobs.get(prev["name"])) == "Completed":
            print(f"[skip] {key}: already completed ({prev['name']})", flush=True)
            results[key] = "Completed"
            continue
        name = submit(ml, environment=environment, dataset=dataset, experiment=exp,
                      catalog=cat, splits=args.split, tag=args.tag, extra_args=extra,
                      compute=args.compute, git_ref=args.git_ref)
        print(f"[submitted] {key} -> {name} ({args.compute or 'default compute'})", flush=True)
        if not args.parallel:
            results[key] = wait_for(ml, name, args.poll,
                                    log=lambda msg: print(msg, flush=True))
    if not args.parallel:
        print("[queue] done: " + ", ".join(f"{k}={v}" for k, v in results.items()), flush=True)
        return 0 if all(v == "Completed" for v in results.values()) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
