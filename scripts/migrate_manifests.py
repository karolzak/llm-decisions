"""Re-stamp run manifests with the current config hash after a hashing-scheme change.

Only re-stamps when (a) the run's experiment/catalog still resolve, and (b) every stored
prediction satisfies the current prediction logic (valid distribution, predicted == argmax,
non-completed responses recorded as errors, verbalized confidence in [0, 1]).
Never touches runs with an active lock.

Usage:
    uv run python scripts/migrate_manifests.py --dataset ace           # dry run
    uv run python scripts/migrate_manifests.py --dataset ace --apply
"""

from __future__ import annotations

import argparse
import json
import math

from edb.config import config_hash
from edb.runners.predict import read_predictions, resolve_runs
from edb.settings import get_settings


def violations(preds) -> list[str]:
    bad = []
    for p in preds.values():
        if not p.ok:
            continue
        probs = p.probabilities or {}
        top = max(probs.values()) if probs else None
        if not probs or any(not math.isfinite(v) or v < 0 for v in probs.values()):
            bad.append(f"{p.example_id}: invalid distribution")
        elif probs.get(p.predicted) != top:
            bad.append(f"{p.example_id}: predicted is not an argmax label")
        if p.extras.get("status") not in (None, "completed"):
            bad.append(f"{p.example_id}: status {p.extras.get('status')} stored as success")
        c = p.extras.get("verbalized_confidence")
        if c is not None and not 0 <= c <= 1:
            bad.append(f"{p.example_id}: confidence {c}")
    return bad


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-stamp run manifests")
    parser.add_argument("--dataset", default="ace")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--exclude", nargs="*", default=[],
                        help="run_id prefixes to skip (e.g. runs still being written)")
    args = parser.parse_args()

    root = get_settings().artifacts_dir / "runs" / args.dataset
    for manifest_path in sorted(root.glob("*/*/manifest.json")):
        split_dir = manifest_path.parent
        if split_dir.name.count(".archived-") or any(
                split_dir.parent.name.startswith(x) for x in args.exclude):
            continue
        if (split_dir / ".lock").exists():
            print(f"[locked]  {split_dir}")
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        exp_id, role = manifest["run_id"].split("__")
        spec = resolve_runs(exp_id, args.dataset, [role])[0]
        snapshot = spec.snapshot()
        new_hash = config_hash(snapshot)
        if manifest["config_hash"] == new_hash:
            print(f"[ok]      {manifest['run_id']}/{split_dir.name}")
            continue
        bad = violations(read_predictions(split_dir / "predictions.jsonl"))
        if bad:
            print(f"[blocked] {manifest['run_id']}/{split_dir.name}: {bad[:3]} ...")
            continue
        print(f"[restamp] {manifest['run_id']}/{split_dir.name}: "
              f"{manifest['config_hash']} -> {new_hash}")
        if args.apply:
            manifest.update(config=snapshot, config_hash=new_hash,
                            previous_config_hashes=[*manifest.get("previous_config_hashes", []),
                                                    manifest["config_hash"]])
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
