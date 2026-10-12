"""Download raw data and build frozen splits for a dataset.

Usage:
    uv run python scripts/prepare_data.py --dataset ace
    uv run python scripts/prepare_data.py --dataset banking77
    uv run python scripts/prepare_data.py --dataset ace --resplit   # re-freeze (rarely!)
"""

from __future__ import annotations

import argparse
import json

from edb.datasets.store import prepare_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare dataset splits")
    parser.add_argument("--dataset", required=True, help="configs/datasets/<name>.yaml")
    parser.add_argument("--resplit", action="store_true", help="recompute frozen split assignment")
    parser.add_argument("--refresh", action="store_true", help="re-download raw source files")
    args = parser.parse_args()
    manifest = prepare_dataset(args.dataset, resplit=args.resplit, refresh=args.refresh)
    print(json.dumps({k: manifest[k] for k in ("dataset", "splits", "group_overlap")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
