"""Fetch and parse raw dataset files (url/local; json/jsonl/csv)."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import httpx

from edb.config import DatasetConfig, FileSource
from edb.settings import REPO_ROOT, get_settings


def raw_dir(dataset: str) -> Path:
    return get_settings().data_dir / "raw" / dataset


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_file(dataset: str, split: str, source: FileSource, refresh: bool = False) -> Path:
    """Return a local path for a source split, downloading it into data/raw if needed."""
    if source.path:
        p = Path(source.path)
        return p if p.is_absolute() else REPO_ROOT / p
    target = raw_dir(dataset) / f"{split}.{source.format}"
    if target.exists() and not refresh:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    with httpx.stream("GET", source.url, follow_redirects=True, timeout=120) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    tmp.replace(target)
    return target


def _dig(obj: Any, dotted: str | None) -> Any:
    for part in (dotted or "").split("."):
        if part:
            obj = obj[part]
    return obj


def read_records(path: Path, source: FileSource) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if source.format == "json":
        records = _dig(json.loads(text), source.records_path)
    elif source.format == "jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        records = list(csv.DictReader(io.StringIO(text)))
    if not isinstance(records, list):
        raise TypeError(f"{path}: expected a list of records (check `records_path`)")
    return records


def load_raw(
    cfg: DatasetConfig, refresh: bool = False
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    """Load raw records for every configured source split. Returns (records, sha256 by split)."""
    records, hashes = {}, {}
    for split, source in cfg.files.items():
        path = fetch_file(cfg.name, split, source, refresh=refresh)
        hashes[split] = sha256_file(path)
        records[split] = read_records(path, source)
    return records, hashes
