"""Prepare, freeze and load dataset splits.

Frozen split assignments are committed to `configs/datasets/splits/<dataset>.json`
(example ids per derived split + source file hashes), so every machine regenerates the
exact same splits from the raw sources. Processed examples are cached as JSONL in
`data/processed/<dataset>/`.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from edb.config import AUTO, DatasetConfig, load_dataset_config
from edb.datasets.render import render_state, template_fields
from edb.datasets.sources import load_raw
from edb.datasets.splits import assign_splits
from edb.schemas import Example
from edb.settings import get_settings


def assignment_path(dataset: str) -> Path:
    return get_settings().configs_dir / "datasets" / "splits" / f"{dataset}.json"


def processed_dir(dataset: str) -> Path:
    return get_settings().data_dir / "processed" / dataset


def build_examples(cfg: DatasetConfig, split: str, records: list[dict[str, Any]]) -> list[Example]:
    keep = cfg.fields.keep or template_fields(cfg.state.template)
    examples = []
    for i, rec in enumerate(records):
        rid = rec.get(cfg.fields.id) if cfg.fields.id else None
        group = rec.get(cfg.fields.group) if cfg.fields.group else None
        examples.append(
            Example(
                id=f"{split}-{rid}" if rid is not None else f"{split}-{i:05d}",
                state=render_state(rec, cfg),
                label=str(rec[cfg.fields.label]).strip(),
                group=_digest(group) if group is not None else None,
                source_split=split,
                source_index=i,
                fields={k: rec.get(k) for k in keep},
            )
        )
    if len({e.id for e in examples}) != len(examples):
        raise ValueError(f"duplicate example ids in split {split!r}")
    return examples


def _digest(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def prepare_dataset(
    name: str, resplit: bool = False, refresh: bool = False
) -> dict[str, Any]:
    cfg = load_dataset_config(name)
    overlap = set(cfg.splits.sealed) & set(cfg.splits.ratios)
    if overlap:
        raise ValueError(f"derived split names collide with sealed splits: {overlap}")

    raw, hashes = load_raw(cfg, refresh=refresh)
    by_source = {split: build_examples(cfg, split, recs) for split, recs in raw.items()}

    observed = sorted({e.label for exs in by_source.values() for e in exs})
    labels = observed if cfg.labels == AUTO else list(cfg.labels)
    unknown = set(observed) - set(labels)
    if unknown:
        raise ValueError(f"labels {sorted(unknown)} not declared in dataset config")

    derived = by_source[cfg.splits.derive_from]
    apath = assignment_path(cfg.name)
    if apath.exists() and not resplit:
        frozen = json.loads(apath.read_text(encoding="utf-8"))
        for split in (cfg.splits.derive_from, *cfg.splits.sealed):
            if frozen["source_sha256"].get(split) != hashes[split]:
                raise RuntimeError(
                    f"source file for split {split!r} changed since splits were frozen "
                    f"({apath}). Investigate, or re-freeze with --resplit."
                )
        split_of = {eid: s for s, ids in frozen["assignments"].items() for eid in ids}
        if set(split_of) != {e.id for e in derived}:
            raise RuntimeError(f"frozen assignment ids do not match source records ({apath})")
    else:
        names = assign_splits(
            labels=[e.label for e in derived],
            groups=[e.group for e in derived] if cfg.fields.group else None,
            ratios=cfg.splits.ratios,
            seed=cfg.splits.seed,
            stratify=cfg.splits.stratify,
        )
        split_of = {e.id: s for e, s in zip(derived, names, strict=True)}
        frozen = {
            "dataset": cfg.name,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "derive_from": cfg.splits.derive_from,
            "ratios": cfg.splits.ratios,
            "seed": cfg.splits.seed,
            "stratify": cfg.splits.stratify,
            "group_field": cfg.fields.group,
            "source_sha256": {s: hashes[s] for s in (cfg.splits.derive_from, *cfg.splits.sealed)},
            "assignments": {
                s: [e.id for e in derived if split_of[e.id] == s] for s in cfg.splits.ratios
            },
        }
        apath.parent.mkdir(parents=True, exist_ok=True)
        apath.write_text(json.dumps(frozen, indent=1) + "\n", encoding="utf-8")

    splits: dict[str, list[Example]] = defaultdict(list)
    for e in derived:
        splits[split_of[e.id]].append(e)
    for sealed in cfg.splits.sealed:
        splits[sealed] = by_source[sealed]

    out = processed_dir(cfg.name)
    out.mkdir(parents=True, exist_ok=True)
    for split, exs in splits.items():
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for e in exs:
                f.write(e.model_dump_json() + "\n")

    manifest = {
        "dataset": cfg.name,
        "prepared_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "labels": labels,
        "source_sha256": hashes,
        "assignment_file": str(apath.relative_to(get_settings().configs_dir.parent)),
        "splits": {
            s: {"n": len(exs), "labels": dict(sorted(Counter(e.label for e in exs).items()))}
            for s, exs in splits.items()
        },
        "group_overlap": _group_overlap(cfg, splits),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _group_overlap(cfg: DatasetConfig, splits: dict[str, list[Example]]) -> dict[str, int]:
    """Number of groups shared between split pairs (derived pairs must be 0)."""
    if not cfg.fields.group:
        return {}
    groups = {s: {e.group for e in exs} for s, exs in splits.items()}
    names = sorted(groups)
    result = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            result[f"{a}|{b}"] = len(groups[a] & groups[b])
    derived = set(cfg.splits.ratios)
    leaks = {k: v for k, v in result.items() if set(k.split("|")) <= derived and v}
    if leaks:
        raise AssertionError(f"group leakage between derived splits: {leaks}")
    return result


def load_manifest(dataset: str) -> dict[str, Any]:
    path = processed_dir(dataset) / "manifest.json"
    if not path.exists():
        prepare_dataset(dataset)
    return json.loads(path.read_text(encoding="utf-8"))


def dataset_labels(dataset: str) -> list[str]:
    return load_manifest(dataset)["labels"]


def load_split(dataset: str, split: str) -> list[Example]:
    path = processed_dir(dataset) / f"{split}.jsonl"
    if not path.exists():
        prepare_dataset(dataset)
    if not path.exists():
        raise FileNotFoundError(f"dataset {dataset!r} has no split {split!r}")
    with path.open(encoding="utf-8") as f:
        return [Example.model_validate_json(line) for line in f if line.strip()]


def sample_examples(examples: list[Example], n: int | None, seed: int = 0) -> list[Example]:
    """Deterministic pseudo-random subset (stable under additions to the pool)."""
    if n is None or n >= len(examples):
        return examples

    def key(e: Example) -> str:
        return hashlib.sha1(f"{seed}:{e.id}".encode()).hexdigest()

    return sorted(examples, key=key)[:n]
