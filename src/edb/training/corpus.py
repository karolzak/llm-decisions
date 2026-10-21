"""Multi-task decision corpus from public classification datasets.

Rows are fetched from the Hugging Face dataset viewer API (`/rows`, JSON; no `datasets` or
pyarrow dependency), cached under CACHE_DIR/training/corpus/, and converted to decisions:

    {"source", "state", "instructions", "options": [{"value", "description"}], "gold"}

Options are stored in canonical (label-index) order; whether training shuffles them is a
property of the training arm. Sources with many labels get a random subset of
`max_options` options per row (always including the gold one), so the model sees option
sets that differ from row to row, as a decision API does.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

from edb.config import _load_yaml, _resolve
from edb.settings import get_settings

ROWS_URL = "https://datasets-server.huggingface.co/rows"
PAGE = 100
PAGE_PAUSE_S = 0.5


class SourceSpec(BaseModel):
    name: str
    dataset: str
    config: str = "default"
    split: str = "train"
    state: str  # str.format template over the row's fields
    label: str  # label field (ClassLabel index, bool or string)
    instructions: list[str]  # templates; one is drawn per row (may use row fields)
    # label (as named by the dataset) -> option value shown to the model
    rename: dict[str, str] = Field(default_factory=dict)
    descriptions: dict[str, str] = Field(default_factory=dict)  # option value -> criterion
    exclude_labels: list[str] = Field(default_factory=list)
    rows: int | None = None  # overrides CorpusSpec.rows_per_source
    note: str | None = None


class CorpusSpec(BaseModel):
    name: str
    description: str = ""
    rows_per_source: int = 3000
    max_options: int = 10
    min_options: int = 2
    # Also draw a random option subset (min_options..all) when a source has <= max_options
    # labels, so option-set sizes vary for every source.
    subsample: bool = False
    max_state_chars: int = 3000
    holdout_fraction: float = 0.05
    seed: int = 0
    sources: list[SourceSpec]


def load_corpus_spec(name: str) -> CorpusSpec:
    return CorpusSpec.model_validate(_load_yaml(_resolve("training", name)))


def corpus_dir() -> Path:
    return get_settings().cache_dir / "training" / "corpus"


def _get(params: dict[str, Any], attempts: int = 14) -> dict[str, Any]:
    """GET with patient retries: parallel jobs share one egress IP, and the anonymous dataset
    viewer API rate-limits it (429) or times out under load."""
    headers = {}
    token = get_settings().hf_token
    if token:
        headers["Authorization"] = "Bearer " + token
    last = None
    for attempt in range(attempts):
        wait = min(180.0, 5.0 * 2 ** attempt)
        try:
            r = httpx.get(ROWS_URL, params=params, headers=headers, timeout=120)
            if r.status_code == 200:
                time.sleep(PAGE_PAUSE_S)
                return r.json()
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            if r.status_code not in (408, 425, 429, 500, 502, 503, 504):
                r.raise_for_status()
            retry_after = r.headers.get("retry-after")
            if retry_after and retry_after.isdigit():
                wait = max(wait, float(retry_after))
        except httpx.TransportError as exc:
            last = f"{type(exc).__name__}: {exc}"
        wait *= random.uniform(0.75, 1.25)
        print(f"[corpus] {params.get('dataset')} offset={params.get('offset')}: {last}; "
              f"retry {attempt + 1}/{attempts - 1} in {wait:.0f}s", flush=True)
        time.sleep(wait)
    raise RuntimeError(f"dataset viewer request failed ({last}): {params}")


def fetch_rows(src: SourceSpec, n: int, seed: int) -> tuple[list[dict[str, Any]], dict]:
    """~n rows from random pages of the split (cached), plus the feature schema."""
    path = corpus_dir() / "raw" / f"{src.name}-{n}-{seed}.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return data["rows"], data["features"]
    base = {"dataset": src.dataset, "config": src.config, "split": src.split}
    first = _get({**base, "offset": 0, "length": 1})
    total = first["num_rows_total"]
    pages = list(range(0, total, PAGE))
    rng = random.Random(f"{seed}:{src.name}")
    rng.shuffle(pages)
    rows: list[dict[str, Any]] = []
    for offset in pages:
        if len(rows) >= n:
            break
        page = _get({**base, "offset": offset, "length": PAGE})
        rows += [r["row"] for r in page["rows"]]
    features = {f["name"]: f["type"] for f in first["features"]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"rows": rows[:n], "features": features}), encoding="utf-8")
    return rows[:n], features


def label_key(raw: Any) -> str:
    """Label as text: booleans lowercased (`true`/`false`), everything else kept as is."""
    return str(raw).lower() if isinstance(raw, bool) else str(raw)


def label_names(src: SourceSpec, features: dict[str, Any], rows: list[dict]) -> list[str]:
    ftype = features.get(src.label, {})
    if ftype.get("_type") == "ClassLabel":
        return list(ftype["names"])
    return sorted({label_key(r[src.label]) for r in rows})


def to_decisions(src: SourceSpec, spec: CorpusSpec, rows: list[dict],
                 features: dict[str, Any]) -> list[dict[str, Any]]:
    names = label_names(src, features, rows)
    values = [src.rename.get(n, n) for n in names]
    allowed = [i for i, n in enumerate(names) if n not in src.exclude_labels]
    class_label = features.get(src.label, {}).get("_type") == "ClassLabel"
    rng = random.Random(f"{spec.seed}:{src.name}:rows")
    out = []
    for row in rows:
        raw = row[src.label]
        idx = raw if class_label else names.index(label_key(raw))
        if idx < 0 or idx not in allowed:
            continue  # unlabeled (-1) or excluded
        state = src.state.format(**row)[: spec.max_state_chars]
        k = len(allowed)
        if k > spec.max_options:
            k = rng.randint(spec.min_options, spec.max_options)
        elif spec.subsample:
            k = rng.randint(spec.min_options, k)
        if k < len(allowed):
            chosen = sorted([idx, *rng.sample([i for i in allowed if i != idx], k - 1)])
        else:
            chosen = allowed
        instructions = rng.choice(src.instructions).format(**row)
        out.append({
            "source": src.name,
            "state": state,
            "instructions": instructions,
            "options": [{"value": values[i], "description": src.descriptions.get(values[i])}
                        for i in chosen],
            "gold": values[idx],
        })
    return out


def build_corpus(name: str = "corpus", force: bool = False) -> dict[str, Path]:
    """Build (or reuse) train/holdout JSONL files; returns their paths."""
    spec = load_corpus_spec(name)
    out = corpus_dir() / spec.name
    paths = {"train": out / "train.jsonl", "holdout": out / "holdout.jsonl"}
    if not force and all(p.exists() for p in paths.values()):
        return paths
    rows: list[dict[str, Any]] = []
    stats = {}
    for src in spec.sources:
        raw, features = fetch_rows(src, src.rows or spec.rows_per_source, spec.seed)
        decisions = to_decisions(src, spec, raw, features)
        stats[src.name] = len(decisions)
        rows += decisions
        print(f"[corpus] {src.name}: {len(decisions)} decisions", flush=True)
    rng = random.Random(spec.seed)
    rng.shuffle(rows)
    n_hold = int(len(rows) * spec.holdout_fraction)
    out.mkdir(parents=True, exist_ok=True)
    for split, part in (("holdout", rows[:n_hold]), ("train", rows[n_hold:])):
        with paths[split].open("w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "stats.json").write_text(json.dumps({"spec": spec.model_dump(), "rows": stats,
                                                "train": len(rows) - n_hold,
                                                "holdout": n_hold}, indent=1), encoding="utf-8")
    return paths


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
