"""YAML configuration models and resolution.

Configs are referenced by name (resolved under `configs/<kind>/<name>.yaml`) or by path.
Experiments reference datasets/models/catalogs by name, and label catalogs by *role*
(e.g. `short`, `verbose`) which each dataset maps to concrete catalogs, so the same
experiment definition runs unchanged against any dataset.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from edb.schemas import LabelCatalog, LabelOption, TaskSpec
from edb.settings import get_settings

AUTO = "auto"


# --------------------------------------------------------------------------- datasets
class FileSource(BaseModel):
    url: str | None = None
    path: str | None = None  # relative to repo root or absolute
    format: Literal["json", "jsonl", "csv"] = "json"
    # Dotted path to the record list inside a JSON document (e.g. "scenarios").
    records_path: str | None = None

    @model_validator(mode="after")
    def _one_location(self) -> FileSource:
        if bool(self.url) == bool(self.path):
            raise ValueError("exactly one of `url` or `path` must be set")
        return self


class FieldFormatter(BaseModel):
    """How a raw field is rendered into the state template."""

    type: Literal["text", "mapping", "list", "json"] = "text"
    item_template: str = "{key}: {value}"  # mapping: {key},{value}; list: {index},{value}
    joiner: str = "\n"


class StateConfig(BaseModel):
    # `str.format` template over record fields, e.g. "Clauses:\n{clauses}\n\nScenario:\n{text}"
    template: str
    formatters: dict[str, FieldFormatter] = Field(default_factory=dict)


class FieldsConfig(BaseModel):
    label: str
    id: str | None = None  # optional stable id field in the source records
    group: str | None = None  # records sharing this value never straddle derived splits
    keep: list[str] | None = None  # raw fields copied to Example.fields (default: template fields)


class SplitConfig(BaseModel):
    sealed: list[str] = Field(default_factory=lambda: ["test"])  # used as-is, never re-split
    derive_from: str = "train"
    ratios: dict[str, float] = Field(
        default_factory=lambda: {"train": 0.6, "dev": 0.2, "calibration": 0.2}
    )
    seed: int = 0
    stratify: bool = True

    @model_validator(mode="after")
    def _ratios_sum(self) -> SplitConfig:
        if abs(sum(self.ratios.values()) - 1.0) > 1e-6:
            raise ValueError(f"split ratios must sum to 1, got {self.ratios}")
        return self


class DatasetConfig(BaseModel):
    name: str
    description: str = ""
    license: str | None = None
    homepage: str | None = None
    files: dict[str, FileSource]  # source split name -> file
    fields: FieldsConfig
    state: StateConfig
    labels: list[str] | Literal["auto"] = AUTO
    task: TaskSpec
    splits: SplitConfig = Field(default_factory=SplitConfig)
    # role -> catalog name (configs/label_catalogs/<name>.yaml) or "auto" (labels as-is)
    label_catalogs: dict[str, str] = Field(default_factory=lambda: {"short": AUTO})

    @model_validator(mode="after")
    def _check_splits(self) -> DatasetConfig:
        for split in [*self.splits.sealed, self.splits.derive_from]:
            if split not in self.files:
                raise ValueError(f"split {split!r} has no entry under `files`")
        return self

    @property
    def evaluation_splits(self) -> list[str]:
        return [*self.splits.ratios, *self.splits.sealed]


# --------------------------------------------------------------------------- models
class Pricing(BaseModel):
    """USD per `per_tokens` tokens."""

    input: float = 0.0
    cached_input: float | None = None
    cache_write: float | None = None
    output: float = 0.0
    per_tokens: int = 1_000_000
    source: str | None = None


class ModelConfig(BaseModel):
    name: str
    provider: Literal["azure_openai", "azure_foundry", "openai", "local_hf"]
    deployment: str | None = None  # Azure deployment / API model name
    hf_model_id: str | None = None
    pricing: Pricing | None = None
    gpu_hourly_price_usd: float | None = None
    params: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- experiments
class MethodConfig(BaseModel):
    type: str
    model: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class RunConfig(BaseModel):
    concurrency: int = 4
    timeout_s: float = 180.0
    max_retries: int = 8
    # Client-side request pacing (requests/minute across all workers); match the deployment's
    # rate limit to avoid burning retries on 429s.
    max_rpm: float | None = None


class ExperimentConfig(BaseModel):
    id: str
    description: str = ""
    leaderboards: list[str] = Field(default_factory=list)  # e.g. product, controlled
    # zero_shot: no task labels used; supervised: fits on the dataset's train labels.
    supervision: Literal["zero_shot", "supervised"] = "zero_shot"
    dataset: str | None = None  # default dataset; overridable from the CLI
    label_catalogs: list[str] = Field(default_factory=lambda: ["short"])  # roles
    method: MethodConfig
    run: RunConfig = Field(default_factory=RunConfig)
    enabled: bool = True
    notes: str | None = None


# --------------------------------------------------------------------------- loading
def _resolve(kind: str, name_or_path: str | Path) -> Path:
    p = Path(name_or_path)
    if p.suffix in {".yaml", ".yml"} and p.exists():
        return p
    candidate = get_settings().configs_dir / kind / f"{name_or_path}.yaml"
    if not candidate.exists():
        raise FileNotFoundError(f"no {kind} config named {name_or_path!r} ({candidate})")
    return candidate


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_dataset_config(name: str | Path) -> DatasetConfig:
    return DatasetConfig.model_validate(_load_yaml(_resolve("datasets", name)))


def load_model_config(name: str | Path) -> ModelConfig:
    return ModelConfig.model_validate(_load_yaml(_resolve("models", name)))


def load_experiment_config(name: str | Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(_load_yaml(_resolve("experiments", name)))


def list_experiments() -> list[ExperimentConfig]:
    folder = get_settings().configs_dir / "experiments"
    return [load_experiment_config(p) for p in sorted(folder.glob("*.yaml"))]


def load_label_catalog(dataset: DatasetConfig, role: str, labels: list[str]) -> LabelCatalog:
    """Resolve a catalog role for a dataset. `labels` is the dataset's label set."""
    if role not in dataset.label_catalogs:
        raise KeyError(
            f"dataset {dataset.name!r} has no {role!r} label catalog "
            f"(available: {sorted(dataset.label_catalogs)})"
        )
    ref = dataset.label_catalogs[role]
    if ref == AUTO:
        catalog = LabelCatalog(
            name=f"{dataset.name}_{role}_auto", labels=[LabelOption(value=v) for v in labels]
        )
    else:
        catalog = LabelCatalog.model_validate(_load_yaml(_resolve("label_catalogs", ref)))
    if sorted(catalog.values) != sorted(labels):
        raise ValueError(
            f"catalog {catalog.name!r} values {catalog.values} != dataset labels {labels}"
        )
    return catalog


def config_hash(*objs: BaseModel | dict[str, Any]) -> str:
    payload = [o.model_dump(mode="json") if isinstance(o, BaseModel) else o for o in objs]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
