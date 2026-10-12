"""Validate every shipped config so typos fail in CI rather than mid-run."""

import json

import pytest

from edb.config import (
    list_experiments,
    load_dataset_config,
    load_label_catalog,
    load_model_config,
)
from edb.providers.registry import method_types
from edb.settings import REPO_ROOT

CONFIGS = REPO_ROOT / "configs"
DATASETS = sorted(p.stem for p in (CONFIGS / "datasets").glob("*.yaml"))


@pytest.mark.parametrize("name", DATASETS)
def test_dataset_configs_and_catalogs(name):
    cfg = load_dataset_config(name)
    frozen = CONFIGS / "datasets" / "splits" / f"{name}.json"
    assert frozen.exists(), f"splits for {name} are not frozen (run prepare_data.py)"
    data = json.loads(frozen.read_text(encoding="utf-8"))
    assert set(data["assignments"]) == set(cfg.splits.ratios)
    ids = [i for v in data["assignments"].values() for i in v]
    assert len(ids) == len(set(ids))
    if cfg.labels != "auto":
        for role in cfg.label_catalogs:
            load_label_catalog(cfg, role, list(cfg.labels))


def test_experiments_reference_valid_models_and_methods():
    known = set(method_types())
    experiments = list_experiments()
    assert experiments
    ids = [e.id for e in experiments]
    assert len(ids) == len(set(ids))
    for exp in experiments:
        if exp.method.model:
            load_model_config(exp.method.model)
        if exp.method.model and load_model_config(exp.method.model).provider == "local_hf":
            continue
        assert exp.method.type in known, exp.id
