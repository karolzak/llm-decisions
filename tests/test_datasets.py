import json
from collections import Counter

import pytest

from edb.config import FieldFormatter, load_dataset_config
from edb.datasets.render import format_field, render_state
from edb.datasets.splits import assign_splits, folds_for_ratios
from edb.datasets.store import assignment_path, load_split, prepare_dataset, sample_examples


def test_format_mapping_and_list():
    fmt = FieldFormatter(type="mapping", item_template="[{key}] {value}", joiner=" | ")
    assert format_field({"a": "x", "b": "y"}, fmt) == "[a] x | [b] y"
    fmt = FieldFormatter(type="list", item_template="{index}. {value}")
    assert format_field(["x", "y"], fmt) == "1. x\n2. y"


def test_render_state_missing_field(sandbox):
    cfg = load_dataset_config("toy")
    assert render_state({"topic": "t", "text": "hi"}, cfg) == "Topic t: hi"
    with pytest.raises(KeyError):
        render_state({"text": "hi"}, cfg)


def test_folds_for_ratios():
    assert folds_for_ratios({"a": 0.6, "b": 0.2, "c": 0.2}) == {"a": 3, "b": 1, "c": 1}
    assert folds_for_ratios({"a": 0.5, "b": 0.5}) == {"a": 1, "b": 1}
    assert folds_for_ratios({"a": 0.7, "b": 0.15, "c": 0.15}) == {"a": 14, "b": 3, "c": 3}


def test_assign_splits_groups_stratification_determinism():
    labels = [["x", "y", "z"][i % 3] for i in range(600)]
    groups = [f"g{i // 2}" for i in range(600)]
    ratios = {"train": 0.6, "dev": 0.2, "calibration": 0.2}
    a = assign_splits(labels, groups, ratios, seed=3)
    assert a == assign_splits(labels, groups, ratios, seed=3)
    assert a != assign_splits(labels, groups, ratios, seed=4)
    split_of_group = {}
    for g, s in zip(groups, a, strict=True):
        assert split_of_group.setdefault(g, s) == s
    sizes = Counter(a)
    assert abs(sizes["train"] - 360) <= 6 and abs(sizes["dev"] - 120) <= 6


def test_prepare_freeze_and_reuse(sandbox):
    m1 = prepare_dataset("toy")
    assert m1["labels"] == ["alpha", "beta", "gamma"]
    assert sum(m1["splits"][s]["n"] for s in ("train", "dev", "calibration")) == 300
    assert m1["splits"]["test"]["n"] == 30
    for pair, n in m1["group_overlap"].items():
        if "test" not in pair:
            assert n == 0
    frozen = assignment_path("toy").read_text(encoding="utf-8")
    prepare_dataset("toy")
    assert assignment_path("toy").read_text(encoding="utf-8") == frozen

    dev = load_split("toy", "dev")
    assert dev[0].state.startswith("Topic t")
    assert sample_examples(dev, 5, seed=1) == sample_examples(dev, 5, seed=1)


def test_prepare_detects_changed_source(sandbox):
    prepare_dataset("toy")
    src = sandbox / "src" / "train.csv"
    src.write_text(src.read_text(encoding="utf-8") + "extra,t99,alpha\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed since splits were frozen"):
        prepare_dataset("toy")
    m = prepare_dataset("toy", resplit=True)
    assert json.loads(assignment_path("toy").read_text())["source_sha256"]["train"] == \
        m["source_sha256"]["train"]
