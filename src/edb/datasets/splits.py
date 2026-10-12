"""Deterministic, stratified, group-aware split assignment."""

from __future__ import annotations

import random
from collections.abc import Sequence
from fractions import Fraction

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def folds_for_ratios(ratios: dict[str, float], max_folds: int = 100) -> dict[str, int]:
    """Smallest fold count that expresses all ratios exactly, e.g. 0.6/0.2/0.2 -> 3/1/1 of 5."""
    fracs = {k: Fraction(v).limit_denominator(max_folds) for k, v in ratios.items()}
    for n in range(2, max_folds + 1):
        if all((f * n).denominator == 1 for f in fracs.values()):
            return {k: int(f * n) for k, f in fracs.items()}
    raise ValueError(f"cannot express ratios {ratios} with <= {max_folds} folds")


def assign_splits(
    labels: Sequence[str],
    groups: Sequence[str] | None,
    ratios: dict[str, float],
    seed: int,
    stratify: bool = True,
) -> list[str]:
    """Assign each item to a split name.

    Items with the same group always land in the same split; label proportions are kept
    as close as possible to the overall distribution when `stratify` is set.
    """
    n = len(labels)
    fold_counts = folds_for_ratios(ratios)
    n_folds = sum(fold_counts.values())
    y = np.array(labels if stratify else ["_"] * n)
    g = np.array(groups if groups is not None else [str(i) for i in range(n)])

    fold_of = np.empty(n, dtype=int)
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (_, idx) in enumerate(skf.split(np.zeros(n), y, g)):
        fold_of[idx] = fold

    order = list(range(n_folds))
    random.Random(seed).shuffle(order)
    split_of_fold: dict[int, str] = {}
    cursor = 0
    for name, count in fold_counts.items():
        for fold in order[cursor : cursor + count]:
            split_of_fold[fold] = name
        cursor += count
    return [split_of_fold[f] for f in fold_of]
