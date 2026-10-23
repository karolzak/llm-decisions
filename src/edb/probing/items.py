"""Base items for probes: real decisions drawn from the benchmark datasets.

Probes perturb these items (reorder, rename or duplicate options, change the question, ...)
so their effect on a target can be read against the unperturbed decision.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

from edb.config import load_dataset_config, load_label_catalog
from edb.datasets.store import dataset_labels, load_split, sample_examples
from edb.probing.api import Option

DEFAULT_SOURCES = ("ace", "banking77")


@dataclass(frozen=True)
class BaseItem:
    id: str  # "<dataset>:<example id>"
    source: str
    state: str
    instructions: str
    options: tuple[Option, ...]  # catalog order
    gold: str

    @property
    def values(self) -> list[str]:
        return [o.value for o in self.options]

    def option(self, value: str) -> Option:
        return next(o for o in self.options if o.value == value)


def load_items(source: str, n: int, split: str = "dev", role: str | None = None,
               seed: int = 0) -> list[BaseItem]:
    """`n` examples of `source`, with its `verbose` catalog when it has one."""
    ds = load_dataset_config(source)
    role = role or ("verbose" if "verbose" in ds.label_catalogs else "short")
    catalog = load_label_catalog(ds, role, dataset_labels(source))
    options = tuple(Option(o.value, o.description if o.has_description else None)
                    for o in catalog.labels)
    return [BaseItem(id=f"{source}:{e.id}", source=source, state=e.state,
                     instructions=ds.task.instructions, options=options, gold=e.label)
            for e in sample_examples(load_split(source, split), n, seed=seed)]


def rng_for(*parts: object) -> random.Random:
    return random.Random(":".join(map(str, parts)))


def subset(item: BaseItem, k: int, seed: int = 0, shuffle: bool = True) -> BaseItem:
    """The gold option plus k-1 distractors. Distractors are nested across k (the first k-1
    of one fixed per-item permutation), so a larger subset always contains a smaller one."""
    if k >= len(item.options):
        opts = list(item.options)
    else:
        rng = rng_for("subset", seed, item.id)
        distractors = [o for o in item.options if o.value != item.gold]
        rng.shuffle(distractors)
        opts = [item.option(item.gold), *distractors[: k - 1]]
    if shuffle:
        rng_for("order", seed, item.id, k).shuffle(opts)
    return replace(item, options=tuple(opts))
