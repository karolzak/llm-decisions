"""Annotation-artifact diagnostic: can a linear probe predict labels from partial inputs?

Fits a logistic-regression probe on embeddings of (a) each state-template field alone,
(b) all fields, and (c) the full rendered state, scored by group-aware cross-validation on
the train split only (no dev/test data touched). If a single field that should be
insufficient on its own (e.g. the scenario without the clauses) predicts the label far
above chance, the dataset has label-correlated shortcuts.

Usage:
    uv run python scripts/probe_artifacts.py --dataset ace
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from edb.config import load_dataset_config, load_model_config
from edb.datasets.render import format_field, template_fields
from edb.datasets.store import load_split
from edb.embeddings import Embedder
from edb.providers.azure_auth import async_openai_client
from edb.providers.embeddings import combine_views
from edb.settings import get_settings


async def embed_views(embedder: Embedder, texts: dict[str, list[str]]) -> dict[str, np.ndarray]:
    return {name: await embedder.embed_cached(t) for name, t in texts.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe for annotation artifacts")
    parser.add_argument("--dataset", default="ace")
    parser.add_argument("--split", default="train")
    parser.add_argument("--model", default="text-embedding-3-large")
    parser.add_argument("--C", type=float, default=0.01)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    cfg = load_dataset_config(args.dataset)
    model = load_model_config(args.model)
    examples = load_split(args.dataset, args.split)
    fields = template_fields(cfg.state.template)
    texts = {"state": [e.state for e in examples]}
    for f in fields:
        texts[f] = [format_field(e.fields[f], cfg.state.formatters.get(f)) for e in examples]

    async def run():
        client = async_openai_client(get_settings(), max_retries=10)
        try:
            return await embed_views(Embedder(client, model.deployment), texts)
        finally:
            await client.close()

    emb = asyncio.run(run())
    labels = sorted({e.label for e in examples})
    y = np.array([labels.index(e.label) for e in examples])
    groups = np.array([e.group or e.id for e in examples])
    folds = list(GroupKFold(n_splits=args.folds).split(y, y, groups))
    majority = Counter(y).most_common(1)[0][1] / len(y)

    feature_sets = {f"field:{f}": emb[f] for f in fields}
    if len(fields) > 1:
        feature_sets["fields:all"] = combine_views([emb[f] for f in fields])
    feature_sets["state"] = emb["state"]

    results = {"dataset": args.dataset, "split": args.split, "n": len(y),
               "majority_baseline": round(majority, 4), "probes": {}}
    for name, X in feature_sets.items():
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=args.C, max_iter=3000))
        pred = cross_val_predict(clf, X, y, cv=folds)
        per_class = {labels[k]: round(float((pred[y == k] == k).mean()), 3)
                     for k in range(len(labels))}
        results["probes"][name] = {"cv_accuracy": round(float((pred == y).mean()), 4),
                                   "recall_per_class": per_class}
        print(f"{name:28s} cv_acc={results['probes'][name]['cv_accuracy']:.3f}  {per_class}")
    print(f"{'majority baseline':28s} {majority:.3f}")

    out = get_settings().artifacts_dir / "analysis" / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    (out / "artifact_probe.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved {out / 'artifact_probe.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
