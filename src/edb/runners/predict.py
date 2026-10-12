"""Resumable, concurrent prediction runner.

Layout: artifacts/runs/<dataset>/<run_id>/<split>/{predictions.jsonl, manifest.json}
where run_id = "<experiment id>__<label catalog role>".

Predictions are appended as they complete; re-running skips finished examples, so
interrupted or rate-limited runs can simply be restarted.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tqdm.asyncio import tqdm

from edb import __version__
from edb.config import (
    DatasetConfig,
    ExperimentConfig,
    ModelConfig,
    config_hash,
    load_dataset_config,
    load_experiment_config,
    load_label_catalog,
    load_model_config,
)
from edb.datasets.store import dataset_labels, load_split, sample_examples
from edb.prompts import DEFAULT_SYSTEM_PROMPT
from edb.providers.base import MethodContext
from edb.providers.registry import build_method
from edb.schemas import LabelCatalog, Prediction
from edb.settings import get_settings


@dataclass
class RunSpec:
    experiment: ExperimentConfig
    dataset: DatasetConfig
    catalog_role: str
    catalog: LabelCatalog
    model: ModelConfig | None

    @property
    def run_id(self) -> str:
        return f"{self.experiment.id}__{self.catalog_role}"

    def snapshot(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment.model_dump(mode="json", exclude={"run"}),
            "dataset": self.dataset.name,
            "task": self.dataset.task.model_dump(mode="json"),
            "state_template": self.dataset.state.template,
            "default_system_prompt": DEFAULT_SYSTEM_PROMPT,
            "catalog_role": self.catalog_role,
            "catalog": self.catalog.model_dump(mode="json"),
            "model": self.model.model_dump(mode="json") if self.model else None,
        }

    def context(self, concurrency: int | None = None) -> MethodContext:
        run = self.experiment.run.model_copy(
            update={"concurrency": concurrency or self.experiment.run.concurrency}
        )
        return MethodContext(
            run_id=self.run_id,
            dataset=self.dataset,
            catalog=self.catalog,
            model=self.model,
            params=self.experiment.method.params,
            run=run,
            settings=get_settings(),
        )


def resolve_runs(
    experiment: str | Path, dataset: str | None = None, catalogs: list[str] | None = None
) -> list[RunSpec]:
    exp = load_experiment_config(experiment)
    ds_name = dataset or exp.dataset
    if not ds_name:
        raise ValueError(f"experiment {exp.id!r} has no default dataset; pass --dataset")
    ds = load_dataset_config(ds_name)
    labels = dataset_labels(ds.name)
    model = load_model_config(exp.method.model) if exp.method.model else None
    specs = []
    for role in catalogs or exp.label_catalogs:
        if role not in ds.label_catalogs:
            print(f"[skip] dataset {ds.name!r} has no {role!r} catalog for {exp.id}")
            continue
        specs.append(RunSpec(exp, ds, role, load_label_catalog(ds, role, labels), model))
    return specs


def run_dir(dataset: str, run_id: str, split: str) -> Path:
    return get_settings().artifacts_dir / "runs" / dataset / run_id / split


def read_predictions(path: Path) -> dict[str, Prediction]:
    """Latest prediction per example id."""
    preds: dict[str, Prediction] = {}
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    p = Prediction.model_validate_json(line)
                    preds[p.example_id] = p
    return preds


async def run_split(
    spec: RunSpec,
    split: str,
    limit: int | None = None,
    concurrency: int | None = None,
    retry_errors: bool = False,
    force: bool = False,
    sample_seed: int = 0,
) -> dict[str, Any]:
    out = run_dir(spec.dataset.name, spec.run_id, split)
    manifest_path, preds_path = out / "manifest.json", out / "predictions.jsonl"
    chash = config_hash(spec.snapshot())

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["config_hash"] != chash:
            if not force:
                raise RuntimeError(
                    f"{out} was produced with a different config ({manifest['config_hash']} != "
                    f"{chash}). Use --force to archive it and start over."
                )
            out.rename(out.with_name(f"{split}.archived-{datetime.now(UTC):%Y%m%dT%H%M%S}"))
            manifest = None
    else:
        manifest = None
    out.mkdir(parents=True, exist_ok=True)
    manifest = manifest or {
        "run_id": spec.run_id,
        "dataset": spec.dataset.name,
        "split": split,
        "config_hash": chash,
        "config": spec.snapshot(),
        "leaderboards": spec.experiment.leaderboards,
        "sessions": [],
    }

    examples = sample_examples(load_split(spec.dataset.name, split), limit, seed=sample_seed)
    done = read_predictions(preds_path)
    todo = [e for e in examples if e.id not in done or (retry_errors and not done[e.id].ok)]

    ctx = spec.context(concurrency)
    session: dict[str, Any] = {
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "package_version": __version__,
        "requested": len(examples),
        "submitted": len(todo),
        "concurrency": ctx.run.concurrency,
    }
    if todo:
        method = build_method(spec.experiment.method.type, ctx)
        sem = asyncio.Semaphore(ctx.run.concurrency)
        t0 = time.perf_counter()
        try:
            await method.setup()
            with preds_path.open("a", encoding="utf-8") as sink:

                async def one(example):
                    async with sem:
                        pred = await method.predict(example)
                    sink.write(pred.model_dump_json() + "\n")
                    sink.flush()
                    return pred

                await tqdm.gather(*(one(e) for e in todo), desc=f"{spec.run_id}/{split}")
        finally:
            await method.aclose()
        session["wall_s"] = round(time.perf_counter() - t0, 3)
    session["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    manifest["sessions"].append(session)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    final = read_predictions(preds_path)
    selected = [final[e.id] for e in examples if e.id in final]
    ok = [p for p in selected if p.ok]
    correct = sum(p.predicted == p.gold for p in ok)
    return {
        "run_id": spec.run_id,
        "split": split,
        "n": len(selected),
        "errors": len(selected) - len(ok),
        "accuracy": round(correct / len(selected), 4) if selected else None,
        "cost_usd": round(sum(p.cost_usd or 0 for p in selected), 4),
        "dir": str(out),
    }
