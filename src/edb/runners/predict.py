"""Resumable, concurrent prediction runner.

Layout: artifacts/runs/<dataset>/<run_id>/<split>/{predictions.jsonl, manifest.json}
where run_id = "<experiment id>__<label catalog role>".

Predictions are appended as they complete; re-running skips finished examples, so
interrupted or rate-limited runs can simply be restarted.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from contextlib import contextmanager
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
from edb.datasets.store import dataset_fingerprint, dataset_labels, load_split, sample_examples
from edb.local import SCORING_VERSION as LOCAL_SCORING_VERSION
from edb.prompts import ANSWER_JSON_CONFIDENCE, ANSWER_LABEL, DEFAULT_SYSTEM_PROMPT
from edb.providers.base import PREDICTION_LOGIC_VERSION, MethodContext
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
        """Everything that can change predictions (hashed to guard against mixing outputs)."""
        return {
            "logic_version": PREDICTION_LOGIC_VERSION,
            "experiment": self.experiment.model_dump(
                mode="json", include={"id", "method"}
            ),
            "dataset": self.dataset.name,
            "data_fingerprint": dataset_fingerprint(self.dataset),
            "task": self.dataset.task.model_dump(mode="json"),
            "state_template": self.dataset.state.template,
            "prompts": {
                "system": DEFAULT_SYSTEM_PROMPT,
                "answer_label": ANSWER_LABEL,
                "answer_json_confidence": ANSWER_JSON_CONFIDENCE,
            },
            "catalog_role": self.catalog_role,
            "catalog": self.catalog.model_dump(mode="json"),
            "model": self.model.model_dump(mode="json", exclude={"pricing"})
            if self.model else None,
            **({"local_scoring_version": LOCAL_SCORING_VERSION}
               if self.model and self.model.provider == "local_hf" else {}),
            **({"adapter_sha256": adapter_fingerprint(adapter)}
               if (adapter := self.experiment.method.params.get("adapter")) else {}),
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


def adapter_fingerprint(adapter: str) -> str | None:
    """sha256 of a trained adapter's weights (None if not trained yet), so predictions made
    with a retrained adapter never mix with older ones."""
    path = Path(adapter)
    path = (path if path.is_absolute() else get_settings().artifacts_dir / path)
    weights = path / "adapter.safetensors"
    return hashlib.sha256(weights.read_bytes()).hexdigest()[:16] if weights.exists() else None


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


def _parse_lines(path: Path) -> list[Prediction]:
    """All records in append order. A truncated final line (crash mid-write) is skipped."""
    if not path.exists():
        return []
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    records = []
    for i, line in enumerate(lines):
        try:
            records.append(Prediction.model_validate_json(line))
        except ValueError:
            if i == len(lines) - 1:
                print(f"[warn] ignoring truncated last record in {path}")
                break
            raise
    return records


def read_prediction_history(path: Path) -> list[Prediction]:
    return _parse_lines(path)


def read_predictions(path: Path) -> dict[str, Prediction]:
    """Latest prediction per example id."""
    return {p.example_id: p for p in _parse_lines(path)}


@contextmanager
def run_lock(out: Path):
    """Exclusive lock so two processes never write the same run split."""
    lock = out / ".lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(
            f"{out} is locked by another run ({lock.read_text(encoding='utf-8')}). "
            f"If no run is active (e.g. after a crash), delete {lock}."
        ) from None
    os.write(fd, f"pid={os.getpid()} since={datetime.now(UTC).isoformat()}".encode())
    os.close(fd)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


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
    snapshot = spec.snapshot()
    chash = config_hash(snapshot)

    if (out / ".lock").exists():
        raise RuntimeError(f"{out} is locked by another run; delete {out / '.lock'} if stale")
    manifest = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    elif preds_path.exists() and not force:
        raise RuntimeError(f"{preds_path} exists without a manifest; its config is unknown. "
                           "Move it away or use --force to archive it.")
    if manifest is not None and manifest["config_hash"] != chash:
        if not force:
            raise RuntimeError(
                f"{out} was produced with a different config ({manifest['config_hash']} != "
                f"{chash}). Use --force to archive it and start over."
            )
        manifest = None
    if manifest is None and out.exists() and any(out.iterdir()):
        if not force:
            raise RuntimeError(f"{out} is not empty; use --force to archive it")
        out.rename(out.with_name(f"{split}.archived-{datetime.now(UTC):%Y%m%dT%H%M%S}"))
    out.mkdir(parents=True, exist_ok=True)

    with run_lock(out):
        return await _run_locked(spec, split, out, manifest, snapshot, chash, limit,
                                 concurrency, retry_errors, sample_seed)


async def _run_locked(
    spec: RunSpec, split: str, out: Path, manifest: dict[str, Any] | None,
    snapshot: dict[str, Any], chash: str, limit: int | None, concurrency: int | None,
    retry_errors: bool, sample_seed: int,
) -> dict[str, Any]:
    manifest_path, preds_path = out / "manifest.json", out / "predictions.jsonl"
    manifest = manifest or {
        "run_id": spec.run_id,
        "dataset": spec.dataset.name,
        "split": split,
        "config_hash": chash,
        "config": snapshot,
        "sessions": [],
    }
    manifest["leaderboards"] = spec.experiment.leaderboards
    _write_manifest(manifest_path, manifest)  # before any API call: config is always known

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
        try:
            t_setup = time.perf_counter()
            await method.setup()
            session["setup_s"] = round(time.perf_counter() - t_setup, 3)
            session["method_info"] = method.info
            t0 = time.perf_counter()
            with preds_path.open("a", encoding="utf-8") as sink:

                async def one(example):
                    async with sem:
                        pred = await method.predict(example)
                    sink.write(pred.model_dump_json() + "\n")
                    sink.flush()
                    return pred

                await tqdm.gather(*(one(e) for e in todo), desc=f"{spec.run_id}/{split}")
            session["wall_s"] = round(time.perf_counter() - t0, 3)
        finally:
            await method.aclose()
    session["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    manifest["sessions"].append(session)
    _write_manifest(manifest_path, manifest)

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
