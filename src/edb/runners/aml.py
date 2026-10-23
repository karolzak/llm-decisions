"""Azure ML job submission/collection for runs on the GPU cluster."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from edb.settings import REPO_ROOT, get_settings

ENV_DIR = REPO_ROOT / "infra" / "aml" / "env"
ENV_NAME = "edb-base"
# Plain public base image: no custom image build (the workspace has no reachable container
# registry under the private-storage policy). run_job.sh installs the locked requirements.
BASE_IMAGE = "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest"
TERMINAL = {"Completed", "Failed", "Canceled", "NotResponding"}
# Folders under a job's ARTIFACTS_DIR that scripts/aml_collect.py downloads.
ARTIFACT_DIRS = ("runs/", "latency/", "probes/", "training/")


def status_of(job) -> str:
    """Job status as a plain string; the SDK may return an enum (e.g. JobStatus.FAILED)."""
    raw = job.status
    value = getattr(raw, "value", None)
    if isinstance(value, str):
        return value
    text = str(raw)
    if text.startswith("JobStatus."):
        return "".join(part.capitalize() for part in text.split(".", 1)[1].split("_"))
    return text


def _credential():
    from azure.identity import AzureCliCredential, ChainedTokenCredential, DefaultAzureCredential

    return ChainedTokenCredential(
        AzureCliCredential(process_timeout=60),
        DefaultAzureCredential(exclude_interactive_browser_credential=True),
    )


def ml_client():
    from azure.ai.ml import MLClient

    s = get_settings()
    s.require("azure_subscription_id", "azure_resource_group", "aml_workspace")
    credential = _credential()
    return MLClient(credential, s.azure_subscription_id, s.azure_resource_group, s.aml_workspace)


def ensure_environment(ml) -> str:
    """Image-only environment (no build); the job installs pinned requirements at start."""
    from azure.ai.ml.entities import Environment
    from azure.core.exceptions import ResourceNotFoundError

    version = hashlib.sha256(BASE_IMAGE.encode()).hexdigest()[:12]
    try:
        ml.environments.get(ENV_NAME, version)
    except ResourceNotFoundError:
        ml.environments.create_or_update(Environment(
            name=ENV_NAME, version=version, image=BASE_IMAGE,
            description="Base image; infra/aml/run_job.sh installs uv.lock-pinned requirements",
        ))
    return f"azureml:{ENV_NAME}:{version}"


def output_path(tag: str, dataset: str, run_id: str) -> str:
    return f"azureml://datastores/workspaceblobstore/paths/edb/{tag}/{dataset}/{run_id}/"


def jobs_log() -> Path:
    return get_settings().artifacts_dir / "aml" / "jobs.jsonl"


def record(entry: dict[str, Any]) -> None:
    path = jobs_log()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def read_log(tag: str | None = None) -> list[dict[str, Any]]:
    path = jobs_log()
    if not path.exists():
        return []
    entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [e for e in entries if tag is None or e["tag"] == tag]


def job_key(experiment: str, catalog: str) -> str:
    return f"{experiment}__{catalog}"


def git_source(ref: str) -> tuple[str, str]:
    """(commit sha, tarball URL) for `ref`, which must already be pushed to a public GitHub
    `origin`. Jobs submitted from source download that exact commit, so no code is uploaded
    to the workspace storage."""
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()

    sha = git("rev-parse", "--verify", f"{ref}^{{commit}}")
    if not git("branch", "-r", "--contains", sha):
        raise RuntimeError(f"commit {sha[:12]} is not on any remote branch; push it first")
    m = re.match(r"(?:https://github\.com/|git@github\.com:)([^/]+)/(.+?)(?:\.git)?$",
                 git("remote", "get-url", "origin"))
    if not m:
        raise RuntimeError("origin is not a GitHub repository")
    return sha, f"https://codeload.github.com/{m.group(1)}/{m.group(2)}/tar.gz/{sha}"


def submit(ml, *, environment: str, dataset: str, experiment: str, catalog: str,
           splits: list[str], tag: str, extra_args: list[str], compute: str | None = None,
           git_ref: str | None = None) -> str:
    """Submit one job running a single experiment x catalog unit."""
    args = ["scripts/run_experiment.py", "--experiment", experiment, "--catalog", catalog,
            "--dataset", dataset, "--split", *splits, *extra_args]
    return submit_script(ml, environment=environment, group=dataset,
                         key=job_key(experiment, catalog), args=args, tag=tag,
                         compute=compute, splits=splits, git_ref=git_ref)


def submit_script(ml, *, environment: str, group: str, key: str, args: list[str], tag: str,
                  compute: str | None = None, splits: list[str] | None = None,
                  git_ref: str | None = None) -> str:
    """Submit one job running `python <args>` (a repo script) with a resumable output folder
    at edb/<tag>/<group>/<key>/ that mirrors ARTIFACTS_DIR (runs/, probes/, training/, ...).

    `git_ref`: run a pushed commit downloaded from GitHub instead of uploading the local
    working tree (for machines that cannot reach the workspace storage)."""
    from azure.ai.ml import Output, command
    from azure.ai.ml.entities import UserIdentityConfiguration

    compute = compute or get_settings().aml_compute
    if not compute:
        raise RuntimeError("AML_COMPUTE not set (see .env.example) and no --compute given")
    run = "bash infra/aml/run_job.sh ${{outputs.artifacts}} -- " + " ".join(args)
    code: str | None = str(REPO_ROOT)
    sha = None
    if git_ref:
        sha, url = git_source(git_ref)
        code = None
        run = ("(command -v curl >/dev/null || (apt-get update -qq && apt-get install -y -qq "
               "curl ca-certificates)) && mkdir -p repo && "
               f"curl -fsSL --retry 5 {url} | tar xz --strip-components=1 -C repo && "
               f"cd repo && {run}")
    job = command(
        code=code,
        command=run,
        environment=environment,
        compute=compute,
        # Storage is key-less (identity-based datastores): jobs access data as the submitter.
        identity=UserIdentityConfiguration(),
        outputs={"artifacts": Output(type="uri_folder", mode="rw_mount",
                                     path=output_path(tag, group, key))},
        experiment_name=f"edb-{group}-{tag}",
        display_name=f"{key}-{'-'.join(splits)}" if splits else key,
        tags={"run_id": key, "splits": ",".join(splits or []), "tag": tag,
              **({"git_sha": sha} if sha else {})},
    )
    created = ml.jobs.create_or_update(job)
    record({"name": created.name, "run_id": key, "dataset": group, "splits": splits or [],
            "tag": tag, "args": args, "compute": compute, "studio_url": created.studio_url,
            "git_sha": sha, "submitted_at": datetime.now(UTC).isoformat(timespec="seconds")})
    return created.name


def wait_for(ml, name: str, poll_s: float = 30.0, log=print) -> str:
    """Block until job `name` reaches a terminal state; returns that state."""
    last = None
    while True:
        status = status_of(ml.jobs.get(name))
        if status != last:
            log(f"{datetime.now(UTC):%H:%M:%S} {name}: {status}")
            last = status
        if status in TERMINAL:
            return status
        time.sleep(poll_s)


def collect(ml, entries: list[dict[str, Any]], dest_root: Path, only_terminal: bool = True
            ) -> list[dict[str, Any]]:
    """Download each job's `runs/` (and `latency/`) into `dest_root`, merging per run split."""
    statuses = []
    latest: dict[str, dict[str, Any]] = {}
    for e in entries:  # the latest submission per (tag, dataset, run_id) wins
        latest[f"{e['tag']}/{e['dataset']}/{e['run_id']}"] = e
    for e in latest.values():
        job = ml.jobs.get(e["name"])
        status = {"run_id": e["run_id"], "job": e["name"], "status": status_of(job)}
        statuses.append(status)
        if only_terminal and status["status"] not in TERMINAL:
            continue
        try:
            n = download_output(ml, job.outputs["artifacts"].path, dest_root)
        except Exception as exc:  # noqa: BLE001
            status["download_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            continue
        status["collected"] = n > 0
        status["files"] = n
    return statuses


def download_output(ml, output_uri: str, dest_root: Path) -> int:
    """Copy the artifact folders (`runs/`, `latency/`, `probes/`, `training/`) from a job's
    blob output folder into `dest_root`.

    Reads the datastore directly with the caller's Entra identity (the SDK's job download
    does not fetch outputs bound to a custom datastore path).
    """
    from azure.storage.blob import BlobServiceClient

    m = re.match(r"azureml://datastores/([^/]+)/paths/(.+)", output_uri)
    if not m:
        raise ValueError(f"unsupported output uri {output_uri!r}")
    datastore, prefix = m.group(1), m.group(2).rstrip("/") + "/"
    ds = ml.datastores.get(datastore)
    svc = BlobServiceClient(f"https://{ds.account_name}.blob.core.windows.net",
                            credential=_credential())
    container = svc.get_container_client(ds.container_name)
    count = 0
    for sub in ARTIFACT_DIRS:
        for blob in container.list_blobs(name_starts_with=prefix + sub, include=["metadata"]):
            if (blob.metadata or {}).get("hdi_isfolder") == "true" or blob.name.endswith(".lock"):
                continue
            target = dest_root / blob.name[len(prefix):]
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as f:
                container.download_blob(blob.name).readinto(f)
            count += 1
    return count
