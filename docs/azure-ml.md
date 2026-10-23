# Running the C-series on Azure ML

C-series experiments (gpt-oss-20b readouts) run as Azure ML command jobs, one job per
experiment × label catalog, on a GPU compute cluster. All names come from `.env`
(`AZURE_SUBSCRIPTION_ID`, `AZURE_RESOURCE_GROUP`, `AML_WORKSPACE`, `AML_COMPUTE`).

## Hardware

gpt-oss-20b is loaded dequantized to bf16 (about 39 GB of VRAM), so one 80 GB GPU per job is
enough. We used `Standard_NC24ads_A100_v4` (1× A100 80 GB) on a **low-priority** cluster that
scales from 0 nodes. H100 sizes (`Standard_NC40ads_H100_v5`) also work. The repository's cost
numbers use the on-demand A100 price (`configs/models/gpt-oss-20b.yaml`).

## One-time setup

1. **Workspace and cluster.** Create an Azure ML workspace and an `AmlCompute` cluster with a
   system-assigned managed identity, minimum 0 nodes and a short idle timeout, e.g. with the
   Python SDK (`azure.ai.ml.entities.AmlCompute(name=..., size="Standard_NC24ads_A100_v4",
   tier="low_priority", min_instances=0, max_instances=5,
   identity=IdentityConfiguration(type="system_assigned"))`). Low-priority and dedicated GPU
   quota are separate per region; check quota first.
2. **Storage access.** Jobs run with the submitting user's identity
   (`UserIdentityConfiguration`) and write outputs to the workspace's default blob datastore.
   `scripts/aml_collect.py` then downloads them directly from blob storage.
   - Grant *Storage Blob Data Contributor* on the workspace storage account to yourself and to
     the cluster's managed identity.
   - If key-based storage access is disabled, switch the workspace to identity-based datastores
     (`systemDatastoresAuthMode: identity`).
3. **Private storage (only if your organisation requires it).** If the storage account has no
   public network access, give the workspace a managed virtual network with internet outbound,
   so jobs can still reach Hugging Face and PyPI:
   `uv run --extra aml python infra/aml/enable_managed_vnet.py`. It recreates the cluster with
   the same settings; grant the new identity the blob role again. The machine running
   `aml_collect.py` also needs network access to the storage account (e.g. a firewall or
   network-perimeter rule for your IP).

`infra/aml/provision.ps1` is an optional script for an *interactive* GPU compute instance (it
checks quota and creates a workspace and compute instance). The benchmark itself only needs
the cluster.

## How a job runs

[`infra/aml/run_job.sh`](../infra/aml/run_job.sh) is the entry point:

1. **Bootstrap.** On a public base image, it installs uv, Python 3.12 and the `uv.lock`-pinned
   requirements from `infra/aml/env/requirements.txt` (about 1–2 minutes; no image build).
2. **Run on local disk.** Each run unit has a fixed output path
   (`edb/<tag>/<dataset>/<experiment>__<catalog>/` on the default datastore). The script
   restores any previous output, runs `scripts/run_experiment.py` on local disk, and syncs back
   every 3 minutes and on exit.
3. **Resume.** A pre-empted (low-priority) or resubmitted job resumes where it stopped.
4. **Model download.** The model is downloaded from Hugging Face with retries and backoff
   (`src/edb/local/model.py`). Many parallel jobs share one egress IP, so anonymous downloads
   can hit rate limits.

Regenerate the requirements file after dependency changes:

```powershell
uv export --extra local --no-hashes --no-emit-project --no-header --no-annotate --format requirements-txt -o infra/aml/env/requirements.txt
```

## Submitting and collecting

```powershell
uv sync --extra aml

# 1. pilot on a small dev subset (separate tag and local root, so subsets never mix with full runs)
uv run --extra aml python scripts/aml_submit.py --experiment c6_oss_letter_readout c8_oss_reason_then_read --catalog short --split dev --limit 20 --tag pilot --parallel
uv run --extra aml python scripts/aml_collect.py --tag pilot --dest artifacts/pilot --wait
uv run python scripts/check_pilot.py --root artifacts/pilot --dataset ace --split dev --experiment c6_oss_letter_readout c8_oss_reason_then_read --catalog short --min-acc chance

# 2. full runs; AML queues jobs and scales the cluster up to its node limit
uv run --extra aml python scripts/aml_submit.py --dataset banking77 --experiment c1_oss_autoregressive c2_oss_sequence_scoring --split calibration test --allow-test --tag main --parallel
uv run --extra aml python scripts/aml_collect.py --tag main --wait      # downloads into artifacts/runs
uv run --extra aml python scripts/aml_collect.py --tag main --status-only
```

**Submission modes.**
- **Sequential (default):** one job at a time, so a warm node is reused.
- **`--parallel`:** submits all units at once, one node each. Submit the longest jobs first,
  because AML starts queued jobs roughly in submission order.

In both modes, units whose latest job under the same tag and splits completed are skipped.

**Other options.**
- `--compute` overrides `AML_COMPUTE`.
- `--dry-run` prints the plan without submitting.

**Job log.** Every submission is appended to `artifacts/aml/jobs.jsonl`. Re-running the same
command resubmits every unit that has not completed, and it resumes from its outputs. That
includes jobs that are still running, so wait for them (or cancel them) first.

**Pilot gate.** `check_pilot.py` requires, for every expected run: predictions exist, zero
errors, every distribution sums to 1, and (optionally) accuracy above chance. It also projects
the full-run time from pilot latencies.

## Other scripts as jobs (R-series)

`scripts/aml_run.py` submits any repo script as a job, with the same bootstrap, resume and
sync. The output folder `edb/<tag>/<group>/<key>/` mirrors `ARTIFACTS_DIR`, and
`aml_collect.py` downloads its `runs/`, `latency/`, `probes/` and `training/` folders.

```powershell
uv run --extra aml python scripts/aml_run.py --key probes-qwen3.5-9b --tag replication -- scripts/run_probes.py --target qwen3.5-9b
uv run --extra aml python scripts/aml_run.py --key r3-ce_shuffle --tag replication -- scripts/train_decider.py --arm ce_shuffle --split calibration test --allow-test
uv run --extra aml python scripts/aml_collect.py --tag replication --wait
```

Submitting uploads the repository snapshot to the workspace storage account, so the submitting
machine needs the same network access to it as `aml_collect.py` (see *Private storage*).

## When the workspace storage is unreachable

A network security perimeter on the workspace storage only admits listed public IPs and
resources of the subscription. A corporate **split-tunnel VPN** that routes Azure ranges
(e.g. Azure Storage `57.150.0.0/15`) through an Azure-hosted gateway defeats IP rules: the
storage account no longer sees the laptop's public IP, and adding the gateway's IP does not
help either (its traffic reaches Storage over the Microsoft backbone). Check the route with
`Find-NetRoute -RemoteIPAddress <storage IP>`. The subscription's activity log
(`httpRequest.clientIpAddress`) shows the IP Azure sees for management calls.

Jobs themselves are unaffected (they run inside the subscription), and the workspace API is
not behind the perimeter. Two options keep everything else working without touching
storage from the laptop:

- **`--git-ref <branch>`** (`aml_run.py`, `aml_submit.py`): the job downloads that pushed
  commit from GitHub (`origin` must be public) instead of uploading the working tree.
- **Job summaries as tags**: on exit every job attaches a compact summary (all metrics,
  example-by-example agreement with Decision-1 from `configs/reference/`, probe and training
  summaries), its exit code and log tail as MLflow tags. `scripts/aml_summaries.py --tag <tag>`
  reassembles them into `artifacts/aml/summaries/<tag>/` and installs probe summaries, so
  `probe_report.py` and `replication_report.py` work without the raw files. For each unit it
  reads the latest submission, or the latest one not cancelled (so a cancelled duplicate does
  not hide a finished job). Full outputs stay
  in the datastore for `aml_collect.py` once storage is reachable again.

```powershell
git push origin <branch>
uv run --extra aml python scripts/aml_run.py --key r3-ce_shuffle --tag replication --git-ref <branch> -- scripts/train_decider.py --arm ce_shuffle --split calibration test --allow-test
uv run --extra aml python scripts/aml_summaries.py --tag replication --tail
uv run python scripts/replication_report.py --tag replication
```

Low-priority jobs can be pre-empted and are then restarted by Azure ML. Evaluation and probes
resume from the synced output; training arms with `checkpoint_every` (all 9B arms since R9)
resume from their last checkpoint, others restart from scratch.

Training jobs can also run the question-type and unseen-benchmark evaluations with the new
adapter (`train_decider.py --r14 --r19`); their metrics come back in the job summary.
