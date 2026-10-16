# Setup and running the benchmark

Everything environment-specific comes from `.env` (copy [`.env.example`](../.env.example)). Nothing about a
particular Azure subscription is stored in the repository. Everything experiment-specific
(datasets, label catalogs, models, prompts, experiments) is a YAML config under `configs/`.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (provisions Python 3.12 and a local `.venv`).
- Azure CLI, signed in with `az login` (authentication is Microsoft Entra ID only, through
  `DefaultAzureCredential`; no API keys).
- A Microsoft Foundry (AI Services) resource and project with these deployments. Deployment
  names are set in `configs/models/*.yaml`.

  | Deployment (default name) | Model | Used by |
  |---|---|---|
  | `gpt-6-luna` | GPT-6 Luna (Azure OpenAI, Responses API) | A1, A1v |
  | `Decision-1` | Microsoft-Decision-1 v1 | B1 |
  | `text-embedding-3-large` | text-embedding-3-large | S1 |

  Your identity needs a data-plane role on the resource, e.g. *Cognitive Services User*.
- Optional, for the C-series: an Azure ML workspace with a GPU cluster (see
  [azure-ml.md](azure-ml.md)), or any Linux machine with an 80 GB GPU.
- Optional, for A2: an OpenAI platform key (`OPENAI_API_KEY`). Azure does not serve the
  OpenAI Decisions API yet.

## Install and configure

```powershell
uv sync                                  # API methods, reporting, tests
uv sync --extra local --extra aml        # + gpt-oss-20b stack (Linux GPU) and Azure ML SDK
Copy-Item .env.example .env              # then fill in the values below
az login
uv run python scripts/run_preflight.py --skip-gpu
uv run pytest
```

| `.env` variable | Purpose |
|---|---|
| `AZURE_SUBSCRIPTION_ID`, `AZURE_RESOURCE_GROUP` | Azure ML client (C-series jobs) |
| `AZURE_TENANT_ID` | optional, pins the Entra tenant |
| `AZURE_AI_RESOURCE`, `AZURE_AI_PROJECT` | Foundry resource + project. Endpoints are derived as `https://<resource>.openai.azure.com/openai/v1/` (Responses / embeddings) and `https://<resource>.services.ai.azure.com/providers/microsoft/v1/systemone` (Decision-1) |
| `AZURE_OPENAI_BASE_URL`, `AZURE_AI_ENDPOINT` | optional explicit endpoint overrides |
| `AML_WORKSPACE`, `AML_COMPUTE` | Azure ML workspace and GPU cluster (C-series) |
| `OPENAI_API_KEY` | only for A2 (OpenAI Decisions API) |
| `HF_TOKEN` | optional, for Hugging Face downloads |

The preflight script checks the token, prepares the dataset, sends one real request through
every enabled API experiment, probes A2/A3 feasibility and (without `--skip-gpu`) the local GPU
stack. It writes results to `artifacts/preflight/`.

**Rate limits.** Each experiment config has `run.max_rpm` and `run.concurrency`. Keep them below
your deployment's limits. Decision-1 deployments can have low limits (ours allowed 10 requests
per minute, so `b1_decision1` uses `max_rpm: 9`).

**Package versions.** `pyproject.toml` pins `exclude-newer` to a fixed snapshot date for
reproducibility. `uv.lock` was resolved through a PyPI mirror; if `uv sync` cannot reach the
mirror's URLs, run `uv lock` once against your own index (the snapshot date keeps the same
versions).

## Workflow

```powershell
# 1. data: download + frozen grouped/stratified splits (train / dev / calibration; test sealed)
uv run python scripts/prepare_data.py --dataset ace

# 2. predictions (resumable: re-run to continue; --retry-errors redoes failed requests)
uv run python scripts/run_experiment.py --experiment a1_gpt6_luna b1_decision1 --split dev calibration
uv run python scripts/run_experiment.py --experiment b1_decision1 --catalog verbose --limit 100
uv run python scripts/run_experiment.py --dataset banking77 --experiment s1_emb_logreg --split calibration test --allow-test

# 3. optional latency protocol (defaults: 20 warmups, 200 examples, 5 reps, concurrency 1 and 8)
uv run python scripts/run_latency.py --experiment a1_gpt6_luna --catalog short

# 4. offline C9 cascade from stored C6+ and C8 runs
uv run python scripts/build_cascade.py --dataset ace --fast c6p_oss_letter_debiased --slow c8_oss_reason_then_read

# 5. report: artifacts/reports/<dataset>/<split>/report.md (+ CSVs, plots, metrics.json)
uv run python scripts/build_report.py --dataset ace --split test
```

The `test` split is sealed: runs on it need `--allow-test`.

C-series experiments use the same `run_experiment.py` on a GPU machine. In practice we run them
as Azure ML jobs ([azure-ml.md](azure-ml.md)).

### Outputs and run safety

- Predictions: `artifacts/runs/<dataset>/<experiment>__<catalog>/<split>/predictions.jsonl`, next
  to a `manifest.json` with the full config snapshot and its hash.
- The manifest is written before any model call. The hash covers the method, prompts, catalog,
  model, data fingerprint and `PREDICTION_LOGIC_VERSION` (`src/edb/providers/base.py`); local
  models also include `SCORING_VERSION` (`src/edb/local/__init__.py`). Bump these when
  post-processing changes.
- A run with a different config hash refuses to append (use `--force` to archive the old run).
  A per-split lock stops two processes writing the same split.
- After a hashing-scheme change, `scripts/migrate_manifests.py` verifies stored predictions and
  re-stamps manifests.
- `--limit` / `--sample-seed` take a deterministic subset. Re-running with another limit adds
  to the same file. Reports evaluate every stored prediction, and each leaderboard is
  intersected to the examples shared by all its runs.

### Reports

`build_report.py` writes two separate leaderboards, chosen by each experiment's `leaderboards`
field:

- `product`: products and baselines (A, B, S) on hosted APIs.
- `controlled`: open-weight readouts (C) on gpt-oss-20b, same weights throughout.

They are never merged or ranked against each other. Reports read the current experiment
config to group runs; the grouping is not part of the config hash, so regrouping needs no
re-run.

Each leaderboard shows:
- accuracy, macro-F1, NLL, Brier, ECE and AURC;
- accuracy on the most confident 80% / 90%;
- the temperature `T` fitted on the `calibration` split, with temperature-scaled NLL / ECE;
- latency and cost.

Selective-prediction thresholds are also fitted on calibration and applied unchanged. Runs are
labelled `supervised` (fitted on train labels) or `zero_shot`.

## Embedding baselines (S1)

All S1 heads share one method (`type: embedding`). The example is embedded live at prediction
time (honest latency and cost); train-split embeddings are cached in
`artifacts/cache/embeddings/`.

- **Learned heads** (`centroid`, `knn`, `logreg`) are fit on the `train` split only. `logreg`
  picks its regularisation by group-aware cross-validation within train, so dev, calibration
  and test stay untouched.
- **`views: fields`** embeds each state-template field separately (ACE: clauses, scenario) and
  uses `[a, b, a*b, |a-b|]` features.

## Adding a dataset

1. Create `configs/datasets/<name>.yaml`:
   - `files`: url or local path; json/jsonl/csv; `records_path` for nested JSON.
   - `fields`: label, plus optional id and group.
   - `state.template`: `{field}` placeholders, with `mapping` / `list` / `json` formatters.
   - `labels`: a list or `auto`.
   - `task.instructions`, `splits`, and `label_catalogs` (role → catalog file or `auto`).
2. Optional: `configs/label_catalogs/<catalog>.yaml` with a description per label (the
   `verbose` role).
3. `uv run python scripts/prepare_data.py --dataset <name>`, then commit
   `configs/datasets/splits/<name>.json` (the frozen split assignment).
4. Run any experiment with `--dataset <name>`.

To vet a new dataset for shortcuts, run the artifact probe. It fits grouped-CV logistic
regression on embeddings of each state field alone, all fields, and the full state, using the
train split only:

```powershell
uv run python scripts/probe_artifacts.py --dataset <name>
```
