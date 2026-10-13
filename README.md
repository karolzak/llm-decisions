# Enterprise Decisions Benchmark

Benchmark of bounded-decision mechanisms on enterprise text decisions: autoregressive
generation vs. decision APIs (OpenAI Decisions, Microsoft-Decision-1) vs. custom
non-generative readouts on `openai/gpt-oss-20b`. See [exp-plan.md](exp-plan.md).

Everything is **dataset-agnostic**: datasets, label catalogs, task instructions, models and
experiments are YAML configs; code never hard-codes labels or prompts for a specific dataset.

## Layout

```
configs/
  datasets/           <name>.yaml (sources, fields, state template, task, splits, catalogs)
  datasets/splits/    frozen split assignments (committed; reproducible on every machine)
  label_catalogs/     label sets with optional semantic descriptions (short vs. verbose)
  models/             deployments + pricing
  experiments/        method + model + catalog roles + run settings (rate limits, concurrency)
src/edb/
  schemas.py          Example, LabelCatalog, TaskSpec, Prediction, Usage
  config.py           config models + loading
  prompts.py          shared autoregressive prompt construction
  datasets/           sources (url/local; json/jsonl/csv), state rendering, grouped splits
  providers/          decision methods (A1, A2, B1, S1), auth, retries, rate limiting
  calibration/        accuracy, macro-F1, NLL, Brier, ECE, risk/coverage, temperature scaling
  runners/            resumable prediction runner, latency benchmark
  reporting/          leaderboards, reliability + risk/coverage plots, markdown report
scripts/              prepare_data, run_preflight, run_experiment, run_latency, build_report
infra/aml/            Azure ML GPU provisioning (parked)
artifacts/            outputs (git-ignored)
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) (provisions Python 3.12 and a local `.venv`).

```powershell
uv sync
Copy-Item .env.example .env
az login --tenant 16b3c013-d300-468d-ac64-7eda0820b6d3
uv run python scripts/run_preflight.py --skip-gpu
uv run pytest
```

### Azure

| What | Value |
|---|---|
| Subscription / RG | `005cc313-…` / `kz-exp-rg` (swedencentral) |
| Foundry resource / project | `kz-exp-foundry` / `proj-default` |
| `gpt-6-luna` | GlobalStandard, 250 RPM / 250K TPM |
| `Decision-1` (Microsoft-Decision-1 v1) | GlobalStandard, **10 RPM** |
| `text-embedding-3-large` | GlobalStandard, 282K TPM |

Auth is Entra ID only (local auth is disabled): `DefaultAzureCredential`, scope
`https://cognitiveservices.azure.com/.default`.

- gpt-6-luna (Responses API): `https://kz-exp-foundry.openai.azure.com/openai/v1/`
- Microsoft-Decision-1: `https://kz-exp-foundry.services.ai.azure.com/providers/microsoft/v1/systemone`

### Package index note

The user-level uv config uses `packagefeedproxy.microsoft.io`, which lags PyPI. `pyproject.toml`
pins `exclude-newer` to a snapshot the proxy serves. The OpenAI Decisions API is called over raw
HTTP until `openai>=3.26.0` (which adds `client.decisions`) is available.

## Workflow

```powershell
# 1. data: download + frozen grouped/stratified splits (train/dev/calibration; test sealed)
uv run python scripts/prepare_data.py --dataset ace

# 2. predictions (resumable; re-run to continue, --retry-errors to redo failures)
uv run python scripts/run_experiment.py --experiment a1_gpt6_luna b1_decision1 --split dev calibration
uv run python scripts/run_experiment.py --experiment b1_decision1 --catalog verbose --limit 100

# 3. latency (plan defaults: 20 warmups, 200 examples, 5 reps, concurrency 1 and 8)
uv run python scripts/run_latency.py --experiment a1_gpt6_luna --catalog short

# 4. report: artifacts/reports/<dataset>/<split>/report.md (+ CSVs, plots, metrics.json)
uv run python scripts/build_report.py --dataset ace --split dev

# final evaluation only:
uv run python scripts/run_experiment.py --experiment ... --split test --allow-test
```

Outputs: `artifacts/runs/<dataset>/<experiment>__<catalog role>/<split>/predictions.jsonl` plus a
manifest with the config snapshot/hash. The manifest is written before any API call. The hash
covers the method, prompts, catalog, model and data fingerprint, plus
`PREDICTION_LOGIC_VERSION` in `providers/base.py` (bump it when post-processing changes).
Changing any of these refuses to mix outputs (use `--force` to archive the old run), and a
per-split lock prevents two processes from writing the same split. After a hashing-scheme
change, `scripts/migrate_manifests.py` verifies stored predictions and re-stamps manifests.

Experiments declare `supervision: zero_shot | supervised`. Supervised runs fit on the dataset's
train labels and are labelled as such in the leaderboards.

## Experiments

| ID | Config | Status |
|---|---|---|
| A1 | `a1_gpt6_luna` (label only, one-hot) and `a1v_gpt6_luna_verbalized` (JSON + verbalized confidence) | ready |
| A2 | `a2_openai_decisions` | **disabled**: Azure returns 404 for `/openai/v1/decisions`; needs `OPENAI_API_KEY` (OpenAI platform) |
| A3 | – | **infeasible on Azure**: gpt-6-luna rejects logprobs ("not supported with reasoning models") |
| B1 | `b1_decision1` | ready (10 RPM → paced with `max_rpm: 9`) |
| C1–C5 | – | parked until GPU quota is granted |
| S1 | `s1_emb_label` (zero-shot), `s1_emb_centroid`, `s1_emb_knn`, `s1_emb_logreg`, `s1_emb_logreg_fields` | ready (`text-embedding-3-large`; train-split embeddings cached in `artifacts/cache/embeddings/`) |

Each experiment runs once per label catalog role (`short`, `verbose`) available for the dataset.

S1 heads share one method (`type: embedding`): the example is embedded live at prediction time
(honest latency/cost). Learned heads (`centroid`, `knn`, `logreg`) are fit on the `train` split
only; `logreg` picks its regularisation by group-aware cross-validation within train, so
dev/calibration/test stay untouched. `views: fields` embeds each state-template field
separately (ACE: clauses, scenario) and uses `[a, b, a*b, |a-b|]` features.

## Adding a dataset

1. Create `configs/datasets/<name>.yaml`: `files` (url or local path; json/jsonl/csv;
   `records_path` for nested JSON), `fields` (label, optional id/group), a `state.template`
   (`{field}` placeholders, with `mapping`/`list`/`json` formatters), `labels` (list or `auto`),
   `task.instructions`, `splits`, and `label_catalogs` (role → catalog file or `auto`).
2. Optional: `configs/label_catalogs/<catalog>.yaml` with semantic descriptions.
3. `uv run python scripts/prepare_data.py --dataset <name>` and commit
   `configs/datasets/splits/<name>.json`.
4. Run any experiment with `--dataset <name>`.

`banking77.yaml` (77 intents, auto labels) is included as the fallback dataset.

To vet a new dataset for shortcuts, run the artifact probe. It fits grouped-CV
logistic-regression probes on embeddings of each state-template field alone, all fields, and
the full state, using the train split only:

```powershell
uv run python scripts/probe_artifacts.py --dataset <name>
```

## Datasets

ACE (Fujitsu Research, CC BY 4.0) has 3,600 train / 1,100 test examples labelled `Compliant`,
`Non-Compliant` or `Not-Applicable`. The train labels agree 100% with the
`DeepSeek_prediction` teacher field, so the data was apparently filtered for agreement with a
strong LLM. That makes it relatively easy for frontier models. ACE deliberately reuses scenarios
with different clauses (usually with a different label): 550 repeated-scenario groups in train. Derived splits are therefore grouped
by `scenario_text` (no group shared between train/dev/calibration). Splits: train 2161 /
dev 719 / calibration 720. The official test split also shares 360 scenarios with train (always
with different clauses). This is inherent to the benchmark and is reported in the data manifest.

ACE artifact probe (`probe_artifacts.py`, 5-fold grouped CV on train, chance ≈ 0.33):

| Probe input | CV accuracy |
|---|---|
| clauses only | 0.349 |
| scenario only | 0.600 |
| clauses + scenario `[a, b, a*b, \|a-b\|]` | 0.892 |
| full state | 0.850 |

So the scenario wording alone leaks Compliant vs. Non-Compliant (a hypothesis-only artifact),
and a linear probe on frozen embeddings roughly matches frontier LLMs. ACE is therefore weak at
separating decision mechanisms.

The official test split differs from the train distribution. gpt-6-luna reaches ~93% on the
train-derived calibration split but ~70% on test, and Compliant/Non-Compliant recall drops from
~0.91 to ~0.60 (Not-Applicable is unaffected). The test split was apparently not
teacher-filtered and contains contested scenarios. As a result, temperature scaling and
selective-prediction thresholds fitted on the train-derived calibration split are
overconfident on test. Supervised embedding probes reach 0.80–0.84 on test (0.83 on scenarios
never seen in train): they learn ACE's labelling conventions, which zero-shot methods cannot.

Microsoft-Decision-1 shifts much less (verbose labels: 0.79 calibration, 0.74 test) and beats
gpt-6-luna zero-shot on test (0.741 vs. ~0.70). It depends heavily on the label descriptions:
with short labels it almost never predicts `Not-Applicable` (recall ~0.09; 0.52 with verbose
descriptions) and over-predicts `Non-Compliant` (recall ~0.98). In ~1.3% of its answers the top
two probabilities are exactly tied; its own choice (catalog order) is kept and used in metrics.

Note on `--limit`: re-running a split with a different `--limit` or `--sample-seed` adds to the
same prediction file. Reports evaluate every stored prediction, and each leaderboard is
intersected to the examples shared by all its runs.

## Azure ML GPU (parked)

`infra/aml/` holds the provisioning/bootstrap scripts for gpt-oss-20b runs; parked until GPU
quota is granted. Recommended size: `Standard_NC24ads_A100_v4` (1× A100 80 GB).
