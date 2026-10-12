# Enterprise Decisions Benchmark

Benchmark of bounded-decision mechanisms on enterprise text decisions: autoregressive
generation vs. decision APIs (OpenAI Decisions, Microsoft-Decision-1) vs. custom
non-generative readouts on `openai/gpt-oss-20b`. See [exp-plan.md](exp-plan.md).

## Layout

```
configs/      datasets, models, experiments, label catalogs (short vs. verbose criteria)
src/edb/      package: settings, providers (Azure), local (gpt-oss), datasets, calibration,
              runners, reporting
scripts/      run_preflight.py (more to come: prepare_data, run_experiment, build_report)
infra/aml/    Azure ML workspace + GPU compute instance provisioning
tests/
artifacts/    run outputs (git-ignored)
```

## Setup (Windows / laptop: API experiments)

Requires [uv](https://docs.astral.sh/uv/). uv provisions Python 3.12 and a local `.venv`.

```powershell
uv sync                       # creates .venv with core + dev deps
Copy-Item .env.example .env   # adjust if needed
az login --tenant 16b3c013-d300-468d-ac64-7eda0820b6d3
uv run python scripts/run_preflight.py --skip-gpu
uv run pytest
```

### Azure resources

| What | Value |
|---|---|
| Subscription | `karzak-MCAPS-Hybrid-Int-REQ-172014-2026` (`005cc313-…`) |
| Resource group | `kz-exp-rg` (swedencentral) |
| Foundry resource / project | `llm-decisions-resource` / `llm-decisions` |
| Deployments | `gpt-6-luna` (GlobalStandard, 500), `Decision-1` = Microsoft-Decision-1 (GlobalStandard, 50) |

Local (API-key) auth is disabled on the resource, so all calls use Entra ID tokens
(`DefaultAzureCredential`, scope `https://cognitiveservices.azure.com/.default`). Your identity
needs a data-plane role such as **Cognitive Services User** on the resource.

Endpoints used:

- gpt-6-luna (Responses API): `https://llm-decisions-resource.openai.azure.com/openai/v1/`
- Microsoft-Decision-1: `https://llm-decisions-resource.services.ai.azure.com/providers/microsoft/v1/systemone`
- OpenAI Decisions API probe (A2): `…/openai/v1/decisions`

### Package index note

The user-level uv config points at `packagefeedproxy.microsoft.io`, which lags PyPI by
several days. `pyproject.toml` pins `exclude-newer` to a snapshot the proxy covers; bump it
(and `openai>=3.26.0` for the SDK's `client.decisions`) when needed, then `uv lock`.

## Setup (Azure ML GPU compute: gpt-oss-20b experiments)

```powershell
powershell -ExecutionPolicy Bypass -File .\infra\aml\provision.ps1 -CheckOnly   # quota only
powershell -ExecutionPolicy Bypass -File .\infra\aml\provision.ps1 -Size Standard_NC24ads_A100_v4
```

`provision.ps1` creates the workspace (`kz-exp-aml`), a compute instance with idle shutdown
(60 min) and a system-assigned identity, and grants that identity **Cognitive Services User**
on the Foundry resource. Then, on the compute instance terminal:

```bash
git clone <this repo> && cd llm-decisions
bash infra/aml/bootstrap-env.sh   # uv sync --extra local, model download, preflight
```

> Azure ML **dedicated GPU quota is 0** for A10/A100/H100 families in swedencentral on this
> subscription. Request quota (Azure ML studio → Quotas) for the family you choose before
> provisioning.

## Dataset

ACE (Fujitsu Research, CC BY 4.0): 3,600 train / 1,100 test, labels `Compliant`,
`Non-Compliant`, `Not-Applicable`. The official test split stays sealed; train is split
60/20/20 into train/dev/calibration (see `configs/datasets/ace.yaml`).
