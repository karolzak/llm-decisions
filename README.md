# LLM decisions benchmark

How do "decision models" such as Jev, the OpenAI Decisions API and Microsoft-Decision-1
make bounded decisions, and how far can you get without one? A **bounded decision** gives the
model a state (e.g. contract clauses plus a scenario, or a customer message) and a fixed list
of options, and asks for a probability per option instead of free text.

This repository is a reproducible, dataset-agnostic benchmark of the ways to get such
decisions. It measures **accuracy, calibration** (do the probabilities mean what they say?),
**selective prediction** (can it abstain when unsure?), **latency** and **cost**.

## Two separate experiments

| | Part 1: products and baselines | Part 2: open-weight model |
|---|---|---|
| Experiments | A (GPT-6 Luna), B (Microsoft-Decision-1), S (embeddings) | C1–C9 on `openai/gpt-oss-20b` |
| What we control | Only the request; the hosted models are opaque | Everything: same weights in every run; only the *readout* changes |
| Question | Which product works best for bounded decisions? | Holding the model fixed, what does the readout alone buy? |

A **readout** is how a decision is extracted from a model: generate text and parse it, score
each label's likelihood, read one token's probabilities, or apply a small trained layer to
internal activations. The two parts use different models and infrastructure, so they are
reported separately and never ranked against each other.

## Models

| Model | Where | Used as |
|---|---|---|
| GPT-6 Luna | Azure OpenAI (Responses API, reasoning `medium`) | A1: generate the label; A1v: label + verbalized confidence |
| Microsoft-Decision-1 v1 | Microsoft Foundry decision API (`/providers/microsoft/v1/systemone`) | B1: state + options → probability per option |
| text-embedding-3-large | Azure OpenAI | S1: label similarity (zero-shot), centroid / kNN / logistic regression (supervised) |
| gpt-oss-20b | Open weights (Hugging Face), bf16 on one 80 GB GPU via Azure ML | C1–C9 readouts (reasoning effort `low`) |
| Qwen3.5-9B / Qwen3.5-2B | Open weights, bf16 via Azure ML | R2 frozen letter readout (9B is Decision-1's base); R3 LoRA training arms (2B) |
| GPT-6 Luna Decisions API | OpenAI platform | A2: implemented but disabled (not served by Azure yet) |

## Experiments

| ID | What it does | Supervision |
|---|---|---|
| A1 / A1v | GPT-6 Luna generates the label (A1v adds a verbalized confidence) | zero-shot |
| A2 | OpenAI Decisions API on GPT-6 Luna: **disabled**, Azure returns 404 | zero-shot |
| A3 | Logprob readout on GPT-6 Luna: **infeasible**, reasoning models reject logprobs | – |
| B1 | Microsoft-Decision-1 choice question | zero-shot |
| S1 | Embeddings: label similarity; centroid, kNN, logistic regression (also per field) | zero-shot / supervised |
| C1 | Generate with reasoning, parse the label (one-hot) | zero-shot |
| C2 / C3 | Score each label as the answer continuation; C3 shares the prompt's KV cache | zero-shot |
| C2+ | C2 minus a content-free prior | zero-shot |
| C4 | Cross-encoder: Yes/No per candidate | zero-shot |
| C5 | Listwise linear head on frozen hidden states of the listed options | supervised |
| C6 / C6+ | List options, read the option-letter distribution in one pass (C6+: order-averaged + prior) | zero-shot |
| C7 | Early exit: linear head on the answer position after 16–20 of 24 layers | supervised |
| C8 | Reason first, then read the option-letter probabilities | zero-shot |
| C9 | Offline cascade: C6+, escalate to C8 when unsure | zero-shot |
| D2 | Diagnostic: unbatched C2 (bf16 numerics check, no leaderboard) | – |
| R0 | Probes: perturbed requests to Decision-1 (Jev / OpenAI Decisions with a key) + forensics on the returned numbers | – |
| R2 | Frozen Qwen3.5-9B (Decision-1's base) and Qwen3.5-2B, option-letter readout, compared example by example with Decision-1 | zero-shot |
| R3 | Qwen3.5-2B + LoRA trained on 8 other public tasks (shuffled vs. fixed option order, CE vs. Brier loss), evaluated on ACE/BANKING77 | zero-shot (not trained on these tasks) |

Every experiment runs once per label catalog: `short` (label names) and, where available,
`verbose` (names + one-sentence criteria). Plain-language explanations with a diagram per
experiment: [docs/methods.md](docs/methods.md). The R-series (how decision models work):
[docs/replication.md](docs/replication.md); results:
[Decision-1 probes](docs/results/2026-10-20-decision1-probes/README.md) and
[frozen base + training arms](docs/results/2026-10-21-replication/README.md).

## Datasets

| Dataset | Task | Options | Splits (train / dev / calibration / test) | License |
|---|---|---|---|---|
| [ACE](https://github.com/FujitsuResearch/Fujitsu-Assessing-Compliance-in-Enterprise-Dataset) (Fujitsu Research) | Is the scenario compliant with the contract clauses? | 3: `Compliant`, `Non-Compliant`, `Not-Applicable`; short + verbose catalogs | 2,161 / 719 / 720 / 1,100 official test | CC BY 4.0 |
| [BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets) (PolyAI) | Which banking-support intent matches the message? | 77 intents (short catalog only) | 6,002 / 2,001 / 2,000 / 3,080 official test | CC BY 4.0 |

Dev and calibration are carved out of the official train set with frozen, grouped splits
(`configs/datasets/splits/`); the official test set stays sealed until final runs.

**ACE caveats.**
- The scenario text alone partly gives away the label: a probe on it scores 0.60 vs. 0.33
  chance.
- The test set is noticeably harder than the train-derived splits.

Details: [ACE baseline](docs/results/2026-10-13-ace-baseline/README.md).

New datasets need only a YAML config: see [docs/setup.md](docs/setup.md#adding-a-dataset).

## Key results (test sets)

| Part 1 | ACE | BANKING77 |
|---|---|---|
| GPT-6 Luna (best variant) | 0.707 | 0.829 |
| Microsoft-Decision-1 (best catalog) | **0.741** | **0.870** |
| Embeddings + logistic regression (supervised) | 0.844 | 0.944 |

| Part 2 (gpt-oss-20b) | ACE | BANKING77 |
|---|---|---|
| C1 generate (baseline) | 0.757 | 0.806 |
| C6 one-pass letter readout | 0.589 | 0.791 |
| C8 reason, then read | 0.753 | **0.831** |
| C9 cascade | **0.769** | 0.827 |
| C7 early-exit head (supervised) | 0.931 | 0.908 |

**Findings.**
- **Decision-1** is the best zero-shot product on both datasets, and its probabilities are
  calibrated out of the box.
- **Within gpt-oss-20b**, a one-pass readout nearly matches generation on intent routing (at
  1/8 of the latency), but collapses on compliance, which needs reasoning.
- **The decision is already in the mid-layer activations:** a linear head trained on them
  reaches 0.93 on ACE.
- **How Decision-1 works (R-series):** it is one forward pass over listed options with a
  bf16 probability readout, blind to option positions. On ACE its decisions match its
  *frozen* base model (Qwen3.5-9B read in one pass: 0.741 vs. 0.741). Post-training pays
  off on many-way routing (BANKING77 +12 points) and calibration. Small decision-trained
  models reproduce most of its traits; see
  [docs/results/2026-10-21-replication](docs/results/2026-10-21-replication/README.md).

Full results, calibration metrics, latency and cost:
[docs/results/2026-10-14-c-series-banking77](docs/results/2026-10-14-c-series-banking77/README.md).

## Repository map

```
README.md               this overview
exp-plan.md             original experiment plan and hypotheses (H1–H6)
presentation.html       slide deck for the team (open in a browser; arrow keys to navigate)
docs/
  methods.md            how every method works, one diagram per experiment
  replication.md        R-series: probes, frozen-base comparison and training arms
  setup.md              install, .env, workflow, outputs, adding a dataset
  azure-ml.md           running the C-series as Azure ML GPU jobs
  results/              dated result snapshots (write-ups, generated reports, plots)
configs/
  datasets/             dataset configs + frozen split assignments (splits/)
  label_catalogs/       label sets with optional descriptions (short vs. verbose)
  models/               deployments, local model settings, pricing
  experiments/          one YAML per experiment: method, model, catalogs, leaderboard, rate limits
  probe_targets/        decision APIs and local readouts to probe (decision-1, jev, ...)
  training/             R3 corpus sources and training arms
src/edb/
  datasets/             loading, state rendering, grouped/stratified splits
  providers/            API methods (A, B, S), auth, retries, rate limiting
  local/                gpt-oss-20b readouts (C1–C8) and the chat-model letter readout (R2/R3)
  probing/              vendor-neutral decision-API client, probes, forensics, probe report
  training/             LoRA, decision corpus, training loop (R3)
  calibration/          metrics (accuracy, NLL, Brier, ECE, AURC) and temperature scaling
  runners/              resumable prediction runner, latency benchmark, Azure ML submission
  reporting/            leaderboards, reliability and risk-coverage plots, markdown reports
scripts/                prepare_data, run_preflight, run_experiment, run_latency, build_cascade,
                        build_report, aml_submit, aml_run, aml_collect, check_pilot,
                        probe_artifacts, run_probes, probe_report, compare_runs, train_decider
infra/aml/              Azure ML job entry point, pinned requirements, optional setup scripts
tests/                  unit tests (incl. every gpt-oss readout on a tiny random gpt-oss model)
artifacts/              run outputs, caches and reports (git-ignored)
```

## Quick start

```powershell
uv sync
Copy-Item .env.example .env        # fill in your Foundry resource / project (and Azure ML for the C-series)
az login
uv run python scripts/run_preflight.py --skip-gpu
uv run python scripts/prepare_data.py --dataset ace
uv run python scripts/run_experiment.py --experiment b1_decision1 --split dev --limit 50
uv run python scripts/build_report.py --dataset ace --split dev
```

No Azure subscription details are stored in the repository: everything environment-specific
comes from `.env`. Full instructions are in [docs/setup.md](docs/setup.md); GPU runs are in
[docs/azure-ml.md](docs/azure-ml.md).
