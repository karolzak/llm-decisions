# Experiment Plan: Enterprise Text Decisions Benchmark

> This is the original plan. What was actually built and run, and any deviations (A2 disabled
> because Azure does not serve the OpenAI Decisions API; A3 infeasible because GPT-6 Luna
> rejects logprobs; added readouts C2+, C6–C9, a second dataset, BANKING77), are described in
> the [README](README.md), [docs/methods.md](docs/methods.md) and
> [docs/results/](docs/results/). The follow-up study of *how* decision models work (R-series),
> with its findings and next experiments, is in [docs/replication.md](docs/replication.md).

## Background context

This experiment plan comes from recent news about TypeSafe AI's Jev decision model (a "System One" model) and the OpenAI Decisions API, which followed two weeks later.
The goal is to set some baselines and try out different approaches to figure out what Jev/Decisions API is doing behind the scenes and get a better understanding/intuition of how it performs.

## Objective

Build a reproducible benchmark comparing:

1. GPT-6 Luna — standard autoregressive classification
2. OpenAI Decisions API
3. Microsoft-Decision-1 on Azure AI Foundry
4. Open-weight autoregressive baseline using `openai/gpt-oss-20b`
5. Custom non-generative decision approaches using the same `gpt-oss-20b`

Primary metrics:

- Accuracy
- Calibration
- Latency
- Cost

The key experimental question is:

> Holding the underlying model constant, can alternative decision/readout
> mechanisms outperform autoregressive generation for bounded decisions?

## Dataset

Primary: ACE — Assessing Compliance in Enterprise

- 4,700 enterprise contract-compliance scenarios
- 3,600 train / 1,100 test
- Labels:
  - Compliant
  - Non-Compliant
  - Not-Applicable
- Reported model accuracy leaves substantial headroom
- Enterprise/compliance setting
- CC BY 4.0

Fallback: BANKING77

- 13,083 banking-support queries
- 77 fine-grained intents
- Useful if ACE's three choices are too easy for distinguishing approaches

## Core experiment matrix

NOTE: A1 should be run no matter what but A2 and A3 should only be implemented and run after research checking the feasibility based on the availability of dependencies.

| ID | Model | Method |
|---|---|---|
| A1 | GPT-6 Luna on Azure OpenAI | Same autoregressive baseline |
| A2 | GPT-6 Luna on Azure OpenAI | Decisions API (if available) |
| A3 | GPT-6 Luna on Azure OpenAI | Custom, Jev inspired solution based on OpenAI API outputs (logits, etc.?) |
| B1 | Microsoft-Decision-1 | Azure System One API |
| C1 | gpt-oss-20b | Autoregressive classification |
| C2 | gpt-oss-20b | Candidate sequence likelihood |
| C3 | gpt-oss-20b | Shared-state / KV candidate scoring |
| C4 | gpt-oss-20b | Cross-encoder candidate scoring |
| C5 | gpt-oss-20b | Listwise decision head |
| S1 | Embedding model | Bi-encoder similarity baseline |

## Important comparison structure

Produce TWO leaderboards.

### Product leaderboard

Compare:

GPT-6 Luna Azure OpenAI API experiments series
vs
Microsoft-Decision-1
vs
gpt-oss approaches
vs
Bi-encode similarity baseline for sanity checking

Question:

> What works best in practice?

### Controlled readout leaderboard for checking different jev-inspired approaches

Compare only:

gpt-oss-20b autoregressive
vs
gpt-oss-20b sequence scoring
vs
gpt-oss-20b KV scoring
vs
gpt-oss-20b cross-encoder
vs
gpt-oss-20b listwise head

Question:

> If the underlying model is identical, what does changing the
> decision mechanism actually buy us?

## Custom approaches

### C2 — Candidate sequence likelihood

Instead of asking the model to generate:

"The correct answer is Non-Compliant"

score each candidate directly:

score(Compliant | state)
score(Non-Compliant | state)
score(Not-Applicable | state)

Then:

P(choice) = softmax(scores)

This is the simplest approximation of a decision-model readout.

### C3 — Shared-state / KV scoring

Encode the expensive state once:

state -> transformer -> KV cache

Then branch:

KV -> candidate A
KV -> candidate B
KV -> candidate C

Score all candidates.

This tests whether Jev-like shared-state computation provides a meaningful
latency advantage.

Both (C2 and C3 in the README) should produce approximately identical scores.

### C4 — Cross-encoder scoring

For every candidate:

(state, candidate description)
        |
        v
   transformer
        |
        v
      score

This allows deeper semantic interaction between the state and each option.

### C5 — Listwise decision head

Present all candidates simultaneously:

STATE

[A] Compliant
[B] Non-Compliant
[C] Not-Applicable

Extract hidden representations associated with A/B/C and apply the same
small classification head:

h_A -> head -> logit_A
h_B -> head -> logit_B
h_C -> head -> logit_C

softmax(logits)

This is probably the most interesting Jev-inspired experiment.

It allows the options to interact while avoiding autoregressive generation.

Initially freeze the base model and train only the small decision head.

## Verbose option experiment

Run every relevant method twice.

### Short labels

A = Compliant
B = Non-Compliant
C = Not-Applicable

### Semantic criteria

A:
"The described scenario adheres to all governing clauses, including
applicable obligations, conditions, exceptions, and timing requirements."

B:
"At least one governing clause is violated after applying relevant
conditions, exceptions, dependencies, and timing requirements."

C:
"The supplied clauses do not govern the described scenario, so compliance
or violation cannot be established from them."

This tests whether decision models benefit from semantic descriptions of
the decision space rather than opaque labels.

## Hypotheses

H1:
Bounded Decisions APIs improve calibration over autoregressive generation
without materially reducing accuracy.

H2:
On the same gpt-oss-20b model, candidate scoring improves calibration over
autoregressive generation.

H3:
Verbose decision criteria outperform short labels.

H4:
KV reuse produces equivalent predictions to independent candidate scoring
while reducing latency.

H5:
A lightweight listwise decision head closes part of the gap to purpose-built
decision models.

H6:
Calibrated confidence enables useful selective prediction:

high confidence -> automate
low confidence -> escalate

## Calibration metrics

Measure:

- Accuracy
- Macro-F1
- Negative log likelihood
- Brier score
- Expected Calibration Error
- Reliability diagrams
- Risk/coverage curves
- Accuracy at 80%, 90%, 95% coverage

Fit temperature scaling using a dedicated calibration split.

Never calibrate on the test set.

## Latency

Measure:

- p50
- p95
- first-token latency for autoregressive methods
- total request latency
- GPU compute time for local models

Run:

- 20 warmups
- 200-example latency subset
- 5 repetitions
- concurrency 1
- concurrency 8

## Cost

For APIs record actual:

- input tokens
- cached tokens
- output tokens
- provider usage
- request cost

For local models:

cost =
GPU wall-clock seconds
×
GPU hourly price

Also report:

- examples/sec
- tokens/sec
- GPU-seconds/example
- peak VRAM

## Dataset splits

ACE official test remains sealed.

Split the 3,600 training examples:

60% train
20% development
20% calibration

Use:

train -> learned heads
development -> prompts/hyperparameters
calibration -> temperature scaling / thresholds
test -> final evaluation only

## Repository

enterprise-decisions-benchmark/
├── README.md
├── pyproject.toml
├── configs/
│   ├── datasets/
│   ├── models/
│   ├── experiments/
│   └── label_catalogs/
├── src/edb/
│   ├── schemas.py
│   ├── datasets/
│   ├── providers/
│   ├── local/
│   ├── calibration/
│   ├── runners/
│   └── reporting/
├── scripts/
│   ├── prepare_data.py
│   ├── run_preflight.py
│   ├── run_experiment.py
│   └── build_report.py
├── tests/
└── artifacts/

## Implementation order

1. Dataset adapter + frozen splits
2. Common Prediction schema
3. GPT-6 Luna autoregressive baseline
4. gpt-oss-20b autoregressive baseline
5. OpenAI Decisions adapter
6. Microsoft-Decision-1 adapter
7. Candidate sequence scoring
8. KV-cache scoring
9. Cross-encoder scoring
10. Listwise decision head
11. Calibration
12. Latency/cost benchmark
13. Final comparison report
