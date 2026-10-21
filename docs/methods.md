# How each method works

This page explains every decision mechanism in the benchmark, with one diagram per experiment.
Results: [C-series + BANKING77 (2026-10-14)](results/2026-10-14-c-series-banking77/README.md).

All methods answer the same bounded question: given a **state** (e.g. contract clauses plus a
scenario, or a customer message) and a fixed list of **K options** (labels, optionally with a
description), return a probability for each option.

The methods fall into separate parts that are evaluated and reported separately, never
ranked against each other:

- **Part 1 — products and baselines (A, B, S):** hosted models called through APIs (GPT-6 Luna,
  Microsoft-Decision-1, `text-embedding-3-large`). We control only the request.
- **Part 2 — open-weight model (C):** one local model, `openai/gpt-oss-20b`, with identical
  weights in every experiment. Only the readout changes.
- **Part 3 — replication study (R):** probes of Decision-1 itself, its frozen base model
  (Qwen3.5-9B) and small trained deciders, to understand how decision models work.

## Key terms

- **Readout:** how a decision is extracted from a model. The same network can be asked to write
  the answer as text, have each label's likelihood scored, have one token's probabilities read
  after listing the options, or have a small trained layer applied to its internal activations.
  Each of these is a different readout of the same model.
- **Calibration:** whether the probabilities mean what they say. If a method says "90% sure" on
  100 decisions, about 90 of them should be correct. Accuracy says how often the top answer is
  right; calibration says whether the confidence can drive a policy (automate above a
  threshold, escalate below it). We measure it with ECE (gap between stated confidence and
  observed accuracy) and NLL (log-loss), and fix systematic over/under-confidence with
  *temperature scaling* fitted on the separate calibration split. AURC measures a related
  property: whether errors get lower confidence than correct answers.
- **Zero-shot vs. supervised:** supervised methods fit parameters on the dataset's `train`
  labels and are marked as such in every leaderboard. Zero-shot methods use no labels.

# Part 1 — Products and baselines (A, B, S)

| Experiment | Model | What happens per decision | Trained on labels? |
|---|---|---|---|
| A1, A1v | GPT-6 Luna (Azure OpenAI) | 1 generation: reasoning + answer text, parsed to a label | no |
| B1 (A2 blocked on Azure) | Microsoft-Decision-1 (Azure AI Foundry) | 1 decision-API call returning a probability per option | no |
| S1 | text-embedding-3-large | 1 embedding call + similarity or a small classifier | label similarity: no; centroid/kNN/logreg: **yes** |

## A1 / A1v — GPT-6 Luna, generate the label (API)

```mermaid
flowchart LR
  P["Prompt:<br/>task + labels + state"] --> L["GPT-6 Luna<br/>Responses API<br/>reasoning = medium"]
  L --> T["reasoning tokens<br/>(hidden)"]
  T --> A["answer text"]
  A --> X{"parse"}
  X -- "A1: label only" --> O1["one-hot<br/>P = 1.0 for the label"]
  X -- "A1v: JSON label + confidence" --> O2["confidence c on the label,<br/>(1 - c) spread over the rest"]
```

The standard way to classify with an LLM. A1 returns only a label, so its "probabilities" are
one-hot. A1v asks the model to also state a confidence (verbalized confidence). GPT-6 Luna does
not expose logprobs for reasoning models, so a Jev-style readout on top of it (A3) is not
possible on Azure.

## B1 — Microsoft-Decision-1 (decision API)

```mermaid
flowchart LR
  S["state"] --> D["Microsoft-Decision-1<br/>choice question"]
  O["options<br/>(+ descriptions)"] --> D
  D --> P["probability per option"]
```

A purpose-built decision model on Azure AI Foundry: the state and the options go in, a
distribution comes out. No text is generated or parsed. Its internals are not documented. The
OpenAI Decisions API (A2, the same idea on GPT-6 Luna) returns 404 on Azure and is disabled.

## S1 — Embedding baselines

```mermaid
flowchart LR
  S["state"] --> E["text-embedding-3-large"]
  E --> V["state vector"]
  V --> Z["label: cosine similarity to<br/>embedded label text (zero-shot)"]
  V --> C["centroid: similarity to class means"]
  V --> K["kNN: similarity-weighted vote<br/>of 25 nearest train examples"]
  V --> R["logreg: logistic regression"]
  F["per-field vectors a, b<br/>[a, b, a*b, |a-b|]"] --> RF["logreg_fields"]
```

Bi-encoder baselines: the state is embedded once; options are not seen by the encoder (except in
zero-shot label similarity). Centroid, kNN and logistic-regression heads are fit on train
(`supervised`). `logreg_fields` embeds each state field separately (ACE: clauses and scenario)
so the probe can model their interaction.

---

# Part 2 — Open-weight model: gpt-oss-20b readouts (C)

| Readout family | Experiments | Model calls per decision | Trained on labels? |
|---|---|---|---|
| Generate text, parse the label | C1 | 1 generation (reasoning + answer) | no |
| Score each label as a continuation | C2, C3, C2+ | K label continuations (C3 shares the prompt) | no |
| Judge each candidate Yes/No | C4 | K short continuations (shared prompt) | no |
| List the options, read one token | C6, C6+ | 1 forward pass (C6+: 3–4 orderings) | no |
| Reason, then read one token | C8 | 1 generation of the reasoning + 1 read | no |
| Cascade | C9 | C6+ always, C8 only when unsure | no (threshold fit on calibration) |
| Linear head on hidden states | C5, C7 | 1 forward pass (C7: only the first L layers) | **yes** (train split) |

## Shared prompt conventions (gpt-oss-20b)

All C-methods use the same local model (`openai/gpt-oss-20b`: 24 layers, bf16 on one A100 80 GB)
and the same [harmony](https://github.com/openai/harmony) chat format: a system message (fixed
date, reasoning effort `low`), developer instructions (the task plus either the label list or a
listed-options instruction) and the state as the user message. The prompts are generic and come
from the dataset config; nothing is specific to a dataset.

gpt-oss writes a private **analysis** channel (its reasoning) before the **final** channel (the
answer). Several methods skip the reasoning entirely by placing the model directly at the start
of the final answer (`<|start|>assistant<|channel|>final<|message|>`) and reading probabilities
there.

Two prompt layouts are used:

```text
Label prompt (C1, C2, C3, C2+)            Listed-options prompt (C5, C6, C6+, C7, C8)
------------------------------            -------------------------------------------
task instructions                         task instructions
allowed labels (+ descriptions)           "answer with the letter of the correct option"
state                                     state
                                          Options:
                                          [A] Compliant: <description>
                                          [B] Non-Compliant: <description>
                                          [C] Not-Applicable: <description>
-> answer is the label text               -> answer is a single marker token (A..Z, or 1..K for K > 26)
```

Every marker (`A`–`Z`, `1`–`100`) is a single token in the gpt-oss tokenizer, so one next-token
distribution gives a probability for every option.

## C1 — Autoregressive generation (gpt-oss-20b)

```mermaid
flowchart LR
  P["Label prompt<br/>+ state"] --> M["gpt-oss-20b<br/>all 24 layers"]
  M --> R["generate reasoning<br/>(analysis channel)"]
  R --> F["generate final answer<br/>e.g. 'Non-Compliant'"]
  F --> X{"exact catalog label?"}
  X -- yes --> O["one-hot probabilities"]
  X -- no --> E["error: invented label"]
```

The standard way to classify with an LLM, and the reference point for every other C readout. Greedy
decoding with reasoning effort `low` (median 29 generated tokens on BANKING77, 67–70 on ACE,
reasoning included). The answer can be any text,
so it can name a label that does not exist (this happened 30 times on BANKING77, e.g.
`get_virtual_card` instead of `getting_virtual_card`).

## C2 — Candidate sequence likelihood

```mermaid
flowchart LR
  P["Label prompt + state<br/>+ start of final answer"] --> S1["... + 'Compliant' + end"]
  P --> S2["... + 'Non-Compliant' + end"]
  P --> S3["... + 'Not-Applicable' + end"]
  S1 --> L1["log p(label tokens)"]
  S2 --> L2["log p(label tokens)"]
  S3 --> L3["log p(label tokens)"]
  L1 --> SM["softmax<br/>over K labels"]
  L2 --> SM
  L3 --> SM
```

No generation: the model is placed at the start of its final answer and each label is *forced*
as the continuation (followed by the end-of-answer token). The summed token log-probability of
each label becomes its score, and a softmax over the scores gives the distribution. Each
candidate is a separate sequence (batched, 16 per batch), so the prompt is processed K times.

## C3 — Shared-prefix (KV-cache) scoring

```mermaid
flowchart LR
  P["Label prompt + state<br/>+ start of final answer"] --> KV["encode ONCE<br/>keep the KV cache"]
  KV --> B1["branch: 'Compliant' + end"]
  KV --> B2["branch: 'Non-Compliant' + end"]
  KV --> B3["branch: 'Not-Applicable' + end"]
  B1 --> SM["same scores as C2<br/>softmax over K labels"]
  B2 --> SM
  B3 --> SM
```

Mathematically identical to C2, but the shared prompt is encoded once and its key/value cache is
reused for every candidate. Only the label tokens are processed per candidate. This tests whether
"encode the state once, score the options cheaply" (the shared-state idea behind decision
models) changes the answers. Within bf16 rounding it does not.

## C2+ — Sequence likelihood with a content-free prior

```mermaid
flowchart LR
  subgraph setup["once, at setup"]
    N["Label prompt with state = 'N/A'"] --> PR["prior: log p(label | no content)"]
  end
  P["Label prompt + real state"] --> C2["C2 scores<br/>log p(label | state)"]
  C2 --> D["subtract the prior per label"]
  PR --> D
  D --> SM["softmax over K labels"]
```

Contextual calibration (Zhao et al., 2021): measure how much the model likes each label when the
input carries no information, and subtract that bias. Known flaw: the content-free input `N/A`
reads as "not applicable", so on ACE the prior almost entirely favours the `Not-Applicable` label
and the correction removes that label from the predictions (see the results write-up).

## C4 — Cross-encoder (per-candidate Yes/No)

```mermaid
flowchart LR
  P["'Is this candidate the correct decision?<br/>Answer Yes or No' + state"] --> KV["shared prefix<br/>KV cache"]
  KV --> C1["Candidate: Compliant (+ description)"]
  KV --> C2["Candidate: Non-Compliant (+ description)"]
  KV --> C3["Candidate: Not-Applicable (+ description)"]
  C1 --> M1["log p(Yes) - log p(No)"]
  C2 --> M2["log p(Yes) - log p(No)"]
  C3 --> M3["log p(Yes) - log p(No)"]
  M1 --> SM["softmax over margins"]
  M2 --> SM
  M3 --> SM
```

Like a reranker: the model judges each candidate on its own and never sees the alternatives.
The Yes-vs-No margin is the candidate's score.

## C5 — Listwise head on frozen hidden states (supervised)

```mermaid
flowchart LR
  P["Listed-options prompt<br/>state + [A] ... [B] ... [C] ..."] --> M["one forward pass<br/>frozen gpt-oss-20b"]
  M --> H["hidden state at the end of each option line<br/>+ at the answer position (layer L)"]
  H --> W["one shared linear scorer<br/>w · [h_option, h_option * h_answer]"]
  W --> SM["softmax over options"]
  T["train labels"] -. "fit w; choose layer L and L2 on dev" .-> W
```

All options are in the prompt, so because of causal attention each option's representation has
read the state and the options before it (listwise). The base model stays frozen; only a single
weight vector shared by all options is trained. The layer is chosen on dev from {8, 12, 16, 20, 24}:
layer 16 on ACE and 24 on BANKING77.

## C6 — Option-letter readout (one forward pass)

```mermaid
flowchart LR
  P["Listed-options prompt<br/>state + [A] ... [B] ... [C] ...<br/>+ start of final answer"] --> M["ONE forward pass<br/>gpt-oss-20b"]
  M --> V["next-token distribution<br/>over the whole vocabulary"]
  V --> K["keep only the marker tokens<br/>A, B, C (or 1..77)"]
  K --> SM["renormalise<br/>= P(option)"]
```

The closest zero-shot analogue of a "System One" decision: one parallel pass over the prompt, no
generated tokens, a full distribution over the options at once. The cost does not depend on the
number of options beyond listing them in the prompt.

## C6+ — Debiased letter readout

```mermaid
flowchart LR
  S["state"] --> R1["order 1: A=Compliant, B=Non-Compliant, C=Not-Applicable"]
  S --> R2["order 2: rotated"]
  S --> R3["order 3: rotated"]
  R1 --> L["log P(option) per ordering<br/>(shared-prefix KV)"]
  R2 --> L
  R3 --> L
  L --> AV["average per option<br/>- content-free prior"]
  AV --> SM["softmax"]
```

Removes two known biases of letter readouts: **position** bias (averages over cyclic orderings of
the options: 3 on ACE, 4 on BANKING77) and **label prior** bias (subtracts the content-free prior,
as in C2+, with the same `N/A` flaw on ACE).

## C7 — Early-exit question head (supervised)

```mermaid
flowchart LR
  P["Listed-options prompt"] --> L1["layers 1..L only<br/>(L = 16 or 20 of 24)"]
  L1 --> H["hidden state at the<br/>answer position"]
  H --> W["multinomial logistic head<br/>(hidden size -> K)"]
  W --> SM["P(option)"]
  T["train labels"] -. "fit; choose L and L2 on dev" .-> W
  L1 -. "skipped" .-> X["layers L+1..24<br/>and LM head"]
  classDef skipped fill:#f4f4f4,stroke:#999,stroke-dasharray:4 3,color:#777
  class X skipped
```

Tests whether the decision is already formed partway through the network. The forward pass is
truncated at layer L, so inference is cheaper than a full pass. L is chosen on dev from
{12, 16, 20}: 16 (ACE short), 20 (ACE verbose, BANKING77).

## C8 — Reason, then read

```mermaid
flowchart LR
  P["Listed-options prompt"] --> G["generate reasoning<br/>(analysis channel)"]
  G --> S["stop as soon as the model<br/>opens the final channel"]
  S --> R["read next-token probabilities<br/>of the option markers"]
  R --> SM["P(option)"]
```

Keeps the benefit of reasoning but replaces sampling the answer with a probability readout at the
answer position. This turns a one-hot answer into a real distribution (for confidence
thresholds), and the answer can never be an invented label.

## C9 — Adaptive cascade (offline)

```mermaid
flowchart LR
  X["example"] --> F["C6+ fast readout"]
  F --> Q{"max P >= t ?"}
  Q -- "yes" --> A["use the C6+ answer"]
  Q -- "no" --> S["C8 reason-then-read"]
  S --> B["use the C8 answer"]
```

Answer quickly when the fast readout is confident; spend reasoning only on hard cases. Built from
the stored C6+ and C8 predictions by `scripts/build_cascade.py` (no new model calls). The
threshold `t` is fit on the calibration split: the lowest escalation rate whose accuracy is within
0.5 points of C8 alone. Latency and cost are C6+ plus C8 for the escalated examples.

## D2 — Unbatched sequence scoring (diagnostic)

```mermaid
flowchart LR
  P["Label prompt + state"] --> S1["forward: + 'Compliant'"]
  P --> S2["forward: + 'Non-Compliant'"]
  P --> S3["forward: + 'Not-Applicable'"]
  S1 --> C["compare with batched C2<br/>and KV-shared C3"]
  S2 --> C
  S3 --> C
```

C2 with one forward pass per candidate (no batching or padding). It is not on the leaderboard;
it showed that C2 vs. C3 differences come from bf16 numerics that depend on the batch shape
(median about 0.1 nats), not from a bug.

---

# Part 3 — How do decision models work? (R-series)

Full description, hypotheses and commands: [replication.md](replication.md).

```mermaid
flowchart LR
  subgraph R0["R0 probes (API)"]
    P["perturbed requests:<br/>reorder, rename, duplicate options,<br/>change the question, empty state,<br/>2 questions, K = 2..256"] --> D1["Decision-1<br/>(Jev, OpenAI Decisions)"]
    D1 --> F["forensics: bf16 logit grid,<br/>confidence formula, tokens"]
  end
  subgraph R2["R2 frozen base"]
    Q9["Qwen3.5-9B, untrained<br/>letter readout"] --> CMP["example-by-example<br/>comparison with Decision-1"]
  end
  subgraph R3["R3 training arms"]
    C["8 public tasks<br/>(not ACE / BANKING77)"] --> L["Qwen3.5-2B + LoRA<br/>shuffle vs. fixed order,<br/>CE vs. Brier"]
    L --> E["zero-shot ACE / BANKING77<br/>+ the same probes"]
  end
```

## R2 / R3 — Chat-model letter readout (`local_chat_readout`)

```mermaid
flowchart LR
  P["one user message:<br/>&lt;state&gt; &lt;question&gt;<br/>A. option: criterion<br/>B. ..."] --> T["model's own chat template<br/>(thinking off)"]
  T --> M["ONE forward pass"]
  M --> V["next-token distribution"]
  V --> S["sum every token that decodes<br/>to each letter (A, ' A', ...)"]
  S --> SM["renormalise over the K letters"]
```

The recipe of the open "System One" replicas that read instruct models (Cygnet, SemIf, Quyet,
H2O-Lightning). Unlike C6, it uses a non-reasoning chat template (no skipped analysis channel)
and sums all token variants of each letter. More than 26 options are scored in groups, then
the group winners against each other. R3 trains a LoRA adapter on exactly this prompt and
readout.
