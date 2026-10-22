# Question types, shared state, paraphrases and option order: R14, R16, R6, R18 (2026-10-22)

Follow-up to probe finding 10 ([2026-10-20-decision1-probes](../2026-10-20-decision1-probes/README.md#what-this-says-about-the-architecture)):
the decision APIs answer three question types (`choice`, `score`, `noul`) about one state.
These experiments ask how Decision-1 answers `score` and `noul` questions, how it can answer
many questions per request quickly, and how robust it is to rewording.
Plan and decision rules: [replication.md](../../replication.md#next-experiments).

| Experiment | What | Data |
|---|---|---|
| R14 | Decision-1 vs. the frozen Qwen3.5-9B with 4 yes/no and 3 score templates | 200 single-question requests ([items.jsonl](items.jsonl), Decision-1 answers in [decision1.jsonl](decision1.jsonl)) |
| R16 | Many questions about one state from one cached prompt prefix, batched | 4 BANKING77 states × 64 questions, frozen 9B on one A100 |
| R6 | Three paraphrases of the question plus a minimal one ("Choose the best option.") | 50 dev items each from ACE and BANKING77 (10-option subsets); Decision-1 and frozen 9B |
| R18 | Does Decision-1 reorder the options? Listing-order questions, and duplicates named to sort first | 96 Decision-1 requests; frozen 9B probes with sorted options |

## R14: per-type templates

**Items:**
- **BANKING77 yes/no:** 40 dev messages; "Is the customer asking about *intent*?", where
  *intent* is the true intent half of the time. Each is asked twice: plain, and with true/false
  criteria.
- **SST-5 reviews:** 40 validation reviews, each asked as a 5-level score, a 3-level score,
  and the yes/no question "Is this movie review positive?". Reviews labelled neutral have no
  gold answer for the yes/no question.
- SST-5 is not in any training mix.

**Decision-1 on these items:**

| Subset | Accuracy | Mean confidence | Billed input tokens |
|---|---|---|---|
| BANKING77 yes/no, plain | 0.925 | 0.968 | 41 |
| BANKING77 yes/no, with criteria | 0.925 | 0.974 | 65 |
| SST yes/no (non-neutral reviews) | 0.931 | 0.939 | 46 |
| SST 3-level score | 0.700 | 0.845 | 55 |
| SST 5-level score | 0.450 | 0.687 | 61 |

- **The bf16 grid holds at scale:** 198 of 200 answers have log-scores on the 0.125 grid.
- **Criteria barely matter:** adding true/false criteria changes the yes/no log-odds very
  little (r = 0.994, mean shift −0.17, same answer on all 40).
- **`score` tracks the gold level:** r = 0.81 with the gold index, on both 3 and 5 levels.

**Frozen Qwen3.5-9B with each template vs. Decision-1** (job `bubbly_oil_14xcdn1wxy`):

Yes/no:

| Template | Log-odds r with Decision-1 (BANKING77 plain / criteria / SST) | Mean abs. log-odds diff | Same answer | Accuracy (BANKING77 plain / SST) |
|---|---|---|---|---|
| letters, `false` listed first (our default) | 0.969 / 0.987 / 0.952 | 1.87 / 1.03 / 0.83 | 0.975 | 0.900 / 0.931 |
| letters, `true` listed first | 0.960 / 0.987 / 0.963 | 1.36 / 0.93 / 1.07 | 0.95–1.00 | 0.925 / 0.862 |
| `true` / `false` words | 0.961 / 0.977 / 0.962 | 1.29 / 1.23 / **0.69** | 0.95–0.975 | 0.875 / 0.931 |
| `yes` / `no` words | 0.960 / 0.968 / 0.956 | 1.64 / 1.95 / 0.86 | 0.95–1.00 | 0.875 / 0.931 |

Score:

| Template | Mean TV to Decision-1 (3 / 5 levels) | Same answer (3 / 5) | Expected-score r (3 / 5) | Accuracy (3 / 5) |
|---|---|---|---|---|
| letters A, B, C… (our default) | 0.193 / 0.247 | 0.78 / 0.75 | 0.949 / 0.956 | 0.625 / 0.450 |
| digits `0.`, `1.`… as options | **0.138** / 0.244 | **0.90** / 0.73 | **0.964** / 0.958 | 0.700 / 0.450 |
| a `0 = …` scale | 0.149 / **0.237** | **0.90** / **0.80** | 0.955 / **0.965** | 0.650 / 0.475 |

- **No template is clearly Decision-1's.** Every template correlates 0.95–0.99 with Decision-1's
  log-odds and gives the same yes/no answer on 95–100% of items. The R14 decision rule
  ("one template clearly higher") is not met.
- **Digit-labelled score levels fit better than letters on 3 levels.** TV 0.14–0.15 vs. 0.19,
  same answer 90% vs. 78%. They are equal on 5 levels. This is consistent with Decision-1
  keying levels by digit (finding 10).
- **The frozen base already answers yes/no and score questions like Decision-1:** same
  accuracy within one or two items, high correlation. This extends F7 (the judgement is the
  base model's) from choice to all three types.
- **After training on all three types (R15, `truefalse` + `digits`):**
  - **Score:** moves slightly towards Decision-1 (5 levels: same answer 0.725 → 0.825).
  - **Yes/no:** log-odds become more extreme than Decision-1's (mean abs. difference on
    BANKING77 1.23–1.29 → 1.89–2.30).
  - See [replication.md](../../replication.md) (F22).

## R16: many questions from one state

`ChatLetterReadout.answer_shared` encodes the prompt prefix the questions share (template
and state) once. It then reads up to 32 questions per forward pass, with the cached prefix
repeated across the batch and the question suffixes right-padded. This includes Qwen3.5's
linear-attention conv and recurrent states.

| Frozen 9B, 64 questions per state (62 yes/no, 1 choice, 1 score) | One at a time | Shared prefix, sequential (R16) | Shared prefix, batched (R16b) |
|---|---|---|---|
| Wall time per state (A100) | 3.8–4.2 s | 3.9 s | **0.83–0.89 s** |
| Prompt tokens processed | 5,531–5,995 | 2,822–2,847 | 2,822–2,847 |
| Max. probability difference vs. one at a time | – | 0.011–0.044 | 0.008–0.048 |
| Same answer as one at a time | – | 98–100% | 98–100% |

- **Decision-1's speed is matched in order of magnitude:** 64 questions in under 0.9 s of GPU
  time, against Decision-1's 0.99 s for 64 yes/no questions measured from our client (which
  includes the network round trip). This is on reference (non-fused) kernels for the
  linear-attention layers, so a tuned server would be faster.
- **Caching alone does not explain the speed.** It halves the tokens but not the time; batching
  the questions does.
- **The bf16 differences match Decision-1's noise.** In fp32 on CPU, shared and separate answers
  agree to within 3×10⁻⁶. In bf16 on the GPU they differ by up to 0.05, which is the size of
  Decision-1's request-to-request variation (probe finding 7). This supports "batched bf16
  serving" as the source of that variation.

## R6: paraphrased questions

Decision-1 run: 500 requests, 0 errors. Frozen 9B: job `silly_head_4hdt71ddk0`, grouped-letter readout, default template.

| | Decision-1, ACE | Frozen 9B, ACE | Decision-1, BANKING77 | Frozen 9B, BANKING77 |
|---|---|---|---|---|
| Accuracy, original wording | 0.84 | 0.88 | 0.96 | 0.96 |
| Accuracy, the three paraphrases | 0.82 / 0.84 / 0.90 | 0.84 / 0.86 / 0.86 | 0.96 / 0.96 / 0.94 | 0.94 / 0.94 / 0.96 |
| Decision flips per paraphrase, mean | **6.7%** | **2.7%** | **0.7%** | **1.3%** |
| Items with one answer under all four wordings | 0.88 | 0.96 | 0.98 | 0.98 |
| Mean TV to the original, paraphrases | 0.033–0.084 | 0.036–0.045 | 0.007–0.011 | 0.019–0.029 |
| Flips with the minimal instruction ("Choose the best option.") | 8% | 2% | 0% | 4% |

- **Decision-1 is not more robust to rewording than its frozen base.** On ACE it flips more
  (6.7% vs. 2.7%); on BANKING77 both flip about 1% (Decision-1's distributions move less:
  TV 0.01 vs. 0.02–0.03). The R6 decision rule ("Decision-1 flips ≪ frozen 9B") is not met.
- So robustness to wording is not what post-training adds; the base already has it. The large
  moves of the frozen base in F7 come from prompt *layout* (where the state, question and
  options sit), not from rewording the question.
- Part of Decision-1's ACE flips may be serving noise: separate identical requests already
  differ by up to 0.04 (probe finding 7), enough to flip borderline ACE items. The frozen 9B
  runs deterministically.
- Microsoft's published 1.3% is a different measure: the average decision-flip rate over eight
  request perturbations (reordered options, paraphrased option descriptions, changed keys,
  formatting noise), with zero flips for reordering and description paraphrases. It does not
  report rewording of the question itself, so it cannot be compared directly with these rates.
  Its zero reorder flips are what sorting produces (R18).
- 50 items per dataset: one flip is 2 points.

## R18: Decision-1 sorts the options

[r18_option_order.py](../../../scripts/r18_option_order.py), 96 Decision-1 requests
([r18-decision1.jsonl](r18-decision1.jsonl)).

**Listing-order questions.** 5 options with mixed-case names, 4 random orders × 4 name sets.
Share of answers matching each ordering:

| Question | Order we sent | Case-insensitive sorted | Byte-wise sorted |
|---|---|---|---|
| "Which option is listed first?" | 0.06 | 0.25 | **1.00** |
| "… listed second?" | 0.31 | 0.25 | **1.00** |
| "… listed last?" | 0.19 | 1.00 | **1.00** |

Decision-1 shows the model the options in byte-wise sorted order (uppercase before
lowercase).

**Copies named to sort first.** 16 BANKING77 items, 6 options, the correct option duplicated:

| Copy's name | Original's share of the pair |
|---|---|
| `aardvark` (unrelated) | 0.98 |
| `0_<name>` | 0.61 |
| `A <name>` | 0.49 |
| R7 for comparison: `<name> (copy)`, `<name>-2` (sort after) | 0.85–0.91 |

**The frozen 9B with sorted options** (`option_order: sorted`, job `sleepy_pin_6xyq9h01mj`,
16 items per dataset):

| Probe | Decision-1 | Frozen 9B, request order | Frozen 9B, sorted |
|---|---|---|---|
| Order flips, ACE / BANKING77 | 0 / 0 | 7.5% / 3.1% | 0 / 0 |
| "Listed first / last" vs. request order (BANKING77) | 0.50 / 0.50 | 0.88 / 1.00 | 0.50 / 0.50 |
| Fitting name's share, copy before / after (ACE) | 0.84 / 0.84 | 0.97 / 0.41 | 0.97 / 0.97 |
| Fitting name's share, copy before / after (BANKING77) | 0.86 / 0.85 | 0.88 / 0.70 | 0.88 / 0.88 |
| `(copy)`: original's share (ACE / BANKING77) | 0.85 / 0.91 | – | 0.96 / 0.84 |
| Unrelated copy: original's share (ACE / BANKING77) | 0.78 / 0.99 | – | 0.98 / 0.99 |
| Two opaque copies: first's share (ACE / BANKING77) | 0.45 / 0.42 | 0.92 / 0.39 | 0.35 / 0.49 |
| Empty ACE state | Not-Applicable, p ≈ 0.99 | Not-Applicable | Not-Applicable, p 0.95–0.99 |
| Empty BANKING77 state, normalised entropy | 0.95 | 0.75 | 0.73–0.90 |
| Paraphrase flips, ACE / BANKING77 | 6.7% / 0.7% (50 items) | 2.7% / 1.3% (50 items) | 6.3% / 0% (16 items) |

- **Sorting alone reproduces Decision-1's order and name behaviour** with an untrained base.
  The remaining probe difference is the flatter empty-state answer on BANKING77.
- So order invariance is a property of Decision-1's serving code, and the name preference is
  the base model's first-position preference applied to sorted names. Neither needs training.

## Reproduce

```powershell
uv run python scripts/r14_question_types.py build
uv run python scripts/r14_question_types.py decision1          # ~25 min, rate-limited
uv run --extra aml python scripts/aml_run.py --key r14-qwen9b --tag replication --git-ref replication-study -- scripts/r14_question_types.py local --model qwen3.5-9b
uv run --extra aml python scripts/aml_run.py --key r16b-qwen9b --tag replication --git-ref replication-study -- scripts/r16_shared_state.py --model qwen3.5-9b --states 4 --questions 64
uv run python scripts/run_probes.py --target decision-1 --probe paraphrase --items 50
uv run python scripts/r18_option_order.py run                  # ~15 min, rate-limited
uv run --extra aml python scripts/aml_run.py --key probes-qwen9b-sorted --tag replication --git-ref replication-study -- scripts/run_probes.py --target qwen3.5-9b-sorted --probe order letter_names duplicate question empty_state duplicate_names paraphrase --items 16
```
