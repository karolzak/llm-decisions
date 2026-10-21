# How do decision models work? Replication study (R-series)

The first round (A/B/S, C1–C9) asked *how well* decision models decide. This round asks *how
they work*, in three parts that each test a specific hypothesis about Microsoft-Decision-1
(and, with an API key, Jev and the OpenAI Decisions API).

| Part | Question | What runs | Cost |
|---|---|---|---|
| 1. Probes | How is one call computed: what is the readout, is the state shared across questions, what tokenizer and hidden prompt? | Perturbed requests to the decision API, plus forensics on the returned numbers | API only (≈ 850 requests, < $0.10) |
| 2. Frozen base | How much of Decision-1 is already in its base model? | Qwen3.5-9B (Decision-1's base), untrained, read like Decision-1, on the same datasets and probes | ≈ 2–3 GPU-hours |
| 3. Training arms | What does decision training change? | Small LoRA runs on Qwen3.5-2B that differ in one setting (option shuffling, loss), evaluated zero-shot and probed | ≈ 1.5 GPU-hours per arm |

Parts 1–3 have run. [What we learned](#findings-so-far) and
[what to run next](#next-experiments) are at the end of this page.

## Background: what is known

- **Microsoft-Decision-1** is Qwen3.5-9B, "post-trained for fast, single-pass decision
  scoring" ([announcement](https://commandline.microsoft.com/microsoft-decision-1-model-foundry/)).
  Microsoft reports no decision flips when options are reversed or shuffled.
- **Jev** (TypeSafe) is trained with "Reinforcement Learning for Calibrated Decisions" on
  synthetic data. It "ingests the state once and evaluates every question against it in
  parallel", allows up to 255 options and admits a lean toward the first option
  ([docs](https://docs.typesafe.ai/model-jaggedness/jev-1.13)). Its architecture is unpublished.
- **Open replicas** use one of two single-pass readouts:
  - **Letter readout:** options shown as A, B, C…; the LM head's next-token probability of
    each letter is read at one position (Cygnet, SemIf, Quyet, H2O-Lightning; our C6).
  - **Pointer head:** the LM head is dropped; a ~1M-parameter head scores each option's own
    hidden state against the answer position (strands-decider; our C5).

  Strands found that re-shuffling options in every training example is what removes the
  position bias.
- **Already in our B1 outputs:** exactly one output token per question, all options in the
  prompt (input tokens grow with K), `confidence = (p_max − 1/K)/(1 − 1/K)` (Jev's formula),
  and log-probability gaps that are exact multiples of 0.125 (bf16 logits of magnitude 16–32).

## Part 1: probes

Code: [`src/edb/probing/`](../src/edb/probing/). One vendor-neutral request type
(`DecisionRequest`: a state plus typed `choice` / `noul` / `score` questions), one adapter per
wire format, and probes written once against that type:

| Adapter | Wire format | Targets (`configs/probe_targets/`) |
|---|---|---|
| `SystemOneClient` | TypeSafe `POST …/systemone` (questions keyed by name, `criteria`) | `decision-1` (Entra ID), `jev` (`TYPESAFE_API_KEY`); also any open replica's server |
| `OpenAIDecisionsClient` | OpenAI `POST /v1/decisions` (`predicate` / `choice` / `score` with `levels`) | `openai-decisions` (`OPENAI_API_KEY`) |
| `LocalReadoutClient` | Local model, letter readout (Part 2/3) | `qwen3.5-9b`, `qwen3.5-2b`, `r3-*` |

Adding a target is a YAML file. Base items are real decisions from the `dev` splits of ACE (3
options with criteria) and BANKING77 (77 intents, often subsets), so the sealed test sets are
never touched.

### What each probe tests

| Probe | Change | Letter readout predicts | Pointer head predicts | Key metrics |
|---|---|---|---|---|
| `order` | All 6 orders of ACE's options; 5 orders of 6 BANKING77 options | A lean toward early letters unless order was augmented in training | Nearly order-free (strands: ~0.016 shift on reversal) | `flip_rate_vs_original`, `first_vs_last_log_lift`, chosen vs. gold mean position |
| `letter_names` | Options renamed to opaque ids, to `A, B, C` in order, or to letters shifted by one (first option named `B`); meaning kept in the description | Names that collide with its internal letters can pull the choice (`collision_pull`) | Unaffected by names | accuracy per naming, `agree_with_neutral_*`, `collision_pull` |
| `duplicate` | A copy of the gold option (same description, value `…-2`) placed after or before it | The earlier copy wins in both placements | Near-equal split (`median_abs_log_ratio` ≈ 0) | `earlier_copy_share_*`, `original_name_share_*`, `mass_conservation` |
| `cardinality` | 2 … 77 BANKING77 options (nested subsets), plus 255 and 256 | A break between 26 and 27 options (letters run out); extra passes for large K | Smooth in K | accuracy, tokens per option, latency per option, `limits` |
| `question` | Same state/options, different question: original, "does NOT", "listed first", "listed last" | Instruction-tuned readouts follow the question | Trained on one question per task, it may ignore the question | accuracy per question, `same_as_match_*` |
| `empty_state` | `""`, `N/A`, `(no information provided)`, lorem ipsum | Shows the letter/position prior | – | normalised entropy, P(first) vs P(last) |
| `multi_question` | 1 vs. 2 questions per request (and the same question twice) | – | – | `shared_tokens_mean` (state billed once?), answer change alone vs. combined |
| `determinism` | 5 identical repeats | – | – | identical share, max probability difference |
| `tokens` | Only the state changes (digits, CJK, emoji, code, `apple ×n`) | – | – | best-fitting tokenizer, hidden prompt size |
| `duplicate_names` (R7) | Two identical copies of the gold option under different names: original renamed `…-2` with the copy getting the plain name (before/after), both opaque ids, copy named `… (copy)` or an unrelated word | Position decides between identical options | Content is identical, so names decide | `plain_name_share_*`, `first_share_neutral`, `original_share_*` |
| `long_options` (R8) | 4 BANKING77 options with opaque ids; the intent sits at the start or end of a description padded with 0–150 neutral sentences | – | – | billing per filler sentence, accuracy start vs. end |

**Forensics** run on every stored answer (`forensics.py`): which log-probability grid the
answers sit on (bf16 fingerprint), which formula produces `confidence`, output-token counts and
latency.

```powershell
uv run python scripts/run_probes.py --target decision-1 --dry-run         # request plan
uv run python scripts/run_probes.py --target decision-1                   # ~95 min at 9 req/min
uv run python scripts/run_probes.py --target decision-1 --analyze-only    # re-analyse
uv run python scripts/probe_report.py --target decision-1 qwen3.5-9b r3-ce-shuffle
```

Outputs: `artifacts/probes/<target>/<probe>.jsonl` (one record per request; resumable),
`summary.json`, and `artifacts/probes/report.md` (targets side by side).

## Part 2: the frozen base

`local_chat_readout` ([`chat_readout.py`](../src/edb/local/chat_readout.py)) reads any chat
model the way the open replicas read frozen instruct models: one user message with the state,
the question and labelled options through the model's own chat template (thinking off), one
forward pass, probability summed over every token that decodes to each label (`A`, ` A`, …),
renormalised over the labels. Option labels (`readout: {labels: ...}`):
- `letters` (default): A–Z. More than 26 options are scored in groups and then group
  winners (P(option) = P(group winner) × P(option | group)).
- `codes`: A–Z, then single-token two-letter codes (`AA`, `AB`, …). Up to 588 options fit one
  pass on Qwen3.5. Identical to `letters` for ≤ 26 options.
- `numbers`: `1.`, `2.`, …; multi-digit labels are scored exactly:
  P(`3`) × P(`7` | `3`) × P(next token is not a digit | `37`). Any number of options in one
  pass.

- `r2_qwen35_9b_letter`: Qwen3.5-9B, Decision-1's base, on ACE and BANKING77 (same catalog
  order and descriptions as B1). Leaderboard `replication`, next to B1.
- `r2_qwen35_9b_letter_markdown` / `_json`: the same model and readout with two other prompt
  layouts (`readout: {template: ...}`), to measure how much the prompt alone moves the
  decisions. Run them in one job: `scripts/job_summary.py` then compares every pair of runs
  of the job (`pairwise`).
- `qwen3.5-9b` probe target: the same model through every Part 1 probe.
- [`scripts/compare_runs.py`](../scripts/compare_runs.py): example-by-example comparison with
  Decision-1: top-1 agreement and Cohen's kappa, shared mistakes, error correlation,
  correlation of P(gold), Jensen–Shannon divergence and prediction mix.

**Reading it.**
- High agreement and shared mistakes beyond what the two accuracies imply mean Decision-1
  kept its base model's judgement and post-training mostly tuned the readout (format,
  calibration, bias).
- Low agreement means post-training changed the decisions themselves.
- Probe differences between `qwen3.5-9b` and `decision-1` show what post-training targeted.
  For example, an order bias present in the base but absent in Decision-1 was trained away.

```powershell
uv run --extra aml python scripts/aml_submit.py --experiment r2_qwen35_9b_letter --split calibration test --allow-test --tag replication
uv run --extra aml python scripts/aml_submit.py --dataset banking77 --experiment r2_qwen35_9b_letter --split calibration test --allow-test --tag replication
uv run --extra aml python scripts/aml_run.py --key probes-qwen3.5-9b -- scripts/run_probes.py --target qwen3.5-9b
uv run --extra aml python scripts/aml_collect.py --tag replication --wait
uv run python scripts/compare_runs.py --dataset ace --split test --runs r2_qwen35_9b_letter__verbose
```

## Part 3: what training changes

[`src/edb/training/`](../src/edb/training/): small, controlled training runs on Qwen3.5-2B
(the same family as Decision-1's base), each differing from `ce_shuffle` in one setting
([`configs/training/arms.yaml`](../configs/training/arms.yaml)):

| Arm | Loss | Option order in training | Tests |
|---|---|---|---|
| `frozen` | – | – | Untrained control (`r2_qwen35_2b_letter`, target `qwen3.5-2b`) |
| `ce_shuffle` | cross-entropy | re-shuffled every example | Standard decision fine-tuning |
| `ce_fixed` | cross-entropy | canonical label order | Does order augmentation remove position bias (Microsoft's "zero flips")? |
| `brier_shuffle` | Brier score | re-shuffled every example | Does a proper-scoring-rule objective (a differentiable stand-in for RLCD) give probabilities that are calibrated without temperature scaling, as Decision-1's are? |

- **Corpus** ([`configs/training/corpus.yaml`](../configs/training/corpus.yaml)): 3,000 rows
  each from AG News, DBpedia, Yahoo Answers topics, Emotion, SST-2, MNLI, BoolQ and CLINC150,
  fetched through the Hugging Face dataset viewer API. Sources with many labels get 2–10
  options per row, so option sets vary like real requests. **ACE and BANKING77 are not in
  training**, so their evaluation stays zero-shot, like Decision-1's. Caveat: CLINC150
  contains banking intents (similar domain, different data).
- **Training:** LoRA (r = 16 on every linear layer of the decoder blocks, including the
  linear-attention projections; [`lora.py`](../src/edb/training/lora.py), no extra
  dependency), 1,500 steps × 16 examples, on the exact prompt the evaluation reads.
- **Evaluation:** `r3_<arm>` on ACE and BANKING77 (leaderboard `replication`) and the
  `r3-<arm>` probe target through the order, naming, duplicate, question, empty-state and
  cardinality probes.

```powershell
uv run --extra aml python scripts/aml_run.py --key r3-ce_shuffle -- scripts/train_decider.py --arm ce_shuffle --split calibration test --allow-test
uv run --extra aml python scripts/aml_run.py --key r3-frozen -- scripts/train_decider.py --arm frozen --split calibration test --allow-test
uv run --extra aml python scripts/aml_collect.py --tag replication --wait
uv run python scripts/build_report.py --dataset ace --split test
uv run python scripts/probe_report.py --target decision-1 qwen3.5-2b r3-ce-shuffle r3-ce-fixed r3-brier-shuffle
```

A retrained adapter changes the run's config hash (the adapter's weight hash is part of it), so
predictions from an older adapter are archived rather than mixed in.

## Collecting results

```powershell
uv run --extra aml python scripts/aml_collect.py --tag replication --wait         # raw outputs (needs storage access)
uv run --extra aml python scripts/aml_summaries.py --tag replication              # summaries from job tags (no storage access)
uv run python scripts/replication_report.py --tag replication                     # artifacts/reports/replication/
```

If the workspace storage is not reachable from your machine (e.g. behind a corporate VPN),
add `--git-ref <pushed branch>` to every `aml_run.py` / `aml_submit.py` command and use
`aml_summaries.py`; see [azure-ml.md](azure-ml.md#when-the-workspace-storage-is-unreachable).
Results of the first full round: [results/2026-10-21-replication](results/2026-10-21-replication/README.md).

## Findings so far

Evidence: [Decision-1 probes](results/2026-10-20-decision1-probes/README.md) (Part 1) and
[frozen base + training arms](results/2026-10-21-replication/README.md) (Parts 2–3).
Strength: **strong** = follows directly from the data, **likely** = best explanation,
**open** = untested.

| # | Finding | Strength |
|---|---|---|
| F1 | The answer is read at one position per question; the state is billed once per request; the questions of a request are evaluated together | strong |
| F2 | Probabilities are a plain softmax over one bf16 score per option, with no post-hoc temperature, token summing, order averaging or grouping, up to 255 options | strong |
| F3 | Options are scored jointly (the score gap between two options changes when others are added), but their order hardly reaches the model, and it cannot say which option is listed where | strong |
| F4 | Between two identical options, the one whose name best fits the description wins (`…-2`, `(copy)` or unrelated names lose 78–99% of the pair), wherever it sits. With opaque names the split is near even. No local model, trained or not, does this: they favour the earlier copy | strong (R7) |
| F5 | Limits are TypeSafe's System One limits (2–255 options, 2–10 score levels, ≤ 255 questions). 255 also caps questions, so it is most likely a request-format constant, not a label vocabulary | strong / likely |
| F6 | Qwen3.5 tokenizer; ≈ 21 tokens of hidden template; ≈ 2 billed tokens per option beyond its name and description (a one-token label plus a separator) | strong |
| F7 | On ACE, Decision-1 is indistinguishable from its frozen base: same accuracy, 81–83% agreement, while prompt layout alone moves the frozen base's decisions by 15–35%. A question-first prompt makes the frozen base *beat* Decision-1 (0.775 vs. 0.619 names only) | strong |
| F8 | On BANKING77, Decision-1 is 12 points better (0.870 vs. 0.750) and calibrated (T 1.02 vs. 1.41). Part of the gap may be our grouped letter readout and prompt | likely (R4, R5 test it) |
| F9 | Decision training of a 2B model on 8 other tasks makes it position-blind with or without option shuffling, flattens the empty-state prior and improves negation; Brier loss calibrates best on BANKING77. It does not reach 0% order flips (6–11% on ACE) | strong (one seed) |
| F10 | Letters cap a single pass at 26 options. Single-token two-letter codes (562 in Qwen3.5) or exact multi-digit numbers lift that without training; gpt-oss already has single-token numbers 1–999 | strong |
| F11 | Long option descriptions are billed in full and read to the end (up to ≈ 1,650 tokens per option tested): no fixed per-option slots | strong (R8) |
| F12 | For the frozen 9B, one-pass two-letter codes are *worse* than grouped letters on BANKING77 (0.706 vs. 0.750), and number labels change only 7% of ACE decisions (0.646 vs. 0.647 names only). Grouping is not what holds the frozen base back | strong (R4, numbers on BANKING77 pending) |

**Working model:** Decision-1 = Qwen3.5-9B's judgement + a trained decision interface
(compact format, one score per option read jointly and order-free, calibration on
routing-style tasks) + better many-way routing.

**Open questions:**
1. How much of the BANKING77 gain is post-training, and how much our readout and prompt (F8)?
   R4 so far: not the grouping (F12).
2. Does plain decision training of the same 9B reproduce Decision-1's whole profile, or is
   something architectural needed (0% order flips, F3, F4)?
3. ~~What causes the name preference (F4)?~~ Answered by R7: the name that fits the
   description wins. Open: is it learned in post-training? (R7 on the frozen 9B is running.)
4. What are the option labels (F5, F6)? R8 rules out fixed per-option slots (F11).

## Next experiments

Ordered by value for cost. R4 and R7 (frozen 9B) are running; R6 needs only the Decision-1
API (≈ 9 requests/min, a few cents) and can run in parallel with the GPU work.

| ID | Experiment | Question it answers | Decision rule | Cost |
|---|---|---|---|---|
| **R4** (partly done) | Frozen 9B on BANKING77 with one-pass `codes` and `numbers` labels vs. grouped `letters`; `numbers` also on ACE | Is part of the BANKING77 gap the grouped readout (F8)? Do number labels change ACE? | If a one-pass scheme closes most of the 12 points, F8 is mostly readout. So far: codes 0.706 < letters 0.750; numbers on ACE ≈ letters (F12) | ≈ 3 GPU-h (3 jobs) |
| **R5** | Frozen 9B on BANKING77: best label scheme × the three prompt layouts | Upper bound of the untrained base on routing; settles open question 1 | Gap to Decision-1 < 3 points: post-training adds calibration, not accuracy | ≈ 1–3 GPU-h |
| **R6** | Decision-1 robustness: paraphrased instructions and criteria, opaque option names, reversed order (≈ 50 dev items per dataset); the same perturbations on the frozen 9B | Microsoft reports 1.3% flips; the frozen base moves 15–35% with layout. Is robustness to wording the core of post-training? | Decision-1 flips ≪ frozen 9B flips under the same perturbations | API ≈ 600 requests + ≈ 0.5 GPU-h |
| **R7** (done on Decision-1; frozen 9B running) | Name-preference dissection: original renamed `…-2` and copy given the original name; both renamed to neutral ids; copy suffixed `-copy` / `(duplicate)` / an unrelated name | Is F4 a "suffix = duplicate" heuristic, name–description consistency, or first-seen preference? | Result: mass follows the name that fits the description, not the slot (F4) | API ≈ 200 requests |
| **R8** (done) | Long options: descriptions of 50–2,000 tokens, with the deciding fact at the start vs. the end of the description; very long option names | Are options truncated or held in fixed per-option slots (F5, F6)? | Result: billed in full, late facts used, no slots (F11) | API ≈ 100 requests |
| **R9** | Decision-trained **9B**: LoRA on Qwen3.5-9B with the best recipe (shuffled options, Brier loss, best labels and prompt from R4/R5, broader routing data), evaluated zero-shot and probed | Open question 2: does plain training of the same base reproduce Decision-1 (BANKING77 gain, T ≈ 1, 0% order flips, position blindness)? | Matches on most → no special architecture needed | ≈ 6 GPU-h |
| **R10** | Option-token arm (2B): 255 new label tokens `<opt_k>` added to the vocabulary and trained, vs. the `codes` arm | Open question 4: do dedicated label tokens help one-pass scoring of many options and order invariance? | Better order invariance or BANKING77 → dedicated tokens matter | ≈ 2 GPU-h |
| **R11** | Strict fixed-order control (2B): each source always lists its labels in one fixed order, so the gold class sits at a fixed position | `ce_fixed` kept varied positions, so it could not show order bias. Does position bias appear when position predicts the label? | Bias appears → shuffling matters only when positions are informative | ≈ 1.5 GPU-h |
| **R12** | Seeds: `ce_shuffle`, `ce_fixed`, `brier_shuffle` with two more seeds | Are F9's differences between arms (≈ 1–5 points, probe shares) larger than seed noise? | Keep only differences that hold across seeds | ≈ 9 GPU-h |
| **R13** | Jev and the OpenAI Decisions API through the same probes, plus the 256-question test | Same family? Same limits, readout fingerprint and position blindness? | – | needs API keys |

**Changed or dropped from earlier plans:**
- **Grouping options into passes of < 30** is kept only as the `letters` baseline. R4 tests
  one-pass alternatives instead, since Decision-1 shows no grouping (F2).
- **Option isolation through attention masks** is not planned. Qwen3.5 has 24 linear-attention
  (recurrent) layers out of 32, so options cannot be hidden from each other by masking
  alone. And F3 shows Decision-1 does not score options in isolation anyway.
- **More frozen-model prompt engineering** is limited to R5. F7 already shows the frozen base
  is at least as good as Decision-1 on ACE.

**Dependencies:** R4 → R5 → R9 (R9 uses the best labels and prompt). R6–R8 and R13 are
independent; R10–R12 reuse the Part 3 training setup.
