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
| F3 | Options are scored jointly (the score gap between two options changes when others are added), but their order does not reach the model: the serving code sorts them byte-wise by name first (F20), so the model cannot say which option is listed where in the request | strong |
| F4 | Between two identical options, the one whose name best fits the description wins (`…-2`, `(copy)` or unrelated names lose 78–99% of the pair), wherever it sits. With opaque names the split is near even (first copy 0.45 / 0.42). The frozen 9B instead follows position on ACE: it gives 0.92 to the first of two opaque copies, and swapping the two copies moves the plain name's share from 0.97 to 0.41 (Decision-1: 0.84 both ways). On BANKING77 it is partly name-driven (0.88 → 0.70). The 2B arms also favour the earlier copy. So the order-free name preference comes from post-training | strong (R7); partly sorting (F20): the fitting name also sorts first in every R7 variant |
| F5 | Limits are TypeSafe's System One limits (2–255 options, 2–10 score levels, ≤ 255 questions). 255 also caps questions, so it is most likely a request-format constant, not a label vocabulary | strong / likely |
| F6 | Qwen3.5 tokenizer; ≈ 21 tokens of hidden template; ≈ 2 billed tokens per option beyond its name and description (a one-token label plus a separator) | strong |
| F7 | On ACE, Decision-1 is indistinguishable from its frozen base: same accuracy, 81–83% agreement, while prompt layout alone moves the frozen base's decisions by 15–35%. A question-first prompt makes the frozen base *beat* Decision-1 (0.775 vs. 0.619 names only) | strong |
| F8 | On BANKING77, Decision-1 is 12 points better (0.870 vs. 0.750) and calibrated (T 1.02 vs. 1.41). It is not our readout or prompt: no label scheme (R4) or prompt layout (R5) lifts the frozen base above 0.750. Decision-1 scores *lower* on our calibration split (taken from BANKING77's official train set) than on test (0.847 vs. 0.870), like the frozen base (0.723 vs. 0.750), so there is no sign it memorised BANKING77's train set | strong |
| F9 | Decision training of a 2B model on 8 other tasks makes it position-blind with or without option shuffling, flattens the empty-state prior and improves negation; Brier loss calibrates best on BANKING77. It does not reach 0% order flips (6–11% on ACE) | strong (one seed) |
| F10 | Letters cap a single pass at 26 options. Single-token two-letter codes (562 in Qwen3.5) or exact multi-digit numbers lift that without training; gpt-oss already has single-token numbers 1–999 | strong |
| F11 | Long option descriptions are billed in full and read to the end (up to ≈ 1,650 tokens per option tested): no fixed per-option slots. The frozen 9B does the same (0.94 accuracy with the fact at the end of 6.7k-token prompts), so this is no evidence of special training | strong (R8) |
| F12 | For the frozen 9B, both one-pass schemes are *worse* than grouped letters on BANKING77: codes 0.706, numbers 0.681, letters 0.750 (numbers also take ≈ 25× longer). Number labels change only 7% of ACE decisions (0.646 vs. 0.647 names only). Grouping is not what holds the frozen base back | strong (R4) |
| F13 | Prompt layout has no single best: on BANKING77 the default layout is best (0.750) and markdown worst (0.697), the reverse of ACE (F7). The ranking is the same on the calibration and test splits | strong (R5) |
| F14 | Decision training of the 9B (R9) reproduces most of the *interface*: 0 order flips (16 items per dataset), the fitting name wins between duplicates, a flat answer on empty BANKING77 states, one pass over > 26 options, better calibration (T 1.27). It does **not** reproduce the accuracy: BANKING77 0.749 (no gain), ACE 0.573 / 0.606 (below the frozen base). It still knows which option is listed last (1.00) | strong (one run) |
| F15 | The three question types share the one-token, bf16-softmax readout but have their own templates. `score` is a softmax over levels keyed by digit index (`"0"`…`"9"`, ≈ 1 billed token per level), post-processed into `score` = expected index; it differs from a `choice` over the same texts on ambiguous items (TV 0.40). `noul` returns only P(true), a two-way softmax that is close to, but not the same as, a 2-option choice (log-odds 0.5–1.4 apart) | strong (mechanism) / likely (digit labels); exploratory, 3 states |
| F16 | The frozen 9B answers `noul` and `score` questions like Decision-1 under any reasonable template: log-odds r = 0.95–0.99, the same yes/no answer on 95–100% of items, the same accuracy within 1–2 items. No template is clearly Decision-1's; digit-labelled score levels fit better than letters on 3 levels (same answer 90% vs. 78%) | strong (R14, 200 items) |
| F17 | Decision-1's multi-question speed is reproduced by encoding the state once and batching the questions over the cached prefix: 64 questions in 0.83–0.89 s on one A100 (Decision-1: 0.99 s), 4.5× faster than one at a time. Batched bf16 answers differ from one-at-a-time answers by up to 0.05, the size of Decision-1's request-to-request variation | strong (R16) |
| F18 | Decision-1 is not more robust to rewording the question than its frozen base: decision flips under paraphrase 6.7% vs. 2.7% (ACE), 0.7% vs. 1.3% (BANKING77). Rewording moves both little; prompt layout (F7) is what moves the frozen base | strong (R6, 50 items per dataset) |
| F19 | Teaching a "none / not applicable" option (empty and mismatched states pick it when listed) instead of uniform answers fixes R9's ACE loss: R9b-kl scores 0.700 / 0.750 on ACE (Decision-1 0.619 / 0.741) and answers empty ACE states with Not-Applicable like Decision-1. BANKING77 stays at the frozen base's level (0.733) | strong (R9b, one run) |
| F20 | Decision-1 sorts choice options byte-wise by name (uppercase before lowercase) before the model sees them: "which option is listed first / second / last?" matches sorted order in 48 of 48 requests (request order 6–31%). This alone explains 0% reorder flips and the chance-level answers about listing order. In sorted order the model keeps a mild first-position preference: a copy named to sort first takes about half of the pair (`0_<name>` 0.39, `A <name>` 0.51) unless its name is unrelated (`aardvark` 0.02) | strong (R18, 96 requests) |

**Working model:** Decision-1 = Qwen3.5-9B's judgement + a trained decision interface
(compact per-type templates, one score per option or level read jointly, calibration on
routing-style tasks) + better many-way routing. The serving code sorts choice options
byte-wise before the model sees them (F20) and answers all questions about one state by
encoding the state once and batching the questions in bf16 (F17). The base already supplies
the judgement for all three question types (F7, F16) and robustness to rewording (F18).

**Open questions:**
1. ~~How much of the BANKING77 gain is post-training, and how much our readout and prompt
   (F8)?~~ Answered by R4 and R5: none of it is readout or prompt. Open: which training data
   produces it; R9's routing mix (CLINC150, MASSIVE, DBpedia level-3) did not (F14). R17
   tests domain-matched routing data.
2. ~~Does plain decision training of the same 9B reproduce Decision-1's whole profile, or is
   something architectural needed?~~ Mostly answered: order invariance and position
   blindness come from sorting (F20); ACE and the empty-state behaviour are reproduced
   (F19). The BANKING77 gain is the remaining gap.
3. ~~What causes the name preference (F4)?~~ Answered by R7 and R18: name fit (unrelated
   names lose even when they sort first) plus a first-position preference in sorted order
   (F20).
4. What are the option labels (F5, F6)? R8 rules out fixed per-option slots (F11).

## Next experiments

Ordered by value for cost. R4–R9, R9b, R14, R16 and R18 are done; R15 and R17 are running.
R10–R12 are on hold (2B mechanism checks; R10 and R11 mostly answered by F20). Training on
Decision-1's own answers (distillation) is not planned until its terms of use are checked.
API-only experiments run at ≈ 9 requests/min (the deployment's rate limit).

| ID | Experiment | Question it answers | Decision rule | Cost |
|---|---|---|---|---|
| **R4** (done) | Frozen 9B on BANKING77 with one-pass `codes` and `numbers` labels vs. grouped `letters`; `numbers` also on ACE | Is part of the BANKING77 gap the grouped readout (F8)? Do number labels change ACE? | Result: letters 0.750 > codes 0.706 > numbers 0.681; numbers on ACE ≈ letters (F12) | ≈ 3 GPU-h (3 jobs) |
| **R5** (done) | Frozen 9B on BANKING77: grouped letters × the three prompt layouts | Upper bound of the untrained base on routing; settles open question 1 | Result: default 0.750, json 0.744, markdown 0.697; gap to Decision-1 stays 12 points (F8, F13) | ≈ 1–3 GPU-h |
| **R6** (done) | Paraphrase robustness: three meaning-preserving rewordings of the question plus a minimal one, 50 dev items per dataset, Decision-1 and the frozen 9B (`paraphrase` probe) | Microsoft reports 1.3% flips; the frozen base moves 15–35% with layout. Is robustness to wording the core of post-training? | Result: no. Decision-1 flips 6.7% (ACE) / 0.7% (BANKING77), the frozen 9B 2.7% / 1.3% (F18) | API 500 requests + 0.3 GPU-h |
| **R7** (done) | Name-preference dissection: original renamed `…-2` and copy given the original name; both renamed to neutral ids; copy suffixed `-copy` / `(duplicate)` / an unrelated name | Is F4 a "suffix = duplicate" heuristic, name–description consistency, or first-seen preference? | Result: Decision-1's mass follows the name that fits the description, not the slot; the frozen 9B follows the slot on ACE (F4) | API ≈ 200 requests + ≈ 0.5 GPU-h |
| **R8** (done) | Long options: descriptions of 50–2,000 tokens, with the deciding fact at the start vs. the end of the description; very long option names | Are options truncated or held in fixed per-option slots (F5, F6)? | Result: billed in full, late facts used, no slots; the frozen 9B behaves the same (F11) | API ≈ 100 requests + ≈ 0.5 GPU-h |
| **R9** (done) | Decision-trained **9B**: LoRA on Qwen3.5-9B with the best recipe (shuffled options, Brier loss, one-pass code labels, broader routing data, KL to the base, interface augmentations), evaluated zero-shot and probed | Open question 2: does plain training of the same base reproduce Decision-1 (BANKING77 gain, T ≈ 1, 0% order flips, position blindness, F4)? | Result: interface yes, accuracy and position blindness no (F14) | 78 min training + eval |
| **R9b** (done) | R9 corrected ([arms-r9b.yaml](../configs/training/arms-r9b.yaml)): a "none" option is taught. Empty states target it when listed (added to 60% of them), else uniform; states from another task plus a "none" option target "none" (6%); a "none" distractor on ordinary rows keeps the gold target (10%). Two arms: KL 1.0 (also on the distractor rows) vs. no KL | Was R9's ACE loss the empty-state target? Does the KL anchor keep position awareness? | Result: the none target restores ACE (F19). The no-KL arm is still position-aware, so KL is not the cause; sorting is (F20) | ≈ 3 GPU-h (2 arms) |
| **R10** | Option-token arm (2B): 255 new label tokens `<opt_k>` added to the vocabulary and trained, vs. the `codes` arm | Open question 4: do dedicated label tokens help one-pass scoring of many options and order invariance? | Better order invariance or BANKING77 → dedicated tokens matter | ≈ 2 GPU-h |
| **R11** | Strict fixed-order control (2B): each source always lists its labels in one fixed order, so the gold class sits at a fixed position | `ce_fixed` kept varied positions, so it could not show order bias. Does position bias appear when position predicts the label? | Bias appears → shuffling matters only when positions are informative | ≈ 1.5 GPU-h |
| **R12** | Seeds: `ce_shuffle`, `ce_fixed`, `brier_shuffle` with two more seeds | Are F9's differences between arms (≈ 1–5 points, probe shares) larger than seed noise? | Keep only differences that hold across seeds | ≈ 9 GPU-h |
| **R13** | Jev and the OpenAI Decisions API through the same probes, plus the 256-question test | Same family? Same limits, readout fingerprint and position blindness? | – | needs API keys |
| **R14** (done) | Question-type templates on the frozen 9B: 4 `noul` templates (letters in both orders, true/false, yes/no) and 3 `score` templates (letters, digit options, a `0 = …` scale) vs. Decision-1 on 200 single-question items | Can the hidden per-type templates be recovered? (F15) | Result: no template stands out (r = 0.95–0.99 for all); digit-labelled score levels fit slightly better. Adopted: `truefalse` and `digits` (F16) | API 200 requests + 0.5 GPU-h |
| **R15** (next) | Train all three types ([corpus-r15.yaml](../configs/training/corpus-r15.yaml)): the R9 choice mix plus score rows (Yelp stars, STS-B similarity; level wordings vary) with digit labels and a ranked-probability-score term, and noul rows (BoolQ, MNLI entailment, "is this about X?" from AG News and emotion) with the true/false template. SST-5 is kept out (R14 uses it) | Decision-1 is trained to answer every type; R9 only saw choices. Does multi-type training keep the base's noul/score judgement (F16) while adding calibration? | R14 items: noul and score agreement with Decision-1 at least the frozen base's; choice results not worse than R9b's | ≈ 2 GPU-h |
| **R16** (done) | Multi-question serving: encode the state once, then read every question from the cached prefix, batched (`answer_shared`) | Reproduces "state billed once, 64 questions < 1 s" (F1) at inference time | Result: 64 questions in 0.83–0.89 s on one A100, 4.5× faster than one at a time; answers equal in fp32, within bf16 noise on GPU (F17) | 0.5 GPU-h |
| **R17** (running) | Domain-matched routing data: R9b-kl's recipe on [corpus-r17](../configs/training/corpus-r17.yaml) = the R9 mix + Bitext retail-banking (26 banking intents such as `activate_card`) and customer-support intents, 4,000 rows each. Synthetic data, not BANKING77's: 2 generic phrases shared with BANKING77 train/calibration, none with test; intent names partly identical (e.g. `cancel_transfer`) | Open question 1: is the 12-point BANKING77 gap routing data from the same domain? R9's broader routing data did not move it | BANKING77 well above R9b-kl → domain data closes the gap (a zero-shot claim on BANKING77 then needs this caveat); no change → the gap is not data volume or domain | ≈ 2 GPU-h |
| **R18** (done) | Option order: 5 mixed-case options in random orders, "which option is listed first / second / last?"; duplicate options with the copy named to sort first (`aardvark`, `0_<name>`, `A <name>`) | Does Decision-1 canonicalise option order? Is R7's name effect name fit or sorted position? | Result: byte-wise sorted (48/48); name fit for unrelated names, sorted position for near-identical ones (F20) | API 96 requests |

**Changed or dropped from earlier plans:**
- **Grouping options into passes of < 30** is kept only as the `letters` baseline. R4 tests
  one-pass alternatives instead, since Decision-1 shows no grouping (F2).
- **Option isolation through attention masks** is not planned. Qwen3.5 has 24 linear-attention
  (recurrent) layers out of 32, so options cannot be hidden from each other by masking
  alone. And F3 shows Decision-1 does not score options in isolation anyway.
- **More frozen-model prompt engineering** is limited to R5. F7 already shows the frozen base
  is at least as good as Decision-1 on ACE.

**Dependencies:** R9 → R9b. R6–R8 and R13 are independent; R10–R12 reuse the Part 3
training setup.

## Can we replicate Decision-1 on Qwen3.5-9B?

**Status (2026-10-22):**
- **The target is specified:** F1–F20 define the interface and behaviour to match.
- **R9b reproduces Decision-1 except on BANKING77.**
  - **ACE:** above Decision-1 (no-KL arm 0.738 / 0.767), calibrated.
  - **Interface:** the empty-state behaviour, duplicate handling and one-pass reading of
    many options match.
  - **Order invariance:** comes from sorting the options (F20), which the readout now does
    too.
  - **What is missing:** the 12-point BANKING77 gain and its calibration (T 1.02).
  - **Running:** R15 (all three question types) and R17 (domain-matched routing data).

**What a replica must match** (Decision-1 vs. the untrained base and our trained 9Bs). Since
R18, the order rows are reproduced by sorting the options (F20), as Decision-1 does:

| Target | Decision-1 | Frozen 9B | 2B arms | R9 (9B trained) | R9b-kl | R9b-nokl |
|---|---|---|---|---|---|---|
| ACE accuracy, short / verbose | 0.619 / 0.741 | 0.647 / 0.741 (up to 0.775 / 0.767 with another layout) | 0.49–0.55 | 0.573 / 0.606 | 0.700 / 0.750 | **0.738 / 0.767** |
| BANKING77 accuracy | 0.870 | 0.750 | 0.64–0.65 | 0.749 | 0.733 | 0.742 |
| BANKING77 fitted temperature | 1.02 | 1.41 | 1.17–1.46 | 1.27 | 1.26 | 1.50 |
| ACE fitted temperature, short / verbose | 1.65 / 2.75 | 0.64–2.51 (by layout) | 1.4–2.2 | 1.88 / 1.47 | 1.11 / 0.99 | 1.07 / 1.02 |
| ACE decisions flipped by reordering (request order) | 0% (sorted) | 7.5% | 6–11% | 0% | 7.5% | 10% |
| "Which option is listed last?" (BANKING77, request order) | 0.50 (1.00 in sorted order) | 1.00 | 0.47–0.53 | 1.00 | 1.00 | 0.97 |
| Two identical options: the name that fits the description wins | yes (0.84 both orders; partly sorting) | no on ACE: position wins (0.97 → 0.41 when swapped) | no | yes, stronger (0.97 / 0.98) | yes (0.98 / 0.97) | yes (0.998 / 0.988) |
| Two identical opaque options: share of the first (ACE / BANKING77) | 0.45 / 0.42 | 0.92 / 0.39 | – | 0.42 / 0.34 | 0.43 / 0.44 | 0.50 / 0.38 |
| Empty ACE state: top answer | Not-Applicable, p ≈ 0.99 | Not-Applicable | – | near-uniform | Not-Applicable, p 0.98–0.99 | Not-Applicable, p 0.98–0.995 |
| Empty state, normalised entropy (BANKING77) | 0.95 | 0.75 | 0.82–0.88 | 0.99 | 0.98 | 0.98 |
| Up to 255 options in one pass | yes | no (groups of 26) | no | yes (codes, trained to 100) | yes | yes |

**R9 design, updated with the findings:**
- **Base and adapter:** Qwen3.5-9B, light LoRA, plus a KL term to the frozen base so its
  ACE judgement is kept (F7).
- **Readout:** one label token per option. Train on 2–100 options per row so one-pass
  reading of many options is learned (F10, F12).
- **Data:** the Part 3 mix plus more routing tasks. Add:
  - decoy and near-duplicate options with variant or mismatched names (F4);
  - negated and paraphrased questions;
  - empty or uninformative states with uniform targets;
  - options re-shuffled every example.
- **Loss:** cross-entropy plus Brier (F9).

**Pass criteria:**
- within 3 points of Decision-1 on ACE and BANKING77;
- BANKING77 fitted T within 0.9–1.2;
- under 2% decision flips under reordering;
- position-blind ("listed last" ≈ 0.5);
- the name that fits the description wins (F4);
- R6's paraphrase flip rate, once measured, as an extra robustness target.

**As launched** ([arms-r9.yaml](../configs/training/arms-r9.yaml),
[corpus-r9.yaml](../configs/training/corpus-r9.yaml)):
- Qwen3.5-9B, LoRA r=16, 600 steps × 16 examples, rows up to 2,048 tokens. A checkpoint
  every 50 steps lets a pre-empted job resume.
- Code labels (A–Z, then single-token pairs). This is the only scheme that reads 100+
  options in one pass. Untrained, codes scored 4.4 points below grouped letters on
  BANKING77 (R4); whether training closes that gap is part of what R9 measures.
- Data: the Part 3 sources plus MASSIVE intents (60 classes), DBpedia level-3 (219 classes)
  and 20 Newsgroups. Option subsets of 2–100 are drawn per row.
- Loss: CE + Brier, plus 0.3 × KL(frozen base ‖ adapter) on ordinary and paraphrased rows.
- Augmentations (share of examples): decoy duplicate 8% (target: the original name),
  negated question 5% (2 options, target: the wrong one), uninformative state 4% (uniform
  target), paraphrased question 25%.
- Not tuned on R5. R5's best prompt can be applied at evaluation time afterwards.
- Evaluated zero-shot on ACE and BANKING77 (calibration + test), then probed with the
  same probe set as Decision-1, including R7 and R8.

**R9 result** (job `good_rabbit_q2nbt4jdq4`; test split unless noted; probes use 16
items per dataset):
- **Training:** 600 steps in 78 min; accuracy on the corpus holdout rose from 0.873 to
  0.907 (ECE 0.054 → 0.032).
- **Accuracy:** BANKING77 0.749 (frozen base 0.750, Decision-1 0.870). ACE 0.573 short /
  0.606 verbose (frozen base 0.647 / 0.741). Fails the 3-point criterion on all three.
- **Calibration:** BANKING77 ECE 0.084 (frozen 0.097), T 1.27 (frozen 1.41). Fails 0.9–1.2.
- **Interface:** 0 order flips on ACE and BANKING77; the fitting name wins between
  duplicates in both orders (0.97 / 0.98 on ACE, stronger than Decision-1's 0.84); 0.42 to
  the first of two opaque ACE copies (Decision-1 0.45, frozen 0.92); no grouping jump at
  27 options. Passes.
- **Position:** still answers "which option is listed last?" correctly (1.00), and its
  first-vs-last log lift is −0.31 on both datasets (Decision-1 ≈ 0). Fails.
- **Agreement with Decision-1** (kappa): 0.68 / 0.60 / 0.82 on ACE short / verbose /
  BANKING77 (frozen base 0.66 / 0.68 / 0.81).
- **Leakage check:** the R9 corpus shares no text with ACE and 3 generic utterances with
  BANKING77 ("I lost my phone" and two variants, from CLINC150 under a different label),
  out of ≈ 13,000. CLINC150 does include a banking domain (other texts, other labels).

**Why ACE dropped** (evidence strong, cause likely):
- Decision-1 answers an empty ACE state with **Not-Applicable** (p ≈ 0.99); R9 was trained to
  answer uninformative states with a uniform distribution and now does so (entropy 0.99).
- On real ACE items R9 predicts Not-Applicable less often than Decision-1 (6.9% vs. 15.6%
  verbose, 1.2% vs. 2.8% short) and Non-Compliant more often (69.5% vs. 55.8%). ACE's
  gold labels are one third Not-Applicable.
- So the uniform empty-state target was the wrong target where a "none" option exists.
  R9b tests this.

**R9b-kl result** (job `teal_spade_1wfr7s433f`; test split; probes 16 items per dataset):
- **ACE recovered:** 0.700 short / 0.750 verbose (R9 0.573 / 0.606; Decision-1 0.619 / 0.741),
  and calibrated (fitted T 1.11 / 0.99). Empty ACE states now get Not-Applicable (p 0.98–0.99),
  as from Decision-1. So R9's ACE loss came from the uniform empty-state target. Not-Applicable
  is now predicted more often than by Decision-1 (14.6% vs. 2.8% short, 20.5% vs. 15.6% verbose).
- **BANKING77 unchanged:** 0.733 (R9 0.749, frozen 0.750), T 1.26.
- **Interface:** the fitting name wins between duplicates (0.97–0.99); two opaque copies split
  evenly (0.43 / 0.44, Decision-1 0.45 / 0.42). ACE order flips are back at 7.5% (R9: 0%),
  likely the strong KL pull to the base; BANKING77 0%. It still knows which option is listed
  last (1.00).
- **Agreement with Decision-1** (kappa): 0.58 / 0.66 / 0.81 (ACE short / verbose / BANKING77).

**R9b-nokl result** (job `keen_drop_9g1svz05bs`; same data, no KL):
- **Best ACE so far:** 0.738 short / 0.767 verbose, calibrated (T 1.07 / 1.02). Empty ACE
  states → Not-Applicable (p 0.98–0.995).
- **BANKING77:** 0.742, but less calibrated than with KL (T 1.50 vs. 1.26, raw ECE 0.126 vs.
  0.069).
- **Still position-aware** ("listed last" 0.97) and 10% ACE order flips. So the KL anchor is
  *not* what kept R9/R9b position-aware; the hypothesis is rejected. R18 then showed that
  Decision-1's position blindness comes from sorting the options (F20), so a replica sorts
  them too and needs no learned blindness.
- **Agreement with Decision-1** (kappa): 0.58 / 0.67 / 0.81.

**Next:**
- **Order and position:** sort choice options byte-wise in the readout (`option_order: sorted`,
  F20). This makes order flips 0 and listing-order questions follow sorted order, as for
  Decision-1. Checked on the frozen 9B (job `sleepy_pin_6xyq9h01mj`).
- **ACE:** solved by the "none" target (F19); without KL, ACE is above Decision-1.
- **BANKING77 accuracy:** broader routing data did not transfer. More intent datasets
  (keeping BANKING77 out entirely) are the remaining lever; if none helps, Decision-1's
  routing gain likely comes from data we cannot reconstruct.
- **Calibration:** a different Brier weight, or a scoring-rule-only loss.
