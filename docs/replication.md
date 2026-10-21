# How do decision models work? Replication study (R-series)

The first round (A/B/S, C1–C9) asked *how well* decision models decide. This round asks *how
they work*, in three parts that each test a specific hypothesis about Microsoft-Decision-1
(and, with an API key, Jev and the OpenAI Decisions API).

| Part | Question | What runs | Cost |
|---|---|---|---|
| 1. Probes | How is one call computed: what is the readout, is the state shared across questions, what tokenizer and hidden prompt? | Perturbed requests to the decision API, plus forensics on the returned numbers | API only (≈ 850 requests, < $0.10) |
| 2. Frozen base | How much of Decision-1 is already in its base model? | Qwen3.5-9B (Decision-1's base), untrained, read like Decision-1, on the same datasets and probes | ≈ 2–3 GPU-hours |
| 3. Training arms | What does decision training change? | Small LoRA runs on Qwen3.5-2B that differ in one setting (option shuffling, loss), evaluated zero-shot and probed | ≈ 1.5 GPU-hours per arm |

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
the question and lettered options through the model's own chat template (thinking off), one
forward pass, probability summed over every token that decodes to each letter (`A`, ` A`, …),
renormalised over the letters. More than 26 options are scored in groups and then group
winners (P(option) = P(group winner) × P(option | group)).

- `r2_qwen35_9b_letter`: Qwen3.5-9B, Decision-1's base, on ACE and BANKING77 (same catalog
  order and descriptions as B1). Leaderboard `replication`, next to B1.
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
