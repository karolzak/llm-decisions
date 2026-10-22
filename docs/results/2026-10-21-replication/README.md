# How decision models work: replication results (2026-10-21)

Parts 2 and 3 of the [replication study](../../replication.md); Part 1 (probes of
Decision-1 alone) is in [2026-10-20-decision1-probes](../2026-10-20-decision1-probes/README.md).
Generated tables: [report-test.md](report-test.md) (accuracy, calibration, example-by-example
agreement with Decision-1, training curves) and [probe-report.md](probe-report.md) (every
probe, all targets side by side).

| Target | What it is |
|---|---|
| Decision-1 | Microsoft-Decision-1 (post-trained Qwen3.5-9B), from the B1 runs and Part 1 probes |
| Qwen3.5-9B frozen | Decision-1's base model, no training, option-letter readout (`local_chat_readout`) |
| Qwen3.5-2B frozen | The base of the training arms, same readout |
| 2B + LoRA arms | Qwen3.5-2B, LoRA r = 16, 1,500 × 16 examples from 8 public tasks (**not** ACE or BANKING77), same readout. One setting differs per arm: loss (cross-entropy vs. Brier) and option order (re-shuffled every example vs. fixed) |

All numbers are on the sealed test sets (ACE 1,100, BANKING77 3,080); temperatures are fit
on the calibration split. GPU work: about 14 A100-hours on Azure ML (low priority).

## TL;DR

1. **On ACE, Decision-1 decides like its frozen base model.** An untrained Qwen3.5-9B,
   read with a plain letter readout, scores exactly what Decision-1 scores with criteria text
   (0.741 vs. 0.741) and slightly more with label names (0.647 vs. 0.619). The two agree on
   81–83% of examples (kappa 0.66–0.68), and when both are wrong they give the *same* wrong
   answer 91% of the time.
   - **Prompt control:** changing only the prompt layout changes 15–35% of the frozen
     model's own ACE decisions, more than it differs from Decision-1 (17–19%).
   - **A better prompt beats Decision-1:** with question and options before the state, the
     frozen base scores 0.775 / 0.767 (names / criteria), above Decision-1's 0.619 / 0.741.
   - Post-training did not make Decision-1 better at compliance reasoning; its judgement
     there is the base model's (see [prompt sensitivity](#prompt-sensitivity-of-the-frozen-base)).
2. **On BANKING77, post-training adds 12 points and fixes calibration.** Frozen 9B: 0.750,
   raw ECE 0.097, fitted temperature 1.41. Decision-1: 0.870, ECE 0.015, temperature 1.02.
   This is the kind of task decision training data is made of (many-way routing over named
   options), and it is where the gain is. It is not our readout or prompt: one-pass labels
   (R4) and other prompt layouts (R5) all score at or below 0.750 (see
   [follow-up](#follow-up-2026-10-22-r4-r5-r9)).
3. **Decision training on *other* tasks transfers.** Qwen3.5-2B, trained only on 8 public
   classification tasks, goes from 0.38 to 0.49–0.55 on ACE and from 0.51 to 0.64–0.65 on
   BANKING77, and its probabilities become usable (ACE temperature 7.7 → 1.4–1.8).
4. **Decision training, with or without option shuffling, makes a 2B model position-blind.**
   Asked "which option is listed last?" on BANKING77, both frozen models answer correctly
   (1.00), while Decision-1 is at chance (0.50), and so is every trained 2B arm, including
   the fixed-order control (0.53 / 0.50 / 0.47). **But this is not how Decision-1 gets
   there:** it sorts the options before the model sees them (R18,
   [probe finding 11](../2026-10-20-decision1-probes/README.md)), and against sorted order
   it answers correctly. Our trained 9Bs stayed position-aware (R9, R9b). Every 2B arm also
   flips less when options are reordered (ACE 31% → 6–11%; Decision-1 0% from sorting) and
   is closer to uniform on an empty state. What shuffling *does* add, against the
   fixed-order control: better answers to "which option does NOT…" (0.63 → 0.78;
   Decision-1 1.00), less need for temperature correction on ACE (T 2.20 → 1.80 short,
   1.87 → 1.41 verbose) and +5 points on ACE verbose.
5. **A proper scoring rule calibrates in the weights.** The Brier arm matches cross-entropy's
   accuracy and needs less temperature correction on BANKING77 (raw ECE 0.062 vs. 0.128;
   T 1.17 vs. 1.40), the direction of Decision-1's "calibrated out of the box" (T 1.02). On
   ACE the difference is small and mixed. This is consistent with, not proof of, an
   RLCD-style objective.
6. **What we did not reproduce here:** Decision-1 prefers the *original* option name over an
   exact copy named `…-2`, wherever the copy sits (0.84–0.85 of the pair's mass). Every model
   in this round, trained or not, prefers the *earlier* copy instead. R18 later explained
   it: Decision-1 sorts options, the plain name sorts first, and the frozen 9B with sorted
   options shows the same preference (0.97 / 0.88). Decision-1
   also bills the state once across questions and serves bf16 logits; our readout does
   neither by construction.

## Part 2: Decision-1 vs. its frozen base

| Test accuracy | ACE short | ACE verbose | BANKING77 |
|---|---|---|---|
| Decision-1 | 0.619 | **0.741** | **0.870** |
| Qwen3.5-9B frozen | **0.647** | **0.741** | 0.750 |
| gpt-oss-20b letter readout (C6, for context) | 0.471 | 0.589 | 0.791 |

| Calibration (raw ECE / fitted T) | ACE short | ACE verbose | BANKING77 |
|---|---|---|---|
| Decision-1 | 0.222 / 2.75 | 0.106 / 1.65 | **0.015 / 1.02** |
| Qwen3.5-9B frozen | 0.188 / 1.61 | 0.106 / 1.33 | 0.097 / 1.41 |

| Agreement with Decision-1 (test) | ACE short | ACE verbose | BANKING77 |
|---|---|---|---|
| Top-1 agreement (kappa) | 0.83 (0.66) | 0.81 (0.68) | 0.81 (0.81) |
| Same wrong answer when both wrong | 0.91 | 0.91 | 0.69 |
| Correlation of P(gold), Spearman | 0.87 | 0.75 | 0.78 |
| gpt-oss C6 kappa, for context | 0.35 | 0.42 | 0.85 |

**Probes, Decision-1 vs. its base** (what post-training changed):

| Probe | Decision-1 | Qwen3.5-9B frozen | Reading |
|---|---|---|---|
| ACE decision flips across option orders | 0% | 7.5% | Decision-1 sorts the options (R18), so reordering is a no-op |
| "Which option is listed last?" (BANKING77) | 0.50 | 1.00 | sorted order differs from the order sent; against sorted order Decision-1 answers 1.00 |
| Duplicate gold option placed *before* the original: share of the original name (ACE) | 0.84 | 0.41 | the plain name sorts first; the frozen base with sorted options gives 0.97 |
| Empty state, normalised entropy (BANKING77) | 0.95 | 0.75 | Decision-1 is closer to "I don't know" |
| "Which option does NOT…" (BANKING77) | 1.00 | 0.97 | both follow the question |
| Names that look like answer letters | no effect | no effect | – |

### Prompt sensitivity of the frozen base

To check whether the ACE differences between Decision-1 and its base come from post-training
or just from a different prompt, the frozen Qwen3.5-9B was re-run on ACE with two more prompt
layouts. Options, option order and the letter readout stay the same:
- **default:** state, question, then lettered options, in XML-style tags;
- **markdown:** question and options first as `(A) …`, state last;
- **json:** the whole request as one JSON object.

| ACE test | Accuracy (names / criteria) | Fitted T (names / criteria) | Agreement with Decision-1 |
|---|---|---|---|
| Decision-1 | 0.619 / 0.741 | 2.75 / 1.65 | – |
| Frozen 9B, default prompt | 0.647 / 0.741 | 1.61 / 1.33 | 83% / 81% |
| Frozen 9B, JSON prompt | 0.596 / 0.678 | 2.51 / 1.78 | 80% / 79% |
| Frozen 9B, markdown prompt | **0.775 / 0.767** | 0.64 / 0.82 | 63% / 69% |

| Frozen 9B, prompt vs. prompt | Agreement (names / criteria) |
|---|---|
| default vs. JSON | 85% / 85% |
| default vs. markdown | 67% / 69% |
| JSON vs. markdown | 65% / 68% |

- **The prompt matters more than post-training on ACE.** Layout alone changes 15–35% of
  the frozen model's decisions and moves its accuracy by up to 18 points (names only).
  Decision-1 sits within that spread, below the best prompt.
- **Calibration on ACE also depends on the prompt:** the fitted temperature ranges from
  0.64 (underconfident) to 2.51 (overconfident) without any training.
- **Decision-1 agrees most with the state-first prompts**, consistent with a hidden template
  that puts the state first (required to share it across questions).
- The default prompt rerun reproduced the earlier run exactly. BANKING77 was not re-run
  with other prompts, so its 12-point gap could also be partly prompt.

## Part 3: what decision training changes

| Test accuracy | ACE short | ACE verbose | BANKING77 |
|---|---|---|---|
| Qwen3.5-2B frozen | 0.378 | 0.388 | 0.507 |
| CE, shuffled options | 0.542 | 0.544 | 0.645 |
| CE, fixed option order | 0.529 | 0.491 | 0.641 |
| Brier, shuffled options | 0.548 | 0.511 | 0.650 |

| Calibration (raw ECE / fitted T) | ACE short | ACE verbose | BANKING77 |
|---|---|---|---|
| Qwen3.5-2B frozen | 0.557 / 7.70 | 0.477 / 6.22 | 0.136 / 1.50 |
| CE, shuffled options | 0.164 / 1.80 | 0.172 / 1.41 | 0.128 / 1.40 |
| CE, fixed option order | 0.193 / 2.20 | 0.260 / 1.87 | 0.140 / 1.46 |
| Brier, shuffled options | 0.142 / 1.58 | 0.183 / 1.44 | 0.062 / 1.17 |

| Probe | Decision-1 | 2B frozen | CE shuffled | CE fixed | Brier shuffled |
|---|---|---|---|---|---|
| "Listed last?" (BANKING77) | 0.50 | 1.00 | 0.53 | 0.50 | 0.47 |
| "Listed first?" (BANKING77) | 0.50 | 0.56 | 0.53 | 0.53 | 0.53 |
| ACE decision flips across option orders | 0% | 31% | 11% | 6% | 7.5% |
| BANKING77 first-vs-last log lift | 0.02 | 2.86 | 0.59 | 1.17 | 1.18 |
| "Does NOT…" accuracy (BANKING77) | 1.00 | 0.50 | 0.78 | 0.63 | 0.84 |
| Empty state, normalised entropy (BANKING77) | 0.95 | 0.67 | 0.82 | 0.85 | 0.88 |
| Duplicate before original: original-name share (ACE) | 0.84 | 0.44 | 0.26 | 0.29 | 0.34 |

Note on the fixed-order arm: options keep each dataset's label order, but label sets and
option subsets differ between rows and sources, so the gold option still appears in many
positions. The control removes order *randomisation*, not position diversity; a corpus with
one fixed label set per position would be the stricter test.

Held-out corpus accuracy (8 training tasks, options shuffled) plateaus at 0.88–0.89 after
about 500 steps in every arm, so 1,500 steps is enough at this size.

## What this says about how decision models are built

- **The base model supplies the judgement.** On a reasoning-heavy task (ACE), a 9B decision
  model is exactly as good as its frozen base read in one pass; a 2B decision model trained
  on generic decisions stays far behind. Decision training does not add reasoning.
- **Decision training supplies part of the interface.** In our 2B arms it makes the model
  blind to option positions, reduces order sensitivity, improves answers to negated
  questions, flattens the prior on empty input and calibrates the probabilities. Decision-1,
  however, gets its order invariance and listing-order blindness from sorting the options
  (R18), not from training; the frozen 9B with sorted options already matches it there.
  Option shuffling and a proper scoring rule refine the trained arms (negation,
  calibration).
- **Gains concentrate on tasks that look like the training data.** Many-way routing over
  named options (BANKING77: +12 points over the base for Decision-1, +14 for our 2B arms) is
  where training helps most.
- **Still unexplained (as of this round):** Decision-1's preference for the original option
  name over a renamed duplicate, and how Decision-1 handles 77 options without the grouping
  our readout needs. Both were addressed later: the name preference comes from sorting the
  options (R18), and code labels read 77+ options in one pass (R9).

## Follow-up (2026-10-22): R4, R5, R9

**R4 and R5: the frozen 9B's BANKING77 ceiling is 0.750** (test, n = 3,080; temperature fit
on the calibration split):

| Frozen Qwen3.5-9B on BANKING77 | Accuracy | Raw ECE | T |
|---|---|---|---|
| Letters, grouped by 26, default prompt | **0.750** | 0.097 | 1.41 |
| Letters, json prompt | 0.744 | 0.127 | 1.68 |
| Letters, markdown prompt (best on ACE) | 0.697 | 0.100 | 1.39 |
| Two-letter codes, one pass | 0.706 | 0.175 | 1.66 |
| Numbers (exact digit scoring), one pass | 0.681 | 0.126 | 1.34 |
| *Decision-1* | *0.870* | *0.015* | *1.02* |

- The ranking of prompt layouts is the same on the calibration split, so the best layout was
  not chosen on test data.
- Decision-1 scores lower on the calibration split (taken from BANKING77's official train set)
  than on test (0.847 vs. 0.870), like the frozen base (0.723 vs. 0.750): no sign that it
  memorised BANKING77's train set.

**R9: Qwen3.5-9B with decision training.** LoRA r = 16, 600 × 16 examples from 11 public
tasks (the 8 above plus MASSIVE, DBpedia level-3, 20 Newsgroups; 2–100 options per row; no
ACE or BANKING77), code labels, CE + Brier, KL to the frozen base, and interface
augmentations (decoy duplicates, negated questions, uninformative states with uniform
targets, paraphrases). 78 min on one A100. Full design:
[replication.md](../../replication.md#can-we-replicate-decision-1-on-qwen35-9b).

| Test split | Decision-1 | Frozen 9B | R9 |
|---|---|---|---|
| ACE accuracy, short / verbose | 0.619 / 0.741 | 0.647 / 0.741 | 0.573 / 0.606 |
| BANKING77 accuracy | 0.870 | 0.750 | 0.749 |
| BANKING77 raw ECE / T | 0.015 / 1.02 | 0.097 / 1.41 | 0.084 / 1.27 |
| Kappa with Decision-1, ACE short / verbose / BANKING77 | – | 0.66 / 0.68 / 0.81 | 0.68 / 0.60 / 0.82 |
| Order flips, ACE / BANKING77 (16 items) | 0 / 0 | 0.075 / 0.031 | 0 / 0 |
| "Which option is listed last?" (BANKING77) | 0.50 | 1.00 | 1.00 |
| Duplicates: fitting name's share, copy before / after (ACE) | 0.84 / 0.84 | 0.97 / 0.41 | 0.97 / 0.98 |
| Two opaque copies: first copy's share (ACE) | 0.45 | 0.92 | 0.42 |
| Empty state, normalised entropy (BANKING77) | 0.95 | 0.75 | 0.99 |
| Empty ACE state: top answer | Not-Applicable (p ≈ 0.99) | Not-Applicable | near-uniform (p ≈ 0.35) |

- **Interface reproduced:** no order flips, the fitting name wins between duplicates, no
  grouping, better calibration than the base.
- **Accuracy not reproduced:** no BANKING77 gain, and ACE falls below the frozen base.
- **Position awareness kept:** R9 still knows which option is listed last. The 2B arms lost
  it. R9b showed the KL term is not the cause, and R18 that Decision-1's blindness comes
  from sorting the options, so a replica sorts them instead.
- **Likely cause of the ACE drop:** R9 learned to answer uninformative states uniformly,
  where Decision-1 picks Not-Applicable. On real ACE items R9 predicts Not-Applicable less
  often than Decision-1 (6.9% vs. 15.6% verbose) and Non-Compliant more often (69.5% vs.
  55.8%); a third of ACE's gold labels are Not-Applicable. R9b tests this.
- **Leakage check:** the R9 corpus shares no text with ACE, and 3 generic utterances with
  BANKING77 (from CLINC150, under a different label) out of ≈ 13,000.

## Caveats

- **One training run per arm**, 16 probe items per dataset. Probe differences of a few items
  are noise; only large moves (e.g. ~1.0 → ~0.5) are meaningful.
- **Our 2B arms are not a Decision-1 replica.** Base size, data (8 public tasks, 23k rows)
  and steps are orders of magnitude smaller. They show what each training choice *does*, not
  how far the recipe scales.
- **BANKING77 readout:** our letter readout scores 77 options in groups of 26 plus a final
  pass, unlike Decision-1. R4 shows the one-pass alternatives score lower on the frozen
  base, so this does not explain the gap.
- **ACE test ≠ calibration distribution** (see the ACE baseline), so ACE temperatures and
  calibrated ECE are less reliable than BANKING77's.
- **Prompt sensitivity:** the frozen models were run with our own prompt templates. On ACE the
  choice of template alone moves the 9B's accuracy by up to 18 points; none of our templates
  is Decision-1's hidden one.
- The `ce_fixed` job was pre-empted once (low-priority VM) and retrained from scratch.

## Reproduce

```powershell
git push origin replication-study
uv run --extra aml python scripts/aml_submit.py --experiment r2_qwen35_9b_letter --dataset ace --split calibration test --allow-test --parallel --tag replication --git-ref replication-study
uv run --extra aml python scripts/aml_submit.py --experiment r2_qwen35_9b_letter --dataset banking77 --split calibration test --allow-test --parallel --tag replication --git-ref replication-study
uv run --extra aml python scripts/aml_run.py --key probes-qwen3.5-9b --tag replication --git-ref replication-study -- scripts/run_probes.py --target qwen3.5-9b
uv run --extra aml python scripts/aml_run.py --key r2-prompt-variants --tag replication --git-ref replication-study -- scripts/run_experiment.py --experiment r2_qwen35_9b_letter r2_qwen35_9b_letter_markdown r2_qwen35_9b_letter_json --dataset ace --split calibration test --allow-test
foreach ($arm in 'ce_shuffle','ce_fixed','brier_shuffle','frozen') { uv run --extra aml python scripts/aml_run.py --key "r3-$arm" --tag replication --git-ref replication-study -- scripts/train_decider.py --arm $arm --split calibration test --allow-test }
uv run --extra aml python scripts/aml_run.py --key r9-qwen9b --tag replication --git-ref replication-study -- scripts/train_decider.py --arms-file arms-r9 --arm r9_qwen9b --split calibration test --allow-test
uv run --extra aml python scripts/aml_summaries.py --tag replication
uv run python scripts/replication_report.py --tag replication
```
