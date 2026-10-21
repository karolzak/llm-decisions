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
   answer 91% of the time. Post-training did not make Decision-1 better at compliance
   reasoning; its judgement there is the base model's.
2. **On BANKING77, post-training adds 12 points and fixes calibration.** Frozen 9B: 0.750,
   raw ECE 0.097, fitted temperature 1.41. Decision-1: 0.870, ECE 0.015, temperature 1.02.
   This is the kind of task decision training data is made of (many-way routing over named
   options), and it is where the gain is. Part of the gap may be our readout: 77 options need
   four passes of at most 26 letters, while Decision-1 shows no sign of such grouping.
3. **Decision training on *other* tasks transfers.** Qwen3.5-2B, trained only on 8 public
   classification tasks, goes from 0.38 to 0.49–0.55 on ACE and from 0.51 to 0.64–0.65 on
   BANKING77, and its probabilities become usable (ACE temperature 7.7 → 1.4–1.8).
4. **Decision training, with or without option shuffling, reproduces Decision-1's position
   blindness.** Asked "which option is listed last?" on BANKING77, both frozen models answer
   correctly (1.00), while Decision-1 is at chance (0.50), and so is every trained arm,
   including the fixed-order control (0.53 / 0.50 / 0.47). Training on many tasks where only
   the content of an option decides the answer is enough; option shuffling is not what
   causes it. Every arm also flips less when options are reordered (ACE 31% → 6–11%;
   Decision-1 0%) and is closer to uniform on an empty state. What shuffling *does* add,
   against the fixed-order control: better answers to "which option does NOT…" (0.63 → 0.78;
   Decision-1 1.00), less need for temperature correction on ACE (T 2.20 → 1.80 short,
   1.87 → 1.41 verbose) and +5 points on ACE verbose.
5. **A proper scoring rule calibrates in the weights.** The Brier arm matches cross-entropy's
   accuracy and needs less temperature correction on BANKING77 (raw ECE 0.062 vs. 0.128;
   T 1.17 vs. 1.40), the direction of Decision-1's "calibrated out of the box" (T 1.02). On
   ACE the difference is small and mixed. This is consistent with, not proof of, an
   RLCD-style objective.
6. **What we did not reproduce:** Decision-1 prefers the *original* option name over an exact
   copy named `…-2`, wherever the copy sits (0.84–0.85 of the pair's mass). Every local model,
   trained or not, prefers the *earlier* copy instead. Decision-1 also bills the state once
   across questions and serves bf16 logits; our readout does neither by construction.

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
| ACE decision flips across option orders | 0% | 7.5% | order sensitivity trained away |
| "Which option is listed last?" (BANKING77) | 0.50 | 1.00 | Decision-1 cannot see positions |
| Duplicate gold option placed *before* the original: share of the original name (ACE) | 0.84 | 0.41 | Decision-1 reads names, the base reads positions |
| Empty state, normalised entropy (BANKING77) | 0.95 | 0.75 | Decision-1 is closer to "I don't know" |
| "Which option does NOT…" (BANKING77) | 1.00 | 0.97 | both follow the question |
| Names that look like answer letters | no effect | no effect | – |

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
- **Decision training supplies the interface.** It makes the model blind to option positions,
  reduces order sensitivity, improves answers to negated questions, flattens the prior on
  empty input and calibrates the probabilities. Each of these traits moves towards
  Decision-1 in our small arms with the generic recipe the open replicas publish (a
  probability readout over listed options, multi-task decision data). Option shuffling and a
  proper scoring rule refine it (negation, calibration) but are not needed for position
  blindness.
- **Gains concentrate on tasks that look like the training data.** Many-way routing over
  named options (BANKING77: +12 points over the base for Decision-1, +14 for our 2B arms) is
  where training helps most.
- **Still unexplained:** Decision-1's preference for the original option name over a
  renamed duplicate, which no recipe here reproduces; and how Decision-1 handles 77 options
  without the grouping our readout needs.

## Caveats

- **One training run per arm**, 16 probe items per dataset. Probe differences of a few items
  are noise; only large moves (e.g. ~1.0 → ~0.5) are meaningful.
- **Our 2B arms are not a Decision-1 replica.** Base size, data (8 public tasks, 23k rows)
  and steps are orders of magnitude smaller. They show what each training choice *does*, not
  how far the recipe scales.
- **BANKING77 readout:** our letter readout scores 77 options in groups of 26 plus a final
  pass, which is a different (and probably weaker) readout than Decision-1's.
- **ACE test ≠ calibration distribution** (see the ACE baseline), so ACE temperatures and
  calibrated ECE are less reliable than BANKING77's.
- The `ce_fixed` job was pre-empted once (low-priority VM) and retrained from scratch.

## Reproduce

```powershell
git push origin replication-study
uv run --extra aml python scripts/aml_submit.py --experiment r2_qwen35_9b_letter --dataset ace --split calibration test --allow-test --parallel --tag replication --git-ref replication-study
uv run --extra aml python scripts/aml_submit.py --experiment r2_qwen35_9b_letter --dataset banking77 --split calibration test --allow-test --parallel --tag replication --git-ref replication-study
uv run --extra aml python scripts/aml_run.py --key probes-qwen3.5-9b --tag replication --git-ref replication-study -- scripts/run_probes.py --target qwen3.5-9b
foreach ($arm in 'ce_shuffle','ce_fixed','brier_shuffle','frozen') { uv run --extra aml python scripts/aml_run.py --key "r3-$arm" --tag replication --git-ref replication-study -- scripts/train_decider.py --arm $arm --split calibration test --allow-test }
uv run --extra aml python scripts/aml_summaries.py --tag replication
uv run python scripts/replication_report.py --tag replication
```
