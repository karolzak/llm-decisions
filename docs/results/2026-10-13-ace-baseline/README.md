# ACE baseline results (2026-10-13)

> The C-series results (H2, H4, H5) and the BANKING77 rerun are in the follow-up:
> [C-series readouts + BANKING77 (2026-10-14)](../2026-10-14-c-series-banking77/README.md).

Dataset: ACE (Fujitsu, CC BY 4.0), 3 labels. Evaluated on the **official test split**
(1,100 examples, sealed until these runs). Calibration (temperature, selective thresholds) is fit
on the train-derived `calibration` split (720 examples). Every experiment ran on all 1,100 test
and 720 calibration examples. Total API spend for all ACE runs: about $2.8.

Artifacts in this folder: `report-test.md` (generated report), `leaderboard_product.csv`,
`reliability_product.png`, `risk_coverage_product.png`, `artifact_probe.json`. Raw predictions
are in `artifacts/runs/ace/` (git-ignored).

## Test leaderboard

| Run | Supervision | Acc | Macro-F1 | NLL (raw → temp-scaled) | ECE (raw → temp-scaled) | AURC | p50 latency | $/1k |
|---|---|---|---|---|---|---|---|---|
| Embedding logreg, per-field `[a,b,a*b,\|a-b\|]` | supervised | **0.844** | 0.850 | 0.360 → 0.356 | 0.046 → 0.036 | **0.038** | 0.19 s | 0.09 |
| Embedding logreg, full state | supervised | 0.800 | 0.800 | 0.465 → 0.468 | 0.036 → 0.043 | 0.067 | 0.20 s | 0.09 |
| **Decision-1, verbose criteria** | zero-shot | **0.741** | 0.730 | 0.727 → 0.644 | 0.106 → 0.065 | 0.154 | 0.64 s | **0.04** |
| GPT-6 Luna, verbalized confidence, short | zero-shot | 0.707 | 0.718 | 0.889 → 0.886 | 0.189 → 0.188 | 0.137 | 2.2 s | 0.23 |
| GPT-6 Luna, label only, short | zero-shot | 0.701 | 0.710 | 8.19 → 1.06 | 0.296 → 0.231 | 0.297 | 2.1 s | 0.20 |
| GPT-6 Luna, label only, verbose | zero-shot | 0.690 | 0.698 | 8.49 → 1.06 | 0.307 → 0.229 | 0.308 | 2.2 s | 0.24 |
| GPT-6 Luna, verbalized, verbose | zero-shot | 0.685 | 0.694 | 0.948 → 0.942 | 0.205 → 0.203 | 0.156 | 2.7 s | 0.28 |
| Decision-1, short labels | zero-shot | 0.619 | 0.527 | 1.154 → 0.836 | 0.222 → 0.039 | 0.242 | 0.78 s | 0.03 |
| Embedding centroid | supervised | 0.522 | 0.520 | 1.048 → 1.036 | 0.132 → 0.022 | 0.392 | 0.20 s | 0.09 |
| Embedding kNN (k=25) | supervised | 0.401 | 0.405 | 1.283 → 1.104 | 0.204 → 0.089 | 0.618 | 0.20 s | 0.09 |
| Embedding zero-shot label similarity (short / verbose) | zero-shot | 0.345 / 0.342 | – | – | – | – | 0.20 s | 0.09 |

"Label only" GPT-6 Luna outputs are one-hot, so their raw NLL reflects a 1e-6 probability floor.
Latencies are per-request from the bulk runs (rate-limited, concurrency 4–8), not from the
dedicated latency protocol.

Selective prediction on test (thresholds fit on calibration, target 80% coverage):

| Run | Realized coverage | Accuracy |
|---|---|---|
| Embedding logreg (per-field) | 0.744 | 0.936 |
| Decision-1, verbose | 0.767 | 0.777 |
| GPT-6 Luna, verbalized, short | 0.571 | 0.822 |
| GPT-6 Luna, label only | ~1.0 | 0.70 (no usable confidence) |

## Findings

1. **The train-derived splits do not represent the test split.** ACE train labels agree 100%
   with the `DeepSeek_prediction` teacher field, so train was apparently filtered for
   agreement with an LLM. Accuracy drops from calibration to test:

   | Method | Calibration | Test | Drop |
   |---|---|---|---|
   | GPT-6 Luna (all four variants) | 0.92–0.94 | 0.69–0.71 | 23 pts |
   | Decision-1, verbose | 0.79 | 0.74 | 5 pts |
   | Embedding logreg | 0.86–0.91 | 0.80–0.84 | 6 pts |

   The LLM loses almost everything on Compliant vs. Non-Compliant (recall about 0.91 → 0.60);
   Not-Applicable is unaffected. Spot-checked test errors are genuinely contested scenarios.
   Consequence: confidence thresholds fit on the train-derived calibration split are
   miscalibrated on test (GPT-6 Luna automates 57% instead of 80%).
2. **ACE contains label shortcuts.** In a grouped 5-fold CV probe on train (chance 0.33):

   | Probe input | CV accuracy |
   |---|---|
   | clauses only | 0.349 |
   | scenario only | 0.600 (hypothesis-only artifact) |
   | clauses + scenario with interaction features | 0.892 |

3. **Supervised embedding probes beat the zero-shot LLM on test** (0.84 vs. 0.70), including
   0.83 on test scenarios never seen in any derived split. This is not leakage: the probes learn
   ACE's labelling conventions, which zero-shot methods cannot see. Leaderboards now label
   supervised vs. zero-shot runs.
4. **Decision-1 is the best zero-shot method on test,** and the cheapest and fastest of the
   decision APIs. It relies heavily on criteria text: verbose descriptions add 12 points
   (0.62 → 0.74). With short labels it almost never predicts Not-Applicable (recall 0.09 vs.
   0.52 with verbose) and over-predicts Non-Compliant (recall about 0.98). About 1.3% of its
   answers have exactly tied top-two probabilities (broken in catalog order).
5. **Verbose criteria help Decision-1 but not GPT-6 Luna** (−1 to −2 points).
6. **Calibration.** Verbalized confidence makes GPT-6 Luna usable for selective prediction
   (AURC 0.14 vs. 0.30 for one-hot), but its calibration does not transfer under the shift.
   Decision-1's probabilities are better calibrated after temperature scaling (ECE 0.065), and
   the embedding probes are best calibrated (ECE ≤ 0.046).

## Hypotheses status (ACE)

| Hypothesis | Status |
|---|---|
| H1 Decision APIs improve calibration without losing accuracy | Supported for Decision-1 vs. GPT-6 Luna (different models). The same-model A2 (OpenAI Decisions) is blocked on Azure. |
| H2 Candidate scoring beats AR calibration (same model) | Pending: C-series. |
| H3 Verbose criteria beat short labels | Supported for Decision-1 (+12 pts); not for GPT-6 Luna. |
| H4 KV reuse ≈ independent scoring, lower latency | Pending: C2/C3. |
| H5 Listwise head closes the gap | Pending: C5. Note that supervised probes already beat zero-shot APIs on ACE. |
| H6 Useful selective prediction | Supported in principle (logreg 0.94 accuracy at 74% coverage). Thresholds do not transfer across the calibration→test shift. |

## Caveats

- Train-derived calibration does not match test, so temperature-scaled numbers on test are
  shift-affected. Fitting calibration on held-out test data would require splitting the
  official test set.
- Six GPT-6 Luna test answers (3 per A1/A1v variant) exhausted the 4,096-token output budget on
  reasoning and count as errors.
- The dedicated latency protocol (warmups, 200 examples × 5 reps, concurrency 1/8) has not been
  run for the API methods. Decision-1's 10 RPM limit makes concurrency 8 meaningless.
- The dev split was only used with 40-example subsets.

## Bugs found and fixed during this phase

- **Selective prediction:**
  - Off-by-one in the coverage threshold.
  - Thresholds were fit after dropping failed calibration requests.
  - Floating-point noise in temperature-scaled one-hot confidences created a spurious ranking
    (inflated AURC for one run). Confidences are now rounded before ranking.
- **Predictions and data integrity:**
  - The canonical prediction is now the argmax; ties keep the provider's choice.
  - Malformed distributions and incomplete responses are recorded as errors.
  - Verbalized-confidence validation.
  - Auto labels no longer read sealed splits.
- **Run safety:**
  - Manifest written before API calls.
  - Config hash includes data provenance and a logic version.
  - Per-split lock; stale processed data and split-policy changes are detected.
  - A truncated last line no longer blocks resuming.
  - Spend tracking includes retried calls.
- **Performance:**
  - Latency benchmark set up the method once per concurrency level.
  - Logreg fits are now cached.
