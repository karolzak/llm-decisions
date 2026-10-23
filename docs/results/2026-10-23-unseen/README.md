# Unseen benchmarks: R19 (candidate replica) and R20 (training scale) (2026-10-23)

All earlier trained arms were compared on ACE and BANKING77 test. R19 evaluates models on three
benchmarks that no training mix contains and that were not used to choose anything:

| Benchmark | Source | Items | Options | Task |
|---|---|---|---|---|
| MInDS-14 | `PolyAI/minds14`, en-US | 210 (15 per intent) | 14 e-banking intents | route a transcribed customer request |
| ATIS | `tuetschek/atis`, test | 238 (≤ 30 per intent; combined intents dropped) | 14 airline-travel intents | route a traveller's question |
| ContractNLI | `kiddothe2b/contract-nli`, contractnli_a test | 201 (67 per label) | entailment / contradiction / neutral, with descriptions | does a contract excerpt entail, contradict or leave open a statement? |

- **Items and answers:** [items.jsonl](items.jsonl); Decision-1's answers in
  [decision1.jsonl](decision1.jsonl).
- **Script:** [r19_unseen_eval.py](../../../scripts/r19_unseen_eval.py).
- **Overlap check:** no item text appears in any training corpus (r9, r15, r17).
- **Readout:** local models use code labels with options sorted byte-wise by name, as
  Decision-1 does (F20). The option lists here are already in that order.
- **Temperature:** "T" is fitted on the same items, as a diagnostic only.

## Decision-1 vs. the untrained base

| | Decision-1 | Qwen3.5-9B untrained (sorted options) |
|---|---|---|
| MInDS-14 accuracy / ECE / T | **0.971** / 0.036 / 0.71 | 0.914 / 0.029 / 1.05 |
| ATIS accuracy / ECE / T | **0.929** / 0.064 / 0.74 | 0.840 / 0.082 / 1.03 |
| ContractNLI accuracy / ECE / T | 0.776 / 0.104 / 1.60 | 0.756 / 0.086 / 1.31 |
| Agreement with Decision-1 (MInDS-14 / ATIS / ContractNLI) | – | 93% / 89% / 91% (kappa 0.92 / 0.88 / 0.86) |
| Same answer when both are wrong | – | 80% / 81% / 95% |

- **Decision-1's routing gain is not specific to BANKING77.** On two intent sets that our
  training never saw, and that are not known to be in Decision-1's, it beats its base by 6
  (MInDS-14) and 9 (ATIS) points, against 12 on BANKING77. So "BANKING77 in its training
  data" cannot be the whole explanation (F8). Decision-1 gained general many-way routing
  skill.
- **On the reasoning task the gain is small** (+2 points on ContractNLI, 201 items), as on
  ACE (F7).
- **Calibration differs by task type:**
  - On the intent sets Decision-1 is *underconfident* (T ≈ 0.7, fitted in-sample), while the
    base is calibrated.
  - On ContractNLI Decision-1 is overconfident (T 1.60), more than the base (1.31).
  - This matches ACE (overconfident) and contrasts with BANKING77 (T 1.02).
- **The two models are close in decisions:** 89–93% agreement, higher than on ACE (81–83%).

## Candidate replica (R19) and scale (R20)

### R19: the candidate replica

R9b-kl's recipe, retrained with the identical config (job `loving_stone_j3bvyg5zh7`), read
with byte-sorted options. It was chosen before any of these results by the lowest mean
calibration-split NLL over ACE short / verbose / BANKING77 among our trained 9B arms (R9b-kl
0.734, R17 0.742, R9b-nokl 0.771, R15 0.801, R9 0.974).

**Unseen benchmarks:**

| | Decision-1 | Candidate | Untrained 9B |
|---|---|---|---|
| MInDS-14 accuracy / ECE / T | **0.971** / 0.036 / 0.71 | 0.933 / 0.040 / 0.86 | 0.914 / 0.029 / 1.05 |
| ATIS accuracy / ECE / T | **0.929** / 0.064 / 0.74 | 0.861 / 0.095 / 0.88 | 0.840 / 0.082 / 1.03 |
| ContractNLI accuracy / ECE / T | 0.776 / 0.104 / 1.60 | **0.796** / 0.048 / 1.25 | 0.756 / 0.086 / 1.31 |
| Agreement with Decision-1 (kappa) | – | 95% / 90% / 94% (0.94 / 0.89 / 0.91) | 93% / 89% / 91% (0.92 / 0.88 / 0.86) |

- **Training helps on every unseen set:** +1.9 (MInDS-14), +2.1 (ATIS) and +4.0 points
  (ContractNLI) over the untrained base, and the candidate agrees more with Decision-1.
- **On contract reasoning the candidate is above Decision-1** (0.796 vs. 0.776; 201 items, so
  ±3 points is within sampling noise), and better calibrated (ECE 0.048 vs. 0.104).
- **On routing it closes a quarter to a third of the gap:** Decision-1 stays 3.8 (MInDS-14) and 6.8
  (ATIS) points ahead, the same picture as BANKING77.

**Rerun check (same config and seed as R9b-kl):**

| | ACE short / verbose (test) | BANKING77 (test) | BANKING77 T |
|---|---|---|---|
| R9b-kl | 0.700 / 0.750 | 0.733 | 1.26 |
| R19 candidate (rerun) | 0.668 / 0.723 | 0.736 | 1.24 |

- **ACE moves by about 3 points between identical runs**, BANKING77 by 0.3.
  - The causes are GPU nondeterminism and resuming after pre-emption.
  - So ACE differences between our arms of up to about 3 points are not meaningful. That
    covers R9b-nokl's "best ACE", R17's 0.780 verbose, and R15's 3.7–4.8 point ACE drop.
- **The probes with sorted options** (16 items per dataset):
  - 0 order flips;
  - "listed first / last" 0.50 / 0.50 against request order on BANKING77;
  - the fitting name wins between duplicates in both orders (0.98 / 0.99);
  - empty ACE states → Not-Applicable (p 0.91–0.99);
  - empty BANKING77 states flat (entropy 0.99–1.00);
  - paraphrase flips 2.1% (ACE) / 0% (BANKING77).
- **ACE and BANKING77, test:**
  - ACE 0.668 / 0.723 (kappa with Decision-1 0.63 / 0.69);
  - BANKING77 0.736 (kappa 0.81), T 1.24.

### R20: training scale

The R17 recipe with 5× the steps (3,000 × 16 = 48,000 examples, about one pass over
corpus-r17), LoRA rank 64 and KL 0.3. Job `epic_mango_qnkc1hqnlh`, 5.9 h of training on one
A100, read with byte-sorted options.

| | Decision-1 | Untrained 9B | Candidate (R19, 600 steps) | R20 (3,000 steps) |
|---|---|---|---|---|
| BANKING77 test accuracy / T | **0.870** / 1.02 | 0.750 / 1.41 | 0.736 / 1.24 | 0.744 / **1.15** |
| ACE test, short / verbose | 0.619 / 0.741 | 0.647 / 0.741 | 0.668 / 0.723 | **0.586 / 0.561** |
| MInDS-14 | 0.971 | 0.914 | 0.933 | **0.967** |
| ATIS | **0.929** | 0.840 | 0.861 | 0.815 |
| ContractNLI | 0.776 | 0.756 | **0.796** | 0.786 |
| Agreement with Decision-1 (MInDS-14 / ATIS / ContractNLI) | – | 93% / 89% / 91% | 95% / 90% / 94% | 98% / 87% / 93% |
| Corpus holdout accuracy | – | – | 0.887 | 0.930 |

- **More training does not close the BANKING77 gap.** 0.744, the same as every 600-step arm
  (0.733–0.753). It does bring BANKING77 calibration into the target band (T 1.15).
- **It matches Decision-1 on MInDS-14** (0.967 vs. 0.971), but not on ATIS (0.815, below the
  untrained base). MInDS-14 is e-banking, the domain of the Bitext retail-banking data in
  corpus-r17; ATIS's airline domain is in no training source. So the gain looks
  domain-specific rather than the general routing skill Decision-1 shows (F25).
  - R20 changed four things at once (corpus, steps, rank, KL), and R17 was never run on
    these sets, so data and scale cannot be separated here.
- **It damages ACE badly:** 0.586 / 0.561, below the untrained base. Predictions collapse
  onto Non-Compliant (72–76%, against Decision-1's 56–64%), and Not-Applicable falls to 6–7%.
  This is the R9 failure pattern. A weak anchor (KL 0.3) over long routing-heavy training
  pulls the model away from the base's compliance judgement. Contract NLI (0.786) holds up,
  so the loss is specific to ACE's three-way compliance labels.
- **The interface stays right:**
  - 0 order flips (sorted options);
  - the fitting name gets 1.00 of the pair;
  - empty ACE states → Not-Applicable;
  - empty BANKING77 states flat;
  - paraphrase flips 2% / 0%.

## What R19 and R20 say

- **Decision-1's routing skill is general.** It holds on unseen intent sets, so it is not
  BANKING77-specific data (F25).
- **Our recipe recovers part of that skill.** At 600 steps it gains 2–4 points on unseen
  sets without hurting reasoning. At 5× it reaches Decision-1 on MInDS-14 but not on ATIS or
  BANKING77, and it loses ACE. Neither setting reproduces Decision-1's combination: +6 to +12
  points on routing *and* ACE intact.
- **What Decision-1's training likely has that ours lacks (speculation):**
  - routing data spanning many more domains, not more steps on a few;
  - an objective that keeps the base's reasoning while it learns, e.g. RL against calibrated
    scores rather than CE + Brier.
