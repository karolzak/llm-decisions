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

R20_PENDING
