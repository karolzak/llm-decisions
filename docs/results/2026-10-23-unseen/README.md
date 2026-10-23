# Unseen benchmarks: R19 (candidate replica), R20 (training scale), R21 (many domains) and R22 (seeds) (2026-10-23)

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
- **Overlap check:** no item text appears in any training corpus (r9, r15, r17, r21) or in
  the R24 synthetic data.
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

- **Training helps a little on the unseen sets:** +1.9 (MInDS-14), +2.1 (ATIS) and +4.0 points
  (ContractNLI) over the untrained base in this run, and the candidate agrees more with
  Decision-1. The intent-set gains are within seed noise (R22: seed 1 equals the base on ATIS).
- **On contract reasoning the candidate is above Decision-1** (0.796 vs. 0.776; 201 items, so
  ±3 points is within sampling noise), and better calibrated (ECE 0.048 vs. 0.104).
- **On routing it closes a quarter to a third of the gap:** Decision-1 stays 3.8 (MInDS-14) and
  6.7 (ATIS) points ahead, the same picture as BANKING77.

**Rerun check (same config and seed as R9b-kl):**

| | ACE short / verbose (test) | BANKING77 (test) | BANKING77 T |
|---|---|---|---|
| R9b-kl | 0.700 / 0.750 | 0.733 | 1.26 |
| R19 candidate (rerun) | 0.668 / 0.723 | 0.736 | 1.24 |

- **ACE moves by about 3 points between identical runs**, BANKING77 by 0.3.
  - The causes are GPU nondeterminism and resuming after pre-emption.
  - R22 below shows the full seed spread is larger: up to 7.7 points on ACE. ACE differences
    between single runs below that are not meaningful (R9b-nokl's "best ACE", R17's 0.780
    verbose, R15's 3.7–4.8 point ACE drop).
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

### R22: seed noise of the candidate recipe

Seeds 1 and 2 (jobs `teal_double_g29crmt1qc`, `amusing_oregano_h5zrsnys5s`), the same config as
R9b-kl and the R19 rerun (seed 0):

| Run | ACE short / verbose | BANKING77 | MInDS-14 | ATIS | ContractNLI |
|---|---|---|---|---|---|
| R9b-kl (seed 0) | 0.700 / 0.750 | 0.733 | – | – | – |
| R19 rerun (seed 0) | 0.668 / 0.723 | 0.736 | 0.933 | 0.861 | 0.796 |
| Seed 1 | 0.689 / 0.729 | 0.734 | 0.929 | 0.840 | 0.811 |
| Seed 2 | 0.726 / 0.800 | 0.749 | 0.948 | 0.878 | 0.781 |
| Range (sd) | 5.7 (0.024) / 7.7 (0.035) | 1.6 (0.007) | 1.9 | 3.8 | 3.0 |
| *Decision-1* | *0.619 / 0.741* | *0.870* | *0.971* | *0.929* | *0.776* |

- **ACE is noisy:** up to 7.7 points between seeds of one recipe. Single-run ACE differences
  smaller than that say nothing about the recipe.
- **Routing is stable:** BANKING77 1.6 points, unseen intent sets 2–4. Even the best seed stays
  12 (BANKING77), 2.4 (MInDS-14) and 5.0 (ATIS) points below Decision-1.
- **The candidate recipe beats Decision-1 on ACE short in every seed** (0.668–0.726 vs. 0.619;
  verbose 0.723–0.800 vs. 0.741, below it in 2 of 4 seeds) and on ContractNLI in 3 of 3 seeds (0.781–0.811 vs. 0.776).

### R21: routing data from many domains

The candidate recipe with corpus-r21 (job `ashy_prune_61jq2yjk6x`). It adds Bitext intents from
six customer-service domains, Bitext customer support and SNIPS, 2,500 rows each. Banking /
finance and travel were excluded so as not to match BANKING77, MInDS-14 and ATIS.

| | ACE short / verbose | BANKING77 / T | MInDS-14 | ATIS | ContractNLI |
|---|---|---|---|---|---|
| Candidate recipe, seed range | 0.668–0.726 / 0.723–0.800 | 0.733–0.749 / 1.24–1.29 | 0.929–0.948 | 0.840–0.878 | 0.781–0.811 |
| R21 | 0.745 / 0.801 | 0.747 / 1.33 | 0.952 | 0.840 | 0.811 |

- **Routing does not move beyond seed noise:**
  - BANKING77 and ATIS are inside the range;
  - MInDS-14 is 0.5 points (one item) above its top.
  - The decision rule ("BANKING77 and ATIS above the candidate by more than the seed spread")
    fails.
- **ACE and ContractNLI stay intact**, at or above the top of the seed range.
- So ≈ 20,000 routing rows from 8 more domains, at the same 600 steps, do not reproduce
  Decision-1's general routing skill.

## What R19–R22 say

- **Decision-1's routing skill is general.** It holds on unseen intent sets, so it is not
  BANKING77-specific data (F25).
- **Our recipe recovers little of that skill.** At 600 steps it gains 2–4 points on unseen
  sets (partly within seed noise) without hurting reasoning. At 5× it reaches Decision-1 on MInDS-14 but not on ATIS or
  BANKING77, and it loses ACE. Neither setting reproduces Decision-1's combination: +6 to +12
  points on routing *and* ACE intact.
- **More domains at this scale do not help** (R21, 8 more routing sets).
- **What Decision-1's training may have that ours lacks (speculation), now being tested:**
  - routing data at a far larger scale and breadth (R24: 43k synthetic messages from 199
    domains);
  - a different objective (R23: exact on-policy RL with a KL anchor to the base).
