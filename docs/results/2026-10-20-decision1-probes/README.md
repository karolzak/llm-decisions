# How Microsoft-Decision-1 computes a decision: probe results (2026-10-20)

Part 1 of the [replication study](../../replication.md): 855 perturbed requests to
Microsoft-Decision-1, built from 16 `dev` items each of ACE and BANKING77. Full tables:
[probe-report.md](probe-report.md); raw summary: [summary-decision-1.json](summary-decision-1.json).
Parts 2 and 3 (frozen Qwen3.5-9B, training arms) are implemented but not yet run.

## TL;DR

1. **One forward pass, single output position per question, state shared across questions.**
   Every answer costs one output token per question. Latency is flat in the number of options
   (≈ 470 ms from 2 to 77 options, +0.14 ms per option). Two questions in one request bill the
   state once (718 of ≈ 870 tokens shared); each extra question adds only its own ≈ 116 tokens.
2. **Probabilities are a raw softmax over bf16 logits.** 99.8% of answers sit exactly on a bf16
   grid. 70% are on the 0.125 grid, i.e. logits of magnitude 16–32. A frozen Qwen3.5 letter
   readout puts its top answer logit at ≈ 27, inside the same range. No fitted temperature
   is applied afterwards, so the calibration is in the weights. `confidence` is always Jev's
   formula `(p_max − 1/K)/(1 − 1/K)`.
3. **It never sees option order.** Zero decision flips across all 6 orders of ACE's options and
   5 orders of BANKING77 options, and no first-vs-last lift (−0.005 / +0.02 log). Asked "which
   option is listed first/last?" it scores chance (0.50 on BANKING77) and answers by content
   instead. The position of an option appears not to reach the decision.
4. **But it reads option names, and the question.**
   - A copy of the gold option with the same description but the name `…-2` gets only
     ≈ 16% of the pair's mass, whichever comes first. The name matters, the position does not.
   - "Which option does NOT …" is answered correctly 100% of the time.
5. **No sign of a 26-letter readout.**
   - Tokens per option stay ≈ 10 across the 26 → 27 boundary, and latency does not jump.
   - Option names that look like answer letters (first option named `B`, …) cause no
     confusion (`collision_pull` = 0).
   - The limit is 255 options, the same as Jev: 256 is rejected with HTTP 400.
6. **Qwen tokenizer.** Billed tokens fit the Qwen3.5 tokenizer better than gpt-oss's (0.67 vs
   0.33 exact). Digits cost one token each. The hidden prompt around a 2-option question is
   ≈ 39 tokens.
7. **Not deterministic.** Repeats of an identical request differ by up to 0.04 in probability
   (never in the chosen option), as expected from batched bf16 serving. The same question asked
   twice *within* one request gets identical answers, so the questions of a request are
   computed together in one batch.

## What this says about the architecture

| Observation | Plain letter readout (H2O, Quyet, Cygnet) | Pointer head (strands) | Option-slot tokens + order isolation |
|---|---|---|---|
| bf16 logits of LM-head magnitude, no temperature | ✓ | possible, not natural | ✓ |
| No position effect, can't say which option is first | ✗ (sees order) | ✗ (causal: ~0.016 shift) | ✓ |
| No 26 → 27 break, flat latency to 77, 255 cap | ✗ (needs grouping) | ✓ | ✓ |
| Reads option names and the question | ✓ | ✓ | ✓ |
| State billed once, questions batched | via prefix cache | via prefix cache | ✓ |

The best fit is a Qwen3.5-9B whose answer is read from the LM head over **per-option
index tokens** (up to 255 of them), with options **isolated from each other's position**:
each option attends to the state and question but not to its neighbours, or options are
otherwise encoded order-free. Training (Microsoft: "post-trained for single-pass decision
scoring") would then put calibrated mass on those tokens. This is a hypothesis from black-box
evidence.

Two checks narrow it down:
- **Part 2:** a frozen Qwen3.5-9B letter readout should show the order sensitivity Decision-1
  lacks. Its agreement with Decision-1 measures how much judgement post-training kept.
- **Part 3:** the `ce_fixed` arm tests whether order bias appears without order augmentation,
  and `brier_shuffle` whether a scoring-rule loss gives temperature-free calibration.

## Reproduce

```powershell
uv run python scripts/run_probes.py --target decision-1          # ~95 min at 9 requests/min
uv run python scripts/probe_report.py --target decision-1
```
