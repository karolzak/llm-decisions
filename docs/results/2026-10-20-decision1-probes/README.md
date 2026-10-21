# How Microsoft-Decision-1 computes a decision: probe results (2026-10-20)

Part 1 of the [replication study](../../replication.md): 854 perturbed requests to
Microsoft-Decision-1 (852 answered; the other 2 are the deliberate 256-option test). Most
probes use 16 `dev` items each from ACE and BANKING77; the repeat test uses 6, the empty-state
test 1 item per dataset, and the token test synthetic inputs. Full tables:
[probe-report.md](probe-report.md); raw summary: [summary-decision-1.json](summary-decision-1.json).
Parts 2 and 3 (frozen Qwen3.5-9B, training arms): [2026-10-21-replication](../2026-10-21-replication/README.md).

All requests were sent from a laptop over a VPN to a Foundry deployment, so every latency
below includes about 400 ms of network and service overhead.

## TL;DR

1. **The answer is read at one position per question, and the state is billed once per
   request.**
   - Output tokens always equal the number of questions: 820 one-question requests billed 1,
     and 32 two-question requests billed 2. That holds even for long BANKING77 option names
     (e.g. `balance_not_updated_after_bank_transfer`), so the answer is not generated as text.
   - Adding a yes/no question to a choice question costs exactly 25 more input tokens on all
     16 items (instead of the yes/no request's ≈ 743), and asking the same choice question
     twice adds exactly 116. The state and hidden prompt are billed once: 718 of a mean 833
     tokens are shared.
   - The same question asked twice in one request gets bit-identical answers, while repeated
     separate requests differ (finding 7). This is consistent with all questions of a request
     being computed together. Billing alone cannot show whether the state is also *computed*
     once.
   - 64 yes/no questions in one request were answered in 986 ms and billed 64 output tokens.
     A request may carry at most 255 questions (finding 5).
   - Latency is flat from 2 to 77 options (median 454–481 ms, 65 → 815 input tokens). This
     only rules out mechanisms that would add tens of milliseconds, such as generating option
     names token by token. A single forward pass is the simplest explanation, but the timing
     cannot confirm it.
2. **Probabilities are a softmax over bf16 logits, with no arbitrary temperature applied
   afterwards.**
   - In 99.8% of answers, every log-probability gap is an exact multiple of a power-of-two
     step. For random values this happens in about 4% of 2-option answers, and far less
     with more options. For comparison, 97% of our own Qwen3.5-9B answers (fp32 output head)
     sit on no grid.
   - 94% sit on the 0.125 grid (or a coarser one), the spacing of bf16 numbers between 16
     and 32. 6% sit only on the 0.0625 grid (some logit between 8 and 16). Language-model
     answer logits fall in this range: a frozen Qwen3.5-0.8B letter readout, checked on CPU,
     puts its top letter logit at about 27.
   - The grid holds at every number of options, so all options of a question are scored in
     one softmax, never in groups:

     | Options | Answers | On a bf16 grid |
     |---|---|---|
     | 2 | 162 | 100% |
     | 3–8 | 578 | 100% |
     | 9–40 | 64 | 100% |
     | 77 | 46 | 96% |
     | 255 | 2 | 100% |

   - **Ruled out:** any processing between the model's scores and the returned
     probabilities other than a plain softmax. That excludes:
     - summing several tokens per option (`A` + ` A`), as our readout and Cygnet do;
     - averaging over option orders;
     - scoring in groups and then the group winners;
     - post-hoc calibration.

     Each would put the gaps off the grid.
   - **Not a first-token readout either:** `Compliant` and `Compliant-2` start with the same
     token, so reading each option's first name token would give them identical scores.
     Decision-1 splits them 84% / 16% (finding 4).
   - A post-hoc temperature divides every gap by T, which would break the grid unless T is a
     power of two. Any calibration therefore comes from the network itself. Decision-1 is
     calibrated out of the box on BANKING77 (fitted T = 1.02) but not on ACE (T = 1.65 with
     criteria, 2.75 without; see the [C-series results](../2026-10-14-c-series-banking77/README.md)).
   - `confidence` equals Jev's documented formula `(p_max − 1/K)/(1 − 1/K)` on every choice
     answer.
3. **Option order hardly matters, but which other options are present does.**
   - Reordering options never flipped the decision: 0 of 80 reorderings on ACE (all 6 orders
     of 3 options) and 0 of 64 on BANKING77 (5 orders of 6 options). There is no first-vs-last
     preference (log lift −0.005 on ACE, +0.02 on BANKING77).
   - On BANKING77, reordering moves the probabilities no more than repeating the same request
     does (mean total-variation distance 0.0035 vs. 0.006). On ACE it moves them about 2.6×
     more (0.014 vs. 0.005, at most 0.10), so a small order effect remains there.
   - **The options are scored jointly, not each on its own.** If each option were scored in
     isolation (state + question + that option only), the score gap between two options
     could not depend on the other options present. It does, increasingly so as more options
     are added. Change in the score gap between two options present in both requests:

     | Change to the request | Gaps compared | Exactly unchanged | Median change | 90th percentile |
     |---|---|---|---|---|
     | None (identical request resent) | 1,868 | 43% | 0.06 | 0.25 |
     | Options reordered (BANKING77) | 320 | 49% | 0.13 | 0.25 |
     | Options reordered (ACE) | 160 | 37% | 0.13 | 0.25 |
     | Options added: 2 → 4 | 16 | 25% | 0.25 | 0.75 |
     | Options added: 8 → 16 | 112 | 1% | 0.81 | 1.63 |
     | Options added: 26 → 77 | 400 | 0.5% | 1.50 | 3.19 |
     | A copy of the correct option added | 160 | 2.5% | 0.56–0.63 | 1.38 |

     The added-option rows also change the order (each option-subset size is shuffled
     separately), but reordering alone stays at the noise level.
   - Asked "Ignore the message. Which option is listed first?" (or last) about two options,
     it picked the same option in both orders in all 32 cases. It does not read positions.
     Which option it picks is unrelated to the correct intent (correct-intent option in
     50%), so it is not simply answering the original question either.
   - On ACE this question was asked in one fixed order only. It answered `Compliant` (listed
     first) 16/16 and `Not-Applicable` (listed last) 12/16, which cannot separate position
     from content.
4. **It reads option names and follows the question.**
   - A copy of the correct option with the same description but the name `…-2` gets only
     15–17% of the pair's probability. This holds whether the copy is placed before or after
     the original, on both datasets (original-name share 0.83–0.85). Adding the copy leaves
     the pair's total mass close to the original option's mass alone (ratio 1.02–1.09).
   - "Which option does NOT describe what the customer wants?" is answered correctly in all
     32 two-option BANKING77 cases.
5. **Its limits are exactly TypeSafe's System One limits, and it scores up to 255 options
   without grouping.**

   | Request | Response |
   |---|---|
   | Choice with 1 or 256 options | HTTP 400: "choice question … requires 2-255 options" |
   | Score with 11 levels (10 accepted) | HTTP 400: "score question … requires 2-10 levels" |
   | 256 questions (64 accepted) | HTTP 422: "invalid TypeSafe request: request has 256 questions; the limit is 255" |

   - The first two match Jev's documented limits: "up to 255 options", score "up to 10"
     levels. The third names TypeSafe in the error.
   - The two error formats differ: Azure-style `unsupported_request_argument` versus a bare
     `{"detail": …}`. That suggests two validation layers: a Foundry gateway, and a backend
     that validates requests against the TypeSafe request format.
   - All 255 options are scored in one softmax (finding 2). The two 255-option requests billed
     3,550–3,551 input tokens and returned in 0.57–0.63 s.
   - Accuracy holds from 26 to 27 options (both 0.94 on 16 items), and input tokens grow by
     ≈ 10 per option on both sides of 26. Our own letter readout must split more than 26
     options into groups plus a final pass, which bills ≈ 200 extra tokens at 27 options;
     Decision-1 shows nothing like that.
   - **The letter-naming probe is inconclusive.** Naming options like answer letters (the
     first option called `B`, …) pulled no answer toward the clashing name: 2 of 32 answers
     differed from opaque-id naming, neither toward the clash. But our frozen Qwen3.5-9B,
     which *does* read letters, was unaffected too, so this probe cannot detect letter
     readouts.
6. **Qwen tokenizer, and a very short hidden prompt.**
   - With only the state varying, billed tokens equal the Qwen3.5 token count of the state
     plus a constant on all 18 inputs: 39, or 38 for the 6 states ending in a space (the
     space merges with the template that follows). With the gpt-oss tokenizer the offset
     scatters from 37 to 48.
   - Digits are one token each, as in Qwen: `1234567890` bills 9 more tokens than `x`
     (Qwen: 10 tokens; gpt-oss: 4).
   - Everything except the state costs 39 tokens, of which 18 are our own question and option
     text. That leaves ≈ 21 tokens of hidden formatting, far less than a typical instruction
     prompt.
7. **Not deterministic across requests.**
   - Five repeats of the same request differ by up to 0.04 in a single option's probability
     (mean total-variation distance 0.005–0.006). The chosen option never changed (12 items).
   - This is consistent with batched bf16 GPU serving, where results depend slightly on what
     else is in the batch.

## What this says about the architecture

Confidence levels: **strong** (follows directly from the data), **likely** (best
explanation), **speculation** (plausible, untested).

```text
request -> compact template (~21 hidden tokens) -> Qwen3.5-9B (bf16, batched serving)
           state, billed once --+--> question 1 + all its options -> one bf16 score per option -> softmax
                                +--> question 2 + all its options -> one bf16 score per option -> softmax
```

- **Base model and format (strong).** It uses the Qwen3.5 tokenizer. Microsoft says it is a
  post-trained Qwen3.5-9B. It needs only ≈ 21 tokens of hidden formatting, so it was trained
  on its own compact format rather than steered by a long instruction prompt.
- **Readout (strong).**
  - Exactly one bf16 score per option, softmaxed over that question's options only, with
    nothing else applied afterwards (finding 2).
  - All options are scored together, up to 255, never in groups.
  - The score is not read from an option name's first token.
- **Score source (likely).** The scores' size (16–32) matches a language model's output-layer
  logits, so the readout is probably the regular output layer read at one token per option.
  A bf16 classifier head of similar scale cannot be excluded.
- **Option tokens (speculation).** Qwen splits numbers into single digits and there are only
  26 letters, so 255 single-token option labels would need a dedicated set of option tokens.
- **Options are encoded together and ordering hardly reaches them (strong).**
  - Each option's score depends on the other options present, so options are not scored in
    isolation (finding 3).
  - Reordering moves scores no more than resending the request does (on ACE slightly more),
    and the model cannot say which option is listed first.
  - Order robustness is in the model itself, not obtained by averaging over orders at
    answer time, which would break the grid.
- **Questions (likely).** Each question is evaluated as its own branch from the shared state:
  - Duplicated questions are bit-identical, which rules out one sequence where the second
    copy would see the first.
  - A request bills only one extra question's tokens per added question.
- **Limits (strong).** The limits are exactly TypeSafe's System One limits (finding 5).

**What Parts 2 and 3 added** ([2026-10-21-replication](../2026-10-21-replication/README.md)):
- **Position blindness:** a plain letter readout over ordinary causal attention can become
  blind to positions through decision training.
  - The frozen base, read with a letter readout, answers "which option is listed last?"
    correctly (1.00).
  - Decision-1 and every Qwen3.5-2B arm we trained on 8 other tasks answer at chance
    (0.47–0.53), with or without option shuffling.
- **Order invariance:** our trained arms still flip 6–11% of ACE decisions under
  reordering, against Decision-1's 0%.
- **Names over positions:** every local model, trained or not, gives the earlier of two
  identical options the larger share. Decision-1 prefers the original name wherever the
  copy sits.
- **What would explain the last two:** options that cannot see each other's positions, e.g.
  sharing position indices. Such a design is order-invariant by construction while still
  letting the answer compare all options. Neither it nor any alternative is confirmed.

**Not settled by black-box probing:**
- what the option tokens or head are;
- whether the state is computed once or only billed once;
- whether there is exactly one forward pass;
- whether training used reinforcement learning, fine-tuning or distillation.

## What this says about how it was trained

- **Calibration is trained in, and only for some tasks (strong).**
  - No calibration step follows the model (finding 2), so its calibration comes from
    training.
  - It is calibrated on BANKING77 (fitted T = 1.02) but overconfident on ACE (T = 1.65–2.75),
    *more* so than its untrained base read in one pass (T = 1.33–1.61).
  - The calibration objective worked on routing-style data and did not carry over to
    compliance reasoning.
- **Light-touch post-training that kept the base model's judgement (likely).**
  - The answer scores keep the output layer's natural size.
  - On ACE it matches its base model's accuracy and mistakes (next section).
  - The changes are concentrated in the interface: format, order robustness, option-set
    scoring and calibration on familiar tasks.
- **Many-way classification in the training data (likely).** It gains 12 points over its base
  on BANKING77 (next section).
- **The name preference (speculation).** Preferring an option's original name over an
  identical renamed copy may come from training data with near-duplicate or decoy options.

## Is Decision-1 just Qwen3.5-9B with recalibrated probabilities?

Partly. Its judgement on ACE is the base model's, but post-training changed more than the
probabilities. The comparison model is `Qwen/Qwen3.5-9B` as released (the chat model, read
in one pass with thinking off) through our own letter readout, not Decision-1's prompt.

**Where Decision-1 and its base agree:**

| ACE test | Decision-1 | Qwen3.5-9B frozen |
|---|---|---|
| Accuracy with criteria / names only | 0.741 / 0.619 | 0.741 / 0.647 |
| Same decision | 81% / 83% (kappa 0.68 / 0.66) | |
| Same wrong answer when both are wrong | 91% / 91% | |

For comparison, gpt-oss-20b read the same way agrees with Decision-1 at kappa 0.35–0.42.

**Why it is more than recalibration.** Temperature scaling, or any recalibration that keeps
the ranking of options, cannot change which option wins. Yet:

1. **Decisions differ on 17–19% of examples** (ACE and BANKING77).
   - Some of this is our different prompt: renaming options alone changed up to 2 of 16
     answers (13%) in the probes.
   - Measuring how many decisions a prompt change alone flips for the frozen model would
     settle this; it has not been run.
2. **BANKING77 accuracy is 12 points higher:** 0.870 vs. 0.750. Decision-1 alone is right on
   13.6% of examples, the base alone on 1.6%. Our readout scores 77 options in groups of 26,
   which may cost the base some accuracy.
3. **Calibration changed unevenly.**
   - BANKING77 improved: raw ECE 0.097 → 0.015, T 1.41 → 1.02.
   - ACE got more overconfident: T 1.33 → 1.65 with criteria (raw ECE unchanged at 0.106),
     and T 1.61 → 2.75 without (ECE 0.188 → 0.222).
4. **Behaviour changed in ways no recalibration can produce:**

   | Probe | Decision-1 | Qwen3.5-9B frozen |
   |---|---|---|
   | "Which option is listed last?" (BANKING77) | 0.50 | 1.00 |
   | ACE decisions flipped by reordering | 0% | 7.5% |
   | Copy placed before the original: original name's share (ACE) | 0.84 | 0.41 |
   | Empty state, normalised entropy (ACE / BANKING77) | 0.86 / 0.95 | 0.21 / 0.75 |

   It also works from a ≈ 21-token hidden template and scores up to 255 options in one
   softmax.

In short: Decision-1 = Qwen3.5-9B's judgement + a trained decision interface (compact
format, joint option scoring that ignores order, calibration on routing-style tasks) + better
many-way routing.

## Where the 255 limit comes from

- **It is TypeSafe's System One contract (strong).**
  - Decision-1 enforces exactly Jev's documented per-question limits: 2–255 choice options,
    2–10 score levels.
  - It caps a request at 255 questions, with the error "invalid TypeSafe request".
  - It uses the same request schema and the same confidence formula.
  - Open implementations of the same API inherit the cap too: strands-decider's docs call
    "the schema's 255" its only option limit.
- **Microsoft added a cap Jev's docs don't have (strong).** Jev's documentation states "no
  separate limit on the number of questions", while Decision-1 rejects the 256th question.
  The same number appears for both options and questions.
- **Not a capacity limit of the model (strong).** 255 options cost ≈ 3,550 tokens, far below
  the context length, and are still scored in one softmax.
- **Not letters or numbers (strong).** Letters give 26 single-token labels, and Qwen spells
  numbers digit by digit.
- **Why 255 (speculation).** 255 = 2⁸ − 1 is the largest index that fits in one byte with
  one value reserved (e.g. for padding or "no option"). It is the classic signature of an
  8-bit index, or a table of 256 option tokens with one reserved. Qwen3.5's embedding table
  has 243 unused rows (248,320 rows for 248,077 tokens), so the existing spare rows alone
  would not provide 255 option tokens.

**Checks that would narrow it down:** whether Jev also rejects the 256th question; and
whether option names or descriptions above some length are truncated, which would hint at
per-option slots.

## Reproduce

```powershell
uv run python scripts/run_probes.py --target decision-1          # ~95 min at 9 requests/min
uv run python scripts/probe_report.py --target decision-1
```
