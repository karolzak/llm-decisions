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
   - A post-hoc temperature divides every gap by T, which would break the grid unless T is a
     power of two. Any calibration therefore comes from the network itself. Decision-1 is
     calibrated out of the box on BANKING77 (fitted T = 1.02) but not on ACE (T = 1.65 with
     criteria, 2.75 without; see the [C-series results](../2026-10-14-c-series-banking77/README.md)).
   - `confidence` equals Jev's documented formula `(p_max − 1/K)/(1 − 1/K)` on every choice
     answer.
3. **It cannot tell which option is listed where, and reordering never changes its
   decision.**
   - Reordering options never flipped the decision: 0 of 80 reorderings on ACE (all 6 orders
     of 3 options) and 0 of 64 on BANKING77 (5 orders of 6 options). There is no first-vs-last
     preference (log lift −0.005 on ACE, +0.02 on BANKING77).
   - On BANKING77, reordering moves the probabilities no more than repeating the same request
     does (mean total-variation distance 0.0035 vs. 0.006). On ACE it moves them about 2.6×
     more (0.014 vs. 0.005, at most 0.10), so a small order effect remains there.
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
5. **Up to 255 options, with no sign of the grouping a 26-letter readout would need.**
   - 256 options are rejected: HTTP 400, "choice question decision requires 2-255 options".
     255 is also Jev's documented limit.
   - Accuracy holds from 26 to 27 options (both 0.94 on 16 items), and input tokens grow by
     ≈ 10 per option on both sides of 26. Our own letter readout must split more than 26
     options into groups plus a final pass, which bills ≈ 200 extra tokens at 27 options;
     Decision-1 shows nothing like that.
   - This does not rule out a readout over other single tokens (numbers, reserved tokens).
     Naming options like answer letters (the first option called `B`, …) pulled no answer
     toward the clashing name: 2 of 32 answers differed from opaque-id naming, neither
     toward the clash. But this probe cannot detect letter readouts anyway: our frozen
     Qwen3.5-9B, which *does* read letters, was unaffected too.
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

**Supported by the evidence:**
- **Prompt and readout:** a decoder LM with the Qwen3.5 tokenizer reads the state, the
  question and all the options. The answer is read at one position per question, as bf16
  logits over the options, with no arbitrary post-hoc temperature. A request's questions are
  evaluated together.
- **Interface:** it implements TypeSafe's System One contract: the same request schema, the
  same 255-option limit and the same confidence formula.

**What Parts 2 and 3 added:**
- **Position blindness comes from decision training.** The frozen base model, read with a
  plain letter readout, *can* answer "which option is listed last?" (1.00) and flips 7.5%
  of ACE decisions under reordering. Decision-1 cannot answer it (0.50) and never flips.
  Qwen3.5-2B trained with the same letter readout on 8 other tasks also lost the ability
  (0.47–0.53), with or without option shuffling. No special architecture, such as isolating
  options from each other, is needed to explain it.
- **Still unexplained:** Decision-1 prefers the original option name over an identical renamed
  copy wherever the copy sits. Every local model we tried, trained or not, prefers the
  earlier copy instead.

**Not settled by black-box probing:** whether the readout uses letters, numbers, reserved
option tokens or a separate head; whether the state is computed once or only billed once;
whether there is exactly one forward pass.

## Reproduce

```powershell
uv run python scripts/run_probes.py --target decision-1          # ~95 min at 9 requests/min
uv run python scripts/probe_report.py --target decision-1
```
