# How Microsoft-Decision-1 computes a decision: probe results (2026-10-20)

Part 1 of the [replication study](../../replication.md): 854 perturbed requests to
Microsoft-Decision-1 (852 answered; the other 2 are the deliberate 256-option test), plus 272
follow-up requests for findings 8 and 9 (R7, R8), 12 for finding 10 (question types) and 96
for finding 11 (R18, option order). Most
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
     step (1,124 answers of the original probe set; 99.9% of 1,624 after adding R6–R8).
     For random values this happens in about 4% of 2-option answers, and far less
     with more options. For comparison, 97% of our own Qwen3.5-9B answers (fp32 output head)
     sit on no grid.
   - 94% sit on the 0.125 grid (or a coarser one; 97% of the 1,624), the spacing of bf16
     numbers between 16
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
     it picked the same option in both orders in all 32 cases. It does not read the order
     we send. **R18 (finding 11) shows why:** the option it picks is the first (or last)
     in byte-wise sorted order, 32 of 32 here and 48 of 48 in R18. Decision-1 sorts the
     options before the model sees them, so every reordering produces the same prompt.
     That explains the 0 flips above. The slightly larger ACE movement (0.014 vs. 0.005)
     is not explained by sorting; it may be serving noise that depends on the batch.
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
8. **Between two identical options, the one whose name fits wins, not the original or the
   first (R7).** The correct option is duplicated with its description kept, under different
   names (160 requests, 16 items per dataset):

   | Naming of the two identical options | Share of the pair (ACE / BANKING77) |
   |---|---|
   | Original renamed `…-2`, copy gets the original name, copy listed after | plain name 0.84 / 0.85 |
   | Same, copy listed before | plain name 0.84 / 0.86 |
   | Original kept, copy named `… (copy)` | original 0.85 / 0.91 |
   | Original kept, copy named `zebra_crossing` (unrelated) | original 0.78 / 0.99 |
   | Both renamed to opaque ids (`opt_3f2a`) | first-listed 0.45 / 0.42 |

   - The probability follows the **plain name**, wherever it sits in the request and
     whichever copy it is on. Names that look like variants (`-2`, `(copy)`) or don't
     match the description lose mass.
   - **Caveat from R18 (finding 11):** in every variant above, the plain name also sorts
     *first* byte-wise (`X` < `X-2` < `X (copy)`, and `zebra_crossing` sorts late), and
     Decision-1 shows options in sorted order. When the copy is named to sort first, an
     unrelated name still loses (original 0.98), but near-identical names split about
     evenly (`0_<name>`: original 0.61; `A <name>`: 0.49). So the effect is name fit
     *plus* a preference for the first option in sorted order.
   - With no name information, the split is near even, with a slight lean to the
     *later* copy.
   - So finding 4's "original name" preference is partly a preference for the name that
     best fits the description, and partly sorted position.
   - **The untrained Qwen3.5-9B with options in request order does not do this** (same
     probe, our letter readout): it gives 0.92 / 0.39 of the pair to the first of two opaque
     copies, and swapping the copies moves the plain name's share from 0.97 to 0.41 on ACE
     (0.88 → 0.70 on BANKING77).
   - **With options sorted like Decision-1's (finding 11), it does:** the plain name gets
     0.97 / 0.88 in both orders, `(copy)` loses (original 0.96 / 0.84), and an unrelated
     copy loses (0.98 / 0.99). So the order-free name preference comes from sorting plus the
     base model's own preference for the first option, not from post-training.
9. **Long option descriptions are read in full, with no per-option limit (R8).** BANKING77
   items with 4 options named by opaque ids; each description holds the real intent, plus
   neutral filler of 0–150 sentences (up to ≈ 1,650 tokens per option, 6,700 per request),
   with the intent at the start or at the end:

   | Filler per option | Billed tokens | Accuracy, intent at start | Accuracy, intent at end |
   |---|---|---|---|
   | none | 97 | 1.00 | – |
   | 10 sentences | 537 | 0.94 | 0.88 |
   | 40 sentences | 1,857 | 0.94 | 0.88 |
   | 150 sentences | 6,697 | 0.94 | 0.94 |

   - Billing grows by exactly 11 tokens per filler sentence per option, with no cut-off.
   - Facts at the end of a 1,650-token description are used as well as facts at the start
     (16 items, differences of 1 item).
   - So there are no fixed per-option slots that truncate long options.
   - The untrained Qwen3.5-9B behaves the same (0.94–1.00 accuracy at every length, intent
     at the start or end), so this capability comes from the base model.
10. **The three question types share one readout but not one template (exploratory: 3
    states, 14 question variants each, plus 9 single-question requests; 2026-10-22).** The
    request schema offers `choice` (named options with criteria), `score` (an ordered list
    of level texts) and `noul` (a yes/no question, criteria optional):

    | | `choice` | `score` | `noul` |
    |---|---|---|---|
    | Answer fields | probabilities by option name, choice, confidence | probabilities by level **index** `"0"`…`"K-1"` + a `legend` to the texts, `score`, confidence | `noul` = P(true) only, no confidence |
    | Output tokens | 1 | 1 | 1 |
    | Billed per option / level beyond its text | ≈ 2 | ≈ 1 (fits 2, 3 and 5 levels exactly) | – |
    | Log-scores on the bf16 0.125 grid | yes (all 15 answers) | yes (all 18 answers) | yes: log-odds of all 41 answers (32 from earlier probes) |
    | Listing order | does not matter (TV ≤ 0.03) | the distribution follows the level texts (TV 0.01–0.08 when reversed), so `score` = Σ index × p flips with the order | – |

    - **`score` is a post-processed choice over digit-indexed levels.** `score` equals the
      expected index Σ i·p_i in the request's order, and confidence the normalised maximum
      probability, as for choice.
    - **But score and choice are not the same prompt.** With the same 5 level texts, the
      ambiguous review ("clean but the staff were rude") gets 0.79 on *mixed* as a score
      and 0.57 on *negative* as a choice (TV 0.40). Clear reviews differ little (TV
      0.004–0.11).
    - **`noul` is a two-way softmax with its own template.** Its log-odds differ by 0.5–1.4
      from 2-option choices (true/false with criteria, yes/no), which agree with each other
      to within 0.5 and do not depend on order. Adding true/false criteria to a `noul`
      moves the log-odds by up to 1.0. Negating the question mirrors the answer (log-odds
      −4.5 ↔ +5.1, +7.25 ↔ −6.9, −6.5 ↔ +6.6).
    - Script and raw responses: session scratch (`score_noul_probe.py`), not in the repo; the
      numbers above are the complete result.
    - Follow-ups on 200 items (R14: templates; R16: many questions per state; R6:
      paraphrases): [2026-10-22-question-types](../2026-10-22-question-types/README.md).
11. **Choice options are sorted byte-wise before the model sees them (R18, 96 requests).**
    - **Position questions follow sorted order.** Five options with mixed-case names
      (`apple`, `Banana`, `cherry`, …), sent in random orders. "Which option is listed
      first / second / last?" picked the option at that place in **byte-wise sorted order
      in 48 of 48** requests. Case-insensitive order fits 25% (first, second), and the order
      we sent 6–31%. Byte order puts uppercase before lowercase (`Banana` < `apple`).
    - **This explains** the 0 decision flips under reordering (finding 3), the model's
      inability to say which option we listed first (it sees a different order), and the
      "position blindness" in the probes. Order invariance is a property of the serving
      code, not necessarily of the model.
    - **Likely mechanism:**
      - The System One request sends choice options as a JSON *object* keyed by option
        name. JSON objects have no defined order, and a server that parses them into a
        sorted map gives exactly byte-wise order.
      - Score levels are sent as a JSON *list*. That keeps their order, which fits finding
        10: reversing the levels reverses the returned `score`.
      - So the sorting may be a side effect of the request format rather than a deliberate
        design.
    - **The model still has a mild first-position preference in sorted order.** Two
      identical options, with the copy named to sort first:

      | Copy's name | Original's share of the pair (BANKING77, 16 items) |
      |---|---|
      | `aardvark` (unrelated) | 0.98 |
      | `0_<name>` | 0.61 |
      | `A <name>` | 0.49 |
      | for comparison, R7: `<name> (copy)` or `<name>-2` (sorts after) | 0.85–0.91 |

    - Script and raw responses: [r18_option_order.py](../../../scripts/r18_option_order.py),
      [r18-decision1.jsonl](../2026-10-22-question-types/r18-decision1.jsonl).

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
- **Options are encoded together, in sorted order (strong).**
  - Each option's score depends on the other options present, so options are not scored in
    isolation (finding 3).
  - The serving code sorts choice options byte-wise by name before building the prompt
    (finding 11). So reordering produces the same prompt, and the model cannot say which
    option was listed first in the request.
  - Order robustness therefore needs no averaging over orders (which would break the grid)
    and no special architecture.
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
  - **Revised by R18:** Decision-1's chance level is measured against the order we send.
    Against byte-sorted order it answers correctly (finding 11). So its "blindness" comes
    from sorting, while the 2B arms' blindness was learned. Our 9B arms (R9, R9b) stayed
    position-aware.
- **Order invariance:** our trained arms still flip 6–11% of ACE decisions under
  reordering, against Decision-1's 0% (which sorting explains).
- **Names over positions:** every local model, trained or not, gives the earlier of two
  identical options the larger share. Decision-1 gives it to the copy whose name best fits
  the description (finding 8), partly because that copy also sorts first (finding 11).
- **What explains these (strong, R18):** sorting the options. No special architecture is
  needed for order invariance.

**Not settled by black-box probing:**
- what the option tokens or head are;
- whether the state is computed once or only billed once;
- whether there is exactly one forward pass;
- whether training used reinforcement learning, fine-tuning or distillation.

## What this says about how it was trained

- **Calibration is trained in, and only for some tasks (strong).**
  - No calibration step follows the model (finding 2), so its calibration comes from
    training.
  - It is calibrated on BANKING77 (fitted T = 1.02) but overconfident on ACE (T = 1.65–2.75).
    Its untrained base, read in one pass, ranges from T = 0.64 to 2.51 on ACE depending only
    on the prompt layout (next section), so ACE shows no calibration gain from training.
  - The calibration objective worked on routing-style data and did not carry over to
    compliance reasoning.
- **Light-touch post-training that kept the base model's judgement (likely).**
  - The answer scores keep the output layer's natural size.
  - On ACE its accuracy and decisions are within what changing the prompt of its untrained
    base produces, and a better prompt makes the untrained base more accurate than
    Decision-1 (next section).
  - The changes are concentrated in the interface: format, option-set scoring and
    calibration on familiar tasks. Order robustness comes from sorting the options
    (finding 11).
- **Many-way classification in the training data (likely).** It gains 12 points over its base
  on BANKING77 (next section).
- **The name preference (strong: not learned).** The frozen base with sorted options shows
  it too (finding 8): unrelated names lose on content, and variant names (`-2`, `(copy)`)
  lose because they sort after the plain name and the base prefers the first option.

## Is Decision-1 just Qwen3.5-9B with recalibrated probabilities?

**On ACE, nothing measurable separates it from its base model. On BANKING77, post-training
clearly changed more than calibration (+12 points). Most probe differences turned out to
come from the serving code sorting the options (finding 11), not from training.** The
comparison model is `Qwen/Qwen3.5-9B` as released (the chat model, read in one pass with
thinking off) through our own letter readout, not Decision-1's hidden prompt.

### ACE: the differences are within what a prompt change alone produces

Same frozen Qwen3.5-9B, same options in the same order, same letter readout. Only the
prompt layout changes:
- **default:** state, question, then options, in XML-style tags;
- **markdown:** question and options first, state last;
- **json:** the whole request as one JSON object.

| ACE test | Accuracy (names only / with criteria) | Agreement with Decision-1 (names / criteria) |
|---|---|---|
| Decision-1 | 0.619 / 0.741 | – |
| Qwen3.5-9B frozen, default prompt | 0.647 / 0.741 | 83% / 81% |
| Qwen3.5-9B frozen, JSON prompt | 0.596 / 0.678 | 80% / 79% |
| Qwen3.5-9B frozen, markdown prompt | **0.775 / 0.767** | 63% / 69% |

| Frozen 9B with one prompt vs. another (ACE test) | Agreement (names / criteria) |
|---|---|
| default vs. JSON | 85% / 85% |
| default vs. markdown | 67% / 69% |
| JSON vs. markdown | 65% / 68% |

- **Prompt layout alone changes 15–35% of the frozen model's ACE decisions.** Decision-1
  differs from the default-prompt frozen model on 17–19%, inside that range. The ACE
  decision differences cannot be attributed to post-training.
- **A different prompt makes the frozen base beat Decision-1 on ACE:** 0.775 vs. 0.619 with
  names only, 0.767 vs. 0.741 with criteria. Post-training did not add compliance ability;
  the base model already has more than Decision-1 shows.
- **Calibration varies with the prompt too.** The fitted temperature of the frozen model on
  ACE ranges from 0.64 (markdown, underconfident) to 2.51 (JSON names only, overconfident).
  Decision-1's ACE temperatures (1.65 / 2.75) are inside that range.
- **Decision-1 agrees most with the state-first prompts** (default, JSON: 79–83%) and least
  with the state-last one (markdown: 63–69%). That is consistent with Decision-1's hidden
  template putting the state first. Sharing one state across several questions (finding 1)
  requires that anyway.
- The default prompt, rerun in this job, reproduced the earlier run exactly (0.6473 /
  0.7409), so the readout is deterministic. BANKING77 was not re-run with other prompts.

For comparison, gpt-oss-20b read the same way agrees with Decision-1 at kappa 0.35–0.42,
against 0.66–0.68 for the default-prompt frozen 9B.

### Where post-training clearly changed more than calibration

Temperature scaling, or any recalibration that keeps the ranking of options, cannot change
which option wins. Yet:

1. **BANKING77 accuracy is 12 points higher:** 0.870 vs. 0.750 (default prompt).
   - Decision-1 alone is right on 13.6% of examples, the base alone on 1.6%.
   - Caveat: our readout scores 77 options in groups of 26. R4 and R5 later showed that
     no other label scheme or prompt layout lifts the frozen base above 0.750, so the gap
     is not our readout or prompt.
2. **BANKING77 calibration improved:** raw ECE 0.097 → 0.015, temperature 1.41 → 1.02.
3. **Behaviour changed in ways no recalibration can produce:**

   | Probe | Decision-1 | Qwen3.5-9B frozen | Frozen, options sorted like Decision-1 |
   |---|---|---|---|
   | "Which option is listed last?" (BANKING77) | 0.50 (1.00 against sorted order, finding 11) | 1.00 | 0.50 |
   | ACE decisions flipped by reordering | 0% (sorted options) | 7.5% | 0% |
   | Copy placed before the original: original name's share (ACE) | 0.84 | 0.41 | 0.97 |
   | Empty state, normalised entropy (ACE / BANKING77) | 0.86 / 0.95 | 0.21 / 0.75 | 0.08–0.19 / 0.73–0.90 |

   The first three rows come from the serving code sorting the options (finding 11), not
   from training: the frozen base with sorted options reproduces them. Only the
   flatter empty-state answer on BANKING77 remains a behavioural change. It also works from
   a ≈ 21-token hidden template and scores up to 255 options in one softmax.

In short: Decision-1 = Qwen3.5-9B's judgement, unchanged on compliance reasoning, + a
trained decision interface (compact format, joint option scoring, calibration on
routing-style tasks) + options sorted before scoring + better many-way routing.

## Where the 255 limit comes from

- **It is TypeSafe's System One contract (strong).**
  - Decision-1 enforces exactly Jev's documented per-question limits: 2–255 choice options,
    2–10 score levels.
  - It caps a request at 255 questions, with the error "invalid TypeSafe request".
  - It uses the same request schema and the same confidence formula.
  - Open implementations of the same API inherit the cap too: strands-decider's docs call
    "the schema's 255" its only option limit, although its pointer head has no per-option
    parameters and could score any number of options.
- **The same 255 caps questions, which need no label tokens (strong).** Jev's documentation
  states "no separate limit on the number of questions", while Decision-1 rejects the 256th
  question. A limit shared by options and questions looks like a constant of the request
  format (e.g. counts or indices stored in one byte), not of the vocabulary.
- **Not a capacity limit of the model (strong).** 255 options cost ≈ 3,550 tokens, far below
  the context length, and are still scored in one softmax.
- **Why 255 (speculation).** 255 = 2⁸ − 1 is the largest value of one unsigned byte, with one
  value left for "none" or padding: a common engineering default for a per-request index.
  TypeSafe's launch post says applications needing more options chain two stages, so the cap
  is a product decision.
- **The score cap likely comes from the tokenizer (likely, finding 10).** Score questions
  allow at most 10 levels, and Qwen has exactly 10 single-digit tokens (`0`–`9`). The API's
  own answer keys score levels by digit index (`"0"`…`"9"` with a `legend` to the level
  texts), and each level costs about 1 token beyond its text. Reading score levels as
  digit tokens would explain the cap.

### What the 255 option labels could be

Whatever Decision-1 reads, it reads exactly one bf16 score per option (finding 2). The
billing also limits what an option costs. Fitted over 530 requests, each option is billed
its name's tokens, plus its description's tokens, plus **about 2 more** (2 is the best
integer fit; individual requests scatter by a few tokens). So the label and its separator
take about 2 tokens: something like `<label>` + newline. That rules out multi-token labels
such as `Option 137:`.

| Possible labels | Vocabulary change | What Qwen3.5-9B offers | Drawback |
|---|---|---|---|
| Existing single tokens | None | 26 + 26 letters, 10 digits, 562 two-letter codes (`AA`, `AB`, …), thousands of CJK characters | Labels keep their old meanings, and option names can collide with labels |
| Repurposed unused rows | New tokens on existing rows | 243 spare embedding rows (248,320 rows for 248,077 tokens), plus ≈ 30 multimodal special tokens that a text-only service never uses | Spare rows alone fall short of 255 |
| New tokens | Embedding and output tables grow by 255–256 rows | Routine in fine-tuning | The new rows must be trained; LoRA on the existing layers alone does not train them |

- **Not plain letters or numbers (strong).** Letters give 26 single-token labels, and Qwen
  spells numbers digit by digit.
- **Not distinguishable from outside.** The API never returns tokens. The tokenizer check
  (finding 6) only covers our text; label tokens would sit in the hidden template.
- **Not part of calibration.** Choosing or adding label tokens belongs to format training.
  Calibration only adjusts weights so the probabilities on those labels are honest.

**Checks:**
- Long descriptions are not truncated (finding 9), so there are no fixed per-option slots.
- Still open: whether Jev also rejects the 256th question.

## Reproduce

```powershell
uv run python scripts/run_probes.py --target decision-1          # ~95 min at 9 requests/min
uv run python scripts/run_probes.py --target decision-1 --probe duplicate_names long_options   # R7, R8: ~30 min
uv run python scripts/probe_report.py --target decision-1
```
