# How decision models work under the hood

Microsoft-Decision-1 looks like an ordinary open LLM (Qwen3.5-9B) used as a classifier (prompt
based), with a thin layer of engineering and training on top:

- the options are sorted alphabetically before the model sees them;
- each option is assigned a single token to be represented with (e.g. A, B, C..);
- context + question + options are fed into the LLM;
- logits scores of the next token are filtered by tokens we used for options list;
- the API returns a plain softmax of those scores.

Most of its judgement matches its base model run without any extra training. What its training
clearly adds is better choosing among many options (for example, picking 1 of 77 customer
intents) and well-calibrated probabilities on some tasks. Nobody outside Microsoft knows how that
training was done, and we could not reproduce that part.

## Background

- **What a "decision model" is:** you send a situation (the "state") and a fixed list of
  options, and you get back a probability for each option instead of free text. Examples are
  Microsoft-Decision-1 (on Azure AI Foundry), TypeSafe's Jev and the OpenAI Decisions API.
- **What is public:** Microsoft says only that Decision-1 is Qwen3.5-9B "post-trained for fast,
  single-pass decision scoring".
- **What we tested:** only Decision-1. Jev and the OpenAI Decisions API were not tested because
  we had no API keys.
- **How we tested it:**
  1. **About 2,000 deliberately altered requests to the API.** We reordered, renamed and
     duplicated options, changed the question, sent empty inputs, used 2–256 options and put
     many questions in one request. Then we analysed the returned probabilities and the token
     counts we were billed for.
  2. **Comparison with the base model.** We ran Qwen3.5-9B, without any extra training, in the
     same way and compared the two decision by decision.
  3. **Our own replicas.** We fine-tuned Qwen3.5-9B with LoRA (a lightweight fine-tuning
     method) to see which behaviours we could reproduce.
- **Datasets:**
  - ACE: is a scenario compliant with contract clauses? 3 options.
  - BANKING77: which of 77 banking intents a customer message matches.
  - Three sets we never used for tuning: MInDS-14 and ATIS (intent sets) and ContractNLI
    (contract reasoning).
- **How we measure calibration (whether its confidence can be trusted).** A model is calibrated
  if its probabilities can be taken at face value: when it says "90% sure", it is right about
  90% of the time.
  - We take the probabilities Decision-1 returns and ask how much they would have to be
    sharpened or flattened to match how often it is actually right. That adjustment is one
    number, the **temperature T**.
  - To find it, we convert the probabilities back to scores (log), divide them by T, re-apply
    the softmax and pick the T that best fits the correct answers on labelled data. This never
    changes which option wins, only how confident the answer is.
  - **T ≈ 1:** honest. **T > 1:** overconfident (its probabilities have to be flattened).
    **T < 1:** underconfident (they have to be sharpened).
  - T is a measurement we make on the outputs. It is not a setting inside Decision-1.

## How we think a request is processed

```text
request ──► options sorted alphabetically by name (serving code)
        ──► short hidden template (~21 tokens) + Qwen3.5-9B, post-trained, in bf16 precision
              state (billed once) ─┬─► question 1 + all its options ─► one score per option ─► softmax ─► probabilities
                                   └─► question 2 + all its options ─► one score per option ─► softmax ─► probabilities
```

Each step is rated below by how sure we are.

## Confirmed by measurement

**How the answer is computed**

- **No text is generated.** Each question is billed exactly one output token, even when option
  names are long.
- **The probabilities are a plain softmax over one score per option.**
  - In 99.8% of answers, the gaps between returned log-probabilities are exact multiples of a
    step of the 16-bit floating-point format (bf16) the model computes in.
  - Any post-processing would break that pattern: a temperature, averaging over option orders,
    scoring in groups, or combining tokens.
  - The pattern holds up to 255 options, so all options are scored in one go.
- **Calibration is not a separate step after the model.** Since nothing is applied after the
  softmax, any calibration must come from training.
- **Options are scored together, not one by one.** The score gap between two options changes
  when other options are added.
- **The `confidence` field** equals `(p_max − 1/K) / (1 − 1/K)`, the formula TypeSafe
  documents for Jev (K is the number of options).

**How the input is handled**

- **Options are sorted by name before the model sees them.** The order is byte-wise, so
  uppercase comes before lowercase, and the sort key is the name, not the description.
  - Asked "which option is listed first / second / last?", it answered according to sorted
    order in 48 of 48 requests.
  - This is why reordering options never changes its decisions (Microsoft's "zero flips"
    claim). The order robustness comes from the serving code, not from the model.
- **Tokenizer:** it uses Qwen3.5's tokenizer, with only about 21 tokens of hidden prompt. So it
  was trained on a compact format, not steered by long instructions.
- **Long option descriptions are read in full.** We tested up to about 1,650 tokens per option,
  and facts at the very end were still used.
- **Several questions in one request share the state.**
  - The state is billed once; each extra question adds only its own tokens.
  - The same question asked twice in one request gets bit-identical answers.

**API contract**

- **The limits are exactly those of TypeSafe's "System One" API (the API Jev is served
  through):** 2–255 options, 2–10 score levels and at most 255 questions.
  - Exceeding the question limit returns the error "invalid TypeSafe request".
  - So Decision-1 sits behind a TypeSafe-compatible request layer. This says nothing about
    whether the models themselves are related.
- **Results vary slightly between identical requests:** up to 0.04 in one option's
  probability, though the top answer never changed. This is typical of GPU serving that
  processes requests in batches.

## Strong evidence: most of the judgement comes from the base model

Compared with Qwen3.5-9B run without any extra training:

- **Contract compliance (ACE):** both score 0.741 when option descriptions are given, and they
  make the same decision 81–83% of the time.
  - Changing only the prompt layout changes 15–35% of the base model's decisions.
  - One layout makes the untrained base *better* than Decision-1 (0.767–0.775). That layout
    was chosen on a separate data split, not on the test set.
- **Yes/no questions:** the same answer on 95–100% of 200 items, and nearly identical
  probabilities (correlation 0.95–0.99).
- **Order and naming behaviour:** with its options sorted the same way, the untrained base
  reproduces Decision-1's behaviour without any training. This was checked on only 16 items
  per dataset.

**What post-training clearly adds:**

- **Choosing among many options:**
  - +12 points on BANKING77 (0.870 vs. 0.750);
  - +5.7 on MInDS-14 and +8.8 on ATIS, sets we never used for tuning. So this is a general
    skill, not memorisation of BANKING77.
  - Our readout and prompt variants of the base never closed this gap, so the gain is not an
    artefact of how we ran the base.
- **Calibration, but only on some tasks** (T as defined in Background: 1 = honest,
  \>1 = overconfident, <1 = underconfident):
  - **BANKING77:** T = 1.02. Well calibrated: when it says 90%, it is right about 90% of the
    time. Its untrained base needs T = 1.41 (overconfident).
  - **Unseen intent sets (MInDS-14, ATIS):** T ≈ 0.7. Underconfident: it is right more often
    than it says. This T was fitted on the same items it is reported on.
  - **Contract tasks:** ACE T = 1.65 with option descriptions and 2.75 without; ContractNLI
    T = 1.60. Overconfident: it is right less often than it says. On ACE the untrained base
    ranges from 0.64 to 2.51 depending only on the prompt layout, so training shows no
    calibration gain there.
  - **In practice:** its probabilities can drive an automate-or-escalate threshold on routing
    tasks like BANKING77. On compliance-style tasks, recalibrate them on your own labelled
    data first.

**What it does not appear to add:** robustness to rewording the question. Rewording flipped
6.7% of Decision-1's ACE decisions against 2.7% for the base (50 items per dataset), so this is
a likely finding rather than a firm one.

## Likely: the best explanation, not proven

- **The scores are the model's normal next-token predictions, read once per option label.**
  - The score sizes (about 16–32) are typical for those predictions.
  - A separate classifier layer can't be ruled out.
- **Each option gets a short label.**
  - Options are billed about 2 tokens beyond their name and description: likely a label plus
    a separator.
  - The labels can't be plain letters (only 26) or numbers (Qwen splits numbers into single
    digits).
  - They could be two-letter codes, or reused or newly added tokens.
- **The sorting is probably a side effect of the request format.** Options are sent as a JSON
  object keyed by name, and a server that parses it into a sorted map produces exactly this
  byte-wise order. Score levels are sent as a list and keep their order.
- **Score levels are probably read as the digit tokens 0–9.** That would explain the 10-level
  cap.
- **One forward pass per request.**
  - Microsoft says "single-pass", and latency stays flat from 2 to 77 options (about
    450–480 ms). 255 options take about 0.6 s.
  - Timing alone can't prove it.
- **Questions are computed in parallel as branches from a shared state.**
  - The bit-identical duplicates rule out the questions being processed one after another in a
    single sequence.
  - We reproduced the speed this way: 64 questions in about 0.85 s on one A100 GPU, against
    Decision-1's ≈ 0.6 s of server time.
  - Billing proves only that the state is *charged* once, not that it is *computed* once.
- **Post-training was light** and kept the base model's judgement. This is inferred from how
  closely Decision-1 matches its base.

## Unknown or not testable from outside

- The exact option labels and hidden template.
- How it was trained (fine-tuning, reinforcement learning or distillation) and on what data. In
  particular, what produces the gain on many options.
- Whether BANKING77's own training data was in its training. The gains on unseen sets show this
  can't be the whole explanation.
- Whether Jev and the OpenAI Decisions API work the same way.
  - Jev's documentation says it reads the state once, answers all questions in parallel,
    allows up to 255 options, leans slightly toward the first option and is trained with
    reinforcement learning for calibrated decisions.
  - Decision-1 shares Jev's API limits and confidence formula, and in sorted order it also
    leans slightly toward the first option. Beyond that, a shared mechanism is an assumption.

## Can it be reproduced with open models?

- **The interface: yes, mostly without training.** That means sorting the options, reading all
  options from one forward pass of the open base model, and computing the state once for
  several questions.
- **Contract reasoning: yes.** The untrained base already matches Decision-1, and our
  fine-tuned versions match or beat it on ACE and ContractNLI (on ContractNLI by a small
  margin, on 201 items).
- **The gain on many options: no.**
  - Our fine-tuned Qwen3.5-9B models stay 12–14 points below Decision-1 on BANKING77 and
    several points below on the unseen intent sets.
  - One run matched it on MInDS-14 only.
  - **What we tried, one run each:** more and broader real routing data, data from
    BANKING77's own domain, 43,000 synthetic messages from 199 domains, 5× longer training,
    and reinforcement learning with an accuracy reward. We judged each against run-to-run
    noise measured over 4 runs.
  - **Not yet tried:** much larger-scale training, full fine-tuning instead of LoRA,
    distillation from a stronger model, and a reinforcement-learning reward for calibration
    (the approach Jev's documentation describes).
