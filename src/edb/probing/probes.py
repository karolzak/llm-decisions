"""Probe definitions: perturbations of real decisions, and what each one reveals.

A probe turns base items into `ProbeCase`s (one request each) and turns the stored records
back into metrics. Records are plain dicts (see `runner.record`), so analysis also works
on outputs collected from another machine.

Two candidate designs for a decision model's single-pass readout make different predictions:

    letter readout  options shown as A, B, C...; the model's next-token probability of each
                    letter is read at one answer position (H2O, Quyet, Cygnet, our C6)
    pointer head    the LM head is dropped; a small head scores each option's own hidden
                    state against the answer position (strands-decider, our C5)

Each probe's docstring states what it measures and what each design predicts.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import string
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

import numpy as np

from edb.probing.api import DecisionRequest, Option, Question
from edb.probing.items import BaseItem, rng_for, subset

QNAME = "decision"
LETTERS = string.ascii_uppercase


@dataclass
class ProbeCase:
    probe: str
    case_id: str  # unique within the probe, stable across runs
    item_id: str
    variant: str
    request: DecisionRequest
    meta: dict[str, Any] = field(default_factory=dict)


def choice_request(item: BaseItem, options: list[Option] | None = None,
                   instructions: str | None = None, state: Any = None) -> DecisionRequest:
    q = Question(QNAME, instructions or item.instructions, "choice",
                 list(options if options is not None else item.options))
    return DecisionRequest(item.state if state is None else state, [q])


# --------------------------------------------------------------------------- record helpers
def probs_of(rec: dict[str, Any], qname: str = QNAME) -> dict[str, float] | None:
    resp = rec.get("response")
    if not resp:
        return None
    a = (resp.get("answers") or {}).get(qname)
    if not a or not a.get("probabilities"):
        return None
    return a["probabilities"]


def top(p: dict[str, float]) -> str:
    return max(p, key=p.get)


def order_of(rec: dict[str, Any], qname: str = QNAME) -> list[str]:
    for q in rec["request"]["questions"]:
        if q["name"] == qname:
            return [o["value"] for o in q["options"]]
    raise KeyError(qname)


def source_of(rec: dict[str, Any]) -> str:
    return rec["item_id"].split(":", 1)[0]


def by_source(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        out[source_of(r)].append(r)
    return dict(out)


def mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return round(float(np.mean(xs)), 4) if xs else None


def tv(p: dict[str, float], q: dict[str, float]) -> float:
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in set(p) | set(q))


def norm_entropy(p: dict[str, float]) -> float:
    vals = np.array([v for v in p.values() if v > 0])
    return float(-(vals * np.log(vals)).sum() / math.log(len(p))) if len(p) > 1 else 0.0


# --------------------------------------------------------------------------- base class
class Probe(ABC):
    name: ClassVar[str]
    summary: ClassVar[str]

    def __init__(self, seed: int = 0):
        self.seed = seed

    @abstractmethod
    def cases(self, items: dict[str, list[BaseItem]]) -> list[ProbeCase]: ...

    @abstractmethod
    def analyze(self, records: list[dict[str, Any]]) -> dict[str, Any]: ...

    def case(self, item_id: str, variant: str, request: DecisionRequest,
             **meta: Any) -> ProbeCase:
        return ProbeCase(self.name, f"{item_id}|{variant}", item_id, variant, request, meta)


# --------------------------------------------------------------------------- probes
class OrderProbe(Probe):
    """Reorder the options. Microsoft reports zero decision flips when options are reversed
    or shuffled; Jev's docs admit a lean toward the first option. A letter readout without
    order augmentation in training leans toward early letters; a pointer head trained with
    shuffled options is nearly order-free (strands: reversal moves probabilities ~0.016).

    ACE: all 6 orders of its 3 options. BANKING77: 6-option subsets in 5 orders."""

    name = "order"
    summary = "Does reordering the options change the decision? Is there a position bias?"
    k_banking = 6

    def cases(self, items):
        out = []
        for item in items.get("ace", []):
            for perm in itertools.permutations(range(len(item.options))):
                opts = [item.options[i] for i in perm]
                out.append(self.case(item.id, "perm_" + "".join(map(str, perm)),
                                     choice_request(item, opts), gold=item.gold,
                                     identity=perm == tuple(range(len(perm)))))
        for item in items.get("banking77", []):
            sub = subset(item, self.k_banking, self.seed)
            base = list(sub.options)
            k = len(base)
            orders = {"identity": base, "reversed": base[::-1],
                      "rotate2": base[2:] + base[:2], "rotate4": base[4:] + base[:4]}
            shuffled = base[:]
            rng_for("order-probe", self.seed, item.id).shuffle(shuffled)
            orders["shuffled"] = shuffled
            for name, opts in orders.items():
                out.append(self.case(item.id, name, choice_request(sub, opts), gold=item.gold,
                                     identity=name == "identity", k=k))
        return out

    def analyze(self, records):
        out = {}
        for source, recs in by_source(records).items():
            items: dict[str, list[dict]] = defaultdict(list)
            for r in recs:
                if probs_of(r):
                    items[r["item_id"]].append(r)
            flips, tvs, agree_all, acc = [], [], [], []
            chosen_pos, gold_pos = [], []
            first, last = defaultdict(list), defaultdict(list)
            for item_id, rs in items.items():
                ident = next((r for r in rs if r["meta"].get("identity")), None)
                for r in rs:
                    p, order = probs_of(r), order_of(r)
                    acc.append(top(p) == r["meta"]["gold"])
                    chosen_pos.append(order.index(top(p)) / max(1, len(order) - 1))
                    gold_pos.append(order.index(r["meta"]["gold"]) / max(1, len(order) - 1))
                    first[(item_id, order[0])].append(p[order[0]])
                    last[(item_id, order[-1])].append(p[order[-1]])
                    if ident is not None and r is not ident:
                        flips.append(top(p) != top(probs_of(ident)))
                        tvs.append(tv(p, probs_of(ident)))
                agree_all.append(len({top(probs_of(r)) for r in rs}) == 1)
            lifts = [math.log(max(1e-9, np.mean(first[key]))) -
                     math.log(max(1e-9, np.mean(last[key]))) for key in first if key in last]
            out[source] = {
                "items": len(items),
                "accuracy": mean(acc),
                "flip_rate_vs_original": mean(flips),
                "items_consistent_all_orders": mean(agree_all),
                "mean_tv_vs_original": mean(tvs),
                # 0 = first option, 1 = last option; compare chosen vs gold mean position.
                "mean_chosen_position": mean(chosen_pos),
                "mean_gold_position": mean(gold_pos),
                # log P(option | listed first) - log P(option | listed last), same item/option.
                "first_vs_last_log_lift": mean(lifts),
            }
        return out


class LetterNamesProbe(Probe):
    """Rename the options while keeping their meaning in the description:

        neutral          opaque ids (opt_xxxx)
        letters_aligned  A, B, C... in listed order
        letters_shifted  the first option is named B, the second C, ..., the last A

    A letter readout that shows options as "A. <name>" sees "A. B: ..." under
    letters_shifted; if names and internal letters collide, its choice is pulled toward the
    option *named* like the internal letter of the intended position (`collision_pull`).
    A pointer head reads each option's content and is unaffected by names."""

    name = "letter_names"
    summary = "Do option names that look like answer letters confuse the readout?"
    k_banking = 4

    def _renamed(self, item: BaseItem, scheme: str) -> list[Option]:
        k = len(item.options)
        opts, used = [], set()
        for i, o in enumerate(item.options):
            meaning = o.value + (f": {o.description}" if o.description else "")
            if scheme == "neutral":
                salt = 0
                while (name := f"opt_{hash_str(f'{item.id}:{o.value}:{salt}') % 16**4:04x}") \
                        in used:
                    salt += 1
                used.add(name)
            elif scheme == "letters_aligned":
                name = LETTERS[i]
            else:
                name = LETTERS[(i + 1) % k]
            opts.append(Option(name, meaning))
        return opts

    def cases(self, items):
        out = []
        pool = [*items.get("ace", []), *(subset(i, self.k_banking, self.seed)
                                         for i in items.get("banking77", []))]
        for item in pool:
            if len(item.options) > 26:
                continue
            originals = item.values
            out.append(self.case(item.id, "original", choice_request(item),
                                 gold=item.gold, originals=originals))
            for scheme in ("neutral", "letters_aligned", "letters_shifted"):
                opts = self._renamed(item, scheme)
                out.append(self.case(item.id, scheme, choice_request(item, opts), gold=item.gold,
                                     originals=originals))
        return out

    def analyze(self, records):
        out = {}
        for source, recs in by_source(records).items():
            items: dict[str, dict[str, dict]] = defaultdict(dict)
            for r in recs:
                if probs_of(r):
                    items[r["item_id"]][r["variant"]] = r
            acc: dict[str, list[bool]] = defaultdict(list)
            agree: dict[str, list[bool]] = defaultdict(list)
            pull = []
            for vs in items.values():
                pos = {}
                for variant, r in vs.items():
                    order, p = order_of(r), probs_of(r)
                    pos[variant] = order.index(top(p))
                    acc[variant].append(r["meta"]["originals"][pos[variant]] == r["meta"]["gold"])
                if "neutral" not in pos:
                    continue
                for variant in ("original", "letters_aligned", "letters_shifted"):
                    if variant in pos:
                        agree[variant].append(pos[variant] == pos["neutral"])
                if "letters_shifted" in pos and pos["letters_shifted"] != pos["neutral"]:
                    k = len(vs["letters_shifted"]["meta"]["originals"])
                    # The option named like the neutral choice's position letter sits one
                    # position earlier under the shifted naming.
                    pull.append(pos["letters_shifted"] == (pos["neutral"] - 1) % k)
            out[source] = {
                "items": len(items),
                **{f"accuracy_{v}": mean(a) for v, a in sorted(acc.items())},
                **{f"agree_with_neutral_{v}": mean(a) for v, a in sorted(agree.items())},
                "shifted_disagreements": len(pull),
                "collision_pull": mean(pull),
            }
        return out


def hash_str(text: str) -> int:
    return int(hashlib.sha1(text.encode()).hexdigest()[:12], 16)


class DuplicateProbe(Probe):
    """Add a copy of the gold option (same description, value suffixed "-2"):

        dup_after   [..., gold, gold-2, ...]
        dup_before  [..., gold-2, gold, ...]

    A pointer head scores options by content, so the two copies get near-equal mass
    (`median_abs_log_ratio` near 0). A letter readout with a position prior favours whichever
    copy comes first in both variants (`earlier_copy_share` > 0.5 in both). A preference for
    the original name in both variants points to the readout reading option names."""

    name = "duplicate"
    summary = "How is probability split between two identical options?"
    k_banking = 4

    def cases(self, items):
        out = []
        pool = [*items.get("ace", []), *(subset(i, self.k_banking, self.seed)
                                         for i in items.get("banking77", []))]
        for item in pool:
            gold = item.option(item.gold)
            copy = Option(gold.value + "-2", gold.description)
            i = item.values.index(item.gold)
            base = list(item.options)
            out.append(self.case(item.id, "base", choice_request(item, base), gold=item.gold,
                                 copy=copy.value))
            after = base[: i + 1] + [copy] + base[i + 1:]
            before = base[:i] + [copy] + base[i:]
            out.append(self.case(item.id, "dup_after", choice_request(item, after),
                                 gold=item.gold, copy=copy.value))
            out.append(self.case(item.id, "dup_before", choice_request(item, before),
                                 gold=item.gold, copy=copy.value))
        return out

    def analyze(self, records):
        out = {}
        for source, recs in by_source(records).items():
            items: dict[str, dict[str, dict]] = defaultdict(dict)
            for r in recs:
                if probs_of(r):
                    items[r["item_id"]][r["variant"]] = r
            earlier: dict[str, list[float]] = defaultdict(list)
            orig_share: dict[str, list[float]] = defaultdict(list)
            log_ratio, conserve = [], []
            for vs in items.values():
                for variant in ("dup_after", "dup_before"):
                    r = vs.get(variant)
                    if r is None:
                        continue
                    p, gold, copy = probs_of(r), r["meta"]["gold"], r["meta"]["copy"]
                    total = p[gold] + p[copy]
                    if total <= 0:
                        continue
                    first = gold if variant == "dup_after" else copy
                    earlier[variant].append(p[first] / total)
                    orig_share[variant].append(p[gold] / total)
                    log_ratio.append(abs(math.log(max(p[gold], 1e-12) / max(p[copy], 1e-12))))
                    if "base" in vs:
                        conserve.append(total / max(probs_of(vs["base"])[gold], 1e-12))
            out[source] = {
                "items": len(items),
                "earlier_copy_share_dup_after": mean(earlier["dup_after"]),
                "earlier_copy_share_dup_before": mean(earlier["dup_before"]),
                "original_name_share_dup_after": mean(orig_share["dup_after"]),
                "original_name_share_dup_before": mean(orig_share["dup_before"]),
                "median_abs_log_ratio": round(float(np.median(log_ratio)), 4) if log_ratio
                else None,
                # (P(gold) + P(copy)) with the duplicate / P(gold) without it.
                "mass_conservation": mean(conserve),
            }
        return out


class CardinalityProbe(Probe):
    """Grow the option list (BANKING77, nested subsets of K options) and test the limit.

    One forward pass over a listed-options prompt costs input tokens linear in K and latency
    nearly flat in K. A letter readout runs out of single letters after 26: look for a
    discontinuity between K=26 and K=27 in accuracy or tokens per option. Jev documents a
    255-option limit and a two-stage pass for large K (extra latency); K=255/256 probe it."""

    name = "cardinality"
    summary = "How do accuracy, tokens and latency scale with the number of options?"
    ks = (2, 4, 8, 16, 26, 27, 40, 77)
    limit_ks = (255, 256)
    limit_items = 2

    def cases(self, items):
        out = []
        bank = items.get("banking77", [])
        for item in bank:
            for k in self.ks:
                sub = subset(item, k, self.seed)
                out.append(self.case(item.id, f"k{k:03d}", choice_request(sub), gold=item.gold,
                                     k=len(sub.options)))
        for item in bank[: self.limit_items]:
            for k in self.limit_ks:
                extra = [Option(f"extra_option_{i:03d}", f"An unrelated category number {i}.")
                         for i in range(k - len(item.options))]
                opts = [*item.options, *extra]
                rng_for("limit", self.seed, item.id, k).shuffle(opts)
                out.append(self.case(item.id, f"k{k:03d}", choice_request(item, opts),
                                     gold=item.gold, k=k, limit=True))
        return out

    def analyze(self, records):
        rows = defaultdict(lambda: defaultdict(list))
        limits = {}
        for r in records:
            k = r["meta"]["k"]
            if r["meta"].get("limit"):
                limits[f"k{k}"] = "error: " + (r.get("error") or "")[:120] if r.get("error") \
                    else "accepted"
                continue
            row = rows[k]
            row["errors"].append(r.get("error") is not None)
            p = probs_of(r)
            if p is None:
                continue
            row["acc"].append(top(p) == r["meta"]["gold"])
            row["p_gold"].append(p.get(r["meta"]["gold"], 0.0))
            resp = r["response"]
            if resp.get("input_tokens") is not None:
                row["tokens"].append(resp["input_tokens"])
            row["latency"].append(resp["latency_ms"])
        per_k = {k: {"n": len(v["acc"]), "error_rate": mean(v["errors"]),
                     "accuracy": mean(v["acc"]), "mean_p_gold": mean(v["p_gold"]),
                     "input_tokens": mean(v["tokens"]),
                     "latency_p50_ms": round(float(np.median(v["latency"])), 1)
                     if v["latency"] else None}
                 for k, v in sorted(rows.items())}
        ks = [k for k in per_k if per_k[k]["input_tokens"] is not None]
        slope = intercept = None
        if len(ks) >= 2:
            slope, intercept = np.polyfit(ks, [per_k[k]["input_tokens"] for k in ks], 1)
        lat = [k for k in per_k if per_k[k]["latency_p50_ms"] is not None]
        lat_slope = np.polyfit(lat, [per_k[k]["latency_p50_ms"] for k in lat], 1)[0] \
            if len(lat) >= 2 else None

        def step(a: int, b: int) -> float | None:
            if a in per_k and b in per_k and per_k[a]["input_tokens"] and \
                    per_k[b]["input_tokens"]:
                return round((per_k[b]["input_tokens"] - per_k[a]["input_tokens"]) / (b - a), 2)
            return None

        return {
            "per_k": per_k,
            "tokens_per_option": round(float(slope), 2) if slope is not None else None,
            "tokens_at_k0": round(float(intercept), 1) if intercept is not None else None,
            "tokens_per_option_16_to_26": step(16, 26),
            "tokens_per_option_26_to_27": step(26, 27),
            "latency_ms_per_option": round(float(lat_slope), 2) if lat_slope is not None
            else None,
            "limits": limits,
        }


class QuestionProbe(Probe):
    """Change only the question (BANKING77: 2 options; ACE: its 3 options):

        match         the dataset's own question
        not_match     "Which option does NOT ..." (2 options, so exactly one is right)
        listed_first  "Ignore the state. Which option is listed first?"
        listed_last   "... listed last?"

    Strands found that a pointer head trained on one fixed question per task stops reading
    the question (answers `match` regardless); instruction-tuned letter readouts follow it."""

    name = "question"
    summary = "Does the model read the question, or only the state and options?"
    questions = {
        "not_match": "Which option does NOT describe what the customer wants?",
        "listed_first": "Ignore the message. Which option is listed first in the list of "
                        "options?",
        "listed_last": "Ignore the message. Which option is listed last in the list of "
                       "options?",
    }

    def cases(self, items):
        out = []
        for item in items.get("banking77", []):
            sub = subset(item, 2, self.seed, shuffle=False)  # [gold, distractor]
            for order_name, opts in (("gold_first", list(sub.options)),
                                     ("gold_last", list(sub.options)[::-1])):
                out.append(self.case(item.id, f"match|{order_name}", choice_request(sub, opts),
                                     gold=item.gold, question="match"))
                for qn, text in self.questions.items():
                    out.append(self.case(item.id, f"{qn}|{order_name}",
                                         choice_request(sub, opts, instructions=text),
                                         gold=item.gold, question=qn))
        for item in items.get("ace", []):
            for qn in ("listed_first", "listed_last"):
                text = self.questions[qn].replace("the message", "the clauses and scenario")
                out.append(self.case(item.id, qn, choice_request(item, instructions=text),
                                     gold=item.gold, question=qn))
        return out

    def analyze(self, records):
        out = {}
        for source, recs in by_source(records).items():
            match = {r["case_id"].replace("|match|", "|", 1): top(probs_of(r))
                     for r in recs if r["meta"]["question"] == "match" and probs_of(r)}
            acc: dict[str, list[bool]] = defaultdict(list)
            same: dict[str, list[bool]] = defaultdict(list)
            for r in recs:
                p = probs_of(r)
                if p is None:
                    continue
                qn, order, gold, choice = r["meta"]["question"], order_of(r), r["meta"]["gold"], \
                    top(p)
                expected = {"match": gold, "listed_first": order[0], "listed_last": order[-1],
                            "not_match": next((v for v in order if v != gold), None)}[qn]
                acc[qn].append(choice == expected)
                key = r["case_id"].replace(f"|{qn}|", "|", 1)
                if qn != "match" and key in match:
                    same[qn].append(choice == match[key])
            out[source] = {
                **{f"accuracy_{q}": mean(a) for q, a in sorted(acc.items())},
                **{f"same_as_match_{q}": mean(a) for q, a in sorted(same.items())},
            }
        return out


class EmptyStateProbe(Probe):
    """Send states that carry no information. The answer is the readout's prior: a
    letter readout shows its letter/position prior, a trained decision model ideally stays
    near uniform (high normalised entropy). Empty strings may also be rejected outright."""

    name = "empty_state"
    summary = "What does the model answer when the state carries no information?"
    states = {"empty": "", "na": "N/A", "no_info": "(no information provided)",
              "lorem": "Lorem ipsum dolor sit amet, consectetur adipiscing elit."}
    k_banking = 8

    def cases(self, items):
        out = []
        templates = []
        if items.get("ace"):
            item = items["ace"][0]
            base = list(item.options)
            for s in range(len(base)):
                templates.append((item, f"rot{s}", base[s:] + base[:s]))
        if items.get("banking77"):
            sub = subset(items["banking77"][0], self.k_banking, self.seed)
            templates.append((sub, "identity", list(sub.options)))
            templates.append((sub, "reversed", list(sub.options)[::-1]))
        for item, order_name, opts in templates:
            for sname, state in self.states.items():
                out.append(self.case(f"{item.source}:prior", f"{sname}|{order_name}",
                                     choice_request(item, opts, state=state), state=sname))
        return out

    def analyze(self, records):
        out = {}
        for source, recs in by_source(records).items():
            per_state = {}
            for sname in self.states:
                rs = [r for r in recs if r["meta"]["state"] == sname]
                ok = [r for r in rs if probs_of(r)]
                if not rs:
                    continue
                pos = defaultdict(list)
                vals = defaultdict(list)
                for r in ok:
                    p, order = probs_of(r), order_of(r)
                    for j, v in enumerate(order):
                        pos[j].append(p[v])
                        vals[v].append(p[v])
                k = len(pos) or 1
                per_state[sname] = {
                    "errors": sum(r.get("error") is not None for r in rs),
                    "norm_entropy": mean([norm_entropy(probs_of(r)) for r in ok]),
                    "p_first_position": mean(pos.get(0, [])),
                    "p_last_position": mean(pos.get(k - 1, [])),
                    "uniform": round(1 / k, 4),
                    "top_value": max(vals, key=lambda v: np.mean(vals[v])) if vals else None,
                    "top_value_p": mean(vals[max(vals, key=lambda v: np.mean(vals[v]))])
                    if vals else None,
                }
            out[source] = per_state
        return out


class MultiQuestionProbe(Probe):
    """Ask two questions about one state in one request, and compare with asking each alone.

    If the state is encoded once and each question branches from it (Jev: "ingests the state
    once and evaluates every question against it in parallel"), the combined request costs
    fewer input tokens than two requests (`shared_tokens` ~ the state's length) and each
    answer is identical to the single-question answer (no cross-question attention)."""

    name = "multi_question"
    summary = "Is the state encoded once and shared across questions in one request?"
    second = Question("mentions_money", "The scenario involves a payment, fee, price or other "
                      "monetary amount.", "noul")

    def cases(self, items):
        out = []
        for item in items.get("ace", []):
            q1 = choice_request(item).questions[0]
            q1b = replace(q1, name="decision_copy")
            variants = {"q1": [q1], "q2": [self.second], "q1q2": [q1, self.second],
                        "q1q1": [q1, q1b]}
            for name, qs in variants.items():
                out.append(self.case(item.id, name, DecisionRequest(item.state, list(qs)),
                                     gold=item.gold))
        return out

    def analyze(self, records):
        items: dict[str, dict[str, dict]] = defaultdict(dict)
        for r in records:
            if r.get("response"):
                items[r["item_id"]][r["variant"]] = r
        shared, marginal, frac_q1, same1, same2, dup_same, lat_ratio = [], [], [], [], [], [], []

        def tok(r):
            return r["response"].get("input_tokens")

        for vs in items.values():
            if {"q1", "q2", "q1q2"} <= set(vs) and None not in (tok(vs["q1"]), tok(vs["q2"]),
                                                               tok(vs["q1q2"])):
                s = tok(vs["q1"]) + tok(vs["q2"]) - tok(vs["q1q2"])
                shared.append(s)
                frac_q1.append(s / tok(vs["q1"]))
                lat_ratio.append(vs["q1q2"]["response"]["latency_ms"] /
                                 vs["q1"]["response"]["latency_ms"])
            if {"q1", "q1q1"} <= set(vs) and None not in (tok(vs["q1"]), tok(vs["q1q1"])):
                marginal.append(tok(vs["q1q1"]) - tok(vs["q1"]))
            if {"q1", "q1q2"} <= set(vs) and probs_of(vs["q1"]) and probs_of(vs["q1q2"]):
                same1.append(max(abs(probs_of(vs["q1"])[k] - probs_of(vs["q1q2"])[k])
                                 for k in probs_of(vs["q1"])))
            if {"q2", "q1q2"} <= set(vs) and probs_of(vs["q2"], "mentions_money") and \
                    probs_of(vs["q1q2"], "mentions_money"):
                a, b = probs_of(vs["q2"], "mentions_money"), probs_of(vs["q1q2"], "mentions_money")
                same2.append(abs(a["true"] - b["true"]))
            if "q1q1" in vs and probs_of(vs["q1q1"]) and probs_of(vs["q1q1"], "decision_copy"):
                a, b = probs_of(vs["q1q1"]), probs_of(vs["q1q1"], "decision_copy")
                dup_same.append(max(abs(a[k] - b[k]) for k in a))
        return {
            "items": len(items),
            # tokens(q1) + tokens(q2) - tokens(q1 and q2 together): billed once if shared.
            "shared_tokens_mean": mean(shared),
            "shared_fraction_of_q1_request": mean(frac_q1),
            "marginal_tokens_same_question_again": mean(marginal),
            "max_abs_dp_q1_alone_vs_combined": round(max(same1), 6) if same1 else None,
            "max_abs_dp_q2_alone_vs_combined": round(max(same2), 6) if same2 else None,
            "max_abs_dp_duplicate_question": round(max(dup_same), 6) if dup_same else None,
            "latency_ratio_combined_vs_q1": mean(lat_ratio),
        }


class DeterminismProbe(Probe):
    """Repeat identical requests. Identical probabilities on every repeat point to
    deterministic, batch-invariant serving; small differences to batched GPU numerics."""

    name = "determinism"
    summary = "Are repeated identical requests answered identically?"
    repeats = 5
    n_items = 6

    def cases(self, items):
        out = []
        for source in ("ace", "banking77"):
            for item in items.get(source, [])[: self.n_items]:
                for i in range(self.repeats):
                    out.append(self.case(item.id, f"rep{i}", choice_request(item), gold=item.gold))
        return out

    def analyze(self, records):
        out = {}
        for source, recs in by_source(records).items():
            items: dict[str, list[dict]] = defaultdict(list)
            for r in recs:
                if probs_of(r):
                    items[r["item_id"]].append(probs_of(r))
            identical, max_dp, flips = [], [], []
            for ps in items.values():
                keys = list(ps[0])
                arr = np.array([[p[k] for k in keys] for p in ps])
                spread = float((arr.max(0) - arr.min(0)).max())
                identical.append(spread == 0.0)
                max_dp.append(spread)
                flips.append(len({top(p) for p in ps}) > 1)
            out[source] = {"items": len(items), "identical_all_repeats": mean(identical),
                           "max_abs_dp": round(max(max_dp), 8) if max_dp else None,
                           "choice_flips": mean(flips)}
        return out


TOKEN_STATES = {
    "x": "x",
    "hello": "hello",
    "hello_world": "Hello, World!",
    "spaces": "a        b",
    "digits": "1234567890",
    "pi": "3.14159265358979",
    "japanese": "東京は日本の首都です。",
    "russian": "Привет, как дела?",
    "emoji": "🙂🚀🔥",
    "code": "def f(x):\n    return x * 2\n",
    "url": "https://example.com/a/b?c=d",
    "sql": "SELECT * FROM users WHERE id = 42;",
    **{f"apple_x{n:02d}": "apple " * n for n in (1, 2, 4, 8, 16, 32)},
}


class TokensProbe(Probe):
    """Token accounting. Vary only the state (strings that different tokenizers split
    differently) and regress the billed input tokens on each candidate tokenizer's count:
    an exact fit identifies the tokenizer, and the intercept is the size of the hidden
    prompt around the state (system text, question, options, formatting)."""

    name = "tokens"
    summary = "Which tokenizer is billed, and how large is the hidden prompt?"
    options = (Option("yes", "The text is in English."), Option("no", "It is not."))

    def cases(self, items):
        q = Question(QNAME, "Is the text in English?", "choice", list(self.options))
        return [self.case(f"tokens:{name}", "state", DecisionRequest(text, [q]), state=name)
                for name, text in TOKEN_STATES.items()]

    def analyze(self, records, tokenizers: dict[str, Any] | None = None):
        billed = {r["meta"]["state"]: r["response"].get("input_tokens")
                  for r in records if r.get("response")}
        billed = {k: v for k, v in billed.items() if v is not None}
        out: dict[str, Any] = {"billed_input_tokens": billed}
        if not billed:
            return out
        apple = sorted((int(k.split("x")[-1]), v) for k, v in billed.items()
                       if k.startswith("apple_x"))
        if len(apple) >= 2:
            out["tokens_per_apple"] = round(float(np.polyfit(*zip(*apple, strict=True), 1)[0]),
                                            3)
        fits = {}
        for tname, tok in (tokenizers or {}).items():
            xs = [len(tok.encode(TOKEN_STATES[k], add_special_tokens=False)) for k in billed]
            ys = list(billed.values())
            offsets = [y - x for x, y in zip(xs, ys, strict=True)]
            mode = max(set(offsets), key=offsets.count)
            fits[tname] = {"hidden_prompt_tokens": mode,
                           "exact_fit_fraction": round(offsets.count(mode) / len(offsets), 3)}
        if fits:
            out["tokenizer_fits"] = fits
            out["best_tokenizer"] = max(fits, key=lambda t: fits[t]["exact_fit_fraction"])
        return out


PROBES: dict[str, type[Probe]] = {p.name: p for p in (
    OrderProbe, LetterNamesProbe, DuplicateProbe, CardinalityProbe, QuestionProbe,
    EmptyStateProbe, MultiQuestionProbe, DeterminismProbe, TokensProbe)}
