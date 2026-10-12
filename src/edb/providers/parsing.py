"""Map free-form model output onto a label set."""

from __future__ import annotations

import re
from dataclasses import dataclass

_STRIP = " \t\r\n\"'`*.:;,!()[]{}<>"


@dataclass
class ParsedLabel:
    label: str | None
    status: str  # exact | normalized | search_unique | search_last | failed


def _norm(s: str) -> str:
    return re.sub(r"[\s_]+", " ", s.strip(_STRIP)).casefold()


def parse_label(text: str | None, values: list[str]) -> ParsedLabel:
    if not text or not text.strip():
        return ParsedLabel(None, "failed")
    stripped = text.strip()
    if stripped in values:
        return ParsedLabel(stripped, "exact")
    by_norm = {_norm(v): v for v in values}
    if _norm(stripped) in by_norm:
        return ParsedLabel(by_norm[_norm(stripped)], "normalized")

    # Mentions bounded by non-word/non-hyphen characters, so "Compliant" does not match
    # inside "Non-Compliant".
    hits: list[tuple[int, str]] = []
    for v in values:
        pattern = rf"(?<![\w-]){re.escape(v)}(?![\w-])"
        hits += [(m.start(), v) for m in re.finditer(pattern, stripped, flags=re.IGNORECASE)]
    distinct = {v for _, v in hits}
    if len(distinct) == 1:
        return ParsedLabel(distinct.pop(), "search_unique")
    if hits:
        return ParsedLabel(max(hits)[1], "search_last")
    return ParsedLabel(None, "failed")
