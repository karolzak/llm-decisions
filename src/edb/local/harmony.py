"""Deterministic harmony-format prompts for gpt-oss.

The stock chat template embeds today's date, which would make prompts (and results) depend on
the run date. We build the same structure with a fixed date instead.
"""

from __future__ import annotations

import re

SYSTEM_TEMPLATE = (
    "You are ChatGPT, a large language model trained by OpenAI.\n"
    "Knowledge cutoff: 2024-06\n"
    "Current date: {date}\n\n"
    "Reasoning: {reasoning}\n\n"
    "# Valid channels: analysis, commentary, final. Channel must be included for every message."
)
FIXED_DATE = "2026-10-01"
ASSISTANT_START = "<|start|>assistant"
FINAL_PREFIX = "<|channel|>final<|message|>"
RETURN = "<|return|>"


def render(
    instructions: str, user: str, reasoning: str = "low", date: str = FIXED_DATE
) -> str:
    """Conversation up to (and including) `<|start|>assistant`."""
    return (
        f"<|start|>system<|message|>{SYSTEM_TEMPLATE.format(date=date, reasoning=reasoning)}"
        "<|end|>"
        f"<|start|>developer<|message|># Instructions\n\n{instructions}\n\n<|end|>"
        f"<|start|>user<|message|>{user}<|end|>"
        f"{ASSISTANT_START}"
    )


_FINAL_RE = re.compile(
    r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|<\|call\|>|$)", re.S
)


def final_message(generated: str) -> str | None:
    """Text of the last final-channel message in generated output (None if absent)."""
    matches = _FINAL_RE.findall(generated)
    return matches[-1].strip() if matches else None


def analysis_text(generated: str) -> str:
    m = re.search(r"<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>", generated, re.S)
    return m.group(1).strip() if m else ""
