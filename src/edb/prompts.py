"""Prompt construction shared by all autoregressive methods (API and local)."""

from __future__ import annotations

from edb.schemas import LabelCatalog, TaskSpec

DEFAULT_SYSTEM_PROMPT = """You are a careful analyst making a bounded decision.

Task: {instructions}

Options:
{options}

{answer_instruction}"""

ANSWER_LABEL = (
    "Answer with exactly one option label, copied verbatim from the list above, "
    "and nothing else."
)
ANSWER_JSON_CONFIDENCE = (
    "Respond with a JSON object with `label` (exactly one option label, copied verbatim) "
    "and `confidence` (your probability, from 0 to 1, that the label is correct)."
)


def render_options(catalog: LabelCatalog) -> str:
    return "\n".join(
        f"- {o.value}: {o.description}" if o.has_description else f"- {o.value}"
        for o in catalog.labels
    )


def system_prompt(
    task: TaskSpec, catalog: LabelCatalog, verbalized_confidence: bool = False
) -> str:
    template = task.system_prompt or DEFAULT_SYSTEM_PROMPT
    return template.format(
        instructions=task.instructions,
        options=render_options(catalog),
        answer_instruction=ANSWER_JSON_CONFIDENCE if verbalized_confidence else ANSWER_LABEL,
    )


def confidence_schema(values: list[str]) -> dict:
    return {
        "type": "json_schema",
        "name": "decision",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "label": {"type": "string", "enum": values},
                "confidence": {"type": "number"},
            },
            "required": ["label", "confidence"],
            "additionalProperties": False,
        },
    }
