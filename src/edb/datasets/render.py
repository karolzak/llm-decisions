"""Render raw records into the textual decision `state` using the dataset config."""

from __future__ import annotations

import json
import string
from typing import Any

from edb.config import DatasetConfig, FieldFormatter


def template_fields(template: str) -> list[str]:
    return [name for _, name, _, _ in string.Formatter().parse(template) if name]


def format_field(value: Any, fmt: FieldFormatter | None) -> str:
    fmt = fmt or FieldFormatter()
    if fmt.type == "mapping":
        if not isinstance(value, dict):
            raise TypeError(f"mapping formatter expects a dict, got {type(value).__name__}")
        items = (fmt.item_template.format(key=k, value=v) for k, v in value.items())
        return fmt.joiner.join(items)
    if fmt.type == "list":
        if not isinstance(value, list):
            raise TypeError(f"list formatter expects a list, got {type(value).__name__}")
        items = (fmt.item_template.format(index=i + 1, value=v) for i, v in enumerate(value))
        return fmt.joiner.join(items)
    if fmt.type == "json":
        return json.dumps(value, ensure_ascii=False, indent=2)
    return "" if value is None else str(value)


def render_state(record: dict[str, Any], cfg: DatasetConfig) -> str:
    names = template_fields(cfg.state.template)
    missing = [n for n in names if n not in record]
    if missing:
        raise KeyError(f"record is missing template fields {missing}")
    values = {n: format_field(record[n], cfg.state.formatters.get(n)) for n in names}
    return cfg.state.template.format(**values).strip()
