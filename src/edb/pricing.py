from __future__ import annotations

from edb.config import Pricing
from edb.schemas import Usage


def api_cost(usage: Usage | None, pricing: Pricing | None) -> float | None:
    """USD cost of one request. Input tokens include cached / cache-write tokens."""
    if usage is None or pricing is None:
        return None
    cached = usage.cached_tokens if pricing.cached_input is not None else 0
    written = usage.cache_write_tokens if pricing.cache_write is not None else 0
    fresh = max(0, usage.input_tokens - cached - written)
    total = (
        fresh * pricing.input
        + cached * (pricing.cached_input or 0.0)
        + written * (pricing.cache_write or 0.0)
        + usage.output_tokens * pricing.output
    )
    return total / pricing.per_tokens


def gpu_cost(gpu_seconds: float, hourly_price: float | None) -> float | None:
    return None if hourly_price is None else gpu_seconds / 3600.0 * hourly_price
