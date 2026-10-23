"""Retry classification shared by all remote methods."""

from __future__ import annotations

import asyncio
import random

import httpx
import openai
from azure.core.exceptions import ClientAuthenticationError

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class HTTPError(RuntimeError):
    def __init__(self, status: int, body: str, headers: httpx.Headers):
        super().__init__(f"HTTP {status}: {body[:500]}")
        self.status = status
        self.headers = headers


def raise_for_status(response: httpx.Response) -> None:
    if response.status_code >= 400:
        raise HTTPError(response.status_code, response.text, response.headers)


def _retry_after(headers: httpx.Headers | None) -> float | None:
    if not headers:
        return None
    for key, scale in (("retry-after-ms", 1000.0), ("x-ms-retry-after-ms", 1000.0),
                       ("retry-after", 1.0)):
        value = headers.get(key)
        if value:
            try:
                return float(value) / scale
            except ValueError:
                continue
    return None


def classify(exc: BaseException) -> tuple[bool, float | None]:
    """Return (retryable, server-suggested delay seconds)."""
    if isinstance(exc, HTTPError):
        return exc.status in RETRYABLE_STATUS, _retry_after(exc.headers)
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code in RETRYABLE_STATUS, _retry_after(exc.response.headers)
    if isinstance(exc, openai.APIConnectionError | openai.APITimeoutError):
        return True, None
    if isinstance(exc, httpx.TransportError | asyncio.TimeoutError | TimeoutError):
        return True, None
    # Entra token fetches (e.g. an `az` CLI subprocess timing out) fail transiently.
    if isinstance(exc, ClientAuthenticationError):
        return True, None
    return False, None


def backoff_delay(attempt: int, suggested: float | None, cap: float = 60.0) -> float:
    if suggested is not None:
        return min(cap, suggested) + random.uniform(0, 0.5)
    return min(cap, 2.0**attempt) * random.uniform(0.5, 1.0)
