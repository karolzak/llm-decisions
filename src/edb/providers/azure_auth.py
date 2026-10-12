"""Entra ID authenticated clients for Azure OpenAI and Foundry endpoints.

Local (API key) auth is disabled on the Foundry resource, so every call uses a bearer
token from `DefaultAzureCredential` (`az login` locally, managed identity on Azure).
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from functools import lru_cache

import httpx
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AsyncOpenAI, OpenAI

from edb.settings import Settings, get_settings


@lru_cache
def get_credential() -> DefaultAzureCredential:
    return DefaultAzureCredential(exclude_interactive_browser_credential=True)


@lru_cache
def _provider(scope: str) -> Callable[[], str]:
    return get_bearer_token_provider(get_credential(), scope)


def token_provider(settings: Settings | None = None) -> Callable[[], str]:
    settings = settings or get_settings()
    return _provider(settings.azure_token_scope)


class BearerAuth(httpx.Auth):
    """httpx auth (sync + async) that injects a fresh bearer token on every request."""

    def __init__(self, get_token: Callable[[], str]):
        self._get_token = get_token

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        request.headers["Authorization"] = f"Bearer {self._get_token()}"
        yield request


def openai_client(settings: Settings | None = None, **kwargs) -> OpenAI:
    settings = settings or get_settings()
    return OpenAI(base_url=settings.openai_base_url, api_key=token_provider(settings), **kwargs)


def async_openai_client(settings: Settings | None = None, **kwargs) -> AsyncOpenAI:
    settings = settings or get_settings()
    provider = token_provider(settings)

    async def _token() -> str:
        return provider()

    return AsyncOpenAI(base_url=settings.openai_base_url, api_key=_token, **kwargs)


_JSON_HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}


def foundry_http_client(settings: Settings | None = None, timeout: float = 60.0) -> httpx.Client:
    """Sync HTTP client for non-OpenAI Foundry routes (e.g. Microsoft-Decision-1)."""
    settings = settings or get_settings()
    return httpx.Client(auth=BearerAuth(token_provider(settings)), timeout=timeout,
                        headers=_JSON_HEADERS)


def foundry_async_http_client(
    settings: Settings | None = None, timeout: float = 60.0, max_connections: int = 64
) -> httpx.AsyncClient:
    settings = settings or get_settings()
    return httpx.AsyncClient(
        auth=BearerAuth(token_provider(settings)),
        timeout=timeout,
        headers=_JSON_HEADERS,
        limits=httpx.Limits(max_connections=max_connections),
    )
