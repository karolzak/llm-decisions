"""Entra ID authenticated clients for Azure OpenAI and Foundry endpoints.

The Foundry resource has local (API key) auth disabled, so every call uses a
bearer token from `DefaultAzureCredential` (e.g. `az login` locally, managed
identity on Azure ML compute).
"""

from functools import lru_cache

import httpx
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AsyncOpenAI, OpenAI

from edb.settings import Settings, get_settings


@lru_cache
def get_credential() -> DefaultAzureCredential:
    return DefaultAzureCredential(exclude_interactive_browser_credential=True)


def token_provider(settings: Settings | None = None):
    settings = settings or get_settings()
    return get_bearer_token_provider(get_credential(), settings.azure_token_scope)


def openai_client(settings: Settings | None = None, **kwargs) -> OpenAI:
    settings = settings or get_settings()
    return OpenAI(base_url=settings.openai_base_url, api_key=token_provider(settings), **kwargs)


def async_openai_client(settings: Settings | None = None, **kwargs) -> AsyncOpenAI:
    settings = settings or get_settings()
    sync_provider = token_provider(settings)

    async def _provider() -> str:
        return sync_provider()

    return AsyncOpenAI(base_url=settings.openai_base_url, api_key=_provider, **kwargs)


def foundry_http_client(settings: Settings | None = None, timeout: float = 60.0) -> httpx.Client:
    """HTTP client for non-OpenAI Foundry routes (e.g. Microsoft-Decision-1 `systemone`)."""
    settings = settings or get_settings()
    provider = token_provider(settings)

    def _auth(request: httpx.Request) -> httpx.Request:
        request.headers["Authorization"] = f"Bearer {provider()}"
        return request

    return httpx.Client(
        auth=_auth,
        timeout=timeout,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
