"""Method type registry (`method.type` in experiment configs)."""

from __future__ import annotations

from edb.providers.base import DecisionMethod, MethodContext


def _registry() -> dict[str, type[DecisionMethod]]:
    from edb.providers.azure_openai_ar import AzureOpenAIAutoregressive
    from edb.providers.decision1 import MicrosoftDecision1
    from edb.providers.embeddings import EmbeddingSimilarity
    from edb.providers.openai_decisions import OpenAIDecisions

    methods = [AzureOpenAIAutoregressive, MicrosoftDecision1, OpenAIDecisions, EmbeddingSimilarity]
    return {m.type: m for m in methods}


def method_types() -> list[str]:
    return sorted(_registry())


def build_method(method_type: str, ctx: MethodContext) -> DecisionMethod:
    registry = _registry()
    if method_type not in registry:
        raise KeyError(f"unknown method type {method_type!r}; known: {sorted(registry)}")
    return registry[method_type](ctx)
