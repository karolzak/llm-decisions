"""Method type registry (`method.type` in experiment configs)."""

from __future__ import annotations

from edb.providers.base import DecisionMethod, MethodContext

LOCAL_TYPES = ("local_autoregressive", "local_sequence_scoring", "local_cross_encoder",
               "local_listwise_head", "local_letter_readout", "local_early_exit_head",
               "local_reason_then_read")


def _registry(include_local: bool = False) -> dict[str, type[DecisionMethod]]:
    from edb.providers.azure_openai_ar import AzureOpenAIAutoregressive
    from edb.providers.decision1 import MicrosoftDecision1
    from edb.providers.embeddings import EmbeddingMethod
    from edb.providers.openai_decisions import OpenAIDecisions

    methods = [AzureOpenAIAutoregressive, MicrosoftDecision1, OpenAIDecisions, EmbeddingMethod]
    registry = {m.type: m for m in methods}
    if include_local:  # imports torch/transformers: only when a local method is requested
        from edb.local.methods import local_method_types

        registry.update(local_method_types())
    return registry


def method_types() -> list[str]:
    return sorted([*_registry(), *LOCAL_TYPES])


def build_method(method_type: str, ctx: MethodContext) -> DecisionMethod:
    registry = _registry(include_local=method_type in LOCAL_TYPES)
    if method_type not in registry:
        raise KeyError(f"unknown method type {method_type!r}; known: {sorted(registry)}")
    return registry[method_type](ctx)
