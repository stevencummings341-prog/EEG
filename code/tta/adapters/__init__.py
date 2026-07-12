"""TTA model adapters."""

from code.tta.adapters.base import ModelAdapter, PretrainedModelAdapterTemplate
from code.tta.adapters.embedding_only import EmbeddingOnlyAdapter
from code.tta.adapters.registry import build_adapter, list_adapters, register_adapter

__all__ = [
    "ModelAdapter",
    "PretrainedModelAdapterTemplate",
    "EmbeddingOnlyAdapter",
    "build_adapter",
    "list_adapters",
    "register_adapter",
]
