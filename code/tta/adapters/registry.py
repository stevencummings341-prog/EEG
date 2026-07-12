"""Adapter registry — name -> factory / class."""

from __future__ import annotations

from typing import Callable, Dict, Type, Union

from code.tta.adapters.base import ModelAdapter

AdapterFactory = Union[Type[ModelAdapter], Callable[..., ModelAdapter]]

_REGISTRY: Dict[str, AdapterFactory] = {}


def register_adapter(name: str, factory: AdapterFactory) -> None:
    key = (name or "").strip().lower()
    if not key:
        raise ValueError("adapter name must be non-empty")
    _REGISTRY[key] = factory


def get_adapter_factory(name: str) -> AdapterFactory:
    key = (name or "").strip().lower()
    if key not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "(empty)"
        raise KeyError(f"unknown model_adapter '{name}'. Known: {known}")
    return _REGISTRY[key]


def build_adapter(name: str, **kwargs) -> ModelAdapter:
    factory = get_adapter_factory(name)
    return factory(**kwargs) if callable(factory) else factory(**kwargs)  # type: ignore[misc]


def list_adapters() -> list[str]:
    return sorted(_REGISTRY)


def _register_builtins() -> None:
    # Lazy imports keep method backends free of concrete model deps.
    from code.tta.adapters.embedding_only import EmbeddingOnlyAdapter
    from code.tta.adapters.baseline_torch import BaselineTorchAdapter

    register_adapter("embedding_only", EmbeddingOnlyAdapter)
    register_adapter("eegnet", lambda **kw: BaselineTorchAdapter(model_name="eegnet", **kw))
    register_adapter("deepconvnet", lambda **kw: BaselineTorchAdapter(model_name="deepconvnet", **kw))
    register_adapter("fbcnet", lambda **kw: BaselineTorchAdapter(model_name="fbcnet", **kw))
    register_adapter("baseline_torch", BaselineTorchAdapter)


_register_builtins()
