"""Shared exceptions for the TTA backend."""

from __future__ import annotations


class UnsupportedAdapterFeature(Exception):
    """Raised when a ModelAdapter does not implement an optional capability.

    Callers must treat this as a soft capability miss (choose another code path
    or fail with a clear message), never as a silent fallback to EEGNet.

    Prefer raising / catching :class:`UnsupportedCapabilityError` in new code —
    it is a subclass of this exception, so existing ``except
    UnsupportedAdapterFeature`` call sites keep working unchanged.
    """


class UnsupportedCapabilityError(UnsupportedAdapterFeature):
    """Raised when a required, explicitly-queried adapter capability is missing.

    This is the preferred exception for new capability-aware code (see
    ``code.tta.adapters.base.AdapterCapabilities`` / ``require_capability``).
    It subclasses ``UnsupportedAdapterFeature`` for backward compatibility with
    existing ``except UnsupportedAdapterFeature`` handlers.
    """


class CheckpointLoadError(Exception):
    """Raised when a model checkpoint cannot be located, read, or applied.

    Covers missing files, corrupt/incompatible state dicts, and shape
    mismatches between a checkpoint and the constructed model.
    """


class InputValidationError(Exception):
    """Raised when input arrays/tensors fail shape, dtype, or rank checks.

    Used by adapters (``validate_input_shape``) and feature sources
    (``ModelInferenceSource``) to fail fast on malformed batches instead of
    letting a confusing error surface deep inside a forward pass.
    """


class LabelLeakageError(Exception):
    """Raised when a label-free TTA method attempts to use target true labels."""


class FeatureSourceError(Exception):
    """Raised when embeddings / checkpoints cannot be resolved or loaded."""


class TTAConfigError(Exception):
    """Raised for invalid Round-1 / smoke configuration."""
