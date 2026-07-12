"""ModelAdapter protocol — all capabilities are optional.

Future pretrained models (feature-only, backbone+head, logits-only, embedding
npz, foundation models, senior custom checkpoints, …) should implement a thin
adapter and register it. Do **not** hard-code a specific future architecture here.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np

from code.tta.exceptions import UnsupportedAdapterFeature, UnsupportedCapabilityError


@dataclass(frozen=True)
class AdapterCapabilities:
    """Explicit, queryable capability flags for a :class:`ModelAdapter`.

    Feature sources and methods must check these flags (or catch
    ``UnsupportedCapabilityError``) instead of relying on ``soft_call``
    silently substituting a default in production code paths — silent
    fallbacks hide missing capabilities and make failures hard to diagnose.

    Flags:
        features: ``forward_features`` returns real embeddings.
        logits: ``forward_logits`` returns real class logits.
        probabilities: ``predict_proba`` returns real class probabilities.
        classifier_weights: ``get_classifier_weights`` returns a real weight matrix.
        source_prototypes: ``get_source_prototypes`` returns precomputed prototypes.
        metadata: ``get_model_metadata`` returns meaningful metadata (default True;
            every adapter has at least ``{"adapter_name": ...}``).
        checkpoint_loading: ``load_checkpoint`` can load real weights from disk.
        input_validation: ``validate_input_shape`` performs a real shape check.
    """

    features: bool = False
    logits: bool = False
    probabilities: bool = False
    classifier_weights: bool = False
    source_prototypes: bool = False
    metadata: bool = True
    checkpoint_loading: bool = False
    input_validation: bool = False

    def as_dict(self) -> Dict[str, bool]:
        return asdict(self)


class ModelAdapter(ABC):
    """Unified optional interface consumed by the TTA backend.

    Method backends must only call these hooks (or FeatureBundle fields).
    They must never import concrete model classes (EEGNet, …).
    """

    name: str = "base"

    # ---- capability reporting ---------------------------------------------------
    def capabilities(self) -> AdapterCapabilities:
        """Return explicit capability flags for this adapter instance.

        Base default declares no live-forward / checkpoint capabilities —
        only ``metadata`` (every adapter implements ``get_model_metadata``).
        Concrete adapters must override this to reflect what they actually
        support so callers can fail fast with a typed error instead of
        discovering a missing capability deep inside a forward pass.
        """
        return AdapterCapabilities()

    # ---- optional forward paths -------------------------------------------------
    def forward_features(self, x: Any) -> np.ndarray:
        raise UnsupportedCapabilityError(
            f"{type(self).__name__} does not support forward_features()"
        )

    def forward_logits(self, x: Any) -> np.ndarray:
        raise UnsupportedCapabilityError(
            f"{type(self).__name__} does not support forward_logits()"
        )

    def predict_proba(self, x: Any) -> np.ndarray:
        raise UnsupportedCapabilityError(
            f"{type(self).__name__} does not support predict_proba()"
        )

    # ---- optional weight / prototype access ------------------------------------
    def get_classifier_weights(self) -> np.ndarray:
        """Return classifier weight matrix [n_classes, feature_dim] if available."""
        raise UnsupportedCapabilityError(
            f"{type(self).__name__} does not support get_classifier_weights()"
        )

    def get_source_prototypes(self) -> Dict[int, np.ndarray]:
        """Return class -> prototype vector if the adapter already has them."""
        raise UnsupportedCapabilityError(
            f"{type(self).__name__} does not support get_source_prototypes()"
        )

    def get_feature_dim(self) -> int:
        raise UnsupportedCapabilityError(
            f"{type(self).__name__} does not support get_feature_dim()"
        )

    def get_model_metadata(self) -> Dict[str, Any]:
        """Channel count, sfreq, input length, preprocessing tags, etc."""
        return {"adapter_name": self.name}

    # ---- optional lifecycle ----------------------------------------------------
    def load_checkpoint(self, path: str) -> None:
        raise UnsupportedCapabilityError(
            f"{type(self).__name__} does not support load_checkpoint()"
        )

    def to(self, device: Any) -> "ModelAdapter":
        return self

    def eval(self) -> "ModelAdapter":
        return self

    def validate_input_shape(self, x: Any) -> None:
        """Optional shape check; default is a no-op."""
        return None


# --------------------------------------------------------------------------- #
# Template for a future pretrained-model adapter (NOT implemented).
# --------------------------------------------------------------------------- #
class PretrainedModelAdapterTemplate(ModelAdapter):
    """DOCSTRING-ONLY TEMPLATE — do not instantiate as a real adapter.

    When a senior / foundation / custom pretrained model arrives:

    1. Create ``code/tta/adapters/<your_name>.py`` implementing ``ModelAdapter``.
    2. Implement only the capabilities the checkpoint actually provides
       (features-only, logits, classifier weights, …). Leave the rest raising
       ``UnsupportedCapabilityError`` and override ``capabilities()`` to report
       ``False`` for them.
    3. Register via ``register_adapter("<name>", YourAdapter)``.
    4. Point config ``model_adapter: <name>`` (+ checkpoint / preprocess fields).
    5. Do **not** rewrite TTA methods / evaluators.

    Do not assume fixed layer names, fixed feature dims, fixed channel counts,
    or a particular checkpoint format in the TTA backend itself.
    """

    name = "pretrained_template"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "PretrainedModelAdapterTemplate is documentation only. "
            "Add a real adapter module when the pretrained model is available."
        )


def require_capability(
    adapter: ModelAdapter, capability: str, *, context: str = ""
) -> AdapterCapabilities:
    """Fail fast with ``UnsupportedCapabilityError`` unless ``adapter`` declares
    support for ``capability`` via :meth:`ModelAdapter.capabilities`.

    This is the production-safe alternative to ``soft_call``: it never
    substitutes a silent default, so missing capabilities surface immediately
    with a clear, typed error instead of ``None``/zeros propagating downstream.

    Returns the adapter's ``AdapterCapabilities`` (so callers can reuse it
    without a second ``capabilities()`` call).
    """
    caps = adapter.capabilities()
    if not getattr(caps, capability, False):
        suffix = f" ({context})" if context else ""
        raise UnsupportedCapabilityError(
            f"{type(adapter).__name__} does not support required capability "
            f"'{capability}'{suffix}. capabilities={caps.as_dict()}"
        )
    return caps


def soft_call(adapter: ModelAdapter, method: str, *args: Any, default: Any = None) -> Any:
    """**Test-only convenience helper.** Call an optional adapter method and
    return ``default`` on ``UnsupportedAdapterFeature`` (which includes the
    newer ``UnsupportedCapabilityError``).

    Do NOT use this in production feature-extraction paths such as
    ``ModelInferenceSource``: silently substituting a default hides a missing
    capability instead of failing fast. Production code must check
    ``adapter.capabilities()`` — see :func:`require_capability` — and raise a
    typed error when a required capability is absent.
    """
    fn = getattr(adapter, method, None)
    if fn is None:
        return default
    try:
        return fn(*args)
    except UnsupportedAdapterFeature:
        return default
