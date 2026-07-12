"""Mock ``ModelAdapter`` wrapping ``MockEEGModel`` — **TEST FIXTURE ONLY**.

Provides three capability profiles ("A"/"B"/"C") so tests can exercise
``ModelInferenceSource`` capability fail-fast behavior without any production
adapter code. This adapter must NEVER be registered in
``code.tta.adapters.registry`` as a production adapter.

  * Profile A: full        — features, logits, probabilities, classifier weights, metadata.
  * Profile B: partial     — features + logits (+ probabilities), NO classifier weights.
  * Profile C: logits-only — logits (+ probabilities) only, NO features
                              (no fake embeddings are ever fabricated).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from code.tta.adapters.base import AdapterCapabilities, ModelAdapter
from code.tta.exceptions import (
    CheckpointLoadError,
    InputValidationError,
    UnsupportedCapabilityError,
)
from tests.tta.support.mock_model import MockModelConfig, build_mock_model, save_mock_checkpoint

PathLike = Union[str, Path]
PROFILES = ("A", "B", "C")


class MockModelAdapter(ModelAdapter):
    """TEST FIXTURE ONLY — see module docstring for capability profiles."""

    name = "mock_test_adapter"

    def __init__(
        self,
        *,
        n_channels: int = 6,
        n_times: int = 32,
        n_classes: int = 2,
        feature_dim: int = 10,
        profile: str = "A",
        device: str = "cpu",
        seed: int = 0,
        model: Optional[Any] = None,
    ) -> None:
        if profile not in PROFILES:
            raise ValueError(f"unknown mock profile '{profile}', expected one of {PROFILES}")
        self.profile = profile
        self.n_channels = int(n_channels)
        self.n_times = int(n_times)
        self.n_classes = int(n_classes)
        self.cfg = MockModelConfig(
            n_channels=self.n_channels,
            n_times=self.n_times,
            n_classes=self.n_classes,
            feature_dim=int(feature_dim),
        )
        self._model = model if model is not None else build_mock_model(self.cfg, seed=seed)
        self._device = torch.device(device)
        self._model.to(self._device)
        self._model.eval()

    def capabilities(self) -> AdapterCapabilities:
        if self.profile == "A":
            return AdapterCapabilities(
                features=True,
                logits=True,
                probabilities=True,
                classifier_weights=True,
                source_prototypes=False,
                metadata=True,
                checkpoint_loading=True,
                input_validation=True,
            )
        if self.profile == "B":
            return AdapterCapabilities(
                features=True,
                logits=True,
                probabilities=True,
                classifier_weights=False,
                source_prototypes=False,
                metadata=True,
                checkpoint_loading=True,
                input_validation=True,
            )
        # Profile C: logits-only.
        return AdapterCapabilities(
            features=False,
            logits=True,
            probabilities=True,
            classifier_weights=False,
            source_prototypes=False,
            metadata=True,
            checkpoint_loading=True,
            input_validation=True,
        )

    def _as_tensor(self, x: Any) -> torch.Tensor:
        arr = np.asarray(x, dtype=np.float32)
        t = torch.from_numpy(arr)
        if t.dim() == 2:
            t = t.unsqueeze(0)
        return t.to(self._device)

    def forward_features(self, x: Any) -> np.ndarray:
        if self.profile == "C":
            raise UnsupportedCapabilityError(
                "MockModelAdapter(profile='C') does not support forward_features() "
                "(logits-only profile; this fixture never fabricates fake embeddings)."
            )
        with torch.inference_mode():
            out = self._model(self._as_tensor(x))
            return out["features"].detach().cpu().numpy()

    def forward_logits(self, x: Any) -> np.ndarray:
        with torch.inference_mode():
            out = self._model(self._as_tensor(x))
            return out["logits"].detach().cpu().numpy()

    def predict_proba(self, x: Any) -> np.ndarray:
        logits = self.forward_logits(x)
        m = logits.max(axis=-1, keepdims=True)
        ex = np.exp(logits - m)
        return (ex / ex.sum(axis=-1, keepdims=True)).astype(np.float32)

    def get_classifier_weights(self) -> np.ndarray:
        if self.profile != "A":
            raise UnsupportedCapabilityError(
                f"MockModelAdapter(profile='{self.profile}') does not expose "
                "classifier weights."
            )
        return self._model.classifier.weight.detach().cpu().numpy().astype(np.float32)

    def get_feature_dim(self) -> int:
        if self.profile == "C":
            raise UnsupportedCapabilityError(
                "MockModelAdapter(profile='C') has no feature_dim (logits-only)."
            )
        return int(self.cfg.feature_dim)

    def get_model_metadata(self) -> Dict[str, Any]:
        return {
            "adapter_name": self.name,
            "profile": self.profile,
            "n_channels": self.n_channels,
            "n_times": self.n_times,
            "n_classes": self.n_classes,
        }

    def load_checkpoint(self, path: str) -> None:
        p = Path(path)
        if not p.is_file():
            raise CheckpointLoadError(f"checkpoint not found: {p}")
        try:
            state = torch.load(p, map_location=self._device)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            self._model.load_state_dict(state)
        except CheckpointLoadError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize to typed error
            raise CheckpointLoadError(f"failed to load checkpoint {p}: {exc}") from exc

    def to(self, device: Any) -> "MockModelAdapter":
        self._device = torch.device(device)
        self._model.to(self._device)
        return self

    def eval(self) -> "MockModelAdapter":
        self._model.eval()
        return self

    def validate_input_shape(self, x: Any) -> None:
        arr = np.asarray(x)
        if arr.ndim == 2:
            c, t = arr.shape
        elif arr.ndim == 3:
            _, c, t = arr.shape
        else:
            raise InputValidationError(f"expected [C,T] or [B,C,T], got shape {arr.shape}")
        if c != self.n_channels or t != self.n_times:
            raise InputValidationError(
                f"input shape channels/times ({c},{t}) != "
                f"adapter ({self.n_channels},{self.n_times})"
            )


def save_checkpoint_for_adapter(adapter: MockModelAdapter, path: PathLike) -> None:
    """Save ``adapter``'s underlying model weights to ``path``."""
    save_mock_checkpoint(adapter._model, path)


def state_dicts_equal(a: MockModelAdapter, b: MockModelAdapter) -> bool:
    """Return True iff every parameter tensor in ``a`` and ``b`` is identical."""
    sa, sb = a._model.state_dict(), b._model.state_dict()
    if sa.keys() != sb.keys():
        return False
    return all(torch.equal(sa[k], sb[k]) for k in sa)
