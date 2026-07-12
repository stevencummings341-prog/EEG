"""Example adapter wrapping existing project baseline torch models.

This is an **example / smoke tool only**. The TTA method backend must not import
EEGNet / DeepConvNet / FBCNet directly — only this adapter may.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from code.tta.adapters.base import AdapterCapabilities, ModelAdapter
from code.tta.exceptions import CheckpointLoadError, InputValidationError, UnsupportedCapabilityError


class BaselineTorchAdapter(ModelAdapter):
    """Optional live wrapper around ``code.models.registry.build_model``.

    Round-1 smoke primarily uses embedding replay, so this adapter is mainly for
    interface tests and future live-inference scaffolding.
    """

    def __init__(
        self,
        *,
        model_name: str = "eegnet",
        n_channels: int = 58,
        n_times: int = 1000,
        n_classes: int = 2,
        sfreq: int = 250,
        params: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
        model: Any = None,
    ) -> None:
        self.model_name = (model_name or "eegnet").lower().strip()
        self.name = self.model_name
        self.n_channels = int(n_channels)
        self.n_times = int(n_times)
        self.n_classes = int(n_classes)
        self.sfreq = int(sfreq)
        self.params = dict(params or {})
        self._device_str = device
        self._model = model
        if self._model is None:
            from code.models.registry import build_model

            self._model = build_model(
                self.model_name,
                n_channels=self.n_channels,
                n_times=self.n_times,
                n_classes=self.n_classes,
                sfreq=self.sfreq,
                params=self.params,
            )
        self.to(device)
        self.eval()
        clf = getattr(self._model, "classifier", None)
        self._has_classifier_weights = clf is not None and hasattr(clf, "weight")

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            features=True,
            logits=True,
            probabilities=True,
            classifier_weights=self._has_classifier_weights,
            source_prototypes=False,
            metadata=True,
            checkpoint_loading=True,
            input_validation=True,
        )

    def _as_tensor(self, x: Any):
        import torch

        if isinstance(x, np.ndarray):
            t = torch.from_numpy(x.astype(np.float32))
        else:
            t = x
        if t.dim() == 2:
            t = t.unsqueeze(0)
        return t.to(self._device)

    def forward_features(self, x: Any) -> np.ndarray:
        import torch

        with torch.no_grad():
            out = self._model(self._as_tensor(x))
            feats = out["features"]
            return feats.detach().cpu().numpy()

    def forward_logits(self, x: Any) -> np.ndarray:
        import torch

        with torch.no_grad():
            out = self._model(self._as_tensor(x))
            return out["logits"].detach().cpu().numpy()

    def predict_proba(self, x: Any) -> np.ndarray:
        logits = self.forward_logits(x)
        m = logits.max(axis=-1, keepdims=True)
        ex = np.exp(logits - m)
        return (ex / ex.sum(axis=-1, keepdims=True)).astype(np.float32)

    def get_classifier_weights(self) -> np.ndarray:
        clf = getattr(self._model, "classifier", None)
        if clf is None or not hasattr(clf, "weight"):
            raise UnsupportedCapabilityError(
                f"{self.model_name} has no .classifier.weight"
            )
        return clf.weight.detach().cpu().numpy().astype(np.float32)

    def get_feature_dim(self) -> int:
        dim = getattr(self._model, "feature_dim", None)
        if dim is None:
            raise UnsupportedCapabilityError(
                f"{self.model_name} does not expose feature_dim"
            )
        return int(dim)

    def get_model_metadata(self) -> Dict[str, Any]:
        meta = {
            "adapter_name": self.name,
            "model_name": self.model_name,
            "n_channels": self.n_channels,
            "n_times": self.n_times,
            "n_classes": self.n_classes,
            "sfreq": self.sfreq,
            "device": self._device_str,
        }
        try:
            meta["feature_dim"] = self.get_feature_dim()
        except UnsupportedCapabilityError:
            pass
        return meta

    def load_checkpoint(self, path: str) -> None:
        import torch

        p = Path(path)
        if not p.is_file():
            raise CheckpointLoadError(f"checkpoint not found: {p}")
        try:
            state = torch.load(p, map_location=self._device)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            elif isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self._model.load_state_dict(state)
        except CheckpointLoadError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize to typed error
            raise CheckpointLoadError(
                f"failed to load checkpoint {p}: {exc}"
            ) from exc

    def to(self, device: Any) -> "BaselineTorchAdapter":
        import torch

        self._device = torch.device(device if device is not None else "cpu")
        self._device_str = str(self._device)
        self._model.to(self._device)
        return self

    def eval(self) -> "BaselineTorchAdapter":
        self._model.eval()
        return self

    def validate_input_shape(self, x: Any) -> None:
        arr = np.asarray(x)
        if arr.ndim == 2:
            c, t = arr.shape
        elif arr.ndim == 3:
            _, c, t = arr.shape
        else:
            raise InputValidationError(
                f"expected [C,T] or [B,C,T], got shape {arr.shape}"
            )
        if c != self.n_channels or t != self.n_times:
            raise InputValidationError(
                f"input shape channels/times ({c},{t}) != "
                f"adapter ({self.n_channels},{self.n_times})"
            )
