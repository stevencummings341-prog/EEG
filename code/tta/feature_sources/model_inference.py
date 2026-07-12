"""Live model-inference feature source.

Runs a frozen ``ModelAdapter`` over target (and optionally source) data to
produce a :class:`FeatureBundle` — checkpoint → adapter load → ``eval()`` +
``torch.inference_mode()`` → batched forward → concatenate → validate →
FeatureBundle.

This layer is intentionally dumb: it never trains, backprops, or adapts
anything. All TTA logic (prototype updates, filtering, …) lives in
``code.tta.methods.*``. If a required capability (features / logits /
probabilities / classifier weights / checkpoint loading) is missing on the
adapter, this source fails fast with a typed error from ``code.tta.exceptions``
instead of silently returning ``None`` / fabricated data — a T3A-style method
that needs embeddings must never receive fake ones.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, Optional

import numpy as np

from code.tta.adapters.base import ModelAdapter, require_capability
from code.tta.exceptions import CheckpointLoadError, InputValidationError, UnsupportedCapabilityError
from code.tta.feature_sources.base import FeatureBundle, FeatureSource, make_cell_id


def _to_numpy(batch: Any) -> np.ndarray:
    if isinstance(batch, np.ndarray):
        return batch
    detach = getattr(batch, "detach", None)
    if callable(detach):
        return detach().cpu().numpy()
    return np.asarray(batch)


def _iter_raw_batches(x: Any, batch_size: int) -> Iterator[np.ndarray]:
    """Yield numpy batches from a numpy/tensor array or a pre-batched iterable.

    Accepts:
      * numpy array (or tensor with ``.shape``/``.detach``) shaped ``[N, C, T]``
        (or ``[C, T]`` for a single sample) — sliced into chunks of
        ``batch_size``.
      * any other iterable yielding batches already (e.g. a torch
        ``DataLoader`` or a plain list of numpy/torch batches) — consumed
        as-is, each element converted to numpy.
    """
    if isinstance(x, np.ndarray) or hasattr(x, "shape"):
        arr = _to_numpy(x)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        n = arr.shape[0]
        for start in range(0, n, batch_size):
            yield arr[start : start + batch_size]
        return

    for batch in x:
        yield _to_numpy(batch)


class ModelInferenceSource(FeatureSource):
    """Real live-inference :class:`FeatureSource` backed by a ``ModelAdapter``.

    Parameters
    ----------
    adapter:
        A constructed :class:`ModelAdapter` (already built; checkpoint may be
        loaded lazily via ``load_cell(checkpoint_path=...)``).
    device:
        Passed to ``adapter.to(device)`` before inference.
    batch_size:
        Batch size used when the raw input is a numpy array (ignored for
        inputs that are already batch iterables, e.g. a DataLoader).
    require_features / require_logits / require_probabilities /
    require_classifier_weights:
        Fail fast with ``UnsupportedCapabilityError`` at ``load_cell`` time if
        the adapter does not declare the corresponding capability.
        ``require_features`` defaults to ``True`` because most downstream TTA
        methods (e.g. T3A) need real embeddings; callers that only need
        logits/predictions (e.g. no_tta on a logits-only adapter) must pass
        ``require_features=False`` explicitly — this source will never
        fabricate embeddings to paper over a missing capability.
    dtype:
        Numpy dtype used for feature/logit/prob arrays in the returned bundle.
    """

    name = "model_inference"

    def __init__(
        self,
        adapter: ModelAdapter,
        *,
        device: str = "cpu",
        batch_size: int = 32,
        require_features: bool = True,
        require_logits: bool = False,
        require_probabilities: bool = False,
        require_classifier_weights: bool = False,
        dtype: Any = np.float32,
    ) -> None:
        self.adapter = adapter
        self.device = device
        self.batch_size = int(batch_size)
        self.require_features = bool(require_features)
        self.require_logits = bool(require_logits)
        self.require_probabilities = bool(require_probabilities)
        self.require_classifier_weights = bool(require_classifier_weights)
        self.dtype = dtype

    # ------------------------------------------------------------------ #
    def _load_checkpoint_if_needed(self, checkpoint_path: Optional[str]) -> None:
        if checkpoint_path is None:
            return
        require_capability(
            self.adapter,
            "checkpoint_loading",
            context=f"checkpoint_path={checkpoint_path}",
        )
        try:
            self.adapter.load_checkpoint(str(checkpoint_path))
        except CheckpointLoadError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize to typed error
            raise CheckpointLoadError(
                f"failed to load checkpoint {checkpoint_path}: {exc}"
            ) from exc

    def _check_required_capabilities(self) -> "Any":
        caps = self.adapter.capabilities()
        if self.require_features and not caps.features:
            raise UnsupportedCapabilityError(
                f"{type(self.adapter).__name__} lacks 'features' capability required "
                f"by ModelInferenceSource(require_features=True). "
                f"Pass require_features=False if this cell truly does not need "
                f"embeddings (e.g. no_tta on a logits-only model). "
                f"capabilities={caps.as_dict()}"
            )
        if self.require_logits and not caps.logits:
            raise UnsupportedCapabilityError(
                f"{type(self.adapter).__name__} lacks 'logits' capability required "
                f"by ModelInferenceSource(require_logits=True). capabilities={caps.as_dict()}"
            )
        if self.require_probabilities and not caps.probabilities:
            raise UnsupportedCapabilityError(
                f"{type(self.adapter).__name__} lacks 'probabilities' capability "
                f"required by ModelInferenceSource(require_probabilities=True). "
                f"capabilities={caps.as_dict()}"
            )
        if self.require_classifier_weights and not caps.classifier_weights:
            raise UnsupportedCapabilityError(
                f"{type(self.adapter).__name__} lacks 'classifier_weights' capability "
                f"required by ModelInferenceSource(require_classifier_weights=True). "
                f"capabilities={caps.as_dict()}"
            )
        return caps

    def _validate_batch_shape(self, batch: np.ndarray, caps: Any) -> None:
        if not caps.input_validation:
            return
        try:
            self.adapter.validate_input_shape(batch)
        except InputValidationError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize to typed error
            raise InputValidationError(str(exc)) from exc

    def _forward_batches(
        self, x: Any, caps: Any, *, want_features: bool, want_logits: bool, want_probs: bool
    ) -> Dict[str, Optional[np.ndarray]]:
        if x is None:
            return {"features": None, "logits": None, "probs": None, "n": 0}

        feat_chunks, logit_chunks, prob_chunks = [], [], []
        n_total = 0

        try:
            import torch

            inference_ctx = torch.inference_mode()
        except ImportError:  # pragma: no cover - torch expected in this env
            import contextlib

            inference_ctx = contextlib.nullcontext()

        self.adapter.eval()
        with inference_ctx:
            for batch in _iter_raw_batches(x, self.batch_size):
                batch = np.asarray(batch, dtype=self.dtype)
                self._validate_batch_shape(batch, caps)
                n_total += batch.shape[0]
                if want_features and caps.features:
                    feat_chunks.append(np.asarray(self.adapter.forward_features(batch)))
                if want_logits and caps.logits:
                    logit_chunks.append(np.asarray(self.adapter.forward_logits(batch)))
                if want_probs and caps.probabilities:
                    prob_chunks.append(np.asarray(self.adapter.predict_proba(batch)))

        def _cat(chunks):
            if not chunks:
                return None
            return np.concatenate(chunks, axis=0).astype(self.dtype)

        return {
            "features": _cat(feat_chunks),
            "logits": _cat(logit_chunks),
            "probs": _cat(prob_chunks),
            "n": n_total,
        }

    # ------------------------------------------------------------------ #
    def load_cell(
        self,
        *,
        dataset: str,
        model: str,
        seed: int,
        subject: str,
        source_session: str,
        target_session: str,
        x_target: Any,
        y_target: Optional[np.ndarray] = None,
        x_source: Optional[Any] = None,
        y_source: Optional[np.ndarray] = None,
        checkpoint_path: Optional[str] = None,
        cell_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> FeatureBundle:
        self._load_checkpoint_if_needed(checkpoint_path)
        caps = self._check_required_capabilities()

        self.adapter.to(self.device)
        self.adapter.eval()

        tgt = self._forward_batches(
            x_target,
            caps,
            want_features=True,
            want_logits=True,
            want_probs=True,
        )
        n_target = int(tgt["n"])
        if n_target == 0:
            raise InputValidationError(
                f"x_target for cell (subject={subject}, {source_session}->{target_session}) "
                "produced zero samples."
            )

        src = self._forward_batches(
            x_source,
            caps,
            want_features=True,
            want_logits=False,
            want_probs=False,
        )
        n_source = int(src["n"])

        target_features = tgt["features"]
        target_logits = tgt["logits"]
        target_probs = tgt["probs"]
        source_features = src["features"]

        self._validate_consistency(
            n_target=n_target,
            target_features=target_features,
            target_logits=target_logits,
            target_probs=target_probs,
            y_target=y_target,
        )

        source_labels = None
        if y_source is not None:
            source_labels = np.asarray(y_source, dtype=np.int64).ravel()
            if source_labels.shape[0] != n_source:
                raise InputValidationError(
                    f"y_source length {source_labels.shape[0]} != x_source sample "
                    f"count {n_source}"
                )

        target_y_true = None
        if y_target is not None:
            target_y_true = np.asarray(y_target, dtype=np.int64).ravel()

        if target_logits is not None:
            target_pred = target_logits.argmax(axis=1).astype(np.int64)
        elif target_probs is not None:
            target_pred = target_probs.argmax(axis=1).astype(np.int64)
        else:
            target_pred = None
        target_conf = (
            target_probs.max(axis=1).astype(np.float32) if target_probs is not None else None
        )

        resolved_cell_id = cell_id or make_cell_id(
            dataset, model, seed, subject, source_session, target_session
        )

        merged_meta: Dict[str, Any] = {}
        try:
            merged_meta.update(self.adapter.get_model_metadata())
        except UnsupportedCapabilityError:
            pass
        merged_meta.update(
            {
                "device": self.device,
                "batch_size": self.batch_size,
                "checkpoint_path": str(checkpoint_path) if checkpoint_path else "",
                "capabilities": caps.as_dict(),
            }
        )
        if metadata:
            merged_meta.update(dict(metadata))

        embedding_dim = int(target_features.shape[1]) if target_features is not None else None

        return FeatureBundle(
            cell_id=resolved_cell_id,
            dataset=dataset,
            model=model,
            seed=int(seed),
            subject=subject,
            source_session=source_session,
            target_session=target_session,
            source_features=source_features,
            source_labels=source_labels,
            target_features=target_features,
            target_logits=target_logits,
            target_probs=target_probs,
            target_pred=target_pred,
            target_conf=target_conf,
            target_y_true=target_y_true,
            feature_source=self.name,
            npz_path_resolved="",
            npz_path_original="",
            embedding_dim=embedding_dim,
            n_source=n_source,
            n_target=n_target,
            metadata=merged_meta,
        )

    @staticmethod
    def _validate_consistency(
        *,
        n_target: int,
        target_features: Optional[np.ndarray],
        target_logits: Optional[np.ndarray],
        target_probs: Optional[np.ndarray],
        y_target: Optional[np.ndarray],
    ) -> None:
        if target_features is not None:
            if target_features.ndim != 2 or target_features.shape[0] != n_target:
                raise InputValidationError(
                    f"target_features shape {target_features.shape} inconsistent "
                    f"with n_target={n_target}"
                )
        if target_logits is not None:
            if target_logits.ndim != 2 or target_logits.shape[0] != n_target:
                raise InputValidationError(
                    f"target_logits shape {target_logits.shape} inconsistent "
                    f"with n_target={n_target}"
                )
        if target_probs is not None:
            if target_probs.ndim != 2 or target_probs.shape[0] != n_target:
                raise InputValidationError(
                    f"target_probs shape {target_probs.shape} inconsistent "
                    f"with n_target={n_target}"
                )
            if target_logits is not None and target_probs.shape != target_logits.shape:
                raise InputValidationError(
                    f"target_probs shape {target_probs.shape} != "
                    f"target_logits shape {target_logits.shape}"
                )
        if y_target is not None:
            y_arr = np.asarray(y_target).ravel()
            if y_arr.shape[0] != n_target:
                raise InputValidationError(
                    f"y_target length {y_arr.shape[0]} != n_target={n_target}"
                )
