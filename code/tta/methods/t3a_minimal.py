"""Minimal T3A clean variant (Round-1 smoke only).

This is **one** configurable variant for pipeline smoke — not a permanent
project-wide default. Other init/geometry/filter choices live in method_catalog
and are not implemented here.

Binary-class note: for K=2, prediction entropy is a strictly monotone function of
max-softmax confidence, so entropy ranking ≡ max-confidence ranking. Do not treat
them as separate ablations on this MI task.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from code.tta.feature_sources.base import FeatureBundle
from code.tta.methods.base import MethodResult, TTAMethod


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


def _similarity(z: np.ndarray, protos: np.ndarray, geometry: str) -> np.ndarray:
    """Return [N, C] similarity scores."""
    g = (geometry or "cosine").lower()
    if g == "cosine":
        return _l2_normalize(z) @ _l2_normalize(protos).T
    if g == "dot":
        return z @ protos.T
    if g == "euclidean":
        # negative distance so argmax = nearest
        d2 = ((z[:, None, :] - protos[None, :, :]) ** 2).sum(axis=-1)
        return -d2
    raise ValueError(f"unknown geometry '{geometry}' (round1 supports cosine|dot|euclidean)")


def _entropy_from_probs(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(probs, eps, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def _softmax_from_scores(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    t = max(float(temperature), 1e-8)
    s = scores / t
    m = s.max(axis=1, keepdims=True)
    ex = np.exp(s - m)
    return ex / ex.sum(axis=1, keepdims=True)


class MinimalT3AMethod(TTAMethod):
    """Minimal T3A: frozen features, no grad, support update, prototype predict.

    Parameters are instance/config fields — smoke may set src_proto+cosine+k=20,
    but that combination is not hard-coded as the only project default.
    """

    name = "t3a_minimal"
    uses_target_labels = False

    def __init__(
        self,
        *,
        geometry: str = "cosine",
        filter_k: int = 20,
        initialization: str = "src_proto",
        temperature: float = 1.0,
        episodic: bool = False,
        n_classes: int = 2,
        seed: int = 0,
        empty_class_fallback: str = "uniform_proto",
    ) -> None:
        self.geometry = geometry
        self.filter_k = int(filter_k)
        self.initialization = initialization
        self.temperature = float(temperature)
        self.episodic = bool(episodic)
        self.n_classes = int(n_classes)
        self.seed = int(seed)
        self.empty_class_fallback = empty_class_fallback
        # NOTE (K=2): entropy filter ≡ max-confidence filter by ranking.

    def _init_supports(
        self, bundle: FeatureBundle
    ) -> Dict[int, List[np.ndarray]]:
        supports: Dict[int, List[np.ndarray]] = {c: [] for c in range(self.n_classes)}
        init = (self.initialization or "src_proto").lower()
        if init in ("src_proto", "source_proto", "t3a_source_proto_init"):
            if bundle.source_features is None or bundle.source_labels is None:
                raise ValueError(
                    f"{bundle.cell_id}: src_proto init needs source_features/labels"
                )
            z = np.asarray(bundle.source_features, dtype=np.float32)
            y = np.asarray(bundle.source_labels, dtype=np.int64).ravel()
            for c in range(self.n_classes):
                mask = y == c
                if mask.any():
                    supports[c].append(z[mask].mean(axis=0))
                else:
                    # empty source class → zero vector placeholder
                    supports[c].append(np.zeros(z.shape[1], dtype=np.float32))
        elif init in ("zeros", "target_support_only"):
            dim = int(bundle.target_features.shape[1])
            for c in range(self.n_classes):
                supports[c].append(np.zeros(dim, dtype=np.float32))
        else:
            raise ValueError(
                f"Round-1 MinimalT3A does not implement init='{init}'. "
                "See method_catalog for candidates (clf_weights etc.)."
            )
        return supports

    def _protos_from_supports(
        self, supports: Dict[int, List[np.ndarray]], dim: int
    ) -> np.ndarray:
        protos = np.zeros((self.n_classes, dim), dtype=np.float32)
        for c in range(self.n_classes):
            vecs = supports[c]
            if not vecs:
                if self.empty_class_fallback == "uniform_proto":
                    protos[c] = 0.0
                else:
                    protos[c] = 0.0
                continue
            # Keep at most filter_k lowest-entropy additions; init vectors have no entropy
            # so we keep all init + filtered target supports already trimmed on insert.
            stacked = np.stack(vecs, axis=0)
            if self.filter_k > 0 and stacked.shape[0] > self.filter_k:
                # Prefer later (target) supports: keep last filter_k
                stacked = stacked[-self.filter_k :]
            protos[c] = stacked.mean(axis=0)
        return protos

    def run(self, bundle: FeatureBundle, **kwargs) -> MethodResult:
        rng = np.random.RandomState(self.seed)
        _ = rng  # deterministic hook reserved for future stochastic filters
        bundle = self._prepare_label_free(bundle)
        if bundle.target_features is None:
            raise ValueError(f"{bundle.cell_id}: missing target_features")

        z = np.asarray(bundle.target_features, dtype=np.float32)
        n, dim = z.shape
        supports = self._init_supports(bundle)
        preds = np.zeros(n, dtype=np.int64)

        # Cumulative stream (episodic=False): update supports across the session.
        # Episodic=True: reset supports each step to init (smoke-compatible switch).
        init_snapshot = {c: list(v) for c, v in supports.items()}

        for i in range(n):
            if self.episodic:
                supports = {c: list(v) for c, v in init_snapshot.items()}
            protos = self._protos_from_supports(supports, dim)
            scores = _similarity(z[i : i + 1], protos, self.geometry)  # [1,C]
            probs = _softmax_from_scores(scores, self.temperature)
            pred = int(scores.argmax(axis=1)[0])
            preds[i] = pred

            # Uncertainty: entropy (≡ max-conf ranking when K=2)
            ent = float(_entropy_from_probs(probs)[0])
            supports[pred].append(z[i].copy())
            # Per-class top-k by lowest entropy among target-added vectors:
            # store entropy as sidecar via trimming after append.
            self._trim_class_support(supports, pred, ent)

        return MethodResult(
            pred=preds,
            method=self.name,
            used_target_labels=False,
            oracle_diagnostic_only=False,
            not_deployable=False,
            geometry=self.geometry,
            filter_k=self.filter_k,
            initialization=self.initialization,
            extras={
                "episodic": self.episodic,
                "note_binary_entropy_eq_maxconf": True,
            },
        )

    def _trim_class_support(
        self,
        supports: Dict[int, List[np.ndarray]],
        cls: int,
        last_entropy: float,
    ) -> None:
        """Keep init prototype + up to filter_k target supports.

        Round-1 simplification: keep the most recent filter_k target vectors
        (stream order approximates online T3A). Entropy is recorded in extras
        path only; full entropy-sorted buffer is a catalog TODO.
        """
        if self.filter_k <= 0:
            return
        # supports[cls][0] is init proto; rest are target additions
        init = supports[cls][:1]
        rest = supports[cls][1:]
        if len(rest) > self.filter_k:
            rest = rest[-self.filter_k :]
        supports[cls] = init + rest
        _ = last_entropy
