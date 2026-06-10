"""Feature-space alignment transforms for Step-2 no-learning adaptation.

Every transform follows the SAME fit/transform contract and the SAME fairness
rule:

  * ``fit(X)`` estimates the alignment statistics from a set of trials ``X``
    (shape ``[N, C, T]``, C=58, T=1000). It is called either on the **source
    train split** (to align the source) or on the **target/test X** (unlabeled,
    to align the target). It NEVER sees labels.
  * ``transform(X)`` applies the fitted statistics to ``X`` and returns an array
    of the SAME shape ``[N, C, T]`` (so the downstream model is unchanged).
  * ``summary()`` returns a small JSON-serializable dict logged into the result
    CSV (``source_alignment_stats`` / ``target_alignment_stats``).

Implemented methods (the four feature-space ones; ``bn_statistics_adaptation``
lives in ``bn_adaptation.py`` and ``none_reference`` is pulled from baseline_v1
by the summarizer):

  * ``session_zscore``        — per-channel mean/std normalization.
  * ``euclidean_alignment``   — whiten by the inverse square root of the
                                Euclidean (arithmetic) mean of trial covariances.
                                ``R^{-1/2}`` computed via symmetric eigh.
  * ``riemannian_alignment``  — whiten by the inverse square root of the
                                LOG-EUCLIDEAN mean of trial covariances (a stable
                                SPD mean; ``expm(mean_i logm(C_i))``), computed
                                via eigh. We do NOT depend on pyriemann; if it is
                                importable we only record that fact.
  * ``filterbank_reweighting``— conservative spectral alignment: decompose the
                                signal into μ/β (+θ/low-γ) FIR sub-bands and
                                reweight each band so the target band-power
                                profile matches the source band-power profile.

Numerical safety: covariance matrices get an ``eps`` ridge + optional diagonal
``shrinkage`` toward ``tr(R)/C * I`` before any eigh/inverse, and eigenvalues are
clipped to ``>= eps`` so no inverse-square-root or log is ever taken of a
non-positive eigenvalue (guards against singular / rank-deficient covariances).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

EPS_DEFAULT = 1e-5
SHRINKAGE_DEFAULT = 0.1


def _check_xshape(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 3:
        raise ValueError(f"alignment expects X of shape [N, C, T]; got {X.shape}")
    return X


# --------------------------------------------------------------------------- #
# SPD helpers (all via symmetric eigh, never matrix-power on raw matrices)
# --------------------------------------------------------------------------- #
def _regularize(M: np.ndarray, eps: float, shrinkage: float) -> np.ndarray:
    """Ridge + diagonal shrinkage toward tr(M)/C * I (keeps M SPD / well-conditioned)."""
    c = M.shape[0]
    M = 0.5 * (M + M.T)
    tr = float(np.trace(M)) / c
    if shrinkage > 0.0:
        M = (1.0 - shrinkage) * M + shrinkage * tr * np.eye(c)
    M = M + eps * np.eye(c)
    return M


def _eigh_clip(M: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    w, V = np.linalg.eigh(0.5 * (M + M.T))
    w = np.clip(w, eps, None)
    return w, V


def inv_sqrt_spd(M: np.ndarray, eps: float = EPS_DEFAULT, shrinkage: float = 0.0) -> np.ndarray:
    """Symmetric inverse square root ``M^{-1/2}`` of an SPD matrix via eigh."""
    Mr = _regularize(M, eps, shrinkage)
    w, V = _eigh_clip(Mr, eps)
    return (V * (1.0 / np.sqrt(w))) @ V.T


def logm_spd(M: np.ndarray, eps: float = EPS_DEFAULT, shrinkage: float = 0.0) -> np.ndarray:
    """Matrix logarithm of an SPD matrix via eigh (eigenvalues clipped > 0)."""
    Mr = _regularize(M, eps, shrinkage)
    w, V = _eigh_clip(Mr, eps)
    return (V * np.log(w)) @ V.T


def expm_sym(M: np.ndarray) -> np.ndarray:
    """Matrix exponential of a symmetric matrix via eigh."""
    w, V = np.linalg.eigh(0.5 * (M + M.T))
    return (V * np.exp(w)) @ V.T


def trial_covariances(X: np.ndarray) -> np.ndarray:
    """Per-trial spatial covariance ``[N, C, C]`` (time-mean removed, unbiased)."""
    X = _check_xshape(X)
    n, c, t = X.shape
    Xc = X - X.mean(axis=2, keepdims=True)
    covs = np.einsum("nct,ndt->ncd", Xc, Xc) / max(t - 1, 1)
    return covs


# --------------------------------------------------------------------------- #
# 1. Channel-wise z-score
# --------------------------------------------------------------------------- #
class ChannelZScore:
    """Per-channel z-score: ``(x - mean_c) / std_c`` (mean/std over trials*time)."""

    name = "session_zscore"

    def __init__(self, eps: float = 1e-8):
        self.eps = float(eps)
        self.mean_: Optional[np.ndarray] = None  # [C]
        self.std_: Optional[np.ndarray] = None   # [C]

    def fit(self, X: np.ndarray) -> "ChannelZScore":
        X = _check_xshape(X)
        self.mean_ = X.mean(axis=(0, 2))
        self.std_ = X.std(axis=(0, 2))
        self.std_ = np.where(self.std_ < self.eps, 1.0, self.std_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("ChannelZScore.transform called before fit().")
        X = _check_xshape(X)
        out = (X - self.mean_[None, :, None]) / self.std_[None, :, None]
        return out.astype(np.float32)

    def summary(self) -> Dict[str, object]:
        return {
            "method": self.name,
            "mean_abs_mean": float(np.mean(np.abs(self.mean_))),
            "mean_std": float(np.mean(self.std_)),
            "min_std": float(np.min(self.std_)),
            "n_channels": int(self.mean_.shape[0]),
        }


# --------------------------------------------------------------------------- #
# 2. Euclidean Alignment (He & Wu 2020)
# --------------------------------------------------------------------------- #
class EuclideanAlignment:
    """Whiten trials by ``R^{-1/2}`` where R = arithmetic mean of trial covs.

    After transform the mean covariance of the aligned trials is ~identity, so
    every domain is recentered to the same reference -> reduces cross-session
    covariance shift. ``R`` gets eps + shrinkage before the eigh inverse-sqrt.
    """

    name = "euclidean_alignment"

    def __init__(self, eps: float = EPS_DEFAULT, shrinkage: float = SHRINKAGE_DEFAULT):
        self.eps = float(eps)
        self.shrinkage = float(shrinkage)
        self.W_: Optional[np.ndarray] = None    # [C, C] = R^{-1/2}
        self.ref_trace_: float = float("nan")
        self.cond_: float = float("nan")

    def fit(self, X: np.ndarray) -> "EuclideanAlignment":
        covs = trial_covariances(X)
        R = covs.mean(axis=0)
        self.ref_trace_ = float(np.trace(R))
        self.W_ = inv_sqrt_spd(R, eps=self.eps, shrinkage=self.shrinkage)
        Rr = _regularize(R, self.eps, self.shrinkage)
        w, _ = _eigh_clip(Rr, self.eps)
        self.cond_ = float(w.max() / w.min())
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.W_ is None:
            raise RuntimeError("EuclideanAlignment.transform called before fit().")
        X = _check_xshape(X)
        out = np.einsum("cd,ndt->nct", self.W_, X)
        return out.astype(np.float32)

    def summary(self) -> Dict[str, object]:
        return {
            "method": self.name,
            "matrix_shape": list(self.W_.shape),
            "ref_trace": self.ref_trace_,
            "cond_number": self.cond_,
            "eps": self.eps,
            "shrinkage": self.shrinkage,
        }


# --------------------------------------------------------------------------- #
# 3. Riemannian Alignment (log-Euclidean SPD mean)
# --------------------------------------------------------------------------- #
class RiemannianAlignment:
    """Whiten trials by ``G^{-1/2}`` where G = log-Euclidean mean of trial covs.

    Implementation (documented, numpy/scipy only, NO pyriemann dependency):
      G = expm( (1/N) * sum_i logm(C_i) ),  C_i = per-trial covariance.
    The log-Euclidean mean is a stable, closed-form SPD mean (each logm/expm via
    symmetric eigh with eigenvalue clipping). This differs from Euclidean
    Alignment, which uses the arithmetic mean of covariances. If ``pyriemann`` is
    importable we record ``pyriemann_available=True`` but never call it.
    """

    name = "riemannian_alignment"

    def __init__(self, eps: float = EPS_DEFAULT, shrinkage: float = SHRINKAGE_DEFAULT):
        self.eps = float(eps)
        self.shrinkage = float(shrinkage)
        self.W_: Optional[np.ndarray] = None    # [C, C] = G^{-1/2}
        self.mean_trace_: float = float("nan")
        self.cond_: float = float("nan")
        self.pyriemann_available_ = _pyriemann_available()

    def fit(self, X: np.ndarray) -> "RiemannianAlignment":
        covs = trial_covariances(X)
        log_sum = None
        for ci in covs:
            li = logm_spd(ci, eps=self.eps, shrinkage=self.shrinkage)
            log_sum = li if log_sum is None else log_sum + li
        log_mean = log_sum / covs.shape[0]
        G = expm_sym(log_mean)
        self.mean_trace_ = float(np.trace(G))
        self.W_ = inv_sqrt_spd(G, eps=self.eps, shrinkage=0.0)
        w, _ = _eigh_clip(_regularize(G, self.eps, 0.0), self.eps)
        self.cond_ = float(w.max() / w.min())
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.W_ is None:
            raise RuntimeError("RiemannianAlignment.transform called before fit().")
        X = _check_xshape(X)
        out = np.einsum("cd,ndt->nct", self.W_, X)
        return out.astype(np.float32)

    def summary(self) -> Dict[str, object]:
        return {
            "method": self.name,
            "spd_mean": "log_euclidean",
            "matrix_shape": list(self.W_.shape),
            "mean_trace": self.mean_trace_,
            "cond_number": self.cond_,
            "eps": self.eps,
            "shrinkage": self.shrinkage,
            "pyriemann_available": bool(self.pyriemann_available_),
        }


def _pyriemann_available() -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec("pyriemann") is not None
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# 4. Filter-bank reweighting (conservative spectral alignment)
# --------------------------------------------------------------------------- #
class FilterBankReweight:
    """Conservative band-power alignment over μ/β (+θ/low-γ) FIR sub-bands.

    Pipeline (identical for source and target so train/test stay consistent):
      1. decompose ``x`` into ``len(bands)`` band-pass views with fixed
         linear-phase FIR kernels (firwin, Hamming);
      2. estimate each band's mean power ``P_b = mean(view_b ** 2)``;
      3. rebuild ``x' = sum_b w_b * view_b`` with ``w_b = sqrt(P_ref_b / P_b)``.

    ``fit(X)`` stores the band-power profile of ``X`` as the reference ``P_ref``.
      * Fitted on the SOURCE TRAIN split -> the stored ``P_ref`` is the source
        profile; applied to source train (w≈1) and source val (≈1).
      * For the TARGET, we transform target X **with the SOURCE P_ref** so the
        target band powers (estimated from the unlabeled target X) are reweighted
        toward the source profile — i.e. the source-trained model sees a target
        whose μ/β balance matches the source. This is the conservative version:
        a single scalar gain per band, clipped to ``[w_min, w_max]`` for
        stability; no per-channel or per-trial reweighting, no learning.

    ``target_alignment_stats`` therefore records the gains actually applied to the
    target (``used_target_x_for_stats=True`` because the target band powers come
    from target X).
    """

    name = "filterbank_reweighting"

    def __init__(self, sfreq: int = 250, bands: Optional[List[Tuple[float, float]]] = None,
                 taps: int = 125, w_min: float = 0.5, w_max: float = 2.0, eps: float = 1e-8):
        self.sfreq = int(sfreq)
        self.bands = bands or [(4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 40.0)]
        self.taps = taps if taps % 2 == 1 else taps + 1
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        self.eps = float(eps)
        self._kernels = self._design_kernels()
        self.p_ref_: Optional[np.ndarray] = None   # [n_bands]
        self.last_weights_: Optional[np.ndarray] = None

    def _design_kernels(self) -> np.ndarray:
        from scipy.signal import firwin
        nyq = self.sfreq / 2.0
        ks = []
        for lo, hi in self.bands:
            hi = min(hi, nyq - 1.0)
            if hi <= lo:
                raise ValueError(f"FilterBankReweight: invalid band ({lo},{hi}) at fs={self.sfreq}")
            ks.append(firwin(self.taps, [lo, hi], pass_zero=False, fs=self.sfreq).astype(np.float64))
        return np.stack(ks, axis=0)   # [n_bands, taps]

    def _bandpass(self, X: np.ndarray) -> np.ndarray:
        """Return band views ``[n_bands, N, C, T]`` via same-length FIR convolution."""
        from scipy.signal import fftconvolve
        X = _check_xshape(X)
        views = []
        for k in self._kernels:
            kk = k[None, None, :]
            v = fftconvolve(X, kk, mode="same", axes=-1)
            views.append(v)
        return np.stack(views, axis=0)

    @staticmethod
    def _band_power(views: np.ndarray) -> np.ndarray:
        # views [n_bands, N, C, T] -> [n_bands]
        return np.mean(views ** 2, axis=(1, 2, 3))

    def fit(self, X: np.ndarray) -> "FilterBankReweight":
        views = self._bandpass(X)
        self.p_ref_ = self._band_power(views)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.p_ref_ is None:
            raise RuntimeError("FilterBankReweight.transform called before fit().")
        views = self._bandpass(X)
        p_x = self._band_power(views)
        w = np.sqrt(self.p_ref_ / (p_x + self.eps))
        w = np.clip(w, self.w_min, self.w_max)
        self.last_weights_ = w
        out = np.tensordot(w, views, axes=([0], [0]))   # [N, C, T]
        return out.astype(np.float32)

    def summary(self) -> Dict[str, object]:
        return {
            "method": self.name,
            "bands_hz": [list(b) for b in self.bands],
            "band_power_ref": None if self.p_ref_ is None else [float(p) for p in self.p_ref_],
            "applied_weights": None if self.last_weights_ is None else [float(x) for x in self.last_weights_],
            "w_clip": [self.w_min, self.w_max],
            "taps": self.taps,
        }


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
ALIGNMENT_METHODS: List[str] = [
    "session_zscore",
    "euclidean_alignment",
    "riemannian_alignment",
    "filterbank_reweighting",
]


def make_alignment_method(name: str, *, sfreq: int = 250, params: Optional[Dict] = None):
    """Construct a feature-space alignment transform by name (fresh, unfitted)."""
    key = (name or "").lower().strip()
    params = dict(params or {})
    if key == "session_zscore":
        return ChannelZScore(eps=params.get("zscore_eps", 1e-8))
    if key == "euclidean_alignment":
        return EuclideanAlignment(eps=params.get("eps", EPS_DEFAULT),
                                  shrinkage=params.get("shrinkage", SHRINKAGE_DEFAULT))
    if key == "riemannian_alignment":
        return RiemannianAlignment(eps=params.get("eps", EPS_DEFAULT),
                                   shrinkage=params.get("shrinkage", SHRINKAGE_DEFAULT))
    if key == "filterbank_reweighting":
        return FilterBankReweight(
            sfreq=sfreq,
            bands=params.get("bands"),
            taps=params.get("taps", 125),
            w_min=params.get("w_min", 0.5),
            w_max=params.get("w_max", 2.0),
        )
    raise ValueError(f"unknown alignment method '{name}'. Known: {ALIGNMENT_METHODS}")
