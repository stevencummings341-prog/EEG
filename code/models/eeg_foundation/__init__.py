"""EEG foundation models: S4 / DINO-DualCD family (5 variants).

Ported from the advisor-supplied `models_eeg_foundation/` package (2026-08-04) and
wrapped for this project in `adapter.py`. Experiments must go through the model
registry (``code/models/registry.build_model``) or `build_eeg_foundation`, never
through the raw classes, so the `{logits, features, confidence}` contract and the
``[B, C, T]`` tensor convention hold everywhere.

Layout:
  * ``s4_layers.py`` — S4 (HiPPO-LegS init + FFT convolution), pure PyTorch
  * ``pooling.py``   — flatten / attention / temporal-binned pooling
  * ``encoders.py``  — ShallowNet CNN stem + S4 or Transformer encoder
  * ``losses.py``    — DINO / iBOT / DKoleo / PrototypeBank / OrthogonalMask / perturbation
  * ``models.py``    — the 5 model variants
  * ``adapter.py``   — project contract + training hooks + per-trial normalization
"""

from .adapter import (
    DEFAULT_LR,
    VARIANT_NAMES,
    VARIANTS,
    EEGFoundationClassifier,
    EEGFoundationConfig,
    build_eeg_foundation,
    normalize_trials,
)
from .models import (
    S4ERP,
    MultiViewGenerator,
    UnifiedDINODualCD_S4_Flatten,
    UnifiedDINODualCD_S4_Pos,
    UnifiedDINODualCD_S4_Timepatch,
    UnifiedDINODualCD_Transformer,
)

__all__ = [
    "EEGFoundationConfig", "EEGFoundationClassifier", "build_eeg_foundation",
    "normalize_trials", "VARIANTS", "VARIANT_NAMES", "DEFAULT_LR",
    "S4ERP", "MultiViewGenerator",
    "UnifiedDINODualCD_S4_Flatten", "UnifiedDINODualCD_S4_Pos",
    "UnifiedDINODualCD_S4_Timepatch", "UnifiedDINODualCD_Transformer",
]
