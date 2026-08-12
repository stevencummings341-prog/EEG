"""EEG/ERP Foundation Models - Self-contained model package.

5 model variants with DualCD (OrthogonalMask + DualPerturbation):
  - S4ERP: standalone S4 (supervised baseline, no DualCD)
  - UnifiedDINODualCD_S4_Flatten: DINO + DualCD + S4 + flatten
  - UnifiedDINODualCD_S4_Pos: DINO + DualCD + S4 + attention pooling
  - UnifiedDINODualCD_S4_Timepatch: DINO + DualCD + S4 + temporal binned pooling
  - UnifiedDINODualCD_Transformer: DINO + DualCD + Transformer + flatten (original)
"""

from .s4_layers import S4Layer, S4Block, S4Encoder
from .pooling import FlattenPooling, AttentionPooling, TemporalBinnedPooling
from .encoders import ShallowNetEmbedding, S4ERPEncoder, TransformerERPEncoder
from .losses import (
    DINOLoss, ProjectionHead, IBOTHead, DKoleoLoss, PrototypeBank,
    OrthogonalMask, intra_class_perturbation, inter_class_perturbation,
)
from .models import (
    S4ERP,
    UnifiedDINODualCD_S4_Flatten,
    UnifiedDINODualCD_S4_Pos,
    UnifiedDINODualCD_S4_Timepatch,
    UnifiedDINODualCD_Transformer,
)

__all__ = [
    "S4Layer", "S4Block", "S4Encoder",
    "FlattenPooling", "AttentionPooling", "TemporalBinnedPooling",
    "ShallowNetEmbedding", "S4ERPEncoder", "TransformerERPEncoder",
    "DINOLoss", "ProjectionHead", "IBOTHead", "DKoleoLoss", "PrototypeBank",
    "OrthogonalMask", "intra_class_perturbation", "inter_class_perturbation",
    "S4ERP",
    "UnifiedDINODualCD_S4_Flatten",
    "UnifiedDINODualCD_S4_Pos",
    "UnifiedDINODualCD_S4_Timepatch",
    "UnifiedDINODualCD_Transformer",
]
