"""Self-supervised losses for DINO-style training.

Components:
  - DINOLoss: self-distillation with centering and sharpening
  - IBOTHead: patch-level masked prediction head
  - DKoleoLoss: anti-collapse contrastive loss (KoLeo)
  - ProjectionHead: MLP projection for DINO/iBOT
  - PrototypeBank: per-class prototype bank with EMA update
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


class DINOLoss(nn.Module):
    """DINOv2-style self-distillation loss.

    Student: softmax(z_student / tau_s)
    Teacher: softmax((z_teacher - center) / tau_t)
    Loss: -teacher_probs * log(student_probs)

    Parameters
    ----------
    out_dim : int
        Projection dimension (default 256).
    student_temp : float
        Student temperature (default 0.1).
    teacher_temp : float
        Teacher temperature (default 0.04).
    center_momentum : float
        EMA momentum for center update (default 0.9).
    """

    def __init__(self, out_dim: int = 256, student_temp: float = 0.1,
                 teacher_temp: float = 0.04, center_momentum: float = 0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output: torch.Tensor, teacher_output: torch.Tensor,
                update_center: bool = True) -> torch.Tensor:
        student_probs = F.log_softmax(student_output / self.student_temp, dim=-1)
        teacher_probs = F.softmax((teacher_output - self.center) / self.teacher_temp, dim=-1)
        loss = -(teacher_probs * student_probs).sum(dim=-1).mean()

        if update_center:
            self.update_center(teacher_output)
        return loss

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor):
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center.mul_(self.center_momentum).add_(batch_center, alpha=1 - self.center_momentum)


class ProjectionHead(nn.Module):
    """MLP projection head for DINO.

    Linear -> GELU -> Linear
    """

    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class IBOTHead(nn.Module):
    """Prediction head for iBOT masked patch reconstruction.

    Linear -> GELU -> Linear
    """

    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class DKoleoLoss(nn.Module):
    """KoLeo (Kozachenko-Leonenko) anti-collapse loss.

    Encourages representations to spread uniformly in feature space
    by minimizing the entropy of nearest-neighbor distances.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """(B, D) -> scalar loss"""
        # Pairwise distances
        dists = torch.cdist(features, features)  # (B, B)
        # Set diagonal to infinity
        dists = dists + torch.eye(dists.size(0), device=dists.device) * 1e6
        # Min distance per sample
        min_dists, _ = dists.min(dim=1)  # (B,)
        # KoLeo loss: minimize log of min distances
        return -torch.log(min_dists + self.eps).mean()


class PrototypeBank(nn.Module):
    """Per-class prototype bank for familiarity estimation.

    Maintains K prototypes per class, updated via EMA from teacher embeddings.
    Computes distance-based entropy for anomaly detection.

    Parameters
    ----------
    num_classes : int
        Number of classes.
    feature_dim : int
        Feature dimension.
    k : int
        Prototypes per class.
    """

    def __init__(self, num_classes: int, feature_dim: int, k: int = 5):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.k = k
        self.register_buffer(
            "prototypes", torch.randn(num_classes, k, feature_dim) * 0.01
        )

    @torch.no_grad()
    def update(self, features: torch.Tensor, labels: torch.Tensor,
               momentum: float = 0.9):
        """Update prototypes with EMA from teacher embeddings."""
        for c in range(self.num_classes):
            mask = labels == c
            if mask.any():
                class_z = features[mask]
                n = min(self.k, class_z.size(0))
                self.prototypes[c, :n] = (
                    momentum * self.prototypes[c, :n] + (1 - momentum) * class_z[:n]
                )

    def forward(self, z_causal: torch.Tensor) -> torch.Tensor:
        """Compute entropy of distance-based assignment.

        Returns: (B,) entropy per sample (lower = more familiar).
        """
        B, D = z_causal.shape
        all_protos = self.prototypes.reshape(-1, D)
        dists = torch.cdist(z_causal.unsqueeze(0), all_protos.unsqueeze(0)).squeeze(0)
        soft_assign = F.softmax(-dists, dim=-1)
        entropy = -torch.sum(soft_assign * torch.log(soft_assign + 1e-8), dim=-1)
        return entropy


# ── OrthogonalMask (DualCD Core) ─────────────────────────────────────────────

class OrthogonalMask(nn.Module):
    """Split features into causal and spurious components.

    z_causal:   features correlated with label (task-relevant)
    z_spurious: features uncorrelated with label (confounders)

    Constraint: z_causal ⊥ z_spurious (approximately, via complementary masks)
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = nn.Linear(d_model, d_model)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        h : (B, D) where D = patch_num * d_model

        Returns
        -------
        (z_causal, z_spurious), each (B, D)
        """
        B, D = h.shape
        d_model = self.mlp.in_features
        patch_num = D // d_model
        h_patches = h.reshape(B, patch_num, d_model)
        M = self.mlp(h_patches)
        mask_causal = torch.sigmoid(M)
        mask_spurious = 1.0 - mask_causal
        z_causal = (mask_causal * h_patches).reshape(B, D)
        z_spurious = (mask_spurious * h_patches).reshape(B, D)
        return z_causal, z_spurious


# ── Dual Perturbation ────────────────────────────────────────────────────────

def intra_class_perturbation(z_spurious: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Replace z_spurious with a random same-class sample's z_spurious."""
    result = z_spurious.clone()
    for i in range(len(z_spurious)):
        same_class = (y == y[i]).nonzero(as_tuple=True)[0]
        candidates = same_class[same_class != i]
        if len(candidates) > 0:
            j = candidates[torch.randint(len(candidates), (1,)).item()]
            result[i] = z_spurious[j]
    return result


def inter_class_perturbation(z_causal: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Replace with a random other-class sample's z_causal."""
    result = torch.zeros_like(z_causal)
    for i in range(len(z_causal)):
        other_class = (y != y[i]).nonzero(as_tuple=True)[0]
        if len(other_class) > 0:
            j = other_class[torch.randint(len(other_class), (1,)).item()]
            result[i] = z_causal[j]
    return result
