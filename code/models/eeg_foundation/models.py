"""EEG/ERP foundation model variants with DualCD.

Model hierarchy (all include OrthogonalMask + DualPerturbation):
  UnifiedDINODualCD_Transformer (original)
    ├─ UnifiedDINODualCD_S4_Flatten:   Transformer->S4, pooling unchanged
    ├─ UnifiedDINODualCD_S4_Pos:       Transformer->S4 + flatten->attention pooling
    └─ UnifiedDINODualCD_S4_Timepatch: Transformer->S4 + flatten->temporal binned pooling

  Standalone (no DualCD):
    └─ S4ERP: supervised S4 baseline

All DINO models share:
  - DINO self-distillation + iBOT + DKoleo
  - OrthogonalMask (z_causal / z_spurious separation)
  - DualPerturbation (intra-class + inter-class)
  - PrototypeBank (EMA updated)
  - compute_loss(x, y, epoch) -> (loss, parts_dict)

Provenance: ported from the advisor-supplied `models_eeg_foundation/` package
(2026-08-04). Deviations from that package are listed in this directory's
`README.md` (section "Porting notes"). The project-facing wrapper that adds the
`{logits, features, confidence}` contract lives in `adapter.py`; do not call these
classes directly from experiments.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .encoders import ShallowNetEmbedding, TransformerERPEncoder
from .pooling import FlattenPooling, AttentionPooling, TemporalBinnedPooling
from .s4_layers import S4Encoder as S4EncoderCore
from .losses import (
    DINOLoss, ProjectionHead, IBOTHead, DKoleoLoss, PrototypeBank,
    OrthogonalMask, intra_class_perturbation, inter_class_perturbation,
)


# ── Multi-View Generator ────────────────────────────────────────────────────

class MultiViewGenerator:
    """Generate 6 views from a single EEG trial.

    Teacher views (2): global_a (identity), global_b (noise)
    Student views (4): local_time_1, local_time_2, local_freq_1, local_freq_2

    ``low_band`` / ``high_band`` are the two band-pass views (Hz). Defaults are the
    ERP values from the original package (4-12 / 12-30). For motor imagery set them
    to the mu/beta bands (8-13 / 13-30) via the model config — the two attributes are
    read at every ``generate`` call, so they can also be overridden after construction.
    """

    def __init__(self, seq_len: int = 170, sfreq: float = 200.0,
                 low_band: Tuple[float, float] = (4.0, 12.0),
                 high_band: Tuple[float, float] = (12.0, 30.0),
                 noise_std: float = 0.1):
        self.seq_len = seq_len
        self.sfreq = sfreq
        self.low_band = tuple(low_band)
        self.high_band = tuple(high_band)
        self.noise_std = float(noise_std)

    def _bandpass_fft(self, x, low, high):
        B, T, C = x.shape
        X_fft = torch.fft.rfft(x, dim=1)
        freqs = torch.fft.rfftfreq(T, d=1.0 / self.sfreq, device=x.device)
        mask = ((freqs >= low) & (freqs <= high)).float().unsqueeze(0).unsqueeze(-1)
        return torch.fft.irfft(X_fft * mask, n=T, dim=1)

    def generate(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, T, C = x.shape
        crop_len = T // 2
        lt1 = torch.zeros_like(x)
        lt2 = torch.zeros_like(x)
        for i in range(B):
            s1 = torch.randint(0, T - crop_len + 1, (1,)).item()
            lt1[i, :crop_len, :] = x[i, s1:s1 + crop_len, :]
            s2 = torch.randint(0, T - crop_len + 1, (1,)).item()
            while s2 == s1 and T - crop_len > 0:
                s2 = torch.randint(0, T - crop_len + 1, (1,)).item()
            lt2[i, :crop_len, :] = x[i, s2:s2 + crop_len, :]
        return {
            "global_a": x,
            "global_b": x + torch.randn_like(x) * self.noise_std,
            "local_time_1": lt1, "local_time_2": lt2,
            "local_freq_1": self._bandpass_fft(x, *self.low_band),
            "local_freq_2": self._bandpass_fft(x, *self.high_band),
        }


# ── Base DualCD Model ───────────────────────────────────────────────────────

class _BaseDualCDModel(nn.Module):
    """Shared logic for all DualCD variants."""

    def __init__(self, config, d_model, lambda_intra, dino_out_dim,
                 proto_k, teacher_momentum):
        super().__init__()
        self.d_model = d_model
        self.num_classes = config.num_classes
        self.lambda_intra = lambda_intra
        self.teacher_momentum = teacher_momentum
        sfreq = getattr(config, "sampling_rate", 200.0)
        self.multi_view = MultiViewGenerator(
            seq_len=config.seq_len, sfreq=sfreq,
            low_band=getattr(config, "view_low_band", (4.0, 12.0)),
            high_band=getattr(config, "view_high_band", (12.0, 30.0)),
        )
        self.dino_loss = DINOLoss(out_dim=dino_out_dim)
        self.dkoleo_loss = DKoleoLoss()

    def _update_teacher_ema(self):
        m = self.teacher_momentum
        for sp, tp in zip(self.student_embedding.parameters(), self.teacher_embedding.parameters()):
            tp.data.mul_(m).add_(sp.data, alpha=1 - m)
        for sp, tp in zip(self.student_encoder.parameters(), self.teacher_encoder.parameters()):
            tp.data.mul_(m).add_(sp.data, alpha=1 - m)
        for sp, tp in zip(self.student_proj.parameters(), self.teacher_proj.parameters()):
            tp.data.mul_(m).add_(sp.data, alpha=1 - m)

    def _encode_student_flat(self, x):
        z = self.student_embedding(x)
        z = self.student_encoder(z)
        return self.student_pool(z)

    @torch.no_grad()
    def _encode_teacher_flat(self, x):
        z = self.teacher_embedding(x)
        z = self.teacher_encoder(z)
        return self.teacher_pool(z)

    def train(self, mode=True):
        super().train(mode)
        self.teacher_embedding.eval()
        self.teacher_encoder.eval()
        self.teacher_proj.eval()
        return self

    def compute_loss(self, x, y, epoch=0, warmup_epochs=5):
        """Unified training loss: DINO + iBOT + DKoleo + DualCD + Prototype.

        Returns: (total_loss, loss_parts_dict)
        """
        B = x.shape[0]
        views = self.multi_view.generate(x)

        # Teacher encodes global views
        with torch.no_grad():
            h_t_a = self._encode_teacher_flat(views["global_a"])
            h_t_b = self._encode_teacher_flat(views["global_b"])
            h_t_a_patch = self.teacher_embedding(views["global_a"])
            h_t_a_patch = self.teacher_encoder(h_t_a_patch)

        # Student encodes local views
        h_lt1 = self._encode_student_flat(views["local_time_1"])
        h_lt2 = self._encode_student_flat(views["local_time_2"])
        h_lf1 = self._encode_student_flat(views["local_freq_1"])
        h_lf2 = self._encode_student_flat(views["local_freq_2"])
        h_global = self._encode_student_flat(views["global_a"])

        # DINO projections
        z_t_a = self.teacher_proj(h_t_a)
        z_t_b = self.teacher_proj(h_t_b)
        z_s = [self.student_proj(h) for h in [h_lt1, h_lt2, h_lf1, h_lf2]]

        # DINO loss
        dino_loss = sum(
            self.dino_loss(z, z_t_a, update_center=False) +
            self.dino_loss(z, z_t_b, update_center=False)
            for z in z_s
        ) / 8.0
        self.dino_loss.update_center(torch.cat((z_t_a, z_t_b), dim=0))

        # iBOT
        P = self.student_pool.patch_num if hasattr(self.student_pool, 'patch_num') else h_t_a_patch.shape[1]
        num_masked = P // 2
        mask_idx = torch.randperm(P, device=x.device)[:num_masked]
        mask_bool = torch.zeros(P, device=x.device, dtype=torch.bool)
        mask_bool[mask_idx] = True

        h_s_emb = self.student_embedding(views["global_a"])
        h_s_masked = h_s_emb.clone()
        mtok = self.mask_token.expand(B, -1, -1).expand(-1, num_masked, -1)
        h_s_masked[:, mask_bool, :] = mtok[:, :h_s_masked[:, mask_bool, :].shape[1], :]
        h_s_masked = self.student_encoder(h_s_masked)

        pred = F.normalize(self.ibot_head(h_s_masked[:, mask_bool, :]), dim=-1)
        target = F.normalize(h_t_a_patch[:, mask_bool, :].detach(), dim=-1)
        ibot_loss = F.mse_loss(pred, target)

        # DKoleo
        dkoleo_loss = self.dkoleo_loss(self.student_proj(h_global))

        # OrthogonalMask -> z_causal, z_spurious
        z_causal, z_spurious = self.mask(h_global)
        logits = self.classifier(z_causal)
        base_loss = F.cross_entropy(logits, y)

        # Dual perturbation
        warm = min(1.0, max(0.0, (epoch + 1) / max(warmup_epochs, 1)))
        z_sp_intra = intra_class_perturbation(z_spurious.detach(), y)
        loss_intra = F.cross_entropy(self.classifier(z_causal + z_sp_intra), y)
        z_sp_inter = inter_class_perturbation(z_causal.detach(), y)
        loss_inter = F.cross_entropy(self.classifier(z_causal + z_sp_inter), y)
        perturb_loss = warm * (self.lambda_intra * loss_intra + (1 - self.lambda_intra) * loss_inter)

        # Prototype loss
        proto_loss = self.prototypes(z_causal).mean()

        total = dino_loss + ibot_loss + 0.1 * dkoleo_loss + 0.5 * base_loss + 0.5 * perturb_loss + 0.1 * proto_loss

        return total, {
            "dino": dino_loss, "ibot": ibot_loss, "dkoleo": dkoleo_loss,
            "base": base_loss, "perturb": perturb_loss, "proto": proto_loss,
        }

    def update_prototypes(self, x, y):
        """Update prototypes with teacher embeddings."""
        with torch.no_grad():
            h = self._encode_teacher_flat(x)
            self.prototypes.update(h, y)

    def update_ema(self):
        """Update teacher EMA."""
        self._update_teacher_ema()

    def encode(self, x, **kwargs):
        return self._encode_student_flat(x)

    def forward(self, x, **kwargs):
        z = self._encode_student_flat(x)
        z_causal, z_spurious = self.mask(z)
        logits = self.classifier(z_causal)
        return logits


# ── UnifiedDINODualCD + Transformer (Original) ───────────────────────────────

class UnifiedDINODualCD_Transformer(_BaseDualCDModel):
    """Original: DINO + DualCD + Transformer backbone + flatten pooling."""

    def __init__(self, config, d_model=128, n_layers=6, n_heads=8,
                 d_ff=256, dropout=0.1, lambda_intra=0.5, dino_out_dim=256,
                 proto_k=5, teacher_momentum=0.996):
        super().__init__(config, d_model, lambda_intra, dino_out_dim, proto_k, teacher_momentum)
        c_in = config.num_channels
        seq_len = config.seq_len
        patch_num = (seq_len - 28) // 2 + 1
        self.feature_dim = patch_num * d_model

        self.student_embedding = ShallowNetEmbedding(c_in, d_model, dropout)
        self.student_encoder = TransformerERPEncoder(d_model, n_layers, n_heads, d_ff, dropout)
        self.student_pool = FlattenPooling(d_model, patch_num)
        self.student_pool.patch_num = patch_num

        self.teacher_embedding = ShallowNetEmbedding(c_in, d_model, dropout)
        self.teacher_encoder = TransformerERPEncoder(d_model, n_layers, n_heads, d_ff, dropout)
        self.teacher_pool = FlattenPooling(d_model, patch_num)

        self._init_teacher()
        self._init_heads(dino_out_dim, proto_k)

    def _init_teacher(self):
        self.teacher_embedding.load_state_dict(self.student_embedding.state_dict())
        self.teacher_encoder.load_state_dict(self.student_encoder.state_dict())
        for p in self.teacher_embedding.parameters(): p.requires_grad = False
        for p in self.teacher_encoder.parameters(): p.requires_grad = False

    def _init_heads(self, dino_out_dim, proto_k):
        self.student_proj = ProjectionHead(self.feature_dim, 512, dino_out_dim)
        self.teacher_proj = ProjectionHead(self.feature_dim, 512, dino_out_dim)
        self.teacher_proj.load_state_dict(self.student_proj.state_dict())
        for p in self.teacher_proj.parameters(): p.requires_grad = False
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
        self.ibot_head = IBOTHead(self.d_model, self.d_model * 4, self.d_model)
        self.mask = OrthogonalMask(self.d_model)
        self.prototypes = PrototypeBank(self.num_classes, self.feature_dim, k=proto_k)
        self.classifier = nn.Linear(self.feature_dim, self.num_classes)


# ── UnifiedDINODualCD + S4 + Flatten ─────────────────────────────────────────

class UnifiedDINODualCD_S4_Flatten(_BaseDualCDModel):
    """DINO + DualCD + S4 backbone + flatten pooling."""

    def __init__(self, config, d_model=128, n_layers=4, state_dim=8,
                 d_ff=256, dropout=0.1, lambda_intra=0.5, dino_out_dim=256,
                 proto_k=5, teacher_momentum=0.996):
        super().__init__(config, d_model, lambda_intra, dino_out_dim, proto_k, teacher_momentum)
        c_in = config.num_channels
        seq_len = config.seq_len
        patch_num = (seq_len - 28) // 2 + 1
        self.feature_dim = patch_num * d_model

        self.student_embedding = ShallowNetEmbedding(c_in, d_model, dropout)
        self.student_encoder = S4EncoderCore(d_model, n_layers, state_dim, d_ff, dropout)
        self.student_pool = FlattenPooling(d_model, patch_num)
        self.student_pool.patch_num = patch_num

        self.teacher_embedding = ShallowNetEmbedding(c_in, d_model, dropout)
        self.teacher_encoder = S4EncoderCore(d_model, n_layers, state_dim, d_ff, dropout)
        self.teacher_pool = FlattenPooling(d_model, patch_num)

        self._init_teacher()
        self._init_heads(dino_out_dim, proto_k)

    def _init_teacher(self):
        self.teacher_embedding.load_state_dict(self.student_embedding.state_dict())
        self.teacher_encoder.load_state_dict(self.student_encoder.state_dict())
        for p in self.teacher_embedding.parameters(): p.requires_grad = False
        for p in self.teacher_encoder.parameters(): p.requires_grad = False

    def _init_heads(self, dino_out_dim, proto_k):
        self.student_proj = ProjectionHead(self.feature_dim, 512, dino_out_dim)
        self.teacher_proj = ProjectionHead(self.feature_dim, 512, dino_out_dim)
        self.teacher_proj.load_state_dict(self.student_proj.state_dict())
        for p in self.teacher_proj.parameters(): p.requires_grad = False
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
        self.ibot_head = IBOTHead(self.d_model, self.d_model * 4, self.d_model)
        self.mask = OrthogonalMask(self.d_model)
        self.prototypes = PrototypeBank(self.num_classes, self.feature_dim, k=proto_k)
        self.classifier = nn.Linear(self.feature_dim, self.num_classes)


# ── UnifiedDINODualCD + S4 + Attention Pooling ───────────────────────────────

class UnifiedDINODualCD_S4_Pos(_BaseDualCDModel):
    """DINO + DualCD + S4 backbone + attention pooling."""

    def __init__(self, config, d_model=128, n_layers=4, state_dim=8,
                 d_ff=256, dropout=0.1, lambda_intra=0.5, dino_out_dim=256,
                 proto_k=5, teacher_momentum=0.996):
        super().__init__(config, d_model, lambda_intra, dino_out_dim, proto_k, teacher_momentum)
        c_in = config.num_channels
        seq_len = config.seq_len
        patch_num = (seq_len - 28) // 2 + 1
        self.feature_dim = d_model

        pool = AttentionPooling(d_model, patch_num)
        pool.patch_num = patch_num

        self.student_embedding = ShallowNetEmbedding(c_in, d_model, dropout)
        self.student_encoder = S4EncoderCore(d_model, n_layers, state_dim, d_ff, dropout)
        self.student_pool = pool

        self.teacher_embedding = ShallowNetEmbedding(c_in, d_model, dropout)
        self.teacher_encoder = S4EncoderCore(d_model, n_layers, state_dim, d_ff, dropout)
        self.teacher_pool = AttentionPooling(d_model, patch_num)
        self.teacher_pool.patch_num = patch_num

        self._init_teacher()
        self._init_heads(dino_out_dim, proto_k)

    def _init_teacher(self):
        self.teacher_embedding.load_state_dict(self.student_embedding.state_dict())
        self.teacher_encoder.load_state_dict(self.student_encoder.state_dict())
        self.teacher_pool.load_state_dict(self.student_pool.state_dict())
        for p in self.teacher_embedding.parameters(): p.requires_grad = False
        for p in self.teacher_encoder.parameters(): p.requires_grad = False
        for p in self.teacher_pool.parameters(): p.requires_grad = False

    def _init_heads(self, dino_out_dim, proto_k):
        self.student_proj = ProjectionHead(self.feature_dim, 512, dino_out_dim)
        self.teacher_proj = ProjectionHead(self.feature_dim, 512, dino_out_dim)
        self.teacher_proj.load_state_dict(self.student_proj.state_dict())
        for p in self.teacher_proj.parameters(): p.requires_grad = False
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
        self.ibot_head = IBOTHead(self.d_model, self.d_model * 4, self.d_model)
        self.mask = OrthogonalMask(self.d_model)
        self.prototypes = PrototypeBank(self.num_classes, self.feature_dim, k=proto_k)
        self.classifier = nn.Linear(self.feature_dim, self.num_classes)


# ── UnifiedDINODualCD + S4 + Temporal Binned Pooling ─────────────────────────

class UnifiedDINODualCD_S4_Timepatch(_BaseDualCDModel):
    """DINO + DualCD + S4 backbone + temporal binned pooling."""

    def __init__(self, config, d_model=128, n_layers=4, state_dim=8,
                 d_ff=256, dropout=0.1, lambda_intra=0.5, dino_out_dim=256,
                 proto_k=5, teacher_momentum=0.996,
                 bin_boundaries_ms=None, use_std=True):
        super().__init__(config, d_model, lambda_intra, dino_out_dim, proto_k, teacher_momentum)
        c_in = config.num_channels
        seq_len = config.seq_len
        sfreq = getattr(config, "sampling_rate", 200.0)
        patch_num = (seq_len - 28) // 2 + 1

        pool = TemporalBinnedPooling(d_model, patch_num, seq_len, sfreq, bin_boundaries_ms, use_std)
        pool.patch_num = patch_num
        self.feature_dim = pool.out_dim

        self.student_embedding = ShallowNetEmbedding(c_in, d_model, dropout)
        self.student_encoder = S4EncoderCore(d_model, n_layers, state_dim, d_ff, dropout)
        self.student_pool = pool

        self.teacher_embedding = ShallowNetEmbedding(c_in, d_model, dropout)
        self.teacher_encoder = S4EncoderCore(d_model, n_layers, state_dim, d_ff, dropout)
        self.teacher_pool = TemporalBinnedPooling(d_model, patch_num, seq_len, sfreq, bin_boundaries_ms, use_std)
        self.teacher_pool.patch_num = patch_num

        self._init_teacher()
        self._init_heads(dino_out_dim, proto_k)

    def _init_teacher(self):
        self.teacher_embedding.load_state_dict(self.student_embedding.state_dict())
        self.teacher_encoder.load_state_dict(self.student_encoder.state_dict())
        for p in self.teacher_embedding.parameters(): p.requires_grad = False
        for p in self.teacher_encoder.parameters(): p.requires_grad = False

    def _init_heads(self, dino_out_dim, proto_k):
        self.student_proj = ProjectionHead(self.feature_dim, 512, dino_out_dim)
        self.teacher_proj = ProjectionHead(self.feature_dim, 512, dino_out_dim)
        self.teacher_proj.load_state_dict(self.student_proj.state_dict())
        for p in self.teacher_proj.parameters(): p.requires_grad = False
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
        self.ibot_head = IBOTHead(self.d_model, self.d_model * 4, self.d_model)
        self.mask = OrthogonalMask(self.d_model)
        self.prototypes = PrototypeBank(self.num_classes, self.feature_dim, k=proto_k)
        self.classifier = nn.Linear(self.feature_dim, self.num_classes)


# ── S4ERP Standalone (no DualCD) ─────────────────────────────────────────────

class S4ERP(nn.Module):
    """Standalone S4 model for supervised classification. No DINO/DualCD."""

    def __init__(self, config, d_model=128, n_layers=4, state_dim=8,
                 d_ff=256, dropout=0.1):
        super().__init__()
        self.patch_num = (config.seq_len - 28) // 2 + 1
        self.embedding = ShallowNetEmbedding(config.num_channels, d_model, dropout)
        self.encoder = S4EncoderCore(d_model, n_layers, state_dim, d_ff, dropout)
        self.classifier = nn.Sequential(
            nn.GELU(), nn.Dropout(dropout), nn.Flatten(),
            nn.Linear(d_model * self.patch_num, config.num_classes),
        )

    def forward(self, x, **kwargs):
        z = self.embedding(x)
        z = self.encoder(z)
        return {"logits": self.classifier(z), "z_inv": z.reshape(z.size(0), -1)}

    def encode(self, x, **kwargs):
        z = self.embedding(x)
        z = self.encoder(z)
        return z.reshape(z.size(0), -1)
