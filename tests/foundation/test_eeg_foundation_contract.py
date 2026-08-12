"""Contract tests for the 5 ported EEG foundation models (CPU, tiny shapes).

Checks that the advisor-supplied package really is interchangeable with the EEGNet
family inside this project:

  * the registry can build every variant by name;
  * ``forward([B, C, T])`` returns ``{logits, features, confidence}`` with the right
    shapes for BOTH datasets' channel counts (WBCIC 58ch, SHU 32ch);
  * DualCD variants expose the training hooks and produce a finite, backward-able loss;
  * ``after_optimizer_step`` moves the teacher weights (EMA) and prototypes;
  * per-trial normalization is fit-free (test trials unaffected by other trials).

Shapes are deliberately small (n_times=200) so this stays a fast CPU test. The real
parameter counts at n_times=1000 are asserted only for the cheap variant.

Run: python -m pytest tests/foundation/test_eeg_foundation_contract.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from code.models.eeg_foundation import VARIANT_NAMES, normalize_trials  # noqa: E402
from code.models.registry import build_model  # noqa: E402

DIMS = dict(n_times=200, n_classes=2, sfreq=250)
CHANNELS = [58, 32]  # WBCIC-SHU, SHU 2022


def _params(variant: str) -> dict:
    """Small-but-valid structural params (timepatch needs trial-length bins)."""
    p = {"d_model": 16, "n_layers": 1, "d_ff": 32, "dino_out_dim": 8, "proto_k": 2}
    if variant == "dualcd_s4_timepatch":
        p["bin_boundaries_ms"] = [0, 200, 400, 800]
    return p


@pytest.mark.parametrize("variant", VARIANT_NAMES)
@pytest.mark.parametrize("n_channels", CHANNELS)
def test_forward_contract(variant: str, n_channels: int):
    torch.manual_seed(0)
    model = build_model(variant, n_channels=n_channels, params=_params(variant), **DIMS)
    x = torch.randn(4, n_channels, DIMS["n_times"])

    model.eval()
    with torch.no_grad():
        out = model(x)

    assert set(out) == {"logits", "features", "confidence"}
    assert out["logits"].shape == (4, DIMS["n_classes"])
    assert out["features"].shape == (4, model.feature_dim)
    assert out["confidence"] is None
    assert torch.isfinite(out["logits"]).all()

    # 4D [B,1,C,T] must give identical logits to 3D [B,C,T].
    with torch.no_grad():
        out4 = model(x.unsqueeze(1))
    assert torch.allclose(out["logits"], out4["logits"], atol=1e-6)


@pytest.mark.parametrize("variant", VARIANT_NAMES)
def test_training_step_is_finite_and_differentiable(variant: str):
    torch.manual_seed(0)
    model = build_model(variant, n_channels=32, params=_params(variant), **DIMS)
    x = torch.randn(6, 32, DIMS["n_times"])
    y = torch.tensor([0, 1, 0, 1, 0, 1])

    model.train()
    loss, parts = model.training_step(x, y, epoch=0)
    assert torch.isfinite(loss), f"{variant}: non-finite loss"
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads, f"{variant}: no gradients reached any trainable parameter"
    assert all(torch.isfinite(g).all() for g in grads)

    expected = ({"dino", "ibot", "dkoleo", "base", "perturb", "proto"}
                if model.uses_custom_loss else {"ce"})
    assert set(parts) == expected


def test_dualcd_hooks_update_teacher_and_prototypes():
    torch.manual_seed(0)
    model = build_model("dualcd_s4_pos", n_channels=32,
                        params=_params("dualcd_s4_pos"), **DIMS)
    assert model.uses_custom_loss

    x = torch.randn(6, 32, DIMS["n_times"])
    y = torch.tensor([0, 1, 0, 1, 0, 1])
    inner = model.model
    teacher_before = inner.teacher_encoder.layers[0].s4.B.detach().clone()
    protos_before = inner.prototypes.prototypes.detach().clone()

    model.train()
    loss, _ = model.training_step(x, y, epoch=0)
    loss.backward()
    # Give the student a non-trivial step so the EMA has something to pull towards.
    with torch.no_grad():
        for p in inner.student_encoder.parameters():
            if p.grad is not None:
                p.add_(-0.5 * p.grad)
    model.after_optimizer_step(x, y)

    assert not torch.allclose(inner.teacher_encoder.layers[0].s4.B, teacher_before), \
        "teacher EMA did not move"
    assert not torch.allclose(inner.prototypes.prototypes, protos_before), \
        "prototype bank did not update"
    # Teacher must stay frozen w.r.t. autograd.
    assert all(not p.requires_grad for p in inner.teacher_encoder.parameters())


def test_s4erp_has_no_dualcd_hooks_active():
    model = build_model("s4erp", n_channels=32, params=_params("s4erp"), **DIMS)
    assert not model.uses_custom_loss
    x = torch.randn(2, 32, DIMS["n_times"])
    y = torch.tensor([0, 1])
    model.after_optimizer_step(x, y)  # must be a no-op, not an error


def test_normalize_trials_is_fit_free():
    torch.manual_seed(0)
    x = torch.randn(8, 32, DIMS["n_times"]) * 30.0 + 5.0

    z = normalize_trials(x, "per_sample_zscore")
    assert torch.allclose(z.mean(dim=-1), torch.zeros(8, 32), atol=1e-5)
    assert torch.allclose(z.std(dim=-1), torch.ones(8, 32), atol=1e-3)

    # Removing other trials must not change a given trial's normalization: this is
    # what makes it legal to apply on the test split.
    z_single = normalize_trials(x[2:3], "per_sample_zscore")
    assert torch.allclose(z[2:3], z_single, atol=1e-6)

    assert torch.allclose(normalize_trials(x, "none"), x)
    with pytest.raises(ValueError):
        normalize_trials(x, "minmax")


def test_feature_dim_matches_advisor_table_at_full_length():
    """Sanity-check the real MI shapes (32ch, 1000 samples) used for both datasets."""
    model = build_model("dualcd_s4_timepatch", n_channels=32, n_times=1000,
                        n_classes=2, sfreq=250,
                        params={"bin_boundaries_ms": [0, 500, 1000, 1500, 2000, 3000, 4000]})
    # 6 bins x d_model(128) x 2 (mean+std)
    assert model.feature_dim == 6 * 128 * 2
    n_params = sum(p.numel() for p in model.parameters())
    assert 2.5e6 < n_params < 4.5e6, n_params  # advisor table: ~3.3M
