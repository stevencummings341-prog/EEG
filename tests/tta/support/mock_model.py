"""Tiny deterministic PyTorch model — **TEST FIXTURE ONLY**.

This is NOT a research model and must never be imported by production
``code/tta`` or ``code/models`` modules. It exists solely so tests can
exercise the real ``ModelInferenceSource`` -> ``FeatureBundle`` -> TTA method
-> evaluator path end-to-end without depending on the (slow, larger) project
baseline models. Configurable channel/time/feature/class dims keep it CPU-only
and fast; weights are seeded for determinism.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union

import torch
import torch.nn as nn

PathLike = Union[str, Path]


@dataclass
class MockModelConfig:
    n_channels: int = 6
    n_times: int = 32
    n_classes: int = 2
    feature_dim: int = 10


class MockEEGModel(nn.Module):
    """Deterministic linear encoder + classifier — TEST FIXTURE ONLY.

    ``forward(x)`` accepts ``[B, C, T]`` (or ``[C, T]`` for a single sample)
    and returns ``{"features": [B, feature_dim], "logits": [B, n_classes],
    "confidence": None}``, mirroring the shared contract documented in
    ``code/models/registry.py`` closely enough for adapter-layer tests
    without importing any production model.
    """

    def __init__(self, cfg: MockModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.feature_dim = cfg.feature_dim
        in_dim = cfg.n_channels * cfg.n_times
        self.encoder = nn.Linear(in_dim, cfg.feature_dim)
        self.classifier = nn.Linear(cfg.feature_dim, cfg.n_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        flat = x.reshape(x.shape[0], -1).float()
        features = torch.tanh(self.encoder(flat))
        logits = self.classifier(features)
        return {"features": features, "logits": logits, "confidence": None}


def build_mock_model(cfg: MockModelConfig, *, seed: int = 0) -> MockEEGModel:
    """Construct a ``MockEEGModel`` with deterministic weights for ``seed``."""
    gen = torch.Generator().manual_seed(int(seed))
    model = MockEEGModel(cfg)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.empty(p.shape).uniform_(-0.5, 0.5, generator=gen))
    model.eval()
    return model


def save_mock_checkpoint(model: MockEEGModel, path: PathLike) -> None:
    """Save a checkpoint shaped like real project checkpoints.

    Uses the same ``{"model_state_dict": ...}`` wrapper that
    ``BaselineTorchAdapter.load_checkpoint`` (and the mock adapter) expect.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, p)
