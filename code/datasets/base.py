"""Dataset abstractions for all MI-EEG datasets.

Every dataset adapter returns a session-indexed view so experiments can be reused
across WBCIC-SHU, SHU 2022, and future datasets without rewriting protocols.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Session:
    """Standard in-memory representation of one subject/session."""

    subject_id: str
    session_id: str
    X: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    npz_path: Optional[Path] = None
    n_channels: int = 0
    n_timepoints: int = 0
    sfreq: float = 250.0
    ch_names: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)

    def has_arrays(self) -> bool:
        return self.X is not None and self.y is not None


class BaseDataset:
    """Base class for dataset adapters.

    Subclasses should expose metadata without touching raw files when possible.
    Heavy preprocessing must be a separate explicit experiment, never hidden inside
    `load()`.
    """

    def __init__(self, config: dict):
        self.config = dict(config or {})
        self.name = self.config["name"]
        self.data_dir = Path(self.config["data_dir"])
        self.n_channels = int(self.config["n_channels"])
        self.n_classes = int(self.config.get("n_classes", 2))
        self.sfreq = float(self.config.get("sfreq", 250.0))
        self.channels = list(self.config.get("channels", []))

    def load(self) -> Dict[Tuple[str, str], Session]:
        raise NotImplementedError

    def get_subjects(self) -> List[str]:
        return sorted({sub for sub, _ in self.load().keys()})

    def describe(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "data_dir": str(self.data_dir),
            "n_channels": self.n_channels,
            "n_classes": self.n_classes,
            "sfreq": self.sfreq,
        }
