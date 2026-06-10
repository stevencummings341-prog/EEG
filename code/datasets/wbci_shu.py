"""WBCIC-SHU 2025 dataset adapter.

Formal training entry: preprocessed per-session `.npz` files listed in the
processed manifest. Raw BDF and derivative MAT files remain external read-only
references.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .base import BaseDataset, Session


class WBCICSHUDataset(BaseDataset):
    """Adapter for WBCIC-SHU 2C processed `.npz` sessions."""

    def load(self) -> Dict[Tuple[str, str], Session]:
        manifest = Path(self.config.get("manifest", ""))
        if not manifest.exists():
            raise FileNotFoundError(f"WBCIC manifest not found: {manifest}")
        status_filter = set(self.config.get("status_filter", ["ok"]))
        sessions: Dict[Tuple[str, str], Session] = {}
        with manifest.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if status_filter and row.get("status") not in status_filter:
                    continue
                npz_path = Path(row.get("npz_path", ""))
                if not npz_path.exists():
                    continue
                subject = row["subject_id"]
                session = row["session_id"]
                sessions[(subject, session)] = Session(
                    subject_id=subject,
                    session_id=session,
                    npz_path=npz_path,
                    n_channels=self.n_channels,
                    n_timepoints=int(self.config.get("n_timepoints", 1000)),
                    sfreq=self.sfreq,
                    ch_names=self.channels,
                    metadata={"status": row.get("status"), "source": "processed_manifest"},
                )
        if not sessions:
            raise ValueError(f"No WBCIC sessions matched manifest={manifest}")
        return sessions

    @staticmethod
    def load_arrays(session: Session):
        """Load X/y lazily for one WBCIC processed session."""
        if session.npz_path is None:
            raise ValueError("session has no npz_path")
        with np.load(session.npz_path, allow_pickle=False) as data:
            return data["X"].astype(np.float32), data["y"].astype(np.int64)
