"""SHU 2022 dataset adapter.

External SHU raw root comes from dataset / paths config (`data_dir`), never
hard-coded. That tree is read-only. This adapter indexes BIDS-like EDF/MAT/events
files and exposes their metadata; converting raw sessions into standardized
`.npz` must be done by an explicit preprocessing experiment.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Tuple

from .base import BaseDataset, Session

_PATTERN = re.compile(r"^(sub-\d+)_+(ses-\d+)_task_motorimagery_(eeg|events)\.(edf|mat|tsv)$")


class SHUDataset(BaseDataset):
    """Read-only index for Ma et al. SHU motor imagery dataset."""

    def load(self) -> Dict[Tuple[str, str], Session]:
        if not self.data_dir.exists():
            raise FileNotFoundError(f"SHU data_dir not found: {self.data_dir}")
        grouped: Dict[Tuple[str, str], Dict[str, Path]] = {}
        for path in self.data_dir.iterdir():
            m = _PATTERN.match(path.name)
            if not m:
                continue
            subject, session, kind, ext = m.groups()
            grouped.setdefault((subject, session), {})[f"{kind}_{ext}"] = path
        sessions: Dict[Tuple[str, str], Session] = {}
        for (subject, session), files in sorted(grouped.items()):
            sessions[(subject, session)] = Session(
                subject_id=subject,
                session_id=session,
                n_channels=self.n_channels,
                n_timepoints=int(self.config.get("n_timepoints", 1000)),
                sfreq=self.sfreq,
                ch_names=self.channels,
                metadata={"files": {k: str(v) for k, v in sorted(files.items())}},
            )
        if not sessions:
            raise ValueError(f"No SHU sessions found under {self.data_dir}")
        return sessions
