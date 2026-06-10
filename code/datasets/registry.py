"""Dataset registry for the modular framework."""
from __future__ import annotations

from typing import Dict, Type

from .base import BaseDataset
from .shu import SHUDataset
from .wbci_shu import WBCICSHUDataset

DATASETS: Dict[str, Type[BaseDataset]] = {
    "shu": SHUDataset,
    "wbci_shu": WBCICSHUDataset,
    "wbcic_shu": WBCICSHUDataset,
}


def build_dataset(config: dict) -> BaseDataset:
    """Build a dataset adapter by `config.name`."""
    name = str(config.get("name", "")).lower().strip()
    if name not in DATASETS:
        raise ValueError(f"unknown dataset '{name}'. Known datasets: {sorted(DATASETS)}")
    return DATASETS[name](config)
