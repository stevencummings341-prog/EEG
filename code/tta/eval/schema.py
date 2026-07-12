"""Result-row schema for Phase 3 TTA (aligned with project snake_case)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

RESULT_COLUMNS: List[str] = [
    "dataset",
    "model_adapter",
    "method",
    "seed",
    "subject",
    "source_session",
    "target_session",
    "cell_id",
    "feature_source",
    "n_source",
    "n_target",
    "acc",
    "balanced_acc",
    "delta_vs_no_tta",
    "negative_transfer",
    "used_target_labels",
    "oracle_diagnostic_only",
    "not_deployable",
    "geometry",
    "filter_k",
    "initialization",
    "npz_path_resolved",
    "failure_reason",
]


def empty_result_row(**overrides: Any) -> Dict[str, Any]:
    row = {c: "" for c in RESULT_COLUMNS}
    row.update(
        {
            "acc": float("nan"),
            "balanced_acc": float("nan"),
            "delta_vs_no_tta": float("nan"),
            "negative_transfer": False,
            "used_target_labels": False,
            "oracle_diagnostic_only": False,
            "not_deployable": False,
            "n_source": 0,
            "n_target": 0,
            "filter_k": "",
        }
    )
    row.update(overrides)
    return row


def validate_result_row(row: Dict[str, Any]) -> List[str]:
    """Return list of missing required columns (empty if OK)."""
    return [c for c in RESULT_COLUMNS if c not in row]
