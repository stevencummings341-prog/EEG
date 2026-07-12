"""tests for evaluator schema."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.tta.eval.schema import RESULT_COLUMNS, empty_result_row, validate_result_row


def test_schema_columns_complete():
    row = empty_result_row(failure_reason="unit_test")
    assert validate_result_row(row) == []
    for c in (
        "cell_id",
        "used_target_labels",
        "oracle_diagnostic_only",
        "not_deployable",
        "delta_vs_no_tta",
        "failure_reason",
    ):
        assert c in RESULT_COLUMNS
    assert row["failure_reason"] == "unit_test"


def test_validate_detects_missing():
    bad = {"acc": 0.5}
    missing = validate_result_row(bad)
    assert "cell_id" in missing
    assert "method" in missing
