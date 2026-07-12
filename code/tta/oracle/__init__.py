"""Oracle diagnostic modules (Round-1: minimal target-label proto + guard)."""

from code.tta.oracle.base import PROVISIONAL_ORACLE_NOTES, OracleMethod
from code.tta.oracle.label_guard import (
    assert_label_free_bundle,
    assert_method_result_flags,
    run_label_free,
    run_oracle,
)
from code.tta.oracle.target_label_proto import TargetLabelProtoOracle

__all__ = [
    "PROVISIONAL_ORACLE_NOTES",
    "OracleMethod",
    "TargetLabelProtoOracle",
    "assert_label_free_bundle",
    "assert_method_result_flags",
    "run_label_free",
    "run_oracle",
]
