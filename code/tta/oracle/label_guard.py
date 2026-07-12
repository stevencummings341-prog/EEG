"""Label-leakage guards for label-free vs Oracle paths."""

from __future__ import annotations

from typing import Any, Optional

from code.tta.exceptions import LabelLeakageError
from code.tta.feature_sources.base import FeatureBundle
from code.tta.methods.base import MethodResult, TTAMethod


def assert_label_free_bundle(
    bundle: FeatureBundle,
    *,
    method_name: str,
    y_true_used: bool = False,
) -> None:
    """Fail if a label-free method is about to consume target true labels."""
    if y_true_used:
        raise LabelLeakageError(
            f"label-free method '{method_name}' must not use target true labels "
            f"(cell_id={bundle.cell_id})."
        )


def assert_method_result_flags(result: MethodResult, *, expect_oracle: bool) -> None:
    if expect_oracle:
        if not result.used_target_labels:
            raise LabelLeakageError(
                f"Oracle result '{result.method}' missing used_target_labels=True"
            )
        if not result.oracle_diagnostic_only or not result.not_deployable:
            raise LabelLeakageError(
                f"Oracle result '{result.method}' must set "
                "oracle_diagnostic_only=True and not_deployable=True"
            )
    else:
        if result.used_target_labels:
            raise LabelLeakageError(
                f"label-free result '{result.method}' has used_target_labels=True"
            )


def run_label_free(method: TTAMethod, bundle: FeatureBundle, **kwargs) -> MethodResult:
    """Execute a label-free method with leakage checks.

    Interface-level guarantee (not just convention): ``target_y_true`` is
    stripped from the bundle **before** ``method.run`` is ever called, via
    ``bundle.freeze_for_label_free()``. Methods that forget to call
    ``self._prepare_label_free`` internally (or that read
    ``bundle.target_y_true`` directly) still cannot observe target labels,
    because the guard already removed them upstream of ``method.run``.
    """
    if getattr(method, "uses_target_labels", False):
        raise LabelLeakageError(
            f"method '{method.name}' declares uses_target_labels=True; "
            "use the Oracle path instead."
        )
    # Detect misuse: if caller passes y_true explicitly for adaptation.
    if kwargs.pop("target_y_true", None) is not None:
        raise LabelLeakageError(
            f"label-free method '{method.name}' received target_y_true kwarg."
        )
    if kwargs.pop("use_target_labels", False):
        raise LabelLeakageError(
            f"label-free method '{method.name}' received use_target_labels=True."
        )
    safe_bundle = bundle.freeze_for_label_free()
    result = method.run(safe_bundle, **kwargs)
    assert_method_result_flags(result, expect_oracle=False)
    return result


def run_oracle(method: Any, bundle: FeatureBundle, **kwargs) -> MethodResult:
    """Execute an Oracle diagnostic with mandatory leakage flags."""
    if bundle.target_y_true is None:
        raise LabelLeakageError(
            f"Oracle '{getattr(method, 'name', method)}' needs target_y_true "
            f"on cell {bundle.cell_id}"
        )
    result = method.run(bundle, **kwargs)
    assert_method_result_flags(result, expect_oracle=True)
    return result
