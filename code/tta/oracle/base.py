"""Oracle diagnostic base — always non-deployable when using target labels."""

from __future__ import annotations

from abc import ABC, abstractmethod

from code.tta.feature_sources.base import FeatureBundle
from code.tta.methods.base import MethodResult


class OracleMethod(ABC):
    """Offline upper-bound diagnostics only. Never treat as deployable TTA."""

    name: str = "oracle_base"
    uses_target_labels: bool = True

    @abstractmethod
    def run(self, bundle: FeatureBundle, **kwargs) -> MethodResult:
        ...

    def _tagged(self, result: MethodResult) -> MethodResult:
        result.used_target_labels = True
        result.oracle_diagnostic_only = True
        result.not_deployable = True
        return result


# Provisional decision notes (NOT hard rules):
# After pretrained-model integration, revisit these thresholds.
#   * Oracle mean improvement > +3pp  -> may justify expanding T3A / safe-T3A
#   * Oracle mean improvement < +1pp  -> large-scale T3A may not be worthwhile
#   * Between +1pp and +3pp           -> cautious small-scale validation only
# Also apply risk checks (negative transfer, Fisher recovery) per PHASE3_ROUTE_PLAN.
PROVISIONAL_ORACLE_NOTES = (
    "Current Oracle thresholds (+3pp / +1pp) are provisional and should be "
    "revisited after pretrained model integration."
)
