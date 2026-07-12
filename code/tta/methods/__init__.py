"""TTA methods (Round-1: no_tta + minimal T3A)."""

from code.tta.methods.base import MethodResult, TTAMethod
from code.tta.methods.no_tta import NoTTAMethod
from code.tta.methods.t3a_minimal import MinimalT3AMethod

__all__ = ["MethodResult", "TTAMethod", "NoTTAMethod", "MinimalT3AMethod"]
