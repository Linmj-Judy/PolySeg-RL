"""Shared validation and failure accounting for PolySeg feedback calls.

A scorer may raise FeedbackQueryError with an authoritative billed cost. For
an ordinary exception, cancellation or malformed response the cost is unknown:
the reserved upper bound remains charged, rather than inventing a free query.
"""

from __future__ import annotations

import math
from typing import Any


def is_validation_batch(data: Any) -> bool:
    """Read the trainer-owned phase through DataProto or TensorDict transport."""
    import numpy as np

    if hasattr(data, "meta_info"):
        value = data.meta_info.get("validate", data.non_tensor_batch.get("validate", False))
    else:
        from verl.utils.tensordict_utils import get_non_tensor_data

        value = get_non_tensor_data(data, "validate", False)
    values = np.asarray(value).reshape(-1)
    if values.size == 0:
        return False
    if any(not isinstance(v, (bool, np.bool_)) for v in values):
        raise ValueError("validate must contain boolean phase markers")
    if not np.all(values == values[0]):
        raise ValueError("a reward batch cannot mix training and validation phases")
    return bool(values[0])


class FeedbackQueryError(RuntimeError):
    """A failed feedback request with a known non-negative billed query cost."""

    def __init__(self, message: str, *, query_cost: float):
        self.query_cost = finite_nonnegative(query_cost, "query_cost")
        super().__init__(message)


def finite_nonnegative(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative number") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


def validate_feedback_cost(info: dict[str, Any], reserved_cost: float) -> float:
    cost = finite_nonnegative(info.get("query_cost", reserved_cost), "query_cost")
    if cost > reserved_cost + 1e-8:
        raise ValueError("PolySeg scorer query_cost exceeds reserved segment_cost")
    return cost


def failed_feedback_cost(error: BaseException, reserved_cost: float) -> tuple[float, str]:
    """Return conservative cost and its provenance for an unsuccessful call."""
    if isinstance(error, FeedbackQueryError) and error.query_cost <= reserved_cost + 1e-8:
        return error.query_cost, "reported_failure_cost"
    return reserved_cost, "unknown_cost_reserved_upper_bound"


def feedback_value(result: Any, protocol: str) -> tuple[float, dict[str, Any]]:
    potential = protocol in {"utility_delta", "prm_potential"}
    if isinstance(result, dict):
        key = "utility" if potential else "score" if "score" in result else "utility"
        if key not in result:
            raise KeyError(f"PolySeg {protocol} scorer must return '{key}'")
        info = dict(result)
        value = float(result[key])
    else:
        if potential:
            raise TypeError("PolySeg utility/potential scorer must return {'utility': ...}; a scalar is ambiguous")
        value = float(result)
        info = {"score": value}
    if not math.isfinite(value):
        raise ValueError("PolySeg feedback value must be finite")
    return value, info


def final_accuracy(info: dict[str, Any], protocol: str) -> float | None:
    """Missing correctness is not an incorrect answer or a utility estimate."""
    value = info.get("acc", info.get("accuracy"))
    if value is None and protocol in {"terminal", "binary_local"} and "score" in info:
        value = float(info["score"]) > 0.5
    if value is None:
        return None
    accuracy = float(value)
    if not math.isfinite(accuracy) or not 0.0 <= accuracy <= 1.0:
        raise ValueError("final accuracy must be finite and in [0, 1]")
    return accuracy
