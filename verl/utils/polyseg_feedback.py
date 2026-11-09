"""Shared validation and failure accounting for PolySeg feedback calls.

A scorer may raise FeedbackQueryError with an authoritative billed cost. For
an ordinary exception, cancellation or malformed response the cost is unknown:
the reserved upper bound remains charged, rather than inventing a free query.
"""

from __future__ import annotations

import math
from typing import Any


BILLING_RECEIPT_TOLERANCE = 1e-8


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


def validate_billing_receipt(
    receipt: Any,
    *,
    reported_cost: Any,
    reserved_cost: Any,
) -> tuple[dict[str, Any] | None, str]:
    """Validate the provider evidence needed for a lower-cost settlement.

    A scorer-controlled ``query_cost`` is not an accounting receipt.  A lower
    bill is accepted only when the provider supplies a non-empty stable query
    id and a finite ``billed_cost`` that agrees with the scorer's reported
    cost and fits inside the reservation.  Invalid or missing receipts are
    deliberately represented as a status instead of raising: callers charge
    the conservative reservation upper bound and retain the audit reason.
    """
    if receipt is None:
        return None, "missing"
    if not isinstance(receipt, dict):
        return None, "invalid_type"
    provider_query_id = receipt.get("provider_query_id")
    if not isinstance(provider_query_id, str) or not provider_query_id.strip():
        return None, "missing_provider_query_id"
    try:
        billed_cost = finite_nonnegative(receipt.get("billed_cost"), "billing_receipt.billed_cost")
        reported = finite_nonnegative(reported_cost, "query_cost")
        reserved = finite_nonnegative(reserved_cost, "reserved_cost")
    except ValueError:
        return None, "invalid_billed_cost"
    if billed_cost > reserved + BILLING_RECEIPT_TOLERANCE:
        return None, "billed_cost_exceeds_reservation"
    if not math.isclose(
        billed_cost,
        reported,
        rel_tol=0.0,
        abs_tol=BILLING_RECEIPT_TOLERANCE,
    ):
        return None, "billed_cost_mismatch"
    normalized = {
        "provider_query_id": provider_query_id.strip(),
        "billed_cost": billed_cost,
    }
    # Preserve a provider name when supplied, but never copy arbitrary
    # response objects into a durable journal row.
    provider = receipt.get("provider")
    if isinstance(provider, str) and provider.strip():
        normalized["provider"] = provider.strip()
    return normalized, "accepted"


def settle_feedback_cost(
    info: dict[str, Any],
    *,
    reported_cost: float,
    reserved_cost: float,
    policy: str,
) -> float:
    """Return the authoritative billed amount and annotate receipt status."""
    if policy not in {"reserve_upper", "reported_with_receipt"}:
        raise ValueError(f"unsupported feedback cost policy: {policy!r}")
    receipt, status = validate_billing_receipt(
        info.get("billing_receipt"),
        reported_cost=reported_cost,
        reserved_cost=reserved_cost,
    )
    info["billing_receipt_status"] = (
        "accepted" if status == "accepted" and policy == "reported_with_receipt"
        else "valid_not_used" if status == "accepted" else status
    )
    if receipt is not None:
        info["billing_receipt"] = receipt
    if policy == "reported_with_receipt" and receipt is not None:
        return float(receipt["billed_cost"])
    return float(reserved_cost)


def billing_receipt_journal_entries(infos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create JSON-safe per-segment receipt evidence for the feedback journal."""
    entries: list[dict[str, Any]] = []
    for segment_id, info in enumerate(infos):
        if "billing_receipt" not in info and "billing_receipt_status" not in info:
            continue
        receipt = info.get("billing_receipt")
        entry: dict[str, Any] = {
            "segment_id": int(segment_id),
            "status": str(info.get("billing_receipt_status", "missing")),
        }
        if isinstance(receipt, dict):
            query_id = receipt.get("provider_query_id")
            billed = receipt.get("billed_cost")
            if isinstance(query_id, str) and query_id.strip():
                entry["provider_query_id"] = query_id.strip()
            if type(billed) in (int, float) and math.isfinite(float(billed)) and float(billed) >= 0:
                entry["billed_cost"] = float(billed)
            provider = receipt.get("provider")
            if isinstance(provider, str) and provider.strip():
                entry["provider"] = provider.strip()
        reported = info.get("query_cost")
        if type(reported) in (int, float) and math.isfinite(float(reported)) and float(reported) >= 0:
            entry["reported_cost"] = float(reported)
        entries.append(entry)
    return entries


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
