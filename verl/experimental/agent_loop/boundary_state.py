"""Recoverable boundary-state contract for tool-agent trajectories.

An event/segment identifier is only a reference.  A science harness must
provide durable state artifacts before a boundary can be used for a restore or
fresh-branch experiment.  This module is intentionally independent from a
specific provider (OpenAI4S, a notebook kernel, or a local tool runtime).
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from collections.abc import Mapping
from typing import Any

SCHEMA_VERSION = "polyseg-boundary-state-v1"
_REQUIRED = (
    "boundary_id", "snapshot_id", "trajectory_id", "session_id", "prefix_token_ids",
    "action_mask", "tool_ledger", "boundary_state", "boundary_hidden", "kernel_state",
    "workspace_hashes", "artifact_hashes", "dependency_hashes", "rng_state",
    "execution_budget_initial", "execution_budget_remaining", "policy_version",
    "tokenizer_version", "sampling_config",
)
_STATE_FIELDS = ("boundary_state", "boundary_hidden", "kernel_state")


class BoundaryStateContractError(ValueError):
    """Raised when a provider returns a non-recoverable boundary projection."""


class BoundaryStateProviderError(ValueError):
    """Raised when a configured boundary-state provider cannot be loaded."""


def _json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (tuple, list)):
        return [_json(v) for v in value]
    if isinstance(value, set):
        return sorted((_json(v) for v in value), key=lambda item: json.dumps(item, sort_keys=True))
    if hasattr(value, "tolist"):
        return _json(value.tolist())
    if hasattr(value, "item"):
        try:
            return _json(value.item())
        except (TypeError, ValueError):
            pass
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(_json(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _config_value(source: Any, name: str, default: Any = None) -> Any:
    if source is None:
        return default
    try:
        value = source.get(name, default)
    except (AttributeError, TypeError):
        value = getattr(source, name, default)
    return default if value is None else value


def _rollout_config(config: Any) -> Any:
    actor_rollout_ref = _config_value(config, "actor_rollout_ref", None)
    rollout = _config_value(actor_rollout_ref, "rollout", None)
    return rollout if rollout is not None else config


def boundary_snapshots_required(config: Any) -> bool:
    """Return whether this rollout must provide recoverable snapshots."""

    return bool(_config_value(_rollout_config(config), "polyseg_require_boundary_snapshots", False))


def load_boundary_state_provider(config: Any) -> Any | None:
    """Load the harness-owned provider configured for tool-agent rollouts.

    A provider is intentionally imported in the worker process so it can hold
    runtime handles to the session/CAS/kernel service.  The provider factory
    receives the full trainer config and must expose ``capture`` or
    ``capture_boundary_state``.  Required snapshot runs fail during worker
    construction when no provider path is configured.
    """

    rollout = _rollout_config(config)
    path = str(_config_value(rollout, "polyseg_boundary_state_provider_path", "") or "").strip()
    required = boundary_snapshots_required(config)
    if not path:
        if required:
            raise BoundaryStateProviderError(
                "polyseg_require_boundary_snapshots=true but "
                "polyseg_boundary_state_provider_path is empty"
            )
        return None
    module_name, separator, attribute = path.partition(":")
    if not separator:
        module_name, _, attribute = path.rpartition(".")
    if not module_name or not attribute:
        raise BoundaryStateProviderError(
            "polyseg_boundary_state_provider_path must be module:attribute or module.attribute"
        )
    try:
        module = importlib.import_module(module_name)
        factory = getattr(module, attribute)
    except (ImportError, AttributeError) as exc:
        raise BoundaryStateProviderError(f"boundary-state provider {path!r} was not found") from exc
    provider = factory(config) if callable(factory) else factory
    if not callable(getattr(provider, "capture_boundary_state", None)) and not callable(
        getattr(provider, "capture", None)
    ):
        raise BoundaryStateProviderError(
            f"boundary-state provider {path!r} must expose capture_boundary_state or capture"
        )
    return provider


def _artifact(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BoundaryStateContractError(f"{name}_missing_or_not_mapping")
    reference, sha = value.get("artifact_ref"), value.get("sha256")
    if not isinstance(reference, str) or not reference:
        raise BoundaryStateContractError(f"{name}_artifact_ref_missing")
    if not isinstance(sha, str) or len(sha) != 64 or any(char not in "0123456789abcdef" for char in sha.lower()):
        raise BoundaryStateContractError(f"{name}_sha256_invalid")
    if value.get("recoverable") is not True:
        raise BoundaryStateContractError(f"{name}_not_recoverable")
    return {"artifact_ref": reference, "sha256": sha.lower(), "recoverable": True}


def validate_boundary_snapshot(raw: Mapping[str, Any], *, require_restore_audit: bool = False) -> dict[str, Any]:
    """Validate and normalize one provider-owned recoverable snapshot.

    State IDs alone are rejected: all three runtime projections need a durable
    artifact reference, content hash, and explicit recoverability attestation.
    """

    if not isinstance(raw, Mapping):
        raise BoundaryStateContractError("snapshot_not_mapping")
    result = dict(raw)
    if result.get("schema_version") != SCHEMA_VERSION:
        raise BoundaryStateContractError("schema_version_mismatch")
    missing = [key for key in _REQUIRED if key not in result]
    if missing:
        raise BoundaryStateContractError("missing_fields:" + ",".join(missing))
    prefix, mask = result["prefix_token_ids"], result["action_mask"]
    if not isinstance(prefix, list) or any(type(item) is not int or item < 0 for item in prefix):
        raise BoundaryStateContractError("invalid_prefix_token_ids")
    if not isinstance(mask, list) or len(mask) != len(prefix) or any(
        type(item) not in (bool, int) or int(item) not in (0, 1) for item in mask
    ):
        raise BoundaryStateContractError("invalid_action_mask")
    for key in _STATE_FIELDS:
        result[key] = _artifact(result[key], key)
    for key in ("workspace_hashes", "artifact_hashes", "dependency_hashes", "sampling_config", "tool_ledger"):
        if not isinstance(result[key], Mapping):
            raise BoundaryStateContractError(f"{key}_not_mapping")
    if not isinstance(result["rng_state"], str) or not result["rng_state"]:
        raise BoundaryStateContractError("rng_state_missing")
    try:
        initial = float(result["execution_budget_initial"])
        remaining = float(result["execution_budget_remaining"])
    except (TypeError, ValueError) as exc:
        raise BoundaryStateContractError("invalid_execution_budget") from exc
    if not all(math.isfinite(value) for value in (initial, remaining)) or initial < 0 or remaining < 0 or remaining > initial:
        raise BoundaryStateContractError("invalid_execution_budget")
    payload = dict(result)
    payload.pop("captured_state_digest", None)
    expected = content_hash(payload)
    if result.get("captured_state_digest") != expected:
        raise BoundaryStateContractError("captured_state_digest_mismatch")
    if require_restore_audit:
        restore_audit = result.get("restore_audit")
        if not isinstance(restore_audit, Mapping) or restore_audit.get("status") != "passed":
            raise BoundaryStateContractError("restore_audit_missing_or_failed")
    result["captured_state_digest"] = expected
    return result


def audit_restore(snapshot: Mapping[str, Any], restored: Mapping[str, Any] | None) -> dict[str, Any]:
    """Compare a restore result with the captured snapshot, fail closed."""

    reasons: list[str] = []
    try:
        expected = validate_boundary_snapshot(snapshot)
    except BoundaryStateContractError as exc:
        return {"status": "failed", "snapshot_digest": None, "restored_digest": None, "reasons": [str(exc)]}
    if restored is None:
        reasons.append("restore_returned_none")
    else:
        try:
            actual = validate_boundary_snapshot(restored)
        except BoundaryStateContractError as exc:
            actual = None
            reasons.append(str(exc))
        if actual is not None:
            for key in _REQUIRED + ("captured_state_digest",):
                if canonical_json(actual.get(key)) != canonical_json(expected.get(key)):
                    reasons.append(f"{key}_mismatch")
    return {
        "status": "passed" if not reasons else "failed",
        "snapshot_digest": expected["captured_state_digest"],
        "restored_digest": None if restored is None else content_hash(restored),
        "reasons": sorted(set(reasons)),
    }


__all__ = [
    "SCHEMA_VERSION",
    "BoundaryStateContractError",
    "BoundaryStateProviderError",
    "canonical_json",
    "content_hash",
    "boundary_snapshots_required",
    "load_boundary_state_provider",
    "validate_boundary_snapshot",
    "audit_restore",
]
