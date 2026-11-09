"""Durable accounting of completed feedback trajectories, before transport.

This journal counts scorer calls (not suffix samples) and their reported cost.
It is evidence for successful runs; failed attempts retain their own journal
and cannot be promoted to research results. All workers must share the path.
"""

import fcntl
import hashlib
import json
import math
import os
import time
import uuid
from pathlib import Path


def _ledger_lock_path(path: Path) -> Path:
    return path.with_name(path.name + ".lock")


def _read_ledger_locked(path: Path) -> dict:
    """Read one complete shared-ledger state while holding its writer lock."""

    if not path.is_file():
        raise FileNotFoundError(path)
    with _ledger_lock_path(path).open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            with path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    if not isinstance(value, dict):
        raise ValueError("PolySeg feedback ledger must contain a JSON object")
    return value


def snapshot_feedback_ledger(path, destination) -> dict:
    """Persist an atomic checkpoint copy of the shared feedback ledger.

    The snapshot is intentionally a complete state document, including the
    ledger identity and idempotency maps.  It is used only for resume audits;
    the live ledger remains the source of truth for reward workers.
    """

    source = Path(path)
    target = Path(destination)
    state = _read_ledger_locked(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, allow_nan=False, sort_keys=True, separators=(",", ":"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return state


def verify_feedback_ledger_snapshot(path, snapshot_path) -> dict:
    """Require a live ledger to match its checkpoint snapshot exactly."""

    live = _read_ledger_locked(Path(path))
    with Path(snapshot_path).open("r", encoding="utf-8") as handle:
        saved = json.load(handle)
    if not isinstance(saved, dict):
        raise ValueError("PolySeg feedback ledger checkpoint must contain a JSON object")
    # Ignore no fields: reservations/settlements are part of the idempotency
    # contract and silently dropping them can double-charge a resumed run.
    if live != saved:
        raise RuntimeError(
            "live PolySeg feedback ledger differs from checkpoint snapshot; "
            "refusing to resume with ambiguous feedback accounting"
        )
    return live


def restore_feedback_ledger_snapshot(path, snapshot_path) -> dict:
    """Restore the exact ledger state captured by a checkpoint."""

    source = Path(path)
    with Path(snapshot_path).open("r", encoding="utf-8") as handle:
        saved = json.load(handle)
    if not isinstance(saved, dict):
        raise ValueError("PolySeg feedback ledger checkpoint must contain a JSON object")
    required = ("schema_version", "ledger_id", "total", "consumed", "reserved", "reservations", "settled")
    if any(key not in saved for key in required):
        raise ValueError("PolySeg feedback ledger checkpoint is incomplete")
    if not source.is_file():
        raise RuntimeError("live PolySeg feedback ledger is missing while a checkpoint exists")
    with _ledger_lock_path(source).open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            with source.open("r", encoding="utf-8") as handle:
                live = json.load(handle)
            if not isinstance(live, dict):
                raise ValueError("live PolySeg feedback ledger must contain a JSON object")
            if live.get("schema_version") != saved.get("schema_version") or live.get("ledger_id") != saved.get("ledger_id"):
                raise RuntimeError("live PolySeg feedback ledger identity differs from checkpoint snapshot")
            if not math.isclose(float(live.get("total")), float(saved.get("total")), rel_tol=0.0, abs_tol=1e-6):
                raise RuntimeError("live PolySeg feedback ledger total differs from checkpoint snapshot")
            temporary = source.with_name(f".{source.name}.{os.getpid()}.resume.tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(saved, handle, allow_nan=False, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, source)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return saved


def configured_feedback_ledger_path(config):
    """Resolve the shared ledger path from a trainer config, if configured."""

    reward = config.get("reward", {}) if config is not None else {}
    kwargs = reward.get("reward_kwargs", {}) if hasattr(reward, "get") else {}
    value = kwargs.get("feedback_ledger_path") if hasattr(kwargs, "get") else None
    return None if value in (None, "") else str(value)


def configured_feedback_journal_path(config):
    """Resolve the append-only trajectory journal path from a trainer config."""

    reward = config.get("reward", {}) if config is not None else {}
    kwargs = reward.get("reward_kwargs", {}) if hasattr(reward, "get") else {}
    value = kwargs.get("feedback_journal_path") if hasattr(kwargs, "get") else None
    return None if value in (None, "") else str(value)


def _load_journal_cursor(snapshot_path) -> dict:
    with Path(snapshot_path).open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    if not isinstance(state, dict) or state.get("schema_version") != 1:
        raise ValueError("invalid PolySeg feedback-journal checkpoint schema")
    size_bytes = state.get("size_bytes")
    record_count = state.get("record_count")
    prefix_sha256 = state.get("prefix_sha256")
    if type(size_bytes) is not int or size_bytes < 0:
        raise ValueError("feedback-journal checkpoint size_bytes must be a non-negative integer")
    if type(record_count) is not int or record_count < 0:
        raise ValueError("feedback-journal checkpoint record_count must be a non-negative integer")
    if not isinstance(prefix_sha256, str) or len(prefix_sha256) != 64:
        raise ValueError("feedback-journal checkpoint prefix_sha256 is invalid")
    return state


def _journal_prefix_state(handle, size_bytes: int) -> tuple[int, str]:
    handle.seek(0)
    digest = hashlib.sha256()
    record_count = 0
    remaining = size_bytes
    last_byte = b""
    while remaining:
        chunk = handle.read(min(1024 * 1024, remaining))
        if not chunk:
            raise RuntimeError("feedback journal ended before checkpoint cursor")
        digest.update(chunk)
        record_count += chunk.count(b"\n")
        last_byte = chunk[-1:]
        remaining -= len(chunk)
    if size_bytes and (last_byte != b"\n" or record_count == 0):
        raise RuntimeError("feedback journal checkpoint does not end at a complete record")
    return record_count, digest.hexdigest()


def snapshot_feedback_journal(path, destination) -> dict:
    """Record a durable journal cursor beside a trainer checkpoint.

    A resumed run may have appended records after its last checkpoint before
    being interrupted.  The cursor lets startup verify the checkpoint prefix
    and remove only that uncheckpointed suffix before replaying the batch.
    """

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open("r+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0, os.SEEK_END)
            size_bytes = handle.tell()
            record_count, prefix_sha256 = _journal_prefix_state(handle, size_bytes)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    state = {
        "schema_version": 1,
        "journal_path": str(source.resolve()),
        "size_bytes": size_bytes,
        "record_count": record_count,
        "prefix_sha256": prefix_sha256,
    }
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, allow_nan=False, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return state


def verify_feedback_journal_snapshot(path, snapshot_path) -> dict:
    """Verify the checkpoint prefix and return its cursor plus trailing size."""

    source = Path(path)
    state = _load_journal_cursor(snapshot_path)
    expected_path = state.get("journal_path")
    if expected_path is not None and str(source.resolve()) != expected_path:
        raise RuntimeError("feedback journal checkpoint belongs to a different journal path")
    if not source.is_file():
        raise RuntimeError("feedback journal is missing while a checkpoint cursor exists")
    with source.open("rb") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0, os.SEEK_END)
            current_size = handle.tell()
            if current_size < state["size_bytes"]:
                raise RuntimeError("feedback journal is shorter than its checkpoint cursor")
            record_count, prefix_sha256 = _journal_prefix_state(handle, state["size_bytes"])
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    if record_count != state["record_count"] or prefix_sha256 != state["prefix_sha256"]:
        raise RuntimeError("feedback journal prefix differs from checkpoint cursor")
    return {**state, "current_size_bytes": current_size, "trailing_bytes": current_size - state["size_bytes"]}


def restore_feedback_journal_snapshot(path, snapshot_path) -> dict:
    """Drop post-checkpoint journal rows after verifying the checkpoint prefix."""

    source = Path(path)
    state = _load_journal_cursor(snapshot_path)
    expected_path = state.get("journal_path")
    if expected_path is not None and str(source.resolve()) != expected_path:
        raise RuntimeError("feedback journal checkpoint belongs to a different journal path")
    if not source.is_file():
        raise RuntimeError("feedback journal is missing while a checkpoint cursor exists")
    with source.open("r+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0, os.SEEK_END)
            current_size = handle.tell()
            if current_size < state["size_bytes"]:
                raise RuntimeError("feedback journal is shorter than its checkpoint cursor")
            record_count, prefix_sha256 = _journal_prefix_state(handle, state["size_bytes"])
            if record_count != state["record_count"] or prefix_sha256 != state["prefix_sha256"]:
                raise RuntimeError("feedback journal prefix differs from checkpoint cursor")
            if current_size > state["size_bytes"]:
                handle.seek(state["size_bytes"])
                handle.truncate()
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return {**state, "current_size_bytes": current_size, "truncated_bytes": max(0, current_size - state["size_bytes"])}


RESOURCE_ACCOUNTING_KEYS = (
    "prompt_tokens",
    "response_tokens",
    "action_tokens",
    "tool_observation_tokens",
    "total_tokens",
    "tool_calls",
    "tool_successes",
    "tool_failures",
)


def append_feedback_record(path, *, attempt_id, ledger_id, evaluation, protocol,
                           query_cost, query_count, segment_count, skipped_count,
                           budget_scope, global_budget, resource_accounting=None,
                           billing_receipts=None, feedback_contract=None):
    if not path:
        return
    if not attempt_id:
        raise ValueError("feedback journal requires an attempt_id")
    row = {
        "schema_version": 1,
        "attempt_id": attempt_id,
        "event_id": uuid.uuid4().hex,
        "ledger_id": ledger_id,
        "phase": "evaluation" if evaluation else "training",
        "time_ns": time.time_ns(),
        "pid": os.getpid(),
        "protocol": protocol,
        "query_cost": float(query_cost),
        "query_count": int(query_count),
        "segment_count": int(segment_count),
        "skipped_count": int(skipped_count),
        "budget_scope": "evaluation" if evaluation else budget_scope,
        "global_budget": None if global_budget is None else {
            "total": global_budget.total,
            "consumed": global_budget.consumed,
            "reserved": global_budget.reserved,
        },
    }
    if feedback_contract is not None:
        if not isinstance(feedback_contract, dict):
            raise TypeError("feedback_contract must be a mapping")
        required = ("protocol", "family", "label_semantics", "estimator", "comparator")
        if any(not isinstance(feedback_contract.get(key), str) or not feedback_contract[key].strip() for key in required):
            raise ValueError("feedback_contract is missing a required semantic field")
        if str(feedback_contract["protocol"]).strip() != str(protocol).strip():
            raise ValueError("feedback_contract protocol disagrees with journal protocol")
        row["feedback_protocol_contract"] = {
            key: str(feedback_contract[key]).strip() for key in required
        }
    if resource_accounting is not None:
        if not isinstance(resource_accounting, dict):
            raise TypeError("resource_accounting must be a mapping")
        normalized = {}
        for key in RESOURCE_ACCOUNTING_KEYS:
            value = resource_accounting.get(key)
            if type(value) is not int or value < 0:
                raise ValueError(f"resource accounting field {key} must be a non-negative integer")
            normalized[key] = value
        row["resource_accounting"] = normalized
    if billing_receipts is not None:
        if not isinstance(billing_receipts, list):
            raise TypeError("billing_receipts must be a list")
        normalized_receipts = []
        for receipt in billing_receipts:
            if not isinstance(receipt, dict):
                raise TypeError("each billing receipt journal entry must be an object")
            status = receipt.get("status")
            if status not in {"accepted", "valid_not_used", "missing", "invalid_type", "missing_provider_query_id",
                              "invalid_billed_cost", "billed_cost_exceeds_reservation",
                              "billed_cost_mismatch"}:
                raise ValueError("invalid billing receipt status")
            segment_id = receipt.get("segment_id")
            if type(segment_id) is not int or segment_id < 0:
                raise ValueError("billing receipt segment_id must be a non-negative integer")
            item = {"segment_id": segment_id, "status": status}
            if status in {"accepted", "valid_not_used"}:
                query_id = receipt.get("provider_query_id")
                billed_cost = receipt.get("billed_cost")
                if not isinstance(query_id, str) or not query_id.strip():
                    raise ValueError("valid billing receipt requires provider_query_id")
                if type(billed_cost) not in (int, float) or not math.isfinite(float(billed_cost)) or billed_cost < 0:
                    raise ValueError("valid billing receipt requires billed_cost")
                item.update({"provider_query_id": query_id.strip(), "billed_cost": float(billed_cost)})
            if "reported_cost" in receipt:
                reported_cost = receipt["reported_cost"]
                if type(reported_cost) not in (int, float) or not math.isfinite(float(reported_cost)) or reported_cost < 0:
                    raise ValueError("billing receipt reported_cost must be non-negative")
                item["reported_cost"] = float(reported_cost)
            if "provider" in receipt:
                provider = receipt["provider"]
                if not isinstance(provider, str) or not provider.strip():
                    raise ValueError("billing receipt provider must be a non-empty string")
                item["provider"] = provider.strip()
            normalized_receipts.append(item)
        if len(normalized_receipts) > int(query_count):
            raise ValueError("billing receipt entries exceed query_count")
        row["billing_receipts"] = normalized_receipts
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # flock covers the complete write even if a filesystem splits it. A
    # missing/failed journal write fails the run, never silently drops cost.
    with target.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
