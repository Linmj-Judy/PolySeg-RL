"""Durable accounting of completed feedback trajectories, before transport.

This journal counts scorer calls (not suffix samples) and their reported cost.
It is evidence for successful runs; failed attempts retain their own journal
and cannot be promoted to research results. All workers must share the path.
"""

import fcntl
import json
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


def configured_feedback_ledger_path(config):
    """Resolve the shared ledger path from a trainer config, if configured."""

    reward = config.get("reward", {}) if config is not None else {}
    kwargs = reward.get("reward_kwargs", {}) if hasattr(reward, "get") else {}
    value = kwargs.get("feedback_ledger_path") if hasattr(kwargs, "get") else None
    return None if value in (None, "") else str(value)


def append_feedback_record(path, *, attempt_id, ledger_id, evaluation, protocol,
                           query_cost, query_count, segment_count, skipped_count,
                           budget_scope, global_budget):
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
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # flock covers the complete write even if a filesystem splits it. A
    # missing/failed journal write fails the run, never silently drops cost.
    with target.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
