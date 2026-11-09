import json
import multiprocessing as mp

import pytest

from verl.trainer.ppo.polyseg import FileBackedSegmentRewardBudget
from verl.utils.polyseg_journal import (
    restore_feedback_journal_snapshot,
    restore_feedback_ledger_snapshot,
    snapshot_feedback_journal,
    snapshot_feedback_ledger,
    verify_feedback_journal_snapshot,
    verify_feedback_ledger_snapshot,
)


def _reserve_and_commit_shared_budget(args):
    """Worker used by the cross-process ledger contention regression."""

    path, reservation_id = args
    budget = FileBackedSegmentRewardBudget(path, 4.0)
    try:
        token = budget.reserve(1.0, reservation_id=reservation_id)
        # Exercise settlement through a second object in the same worker.  A
        # real reward callback may not retain the object that made the hold.
        reloaded = FileBackedSegmentRewardBudget(path, 4.0)
        reloaded.commit(token, cost=1.0)
        return "committed"
    except (RuntimeError, ValueError, KeyError):
        return "rejected"


def test_feedback_ledger_snapshot_round_trip_and_mismatch_fails_closed(tmp_path):
    live = tmp_path / "ledger.json"
    snapshot = tmp_path / "checkpoint" / "polyseg_feedback_ledger.json"
    budget = FileBackedSegmentRewardBudget(live, 3.0)
    token = budget.reserve(1.0, reservation_id="r1")
    budget.commit(token, cost=1.0)

    state = snapshot_feedback_ledger(live, snapshot)
    assert state["ledger_id"] == budget.ledger_id
    assert verify_feedback_ledger_snapshot(live, snapshot)["consumed"] == 1.0

    budget.consume(1.0)
    with pytest.raises(RuntimeError, match="differs from checkpoint"):
        verify_feedback_ledger_snapshot(live, snapshot)


def test_feedback_ledger_snapshot_is_complete_json(tmp_path):
    live = tmp_path / "ledger.json"
    snapshot = tmp_path / "snapshot.json"
    FileBackedSegmentRewardBudget(live, 2.0)
    snapshot_feedback_ledger(live, snapshot)
    saved = json.loads(snapshot.read_text())
    assert {"ledger_id", "total", "consumed", "reserved", "reservations", "settled"} <= saved.keys()


def test_feedback_ledger_restore_rolls_back_post_checkpoint_settlements(tmp_path):
    live = tmp_path / "ledger.json"
    snapshot = tmp_path / "checkpoint" / "polyseg_feedback_ledger.json"
    budget = FileBackedSegmentRewardBudget(live, 4.0)
    first = budget.reserve(1.0, reservation_id="checkpoint-query")
    budget.commit(first, cost=1.0)
    expected = snapshot_feedback_ledger(live, snapshot)

    # Simulate feedback that completed after the checkpoint but before a
    # process restart.  Resume must roll this state back before replay.
    second = budget.reserve(1.0, reservation_id="post-checkpoint-query")
    budget.commit(second, cost=1.0)
    assert json.loads(live.read_text())["consumed"] == 2.0
    with pytest.raises(RuntimeError, match="differs from checkpoint"):
        verify_feedback_ledger_snapshot(live, snapshot)

    restored = restore_feedback_ledger_snapshot(live, snapshot)
    assert restored == expected
    assert verify_feedback_ledger_snapshot(live, snapshot) == expected
    assert json.loads(live.read_text())["consumed"] == 1.0


def test_shared_feedback_ledger_enforces_budget_across_processes(tmp_path):
    """Concurrent workers may reserve at most the configured global budget."""

    live = tmp_path / "shared-ledger.json"
    ctx = mp.get_context("fork")
    requests = [(str(live), f"worker-{index}") for index in range(12)]
    with ctx.Pool(processes=6) as pool:
        outcomes = pool.map(_reserve_and_commit_shared_budget, requests)

    assert outcomes.count("committed") == 4
    assert outcomes.count("rejected") == 8
    final = FileBackedSegmentRewardBudget(live, 4.0)
    assert final.consumed == pytest.approx(4.0)
    assert final.reserved == pytest.approx(0.0)
    assert final.remaining == pytest.approx(0.0)


def test_feedback_journal_cursor_truncates_post_checkpoint_rows(tmp_path):
    journal = tmp_path / "feedback.jsonl"
    journal.write_bytes(b'{"event_id":"one"}\n{"event_id":"two"}\n')
    snapshot = tmp_path / "checkpoint" / "polyseg_feedback_journal.json"

    state = snapshot_feedback_journal(journal, snapshot)
    assert state["record_count"] == 2
    journal.write_bytes(journal.read_bytes() + b'{"event_id":"three"}\n')

    checked = verify_feedback_journal_snapshot(journal, snapshot)
    assert checked["trailing_bytes"] > 0
    restored = restore_feedback_journal_snapshot(journal, snapshot)
    assert restored["truncated_bytes"] == checked["trailing_bytes"]
    assert journal.read_bytes() == b'{"event_id":"one"}\n{"event_id":"two"}\n'
    assert verify_feedback_journal_snapshot(journal, snapshot)["trailing_bytes"] == 0


def test_feedback_journal_cursor_rejects_changed_checkpoint_prefix(tmp_path):
    journal = tmp_path / "feedback.jsonl"
    journal.write_bytes(b'{"event_id":"one"}\n')
    snapshot = tmp_path / "checkpoint.json"
    snapshot_feedback_journal(journal, snapshot)
    journal.write_bytes(b'{"event_id":"two"}\n')
    with pytest.raises(RuntimeError, match="prefix differs"):
        verify_feedback_journal_snapshot(journal, snapshot)
