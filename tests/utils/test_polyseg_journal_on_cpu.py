import json

import pytest

from verl.trainer.ppo.polyseg import FileBackedSegmentRewardBudget
from verl.utils.polyseg_journal import snapshot_feedback_ledger, verify_feedback_ledger_snapshot


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
