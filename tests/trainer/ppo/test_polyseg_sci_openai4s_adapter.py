"""Contract tests for the OpenAI4S/PolySeg-Sci integration boundary."""

from pathlib import Path
from dataclasses import replace
import asyncio
import sqlite3
import sys
import threading

import torch

import pytest

from verl.trainer.ppo.polyseg_sci import (
    BoundarySnapshot,
    BranchExecutionError,
    CausalSegmenter,
    EndpointObservation,
    EndpointSchema,
    EndpointSpec,
    FeatureMapContract,
    FidelitySpec,
    RestoreAudit,
    StateAuditOracle,
)
import verl.trainer.ppo.polyseg_openai4s_provider as openai4s_provider
from verl.trainer.ppo.polyseg_openai4s_provider import OpenAI4SScienceRoundProvider
from verl.trainer.ppo.polyseg_sci_provider import ScienceRoundRequest
from verl.trainer.ppo.polyseg_sci_openai4s import (
    OpenAI4SAdapterError,
    OpenAI4SStoreAdapter,
    OpenAI4STrajectoryAdapter,
    write_sidecar,
)


def _schema() -> EndpointSchema:
    fidelity = FidelitySpec("score-hf", "HF", "gold", "v1", 2.0, kind="gold")
    return EndpointSchema(
        (EndpointSpec("score", "score", "a.u.", valid_range=(0.0, 1.0), hf_fidelity_id="score-hf", fidelities=(fidelity,)),),
        objective_mode="preference",
        weights=(1.0,),
    )


def _adapter() -> OpenAI4STrajectoryAdapter:
    fmap = FeatureMapContract.fit([[0.0], [1.0]], ["progress"])
    return OpenAI4STrajectoryAdapter(
        _schema(), fmap, segmenter=CausalSegmenter(m=2, horizon=4), state_oracle=StateAuditOracle()
    )


def test_action_ledger_keeps_policy_token_alignment_and_excludes_tool_results():
    adapter = _adapter()
    groups = (
        {
            "group_id": "g0",
            "kind": "tool",
            "events": (
                {"type": "proposed", "action_id": "call-1", "tool_name": "simulate", "created_at": 1700000000000},
                {"type": "result", "action_id": "call-1", "event_id": "return-1", "created_at": 1700000001000},
                {"type": "model_call", "result": {"usage": 3}},
            ),
        },
    )
    records = adapter.adapt_action_ledger(
        groups,
        policy_actions=(
            {"action_id": "tok-0", "token_start": 0, "token_end": 2, "logprob": -0.2},
            {"action_id": "tok-1", "token_start": 2, "token_end": 4, "logprob": -0.3},
        ),
        token_count=4,
    )
    assert [record.is_policy_action for record in records] == [True, True, False]
    assert records[-1].tool_name == "simulate"
    assert records[-1].tool_return_id == "return-1"
    assert records[-1].timestamp == pytest.approx(1700000001.0)
    assert adapter.action_mask(records, 4) == (True, True, True, True)


def test_action_ledger_rejects_zero_length_policy_token_span():
    adapter = _adapter()
    with pytest.raises(OpenAI4SAdapterError, match="policy token span"):
        adapter.adapt_action_ledger(
            (),
            policy_actions=({"action_id": "tok-0", "token_start": 2, "token_end": 2},),
            token_count=4,
        )


@pytest.mark.parametrize(
    "policy_action",
    [
        {"action_id": "tok-0", "token_start": 0.5, "token_end": 1.5},
        {"action_id": "tok-0", "token_start": False, "token_end": 1},
    ],
)
def test_action_ledger_rejects_coercible_policy_token_indices(policy_action):
    with pytest.raises(OpenAI4SAdapterError, match="policy token_start"):
        _adapter().adapt_action_ledger((), policy_actions=(policy_action,), token_count=4)


def test_action_ledger_keeps_completed_tool_events_with_call_identity():
    adapter = _adapter()
    records = adapter.adapt_action_ledger(
        ({"group_id": "g0", "events": ({"type": "proposed", "action_id": "call-1"}, {"type": "completed", "action_id": "call-1", "event_id": "return-1"})},),
        policy_actions=({"action_id": "tok-0", "token_start": 0, "token_end": 1},),
        token_count=1,
    )
    assert records[-1].tool_call_id == "call-1"
    assert records[-1].tool_return_id == "return-1"


def test_action_ledger_rejects_overlapping_policy_token_spans():
    adapter = _adapter()
    with pytest.raises(OpenAI4SAdapterError, match="spans overlap"):
        adapter.adapt_action_ledger(
            (),
            policy_actions=(
                {"action_id": "a0", "token_start": 0, "token_end": 2},
                {"action_id": "a1", "token_start": 1, "token_end": 3},
            ),
            token_count=3,
        )


def test_segment_and_rollout_fields_are_token_aligned_and_fail_closed():
    adapter = _adapter()
    actions = adapter.adapt_action_ledger(
        (),
        policy_actions=({"action_id": "a0", "token_start": 0, "token_end": 4},),
        token_count=4,
    )
    mask, boundaries = adapter.segment_actions(actions, token_count=4)
    fields = adapter.science_rollout_fields(
        action_mask=mask,
        boundaries=boundaries,
        corrected_boundary_values=[[0.0], [0.4], [0.8]],
    )
    assert fields["segment_ids"].tolist() == [0, 0, 1, 1]
    assert fields["science_boundary_values"].shape == (3, 1)
    with pytest.raises(OpenAI4SAdapterError, match="incomplete endpoint"):
        adapter.science_rollout_fields(
            action_mask=mask,
            boundaries=boundaries,
            corrected_boundary_values=[[0.0], [0.4], [0.8]],
            endpoint_mask=[[True], [False], [True]],
        )


def test_capture_restore_and_fresh_branch_contract():
    adapter = _adapter()
    boundary = adapter.segmenter.segment((True, True, True, True))[0]
    checkpoint_calls = []

    def checkpoint(root, **kwargs):
        checkpoint_calls.append((root, kwargs))
        return {"checkpoint_id": "cp-0"}

    snapshot = adapter.capture_boundary(
        root_frame_id="root",
        branch_id="root",
        boundary=boundary,
        task_id="task",
        trajectory_id="traj",
        run_id="run",
        session_id="session",
        prefix="prefix",
        action_mask=(True, True, True, True),
        old_log_probs=(-1.0, -1.0, -1.0, -1.0),
        tool_ledger=(),
        state={
            "workspace_hashes": {"a": "h"},
            "dependency_hashes": {"code": "d"},
            "rng_state": "rng",
            "execution_budget_initial": 10,
            "execution_budget_remaining": 7,
            "policy_version": "p1",
            "tokenizer_version": "t1",
            "sampling_config": {"temperature": 0.0},
        },
        checkpoint_fn=checkpoint,
    )
    assert snapshot.snapshot_id == "cp-0"
    assert checkpoint_calls[0][1]["metadata"]["feature_map_hash"] == adapter.feature_map.fingerprint

    def restore(value):
        return {
            "boundary_id": value.boundary_id,
            "snapshot_id": value.snapshot_id,
            "policy_version": value.policy_version,
            "feature_map_hash": value.feature_map_hash,
            "segment_rule_version": value.segment_rule_version,
            "task_id": value.task_id,
            "trajectory_id": value.trajectory_id,
            "run_id": value.run_id,
            "session_id": value.session_id,
            "tokenizer_version": value.tokenizer_version,
            "dependency_hashes": dict(value.dependency_hashes),
            "sampling_config": dict(value.sampling_config),
            "rng_state": value.rng_state,
            "snapshot_digest": value.digest,
            "execution_budget_initial": value.execution_budget_initial,
            "execution_budget_remaining": value.execution_budget_remaining,
            "artifact_hashes": dict(value.workspace_hashes),
            "prefix": value.prefix,
            "action_mask": value.action_mask,
            "old_log_probs": value.old_log_probs,
            "tool_ledger": value.tool_ledger,
            "rng_independent": True,
        }

    audit = adapter.restore_audit(snapshot, restore, cost=0.5)
    assert audit.passed
    branches = adapter.fresh_branches(
        snapshot,
        restore_audit=audit,
        branch_count=2,
        policy_version="p1",
        branch_fn=lambda _snapshot, rng_id, remaining: {
            "branch_id": f"{_snapshot.boundary_id}:fresh:{rng_id}",
            "policy_version": "p1",
            "rng_id": rng_id,
            "rng_independent": True,
            "execution_budget_remaining": remaining - 1,
            "observations": (EndpointObservation("score", "score-hf", 0.5, evaluator_id="gold", evaluator_version="v1"),),
        },
    )
    assert len(branches) == 2
    assert len({branch.rng_id for branch in branches}) == 2
    assert all(branch.status == "completed" for branch in branches)


def test_openai4s_branch_callback_preserves_evaluator_failure_class():
    adapter = _adapter()
    snapshot = BoundarySnapshot(
        "task", "traj", "run", "session", "b0", "prefix", (True,), (-1.0,), (), {}, {}, "rng", 10, 6,
        "policy-v1", "tok-v1", {}, adapter.feature_map.version, adapter.feature_map.fingerprint,
        adapter.segmenter.rule_version, "snap",
    )
    audit = adapter.state_oracle.audit(
        snapshot,
        {
            "boundary_id": "b0", "snapshot_id": "snap", "policy_version": "policy-v1",
            "feature_map_hash": adapter.feature_map.fingerprint, "segment_rule_version": adapter.segmenter.rule_version,
            "task_id": "task", "trajectory_id": "traj", "run_id": "run", "session_id": "session",
            "tokenizer_version": "tok-v1", "dependency_hashes": {}, "sampling_config": {}, "rng_state": "rng",
            "execution_budget_remaining": 6, "artifact_hashes": {}, "rng_independent": True,
        },
    )
    callback = adapter.openai4s_branch_callback(
        root_frame_id="root",
        snapshot=snapshot,
        fork_fn=lambda _root, **kwargs: {"branch_id": kwargs["branch_id"]},
        rollout_fn=lambda _branch, current, rng_id, remaining: {
            "artifact": "ok",
            "policy_version": current.policy_version,
            "rng_id": rng_id,
            "rng_independent": True,
            "execution_budget_remaining": remaining,
        },
        evaluate_fn=lambda _artifact: (_ for _ in ()).throw(ValueError("evaluator unavailable")),
    )
    result = adapter.fresh_branches(snapshot, restore_audit=audit, branch_count=1, branch_fn=callback, policy_version="policy-v1")
    assert result[0].failure_class == "evaluator_failure"


def test_openai4s_branch_callback_rejects_missing_runtime_attestations():
    adapter = _adapter()
    snapshot = BoundarySnapshot(
        "task", "traj", "run", "session", "b0", "prefix", (True,), (-1.0,), (), {}, {}, "rng", 10, 6,
        "policy-v1", "tok-v1", {}, adapter.feature_map.version, adapter.feature_map.fingerprint,
        adapter.segmenter.rule_version, "snap",
    )
    audit = adapter.state_oracle.audit(
        snapshot,
        {
            "boundary_id": "b0", "snapshot_id": "snap", "policy_version": "policy-v1",
            "feature_map_hash": adapter.feature_map.fingerprint, "segment_rule_version": adapter.segmenter.rule_version,
            "task_id": "task", "trajectory_id": "traj", "run_id": "run", "session_id": "session",
            "tokenizer_version": "tok-v1", "dependency_hashes": {}, "sampling_config": {}, "rng_state": "rng",
            "execution_budget_remaining": 6, "artifact_hashes": {},
        },
    )
    callback = adapter.openai4s_branch_callback(
        root_frame_id="root",
        snapshot=snapshot,
        fork_fn=lambda _root, **kwargs: {"branch_id": kwargs["branch_id"]},
        rollout_fn=lambda *_args: {"artifact": "ok"},
        evaluate_fn=lambda _artifact: (),
    )
    result = adapter.fresh_branches(
        snapshot,
        restore_audit=audit,
        branch_count=1,
        branch_fn=callback,
        policy_version="policy-v1",
    )
    assert result[0].failure_class == "branch_failure"
    assert "attest" in (result[0].failure_reason or "")


def test_adapter_restore_requires_captured_state_projections():
    adapter = _adapter()
    snapshot = BoundarySnapshot(
        "task", "traj", "run", "session", "b0", "prefix", (True,), (-1.0,), (), {}, {}, "rng", 10, 6,
        "policy-v1", "tok-v1", {}, adapter.feature_map.version, adapter.feature_map.fingerprint,
        adapter.segmenter.rule_version, "snap",
    )
    restored = {
        "boundary_id": "b0", "snapshot_id": "snap", "policy_version": "policy-v1",
        "feature_map_hash": adapter.feature_map.fingerprint, "segment_rule_version": adapter.segmenter.rule_version,
        "task_id": "task", "trajectory_id": "traj", "run_id": "run", "session_id": "session",
        "tokenizer_version": "tok-v1", "dependency_hashes": {}, "sampling_config": {}, "rng_state": "rng",
        "execution_budget_remaining": 6, "artifact_hashes": {},
    }
    audit = adapter.restore_audit(snapshot, lambda _snapshot: restored)
    assert not audit.passed
    assert "prefix_missing" in audit.reasons


def test_capture_requires_explicit_runtime_state_and_trajectory_is_auditable():
    adapter = _adapter()
    boundary = adapter.segmenter.segment((True, True, True, True))[0]
    with pytest.raises(OpenAI4SAdapterError, match="missing required fields"):
        adapter.capture_boundary(
            root_frame_id="root",
            branch_id="root",
            boundary=boundary,
            task_id="task",
            trajectory_id="traj",
            run_id="run",
            session_id="session",
            prefix="prefix",
            action_mask=(True, True, True, True),
            old_log_probs=(-1.0,) * 4,
            tool_ledger=(),
            state={},
            checkpoint_fn=lambda *_args, **_kwargs: {"checkpoint_id": "cp"},
        )
    trajectory = adapter.trajectory(
        metadata={
            "task_id": "task",
            "trajectory_id": "traj",
            "prompt_id": "prompt",
            "run_id": "run",
            "session_id": "session",
            "actor_checkpoint": "actor",
            "tokenizer_version": "tok",
            "policy_version": "policy",
            "sampling_config": {},
            "terminal_evaluator_id": "gold",
            "terminal_evaluator_version": "v1",
            "execution_budget_initial": 10,
            "execution_budget_remaining": 4,
            "trajectory_complete": True,
        },
        terminal_observations=(EndpointObservation("score", "score-hf", 0.8, evaluator_id="gold", evaluator_version="v1"),),
        actions=(),
        terminal_cost=2.0,
    )
    assert trajectory.research_eligible is False
    assert trajectory.fingerprint


def test_capture_boundary_rejects_coercible_masks_and_log_probs():
    adapter = _adapter()
    boundary = adapter.segmenter.segment((True, True, True, True))[0]
    checkpoint_calls = []
    common = dict(
        root_frame_id="root",
        branch_id="root",
        boundary=boundary,
        task_id="task",
        trajectory_id="traj",
        run_id="run",
        session_id="session",
        prefix="prefix",
        action_mask=(True, True, True, True),
        old_log_probs=(-1.0,) * 4,
        tool_ledger=(),
        state={
            "workspace_hashes": {}, "dependency_hashes": {}, "rng_state": "rng",
            "execution_budget_initial": 10, "execution_budget_remaining": 7,
            "policy_version": "p1", "tokenizer_version": "t1", "sampling_config": {},
        },
        checkpoint_fn=lambda *_args, **_kwargs: checkpoint_calls.append(True) or {"checkpoint_id": "cp"},
    )
    with pytest.raises(OpenAI4SAdapterError, match="action_mask entry"):
        adapter.capture_boundary(**{**common, "action_mask": (1, True, True, True)})
    with pytest.raises(OpenAI4SAdapterError, match="old_log_probs entry"):
        adapter.capture_boundary(**{**common, "old_log_probs": ("-1.0",) * 4})
    for key in ("execution_budget_initial", "execution_budget_remaining"):
        with pytest.raises(OpenAI4SAdapterError, match=key):
            adapter.capture_boundary(**{**common, "state": {**common["state"], key: True}})
    assert checkpoint_calls == []


def test_trajectory_rejects_coercible_boolean_metadata():
    adapter = _adapter()
    metadata = {
        "task_id": "task", "trajectory_id": "traj", "prompt_id": "prompt", "run_id": "run",
        "session_id": "session", "actor_checkpoint": "actor", "tokenizer_version": "tok",
        "policy_version": "policy", "sampling_config": {}, "terminal_evaluator_id": "gold",
        "terminal_evaluator_version": "v1", "execution_budget_initial": 10,
        "execution_budget_remaining": 4, "trajectory_complete": True,
    }
    observation = EndpointObservation("score", "score-hf", 0.8, evaluator_id="gold", evaluator_version="v1")
    with pytest.raises(OpenAI4SAdapterError, match="trajectory_complete"):
        adapter.trajectory(
            metadata={**metadata, "trajectory_complete": "false"},
            terminal_observations=(observation,), actions=(), terminal_cost=1.0,
        )
    with pytest.raises(OpenAI4SAdapterError, match="research_eligible"):
        adapter.trajectory(
            metadata=metadata, terminal_observations=(observation,), actions=(), terminal_cost=1.0,
            research_eligible="false",
        )


def _contract_snapshot():
    adapter = _adapter()
    return BoundarySnapshot(
        "task", "traj", "run", "session", "b0", "prefix", (True,), (-1.0,), (), {}, {}, "rng", 10, 6,
        "policy-v1", "tok-v1", {}, adapter.feature_map.version, adapter.feature_map.fingerprint,
        adapter.segmenter.rule_version, "snap",
    )


@pytest.mark.parametrize("cost", [True, "0.5", -1.0, float("nan"), float("inf")])
def test_restore_rejects_invalid_cost_before_callback_and_oracle_never_passes(cost):
    snapshot = _contract_snapshot()
    calls = []
    audit = _adapter().restore_audit(snapshot, lambda _: calls.append(True), cost=cost)
    assert not audit.passed and "invalid_restore_cost" in audit.reasons
    assert calls == []
    restored = snapshot.to_dict()
    restored["artifact_hashes"] = snapshot.workspace_hashes
    direct = StateAuditOracle().audit(snapshot, restored, cost=cost, require_captured_state=True)
    assert not direct.passed and "invalid_restore_cost" in direct.reasons
    assert direct.cost == 0.0


@pytest.mark.parametrize("field", ["execution_budget_initial", "execution_budget_remaining"])
def test_restore_rejects_boolean_budget_even_when_python_values_compare_equal(field):
    snapshot = replace(_contract_snapshot(), execution_budget_initial=1.0, execution_budget_remaining=1.0)
    restored = snapshot.to_dict()
    restored["artifact_hashes"] = snapshot.workspace_hashes
    restored["snapshot_digest"] = snapshot.digest
    restored[field] = True
    audit = StateAuditOracle().audit(snapshot, restored, require_captured_state=True)
    assert not audit.passed
    assert audit.reasons == ("%s_mismatch" % ("initial_budget" if field.endswith("initial") else "remaining_budget"),)


@pytest.mark.parametrize("field", ["cost", "execution_budget_remaining"])
@pytest.mark.parametrize("value", [True, "1.0"])
def test_fresh_branch_rejects_coercible_accounting_without_emitting_labels(field, value):
    snapshot = _contract_snapshot()
    audit = RestoreAudit(snapshot.boundary_id, "passed", snapshot.digest, snapshot.digest, 0.0)

    def branch(current, rng_id, remaining):
        return {
            "branch_id": f"{current.boundary_id}:fresh:{rng_id}", "rng_id": rng_id,
            "rng_independent": True, "policy_version": current.policy_version,
            "execution_budget_remaining": remaining, "cost": 1.0,
            "observations": (EndpointObservation("score", "score-hf", 0.5, evaluator_id="gold", evaluator_version="v1"),),
            field: value,
        }

    results = _adapter().fresh_branches(
        snapshot, restore_audit=audit, branch_count=1, branch_fn=branch, policy_version=snapshot.policy_version,
    )
    assert results[0].status == "failed"
    assert results[0].observations == ()
    assert results[0].failure_class == "branch_failure"


@pytest.mark.parametrize("field", ["cost", "execution_budget_remaining"])
def test_openai4s_branch_rejects_invalid_accounting_before_evaluator(field):
    from verl.trainer.ppo.polyseg_sci import BranchExecutionError

    snapshot = _contract_snapshot()
    evaluator_calls = []
    callback = _adapter().openai4s_branch_callback(
        root_frame_id="root", snapshot=snapshot,
        fork_fn=lambda _, **kwargs: {"branch_id": kwargs["branch_id"]},
        rollout_fn=lambda branch, current, rng, remaining: {
            "policy_version": current.policy_version, "rng_id": rng, "rng_independent": True,
            "execution_budget_remaining": remaining, "cost": 1.0, field: True,
        },
        evaluate_fn=lambda _: evaluator_calls.append(True),
    )
    with pytest.raises(BranchExecutionError, match="invalid"):
        callback(snapshot, "independent-rng", 6.0)
    assert evaluator_calls == []


@pytest.mark.parametrize("field,value", [("action_mask", [1, 1, 1, 1]), ("endpoint_mask", [[1], [1], [1]])])
def test_rollout_fields_reject_coercible_masks(field, value):
    adapter = _adapter()
    kwargs = dict(
        action_mask=(True,) * 4, boundaries=adapter.segmenter.segment((True,) * 4),
        corrected_boundary_values=[[0.0], [0.4], [0.8]],
    )
    kwargs[field] = value
    with pytest.raises(OpenAI4SAdapterError, match="boolean"):
        adapter.science_rollout_fields(**kwargs)


def test_trajectory_requires_complete_hf_endpoint_vector():
    adapter = _adapter()
    metadata = {
        "task_id": "task",
        "trajectory_id": "traj",
        "prompt_id": "prompt",
        "run_id": "run",
        "session_id": "session",
        "actor_checkpoint": "actor",
        "tokenizer_version": "tok",
        "policy_version": "policy",
        "sampling_config": {},
        "terminal_evaluator_id": "gold",
        "terminal_evaluator_version": "v1",
        "execution_budget_initial": 10,
        "execution_budget_remaining": 4,
    }
    with pytest.raises(OpenAI4SAdapterError, match="incomplete"):
        adapter.trajectory(
            metadata=metadata,
            terminal_observations=(),
            actions=(),
            terminal_cost=1.0,
        )


def test_trajectory_preserves_endpoint_level_evaluator_provenance_for_vector_schema():
    adapter = _adapter()
    metadata = {
        "task_id": "task", "trajectory_id": "traj-vector", "prompt_id": "prompt",
        "run_id": "run", "session_id": "session", "actor_checkpoint": "actor",
        "tokenizer_version": "tok", "policy_version": "policy", "sampling_config": {},
        # Scalar fields remain required for compatibility; endpoint maps are
        # the authoritative form when a vector schema uses heterogeneous HF
        # evaluator services.
        "terminal_evaluator_id": "vector", "terminal_evaluator_version": "vector-v1",
        "terminal_evaluator_ids": {"score": "gold"},
        "terminal_evaluator_versions": {"score": "v1"},
        "execution_budget_initial": 10, "execution_budget_remaining": 4,
        "trajectory_complete": True,
    }
    trajectory = adapter.trajectory(
        metadata=metadata,
        terminal_observations=(EndpointObservation("score", "score-hf", 0.7, evaluator_id="gold", evaluator_version="v1"),),
        actions=(),
        terminal_cost=1.0,
    )
    assert trajectory.terminal_evaluator_ids == {"score": "gold"}
    assert trajectory.terminal_evaluator_versions == {"score": "v1"}


def test_formal_fresh_branch_requires_evaluator_provenance():
    adapter = _adapter()
    snapshot = BoundarySnapshot(
        "task", "traj", "run", "session", "b0", "prefix", (True,), (-1.0,), (), {}, {}, "rng", 10, 6,
        "policy-v1", "tok-v1", {}, adapter.feature_map.version, adapter.feature_map.fingerprint,
        adapter.segmenter.rule_version, "snap",
    )
    restored = {
        "boundary_id": "b0", "snapshot_id": "snap", "policy_version": "policy-v1",
        "feature_map_hash": adapter.feature_map.fingerprint, "segment_rule_version": adapter.segmenter.rule_version,
        "task_id": "task", "trajectory_id": "traj", "run_id": "run", "session_id": "session",
        "tokenizer_version": "tok-v1", "dependency_hashes": {}, "sampling_config": {}, "rng_state": "rng",
        "snapshot_digest": snapshot.digest, "execution_budget_initial": snapshot.execution_budget_initial,
        "execution_budget_remaining": 6, "artifact_hashes": {}, "prefix": "prefix",
        "action_mask": (True,), "old_log_probs": (-1.0,), "tool_ledger": (),
    }
    audit = adapter.state_oracle.audit(snapshot, restored, require_captured_state=True)
    result = adapter.fresh_branches(
        snapshot,
        restore_audit=audit,
        branch_count=1,
        policy_version="policy-v1",
        branch_fn=lambda _snapshot, rng_id, remaining: {
            "branch_id": f"b0:fresh:{rng_id}", "policy_version": "policy-v1", "rng_id": rng_id,
            "rng_independent": True, "execution_budget_remaining": remaining,
            "observations": (EndpointObservation("score", "score-hf", 0.5),),
        },
    )
    assert result[0].status == "failed" and "provenance" in (result[0].failure_reason or "")


def test_sidecar_is_atomic_and_hashed(tmp_path: Path):
    path = tmp_path / "run" / "trajectory.json"
    digest = write_sidecar(path, {"b": 2, "a": 1})
    assert path.exists()
    assert digest
    assert path.read_text() == '{"a":1,"b":2}'


def test_store_facade_uses_real_openai4s_projection_signatures():
    class Store:
        def list_action_groups(self, root, **kwargs):
            assert root == "root" and kwargs["include_events"] is True
            return [{"group_id": "g"}]

        def get_session_checkpoint(self, checkpoint_id):
            return {"checkpoint_id": checkpoint_id}

    class Domain:
        def create_checkpoint(self, root, **kwargs):
            assert root == "root" and kwargs["branch_id"] == "root"
            return {"checkpoint_id": "cp"}

        def fork_branch(self, root, **kwargs):
            assert root == "root"
            return {"branch_id": kwargs["branch_id"]}

    facade = OpenAI4SStoreAdapter(Store(), Domain(), root_frame_id="root")
    assert facade.action_groups() == ({"group_id": "g"},)
    assert facade.create_checkpoint(reason="boundary")["checkpoint_id"] == "cp"
    assert facade.fork_branch(from_checkpoint_id="cp", branch_id="br")["branch_id"] == "br"
    assert facade.checkpoint("cp")["checkpoint_id"] == "cp"


def test_capture_boundary_uses_real_openai4s_checkpoint_and_cas(tmp_path: Path):
    # Import the pinned harness checkout directly; the adapter itself remains
    # dependency-free and only uses the public duck-typed service contract.
    harness_root = Path(__file__).resolve().parents[4] / "repos" / "harness" / "OpenAI4S"
    sys.path.insert(0, str(harness_root))
    try:
        from openai4s.server.session_branching import SessionBranchingService
        from openai4s.storage.snapshots import SessionSnapshotRepository, WorkspaceCAS

        connection = sqlite3.connect(tmp_path / "openai4s.sqlite")
        connection.row_factory = sqlite3.Row
        repository = SessionSnapshotRepository(connection, threading.RLock(), clock_ms=lambda: 1000)
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "state.txt").write_text("boundary", encoding="utf-8")
        state = {
            "workspace_hashes": {"state.txt": "sha256:external"},
            "dependency_hashes": {"code": "sha256:code"},
            "rng_state": "rng-v1",
            "execution_budget_initial": 10,
            "execution_budget_remaining": 7,
            "policy_version": "policy-v1",
            "tokenizer_version": "tok-v1",
            "sampling_config": {"temperature": 0.0},
        }
        service = SessionBranchingService(
            repository,
            WorkspaceCAS(tmp_path / "cas"),
            workspace=lambda _root, _branch: workspace,
            read_state=lambda _root, _branch: state,
        )
        adapter = _adapter()
        boundary = adapter.segmenter.segment((True, True, True, True))[0]
        snapshot = adapter.capture_boundary(
            root_frame_id="root",
            branch_id="root",
            boundary=boundary,
            task_id="task",
            trajectory_id="traj",
            run_id="run",
            session_id="session",
            prefix="prefix",
            action_mask=(True, True, True, True),
            old_log_probs=(-1.0,) * 4,
            tool_ledger=(),
            state=state,
            checkpoint_fn=service.create_checkpoint,
        )
        assert snapshot.snapshot_id.startswith("cp-")
        assert repository.get_checkpoint(snapshot.snapshot_id)["workspace_tree_id"]
    finally:
        if "connection" in locals():
            connection.close()
        if str(harness_root) in sys.path:
            sys.path.remove(str(harness_root))


def test_openai4s_provider_fixture_stays_in_test_path(monkeypatch):
    """Exercise the provider bridge without putting fixture scores in experiments."""

    class Verification:
        passed = True
        reasons = ()
        checked_fields = ("ledger",)

    class Result:
        corrected_boundary_values = torch.tensor([[0.0], [0.5], [0.9]])
        segment_gradient = torch.tensor([[0.1], [0.2]])
        verification = Verification()
        ledger = {"spent": 1.0}
        audit = {"fixture": True}
        boundary_records = ()
        covariance_by_boundary = {}

    monkeypatch.setattr(openai4s_provider, "run_science_round", lambda **_: Result())
    adapter = _adapter()
    action_mask = (True, True, True, True)
    boundaries = adapter.segmenter.segment(action_mask)

    class Schema:
        dimension = 1
        fingerprint = ""

    class Adapter:
        schema = Schema()

        def science_rollout_fields(self, *, action_mask, boundaries, corrected_boundary_values, endpoint_mask):
            assert tuple(action_mask) == action_mask
            assert len(boundaries) == 2
            assert corrected_boundary_values.shape == (3, 1)
            return {
                "segment_ids": torch.tensor([0, 0, 1, 1]),
                "segment_mask": torch.ones(2, dtype=torch.bool),
                "science_boundary_values": corrected_boundary_values,
                "science_boundary_mask": torch.ones(2, dtype=torch.bool),
                "science_endpoint_mask": endpoint_mask,
            }

    response = asyncio.run(
        OpenAI4SScienceRoundProvider().execute(
            ScienceRoundRequest(
                output=None,
                outputs=(),
                validate=False,
                context={
                    "openai4s_science_round": {
                        "adapter": Adapter(),
                        "action_mask": action_mask,
                        "boundaries": boundaries,
                        "round_kwargs": {"fixture": True, "boundary_ids": ("b0",)},
                        "research_eligible": False,
                    }
                },
            )
        )
    )
    assert response.research_eligible is False
    assert response.audit["provider"] == "openai4s"
    assert response.audit["verification"]["passed"] is True


def test_openai4s_branch_rejects_bad_identity_before_rollout_and_evaluator():
    snapshot = _contract_snapshot()
    rollout_calls = []
    evaluator_calls = []

    for fork_result in ({}, {"branch_id": "other-branch"}):
        callback = _adapter().openai4s_branch_callback(
            root_frame_id="root",
            snapshot=snapshot,
            fork_fn=lambda _root, _result=fork_result, **kwargs: _result,
            rollout_fn=lambda *_args: rollout_calls.append(True) or {},
            evaluate_fn=lambda _artifact: evaluator_calls.append(True) or (),
        )
        with pytest.raises(BranchExecutionError, match="branch identit"):
            callback(snapshot, "independent-rng", snapshot.execution_budget_remaining)
    assert rollout_calls == []
    assert evaluator_calls == []


def test_restore_audit_rejects_missing_captured_evidence_fields():
    snapshot = _contract_snapshot()
    restored = snapshot.to_dict()
    for key in ("snapshot_digest", "execution_budget_initial", "artifact_hashes"):
        restored.pop(key, None)
    audit = _adapter().restore_audit(snapshot, lambda _snapshot: restored, evidence_kind="durable")
    assert not audit.passed
    assert {"snapshot_digest_missing", "initial_budget_missing", "artifact_hashes_missing"} <= set(
        audit.reasons
    )
    assert not audit.research_eligible


def test_trajectory_requires_explicit_trajectory_complete():
    metadata = {
        "task_id": "task", "trajectory_id": "traj", "prompt_id": "prompt", "run_id": "run",
        "session_id": "session", "actor_checkpoint": "actor", "tokenizer_version": "tok",
        "policy_version": "policy", "sampling_config": {}, "terminal_evaluator_id": "gold",
        "terminal_evaluator_version": "v1", "execution_budget_initial": 10,
        "execution_budget_remaining": 4,
    }
    with pytest.raises(OpenAI4SAdapterError, match="trajectory_complete"):
        _adapter().trajectory(
            metadata=metadata,
            terminal_observations=(
                EndpointObservation("score", "score-hf", 0.8, evaluator_id="gold", evaluator_version="v1"),
            ),
            actions=(),
            terminal_cost=1.0,
        )


@pytest.mark.parametrize("row_count", [2, 4])
def test_rollout_fields_reject_truncated_or_oversized_boundary_matrix(row_count):
    adapter = _adapter()
    boundaries = adapter.segmenter.segment((True,) * 4)
    values = [[0.0]] * row_count
    with pytest.raises(OpenAI4SAdapterError, match="shape"):
        adapter.science_rollout_fields(
            action_mask=(True,) * 4,
            boundaries=boundaries,
            corrected_boundary_values=values,
        )


def test_protocol_self_attested_restore_cannot_be_promoted_to_research_evidence():
    snapshot = _contract_snapshot()
    passed = RestoreAudit(
        snapshot.boundary_id, "passed", snapshot.digest, snapshot.digest, 0.0,
        evidence_kind="self_attested",
    )
    assert passed.passed
    assert not passed.research_eligible
    oracle_audit = StateAuditOracle().audit(
        snapshot, snapshot.to_dict(), evidence_kind="self_attested", require_captured_state=True
    )
    assert not oracle_audit.research_eligible
