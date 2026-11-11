"""Checkpoint contracts for the driver-owned PolySeg-Sci state."""

import copy

import pytest
import torch

from verl.trainer.ppo.polyseg_sci import (
    BudgetLedger,
    DurableBudgetLedger,
    BudgetedQuerySampler,
    BoundarySnapshot,
    EndpointObservation,
    EndpointSchema,
    EndpointSpec,
    FeatureMapContract,
    FidelitySpec,
    FidelityFeatureMapContract,
    PolynomialVectorPredictor,
    QueryCandidate,
    ScienceCheckpointState,
    ScienceDependenceState,
    PolySegScienceRuntime,
    prior_rollout_records_from_extra_fields,
    run_science_round,
)


def _schema() -> EndpointSchema:
    def fidelity(endpoint: str, level: str, cost: float) -> FidelitySpec:
        return FidelitySpec(f"{endpoint}-{level.lower()}", level, f"eval-{endpoint}", f"{level.lower()}-v1", cost, kind="gold" if level == "HF" else "proxy")

    endpoints = tuple(
        EndpointSpec(
            endpoint_id,
            endpoint_id,
            "unit",
            hf_fidelity_id=f"{endpoint_id}-hf",
            fidelities=(fidelity(endpoint_id, "LF", 1.0), fidelity(endpoint_id, "HF", 3.0)),
        )
        for endpoint_id in ("a", "b")
    )
    return EndpointSchema(endpoints, objective_mode="preference", weights=(0.5, 0.5))


def test_predictor_restore_reproduces_predictions_and_rejects_feature_drift():
    schema = _schema()
    feature_map = FeatureMapContract.fit([[0.0], [1.0], [0.5], [0.25]], ["progress"])
    predictor = PolynomialVectorPredictor(schema, feature_map, q=1, ridge_lambda=1e-3)
    features = [[0.0], [1.0], [0.5], [0.25]]
    predictor.fit(features, [[1.0, 2.0], [2.0, 4.0], [1.5, 3.0], [1.25, 2.5]], groups=["x", "y", "x", "y"], cross_fit=True)
    state = predictor.state_dict()
    restored = PolynomialVectorPredictor.from_state_dict(state)
    assert torch.allclose(restored.predict(features), predictor.predict(features))
    changed_map = FeatureMapContract.fit([[0.0], [2.0]], ["progress"])
    with pytest.raises(ValueError, match="feature-map fingerprint"):
        PolynomialVectorPredictor.from_state_dict(state, feature_map=changed_map)


def test_science_checkpoint_roundtrip_preserves_ledger_propensity_and_dual_free_state():
    schema = _schema()
    feature_map = FeatureMapContract.fit([[0.0], [1.0]], ["progress"])
    predictor = PolynomialVectorPredictor(schema, feature_map, q=0)
    predictor.fit([[0.0], [1.0]], [[1.0, 2.0], [2.0, 4.0]])
    sampler = BudgetedQuerySampler()
    candidate = QueryCandidate("boundary-0", "a", 1.0, 2.0, 1.0, 1.0)
    sampler.sample([candidate], budget=1.0, seed=7)
    ledger = BudgetLedger(10.0)
    ledger.record("query", 1.0, planned=1.0, reference="boundary-0::a")
    state = ScienceCheckpointState(
        schema=schema,
        feature_map=feature_map,
        predictor=predictor,
        sampler=sampler,
        ledger=ledger,
        dependency_state={"hac": [1.0]},
        rng_state={"python": (3, (1, 2, 3), None)},
        heldout_manifest={"tasks": ["heldout-1"]},
        evaluator_hashes={"a": "hf-a-v1"},
        sidecar_location="audit/round.json",
    )
    restored = ScienceCheckpointState.from_state_dict(copy.deepcopy(state.state_dict()))
    assert restored.fingerprint == state.fingerprint
    assert restored.complete is False  # in-memory/non-held-out state cannot earn research resume status
    assert restored.ledger.spent == 1.0
    assert len(restored.sampler.history) == 1
    assert restored.predictor is not None
    tampered = state.state_dict()
    tampered["feature_map"] = dict(tampered["feature_map"], lower=[-2.0])
    with pytest.raises(ValueError):
        ScienceCheckpointState.from_state_dict(tampered)


def test_durable_budget_ledger_is_atomic_and_reference_idempotent(tmp_path):
    path = tmp_path / "science-ledger.json"
    first = DurableBudgetLedger(3.0, path, ledger_id="run-1")
    first.commit("query", 1.0, planned=1.0, reference="query-1")
    # A fresh process object observes the persisted state and a retry does not
    # charge the same operation twice.
    second = DurableBudgetLedger(3.0, path, ledger_id="run-1")
    second.commit("query", 1.0, planned=1.0, reference="query-1")
    assert second.spent == 1.0
    assert len(second.events) == 1
    restored = DurableBudgetLedger.from_state_dict(second.state_dict())
    assert restored.spent == 1.0
    with pytest.raises(ValueError, match="different amount"):
        restored.commit("query", 2.0, reference="query-1")


def test_dependence_state_keeps_trajectory_units_and_hac_diagnostics():
    state = ScienceDependenceState(hac_lag=2, block_size=2, bootstrap_repetitions=100, bootstrap_seed=3)
    state.append("trajectory-a", [[1.0, 2.0], [2.0, 3.0], [1.5, 2.5]])
    state.append("trajectory-a", [[1.75, 2.75]])
    state.append("trajectory-b", [[4.0, 5.0], [5.0, 6.0]])
    diagnostics = state.diagnostics()
    assert diagnostics["trajectory_count"] == 2
    assert diagnostics["segment_count"] == 6
    assert len(diagnostics["kappa_seg_upper"]) == 2
    assert diagnostics["bootstrap_repetitions"] == 100
    restored = ScienceDependenceState.from_state_dict(state.state_dict())
    assert restored.diagnostics()["cluster_ess"] == diagnostics["cluster_ess"]
    with pytest.raises(ValueError, match="trajectory_id"):
        ScienceDependenceState(trajectories=[{"trajectory_id": "", "residuals": [[1]], "weights": [1]}])


def test_science_runtime_requires_fit_and_restores_complete_checkpoint(tmp_path):
    schema = _schema()
    feature_map = FeatureMapContract.fit([[0.0], [1.0]], ["progress"])
    runtime = PolySegScienceRuntime(
        schema,
        feature_map,
        DurableBudgetLedger(5.0, tmp_path / "ledger.json", ledger_id="runtime"),
        heldout_manifest={"split": "heldout-v1"},
        evaluator_hashes={"a": "hf-a-v1", "b": "hf-b-v1"},
        sidecar_location="sidecar.json",
    )
    with pytest.raises(RuntimeError, match="fitted predictor"):
        runtime.prepare_round({"boundary_features": [[0.5]]})
    runtime.fit_predictor([[0.0], [1.0]], [[1.0, 2.0], [2.0, 4.0]])
    prepared = runtime.prepare_round({"boundary_features": [[0.5]]})
    assert prepared["boundary_predictions"].shape == (1, 2)
    checkpoint = runtime.checkpoint_state()
    assert checkpoint.complete
    restored = PolySegScienceRuntime.from_checkpoint(checkpoint.state_dict())
    assert torch.allclose(restored.predict([[0.5]]), runtime.predict([[0.5]]))


def test_checkpoint_envelope_restores_explicit_fidelity_feature_map(tmp_path):
    schema = _schema()
    feature_map = FeatureMapContract.fit([[0.0], [1.0]], ["progress"])
    slots = ("a::a-lf", "b::b-lf")
    fidelity_map = FidelityFeatureMapContract.fit(
        [[0.1, 20.0], [0.8, 60.0]],
        [[True, True], [True, True]],
        slots,
    )
    runtime = PolySegScienceRuntime(
        schema,
        feature_map,
        DurableBudgetLedger(10.0, tmp_path / "ledger.json", ledger_id="fidelity-envelope"),
        fidelity_feature_map=fidelity_map,
        heldout_manifest={"split": "heldout-v1"},
        evaluator_hashes={"a": "hf-a-v1", "b": "hf-b-v1"},
        sidecar_location="sidecar.json",
    )
    runtime.fit_predictor(
        [[0.0], [1.0]],
        [[1.0, 2.0], [2.0, 4.0]],
        fidelity_values=[[0.1, 20.0], [0.8, 60.0]],
        fidelity_mask=[[True, True], [True, True]],
    )
    state = runtime.checkpoint_state()
    assert state.fidelity_feature_map is not None
    assert state.state_dict()["fidelity_feature_map"]["slot_ids"] == list(slots)
    restored = ScienceCheckpointState.from_state_dict(state.state_dict())
    assert restored.fidelity_feature_map is not None
    assert restored.fidelity_feature_map.fingerprint == fidelity_map.fingerprint

    legacy = state.state_dict()
    legacy.pop("fidelity_feature_map")
    restored_legacy = ScienceCheckpointState.from_state_dict(legacy)
    assert restored_legacy.fidelity_feature_map is not None


def test_runtime_ingests_prior_rollouts_without_heldout_leakage_and_restores_rows(tmp_path):
    schema = _schema()
    feature_map = FeatureMapContract.fit([[0.0], [1.0]], ["progress"])
    runtime = PolySegScienceRuntime(
        schema,
        feature_map,
        DurableBudgetLedger(10.0, tmp_path / "ledger.json", ledger_id="prior-runtime"),
        heldout_manifest={"trajectory_ids": ["heldout-traj"], "task_ids": ["heldout-task"]},
        evaluator_hashes={"a": "hf-a-v1", "b": "hf-b-v1"},
        sidecar_location="sidecar.json",
    )
    records = [
        {
            "trajectory_id": "train-traj-a",
            "task_id": "train-task-a",
            "split": "train",
            "boundary_features": [[0.0], [0.25]],
            "hf_targets": [[1.0, 2.0], [1.25, 2.5]],
        },
        {
            "trajectory_id": "train-traj-b",
            "task_id": "train-task-b",
            "split": "prior",
            "boundary_features": [[0.75], [1.0]],
            "hf_targets": [[1.75, 3.5], [2.0, 4.0]],
        },
    ]
    predictor = runtime.fit_from_prior_rollouts(records, cross_fit=True)
    assert predictor.frozen
    assert runtime.prior_data_manifest["sample_count"] == 4
    assert runtime.prior_data_manifest["trajectory_count"] == 2
    assert runtime.prior_data_manifest["cross_fit"] is True
    with pytest.raises(ValueError, match="held-out manifest"):
        runtime.ingest_prior_batch(
            [{
                "trajectory_id": "heldout-traj",
                "task_id": "heldout-task",
                "split": "train",
                "boundary_features": [[0.5]],
                "hf_targets": [[1.5, 3.0]],
            }]
        )
    restored = PolySegScienceRuntime.from_checkpoint(runtime.checkpoint_state().state_dict())
    assert restored.prior_data_manifest["sample_count"] == 4
    assert torch.allclose(restored.predict([[0.5]]), runtime.predict([[0.5]]))


def test_prior_rollout_transport_adapter_reconstructs_hf_observations_and_rejects_ineligible():
    hf_a = {
        "endpoint_id": "a",
        "fidelity_id": "a-hf",
        "value": 1.25,
        "status": "observed",
        "evaluator_id": "eval-a",
        "evaluator_version": "hf-v1",
    }
    hf_b = {
        "endpoint_id": "b",
        "fidelity_id": "b-hf",
        "value": 2.5,
        "status": "observed",
        "evaluator_id": "eval-b",
        "evaluator_version": "hf-v1",
    }
    records = prior_rollout_records_from_extra_fields(
        [
            {
                "trajectory_id": "traj-transport",
                "task_id": "task-transport",
                "split": "train",
                "science_research_eligible": True,
                "boundary_features": [[0.1], [0.2]],
                "terminal_endpoint_observations": [hf_a, hf_b],
            }
        ]
    )
    assert len(records) == 1
    assert records[0]["observation_broadcast"] is True
    assert len(records[0]["observations"]) == 2
    assert all(isinstance(item, EndpointObservation) for item in records[0]["observations"])
    with pytest.raises(ValueError, match="science_research_eligible"):
        prior_rollout_records_from_extra_fields(
            [
                {
                    "trajectory_id": "traj-ineligible",
                    "task_id": "task-transport",
                    "split": "train",
                    "science_research_eligible": False,
                    "boundary_features": [[0.1]],
                    "terminal_endpoint_observations": [hf_a, hf_b],
                }
            ]
        )


def test_prior_rollout_lf_mf_observations_feed_predictor_with_explicit_masks(tmp_path):
    schema = _schema()
    feature_map = FeatureMapContract.fit([[0.0], [1.0]], ["progress"])
    runtime = PolySegScienceRuntime(
        schema,
        feature_map,
        DurableBudgetLedger(20.0, tmp_path / "ledger.json", ledger_id="lf-mf-runtime"),
    )

    def row(traj, progress, a_hf, b_hf, a_lf, b_lf):
        return {
            "trajectory_id": traj,
            "task_id": traj,
            "split": "train",
            "science_research_eligible": True,
            "boundary_features": [[progress]],
            "terminal_endpoint_observations": [
                {"endpoint_id": "a", "fidelity_id": "a-lf", "value": a_lf, "status": "observed", "evaluator_id": "eval-a", "evaluator_version": "lf-v1"},
                {"endpoint_id": "b", "fidelity_id": "b-lf", "value": b_lf, "status": "observed", "evaluator_id": "eval-b", "evaluator_version": "lf-v1"},
                {"endpoint_id": "a", "fidelity_id": "a-hf", "value": a_hf, "status": "observed", "evaluator_id": "eval-a", "evaluator_version": "hf-v1"},
                {"endpoint_id": "b", "fidelity_id": "b-hf", "value": b_hf, "status": "observed", "evaluator_id": "eval-b", "evaluator_version": "hf-v1"},
            ],
        }

    records = [
        row("lf-mf-0", 0.0, 0.2, 20.0, 0.1, 10.0),
        row("lf-mf-1", 1.0, 0.8, 80.0, 0.7, 70.0),
    ]
    runtime.ingest_prior_rollout_extra_fields(records, fit=True)
    assert runtime.predictor.fidelity_feature_map is not None
    assert runtime.predictor.fidelity_feature_map.slot_ids == (
        "a::a-lf",
        "b::b-lf",
    )
    prepared = runtime.prepare_round(
        {
            "boundary_ids": ("b0",),
            "boundary_features": [[0.5]],
            "fidelity_features": [[0.4, 40.0]],
            "fidelity_feature_mask": [[True, True]],
            "initial_boundary_features": [[0.0]],
            "initial_fidelity_features": [[0.1, 10.0]],
            "initial_fidelity_mask": [[True, True]],
            "terminal_boundary_features": [[1.0]],
            "terminal_boundary_fidelity_features": [[0.9, 90.0]],
            "terminal_boundary_fidelity_mask": [[True, True]],
        }
    )
    assert prepared["boundary_predictions"].shape == (3, 2)
    assert "fidelity_features" not in prepared
    assert "fidelity_feature_mask" not in prepared
    assert "initial_fidelity_features" not in prepared
    assert "initial_fidelity_mask" not in prepared
    assert "terminal_boundary_fidelity_features" not in prepared
    assert "terminal_boundary_fidelity_mask" not in prepared


def test_prior_rollout_transport_deduplicates_broadcast_outputs_and_rejects_new_rows_after_fit(tmp_path):
    schema = _schema()
    feature_map = FeatureMapContract.fit([[0.0], [1.0]], ["progress"])
    row = {
        "trajectory_id": "broadcast-traj",
        "task_id": "broadcast-task",
        "split": "prior",
        "science_research_eligible": True,
        "boundary_features": [[0.0], [1.0]],
        "terminal_endpoint_observations": [
            {
                "endpoint_id": "a",
                "fidelity_id": "a-hf",
                "value": 1.0,
                "status": "observed",
                "evaluator_id": "eval-a",
                "evaluator_version": "hf-v1",
            },
            {
                "endpoint_id": "b",
                "fidelity_id": "b-hf",
                "value": 2.0,
                "status": "observed",
                "evaluator_id": "eval-b",
                "evaluator_version": "hf-v1",
            },
        ],
    }
    # The AgentLoop attaches one envelope to every output.  Transport-level
    # deduplication must preserve one scientific sample rather than count the
    # same terminal evaluator twice.
    assert len(prior_rollout_records_from_extra_fields([row, copy.deepcopy(row)])) == 1
    runtime = PolySegScienceRuntime(
        schema,
        feature_map,
        DurableBudgetLedger(10.0, tmp_path / "ledger.json", ledger_id="dedupe-runtime"),
    )
    runtime.ingest_prior_rollout_extra_fields([row, copy.deepcopy(row)], fit=True)
    assert runtime.prior_data_manifest["sample_count"] == 2
    # Replaying the same aggregated batch is idempotent even when the caller
    # still asks for fitting; it must not refit or append rows.
    runtime.ingest_prior_rollout_extra_fields([row, copy.deepcopy(row)], fit=True)
    assert runtime.prior_data_manifest["sample_count"] == 2
    changed = copy.deepcopy(row)
    changed["trajectory_id"] = "new-traj"
    with pytest.raises(RuntimeError, match="new prior rollout rows"):
        runtime.ingest_prior_rollout_extra_fields([changed], fit=False)


def test_runtime_refreshes_frozen_predictor_between_training_iterations(tmp_path):
    schema = _schema()
    feature_map = FeatureMapContract.fit([[0.0], [1.0]], ["progress"])
    runtime = PolySegScienceRuntime(
        schema,
        feature_map,
        DurableBudgetLedger(10.0, tmp_path / "refresh-ledger.json", ledger_id="refresh-runtime"),
    )
    runtime.fit_from_prior_rollouts(
        [
            {
                "trajectory_id": "iteration-0",
                "task_id": "task-0",
                "split": "train",
                "boundary_features": [[0.0], [1.0]],
                "hf_targets": [[0.0, 0.0], [1.0, 10.0]],
            }
        ]
    )
    first_predictor = runtime.predictor
    runtime.refresh_from_prior_rollouts(
        [
            {
                "trajectory_id": "iteration-1",
                "task_id": "task-1",
                "split": "train",
                "boundary_features": [[0.5]],
                "hf_targets": [[8.0, 80.0]],
            }
        ]
    )
    assert runtime.predictor is not first_predictor
    assert runtime.prior_data_manifest["sample_count"] == 3
    assert runtime.prior_data_manifest["batch_count"] == 2
    assert runtime.prior_data_manifest["trajectory_ids"] == ["iteration-0", "iteration-1"]
    with pytest.raises(RuntimeError, match="refresh a predictor"):
        PolySegScienceRuntime(
            schema,
            feature_map,
            DurableBudgetLedger(10.0, tmp_path / "unfitted-ledger.json", ledger_id="unfitted-refresh"),
        ).refresh_from_prior_rollouts([])


def test_runtime_ingests_serialized_rollout_extra_fields_with_hf_provenance(tmp_path):
    schema = _schema()
    feature_map = FeatureMapContract.fit([[0.0], [1.0]], ["progress"])
    runtime = PolySegScienceRuntime(
        schema,
        feature_map,
        DurableBudgetLedger(10.0, tmp_path / "ledger.json", ledger_id="transport-runtime"),
        heldout_manifest={"trajectory_ids": ["heldout-traj"]},
        evaluator_hashes={"a": "hf-v1", "b": "hf-v1"},
    )
    extra_fields = [
        {
            "trajectory_id": "train-transport",
            "task_id": "task-transport",
            "split": "prior",
            "science_research_eligible": True,
            "boundary_features": [[0.0], [1.0]],
            "terminal_endpoint_observations": [
                {
                    "endpoint_id": "a",
                    "fidelity_id": "a-hf",
                    "value": 1.0,
                    "status": "observed",
                    "evaluator_id": "eval-a",
                    "evaluator_version": "hf-v1",
                },
                {
                    "endpoint_id": "b",
                    "fidelity_id": "b-hf",
                    "value": 2.0,
                    "status": "observed",
                    "evaluator_id": "eval-b",
                    "evaluator_version": "hf-v1",
                },
            ],
        }
    ]
    runtime.ingest_prior_rollout_extra_fields(extra_fields, fit=True)
    assert runtime.fitted
    assert runtime.prior_data_manifest["sample_count"] == 2
    assert runtime.prior_data_manifest["trajectory_ids"] == ["train-transport"]


def test_runtime_prepare_and_science_round_use_one_fitted_prior_predictor(tmp_path):
    schema = _schema()
    feature_map = FeatureMapContract.fit([[0.0], [1.0]], ["progress"])
    runtime = PolySegScienceRuntime(
        schema,
        feature_map,
        DurableBudgetLedger(20.0, tmp_path / "ledger.json", ledger_id="e2e-runtime"),
        heldout_manifest={"trajectory_ids": ["heldout"]},
        evaluator_hashes={"a": "hf-a-v1", "b": "hf-b-v1"},
        sidecar_location="sidecar.json",
    )
    runtime.fit_from_prior_rollouts([
        {
            "trajectory_id": "train-a",
            "task_id": "task-a",
            "split": "train",
            "boundary_features": [[0.0], [0.5], [1.0]],
            "hf_targets": [[1.0, 2.0], [1.5, 3.0], [2.0, 4.0]],
        }
    ])
    snapshot = BoundarySnapshot(
        "task", "traj", "run", "session", "b0", "prefix", (True,), (-1.0,), (), {}, {},
        "rng", 10.0, 8.0, "policy-v1", "tok-v1", {}, "fm", "fm-hash", "seg-v1", "snap0"
    )
    candidates = (
        QueryCandidate("b0", "a", 1.0, 1.0, 1.0, 1.0),
        QueryCandidate("b0", "b", 1.0, 1.0, 1.0, 1.0),
    )
    prepared = runtime.prepare_round({
        "boundary_ids": ("b0",),
        "boundary_features": [[0.5]],
        "initial_boundary_features": [[0.0]],
        "terminal_boundary_features": [[1.0]],
    }, policy_version="policy-v1")
    assert prepared["boundary_predictions"].shape == (3, 2)

    def restore(value):
        return {**value.to_dict(), "snapshot_digest": value.digest, "artifact_hashes": value.workspace_hashes}

    def branch(value, rng_id, remaining):
        return {
            "branch_id": f"{value.boundary_id}:fresh:{rng_id}",
            "policy_version": "policy-v1",
            "rng_id": rng_id,
            "rng_independent": True,
            "execution_budget_remaining": remaining,
            "cost": 1.0,
            "observations": (
                EndpointObservation("a", "a-hf", 1.5, evaluator_id="eval-a", evaluator_version="hf-v1"),
                EndpointObservation("b", "b-hf", 3.0, evaluator_id="eval-b", evaluator_version="hf-v1"),
            ),
        }

    result = run_science_round(
        **prepared,
        boundary_snapshots={"b0": snapshot},
        boundary_intervals=[(0, 2)],
        terminal_values=[2.0, 4.0],
        segment_scores=torch.ones((2, 1)),
        candidates=candidates,
        query_budget=2.0,
        branch_fn=branch,
        restore_fn=restore,
        branch_count=1,
        planned_branch_cost=1.0,
        terminal_evaluator_ids={"a": "eval-a", "b": "eval-b"},
        terminal_evaluator_versions={"a": "hf-v1", "b": "hf-v1"},
        response_model_validated=True,
    )
    runtime.record_round(result)
    assert result.verification.passed
    assert result.corrected_boundary_values.shape == (3, 2)
    assert runtime.checkpoint_state().complete
