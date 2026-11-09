"""OpenAI4S to PolySeg-Sci v4 adapter.

The science primitives intentionally do not import OpenAI4S.  This module is
the narrow integration boundary: it consumes the durable action-ledger and
checkpoint projections exposed by OpenAI4S (or an equivalent test double),
and emits only the immutable records understood by ``polyseg_sci``.  The
callbacks are deliberately explicit so a failed restore, fork, rollout, or
evaluator remains a typed audit event instead of being converted into a fake
scientific label.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any, Protocol

import torch

from verl.trainer.ppo.polyseg_sci import (
    ActionRecord,
    BoundarySnapshot,
    CausalSegmenter,
    EndpointObservation,
    EndpointSchema,
    EvaluatorExecutionError,
    BranchExecutionError,
    FeatureMapContract,
    FreshBranchRunner,
    RESTORE_EVIDENCE_KINDS,
    RestoreAudit,
    SegmentBoundary,
    StateAuditOracle,
    TrajectoryRecord,
    canonical_json,
    content_hash,
)


OPENAI4S_ADAPTER_VERSION = "polyseg-sci-openai4s-adapter-v2"
RESTORE_CONTRACT_VERSION = "openai4s-checkpoint-fork-v2"


class CheckpointFn(Protocol):
    def __call__(self, root_frame_id: str, **kwargs: Any) -> Mapping[str, Any]: ...


class RestoreFn(Protocol):
    def __call__(self, snapshot: BoundarySnapshot) -> Mapping[str, Any] | None: ...


class ForkFn(Protocol):
    def __call__(self, root_frame_id: str, **kwargs: Any) -> Mapping[str, Any]: ...


class OpenAI4SAdapterError(RuntimeError):
    """Raised when an OpenAI4S record cannot satisfy the science contract."""


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first(mapping: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return default


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _strict_real(value: Any, name: str) -> float:
    """Accept only finite numeric scalars at the adapter boundary."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise OpenAI4SAdapterError(f"{name} must be a finite real number")
    number = float(value)
    if not math.isfinite(number):
        raise OpenAI4SAdapterError(f"{name} must be a finite real number")
    return number


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise OpenAI4SAdapterError(f"{name} must be boolean")
    return value


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise OpenAI4SAdapterError(f"{name} must be an integer >= {minimum}")
    return value


def _events(group: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    raw = group.get("events", ())
    if isinstance(raw, Mapping):
        raw = (raw,)
    return tuple(item for item in raw if isinstance(item, Mapping))


def _event_action_id(event: Mapping[str, Any], group: Mapping[str, Any]) -> str | None:
    value = _first(event, "action_id", "tool_call_id", "call_id", "event_id")
    event_type = str(_first(event, "type", "event_type", default="")).lower()
    # These are lifecycle/telemetry records, not actions.  Falling back to a
    # group id for them would create phantom policy/tool actions and duplicate
    # segment accounting.
    if value is None and event_type in {"model_call", "telemetry", "completed", "failed", "cancelled"}:
        return None
    if value is None:
        value = _first(group, "group_id", "action_group_id")
    return None if value is None else str(value)


def _tool_name(event: Mapping[str, Any]) -> str | None:
    value = _first(event, "tool_name", "name")
    args = _mapping(_first(event, "canonical_arguments", "arguments", default={}))
    if value is None:
        value = _first(args, "tool_name", "name")
    return None if value is None else str(value)


def _artifact_refs(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        value = _first(value, "artifact_refs", "artifacts", "artifact_ids", default=())
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(str(item) for item in value if item is not None)


def _event_cost(event: Mapping[str, Any], group: Mapping[str, Any]) -> float:
    raw = _first(event, "execution_cost", "cost", "cost_usd")
    if raw is None:
        raw = _first(group, "execution_cost", "cost", "cost_usd", default=0.0)
    try:
        value = _strict_real(raw, "action-ledger cost")
    except OpenAI4SAdapterError as exc:
        raise OpenAI4SAdapterError(f"invalid action-ledger cost: {raw!r}") from exc
    if value < 0:
        raise OpenAI4SAdapterError(f"invalid action-ledger cost: {raw!r}")
    return value


def _event_timestamp(event: Mapping[str, Any], group: Mapping[str, Any]) -> float | None:
    value = _first(event, "timestamp", "created_at", "created_at_ms")
    if value is None:
        value = _first(group, "timestamp", "created_at", "created_at_ms")
    if value is None:
        return None
    # OpenAI4S storage uses integer epoch milliseconds in some projections.
    number = _strict_real(value, "action-ledger timestamp")
    if abs(number) > 1e11:
        number /= 1000.0
    if not math.isfinite(number):
        raise OpenAI4SAdapterError("action-ledger timestamp is not finite")
    return number


class OpenAI4STrajectoryAdapter:
    """Convert OpenAI4S ledger/checkpoint projections into science records.

    ``groups`` are the dictionaries returned by
    ``Store.list_action_groups(..., include_events=True)``.  Token-level policy
    records are supplied by the rollout worker because OpenAI4S deliberately
    stores provider wire messages, not the tokenizer-specific token alignment
    needed by PPO.  Tool records are recovered from the same groups and are
    always marked ``is_policy_action=False``.
    """

    def __init__(
        self,
        schema: EndpointSchema,
        feature_map: FeatureMapContract,
        *,
        segmenter: CausalSegmenter,
        state_oracle: StateAuditOracle | None = None,
        restore_contract_version: str = RESTORE_CONTRACT_VERSION,
    ) -> None:
        self.schema = schema
        self.feature_map = feature_map
        self.segmenter = segmenter
        self.state_oracle = state_oracle or StateAuditOracle()
        self.restore_contract_version = restore_contract_version

    @staticmethod
    def _policy_records(policy_actions: Sequence[Mapping[str, Any]], token_count: int | None) -> list[ActionRecord]:
        if token_count is not None:
            _strict_int(token_count, "token_count", minimum=1)
        records: list[ActionRecord] = []
        for index, raw in enumerate(policy_actions):
            action = _mapping(raw)
            start = _strict_int(_first(action, "token_start", "start", default=index), "policy token_start")
            end = _strict_int(_first(action, "token_end", "end", default=start + 1), "policy token_end")
            # A policy action must cover at least one response token.  Tool
            # lifecycle records are allowed to use a 0:0 span, but those are
            # emitted separately and are never passed through this method.
            if start < 0 or end <= start or (token_count is not None and end > token_count):
                raise OpenAI4SAdapterError(f"policy token span is outside response: {start}:{end}")
            action_id = str(_first(action, "action_id", "id", default=f"policy:{start}:{end}"))
            records.append(
                ActionRecord(
                    action_id=action_id,
                    action_type=str(_first(action, "action_type", "type", default="policy_token")),
                    is_policy_action=True,
                    token_start=start,
                    token_end=end,
                    logprob=_first(action, "logprob"),
                    old_logprob=_first(action, "old_logprob"),
                    observation_ref=_first(action, "observation_ref"),
                    artifact_refs=_artifact_refs(action),
                    execution_cost=_event_cost(action, {}),
                    remaining_budget=_first(action, "remaining_budget"),
                    timestamp=_event_timestamp(action, {}),
                    action_mask=True,
                )
            )
        ordered = sorted(records, key=lambda item: (item.token_start, item.token_end, item.action_id))
        for previous, current in zip(ordered, ordered[1:]):
            if previous.token_end > current.token_start:
                raise OpenAI4SAdapterError(
                    f"policy action spans overlap: {previous.action_id!r} and {current.action_id!r}"
                )
        return records

    def adapt_action_ledger(
        self,
        groups: Sequence[Mapping[str, Any]],
        *,
        policy_actions: Sequence[Mapping[str, Any]],
        token_count: int | None = None,
    ) -> tuple[ActionRecord, ...]:
        """Return policy-token plus deduplicated tool action records.

        A proposed tool call and its later result are one logical action.  The
        result is merged into the proposed record, so tool observations cannot
        accidentally increase the policy-action count used for segmentation.
        """

        records = self._policy_records(policy_actions, token_count)
        tool_rows: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for group in groups:
            group_map = _mapping(group)
            for event in _events(group_map):
                event_type = str(_first(event, "type", "event_type", default="")).lower()
                # Lifecycle records without an action identity are telemetry,
                # but a provider may encode a tool result/failure as a
                # ``completed``/``failed`` event carrying the original call
                # id.  Keep those records so the call and return remain one
                # auditable tool action.
                if event_type in {"model_call", "telemetry", "completed", "failed", "cancelled"} and _event_action_id(event, group_map) is None:
                    continue
                action_id = _event_action_id(event, group_map)
                if action_id is None:
                    continue  # model_call/telemetry events are not actions
                row = tool_rows.setdefault(action_id, {"action_id": action_id})
                if action_id not in order:
                    order.append(action_id)
                event_type = event_type or "tool"
                if event_type in {"proposed", "call", "tool_call"}:
                    row["action_type"] = "tool_call"
                    row["tool_name"] = _tool_name(event)
                    row["tool_call_id"] = _first(event, "tool_call_id", "action_id")
                if event_type in {"result", "returned", "outcome", "completed", "tool_result"}:
                    row["action_type"] = row.get("action_type", "tool_call")
                    row["tool_return_id"] = _first(event, "tool_return_id", "wire_id", "event_id")
                    row["observation_ref"] = _first(event, "observation_ref", "result_ref", "event_id")
                row["artifact_refs"] = tuple(dict.fromkeys((*row.get("artifact_refs", ()), *_artifact_refs(event))))
                # ``cost_usd`` on an action group is provider billing, not a
                # per-tool execution charge.  Only an event's explicit
                # execution cost belongs on the action record; otherwise a
                # proposed event and its result would count the group charge
                # twice.
                row["execution_cost"] = row.get("execution_cost", 0.0) + _event_cost(event, {})
                timestamp = _event_timestamp(event, group_map)
                if timestamp is not None:
                    row["timestamp"] = timestamp
                remaining = _first(event, "remaining_budget", "execution_budget_remaining")
                if remaining is not None:
                    row["remaining_budget"] = remaining
        for action_id in order:
            row = tool_rows[action_id]
            records.append(
                ActionRecord(
                    action_id=action_id,
                    action_type=str(row.get("action_type", "tool")),
                    is_policy_action=False,
                    token_start=0,
                    token_end=0,
                    tool_name=row.get("tool_name"),
                    tool_call_id=row.get("tool_call_id"),
                    tool_return_id=row.get("tool_return_id"),
                    observation_ref=row.get("observation_ref"),
                    artifact_refs=tuple(row.get("artifact_refs", ())),
                    execution_cost=float(row.get("execution_cost", 0.0)),
                    remaining_budget=row.get("remaining_budget"),
                    timestamp=row.get("timestamp"),
                    action_mask=False,
                )
            )
        return tuple(records)

    @staticmethod
    def action_mask(actions: Sequence[ActionRecord], token_count: int) -> tuple[bool, ...]:
        _strict_int(token_count, "token_count", minimum=1)
        mask = [False] * token_count
        for action in actions:
            if not action.is_policy_action:
                continue
            if action.token_end > token_count:
                raise OpenAI4SAdapterError("policy action exceeds token_count")
            for position in range(action.token_start, action.token_end):
                mask[position] = True
        if not any(mask):
            raise OpenAI4SAdapterError("OpenAI4S ledger contains no policy actions")
        return tuple(mask)

    def segment_actions(
        self,
        actions: Sequence[ActionRecord],
        *,
        token_count: int,
        tool_return_positions: Sequence[int] | None = None,
        event_positions: Sequence[int] | None = None,
    ) -> tuple[tuple[bool, ...], tuple[SegmentBoundary, ...]]:
        """Derive token-aligned causal boundaries from the ledger records."""

        mask = self.action_mask(actions, token_count)
        boundaries = tuple(
            self.segmenter.segment(
                mask,
                tool_return_positions=tool_return_positions,
                event_positions=event_positions,
            )
        )
        return mask, boundaries

    def science_rollout_fields(
        self,
        *,
        action_mask: Sequence[bool],
        boundaries: Sequence[SegmentBoundary],
        corrected_boundary_values: Any,
        endpoint_mask: Any | None = None,
    ) -> dict[str, torch.Tensor]:
        """Build the tensor fields consumed by ``polyseg_sci`` in ``verl``.

        Values are supplied by the frozen predictor/residual-correction stage;
        this method only validates shape and assigns each policy token to its
        causal segment.  It never turns missing endpoint values into zeros.
        """

        values = torch.as_tensor(corrected_boundary_values, dtype=torch.float32)
        if values.ndim != 2 or values.shape[0] != len(boundaries) + 1 or values.shape[1] != self.schema.dimension:
            raise OpenAI4SAdapterError(
                "corrected_boundary_values must have shape [segments+1, endpoint_dimension]"
            )
        if not torch.isfinite(values).all():
            raise OpenAI4SAdapterError("science boundary values must be finite after correction")
        ordered_boundaries = tuple(boundaries)
        expected_ids = tuple(range(len(ordered_boundaries)))
        if tuple(boundary.segment_id for boundary in ordered_boundaries) != expected_ids:
            raise OpenAI4SAdapterError("causal boundary segment IDs must be unique and contiguous")
        previous_end = -1
        for boundary in ordered_boundaries:
            if boundary.start < previous_end:
                raise OpenAI4SAdapterError("causal boundaries overlap")
            previous_end = boundary.end
        token_mask = torch.as_tensor(action_mask)
        if token_mask.dtype != torch.bool or token_mask.ndim != 1:
            raise OpenAI4SAdapterError("action_mask must be a boolean token vector")
        segment_ids = torch.full((token_mask.numel(),), -1, dtype=torch.long)
        for boundary in ordered_boundaries:
            if boundary.end > token_mask.numel():
                raise OpenAI4SAdapterError("causal boundary exceeds token-aligned action mask")
            positions = torch.arange(boundary.start, boundary.end, dtype=torch.long)
            positions = positions[token_mask[positions]]
            if positions.numel() == 0:
                raise OpenAI4SAdapterError("boundary has no policy action tokens")
            segment_ids[positions] = boundary.segment_id
        if torch.any(token_mask & (segment_ids < 0)):
            raise OpenAI4SAdapterError("policy action token is not covered by a causal boundary")
        if endpoint_mask is None:
            observed_mask = torch.ones_like(values, dtype=torch.bool)
        else:
            observed_mask = torch.as_tensor(endpoint_mask)
            if observed_mask.dtype != torch.bool:
                raise OpenAI4SAdapterError("science endpoint mask must contain booleans")
            if observed_mask.shape != values.shape:
                raise OpenAI4SAdapterError("science endpoint mask must match boundary values")
            if not bool(observed_mask.all()):
                raise OpenAI4SAdapterError("incomplete endpoint vectors cannot enter the actor update")
        return {
            "segment_ids": segment_ids,
            "segment_mask": torch.ones(len(ordered_boundaries), dtype=torch.bool),
            "science_boundary_values": values,
            "science_boundary_mask": torch.ones(len(ordered_boundaries), dtype=torch.bool),
            "science_endpoint_mask": observed_mask,
        }

    def capture_boundary(
        self,
        *,
        root_frame_id: str,
        branch_id: str,
        boundary: SegmentBoundary,
        task_id: str,
        trajectory_id: str,
        run_id: str,
        session_id: str,
        prefix: str,
        action_mask: Sequence[bool],
        old_log_probs: Sequence[float],
        tool_ledger: Sequence[Mapping[str, Any]],
        state: Mapping[str, Any],
        checkpoint_fn: CheckpointFn,
        snapshot_cost: float = 0.0,
    ) -> BoundarySnapshot:
        """Create an immutable OpenAI4S checkpoint-backed boundary snapshot."""

        if len(action_mask) != len(old_log_probs):
            raise OpenAI4SAdapterError("action mask and old log probabilities must align")
        normalized_mask = tuple(_strict_bool(value, "action_mask entry") for value in action_mask)
        normalized_log_probs = tuple(_strict_real(value, "old_log_probs entry") for value in old_log_probs)
        snapshot_cost_value = _strict_real(snapshot_cost, "snapshot cost")
        if snapshot_cost_value < 0:
            raise OpenAI4SAdapterError("snapshot cost must be finite and non-negative")
        required = (
            "rng_state",
            "execution_budget_initial",
            "execution_budget_remaining",
            "policy_version",
            "tokenizer_version",
            "sampling_config",
            "workspace_hashes",
            "dependency_hashes",
        )
        missing = [key for key in required if key not in state]
        if missing:
            raise OpenAI4SAdapterError(f"checkpoint state is missing required fields: {missing}")
        initial = _strict_real(state["execution_budget_initial"], "execution_budget_initial")
        remaining = _strict_real(state["execution_budget_remaining"], "execution_budget_remaining")
        if not 0 <= remaining <= initial:
            raise OpenAI4SAdapterError("invalid execution budget in checkpoint state")
        checkpoint = _mapping(
            checkpoint_fn(
                root_frame_id,
                branch_id=branch_id,
                reason=f"polyseg_sci_boundary:{boundary.segment_id}",
                metadata={
                    "adapter_version": OPENAI4S_ADAPTER_VERSION,
                    "trajectory_id": trajectory_id,
                    "boundary_id": f"{trajectory_id}:b{boundary.segment_id}",
                    "feature_map_version": self.feature_map.version,
                    "feature_map_hash": self.feature_map.fingerprint,
                    "segment_rule_version": self.segmenter.rule_version,
                },
            )
        )
        snapshot_id = _first(checkpoint, "checkpoint_id", "snapshot_id", "id")
        if snapshot_id is None:
            raise OpenAI4SAdapterError("OpenAI4S checkpoint response has no immutable checkpoint_id")
        boundary_id = f"{trajectory_id}:b{boundary.segment_id}"
        return BoundarySnapshot(
            task_id=task_id,
            trajectory_id=trajectory_id,
            run_id=run_id,
            session_id=session_id,
            boundary_id=boundary_id,
            prefix=prefix,
            action_mask=normalized_mask,
            old_log_probs=normalized_log_probs,
            tool_ledger=tuple(dict(item) for item in tool_ledger),
            workspace_hashes=dict(_mapping(state["workspace_hashes"])),
            dependency_hashes=dict(_mapping(state["dependency_hashes"])),
            rng_state=str(state["rng_state"]),
            execution_budget_initial=initial,
            execution_budget_remaining=remaining,
            policy_version=str(state["policy_version"]),
            tokenizer_version=str(state["tokenizer_version"]),
            sampling_config=dict(_mapping(state["sampling_config"])),
            feature_map_version=self.feature_map.version,
            feature_map_hash=self.feature_map.fingerprint,
            segment_rule_version=self.segmenter.rule_version,
            snapshot_id=str(snapshot_id),
            snapshot_cost=snapshot_cost_value,
        )

    def restore_audit(
        self,
        snapshot: BoundarySnapshot,
        restore_fn: RestoreFn,
        *,
        cost: float = 0.0,
        evidence_kind: str = "unspecified",
    ) -> RestoreAudit:
        """Restore and audit one checkpoint; exceptions become failures.

        ``evidence_kind`` is a caller trust declaration, not proof: only a
        ``durable`` restore that passes every captured-state check is research
        eligible.  The protocol probe's snapshot echo is ``self_attested``.
        """

        if evidence_kind not in RESTORE_EVIDENCE_KINDS:
            raise OpenAI4SAdapterError("invalid restore evidence kind")
        try:
            cost_value = _strict_real(cost, "restore cost")
            if cost_value < 0:
                raise OpenAI4SAdapterError("restore cost must be non-negative")
        except OpenAI4SAdapterError:
            return RestoreAudit(snapshot.boundary_id, "failed", snapshot.digest, None, 0.0, ("invalid_restore_cost",), evidence_kind)
        try:
            restored = restore_fn(snapshot)
        except Exception as exc:  # noqa: BLE001 - failure is an audit datum
            return RestoreAudit(snapshot.boundary_id, "failed", snapshot.digest, None, max(0.0, cost_value), (f"restore_exception:{type(exc).__name__}",), evidence_kind)
        return self.state_oracle.audit(
            snapshot,
            restored,
            cost=cost_value,
            require_captured_state=True,
            evidence_kind=evidence_kind,
        )

    def fresh_branches(
        self,
        snapshot: BoundarySnapshot,
        *,
        restore_audit: RestoreAudit,
        branch_count: int,
        branch_fn: Callable[[BoundarySnapshot, str, float], Mapping[str, Any]],
        policy_version: str,
        seed: int = 0,
        require_continuation_provenance: bool = False,
    ):
        """Run the common fail-closed fresh-branch contract.

        ``require_continuation_provenance`` stays opt-in: the protocol probe
        deliberately restarts from the original prompt and remains a
        non-research branch, while research continuations must attest a
        ``restored_boundary`` provenance matching the captured prefix and
        frozen sampling configuration.
        """

        return FreshBranchRunner().run(
            snapshot,
            restore_audit,
            branch_count=branch_count,
            branch_fn=branch_fn,
            schema=self.schema,
            policy_version=policy_version,
            seed=seed,
            require_evaluator_provenance=True,
            require_continuation_provenance=require_continuation_provenance,
        )

    @staticmethod
    def openai4s_branch_callback(
        *,
        root_frame_id: str,
        snapshot: BoundarySnapshot,
        fork_fn: ForkFn,
        rollout_fn: Callable[[Mapping[str, Any], BoundarySnapshot, str, float], Mapping[str, Any]],
        evaluate_fn: Callable[[Mapping[str, Any]], Sequence[EndpointObservation]],
    ) -> Callable[[BoundarySnapshot, str, float], Mapping[str, Any]]:
        """Compose OpenAI4S fork -> rollout -> independent evaluator.

        ``rollout_fn`` receives the newly forked branch and must run the frozen
        policy with the supplied RNG id.  It returns an artifact/trajectory
        mapping; the evaluator is called only after that branch completes.
        Explicit exception classes preserve the failure taxonomy used by the
        independent verifier.
        """

        def callback(current: BoundarySnapshot, rng_id: str, remaining: float) -> Mapping[str, Any]:
            branch_id = f"{current.boundary_id}:fresh:{rng_id}"
            try:
                branch = _mapping(
                    fork_fn(
                        root_frame_id,
                        from_checkpoint_id=current.snapshot_id,
                        branch_id=branch_id,
                        name=f"polyseg-sci-{current.boundary_id}-{rng_id}",
                    )
                )
            except TimeoutError:
                raise
            except Exception as exc:  # noqa: BLE001 - classified by runner
                raise BranchExecutionError(f"fork failed: {type(exc).__name__}: {exc}") from exc
            if "branch_id" not in branch:
                raise BranchExecutionError("fork did not attest the requested fresh branch identity")
            returned_branch_id = branch["branch_id"]
            if str(returned_branch_id) != branch_id:
                raise BranchExecutionError("fork returned a branch identity different from the requested fresh branch")
            for checkpoint_key in ("base_checkpoint_id", "from_checkpoint_id"):
                if checkpoint_key in branch and branch[checkpoint_key] is not None:
                    if str(branch[checkpoint_key]) != str(current.snapshot_id):
                        raise BranchExecutionError("fork was not based on the requested boundary checkpoint")
            try:
                artifact = _mapping(rollout_fn(branch, current, rng_id, remaining))
            except TimeoutError:
                raise
            except Exception as exc:  # noqa: BLE001 - classified by runner
                raise BranchExecutionError(f"fresh rollout failed: {type(exc).__name__}: {exc}") from exc
            # Do not manufacture the attestations that FreshBranchRunner
            # requires.  The rollout worker must prove that it actually used
            # the frozen policy, the supplied child RNG, and the inherited
            # execution budget.  A callback that merely returns an artifact
            # is therefore a failed branch rather than an apparently valid
            # scientific continuation.
            if artifact.get("policy_version") != current.policy_version:
                raise BranchExecutionError("fresh rollout did not attest the frozen policy version")
            if artifact.get("rng_id") != rng_id:
                raise BranchExecutionError("fresh rollout did not attest the supplied independent RNG id")
            if artifact.get("rng_independent") is not True:
                raise BranchExecutionError("fresh rollout did not attest independent future randomness")
            if "execution_budget_remaining" not in artifact:
                raise BranchExecutionError("fresh rollout did not attest its remaining execution budget")
            try:
                branch_remaining = _strict_real(artifact["execution_budget_remaining"], "branch execution budget")
            except OpenAI4SAdapterError as exc:
                raise BranchExecutionError("fresh rollout returned an invalid remaining execution budget") from exc
            if not math.isfinite(branch_remaining) or not 0 <= branch_remaining <= remaining:
                raise BranchExecutionError("fresh rollout did not inherit a valid remaining execution budget")
            try:
                branch_cost = _strict_real(_first(artifact, "cost", "execution_cost", default=0.0), "branch cost")
                if branch_cost < 0:
                    raise OpenAI4SAdapterError("branch cost must be non-negative")
            except OpenAI4SAdapterError as exc:
                raise BranchExecutionError("fresh rollout returned an invalid branch cost") from exc
            # Reject an explicitly incomplete/truncated terminal artifact
            # *before* invoking the independent evaluator.  Absent fields are
            # left to the opt-in research-provenance check rather than being
            # fabricated into a completion claim.
            if "trajectory_complete" in artifact and artifact.get("trajectory_complete") is not True:
                raise BranchExecutionError("fresh rollout terminal trajectory is incomplete")
            for truncated_key in ("truncated", "generation_truncated"):
                if truncated_key in artifact and artifact.get(truncated_key) is not False:
                    raise BranchExecutionError("fresh rollout terminal observation is truncated")
            finish_reason = artifact.get("finish_reason")
            if finish_reason is not None and str(finish_reason).lower() in {
                "length",
                "max_tokens",
                "max_length",
                "truncated",
            }:
                raise BranchExecutionError("fresh rollout terminal observation hit the token limit")
            # Pass through caller-attested provenance verbatim (or an explicit
            # flat form).  Never synthesize the values FreshBranchRunner
            # compares against the captured boundary.
            provenance = artifact.get("continuation_provenance")
            if provenance is not None and not isinstance(provenance, Mapping):
                raise BranchExecutionError("fresh rollout continuation provenance must be a mapping")
            if provenance is None:
                explicit = {
                    key: artifact[key]
                    for key in (
                        "continuation_mode",
                        "prefix",
                        "prefix_hash",
                        "sampling_config",
                        "sampling_config_hash",
                    )
                    if key in artifact
                }
                provenance = explicit or None
            try:
                observations = tuple(evaluate_fn(artifact))
            except TimeoutError:
                raise
            except Exception as exc:  # noqa: BLE001 - classified by runner
                raise EvaluatorExecutionError(f"gold evaluator failed: {type(exc).__name__}: {exc}") from exc
            returned_result = {
                "branch_id": str(returned_branch_id),
                "policy_version": artifact["policy_version"],
                "rng_id": artifact["rng_id"],
                "rng_independent": artifact["rng_independent"],
                "execution_budget_remaining": branch_remaining,
                "cost": branch_cost,
                "observations": observations,
            }
            if provenance is not None:
                returned_result["continuation_provenance"] = dict(provenance)
            if "trajectory_complete" in artifact:
                returned_result["trajectory_complete"] = artifact["trajectory_complete"]
            return returned_result

        return callback

    def trajectory(
        self,
        *,
        metadata: Mapping[str, Any],
        terminal_observations: Sequence[EndpointObservation],
        actions: Sequence[ActionRecord],
        terminal_cost: float,
        restore_status: str = "not_attempted",
        restore_cost: float = 0.0,
        failure_class: str | None = None,
        research_eligible: bool = False,
    ) -> TrajectoryRecord:
        """Build a validated trajectory envelope from an OpenAI4S run."""

        observations = tuple(terminal_observations)
        observed_ids = tuple(observation.endpoint_id for observation in observations)
        if len(set(observed_ids)) != len(observed_ids) or set(observed_ids) != set(self.schema.endpoint_ids):
            missing = sorted(set(self.schema.endpoint_ids) - set(observed_ids))
            extra = sorted(set(observed_ids) - set(self.schema.endpoint_ids))
            raise OpenAI4SAdapterError(
                f"terminal endpoint vector is incomplete or has extra dimensions; missing={missing}, extra={extra}"
            )
        observations = tuple(
            sorted(observations, key=lambda observation: self.schema.endpoint_index(observation.endpoint_id))
        )
        for observation in observations:
            self.schema.validate_observation(observation, allow_empty_provenance=False)
            endpoint = self.schema.endpoints[self.schema.endpoint_index(observation.endpoint_id)]
            if observation.fidelity_id != endpoint.hf_fidelity_id:
                raise OpenAI4SAdapterError("terminal observations must use the registered HF fidelity")
        required = (
            "task_id", "trajectory_id", "prompt_id", "run_id", "session_id",
            "actor_checkpoint", "tokenizer_version", "policy_version",
            "sampling_config", "terminal_evaluator_id", "terminal_evaluator_version",
            "execution_budget_initial", "execution_budget_remaining", "trajectory_complete",
        )
        missing = [key for key in required if key not in metadata]
        if missing:
            raise OpenAI4SAdapterError(f"trajectory metadata is missing required fields: {missing}")
        # Truncated/completion metadata must never be wrapped as an "observed"
        # scientific terminal label.  The flag is explicit and fail-closed.
        if metadata.get("truncated") is True or metadata.get("generation_truncated") is True:
            raise OpenAI4SAdapterError("truncated terminal observations cannot be recorded as complete")
        finish_reason = metadata.get("finish_reason")
        if finish_reason is not None and str(finish_reason).lower() in {
            "length",
            "max_tokens",
            "max_length",
            "truncated",
        }:
            raise OpenAI4SAdapterError("terminal observations hit the generation token limit")
        if research_eligible is True and metadata.get("trajectory_complete") is not True:
            raise OpenAI4SAdapterError("research_eligible trajectories require trajectory_complete=True")
        terminal_evaluator_id = str(metadata["terminal_evaluator_id"])
        terminal_evaluator_version = str(metadata["terminal_evaluator_version"])
        declared_ids = _mapping(metadata.get("terminal_evaluator_ids"))
        declared_versions = _mapping(metadata.get("terminal_evaluator_versions"))
        if declared_ids or declared_versions:
            if set(declared_ids) != set(self.schema.endpoint_ids) or set(declared_versions) != set(self.schema.endpoint_ids):
                raise OpenAI4SAdapterError("endpoint-level terminal evaluator metadata is incomplete")
            for observation in observations:
                if (
                    str(declared_ids[observation.endpoint_id]) != observation.evaluator_id
                    or str(declared_versions[observation.endpoint_id]) != observation.evaluator_version
                ):
                    raise OpenAI4SAdapterError(
                        "terminal observation provenance does not match endpoint evaluator metadata"
                    )
        else:
            for observation in observations:
                if (
                    observation.evaluator_id != terminal_evaluator_id
                    or observation.evaluator_version != terminal_evaluator_version
                ):
                    raise OpenAI4SAdapterError(
                        "terminal observation provenance does not match trajectory evaluator metadata"
                    )
        return TrajectoryRecord(
            task_id=str(metadata["task_id"]),
            trajectory_id=str(metadata["trajectory_id"]),
            prompt_id=str(metadata["prompt_id"]),
            run_id=str(metadata["run_id"]),
            session_id=str(metadata["session_id"]),
            actor_checkpoint=str(metadata["actor_checkpoint"]),
            tokenizer_version=str(metadata["tokenizer_version"]),
            policy_version=str(metadata["policy_version"]),
            sampling_config=dict(_mapping(metadata["sampling_config"])),
            terminal_observations=observations,
            terminal_evaluator_id=terminal_evaluator_id,
            terminal_evaluator_version=terminal_evaluator_version,
            terminal_cost=_strict_real(terminal_cost, "terminal cost"),
            execution_budget_initial=_strict_real(metadata["execution_budget_initial"], "execution_budget_initial"),
            execution_budget_remaining=_strict_real(metadata["execution_budget_remaining"], "execution_budget_remaining"),
            restore_contract_version=self.restore_contract_version,
            restore_status=restore_status,
            restore_cost=_strict_real(restore_cost, "restore cost"),
            feature_map_version=self.feature_map.version,
            feature_map_hash=self.feature_map.fingerprint,
            segment_rule_version=self.segmenter.rule_version,
            trajectory_complete=_strict_bool(metadata["trajectory_complete"], "trajectory_complete"),
            failure_class=failure_class,
            research_eligible=_strict_bool(research_eligible, "research_eligible"),
            actions=tuple(actions),
            terminal_score=tuple(observation.value for observation in observations),
            segment_rule_hash=content_hash({"segment_rule_version": self.segmenter.rule_version}),
            terminal_evaluator_ids={
                observation.endpoint_id: observation.evaluator_id for observation in observations
            },
            terminal_evaluator_versions={
                observation.endpoint_id: observation.evaluator_version for observation in observations
            },
        )


class OpenAI4SStoreAdapter:
    """Small concrete facade over OpenAI4S ``Store`` and session domain.

    OpenAI4S keeps the Store and the filesystem-aware session domain as
    separate services.  This facade does not guess missing runtime state: it
    exposes the durable ledger/checkpoint/fork operations and leaves restore
    attestation to :meth:`OpenAI4STrajectoryAdapter.restore_audit`.
    """

    def __init__(self, store: Any, session_domain: Any, *, root_frame_id: str, branch_id: str | None = None) -> None:
        if not root_frame_id:
            raise ValueError("root_frame_id is required")
        if not callable(getattr(store, "list_action_groups", None)):
            raise TypeError("store must expose list_action_groups")
        if not callable(getattr(session_domain, "create_checkpoint", None)):
            raise TypeError("session_domain must expose create_checkpoint")
        if not callable(getattr(session_domain, "fork_branch", None)):
            raise TypeError("session_domain must expose fork_branch")
        self.store = store
        self.session_domain = session_domain
        self.root_frame_id = str(root_frame_id)
        self.branch_id = str(branch_id or root_frame_id)

    def action_groups(self, *, turn_id: str | None = None, after_ordinal: int | None = None) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            self.store.list_action_groups(
                self.root_frame_id,
                branch_id=self.branch_id,
                turn_id=turn_id,
                after_ordinal=after_ordinal,
                include_events=True,
            )
        )

    def create_checkpoint(self, root_frame_id: str | None = None, **kwargs: Any) -> Mapping[str, Any]:
        """Match the callback shape expected by ``capture_boundary``.

        OpenAI4S's domain service receives ``root_frame_id`` positionally,
        while this facade historically accepted only keyword arguments.  Keep
        the facade usable directly as a callback and reject cross-session use.
        """
        if root_frame_id is not None and str(root_frame_id) != self.root_frame_id:
            raise OpenAI4SAdapterError("checkpoint root_frame_id does not match the adapter")
        kwargs.pop("branch_id", None)
        return self.session_domain.create_checkpoint(
            self.root_frame_id,
            branch_id=self.branch_id,
            **kwargs,
        )

    def fork_branch(
        self,
        root_frame_id: str | None = None,
        *,
        from_checkpoint_id: str,
        branch_id: str,
        name: str | None = None,
    ) -> Mapping[str, Any]:
        """Match the callback shape expected by ``openai4s_branch_callback``."""
        if root_frame_id is not None and str(root_frame_id) != self.root_frame_id:
            raise OpenAI4SAdapterError("fork root_frame_id does not match the adapter")
        return self.session_domain.fork_branch(
            self.root_frame_id,
            from_checkpoint_id=from_checkpoint_id,
            branch_id=branch_id,
            name=name,
        )

    def checkpoint(self, checkpoint_id: str) -> Mapping[str, Any] | None:
        getter = getattr(self.store, "get_session_checkpoint", None)
        if not callable(getter):
            raise OpenAI4SAdapterError("Store does not expose get_session_checkpoint")
        return getter(checkpoint_id)


def write_sidecar(path: str | Path, payload: Mapping[str, Any]) -> str:
    """Atomically write one auditable JSON sidecar and return its SHA-256."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json(dict(payload)).encode("utf-8")
    digest = content_hash(dict(payload))
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return digest


__all__ = [
    "OPENAI4S_ADAPTER_VERSION",
    "RESTORE_CONTRACT_VERSION",
    "OpenAI4SAdapterError",
    "OpenAI4STrajectoryAdapter",
    "OpenAI4SStoreAdapter",
    "write_sidecar",
]
