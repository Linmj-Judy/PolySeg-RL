"""PolySeg-Sci provider bridge backed by the real OpenAI4S adapter.

OpenAI4S owns the durable Store, action ledger, SessionDomainService,
checkpoint/CAS and branch lifecycle.  This module does not replace those
services and does not invent endpoint values.  The rollout harness supplies a
fully constructed round specification containing the real OpenAI4S adapter,
its runtime callbacks, and the task-specific independent evaluator; this
bridge executes the existing PolySeg round and serializes its verified fields.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from verl.trainer.ppo.polyseg_sci import load_feature_map_contract, run_science_round
from verl.trainer.ppo.polyseg_sci_provider import (
    ScienceRoundRequest,
    ScienceRoundResponse,
)


class OpenAI4SProviderError(RuntimeError):
    """Raised when the rollout has not supplied a real OpenAI4S round spec."""


def _round_spec(request: ScienceRoundRequest) -> Mapping[str, Any]:
    context = request.context
    candidate = context.get("openai4s_science_round") if isinstance(context, Mapping) else None
    if candidate is None:
        extra_fields = getattr(request.output, "extra_fields", {})
        if isinstance(extra_fields, Mapping):
            candidate = extra_fields.get("openai4s_science_round")
    if not isinstance(candidate, Mapping):
        raise OpenAI4SProviderError(
            "OpenAI4S provider requires a rollout-owned openai4s_science_round "
            "spec; no evaluator, checkpoint, or branch callback is inferred"
        )
    return candidate


def _as_audit(result: Any) -> dict[str, Any]:
    verification = getattr(result, "verification", None)
    return {
        "provider": "openai4s",
        "ledger": dict(getattr(result, "ledger", {}) or {}),
        "audit": dict(getattr(result, "audit", {}) or {}),
        "verification": {
            "passed": bool(getattr(verification, "passed", False)),
            "reasons": list(getattr(verification, "reasons", ()) or ()),
            "checked_fields": list(getattr(verification, "checked_fields", ()) or ()),
        },
        "boundary_records": [record.to_dict() for record in getattr(result, "boundary_records", ())],
        "covariance_by_boundary": dict(getattr(result, "covariance_by_boundary", {}) or {}),
    }


def _config_value(source: Any, name: str, default: Any = None) -> Any:
    if source is None:
        return default
    try:
        value = source.get(name, default)
    except (AttributeError, TypeError):
        value = getattr(source, name, default)
    return default if value is None else value


def _science_config(config: Any) -> Any:
    algorithm = _config_value(config, "algorithm", config)
    polyseg = _config_value(algorithm, "polyseg", None)
    return _config_value(polyseg, "science", None)


class OpenAI4SScienceRoundProvider:
    """Execute a PolySeg round using real OpenAI4S-owned runtime objects.

    The spec is deliberately explicit because token spans/log-probs, training
    RNG/state restoration, and task-specific gold evaluation are owned by the
    rollout/task harness rather than by OpenAI4S's generic Store.  Required
    keys are the keyword arguments of :func:`run_science_round`, plus:

    ``adapter``
        An ``OpenAI4STrajectoryAdapter`` instance.
    ``action_mask`` / ``boundaries``
        Output of the adapter's real causal ledger segmentation.
    ``research_eligible`` (optional)
        Harness decision; never inferred from a successful callback alone.
    """

    def __init__(self, config: Any = None):
        self.config = config

    async def execute(self, request: ScienceRoundRequest) -> ScienceRoundResponse:
        spec = _round_spec(request)
        adapter = spec.get("adapter")
        if adapter is None or not callable(getattr(adapter, "science_rollout_fields", None)):
            raise OpenAI4SProviderError("round spec must contain an OpenAI4STrajectoryAdapter")
        science_config = _science_config(self.config)
        expected_schema_hash = str(_config_value(science_config, "endpoint_schema_hash", "") or "")
        if expected_schema_hash:
            actual_schema = getattr(getattr(adapter, "schema", None), "fingerprint", "")
            if actual_schema != expected_schema_hash:
                raise OpenAI4SProviderError("OpenAI4S adapter endpoint schema hash does not match configured hash")
        expected_feature_hash = str(_config_value(science_config, "feature_map_hash", "") or "")
        feature_map_path = str(_config_value(science_config, "feature_map_path", "") or "")
        if feature_map_path:
            try:
                declared_feature_map = load_feature_map_contract(
                    feature_map_path,
                    expected_hash=expected_feature_hash or None,
                )
            except (OSError, ValueError) as exc:
                raise OpenAI4SProviderError(f"invalid configured feature map contract: {exc}") from exc
            expected_feature_hash = declared_feature_map.fingerprint
        if expected_feature_hash:
            actual_feature = getattr(getattr(adapter, "feature_map", None), "fingerprint", "")
            if actual_feature != expected_feature_hash:
                raise OpenAI4SProviderError("OpenAI4S adapter feature map hash does not match configured hash")
        action_mask = spec.get("action_mask")
        boundaries = spec.get("boundaries")
        if action_mask is None or boundaries is None:
            raise OpenAI4SProviderError("round spec must contain OpenAI4S causal action_mask and boundaries")
        round_kwargs = spec.get("round_kwargs")
        if not isinstance(round_kwargs, Mapping):
            raise OpenAI4SProviderError("round spec must contain run_science_round keyword arguments")
        round_kwargs = dict(round_kwargs)
        if request.validate:
            raise OpenAI4SProviderError("OpenAI4S science provider is disabled for validation rollouts")

        # Every spec-level precondition is checked *before* run_science_round
        # so a malformed boundary or terminal contract cannot trigger
        # billable restore/branch/evaluator callbacks.
        boundary_ids = round_kwargs.get("boundary_ids")
        if not isinstance(boundary_ids, Sequence) or isinstance(boundary_ids, (str, bytes)):
            raise OpenAI4SProviderError("round spec boundary_ids must be a sequence")
        if len(boundaries) != len(boundary_ids) + 1:
            raise OpenAI4SProviderError(
                "OpenAI4S boundary contract mismatch: expected one more segment boundary than internal boundary id"
            )
        schema = getattr(adapter, "schema", None)
        dimension = getattr(schema, "dimension", None)
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
            raise OpenAI4SProviderError("adapter endpoint schema dimension is missing or invalid")
        requested_eligible = spec.get("research_eligible", False)
        if not isinstance(requested_eligible, bool):
            raise OpenAI4SProviderError("round spec research_eligible must be boolean")
        terminal_complete = spec.get("trajectory_complete", spec.get("terminal_complete"))
        truncated = spec.get("truncated", spec.get("generation_truncated"))
        finish_reason = spec.get("finish_reason")
        if terminal_complete is not None and not isinstance(terminal_complete, bool):
            raise OpenAI4SProviderError("round spec trajectory_complete must be boolean")
        if truncated is not None and not isinstance(truncated, bool):
            raise OpenAI4SProviderError("round spec truncated must be boolean")
        terminal_reasons: list[str] = []
        if terminal_complete is False:
            terminal_reasons.append("terminal_trajectory_incomplete")
        if truncated is True:
            terminal_reasons.append("terminal_truncated")
        if finish_reason is not None and str(finish_reason).lower() in {
            "length",
            "max_tokens",
            "max_length",
            "truncated",
        }:
            terminal_reasons.append("terminal_token_limit")
        if terminal_reasons:
            raise OpenAI4SProviderError(
                "OpenAI4S round terminal is incomplete: " + ",".join(terminal_reasons)
            )
        if requested_eligible:
            # Research eligibility requires explicit, non-echo evidence.  The
            # hashes/attestations below make the claim auditable; they do not
            # cryptographically prove the harness actually restored state.
            if terminal_complete is not True:
                raise OpenAI4SProviderError(
                    "research-eligible OpenAI4S round requires explicit trajectory_complete=True"
                )
            if spec.get("restore_evidence_kind", "unspecified") != "durable":
                raise OpenAI4SProviderError(
                    "research-eligible OpenAI4S round requires durable restore evidence"
                )
            if spec.get("continuation_mode", "unspecified") != "restored_boundary":
                raise OpenAI4SProviderError(
                    "research-eligible OpenAI4S round requires restored_boundary continuation"
                )
            round_kwargs["require_continuation_provenance"] = True

        # OpenAI4S callbacks can perform filesystem/CAS and evaluator work;
        # keep them off the AgentLoop event loop while preserving the exact
        # callback objects supplied by the harness.
        result = await asyncio.to_thread(run_science_round, **round_kwargs)
        if not result.verification.passed:
            raise OpenAI4SProviderError(
                "OpenAI4S PolySeg round failed independent verification: "
                + ";".join(result.verification.reasons)
            )

        corrected = torch.as_tensor(result.corrected_boundary_values)
        if corrected.ndim != 2 or corrected.shape[0] != len(boundaries) + 1:
            raise OpenAI4SProviderError(
                "corrected boundary values must have one initial row plus one row per segment"
            )
        if corrected.shape[1] != dimension:
            raise OpenAI4SProviderError("corrected boundary values do not match endpoint dimension")
        # run_science_round returns exactly one initial row plus one row per
        # segment.  Do not silently slice a malformed result into a plausible
        # but semantically misaligned actor-update tensor.
        corrected_for_rollout = corrected
        fields = adapter.science_rollout_fields(
            action_mask=action_mask,
            boundaries=boundaries,
            corrected_boundary_values=corrected_for_rollout,
            endpoint_mask=torch.ones_like(corrected_for_rollout, dtype=torch.bool),
        )
        audit = _as_audit(result)
        audit["science_gradient"] = torch.as_tensor(result.segment_gradient).detach().cpu().tolist()
        audit["openai4s_round_spec_id"] = str(spec.get("round_id", ""))
        audit["predictor_provenance"] = str(spec.get("predictor_provenance", ""))
        audit["feature_map_version"] = str(getattr(getattr(adapter, "feature_map", None), "version", ""))
        audit["feature_map_hash"] = str(getattr(getattr(adapter, "feature_map", None), "fingerprint", ""))
        audit["endpoint_schema_hash"] = str(getattr(getattr(adapter, "schema", None), "fingerprint", ""))
        round_audit = dict(audit.get("audit", {}) or {})
        selected_ids = tuple(str(value) for value in boundary_ids)
        restore_evidence_kinds = dict(round_audit.get("restore_evidence_kinds", {}) or {})
        continuation_eligible = dict(round_audit.get("continuation_research_eligible", {}) or {})
        audit["restore_evidence_kinds"] = restore_evidence_kinds
        audit["continuation_research_eligible"] = continuation_eligible
        audit["terminal_completeness"] = {
            "trajectory_complete": terminal_complete,
            "truncated": truncated,
            "finish_reason": finish_reason,
        }
        correction_audit = dict(round_audit.get("correction", {}) or {})
        correction_ready = correction_audit.get("unbiased_under_response_model") is True
        # A successful structural verifier is insufficient for a research
        # claim when response missingness/propensity has not been audited, the
        # restore is only a self-attested echo, or the branch did not continue
        # from the captured boundary under the frozen sampling configuration.
        # Preserve the round for engineering use, but downgrade eligibility
        # instead of allowing a harness flag to relabel it as evidence.
        eligible = requested_eligible
        if requested_eligible:
            if not (
                selected_ids
                and all(restore_evidence_kinds.get(key) == "durable" for key in selected_ids)
            ):
                audit["research_eligibility_rejection"] = "restore_evidence_not_durable"
                eligible = False
            elif not (
                selected_ids
                and all(continuation_eligible.get(key) is True for key in selected_ids)
            ):
                audit["research_eligibility_rejection"] = "continuation_provenance_not_verified"
                eligible = False
            elif not correction_ready:
                audit["research_eligibility_rejection"] = "response_model_not_validated"
                eligible = False
        return ScienceRoundResponse(fields=fields, audit=audit, research_eligible=eligible)


def factory(config: Any = None) -> OpenAI4SScienceRoundProvider:
    """Factory suitable for ``algorithm.polyseg.science.provider_path``."""

    return OpenAI4SScienceRoundProvider(config)


__all__ = ["OpenAI4SProviderError", "OpenAI4SScienceRoundProvider", "factory"]
