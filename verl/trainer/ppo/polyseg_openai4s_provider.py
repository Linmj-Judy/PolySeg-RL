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
import importlib
import inspect
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from verl.trainer.ppo.polyseg_sci import (
    PolySegScienceRuntime,
    load_feature_map_contract,
    load_fidelity_feature_map_contract,
    run_science_round,
)
from verl.trainer.ppo.polyseg_sci_provider import (
    ScienceRoundRequest,
    ScienceRoundResponse,
)


class OpenAI4SProviderError(RuntimeError):
    """Raised when the rollout has not supplied a real OpenAI4S round spec."""


def _round_spec(request: ScienceRoundRequest) -> Mapping[str, Any] | None:
    context = request.context
    candidate = context.get("openai4s_science_round") if isinstance(context, Mapping) else None
    if candidate is None:
        extra_fields = getattr(request.output, "extra_fields", {})
        if isinstance(extra_fields, Mapping):
            candidate = extra_fields.get("openai4s_science_round")
    if candidate is None:
        return None
    if not isinstance(candidate, Mapping):
        raise OpenAI4SProviderError(
            "OpenAI4S provider requires a rollout-owned openai4s_science_round "
            "spec; no evaluator, checkpoint, or branch callback is inferred"
        )
    return candidate


def _load_factory(path: str, config: Any) -> Any | None:
    """Load an optional harness-owned round factory.

    The factory is deliberately outside this repository's OpenAI4S adapter:
    it is the place where a task harness binds session IDs, CAS/checkpoint
    handles, policy rollout metadata, and evaluator callbacks.  It may expose
    either ``factory(request)`` or an object with ``build_science_round``.
    """

    path = str(path or "").strip()
    if not path:
        return None
    module_name, separator, attribute = path.partition(":")
    if not separator:
        module_name, _, attribute = path.rpartition(".")
    if not module_name or not attribute:
        raise OpenAI4SProviderError("science round_factory_path must be module:attribute or module.attribute")
    try:
        module = importlib.import_module(module_name)
        factory = getattr(module, attribute)
    except (ImportError, AttributeError) as exc:
        raise OpenAI4SProviderError(f"science round factory {path!r} was not found") from exc
    if callable(factory):
        try:
            parameters = tuple(inspect.signature(factory).parameters.values())
        except (TypeError, ValueError):
            parameters = ()
        first = next(
            (
                parameter
                for parameter in parameters
                if parameter.kind in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
            ),
            None,
        )
        # A plain request factory is called later with ScienceRoundRequest.
        # Constructor factories use an explicit config-like parameter name;
        # this avoids invoking a request factory once with the wrong object.
        if first is not None and first.name.lower() in {"request", "science_request", "round_request"}:
            return factory
        accepts_constructor_config = bool(parameters) or any(
            parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in parameters
        )
        constructed = factory(config) if accepts_constructor_config else factory()
        if callable(constructed) or callable(getattr(constructed, "build_science_round", None)):
            return constructed
        # A one-shot request factory may accept any object, including the
        # trainer config, and return a mapping immediately.  In that case the
        # imported callable itself is the request factory; do not retain the
        # first mapping as a provider-wide round spec.
        if isinstance(constructed, Mapping):
            return factory
        return constructed
    if callable(getattr(factory, "build_science_round", None)):
        return factory
    raise OpenAI4SProviderError(
        f"science round factory {path!r} must be callable or expose build_science_round"
    )


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
        science_config = _science_config(config)
        self.round_factory = _load_factory(
            _config_value(science_config, "round_factory_path", ""),
            config,
        )
        # This reference is bound only for driver-owned execution.  Worker
        # providers can still receive an explicit per-request runtime, but do
        # not silently construct mutable driver state themselves.
        self.runtime: PolySegScienceRuntime | None = None

    def bind_runtime(self, runtime: PolySegScienceRuntime | None) -> None:
        if runtime is not None and not isinstance(runtime, PolySegScienceRuntime):
            raise TypeError("OpenAI4S provider runtime must be PolySegScienceRuntime")
        self.runtime = runtime

    async def _resolve_round_spec(self, request: ScienceRoundRequest) -> Mapping[str, Any]:
        explicit = _round_spec(request)
        if explicit is not None:
            return explicit
        if self.round_factory is None:
            raise OpenAI4SProviderError(
                "OpenAI4S provider requires an explicit round spec or configured round_factory_path"
            )
        factory = self.round_factory
        builder = getattr(factory, "build_science_round", None)
        if callable(builder):
            candidate = builder(request)
        else:
            candidate = factory(request)
        if inspect.isawaitable(candidate):
            candidate = await candidate
        if not isinstance(candidate, Mapping):
            raise OpenAI4SProviderError(
                "science round factory must return a mapping round specification"
            )
        return candidate

    async def execute(self, request: ScienceRoundRequest) -> ScienceRoundResponse:
        spec = await self._resolve_round_spec(request)
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
        expected_fidelity_hash = str(_config_value(science_config, "fidelity_feature_map_hash", "") or "")
        fidelity_feature_path = str(_config_value(science_config, "fidelity_feature_map_path", "") or "")
        if expected_fidelity_hash and not fidelity_feature_path:
            raise OpenAI4SProviderError(
                "fidelity_feature_map_hash requires fidelity_feature_map_path"
            )
        if fidelity_feature_path:
            try:
                declared_fidelity_map = load_fidelity_feature_map_contract(
                    fidelity_feature_path,
                    expected_hash=expected_fidelity_hash or None,
                )
            except (OSError, ValueError) as exc:
                raise OpenAI4SProviderError(
                    f"invalid configured fidelity feature map contract: {exc}"
                ) from exc
            expected_fidelity_hash = declared_fidelity_map.fingerprint
        if expected_fidelity_hash:
            runtime_candidate = spec.get("runtime")
            actual_fidelity = getattr(
                getattr(adapter, "fidelity_feature_map", None), "fingerprint", ""
            ) or getattr(getattr(runtime_candidate, "fidelity_feature_map", None), "fingerprint", "")
            if actual_fidelity != expected_fidelity_hash:
                raise OpenAI4SProviderError(
                    "OpenAI4S adapter fidelity feature map hash does not match configured hash"
                )
        action_mask = spec.get("action_mask")
        boundaries = spec.get("boundaries")
        if action_mask is None or boundaries is None:
            raise OpenAI4SProviderError("round spec must contain OpenAI4S causal action_mask and boundaries")
        round_kwargs = spec.get("round_kwargs")
        if not isinstance(round_kwargs, Mapping):
            raise OpenAI4SProviderError("round spec must contain run_science_round keyword arguments")
        round_kwargs = dict(round_kwargs)
        # Make the v5 query contract executable when a harness leaves these
        # controls out of its per-round spec. Explicit round values remain
        # authoritative for preregistered sweeps.
        if science_config is not None:
            round_kwargs.setdefault(
                "query_mode", _config_value(science_config, "query_mode", "uniform")
            )
            configured_hard_budget = _config_value(science_config, "hard_budget", False)
            configured_budget_mode = _config_value(science_config, "budget_mode", "expected")
            if not isinstance(configured_hard_budget, bool):
                raise OpenAI4SProviderError("science hard_budget must be boolean")
            if configured_budget_mode not in {"hard", "expected"}:
                raise OpenAI4SProviderError("science budget_mode must be hard or expected")
            round_kwargs.setdefault(
                "hard_budget", configured_hard_budget or configured_budget_mode == "hard"
            )
            round_kwargs.setdefault(
                "min_inclusion_probability",
                _config_value(science_config, "min_inclusion_probability", 0.0),
            )
            round_kwargs.setdefault(
                "branch_count", _config_value(science_config, "branch_count", 1)
            )
        runtime = spec.get("runtime") or self.runtime
        if runtime is not None:
            if not isinstance(runtime, PolySegScienceRuntime):
                raise OpenAI4SProviderError("round spec runtime must be PolySegScienceRuntime")
            try:
                round_kwargs = runtime.prepare_round(round_kwargs, policy_version=round_kwargs.get("policy_version"))
            except (TypeError, ValueError, RuntimeError) as exc:
                raise OpenAI4SProviderError(f"science runtime refused round: {exc}") from exc
            runtime_fidelity_hash = getattr(runtime.fidelity_feature_map, "fingerprint", "")
            if expected_fidelity_hash and runtime_fidelity_hash != expected_fidelity_hash:
                raise OpenAI4SProviderError(
                    "configured fidelity feature map does not match the runtime contract"
                )
        # Durable ledgers use reference-key idempotency.  A stable round ID is
        # therefore required once a provider is used for more than one
        # trajectory; otherwise every round would reuse the legacy
        # ``science_round:*`` references and a retry/next batch could be
        # mistaken for a duplicate event.
        round_id = spec.get("round_id")
        if round_id is None and isinstance(request.context, Mapping):
            round_id = request.context.get("round_id")
        if round_id is None:
            round_id = spec.get("trajectory_id")
        if round_id is not None:
            if isinstance(round_id, (str, bytes)) and str(round_id):
                round_kwargs.setdefault("round_id", str(round_id))
            else:
                raise OpenAI4SProviderError("round_id must be a non-empty string when supplied")
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
        if runtime is not None:
            runtime.record_round(result)
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
        # Prior predictor supervision is owned by the task harness.  Never
        # manufacture it from corrected predictions or from a proxy score.
        # When the harness supplies explicit training-only records, preserve
        # them in the rollout envelope for the driver-owned runtime to ingest
        # after the batch is aggregated.
        prior_records = spec.get("prior_rollout_records")
        if prior_records is not None:
            if not isinstance(prior_records, Sequence) or isinstance(prior_records, (str, bytes)):
                raise OpenAI4SProviderError("prior_rollout_records must be a sequence of explicit training records")
            fields["science_prior_rollouts"] = [dict(record) for record in prior_records]
        elif spec.get("emit_terminal_prior_record") is True:
            # This opt-in path turns the independent terminal HF evaluator
            # result into explicit prior supervision for the next driver
            # round. It is never inferred from corrected predictions, and the
            # broadcast-to-boundaries semantics are recorded so predictor
            # fitting cannot silently treat it as a fresh boundary label.
            trajectory_id = spec.get("trajectory_id")
            task_id = spec.get("task_id")
            split = spec.get("prior_split")
            if not isinstance(trajectory_id, str) or not trajectory_id:
                raise OpenAI4SProviderError(
                    "emit_terminal_prior_record requires a non-empty trajectory_id"
                )
            if not isinstance(task_id, str) or not task_id:
                raise OpenAI4SProviderError(
                    "emit_terminal_prior_record requires a non-empty task_id"
                )
            if not isinstance(split, str) or split.lower() not in {"train", "training", "prior", "prior_train"}:
                raise OpenAI4SProviderError(
                    "emit_terminal_prior_record requires prior_split=train/prior"
                )
            terminal_values = round_kwargs.get("terminal_values")
            terminal_ids = round_kwargs.get("terminal_evaluator_ids", {})
            terminal_versions = round_kwargs.get("terminal_evaluator_versions", {})
            if not isinstance(terminal_values, Sequence) or isinstance(terminal_values, (str, bytes)):
                raise OpenAI4SProviderError("terminal_values are required for terminal prior supervision")
            if len(terminal_values) != dimension:
                raise OpenAI4SProviderError("terminal_values do not match endpoint dimension")
            observations = []
            for index, endpoint in enumerate(schema.endpoints):
                value = terminal_values[index]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise OpenAI4SProviderError("terminal prior values must be finite numeric endpoint values")
                evaluator_id = terminal_ids.get(endpoint.endpoint_id)
                evaluator_version = terminal_versions.get(endpoint.endpoint_id)
                if not isinstance(evaluator_id, str) or not isinstance(evaluator_version, str):
                    raise OpenAI4SProviderError(
                        "terminal prior supervision requires registered evaluator provenance"
                    )
                observations.append(
                    {
                        "endpoint_id": endpoint.endpoint_id,
                        "fidelity_id": endpoint.hf_fidelity_id,
                        "value": float(value),
                        "status": "observed",
                        "evaluator_id": evaluator_id,
                        "evaluator_version": evaluator_version,
                    }
                )
            fields["science_prior_rollouts"] = [
                {
                    "trajectory_id": trajectory_id,
                    "task_id": task_id,
                    "split": split,
                    "is_heldout": bool(spec.get("is_heldout", False)),
                    "science_research_eligible": bool(requested_eligible),
                    "boundary_features": round_kwargs.get("boundary_features"),
                    "fidelity_features": round_kwargs.get(
                        "boundary_fidelity_features", round_kwargs.get("fidelity_features")
                    ),
                    "fidelity_feature_mask": round_kwargs.get(
                        "boundary_fidelity_mask", round_kwargs.get("fidelity_feature_mask")
                    ),
                    "terminal_endpoint_observations": observations,
                }
            ]
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
        # The provider may have attached explicit prior supervision before the
        # independent correction/restore audit was finalized.  Propagate the
        # final eligibility decision into every nested record so the driver
        # cannot ingest a row marked eligible merely because the harness's
        # initial request asked for eligibility.  Keep the records copied and
        # immutable from the caller's point of view.
        prior_payload = fields.get("science_prior_rollouts")
        if prior_payload is not None:
            if not isinstance(prior_payload, Sequence) or isinstance(prior_payload, (str, bytes)):
                raise OpenAI4SProviderError("science_prior_rollouts must be a sequence of mappings")
            normalized_prior = []
            for index, record in enumerate(prior_payload):
                if not isinstance(record, Mapping):
                    raise OpenAI4SProviderError(
                        f"science_prior_rollouts[{index}] must be a mapping"
                    )
                copied = dict(record)
                copied["science_research_eligible"] = bool(eligible)
                normalized_prior.append(copied)
            fields["science_prior_rollouts"] = normalized_prior
        return ScienceRoundResponse(fields=fields, audit=audit, research_eligible=eligible)


def factory(config: Any = None) -> OpenAI4SScienceRoundProvider:
    """Factory suitable for ``algorithm.polyseg.science.provider_path``."""

    return OpenAI4SScienceRoundProvider(config)


__all__ = ["OpenAI4SProviderError", "OpenAI4SScienceRoundProvider", "factory"]
