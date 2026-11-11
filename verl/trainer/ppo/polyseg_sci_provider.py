"""Production seam for executing PolySeg-Sci rounds before PPO advantage math.

The provider deliberately lives outside the advantage estimator.  Rollout and
reward code supplies a frozen trajectory representation and the provider
returns already verified transport fields.  A configured provider may execute
OpenAI4S or another external harness; the default path is disabled and never
changes legacy PolySeg behavior.
"""

from __future__ import annotations

import asyncio
import importlib
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import torch

from verl.trainer.ppo.polyseg_sci import load_endpoint_schema


SCIENCE_FIELD_NAMES = (
    "segment_ids",
    "segment_mask",
    "science_boundary_values",
    "science_boundary_mask",
    "science_endpoint_mask",
)


@dataclass(frozen=True)
class ScienceRoundRequest:
    """Immutable request envelope passed to a configured science harness."""

    output: Any
    outputs: tuple[Any, ...]
    validate: bool
    context: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.validate, bool):
            raise ValueError("validate must be boolean")
        if not isinstance(self.context, Mapping):
            raise ValueError("science provider context must be a mapping")


@dataclass(frozen=True)
class ScienceRoundResponse:
    """Provider result before it is copied into AgentLoop extra fields."""

    fields: Mapping[str, Any]
    audit: Mapping[str, Any] = field(default_factory=dict)
    research_eligible: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.research_eligible, bool):
            raise ValueError("research_eligible must be boolean")
        missing = [name for name in SCIENCE_FIELD_NAMES if name not in self.fields]
        if missing:
            raise ValueError(f"science provider response is missing fields: {missing}")
        values = torch.as_tensor(self.fields["science_boundary_values"])
        if values.ndim != 2 or values.shape[0] < 2 or not torch.isfinite(values).all():
            raise ValueError("science provider boundary values must be finite [segments+1, endpoints]")
        endpoint_mask = torch.as_tensor(self.fields["science_endpoint_mask"])
        boundary_mask = torch.as_tensor(self.fields["science_boundary_mask"]).reshape(-1)
        segment_mask = torch.as_tensor(self.fields["segment_mask"]).reshape(-1)
        segment_ids = torch.as_tensor(self.fields["segment_ids"]).reshape(-1)
        if endpoint_mask.dtype != torch.bool or boundary_mask.dtype != torch.bool or segment_mask.dtype != torch.bool:
            raise ValueError("science provider masks must contain booleans")
        if segment_ids.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
            raise ValueError("science provider segment_ids must contain integers")
        if endpoint_mask.shape != values.shape:
            raise ValueError("science provider endpoint mask must match boundary values")
        if boundary_mask.numel() != values.shape[0] - 1:
            raise ValueError("science provider boundary mask must match segment axis")
        if segment_mask.numel() != values.shape[0] - 1:
            raise ValueError("science provider segment_mask must match segment axis")
        if torch.any(boundary_mask & ~segment_mask):
            raise ValueError("active science boundaries require active segment_mask entries")
        if torch.any(boundary_mask[:, None] & ~endpoint_mask[:-1]) or torch.any(
            boundary_mask[:, None] & ~endpoint_mask[1:]
        ):
            raise ValueError("active science boundaries require complete endpoint vectors")
        if segment_ids.numel() == 0 or torch.any(segment_ids < -1):
            raise ValueError("science provider segment_ids contain invalid values")
        if torch.any(segment_ids >= values.shape[0] - 1):
            raise ValueError("science provider segment_ids exceed the macro segment axis")
        if not isinstance(self.audit, Mapping):
            raise ValueError("science provider audit must be a mapping")


class ScienceRoundProvider(Protocol):
    """Async provider protocol implemented by an external science harness."""

    async def execute(self, request: ScienceRoundRequest) -> ScienceRoundResponse:
        """Run one frozen-policy science round and return verified fields."""


class DisabledScienceRoundProvider:
    """Provider used when PolySeg-Sci is disabled."""

    async def execute(self, request: ScienceRoundRequest) -> ScienceRoundResponse:
        raise RuntimeError("disabled science provider must not be executed")


class CallbackScienceRoundProvider:
    """Adapter for tests and harnesses exposing a sync or async callback."""

    def __init__(self, callback: Any):
        if not callable(callback):
            raise TypeError("science provider callback must be callable")
        self.callback = callback

    async def execute(self, request: ScienceRoundRequest) -> ScienceRoundResponse:
        result = self.callback(request)
        if asyncio.iscoroutine(result):
            result = await result
        if isinstance(result, ScienceRoundResponse):
            return _enforce_research_eligibility(result)
        if not isinstance(result, Mapping):
            raise TypeError("science provider callback must return ScienceRoundResponse or a mapping")
        fields = result.get("fields", result)
        audit = result.get("audit", {}) if isinstance(result, Mapping) else {}
        eligible = result.get("research_eligible", False) if isinstance(result, Mapping) else False
        # A callback is an integration seam, not an independent scientific
        # verifier.  Keep the explicit harness decision, but require the
        # response-missingness audit before allowing it to become research
        # evidence.  This mirrors the OpenAI4S provider's fail-closed gate
        # for custom providers as well.
        return _enforce_research_eligibility(
            ScienceRoundResponse(fields=fields, audit=audit, research_eligible=eligible)
        )


def _correction_audit(audit: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the correction audit regardless of provider envelope shape."""

    direct = audit.get("correction")
    if isinstance(direct, Mapping):
        return direct
    nested = audit.get("audit")
    if isinstance(nested, Mapping):
        correction = nested.get("correction")
        if isinstance(correction, Mapping):
            return correction
    return audit


def _enforce_research_eligibility(response: ScienceRoundResponse) -> ScienceRoundResponse:
    """Downgrade an unaudited harness claim before it reaches rollout fields."""

    if not response.research_eligible:
        return response
    if _correction_audit(response.audit).get("unbiased_under_response_model") is True:
        return response
    audit = dict(response.audit)
    audit.setdefault("research_eligibility_rejection", "response_model_not_validated")
    return ScienceRoundResponse(fields=response.fields, audit=audit, research_eligible=False)


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


def science_execution_owner(config: Any) -> str:
    """Return the configured owner of mutable science-round execution.

    The default remains ``worker`` for compatibility.  A driver-owned runtime
    must opt into ``driver`` so one manager, rather than each Ray worker,
    mutates query-sampler, dependence, and budget state.
    """

    owner = str(_config_value(_science_config(config), "execution_owner", "worker") or "worker")
    if owner not in {"worker", "driver"}:
        raise ValueError("PolySeg-Sci execution_owner must be worker or driver")
    return owner


def science_enabled(config: Any) -> bool:
    algorithm = _config_value(config, "algorithm", config)
    estimator = str(_config_value(algorithm, "adv_estimator", ""))
    science = _science_config(config)
    enabled = _config_value(science, "enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("science enabled flag must be boolean")
    return estimator == "polyseg_sci" or enabled


def load_science_round_provider(config: Any) -> ScienceRoundProvider | None:
    """Load the configured provider, failing closed for enabled science runs."""

    if not science_enabled(config):
        return None
    science = _science_config(config)
    path = str(_config_value(science, "provider_path", "") or "").strip()
    if not path:
        raise ValueError(
            "PolySeg-Sci is enabled but algorithm.polyseg.science.provider_path is empty; "
            "production rollout must provide a ScienceRoundProvider"
        )
    schema_path = str(_config_value(science, "endpoint_schema_path", "") or "").strip()
    if not schema_path:
        raise ValueError(
            "PolySeg-Sci is enabled but algorithm.polyseg.science.endpoint_schema_path is empty; "
            "a pre-registered endpoint schema is required"
        )
    schema = load_endpoint_schema(schema_path)
    configured_hash = str(_config_value(science, "endpoint_schema_hash", "") or "")
    if configured_hash and configured_hash != schema.fingerprint:
        raise ValueError("science endpoint_schema_hash does not match endpoint_schema_path")
    module_name, separator, attribute = path.partition(":")
    if not separator:
        module_name, _, attribute = path.rpartition(".")
    if not module_name or not attribute:
        raise ValueError("science provider_path must be module:attribute or module.attribute")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute, None)
    if factory is None:
        raise ValueError(f"science provider factory {path!r} was not found")
    provider = factory(config) if callable(factory) else factory
    if not hasattr(provider, "execute") or not callable(provider.execute):
        raise TypeError(f"science provider {path!r} does not implement async execute(request)")
    return provider


def attach_science_response(outputs: Sequence[Any], response: ScienceRoundResponse) -> None:
    """Attach one verified envelope to every persisted output in a trajectory."""

    # ScienceRoundResponse validates this field at construction.  Preserve the
    # validated value without a second truthiness conversion at the transport
    # boundary.
    eligible = response.research_eligible
    audit = dict(response.audit)
    if eligible and _correction_audit(audit).get("unbiased_under_response_model") is not True:
        eligible = False
        audit.setdefault("research_eligibility_rejection", "response_model_not_validated")
    for output in outputs:
        extra_fields = getattr(output, "extra_fields", None)
        if not isinstance(extra_fields, dict):
            raise TypeError("agent output must expose a mutable extra_fields mapping")
        for key, value in response.fields.items():
            if isinstance(value, np.ndarray):
                value = np.array(value, copy=True)
            elif isinstance(value, torch.Tensor):
                value = value.detach().cpu().clone()
            extra_fields[key] = value
        extra_fields["science_round_audit"] = audit
        extra_fields["science_research_eligible"] = eligible


def attach_science_response_to_batch(batch: Any, index: int, response: ScienceRoundResponse) -> None:
    """Attach one driver-executed response to a row of an aggregated batch.

    Ray workers normally attach fields to ``AgentLoopOutput`` before padding.
    Driver-owned execution happens after worker aggregation, where only the
    serialized ``extra_fields`` object array is available.  Keep this helper
    explicit so a driver cannot accidentally mutate a different row or lose
    the final eligibility/audit envelope.
    """

    non_tensor = getattr(batch, "non_tensor_batch", None)
    if not isinstance(non_tensor, Mapping):
        raise TypeError("driver science attachment requires batch.non_tensor_batch")
    extras = non_tensor.get("extra_fields")
    if extras is None or not hasattr(extras, "__len__"):
        raise ValueError("driver science attachment requires serialized extra_fields rows")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= len(extras):
        raise IndexError("driver science row index is outside the batch")
    row = extras[index]
    row = dict(row) if isinstance(row, Mapping) else {}
    eligible = response.research_eligible
    audit = dict(response.audit)
    if eligible and _correction_audit(audit).get("unbiased_under_response_model") is not True:
        eligible = False
        audit.setdefault("research_eligibility_rejection", "response_model_not_validated")
    for key, value in response.fields.items():
        if isinstance(value, np.ndarray):
            value = np.array(value, copy=True)
        elif isinstance(value, torch.Tensor):
            value = value.detach().cpu().clone()
        row[key] = value
    row["science_round_audit"] = audit
    row["science_research_eligible"] = eligible
    extras[index] = row


def materialize_science_batch_fields(batch: Any) -> None:
    """Pack driver-attached ragged science fields into padded TensorDict axes."""

    non_tensor = getattr(batch, "non_tensor_batch", None)
    tensor_batch = getattr(batch, "batch", None)
    if not isinstance(non_tensor, Mapping) or tensor_batch is None:
        raise TypeError("science batch materialization requires DataProto-like batch")
    extras = non_tensor.get("extra_fields")
    if extras is None or len(extras) == 0:
        raise ValueError("science batch materialization requires extra_fields rows")
    rows = [item if isinstance(item, Mapping) else {} for item in extras]
    required = tuple(SCIENCE_FIELD_NAMES)
    if any(any(key not in row for key in required) for row in rows):
        raise ValueError("driver science provider did not return a complete field envelope for every row")
    values = [torch.as_tensor(row["science_boundary_values"], dtype=torch.float32) for row in rows]
    if any(value.ndim != 2 or value.shape[0] < 2 or not torch.isfinite(value).all() for value in values):
        raise ValueError("driver science boundary values are not finite [segments+1,endpoints]")
    endpoint_dim = int(values[0].shape[1])
    if any(int(value.shape[1]) != endpoint_dim for value in values):
        raise ValueError("driver science endpoint dimensions differ across rows")
    max_boundaries = max(int(value.shape[0]) for value in values)
    padded_values: list[torch.Tensor] = []
    padded_boundary_masks: list[torch.Tensor] = []
    padded_endpoint_masks: list[torch.Tensor] = []
    for row, value in zip(rows, values, strict=True):
        boundary_mask = torch.as_tensor(row["science_boundary_mask"], dtype=torch.bool).reshape(-1)
        endpoint_mask = torch.as_tensor(row["science_endpoint_mask"], dtype=torch.bool)
        segment_mask = torch.as_tensor(row["segment_mask"], dtype=torch.bool).reshape(-1)
        if boundary_mask.numel() != value.shape[0] - 1 or segment_mask.numel() != value.shape[0] - 1:
            raise ValueError("driver science boundary/segment masks do not align")
        if endpoint_mask.shape != value.shape:
            raise ValueError("driver science endpoint mask does not align")
        pad_rows = max_boundaries - int(value.shape[0])
        if pad_rows:
            value = torch.cat((value, value[-1:].expand(pad_rows, -1)), dim=0)
            endpoint_mask = torch.cat(
                (endpoint_mask, torch.zeros(pad_rows, endpoint_dim, dtype=torch.bool)), dim=0
            )
            boundary_mask = torch.cat((boundary_mask, torch.zeros(pad_rows, dtype=torch.bool)), dim=0)
        padded_values.append(value)
        padded_boundary_masks.append(boundary_mask)
        padded_endpoint_masks.append(endpoint_mask)
    tensor_batch["science_boundary_values"] = torch.stack(padded_values, dim=0)
    tensor_batch["science_boundary_mask"] = torch.stack(padded_boundary_masks, dim=0)
    tensor_batch["science_endpoint_mask"] = torch.stack(padded_endpoint_masks, dim=0)
    # The estimator accepts segment IDs from either axis.  Driver execution
    # must provide the token-aligned form when the workers did not already
    # materialize it (for example, after a custom rollout adapter).
    response_width = int(tensor_batch["responses"].shape[1])
    padded_ids = []
    for row in rows:
        ids = torch.as_tensor(row["segment_ids"], dtype=torch.long).reshape(-1)
        if ids.numel() > response_width:
            raise ValueError("driver science segment_ids exceed response width")
        padded = torch.full((response_width,), -1, dtype=torch.long)
        padded[: ids.numel()] = ids
        padded_ids.append(padded)
    serialized_ids = torch.stack(padded_ids, dim=0)
    if "segment_ids" in tensor_batch.keys():
        existing_ids = torch.as_tensor(tensor_batch["segment_ids"], dtype=torch.long)
        if existing_ids.shape != serialized_ids.shape or not torch.equal(existing_ids, serialized_ids):
            raise ValueError("driver science segment_ids disagree with worker token metadata")
    else:
        tensor_batch["segment_ids"] = serialized_ids


__all__ = [
    "SCIENCE_FIELD_NAMES",
    "ScienceRoundRequest",
    "ScienceRoundResponse",
    "ScienceRoundProvider",
    "DisabledScienceRoundProvider",
    "CallbackScienceRoundProvider",
    "load_science_round_provider",
    "science_execution_owner",
    "science_enabled",
    "attach_science_response",
    "attach_science_response_to_batch",
    "materialize_science_batch_fields",
]
