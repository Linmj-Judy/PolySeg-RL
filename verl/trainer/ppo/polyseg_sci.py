"""Auditable PolySeg-Sci primitives for multi-fidelity science-agent RL.

This module is deliberately separate from :mod:`polyseg`.  The original
module implements the scalar, terminal-feedback trainer used by the existing
verl experiments.  PolySeg-Sci adds the v4 contract without changing those
experiments: vector-valued endpoints, fidelity-aware observations, causal
boundaries, frozen multi-output polynomial prediction, positive-propensity
residual correction, fresh-branch records, and a complete training-budget
ledger.

The objects in this file are data and estimator primitives.  They do not call
OpenAI4S, a network service, or a scientific evaluator themselves.  A harness
adapter supplies snapshots and evaluator callbacks and receives immutable JSON
records that an independent verifier can inspect.  This separation prevents
proxy scores or actor self-reports from silently becoming gold labels.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass, field, replace
from itertools import combinations
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch


SCI_METHOD_VERSION = "polyseg-sci-v4.2-evidence-bound-20261107"
# Restore/branch/trajectory evidence contract version.  Bumped whenever a
# research-eligibility rule changes so old artifacts cannot be silently
# reinterpreted as evidence under the current protocol.
SCI_CONTRACT_VERSION = "polyseg-sci-evidence-contract-v2"
ENDPOINT_SCHEMA_VERSION = "endpoint-schema-v1"
FEATURE_MAP_VERSION = "polyseg-sci-frozen-summary-v1"
SEGMENT_RULE_VERSION = "polyseg-sci-causal-boundary-v1"
# Evidence grades for a boundary restore.  ``durable`` means the restore came
# from a real checkpoint/CAS projection; ``self_attested`` means the response
# merely echoed a snapshot (e.g. the protocol probe).  Echoes are structurally
# useful but never earn research eligibility.
RESTORE_EVIDENCE_KINDS = ("durable", "self_attested", "unspecified")
# Continuation provenance modes for a fresh branch.  Only
# ``restored_boundary`` is admissible as scientific branch evidence; the
# protocol probe's "restart from the original prompt" is explicitly a
# non-evidence mode.
CONTINUATION_MODES = ("restored_boundary", "original_prompt_restart")
RESEARCH_CONTINUATION_MODES = frozenset({"restored_boundary"})
FAILURE_CLASSES = frozenset(
    {
        "restore_failure",
        "branch_failure",
        "evaluator_failure",
        "timeout",
        "invalid_state",
        "invalid_endpoint",
    }
)


class BranchExecutionError(RuntimeError):
    """Explicit fresh-branch failure (kept separate from evaluator errors)."""


class EvaluatorExecutionError(RuntimeError):
    """The independent endpoint evaluator failed after a branch ran."""


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _strict_real(value: Any) -> bool:
    """Return true only for finite real scalars, excluding booleans."""

    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def _strict_nonnegative_int(value: Any, name: str) -> int:
    """Parse persisted counters without silently truncating fractions."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _canonical(value: Any) -> Any:
    """Convert records into deterministic JSON-compatible values."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (tuple, list)):
        return [_canonical(v) for v in value]
    if isinstance(value, set):
        return sorted((_canonical(v) for v in value), key=lambda x: json.dumps(x, sort_keys=True))
    if hasattr(value, "item"):
        try:
            return _canonical(value.item())
        except (TypeError, ValueError):
            pass
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def prefix_hash(prefix: str) -> str:
    """Digest the token prefix a fresh branch claims to continue from."""

    if not isinstance(prefix, str):
        raise ValueError("prefix must be a string")
    return content_hash({"prefix": prefix})


def sampling_config_hash(sampling_config: Mapping[str, Any]) -> str:
    """Digest a sampling configuration for continuation provenance checks."""

    if not isinstance(sampling_config, Mapping):
        raise ValueError("sampling_config must be a mapping")
    return content_hash(dict(sampling_config))


def verify_continuation_provenance(
    provenance: Mapping[str, Any] | None,
    *,
    prefix: str,
    sampling_config: Mapping[str, Any],
    policy_version: str | None = None,
) -> tuple[str, ...]:
    """Return reasons a fresh-branch continuation is *not* research evidence.

    The contract is intentionally evidence-first: a branch may attest either
    the raw ``prefix``/``sampling_config`` it used or their digests.  An empty
    result means the branch demonstrably continued from the captured boundary
    under the captured sampling configuration.  ``original_prompt_restart`` is
    a valid protocol-probe mode but is never scientific branch evidence.
    """

    reasons: list[str] = []
    if not isinstance(provenance, Mapping):
        return ("continuation_provenance_missing",)
    mode = provenance.get("continuation_mode")
    if mode not in CONTINUATION_MODES:
        reasons.append("continuation_mode_invalid")
    elif mode not in RESEARCH_CONTINUATION_MODES:
        reasons.append(f"continuation_mode_not_research_eligible:{mode}")
    expected_prefix_hash = prefix_hash(prefix)
    claimed_prefix_hash = provenance.get("prefix_hash")
    if claimed_prefix_hash is None:
        claimed_prefix = provenance.get("prefix")
        if not isinstance(claimed_prefix, str):
            reasons.append("continuation_prefix_hash_missing")
        elif prefix_hash(claimed_prefix) != expected_prefix_hash:
            reasons.append("continuation_prefix_mismatch")
    elif str(claimed_prefix_hash) != expected_prefix_hash:
        reasons.append("continuation_prefix_hash_mismatch")
    expected_sampling_hash = sampling_config_hash(sampling_config)
    claimed_sampling_hash = provenance.get("sampling_config_hash")
    if claimed_sampling_hash is None:
        claimed_sampling = provenance.get("sampling_config")
        if not isinstance(claimed_sampling, Mapping):
            reasons.append("continuation_sampling_config_hash_missing")
        elif sampling_config_hash(claimed_sampling) != expected_sampling_hash:
            reasons.append("continuation_sampling_config_mismatch")
    elif str(claimed_sampling_hash) != expected_sampling_hash:
        reasons.append("continuation_sampling_config_hash_mismatch")
    if policy_version is not None:
        claimed_policy = provenance.get("policy_version")
        if claimed_policy is None:
            reasons.append("continuation_policy_version_missing")
        elif str(claimed_policy) != str(policy_version):
            reasons.append("continuation_policy_version_mismatch")
    return tuple(reasons)


@dataclass(frozen=True)
class FidelitySpec:
    """One measurement protocol for one endpoint dimension."""

    fidelity_id: str
    level: str
    evaluator_id: str
    evaluator_version: str
    cost: float
    calibrated_to_hf: bool = False
    kind: str = "proxy"

    def __post_init__(self) -> None:
        if not self.fidelity_id or not self.evaluator_id or not self.evaluator_version:
            raise ValueError("fidelity identifiers and evaluator version are required")
        if self.level not in {"LF", "MF", "HF"}:
            raise ValueError("fidelity level must be LF, MF, or HF")
        if not _strict_real(self.cost) or self.cost <= 0:
            raise ValueError("fidelity cost must be finite and positive")
        if not isinstance(self.calibrated_to_hf, bool):
            raise ValueError("calibrated_to_hf must be boolean")
        if self.kind not in {"gold", "proxy", "pseudo_label"}:
            raise ValueError("fidelity kind must be gold, proxy, or pseudo_label")
        if self.level == "HF" and self.kind != "gold":
            raise ValueError("HF fidelity must be a gold evaluator")


@dataclass(frozen=True)
class EndpointSpec:
    """Pre-registered scientific endpoint dimension."""

    endpoint_id: str
    attribute: str
    unit: str
    direction: str = "maximize"
    valid_range: tuple[float, float] | None = None
    hf_fidelity_id: str = "hf"
    missing_policy: str = "explicit_status"
    normalization: str = "none"
    constraint: str | None = None
    threshold: float | None = None
    priority: float = 1.0
    fidelities: tuple[FidelitySpec, ...] = ()
    target_value: float | None = None
    detection_limit: float | None = None
    quantification_limit: float | None = None
    replicate_error: float | None = None
    evaluator_reference_hash: str = ""
    calibration_dataset_hash: str = ""
    proxy_only: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.proxy_only, bool):
            raise ValueError("endpoint proxy_only must be boolean")
        for name, value in (("priority", self.priority), ("threshold", self.threshold), ("target_value", self.target_value), ("detection_limit", self.detection_limit), ("quantification_limit", self.quantification_limit), ("replicate_error", self.replicate_error)):
            if value is not None and not _strict_real(value):
                raise ValueError(f"endpoint {name} must be a finite real number")
        if not self.endpoint_id or not self.attribute:
            raise ValueError("endpoint_id and attribute are required")
        if self.direction not in {"maximize", "minimize", "target_range"}:
            raise ValueError("direction must be maximize, minimize, or target_range")
        if self.direction == "target_range" and self.target_value is None:
            raise ValueError("target_range endpoint requires target_value")
        if self.valid_range is not None:
            lo, hi = self.valid_range
            if not (_strict_real(lo) and _strict_real(hi) and lo <= hi):
                raise ValueError("valid_range must be finite and ordered")
        if self.missing_policy not in {"explicit_status", "censored", "interval"}:
            raise ValueError("missing_policy is not supported")
        if self.normalization not in {"none", "standardize"}:
            raise ValueError("normalization must be none or standardize")
        if not _finite(self.priority) or self.priority < 0:
            raise ValueError("endpoint priority must be finite and non-negative")
        ids = {f.fidelity_id for f in self.fidelities}
        if len(ids) != len(self.fidelities):
            raise ValueError("fidelity IDs must be unique within an endpoint")
        if self.proxy_only:
            if self.hf_fidelity_id in ids:
                raise ValueError("proxy_only endpoint must not include an HF fidelity")
        else:
            if self.hf_fidelity_id not in ids:
                raise ValueError(f"missing HF fidelity {self.hf_fidelity_id!r}")
            hf = next(f for f in self.fidelities if f.fidelity_id == self.hf_fidelity_id)
            if hf.level != "HF" or hf.kind != "gold":
                raise ValueError("hf_fidelity_id must identify an HF gold fidelity")
        if self.constraint is not None and self.constraint not in {"lower", "upper"}:
            raise ValueError("constraint must be lower or upper")
        if self.constraint is not None and self.threshold is None:
            raise ValueError("a constrained endpoint requires a threshold")
        if self.threshold is not None and not _finite(self.threshold):
            raise ValueError("endpoint threshold must be finite")
        for name, value in (("detection_limit", self.detection_limit), ("quantification_limit", self.quantification_limit), ("replicate_error", self.replicate_error)):
            if value is not None and (not _finite(value) or (name == "replicate_error" and value < 0)):
                raise ValueError(f"endpoint {name} must be finite and non-negative when supplied")
        if self.detection_limit is not None and self.quantification_limit is not None and self.quantification_limit < self.detection_limit:
            raise ValueError("quantification_limit must not be below detection_limit")
        if not isinstance(self.evaluator_reference_hash, str) or not isinstance(self.calibration_dataset_hash, str):
            raise ValueError("endpoint provenance hashes must be strings")
        if self.proxy_only and any(f.level == "HF" for f in self.fidelities):
            raise ValueError("proxy_only endpoint cannot declare an HF fidelity")
        if self.target_value is not None and not _finite(self.target_value):
            raise ValueError("endpoint target_value must be finite")
        if self.target_value is not None and self.valid_range is not None:
            lo, hi = self.valid_range
            if not lo <= float(self.target_value) <= hi:
                raise ValueError("endpoint target_value must lie inside valid_range")


@dataclass(frozen=True)
class EndpointSchema:
    """Immutable vector-endpoint contract used by a training run."""

    endpoints: tuple[EndpointSpec, ...]
    objective_mode: str = "preference"
    weights: tuple[float, ...] | None = None
    schema_version: str = ENDPOINT_SCHEMA_VERSION
    primary_objective: str | None = None

    def __post_init__(self) -> None:
        if not self.endpoints:
            raise ValueError("endpoint schema must contain at least one endpoint")
        ids = [e.endpoint_id for e in self.endpoints]
        if len(set(ids)) != len(ids):
            raise ValueError("endpoint IDs must be unique")
        if self.objective_mode not in {"pareto", "preference", "constraint"}:
            raise ValueError("objective_mode must be pareto, preference, or constraint")
        if self.objective_mode == "preference":
            if self.weights is None or len(self.weights) != len(ids):
                raise ValueError("preference mode requires one weight per endpoint")
            if any(not _strict_real(w) or w < 0 for w in self.weights) or sum(self.weights) <= 0:
                raise ValueError("preference weights must be finite, non-negative, and nonzero")
        elif self.weights is not None and len(self.weights) != len(ids):
            raise ValueError("weights must have one entry per endpoint")
        if self.objective_mode == "constraint" and not any(e.constraint for e in self.endpoints):
            raise ValueError("constraint mode requires at least one constrained endpoint")
        if self.objective_mode == "constraint" and self.primary_objective is None:
            raise ValueError("constraint mode requires an explicit primary_objective endpoint")
        if self.primary_objective is not None:
            if self.primary_objective not in ids:
                raise ValueError("primary_objective must name an endpoint")
            if self.endpoints[ids.index(self.primary_objective)].constraint is not None:
                raise ValueError("primary_objective must be an unconstrained endpoint")
            if (
                self.objective_mode == "constraint"
                and self.endpoints[ids.index(self.primary_objective)].direction == "target_range"
            ):
                raise ValueError("target_range cannot be the primary constraint objective")

    @property
    def endpoint_ids(self) -> tuple[str, ...]:
        return tuple(e.endpoint_id for e in self.endpoints)

    @property
    def dimension(self) -> int:
        return len(self.endpoints)

    @property
    def fingerprint(self) -> str:
        return content_hash(self.to_dict())

    def endpoint_index(self, endpoint_id: str) -> int:
        try:
            return self.endpoint_ids.index(endpoint_id)
        except ValueError as exc:
            raise KeyError(f"unknown endpoint {endpoint_id!r}") from exc

    def to_dict(self) -> dict[str, Any]:
        return _canonical(asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EndpointSchema":
        """Reconstruct a schema from its canonical JSON representation.

        Config files are intentionally plain JSON/YAML, while runtime code
        uses immutable dataclasses.  Keeping this conversion here makes the
        trainer and independent audit tools use exactly the same contract.
        Unknown fields are rejected by the dataclass constructors rather than
        silently ignored.
        """

        if not isinstance(payload, Mapping):
            raise ValueError("endpoint schema must be a mapping")
        raw_endpoints = payload.get("endpoints")
        if not isinstance(raw_endpoints, Sequence) or isinstance(raw_endpoints, (str, bytes)):
            raise ValueError("endpoint schema must contain an endpoints sequence")
        endpoints = []
        for raw_endpoint in raw_endpoints:
            if not isinstance(raw_endpoint, Mapping):
                raise ValueError("endpoint entries must be mappings")
            raw_fidelities = raw_endpoint.get("fidelities", ())
            fidelities = tuple(
                FidelitySpec(**dict(raw_fidelity))
                for raw_fidelity in raw_fidelities
                if isinstance(raw_fidelity, Mapping)
            )
            if len(fidelities) != len(raw_fidelities):
                raise ValueError("fidelity entries must be mappings")
            fields = dict(raw_endpoint)
            fields["valid_range"] = tuple(fields["valid_range"]) if fields.get("valid_range") is not None else None
            fields["fidelities"] = fidelities
            endpoints.append(EndpointSpec(**fields))
        weights = payload.get("weights")
        return cls(
            endpoints=tuple(endpoints),
            objective_mode=str(payload.get("objective_mode", "preference")),
            weights=tuple(weights) if weights is not None else None,
            schema_version=str(payload.get("schema_version", ENDPOINT_SCHEMA_VERSION)),
            primary_objective=payload.get("primary_objective"),
        )

    def validate_vector(self, values: Sequence[float | None], statuses: Sequence[str] | None = None) -> None:
        if len(values) != self.dimension:
            raise ValueError("endpoint vector dimension does not match schema")
        if statuses is not None and len(statuses) != self.dimension:
            raise ValueError("endpoint status vector dimension does not match schema")
        for i, value in enumerate(values):
            status = "observed" if statuses is None else statuses[i]
            if status in {"observed", "censored", "interval"}:
                endpoint = self.endpoints[i]
                if status == "censored" and endpoint.missing_policy != "censored":
                    raise ValueError(f"endpoint {endpoint.endpoint_id} is not registered for censored observations")
                if status == "interval" and endpoint.missing_policy != "interval":
                    raise ValueError(f"endpoint {endpoint.endpoint_id} is not registered for interval observations")
                if value is None or not _finite(value):
                    raise ValueError(f"endpoint {self.endpoints[i].endpoint_id} has invalid observed value")
                if endpoint.valid_range is not None:
                    lo, hi = endpoint.valid_range
                    if float(value) < lo or float(value) > hi:
                        raise ValueError(f"endpoint {self.endpoints[i].endpoint_id} is outside valid range")
            elif status == "below_detection_limit":
                endpoint = self.endpoints[i]
                if value is not None or endpoint.detection_limit is None:
                    raise ValueError(f"endpoint {endpoint.endpoint_id} below_detection_limit requires null value and detection limit")
            elif status in {
                "not_measured",
                "not_applicable",
                "assay_failed",
                "scientifically_failed",
                "restore_failed",
                "branch_failed",
                "evaluator_failed",
                "timeout",
            }:
                if value is not None:
                    raise ValueError(f"missing endpoint {self.endpoints[i].endpoint_id} must have a null value")
            else:
                raise ValueError(f"unknown endpoint status {status!r}")

    def validate_observation(self, observation: "EndpointObservation", *, allow_empty_provenance: bool = True) -> None:
        """Validate one observation against its endpoint/fidelity contract."""

        if observation.endpoint_id not in self.endpoint_ids:
            raise ValueError(f"unknown endpoint {observation.endpoint_id!r}")
        endpoint = self.endpoints[self.endpoint_index(observation.endpoint_id)]
        fidelity = next(
            (item for item in endpoint.fidelities if item.fidelity_id == observation.fidelity_id),
            None,
        )
        if fidelity is None:
            raise ValueError(
                f"unknown fidelity {observation.fidelity_id!r} for endpoint {endpoint.endpoint_id!r}"
            )
        if observation.status == "censored" and endpoint.missing_policy != "censored":
            raise ValueError(f"endpoint {endpoint.endpoint_id} is not registered for censored observations")
        if observation.status == "interval" and endpoint.missing_policy != "interval":
            raise ValueError(f"endpoint {endpoint.endpoint_id} is not registered for interval observations")
        failure_statuses = {"restore_failed", "branch_failed", "evaluator_failed", "timeout"}
        if observation.status in failure_statuses and observation.failure_class is None:
            raise ValueError("failed endpoint observations require failure_class")
        if observation.status in {"observed", "censored", "interval"} and observation.failure_class is not None:
            raise ValueError("measured endpoint observations cannot carry failure_class")
        if observation.status in {"observed", "censored", "interval"}:
            if observation.value is None or not _finite(observation.value):
                raise ValueError("measured endpoint observations require a finite value")
            if observation.status in {"censored", "interval"} and (
                observation.lower_bound is None
                or observation.upper_bound is None
                or not _finite(observation.lower_bound)
                or not _finite(observation.upper_bound)
                or observation.lower_bound > observation.upper_bound
                or not observation.lower_bound <= float(observation.value) <= observation.upper_bound
            ):
                raise ValueError("censored or interval observations require ordered bounds containing the value")
            if endpoint.valid_range is not None:
                lo, hi = endpoint.valid_range
                if not lo <= float(observation.value) <= hi:
                    raise ValueError(f"endpoint {endpoint.endpoint_id} is outside valid range")
        elif observation.status == "below_detection_limit":
            if observation.value is not None or observation.detection_limit is None:
                raise ValueError("below_detection_limit observations require a null value and detection limit")
            if endpoint.detection_limit is not None and abs(float(observation.detection_limit) - float(endpoint.detection_limit)) > 1e-8:
                raise ValueError("observation detection limit does not match endpoint schema")
        elif observation.value is not None:
            # A failed or explicitly missing measurement is an audit datum,
            # never a scientific label.  Reject it here so a fresh-branch
            # callback cannot smuggle a finite value through the validator and
            # rely on a later aggregation step to ignore it.
            raise ValueError("failed or missing endpoint observations must have a null value")
        expected_id = fidelity.evaluator_id
        expected_version = fidelity.evaluator_version
        if observation.evaluator_id and observation.evaluator_id != expected_id:
            raise ValueError("endpoint evaluator provenance mismatch")
        if observation.evaluator_version and observation.evaluator_version != expected_version:
            raise ValueError("endpoint evaluator version mismatch")
        if not allow_empty_provenance and (not observation.evaluator_id or not observation.evaluator_version):
            raise ValueError("endpoint evaluator provenance is required")

    def orient_values(self, values: torch.Tensor) -> torch.Tensor:
        """Apply only affine orientation to conditional endpoint means.

        Target-range values are mapped to negative distance from the explicitly
        registered target.  Evaluators should apply this transform to terminal
        outcomes before fitting if strict conditional-mean unbiasedness is
        required, because nonlinear utility does not commute with expectation.
        """

        tensor = torch.as_tensor(values)
        if tensor.shape[-1] != self.dimension:
            raise ValueError("endpoint axis does not match schema")
        result = tensor.clone()
        for index, endpoint in enumerate(self.endpoints):
            if endpoint.direction == "minimize":
                result[..., index] = -tensor[..., index]
            elif endpoint.direction == "target_range":
                result[..., index] = -(tensor[..., index] - float(endpoint.target_value)).abs()
        return result

    def objective_sign(self, endpoint_index: int) -> float:
        """Linear sign for a maximize/minimize primary objective."""

        endpoint = self.endpoints[endpoint_index]
        if endpoint.direction == "target_range":
            raise ValueError("target_range cannot be represented by one fixed gradient sign")
        return -1.0 if endpoint.direction == "minimize" else 1.0

    def primary_index(self, primary_objective: int | None = None) -> int:
        declared = None if self.primary_objective is None else self.endpoint_index(self.primary_objective)
        index = declared if primary_objective is None else primary_objective
        if index is None or isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < self.dimension:
            raise ValueError("constraint mode requires an explicit primary_objective")
        if declared is not None and index != declared:
            raise ValueError("primary_objective conflicts with endpoint schema")
        if self.endpoints[index].constraint is not None:
            raise ValueError("primary_objective must be an unconstrained endpoint")
        return index

    def constraint_coefficients(self, multipliers: torch.Tensor, primary_objective: int | None = None) -> torch.Tensor:
        """Frozen primal coefficients; raw lower/upper bounds determine signs."""
        index = self.primary_index(primary_objective)
        lambdas = torch.as_tensor(multipliers).detach()
        if lambdas.ndim < 1 or lambdas.shape[-1] != self.dimension:
            raise ValueError("constraint multipliers must match endpoint dimension")
        if not torch.isfinite(lambdas).all() or torch.any(lambdas < 0):
            raise ValueError("constraint multipliers must be finite and non-negative")
        coefficients = torch.zeros_like(lambdas)
        for i, endpoint in enumerate(self.endpoints):
            if endpoint.constraint == "lower":
                coefficients[..., i] = lambdas[..., i]
            elif endpoint.constraint == "upper":
                coefficients[..., i] = -lambdas[..., i]
        coefficients[..., index] = self.objective_sign(index)
        return coefficients


def load_endpoint_schema(path: str | Path) -> EndpointSchema:
    """Load and validate a pre-registered endpoint schema from JSON."""

    schema_path = Path(path)
    if not schema_path.is_file():
        raise FileNotFoundError(f"endpoint schema file does not exist: {schema_path}")
    try:
        payload = json.loads(schema_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"endpoint schema is not valid JSON: {schema_path}") from exc
    return EndpointSchema.from_dict(payload)


def load_feature_map_contract(path: str | Path, *, expected_hash: str | None = None) -> FeatureMapContract:
    """Load and optionally fingerprint-check a pre-registered feature map."""

    feature_path = Path(path)
    if not feature_path.is_file():
        raise FileNotFoundError(f"feature map contract file does not exist: {feature_path}")
    try:
        payload = json.loads(feature_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"feature map contract is not valid JSON: {feature_path}") from exc
    feature_map = FeatureMapContract.from_dict(payload)
    if expected_hash and str(expected_hash) != feature_map.fingerprint:
        raise ValueError("feature_map_hash does not match feature_map_path")
    if feature_map.version != FEATURE_MAP_VERSION:
        raise ValueError(
            f"unsupported feature map version {feature_map.version!r}; expected {FEATURE_MAP_VERSION!r}"
        )
    return feature_map


@dataclass(frozen=True)
class EndpointObservation:
    endpoint_id: str
    fidelity_id: str
    value: float | None
    status: str = "observed"
    cost: float = 0.0
    evaluator_id: str = ""
    evaluator_version: str = ""
    failure_class: str | None = None
    evidence_ref: str | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    detection_limit: float | None = None
    replicate_id: str | None = None

    def __post_init__(self) -> None:
        for name, value in (("value", self.value), ("cost", self.cost), ("lower_bound", self.lower_bound), ("upper_bound", self.upper_bound), ("detection_limit", self.detection_limit)):
            if value is not None and not _strict_real(value):
                raise ValueError(f"observation {name} must be a finite real number")
        if not isinstance(self.endpoint_id, str) or not self.endpoint_id:
            raise ValueError("endpoint_id must be a non-empty string")
        if not isinstance(self.fidelity_id, str) or not self.fidelity_id:
            raise ValueError("fidelity_id must be a non-empty string")
        if self.status not in {
            "observed",
            "below_detection_limit",
            "censored",
            "interval",
            "not_measured",
            "not_applicable",
            "assay_failed",
            "scientifically_failed",
            "restore_failed",
            "branch_failed",
            "evaluator_failed",
            "timeout",
        }:
            raise ValueError("unknown endpoint observation status")
        if self.status in {"observed", "censored", "interval"} and (self.value is None or not _strict_real(self.value)):
            raise ValueError(f"a {self.status} endpoint requires a finite value")
        if self.status == "below_detection_limit":
            if self.value is not None or self.detection_limit is None or not _strict_real(self.detection_limit):
                raise ValueError("below_detection_limit requires a finite detection_limit and null value")
        if self.status in {"censored", "interval"} and (self.lower_bound is not None or self.upper_bound is not None):
            if self.lower_bound is None or self.upper_bound is None or not _strict_real(self.lower_bound) or not _strict_real(self.upper_bound) or self.lower_bound > self.upper_bound:
                raise ValueError(f"a {self.status} endpoint requires ordered finite bounds")
            if not self.lower_bound <= float(self.value) <= self.upper_bound:
                raise ValueError(f"a {self.status} endpoint value must lie within bounds")
        if self.status not in {"observed", "below_detection_limit", "censored", "interval"} and self.value is not None:
            # Failed or explicitly missing measurements are audit events, not
            # scientific labels.  Enforce this at construction time as well
            # as in ``EndpointSchema.validate_observation`` so callers cannot
            # pass an invalid observation directly to a correction routine.
            raise ValueError("failed or missing endpoint observations must have a null value")
        if not _strict_real(self.cost) or self.cost < 0:
            raise ValueError("observation cost must be finite and non-negative")
        if self.failure_class is not None and self.failure_class not in FAILURE_CLASSES:
            raise ValueError("unknown failure class")
        if self.status in {"observed", "below_detection_limit", "censored", "interval"} and self.failure_class is not None:
            raise ValueError("measured endpoint observations cannot carry failure_class")
        expected_failure = {
            "restore_failed": "restore_failure",
            "branch_failed": "branch_failure",
            "evaluator_failed": "evaluator_failure",
            "timeout": "timeout",
        }.get(self.status)
        if expected_failure is not None and self.failure_class != expected_failure:
            raise ValueError(f"{self.status} observations require failure_class={expected_failure!r}")


@dataclass(frozen=True)
class ActionRecord:
    action_id: str
    action_type: str
    is_policy_action: bool
    token_start: int
    token_end: int
    logprob: float | None = None
    old_logprob: float | None = None
    tool_name: str | None = None
    tool_call_id: str | None = None
    tool_return_id: str | None = None
    observation_ref: str | None = None
    artifact_refs: tuple[str, ...] = ()
    execution_cost: float = 0.0
    remaining_budget: float | None = None
    # OpenAI4S records a wall-clock timestamp for every ledger event.  Keeping
    # it optional preserves compatibility with offline/replayed trajectories,
    # while adapters that claim a live action-ledger contract must populate it.
    timestamp: float | None = None
    # Token-level boolean retained explicitly in serialized ledgers.  It is
    # redundant with ``is_policy_action`` for compatibility, but makes the
    # action mask contract unambiguous to external auditors.
    action_mask: bool | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.is_policy_action, bool):
            raise ValueError("is_policy_action must be boolean")
        if self.action_mask is not None and not isinstance(self.action_mask, bool):
            raise ValueError("action_mask must be boolean when supplied")
        # Token spans are part of the causal/PPO alignment contract.  Tool
        # lifecycle records may use an empty 0:0 span, but a policy action
        # must cover a strictly positive integer token interval.
        if (
            isinstance(self.token_start, bool)
            or isinstance(self.token_end, bool)
            or not isinstance(self.token_start, int)
            or not isinstance(self.token_end, int)
            or self.token_start < 0
            or self.token_end < self.token_start
            or (self.is_policy_action and self.token_end <= self.token_start)
        ):
            raise ValueError("invalid action token interval")
        if self.action_mask is not None and bool(self.action_mask) != bool(self.is_policy_action):
            raise ValueError("action_mask must agree with is_policy_action")
        if not _strict_real(self.execution_cost) or self.execution_cost < 0:
            raise ValueError("execution cost must be finite and non-negative")
        if self.timestamp is not None and not _strict_real(self.timestamp):
            raise ValueError("action timestamp must be finite when supplied")
        for name, value in (("logprob", self.logprob), ("old_logprob", self.old_logprob)):
            if value is not None and not _strict_real(value):
                raise ValueError(f"{name} must be finite when supplied")
        if self.remaining_budget is not None and (
            not _strict_real(self.remaining_budget) or float(self.remaining_budget) < 0
        ):
            raise ValueError("remaining budget must be finite and non-negative when supplied")


@dataclass(frozen=True)
class SegmentBoundary:
    segment_id: int
    start: int
    end: int
    action_indices: tuple[int, ...]
    boundary_type: str
    duration: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.segment_id, bool)
            or isinstance(self.start, bool)
            or isinstance(self.end, bool)
            or isinstance(self.duration, bool)
            or not all(isinstance(value, int) for value in (self.segment_id, self.start, self.end, self.duration))
            or self.segment_id < 0
            or self.start < 0
            or self.end <= self.start
        ):
            raise ValueError("invalid causal segment boundary")
        if not isinstance(self.boundary_type, str) or not self.boundary_type:
            raise ValueError("boundary_type must be a non-empty string")
        if (
            not self.action_indices
            or any(isinstance(index, bool) or not isinstance(index, int) for index in self.action_indices)
            or len(set(self.action_indices)) != len(self.action_indices)
        ):
            raise ValueError("segment action indices must be unique non-negative integers")
        if self.duration != self.end - self.start:
            raise ValueError("segment duration/action indices are inconsistent")
        if tuple(sorted(self.action_indices)) != self.action_indices:
            raise ValueError("segment action indices must be sorted")
        if min(self.action_indices) < self.start or max(self.action_indices) >= self.end:
            raise ValueError("segment action indices must lie inside the boundary")


class CausalSegmenter:
    """Create fixed or event-defined boundaries from past/current actions only."""

    def __init__(self, *, rule: str = "fixed_action_tokens", m: int = 4, horizon: int | None = None, rule_version: str = SEGMENT_RULE_VERSION):
        if rule not in {"fixed_action_tokens", "tool_return", "event"}:
            raise ValueError("unsupported causal segment rule")
        if isinstance(m, bool) or not isinstance(m, int) or m < 1:
            raise ValueError("m must be positive")
        self.rule = rule
        self.m = m
        if horizon is not None and (isinstance(horizon, bool) or not isinstance(horizon, int)):
            raise ValueError("horizon must be an integer when supplied")
        if rule == "fixed_action_tokens" and (horizon is None or horizon < 1):
            raise ValueError("fixed_action_tokens requires a pre-registered positive horizon")
        self.horizon = horizon
        if self.horizon is not None and self.horizon < self.m:
            raise ValueError("fixed_action_tokens requires m <= the registered horizon")
        self.rule_version = rule_version

    @staticmethod
    def _balanced_slices(length: int, count: int) -> list[tuple[int, int]]:
        count = min(length, count)
        base, rem = divmod(length, count)
        result, start = [], 0
        for i in range(count):
            end = start + base + (1 if i < rem else 0)
            result.append((start, end))
            start = end
        return result

    def segment(
        self,
        action_mask: Sequence[bool],
        *,
        tool_return_positions: Sequence[int] | None = None,
        event_positions: Sequence[int] | None = None,
    ) -> list[SegmentBoundary]:
        normalized_mask: list[bool] = []
        for value in action_mask:
            # Python bool is the serialized contract.  Tensor/NumPy scalar
            # booleans are accepted only after extracting their scalar value;
            # numeric 0/1 and other truthy values must fail closed.
            if isinstance(value, bool):
                normalized_mask.append(value)
            elif isinstance(value, torch.Tensor) and value.ndim == 0 and value.dtype == torch.bool:
                normalized_mask.append(bool(value.item()))
            else:
                raise ValueError("action_mask must contain boolean values")
        mask = normalized_mask
        positions = [i for i, active in enumerate(mask) if active]
        if not positions:
            raise ValueError("causal segmentation requires at least one policy action")
        if self.rule == "fixed_action_tokens":
            if any(position >= self.horizon for position in positions):
                raise ValueError("action mask contains policy actions beyond the registered horizon")
            # The interval size is fixed from the pre-registered horizon, not
            # inferred from the eventual response length.  This is what keeps
            # boundaries causal when an agent terminates early.
            width = max(1, math.ceil(self.horizon / self.m))
            cuts = []
            start = 0
            while start < self.horizon:
                # The bin layout is fixed from the registered horizon, but a
                # realized response may terminate inside its final bin.  Clip
                # the serialized token interval to the observed response so a
                # boundary never references tokens outside ``action_mask``.
                end = min(self.horizon, start + width, len(mask))
                if end <= start:
                    start += width
                    continue
                inside = [p for p in positions if start <= p < end]
                if inside:
                    cuts.append((start, end))
                start = end
        else:
            raw = tool_return_positions if self.rule == "tool_return" else event_positions
            if raw is None or not raw:
                raise ValueError(f"{self.rule} segmentation requires boundary positions")
            if any(isinstance(p, bool) or not isinstance(p, int) or p <= 0 or p > len(mask) for p in raw):
                raise ValueError("causal boundary positions must be in (0, token_length]")
            ordered = list(raw)
            if ordered != sorted(set(ordered)):
                raise ValueError("causal boundary positions must be strictly increasing")
            cuts = []
            start = positions[0]
            for end in ordered:
                inside = [p for p in positions if start <= p < end]
                if inside:
                    cuts.append((start, end))
                    start = end
            if start <= positions[-1]:
                inside = [p for p in positions if start <= p <= positions[-1]]
                if inside:
                    cuts.append((start, positions[-1] + 1))
        result = []
        for segment_id, (start, end) in enumerate(cuts):
            indices = tuple(p for p in positions if start <= p < end)
            if not indices:
                continue
            result.append(SegmentBoundary(segment_id, start, end, indices, self.rule, end - start))
        if not result:
            raise ValueError("causal segmentation produced no segments")
        return result


@dataclass(frozen=True)
class FeatureMapContract:
    """Frozen, hashable normalization contract for boundary features."""

    feature_names: tuple[str, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    version: str = FEATURE_MAP_VERSION
    clip: bool = True

    def __post_init__(self) -> None:
        if not self.feature_names or len(self.feature_names) != len(self.lower) or len(self.lower) != len(self.upper):
            raise ValueError("feature names and bounds must have equal non-zero length")
        if not isinstance(self.clip, bool):
            raise ValueError("feature map clip must be boolean")
        if len(set(self.feature_names)) != len(self.feature_names) or any(not str(name) for name in self.feature_names):
            raise ValueError("feature names must be unique and non-empty")
        for lo, hi in zip(self.lower, self.upper, strict=True):
            if not (_strict_real(lo) and _strict_real(hi) and hi > lo):
                raise ValueError("feature bounds must be finite and strictly ordered")

    @property
    def dimension(self) -> int:
        return len(self.feature_names)

    @property
    def fingerprint(self) -> str:
        return content_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return _canonical(asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureMapContract":
        """Reconstruct a frozen feature map from an audited JSON contract."""

        if not isinstance(payload, Mapping):
            raise ValueError("feature map contract must be a mapping")
        names = payload.get("feature_names")
        lower = payload.get("lower")
        upper = payload.get("upper")
        if not isinstance(names, Sequence) or isinstance(names, (str, bytes)):
            raise ValueError("feature map feature_names must be a sequence")
        if not isinstance(lower, Sequence) or isinstance(lower, (str, bytes)):
            raise ValueError("feature map lower bounds must be a sequence")
        if not isinstance(upper, Sequence) or isinstance(upper, (str, bytes)):
            raise ValueError("feature map upper bounds must be a sequence")
        clip = payload.get("clip", True)
        if not isinstance(clip, bool):
            raise ValueError("feature map clip must be boolean")
        return cls(
            feature_names=tuple(str(name) for name in names),
            lower=tuple(lower),
            upper=tuple(upper),
            version=str(payload.get("version", FEATURE_MAP_VERSION)),
            clip=clip,
        )

    @classmethod
    def fit(cls, features: Sequence[Sequence[float]], names: Sequence[str], *, version: str = FEATURE_MAP_VERSION) -> "FeatureMapContract":
        x = torch.as_tensor(features, dtype=torch.float64)
        if x.ndim != 2 or x.shape[1] != len(names) or x.shape[0] < 1:
            raise ValueError("features must be a non-empty [N,p] matrix")
        if not torch.isfinite(x).all():
            raise ValueError("feature fitting received non-finite values")
        lower = x.min(0).values
        upper = x.max(0).values
        # Constant coordinates are valid; use a one-unit interval centered on
        # the observed value rather than silently dropping the coordinate.
        width = (upper - lower).clamp_min(1.0)
        center = (upper + lower) / 2
        return cls(tuple(names), tuple((center - width / 2).tolist()), tuple((center + width / 2).tolist()), version)

    def transform(self, features: Sequence[Sequence[float]] | torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(features, dtype=torch.float32)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim != 2 or x.shape[1] != self.dimension:
            raise ValueError("features do not match frozen feature map")
        if not torch.isfinite(x).all():
            raise ValueError("feature transform received non-finite values")
        lower = torch.tensor(self.lower, dtype=x.dtype, device=x.device)
        upper = torch.tensor(self.upper, dtype=x.dtype, device=x.device)
        z = 2 * (x - lower) / (upper - lower) - 1
        if self.clip:
            z = z.clamp(-1, 1)
        if not torch.isfinite(z).all():
            raise ValueError("feature transform produced non-finite values")
        return z


def _multi_indices(p: int, q: int) -> list[tuple[int, ...]]:
    if p < 1 or q < 0:
        raise ValueError("feature dimension must be positive and q non-negative")
    result: list[tuple[int, ...]] = []

    def visit(prefix: tuple[int, ...], remaining: int, dim: int) -> None:
        if dim == p - 1:
            for degree in range(remaining + 1):
                result.append(prefix + (degree,))
            return
        for degree in range(remaining + 1):
            visit(prefix + (degree,), remaining - degree, dim + 1)

    visit((), q, 0)
    return result


def _legendre_features(z: torch.Tensor, q: int, basis: str = "total") -> torch.Tensor:
    if z.ndim == 1:
        z = z.unsqueeze(0)
    if z.ndim != 2 or q < 0 or basis not in {"total", "additive"}:
        raise ValueError("invalid polynomial feature arguments")
    values = []
    for dim in range(z.shape[1]):
        one = [torch.ones_like(z[:, dim])]
        if q >= 1:
            one.append(z[:, dim])
        for degree in range(1, q):
            one.append(((2 * degree + 1) * z[:, dim] * one[-1] - degree * one[-2]) / (degree + 1))
        values.append(torch.stack(one, dim=-1))
    if basis == "additive":
        cols = [torch.ones(z.shape[0], dtype=z.dtype, device=z.device)]
        for dim in range(z.shape[1]):
            cols.extend(values[dim][:, degree] for degree in range(1, q + 1))
        return torch.stack(cols, dim=-1)
    cols = []
    for alpha in _multi_indices(z.shape[1], q):
        term = torch.ones(z.shape[0], dtype=z.dtype, device=z.device)
        for dim, degree in enumerate(alpha):
            term = term * values[dim][:, degree]
        cols.append(term)
    return torch.stack(cols, dim=-1)


class PolynomialVectorPredictor:
    """Frozen multi-output ridge predictor over Legendre boundary features."""

    def __init__(self, schema: EndpointSchema, feature_map: FeatureMapContract, *, q: int = 2, basis: str = "total", ridge_lambda: float | Sequence[float] = 1.0):
        # ``q`` controls the basis cardinality.  It is a method contract, so
        # accepting ``1.5``/``True`` and then applying ``int`` would silently
        # select a different predictor than the caller requested.
        if isinstance(q, bool) or not isinstance(q, int) or q < 0:
            raise ValueError("q must be a non-negative integer")
        if basis not in {"total", "additive"}:
            raise ValueError("basis must be total or additive")
        if isinstance(ridge_lambda, Sequence) and not isinstance(ridge_lambda, (str, bytes)):
            if len(ridge_lambda) != schema.dimension or any(not _strict_real(value) or float(value) <= 0 for value in ridge_lambda):
                raise ValueError("per-endpoint ridge lambdas must match endpoint dimension and be positive")
            self.ridge_lambdas = tuple(float(value) for value in ridge_lambda)
        elif _strict_real(ridge_lambda) and float(ridge_lambda) > 0:
            self.ridge_lambdas = tuple(float(ridge_lambda) for _ in range(schema.dimension))
        else:
            raise ValueError("ridge_lambda must be positive")
        self.schema = schema
        self.feature_map = feature_map
        self.q = q
        self.basis = basis
        self.ridge_lambda = float(self.ridge_lambdas[0]) if len(set(self.ridge_lambdas)) == 1 else self.ridge_lambdas
        self.theta: torch.Tensor | None = None
        self.gram: torch.Tensor | None = None
        self.head_grams: tuple[torch.Tensor, ...] = ()
        self.sample_count = 0
        self.fit_groups: tuple[str, ...] = ()
        self.oof_predictions: torch.Tensor | None = None
        self.oof_mask: torch.Tensor | None = None
        # Output normalization is fitted once on the training observations
        # and frozen with the predictor.  It makes the common ridge scale
        # meaningful when endpoint units differ (e.g. pKd versus degrees).
        self.target_center: torch.Tensor | None = None
        self.target_scale: torch.Tensor | None = None

    @property
    def dimension(self) -> int:
        return _legendre_features(torch.zeros(1, self.feature_map.dimension), self.q, self.basis).shape[1]

    def _design(self, features: Sequence[Sequence[float]] | torch.Tensor) -> torch.Tensor:
        z = self.feature_map.transform(features)
        return _legendre_features(z, self.q, self.basis)

    def _fit_matrix(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        theta = torch.zeros(x.shape[1], self.schema.dimension, dtype=x.dtype, device=x.device)
        gram = x.T @ x
        eye = torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
        for endpoint in range(self.schema.dimension):
            valid = mask[:, endpoint]
            if not torch.any(valid):
                continue
            local_x, local_y = x[valid], y[valid, endpoint]
            local_gram = local_x.T @ local_x + self.ridge_lambdas[endpoint] * eye
            theta[:, endpoint] = torch.linalg.solve(local_gram, local_x.T @ local_y)
        return theta, gram

    @staticmethod
    def _target_stats(y: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        center = torch.zeros(y.shape[1], dtype=y.dtype, device=y.device)
        scale = torch.ones(y.shape[1], dtype=y.dtype, device=y.device)
        for endpoint in range(y.shape[1]):
            valid = mask[:, endpoint]
            if torch.any(valid):
                values = y[valid, endpoint]
                center[endpoint] = values.mean()
                # A constant endpoint is valid; a unit scale avoids creating
                # an artificial zero-variance regression problem.
                spread = values.std(unbiased=False)
                scale[endpoint] = torch.where(
                    spread > torch.finfo(y.dtype).eps,
                    spread,
                    torch.ones_like(spread),
                )
        return center, scale

    def fit(
        self,
        features: Sequence[Sequence[float]] | torch.Tensor,
        targets: Sequence[Sequence[float | None]] | torch.Tensor,
        *,
        target_mask: Sequence[Sequence[bool]] | torch.Tensor | None = None,
        groups: Sequence[str] | None = None,
        cross_fit: bool = False,
    ) -> "PolynomialVectorPredictor":
        x = self._design(features)
        # ``None`` is the explicit not-measured representation at the Python
        # boundary.  Convert it to NaN only inside the fit routine, where the
        # endpoint mask makes the missingness semantics explicit.
        if isinstance(targets, torch.Tensor):
            y = targets.to(dtype=x.dtype, device=x.device)
            tensor_targets = True
        else:
            tensor_targets = False
            try:
                rows = list(targets)
                y = torch.tensor(
                    [[float(value) if value is not None else float("nan") for value in row] for row in rows],
                    dtype=x.dtype,
                    device=x.device,
                )
                provided = torch.tensor(
                    [[value is not None for value in row] for row in rows], dtype=torch.bool, device=x.device
                )
            except (TypeError, ValueError, RuntimeError) as exc:
                raise ValueError("targets must be a rectangular numeric/None matrix") from exc
        if y.ndim != 2 or y.shape != (x.shape[0], self.schema.dimension):
            raise ValueError("targets must have shape [N, endpoint_dimension]")
        if target_mask is None:
            if tensor_targets and not torch.isfinite(y).all():
                raise ValueError("tensor targets with missing values require an explicit target_mask")
            # Only None denotes missing data implicitly. Numeric NaN/Inf
            # remains observed and fails the finite-value check below.
            mask = torch.ones_like(y, dtype=torch.bool) if tensor_targets else provided
        else:
            mask = torch.as_tensor(target_mask, dtype=torch.bool, device=x.device)
            if mask.shape != y.shape:
                raise ValueError("target_mask must match targets")
        # Missing values are allowed only where the explicit target mask says
        # they are missing.  An observed NaN/Inf must never become a synthetic
        # zero label through ``nan_to_num``.
        if torch.any(mask & ~torch.isfinite(y)):
            raise ValueError("observed predictor targets must be finite")
        for endpoint_index, endpoint in enumerate(self.schema.endpoints):
            if endpoint.valid_range is not None:
                lo, hi = endpoint.valid_range
                observed = y[mask[:, endpoint_index], endpoint_index]
                if observed.numel() and (torch.any(observed < lo) or torch.any(observed > hi)):
                    raise ValueError(f"predictor target for endpoint {endpoint.endpoint_id} is outside valid range")
        self.target_center, self.target_scale = self._target_stats(y, mask)
        missing_heads = [
            endpoint.endpoint_id
            for endpoint, endpoint_mask in zip(self.schema.endpoints, mask.T, strict=True)
            if not torch.any(endpoint_mask)
        ]
        if missing_heads:
            raise ValueError(
                "predictor fit requires at least one observed target per endpoint; "
                f"missing endpoints: {missing_heads}"
            )
        scaled_y = torch.where(
            mask,
            (y - self.target_center.unsqueeze(0)) / self.target_scale.unsqueeze(0),
            torch.zeros_like(y),
        )
        if groups is not None and len(groups) != x.shape[0]:
            raise ValueError("groups must have one entry per trajectory")
        self.theta, self.gram = self._fit_matrix(x, scaled_y, mask)
        self.head_grams = tuple(
            (x[mask[:, endpoint]].T @ x[mask[:, endpoint]])
            if torch.any(mask[:, endpoint])
            else torch.zeros_like(self.gram)
            for endpoint in range(self.schema.dimension)
        )
        self.sample_count = int(x.shape[0])
        self.fit_groups = tuple(groups or ())
        self.oof_predictions = None
        self.oof_mask = None
        if cross_fit:
            if groups is None:
                raise ValueError("cross_fit requires trajectory/task groups")
            unique = list(dict.fromkeys(str(g) for g in groups))
            if len(unique) < 2:
                raise ValueError("cross_fit requires at least two groups")
            oof = torch.zeros_like(y)
            oof_valid = torch.zeros_like(mask)
            for held_out in unique:
                train = torch.tensor([str(g) != held_out for g in groups], device=x.device)
                held = ~train
                # A cross-fitted prediction is only defined when the fold's
                # training partition contains an observation for every
                # endpoint.  The global fit check above is not sufficient:
                # an endpoint can be present only in the held-out group.  In
                # that case ``_fit_matrix`` would otherwise return a zero
                # head and the OOF mask would incorrectly advertise it as a
                # valid scientific prediction.
                missing_fold = [
                    endpoint.endpoint_id
                    for endpoint, endpoint_mask in zip(self.schema.endpoints, mask[train].T, strict=True)
                    if not torch.any(endpoint_mask)
                ]
                if missing_fold:
                    raise ValueError(
                        "cross_fit requires at least one training observation "
                        f"per endpoint in every fold; held-out group {held_out!r} "
                        f"leaves missing endpoints: {missing_fold}"
                    )
                local_center, local_scale = self._target_stats(y[train], mask[train])
                local_y = torch.where(
                    mask[train],
                    (y[train] - local_center.unsqueeze(0)) / local_scale.unsqueeze(0),
                    torch.zeros_like(y[train]),
                )
                local_theta, _ = self._fit_matrix(x[train], local_y, mask[train])
                oof[held] = (x[held] @ local_theta) * local_scale.unsqueeze(0) + local_center.unsqueeze(0)
                oof_valid[held] = mask[held]
            self.oof_predictions, self.oof_mask = oof.detach(), oof_valid.detach()
        return self

    @property
    def frozen(self) -> bool:
        return self.theta is not None

    def predict(self, features: Sequence[Sequence[float]] | torch.Tensor) -> torch.Tensor:
        if self.theta is None or self.target_center is None or self.target_scale is None:
            raise RuntimeError("predictor must be fitted before prediction")
        prediction = self._design(features) @ self.theta.detach()
        return prediction * self.target_scale.detach().unsqueeze(0) + self.target_center.detach().unsqueeze(0)

    def oof_residual(self, targets: torch.Tensor) -> torch.Tensor:
        if self.oof_predictions is None or self.oof_mask is None:
            raise RuntimeError("cross-fitted predictions are not available")
        y = torch.as_tensor(targets, dtype=self.oof_predictions.dtype, device=self.oof_predictions.device)
        if y.shape != self.oof_predictions.shape:
            raise ValueError("targets do not match OOF predictions")
        if torch.any(self.oof_mask & ~torch.isfinite(y)):
            raise ValueError("observed OOF targets must be finite")
        return torch.where(self.oof_mask, self.oof_predictions - y, torch.zeros_like(y))

    def diagnostics(self) -> dict[str, Any]:
        if self.theta is None or self.gram is None:
            raise RuntimeError("predictor is not fitted")
        # The shared Gram matrix is paired with a different fixed regularizer
        # for each output head.  Report head-specific conditioning instead of
        # accidentally passing a tuple to ``torch`` or collapsing the heads
        # to one arbitrary lambda.
        eye = torch.eye(self.gram.shape[0], dtype=self.gram.dtype, device=self.gram.device)
        head_grams = self.head_grams or tuple(self.gram for _ in self.ridge_lambdas)
        condition_numbers = [
            float(torch.linalg.cond(head_gram + ridge * eye))
            for head_gram, ridge in zip(head_grams, self.ridge_lambdas, strict=True)
        ]
        minimum_eigenvalues = [
            float(torch.linalg.eigvalsh(head_gram + ridge * eye).min())
            for head_gram, ridge in zip(head_grams, self.ridge_lambdas, strict=True)
        ]
        return {
            "q": self.q,
            "basis": self.basis,
            "feature_map_version": self.feature_map.version,
            "feature_map_hash": self.feature_map.fingerprint,
            "endpoint_schema_hash": self.schema.fingerprint,
            "sample_count": self.sample_count,
            "feature_dimension": int(self.theta.shape[0]),
            "endpoint_dimension": self.schema.dimension,
            "target_center": self.target_center.detach().cpu().tolist() if self.target_center is not None else None,
            "target_scale": self.target_scale.detach().cpu().tolist() if self.target_scale is not None else None,
            "condition_number": max(condition_numbers),
            "condition_numbers": condition_numbers,
            "minimum_eigenvalue": min(minimum_eigenvalues),
            "minimum_eigenvalues": minimum_eigenvalues,
            "oof_available": self.oof_predictions is not None,
        }

    def state_dict(self) -> dict[str, Any]:
        if not self.frozen:
            raise RuntimeError("cannot serialize an unfitted predictor")
        return {
            "method_version": SCI_METHOD_VERSION,
            "q": self.q,
            "basis": self.basis,
            "ridge_lambda": self.ridge_lambda,
            "ridge_lambdas": self.ridge_lambdas,
            "theta": self.theta.detach().cpu().tolist(),
            "gram": self.gram.detach().cpu().tolist() if self.gram is not None else None,
            "head_grams": [gram.detach().cpu().tolist() for gram in self.head_grams],
            "target_center": self.target_center.detach().cpu().tolist() if self.target_center is not None else None,
            "target_scale": self.target_scale.detach().cpu().tolist() if self.target_scale is not None else None,
            "schema": self.schema.to_dict(),
            "feature_map": self.feature_map.to_dict(),
            "diagnostics": self.diagnostics(),
        }


@dataclass(frozen=True)
class BoundarySnapshot:
    task_id: str
    trajectory_id: str
    run_id: str
    session_id: str
    boundary_id: str
    prefix: str
    action_mask: tuple[bool, ...]
    old_log_probs: tuple[float, ...]
    tool_ledger: tuple[Mapping[str, Any], ...]
    workspace_hashes: Mapping[str, str]
    dependency_hashes: Mapping[str, str]
    rng_state: str
    execution_budget_initial: float
    execution_budget_remaining: float
    policy_version: str
    tokenizer_version: str
    sampling_config: Mapping[str, Any]
    feature_map_version: str
    feature_map_hash: str
    segment_rule_version: str
    snapshot_id: str
    snapshot_status: str = "captured"
    snapshot_cost: float = 0.0

    def __post_init__(self) -> None:
        if not all(_strict_real(value) for value in (self.execution_budget_initial, self.execution_budget_remaining, self.snapshot_cost)):
            raise ValueError("snapshot budgets/cost must be finite real numbers")
        if not self.boundary_id or not self.snapshot_id:
            raise ValueError("boundary and snapshot IDs are required")
        if len(self.action_mask) != len(self.old_log_probs):
            raise ValueError("action mask and old log probabilities must align")
        if any(not isinstance(value, bool) for value in self.action_mask):
            raise ValueError("action_mask must contain booleans")
        if any(not _strict_real(value) for value in self.old_log_probs):
            raise ValueError("old log probabilities must be finite")
        if not _finite(self.execution_budget_initial) or not _finite(self.execution_budget_remaining):
            raise ValueError("execution budgets must be finite")
        if self.execution_budget_remaining < 0 or self.execution_budget_remaining > self.execution_budget_initial:
            raise ValueError("remaining execution budget is inconsistent")
        if self.snapshot_status not in {"captured", "restored", "restore_failed"}:
            raise ValueError("invalid snapshot status")
        if self.snapshot_cost < 0 or not _finite(self.snapshot_cost):
            raise ValueError("snapshot cost must be finite and non-negative")

    @property
    def digest(self) -> str:
        return content_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return _canonical(asdict(self))


@dataclass(frozen=True)
class RestoreAudit:
    boundary_id: str
    status: str
    snapshot_digest: str
    restored_digest: str | None
    cost: float
    reasons: tuple[str, ...] = ()
    evidence_kind: str = "unspecified"

    def __post_init__(self) -> None:
        if not _strict_real(self.cost):
            raise ValueError("restore audit cost must be a finite real number")
        if self.status not in {"passed", "failed"}:
            raise ValueError("invalid restore audit status")
        if not _finite(self.cost) or self.cost < 0:
            raise ValueError("restore audit cost must be finite and non-negative")
        if self.evidence_kind not in RESTORE_EVIDENCE_KINDS:
            raise ValueError("invalid restore evidence kind")

    @property
    def passed(self) -> bool:
        return self.status == "passed" and not self.reasons

    @property
    def research_eligible(self) -> bool:
        """Structural restore success is not durable proof.

        A response that merely echoed the captured snapshot (the protocol
        probe's ``self_attested`` mode) satisfies ``passed`` but can never
        earn research eligibility.
        """

        return self.passed and self.evidence_kind == "durable"


class StateAuditOracle:
    """Fail-closed checks for restoring a boundary before a fresh branch."""

    def audit(
        self,
        snapshot: BoundarySnapshot,
        restored: Mapping[str, Any] | None,
        *,
        cost: float = 0.0,
        require_captured_state: bool = False,
        evidence_kind: str = "unspecified",
    ) -> RestoreAudit:
        reasons: list[str] = []
        invalid_cost = not _strict_real(cost) or cost < 0
        cost_value = 0.0 if invalid_cost else float(cost)
        if evidence_kind not in RESTORE_EVIDENCE_KINDS:
            raise ValueError("invalid restore evidence kind")
        if invalid_cost:
            reasons.append("invalid_restore_cost")
        if restored is None:
            reasons.append("restore_returned_none")
        elif not isinstance(restored, Mapping):
            # Restore callbacks cross a process/plugin boundary.  A malformed
            # response must become an auditable failed restore rather than an
            # AttributeError that bypasses the boundary contract entirely.
            reasons.append("restore_not_mapping")
        else:
            if restored.get("boundary_id") != snapshot.boundary_id:
                reasons.append("boundary_id_mismatch")
            if restored.get("snapshot_id") != snapshot.snapshot_id:
                reasons.append("snapshot_id_mismatch")
            if restored.get("policy_version") != snapshot.policy_version:
                reasons.append("policy_version_mismatch")
            if restored.get("feature_map_hash") != snapshot.feature_map_hash:
                reasons.append("feature_map_hash_mismatch")
            if restored.get("segment_rule_version") != snapshot.segment_rule_version:
                reasons.append("segment_rule_version_mismatch")
            if restored.get("task_id") != snapshot.task_id:
                reasons.append("task_id_mismatch")
            if restored.get("trajectory_id") != snapshot.trajectory_id:
                reasons.append("trajectory_id_mismatch")
            if restored.get("run_id") != snapshot.run_id:
                reasons.append("run_id_mismatch")
            if restored.get("session_id") != snapshot.session_id:
                reasons.append("session_id_mismatch")
            if restored.get("tokenizer_version") != snapshot.tokenizer_version:
                reasons.append("tokenizer_version_mismatch")
            if "feature_map_version" in restored and restored.get("feature_map_version") != snapshot.feature_map_version:
                reasons.append("feature_map_version_mismatch")
            if require_captured_state and "snapshot_digest" not in restored:
                reasons.append("snapshot_digest_missing")
            elif "snapshot_digest" in restored and restored.get("snapshot_digest") != snapshot.digest:
                reasons.append("snapshot_digest_mismatch")
            # Adapters may return these captured state projections.  Compare
            # them whenever present; legacy replay fixtures may omit them,
            # so omission remains a separate adapter-level completeness
            # decision rather than silently comparing fabricated defaults.
            for field_name, expected in (
                ("prefix", snapshot.prefix),
                ("action_mask", snapshot.action_mask),
                ("old_log_probs", snapshot.old_log_probs),
                ("tool_ledger", snapshot.tool_ledger),
            ):
                if require_captured_state and field_name not in restored:
                    reasons.append(f"{field_name}_missing")
                elif field_name in restored and _canonical(restored.get(field_name)) != _canonical(expected):
                    reasons.append(f"{field_name}_mismatch")
            if restored.get("dependency_hashes") != snapshot.dependency_hashes:
                reasons.append("dependency_hash_mismatch")
            if restored.get("sampling_config") != snapshot.sampling_config:
                reasons.append("sampling_config_mismatch")
            if restored.get("rng_state") != snapshot.rng_state:
                reasons.append("rng_state_mismatch")
            remaining = restored.get("execution_budget_remaining")
            if not _strict_real(remaining) or float(remaining) != snapshot.execution_budget_remaining:
                reasons.append("remaining_budget_mismatch")
            # Never substitute the snapshot's own value for a missing
            # projection: an empty restore response must not pass the captured
            # state check by echoing fabricated defaults.
            if "execution_budget_initial" in restored:
                initial = restored.get("execution_budget_initial")
                if not _strict_real(initial) or float(initial) != snapshot.execution_budget_initial:
                    reasons.append("initial_budget_mismatch")
            elif require_captured_state:
                reasons.append("initial_budget_missing")
            if "artifact_hashes" in restored:
                if restored.get("artifact_hashes") != snapshot.workspace_hashes:
                    reasons.append("workspace_hash_mismatch")
            elif require_captured_state:
                reasons.append("artifact_hashes_missing")
            # Future-randomness independence belongs to the fresh-branch
            # callback (FreshBranchRunner verifies ``rng_id`` and
            # ``rng_independent`` there).  A restore response only needs to
            # reproduce the captured RNG state; requiring the branch attestation
            # here incorrectly conflates the two contracts.
        # A malformed non-mapping restore may not be JSON serializable.  The
        # failed audit itself is the durable record; do not let digesting it
        # re-raise and bypass fail-closed handling.
        restored_digest = content_hash(restored) if isinstance(restored, Mapping) else None
        status = "passed" if not reasons else "failed"
        return RestoreAudit(
            snapshot.boundary_id,
            status,
            snapshot.digest,
            restored_digest,
            max(0.0, cost_value),
            tuple(reasons),
            evidence_kind,
        )


@dataclass(frozen=True)
class TrajectoryRecord:
    """Auditable trajectory envelope produced by a harness adapter."""

    task_id: str
    trajectory_id: str
    prompt_id: str
    run_id: str
    session_id: str
    actor_checkpoint: str
    tokenizer_version: str
    policy_version: str
    sampling_config: Mapping[str, Any]
    terminal_observations: tuple[EndpointObservation, ...]
    terminal_evaluator_id: str
    terminal_evaluator_version: str
    terminal_cost: float
    execution_budget_initial: float
    execution_budget_remaining: float
    restore_contract_version: str
    restore_status: str
    restore_cost: float
    feature_map_version: str
    feature_map_hash: str
    segment_rule_version: str
    trajectory_complete: bool
    failure_class: str | None = None
    research_eligible: bool = False
    actions: tuple[ActionRecord, ...] = ()
    terminal_score: tuple[float | None, ...] | None = None
    segment_rule_hash: str | None = None
    # Endpoint schemas may use different HF evaluator services per vector
    # dimension.  Keep the historical scalar provenance fields, but retain
    # the complete endpoint-level mapping whenever a vector run needs it.
    terminal_evaluator_ids: Mapping[str, str] = field(default_factory=dict)
    terminal_evaluator_versions: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.trajectory_complete, bool) or not isinstance(self.research_eligible, bool):
            raise ValueError("trajectory_complete and research_eligible must be boolean")
        if self.research_eligible and not self.trajectory_complete:
            raise ValueError("research_eligible trajectories must be complete")
        if not self.task_id or not self.trajectory_id or not self.policy_version:
            raise ValueError("trajectory identity and policy version are required")
        if (
            not _strict_real(self.terminal_cost)
            or not _strict_real(self.restore_cost)
            or self.terminal_cost < 0
            or self.restore_cost < 0
        ):
            raise ValueError("trajectory costs must be non-negative")
        if (
            not _strict_real(self.execution_budget_initial)
            or not _strict_real(self.execution_budget_remaining)
            or self.execution_budget_initial < 0
            or self.execution_budget_remaining < 0
            or self.execution_budget_remaining > self.execution_budget_initial
        ):
            raise ValueError("trajectory execution budget is inconsistent")
        if self.failure_class is not None and self.failure_class not in FAILURE_CLASSES:
            raise ValueError("unknown trajectory failure class")
        if self.restore_status not in {"not_attempted", "passed", "failed"}:
            raise ValueError("invalid trajectory restore status")
        if self.terminal_score is not None and any(
            value is not None and not _strict_real(value) for value in self.terminal_score
        ):
            raise ValueError("terminal endpoint scores must be finite or null")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for action in payload.get("actions", ()):
            if action.get("action_mask") is None:
                action["action_mask"] = bool(action["is_policy_action"])
        return _canonical(payload)

    @property
    def fingerprint(self) -> str:
        return content_hash(self.to_dict())


@dataclass(frozen=True)
class BoundaryRecord:
    """One causal boundary and all information used by the query design."""

    boundary_id: str
    trajectory_id: str
    segment_id: int
    start: int
    end: int
    duration: int
    boundary_type: str
    snapshot: BoundarySnapshot
    feature_vector: tuple[float, ...]
    prediction: tuple[float | None, ...]
    oof_residual: tuple[float | None, ...] | None
    query_selected: Mapping[str, bool]
    query_inclusion_probability: Mapping[str, float]
    branch_count: Mapping[str, int]
    gold_or_proxy: Mapping[str, str]
    gold_score: Mapping[str, float | None]
    query_cost: float
    restore_cost: float
    branch_status: Mapping[str, str]
    failure_reason: Mapping[str, str | None]
    budget_remaining_after: float
    # Response propensity and validity are separate from selection propensity:
    # a selected query can fail during restore/branch/evaluation and must stay
    # visible in the audit ledger rather than being treated as an unobserved
    # gold label.
    response_probability: Mapping[str, float | None] = field(default_factory=dict)
    valid_observation: Mapping[str, bool] = field(default_factory=dict)
    # Explicit references retained in the sidecar so an auditor can connect a
    # row back to the immutable harness state and fresh-branch attestations.
    boundary_state_ref: str | None = None
    evidence_state: Mapping[str, Any] = field(default_factory=dict)
    remaining_execution_budget: float | None = None
    branch_policy_version: Mapping[str, str] = field(default_factory=dict)
    branch_rng_id: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.boundary_id, str) or not self.boundary_id:
            raise ValueError("boundary_id must be a non-empty string")
        if not isinstance(self.trajectory_id, str) or not self.trajectory_id:
            raise ValueError("trajectory_id must be a non-empty string")
        if (
            any(isinstance(value, bool) or not isinstance(value, int) for value in (self.segment_id, self.start, self.end, self.duration))
            or self.segment_id < 0
            or self.start < 0
            or self.end <= self.start
            or self.duration != self.end - self.start
        ):
            raise ValueError("boundary indices and duration must be non-negative integers")
        if not isinstance(self.boundary_type, str) or not self.boundary_type:
            raise ValueError("boundary_type must be a non-empty string")
        if not isinstance(self.snapshot, BoundarySnapshot):
            raise ValueError("boundary snapshot must be a BoundarySnapshot")
        for name, values in (
            ("feature_vector", self.feature_vector),
            ("prediction", self.prediction),
            ("oof_residual", self.oof_residual),
        ):
            if values is None and name == "oof_residual":
                continue
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                raise ValueError(f"{name} must be a numeric sequence")
            if name == "feature_vector" and any(value is None for value in values):
                raise ValueError("feature_vector entries must be finite real numbers")
            if any(value is not None and not _strict_real(value) for value in values):
                raise ValueError(f"{name} entries must be finite real numbers or null")
        for name, values in (("query_selected", self.query_selected), ("valid_observation", self.valid_observation)):
            if not isinstance(values, Mapping) or any(not isinstance(key, str) or not isinstance(value, bool) for key, value in values.items()):
                raise ValueError(f"{name} must map string keys to booleans")
        for name, values in (("query_inclusion_probability", self.query_inclusion_probability), ("response_probability", self.response_probability)):
            if not isinstance(values, Mapping) or any(
                not isinstance(key, str)
                or (value is None and name == "query_inclusion_probability")
                or (value is not None and (not _strict_real(value) or not 0 <= float(value) <= 1))
                for key, value in values.items()
            ):
                raise ValueError(f"{name} must map string keys to probabilities")
        if not isinstance(self.branch_count, Mapping) or any(
            not isinstance(key, str) or isinstance(value, bool) or not isinstance(value, int) or value < 0
            for key, value in self.branch_count.items()
        ):
            raise ValueError("branch_count must map string keys to non-negative integers")
        for name, values in (("gold_or_proxy", self.gold_or_proxy), ("branch_status", self.branch_status), ("branch_policy_version", self.branch_policy_version)):
            if not isinstance(values, Mapping) or any(not isinstance(key, str) or not isinstance(value, str) or not value for key, value in values.items()):
                raise ValueError(f"{name} must map string keys to non-empty strings")
        # A restore failure never starts a branch and therefore has no RNG ID.
        if not isinstance(self.branch_rng_id, Mapping) or any(
            not isinstance(key, str) or not isinstance(value, str)
            or (not value and self.branch_status.get(key) == "completed")
            for key, value in self.branch_rng_id.items()
        ):
            raise ValueError("branch_rng_id must map string keys to strings, non-empty for completed branches")
        if not isinstance(self.gold_score, Mapping) or any(
            not isinstance(key, str) or (value is not None and not _strict_real(value))
            for key, value in self.gold_score.items()
        ):
            raise ValueError("gold_score must map string keys to finite real numbers or null")
        if not isinstance(self.failure_reason, Mapping) or any(
            not isinstance(key, str) or (value is not None and not isinstance(value, str))
            for key, value in self.failure_reason.items()
        ):
            raise ValueError("failure_reason must map string keys to strings or null")
        if self.snapshot.boundary_id != self.boundary_id or self.snapshot.trajectory_id != self.trajectory_id:
            raise ValueError("boundary snapshot identity mismatch")
        if (
            not _strict_real(self.query_cost)
            or not _strict_real(self.restore_cost)
            or not _strict_real(self.budget_remaining_after)
            or self.query_cost < 0
            or self.restore_cost < 0
            or self.budget_remaining_after < 0
        ):
            raise ValueError("boundary costs/budget must be non-negative")
        if self.remaining_execution_budget is not None and (
            not _strict_real(self.remaining_execution_budget)
            or self.remaining_execution_budget < 0
            or self.remaining_execution_budget > self.snapshot.execution_budget_remaining
        ):
            raise ValueError("remaining execution budget is inconsistent with the boundary snapshot")

    def to_dict(self) -> dict[str, Any]:
        return _canonical(asdict(self))


class ScientificEndpointEvaluator:
    """Adapter around a pre-registered independent endpoint evaluator."""

    def __init__(self, schema: EndpointSchema, *, evaluator_id: str, evaluator_version: str, evaluate_fn: Callable[[Any], Sequence[EndpointObservation]]):
        if not evaluator_id or not evaluator_version:
            raise ValueError("evaluator identity/version are required")
        self.schema = schema
        self.evaluator_id = evaluator_id
        self.evaluator_version = evaluator_version
        self._evaluate_fn = evaluate_fn

    def evaluate(self, artifact: Any, *, cost: float) -> tuple[EndpointObservation, ...]:
        if not _strict_real(cost) or cost < 0:
            raise ValueError("evaluator cost must be finite and non-negative")
        observations = tuple(self._evaluate_fn(artifact))
        if len({o.endpoint_id for o in observations}) != len(observations):
            raise ValueError("endpoint evaluator returned duplicate endpoint observations")
        for obs in observations:
            if obs.evaluator_id not in {"", self.evaluator_id} or obs.evaluator_version not in {"", self.evaluator_version}:
                raise ValueError("endpoint evaluator provenance mismatch")
            # Validate endpoint identity, fidelity, status, range, and the
            # evaluator version declared by the endpoint schema before
            # attaching this adapter's provenance.  In particular, an HF
            # evaluator must not accidentally emit an LF/MF observation.
            self.schema.validate_observation(obs, allow_empty_provenance=True)
            endpoint = self.schema.endpoints[self.schema.endpoint_index(obs.endpoint_id)]
            fidelity = next(item for item in endpoint.fidelities if item.fidelity_id == obs.fidelity_id)
            if fidelity.evaluator_id != self.evaluator_id or fidelity.evaluator_version != self.evaluator_version:
                raise ValueError("evaluator callback does not match the endpoint fidelity contract")
        return tuple(
            replace(
                o,
                cost=cost if o.cost == 0 else o.cost,
                evaluator_id=self.evaluator_id,
                evaluator_version=self.evaluator_version,
            )
            for o in observations
        )


@dataclass(frozen=True)
class QueryCandidate:
    boundary_id: str
    endpoint_id: str
    sensitivity: float
    residual_risk: float
    scientific_priority: float
    cost: float
    branch_count: int = 1

    @property
    def estimated_gradient_risk(self) -> float:
        """Estimated ``a_b`` in the variance objective.

        The caller is responsible for constructing ``sensitivity`` and
        ``residual_risk`` from held-out data; the product is not presented as
        an oracle or a regret bound.
        """

        return max(0.0, self.sensitivity) * max(0.0, self.residual_risk) * max(0.0, self.scientific_priority)

    @property
    def utility(self) -> float:
        """Backward-compatible alias for the estimated gradient risk."""

        return self.estimated_gradient_risk

    def __post_init__(self) -> None:
        if not _strict_real(self.cost) or any(not _strict_real(value) for value in (self.sensitivity, self.residual_risk, self.scientific_priority)):
            raise ValueError("query candidate values must be finite real numbers")
        if not self.boundary_id or not self.endpoint_id or self.cost <= 0 or not _finite(self.cost):
            raise ValueError("query candidate identity and positive cost are required")
        if isinstance(self.branch_count, bool) or not isinstance(self.branch_count, int) or self.branch_count < 1:
            raise ValueError("branch_count must be a positive integer")
        for value in (self.sensitivity, self.residual_risk, self.scientific_priority):
            if not _finite(value) or value < 0:
                raise ValueError("query utility terms must be finite and non-negative")


@dataclass(frozen=True)
class QueryDecision:
    candidates: tuple[QueryCandidate, ...]
    selected_ids: tuple[str, ...]
    inclusion_probabilities: Mapping[str, float]
    pair_inclusion_probabilities: Mapping[str, float]
    mode: str
    hard_budget: bool
    budget: float
    planned_cost: float
    seed: int
    response_probabilities: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.hard_budget, bool):
            raise ValueError("hard_budget must be boolean")
        if not _strict_real(self.budget) or not _strict_real(self.planned_cost):
            raise ValueError("query decision costs must be finite real numbers")
        if self.mode not in {"uniform", "adaptive"}:
            raise ValueError("query decision mode must be uniform or adaptive")
        if not _finite(self.budget) or self.budget < 0:
            raise ValueError("query decision budget must be finite and non-negative")
        if not _finite(self.planned_cost) or self.planned_cost < 0:
            raise ValueError("planned query cost must be finite and non-negative")
        if self.hard_budget and self.planned_cost > self.budget + 1e-8:
            raise ValueError("hard-budget query decision exceeds budget")
        keys = [BudgetedQuerySampler._key(candidate) for candidate in self.candidates]
        if len(set(keys)) != len(keys):
            raise ValueError("query decision candidates must have unique keys")
        selected = tuple(self.selected_ids)
        if len(set(selected)) != len(selected) or not set(selected).issubset(keys):
            raise ValueError("selected query IDs must be unique candidate keys")
        candidate_by_key = dict(zip(keys, self.candidates, strict=True))
        expected_cost = sum(candidate_by_key[key].cost for key in selected)
        if abs(float(self.planned_cost) - expected_cost) > 1e-8:
            raise ValueError("planned query cost does not match selected candidates")
        for name, values in (
            ("inclusion", self.inclusion_probabilities),
            ("response", self.response_probabilities),
        ):
            unknown = set(values) - set(keys)
            if unknown:
                raise ValueError(f"{name} probabilities contain unknown query keys: {sorted(unknown)}")
        expected_pairs = {"::".join(pair) for pair in combinations(keys, 2)}
        unknown_pairs = set(self.pair_inclusion_probabilities) - expected_pairs
        if unknown_pairs:
            raise ValueError(f"pair inclusion probabilities contain unknown query pairs: {sorted(unknown_pairs)}")
        for name, values in (
            ("inclusion", self.inclusion_probabilities),
            ("pair inclusion", self.pair_inclusion_probabilities),
            ("response", self.response_probabilities),
        ):
            for probability in values.values():
                if not _strict_real(probability) or not 0 <= float(probability) <= 1:
                    raise ValueError(f"{name} probabilities must be finite and in [0,1]")
        inclusion = {key: float(value) for key, value in self.inclusion_probabilities.items()}
        pair_lookup = {
            "::".join(pair): pair for pair in combinations(keys, 2)
        }
        for pair_key, probability in self.pair_inclusion_probabilities.items():
            pair = pair_lookup.get(pair_key)
            if pair is None:
                continue
            left, right = pair
            p_left = inclusion.get(left, 0.0)
            p_right = inclusion.get(right, 0.0)
            lower = max(0.0, p_left + p_right - 1.0)
            upper = min(p_left, p_right)
            if float(probability) < lower - 1e-8 or float(probability) > upper + 1e-8:
                raise ValueError(
                    "pair inclusion probabilities must be consistent with endpoint marginals"
                )
        for key in selected:
            if float(self.inclusion_probabilities.get(key, 0.0)) <= 0:
                raise ValueError("selected queries require positive inclusion probabilities")

    @property
    def selected(self) -> tuple[QueryCandidate, ...]:
        selected = set(self.selected_ids)
        return tuple(c for c in self.candidates if c.boundary_id + "::" + c.endpoint_id in selected)


class BudgetedQuerySampler:
    """Budget-aware sampler that records first- and second-order propensities."""

    @staticmethod
    def _key(candidate: QueryCandidate) -> str:
        return f"{candidate.boundary_id}::{candidate.endpoint_id}"

    def sample(
        self,
        candidates: Sequence[QueryCandidate],
        budget: float,
        *,
        mode: str = "uniform",
        hard_budget: bool = False,
        seed: int = 0,
    ) -> QueryDecision:
        if not _strict_real(budget) or budget < 0:
            raise ValueError("query budget must be finite and non-negative")
        _strict_bool(hard_budget, "hard_budget")
        if mode not in {"uniform", "adaptive"}:
            raise ValueError("query mode must be uniform or adaptive")
        items = tuple(candidates)
        keys = [self._key(c) for c in items]
        if len(set(keys)) != len(keys):
            raise ValueError("query candidate keys must be unique")
        if not items:
            return QueryDecision((), (), {}, {}, mode, hard_budget, float(budget), 0.0, seed)
        rng = random.Random(seed)
        probs: dict[str, float] = {}
        selected: list[str] = []
        if hard_budget:
            # Exact first/second-order probabilities are available for equal
            # cost simple random sampling without replacement.  Other designs
            # are rejected instead of pretending their propensities are known.
            costs = {round(c.cost, 12) for c in items}
            if len(costs) != 1:
                raise ValueError("hard-budget sampler currently requires equal candidate costs")
            k = min(len(items), int(math.floor(budget / items[0].cost + 1e-12)))
            if k == 0:
                return QueryDecision(items, (), {key: 0.0 for key in keys}, {}, mode, True, float(budget), 0.0, seed)
            chosen = set(rng.sample(keys, k))
            p = k / len(items)
            probs = {key: p for key in keys}
            pair_p = (k * (k - 1)) / (len(items) * (len(items) - 1)) if len(items) > 1 else 0.0
            pairs = {"::".join(pair): pair_p for pair in combinations(keys, 2)}
            selected = [key for key in keys if key in chosen]
        else:
            if mode == "uniform":
                raw = [1.0 for _ in items]
            else:
                # The continuous relaxation minimizes sum(a_i / p_i) under
                # sum(cost_i * p_i) <= budget, hence p_i is proportional to
                # sqrt(a_i / cost_i), rather than a_i / cost_i.
                risks = [c.estimated_gradient_risk for c in items]
                if any(not _finite(risk) for risk in risks):
                    raise ValueError("combined gradient risks must be finite")
                log_weights = [
                    0.5 * (math.log(risk) - math.log(c.cost)) if risk > 0 else -math.inf
                    for c, risk in zip(items, risks, strict=True)
                ]
                largest = max(log_weights)
                if largest == -math.inf:
                    raw = [0.0 for _ in log_weights]
                else:
                    raw = [
                        math.exp(weight - largest) if weight != -math.inf else 0.0
                        for weight in log_weights
                    ]
                if any(risk > 0 and weight == 0 for risk, weight in zip(risks, raw, strict=True)):
                    raise ValueError("gradient risk range is too wide for positive floating-point propensities")
            # Solve p_i=min(1, scale * raw_i) for the expected cost.
            def expected(scale: float) -> float:
                return sum(c.cost * min(1.0, scale * weight) for c, weight in zip(items, raw, strict=True))

            positive_cost = sum(c.cost for c, weight in zip(items, raw, strict=True) if weight > 0)
            if budget >= positive_cost:
                probabilities = [1.0 if weight > 0 else 0.0 for weight in raw]
            elif budget > 0:
                lo, hi = 0.0, 1.0
                while expected(hi) < budget:
                    hi *= 2
                    if not math.isfinite(hi):
                        raise ValueError("query allocation scale exceeds floating-point range")
                for _ in range(80):
                    mid = (lo + hi) / 2
                    if expected(mid) <= budget:
                        lo = mid
                    else:
                        hi = mid
                probabilities = [min(1.0, lo * weight) for weight in raw]
            else:
                probabilities = [0.0 for _ in raw]
            for key, p in zip(keys, probabilities, strict=True):
                probs[key] = p
                if rng.random() < p:
                    selected.append(key)
            pairs = {"::".join(pair): probs[pair[0]] * probs[pair[1]] for pair in combinations(keys, 2)}
        planned = sum(c.cost for c in items if self._key(c) in selected)
        return QueryDecision(items, tuple(selected), probs, pairs, mode, hard_budget, float(budget), planned, seed)


@dataclass
class BudgetLedger:
    """Global training-audit ledger; all costs are explicit and additive."""

    budget: float
    consumed: dict[str, float] = field(default_factory=dict)
    reserved: dict[str, float] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    CATEGORIES = frozenset({"terminal", "restore", "branch", "audit", "pilot", "calibration", "selection", "query"})

    def __post_init__(self) -> None:
        if not _strict_real(self.budget) or self.budget < 0:
            raise ValueError("ledger budget must be finite and non-negative")
        if any(not _strict_real(v) for v in self.consumed.values()) or any(not _strict_real(v) for v in self.reserved.values()):
            raise ValueError("ledger amounts must be finite real numbers")
        self.consumed = {str(k): float(v) for k, v in self.consumed.items()}
        self.reserved = {str(k): float(v) for k, v in self.reserved.items()}
        for key in set(self.consumed) | set(self.reserved):
            if key not in self.CATEGORIES or not _finite(self.consumed.get(key, 0)) or not _finite(self.reserved.get(key, 0)) or self.consumed.get(key, 0) < 0 or self.reserved.get(key, 0) < 0:
                raise ValueError("invalid ledger category or amount")
        if self.spent + self.reserved_total > self.budget + 1e-9:
            raise ValueError("ledger starts over budget")

    @property
    def spent(self) -> float:
        return sum(self.consumed.values())

    @property
    def reserved_total(self) -> float:
        return sum(self.reserved.values())

    @property
    def remaining(self) -> float:
        return self.budget - self.spent - self.reserved_total

    def reserve(self, category: str, amount: float, *, reference: str = "") -> None:
        self._check(category, amount)
        if amount > self.remaining + 1e-9:
            raise ValueError("training audit budget exceeded by reservation")
        self.reserved[category] = self.reserved.get(category, 0.0) + float(amount)
        self.events.append({"event": "reserve", "category": category, "amount": float(amount), "reference": reference})

    def commit(self, category: str, amount: float, *, reference: str = "", planned: float | None = None) -> None:
        self._check(category, amount)
        if planned is not None and (not _finite(planned) or planned < 0):
            raise ValueError("planned ledger amount must be finite and non-negative")
        amount = float(amount)
        available_before = self.remaining
        held = min(amount, self.reserved.get(category, 0.0))
        if held:
            self.reserved[category] -= held
        # Only the amount beyond an existing reservation consumes free
        # budget.  Compare against the free budget *before* releasing the
        # reservation; using the post-release value would permit an
        # over-commit whenever ``amount`` exceeded the held reservation.
        if amount - held > available_before + 1e-9:
            if held:
                self.reserved[category] += held
            raise ValueError("training audit budget exceeded")
        self.consumed[category] = self.consumed.get(category, 0.0) + amount
        self.events.append({"event": "commit", "category": category, "amount": amount, "planned": planned, "reference": reference})

    def release(self, category: str, amount: float, *, reference: str = "") -> None:
        """Release an unused reservation without changing consumed cost."""
        self._check(category, amount)
        amount = float(amount)
        if amount > self.reserved.get(category, 0.0) + 1e-9:
            raise ValueError("cannot release more than the category reservation")
        self.reserved[category] = self.reserved.get(category, 0.0) - amount
        self.events.append({"event": "release", "category": category, "amount": amount, "reference": reference})

    def record(self, category: str, actual: float, *, planned: float | None = None, reference: str = "") -> None:
        self.commit(category, actual, planned=planned, reference=reference)

    def _check(self, category: str, amount: float) -> None:
        if category not in self.CATEGORIES:
            raise ValueError(f"unknown ledger category {category!r}")
        if not _strict_real(amount) or amount < 0:
            raise ValueError("ledger amount must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {"budget": self.budget, "consumed": dict(self.consumed), "reserved": dict(self.reserved), "events": list(self.events), "spent": self.spent, "remaining": self.remaining}

    @property
    def fingerprint(self) -> str:
        return content_hash(self.to_dict())


@dataclass(frozen=True)
class BranchResult:
    branch_id: str
    rng_id: str
    status: str
    observations: tuple[EndpointObservation, ...]
    execution_budget_remaining: float
    cost: float
    failure_class: str | None = None
    failure_reason: str | None = None
    # Kept as trailing defaults so existing positional construction and the
    # generic synthetic-math branch contract stay backward compatible.
    continuation_mode: str | None = None
    research_eligible: bool = False

    def __post_init__(self) -> None:
        if not _strict_real(self.execution_budget_remaining) or not _strict_real(self.cost) or self.execution_budget_remaining < 0 or self.cost < 0:
            raise ValueError("branch budgets/cost must be finite and non-negative real numbers")
        if self.continuation_mode is not None and self.continuation_mode not in CONTINUATION_MODES:
            raise ValueError("invalid branch continuation mode")
        if not isinstance(self.research_eligible, bool):
            raise ValueError("branch research_eligible must be boolean")


class FreshBranchRunner:
    """Run independent continuations through a caller-supplied callback."""

    def run(
        self,
        snapshot: BoundarySnapshot,
        restore_audit: RestoreAudit,
        *,
        branch_count: int,
        branch_fn: Callable[[BoundarySnapshot, str, float], Mapping[str, Any]],
        schema: EndpointSchema,
        policy_version: str,
        seed: int = 0,
        require_evaluator_provenance: bool = False,
        require_continuation_provenance: bool = False,
    ) -> tuple[BranchResult, ...]:
        if branch_count < 1:
            raise ValueError("branch_count must be positive")
        _strict_bool(require_continuation_provenance, "require_continuation_provenance")
        if not restore_audit.passed:
            return tuple(
                BranchResult(f"{snapshot.boundary_id}:branch:{i}", "", "restore_failed", (), snapshot.execution_budget_remaining, 0.0, "restore_failure", ";".join(restore_audit.reasons))
                for i in range(branch_count)
            )
        results = []
        for i in range(branch_count):
            rng_id = hashlib.sha256(f"{seed}:{snapshot.boundary_id}:{i}:{policy_version}".encode()).hexdigest()[:24]
            branch_result_id = f"{snapshot.boundary_id}:fresh:{rng_id}"
            try:
                returned = dict(branch_fn(snapshot, rng_id, snapshot.execution_budget_remaining))
                returned_branch_id = returned.get("branch_id")
                if returned_branch_id is None:
                    raise ValueError("branch did not attest its fresh branch identity")
                if str(returned_branch_id) != branch_result_id:
                    raise ValueError("branch did not attest the expected fresh branch identity")
                if returned.get("policy_version") != policy_version:
                    raise ValueError("branch policy version mismatch")
                if returned.get("rng_id") != rng_id:
                    raise ValueError("branch did not attest the supplied independent RNG id")
                if returned.get("rng_independent") is not True:
                    raise ValueError("branch did not attest independent future randomness")
                raw = returned.get("observations", ())
                observations = tuple(raw)
                if any(not isinstance(obs, EndpointObservation) for obs in observations):
                    raise ValueError("branch observations must be EndpointObservation objects")
                ids = {obs.endpoint_id for obs in observations}
                if len(ids) != len(observations):
                    raise ValueError("branch returned duplicate endpoint observations")
                if ids != set(schema.endpoint_ids):
                    missing = sorted(set(schema.endpoint_ids) - ids)
                    extra = sorted(ids - set(schema.endpoint_ids))
                    raise ValueError(
                        "branch returned an incomplete endpoint vector; "
                        f"missing={missing}, extra={extra}"
                    )
                for obs in observations:
                    schema.validate_observation(obs, allow_empty_provenance=not require_evaluator_provenance)
                    endpoint = schema.endpoints[schema.endpoint_index(obs.endpoint_id)]
                    if obs.fidelity_id != endpoint.hf_fidelity_id:
                        raise ValueError("fresh branches must return the registered HF fidelity")
                if "execution_budget_remaining" not in returned:
                    raise ValueError("branch did not attest its remaining execution budget")
                remaining = returned["execution_budget_remaining"]
                if not _strict_real(remaining) or not (0 <= remaining <= snapshot.execution_budget_remaining):
                    raise ValueError("branch did not inherit a valid remaining execution budget")
                cost = returned.get("cost", 0.0)
                if not _strict_real(cost) or cost < 0:
                    raise ValueError("branch cost must be finite and non-negative")
                # Continuation provenance is verified against the captured
                # boundary, never against values supplied by the branch itself.
                # A protocol probe may omit it (and stays non-research); a
                # research continuation must supply it and match exactly.
                provenance = returned.get("continuation_provenance")
                if provenance is None:
                    explicit = {
                        key: returned[key]
                        for key in (
                            "continuation_mode",
                            "prefix",
                            "prefix_hash",
                            "sampling_config",
                            "sampling_config_hash",
                            "policy_version",
                        )
                        if key in returned
                    }
                    provenance = explicit or None
                if require_continuation_provenance:
                    if returned.get("trajectory_complete") is not True:
                        raise ValueError("branch did not attest a complete terminal trajectory")
                    if not isinstance(provenance, Mapping):
                        raise ValueError("branch continuation provenance is missing")
                if provenance is not None and not isinstance(provenance, Mapping):
                    raise ValueError("branch continuation provenance must be a mapping")
                continuation_reasons = (
                    verify_continuation_provenance(
                        provenance,
                        prefix=snapshot.prefix,
                        sampling_config=snapshot.sampling_config,
                        policy_version=policy_version,
                    )
                    if isinstance(provenance, Mapping)
                    else ()
                )
                if require_continuation_provenance and continuation_reasons:
                    raise ValueError(
                        "branch continuation provenance failed: " + ";".join(continuation_reasons)
                    )
                continuation_mode = None
                if isinstance(provenance, Mapping):
                    raw_mode = provenance.get("continuation_mode")
                    continuation_mode = raw_mode if raw_mode in CONTINUATION_MODES else None
                results.append(
                    BranchResult(
                        branch_result_id,
                        rng_id,
                        "completed",
                        observations,
                        remaining,
                        cost,
                        continuation_mode=continuation_mode,
                        research_eligible=(
                            restore_audit.research_eligible
                            and isinstance(provenance, Mapping)
                            and not continuation_reasons
                            and continuation_mode in RESEARCH_CONTINUATION_MODES
                        ),
                    )
                )
            except TimeoutError as exc:
                results.append(BranchResult(branch_result_id, rng_id, "failed", (), snapshot.execution_budget_remaining, 0.0, "timeout", str(exc)))
            except EvaluatorExecutionError as exc:
                results.append(BranchResult(branch_result_id, rng_id, "failed", (), snapshot.execution_budget_remaining, 0.0, "evaluator_failure", str(exc)))
            except BranchExecutionError as exc:
                results.append(BranchResult(branch_result_id, rng_id, "failed", (), snapshot.execution_budget_remaining, 0.0, "branch_failure", str(exc)))
            except Exception as exc:  # callback boundary: failures are data, not fabricated labels
                results.append(BranchResult(branch_result_id, rng_id, "failed", (), snapshot.execution_budget_remaining, 0.0, "branch_failure", f"{type(exc).__name__}: {exc}"))
        return tuple(results)


@dataclass(frozen=True)
class ScienceRoundResult:
    """Auditable output of one frozen-policy PolySeg-Sci round."""

    corrected_boundary_values: torch.Tensor
    segment_gradient: torch.Tensor
    decision: QueryDecision
    observations: Mapping[str, EndpointObservation]
    boundary_records: tuple[BoundaryRecord, ...]
    ledger: Mapping[str, Any]
    verification: "VerificationResult"
    audit: Mapping[str, Any]
    # Preserve endpoint covariance instead of dropping the third return value
    # from mean_observations.  A mapping is used because each boundary may have
    # a different number of successful fresh branches.
    covariance_by_boundary: Mapping[str, tuple[tuple[float, ...], ...]] = field(default_factory=dict)


def run_science_round(
    *,
    schema: EndpointSchema,
    feature_map: FeatureMapContract,
    policy_version: str,
    boundary_ids: Sequence[str],
    boundary_snapshots: Mapping[str, BoundarySnapshot],
    boundary_intervals: Sequence[tuple[int, int]],
    boundary_features: Sequence[Sequence[float]],
    boundary_predictions: torch.Tensor,
    terminal_values: Sequence[float],
    segment_scores: torch.Tensor,
    candidates: Sequence[QueryCandidate],
    query_budget: float,
    branch_fn: Callable[[BoundarySnapshot, str, float], Mapping[str, Any]],
    restore_fn: Callable[[BoundarySnapshot], Mapping[str, Any] | tuple[Mapping[str, Any], float]],
    ledger: BudgetLedger,
    query_mode: str = "uniform",
    hard_budget: bool = False,
    branch_count: int = 1,
    restore_cost: float = 0.0,
    planned_branch_cost: float | None = None,
    audit_cost: float = 0.0,
    seed: int = 0,
    oof_residuals: Sequence[Sequence[float | None]] | None = None,
    terminal_cost: float = 0.0,
    terminal_evaluator_ids: Mapping[str, str] | None = None,
    terminal_evaluator_versions: Mapping[str, str] | None = None,
    response_probabilities: Mapping[str, float] | None = None,
    response_model_validated: bool = False,
    require_continuation_provenance: bool = False,
) -> ScienceRoundResult:
    """Execute one complete frozen-policy science round.

    The caller supplies the already-frozen predictor output and action scores;
    this function owns the causal query/restore/branch/correction/gradient
    path.  ``boundary_ids`` names the *internal* boundaries, so predictions
    have shape ``[len(boundary_ids)+2, R]`` and scores have shape ``[M, D]``.
    Every selected boundary is restored before a fresh branch is run.  Failed
    restores/branches remain audit records and never create a scientific value.
    ``restore_cost`` is the per-selected-boundary restore reservation, and
    ``planned_branch_cost`` is the per-branch reservation.  A selected round
    must provide the latter explicitly so the global ledger can reject an
    over-budget round before invoking callbacks. ``audit_cost`` is the
    per-selected-boundary cost of the state/evidence audit itself.
    """

    # Validate all control flags before constructing a sampler or invoking any
    # restore/branch callback.  A malformed truthy string such as ``"false"``
    # must fail before it can cause billable scientific side effects.
    _strict_bool(response_model_validated, "response_model_validated")
    _strict_bool(require_continuation_provenance, "require_continuation_provenance")
    ids = tuple(str(value) for value in boundary_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("boundary_ids must be unique and non-empty")
    if set(boundary_snapshots) != set(ids):
        raise ValueError("boundary_snapshots must cover exactly the internal boundaries")
    if len(boundary_features) != len(ids):
        raise ValueError("boundary_features must have one row per internal boundary")
    if len(boundary_intervals) != len(ids):
        raise ValueError("boundary_intervals must have one interval per internal boundary")
    if not policy_version:
        raise ValueError("policy_version is required")
    predictions = torch.as_tensor(boundary_predictions, dtype=torch.float64).clone()
    terminal = torch.as_tensor(terminal_values, dtype=torch.float64).reshape(-1)
    if predictions.shape != (len(ids) + 2, schema.dimension):
        raise ValueError("boundary_predictions must have shape [internal_boundaries+2, endpoint_dimension]")
    if terminal.numel() != schema.dimension or not torch.isfinite(terminal).all():
        raise ValueError("terminal_values must match the endpoint schema and be finite")
    schema.validate_vector(terminal.tolist(), ["observed"] * schema.dimension)
    if terminal_evaluator_ids is None or terminal_evaluator_versions is None:
        raise ValueError("terminal evaluator provenance is required for a science round")
    expected_terminal_ids = {
        endpoint.endpoint_id: next(
            fidelity.evaluator_id
            for fidelity in endpoint.fidelities
            if fidelity.fidelity_id == endpoint.hf_fidelity_id
        )
        for endpoint in schema.endpoints
    }
    expected_terminal_versions = {
        endpoint.endpoint_id: next(
            fidelity.evaluator_version
            for fidelity in endpoint.fidelities
            if fidelity.fidelity_id == endpoint.hf_fidelity_id
        )
        for endpoint in schema.endpoints
    }
    if dict(terminal_evaluator_ids) != expected_terminal_ids:
        raise ValueError("terminal evaluator provenance does not match the endpoint schema")
    if dict(terminal_evaluator_versions) != expected_terminal_versions:
        raise ValueError("terminal evaluator provenance does not match the endpoint schema")
    if not torch.isfinite(predictions).all():
        raise ValueError("boundary_predictions must be finite")
    scores = torch.as_tensor(segment_scores, dtype=torch.float64)
    if scores.ndim == 2:
        scores = scores.unsqueeze(0)
    if scores.ndim != 3 or scores.shape[0] != 1 or scores.shape[1] != predictions.shape[0] - 1:
        raise ValueError("segment_scores must have shape [segments, parameter_dim] for one trajectory")
    if (
        not _strict_real(query_budget)
        or query_budget < 0
        or isinstance(branch_count, bool)
        or not isinstance(branch_count, int)
        or branch_count < 1
        or not _strict_real(terminal_cost)
        or terminal_cost < 0
        or not _strict_real(restore_cost)
        or restore_cost < 0
        or not _strict_real(audit_cost)
        or audit_cost < 0
        or (
            planned_branch_cost is not None
            and (not _strict_real(planned_branch_cost) or planned_branch_cost < 0)
        )
    ):
        raise ValueError("invalid science-round budget or branch_count")
    for snapshot_id in ids:
        if boundary_snapshots[snapshot_id].boundary_id != snapshot_id:
            raise ValueError("boundary snapshot identity mismatch")
    for start, end in boundary_intervals:
        if (
            isinstance(start, bool) or isinstance(end, bool)
            or not isinstance(start, int) or not isinstance(end, int)
            or start < 0 or end <= start
        ):
            raise ValueError("boundary_intervals must contain positive integer intervals")

    candidate_keys = {BudgetedQuerySampler._key(candidate) for candidate in candidates}
    expected_internal_keys = {f"{boundary_id}::{endpoint_id}" for boundary_id in ids for endpoint_id in schema.endpoint_ids}
    if not candidate_keys or not candidate_keys.issubset(expected_internal_keys):
        raise ValueError("candidates must be a non-empty subset of internal boundary/endpoint queries")
    # Response propensity is caller-supplied metadata.  Validate it before
    # sampling, reserving budget, or invoking any external callback so a
    # malformed value cannot create a partially committed round.
    if response_probabilities is not None:
        if not isinstance(response_probabilities, Mapping):
            raise ValueError("response_probabilities must be a mapping")
        unknown_response = set(response_probabilities) - candidate_keys
        if unknown_response:
            raise ValueError(f"response probabilities contain unknown query keys: {sorted(unknown_response)}")
        for probability in response_probabilities.values():
            if not _strict_real(probability) or not 0 <= float(probability) <= 1:
                raise ValueError("response probabilities must be finite real numbers in [0,1]")
    decision = BudgetedQuerySampler().sample(
        candidates,
        query_budget,
        mode=query_mode,
        hard_budget=hard_budget,
        seed=seed,
    )
    # Response propensity is a separate missingness mechanism from query
    # selection.  It must be supplied by the harness/audit layer when the
    # caller wants an unbiased response-adjusted claim; the round runner never
    # infers rho=1 merely because an observation happened to arrive.
    response_map = {} if response_probabilities is None else dict(response_probabilities)
    decision = replace(decision, response_probabilities=response_map)

    selected_cost = sum(
        candidate.cost
        for candidate in decision.candidates
        if BudgetedQuerySampler._key(candidate) in decision.selected_ids
    )
    selected_boundaries = {key.split("::", 1)[0] for key in decision.selected_ids}
    if selected_boundaries and planned_branch_cost is None:
        raise ValueError(
            "planned_branch_cost is required when the query design selects a fresh branch"
        )
    planned_restore_total = float(len(selected_boundaries) * restore_cost)
    planned_audit_total = float(len(selected_boundaries) * audit_cost)
    planned_branch_total = (
        0.0
        if planned_branch_cost is None
        else float(len(selected_boundaries) * branch_count * planned_branch_cost)
    )

    # Reserve every known cost before invoking a restore or fresh branch.  A
    # failed reservation rolls back the reservations already made in this
    # round, leaving the caller's ledger unchanged and, crucially, preventing
    # an expensive callback from running without budget coverage.
    reservation_plan = (
        ("terminal", float(terminal_cost), "science_round_terminal"),
        ("query", float(selected_cost), "science_round_queries"),
        ("restore", planned_restore_total, "science_round_restore_plan"),
        ("audit", planned_audit_total, "science_round_audit_plan"),
        ("branch", planned_branch_total, "science_round_branch_plan"),
    )
    reserved: list[tuple[str, float, str]] = []
    baseline_reserved = {
        category: ledger.reserved.get(category, 0.0)
        for category, _, _ in reservation_plan
    }
    try:
        for category, amount, reference in reservation_plan:
            if amount <= 0:
                continue
            ledger.reserve(category, amount, reference=reference)
            reserved.append((category, amount, reference))
    except Exception:
        for category, amount, reference in reversed(reserved):
            ledger.release(category, amount, reference=f"{reference}:rollback")
        raise

    # Terminal and selected-query costs are known before the callbacks and
    # can be consumed immediately.  Restore and branch reservations remain
    # held while their actual callback costs are reconciled below.
    ledger.commit("terminal", float(terminal_cost), planned=float(terminal_cost), reference="science_round_terminal")
    ledger.commit("query", float(selected_cost), planned=float(decision.planned_cost), reference="science_round_queries")

    state_oracle = StateAuditOracle()
    runner = FreshBranchRunner()
    observations: dict[str, EndpointObservation] = {}
    boundary_results: dict[str, tuple[RestoreAudit, tuple[BranchResult, ...]]] = {}
    covariance_by_boundary: dict[str, tuple[tuple[float, ...], ...]] = {}
    restore_evidence_kinds: dict[str, str] = {}
    continuation_modes: dict[str, tuple[str | None, ...]] = {}
    continuation_research_eligible: dict[str, bool] = {}
    restore_spent = 0.0
    audit_spent = 0.0
    branch_spent = 0.0
    try:
        for boundary_id in ids:
            if boundary_id not in selected_boundaries:
                boundary_results[boundary_id] = (RestoreAudit(boundary_id, "failed", boundary_snapshots[boundary_id].digest, None, 0.0, ("not_selected",)), ())
                restore_evidence_kinds[boundary_id] = "unspecified"
                continuation_modes[boundary_id] = ()
                continuation_research_eligible[boundary_id] = False
                continue
            snapshot = boundary_snapshots[boundary_id]
            try:
                restored_response = restore_fn(snapshot)
                if isinstance(restored_response, tuple):
                    restored, observed_restore_cost = restored_response
                else:
                    restored, observed_restore_cost = restored_response, restore_cost
                # The restore response may explicitly grade its own evidence.
                # An echo of the snapshot is ``self_attested`` at best and never
                # becomes research evidence; this flag is a trust declaration,
                # not proof that a durable restore actually occurred.
                evidence_kind = (
                    restored.get("restore_evidence_kind", restored.get("evidence_kind", "unspecified"))
                    if isinstance(restored, Mapping)
                    else "unspecified"
                )
                if evidence_kind not in RESTORE_EVIDENCE_KINDS:
                    raise ValueError("invalid restore evidence kind")
                audit = state_oracle.audit(
                    snapshot,
                    restored,
                    cost=observed_restore_cost,
                    require_captured_state=True,
                    evidence_kind=evidence_kind,
                )
            except Exception as exc:  # restore failures are audit data, never labels
                audit = RestoreAudit(
                    boundary_id,
                    "failed",
                    snapshot.digest,
                    None,
                    max(0.0, float(restore_cost)) if _finite(restore_cost) else 0.0,
                    (f"restore_exception:{type(exc).__name__}",),
                )
            if audit.cost > restore_cost + 1e-9:
                audit = replace(
                    audit,
                    status="failed",
                    reasons=tuple(audit.reasons) + ("restore_cost_exceeds_planned_reservation",),
                )
            # A failed restore prevents every fresh branch for this boundary
            # from running. Release that boundary's branch hold before
            # reconciling an over-cost restore; otherwise an unused branch
            # reservation could make an otherwise affordable failed restore
            # look globally over budget.
            if not audit.passed and planned_branch_cost is not None:
                skipped_branch_hold = float(branch_count * planned_branch_cost)
                if skipped_branch_hold > 0:
                    ledger.release(
                        "branch",
                        skipped_branch_hold,
                        reference=f"science_round_branch_plan:skip:{boundary_id}",
                    )
            # Reconcile the restore before starting any fresh branch.  If its
            # attested cost exceeds the plan and the ledger cannot cover the
            # excess, fail closed without running a branch.
            ledger.commit("restore", audit.cost, planned=float(restore_cost), reference=f"science_round_restore:{boundary_id}")
            ledger.commit("audit", float(audit_cost), planned=float(audit_cost), reference=f"science_round_audit:{boundary_id}")
            restore_spent += audit.cost
            audit_spent += float(audit_cost)
            results = runner.run(
                snapshot,
                audit,
                branch_count=branch_count,
                branch_fn=branch_fn,
                schema=schema,
                policy_version=policy_version,
                seed=seed,
                require_evaluator_provenance=True,
                require_continuation_provenance=require_continuation_provenance,
            )
            reconciled_results: list[BranchResult] = []
            for result in results:
                # A branch callback that reports a cost beyond its planned
                # reservation is an accounting failure, not a partial label.
                ledger.commit(
                    "branch",
                    result.cost,
                    planned=None if planned_branch_cost is None else float(planned_branch_cost),
                    reference=f"science_round_branch:{result.branch_id}",
                )
                branch_spent += result.cost
                if (
                    planned_branch_cost is not None
                    and result.cost > planned_branch_cost + 1e-9
                    and result.status == "completed"
                ):
                    result = replace(
                        result,
                        status="failed",
                        observations=(),
                        failure_class="branch_failure",
                        failure_reason="branch_cost_exceeds_planned_reservation",
                    )
                reconciled_results.append(result)
            results = tuple(reconciled_results)
            boundary_results[boundary_id] = (audit, results)
            restore_evidence_kinds[boundary_id] = audit.evidence_kind
            continuation_modes[boundary_id] = tuple(result.continuation_mode for result in results)
            continuation_research_eligible[boundary_id] = bool(results) and all(
                result.status == "completed" and result.research_eligible for result in results
            )
            values, valid, covariance = mean_observations(results, schema)
            covariance_by_boundary[boundary_id] = tuple(
                tuple(float(value) for value in row) for row in covariance.tolist()
            )
            # HF audits are sampled per boundary *and endpoint*.  A branch may
            # return a valid value for one endpoint while another assay is
            # unavailable or failed.  Keep each valid selected dimension
            # instead of requiring an all-or-nothing vector, otherwise a
            # partial query design silently loses its observed endpoint and
            # becomes a missing propensity correction.
            for endpoint_id, value, endpoint_valid in zip(schema.endpoint_ids, values, valid, strict=True):
                key = f"{boundary_id}::{endpoint_id}"
                if key not in decision.selected_ids or not endpoint_valid or value is None:
                    continue
                endpoint = schema.endpoints[schema.endpoint_index(endpoint_id)]
                fidelity = next(item for item in endpoint.fidelities if item.fidelity_id == endpoint.hf_fidelity_id)
                observations[key] = EndpointObservation(
                    endpoint_id,
                    endpoint.hf_fidelity_id,
                    float(value),
                    evaluator_id=fidelity.evaluator_id,
                    evaluator_version=fidelity.evaluator_version,
                )
    finally:
        # Actual callback costs may be below their plans.  Release only the
        # unused holds; consumed amounts remain in the immutable ledger.
        for category, amount, reference in (
            ("restore", planned_restore_total, "science_round_restore_plan"),
            ("audit", planned_audit_total, "science_round_audit_plan"),
            ("branch", planned_branch_total, "science_round_branch_plan"),
        ):
            held = max(0.0, ledger.reserved.get(category, 0.0) - baseline_reserved.get(category, 0.0))
            if held > 0:
                ledger.release(category, held, reference=f"{reference}:release")

    predictions[-1] = terminal
    corrected_internal, _, correction_audit = residual_correct(
        predictions[1:-1],
        decision,
        observations,
        endpoint_ids=schema.endpoint_ids,
        boundary_ids=ids,
        response_model_validated=response_model_validated,
    )
    corrected = predictions.clone()
    corrected[1:-1] = corrected_internal
    gradient = segment_score_gradient(scores, corrected.unsqueeze(0)).squeeze(0)
    records: list[BoundaryRecord] = []
    for index, boundary_id in enumerate(ids, start=1):
        snapshot = boundary_snapshots[boundary_id]
        start, end = boundary_intervals[index - 1]
        audit, results = boundary_results[boundary_id]
        selected = {endpoint_id: f"{boundary_id}::{endpoint_id}" in decision.selected_ids for endpoint_id in schema.endpoint_ids}
        probabilities = {endpoint_id: float(decision.inclusion_probabilities.get(f"{boundary_id}::{endpoint_id}", 0.0)) for endpoint_id in schema.endpoint_ids}
        aggregate_status = (
            "not_selected"
            if not results
            else "completed"
            if all(result.status == "completed" for result in results)
            else next((result.status for result in results if result.status != "completed"), "failed")
        )
        aggregate_reason = next((result.failure_reason for result in results if result.failure_reason), None)
        statuses: dict[str, str] = {}
        reasons: dict[str, str | None] = {}
        for endpoint_id in schema.endpoint_ids:
            key = f"{boundary_id}::{endpoint_id}"
            if not selected[endpoint_id]:
                statuses[endpoint_id] = "not_selected"
                reasons[endpoint_id] = None
                continue
            if key in observations:
                statuses[endpoint_id] = "completed"
                reasons[endpoint_id] = None
                continue
            # Preserve a typed endpoint failure when the branch completed but
            # the evaluator returned a missing/failed observation.  A restore
            # or rollout failure is represented by the BranchResult status.
            endpoint_failure = next(
                (
                    obs
                    for result in results
                    for obs in result.observations
                    if obs.endpoint_id == endpoint_id and obs.status != "observed"
                ),
                None,
            )
            statuses[endpoint_id] = (
                endpoint_failure.status
                if endpoint_failure is not None
                else aggregate_status
            )
            reasons[endpoint_id] = (
                endpoint_failure.failure_class
                if endpoint_failure is not None
                else aggregate_reason
            )
        scores_for_boundary = {endpoint_id: (None if f"{boundary_id}::{endpoint_id}" not in observations else observations[f"{boundary_id}::{endpoint_id}"].value) for endpoint_id in schema.endpoint_ids}
        records.append(BoundaryRecord(
            boundary_id=boundary_id, trajectory_id=snapshot.trajectory_id, segment_id=index - 1,
            start=start, end=end, duration=end - start, boundary_type="science_round", snapshot=snapshot,
            feature_vector=tuple(float(value) for value in boundary_features[index - 1]),
            prediction=tuple(float(value) for value in predictions[index]),
            oof_residual=None if oof_residuals is None else tuple(oof_residuals[index - 1]),
            query_selected=selected, query_inclusion_probability=probabilities,
            branch_count={endpoint_id: branch_count if selected[endpoint_id] else 0 for endpoint_id in schema.endpoint_ids},
            gold_or_proxy={endpoint_id: ("gold" if scores_for_boundary[endpoint_id] is not None else "proxy") for endpoint_id in schema.endpoint_ids},
            gold_score=scores_for_boundary, query_cost=sum(candidate.cost for candidate in decision.candidates if BudgetedQuerySampler._key(candidate).startswith(f"{boundary_id}::") and BudgetedQuerySampler._key(candidate) in decision.selected_ids),
            restore_cost=audit.cost, branch_status=statuses, failure_reason=reasons,
            budget_remaining_after=ledger.remaining,
            response_probability={
                endpoint_id: decision.response_probabilities.get(f"{boundary_id}::{endpoint_id}")
                for endpoint_id in schema.endpoint_ids
            },
            valid_observation={endpoint_id: f"{boundary_id}::{endpoint_id}" in observations for endpoint_id in schema.endpoint_ids},
            boundary_state_ref=snapshot.digest,
            evidence_state={
                "correction_audit": correction_audit,
                "restore_evidence_kind": audit.evidence_kind,
                "restore_research_eligible": audit.research_eligible,
                "continuation_modes": continuation_modes.get(boundary_id, ()),
                "continuation_research_eligible": continuation_research_eligible.get(boundary_id, False),
            },
            remaining_execution_budget=snapshot.execution_budget_remaining,
            branch_policy_version={endpoint_id: policy_version for endpoint_id in schema.endpoint_ids if selected[endpoint_id]},
            branch_rng_id={endpoint_id: ",".join(result.rng_id for result in results) for endpoint_id in schema.endpoint_ids if selected[endpoint_id]},
        ))

    verification = PolySegSciVerifier().verify(
        schema=schema,
        ledger=ledger,
        decisions=(decision,),
        observations=observations,
        feature_map=feature_map,
        policy_version=policy_version,
        terminal_evaluator_versions=terminal_evaluator_versions,
    )
    return ScienceRoundResult(
        corrected_boundary_values=corrected,
        segment_gradient=gradient,
        decision=decision,
        observations=observations,
        boundary_records=tuple(records),
        ledger=ledger.to_dict(),
        verification=verification,
        audit={
            "correction": correction_audit,
            "restore_spent": restore_spent,
            "audit_spent": audit_spent,
            "branch_spent": branch_spent,
            "covariance_by_boundary": covariance_by_boundary,
            "restore_evidence_kinds": restore_evidence_kinds,
            "continuation_modes": continuation_modes,
            "continuation_research_eligible": continuation_research_eligible,
            "require_continuation_provenance": bool(require_continuation_provenance),
        },
        covariance_by_boundary=covariance_by_boundary,
    )


def mean_observations(results: Sequence[BranchResult], schema: EndpointSchema) -> tuple[tuple[float | None, ...], tuple[bool, ...], torch.Tensor]:
    values: list[float | None] = []
    valid: list[bool] = []
    covariance = torch.zeros(schema.dimension, schema.dimension, dtype=torch.float64)
    for endpoint_id in schema.endpoint_ids:
        # ``branch_count`` is part of the estimator contract.  Averaging only
        # successful branches would silently change n_b after a timeout or
        # crash and turn a failed fresh-continuation audit into a scientific
        # label.  Every branch must complete and report this endpoint before
        # its mean is valid; the caller can still retain valid dimensions from
        # a fully completed vector whose other endpoint assay failed.
        samples = []
        endpoint_complete = bool(results)
        for result in results:
            if result.status != "completed":
                endpoint_complete = False
                break
            matches = [
                obs.value
                for obs in result.observations
                if obs.endpoint_id == endpoint_id and obs.status == "observed" and obs.value is not None
            ]
            if len(matches) != 1:
                endpoint_complete = False
                break
            samples.append(float(matches[0]))
        if not endpoint_complete:
            values.append(None)
            valid.append(False)
            continue
        values.append(float(statistics.fmean(samples)))
        valid.append(True)
    if all(valid):
        rows = []
        for result in results:
            row = []
            for endpoint_id in schema.endpoint_ids:
                match = next((obs.value for obs in result.observations if obs.endpoint_id == endpoint_id and obs.status == "observed"), None)
                if match is None:
                    break
                row.append(float(match))
            if len(row) == schema.dimension:
                rows.append(row)
        if len(rows) > 1:
            covariance = torch.as_tensor(rows, dtype=torch.float64).T.cov(correction=1)
            # ``Tensor.cov`` returns a scalar for a one-dimensional endpoint
            # vector.  Keep the covariance contract [R, R] even when R=1 so
            # downstream vector correction and serialized audit records do
            # not change shape with endpoint dimensionality.
            if covariance.ndim == 0:
                covariance = covariance.reshape(1, 1)
    return tuple(values), tuple(valid), covariance


def residual_correct(
    predictions: torch.Tensor,
    decisions: QueryDecision,
    observations: Mapping[str, EndpointObservation],
    *,
    endpoint_ids: Sequence[str],
    boundary_ids: Sequence[str] | None = None,
    terminal_values: torch.Tensor | None = None,
    terminal_mask: torch.Tensor | None = None,
    selection_probabilities: Mapping[str, float] | None = None,
    response_probabilities: Mapping[str, float] | None = None,
    valid_observation_mask: Mapping[str, bool] | None = None,
    response_model_validated: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Apply the untrimmed inverse-propensity residual correction.

    ``predictions`` is [N,R].  ``observations`` is keyed by
    ``boundary_id::endpoint_id``.  Failed or missing observations leave the
    predictor in place and mark the corresponding audit mask false; no value
    is invented for a failed oracle.
    """

    _strict_bool(response_model_validated, "response_model_validated")
    pred = torch.as_tensor(predictions, dtype=torch.float64)
    if pred.ndim != 2 or pred.shape[1] != len(endpoint_ids):
        raise ValueError("predictions must have shape [N,R]")
    if not torch.isfinite(pred).all():
        raise ValueError("predictions must be finite")
    endpoint_ids = tuple(str(endpoint_id) for endpoint_id in endpoint_ids)
    if len(set(endpoint_ids)) != len(endpoint_ids) or any(not endpoint_id for endpoint_id in endpoint_ids):
        raise ValueError("endpoint_ids must be unique, non-empty identifiers")
    if boundary_ids is not None and len(boundary_ids) != pred.shape[0]:
        raise ValueError("boundary_ids must have one entry per prediction row")
    corrected = pred.clone()
    valid = torch.zeros_like(pred, dtype=torch.bool)
    selected = set(decisions.selected_ids)
    selection_map = decisions.inclusion_probabilities if selection_probabilities is None else selection_probabilities
    candidate_keys = {BudgetedQuerySampler._key(candidate) for candidate in decisions.candidates}
    if len(selected) != len(decisions.selected_ids) or not selected.issubset(candidate_keys):
        raise ValueError("selected queries must be unique members of the candidate set")
    unknown_observations = set(observations) - candidate_keys
    if unknown_observations:
        raise ValueError(f"observations contain unknown query keys: {sorted(unknown_observations)}")
    unknown_selection = set(selection_map) - candidate_keys
    if unknown_selection:
        raise ValueError(f"selection probabilities contain unknown query keys: {sorted(unknown_selection)}")
    for key, probability in selection_map.items():
        if not _finite(probability) or float(probability) < 0 or float(probability) > 1:
            raise ValueError("selection probabilities must be finite and in [0,1]")
    for key in selected:
        if float(selection_map.get(key, 0.0)) <= 0:
            raise ValueError("selected queries require positive selection probabilities")
    response_map = {
        **dict(decisions.response_probabilities),
        **({} if response_probabilities is None else dict(response_probabilities)),
    }
    if any(not _strict_real(value) for value in selection_map.values()):
        raise ValueError("selection probabilities must be finite real numbers")
    if any(not _strict_real(value) for value in response_map.values()):
        raise ValueError("response probabilities must be finite real numbers")
    unknown_response = set(response_map) - candidate_keys
    if unknown_response:
        raise ValueError(f"response probabilities contain unknown query keys: {sorted(unknown_response)}")
    valid_map = {} if valid_observation_mask is None else dict(valid_observation_mask)
    if any(not isinstance(value, bool) for value in valid_map.values()):
        raise ValueError("valid observation mask values must be boolean")
    unknown_valid = set(valid_map) - candidate_keys
    if unknown_valid:
        raise ValueError(f"valid observation mask contains unknown query keys: {sorted(unknown_valid)}")
    audit: dict[str, Any] = {
        "selected": [],
        "failed": [],
        "zero_propensity": [],
        "response_probability_missing": [],
        "response_probability_invalid": [],
        "selection_probabilities": {},
        "response_probabilities": {},
        "valid_observation_mask": {},
        "unbiased_under_response_model": True,
        "response_model_assumption_required": False,
        "response_model_validated": bool(response_model_validated),
    }
    for row in range(pred.shape[0]):
        row_id = str(row) if boundary_ids is None else str(boundary_ids[row])
        for col, endpoint_id in enumerate(endpoint_ids):
            key = f"{row_id}::{endpoint_id}"
            # Candidates with p=0 are deliberately auditable.  They may be
            # left at the frozen prediction for an engineering fallback, but
            # that position cannot be included in an overall unbiasedness
            # claim.  A non-selected candidate with p>0 is the ordinary
            # inverse-propensity I=0 case and remains valid.
            if key not in candidate_keys:
                # Some designs query only a registered subset of endpoint
                # dimensions.  Leave those dimensions outside this
                # correction audit rather than treating an intentionally
                # omitted candidate as a failed response.
                continue
            p = float(selection_map.get(key, 0.0))
            if p <= 0 or not _finite(p):
                audit["zero_propensity"].append(key)
                audit["selection_probabilities"][key] = p if _finite(p) else None
                audit["valid_observation_mask"][key] = False
                audit["unbiased_under_response_model"] = False
                continue
            if key not in selected:
                continue
            if p > 1:
                audit["response_probability_invalid"].append(key)
                audit["unbiased_under_response_model"] = False
                continue
            # Keep the selection event auditable even when no response is
            # returned.  A failed query is still a selected query and must
            # not disappear from the propensity ledger.
            audit["selection_probabilities"][key] = p
            obs = observations.get(key)
            if obs is not None and obs.endpoint_id != endpoint_id:
                audit["failed"].append({"key": key, "status": "endpoint_mismatch"})
                audit["valid_observation_mask"][key] = False
                audit["unbiased_under_response_model"] = False
                continue
            observed = obs is not None and obs.status == "observed" and obs.value is not None and valid_map.get(key, True)
            if key in response_map:
                rho = float(response_map[key])
                if not (0 < rho <= 1) or not _finite(rho):
                    audit["response_probability_invalid"].append(key)
                    audit["unbiased_under_response_model"] = False
                    continue
                # A numeric response propensity is useful for the realized
                # correction, but it is not by itself evidence that the
                # response mechanism is ignorable.  Keep the estimate for
                # continuity while requiring an explicit audit/model
                # attestation before calling the result unbiased.
                if not response_model_validated:
                    audit["response_model_assumption_required"] = True
                    audit["unbiased_under_response_model"] = False
            elif observed:
                # We can still form the realized correction with rho=1, but
                # this is an assumption about the response mechanism, not an
                # observed probability.  Keep the value for engineering
                # continuity while refusing to label it unbiased until the
                # response model (or an explicitly supplied rho) is audited.
                rho = 1.0
                audit["response_model_assumption_required"] = True
                audit["unbiased_under_response_model"] = False
                audit["response_probability_missing"].append(key)
            else:
                audit["response_model_assumption_required"] = True
                audit["unbiased_under_response_model"] = False
                audit["response_probability_missing"].append(key)
                audit["failed"].append({"key": key, "status": None if obs is None else obs.status})
                audit["valid_observation_mask"][key] = False
                audit["selected"].append(
                    {
                        "key": key,
                        "p": p,
                        "selection_probability": p,
                        "response_probability": None,
                        "joint_probability": None,
                        "observed": False,
                        "valid_observation": False,
                    }
                )
                continue
            effective_p = p * rho
            if effective_p <= 0 or effective_p > 1 or not _finite(effective_p):
                audit["response_probability_invalid"].append(key)
                audit["unbiased_under_response_model"] = False
                continue
            audit["response_probabilities"][key] = rho
            audit["valid_observation_mask"][key] = bool(observed)
            if not observed:
                audit["response_model_assumption_required"] = True
                if not response_model_validated:
                    audit["unbiased_under_response_model"] = False
                audit["failed"].append({"key": key, "status": None if obs is None else obs.status})
                audit["selected"].append(
                    {
                        "key": key,
                        "p": p,
                        "selection_probability": p,
                        "response_probability": rho,
                        "joint_probability": effective_p,
                        "observed": False,
                        "valid_observation": False,
                    }
                )
                continue
            corrected[row, col] = pred[row, col] + (float(obs.value) - pred[row, col]) / effective_p
            valid[row, col] = True
            audit["selected"].append(
                {
                    "key": key,
                    "p": p,
                    "selection_probability": p,
                    "response_probability": rho,
                    "joint_probability": effective_p,
                    "value": float(obs.value),
                    "observed": True,
                    "valid_observation": True,
                }
            )
    if terminal_values is not None:
        terminal = torch.as_tensor(terminal_values, dtype=pred.dtype)
        if terminal.shape != pred.shape:
            raise ValueError("terminal_values must match predictions")
        if not torch.isfinite(terminal).all():
            raise ValueError("terminal_values must be finite")
        mask = torch.ones_like(pred, dtype=torch.bool) if terminal_mask is None else torch.as_tensor(terminal_mask, dtype=torch.bool)
        if mask.shape != pred.shape:
            raise ValueError("terminal_mask must match predictions")
        corrected = torch.where(mask, terminal, corrected)
        valid |= mask
    return corrected, valid, audit


def residual_correction_covariance(
    prediction_error: torch.Tensor,
    branch_covariance: torch.Tensor,
    inclusion_probability: float | Sequence[float],
    branch_count: int,
    *,
    pair_inclusion_probabilities: torch.Tensor | Sequence[Sequence[float]] | None = None,
    branch_counts: Sequence[int] | None = None,
    paired_branch_counts: torch.Tensor | Sequence[Sequence[int]] | None = None,
) -> torch.Tensor:
    """Covariance of vector residual correction under endpoint-specific inclusion.

    The scalar form is retained for compatibility.  When endpoint probabilities
    and a joint ``p_rs`` matrix are supplied, diagonal terms use ``p_r`` and
    cross-endpoint terms use the actual pair probability; a product of
    marginals is never substituted for a constrained design.

    Branches are independent replicates. ``paired_branch_counts[r,s]`` is
    the number of shared replicates for the two endpoint means, so their
    noise covariance is ``Sigma[r,s] * n_rs / (n_r * n_s)``. The default
    shared-vector design uses the same replicates for every endpoint;
    unequal counts require explicit overlap counts. Correlation across
    different replicates is outside this helper's model.
    """

    error = torch.as_tensor(prediction_error, dtype=torch.float64).reshape(-1)
    covariance = torch.as_tensor(branch_covariance, dtype=torch.float64)
    dimension = error.numel()
    if covariance.shape != (dimension, dimension):
        raise ValueError("branch_covariance must be square with one row per endpoint")
    if not torch.isfinite(error).all() or not torch.isfinite(covariance).all():
        raise ValueError("covariance inputs must be finite")
    if isinstance(branch_count, bool) or not isinstance(branch_count, int) or branch_count < 1:
        raise ValueError("branch_count must be a positive integer")
    if _strict_real(inclusion_probability):
        if not 0 < float(inclusion_probability) <= 1:
            raise ValueError("inclusion probability must be in (0,1] and branch_count positive")
        if branch_counts is None and paired_branch_counts is None:
            return ((1 - float(inclusion_probability)) / float(inclusion_probability)) * torch.outer(error, error) + covariance / (float(inclusion_probability) * branch_count)
        # A shared query indicator has p_rs=p, including off the diagonal.
        probability = float(inclusion_probability)
        inclusion_probability = [probability] * dimension
        pair_inclusion_probabilities = torch.full((dimension, dimension), probability, dtype=torch.float64)
    if isinstance(inclusion_probability, (str, bytes, bool)):
        raise ValueError("inclusion probability must be a finite real or a numeric vector")
    probabilities = torch.as_tensor(inclusion_probability, dtype=torch.float64).reshape(-1)
    if probabilities.numel() != dimension or not torch.isfinite(probabilities).all() or torch.any((probabilities <= 0) | (probabilities > 1)):
        raise ValueError("endpoint inclusion probabilities must be finite and in (0,1]")
    if pair_inclusion_probabilities is None:
        raise ValueError("pair inclusion probabilities are required for vector covariance")
    pairs = torch.as_tensor(pair_inclusion_probabilities, dtype=torch.float64)
    if pairs.shape != (dimension, dimension) or not torch.isfinite(pairs).all():
        raise ValueError("pair inclusion probabilities must be a finite square matrix")
    if torch.any(pairs < 0) or torch.any(pairs > 1):
        raise ValueError("pair inclusion probabilities must lie in [0,1]")
    if not torch.allclose(pairs, pairs.T, atol=1e-8, rtol=0):
        raise ValueError("pair inclusion probabilities must be symmetric")
    if not torch.allclose(torch.diag(pairs), probabilities, atol=1e-8, rtol=0):
        raise ValueError("pair diagonal must equal endpoint inclusion probabilities")
    off_diagonal = ~torch.eye(dimension, dtype=torch.bool)
    marginal_bound = torch.minimum(probabilities[:, None], probabilities[None, :])
    if torch.any(pairs[off_diagonal] - marginal_bound[off_diagonal] > 1e-8):
        raise ValueError("joint inclusion probabilities must not exceed either endpoint marginal")
    lower_bound = torch.maximum(
        torch.zeros_like(marginal_bound), probabilities[:, None] + probabilities[None, :] - 1.0
    )
    if torch.any(lower_bound[off_diagonal] - pairs[off_diagonal] > 1e-8):
        raise ValueError("joint inclusion probabilities violate the Frechet lower bound")
    if branch_counts is None:
        counts = torch.full((dimension,), branch_count, dtype=torch.float64)
    else:
        if isinstance(branch_counts, torch.Tensor):
            if branch_counts.ndim != 1 or branch_counts.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
                raise ValueError("branch_counts must contain positive integers for every endpoint")
            raw_counts = tuple(branch_counts.tolist())
        else:
            raw_counts = tuple(branch_counts)
        if len(raw_counts) != dimension or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in raw_counts):
            raise ValueError("branch_counts must contain positive integers for every endpoint")
        counts = torch.as_tensor(raw_counts, dtype=torch.float64).reshape(-1)
    if counts.numel() != dimension or not torch.isfinite(counts).all() or torch.any(counts < 1):
        raise ValueError("branch_counts must be positive for every endpoint")
    if paired_branch_counts is None:
        if not torch.all(counts == counts[0]):
            raise ValueError("unequal branch_counts require explicit paired_branch_counts")
        overlaps = torch.full_like(covariance, counts[0].item())
    else:
        raw_overlaps = torch.as_tensor(paired_branch_counts)
        if raw_overlaps.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
            raise ValueError("paired_branch_counts must be an integer matrix")
        if not isinstance(paired_branch_counts, torch.Tensor) and any(
            isinstance(value, bool) for row in paired_branch_counts for value in row
        ):
            raise ValueError("paired_branch_counts must be an integer matrix")
        overlaps = raw_overlaps.to(dtype=torch.float64)
        if (
            overlaps.shape != covariance.shape
            or torch.any(overlaps < 0)
            or not torch.equal(overlaps, overlaps.T)
            or not torch.equal(overlaps.diag(), counts)
            or torch.any(overlaps > torch.minimum(counts[:, None], counts[None, :]))
        ):
            raise ValueError("paired_branch_counts must be symmetric with matching diagonal and feasible overlaps")
    result = torch.zeros_like(covariance)
    for left in range(dimension):
        p_left = probabilities[left]
        result[left, left] = ((1 - p_left) / p_left) * error[left] ** 2 + covariance[left, left] / (p_left * counts[left])
        for right in range(left):
            p_right = probabilities[right]
            p_pair = pairs[left, right]
            joint_factor = p_pair / (p_left * p_right)
            mean_noise_covariance = covariance[left, right] * overlaps[left, right] / (counts[left] * counts[right])
            value = (joint_factor - 1) * error[left] * error[right] + joint_factor * mean_noise_covariance
            result[left, right] = value
            result[right, left] = value
    return result


class MultiFidelityCalibrator:
    """Paired LF/MF-to-HF diagnostics without assuming calibration for free."""

    def __init__(self, *, ridge_lambda: float = 1e-6):
        if not _strict_real(ridge_lambda) or ridge_lambda <= 0:
            raise ValueError("ridge_lambda must be positive")
        self.ridge_lambda = float(ridge_lambda)

    def diagnostics(
        self,
        low_fidelity: Sequence[float],
        high_fidelity: Sequence[float],
        *,
        valid_mask: Sequence[bool] | None = None,
        interval: float = 1.96,
        heldout_low_fidelity: Sequence[float] | None = None,
        heldout_high_fidelity: Sequence[float] | None = None,
        heldout_valid_mask: Sequence[bool] | None = None,
        domain_labels: Sequence[str] | None = None,
        heldout_domain_labels: Sequence[str] | None = None,
    ) -> dict[str, float | int | None]:
        low = torch.as_tensor(low_fidelity, dtype=torch.float64).reshape(-1)
        high = torch.as_tensor(high_fidelity, dtype=torch.float64).reshape(-1)
        if low.shape != high.shape:
            raise ValueError("paired fidelity arrays must have equal length")
        if valid_mask is None:
            mask = torch.ones_like(low, dtype=torch.bool)
        elif isinstance(valid_mask, torch.Tensor):
            if valid_mask.dtype != torch.bool:
                raise ValueError("valid_mask must contain booleans")
            mask = valid_mask.reshape(-1)
        else:
            raw_mask = tuple(valid_mask)
            if any(not isinstance(value, bool) for value in raw_mask):
                raise ValueError("valid_mask must contain booleans")
            mask = torch.as_tensor(raw_mask, dtype=torch.bool)
        if mask.shape != low.shape or int(mask.sum()) < 2:
            raise ValueError("calibration requires at least two valid paired observations")
        if not _strict_real(interval) or not 0 < interval:
            raise ValueError("interval must be finite and positive")
        x, y = low[mask], high[mask]
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError("calibration observations must be finite")
        design = torch.stack([torch.ones_like(x), x], dim=1)
        theta = torch.linalg.solve(design.T @ design + self.ridge_lambda * torch.eye(2, dtype=design.dtype), design.T @ y)
        pred = design @ theta
        residual = y - pred
        variance = residual.var(unbiased=True) if residual.numel() > 1 else torch.zeros(())
        report: dict[str, float | int | None] = {
            "n": int(x.numel()),
            "bias": float((x - y).mean()),
            "mae": float((x - y).abs().mean()),
            "rmse": float((x - y).square().mean().sqrt()),
            "calibrated_rmse": float(residual.square().mean().sqrt()),
            "calibration_intercept": float(theta[0]),
            "calibration_slope": float(theta[1]),
            "interval_coverage": float(((y >= pred - interval * variance.sqrt()) & (y <= pred + interval * variance.sqrt())).to(torch.float64).mean()),
            "heldout_n": None,
            "heldout_mae": None,
            "heldout_rmse": None,
            "extrapolation_fraction": None,
            "domain_shift_available": False,
            "domain_shift_mean_delta": None,
        }
        if heldout_low_fidelity is not None or heldout_high_fidelity is not None:
            if heldout_low_fidelity is None or heldout_high_fidelity is None:
                raise ValueError("held-out calibration requires both fidelity arrays")
            heldout_x = torch.as_tensor(heldout_low_fidelity, dtype=torch.float64).reshape(-1)
            heldout_y = torch.as_tensor(heldout_high_fidelity, dtype=torch.float64).reshape(-1)
            if heldout_valid_mask is None:
                heldout_mask = torch.ones_like(heldout_x, dtype=torch.bool)
            elif isinstance(heldout_valid_mask, torch.Tensor):
                if heldout_valid_mask.dtype != torch.bool:
                    raise ValueError("heldout_valid_mask must contain booleans")
                heldout_mask = heldout_valid_mask.reshape(-1)
            else:
                raw_heldout_mask = tuple(heldout_valid_mask)
                if any(not isinstance(value, bool) for value in raw_heldout_mask):
                    raise ValueError("heldout_valid_mask must contain booleans")
                heldout_mask = torch.as_tensor(raw_heldout_mask, dtype=torch.bool)
            if heldout_x.shape != heldout_y.shape or heldout_mask.shape != heldout_x.shape or int(heldout_mask.sum()) < 1:
                raise ValueError("held-out calibration arrays/mask must align and contain an observation")
            if not torch.isfinite(heldout_x[heldout_mask]).all() or not torch.isfinite(heldout_y[heldout_mask]).all():
                raise ValueError("held-out calibration observations must be finite")
            heldout_pred = theta[0] + theta[1] * heldout_x[heldout_mask]
            heldout_error = heldout_pred - heldout_y[heldout_mask]
            report["heldout_n"] = int(heldout_mask.sum())
            report["heldout_mae"] = float(heldout_error.abs().mean())
            report["heldout_rmse"] = float(heldout_error.square().mean().sqrt())
            report["extrapolation_fraction"] = float(((heldout_x[heldout_mask] < x.min()) | (heldout_x[heldout_mask] > x.max())).to(torch.float64).mean())
        if domain_labels is not None or heldout_domain_labels is not None:
            heldout_length = int(torch.as_tensor(heldout_low_fidelity).reshape(-1).numel()) if heldout_low_fidelity is not None else 0
            if domain_labels is None or heldout_domain_labels is None or len(domain_labels) != len(low) or len(heldout_domain_labels) != heldout_length:
                raise ValueError("domain labels must align with train and held-out calibration rows")
            train_domains = set(str(label) for label in domain_labels)
            heldout_domains = set(str(label) for label in heldout_domain_labels)
            report["domain_shift_available"] = True
            report["domain_shift_mean_delta"] = float(len(heldout_domains - train_domains)) / max(1, len(heldout_domains))
        return report

    def diagnostics_vector(
        self,
        low_fidelity: Sequence[Sequence[float]] | torch.Tensor,
        high_fidelity: Sequence[Sequence[float]] | torch.Tensor,
        *,
        endpoint_ids: Sequence[str] | None = None,
        valid_mask: Sequence[Sequence[bool]] | torch.Tensor | None = None,
        interval: float = 1.96,
    ) -> dict[str, Any]:
        """Report calibration for every endpoint and their residual dependence.

        ``low_fidelity`` and ``high_fidelity`` are paired matrices with shape
        ``[N, R]``.  Each endpoint is fitted independently using the same
        ridge rule as :meth:`diagnostics`; residual covariance and correlation
        are then computed pairwise on rows observed for both endpoints.  This
        keeps endpoint-specific missingness explicit while preserving the
        cross-endpoint dependence needed for vector gradient uncertainty.
        """

        low = torch.as_tensor(low_fidelity, dtype=torch.float64)
        high = torch.as_tensor(high_fidelity, dtype=torch.float64)
        if low.ndim == 1:
            low = low.unsqueeze(1)
        if high.ndim == 1:
            high = high.unsqueeze(1)
        if low.ndim != 2 or high.shape != low.shape:
            raise ValueError("vector calibration arrays must both have shape [N, R]")
        n_rows, dimension = low.shape
        if endpoint_ids is None:
            ids = [str(index) for index in range(dimension)]
        else:
            ids = [str(endpoint_id) for endpoint_id in endpoint_ids]
            if len(ids) != dimension or len(set(ids)) != dimension:
                raise ValueError("endpoint_ids must be unique and match endpoint dimension")
        if valid_mask is None:
            mask = torch.ones_like(low, dtype=torch.bool)
        else:
            if isinstance(valid_mask, torch.Tensor):
                if valid_mask.dtype != torch.bool:
                    raise ValueError("valid_mask must contain booleans")
                mask = valid_mask
            else:
                raw_mask = tuple(valid_mask)
                if any(not isinstance(value, bool) for row in raw_mask for value in row):
                    raise ValueError("valid_mask must contain booleans")
                mask = torch.as_tensor(raw_mask, dtype=torch.bool)
            if mask.shape != low.shape:
                raise ValueError("valid_mask must match vector calibration arrays")
        if torch.any(mask & (~torch.isfinite(low) | ~torch.isfinite(high))):
            raise ValueError("observed calibration values must be finite")
        if not _strict_real(interval) or not 0 < interval:
            raise ValueError("interval must be finite and positive")
        endpoint_reports: dict[str, dict[str, float | int]] = {}
        residuals = torch.full_like(high, float("nan"))
        for index, endpoint_id in enumerate(ids):
            report = self.diagnostics(
                low[:, index],
                high[:, index],
                valid_mask=mask[:, index],
                interval=interval,
            )
            endpoint_reports[endpoint_id] = report
            observed = mask[:, index]
            prediction = report["calibration_intercept"] + report["calibration_slope"] * low[:, index]
            residuals[observed, index] = high[observed, index] - prediction[observed]

        covariance: list[list[float | None]] = []
        correlation: list[list[float | None]] = []
        pair_counts: list[list[int]] = []
        for left in range(dimension):
            covariance_row: list[float | None] = []
            correlation_row: list[float | None] = []
            count_row: list[int] = []
            for right in range(dimension):
                paired = mask[:, left] & mask[:, right]
                count = int(paired.sum())
                count_row.append(count)
                if count < 2:
                    covariance_row.append(None)
                    correlation_row.append(None)
                    continue
                left_residual = residuals[paired, left]
                right_residual = residuals[paired, right]
                centered_left = left_residual - left_residual.mean()
                centered_right = right_residual - right_residual.mean()
                cov = float((centered_left * centered_right).sum() / (count - 1))
                covariance_row.append(cov)
                left_var = float((centered_left.square().sum() / (count - 1)))
                right_var = float((centered_right.square().sum() / (count - 1)))
                if left_var <= 0.0 or right_var <= 0.0:
                    correlation_row.append(None)
                else:
                    correlation_row.append(cov / math.sqrt(left_var * right_var))
            covariance.append(covariance_row)
            correlation.append(correlation_row)
            pair_counts.append(count_row)

        return {
            "endpoint_ids": ids,
            "endpoint_dimension": dimension,
            "row_count": n_rows,
            "endpoints": endpoint_reports,
            "pair_counts": pair_counts,
            "residual_covariance": covariance,
            "residual_correlation": correlation,
        }


def trajectory_bootstrap_ci(
    values: Sequence[float],
    groups: Sequence[str],
    *,
    confidence: float = 0.95,
    repetitions: int = 2000,
    seed: int = 0,
) -> dict[str, float | int]:
    """Task/trajectory-level bootstrap; never resamples adjacent segments."""

    if len(values) != len(groups) or not values:
        raise ValueError("values and groups must have equal non-zero length")
    if not (0 < confidence < 1) or repetitions < 1:
        raise ValueError("invalid bootstrap confidence or repetitions")
    grouped: dict[str, list[float]] = {}
    for value, group in zip(values, groups, strict=True):
        if not _finite(value):
            raise ValueError("bootstrap values must be finite")
        grouped.setdefault(str(group), []).append(float(value))
    summaries = [statistics.fmean(v) for v in grouped.values()]
    rng = random.Random(seed)
    draws = [statistics.fmean(rng.choice(summaries) for _ in summaries) for _ in range(repetitions)]
    draws.sort()
    lower_index = max(0, min(repetitions - 1, int((1 - confidence) * repetitions / 2)))
    upper_index = max(0, min(repetitions - 1, int((1 - (1 - confidence) / 2) * repetitions) - 1))
    return {"estimate": float(statistics.fmean(summaries)), "lower": float(draws[lower_index]), "upper": float(draws[upper_index]), "groups": len(summaries), "repetitions": repetitions}


def segment_score_gradient(
    segment_scores: torch.Tensor,
    boundary_values: torch.Tensor,
    *,
    segment_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return [batch, endpoint, parameter] segment score gradients.

    Values are detached by construction.  The terminal value belongs at
    column ``m`` and the initial baseline at column zero; no terminal label is
    copied into intermediate segments.
    """

    scores = torch.as_tensor(segment_scores)
    values = torch.as_tensor(boundary_values, dtype=scores.dtype, device=scores.device)
    if not torch.isfinite(scores).all() or not torch.isfinite(values).all():
        raise ValueError("segment scores and boundary values must be finite")
    if scores.ndim == 2:
        scores = scores.unsqueeze(0)
    if values.ndim == 2:
        values = values.unsqueeze(0)
    if scores.ndim != 3 or values.ndim != 3 or values.shape[0] != scores.shape[0] or values.shape[1] != scores.shape[1] + 1:
        raise ValueError("scores must be [B,M,D] and boundary_values [B,M+1,R]")
    delta = (values[:, 1:] - values[:, :-1]).detach()
    if segment_mask is not None:
        mask = torch.as_tensor(segment_mask, dtype=torch.bool, device=scores.device)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if mask.shape != scores.shape[:2]:
            raise ValueError("segment_mask must match [B,M]")
        scores = scores * mask.unsqueeze(-1).to(scores.dtype)
    return torch.einsum("bmd,bmr->brd", scores, delta)


def parameter_score_gradients(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    response_start: int,
    segment_ranges: Sequence[tuple[int, int]],
    attention_mask: torch.Tensor | None = None,
    action_mask: torch.Tensor | None = None,
    parameter_names: Sequence[str] | None = None,
) -> torch.Tensor:
    """Compute token-policy score vectors for causal response segments.

    ``segment_ranges`` are half-open offsets in the generated response.  The
    returned row for a segment is

    ``grad_theta sum_{t in segment and action_mask[t]} log pi_theta(y_t | y_<t)``.

    This is a real parameter score in the selected parameter subspace; it is
    not an average token log-probability.  The model is evaluated in ``eval``
    mode for deterministic dropout semantics and its original training mode is
    restored before returning.  Teacher-forced input tokens are used only to
    evaluate the sampled trajectory, so no generated token is replaced by a
    greedy or sampled alternative during the score calculation.
    """

    try:
        ids = torch.as_tensor(input_ids)
    except (TypeError, RuntimeError) as exc:
        raise ValueError("input_ids must be an integer tensor") from exc
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 2:
        raise ValueError("input_ids must have shape [1, sequence_length>=2]")
    if ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("input_ids must be an integer tensor")
    if isinstance(response_start, bool) or not isinstance(response_start, int):
        raise ValueError("response_start must be an integer")
    response_width = ids.shape[1] - response_start
    if response_start < 1 or response_width < 1:
        raise ValueError("response_start must leave at least one response token and one context token")

    def _strict_mask(value: torch.Tensor | Sequence[Any], *, name: str, shape: tuple[int, ...]) -> torch.Tensor:
        try:
            raw = torch.as_tensor(value, device=ids.device)
        except (TypeError, RuntimeError) as exc:
            raise ValueError(f"{name} must contain booleans or binary numeric values") from exc
        if raw.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if raw.dtype == torch.bool:
            return raw
        if raw.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.float16, torch.float32, torch.float64, torch.bfloat16):
            raise ValueError(f"{name} must contain booleans or binary numeric values")
        if not torch.isfinite(raw).all() or not torch.all((raw == 0) | (raw == 1)):
            raise ValueError(f"{name} must contain only 0/1 or boolean values")
        return raw.to(dtype=torch.bool)

    if attention_mask is None:
        attn = torch.ones_like(ids, dtype=torch.bool)
    else:
        attn = _strict_mask(attention_mask, name="attention_mask", shape=tuple(ids.shape))
    if action_mask is None:
        actions = torch.ones(response_width, dtype=torch.bool, device=ids.device)
    else:
        actions = _strict_mask(action_mask, name="action_mask", shape=(response_width,))
    if not isinstance(segment_ranges, Sequence) or not segment_ranges:
        raise ValueError("segment_ranges must be a non-empty sequence")
    normalized_ranges: list[tuple[int, int]] = []
    for value in segment_ranges:
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError("each segment range must be a pair")
        start, end = value
        if any(isinstance(item, bool) or not isinstance(item, int) for item in (start, end)):
            raise ValueError("segment range bounds must be integers")
        if start < 0 or end <= start or end > response_width:
            raise ValueError("segment ranges must be non-empty and inside the response")
        if not bool(actions[start:end].any()):
            raise ValueError("each segment must contain at least one action token")
        normalized_ranges.append((start, end))

    named_parameters = dict(model.named_parameters())
    if parameter_names is None:
        names = [name for name, parameter in named_parameters.items() if parameter.requires_grad]
    else:
        if isinstance(parameter_names, (str, bytes)) or not isinstance(parameter_names, Sequence):
            raise ValueError("parameter_names must be a sequence of names")
        names = list(parameter_names)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("parameter_names must contain non-empty strings")
        if len(set(names)) != len(names) or any(name not in named_parameters for name in names):
            raise ValueError("parameter_names must be unique names present in model.named_parameters()")
        if any(not named_parameters[name].requires_grad for name in names):
            raise ValueError("selected parameters must require gradients")
    if not names:
        raise ValueError("model has no selected parameters requiring gradients")
    parameters = [named_parameters[name] for name in names]
    if any(parameter.device != ids.device for parameter in parameters):
        raise ValueError("input_ids and selected model parameters must share a device")

    module_training_states = {module: module.training for module in model.modules()}
    model.eval()
    try:
        outputs = model(input_ids=ids, attention_mask=attn.to(dtype=torch.long))
        logits = getattr(outputs, "logits", None)
        if logits is None and isinstance(outputs, (tuple, list)) and outputs:
            logits = outputs[0]
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3 or logits.shape[:2] != ids.shape:
            raise ValueError("model forward must return logits with shape [1, sequence_length, vocab]")
        if not logits.requires_grad:
            raise ValueError("model logits do not require gradients")
        token_log_probs = torch.log_softmax(logits[:, :-1], dim=-1).gather(
            -1, ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)[0]
        rows: list[torch.Tensor] = []
        for index, (start, end) in enumerate(normalized_ranges):
            response_offsets = torch.arange(start, end, device=ids.device)
            selected = response_offsets[actions[start:end]]
            # Response offset r is predicted by the logit at absolute position
            # response_start+r-1, whose target is input_ids[response_start+r].
            token_positions = response_start + selected - 1
            if not bool(attn[0, response_start + selected].all()):
                raise ValueError("action tokens must be active in attention_mask")
            segment_log_prob = token_log_probs[token_positions].sum()
            gradients = torch.autograd.grad(
                segment_log_prob,
                parameters,
                retain_graph=index < len(normalized_ranges) - 1,
                allow_unused=True,
            )
            flat = torch.cat(
                [
                    (gradient if gradient is not None else torch.zeros_like(parameter)).reshape(-1)
                    for gradient, parameter in zip(gradients, parameters, strict=True)
                ]
            )
            if not torch.isfinite(flat).all():
                raise ValueError("parameter score gradient is non-finite")
            rows.append(flat.detach())
        return torch.stack(rows, dim=0)
    finally:
        for module, training in module_training_states.items():
            module.train(training)


def scalarize_boundary_values(
    boundary_values: torch.Tensor,
    schema: EndpointSchema,
    *,
    mode: str | None = None,
    pareto_alpha: torch.Tensor | None = None,
    estimates: Sequence[float] | None = None,
    penalty: float = 1.0,
    primary_objective: int | None = None,
    multipliers: Sequence[float] | None = None,
    allow_dual_bootstrap: bool = False,
) -> torch.Tensor:
    """Turn vector boundary values into an explicitly declared PG scalar.

    Pareto mode requires a supplied simplex coefficient from the vector
    gradient step.  This prevents silently replacing a multi-objective claim
    with an after-the-fact scalar reward.
    """

    _strict_bool(allow_dual_bootstrap, "allow_dual_bootstrap")
    values = torch.as_tensor(boundary_values)
    if values.ndim < 1 or values.shape[-1] != schema.dimension or not torch.isfinite(values).all():
        raise ValueError("boundary endpoint axis does not match schema")
    selected_mode = mode or schema.objective_mode
    if selected_mode == "preference":
        if schema.weights is None:
            raise ValueError("preference mode requires pre-registered weights")
        weights = torch.as_tensor(schema.weights, dtype=values.dtype, device=values.device)
        return (schema.orient_values(values) * (weights / weights.sum())).sum(dim=-1)
    if selected_mode == "pareto":
        if pareto_alpha is None:
            raise ValueError("Pareto scalarization requires vector-gradient alpha")
        alpha = torch.as_tensor(pareto_alpha, dtype=values.dtype, device=values.device)
        if alpha.numel() != schema.dimension or torch.any(alpha < 0) or not torch.isclose(alpha.sum(), torch.ones((), device=alpha.device, dtype=alpha.dtype)):
            raise ValueError("pareto_alpha must be a simplex")
        return (schema.orient_values(values) * alpha).sum(dim=-1)
    if selected_mode == "constraint":
        if not _finite(penalty) or penalty < 0:
            raise ValueError("constraint penalty must be finite and non-negative")
        primary_objective = schema.primary_index(primary_objective)
        if multipliers is None:
            if not allow_dual_bootstrap:
                raise ValueError(
                    "constraint scalarization requires persistent multipliers; "
                    "set allow_dual_bootstrap=True only for an explicit compatibility path"
                )
            if estimates is None:
                raise ValueError("dual bootstrap requires endpoint estimates")
            if len(estimates) != schema.dimension:
                raise ValueError("constraint estimates must match endpoint dimension")
            lambdas = torch.zeros(schema.dimension, dtype=values.dtype, device=values.device)
            for i, endpoint in enumerate(schema.endpoints):
                if i == primary_objective or endpoint.constraint is None:
                    continue
                if endpoint.constraint == "lower":
                    lambdas[i] = penalty * max(0.0, float(endpoint.threshold) - float(estimates[i]))
                else:
                    lambdas[i] = penalty * max(0.0, float(estimates[i]) - float(endpoint.threshold))
        else:
            # Keep the caller-owned dual state immutable while constructing
            # the scalarization coefficients.
            lambdas = torch.as_tensor(multipliers, dtype=values.dtype, device=values.device).clone()
            if lambdas.numel() != schema.dimension or torch.any(~torch.isfinite(lambdas)) or torch.any(lambdas < 0):
                raise ValueError("constraint multipliers must be a non-negative endpoint vector")
        lambdas[primary_objective] = 0.0
        coefficients = schema.constraint_coefficients(lambdas, primary_objective=primary_objective)
        return (values * coefficients).sum(dim=-1)
    raise ValueError("mode must be pareto, preference, or constraint")


def boundary_values_to_token_advantages(
    boundary_values: torch.Tensor,
    segment_ids_tensor: torch.Tensor,
    response_mask: torch.Tensor,
    schema: EndpointSchema,
    *,
    mode: str | None = None,
    pareto_alpha: torch.Tensor | None = None,
    estimates: Sequence[float] | None = None,
    primary_objective: int | None = None,
    multipliers: Sequence[float] | None = None,
    allow_dual_bootstrap: bool = False,
) -> torch.Tensor:
    """Scatter declared vector value differences to policy action tokens."""

    values = scalarize_boundary_values(
        boundary_values,
        schema,
        mode=mode,
        pareto_alpha=pareto_alpha,
        estimates=estimates,
        primary_objective=primary_objective,
        multipliers=multipliers,
        allow_dual_bootstrap=allow_dual_bootstrap,
    )
    if values.ndim == 1:
        values = values.unsqueeze(0)
    ids = torch.as_tensor(segment_ids_tensor, device=values.device, dtype=torch.long)
    mask = torch.as_tensor(response_mask, device=values.device, dtype=torch.bool)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    if mask.shape != ids.shape or ids.shape[0] != values.shape[0]:
        raise ValueError("segment IDs/mask do not match boundary value axis")
    if torch.any(mask & (ids < 0)):
        raise ValueError("active response tokens must have a non-negative segment ID")
    observed_max = int(ids[mask].max()) if torch.any(mask) else -1
    if observed_max >= values.shape[1] - 1:
        raise ValueError("segment IDs/mask do not match boundary value axis")
    delta = values[:, 1:] - values[:, :-1]
    safe = ids.clamp(0, delta.shape[1] - 1)
    token = delta.gather(1, safe)
    return token * mask.to(token.dtype)


def _science_field(source: Any, name: str) -> Any:
    """Read a tensor field from either a mapping or a TensorDict-like object."""

    if source is None:
        return None
    try:
        if name in source:
            return source[name]
    except (TypeError, KeyError):
        pass
    return None


def _science_config_value(config: Any, name: str, default: Any = None) -> Any:
    science = config.get("polyseg", {}).get("science", {}) if isinstance(config, Mapping) else getattr(config, "polyseg", {}).get("science", {}) if config is not None else {}
    if isinstance(science, Mapping):
        return science.get(name, default)
    return getattr(science, name, default)


def _science_objective_mode(config: Any) -> str | None:
    """Resolve the v4 objective mode and its legacy policy_update alias."""

    source = config
    if source is not None:
        try:
            algorithm = source.get("algorithm")
        except (AttributeError, TypeError):
            algorithm = getattr(source, "algorithm", None)
        if algorithm is not None:
            source = algorithm
    try:
        has_polyseg = source is not None and source.get("polyseg") is not None
    except (AttributeError, TypeError):
        has_polyseg = getattr(source, "polyseg", None) is not None if source is not None else False
    if has_polyseg:
        objective = _science_config_value(source, "objective_mode", None)
        policy_update = _science_config_value(source, "policy_update", None)
    else:
        try:
            objective = source.get("objective_mode") if source is not None else None
            policy_update = source.get("policy_update") if source is not None else None
        except (AttributeError, TypeError):
            objective = getattr(source, "objective_mode", None) if source is not None else None
            policy_update = getattr(source, "policy_update", None) if source is not None else None
    if objective is None:
        return policy_update
    if objective == "preference" and policy_update not in {None, "preference"}:
        return policy_update
    if policy_update not in {None, "preference", objective}:
        raise ValueError("science objective_mode and policy_update disagree")
    return str(objective)


def compute_polyseg_science_advantage(
    *,
    response_mask: torch.Tensor,
    segment_ids: torch.Tensor,
    boundary_values: torch.Tensor,
    objective_mode: str,
    endpoint_weights: Sequence[float] | None = None,
    pareto_alpha: torch.Tensor | None = None,
    constraint_multipliers: torch.Tensor | None = None,
    constraint_signs: torch.Tensor | None = None,
    primary_objective: int = 0,
    segment_mask: torch.Tensor | None = None,
    normalize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert corrected vector boundary values into token advantages.

    ``boundary_values`` is the science harness output with shape ``[B,M+1,R]``.
    This function intentionally requires an explicit objective mode and
    scalarization data.  There is no terminal-reward fallback: a malformed or
    incomplete science batch fails closed before a policy update.
    """

    _strict_bool(normalize, "science normalize_advantages")
    if isinstance(primary_objective, bool) or not isinstance(primary_objective, int):
        raise ValueError("science primary_objective must be an integer")
    values = torch.as_tensor(boundary_values)
    if values.ndim != 3 or values.shape[1] < 2 or not torch.isfinite(values).all():
        raise ValueError("science boundary values must be finite [batch, segments+1, endpoints]")
    if objective_mode not in {"preference", "pareto", "constraint"}:
        raise ValueError("science objective_mode is invalid")
    ids = torch.as_tensor(segment_ids, device=values.device, dtype=torch.long)
    mask = torch.as_tensor(response_mask, device=values.device, dtype=torch.bool)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    if mask.shape != ids.shape or ids.shape[0] != values.shape[0]:
        raise ValueError("science segment IDs and response mask must align with boundary values")
    if segment_mask is None:
        macro_mask = torch.ones(values.shape[0], values.shape[1] - 1, dtype=torch.bool, device=values.device)
    else:
        macro_mask = torch.as_tensor(segment_mask, device=values.device, dtype=torch.bool)
        if macro_mask.ndim == 1:
            macro_mask = macro_mask.unsqueeze(0)
        if macro_mask.shape != (values.shape[0], values.shape[1] - 1):
            raise ValueError("science segment_mask must have shape [batch, segments]")
    if objective_mode == "preference":
        if endpoint_weights is None or len(endpoint_weights) != values.shape[-1]:
            raise ValueError("science preference mode requires one weight per endpoint")
        if any(not _strict_real(weight) or float(weight) < 0 for weight in endpoint_weights):
            raise ValueError("science endpoint weights must be finite, non-negative, and nonzero")
        weights = torch.as_tensor(endpoint_weights, dtype=values.dtype, device=values.device)
        if torch.any(~torch.isfinite(weights)) or torch.any(weights < 0) or float(weights.sum()) <= 0:
            raise ValueError("science endpoint weights must be finite, non-negative, and nonzero")
        scalar = (values * (weights / weights.sum())).sum(-1)
    elif objective_mode == "pareto":
        if pareto_alpha is None:
            raise ValueError("science Pareto mode requires per-batch simplex coefficients")
        alpha = torch.as_tensor(pareto_alpha, dtype=values.dtype, device=values.device)
        if alpha.ndim == 1:
            alpha = alpha.unsqueeze(0).expand(values.shape[0], -1)
        if alpha.shape != (values.shape[0], values.shape[-1]) or torch.any(~torch.isfinite(alpha)) or torch.any(alpha < 0):
            raise ValueError("science Pareto coefficients must have shape [batch, endpoints]")
        if not torch.allclose(alpha.sum(-1), torch.ones(values.shape[0], device=values.device, dtype=values.dtype)):
            raise ValueError("science Pareto coefficients must lie on the simplex")
        scalar = (values * alpha[:, None, :]).sum(-1)
    else:
        if constraint_multipliers is None:
            raise ValueError("science constraint mode requires explicit multipliers")
        # The driver-owned dual state is an input to advantage computation;
        # never mutate it while forcing the primary coefficient.
        multipliers = torch.as_tensor(constraint_multipliers, dtype=values.dtype, device=values.device).clone()
        if multipliers.ndim == 1:
            # ``expand`` returns a shared-stride view; clone before forcing the
            # primary objective multiplier below.
            multipliers = multipliers.unsqueeze(0).expand(values.shape[0], -1).clone()
        if multipliers.shape != (values.shape[0], values.shape[-1]) or torch.any(~torch.isfinite(multipliers)) or torch.any(multipliers < 0):
            raise ValueError("science constraint multipliers must have shape [batch, endpoints]")
        if not 0 <= primary_objective < values.shape[-1]:
            raise ValueError("science primary_objective is outside endpoint axis")
        multipliers[:, primary_objective] = 1.0
        if constraint_signs is None:
            raise ValueError("science constraint mode requires explicit lower/upper Lagrangian signs")
        else:
            signs = torch.as_tensor(constraint_signs, dtype=values.dtype, device=values.device).reshape(-1)
            if (
                signs.numel() != values.shape[-1]
                or torch.any(~torch.isfinite(signs))
                or torch.any((signs.abs() - 1).abs() > 1e-6)
            ):
                raise ValueError("science constraint signs must be exactly -1 or +1 per endpoint")
        scalar = (values * (multipliers * signs)[..., None, :]).sum(-1)
    delta = (scalar[:, 1:] - scalar[:, :-1]).detach() * macro_mask.to(values.dtype)
    active = mask & (ids >= 0)
    if torch.any(mask & (ids < 0)):
        raise ValueError("active science tokens must have a non-negative segment ID")
    if torch.any(ids[active] >= delta.shape[1]):
        raise ValueError("science segment IDs exceed corrected boundary-value width")
    token_advantages = torch.zeros_like(mask, dtype=values.dtype)
    safe_ids = ids.clamp(0, delta.shape[1] - 1)
    token_advantages = delta.gather(1, safe_ids) * active.to(values.dtype)
    if normalize and torch.any(active):
        selected = token_advantages[active]
        token_advantages = (token_advantages - selected.mean()) / selected.std(unbiased=False).clamp_min(torch.finfo(values.dtype).eps)
        token_advantages = token_advantages * active.to(values.dtype)
    return token_advantages, token_advantages.detach().clone()


def compute_polyseg_science_advantage_registered(
    token_level_rewards: torch.Tensor | None,
    response_mask: torch.Tensor,
    config: Any = None,
    batch: Any = None,
    non_tensor_batch: Any = None,
    values: torch.Tensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Registered ``verl`` adapter for the vector science estimator."""

    boundary_values = _science_field(batch, "science_boundary_values")
    if boundary_values is None:
        boundary_values = _science_field(non_tensor_batch, "science_boundary_values")
    segment_ids = _science_field(batch, "segment_ids")
    if segment_ids is None:
        segment_ids = _science_field(non_tensor_batch, "segment_ids")
    if boundary_values is None or segment_ids is None:
        raise ValueError("polyseg_sci requires rollout-provided science_boundary_values and segment_ids")
    mode = _science_objective_mode(config)
    if mode is None:
        raise ValueError("polyseg_sci requires an explicit objective mode")
    schema_path = _science_config_value(config, "endpoint_schema_path", "")
    schema = load_endpoint_schema(schema_path) if schema_path else None
    configured_schema_hash = _science_config_value(config, "endpoint_schema_hash", "")
    if configured_schema_hash and schema is None:
        raise ValueError("science endpoint_schema_hash requires endpoint_schema_path")
    if configured_schema_hash and schema is not None and str(configured_schema_hash) != schema.fingerprint:
        raise ValueError("science endpoint_schema_hash does not match endpoint_schema_path")
    if schema is not None and str(mode) != schema.objective_mode:
        raise ValueError("science config objective_mode does not match the pre-registered endpoint schema")
    pareto_alpha = _science_field(batch, "science_pareto_alpha")
    if pareto_alpha is None:
        pareto_alpha = _science_field(non_tensor_batch, "science_pareto_alpha")
    multipliers = _science_field(batch, "science_constraint_multipliers")
    if multipliers is None:
        multipliers = _science_field(non_tensor_batch, "science_constraint_multipliers")
    segment_mask = _science_field(batch, "science_boundary_mask")
    if segment_mask is None:
        segment_mask = _science_field(non_tensor_batch, "science_boundary_mask")
    if segment_mask is None:
        segment_mask = _science_field(batch, "segment_mask")
    if segment_mask is None:
        segment_mask = _science_field(non_tensor_batch, "segment_mask")
    boundary_tensor = torch.as_tensor(boundary_values)
    endpoint_mask = _science_field(batch, "science_endpoint_mask")
    if endpoint_mask is None:
        endpoint_mask = _science_field(non_tensor_batch, "science_endpoint_mask")
    if endpoint_mask is not None:
        endpoint_mask = torch.as_tensor(endpoint_mask, dtype=torch.bool, device=boundary_tensor.device)
        expected = boundary_tensor.shape
        if endpoint_mask.shape != expected:
            raise ValueError("science_endpoint_mask must match science_boundary_values")
        # Padding on the macro axis is structural, not an observed endpoint.
        # Every boundary that contributes to an active segment still needs a
        # complete vector; inactive padded rows may be masked out explicitly.
        if segment_mask is None:
            required = torch.ones_like(endpoint_mask)
        else:
            macro_mask = torch.as_tensor(segment_mask, dtype=torch.bool, device=endpoint_mask.device)
            if macro_mask.ndim == 1:
                macro_mask = macro_mask.unsqueeze(0)
            if macro_mask.shape != (endpoint_mask.shape[0], endpoint_mask.shape[1] - 1):
                raise ValueError("science_boundary_mask must align with endpoint boundary axis")
            required = torch.zeros_like(endpoint_mask)
            active_segments = macro_mask.unsqueeze(-1)
            required[:, :-1] |= active_segments
            required[:, 1:] |= active_segments
        # Missing values are permitted only in explicitly inactive padded
        # cells.  NaN is a valid transport marker there; infinities are always
        # rejected because they usually indicate a numerical failure rather
        # than structural padding.
        if torch.any(torch.isinf(boundary_tensor)):
            raise ValueError("science_boundary_values may not contain infinity")
        if torch.any(required & ~endpoint_mask):
            raise ValueError("polyseg_sci received an incomplete active endpoint vector; no value may be fabricated")
        if torch.any(endpoint_mask & ~torch.isfinite(boundary_tensor)):
            raise ValueError("polyseg_sci received a non-finite active endpoint value")
        boundary_tensor = torch.where(endpoint_mask, boundary_tensor, torch.zeros_like(boundary_tensor))
    elif not torch.isfinite(boundary_tensor).all():
        raise ValueError("polyseg_sci science_boundary_values must be finite without an endpoint mask")
    weights = _science_config_value(config, "endpoint_weights", None)
    if schema is not None and weights is not None and schema.weights is not None:
        if any(not _strict_real(value) for value in weights):
            raise ValueError("science endpoint_weights must contain finite real numbers")
        configured_weights = tuple(float(value) for value in weights)
        if configured_weights != tuple(float(value) for value in schema.weights):
            raise ValueError("science endpoint_weights conflict with the pre-registered endpoint schema")
    if weights is None and schema is not None:
        weights = schema.weights
    configured_primary = _science_config_value(config, "primary_objective", None)
    if configured_primary is None:
        primary_objective = 0
    elif isinstance(configured_primary, str):
        if schema is None:
            raise ValueError("string science primary_objective requires an endpoint schema")
        primary_objective = schema.endpoint_index(configured_primary)
    else:
        if isinstance(configured_primary, bool) or not isinstance(configured_primary, int):
            raise ValueError("science primary_objective must be an endpoint index or endpoint id")
        primary_objective = configured_primary
    if schema is not None and str(mode) == "constraint":
        primary_objective = schema.primary_index(None if configured_primary is None else primary_objective)
    if str(mode) == "constraint" and multipliers is None:
        allow_dual_bootstrap = _strict_bool(
            _science_config_value(config, "allow_dual_bootstrap", False),
            "allow_dual_bootstrap",
        )
        if not allow_dual_bootstrap:
            raise ValueError(
                "constraint mode requires persistent science_constraint_multipliers; "
                "temporary per-batch penalties are disabled"
            )
        estimates = _science_field(batch, "science_endpoint_estimates")
        if estimates is None:
            estimates = _science_field(non_tensor_batch, "science_endpoint_estimates")
        if estimates is None or schema is None:
            raise ValueError("science constraint mode requires endpoint estimates and a schema")
        estimate_tensor = torch.as_tensor(estimates, dtype=torch.float32)
        if estimate_tensor.ndim == 1:
            estimate_tensor = estimate_tensor.unsqueeze(0)
        if estimate_tensor.shape[1] != schema.dimension:
            raise ValueError("science endpoint estimates do not match schema dimension")
        multipliers = torch.zeros_like(estimate_tensor)
        multipliers[:, primary_objective] = 1.0
        for endpoint_index, endpoint in enumerate(schema.endpoints):
            if endpoint.constraint == "lower":
                multipliers[:, endpoint_index] = 1 + (float(endpoint.threshold) - estimate_tensor[:, endpoint_index]).clamp_min(0)
            elif endpoint.constraint == "upper":
                multipliers[:, endpoint_index] = 1 + (estimate_tensor[:, endpoint_index] - float(endpoint.threshold)).clamp_min(0)
    if schema is not None and str(mode) == "constraint" and multipliers is not None:
        multiplier_tensor = torch.as_tensor(multipliers)
        if multiplier_tensor.ndim == 1:
            multiplier_tensor = multiplier_tensor.unsqueeze(0)
        unconstrained = [
            index
            for index, endpoint in enumerate(schema.endpoints)
            if index != primary_objective and endpoint.constraint is None
        ]
        if unconstrained and torch.any(multiplier_tensor[:, unconstrained] != 0):
            raise ValueError("science constraint multipliers for unconstrained endpoints must be zero")
    # Preserve raw values for constraint violation semantics.  For preference
    # and Pareto updates, however, every endpoint must first be put on the
    # registered maximize-utility scale (including minimize/target-range
    # endpoints) before scalarization.
    if schema is not None and str(mode) in {"preference", "pareto"}:
        boundary_values = schema.orient_values(boundary_tensor)
    else:
        boundary_values = boundary_tensor
    signs = None
    if schema is not None and str(mode) == "constraint":
        signs = torch.ones(schema.dimension, dtype=torch.float32)
        signs[primary_objective] = schema.objective_sign(primary_objective)
        for endpoint_index, endpoint in enumerate(schema.endpoints):
            if endpoint.constraint == "upper":
                signs[endpoint_index] = -1.0
    normalize = _strict_bool(
        _science_config_value(config, "normalize_advantages", False),
        "normalize_advantages",
    )
    return compute_polyseg_science_advantage(
        response_mask=response_mask,
        segment_ids=segment_ids,
        boundary_values=boundary_values,
        objective_mode=str(mode),
        endpoint_weights=weights,
        pareto_alpha=pareto_alpha,
        constraint_multipliers=multipliers,
        constraint_signs=signs,
        primary_objective=primary_objective,
        segment_mask=segment_mask,
        normalize=normalize,
    )


def pareto_min_norm_direction(gradients: torch.Tensor, *, steps: int = 512, learning_rate: float = 0.2) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a deterministic simplex-constrained local Pareto direction."""

    g = torch.as_tensor(gradients, dtype=torch.float64)
    if g.ndim != 2 or g.shape[0] < 1 or not torch.isfinite(g).all():
        raise ValueError("gradients must have shape [R,D]")
    if g.shape[0] == 1:
        return g[0].to(dtype=gradients.dtype), torch.ones(1, dtype=gradients.dtype, device=gradients.device)
    alpha = torch.full((g.shape[0],), 1.0 / g.shape[0], dtype=g.dtype, device=g.device)
    for _ in range(max(1, steps)):
        direction = alpha @ g
        grad_alpha = 2 * (g @ direction)
        candidate = alpha - learning_rate * grad_alpha
        sorted_candidate = torch.sort(candidate, descending=True).values
        cssv = torch.cumsum(sorted_candidate, dim=0) - 1
        ind = torch.arange(1, candidate.numel() + 1, device=alpha.device, dtype=alpha.dtype)
        rho = torch.nonzero(sorted_candidate - cssv / ind > 0, as_tuple=False)
        rho_i = int(rho[-1].item()) if rho.numel() else 0
        theta = cssv[rho_i] / float(rho_i + 1)
        alpha = (candidate - theta).clamp_min(0)
    return (alpha @ g).to(dtype=gradients.dtype), alpha.to(dtype=gradients.dtype)


def preference_direction(gradients: torch.Tensor, weights: Sequence[float]) -> torch.Tensor:
    g = torch.as_tensor(gradients)
    w = torch.as_tensor(weights, dtype=g.dtype, device=g.device)
    if g.ndim != 2 or not torch.isfinite(g).all() or w.numel() != g.shape[0] or not torch.isfinite(w).all() or torch.any(w < 0) or float(w.sum()) <= 0:
        raise ValueError("invalid preference gradient shapes/weights")
    return (w / w.sum()) @ g


def constrained_direction(
    gradients: torch.Tensor,
    schema: EndpointSchema,
    estimates: Sequence[float],
    *,
    penalty: float = 1.0,
    primary_objective: int | None = None,
    multipliers: Sequence[float] | None = None,
    allow_dual_bootstrap: bool = False,
) -> torch.Tensor:
    """Return a primal direction for a Lagrangian constrained update.

    Lower constraints contribute ``+lambda * grad`` and upper constraints
    contribute ``-lambda * grad``.  Production training must pass projected
    dual state.  The estimate-based penalty path is available only when the
    caller explicitly opts into the compatibility bootstrap.
    """

    _strict_bool(allow_dual_bootstrap, "allow_dual_bootstrap")
    g = torch.as_tensor(gradients)
    if g.ndim != 2 or g.shape[0] != schema.dimension or not torch.isfinite(g).all() or len(estimates) != schema.dimension:
        raise ValueError("gradient/estimate dimensions do not match endpoint schema")
    if any(not _finite(value) for value in estimates):
        raise ValueError("constraint estimates must be finite")
    if not _finite(penalty) or penalty < 0:
        raise ValueError("constraint penalty must be finite and non-negative")
    primary_objective = schema.primary_index(primary_objective)
    if multipliers is None:
        if not allow_dual_bootstrap:
            raise ValueError(
                "constraint direction requires persistent multipliers; "
                "set allow_dual_bootstrap=True only for compatibility"
            )
        lambdas = torch.zeros(schema.dimension, dtype=g.dtype, device=g.device)
        for i, endpoint in enumerate(schema.endpoints):
            if i == primary_objective or endpoint.constraint is None:
                continue
            if endpoint.constraint == "lower":
                lambdas[i] = penalty * max(0.0, float(endpoint.threshold) - float(estimates[i]))
            else:
                lambdas[i] = penalty * max(0.0, float(estimates[i]) - float(endpoint.threshold))
    else:
        lambdas = torch.as_tensor(multipliers, dtype=g.dtype, device=g.device)
        if lambdas.numel() != schema.dimension or torch.any(~torch.isfinite(lambdas)) or torch.any(lambdas < 0):
            raise ValueError("constraint multipliers must be a non-negative endpoint vector")
    direction = schema.objective_sign(primary_objective) * g[primary_objective].clone()
    for i, endpoint in enumerate(schema.endpoints):
        if endpoint.constraint == "lower":
            direction = direction + lambdas[i] * g[i]
        elif endpoint.constraint == "upper":
            direction = direction - lambdas[i] * g[i]
    return direction


def projected_dual_ascent(
    multipliers: torch.Tensor,
    schema: EndpointSchema,
    estimates: Sequence[float],
    *,
    learning_rate: float,
    primary_objective: int | None = None,
) -> torch.Tensor:
    """Update non-negative constraint multipliers using HF estimates.

    For a lower bound the violation is ``threshold - estimate``; for an upper
    bound it is ``estimate - threshold``.  Unconstrained and primary endpoints
    retain zero multiplier.
    """

    if learning_rate <= 0 or not _finite(learning_rate):
        raise ValueError("dual learning_rate must be positive and finite")
    estimate_values = torch.as_tensor(estimates, dtype=torch.float64).reshape(-1)
    if estimate_values.numel() != schema.dimension or not torch.isfinite(estimate_values).all():
        raise ValueError("dual estimates must match endpoint dimension")
    primary_objective = schema.primary_index(primary_objective)
    result = torch.as_tensor(multipliers, dtype=torch.float64).reshape(-1).clone()
    if result.numel() != schema.dimension or torch.any(~torch.isfinite(result)) or torch.any(result < 0):
        raise ValueError("dual multipliers must be a non-negative endpoint vector")
    for i, endpoint in enumerate(schema.endpoints):
        if i == primary_objective:
            result[i] = 0
            continue
        if endpoint.constraint == "lower":
            violation = float(endpoint.threshold) - float(estimate_values[i])
        elif endpoint.constraint == "upper":
            violation = float(estimate_values[i]) - float(endpoint.threshold)
        else:
            violation = 0.0
        result[i] = (result[i] + learning_rate * violation).clamp_min(0)
    # Preserve a tensor caller's dtype/device.  Python sequences have no
    # dtype contract; use floating point so a list such as ``[0, 0]`` cannot
    # silently truncate the projected dual update back to integer values.
    if isinstance(multipliers, torch.Tensor):
        target_dtype, target_device = multipliers.dtype, multipliers.device
    else:
        target_dtype, target_device = torch.float64, result.device
    return result.to(dtype=target_dtype, device=target_device)


@dataclass
class ConstraintDualState:
    """Persistent projected-dual state for constraint-mode training.

    The state is deliberately independent of a rollout batch.  A harness or
    trainer updates it once per audited HF estimate and serializes it with the
    run checkpoint; advantage computation consumes the resulting multiplier
    vector rather than recreating a penalty from the current noisy batch.
    """

    schema_fingerprint: str
    primary_objective: int
    multipliers: torch.Tensor
    learning_rate: float
    update_count: int = 0

    @classmethod
    def initialize(
        cls,
        schema: EndpointSchema,
        *,
        primary_objective: int,
        learning_rate: float,
        initial_multipliers: Sequence[float] | None = None,
    ) -> "ConstraintDualState":
        schema.primary_index(primary_objective)
        if not _strict_real(learning_rate) or learning_rate <= 0:
            raise ValueError("dual learning_rate must be positive and finite")
        if initial_multipliers is not None and any(not _strict_real(value) for value in initial_multipliers):
            raise ValueError("initial dual multipliers must be finite real numbers")
        values = torch.zeros(schema.dimension, dtype=torch.float64) if initial_multipliers is None else torch.as_tensor(initial_multipliers, dtype=torch.float64).reshape(-1)
        if values.numel() != schema.dimension or not torch.isfinite(values).all() or torch.any(values < 0):
            raise ValueError("initial dual multipliers must be finite and non-negative")
        values = values.clone()
        for index, endpoint in enumerate(schema.endpoints):
            if index == primary_objective or endpoint.constraint is None:
                values[index] = 0
        return cls(schema.fingerprint, primary_objective, values, float(learning_rate), 0)

    def update(self, schema: EndpointSchema, estimates: Sequence[float]) -> torch.Tensor:
        if schema.fingerprint != self.schema_fingerprint:
            raise ValueError("dual state schema fingerprint mismatch")
        self.multipliers = projected_dual_ascent(
            self.multipliers,
            schema,
            estimates,
            learning_rate=self.learning_rate,
            primary_objective=self.primary_objective,
        )
        self.update_count += 1
        return self.multipliers.clone()

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_fingerprint": self.schema_fingerprint,
            "primary_objective": self.primary_objective,
            "multipliers": self.multipliers.detach().cpu().tolist(),
            "learning_rate": self.learning_rate,
            "update_count": self.update_count,
        }

    @classmethod
    def from_state_dict(cls, schema: EndpointSchema, payload: Mapping[str, Any]) -> "ConstraintDualState":
        if payload.get("schema_fingerprint") != schema.fingerprint:
            raise ValueError("dual state schema fingerprint mismatch")
        primary_objective = payload["primary_objective"]
        if isinstance(primary_objective, bool) or not isinstance(primary_objective, int):
            raise ValueError("dual primary_objective must be an integer")
        if not _strict_real(payload["learning_rate"]) or payload["learning_rate"] <= 0:
            raise ValueError("dual learning_rate must be positive and finite")
        state = cls.initialize(
            schema,
            primary_objective=primary_objective,
            learning_rate=payload["learning_rate"],
            initial_multipliers=payload["multipliers"],
        )
        state.update_count = _strict_nonnegative_int(payload.get("update_count", 0), "dual state update_count")
        return state


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    reasons: tuple[str, ...]
    checked_fields: tuple[str, ...]


class PolySegSciVerifier:
    """Independent, conservative verifier for serialized science rounds."""

    def verify(
        self,
        *,
        schema: EndpointSchema,
        ledger: BudgetLedger,
        decisions: Sequence[QueryDecision],
        observations: Mapping[str, EndpointObservation],
        feature_map: FeatureMapContract,
        policy_version: str,
        terminal_evaluator_versions: Mapping[str, str] | None = None,
    ) -> VerificationResult:
        reasons: list[str] = []
        checked = ["endpoint_schema", "budget_ledger", "query_propensity", "observation_status", "feature_map"]
        if ledger.spent + ledger.reserved_total > ledger.budget + 1e-8:
            reasons.append("budget_exceeded")
        if feature_map.version != FEATURE_MAP_VERSION or not feature_map.fingerprint:
            reasons.append("feature_map_contract_invalid")
        if not policy_version:
            reasons.append("missing_policy_version")
        all_candidates: dict[str, QueryCandidate] = {}
        for decision in decisions:
            decision_keys = [self._candidate_key(candidate) for candidate in decision.candidates]
            if len(set(decision_keys)) != len(decision_keys):
                reasons.append("duplicate_query_candidate")
            for candidate, key in zip(decision.candidates, decision_keys, strict=True):
                if key in all_candidates:
                    reasons.append(f"duplicate_query_candidate_across_decisions:{key}")
                all_candidates[key] = candidate
            if set(decision.selected_ids) - set(decision_keys):
                reasons.append("selected_query_not_in_candidates")
            if len(set(decision.selected_ids)) != len(decision.selected_ids):
                reasons.append("duplicate_selected_query")
            for key, p in decision.inclusion_probabilities.items():
                if not _finite(p) or p < 0 or p > 1:
                    reasons.append(f"invalid_propensity:{key}")
                if key in decision.selected_ids and p <= 0:
                    reasons.append(f"nonpositive_selected_propensity:{key}")
            for key in decision.selected_ids:
                if key not in decision.inclusion_probabilities:
                    reasons.append(f"selected_propensity_missing:{key}")
            # Cost accounting is local to a decision.  Reusing a key in a
            # later decision must not make an earlier decision's planned cost
            # depend on a global last-write-wins candidate map.
            local_candidates = {
                self._candidate_key(candidate): candidate for candidate in decision.candidates
            }
            expected_cost = sum(
                local_candidates[key].cost
                for key in decision.selected_ids
                if key in local_candidates
            )
            if abs(float(decision.planned_cost) - expected_cost) > 1e-8:
                reasons.append("planned_query_cost_mismatch")
            if decision.planned_cost < -1e-9 or (decision.planned_cost > decision.budget + 1e-8 and decision.hard_budget):
                reasons.append("query_plan_exceeds_budget")
            for key, probability in decision.pair_inclusion_probabilities.items():
                if not _finite(probability) or probability < 0 or probability > 1:
                    reasons.append(f"invalid_pair_propensity:{key}")
                pair = next(
                    (
                        pair
                        for pair in combinations(decision_keys, 2)
                        if "::".join(pair) == key
                    ),
                    None,
                )
                if pair is not None:
                    p_left = float(decision.inclusion_probabilities.get(pair[0], 0.0))
                    p_right = float(decision.inclusion_probabilities.get(pair[1], 0.0))
                    lower = max(0.0, p_left + p_right - 1.0)
                    upper = min(p_left, p_right)
                    if float(probability) < lower - 1e-8 or float(probability) > upper + 1e-8:
                        reasons.append(f"inconsistent_pair_propensity:{key}")
            for key, probability in decision.response_probabilities.items():
                if not _finite(probability) or not 0 < probability <= 1:
                    reasons.append(f"invalid_response_probability:{key}")
        for key, obs in observations.items():
            if obs.status not in {"observed", "censored", "interval"} and obs.value is not None:
                reasons.append(f"failed_observation_has_value:{key}")
            try:
                schema.validate_observation(obs, allow_empty_provenance=True)
            except ValueError as exc:
                reasons.append(f"invalid_observation:{key}:{exc}")
            expected_key = key.rsplit("::", 1)[-1] if "::" in key else ""
            if expected_key != obs.endpoint_id:
                reasons.append(f"observation_key_endpoint_mismatch:{key}")
        selected_keys = {key for decision in decisions for key in decision.selected_ids}
        for key in selected_keys:
            if key not in observations:
                reasons.append(f"selected_observation_missing:{key}")
        for key in observations:
            if key not in all_candidates:
                reasons.append(f"observation_not_in_query_candidates:{key}")
        if terminal_evaluator_versions is not None:
            for endpoint in schema.endpoints:
                if endpoint.proxy_only:
                    reasons.append(f"proxy_only_endpoint_has_no_hf:{endpoint.endpoint_id}")
                    continue
                hf = next(f for f in endpoint.fidelities if f.fidelity_id == endpoint.hf_fidelity_id)
                if terminal_evaluator_versions.get(endpoint.endpoint_id) != hf.evaluator_version:
                    reasons.append(f"terminal_evaluator_version_mismatch:{endpoint.endpoint_id}")
        return VerificationResult(not reasons, tuple(sorted(set(reasons))), tuple(checked))

    @staticmethod
    def _candidate_key(candidate: QueryCandidate) -> str:
        return f"{candidate.boundary_id}::{candidate.endpoint_id}"


# Register lazily with the common estimator registry.  Importing this module
# is already explicit in ``core_algos.get_adv_estimator_fn`` for the science
# estimator, so ordinary scalar PolySeg users do not pay this import cost.
from verl.trainer.ppo.core_algos import AdvantageEstimator, register_adv_est  # noqa: E402


@register_adv_est(AdvantageEstimator.POLYSEG_SCI)
def _registered_polyseg_science_advantage(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
    return compute_polyseg_science_advantage_registered(*args, **kwargs)


__all__ = [
    "SCI_METHOD_VERSION",
    "ENDPOINT_SCHEMA_VERSION",
    "FEATURE_MAP_VERSION",
    "SEGMENT_RULE_VERSION",
    "FidelitySpec",
    "EndpointSpec",
    "EndpointSchema",
    "load_endpoint_schema",
    "load_feature_map_contract",
    "EndpointObservation",
    "ActionRecord",
    "SegmentBoundary",
    "CausalSegmenter",
    "FeatureMapContract",
    "PolynomialVectorPredictor",
    "BoundarySnapshot",
    "TrajectoryRecord",
    "BoundaryRecord",
    "RestoreAudit",
    "StateAuditOracle",
    "ScientificEndpointEvaluator",
    "QueryCandidate",
    "QueryDecision",
    "BudgetedQuerySampler",
    "BudgetLedger",
    "BranchResult",
    "FreshBranchRunner",
    "ScienceRoundResult",
    "run_science_round",
    "mean_observations",
    "residual_correct",
    "residual_correction_covariance",
    "MultiFidelityCalibrator",
    "trajectory_bootstrap_ci",
    "segment_score_gradient",
    "parameter_score_gradients",
    "scalarize_boundary_values",
    "boundary_values_to_token_advantages",
    "pareto_min_norm_direction",
    "preference_direction",
    "constrained_direction",
    "projected_dual_ascent",
    "ConstraintDualState",
    "compute_polyseg_science_advantage",
    "compute_polyseg_science_advantage_registered",
    "VerificationResult",
    "PolySegSciVerifier",
    "canonical_json",
    "content_hash",
    "BranchExecutionError",
    "EvaluatorExecutionError",
]
