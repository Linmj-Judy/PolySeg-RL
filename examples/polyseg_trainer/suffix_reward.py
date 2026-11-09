"""HTTP-backed suffix utility verifier for PolySeg research runs.

The endpoint is an inference service owned by the rollout experiment.  It must
accept ``{"prompt": ..., "prefix": ..., "num_samples": ...}`` and return
``{"suffixes": ["...", ...], "billing_receipt": {...}}``.  The receipt is
optional for compatibility, but required before a lower bill can be accepted.
No local/random fallback is provided: if the
suffix service is unavailable, the run fails instead of silently changing the
feedback semantics.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any
from urllib.request import Request, urlopen

import numpy as np

from verl.utils.reward_score import default_compute_score


class _SuffixSamples(list[str]):
    """List-compatible suffix batch carrying optional provider evidence."""

    def __init__(self, values: list[str], billing_receipt: dict[str, Any] | None = None):
        super().__init__(values)
        self.billing_receipt = billing_receipt


def _strict_suffix_score(data_source: str, solution: str, ground_truth: str, extra_info: dict[str, Any]) -> float:
    """Run the strict verifier, accepting the checkpoint's explicit boxed form.

    The training dataset asks the agent for ``#### number`` while Qwen
    checkpoints frequently emit ``\\boxed{number}``.  This adapter mirrors the
    terminal research scorer's formatting-only canonicalisation and still sends
    the resulting answer through the strict GSM8K verifier.
    """
    result = default_compute_score(data_source, solution, ground_truth, extra_info=extra_info)
    score = float(result.get("score", 0.0)) if isinstance(result, dict) else float(result)
    if score == 0.0 and str(data_source) == "openai/gsm8k":
        matches = re.findall(r"\\(?:boxed|fbox)\s*\{\s*([^{}]+?)\s*\}", solution)
        if matches:
            answer = matches[-1].strip().replace(",", "").replace("$", "")
            if re.fullmatch(r"-?[0-9]+(?:\.[0-9]+)?", answer):
                result = default_compute_score(data_source, f"#### {answer}", ground_truth, extra_info=extra_info)
    return float(result.get("score", 0.0)) if isinstance(result, dict) else float(result)


def _jsonable(value: Any) -> Any:
    """Convert DataProto/object-array values into JSON-compatible containers.

    ``raw_prompt`` can be reconstructed as an object ndarray by the transfer
    queue.  Preserve its nested message structure while removing numpy scalar
    and array wrappers before sending it to the HTTP suffix sampler.
    """
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _normalise_prompt(value: Any) -> Any:
    """Remove a singleton object-array wrapper introduced by batch transport."""
    value = _jsonable(value)
    while isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    return value


def _sample_suffixes(prompt: Any, prefix: str, num_samples: int) -> list[str]:
    endpoint = os.environ.get("POLYSEG_SUFFIX_ENDPOINT")
    if not endpoint:
        raise RuntimeError("POLYSEG_SUFFIX_ENDPOINT is required for utility_delta feedback")
    payload = json.dumps(
        {
            "prompt": _normalise_prompt(prompt),
            "prefix": prefix,
            "num_samples": int(num_samples),
            "cost_per_sample": float(os.environ.get("POLYSEG_SUFFIX_COST_PER_SAMPLE", "1.0")),
            "temperature": float(os.environ.get("POLYSEG_SUFFIX_TEMPERATURE", "1.0")),
            "max_tokens": int(os.environ.get("POLYSEG_SUFFIX_MAX_TOKENS", "1024")),
            "seed": (
                int(os.environ["POLYSEG_SUFFIX_SEED"])
                if os.environ.get("POLYSEG_SUFFIX_SEED") is not None
                else None
            ),
        }
    ).encode("utf-8")
    request = Request(endpoint, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=float(os.environ.get("POLYSEG_SUFFIX_TIMEOUT", "120"))) as response:
        result = json.loads(response.read().decode("utf-8"))
    suffixes = result.get("suffixes") if isinstance(result, dict) else None
    if not isinstance(suffixes, list) or not suffixes or not all(isinstance(item, str) for item in suffixes):
        raise ValueError("suffix endpoint must return a non-empty string list under 'suffixes'")
    if len(suffixes) != int(num_samples):
        raise ValueError(
            f"suffix endpoint returned {len(suffixes)} samples; expected exactly {int(num_samples)}"
        )
    billing_receipt = result.get("billing_receipt") if isinstance(result, dict) else None
    return _SuffixSamples(suffixes, billing_receipt)


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict[str, Any] | None = None):
    extra_info = extra_info or {}
    if str(extra_info.get("polyseg_protocol", "")).strip().lower() != "utility_delta":
        raise RuntimeError("suffix_reward.compute_score requires feedback=utility_delta")

    samples = int(os.environ.get("POLYSEG_SUFFIX_SAMPLES", "4"))
    if samples < 1:
        raise ValueError("POLYSEG_SUFFIX_SAMPLES must be positive")
    remaining = extra_info.get("budget_remaining")
    cost_per_sample = float(os.environ.get("POLYSEG_SUFFIX_COST_PER_SAMPLE", "1.0"))
    requested_cost = samples * cost_per_sample
    if remaining is not None and requested_cost > float(remaining) + 1e-8:
        raise RuntimeError(
            f"suffix query requires {requested_cost} budget but only {remaining} remains; "
            "reduce POLYSEG_SUFFIX_SAMPLES or increase the configured budget"
        )

    prefix = str(extra_info.get("segment_prefix", ""))
    sampled = _sample_suffixes(extra_info.get("raw_prompt"), prefix, samples)
    # Keep compatibility with local tests and custom samplers that return only
    # the suffix list; real providers may additionally return billing evidence.
    if isinstance(sampled, tuple) and len(sampled) == 2:
        suffixes, billing_receipt = sampled
    else:
        suffixes, billing_receipt = sampled, getattr(sampled, "billing_receipt", None)
    outcomes = []
    for suffix in suffixes:
        outcome = _strict_suffix_score(data_source, prefix + suffix, ground_truth, extra_info)
        outcomes.append(float(outcome > 0))
    utility = sum(outcomes) / len(outcomes)
    result = {
        "utility": utility,
        "query_cost": len(suffixes) * cost_per_sample,
        "suffix_samples": len(suffixes),
        "suffix_successes": int(sum(outcomes)),
    }
    if billing_receipt is not None:
        result["billing_receipt"] = billing_receipt
    # Utility and observed final correctness are different signals.  When the
    # manager supplies the complete response on the terminal segment, expose a
    # local exact-verifier diagnostic without counting it as a suffix sample.
    segment_id = int(extra_info.get("segment_id", 0))
    segment_count = int(extra_info.get("segment_count", 1))
    if segment_id == segment_count - 1 and "full_response" in extra_info:
        observed_score = _strict_suffix_score(
            data_source, str(extra_info["full_response"]), ground_truth, extra_info
        )
        result.update({"verifier_score": float(observed_score), "acc": bool(observed_score > 0)})
    return result
