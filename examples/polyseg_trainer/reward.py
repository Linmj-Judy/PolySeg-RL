"""Research verifier contract for PolySeg agent trajectories.

This function deliberately does not manufacture reward from token length or
from the presence of text in a segment.  ``utility_delta`` requires a real
suffix sampler to provide one Monte-Carlo utility per AgentLoop boundary.
"""

from __future__ import annotations

import os
import re
from typing import Any

from verl.utils.reward_score import default_compute_score
from verl.trainer.ppo.polyseg import feedback_protocol_contract


def _canonical_gsm8k_response(solution: str) -> str | None:
    """Map a Qwen-style explicit boxed final answer to GSM8K's marker.

    This is only a formatting adapter: the resulting ``####`` answer is
    passed back through verl's strict GSM8K verifier.  Intermediate numbers
    and unmarked prose are never treated as reward-bearing answers.
    """
    matches = re.findall(r"\\(?:boxed|fbox)\s*\{\s*([^{}]+?)\s*\}", solution)
    if not matches:
        return None
    answer = matches[-1].strip().replace(",", "").replace("$", "")
    if not re.fullmatch(r"-?[0-9]+(?:\.[0-9]+)?", answer):
        return None
    return f"#### {answer}"


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
):
    extra_info = extra_info or {}
    # The same scorer can serve the terminal baseline through verl's naive
    # reward manager, which does not inject PolySeg metadata.
    protocol = str(extra_info.get("polyseg_protocol", "terminal")).strip().lower()
    segment_id = int(extra_info.get("segment_id", 0))

    if protocol == "utility_delta":
        utilities = extra_info.get("suffix_utilities")
        utility = extra_info.get("suffix_utility")
        if utility is None and utilities is not None:
            if not isinstance(utilities, (list, tuple)) or segment_id >= len(utilities):
                raise ValueError("suffix_utilities must contain one utility per rollout segment")
            utility = utilities[segment_id]
        if utility is None:
            raise RuntimeError(
                "utility_delta requires suffix_utility/suffix_utilities from a real suffix sampler; "
                "the research scorer will not infer utility from segment text"
            )
        utility = float(utility)
        if not 0.0 <= utility <= 1.0:
            raise ValueError(f"suffix utility must be in [0, 1], got {utility}")
        return {
            "utility": utility,
            "query_cost": float(extra_info.get("suffix_query_cost", extra_info.get("suffix_samples", 1))),
            "suffix_samples": int(extra_info.get("suffix_samples", 1)),
            "feedback_protocol_contract": feedback_protocol_contract(protocol),
        }

    if protocol in {"terminal", "binary_local", "binary"}:
        full_response = str(extra_info.get("full_response", solution_str))
        # ``binary_local`` queries the exact GSM8K verifier on the observed
        # prefix.  Intermediate prefixes normally receive zero and the final
        # complete prefix receives one when correct; this is a real binary
        # verifier query, not a text/length heuristic.
        verifier_response = (
            str(extra_info.get("segment_prefix", solution_str))
            if protocol in {"binary_local", "binary"}
            else full_response
        )
        raw_score = default_compute_score(
            data_source,
            verifier_response,
            ground_truth,
            extra_info=extra_info,
        )
        # ``math_dapo`` returns a structured verifier result containing both
        # signed reward and strict accuracy.  GSM8K returns a scalar.  Keep
        # both forms explicit so the PolySeg manager can consume ``score``
        # while validation metrics still use ``acc``.
        verifier_info = dict(raw_score) if isinstance(raw_score, dict) else {}
        score = float(verifier_info.get("score", raw_score))
        verifier_mode = "gsm8k_strict" if str(data_source) == "openai/gsm8k" else "math_dapo_strict"
        if float(score) == 0.0 and str(data_source) == "openai/gsm8k":
            canonical = _canonical_gsm8k_response(verifier_response)
            if canonical is not None:
                canonical_score = default_compute_score(data_source, canonical, ground_truth, extra_info=extra_info)
                if isinstance(canonical_score, dict):
                    verifier_info.update(canonical_score)
                    score = float(canonical_score.get("score", 0.0))
                else:
                    score = float(canonical_score)
                verifier_mode = "gsm8k_strict_after_boxed_canonicalization"
        if os.getenv("POLYSEG_DEBUG_REWARD", "0") == "1":
            response = full_response.replace("\n", "\\n")
            print(
                "POLYSEG_DEBUG_REWARD "
                f"segment={segment_id} data_source={data_source!r} gt={ground_truth!r} "
                f"score={float(score):.3f} verifier={verifier_mode} response={response[-1200:]!r}",
                flush=True,
            )
        if protocol in {"binary_local", "binary"}:
            return {
                "score": float(score > 0.5),
                "binary_label": float(score > 0.5),
                "acc": float(verifier_info.get("acc", score > 0.5)),
                "query_cost": float(extra_info.get("binary_query_cost", 1.0)),
                "verifier_mode": verifier_mode,
                "feedback_protocol_contract": feedback_protocol_contract(protocol),
            }
        if verifier_info:
            verifier_info.update({
                "score": score,
                "verifier_mode": verifier_mode,
                "feedback_protocol_contract": feedback_protocol_contract(protocol),
            })
            # GSM8K's strict scorer historically returned a scalar rather
            # than a structured ``{"score", "acc"}`` payload.  The score is
            # binary correctness for this task, so expose the observed
            # per-example accuracy explicitly for validation artifacts.
            if str(data_source) == "openai/gsm8k" and "acc" not in verifier_info:
                verifier_info["acc"] = float(score > 0.5)
            return verifier_info
        if str(data_source) == "openai/gsm8k":
            return {
                "score": float(score),
                "acc": float(score > 0.5),
                "verifier_mode": verifier_mode,
                "feedback_protocol_contract": feedback_protocol_contract(protocol),
            }
        return score

    raise RuntimeError(
        f"PolySeg research scorer has no implicit implementation for protocol {protocol!r}; "
        "provide a real verifier/suffix utility producer"
    )
