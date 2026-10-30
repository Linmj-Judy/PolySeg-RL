import json

import numpy as np

from examples.polyseg_trainer import suffix_reward
from examples.polyseg_trainer.suffix_reward import _jsonable, _normalise_prompt


def test_jsonable_converts_object_array_chat_prompt():
    prompt = np.asarray([[{"role": "user", "content": "Compute 2+2"}]], dtype=object)

    assert _jsonable(prompt) == [[{"role": "user", "content": "Compute 2+2"}]]
    assert _normalise_prompt(prompt) == [{"role": "user", "content": "Compute 2+2"}]


def test_compute_score_accepts_boxed_suffix_via_strict_verifier(monkeypatch):
    monkeypatch.setenv("POLYSEG_SUFFIX_SAMPLES", "1")
    monkeypatch.setattr(suffix_reward, "_sample_suffixes", lambda *args: [r"\boxed{72}"])

    result = suffix_reward.compute_score(
        "openai/gsm8k",
        "",
        "72",
        {"polyseg_protocol": "utility_delta", "raw_prompt": [], "budget_remaining": 1},
    )

    assert result["utility"] == 1.0


def test_suffix_reports_observed_final_correctness_separately(monkeypatch):
    monkeypatch.setenv("POLYSEG_SUFFIX_SAMPLES", "1")
    monkeypatch.setattr(suffix_reward, "_sample_suffixes", lambda *args: [r"\boxed{72}"])
    result = suffix_reward.compute_score(
        "openai/gsm8k",
        "",
        "72",
        {
            "polyseg_protocol": "utility_delta",
            "raw_prompt": [],
            "full_response": r"\boxed{72}",
            "segment_id": 0,
            "segment_count": 1,
            "budget_remaining": 1,
        },
    )
    assert result["utility"] == 1.0
    assert result["acc"] is True and result["verifier_score"] == 1.0


def test_suffix_endpoint_requires_exact_requested_cardinality(monkeypatch):
    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"suffixes": ["a", "b"]}).encode()

    monkeypatch.setenv("POLYSEG_SUFFIX_ENDPOINT", "http://suffix.test")
    monkeypatch.setattr(suffix_reward, "urlopen", lambda *args, **kwargs: _Response())
    try:
        suffix_reward._sample_suffixes([], "", 1)
    except ValueError as exc:
        assert "expected exactly 1" in str(exc)
    else:
        raise AssertionError("over-returning suffix endpoint must be rejected")


def test_strict_suffix_score_accepts_structured_verifier(monkeypatch):
    monkeypatch.setattr(
        suffix_reward,
        "default_compute_score",
        lambda *args, **kwargs: {"score": 1.0, "acc": 1.0, "pred": "72"},
    )
    assert suffix_reward._strict_suffix_score("math_dapo", "answer", "72", {}) == 1.0
