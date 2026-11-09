from examples.polyseg_trainer import reward


def test_terminal_gsm8k_scalar_verifier_materializes_accuracy(monkeypatch):
    monkeypatch.setattr(reward, "default_compute_score", lambda *args, **kwargs: 1.0)
    result = reward.compute_score(
        "openai/gsm8k",
        "#### 4",
        "4",
        {"polyseg_protocol": "terminal", "full_response": "#### 4"},
    )
    assert result["score"] == 1.0
    assert result["acc"] == 1.0
    assert result["verifier_mode"] == "gsm8k_strict"
    assert result["feedback_protocol_contract"]["label_semantics"] == "deterministic_terminal_verifier"


def test_terminal_gsm8k_structured_verifier_preserves_accuracy(monkeypatch):
    monkeypatch.setattr(
        reward,
        "default_compute_score",
        lambda *args, **kwargs: {"score": 0.0, "acc": 0.0, "pred": "wrong"},
    )
    result = reward.compute_score(
        "openai/gsm8k",
        "#### 4",
        "4",
        {"polyseg_protocol": "terminal", "full_response": "#### 4"},
    )
    assert result["score"] == 0.0
    assert result["acc"] == 0.0


def test_binary_verifier_declares_deterministic_semantics(monkeypatch):
    monkeypatch.setattr(reward, "default_compute_score", lambda *args, **kwargs: {"score": 1.0, "acc": True})
    result = reward.compute_score(
        "math_dapo",
        "partial",
        "42",
        {"polyseg_protocol": "binary_local", "segment_prefix": "partial"},
    )
    assert result["binary_label"] == 1.0
    assert result["feedback_protocol_contract"]["label_semantics"] == "deterministic_prefix_verifier"
