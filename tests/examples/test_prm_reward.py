from examples.polyseg_trainer import prm_reward


def test_prm_final_accuracy_accepts_structured_math_dapo_verifier(monkeypatch):
    monkeypatch.setattr(prm_reward, "_load_prm", lambda *args: ("tok", "model", "cpu"))
    monkeypatch.setattr(prm_reward, "_separator_probability", lambda *args: 0.73)
    monkeypatch.setattr(
        prm_reward,
        "default_compute_score",
        lambda *args, **kwargs: {"score": -1.0, "acc": False, "pred": "wrong"},
    )
    result = prm_reward.compute_score(
        "math_dapo",
        "prefix",
        "answer",
        {
            "segment_prefix": "prefix",
            "full_response": "prefix",
            "raw_prompt": [],
            "segment_id": 0,
            "segment_count": 1,
        },
        prm_model_path="unused",
        prm_device="cpu",
    )
    assert result["utility"] == 0.73
    assert result["acc"] == 0.0
    assert result["verifier_score"] == -1.0
    assert result["verifier_pred"] == "wrong"
    assert result["feedback_protocol_contract"]["label_semantics"] == "prm_model_derived_pseudo_label"
