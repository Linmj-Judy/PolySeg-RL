from examples.polyseg_trainer.function_tools import calculator


def test_calculator_default_is_exact(monkeypatch):
    monkeypatch.delenv("POLYSEG_CALCULATOR_NOISE_P", raising=False)
    assert calculator("2 + 3") == "5"


def test_calculator_noise_is_deterministic_and_auditable(monkeypatch):
    monkeypatch.setenv("POLYSEG_CALCULATOR_NOISE_P", "1")
    monkeypatch.setenv("POLYSEG_CALCULATOR_NOISE_SEED", "unit-test")
    first = calculator("2 + 3")
    second = calculator("2 + 3")
    assert first == second == "NOISY_RESULT: 6"


def test_calculator_noise_probability_is_validated(monkeypatch):
    monkeypatch.setenv("POLYSEG_CALCULATOR_NOISE_P", "2")
    assert calculator("2 + 3").startswith("ERROR:")
