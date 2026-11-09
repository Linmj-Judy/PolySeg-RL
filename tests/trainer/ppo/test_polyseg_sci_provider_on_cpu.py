import asyncio

import pytest
import torch
from omegaconf import OmegaConf

import verl.trainer.ppo.polyseg_openai4s_provider as openai4s_provider
from verl.trainer.config.algorithm import PolySegScienceConfig
from verl.trainer.ppo.polyseg_sci_provider import (
    CallbackScienceRoundProvider,
    ScienceRoundRequest,
    ScienceRoundResponse,
    attach_science_response,
    load_science_round_provider,
    science_enabled,
)


def _fields():
    return {
        "segment_ids": torch.tensor([0, 0, 1]),
        "segment_mask": torch.tensor([True, True]),
        "science_boundary_values": torch.tensor([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]),
        "science_boundary_mask": torch.tensor([True, True]),
        "science_endpoint_mask": torch.ones(3, 2, dtype=torch.bool),
    }


def test_provider_response_rejects_incomplete_active_endpoint_vector():
    values = _fields()
    values["science_endpoint_mask"][1, 0] = False
    with pytest.raises(ValueError, match="active science boundaries"):
        ScienceRoundResponse(fields=values)


def test_provider_response_rejects_misaligned_segment_metadata():
    values = _fields()
    values["segment_mask"] = torch.tensor([True])
    with pytest.raises(ValueError, match="segment_mask must match segment axis"):
        ScienceRoundResponse(fields=values)

    values = _fields()
    values["segment_ids"] = torch.tensor([0, 0, 2])
    with pytest.raises(ValueError, match="segment_ids exceed"):
        ScienceRoundResponse(fields=values)

    values = _fields()
    values["science_boundary_mask"] = torch.tensor([True, False])
    values["segment_mask"] = torch.tensor([True, False])
    values["science_boundary_mask"][1] = True
    with pytest.raises(ValueError, match="active science boundaries require active segment_mask"):
        ScienceRoundResponse(fields=values)


def test_provider_response_attaches_verified_fields_and_audit():
    class Output:
        def __init__(self):
            self.extra_fields = {}

    outputs = [Output(), Output()]
    response = ScienceRoundResponse(fields=_fields(), audit={"ledger_fingerprint": "abc"})
    attach_science_response(outputs, response)
    for output in outputs:
        assert torch.equal(output.extra_fields["segment_ids"], torch.tensor([0, 0, 1]))
        assert output.extra_fields["science_round_audit"] == {"ledger_fingerprint": "abc"}
        assert output.extra_fields["science_research_eligible"] is False


def test_attach_science_response_downgrades_unverified_eligibility():
    class Output:
        def __init__(self):
            self.extra_fields = {}

    output = Output()
    response = ScienceRoundResponse(fields=_fields(), audit={"source": "custom"}, research_eligible=True)
    attach_science_response([output], response)
    assert output.extra_fields["science_research_eligible"] is False
    assert output.extra_fields["science_round_audit"]["research_eligibility_rejection"] == (
        "response_model_not_validated"
    )


def test_callback_provider_rejects_coercible_research_eligibility():
    def callback(request):
        return {"fields": _fields(), "research_eligible": "false"}

    with pytest.raises(ValueError, match="research_eligible"):
        asyncio.run(
            CallbackScienceRoundProvider(callback).execute(
                ScienceRoundRequest(output=None, outputs=(), validate=False, context={})
            )
        )


def test_callback_provider_supports_sync_and_async_callbacks():
    def callback(request):
        assert isinstance(request, ScienceRoundRequest)
        return {"fields": _fields(), "audit": {"source": "test"}}

    provider = CallbackScienceRoundProvider(callback)
    response = asyncio.run(
        provider.execute(ScienceRoundRequest(output=None, outputs=(), validate=False, context={}))
    )
    assert response.audit["source"] == "test"

    async def async_callback(request):
        return {"fields": _fields(), "research_eligible": True}

    response = asyncio.run(
        CallbackScienceRoundProvider(async_callback).execute(
            ScienceRoundRequest(output=None, outputs=(), validate=False, context={})
        )
    )
    assert response.research_eligible is False
    assert response.audit["research_eligibility_rejection"] == "response_model_not_validated"

    async def audited_callback(request):
        return {
            "fields": _fields(),
            "audit": {"correction": {"unbiased_under_response_model": True}},
            "research_eligible": True,
        }

    response = asyncio.run(
        CallbackScienceRoundProvider(audited_callback).execute(
            ScienceRoundRequest(output=None, outputs=(), validate=False, context={})
        )
    )
    assert response.research_eligible is True


def test_openai4s_provider_executes_existing_round_and_adapter(monkeypatch):
    class Verification:
        passed = True
        reasons = ()
        checked_fields = ("ledger",)

    class Result:
        corrected_boundary_values = torch.tensor([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
        segment_gradient = torch.tensor([[0.1], [0.2]])
        verification = Verification()
        ledger = {"spent": 3.0}
        audit = {"source": "openai4s"}
        boundary_records = ()
        covariance_by_boundary = {}

    called = {}

    def fake_round(**kwargs):
        called.update(kwargs)
        return Result()

    monkeypatch.setattr(openai4s_provider, "run_science_round", fake_round)

    class Schema:
        dimension = 2
        fingerprint = ""

    class Adapter:
        schema = Schema()

        def science_rollout_fields(self, *, action_mask, boundaries, corrected_boundary_values, endpoint_mask):
            assert len(action_mask) == 3 and len(boundaries) == 2
            assert corrected_boundary_values.shape == (3, 2)
            return _fields()

    request = ScienceRoundRequest(
        output=None,
        outputs=(),
        validate=False,
        context={
            "openai4s_science_round": {
                "adapter": Adapter(),
                "action_mask": [True, True, True],
                "boundaries": ["b0", "b1"],
                "round_kwargs": {
                    "marker": "real-harness-contract",
                    "boundary_ids": ("b0",),
                },
                "round_id": "round-1",
                "research_eligible": False,
            }
        },
    )
    response = asyncio.run(openai4s_provider.OpenAI4SScienceRoundProvider().execute(request))
    assert called == {"marker": "real-harness-contract", "boundary_ids": ("b0",)}
    assert response.audit["provider"] == "openai4s"
    assert response.audit["verification"]["passed"] is True
    assert response.research_eligible is False


class _ProviderSchema:
    dimension = 1
    fingerprint = ""


class _ProviderAdapter:
    schema = _ProviderSchema()

    def science_rollout_fields(self, **_kwargs):
        raise AssertionError("provider must fail before adapter fields are built")


def _provider_spec(**overrides):
    spec = {
        "adapter": _ProviderAdapter(),
        "action_mask": (True, True, True),
        "boundaries": ("b0", "b1"),
        "round_kwargs": {"boundary_ids": ("b0",)},
        "research_eligible": False,
    }
    spec.update(overrides)
    return spec


def _run_provider(monkeypatch, spec, called):
    def fail_round(**kwargs):
        called.append(kwargs)
        raise AssertionError("provider preconditions must fail before run_science_round")

    monkeypatch.setattr(openai4s_provider, "run_science_round", fail_round)
    return asyncio.run(
        openai4s_provider.OpenAI4SScienceRoundProvider().execute(
            ScienceRoundRequest(
                output=None,
                outputs=(),
                validate=False,
                context={"openai4s_science_round": spec},
            )
        )
    )


def test_openai4s_provider_rejects_boundary_shape_before_callbacks(monkeypatch):
    called = []
    spec = _provider_spec(boundaries=("b0", "b1", "b2"))
    with pytest.raises(openai4s_provider.OpenAI4SProviderError, match="boundary contract mismatch"):
        _run_provider(monkeypatch, spec, called)
    assert called == []


def test_openai4s_provider_rejects_truncated_terminal_before_callbacks(monkeypatch):
    called = []
    with pytest.raises(openai4s_provider.OpenAI4SProviderError, match="terminal is incomplete"):
        _run_provider(monkeypatch, _provider_spec(truncated=True), called)
    assert called == []
    with pytest.raises(openai4s_provider.OpenAI4SProviderError, match="terminal is incomplete"):
        _run_provider(monkeypatch, _provider_spec(finish_reason="length"), called)
    assert called == []


def test_openai4s_provider_cannot_promote_self_attested_restore(monkeypatch):
    called = []
    spec = _provider_spec(
        research_eligible=True,
        trajectory_complete=True,
        restore_evidence_kind="self_attested",
        continuation_mode="restored_boundary",
    )
    with pytest.raises(openai4s_provider.OpenAI4SProviderError, match="durable restore evidence"):
        _run_provider(monkeypatch, spec, called)
    assert called == []


def test_openai4s_provider_research_requires_explicit_completion(monkeypatch):
    called = []
    spec = _provider_spec(
        research_eligible=True,
        restore_evidence_kind="durable",
        continuation_mode="restored_boundary",
    )
    with pytest.raises(openai4s_provider.OpenAI4SProviderError, match="trajectory_complete=True"):
        _run_provider(monkeypatch, spec, called)
    assert called == []


def test_enabled_science_requires_explicit_provider_path():
    disabled = OmegaConf.create({"algorithm": {"adv_estimator": "gae", "polyseg": {"science": {}}}})
    assert science_enabled(disabled) is False
    assert load_science_round_provider(disabled) is None

    enabled = OmegaConf.create(
        {"algorithm": {"adv_estimator": "polyseg_sci", "polyseg": {"science": {"enabled": True}}}}
    )
    assert science_enabled(enabled) is True
    with pytest.raises(ValueError, match="provider_path is empty"):
        load_science_round_provider(enabled)


def test_enabled_science_requires_pre_registered_endpoint_schema():
    enabled = OmegaConf.create(
        {
            "algorithm": {
                "adv_estimator": "polyseg_sci",
                "polyseg": {
                    "science": {
                        "enabled": True,
                        "provider_path": "verl.trainer.ppo.polyseg_sci_provider:CallbackScienceRoundProvider",
                    }
                },
            }
        }
    )
    with pytest.raises(ValueError, match="endpoint_schema_path is empty"):
        load_science_round_provider(enabled)


@pytest.mark.parametrize(
    "field,value",
    [
        ("enabled", "false"),
        ("hard_budget", "false"),
        ("require_restore_audit", 1),
        ("allow_dual_bootstrap", "false"),
        ("predictor_q", 1.5),
        ("predictor_q", True),
        ("branch_count", 1.5),
        ("branch_count", True),
        ("predictor_ridge_lambda", True),
        ("train_budget", True),
        ("dual_learning_rate", "0.1"),
        ("endpoint_weights", (True,)),
        ("initial_multipliers", ("1.0",)),
    ],
)
def test_science_config_rejects_coercible_contract_fields(field, value):
    with pytest.raises((ValueError, TypeError)):
        PolySegScienceConfig(**{field: value})


def test_science_enabled_rejects_string_boolean_flag():
    config = OmegaConf.create({"algorithm": {"adv_estimator": "gae", "polyseg": {"science": {"enabled": "false"}}}})
    with pytest.raises(ValueError, match="enabled flag"):
        science_enabled(config)
