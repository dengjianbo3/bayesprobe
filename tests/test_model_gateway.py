import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import bayesprobe.model_gateway as model_gateway_module
from bayesprobe.openai_gateway import (
    OpenAIChatCompletionsModelGateway,
    OpenAIResponsesModelGateway,
)
from bayesprobe.recorded_gateway import RecordedModelGateway
from bayesprobe.model_gateway import (
    DeterministicModelGateway,
    EvidenceJudgmentRepairPolicy,
    ModelGatewayConfig,
    ModelGatewayValidationError,
    ModelInvocationTrace,
    ProviderRequestControls,
    ScriptedModelGateway,
    StructuredModelRequest,
    build_model_gateway,
    evidence_judgment_from_mapping,
    model_gateway_adapter_kind,
)
from bayesprobe.schemas import (
    EvidenceType,
    FrameFit,
    HypothesisCompetition,
    HypothesisCoverage,
    LikelihoodBand,
)


def make_request(
    raw_content: str,
    *,
    target_hypotheses: tuple[str, ...] = ("H1", "H2"),
    source_type: str = "benchmark_stream",
) -> StructuredModelRequest:
    return StructuredModelRequest(
        task="judge_evidence",
        input={
            "raw_content": raw_content,
            "target_hypotheses": list(target_hypotheses),
            "source_type": source_type,
        },
    )


def test_deterministic_gateway_judges_refuting_signal():
    response = DeterministicModelGateway().complete_structured(
        make_request("REFUTES: passage contradicts H1.")
    )

    judgment = evidence_judgment_from_mapping(response)

    assert judgment.evidence_type == EvidenceType.COUNTEREVIDENCE
    assert judgment.likelihoods["H1"] == LikelihoodBand.MODERATELY_DISCONFIRMING
    assert judgment.likelihoods["H2"] == LikelihoodBand.MODERATELY_CONFIRMING
    assert judgment.interpretation == "Deterministic v0.2 interpretation for benchmark_stream."


def test_deterministic_gateway_judges_supporting_signal():
    response = DeterministicModelGateway().complete_structured(
        make_request("SUPPORTS: passage supports H1.")
    )

    judgment = evidence_judgment_from_mapping(response)

    assert judgment.evidence_type == EvidenceType.SUPPORTING
    assert judgment.likelihoods["H1"] == LikelihoodBand.MODERATELY_CONFIRMING
    assert judgment.likelihoods["H2"] == LikelihoodBand.MODERATELY_DISCONFIRMING


def test_deterministic_gateway_supports_explicit_non_hypothesis_id():
    response = DeterministicModelGateway().complete_structured(
        make_request(
            "SUPPORTS D: The parity argument selects answer choice D.",
            target_hypotheses=("A", "B", "C", "D", "E"),
        )
    )

    judgment = evidence_judgment_from_mapping(response)

    assert judgment.likelihoods["D"] == LikelihoodBand.MODERATELY_CONFIRMING
    assert {
        judgment.likelihoods[hypothesis_id]
        for hypothesis_id in ("A", "B", "C", "E")
    } == {LikelihoodBand.MODERATELY_DISCONFIRMING}


def test_deterministic_gateway_judges_anomaly_for_all_targets():
    response = DeterministicModelGateway().complete_structured(
        make_request("ANOMALY: current hypotheses explain this badly.", target_hypotheses=("H1", "H2", "H3"))
    )

    judgment = evidence_judgment_from_mapping(response)

    assert judgment.evidence_type == EvidenceType.ANOMALY
    assert judgment.likelihoods == {
        "H1": LikelihoodBand.MODERATELY_DISCONFIRMING,
        "H2": LikelihoodBand.MODERATELY_DISCONFIRMING,
        "H3": LikelihoodBand.MODERATELY_DISCONFIRMING,
    }


def test_deterministic_gateway_maps_anomaly_to_unresolved_only_for_exclusive_open():
    open_request = make_request("ANOMALY: named candidates miss this result.")
    open_request.input["frame"] = {
        "competition": "exclusive",
        "coverage": "open",
        "frame_version": 2,
    }
    open_judgment = evidence_judgment_from_mapping(
        DeterministicModelGateway().complete_structured(open_request),
        competition=HypothesisCompetition.EXCLUSIVE,
        coverage=HypothesisCoverage.OPEN,
    )
    exhaustive_request = make_request("ANOMALY: this result is underdetermined.")
    exhaustive_request.input["frame"] = {
        "competition": "exclusive",
        "coverage": "exhaustive",
        "frame_version": 1,
    }
    exhaustive_judgment = evidence_judgment_from_mapping(
        DeterministicModelGateway().complete_structured(exhaustive_request),
        competition=HypothesisCompetition.EXCLUSIVE,
        coverage=HypothesisCoverage.EXHAUSTIVE,
    )

    assert open_judgment.frame_fit == FrameFit.SUPPORTS_UNRESOLVED
    assert open_judgment.unresolved_likelihood == LikelihoodBand.MODERATELY_CONFIRMING
    assert exhaustive_judgment.frame_fit == FrameFit.UNDERDETERMINED
    assert exhaustive_judgment.unresolved_likelihood is None


def test_deterministic_gateway_judges_neutral_signal():
    response = DeterministicModelGateway().complete_structured(
        make_request("This signal has no deterministic cue.")
    )

    judgment = evidence_judgment_from_mapping(response)

    assert judgment.evidence_type == EvidenceType.NEUTRAL
    assert judgment.likelihoods == {
        "H1": LikelihoodBand.NEUTRAL,
        "H2": LikelihoodBand.NEUTRAL,
    }


def test_scripted_gateway_records_requests_and_returns_response():
    gateway = ScriptedModelGateway(
        responses={
            "judge_evidence": {
                "evidence_type": "boundary_condition",
                "likelihoods": {"H1": "weakly_disconfirming", "H2": "neutral"},
                "interpretation": "Scripted boundary judgment.",
                "quality_overrides": {"reliability": 0.62},
            }
        }
    )
    request = make_request("No keyword cue.")

    judgment = evidence_judgment_from_mapping(gateway.complete_structured(request))

    assert gateway.requests == [request]
    assert judgment.evidence_type == EvidenceType.BOUNDARY_CONDITION
    assert judgment.likelihoods["H1"] == LikelihoodBand.WEAKLY_DISCONFIRMING
    assert judgment.likelihoods["H2"] == LikelihoodBand.NEUTRAL
    assert judgment.interpretation == "Scripted boundary judgment."
    assert judgment.quality_overrides == {"reliability": 0.62}


def test_structured_model_request_accepts_minimal_call():
    request = StructuredModelRequest(
        task="judge_evidence",
        input={"raw_content": "SUPPORTS: fixture"},
    )

    assert request.task == "judge_evidence"
    assert request.input == {"raw_content": "SUPPORTS: fixture"}
    assert request.prompt_id is None
    assert request.prompt_version is None
    assert request.schema_name is None
    assert request.schema_version is None
    assert request.metadata == {}


def test_structured_model_request_rejects_secret_input_without_echoing_it():
    secret = "sk-provider-secret-value-123"

    with pytest.raises(ValueError) as captured:
        StructuredModelRequest(
            task="judge_evidence",
            input={"Authorization": f"Bearer {secret}"},
        )

    assert "secret" in str(captured.value).lower()
    assert secret not in str(captured.value)


def test_structured_model_request_stores_metadata_and_is_frozen():
    request = StructuredModelRequest(
        task="judge_evidence",
        input={"raw_content": "SUPPORTS: fixture"},
        prompt_id="evidence_judgment",
        prompt_version="v0.1",
        schema_name="EvidenceJudgment",
        schema_version="v0.1",
        metadata={"run_id": "run_1"},
    )

    assert request.prompt_id == "evidence_judgment"
    assert request.prompt_version == "v0.1"
    assert request.schema_name == "EvidenceJudgment"
    assert request.schema_version == "v0.1"
    assert request.metadata == {"run_id": "run_1"}
    with pytest.raises(FrozenInstanceError):
        request.task = "other"


@pytest.mark.parametrize(
    ("kwargs", "expected_message"),
    [
        (
            {"task": 1, "input": {}},
            "structured model request task must be a string",
        ),
        (
            {"task": "", "input": {}},
            "structured model request task must not be empty",
        ),
        (
            {"task": "   ", "input": {}},
            "structured model request task must not be empty",
        ),
        (
            {"task": "judge_evidence", "input": []},
            "structured model request input must be an object",
        ),
        (
            {"task": "judge_evidence", "input": {}, "prompt_id": ""},
            "structured model request prompt_id must not be empty",
        ),
        (
            {"task": "judge_evidence", "input": {}, "prompt_version": " "},
            "structured model request prompt_version must not be empty",
        ),
        (
            {"task": "judge_evidence", "input": {}, "schema_name": "   "},
            "structured model request schema_name must not be empty",
        ),
        (
            {"task": "judge_evidence", "input": {}, "schema_name": 1},
            "structured model request schema_name must be a string",
        ),
        (
            {"task": "judge_evidence", "input": {}, "schema_version": ""},
            "structured model request schema_version must not be empty",
        ),
        (
            {"task": "judge_evidence", "input": {}, "metadata": []},
            "structured model request metadata must be an object",
        ),
    ],
)
def test_structured_model_request_rejects_invalid_metadata(kwargs, expected_message):
    with pytest.raises(ValueError, match=expected_message):
        StructuredModelRequest(**kwargs)


def test_model_invocation_trace_from_request_copies_prompt_schema_metadata():
    request = StructuredModelRequest(
        task="judge_evidence",
        input={"raw_content": "SUPPORTS: fixture"},
        prompt_id="evidence_judgment",
        prompt_version="v0.1",
        schema_name="EvidenceJudgment",
        schema_version="v0.1",
        metadata={"run_id": "run_1", "repair_attempt_index": 1},
    )

    trace = ModelInvocationTrace.from_request(request, adapter_kind="scripted")

    assert trace.task == "judge_evidence"
    assert trace.adapter_kind == "scripted"
    assert trace.prompt_id == "evidence_judgment"
    assert trace.prompt_version == "v0.1"
    assert trace.schema_name == "EvidenceJudgment"
    assert trace.schema_version == "v0.1"
    assert trace.repair_attempt_index == 1
    assert trace.metadata == {"run_id": "run_1"}
    assert trace.to_dict() == {
        "task": "judge_evidence",
        "adapter_kind": "scripted",
        "prompt_id": "evidence_judgment",
        "prompt_version": "v0.1",
        "schema_name": "EvidenceJudgment",
        "schema_version": "v0.1",
        "repair_attempt_index": 1,
        "metadata": {"run_id": "run_1"},
    }


@pytest.mark.parametrize(
    "repair_attempt_index",
    [0, -1, "1"],
)
def test_model_invocation_trace_rejects_invalid_repair_attempt_index(repair_attempt_index):
    request = StructuredModelRequest(
        task="repair_evidence_judgment",
        input={},
        metadata={"repair_attempt_index": repair_attempt_index},
    )

    with pytest.raises(
        ValueError,
        match="model invocation repair_attempt_index must be a positive integer",
    ):
        ModelInvocationTrace.from_request(request, adapter_kind="scripted")


@pytest.mark.parametrize(
    ("kwargs", "expected_message"),
    [
        ({"task": "", "adapter_kind": "scripted"}, "model invocation task must not be empty"),
        (
            {"task": "judge_evidence", "adapter_kind": ""},
            "model invocation adapter_kind must not be empty",
        ),
        (
            {"task": "judge_evidence", "adapter_kind": "scripted", "prompt_id": ""},
            "model invocation prompt_id must not be empty",
        ),
        (
            {"task": "judge_evidence", "adapter_kind": "scripted", "prompt_version": " "},
            "model invocation prompt_version must not be empty",
        ),
        (
            {"task": "judge_evidence", "adapter_kind": "scripted", "schema_name": ""},
            "model invocation schema_name must not be empty",
        ),
        (
            {"task": "judge_evidence", "adapter_kind": "scripted", "schema_version": "   "},
            "model invocation schema_version must not be empty",
        ),
        (
            {"task": "judge_evidence", "adapter_kind": "scripted", "metadata": []},
            "model invocation metadata must be an object",
        ),
    ],
)
def test_model_invocation_trace_rejects_invalid_fields(kwargs, expected_message):
    with pytest.raises(ValueError, match=expected_message):
        ModelInvocationTrace(**kwargs)


def test_model_gateway_adapter_kind_uses_stable_adapter_identities():
    class CustomGateway:
        def complete_structured(self, request):
            return {}

    assert DeterministicModelGateway.adapter_kind == "deterministic"
    assert ScriptedModelGateway(responses={}).adapter_kind == "scripted"
    assert model_gateway_adapter_kind(DeterministicModelGateway()) == "deterministic"
    assert model_gateway_adapter_kind(ScriptedModelGateway(responses={})) == "scripted"
    assert model_gateway_adapter_kind(CustomGateway()) == "CustomGateway"


def test_model_gateway_identity_distinguishes_models_with_the_same_adapter():
    first = build_model_gateway(
        {"kind": "openai_chat_completions", "model": "provider/model-a"}
    )
    second = build_model_gateway(
        {"kind": "openai_chat_completions", "model": "provider/model-b"}
    )

    assert model_gateway_adapter_kind(first) == model_gateway_adapter_kind(second)
    assert model_gateway_module.model_gateway_identity(first) == (
        "openai_model_identity:v1:"
        '{"adapter_kind":"openai_chat_completions",'
        '"model":"provider/model-a",'
        '"provider_origin":"https://api.openai.com"}'
    )
    assert model_gateway_module.model_gateway_identity(second) == (
        "openai_model_identity:v1:"
        '{"adapter_kind":"openai_chat_completions",'
        '"model":"provider/model-b",'
        '"provider_origin":"https://api.openai.com"}'
    )
    assert model_gateway_module.model_gateway_identity(first) != (
        model_gateway_module.model_gateway_identity(second)
    )


def test_model_gateway_identity_has_stable_custom_fallback():
    class CustomGateway:
        adapter_kind = "custom-safe-adapter"

        def complete_structured(self, request):
            return {}

    assert model_gateway_module.model_gateway_identity(
        DeterministicModelGateway()
    ) == "deterministic"
    assert model_gateway_module.model_gateway_identity(CustomGateway()) == (
        "custom-safe-adapter"
    )


def test_scripted_gateway_rejects_missing_task():
    gateway = ScriptedModelGateway(responses={})

    with pytest.raises(ValueError, match="no scripted response"):
        gateway.complete_structured(make_request("No response configured."))


def test_build_model_gateway_defaults_to_deterministic():
    gateway = build_model_gateway()

    judgment = evidence_judgment_from_mapping(
        gateway.complete_structured(make_request("SUPPORTS: evidence supports H1."))
    )

    assert isinstance(gateway, DeterministicModelGateway)
    assert judgment.evidence_type == EvidenceType.SUPPORTING
    assert judgment.likelihoods["H1"] == LikelihoodBand.MODERATELY_CONFIRMING


def test_build_model_gateway_accepts_deterministic_mapping():
    gateway = build_model_gateway({"kind": "deterministic"})

    judgment = evidence_judgment_from_mapping(
        gateway.complete_structured(make_request("This signal has no deterministic cue."))
    )

    assert isinstance(gateway, DeterministicModelGateway)
    assert judgment.evidence_type == EvidenceType.NEUTRAL


def test_build_model_gateway_accepts_scripted_config_and_records_requests():
    gateway = build_model_gateway(
        ModelGatewayConfig(
            kind="scripted",
            responses={
                "judge_evidence": {
                    "evidence_type": "boundary_condition",
                    "likelihoods": {"H1": "weakly_disconfirming", "H2": "neutral"},
                    "interpretation": "Configured scripted judgment.",
                    "quality_overrides": {"reliability": 0.62},
                }
            },
        )
    )

    request = make_request("No keyword cue.")
    judgment = evidence_judgment_from_mapping(gateway.complete_structured(request))

    assert isinstance(gateway, ScriptedModelGateway)
    assert gateway.requests == [request]
    assert judgment.evidence_type == EvidenceType.BOUNDARY_CONDITION
    assert judgment.likelihoods["H1"] == LikelihoodBand.WEAKLY_DISCONFIRMING
    assert judgment.quality_overrides == {"reliability": 0.62}


def test_build_model_gateway_accepts_openai_mapping_without_network():
    gateway = build_model_gateway(
        {
            "kind": "openai",
            "model": "gpt-5.5",
            "api_key_env": "BAYESPROBE_TEST_OPENAI_KEY",
            "timeout_seconds": 12.5,
            "max_output_tokens": 256,
        }
    )

    assert isinstance(gateway, OpenAIResponsesModelGateway)
    assert gateway.config.model == "gpt-5.5"
    assert gateway.config.api_key_env == "BAYESPROBE_TEST_OPENAI_KEY"
    assert gateway.config.timeout_seconds == 12.5
    assert gateway.config.max_output_tokens == 256


def test_build_model_gateway_creates_openai_chat_completions_gateway():
    gateway = build_model_gateway(
        {
            "kind": "openai_chat_completions",
            "model": "provider-model",
            "api_key_env": "PROVIDER_API_KEY",
            "base_url": "https://provider.example/v1",
            "temperature": 0,
            "top_p": 1,
            "thinking": "enabled",
            "reasoning_effort": "max",
        }
    )

    assert isinstance(gateway, OpenAIChatCompletionsModelGateway)
    assert gateway.config.model == "provider-model"
    assert gateway.config.api_key_env == "PROVIDER_API_KEY"
    assert gateway.config.base_url == "https://provider.example/v1"
    assert gateway.config.request_controls == ProviderRequestControls(
        temperature=0,
        top_p=1,
        thinking="enabled",
        reasoning_effort="max",
    )


def test_build_model_gateway_passes_invocation_observer_to_openai_adapter():
    observer = object()

    gateway = build_model_gateway(
        {
            "kind": "openai_chat_completions",
            "model": "provider-model",
        },
        invocation_observer=observer,
    )

    assert gateway.invocation_observer is observer


def test_build_model_gateway_creates_recorded_gateway(tmp_path: Path):
    fixture_path = tmp_path / "recorded.json"
    fixture_path.write_text(
        json.dumps(
            {
                "fixture_name": "recorded_factory",
                "responses": [
                    {
                        "match": {"task": "judge_evidence", "signal_id": "S1"},
                        "response": {
                            "evidence_type": "supporting",
                            "likelihoods": {"H1": "moderately_confirming"},
                            "interpretation": "Recorded factory response.",
                            "quality_overrides": {},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    gateway = build_model_gateway(
        {"kind": "recorded", "fixture_path": str(fixture_path)}
    )

    assert isinstance(gateway, RecordedModelGateway)
    assert gateway.fixture_name == "recorded_factory"


def test_build_model_gateway_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unsupported model gateway kind"):
        build_model_gateway({"kind": "unknown"})


def test_build_model_gateway_rejects_scripted_without_responses():
    with pytest.raises(ValueError, match="scripted model gateway requires responses"):
        build_model_gateway({"kind": "scripted"})


def test_build_model_gateway_rejects_non_object_responses():
    with pytest.raises(ValueError, match="model gateway responses must be an object"):
        build_model_gateway({"kind": "scripted", "responses": []})


def test_build_model_gateway_rejects_openai_without_model():
    with pytest.raises(ValueError, match="openai model gateway requires model"):
        build_model_gateway({"kind": "openai"})


def test_build_model_gateway_rejects_invalid_openai_timeout():
    with pytest.raises(
        ValueError, match="openai model gateway timeout_seconds must be positive"
    ):
        build_model_gateway({"kind": "openai", "model": "gpt-5.5", "timeout_seconds": 0})


def test_evidence_judgment_repair_policy_defaults_to_disabled():
    policy = EvidenceJudgmentRepairPolicy()

    assert policy.max_attempts == 0
    assert policy.repair_task == "repair_evidence_judgment"


def test_evidence_judgment_repair_policy_from_config_accepts_mapping():
    policy = EvidenceJudgmentRepairPolicy.from_config(
        {"max_attempts": 2, "repair_task": "repair_evidence_judgment"}
    )

    assert policy.max_attempts == 2
    assert policy.repair_task == "repair_evidence_judgment"


def test_evidence_judgment_repair_policy_from_config_accepts_existing_policy():
    existing = EvidenceJudgmentRepairPolicy(max_attempts=1)

    assert EvidenceJudgmentRepairPolicy.from_config(existing) is existing


@pytest.mark.parametrize(
    ("config", "expected_message"),
    [
        ([], "judgment repair policy config must be an object"),
        ({"max_attempts": "1"}, "judgment repair max_attempts must be an integer"),
        ({"max_attempts": -1}, "judgment repair max_attempts must be non-negative"),
        ({"repair_task": 1}, "judgment repair task must be a string"),
        ({"repair_task": ""}, "judgment repair task must not be empty"),
        ({"repair_task": "   "}, "judgment repair task must not be empty"),
    ],
)
def test_evidence_judgment_repair_policy_rejects_invalid_config(config, expected_message):
    with pytest.raises(ValueError, match=expected_message):
        EvidenceJudgmentRepairPolicy.from_config(config)


@pytest.mark.parametrize(
    ("payload", "expected_message"),
    [
        ({}, "evidence judgment missing field: evidence_type"),
        ({"evidence_type": "not_a_type"}, "invalid evidence_type"),
        (
            {"evidence_type": "neutral", "likelihoods": []},
            "evidence judgment likelihoods must be an object",
        ),
        (
            {"evidence_type": "neutral", "likelihoods": {"H1": "not_a_band"}},
            "invalid likelihood band for H1",
        ),
        (
            {"evidence_type": "neutral", "quality_overrides": []},
            "evidence judgment quality_overrides must be an object",
        ),
        (
            {"evidence_type": "neutral", "quality_overrides": {"reliability": "high"}},
            "invalid quality override for reliability",
        ),
        (
            {"evidence_type": "neutral", "quality_overrides": {"authority": 0.5}},
            "unsupported quality override metric: authority",
        ),
        (
            {"evidence_type": "neutral", "quality_overrides": {"reliability": -0.1}},
            "quality override reliability must be finite and between 0 and 1",
        ),
        (
            {"evidence_type": "neutral", "quality_overrides": {"reliability": 1.1}},
            "quality override reliability must be finite and between 0 and 1",
        ),
        (
            {"evidence_type": "neutral", "quality_overrides": {"reliability": float("inf")}},
            "quality override reliability must be finite and between 0 and 1",
        ),
        (
            {"evidence_type": "neutral", "quality_overrides": {"reliability": float("nan")}},
            "quality override reliability must be finite and between 0 and 1",
        ),
    ],
)
def test_evidence_judgment_from_mapping_raises_validation_error(payload, expected_message):
    with pytest.raises(ModelGatewayValidationError, match=expected_message):
        evidence_judgment_from_mapping(payload)


def test_native_evidence_judgment_requires_all_v02_fields():
    with pytest.raises(ModelGatewayValidationError, match="missing field: frame_fit"):
        evidence_judgment_from_mapping(
            {
                "evidence_type": "neutral",
                "likelihoods": {"H1": "neutral"},
                "unresolved_likelihood": None,
                "unexplained_observation": None,
                "interpretation": "Incomplete native judgment.",
                "quality_overrides": {},
            },
            competition=HypothesisCompetition.INDEPENDENT,
            coverage=HypothesisCoverage.OPEN,
        )


@pytest.mark.parametrize(
    ("interpretation", "quality_overrides", "message"),
    [
        (True, {}, "interpretation must be a non-empty string"),
        ("   ", {}, "interpretation must be a non-empty string"),
        ("Valid interpretation.", {"reliability": True}, "real finite number"),
        ("Valid interpretation.", {"reliability": "0.5"}, "real finite number"),
    ],
)
def test_native_evidence_judgment_rejects_coerced_text_and_quality_values(
    interpretation,
    quality_overrides,
    message,
):
    with pytest.raises(ModelGatewayValidationError, match=message):
        evidence_judgment_from_mapping(
            {
                "evidence_type": "neutral",
                "likelihoods": {"H1": "neutral"},
                "unresolved_likelihood": None,
                "frame_fit": "underdetermined",
                "unexplained_observation": None,
                "interpretation": interpretation,
                "quality_overrides": quality_overrides,
            },
            competition=HypothesisCompetition.INDEPENDENT,
            coverage=HypothesisCoverage.OPEN,
        )


@pytest.mark.parametrize(
    ("frame_fit", "unresolved_likelihood", "message"),
    [
        ("supports_unresolved", "neutral", "requires a confirming"),
        ("explained_by_named", "weakly_confirming", "cannot confirm unresolved"),
        ("underdetermined", "weakly_confirming", "requires neutral"),
    ],
)
def test_native_evidence_judgment_rejects_incoherent_open_frame_fields(
    frame_fit,
    unresolved_likelihood,
    message,
):
    with pytest.raises(ModelGatewayValidationError, match=message):
        evidence_judgment_from_mapping(
            {
                "evidence_type": "anomaly",
                "likelihoods": {"H1": "moderately_disconfirming"},
                "unresolved_likelihood": unresolved_likelihood,
                "frame_fit": frame_fit,
                "unexplained_observation": "The named claim misses the result.",
                "interpretation": "Native coherence fixture.",
                "quality_overrides": {},
            },
            competition=HypothesisCompetition.EXCLUSIVE,
            coverage=HypothesisCoverage.OPEN,
        )


def test_native_evidence_judgment_requires_null_unresolved_outside_exclusive_open():
    with pytest.raises(ModelGatewayValidationError, match="must be null"):
        evidence_judgment_from_mapping(
            {
                "evidence_type": "supporting",
                "likelihoods": {"H1": "weakly_confirming"},
                "unresolved_likelihood": "neutral",
                "frame_fit": "underdetermined",
                "unexplained_observation": None,
                "interpretation": "Independent frames have no shared reserve.",
                "quality_overrides": {},
            },
            competition=HypothesisCompetition.INDEPENDENT,
            coverage=HypothesisCoverage.OPEN,
        )


def test_evidence_judgment_rejects_secret_provider_payload_without_echoing_it():
    secret = "sk-provider-secret-value-123"

    with pytest.raises(ModelGatewayValidationError) as captured:
        evidence_judgment_from_mapping(
            {
                "evidence_type": "neutral",
                "likelihoods": {"H1": "neutral"},
                "unresolved_likelihood": None,
                "frame_fit": "underdetermined",
                "unexplained_observation": None,
                "interpretation": f"Authorization: Bearer {secret}",
                "quality_overrides": {},
            },
            competition=HypothesisCompetition.INDEPENDENT,
            coverage=HypothesisCoverage.OPEN,
        )

    assert "secret" in str(captured.value).lower()
    assert secret not in str(captured.value)
