import pytest

from bayesprobe.model_gateway import (
    DeterministicModelGateway,
    EvidenceJudgmentRepairPolicy,
    ModelGatewayConfig,
    ModelGatewayValidationError,
    ScriptedModelGateway,
    StructuredModelRequest,
    build_model_gateway,
    evidence_judgment_from_mapping,
)
from bayesprobe.schemas import EvidenceType, LikelihoodBand


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


def test_build_model_gateway_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unsupported model gateway kind"):
        build_model_gateway({"kind": "unknown"})


def test_build_model_gateway_rejects_scripted_without_responses():
    with pytest.raises(ValueError, match="scripted model gateway requires responses"):
        build_model_gateway({"kind": "scripted"})


def test_build_model_gateway_rejects_non_object_responses():
    with pytest.raises(ValueError, match="model gateway responses must be an object"):
        build_model_gateway({"kind": "scripted", "responses": []})


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
    ],
)
def test_evidence_judgment_from_mapping_raises_validation_error(payload, expected_message):
    with pytest.raises(ModelGatewayValidationError, match=expected_message):
        evidence_judgment_from_mapping(payload)
