import json
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from bayesprobe.model_gateway import ModelGatewayValidationError, StructuredModelRequest
from bayesprobe.openai_gateway import (
    EVIDENCE_JUDGMENT_JSON_SCHEMA,
    OpenAIModelGatewayConfig,
    OpenAIResponsesModelGateway,
    build_openai_request_payload,
    parse_openai_structured_response,
)


def make_judge_request() -> StructuredModelRequest:
    return StructuredModelRequest(
        task="judge_evidence",
        input={
            "signal_id": "S1",
            "source_type": "benchmark_stream",
            "source": "fixture",
            "raw_content": "SUPPORTS: source supports H1.",
            "target_hypotheses": ["H1", "H2"],
        },
        prompt_id="evidence_judgment",
        prompt_version="v0.1",
        schema_name="EvidenceJudgment",
        schema_version="v0.1",
        metadata={"run_id": "run_1"},
    )


def test_openai_model_gateway_config_requires_explicit_model():
    config = OpenAIModelGatewayConfig(model="gpt-5.5")

    assert config.model == "gpt-5.5"
    assert config.api_key_env == "OPENAI_API_KEY"
    assert config.timeout_seconds == 30.0
    assert config.max_output_tokens is None
    with pytest.raises(FrozenInstanceError):
        config.model = "other"


@pytest.mark.parametrize(
    ("kwargs", "expected_message"),
    [
        ({"model": ""}, "openai model gateway model must not be empty"),
        ({"model": "   "}, "openai model gateway model must not be empty"),
        ({"model": 1}, "openai model gateway model must be a string"),
        (
            {"model": "gpt-5.5", "api_key_env": ""},
            "openai model gateway api_key_env must not be empty",
        ),
        (
            {"model": "gpt-5.5", "api_key_env": 1},
            "openai model gateway api_key_env must be a string",
        ),
        (
            {"model": "gpt-5.5", "timeout_seconds": 0},
            "openai model gateway timeout_seconds must be positive",
        ),
        (
            {"model": "gpt-5.5", "timeout_seconds": "30"},
            "openai model gateway timeout_seconds must be a number",
        ),
        (
            {"model": "gpt-5.5", "max_output_tokens": 0},
            "openai model gateway max_output_tokens must be positive",
        ),
        (
            {"model": "gpt-5.5", "max_output_tokens": "128"},
            "openai model gateway max_output_tokens must be an integer",
        ),
    ],
)
def test_openai_model_gateway_config_rejects_invalid_values(kwargs, expected_message):
    with pytest.raises(ValueError, match=expected_message):
        OpenAIModelGatewayConfig(**kwargs)


def test_build_openai_request_payload_for_judge_evidence():
    payload = build_openai_request_payload(
        make_judge_request(),
        model="gpt-5.5",
        max_output_tokens=256,
    )

    assert payload["model"] == "gpt-5.5"
    assert payload["max_output_tokens"] == 256
    assert payload["metadata"] == {
        "provider": "openai",
        "model": "gpt-5.5",
        "task": "judge_evidence",
        "prompt_id": "evidence_judgment",
        "prompt_version": "v0.1",
        "schema_name": "EvidenceJudgment",
        "schema_version": "v0.1",
        "run_id": "run_1",
    }
    assert payload["text"]["format"]["type"] == "json_schema"
    assert payload["text"]["format"]["name"] == "EvidenceJudgment"
    assert payload["text"]["format"]["strict"] is True
    assert payload["text"]["format"]["schema"] == EVIDENCE_JUDGMENT_JSON_SCHEMA
    assert payload["input"][0]["role"] == "developer"
    assert "BayesProbe" in payload["input"][0]["content"]
    assert payload["input"][1]["role"] == "user"
    user_payload = json.loads(payload["input"][1]["content"])
    assert user_payload["task"] == "judge_evidence"
    assert user_payload["input"]["signal_id"] == "S1"


def test_build_openai_request_payload_for_repair_evidence_judgment():
    request = StructuredModelRequest(
        task="repair_evidence_judgment",
        input={
            "original_request": {"task": "judge_evidence", "input": {"signal_id": "S1"}},
            "invalid_payload": {"evidence_type": "not_a_type"},
            "validation_error": "invalid evidence_type: not_a_type",
            "attempt_index": 2,
        },
        prompt_id="evidence_judgment_repair",
        prompt_version="v0.1",
        schema_name="EvidenceJudgment",
        schema_version="v0.1",
        metadata={"repair_attempt_index": 2},
    )

    payload = build_openai_request_payload(request, model="gpt-5.5")

    assert "max_output_tokens" not in payload
    assert payload["metadata"]["task"] == "repair_evidence_judgment"
    assert payload["metadata"]["prompt_id"] == "evidence_judgment_repair"
    assert payload["metadata"]["repair_attempt_index"] == "2"
    assert "Repair" in payload["input"][0]["content"]
    assert json.loads(payload["input"][1]["content"])["input"]["attempt_index"] == 2


def test_build_openai_request_payload_rejects_unknown_task():
    request = StructuredModelRequest(task="other_task", input={})

    with pytest.raises(ValueError, match="unsupported openai model task: other_task"):
        build_openai_request_payload(request, model="gpt-5.5")


def valid_payload() -> dict[str, object]:
    return {
        "evidence_type": "supporting",
        "likelihoods": {"H1": "moderately_confirming", "H2": "neutral"},
        "interpretation": "OpenAI fixture judgment.",
        "quality_overrides": {},
    }


def test_parse_openai_structured_response_accepts_direct_dict():
    assert parse_openai_structured_response(valid_payload()) == valid_payload()


def test_parse_openai_structured_response_accepts_json_string():
    assert parse_openai_structured_response(json.dumps(valid_payload())) == valid_payload()


def test_parse_openai_structured_response_accepts_output_text_object():
    response = SimpleNamespace(output_text=json.dumps(valid_payload()))

    assert parse_openai_structured_response(response) == valid_payload()


def test_parse_openai_structured_response_accepts_mapping_output_text():
    response = {"output_text": json.dumps(valid_payload())}

    assert parse_openai_structured_response(response) == valid_payload()


@pytest.mark.parametrize(
    ("response", "expected_message"),
    [
        ("{", "openai structured response was not valid JSON"),
        ("[]", "openai structured response must be an object"),
        (["not", "valid"], "openai structured response must be an object"),
        (SimpleNamespace(output_text=None), "openai structured response text was missing"),
    ],
)
def test_parse_openai_structured_response_rejects_malformed_output(
    response,
    expected_message,
):
    with pytest.raises(ModelGatewayValidationError, match=expected_message):
        parse_openai_structured_response(response)


class FakeResponses:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **payload):
        self.calls.append(payload)
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


class FakeOpenAIClient:
    def __init__(self, response):
        self.responses = FakeResponses(response)


def test_openai_responses_model_gateway_calls_fake_client_and_returns_dict():
    client = FakeOpenAIClient(response=json.dumps(valid_payload()))
    gateway = OpenAIResponsesModelGateway(
        config=OpenAIModelGatewayConfig(model="gpt-5.5", max_output_tokens=128),
        client=client,
    )

    result = gateway.complete_structured(make_judge_request())

    assert result == valid_payload()
    assert len(client.responses.calls) == 1
    assert client.responses.calls[0]["model"] == "gpt-5.5"
    assert client.responses.calls[0]["max_output_tokens"] == 128


def test_openai_responses_model_gateway_propagates_provider_exceptions():
    error = RuntimeError("provider outage")
    gateway = OpenAIResponsesModelGateway(
        config=OpenAIModelGatewayConfig(model="gpt-5.5"),
        client=FakeOpenAIClient(response=error),
    )

    with pytest.raises(RuntimeError, match="provider outage"):
        gateway.complete_structured(make_judge_request())


def test_openai_responses_model_gateway_raises_clear_error_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    gateway = OpenAIResponsesModelGateway(
        config=OpenAIModelGatewayConfig(model="gpt-5.5"),
    )

    with pytest.raises(
        RuntimeError,
        match="OpenAI API key environment variable OPENAI_API_KEY is not set",
    ):
        gateway.complete_structured(make_judge_request())
