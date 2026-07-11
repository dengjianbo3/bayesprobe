import json
import io
import urllib.error
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from bayesprobe.model_gateway import (
    ModelGatewayValidationError,
    ProviderRequestControls,
    StructuredModelRequest,
)
from bayesprobe.openai_gateway import (
    EVIDENCE_JUDGMENT_JSON_SCHEMA,
    OPEN_QUESTION_TASK_FRAME_JSON_SCHEMA,
    PROBE_SIGNAL_JSON_SCHEMA,
    TASK_ADMISSION_DECISION_JSON_SCHEMA,
    OpenAIChatCompletionsModelGateway,
    OpenAIModelGatewayConfig,
    OpenAIResponsesModelGateway,
    build_openai_chat_completions_payload,
    build_openai_request_payload,
    parse_openai_chat_completions_response,
    parse_openai_structured_response,
)
from bayesprobe.provider_telemetry import ProviderInvocationRecord


OPEN_QUESTION_TASK_FRAME_REQUIRED_KEYS = [
    "task_kind",
    "answer_relationship",
    "answer_contract",
    "competition",
    "coverage",
    "hypotheses",
    "coverage_statement",
    "coverage_limitation",
]
FORBIDDEN_OPEN_QUESTION_TASK_FRAME_FIELDS = [
    "id",
    "prior",
    "posterior",
    "api_key",
    "credential",
    "secret",
]


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


def make_probe_request() -> StructuredModelRequest:
    return StructuredModelRequest(
        task="execute_probe",
        input={
            "problem": "Which answer choice is correct?",
            "initial_context": "Use the supplied theorem definitions.",
            "probe": {
                "id": "P1",
                "inquiry_goal": "Discriminate all answer choices.",
                "method": "answer_choice_discrimination",
                "target_hypotheses": ["A", "B"],
            },
            "hypotheses": [
                {"id": "A", "statement": "Choice A is correct.", "posterior": 0.5},
                {"id": "B", "statement": "Choice B is correct.", "posterior": 0.5},
            ],
        },
        prompt_id="probe_execution",
        prompt_version="v0.1",
        schema_name="ProbeSignal",
        schema_version="v0.1",
    )


def make_multiple_choice_request(
    *, task: str = "answer_multiple_choice"
) -> StructuredModelRequest:
    return StructuredModelRequest(
        task=task,
        input={
            "question": "What is 2 + 2?",
            "choices": {"A": "3", "B": "4", "C": "5"},
        },
        prompt_id="direct_multiple_choice",
        prompt_version="v0.1",
        schema_name="MultipleChoiceAnswer",
        schema_version="v0.1",
    )


def make_open_question_frame_request() -> StructuredModelRequest:
    return StructuredModelRequest(
        task="frame_open_question",
        input={
            "question": "How should this claim be tested?",
            "task_context": "Use a frozen task distribution.",
            "supported_task_kinds": ["claim_verification", "design"],
            "supported_competition": ["exclusive", "independent"],
            "supported_coverage": ["exhaustive", "open"],
            "hypothesis_count": {"minimum": 1, "maximum": 6},
        },
        prompt_id="open_question_task_framing",
        prompt_version="v0.2",
        schema_name="OpenQuestionTaskFrame",
        schema_version="v0.2",
    )


def make_repair_task_frame_request() -> StructuredModelRequest:
    return StructuredModelRequest(
        task="repair_task_frame",
        input={
            "original_request": make_open_question_frame_request().input,
            "invalid_payload": {"hypotheses": []},
            "validation_error": "at least two hypotheses are required",
            "attempt_index": 1,
        },
        prompt_id="open_question_task_framing_repair",
        prompt_version="v0.2",
        schema_name="OpenQuestionTaskFrame",
        schema_version="v0.2",
        metadata={"repair_attempt_index": 1},
    )


def make_task_admission_request(
    *, task: str = "assess_task_admission"
) -> StructuredModelRequest:
    return StructuredModelRequest(
        task=task,
        input={
            "question": "What integer satisfies the constraints?",
            "task_context": "Use the supplied theorem.",
            "requested_output_shape": "integer with basis",
            "available_capabilities": [],
        },
        prompt_id="task_admission",
        prompt_version="v0.2",
        schema_name="TaskAdmissionDecision",
        schema_version="v0.2",
    )


@pytest.mark.parametrize(
    "model_request",
    [
        make_task_admission_request(),
        make_task_admission_request(task="repair_task_admission"),
    ],
)
def test_responses_task_admission_uses_strict_decision_schema(model_request):
    payload = build_openai_request_payload(model_request, model="test-model")

    assert payload["text"]["format"] == {
        "type": "json_schema",
        "name": "TaskAdmissionDecision",
        "strict": True,
        "schema": TASK_ADMISSION_DECISION_JSON_SCHEMA,
    }
    assert "task admission" in payload["input"][0]["content"].lower()


@pytest.mark.parametrize(
    "model_request",
    [
        make_task_admission_request(),
        make_task_admission_request(task="repair_task_admission"),
    ],
)
def test_chat_task_admission_includes_exact_required_output(model_request):
    payload = build_openai_chat_completions_payload(model_request, model="test-model")
    required_output = json.loads(payload["messages"][1]["content"])[
        "required_output"
    ]

    assert required_output["type"] == "TaskAdmissionDecision"
    assert required_output["json_schema"] == TASK_ADMISSION_DECISION_JSON_SCHEMA
    assert required_output["required_keys"] == [
        "status",
        "epistemic_basis",
        "proposed_task_kind",
        "answer_contract_outline",
        "clarification_questions",
        "reason",
    ]


def assert_open_question_task_frame_schema(schema: dict) -> None:
    assert schema is OPEN_QUESTION_TASK_FRAME_JSON_SCHEMA
    assert schema["required"] == OPEN_QUESTION_TASK_FRAME_REQUIRED_KEYS
    assert list(schema["properties"]) == OPEN_QUESTION_TASK_FRAME_REQUIRED_KEYS
    serialized_schema = json.dumps(schema).lower()
    for field in FORBIDDEN_OPEN_QUESTION_TASK_FRAME_FIELDS:
        assert f'"{field}"' not in serialized_schema


def assert_open_question_task_frame_required_output(required_output: dict) -> None:
    assert required_output["type"] == "OpenQuestionTaskFrame"
    assert required_output["required_keys"] == OPEN_QUESTION_TASK_FRAME_REQUIRED_KEYS
    assert required_output["json_schema"] == OPEN_QUESTION_TASK_FRAME_JSON_SCHEMA
    serialized_required_output = json.dumps(required_output).lower()
    for field in FORBIDDEN_OPEN_QUESTION_TASK_FRAME_FIELDS:
        assert f'"{field}"' not in serialized_required_output


def test_build_openai_payload_for_open_question_frame():
    payload = build_openai_request_payload(
        make_open_question_frame_request(),
        model="test-model",
    )

    assert payload["text"]["format"]["name"] == "OpenQuestionTaskFrame"
    schema = payload["text"]["format"]["schema"]
    assert_open_question_task_frame_schema(schema)
    assert schema["properties"]["competition"]["enum"] == [
        "exclusive",
        "independent",
    ]
    assert json.loads(payload["input"][1]["content"])["task"] == "frame_open_question"


def test_build_openai_payload_for_task_frame_repair():
    payload = build_openai_request_payload(
        make_repair_task_frame_request(),
        model="test-model",
    )

    assert payload["text"]["format"]["name"] == "OpenQuestionTaskFrame"
    assert_open_question_task_frame_schema(payload["text"]["format"]["schema"])
    assert json.loads(payload["input"][1]["content"])["task"] == "repair_task_frame"


def test_build_chat_payload_for_open_question_frame():
    payload = build_openai_chat_completions_payload(
        make_open_question_frame_request(),
        model="test-model",
    )

    required_output = json.loads(payload["messages"][1]["content"])["required_output"]
    assert_open_question_task_frame_required_output(required_output)
    assert json.loads(payload["messages"][1]["content"])["task"] == "frame_open_question"
    assert payload["response_format"] == {"type": "json_object"}


def test_build_chat_payload_for_task_frame_repair():
    payload = build_openai_chat_completions_payload(
        make_repair_task_frame_request(),
        model="test-model",
    )

    required_output = json.loads(payload["messages"][1]["content"])["required_output"]
    assert_open_question_task_frame_required_output(required_output)
    assert json.loads(payload["messages"][1]["content"])["task"] == "repair_task_frame"
    assert payload["response_format"] == {"type": "json_object"}


def test_openai_model_gateway_config_requires_explicit_model():
    config = OpenAIModelGatewayConfig(model="gpt-5.5")

    assert config.model == "gpt-5.5"
    assert config.api_key_env == "OPENAI_API_KEY"
    assert config.timeout_seconds == 30.0
    assert config.max_output_tokens is None
    with pytest.raises(FrozenInstanceError):
        config.model = "other"


def test_openai_model_gateway_config_accepts_base_url():
    config = OpenAIModelGatewayConfig(
        model="gpt-5.5",
        base_url="https://provider.example/v1",
    )

    assert config.base_url == "https://provider.example/v1"


@pytest.mark.parametrize(
    ("base_url", "expected_message"),
    [
        ("", "openai model gateway base_url must not be empty"),
        ("   ", "openai model gateway base_url must not be empty"),
        (1, "openai model gateway base_url must be a string"),
    ],
)
def test_openai_model_gateway_config_rejects_invalid_base_url(
    base_url,
    expected_message,
):
    with pytest.raises(ValueError, match=expected_message):
        OpenAIModelGatewayConfig(model="gpt-5.5", base_url=base_url)


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
            {"model": "gpt-5.5", "api_key_env": "not-an-env-var"},
            "openai model gateway api_key_env must be an environment variable name",
        ),
        (
            {"model": "gpt-5.5", "api_key_env": "openai_api_key"},
            "openai model gateway api_key_env must be an environment variable name",
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
            {"model": "gpt-5.5", "timeout_seconds": float("nan")},
            "openai model gateway timeout_seconds must be finite and positive",
        ),
        (
            {"model": "gpt-5.5", "timeout_seconds": float("inf")},
            "openai model gateway timeout_seconds must be finite and positive",
        ),
        (
            {"model": "gpt-5.5", "timeout_seconds": float("-inf")},
            "openai model gateway timeout_seconds must be finite and positive",
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


def test_build_openai_request_payload_for_execute_probe():
    payload = build_openai_request_payload(
        make_probe_request(),
        model="gpt-5.5",
        max_output_tokens=512,
    )

    assert payload["text"]["format"] == {
        "type": "json_schema",
        "name": "ProbeSignal",
        "strict": True,
        "schema": PROBE_SIGNAL_JSON_SCHEMA,
    }
    assert "active probe executor" in payload["input"][0]["content"]
    user_payload = json.loads(payload["input"][1]["content"])
    assert user_payload["task"] == "execute_probe"
    assert user_payload["input"]["problem"] == "Which answer choice is correct?"


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


def test_build_openai_chat_completions_payload_uses_common_json_object_shape():
    payload = build_openai_chat_completions_payload(
        make_judge_request(),
        model="provider-model",
        max_output_tokens=256,
    )

    assert payload["model"] == "provider-model"
    assert payload["stream"] is False
    assert payload["max_tokens"] == 256
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["messages"][0]["role"] == "system"
    assert "evidence judgment component" in payload["messages"][0]["content"]
    assert "Return only one JSON object" in payload["messages"][0]["content"]
    assert "Do not copy input fields" in payload["messages"][0]["content"]
    assert payload["messages"][1]["role"] == "user"
    user_payload = json.loads(payload["messages"][1]["content"])
    assert user_payload["task"] == "judge_evidence"
    assert user_payload["required_output"]["json_schema"] == EVIDENCE_JUDGMENT_JSON_SCHEMA
    assert user_payload["required_output"]["required_keys"] == [
        "evidence_type",
        "likelihoods",
        "interpretation",
        "quality_overrides",
    ]


def test_build_openai_chat_completions_payload_includes_explicit_request_controls():
    payload = build_openai_chat_completions_payload(
        make_judge_request(),
        model="deepseek-v4-flash",
        controls=ProviderRequestControls(
            temperature=0,
            top_p=1,
            thinking="enabled",
            reasoning_effort="max",
        ),
    )

    assert payload["temperature"] == 0
    assert payload["top_p"] == 1
    assert payload["thinking"] == {"type": "enabled"}
    assert payload["reasoning_effort"] == "max"


def test_build_openai_chat_completions_payload_omits_unset_request_controls():
    payload = build_openai_chat_completions_payload(
        make_judge_request(),
        model="provider-model",
        controls=ProviderRequestControls(),
    )

    assert "temperature" not in payload
    assert "top_p" not in payload
    assert "thinking" not in payload
    assert "reasoning_effort" not in payload


@pytest.mark.parametrize(
    ("kwargs", "expected_message"),
    [
        ({"temperature": -0.1}, "temperature must be finite and non-negative"),
        ({"temperature": float("nan")}, "temperature must be finite and non-negative"),
        ({"top_p": 0}, "top_p must be finite and in the interval"),
        ({"top_p": 1.1}, "top_p must be finite and in the interval"),
        ({"thinking": ""}, "thinking must not be empty"),
        ({"reasoning_effort": "   "}, "reasoning_effort must not be empty"),
    ],
)
def test_provider_request_controls_reject_invalid_values(kwargs, expected_message):
    with pytest.raises(ValueError, match=expected_message):
        ProviderRequestControls(**kwargs)


def test_build_openai_chat_completions_payload_for_execute_probe():
    payload = build_openai_chat_completions_payload(
        make_probe_request(),
        model="provider-model",
        max_output_tokens=512,
    )

    assert "active probe executor" in payload["messages"][0]["content"]
    assert "raw_content" in payload["messages"][0]["content"]
    user_payload = json.loads(payload["messages"][1]["content"])
    assert user_payload["required_output"] == {
        "type": "ProbeSignal",
        "required_keys": ["raw_content"],
        "json_schema": PROBE_SIGNAL_JSON_SCHEMA,
        "notes": [
            "raw_content must report the inquiry result without posterior updates",
            "do not claim external retrieval or verification unless supplied in the input",
        ],
    }


def test_build_openai_chat_payload_for_multiple_choice_answer():
    payload = build_openai_chat_completions_payload(
        make_multiple_choice_request(),
        model="provider-model",
    )

    assert "multiple-choice" in payload["messages"][0]["content"]
    user_payload = json.loads(payload["messages"][1]["content"])
    assert user_payload["input"] == make_multiple_choice_request().input
    assert user_payload["required_output"]["required_keys"] == [
        "answer_label",
        "choice_probabilities",
        "answer_summary",
    ]


def test_build_openai_payload_for_multiple_choice_repair_uses_same_schema():
    request = make_multiple_choice_request(task="repair_multiple_choice_answer")

    responses_payload = build_openai_request_payload(request, model="gpt-5.5")
    chat_payload = build_openai_chat_completions_payload(
        request,
        model="provider-model",
    )

    assert responses_payload["text"]["format"]["name"] == "MultipleChoiceAnswer"
    assert "Repair" in responses_payload["input"][0]["content"]
    assert "Repair" in chat_payload["messages"][0]["content"]


def test_build_openai_chat_payload_for_python_probe_plan():
    request = StructuredModelRequest(
        task="plan_python_probe",
        input={"probe": {"id": "P1"}, "hypotheses": [{"id": "A"}]},
    )

    payload = build_openai_chat_completions_payload(
        request,
        model="provider-model",
    )

    assert "Python is optional" in payload["messages"][0]["content"]
    required = json.loads(payload["messages"][1]["content"])["required_output"]
    assert required["required_keys"] == [
        "mode",
        "purpose",
        "target_hypotheses",
        "expected_observation",
        "code",
    ]


def test_build_openai_payload_for_python_code_repair():
    request = StructuredModelRequest(
        task="repair_python_probe_code",
        input={"original_code": "bad()", "execution_error": {"exit_code": 1}},
    )

    payload = build_openai_request_payload(request, model="gpt-5.5")

    assert payload["text"]["format"]["name"] == "PythonCodeRepair"
    assert "Repair" in payload["input"][0]["content"]


def valid_payload() -> dict[str, object]:
    return {
        "evidence_type": "supporting",
        "likelihoods": {"H1": "moderately_confirming", "H2": "neutral"},
        "interpretation": "OpenAI fixture judgment.",
        "quality_overrides": {},
    }


def test_parse_openai_structured_response_accepts_direct_dict():
    assert parse_openai_structured_response(valid_payload()) == valid_payload()


def test_parse_openai_structured_response_accepts_direct_probe_signal_dict():
    payload = {"raw_content": "A structured probe signal."}

    assert parse_openai_structured_response(payload) == payload


def test_parse_openai_structured_response_accepts_json_string():
    assert parse_openai_structured_response(json.dumps(valid_payload())) == valid_payload()


def test_parse_openai_structured_response_accepts_output_text_object():
    response = SimpleNamespace(output_text=json.dumps(valid_payload()))

    assert parse_openai_structured_response(response) == valid_payload()


def test_parse_openai_structured_response_accepts_mapping_output_text():
    response = {"output_text": json.dumps(valid_payload())}

    assert parse_openai_structured_response(response) == valid_payload()


def test_parse_openai_structured_response_accepts_object_output_content_text():
    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                content=[
                    SimpleNamespace(text=json.dumps(valid_payload())),
                ]
            )
        ]
    )

    assert parse_openai_structured_response(response) == valid_payload()


def test_parse_openai_structured_response_accepts_mapping_output_content_text():
    response = {
        "output": [
            {
                "content": [
                    {"text": json.dumps(valid_payload())},
                ]
            }
        ]
    }

    assert parse_openai_structured_response(response) == valid_payload()


def test_parse_openai_chat_completions_response_extracts_mapping_message_content():
    response = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(valid_payload()),
                }
            }
        ]
    }

    assert parse_openai_chat_completions_response(response) == valid_payload()


class FakeChatMessage:
    content = json.dumps(valid_payload())


class FakeChatChoice:
    message = FakeChatMessage()


class FakeChatResponse:
    choices = [FakeChatChoice()]


def test_parse_openai_chat_completions_response_extracts_object_message_content():
    assert parse_openai_chat_completions_response(FakeChatResponse()) == valid_payload()


def test_parse_openai_chat_completions_response_reports_exhausted_output_budget():
    response = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {
                    "content": "",
                    "reasoning_content": "reasoning consumed the entire budget",
                },
            }
        ]
    }

    with pytest.raises(
        ModelGatewayValidationError,
        match="exhausted max_tokens before producing structured content",
    ):
        parse_openai_chat_completions_response(response)


def test_parse_openai_structured_response_rejects_provider_envelope_without_text():
    response = {
        "id": "resp_123",
        "status": "completed",
        "output": [],
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }

    with pytest.raises(
        ModelGatewayValidationError,
        match="openai structured response text was missing",
    ):
        parse_openai_structured_response(response)


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


class RecordingOpenAI:
    created_with = []

    def __init__(self, **kwargs):
        self.__class__.created_with.append(kwargs)
        self.responses = FakeResponses(response=json.dumps(valid_payload()))


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


class FakeChatCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **payload):
        self.calls.append(payload)
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(valid_payload()),
                    }
                }
            ]
        }


class FakeChatClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=FakeChatCompletions())


class RecordingInvocationObserver:
    def __init__(self, *, fail: bool = False):
        self.records: list[ProviderInvocationRecord] = []
        self.fail = fail

    def observe(self, record: ProviderInvocationRecord) -> None:
        self.records.append(record)
        if self.fail:
            raise RuntimeError("observer must not break inference")


class SequencedChatCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def create(self, **payload):
        self.calls.append(payload)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class SequencedChatClient:
    def __init__(self, outcomes):
        self.chat = SimpleNamespace(completions=SequencedChatCompletions(outcomes))


class RetryableProviderError(RuntimeError):
    def __init__(self, status_code: int, retry_after: str | None = None):
        super().__init__(f"provider status {status_code}")
        self.status_code = status_code
        headers = {} if retry_after is None else {"Retry-After": retry_after}
        self.response = SimpleNamespace(headers=headers)


def chat_response(*, response_id: str = "chatcmpl_1") -> dict[str, object]:
    return {
        "id": response_id,
        "system_fingerprint": "fp_1",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": json.dumps(valid_payload())},
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18,
            "prompt_tokens_details": {"cached_tokens": 4},
            "completion_tokens_details": {"reasoning_tokens": 3},
        },
    }


def test_openai_chat_completions_model_gateway_calls_fake_client_and_returns_dict():
    client = FakeChatClient()
    gateway = OpenAIChatCompletionsModelGateway(
        config=OpenAIModelGatewayConfig(
            model="provider-model",
            max_output_tokens=128,
        ),
        client=client,
    )

    result = gateway.complete_structured(make_judge_request())

    assert result == valid_payload()
    assert client.chat.completions.calls[0]["model"] == "provider-model"
    assert client.chat.completions.calls[0]["max_tokens"] == 128


def test_chat_gateway_observes_success_with_usage_and_request_context():
    observer = RecordingInvocationObserver()
    client = SequencedChatClient([chat_response()])
    request = StructuredModelRequest(
        task="judge_evidence",
        input=make_judge_request().input,
        prompt_id="evidence_judgment",
        prompt_version="v0.1",
        schema_name="EvidenceJudgment",
        schema_version="v0.1",
        metadata={
            "experiment_id": "experiment_1",
            "arm": "direct_flash",
            "sample_id": "sample_pseudonym",
            "run_id": "run_1",
        },
    )
    gateway = OpenAIChatCompletionsModelGateway(
        config=OpenAIModelGatewayConfig(
            model="provider-model",
            base_url="https://provider.example/v1",
        ),
        client=client,
        invocation_observer=observer,
    )

    assert gateway.complete_structured(request) == valid_payload()
    assert len(observer.records) == 1
    record = observer.records[0]
    assert record.outcome == "success"
    assert record.error_category is None
    assert record.base_host == "provider.example"
    assert record.usage.input_tokens == 10
    assert record.usage.cached_input_tokens == 4
    assert record.usage.reasoning_tokens == 3
    assert record.finish_reason == "stop"
    assert record.response_id == "chatcmpl_1"
    assert record.context.experiment_id == "experiment_1"
    assert record.context.sample_id == "sample_pseudonym"
    assert record.context.attempt_index == 1


def test_chat_gateway_observes_each_retry_and_honors_retry_after():
    observer = RecordingInvocationObserver()
    sleeps = []
    client = SequencedChatClient(
        [
            RetryableProviderError(429, retry_after="2"),
            RetryableProviderError(503),
            chat_response(response_id="chatcmpl_after_retry"),
        ]
    )
    gateway = OpenAIChatCompletionsModelGateway(
        config=OpenAIModelGatewayConfig(model="provider-model"),
        client=client,
        invocation_observer=observer,
        sleep=sleeps.append,
        random_value=lambda: 0,
    )

    assert gateway.complete_structured(make_judge_request()) == valid_payload()
    assert [record.context.attempt_index for record in observer.records] == [1, 2, 3]
    assert [record.outcome for record in observer.records] == ["error", "error", "success"]
    assert [record.error_category for record in observer.records] == [
        "rate_limited",
        "provider_server_error",
        None,
    ]
    assert sleeps == [2.0, 1.0]


def test_chat_gateway_does_not_retry_invalid_structured_output():
    observer = RecordingInvocationObserver()
    client = SequencedChatClient(
        [
            {
                "choices": [
                    {"finish_reason": "stop", "message": {"content": "not-json"}}
                ]
            },
            chat_response(),
        ]
    )
    gateway = OpenAIChatCompletionsModelGateway(
        config=OpenAIModelGatewayConfig(model="provider-model"),
        client=client,
        invocation_observer=observer,
        sleep=lambda _: None,
    )

    with pytest.raises(ModelGatewayValidationError):
        gateway.complete_structured(make_judge_request())

    assert len(client.chat.completions.calls) == 1
    assert len(observer.records) == 1
    assert observer.records[0].error_category == "invalid_response"


def test_chat_gateway_observer_failure_does_not_change_model_result():
    observer = RecordingInvocationObserver(fail=True)
    gateway = OpenAIChatCompletionsModelGateway(
        config=OpenAIModelGatewayConfig(model="provider-model"),
        client=SequencedChatClient([chat_response()]),
        invocation_observer=observer,
    )

    assert gateway.complete_structured(make_judge_request()) == valid_payload()
    assert len(observer.records) == 1


def test_openai_chat_completions_model_gateway_uses_stdlib_fallback_without_openai(
    monkeypatch,
):
    import bayesprobe.openai_gateway as openai_gateway

    monkeypatch.setattr(openai_gateway, "OpenAI", None, raising=False)
    calls = []

    def fake_post_json(url, payload, *, api_key, timeout_seconds):
        calls.append(
            {
                "url": url,
                "payload": payload,
                "api_key": api_key,
                "timeout_seconds": timeout_seconds,
            }
        )
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(valid_payload()),
                    }
                }
            ]
        }

    monkeypatch.setattr(openai_gateway, "_post_json", fake_post_json)
    gateway = OpenAIChatCompletionsModelGateway(
        config=OpenAIModelGatewayConfig(
            model="provider-model",
            base_url="https://provider.example/v1/",
            timeout_seconds=7.5,
            max_output_tokens=64,
        ),
        api_key="provider-secret-123",
    )

    result = gateway.complete_structured(make_judge_request())

    assert result == valid_payload()
    assert calls[0]["url"] == "https://provider.example/v1/chat/completions"
    assert calls[0]["api_key"] == "provider-secret-123"
    assert calls[0]["timeout_seconds"] == 7.5
    assert calls[0]["payload"]["model"] == "provider-model"
    assert calls[0]["payload"]["max_tokens"] == 64


def test_stdlib_http_error_preserves_retry_metadata_and_redacts_key(monkeypatch):
    import bayesprobe.openai_gateway as openai_gateway

    error = urllib.error.HTTPError(
        "https://provider.example/v1/chat/completions",
        503,
        "unavailable",
        {"Retry-After": "3"},
        io.BytesIO(b'{"error":"failed for sk-provider-secret"}'),
    )

    def raise_http_error(*args, **kwargs):
        raise error

    monkeypatch.setattr(urllib.request, "urlopen", raise_http_error)

    with pytest.raises(RuntimeError) as captured:
        openai_gateway._post_json(
            "https://provider.example/v1/chat/completions",
            {"model": "provider-model"},
            api_key="sk-provider-secret",
            timeout_seconds=30,
        )

    assert captured.value.status_code == 503
    assert captured.value.response.headers["Retry-After"] == "3"
    assert "sk-provider-secret" not in str(captured.value)


def test_stdlib_url_timeout_remains_retryable_timeout(monkeypatch):
    import bayesprobe.openai_gateway as openai_gateway

    def raise_timeout(*args, **kwargs):
        raise urllib.error.URLError(TimeoutError("provider read timed out"))

    monkeypatch.setattr(urllib.request, "urlopen", raise_timeout)

    with pytest.raises(TimeoutError, match="provider read timed out"):
        openai_gateway._post_json(
            "https://provider.example/v1/chat/completions",
            {"model": "provider-model"},
            api_key="sk-provider-secret",
            timeout_seconds=30,
        )


def test_openai_responses_model_gateway_propagates_provider_exceptions():
    error = RuntimeError("provider outage")
    gateway = OpenAIResponsesModelGateway(
        config=OpenAIModelGatewayConfig(model="gpt-5.5"),
        client=FakeOpenAIClient(response=error),
    )

    with pytest.raises(RuntimeError, match="provider outage"):
        gateway.complete_structured(make_judge_request())


def test_openai_responses_model_gateway_uses_request_scoped_key_and_base_url(
    monkeypatch,
):
    import bayesprobe.openai_gateway as openai_gateway

    RecordingOpenAI.created_with = []
    monkeypatch.setattr(openai_gateway, "OpenAI", RecordingOpenAI, raising=False)
    gateway = OpenAIResponsesModelGateway(
        config=OpenAIModelGatewayConfig(
            model="gpt-5.5",
            base_url="https://provider.example/v1",
            timeout_seconds=12.5,
        ),
        api_key="sk-request-scoped",
    )

    result = gateway.complete_structured(make_judge_request())

    assert result == valid_payload()
    assert RecordingOpenAI.created_with == [
        {
            "api_key": "sk-request-scoped",
            "timeout": 12.5,
            "base_url": "https://provider.example/v1",
        }
    ]


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
