from copy import deepcopy

import pytest

from bayesprobe.initialization import BayesProbeInitializer, InitializeRunInput
from bayesprobe.ledger import JsonlLedgerStore
from bayesprobe.model_gateway import ScriptedModelGateway, StructuredModelRequest
from bayesprobe.schemas import AnswerChoice, HypothesisRelation, TaskKind
from bayesprobe.task_framing import (
    ExplicitTaskFramer,
    HypothesisSeed,
    ModelTaskFramer,
    RecordedTaskFramer,
    RoutingTaskFramer,
    TaskFramingError,
    TaskFramingInput,
    TaskFramingRepairPolicy,
    task_frame_from_mapping,
)


class QueueModelGateway:
    adapter_kind = "queue_test"

    def __init__(self, responses: list[dict | Exception]) -> None:
        self.responses = list(responses)
        self.requests: list[StructuredModelRequest] = []

    def complete_structured(self, request: StructuredModelRequest) -> dict:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError(f"unexpected model task: {request.task}")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


VALID_OPEN_FRAME = {
    "task_kind": "claim_verification",
    "answer_contract": {
        "objective": "Design a discriminating validation protocol.",
        "required_sections": [
            "hypotheses",
            "experimental_design",
            "controls",
            "metrics",
            "decision_rule",
            "limitations",
        ],
        "decision_form": "experimental_protocol",
        "permits_synthesis": True,
    },
    "hypothesis_relation": "independent",
    "hypotheses": [
        {
            "statement": "Scale has an independent positive effect under matched conditions.",
            "type": "causal_claim",
            "scope": "Matched task, scaffold, and inference budget.",
            "falsifiers": ["The controlled effect is negligible or negative."],
            "predictions": ["Performance increases across matched sizes."],
        },
        {
            "statement": "The apparent scale effect is materially confounded.",
            "type": "confounding_explanation",
            "scope": "Comparisons with unmatched resources.",
            "falsifiers": ["The effect survives matched controls."],
            "predictions": ["The effect shrinks after matching resources."],
        },
    ],
    "coverage_statement": "Covers the target effect and the primary confounder.",
    "coverage_limitation": "Conditional task interactions remain possible.",
}


def test_explicit_framer_uses_structured_choices_without_text_parsing():
    frame = ExplicitTaskFramer().frame(
        TaskFramingInput(
            run_id="run_choices",
            question="Which result follows?",
            answer_choices=[
                AnswerChoice(label="A", text="First result"),
                AnswerChoice(label="B", text="Second result"),
            ],
        )
    )

    assert frame.task_kind == TaskKind.MULTIPLE_CHOICE
    assert frame.hypothesis_frame.relation == HypothesisRelation.EXCLUSIVE_EXHAUSTIVE
    assert [item.id for item in frame.hypothesis_frame.hypotheses] == ["A", "B"]


def test_explicit_framer_parses_english_legacy_choices():
    frame = ExplicitTaskFramer().frame(
        TaskFramingInput(
            run_id="run_en_choices",
            question="Which result follows?\nAnswer Choices:\nA. First result\nB. Second result",
        )
    )

    assert frame.task_kind == TaskKind.MULTIPLE_CHOICE
    assert frame.normalized_question == "Which result follows?"


def test_explicit_framer_parses_chinese_legacy_choices():
    frame = ExplicitTaskFramer().frame(
        TaskFramingInput(
            run_id="run_cn_choices",
            question="哪一项正确？\n答案选项：\nA. 第一项\nB. 第二项",
        )
    )

    assert frame.task_kind == TaskKind.MULTIPLE_CHOICE


def test_explicit_framer_uses_explicit_seeds():
    frame = ExplicitTaskFramer().frame(
        TaskFramingInput(
            run_id="run_seeds",
            question="Which explanation fits?",
            hypothesis_seeds=[
                HypothesisSeed(statement="The first explanation fits.", prior=0.5),
                HypothesisSeed(statement="The second explanation fits.", prior=0.5),
            ],
        )
    )

    assert frame.task_kind == TaskKind.DECISION
    assert [item.initial_prior for item in frame.hypothesis_frame.hypotheses] == [0.5, 0.5]


def test_explicit_framer_rejects_unseeded_open_question():
    with pytest.raises(TaskFramingError, match="requires a model or recorded task framer"):
        ExplicitTaskFramer().frame(
            TaskFramingInput(
                run_id="run_open",
                question="这个命题应该如何验证？",
            )
        )


def test_explicit_framer_can_frame_without_materializing(monkeypatch):
    framer = ExplicitTaskFramer()
    materializations = 0
    original_frame = ExplicitTaskFramer.frame

    def count_materializations(self, input):
        nonlocal materializations
        materializations += 1
        return original_frame(self, input)

    monkeypatch.setattr(ExplicitTaskFramer, "frame", count_materializations)

    assert framer.can_frame(
        TaskFramingInput(
            run_id="run_choices",
            question="Which result follows?",
            answer_choices=[
                AnswerChoice(label="A", text="First result"),
                AnswerChoice(label="B", text="Second result"),
            ],
        )
    )
    assert framer.can_frame(
        TaskFramingInput(
            run_id="run_legacy",
            question="Which result follows? Answer Choices: A. First result B. Second result",
        )
    )
    assert not framer.can_frame(
        TaskFramingInput(run_id="run_open", question="How should this claim be tested?")
    )
    assert materializations == 0


@pytest.mark.parametrize(
    "input",
    [
        TaskFramingInput(
            run_id="run_secret_question",
            question="Which result follows from sk-abcdefghijklmnop?",
            answer_choices=[
                AnswerChoice(label="A", text="First result"),
                AnswerChoice(label="B", text="Second result"),
            ],
        ),
        TaskFramingInput(
            run_id="run_secret_context",
            question="Which result follows?",
            task_context="Use sk-abcdefghijklmnop as a constraint.",
            answer_choices=[
                AnswerChoice(label="A", text="First result"),
                AnswerChoice(label="B", text="Second result"),
            ],
        ),
        TaskFramingInput(
            run_id="run_secret_choice",
            question="Which result follows?",
            answer_choices=[
                AnswerChoice(label="A", text="First result sk-abcdefghijklmnop"),
                AnswerChoice(label="B", text="Second result"),
            ],
        ),
    ],
)
def test_explicit_framer_rejects_secret_caller_input_without_materializing(input):
    framer = ExplicitTaskFramer()

    assert not framer.can_frame(input)
    with pytest.raises(TaskFramingError, match="task framing input must not contain secret material"):
        framer.frame(input)


def test_model_task_framer_rejects_secret_caller_input_before_gateway_call():
    gateway = QueueModelGateway([VALID_OPEN_FRAME])

    with pytest.raises(TaskFramingError, match="task framing input must not contain secret material"):
        ModelTaskFramer(gateway).frame(
            TaskFramingInput(
                run_id="run_model_secret_input",
                question="How should sk-abcdefghijklmnop be tested?",
            )
        )

    assert gateway.requests == []


def test_explicit_framer_wraps_late_task_frame_validation_as_task_framing_error():
    invalid_choice = AnswerChoice(label="A", text="First result").model_copy(
        update={"label": " "}
    )

    with pytest.raises(TaskFramingError, match="invalid explicit task frame fields"):
        ExplicitTaskFramer().frame(
            TaskFramingInput(
                run_id="run_invalid_late_frame",
                question="Which result follows?",
                answer_choices=[
                    invalid_choice,
                    AnswerChoice(label="B", text="Second result"),
                ],
            )
        )


@pytest.mark.parametrize(
    ("metadata", "credential"),
    [
        ({"api_key": "provider-secret-123"}, "provider-secret-123"),
        (
            {"nested": {"Authorization": "Bearer private-value"}},
            "private-value",
        ),
    ],
)
def test_explicit_framer_rejects_nested_secret_metadata_before_materialization(
    metadata,
    credential,
):
    input = TaskFramingInput(
        run_id="run_secret_metadata",
        question="Which result follows?",
        answer_choices=[
            AnswerChoice(label="A", text="First result"),
            AnswerChoice(label="B", text="Second result"),
        ],
        metadata=metadata,
    )
    framer = ExplicitTaskFramer()

    assert not framer.can_frame(input)
    with pytest.raises(TaskFramingError, match="task framing input must not contain secret material") as captured:
        framer.frame(input)

    _assert_secret_free_exception(captured.value, credential)


@pytest.mark.parametrize(
    "input",
    [
        TaskFramingInput(
            run_id="run_secret_choice_identifier",
            question="Which result follows?",
            answer_choices=[
                AnswerChoice(label="api-key", text="First result"),
                AnswerChoice(label="B", text="Second result"),
            ],
        ),
        TaskFramingInput(
            run_id="run_secret_seed_identifier",
            question="Which explanation fits?",
            hypothesis_seeds=[
                HypothesisSeed(id="authorization", statement="First explanation."),
                HypothesisSeed(id="H2", statement="Second explanation."),
            ],
        ),
    ],
)
def test_explicit_framer_rejects_secret_identifier_fields_during_pure_preflight(input):
    framer = ExplicitTaskFramer()

    assert not framer.can_frame(input)
    with pytest.raises(TaskFramingError, match="task framing input must not contain secret material"):
        framer.frame(input)


def test_explicit_framer_allows_ordinary_tokenization_prose():
    frame = ExplicitTaskFramer().frame(
        TaskFramingInput(
            run_id="run_tokenization_prose",
            question="Which tokenization strategy best fits the corpus?",
            task_context="Discuss tokenization concepts for a general audience.",
            answer_choices=[
                AnswerChoice(label="A", text="Word-level tokenization"),
                AnswerChoice(label="B", text="Subword tokenization"),
            ],
        )
    )

    assert frame.task_kind == TaskKind.MULTIPLE_CHOICE


def test_initializer_rejects_secret_metadata_before_run_belief_or_ledger_materialization(
    tmp_path,
):
    ledger_path = tmp_path / "secret-metadata.jsonl"
    initializer = BayesProbeInitializer(ledger=JsonlLedgerStore(ledger_path))

    with pytest.raises(TaskFramingError, match="task framing input must not contain secret material") as captured:
        initializer.initialize(
            InitializeRunInput(
                run_id="run_secret_metadata_ledger",
                problem="Which result follows?",
                answer_choices=[
                    AnswerChoice(label="A", text="First result"),
                    AnswerChoice(label="B", text="Second result"),
                ],
                metadata={"nested": {"authorization": "Bearer private-value"}},
            )
        )

    _assert_secret_free_exception(captured.value, "private-value")
    assert not ledger_path.exists()


@pytest.mark.parametrize(
    "input",
    [
        TaskFramingInput(
            run_id="run_one_choice",
            question="Which result follows?",
            answer_choices=[AnswerChoice(label="A", text="First result")],
        ),
        TaskFramingInput(
            run_id="run_one_seed",
            question="Which explanation fits?",
            hypothesis_seeds=[HypothesisSeed(statement="The only explanation.")],
        ),
        TaskFramingInput(
            run_id="run_conflict",
            question="Which explanation fits?",
            answer_choices=[
                AnswerChoice(label="A", text="First result"),
                AnswerChoice(label="B", text="Second result"),
            ],
            hypothesis_seeds=[
                HypothesisSeed(statement="The first explanation.", prior=0.5),
                HypothesisSeed(statement="The second explanation.", prior=0.5),
            ],
        ),
        TaskFramingInput(
            run_id="run_partial_priors",
            question="Which explanation fits?",
            hypothesis_seeds=[
                HypothesisSeed(statement="The first explanation.", prior=0.5),
                HypothesisSeed(statement="The second explanation."),
            ],
        ),
        TaskFramingInput(
            run_id="run_invalid_priors",
            question="Which explanation fits?",
            hypothesis_seeds=[
                HypothesisSeed(statement="The first explanation.", prior=0.7),
                HypothesisSeed(statement="The second explanation.", prior=0.7),
            ],
        ),
    ],
)
def test_explicit_framer_capability_rejects_invalid_explicit_inputs(input):
    framer = ExplicitTaskFramer()

    assert not framer.can_frame(input)
    with pytest.raises(TaskFramingError):
        framer.frame(input)


@pytest.mark.parametrize(
    "seed",
    [
        HypothesisSeed(statement="First explanation.", prior="0.5"),
        HypothesisSeed(statement="First explanation.", prior=float("nan")),
        HypothesisSeed(statement="First explanation.", prior=float("inf")),
        HypothesisSeed(statement="First explanation.", id=1),
        HypothesisSeed(statement="First explanation.", scope=object()),
        HypothesisSeed(statement="First explanation.", falsifiers=["Valid", 3]),
        HypothesisSeed(statement="First explanation.", predictions="not a list"),
    ],
)
def test_explicit_framer_rejects_malformed_seed_values(seed):
    companion = HypothesisSeed(
        statement="Second explanation.",
        prior=seed.prior if seed.prior is not None else None,
    )
    input = TaskFramingInput(
        run_id="run_malformed_seed",
        question="Which explanation fits?",
        hypothesis_seeds=[seed, companion],
    )
    framer = ExplicitTaskFramer()

    assert not framer.can_frame(input)
    with pytest.raises(TaskFramingError):
        framer.frame(input)


def test_explicit_framer_defaults_valid_independent_seed_credences():
    frame = ExplicitTaskFramer().frame(
        TaskFramingInput(
            run_id="run_independent",
            question="Which conditions apply?",
            task_context="Keep the two conditions separate.",
            hypothesis_relation=HypothesisRelation.INDEPENDENT,
            hypothesis_seeds=[
                HypothesisSeed(statement="The first condition applies."),
                HypothesisSeed(statement="The second condition applies."),
            ],
        )
    )

    assert frame.task_context == "Keep the two conditions separate."
    assert [item.initial_prior for item in frame.hypothesis_frame.hypotheses] == [0.5, 0.5]
    assert frame.hypothesis_frame.rival_sets == {"H1": [], "H2": []}


def test_model_task_framer_calls_gateway_before_returning_frame():
    gateway = ScriptedModelGateway({"frame_open_question": VALID_OPEN_FRAME})

    frame = ModelTaskFramer(gateway).frame(
        TaskFramingInput(
            run_id="run_model",
            question="这个命题应该如何验证？",
            task_context="Use matched conditions.",
        )
    )

    assert [request.task for request in gateway.requests] == ["frame_open_question"]
    assert gateway.requests[0].input == {
        "question": "这个命题应该如何验证？",
        "task_context": "Use matched conditions.",
        "supported_task_kinds": [
            "claim_verification",
            "explanation",
            "diagnosis",
            "design",
            "decision",
        ],
        "supported_relations": ["exclusive_exhaustive", "independent"],
        "hypothesis_count": {"minimum": 2, "maximum": 6},
    }
    assert [item.id for item in frame.hypothesis_frame.hypotheses] == ["H1", "H2"]
    assert [item.initial_prior for item in frame.hypothesis_frame.hypotheses] == [0.5, 0.5]
    assert frame.hypothesis_frame.rival_sets == {"H1": [], "H2": []}
    assert frame.framing_method.value == "model"


def test_model_task_framer_repairs_once_then_accepts_without_retaining_secrets():
    gateway = QueueModelGateway(
        [
            {
                "task_kind": "claim_verification",
                "hypotheses": [],
                "api_key": "sk-abcdefghijklmnop",
            },
            VALID_OPEN_FRAME,
        ]
    )

    frame = ModelTaskFramer(gateway).frame(
        TaskFramingInput(run_id="run_repair", question="How should this be tested?")
    )

    assert [request.task for request in gateway.requests] == [
        "frame_open_question",
        "repair_task_frame",
    ]
    assert gateway.requests[1].metadata["repair_attempt_index"] == 1
    assert "api_key" not in gateway.requests[1].input["invalid_payload"]
    assert "sk-abcdefghijklmnop" not in repr(gateway.requests[1])
    assert frame.framing_trace["repair_attempt_index"] == 1


def test_model_task_framer_repairs_secret_bearing_semantic_payload_without_leaking_it():
    secret = "sk-abcdefghijklmnop"
    secret_payload = deepcopy(VALID_OPEN_FRAME)
    secret_payload["hypotheses"][0]["statement"] = f"The provider leaked {secret}."
    gateway = QueueModelGateway([secret_payload, VALID_OPEN_FRAME])

    frame = ModelTaskFramer(gateway).frame(
        TaskFramingInput(run_id="run_semantic_secret_repair", question="How should this be tested?")
    )

    assert [request.task for request in gateway.requests] == [
        "frame_open_question",
        "repair_task_frame",
    ]
    repair_request = gateway.requests[1]
    assert secret not in repr(repair_request)
    assert secret not in str(repair_request.input["validation_error"])
    assert secret not in repr(repair_request.input["invalid_payload"])
    assert "[REDACTED]" in repair_request.input["invalid_payload"]["hypotheses"][0]["statement"]
    assert secret not in frame.model_dump_json()
    assert frame.framing_trace["repair_attempt_index"] == 1


def test_model_task_framer_fails_closed_after_second_secret_bearing_payload():
    secret = "sk-abcdefghijklmnop"
    secret_payload = deepcopy(VALID_OPEN_FRAME)
    secret_payload["coverage_statement"] = f"The provider leaked {secret}."
    gateway = QueueModelGateway([secret_payload, secret_payload])

    with pytest.raises(TaskFramingError, match="invalid after 1 repair attempt") as captured:
        ModelTaskFramer(gateway).frame(
            TaskFramingInput(run_id="run_second_semantic_secret", question="How should this be tested?")
        )

    assert [request.task for request in gateway.requests] == [
        "frame_open_question",
        "repair_task_frame",
    ]
    assert secret not in repr(gateway.requests[1])
    _assert_secret_free_exception(captured.value, secret)


def test_secret_bearing_model_frame_never_reaches_task_frame_ledger_or_trace(tmp_path):
    secret = "sk-abcdefghijklmnop"
    secret_payload = deepcopy(VALID_OPEN_FRAME)
    secret_payload["answer_contract"]["objective"] = f"Do not expose {secret}."
    ledger = JsonlLedgerStore(tmp_path / "task-framing.jsonl")
    initialized = BayesProbeInitializer(
        ledger=ledger,
        task_framer=ModelTaskFramer(QueueModelGateway([secret_payload, VALID_OPEN_FRAME])),
    ).initialize(
        InitializeRunInput(
            run_id="run_ledger_secret_repair",
            problem="How should this be tested?",
        )
    )

    assert secret not in initialized.task_frame.model_dump_json()
    assert secret not in initialized.belief_state.model_dump_json()
    assert secret not in (tmp_path / "task-framing.jsonl").read_text(encoding="utf-8")


def test_model_task_framer_fails_after_one_invalid_repair():
    gateway = QueueModelGateway(
        [
            {"task_kind": "claim_verification", "hypotheses": []},
            {"task_kind": "claim_verification", "hypotheses": []},
        ]
    )

    with pytest.raises(TaskFramingError, match="invalid after 1 repair attempt"):
        ModelTaskFramer(gateway).frame(
            TaskFramingInput(run_id="run_bad_repair", question="How should this be tested?")
        )


@pytest.mark.parametrize("max_attempts", [2, 3, 100])
def test_task_framing_repair_policy_rejects_more_than_one_repair(max_attempts):
    with pytest.raises(ValueError, match="at most one"):
        TaskFramingRepairPolicy(max_attempts=max_attempts)


@pytest.mark.parametrize(
    "policy,expected_tasks",
    [
        (TaskFramingRepairPolicy(max_attempts=0), ["frame_open_question"]),
        (
            TaskFramingRepairPolicy(max_attempts=1),
            ["frame_open_question", "repair_task_frame"],
        ),
    ],
)
def test_model_task_framer_request_order_is_bounded_to_frame_plus_one_repair(
    policy,
    expected_tasks,
):
    gateway = QueueModelGateway(
        [
            {"task_kind": "claim_verification", "hypotheses": []},
            {"task_kind": "claim_verification", "hypotheses": []},
        ]
    )

    with pytest.raises(TaskFramingError):
        ModelTaskFramer(gateway, repair_policy=policy).frame(
            TaskFramingInput(run_id="run_bounded_repair", question="How should this be tested?")
        )

    assert [request.task for request in gateway.requests] == expected_tasks


@pytest.mark.parametrize(
    "field,value",
    [
        ("statement", 42),
        ("type", ["causal_claim"]),
        ("scope", None),
        ("falsifiers", "not-a-list"),
        ("predictions", ("not-a-native-list",)),
        ("coverage_statement", 42),
        ("coverage_limitation", {"text": "not-a-string"}),
    ],
)
def test_chat_shaped_malformed_provider_semantics_get_one_repair(field, value):
    malformed = deepcopy(VALID_OPEN_FRAME)
    if field in {"coverage_statement", "coverage_limitation"}:
        malformed[field] = value
    else:
        malformed["hypotheses"][0][field] = value
    gateway = QueueModelGateway([malformed, VALID_OPEN_FRAME])

    frame = ModelTaskFramer(gateway).frame(
        TaskFramingInput(run_id=f"run_native_{field}", question="How should this be tested?")
    )

    assert frame.hypothesis_frame.hypotheses[0].statement == (
        VALID_OPEN_FRAME["hypotheses"][0]["statement"]
    )
    assert [request.task for request in gateway.requests] == [
        "frame_open_question",
        "repair_task_frame",
    ]


def test_second_chat_shaped_malformed_provider_frame_fails_closed():
    malformed = deepcopy(VALID_OPEN_FRAME)
    malformed["hypotheses"][0]["falsifiers"] = "not-a-list"
    gateway = QueueModelGateway([malformed, malformed])

    with pytest.raises(TaskFramingError, match="invalid after 1 repair attempt"):
        ModelTaskFramer(gateway).frame(
            TaskFramingInput(run_id="run_native_fail_closed", question="How should this be tested?")
        )

    assert [request.task for request in gateway.requests] == [
        "frame_open_question",
        "repair_task_frame",
    ]


def test_repair_request_uses_shared_redaction_for_forbidden_fields_and_values():
    malformed = deepcopy(VALID_OPEN_FRAME)
    malformed["hypotheses"][0].update(
        {
            "private_key": "private-field-value",
            "password": "password-field-value",
            "credential": "credential-field-value",
            "access_key": "access-field-value",
        }
    )
    gateway = QueueModelGateway([malformed, VALID_OPEN_FRAME])

    ModelTaskFramer(gateway).frame(
        TaskFramingInput(run_id="run_shared_redaction", question="How should this be tested?")
    )

    repair_payload = repr(gateway.requests[1].input)
    for forbidden in (
        "private_key",
        "password",
        "credential",
        "access_key",
        "private-field-value",
        "password-field-value",
        "credential-field-value",
        "access-field-value",
    ):
        assert forbidden not in repair_payload


def test_generic_secret_text_is_redacted_before_repair_and_from_exception_chain():
    secret = "Authorization: Bearer abcdefghijklmnop"
    malformed = deepcopy(VALID_OPEN_FRAME)
    malformed["coverage_statement"] = secret
    gateway = QueueModelGateway([malformed, malformed])

    with pytest.raises(TaskFramingError) as captured:
        ModelTaskFramer(gateway).frame(
            TaskFramingInput(run_id="run_generic_secret", question="How should this be tested?")
        )

    assert secret not in repr(gateway.requests[1])
    _assert_secret_free_exception(captured.value, secret)


def test_model_task_framer_hides_initial_gateway_exception_details():
    secret = "sk-initialgatewaycredential"
    gateway = QueueModelGateway([RuntimeError(f"provider rejected {secret}")])

    with pytest.raises(TaskFramingError) as captured:
        ModelTaskFramer(gateway).frame(
            TaskFramingInput(run_id="run_gateway_initial", question="How should this be tested?")
        )

    _assert_secret_free_exception(captured.value, secret)
    assert str(captured.value) == "task framing model gateway call failed"
    assert [request.task for request in gateway.requests] == ["frame_open_question"]


def test_model_task_framer_hides_repair_gateway_exception_details():
    secret = "sk-repairgatewaycredential"
    gateway = QueueModelGateway(
        [
            {"task_kind": "claim_verification", "hypotheses": []},
            RuntimeError(f"provider rejected {secret}"),
        ]
    )

    with pytest.raises(TaskFramingError) as captured:
        ModelTaskFramer(gateway).frame(
            TaskFramingInput(run_id="run_gateway_repair", question="How should this be tested?")
        )

    _assert_secret_free_exception(captured.value, secret)
    assert str(captured.value) == "task framing model gateway call failed"
    assert [request.task for request in gateway.requests] == [
        "frame_open_question",
        "repair_task_frame",
    ]


def test_task_frame_from_mapping_rejects_provider_owned_beliefs():
    payload = {
        **VALID_OPEN_FRAME,
        "hypotheses": [
            {**VALID_OPEN_FRAME["hypotheses"][0], "id": "provider_h1"},
            VALID_OPEN_FRAME["hypotheses"][1],
        ],
    }

    with pytest.raises(TaskFramingError, match="cannot assign ids or beliefs"):
        task_frame_from_mapping(
            payload,
            run_id="run_strict",
            question="How should this be tested?",
            task_context="",
            method="model",
            trace={},
        )


def test_recorded_task_framer_returns_a_run_specific_deep_copy():
    source_frame = ExplicitTaskFramer().frame(
        TaskFramingInput(
            run_id="fixture",
            question="Which explanation fits?",
            hypothesis_seeds=[
                HypothesisSeed(statement="The first explanation."),
                HypothesisSeed(statement="The second explanation."),
            ],
        )
    )

    frame = RecordedTaskFramer(source_frame).frame(
        TaskFramingInput(run_id="replay", question="How should this be tested?")
    )

    assert frame.task_frame_id == "replay_task_frame"
    assert frame.hypothesis_frame.frame_id == "replay_hypothesis_frame"
    assert frame.normalized_question == "How should this be tested?"
    assert frame.framing_method.value == "recorded"
    assert source_frame.framing_method.value == "explicit"


def test_recorded_task_framer_scopes_trace_to_current_run_and_keeps_provenance():
    source_frame = ModelTaskFramer(
        ScriptedModelGateway({"frame_open_question": VALID_OPEN_FRAME})
    ).frame(
        TaskFramingInput(run_id="fixture_run", question="How should this be tested?")
    )

    frame = RecordedTaskFramer(source_frame).frame(
        TaskFramingInput(run_id="replay_run", question="Replay this question.")
    )

    assert frame.framing_trace["metadata"]["run_id"] == "replay_run"
    assert frame.framing_trace["recorded_from_task_frame_id"] == "fixture_run_task_frame"
    assert frame.framing_trace["source_framing_method"] == "model"
    assert frame.framing_trace["source_trace"]["metadata"]["run_id"] == "fixture_run"


def test_recorded_task_framer_uses_current_normalized_question_and_task_context():
    source_frame = ExplicitTaskFramer().frame(
        TaskFramingInput(
            run_id="fixture_current_input",
            question="Fixture question",
            task_context="Stale fixture context",
            hypothesis_seeds=[
                HypothesisSeed(statement="The first explanation."),
                HypothesisSeed(statement="The second explanation."),
            ],
        )
    )

    frame = RecordedTaskFramer(source_frame).frame(
        TaskFramingInput(
            run_id="replay_current_input",
            question="  Current replay question  ",
            task_context="  Current replay context  ",
        )
    )

    assert frame.normalized_question == "Current replay question"
    assert frame.task_context == "Current replay context"


@pytest.mark.parametrize(
    "input",
    [
        TaskFramingInput(run_id="replay_empty_question", question="   "),
        TaskFramingInput(
            run_id="replay_bad_context",
            question="Current replay question",
            task_context=123,
        ),
    ],
)
def test_recorded_task_framer_rejects_malformed_current_input_with_stable_error(input):
    source_frame = ExplicitTaskFramer().frame(
        TaskFramingInput(
            run_id="fixture_bad_current_input",
            question="Fixture question",
            hypothesis_seeds=[
                HypothesisSeed(statement="The first explanation."),
                HypothesisSeed(statement="The second explanation."),
            ],
        )
    )

    with pytest.raises(TaskFramingError, match="invalid recorded task framing input"):
        RecordedTaskFramer(source_frame).frame(input)


def test_recorded_task_framer_revalidates_the_materialized_task_frame():
    source_frame = ExplicitTaskFramer().frame(
        TaskFramingInput(
            run_id="fixture_invalid_copy",
            question="Fixture question",
            hypothesis_seeds=[
                HypothesisSeed(statement="The first explanation."),
                HypothesisSeed(statement="The second explanation."),
            ],
        )
    )
    invalid_source = source_frame.model_copy(
        update={
            "answer_contract": source_frame.answer_contract.model_copy(
                update={"required_sections": []}
            )
        }
    )

    with pytest.raises(TaskFramingError, match="invalid recorded task frame"):
        RecordedTaskFramer(invalid_source).frame(
            TaskFramingInput(run_id="replay_invalid_copy", question="Current question")
        )


def test_routing_task_framer_keeps_explicit_mcq_off_the_model_gateway():
    gateway = QueueModelGateway([])

    frame = RoutingTaskFramer(
        explicit_framer=ExplicitTaskFramer(),
        open_framer=ModelTaskFramer(gateway),
    ).frame(
        TaskFramingInput(
            run_id="run_routed_mcq",
            question="Which result follows? Answer Choices: A. First B. Second",
        )
    )

    assert frame.task_kind == TaskKind.MULTIPLE_CHOICE
    assert gateway.requests == []


def _assert_secret_free_exception(error: BaseException, secret: str) -> None:
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        assert secret not in str(current)
        assert secret not in repr(current)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
