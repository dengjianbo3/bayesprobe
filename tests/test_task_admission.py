from copy import deepcopy

import pytest

from bayesprobe.model_gateway import ScriptedModelGateway, StructuredModelRequest
from bayesprobe.schemas import AnswerChoice, TaskAdmissionStatus, TaskKind
from bayesprobe.task_admission import (
    ExplicitTaskAdmitter,
    ModelTaskAdmitter,
    RoutingTaskAdmitter,
    TaskAdmissionError,
    TaskAdmissionInput,
)
from bayesprobe.task_framing import HypothesisSeed


ADMITTED_RESPONSE = {
    "status": "admitted",
    "epistemic_basis": [
        "The requested answer can be tested against discriminating claims."
    ],
    "proposed_task_kind": "exact_answer",
    "answer_contract_outline": {
        "objective": "Return the supported integer value.",
        "answer_value_type": "integer",
        "decision_form": "single_value",
        "permits_synthesis": False,
        "required_sections": ["answer", "basis", "uncertainty"],
    },
    "clarification_questions": [],
    "reason": "The task has a verifiable scalar answer.",
}


class QueueModelGateway:
    adapter_kind = "queue_test"

    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.requests: list[StructuredModelRequest] = []

    def complete_structured(self, request: StructuredModelRequest) -> dict:
        self.requests.append(request)
        return self.responses.pop(0)


def mcq_admission_input() -> TaskAdmissionInput:
    return TaskAdmissionInput(
        attempt_id="attempt_mcq",
        question="Which result follows?",
        answer_choices=[
            AnswerChoice(label="A", text="First result"),
            AnswerChoice(label="B", text="Second result"),
        ],
    )


def test_explicit_mcq_is_admitted_without_model_call():
    gateway = ScriptedModelGateway(responses={})
    admitter = RoutingTaskAdmitter(
        explicit_admitter=ExplicitTaskAdmitter(),
        open_admitter=ModelTaskAdmitter(gateway),
    )

    decision = admitter.assess(mcq_admission_input())

    assert decision.status == TaskAdmissionStatus.ADMITTED
    assert decision.proposed_task_kind == TaskKind.MULTIPLE_CHOICE
    assert gateway.requests == []


def test_unseeded_open_task_is_assessed_by_model_without_passive_metadata():
    gateway = ScriptedModelGateway(
        responses={"assess_task_admission": deepcopy(ADMITTED_RESPONSE)}
    )
    admitter = RoutingTaskAdmitter(
        explicit_admitter=ExplicitTaskAdmitter(),
        open_admitter=ModelTaskAdmitter(gateway),
    )
    input = TaskAdmissionInput(
        attempt_id="attempt_open",
        question="What integer satisfies the constraints?",
        task_context="Use the supplied theorem.",
        requested_output_shape="integer with basis",
        model_metadata={"provider_token_count": 42},
    )

    decision = admitter.assess(input)

    assert decision.attempt_id == "attempt_open"
    assert decision.status == TaskAdmissionStatus.ADMITTED
    assert [request.task for request in gateway.requests] == ["assess_task_admission"]
    request_input = gateway.requests[0].input
    assert request_input["question"] == input.question
    assert request_input["task_context"] == input.task_context
    assert request_input["requested_output_shape"] == "integer with basis"
    assert request_input["available_capabilities"] == []
    assert "model_metadata" not in request_input


def test_secret_requested_output_shape_is_rejected_before_gateway_request():
    gateway = QueueModelGateway([])

    with pytest.raises(TaskAdmissionError, match="must not contain secret material"):
        ModelTaskAdmitter(gateway).assess(
            TaskAdmissionInput(
                attempt_id="attempt_secret_output_shape",
                question="What integer satisfies the constraints?",
                requested_output_shape="Authorization: Bearer abcdefghijklmnop1",
            )
        )

    assert gateway.requests == []


@pytest.mark.parametrize(
    "explicit_material",
    [
        {
            "answer_choices": [
                AnswerChoice(label="A", text="Only result"),
            ]
        },
        {
            "answer_choices": [
                AnswerChoice.model_construct(label="", text="Blank result"),
                AnswerChoice(label="B", text="Second result"),
            ]
        },
        {
            "hypothesis_seeds": [
                HypothesisSeed(statement="   "),
                HypothesisSeed(statement="A valid alternative."),
            ]
        },
        {
            "hypothesis_seeds": [
                object(),
                HypothesisSeed(statement="A valid alternative."),
            ]
        },
    ],
    ids=["single_choice", "blank_choice", "blank_seed", "invalid_seed"],
)
def test_invalid_explicit_material_routes_to_model_admission(explicit_material):
    gateway = ScriptedModelGateway(
        responses={"assess_task_admission": deepcopy(ADMITTED_RESPONSE)}
    )
    admitter = RoutingTaskAdmitter(
        explicit_admitter=ExplicitTaskAdmitter(),
        open_admitter=ModelTaskAdmitter(gateway),
    )

    decision = admitter.assess(
        TaskAdmissionInput(
            attempt_id="attempt_invalid_explicit",
            question="What result follows?",
            **explicit_material,
        )
    )

    assert decision.proposed_task_kind == TaskKind.EXACT_ANSWER
    assert [request.task for request in gateway.requests] == [
        "assess_task_admission"
    ]


def test_simultaneous_choices_and_seeds_are_rejected_without_model_call():
    gateway = ScriptedModelGateway(responses={})
    admitter = RoutingTaskAdmitter(
        explicit_admitter=ExplicitTaskAdmitter(),
        open_admitter=ModelTaskAdmitter(gateway),
    )

    with pytest.raises(
        TaskAdmissionError,
        match="answer choices or hypothesis seeds, not both",
    ):
        admitter.assess(
            TaskAdmissionInput(
                attempt_id="attempt_ambiguous_explicit",
                question="Which account is supported?",
                answer_choices=[
                    AnswerChoice(label="A", text="First account"),
                    AnswerChoice(label="B", text="Second account"),
                ],
                hypothesis_seeds=[
                    HypothesisSeed(statement="The first account is supported."),
                    HypothesisSeed(statement="The second account is supported."),
                ],
            )
        )

    assert gateway.requests == []


def test_valid_hypothesis_seeds_are_admitted_without_model_call():
    gateway = ScriptedModelGateway(responses={})
    admitter = RoutingTaskAdmitter(
        explicit_admitter=ExplicitTaskAdmitter(),
        open_admitter=ModelTaskAdmitter(gateway),
    )

    decision = admitter.assess(
        TaskAdmissionInput(
            attempt_id="attempt_valid_seeds",
            question="Which account is supported?",
            hypothesis_seeds=[
                HypothesisSeed(statement="The first account is supported."),
                HypothesisSeed(statement="The second account is supported."),
            ],
        )
    )

    assert decision.proposed_task_kind == TaskKind.CLAIM_VERIFICATION
    assert gateway.requests == []


@pytest.mark.parametrize(
    "task_kind",
    [TaskKind.EXACT_ANSWER, TaskKind.MULTIPLE_CHOICE],
)
def test_hypothesis_seeds_reject_task_kinds_that_require_answer_values(task_kind):
    gateway = ScriptedModelGateway(responses={})
    explicit_admitter = ExplicitTaskAdmitter()
    admitter = RoutingTaskAdmitter(
        explicit_admitter=explicit_admitter,
        open_admitter=ModelTaskAdmitter(gateway),
    )
    input = TaskAdmissionInput(
        attempt_id=f"attempt_invalid_seed_kind_{task_kind.value}",
        question="Which answer is supported?",
        hypothesis_seeds=[
            HypothesisSeed(statement="The first explanation is supported."),
            HypothesisSeed(statement="The second explanation is supported."),
        ],
        model_metadata={"task_kind": task_kind.value},
    )

    with pytest.raises(TaskAdmissionError, match="hypothesis seeds cannot frame"):
        explicit_admitter.can_assess(input)
    with pytest.raises(TaskAdmissionError, match="hypothesis seeds cannot frame"):
        admitter.assess(input)
    assert gateway.requests == []


def test_model_admitter_repairs_once_then_accepts_valid_decision():
    gateway = QueueModelGateway(
        [{"status": "maybe"}, deepcopy(ADMITTED_RESPONSE)]
    )

    decision = ModelTaskAdmitter(gateway).assess(
        TaskAdmissionInput(
            attempt_id="attempt_repair",
            question="What integer satisfies the constraints?",
        )
    )

    assert decision.status == TaskAdmissionStatus.ADMITTED
    assert [request.task for request in gateway.requests] == [
        "assess_task_admission",
        "repair_task_admission",
    ]


def test_model_admitter_fails_closed_after_second_invalid_payload():
    gateway = QueueModelGateway([{"status": "maybe"}, {"status": "still_maybe"}])

    with pytest.raises(
        TaskAdmissionError,
        match="task admission invalid after 1 repair attempt",
    ):
        ModelTaskAdmitter(gateway).assess(
            TaskAdmissionInput(
                attempt_id="attempt_invalid",
                question="What integer satisfies the constraints?",
            )
        )

    assert len(gateway.requests) == 2


def test_secret_bearing_decision_is_rejected_then_repaired_without_secret_retention():
    secret = "sk-abcdefghijklmnop"
    invalid = deepcopy(ADMITTED_RESPONSE)
    invalid["answer_contract_outline"]["objective"] = f"Return {secret}."
    gateway = QueueModelGateway([invalid, deepcopy(ADMITTED_RESPONSE)])

    decision = ModelTaskAdmitter(gateway).assess(
        TaskAdmissionInput(
            attempt_id="attempt_secret_decision_repair",
            question="What integer satisfies the constraints?",
        )
    )

    assert decision.status == TaskAdmissionStatus.ADMITTED
    assert [request.task for request in gateway.requests] == [
        "assess_task_admission",
        "repair_task_admission",
    ]
    assert secret not in repr(gateway.requests[1])
    assert secret not in decision.model_dump_json()


def test_second_secret_bearing_decision_fails_closed():
    invalid = deepcopy(ADMITTED_RESPONSE)
    invalid["reason"] = "password=provider-value-123"
    gateway = QueueModelGateway([invalid, deepcopy(invalid)])

    with pytest.raises(
        TaskAdmissionError,
        match="task admission invalid after 1 repair attempt",
    ):
        ModelTaskAdmitter(gateway).assess(
            TaskAdmissionInput(
                attempt_id="attempt_secret_decision_fail_closed",
                question="What integer satisfies the constraints?",
            )
        )

    assert [request.task for request in gateway.requests] == [
        "assess_task_admission",
        "repair_task_admission",
    ]
