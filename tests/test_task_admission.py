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
