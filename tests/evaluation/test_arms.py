from collections.abc import Mapping

import pytest

from bayesprobe.evaluation.arms import DirectFlashArm
from bayesprobe.evaluation.contracts import EvaluationCase
from bayesprobe.model_gateway import ModelGatewayValidationError, StructuredModelRequest


class SequenceGateway:
    adapter_kind = "sequence"

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.requests: list[StructuredModelRequest] = []

    def complete_structured(self, request: StructuredModelRequest):
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def make_case() -> EvaluationCase:
    return EvaluationCase(
        sample_id="synthetic_1",
        question="What is 2 + 2?\n\nAnswer Choices:\nA. 3\nB. 4\nC. 5",
        choices={"A": "3", "B": "4", "C": "5"},
    )


def valid_answer():
    return {
        "answer_label": "B",
        "choice_probabilities": {"A": 0.05, "B": 0.9, "C": 0.05},
        "answer_summary": "Two plus two equals four.",
    }


def test_direct_arm_sends_only_question_and_choices_and_returns_distribution():
    gateway = SequenceGateway([valid_answer()])
    arm = DirectFlashArm(
        gateway,
        invocation_metadata={
            "experiment_id": "experiment_1",
            "arm": "direct_flash",
        },
    )

    result = arm.run_case(make_case())

    assert result.state == "completed"
    assert result.answer_label == "B"
    assert result.probabilities == {"A": 0.05, "B": 0.9, "C": 0.05}
    assert result.answer_summary == "Two plus two equals four."
    request = gateway.requests[0]
    assert request.task == "answer_multiple_choice"
    assert request.input == {
        "question": make_case().question,
        "choices": make_case().choices,
    }
    assert request.metadata == {
        "experiment_id": "experiment_1",
        "arm": "direct_flash",
        "sample_id": "synthetic_1",
    }
    serialized_input = str(request.input)
    assert "gold" not in serialized_input.lower()
    assert "category" not in serialized_input.lower()
    assert "cais/hle" not in serialized_input.lower()


def test_direct_arm_normalizes_probability_sum_within_tolerance():
    answer = valid_answer()
    answer["choice_probabilities"] = {"A": 0.1, "B": 0.7995, "C": 0.1}

    result = DirectFlashArm(SequenceGateway([answer])).run_case(make_case())

    assert sum(result.probabilities.values()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "invalid_payload",
    [
        {
            "answer_label": "B",
            "choice_probabilities": {"A": 0.1, "B": 0.9},
            "answer_summary": "Missing C.",
        },
        {
            "answer_label": "D",
            "choice_probabilities": {"A": 0.1, "B": 0.8, "C": 0.1},
            "answer_summary": "Unknown label.",
        },
        {
            "answer_label": "B",
            "choice_probabilities": {"A": 0.1, "B": 0.8, "C": 0.1},
            "answer_summary": "",
        },
    ],
)
def test_direct_arm_repairs_one_invalid_structured_answer(invalid_payload):
    gateway = SequenceGateway([invalid_payload, valid_answer()])

    result = DirectFlashArm(gateway).run_case(make_case())

    assert result.state == "completed"
    assert result.answer_label == "B"
    assert [request.task for request in gateway.requests] == [
        "answer_multiple_choice",
        "repair_multiple_choice_answer",
    ]
    repair_input = gateway.requests[1].input
    assert repair_input["invalid_payload"] == invalid_payload
    assert repair_input["question"] == make_case().question
    assert "validation_error" in repair_input
    assert "gold" not in str(repair_input).lower()


def test_direct_arm_marks_second_schema_failure_terminal():
    invalid = {
        "answer_label": "B",
        "choice_probabilities": {"A": 0.2, "B": 0.8},
        "answer_summary": "Still incomplete.",
    }
    gateway = SequenceGateway([invalid, invalid])

    result = DirectFlashArm(gateway).run_case(make_case())

    assert result.state == "terminal_failed"
    assert result.answer_label is None
    assert result.probabilities is None
    assert result.error_category == "structured_output_invalid"
    assert len(gateway.requests) == 2


def test_direct_arm_repairs_gateway_json_validation_failure_once():
    gateway = SequenceGateway(
        [ModelGatewayValidationError("not valid JSON"), valid_answer()]
    )

    result = DirectFlashArm(gateway).run_case(make_case())

    assert result.state == "completed"
    assert gateway.requests[1].input["invalid_payload"] is None


def test_direct_arm_converts_provider_exception_to_scored_failure():
    gateway = SequenceGateway([RuntimeError("provider unavailable")])

    result = DirectFlashArm(gateway).run_case(make_case())

    assert result.state == "terminal_failed"
    assert result.error_category == "provider_error"
    assert len(gateway.requests) == 1


def test_direct_arm_rejects_non_mapping_payload_before_repair():
    gateway = SequenceGateway([["wrong"], valid_answer()])

    result = DirectFlashArm(gateway).run_case(make_case())

    assert result.state == "completed"
    assert len(gateway.requests) == 2
    assert isinstance(gateway.requests[1].input, Mapping)
