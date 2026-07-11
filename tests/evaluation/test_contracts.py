import math
from dataclasses import FrozenInstanceError

import pytest

from bayesprobe.evaluation.contracts import ArmCaseResult, EvaluationCase


def test_evaluation_case_copies_and_freezes_runtime_question_choices():
    choices = {"A": "First", "B": "Second"}
    case = EvaluationCase(
        sample_id="synthetic_1",
        question="Which option is correct?\nAnswer Choices:\nA. First\nB. Second",
        choices=choices,
    )
    choices["A"] = "mutated"

    assert case.choices == {"A": "First", "B": "Second"}
    assert case.choice_labels == ("A", "B")
    with pytest.raises(FrozenInstanceError):
        case.question = "changed"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"sample_id": "", "question": "q", "choices": {"A": "a", "B": "b"}},
            "sample_id must not be empty",
        ),
        (
            {"sample_id": "s", "question": "", "choices": {"A": "a", "B": "b"}},
            "question must not be empty",
        ),
        (
            {"sample_id": "s", "question": "q", "choices": {"A": "a"}},
            "at least two choices",
        ),
        (
            {"sample_id": "s", "question": "q", "choices": {"": "a", "B": "b"}},
            "choice labels must not be empty",
        ),
        (
            {"sample_id": "s", "question": "q", "choices": {"A": "", "B": "b"}},
            "choice texts must not be empty",
        ),
    ],
)
def test_evaluation_case_rejects_invalid_runtime_data(kwargs, message):
    with pytest.raises(ValueError, match=message):
        EvaluationCase(**kwargs)


def test_completed_arm_result_normalizes_distribution_within_tolerance():
    result = ArmCaseResult(
        sample_id="synthetic_1",
        arm="direct_flash",
        state="completed",
        answer_label="B",
        probabilities={"A": 0.2001, "B": 0.7995},
        answer_summary="B follows from the supplied facts.",
    )

    assert math.fsum(result.probabilities.values()) == pytest.approx(1.0)
    assert result.probabilities["B"] > result.probabilities["A"]
    assert result.is_terminal is True


@pytest.mark.parametrize(
    ("probabilities", "message"),
    [
        ({"A": 0.2, "B": 0.7}, "sum to one"),
        ({"A": -0.1, "B": 1.1}, "finite and between zero and one"),
        ({"A": float("nan"), "B": 1.0}, "finite and between zero and one"),
    ],
)
def test_completed_arm_result_rejects_invalid_probability_distribution(
    probabilities,
    message,
):
    with pytest.raises(ValueError, match=message):
        ArmCaseResult(
            sample_id="synthetic_1",
            arm="direct_flash",
            state="completed",
            answer_label="B",
            probabilities=probabilities,
            answer_summary="Summary.",
        )


def test_completed_arm_result_requires_answer_in_probability_keys():
    with pytest.raises(ValueError, match="answer_label must be a probability key"):
        ArmCaseResult(
            sample_id="synthetic_1",
            arm="direct_flash",
            state="completed",
            answer_label="C",
            probabilities={"A": 0.2, "B": 0.8},
            answer_summary="Summary.",
        )


def test_terminal_failed_result_has_no_fabricated_answer_or_distribution():
    result = ArmCaseResult(
        sample_id="synthetic_1",
        arm="bayesprobe_python",
        state="terminal_failed",
        answer_label=None,
        probabilities=None,
        error_category="provider_timeout",
    )

    assert result.answer_label is None
    assert result.probabilities is None
    assert result.error_category == "provider_timeout"
    assert result.is_terminal is True


def test_terminal_failed_result_rejects_answer_payload():
    with pytest.raises(ValueError, match="terminal_failed result must not contain"):
        ArmCaseResult(
            sample_id="synthetic_1",
            arm="bayesprobe_python",
            state="terminal_failed",
            answer_label="A",
            probabilities={"A": 1.0, "B": 0.0},
            error_category="provider_timeout",
        )


def test_arm_result_rejects_nonterminal_or_unknown_state():
    with pytest.raises(ValueError, match="state must be completed or terminal_failed"):
        ArmCaseResult(
            sample_id="synthetic_1",
            arm="direct_flash",
            state="running",
            answer_label=None,
            probabilities=None,
        )
