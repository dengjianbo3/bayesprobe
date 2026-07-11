from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol


TERMINAL_ARM_STATES = frozenset({"completed", "terminal_failed"})


@dataclass(frozen=True)
class EvaluationCase:
    sample_id: str
    question: str
    choices: dict[str, str]

    def __post_init__(self) -> None:
        sample_id = _non_empty_string(self.sample_id, "evaluation case sample_id")
        question = _non_empty_string(self.question, "evaluation case question")
        if not isinstance(self.choices, Mapping):
            raise ValueError("evaluation case choices must be an object")
        if len(self.choices) < 2:
            raise ValueError("evaluation case must contain at least two choices")
        choices: dict[str, str] = {}
        for label, text in self.choices.items():
            if not isinstance(label, str) or not label.strip():
                raise ValueError("evaluation case choice labels must not be empty")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("evaluation case choice texts must not be empty")
            normalized_label = label.strip()
            if normalized_label in choices:
                raise ValueError("evaluation case choice labels must be unique")
            choices[normalized_label] = text.strip()
        object.__setattr__(self, "sample_id", sample_id)
        object.__setattr__(self, "question", question)
        object.__setattr__(self, "choices", choices)

    @property
    def choice_labels(self) -> tuple[str, ...]:
        return tuple(self.choices)


@dataclass(frozen=True)
class ArmCaseResult:
    sample_id: str
    arm: str
    state: str
    answer_label: str | None
    probabilities: dict[str, float] | None
    answer_summary: str | None = None
    error_category: str | None = None
    process_metrics: dict[str, int | float | str | bool | None] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sample_id",
            _non_empty_string(self.sample_id, "arm result sample_id"),
        )
        object.__setattr__(self, "arm", _non_empty_string(self.arm, "arm result arm"))
        if self.state not in TERMINAL_ARM_STATES:
            raise ValueError(
                "arm result state must be completed or terminal_failed"
            )
        if not isinstance(self.process_metrics, Mapping):
            raise ValueError("arm result process_metrics must be an object")
        object.__setattr__(self, "process_metrics", dict(self.process_metrics))

        if self.state == "terminal_failed":
            if self.answer_label is not None or self.probabilities is not None:
                raise ValueError(
                    "terminal_failed result must not contain an answer or probabilities"
                )
            if self.error_category is not None:
                object.__setattr__(
                    self,
                    "error_category",
                    _non_empty_string(
                        self.error_category,
                        "terminal_failed result error_category",
                    ),
                )
            return

        answer_label = _non_empty_string(
            self.answer_label,
            "completed result answer_label",
        )
        if not isinstance(self.probabilities, Mapping) or not self.probabilities:
            raise ValueError("completed result probabilities must be a non-empty object")
        probabilities: dict[str, float] = {}
        for label, value in self.probabilities.items():
            if not isinstance(label, str) or not label.strip():
                raise ValueError("completed result probability labels must not be empty")
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(
                    "completed result probabilities must be finite and between zero and one"
                )
            probabilities[label.strip()] = float(value)
        if answer_label not in probabilities:
            raise ValueError("completed result answer_label must be a probability key")
        total = math.fsum(probabilities.values())
        if total <= 0 or abs(total - 1.0) > 1e-3:
            raise ValueError("completed result probabilities must sum to one within 1e-3")
        object.__setattr__(self, "answer_label", answer_label)
        object.__setattr__(
            self,
            "probabilities",
            {label: value / total for label, value in probabilities.items()},
        )
        if self.answer_summary is not None:
            object.__setattr__(
                self,
                "answer_summary",
                _non_empty_string(
                    self.answer_summary,
                    "completed result answer_summary",
                ),
            )
        if self.error_category is not None:
            raise ValueError("completed result must not contain an error_category")

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_ARM_STATES


class ExperimentArm(Protocol):
    def run_case(self, case: EvaluationCase) -> ArmCaseResult:
        ...


def _non_empty_string(value: Any, owner: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{owner} must not be empty")
    return value.strip()


__all__ = [
    "ArmCaseResult",
    "EvaluationCase",
    "ExperimentArm",
    "TERMINAL_ARM_STATES",
]
