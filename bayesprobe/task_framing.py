from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from bayesprobe.schemas import (
    AnswerChoice,
    AnswerContract,
    FramedHypothesis,
    FramingMethod,
    HypothesisFrame,
    HypothesisRelation,
    TaskFrame,
    TaskKind,
)


@dataclass(frozen=True)
class HypothesisSeed:
    statement: str
    id: str | None = None
    scope: str | None = None
    prior: float | None = None
    falsifiers: list[str] = field(default_factory=list)
    predictions: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TaskFramingInput:
    run_id: str
    question: str
    task_context: str = ""
    answer_choices: list[AnswerChoice] = field(default_factory=list)
    hypothesis_seeds: list[HypothesisSeed] = field(default_factory=list)
    task_kind: TaskKind | None = None
    hypothesis_relation: HypothesisRelation | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class TaskFramer(Protocol):
    def frame(self, input: TaskFramingInput) -> TaskFrame:
        raise NotImplementedError


class TaskFramingError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedAnswerChoiceFrame:
    stem: str
    choices: list[AnswerChoice]


_ANSWER_CHOICES_HEADER_RE = re.compile(
    r"(?:\banswer\s+choices?\s*:|答案选项\s*[：:])",
    re.IGNORECASE,
)
_CHOICE_BLOCK_RE = re.compile(
    r"^\s*([A-Z])\s*[\.\)]\s+(.*?)(?=^\s*[A-Z]\s*[\.\)]\s+|\Z)",
    re.MULTILINE | re.DOTALL,
)
_CHOICE_INLINE_RE = re.compile(
    r"(?:^|\s)([A-Z])\s*[\.\)]\s+(.*?)(?=\s+[A-Z]\s*[\.\)]\s+|\Z)",
    re.DOTALL,
)


class ExplicitTaskFramer:
    def can_frame(self, input: TaskFramingInput) -> bool:
        return bool(
            input.answer_choices
            or input.hypothesis_seeds
            or parse_legacy_answer_choice_frame(input.question) is not None
        )

    def frame(self, input: TaskFramingInput) -> TaskFrame:
        parsed = (
            None
            if input.answer_choices
            else parse_legacy_answer_choice_frame(input.question)
        )
        choices = list(input.answer_choices) if input.answer_choices else (
            list(parsed.choices) if parsed is not None else []
        )
        if choices and input.hypothesis_seeds:
            raise TaskFramingError("provide answer choices or hypothesis seeds, not both")
        if choices:
            normalized_question = parsed.stem if parsed is not None else input.question.strip()
            return _frame_choices(input, choices, normalized_question)
        if input.hypothesis_seeds:
            return _frame_seeds(input)
        raise TaskFramingError(
            "unseeded open question requires a model or recorded task framer"
        )


def parse_legacy_answer_choice_frame(
    question: str,
) -> ParsedAnswerChoiceFrame | None:
    header = _ANSWER_CHOICES_HEADER_RE.search(question)
    if header is None:
        return None
    stem = " ".join(question[:header.start()].split())
    choice_text = question[header.end():].strip()
    matches = list(_CHOICE_BLOCK_RE.finditer(choice_text))
    if len(matches) < 2:
        matches = list(_CHOICE_INLINE_RE.finditer(choice_text))
    parsed = [
        AnswerChoice(label=match.group(1), text=" ".join(match.group(2).split()))
        for match in matches
    ]
    if (
        not stem
        or len(parsed) < 2
        or len({choice.label for choice in parsed}) != len(parsed)
    ):
        return None
    return ParsedAnswerChoiceFrame(stem=stem, choices=parsed)


def _frame_choices(
    input: TaskFramingInput,
    choices: list[AnswerChoice],
    normalized_question: str,
) -> TaskFrame:
    ids = [choice.label for choice in choices]
    priors = [1.0 / len(choices)] * len(choices)
    hypotheses = [
        FramedHypothesis(
            id=choice.label,
            statement=f"Answer choice {choice.label} is correct: {choice.text}",
            type="answer_choice",
            scope=(
                f"Assess whether answer choice {choice.label} correctly answers: "
                f"{normalized_question}"
            ),
            initial_prior=prior,
            falsifiers=[
                f"Another answer choice is better supported than {choice.label}.",
                f"A counterexample rules out answer choice {choice.label}.",
            ],
            predictions=[
                f"Reliable reasoning should make answer choice {choice.label} more plausible than its rivals."
            ],
        )
        for choice, prior in zip(choices, priors, strict=True)
    ]
    return TaskFrame(
        task_frame_id=f"{input.run_id}_task_frame",
        task_kind=TaskKind.MULTIPLE_CHOICE,
        normalized_question=normalized_question,
        task_context=input.task_context,
        answer_contract=AnswerContract(
            objective="Select the best-supported answer choice.",
            required_sections=["selected_answer", "justification"],
            decision_form="answer_choice",
        ),
        hypothesis_frame=HypothesisFrame(
            frame_id=f"{input.run_id}_hypothesis_frame",
            relation=HypothesisRelation.EXCLUSIVE_EXHAUSTIVE,
            hypotheses=hypotheses,
            rival_sets=_rival_sets(ids, HypothesisRelation.EXCLUSIVE_EXHAUSTIVE),
            coverage_statement="The listed answer choices are mutually exclusive and collectively exhaustive.",
        ),
        framing_method=FramingMethod.EXPLICIT,
        framing_trace={"source": "answer_choices"},
    )


def _frame_seeds(input: TaskFramingInput) -> TaskFrame:
    seeds = list(input.hypothesis_seeds)
    if len(seeds) < 2:
        raise TaskFramingError("initialization requires at least two rival hypotheses")
    relation = input.hypothesis_relation or HypothesisRelation.EXCLUSIVE_EXHAUSTIVE
    task_kind = input.task_kind or TaskKind.DECISION
    ids = _hypothesis_ids(seeds)
    priors = _initial_priors(seeds, relation)
    hypotheses = [
        FramedHypothesis(
            id=hypothesis_id,
            statement=_required_seed_text(seed.statement, "hypothesis seed statement"),
            type="explicit_seed",
            scope=(
                seed.scope.strip()
                if seed.scope and seed.scope.strip()
                else f"Initial frame for: {input.question.strip()}"
            ),
            initial_prior=prior,
            falsifiers=(
                list(seed.falsifiers)
                or [f"A reliable signal weakens {hypothesis_id} within the problem frame."]
            ),
            predictions=(
                list(seed.predictions)
                or [
                    f"A reliable signal should make {hypothesis_id} more plausible than its rivals."
                ]
            ),
        )
        for seed, hypothesis_id, prior in zip(seeds, ids, priors, strict=True)
    ]
    coverage_statement = (
        "The explicit hypotheses are mutually exclusive and collectively exhaustive."
        if relation == HypothesisRelation.EXCLUSIVE_EXHAUSTIVE
        else "The explicit hypotheses may coexist and do not exhaust all alternatives."
    )
    return TaskFrame(
        task_frame_id=f"{input.run_id}_task_frame",
        task_kind=task_kind,
        normalized_question=input.question.strip(),
        task_context=input.task_context,
        answer_contract=AnswerContract(
            objective="Assess the explicit hypotheses against available evidence.",
            required_sections=["hypotheses", "evidence", "decision"],
            decision_form="hypothesis_assessment",
            permits_synthesis=relation == HypothesisRelation.INDEPENDENT,
        ),
        hypothesis_frame=HypothesisFrame(
            frame_id=f"{input.run_id}_hypothesis_frame",
            relation=relation,
            hypotheses=hypotheses,
            rival_sets=_rival_sets(ids, relation),
            coverage_statement=coverage_statement,
        ),
        framing_method=FramingMethod.EXPLICIT,
        framing_trace={"source": "hypothesis_seeds"},
    )


def _required_seed_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TaskFramingError(f"{field_name} must not be empty")
    return value.strip()


def _hypothesis_ids(seeds: list[HypothesisSeed]) -> list[str]:
    ids: list[str] = []
    used: set[str] = set()
    for index, seed in enumerate(seeds, start=1):
        preferred_id = seed.id.strip() if seed.id and seed.id.strip() else f"H{index}"
        hypothesis_id = preferred_id
        suffix = 2
        while hypothesis_id in used:
            hypothesis_id = f"{preferred_id}_{suffix}"
            suffix += 1
        ids.append(hypothesis_id)
        used.add(hypothesis_id)
    return ids


def _initial_priors(
    seeds: list[HypothesisSeed],
    relation: HypothesisRelation,
) -> list[float]:
    supplied = [seed.prior is not None for seed in seeds]
    if any(supplied) and not all(supplied):
        raise TaskFramingError("seed priors must be supplied for every seed or none")
    if all(supplied):
        priors = [float(seed.prior) for seed in seeds if seed.prior is not None]
    elif relation == HypothesisRelation.EXCLUSIVE_EXHAUSTIVE:
        priors = [1.0 / len(seeds)] * len(seeds)
    else:
        priors = [0.5] * len(seeds)
    if relation == HypothesisRelation.EXCLUSIVE_EXHAUSTIVE and not math.isclose(
        sum(priors), 1.0, abs_tol=1e-6
    ):
        raise TaskFramingError("exclusive seed priors must sum to one")
    if any(prior < 0 or prior > 1 for prior in priors):
        raise TaskFramingError("seed priors must be between zero and one")
    return priors


def _rival_sets(
    ids: list[str],
    relation: HypothesisRelation,
) -> dict[str, list[str]]:
    if relation == HypothesisRelation.EXCLUSIVE_EXHAUSTIVE:
        return {
            hypothesis_id: [other for other in ids if other != hypothesis_id]
            for hypothesis_id in ids
        }
    return {hypothesis_id: [] for hypothesis_id in ids}


__all__ = [
    "ExplicitTaskFramer",
    "HypothesisSeed",
    "ParsedAnswerChoiceFrame",
    "TaskFramer",
    "TaskFramingError",
    "TaskFramingInput",
    "parse_legacy_answer_choice_frame",
]
