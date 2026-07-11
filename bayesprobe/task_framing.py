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
        try:
            _prepare_explicit_input(input)
        except TaskFramingError:
            return False
        return True

    def frame(self, input: TaskFramingInput) -> TaskFrame:
        prepared = _prepare_explicit_input(input)
        if prepared.choices:
            return _frame_choices(
                input,
                prepared.choices,
                prepared.normalized_question,
                prepared.task_context,
            )
        return _frame_seeds(input, prepared)


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


@dataclass(frozen=True)
class _PreparedExplicitInput:
    normalized_question: str
    task_context: str = ""
    choices: list[AnswerChoice] = field(default_factory=list)
    seeds: list[HypothesisSeed] = field(default_factory=list)
    task_kind: TaskKind = TaskKind.DECISION
    relation: HypothesisRelation = HypothesisRelation.EXCLUSIVE_EXHAUSTIVE
    ids: list[str] = field(default_factory=list)
    priors: list[float] = field(default_factory=list)


def _prepare_explicit_input(input: TaskFramingInput) -> _PreparedExplicitInput:
    normalized_question = _required_question(input.question)
    task_context = _normalize_task_context(input.task_context)
    answer_choices = _required_list(input.answer_choices, "answer_choices")
    raw_seeds = _required_list(input.hypothesis_seeds, "hypothesis_seeds")
    _normalize_task_kind(input.task_kind)
    _normalize_hypothesis_relation(input.hypothesis_relation)
    parsed = (
        None
        if answer_choices
        else parse_legacy_answer_choice_frame(normalized_question)
    )
    choices = list(answer_choices) if answer_choices else (
        list(parsed.choices) if parsed is not None else []
    )
    seeds = [_normalize_seed(seed) for seed in raw_seeds]
    if choices and seeds:
        raise TaskFramingError("provide answer choices or hypothesis seeds, not both")
    if choices:
        normalized_question = parsed.stem if parsed is not None else normalized_question
        _validate_choices(choices)
        return _PreparedExplicitInput(
            normalized_question=normalized_question,
            task_context=task_context,
            choices=choices,
            task_kind=TaskKind.MULTIPLE_CHOICE,
        )
    if seeds:
        relation = _normalize_hypothesis_relation(input.hypothesis_relation)
        task_kind = _normalize_task_kind(input.task_kind)
        priors = _validate_seeds(seeds, relation, task_kind)
        return _PreparedExplicitInput(
            normalized_question=normalized_question,
            task_context=task_context,
            seeds=seeds,
            task_kind=task_kind,
            relation=relation,
            ids=_hypothesis_ids(seeds),
            priors=priors,
        )
    raise TaskFramingError(
        "unseeded open question requires a model or recorded task framer"
    )


def _required_question(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TaskFramingError("question must not be empty")
    return value.strip()


def _normalize_task_context(value: str) -> str:
    if not isinstance(value, str):
        raise TaskFramingError("task_context must be a string")
    return value.strip()


def _required_list(value: Any, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TaskFramingError(f"{field_name} must be a list")
    return list(value)


def _normalize_task_kind(value: TaskKind | None) -> TaskKind:
    if value is None:
        return TaskKind.DECISION
    try:
        return TaskKind(value)
    except (TypeError, ValueError) as error:
        raise TaskFramingError("task_kind must be a valid TaskKind") from error


def _normalize_hypothesis_relation(
    value: HypothesisRelation | None,
) -> HypothesisRelation:
    if value is None:
        return HypothesisRelation.EXCLUSIVE_EXHAUSTIVE
    try:
        return HypothesisRelation(value)
    except (TypeError, ValueError) as error:
        raise TaskFramingError(
            "hypothesis_relation must be a valid HypothesisRelation"
        ) from error


def _normalize_seed(value: Any) -> HypothesisSeed:
    if not isinstance(value, HypothesisSeed):
        raise TaskFramingError("hypothesis seeds must be HypothesisSeed instances")
    return HypothesisSeed(
        statement=_required_seed_text(value.statement, "hypothesis seed statement"),
        id=_normalize_seed_id(value.id),
        scope=_normalize_seed_scope(value.scope),
        prior=_normalize_seed_prior(value.prior),
        falsifiers=_normalize_seed_texts(value.falsifiers, "hypothesis seed falsifier"),
        predictions=_normalize_seed_texts(value.predictions, "hypothesis seed prediction"),
    )


def _normalize_seed_id(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise TaskFramingError("hypothesis seed id must be a non-empty string")
    return value.strip()


def _normalize_seed_scope(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TaskFramingError("hypothesis seed scope must be a string")
    return value.strip() or None


def _normalize_seed_prior(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TaskFramingError("hypothesis seed prior must be a finite number")
    prior = float(value)
    if not math.isfinite(prior) or not 0 <= prior <= 1:
        raise TaskFramingError("hypothesis seed prior must be between zero and one")
    return prior


def _normalize_seed_texts(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise TaskFramingError(f"{field_name}s must be a list")
    return [_required_seed_text(item, field_name) for item in value]


def _validate_choices(choices: list[AnswerChoice]) -> None:
    _validate_hypothesis_count(len(choices))
    if not all(isinstance(choice, AnswerChoice) for choice in choices):
        raise TaskFramingError("answer_choices must contain AnswerChoice instances")
    labels = [choice.label for choice in choices]
    if len(labels) != len(set(labels)):
        raise TaskFramingError("answer choice labels must be unique")


def _validate_seeds(
    seeds: list[HypothesisSeed],
    relation: HypothesisRelation,
    task_kind: TaskKind,
) -> list[float]:
    _validate_hypothesis_count(len(seeds))
    statements = [_required_seed_text(seed.statement, "hypothesis seed statement") for seed in seeds]
    if len({_normalized_semantic_text(statement) for statement in statements}) != len(
        statements
    ):
        raise TaskFramingError("hypothesis seed statements must be semantically distinct")
    if (
        task_kind == TaskKind.MULTIPLE_CHOICE
        and relation != HypothesisRelation.EXCLUSIVE_EXHAUSTIVE
    ):
        raise TaskFramingError("multiple-choice tasks require an exclusive frame")
    return _initial_priors(seeds, relation)


def _validate_hypothesis_count(count: int) -> None:
    if not 2 <= count <= 6:
        raise TaskFramingError("explicit framing requires between two and six hypotheses")


def _normalized_semantic_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _frame_choices(
    input: TaskFramingInput,
    choices: list[AnswerChoice],
    normalized_question: str,
    task_context: str,
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
        task_context=task_context,
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


def _frame_seeds(
    input: TaskFramingInput,
    prepared: _PreparedExplicitInput,
) -> TaskFrame:
    seeds = prepared.seeds
    relation = prepared.relation
    task_kind = prepared.task_kind
    ids = prepared.ids
    priors = prepared.priors
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
        normalized_question=prepared.normalized_question,
        task_context=prepared.task_context,
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
