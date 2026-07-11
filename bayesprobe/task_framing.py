from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from bayesprobe.model_gateway import (
    ModelGateway,
    ModelGatewayValidationError,
    ModelInvocationTrace,
    StructuredModelRequest,
)
from bayesprobe.schemas import (
    AnswerChoice,
    AnswerContract,
    AnswerRelationship,
    AnswerValueType,
    BeliefState,
    FramedHypothesis,
    FramingMethod,
    HypothesisFrame,
    HypothesisCompetition,
    HypothesisCoverage,
    HypothesisRelation,
    TaskFrame,
    TaskAdmissionDecision,
    TaskAdmissionStatus,
    TaskKind,
    is_forbidden_secret_key_name,
    is_secret_like_value,
    redact_secret_material,
)
from bayesprobe.task_admission import (
    TaskAdmissionError,
    _classify_explicit_frame_material,
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
    admission_decision: TaskAdmissionDecision
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


def validate_task_framing_input_security(input: TaskFramingInput) -> None:
    _reject_caller_secret_material(
        {
            "run_id": input.run_id,
            "question": input.question,
            "task_context": input.task_context,
            "answer_choices": list(input.answer_choices),
            "hypothesis_seeds": list(input.hypothesis_seeds),
            "metadata": input.metadata,
            "admission_decision": input.admission_decision,
        }
    )
    if input.admission_decision.status != TaskAdmissionStatus.ADMITTED:
        raise TaskFramingError("task framing requires an admitted decision")


def _reject_caller_secret_material(value: Any) -> None:
    if isinstance(value, str):
        if is_secret_like_value(value):
            raise TaskFramingError("task framing input must not contain secret material")
        return
    if isinstance(value, TaskAdmissionDecision):
        _reject_caller_secret_material(value.__dict__)
        return
    if isinstance(value, AnswerChoice):
        _reject_caller_secret_identifier(value.label)
        _reject_caller_secret_material(value.label)
        _reject_caller_secret_material(value.text)
        return
    if isinstance(value, HypothesisSeed):
        _reject_caller_secret_identifier(value.id)
        _reject_caller_secret_material(value.id)
        _reject_caller_secret_material(value.statement)
        _reject_caller_secret_material(value.scope)
        _reject_caller_secret_material(value.falsifiers)
        _reject_caller_secret_material(value.predictions)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_caller_secret_identifier(str(key))
            _reject_caller_secret_material(str(key))
            _reject_caller_secret_material(item)
        return
    if isinstance(value, list | tuple):
        for item in value:
            _reject_caller_secret_material(item)


def _reject_caller_secret_identifier(value: str | None) -> None:
    if value is not None and is_forbidden_secret_key_name(value):
        raise TaskFramingError("task framing input must not contain secret material")


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
        try:
            if prepared.choices:
                return _frame_choices(
                    input,
                    prepared.choices,
                    prepared.normalized_question,
                    prepared.task_context,
                )
            return _frame_seeds(input, prepared)
        except TaskFramingError:
            raise
        except ValueError:
            raise TaskFramingError("invalid explicit task frame fields") from None


@dataclass(frozen=True)
class TaskFramingRepairPolicy:
    max_attempts: int = 1
    repair_task: str = "repair_task_frame"

    def __post_init__(self) -> None:
        if type(self.max_attempts) is not int:
            raise ValueError("task framing repair max_attempts must be an integer")
        if self.max_attempts < 0:
            raise ValueError("task framing repair max_attempts must be non-negative")
        if self.max_attempts > 1:
            raise ValueError("task framing repair max_attempts permits at most one repair")
        if not isinstance(self.repair_task, str):
            raise ValueError("task framing repair task must be a string")
        if not self.repair_task.strip():
            raise ValueError("task framing repair task must not be empty")


class ModelTaskFramer:
    def __init__(
        self,
        model_gateway: ModelGateway,
        repair_policy: TaskFramingRepairPolicy | None = None,
    ) -> None:
        self._model_gateway = model_gateway
        self._repair_policy = repair_policy or TaskFramingRepairPolicy()

    def frame(self, input: TaskFramingInput) -> TaskFrame:
        validate_task_framing_input_security(input)
        request = _open_frame_request(input)
        payload = self._complete_structured(request)
        try:
            return task_frame_from_mapping(
                payload,
                run_id=input.run_id,
                question=input.question,
                task_context=input.task_context,
                admission_decision=input.admission_decision,
                method=FramingMethod.MODEL,
                trace=_trace_for(request, self._model_gateway),
            )
        except (ValueError, ModelGatewayValidationError) as error:
            return self._repair_or_raise(input, request, payload, error)

    def _complete_structured(
        self,
        request: StructuredModelRequest,
    ) -> dict[str, Any]:
        try:
            return self._model_gateway.complete_structured(request)
        except Exception:
            pass
        raise TaskFramingError("task framing model gateway call failed") from None

    def _repair_or_raise(
        self,
        input: TaskFramingInput,
        original_request: StructuredModelRequest,
        invalid_payload: Any,
        validation_error: ValueError,
    ) -> TaskFrame:
        last_error = validation_error
        for attempt_index in range(1, self._repair_policy.max_attempts + 1):
            request = _repair_frame_request(
                input,
                original_request,
                invalid_payload,
                last_error,
                attempt_index,
                self._repair_policy,
            )
            payload = self._complete_structured(request)
            try:
                return task_frame_from_mapping(
                    payload,
                    run_id=input.run_id,
                    question=input.question,
                    task_context=input.task_context,
                    admission_decision=input.admission_decision,
                    method=FramingMethod.MODEL,
                    trace=_trace_for(request, self._model_gateway),
                )
            except (ValueError, ModelGatewayValidationError) as error:
                last_error = error
                invalid_payload = payload
        raise TaskFramingError(
            "task frame invalid after "
            f"{self._repair_policy.max_attempts} repair attempt"
        ) from None


class RecordedTaskFramer:
    def __init__(self, frame: TaskFrame) -> None:
        self._frame = frame.model_copy(deep=True)

    def frame(self, input: TaskFramingInput) -> TaskFrame:
        validate_task_framing_input_security(input)
        try:
            run_id = _required_seed_text(input.run_id, "run_id")
            question = _required_question(input.question)
            task_context = _normalize_task_context(input.task_context)
        except (TaskFramingError, TypeError, ValueError):
            raise TaskFramingError("invalid recorded task framing input") from None

        try:
            source = _strict_recorded_task_frame(self._frame)
            answer_relationship = source.answer_relationship or (
                AnswerRelationship.SYNTHESIS
                if source.answer_contract.permits_synthesis
                else AnswerRelationship.SELECTION
            )
            answer_contract = _canonicalize_native_frame(
                task_kind=source.task_kind,
                answer_relationship=answer_relationship,
                answer_contract=source.answer_contract,
                competition=source.hypothesis_frame.competition,
                coverage=source.hypothesis_frame.coverage,
                unresolved_alternative_mass=(
                    source.hypothesis_frame.unresolved_alternative_mass
                ),
                hypotheses=source.hypothesis_frame.hypotheses,
                admission_decision=input.admission_decision,
            )
            hypothesis_frame = HypothesisFrame(
                frame_id=f"{run_id}_hypothesis_frame",
                competition=source.hypothesis_frame.competition,
                coverage=source.hypothesis_frame.coverage,
                hypotheses=source.hypothesis_frame.hypotheses,
                rival_sets=source.hypothesis_frame.rival_sets,
                coverage_statement=source.hypothesis_frame.coverage_statement,
                unresolved_alternative_mass=(
                    source.hypothesis_frame.unresolved_alternative_mass
                ),
                coverage_limitation=source.hypothesis_frame.coverage_limitation,
            )
            return TaskFrame(
                schema_version="v0.2",
                task_frame_id=f"{run_id}_task_frame",
                admission_decision_id=input.admission_decision.attempt_id,
                task_kind=source.task_kind,
                answer_relationship=answer_relationship,
                normalized_question=question,
                task_context=task_context,
                answer_contract=answer_contract,
                hypothesis_frame=hypothesis_frame,
                framing_method=FramingMethod.RECORDED,
                framing_trace={
                    "metadata": {"run_id": run_id},
                    "recorded_from_task_frame_id": source.task_frame_id,
                    "source_framing_method": source.framing_method.value,
                    "source_trace": redact_secret_material(source.framing_trace),
                },
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            raise TaskFramingError("invalid recorded task frame") from None


class RoutingTaskFramer:
    def __init__(
        self,
        *,
        explicit_framer: ExplicitTaskFramer,
        open_framer: TaskFramer,
    ) -> None:
        self._explicit_framer = explicit_framer
        self._open_framer = open_framer

    def frame(self, input: TaskFramingInput) -> TaskFrame:
        parsed = (
            None
            if input.answer_choices
            else parse_legacy_answer_choice_frame(input.question)
        )
        choices = input.answer_choices if parsed is None else parsed.choices
        task_kind = input.task_kind or input.admission_decision.proposed_task_kind
        try:
            mode = _classify_explicit_frame_material(
                answer_choices=choices,
                hypothesis_seeds=input.hypothesis_seeds,
                task_kind=task_kind,
            )
        except TaskAdmissionError as error:
            raise TaskFramingError(str(error)) from None
        if mode is not None:
            return self._explicit_framer.frame(input)
        return self._open_framer.frame(input)


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


_TASK_FRAME_FIELDS = {
    "task_kind",
    "answer_relationship",
    "answer_contract",
    "competition",
    "coverage",
    "hypotheses",
    "coverage_statement",
    "coverage_limitation",
}
_HYPOTHESIS_FIELDS = {
    "statement",
    "type",
    "scope",
    "falsifiers",
    "predictions",
    "answer_value",
}
_ANSWER_CONTRACT_FIELDS = {
    "objective",
    "answer_value_type",
    "answer_format",
    "required_sections",
    "decision_form",
    "permits_synthesis",
}
_RECORDED_TASK_FRAME_FIELDS = {
    "schema_version",
    "task_frame_id",
    "admission_decision_id",
    "task_kind",
    "answer_relationship",
    "normalized_question",
    "task_context",
    "answer_contract",
    "hypothesis_frame",
    "framing_method",
    "framing_trace",
}
_RECORDED_HYPOTHESIS_FRAME_FIELDS = {
    "frame_id",
    "competition",
    "coverage",
    "hypotheses",
    "rival_sets",
    "coverage_statement",
    "unresolved_alternative_mass",
    "coverage_limitation",
}
_RECORDED_FRAMED_HYPOTHESIS_FIELDS = {
    "id",
    "statement",
    "type",
    "scope",
    "initial_prior",
    "falsifiers",
    "predictions",
    "answer_value",
}
_PROVIDER_BELIEF_FIELDS = {
    "id",
    "prior",
    "posterior",
    "api_key",
    "authorization",
    "token",
}


def _open_frame_request(input: TaskFramingInput) -> StructuredModelRequest:
    question = _required_question(input.question)
    task_context = _normalize_task_context(input.task_context)
    return StructuredModelRequest(
        task="frame_open_question",
        input={
            "question": question,
            "task_context": task_context,
            "supported_task_kinds": [
                kind.value
                for kind in TaskKind
                if kind != TaskKind.MULTIPLE_CHOICE
            ],
            "admitted_task_kind": input.admission_decision.proposed_task_kind.value,
            "supported_competition": [item.value for item in HypothesisCompetition],
            "supported_coverage": [item.value for item in HypothesisCoverage],
            "hypothesis_count": {"minimum": 1, "maximum": 6},
        },
        prompt_id="open_question_task_framing",
        prompt_version="v0.2",
        schema_name="OpenQuestionTaskFrame",
        schema_version="v0.2",
        metadata={"run_id": input.run_id},
    )


def _repair_frame_request(
    input: TaskFramingInput,
    original_request: StructuredModelRequest,
    invalid_payload: Any,
    validation_error: ValueError,
    attempt_index: int,
    repair_policy: TaskFramingRepairPolicy,
) -> StructuredModelRequest:
    return StructuredModelRequest(
        task=repair_policy.repair_task,
        input={
            "original_request": redact_secret_material(original_request.input),
            "invalid_payload": redact_secret_material(invalid_payload),
            "validation_error": redact_secret_material(str(validation_error)),
            "attempt_index": attempt_index,
            "required_fields": {
                "task_frame": sorted(_TASK_FRAME_FIELDS),
                "hypothesis": sorted(_HYPOTHESIS_FIELDS),
            },
            "supported_task_kinds": [
                kind.value
                for kind in TaskKind
                if kind != TaskKind.MULTIPLE_CHOICE
            ],
            "admitted_task_kind": input.admission_decision.proposed_task_kind.value,
            "supported_competition": [item.value for item in HypothesisCompetition],
            "supported_coverage": [item.value for item in HypothesisCoverage],
            "hypothesis_count": {"minimum": 1, "maximum": 6},
        },
        prompt_id="open_question_task_framing_repair",
        prompt_version="v0.2",
        schema_name="OpenQuestionTaskFrame",
        schema_version="v0.2",
        metadata={"run_id": input.run_id, "repair_attempt_index": attempt_index},
    )


def _trace_for(
    request: StructuredModelRequest,
    model_gateway: ModelGateway,
) -> dict[str, Any]:
    adapter_kind = getattr(model_gateway, "adapter_kind", type(model_gateway).__name__)
    if not isinstance(adapter_kind, str) or not adapter_kind.strip():
        adapter_kind = type(model_gateway).__name__
    return redact_secret_material(
        ModelInvocationTrace.from_request(
            request,
            adapter_kind=adapter_kind,
        ).to_dict()
    )


def task_frame_from_mapping(
    payload: Mapping[str, Any],
    *,
    run_id: str,
    question: str,
    task_context: str,
    admission_decision: TaskAdmissionDecision,
    method: FramingMethod,
    trace: dict[str, Any],
) -> TaskFrame:
    if not isinstance(payload, Mapping):
        raise TaskFramingError("task frame payload must be an object")
    if set(payload) != _TASK_FRAME_FIELDS:
        raise TaskFramingError("task frame payload has missing or unknown fields")

    raw_hypotheses = payload.get("hypotheses")
    if not isinstance(raw_hypotheses, list):
        raise TaskFramingError("task frame hypotheses must be a list")
    for item in raw_hypotheses:
        if not isinstance(item, Mapping):
            raise TaskFramingError("each framed hypothesis must be an object")
        if _PROVIDER_BELIEF_FIELDS.intersection(item):
            raise TaskFramingError("provider hypotheses cannot assign ids or beliefs")
        if set(item) != _HYPOTHESIS_FIELDS:
            raise TaskFramingError("provider hypothesis has missing or unknown fields")

    try:
        if (
            not isinstance(payload["task_kind"], str)
            or not isinstance(payload["answer_relationship"], str)
            or not isinstance(payload["competition"], str)
            or not isinstance(payload["coverage"], str)
        ):
            raise ValueError
        task_kind = TaskKind(payload["task_kind"])
        answer_relationship = AnswerRelationship(payload["answer_relationship"])
        competition = HypothesisCompetition(payload["competition"])
        coverage = HypothesisCoverage(payload["coverage"])
    except (KeyError, TypeError, ValueError):
        raise TaskFramingError("invalid task frame classification") from None
    if task_kind == TaskKind.MULTIPLE_CHOICE:
        raise TaskFramingError("model framing cannot create a multiple-choice task")
    if (
        admission_decision.status != TaskAdmissionStatus.ADMITTED
        or admission_decision.proposed_task_kind != task_kind
    ):
        raise TaskFramingError("task frame must match the admitted task kind")
    _validate_native_hypothesis_count(task_kind, len(raw_hypotheses))

    ids = [f"H{index}" for index in range(1, len(raw_hypotheses) + 1)]
    unresolved_mass = None
    if competition == HypothesisCompetition.EXCLUSIVE:
        if coverage == HypothesisCoverage.OPEN:
            unresolved_mass = 0.50
            priors = _exclusive_open_priors(len(ids), unresolved=unresolved_mass)
        else:
            priors = [1.0 / len(ids)] * len(ids)
    else:
        priors = [0.5] * len(ids)
    answer_contract = _answer_contract_from_mapping(payload.get("answer_contract"))
    normalized_hypotheses = [
        {
            "statement": _native_required_text(item["statement"], "statement"),
            "type": _native_required_text(item["type"], "type"),
            "scope": _native_required_text(item["scope"], "scope"),
            "falsifiers": _native_required_text_list(
                item["falsifiers"], "falsifiers"
            ),
            "predictions": _native_required_text_list(
                item["predictions"], "predictions"
            ),
            "answer_value": _native_answer_value(item["answer_value"]),
        }
        for item in raw_hypotheses
    ]
    answer_contract = _canonicalize_native_frame(
        task_kind=task_kind,
        answer_relationship=answer_relationship,
        answer_contract=answer_contract,
        competition=competition,
        coverage=coverage,
        unresolved_alternative_mass=unresolved_mass,
        hypotheses=normalized_hypotheses,
        admission_decision=admission_decision,
    )
    coverage_statement = _native_required_text(
        payload["coverage_statement"], "coverage_statement"
    )
    coverage_limitation = payload["coverage_limitation"]
    if coverage_limitation is not None:
        coverage_limitation = _native_required_text(
            coverage_limitation, "coverage_limitation"
        )
    hypotheses = [
        FramedHypothesis(
            id=ids[index],
            statement=item["statement"],
            type=item["type"],
            scope=item["scope"],
            initial_prior=priors[index],
            falsifiers=item["falsifiers"],
            predictions=item["predictions"],
            answer_value=item["answer_value"],
        )
        for index, item in enumerate(normalized_hypotheses)
    ]
    try:
        return TaskFrame(
            schema_version="v0.2",
            task_frame_id=f"{run_id}_task_frame",
            admission_decision_id=admission_decision.attempt_id,
            task_kind=task_kind,
            answer_relationship=answer_relationship,
            normalized_question=question.strip(),
            task_context=task_context.strip(),
            answer_contract=answer_contract,
            hypothesis_frame=HypothesisFrame(
                frame_id=f"{run_id}_hypothesis_frame",
                competition=competition,
                coverage=coverage,
                hypotheses=hypotheses,
                rival_sets={
                    hypothesis_id: [other for other in ids if other != hypothesis_id]
                    if competition == HypothesisCompetition.EXCLUSIVE
                    else []
                    for hypothesis_id in ids
                },
                coverage_statement=coverage_statement,
                unresolved_alternative_mass=unresolved_mass,
                coverage_limitation=coverage_limitation,
            ),
            framing_method=method,
            framing_trace=redact_secret_material(trace),
        )
    except ValueError:
        raise TaskFramingError("invalid task frame fields") from None


def _native_required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TaskFramingError(f"provider {field_name} must be a non-empty string")
    return value.strip()


def _native_required_text_list(value: Any, field_name: str) -> list[str]:
    if type(value) is not list or not value:
        raise TaskFramingError(
            f"provider {field_name} must be a non-empty list of non-empty strings"
        )
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise TaskFramingError(
            f"provider {field_name} must be a non-empty list of non-empty strings"
        )
    return [item.strip() for item in value]


def _native_answer_value(value: Any) -> str | int | float | None:
    if value is not None and type(value) not in {str, int, float}:
        raise TaskFramingError("provider answer_value must be scalar or null")
    if isinstance(value, str) and not value.strip():
        raise TaskFramingError("provider answer_value must not be empty")
    return value.strip() if isinstance(value, str) else value


def _answer_value_matches_type(
    value: str | int | float,
    answer_value_type: AnswerValueType,
) -> bool:
    if answer_value_type == AnswerValueType.INTEGER:
        return type(value) is int
    if answer_value_type == AnswerValueType.NUMBER:
        return type(value) in {int, float} and math.isfinite(value)
    if answer_value_type in {
        AnswerValueType.CHOICE_LABEL,
        AnswerValueType.SHORT_TEXT,
        AnswerValueType.STRUCTURED_TEXT,
    }:
        return type(value) is str
    return False


def _validate_native_hypothesis_count(task_kind: TaskKind, count: int) -> None:
    if task_kind == TaskKind.EXACT_ANSWER:
        if not 1 <= count <= 6:
            raise TaskFramingError(
                "exclusive-open framing requires one to six candidates"
            )
        return
    if not 2 <= count <= 6:
        raise TaskFramingError("task frame must contain between 2 and 6 hypotheses")


def _canonicalize_native_frame(
    *,
    task_kind: TaskKind,
    answer_relationship: AnswerRelationship,
    answer_contract: AnswerContract,
    competition: HypothesisCompetition,
    coverage: HypothesisCoverage,
    unresolved_alternative_mass: float | None,
    hypotheses: list[FramedHypothesis] | list[dict[str, Any]],
    admission_decision: TaskAdmissionDecision,
) -> AnswerContract:
    if (
        admission_decision.status != TaskAdmissionStatus.ADMITTED
        or admission_decision.proposed_task_kind != task_kind
    ):
        raise TaskFramingError("task frame must match the admitted task kind")
    outline = admission_decision.answer_contract_outline
    if outline is None:
        raise TaskFramingError("admitted task requires an answer contract outline")
    if answer_contract.answer_value_type != outline.answer_value_type:
        raise TaskFramingError(
            "task frame answer value type must match the admitted contract"
        )
    if _normalized_semantic_text(answer_contract.decision_form) != (
        _normalized_semantic_text(outline.decision_form)
    ):
        raise TaskFramingError(
            "task frame decision form must match the admitted contract"
        )
    if answer_contract.permits_synthesis is not outline.permits_synthesis:
        raise TaskFramingError(
            "task frame synthesis permission must match the admitted contract"
        )
    if task_kind == TaskKind.EXACT_ANSWER:
        framed_sections = {
            _normalized_semantic_text(section)
            for section in answer_contract.required_sections
        }
        admitted_sections = {
            _normalized_semantic_text(section) for section in outline.required_sections
        }
        if not admitted_sections <= framed_sections:
            raise TaskFramingError(
                "task frame must preserve all admitted required sections"
            )
    if (
        answer_relationship == AnswerRelationship.SYNTHESIS
        and not answer_contract.permits_synthesis
    ):
        raise TaskFramingError(
            "task frame synthesis relationship requires synthesis permission"
        )

    _validate_native_hypothesis_count(task_kind, len(hypotheses))
    if task_kind != TaskKind.EXACT_ANSWER:
        return answer_contract.model_copy(update={"objective": outline.objective})
    if answer_relationship != AnswerRelationship.SELECTION:
        raise TaskFramingError("exact-answer framing requires answer selection")
    if (
        competition != HypothesisCompetition.EXCLUSIVE
        or coverage != HypothesisCoverage.OPEN
    ):
        raise TaskFramingError("exact-answer framing requires exclusive-open coverage")
    if unresolved_alternative_mass != 0.50:
        raise TaskFramingError(
            "exact-answer framing requires initial unresolved mass 0.50"
        )
    answer_values = [
        hypothesis.answer_value
        if isinstance(hypothesis, FramedHypothesis)
        else hypothesis["answer_value"]
        for hypothesis in hypotheses
    ]
    if any(value is None for value in answer_values):
        raise TaskFramingError("exact-answer candidates require answer_value")
    if any(
        not _answer_value_matches_type(value, answer_contract.answer_value_type)
        for value in answer_values
        if value is not None
    ):
        raise TaskFramingError(
            "exact-answer candidate values must match answer_value_type"
        )
    if len({(type(value).__name__, value) for value in answer_values}) != len(
        answer_values
    ):
        raise TaskFramingError("exact-answer candidate values must be unique")
    return answer_contract.model_copy(update={"objective": outline.objective})


def _exclusive_open_priors(
    count: int,
    *,
    unresolved: float = 0.50,
) -> list[float]:
    if not 1 <= count <= 6:
        raise TaskFramingError("exclusive-open framing requires one to six candidates")
    if not 0.05 <= unresolved < 1.0:
        raise TaskFramingError("exclusive-open unresolved reserve must be at least 0.05")
    return [round((1.0 - unresolved) / count, 12) for _ in range(count)]


def _answer_contract_from_mapping(payload: Any) -> AnswerContract:
    if not isinstance(payload, Mapping):
        raise TaskFramingError("answer_contract must be an object")
    if set(payload) != _ANSWER_CONTRACT_FIELDS:
        raise TaskFramingError("answer_contract has missing or unknown fields")
    objective = _native_required_text(
        payload["objective"],
        "answer_contract objective",
    )
    decision_form = _native_required_text(
        payload["decision_form"],
        "answer_contract decision_form",
    )
    answer_format = _native_required_text(
        payload["answer_format"],
        "answer_contract answer_format",
    )
    try:
        if not isinstance(payload["answer_value_type"], str):
            raise ValueError
        answer_value_type = AnswerValueType(payload["answer_value_type"])
    except (KeyError, TypeError, ValueError):
        raise TaskFramingError(
            "provider answer_contract answer_value_type is invalid"
        ) from None
    required_sections = _native_required_text_list(
        payload["required_sections"],
        "answer_contract required_sections",
    )
    if len(required_sections) != len(set(required_sections)):
        raise TaskFramingError(
            "provider answer_contract required_sections must be unique"
        )
    permits_synthesis = payload["permits_synthesis"]
    if type(permits_synthesis) is not bool:
        raise TaskFramingError(
            "provider answer_contract permits_synthesis must be a boolean"
        )
    return AnswerContract(
        objective=objective,
        answer_value_type=answer_value_type,
        answer_format=answer_format,
        required_sections=required_sections,
        decision_form=decision_form,
        permits_synthesis=permits_synthesis,
    )


def _strict_recorded_answer_contract(value: Any) -> AnswerContract:
    payload = _recorded_model_payload(
        value,
        AnswerContract,
        {
            "objective",
            "answer_value_type",
            "answer_format",
            "required_sections",
            "decision_form",
            "permits_synthesis",
        },
        "answer contract",
    )
    if type(payload["answer_value_type"]) is not AnswerValueType:
        raise TaskFramingError(
            "recorded answer_value_type must be an AnswerValueType"
        )
    if type(payload["permits_synthesis"]) is not bool:
        raise TaskFramingError("recorded permits_synthesis must be a boolean")
    required_sections = _native_required_text_list(
        payload["required_sections"], "recorded required_sections"
    )
    if len(required_sections) != len(set(required_sections)):
        raise TaskFramingError("recorded required_sections must be unique")
    return AnswerContract(
        objective=_native_required_text(payload["objective"], "recorded objective"),
        answer_value_type=payload["answer_value_type"],
        answer_format=_native_required_text(
            payload["answer_format"], "recorded answer_format"
        ),
        required_sections=required_sections,
        decision_form=_native_required_text(
            payload["decision_form"], "recorded decision_form"
        ),
        permits_synthesis=payload["permits_synthesis"],
    )


def _strict_recorded_task_frame(frame: TaskFrame) -> TaskFrame:
    payload = _recorded_model_payload(
        frame,
        TaskFrame,
        _RECORDED_TASK_FRAME_FIELDS,
        "task frame",
    )
    if type(payload["task_kind"]) is not TaskKind:
        raise TaskFramingError("recorded task_kind must be a TaskKind")
    if payload["schema_version"] not in {"v0.1", "v0.2"}:
        raise TaskFramingError("recorded schema_version must be v0.1 or v0.2")
    if payload["schema_version"] == "v0.1":
        if payload["admission_decision_id"] is not None:
            raise TaskFramingError("recorded v0.1 admission_decision_id must be null")
        if payload["answer_relationship"] is not None:
            raise TaskFramingError("recorded v0.1 answer_relationship must be null")
    else:
        _native_required_text(
            payload["admission_decision_id"],
            "recorded admission_decision_id",
        )
        if type(payload["answer_relationship"]) is not AnswerRelationship:
            raise TaskFramingError(
                "recorded answer_relationship must be an AnswerRelationship"
            )
    if type(payload["framing_method"]) is not FramingMethod:
        raise TaskFramingError("recorded framing_method must be a FramingMethod")
    if type(payload["task_context"]) is not str:
        raise TaskFramingError("recorded task_context must be a string")
    if type(payload["framing_trace"]) is not dict:
        raise TaskFramingError("recorded framing_trace must be an object")

    raw_contract = payload["answer_contract"]
    contract_payload = _strict_recorded_answer_contract(raw_contract)
    return TaskFrame(
        schema_version=payload["schema_version"],
        task_frame_id=_native_required_text(
            payload["task_frame_id"], "recorded task_frame_id"
        ),
        admission_decision_id=payload["admission_decision_id"],
        task_kind=payload["task_kind"],
        answer_relationship=payload["answer_relationship"],
        normalized_question=_native_required_text(
            payload["normalized_question"], "recorded normalized_question"
        ),
        task_context=payload["task_context"],
        answer_contract=contract_payload,
        hypothesis_frame=_strict_recorded_hypothesis_frame(
            payload["hypothesis_frame"]
        ),
        framing_method=payload["framing_method"],
        framing_trace=payload["framing_trace"],
    )


def _strict_recorded_hypothesis_frame(value: Any) -> HypothesisFrame:
    payload = _recorded_model_payload(
        value,
        HypothesisFrame,
        _RECORDED_HYPOTHESIS_FRAME_FIELDS,
        "hypothesis frame",
    )
    if type(payload["competition"]) is not HypothesisCompetition:
        raise TaskFramingError(
            "recorded hypothesis frame competition must be a HypothesisCompetition"
        )
    if type(payload["coverage"]) is not HypothesisCoverage:
        raise TaskFramingError(
            "recorded hypothesis frame coverage must be a HypothesisCoverage"
        )
    if type(payload["hypotheses"]) is not list:
        raise TaskFramingError("recorded hypotheses must be a list")
    if type(payload["rival_sets"]) is not dict:
        raise TaskFramingError("recorded rival_sets must be an object")

    rival_sets: dict[str, list[str]] = {}
    for hypothesis_id, rivals in payload["rival_sets"].items():
        if not isinstance(hypothesis_id, str) or not hypothesis_id.strip():
            raise TaskFramingError("recorded rival_sets keys must be strings")
        if type(rivals) is not list or any(
            not isinstance(rival, str) or not rival.strip() for rival in rivals
        ):
            raise TaskFramingError("recorded rival_sets values must be string lists")
        rival_sets[hypothesis_id] = list(rivals)

    unresolved_mass = payload["unresolved_alternative_mass"]
    if unresolved_mass is not None and (
        type(unresolved_mass) is not float
        or not math.isfinite(unresolved_mass)
        or not 0 <= unresolved_mass <= 1
    ):
        raise TaskFramingError(
            "recorded unresolved_alternative_mass must be a float or null"
        )
    coverage_limitation = payload["coverage_limitation"]
    if coverage_limitation is not None:
        coverage_limitation = _native_required_text(
            coverage_limitation, "recorded coverage_limitation"
        )

    return HypothesisFrame(
        frame_id=_native_required_text(payload["frame_id"], "recorded frame_id"),
        competition=payload["competition"],
        coverage=payload["coverage"],
        hypotheses=[
            _strict_recorded_framed_hypothesis(item)
            for item in payload["hypotheses"]
        ],
        rival_sets=rival_sets,
        coverage_statement=_native_required_text(
            payload["coverage_statement"], "recorded coverage_statement"
        ),
        unresolved_alternative_mass=unresolved_mass,
        coverage_limitation=coverage_limitation,
    )


def _strict_recorded_framed_hypothesis(value: Any) -> FramedHypothesis:
    payload = _recorded_model_payload(
        value,
        FramedHypothesis,
        _RECORDED_FRAMED_HYPOTHESIS_FIELDS,
        "framed hypothesis",
    )
    initial_prior = payload["initial_prior"]
    if (
        type(initial_prior) is not float
        or not math.isfinite(initial_prior)
        or not 0 <= initial_prior <= 1
    ):
        raise TaskFramingError("recorded initial_prior must be a probability float")
    answer_value = payload["answer_value"]
    if answer_value is not None and type(answer_value) not in {str, int, float}:
        raise TaskFramingError("recorded answer_value must be scalar or null")
    return FramedHypothesis(
        id=_native_required_text(payload["id"], "recorded hypothesis id"),
        statement=_native_required_text(
            payload["statement"], "recorded hypothesis statement"
        ),
        type=_native_required_text(payload["type"], "recorded hypothesis type"),
        scope=_native_required_text(payload["scope"], "recorded hypothesis scope"),
        initial_prior=initial_prior,
        falsifiers=_native_required_text_list(
            payload["falsifiers"], "recorded hypothesis falsifiers"
        ),
        predictions=_native_required_text_list(
            payload["predictions"], "recorded hypothesis predictions"
        ),
        answer_value=answer_value,
    )


def _recorded_model_payload(
    value: Any,
    model_type: type[Any],
    exact_fields: set[str],
    label: str,
) -> dict[str, Any]:
    if isinstance(value, model_type):
        payload = dict(value.__dict__)
    elif type(value) is dict:
        payload = dict(value)
    else:
        raise TaskFramingError(f"recorded {label} must be an object")
    if set(payload) != exact_fields:
        raise TaskFramingError(f"recorded {label} has missing or unknown fields")
    return payload


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
    validate_task_framing_input_security(input)
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
        if input.admission_decision.proposed_task_kind != TaskKind.MULTIPLE_CHOICE:
            raise TaskFramingError("explicit choices require multiple-choice admission")
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
        admitted_kind = input.admission_decision.proposed_task_kind
        if input.task_kind is None:
            task_kind = admitted_kind
        if admitted_kind != task_kind:
            raise TaskFramingError("explicit frame must match the admitted task kind")
        try:
            seed_mode = _classify_explicit_frame_material(
                answer_choices=[],
                hypothesis_seeds=raw_seeds,
                task_kind=task_kind,
            )
        except TaskAdmissionError as error:
            raise TaskFramingError(str(error)) from None
        if seed_mode != "seeds":
            raise TaskFramingError("invalid explicit hypothesis frame")
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
            answer_value=choice.label,
        )
        for choice, prior in zip(choices, priors, strict=True)
    ]
    return TaskFrame(
        schema_version="v0.2",
        task_frame_id=f"{input.run_id}_task_frame",
        admission_decision_id=input.admission_decision.attempt_id,
        task_kind=TaskKind.MULTIPLE_CHOICE,
        answer_relationship=AnswerRelationship.SELECTION,
        normalized_question=normalized_question,
        task_context=task_context,
        answer_contract=_contract_from_admission(
            input.admission_decision,
            answer_format="choice label",
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
        schema_version="v0.2",
        task_frame_id=f"{input.run_id}_task_frame",
        admission_decision_id=input.admission_decision.attempt_id,
        task_kind=task_kind,
        answer_relationship=(
            AnswerRelationship.SYNTHESIS
            if input.admission_decision.answer_contract_outline.permits_synthesis
            else AnswerRelationship.SELECTION
        ),
        normalized_question=prepared.normalized_question,
        task_context=prepared.task_context,
        answer_contract=_contract_from_admission(
            input.admission_decision,
            answer_format="structured text",
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


def _contract_from_admission(
    admission_decision: TaskAdmissionDecision,
    *,
    answer_format: str,
) -> AnswerContract:
    outline = admission_decision.answer_contract_outline
    if outline is None:
        raise TaskFramingError("admitted task requires an answer contract outline")
    return AnswerContract(
        objective=outline.objective,
        answer_value_type=outline.answer_value_type,
        answer_format=answer_format,
        required_sections=list(outline.required_sections),
        decision_form=outline.decision_form,
        permits_synthesis=outline.permits_synthesis,
    )


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


def migrate_legacy_belief_state(state: BeliefState) -> BeliefState:
    if state.schema_version == "v0.2":
        return state
    from bayesprobe.migrations import migrate_belief_state_v0_1

    return migrate_belief_state_v0_1(state)


__all__ = [
    "ExplicitTaskFramer",
    "HypothesisSeed",
    "ModelTaskFramer",
    "ParsedAnswerChoiceFrame",
    "RecordedTaskFramer",
    "RoutingTaskFramer",
    "TaskFramer",
    "TaskFramingError",
    "TaskFramingInput",
    "TaskFramingRepairPolicy",
    "migrate_legacy_belief_state",
    "parse_legacy_answer_choice_frame",
    "task_frame_from_mapping",
    "validate_task_framing_input_security",
]
