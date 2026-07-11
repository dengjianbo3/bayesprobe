from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from bayesprobe.model_gateway import (
    ModelGateway,
    ModelInvocationTrace,
    StructuredModelRequest,
)
from bayesprobe.schemas import (
    AnswerChoice,
    AnswerContractOutline,
    AnswerValueType,
    CapabilityDescriptor,
    TaskAdmissionDecision,
    TaskAdmissionStatus,
    TaskKind,
    is_forbidden_secret_key_name,
    is_secret_like_value,
    redact_secret_material,
)

if TYPE_CHECKING:
    from bayesprobe.task_framing import HypothesisSeed


@dataclass(frozen=True)
class TaskAdmissionInput:
    attempt_id: str
    question: str
    task_context: str = ""
    answer_choices: list[AnswerChoice] = field(default_factory=list)
    hypothesis_seeds: list[HypothesisSeed] = field(default_factory=list)
    requested_output_shape: str | None = None
    available_capabilities: list[CapabilityDescriptor] = field(default_factory=list)
    model_metadata: dict[str, Any] = field(default_factory=dict)


class TaskAdmitter(Protocol):
    def assess(self, input: TaskAdmissionInput) -> TaskAdmissionDecision:
        raise NotImplementedError


class TaskAdmissionError(ValueError):
    pass


class ExplicitTaskAdmitter:
    def can_assess(self, input: TaskAdmissionInput) -> bool:
        return bool(input.answer_choices or input.hypothesis_seeds)

    def assess(self, input: TaskAdmissionInput) -> TaskAdmissionDecision:
        _validate_admission_input(input)
        if input.answer_choices:
            task_kind = TaskKind.MULTIPLE_CHOICE
            value_type = AnswerValueType.CHOICE_LABEL
            objective = "Select the best-supported answer choice."
            decision_form = "answer_choice"
        elif input.hypothesis_seeds:
            task_kind = _explicit_seed_task_kind(input)
            value_type = AnswerValueType.STRUCTURED_TEXT
            objective = "Assess the supplied hypotheses against available evidence."
            decision_form = "hypothesis_assessment"
        else:
            raise TaskAdmissionError(
                "explicit task admission requires answer choices or hypothesis seeds"
            )
        return TaskAdmissionDecision(
            attempt_id=_required_text(input.attempt_id, "attempt_id"),
            status=TaskAdmissionStatus.ADMITTED,
            epistemic_basis=[
                "The caller supplied an explicit candidate frame for evidence assessment."
            ],
            proposed_task_kind=task_kind,
            answer_contract_outline=AnswerContractOutline(
                objective=objective,
                answer_value_type=value_type,
                decision_form=decision_form,
                permits_synthesis=task_kind != TaskKind.MULTIPLE_CHOICE,
                required_sections=["answer", "basis", "uncertainty"],
            ),
            reason="An explicit frame is available without model-based admission.",
            model_trace={"source": "explicit_task_admission"},
        )


class ModelTaskAdmitter:
    def __init__(self, model_gateway: ModelGateway) -> None:
        self._model_gateway = model_gateway

    def assess(self, input: TaskAdmissionInput) -> TaskAdmissionDecision:
        _validate_admission_input(input)
        request = _admission_request(input)
        payload = self._complete(request)
        try:
            return _decision_from_mapping(
                payload,
                attempt_id=input.attempt_id,
                trace=_trace_for(request, self._model_gateway),
            )
        except (TaskAdmissionError, TypeError, ValueError):
            repair = _repair_admission_request(input, request, payload)
            repaired_payload = self._complete(repair)
            try:
                return _decision_from_mapping(
                    repaired_payload,
                    attempt_id=input.attempt_id,
                    trace=_trace_for(repair, self._model_gateway),
                )
            except (TaskAdmissionError, TypeError, ValueError):
                raise TaskAdmissionError(
                    "task admission invalid after 1 repair attempt"
                ) from None

    def _complete(self, request: StructuredModelRequest) -> dict[str, Any]:
        try:
            return self._model_gateway.complete_structured(request)
        except Exception:
            raise TaskAdmissionError("task admission model gateway call failed") from None


class RecordedTaskAdmitter:
    def __init__(self, decision: TaskAdmissionDecision) -> None:
        self._decision = decision.model_copy(deep=True)

    def assess(self, input: TaskAdmissionInput) -> TaskAdmissionDecision:
        _validate_admission_input(input)
        payload = self._decision.model_dump(mode="python", exclude={"attempt_id"})
        payload["attempt_id"] = _required_text(input.attempt_id, "attempt_id")
        return TaskAdmissionDecision.model_validate(payload)


class RoutingTaskAdmitter:
    def __init__(
        self,
        *,
        explicit_admitter: ExplicitTaskAdmitter,
        open_admitter: TaskAdmitter,
    ) -> None:
        self._explicit_admitter = explicit_admitter
        self._open_admitter = open_admitter

    def assess(self, input: TaskAdmissionInput) -> TaskAdmissionDecision:
        if self._explicit_admitter.can_assess(input):
            return self._explicit_admitter.assess(input)
        return self._open_admitter.assess(input)


_DECISION_FIELDS = {
    "status",
    "epistemic_basis",
    "proposed_task_kind",
    "answer_contract_outline",
    "clarification_questions",
    "reason",
}


def _decision_from_mapping(
    payload: Mapping[str, Any],
    *,
    attempt_id: str,
    trace: dict[str, Any],
) -> TaskAdmissionDecision:
    if not isinstance(payload, Mapping) or set(payload) != _DECISION_FIELDS:
        raise TaskAdmissionError(
            "task admission payload has missing or unknown fields"
        )
    try:
        return TaskAdmissionDecision.model_validate(
            {
                **dict(payload),
                "attempt_id": _required_text(attempt_id, "attempt_id"),
                "model_trace": redact_secret_material(trace),
            }
        )
    except ValueError:
        raise TaskAdmissionError("invalid task admission decision") from None


def _admission_request(input: TaskAdmissionInput) -> StructuredModelRequest:
    return StructuredModelRequest(
        task="assess_task_admission",
        input={
            "question": _required_text(input.question, "question"),
            "task_context": _optional_text(input.task_context, "task_context"),
            "requested_output_shape": _optional_nullable_text(
                input.requested_output_shape,
                "requested_output_shape",
            ),
            "available_capabilities": redact_secret_material(
                [item.model_dump(mode="json") for item in input.available_capabilities]
            ),
        },
        prompt_id="task_admission",
        prompt_version="v0.2",
        schema_name="TaskAdmissionDecision",
        schema_version="v0.2",
        metadata={"attempt_id": input.attempt_id},
    )


def _repair_admission_request(
    input: TaskAdmissionInput,
    original_request: StructuredModelRequest,
    invalid_payload: Any,
) -> StructuredModelRequest:
    return StructuredModelRequest(
        task="repair_task_admission",
        input={
            "original_request": redact_secret_material(original_request.input),
            "invalid_payload": redact_secret_material(invalid_payload),
            "attempt_index": 1,
            "required_fields": sorted(_DECISION_FIELDS),
        },
        prompt_id="task_admission_repair",
        prompt_version="v0.2",
        schema_name="TaskAdmissionDecision",
        schema_version="v0.2",
        metadata={"attempt_id": input.attempt_id, "repair_attempt_index": 1},
    )


def _trace_for(
    request: StructuredModelRequest,
    model_gateway: ModelGateway,
) -> dict[str, Any]:
    adapter_kind = getattr(model_gateway, "adapter_kind", type(model_gateway).__name__)
    if not isinstance(adapter_kind, str) or not adapter_kind.strip():
        adapter_kind = type(model_gateway).__name__
    return ModelInvocationTrace.from_request(
        request,
        adapter_kind=adapter_kind,
    ).to_dict()


def _explicit_seed_task_kind(input: TaskAdmissionInput) -> TaskKind:
    value = input.model_metadata.get("task_kind")
    if value is None:
        return TaskKind.CLAIM_VERIFICATION
    try:
        return TaskKind(value)
    except (TypeError, ValueError):
        raise TaskAdmissionError("invalid explicit task kind") from None


def _validate_admission_input(input: TaskAdmissionInput) -> None:
    _required_text(input.attempt_id, "attempt_id")
    _required_text(input.question, "question")
    _optional_text(input.task_context, "task_context")
    _optional_nullable_text(input.requested_output_shape, "requested_output_shape")
    _reject_secret_material(
        {
            "attempt_id": input.attempt_id,
            "question": input.question,
            "task_context": input.task_context,
            "answer_choices": input.answer_choices,
            "hypothesis_seeds": input.hypothesis_seeds,
            "available_capabilities": input.available_capabilities,
            "model_metadata": input.model_metadata,
        }
    )


def _reject_secret_material(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if is_forbidden_secret_key_name(str(key)) or is_secret_like_value(str(key)):
                raise TaskAdmissionError(
                    "task admission input must not contain secret material"
                )
            _reject_secret_material(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _reject_secret_material(item)
        return
    if hasattr(value, "model_dump"):
        _reject_secret_material(value.model_dump(mode="python"))
        return
    if hasattr(value, "__dict__"):
        _reject_secret_material(vars(value))
        return
    if isinstance(value, str) and is_secret_like_value(value):
        raise TaskAdmissionError("task admission input must not contain secret material")


def _required_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TaskAdmissionError(f"{field_name} must not be empty")
    return value.strip()


def _optional_text(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TaskAdmissionError(f"{field_name} must be a string")
    return value.strip()


def _optional_nullable_text(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field_name)


__all__ = [
    "ExplicitTaskAdmitter",
    "ModelTaskAdmitter",
    "RecordedTaskAdmitter",
    "RoutingTaskAdmitter",
    "TaskAdmitter",
    "TaskAdmissionError",
    "TaskAdmissionInput",
]
