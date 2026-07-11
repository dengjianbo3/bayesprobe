from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from bayesprobe.evaluation.contracts import ArmCaseResult, EvaluationCase
from bayesprobe.model_gateway import (
    ModelGateway,
    ModelGatewayValidationError,
    StructuredModelRequest,
)
from bayesprobe.provider_telemetry import provider_error_category


_DIRECT_REQUIRED_KEYS = frozenset(
    {"answer_label", "choice_probabilities", "answer_summary"}
)


class DirectFlashArm:
    arm_name = "direct_flash"

    def __init__(
        self,
        model_gateway: ModelGateway,
        *,
        invocation_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._model_gateway = model_gateway
        self._invocation_metadata = dict(invocation_metadata or {})

    def run_case(self, case: EvaluationCase) -> ArmCaseResult:
        metadata = {
            **self._invocation_metadata,
            "arm": self.arm_name,
            "sample_id": case.sample_id,
        }
        request = StructuredModelRequest(
            task="answer_multiple_choice",
            input={"question": case.question, "choices": dict(case.choices)},
            prompt_id="direct_multiple_choice",
            prompt_version="v0.1",
            schema_name="MultipleChoiceAnswer",
            schema_version="v0.1",
            metadata=metadata,
        )
        invalid_payload: Any = None
        try:
            invalid_payload = self._model_gateway.complete_structured(request)
            return _result_from_payload(
                case,
                invalid_payload,
                arm=self.arm_name,
                model_calls=1,
                schema_repairs=0,
            )
        except ModelGatewayValidationError as error:
            validation_error = str(error)
        except (TypeError, ValueError) as error:
            validation_error = str(error)
        except Exception as error:
            return _failed_result(
                case,
                arm=self.arm_name,
                error_category=provider_error_category(error),
                model_calls=1,
                schema_repairs=0,
            )

        repair_request = StructuredModelRequest(
            task="repair_multiple_choice_answer",
            input={
                "question": case.question,
                "choices": dict(case.choices),
                "invalid_payload": invalid_payload,
                "validation_error": validation_error,
            },
            prompt_id="direct_multiple_choice_repair",
            prompt_version="v0.1",
            schema_name="MultipleChoiceAnswer",
            schema_version="v0.1",
            metadata={**metadata, "repair_attempt_index": 1},
        )
        try:
            repaired_payload = self._model_gateway.complete_structured(repair_request)
            return _result_from_payload(
                case,
                repaired_payload,
                arm=self.arm_name,
                model_calls=2,
                schema_repairs=1,
            )
        except (ModelGatewayValidationError, TypeError, ValueError):
            return _failed_result(
                case,
                arm=self.arm_name,
                error_category="structured_output_invalid",
                model_calls=2,
                schema_repairs=1,
            )
        except Exception as error:
            return _failed_result(
                case,
                arm=self.arm_name,
                error_category=provider_error_category(error),
                model_calls=2,
                schema_repairs=1,
            )


def _result_from_payload(
    case: EvaluationCase,
    payload: Any,
    *,
    arm: str,
    model_calls: int,
    schema_repairs: int,
) -> ArmCaseResult:
    if not isinstance(payload, Mapping):
        raise ModelGatewayValidationError(
            "multiple-choice answer payload must be an object"
        )
    if set(payload) != _DIRECT_REQUIRED_KEYS:
        raise ModelGatewayValidationError(
            "multiple-choice answer must contain exactly answer_label, "
            "choice_probabilities, and answer_summary"
        )
    answer_label = payload["answer_label"]
    if not isinstance(answer_label, str) or answer_label not in case.choices:
        raise ModelGatewayValidationError(
            "multiple-choice answer_label must be a supplied choice label"
        )
    probabilities = payload["choice_probabilities"]
    if not isinstance(probabilities, Mapping):
        raise ModelGatewayValidationError(
            "multiple-choice choice_probabilities must be an object"
        )
    if set(probabilities) != set(case.choice_labels):
        raise ModelGatewayValidationError(
            "multiple-choice probability keys must exactly match supplied labels"
        )
    answer_summary = payload["answer_summary"]
    if not isinstance(answer_summary, str) or not answer_summary.strip():
        raise ModelGatewayValidationError(
            "multiple-choice answer_summary must not be empty"
        )
    try:
        return ArmCaseResult(
            sample_id=case.sample_id,
            arm=arm,
            state="completed",
            answer_label=answer_label,
            probabilities=dict(probabilities),
            answer_summary=answer_summary,
            process_metrics={
                "model_calls": model_calls,
                "schema_repairs": schema_repairs,
            },
        )
    except ValueError as error:
        raise ModelGatewayValidationError(str(error)) from error


def _failed_result(
    case: EvaluationCase,
    *,
    arm: str,
    error_category: str,
    model_calls: int,
    schema_repairs: int,
) -> ArmCaseResult:
    return ArmCaseResult(
        sample_id=case.sample_id,
        arm=arm,
        state="terminal_failed",
        answer_label=None,
        probabilities=None,
        error_category=error_category,
        process_metrics={
            "model_calls": model_calls,
            "schema_repairs": schema_repairs,
        },
    )


__all__ = ["DirectFlashArm"]
