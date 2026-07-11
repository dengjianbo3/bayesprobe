from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from typing import Any

from bayesprobe.core import BayesProbeCore
from bayesprobe.evaluation.contracts import ArmCaseResult, EvaluationCase
from bayesprobe.evaluation.python_probe import (
    DockerPythonSandbox,
    PythonAugmentedProbeToolGateway,
    PythonExecutionObserver,
    ResolvedSandboxImage,
)
from bayesprobe.initialization import BayesProbeInitializer, InitializeRunInput
from bayesprobe.model_gateway import (
    EvidenceJudgmentRepairPolicy,
    ModelGateway,
    ModelGatewayValidationError,
    StructuredModelRequest,
    model_gateway_adapter_kind,
)
from bayesprobe.probe_executor import ProbeExecutor
from bayesprobe.probe_planner import ProbePlanner
from bayesprobe.provider_telemetry import provider_error_category
from bayesprobe.question_runner import (
    AutonomousQuestionRunConfig,
    AutonomousQuestionRunResult,
    AutonomousQuestionRunner,
)
from bayesprobe.schemas import SignalKind


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


class BayesProbePythonArm:
    arm_name = "bayesprobe_python"

    def __init__(
        self,
        model_gateway: ModelGateway,
        sandbox: DockerPythonSandbox,
        *,
        image: ResolvedSandboxImage | None = None,
        invocation_metadata: Mapping[str, Any] | None = None,
        execution_observer: PythonExecutionObserver | None = None,
        run_result_observer: Callable[[AutonomousQuestionRunResult], None] | None = None,
    ) -> None:
        self._model_gateway = model_gateway
        self._sandbox = sandbox
        self._image = image
        self._invocation_metadata = dict(invocation_metadata or {})
        self._execution_observer = execution_observer
        self._run_result_observer = run_result_observer
        self.run_config = AutonomousQuestionRunConfig(
            max_cycles=4,
            max_probes_per_cycle=2,
            stop_on_no_probes=True,
            confidence_threshold=None,
            posterior_delta_threshold=None,
        )

    def run_case(self, case: EvaluationCase) -> ArmCaseResult:
        run_id = _case_run_id(self.arm_name, case.sample_id)
        metadata = {
            **self._invocation_metadata,
            "arm": self.arm_name,
            "sample_id": case.sample_id,
            "run_id": run_id,
        }
        contextual_gateway = _ContextualModelGateway(
            self._model_gateway,
            metadata=metadata,
        )
        python_gateway = PythonAugmentedProbeToolGateway(
            contextual_gateway,
            self._sandbox,
            image=self._image,
            execution_observer=self._execution_observer,
        )
        core = BayesProbeCore(
            model_gateway=contextual_gateway,
            judgment_repair_policy=EvidenceJudgmentRepairPolicy(max_attempts=1),
        )
        runner = AutonomousQuestionRunner(
            core=core,
            initializer=BayesProbeInitializer(ledger=core.ledger),
            planner=ProbePlanner(ledger=core.ledger),
            executor=ProbeExecutor(gateway=python_gateway, ledger=core.ledger),
            config=self.run_config,
        )
        try:
            run_result = runner.run_question(
                InitializeRunInput(
                    run_id=run_id,
                    problem=case.question,
                    context="",
                    metadata=metadata,
                )
            )
        except Exception as error:
            return ArmCaseResult(
                sample_id=case.sample_id,
                arm=self.arm_name,
                state="terminal_failed",
                answer_label=None,
                probabilities=None,
                error_category=provider_error_category(error),
                process_metrics=python_gateway.process_metrics,
            )
        self._observe_run_result(run_result)
        projection = run_result.final_answer_projection
        if projection is None or projection.current_best_hypothesis not in case.choices:
            return ArmCaseResult(
                sample_id=case.sample_id,
                arm=self.arm_name,
                state="terminal_failed",
                answer_label=None,
                probabilities=None,
                error_category="invalid_final_hypothesis",
                process_metrics=_run_process_metrics(
                    run_result,
                    python_metrics=python_gateway.process_metrics,
                ),
            )
        final_hypotheses = run_result.final_belief_state.hypotheses_by_id()
        if set(final_hypotheses) != set(case.choice_labels):
            return ArmCaseResult(
                sample_id=case.sample_id,
                arm=self.arm_name,
                state="terminal_failed",
                answer_label=None,
                probabilities=None,
                error_category="hypothesis_space_changed",
                process_metrics=_run_process_metrics(
                    run_result,
                    python_metrics=python_gateway.process_metrics,
                ),
            )
        return ArmCaseResult(
            sample_id=case.sample_id,
            arm=self.arm_name,
            state="completed",
            answer_label=projection.current_best_hypothesis,
            probabilities={
                label: final_hypotheses[label].posterior
                for label in case.choice_labels
            },
            answer_summary=projection.answer,
            process_metrics=_run_process_metrics(
                run_result,
                python_metrics=python_gateway.process_metrics,
            ),
        )

    def _observe_run_result(self, result: AutonomousQuestionRunResult) -> None:
        if self._run_result_observer is None:
            return
        try:
            self._run_result_observer(result)
        except Exception:
            return


class _ContextualModelGateway:
    def __init__(
        self,
        gateway: ModelGateway,
        *,
        metadata: Mapping[str, Any],
    ) -> None:
        self._gateway = gateway
        self._metadata = dict(metadata)
        self.adapter_kind = model_gateway_adapter_kind(gateway)

    def complete_structured(self, request: StructuredModelRequest) -> dict[str, Any]:
        contextual_request = StructuredModelRequest(
            task=request.task,
            input=request.input,
            prompt_id=request.prompt_id,
            prompt_version=request.prompt_version,
            schema_name=request.schema_name,
            schema_version=request.schema_version,
            metadata={**request.metadata, **self._metadata},
        )
        return self._gateway.complete_structured(contextual_request)


def _case_run_id(arm: str, sample_id: str) -> str:
    suffix = hashlib.sha256(f"{arm}:{sample_id}".encode("utf-8")).hexdigest()[:20]
    return f"eval_{arm}_{suffix}"


def _run_process_metrics(
    result: AutonomousQuestionRunResult,
    *,
    python_metrics: Mapping[str, int],
) -> dict[str, int | str]:
    cycles = result.cycle_results
    probes = sum(len(cycle.execution_result.executed_probe_ids) for cycle in cycles)
    signals = [signal for cycle in cycles for signal in cycle.signals]
    evidence_events = [event for cycle in cycles for event in cycle.evidence_events]
    top_sequence = [
        _top_hypothesis_id(result.initial_belief_state),
        *(_top_hypothesis_id(cycle.belief_state) for cycle in cycles),
    ]
    final_answer = (
        result.final_answer_projection.current_best_hypothesis
        if result.final_answer_projection is not None
        else top_sequence[-1]
    )
    first_top_cycle = next(
        (
            cycle_index
            for cycle_index, hypothesis_id in enumerate(top_sequence)
            if hypothesis_id == final_answer
        ),
        len(cycles),
    )
    metrics: dict[str, int | str] = {
        "cycles": len(cycles),
        "probes": probes,
        "active_signals": sum(
            signal.signal_kind == SignalKind.ACTIVE for signal in signals
        ),
        "accepted_evidence_events": sum(
            event.discard_reason is None for event in evidence_events
        ),
        "discarded_evidence_events": sum(
            event.discard_reason is not None for event in evidence_events
        ),
        "evidence_judgment_repairs": sum(
            event.model_trace.get("task") == "repair_evidence_judgment"
            for event in evidence_events
        ),
        "top_answer_reversals": sum(
            previous != current
            for previous, current in zip(top_sequence, top_sequence[1:])
        ),
        "final_answer_first_top_cycle": first_top_cycle,
        "stop_reason": result.stop_reason.value,
    }
    metrics.update(python_metrics)
    return metrics


def _top_hypothesis_id(belief_state: Any) -> str:
    return max(
        belief_state.hypotheses,
        key=lambda hypothesis: hypothesis.posterior,
    ).id


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


__all__ = ["BayesProbePythonArm", "DirectFlashArm"]
