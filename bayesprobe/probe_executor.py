from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from bayesprobe.ledger import JsonlLedgerStore
from bayesprobe.model_gateway import (
    ModelGateway,
    ModelGatewayValidationError,
    StructuredModelRequest,
    model_gateway_adapter_kind,
)
from bayesprobe.schemas import (
    BeliefState,
    ExternalSignal,
    FramingMethod,
    ProbeDesign,
    ProbeSet,
    SignalKind,
)


@dataclass(frozen=True)
class ProbeExecutionContext:
    run_id: str
    cycle_id: str
    belief_state: BeliefState
    metadata: dict[str, Any] = field(default_factory=dict)


class ProbeToolGateway(Protocol):
    def execute_probe(
        self,
        *,
        probe: ProbeDesign,
        context: ProbeExecutionContext,
    ) -> list[ExternalSignal]:
        ...


@dataclass(frozen=True)
class ProbeExecutionResult:
    probe_set: ProbeSet
    signals: list[ExternalSignal]
    executed_probe_ids: list[str]


class DeterministicProbeToolGateway:
    def execute_probe(
        self,
        *,
        probe: ProbeDesign,
        context: ProbeExecutionContext,
    ) -> list[ExternalSignal]:
        cue = _deterministic_content_cue(probe.method)
        targets = ", ".join(probe.target_hypotheses)
        return [
            ExternalSignal(
                id=f"S_{context.cycle_id}_{probe.id}",
                cycle_id=context.cycle_id,
                signal_kind=SignalKind.ACTIVE,
                source_type="deterministic_probe_gateway",
                source=probe.method,
                raw_content=(
                    f"{cue}: Deterministic probe result for {probe.id}; "
                    f"goal={probe.inquiry_goal}; targets={targets}."
                ),
                generated_by_probe=probe.id,
                initial_target_hypotheses=list(probe.target_hypotheses),
            )
        ]


class ModelBackedProbeToolGateway:
    def __init__(self, model_gateway: ModelGateway) -> None:
        self._model_gateway = model_gateway

    def execute_probe(
        self,
        *,
        probe: ProbeDesign,
        context: ProbeExecutionContext,
    ) -> list[ExternalSignal]:
        native_v02 = (
            context.belief_state.schema_version == "v0.2"
            and context.belief_state.task_frame is not None
            and context.belief_state.task_frame.framing_method
            != FramingMethod.LEGACY_MIGRATION
            and context.belief_state.task_frame.framing_trace.get("source")
            != "hypothesis_seeds"
        )
        request = StructuredModelRequest(
            task="execute_probe",
            input={
                "problem": _metadata_text(context, "problem"),
                "task_context": _task_context(context),
                "probe": {
                    "id": probe.id,
                    "inquiry_goal": probe.inquiry_goal,
                    "method": probe.method,
                    "target_hypotheses": list(probe.target_hypotheses),
                    "support_condition": dict(probe.support_condition),
                    "weaken_condition": dict(probe.weaken_condition),
                },
                "hypotheses": [
                    {
                        "id": hypothesis.id,
                        "statement": hypothesis.statement,
                        "scope": hypothesis.scope,
                        "posterior": hypothesis.posterior,
                        "predictions": list(hypothesis.predictions),
                        "falsifiers": list(hypothesis.falsifiers),
                    }
                    for hypothesis in context.belief_state.hypotheses
                ],
            },
            prompt_id="probe_execution",
            prompt_version="v0.2" if native_v02 else "v0.1",
            schema_name="ProbeSignal",
            schema_version="v0.2" if native_v02 else "v0.1",
            metadata={
                "run_id": context.run_id,
                "cycle_id": context.cycle_id,
                "probe_id": probe.id,
            },
        )
        payload = self._model_gateway.complete_structured(request)
        raw_content = _probe_raw_content(payload)
        adapter_kind = model_gateway_adapter_kind(self._model_gateway)
        return [
            ExternalSignal(
                id=f"S_{context.cycle_id}_{probe.id}",
                cycle_id=context.cycle_id,
                signal_kind=SignalKind.ACTIVE,
                source_type="model_probe_gateway",
                source=f"model_gateway:{adapter_kind}",
                raw_content=raw_content,
                generated_by_probe=probe.id,
                initial_target_hypotheses=list(probe.target_hypotheses),
            )
        ]


class ProbeExecutor:
    def __init__(
        self,
        gateway: ProbeToolGateway,
        ledger: JsonlLedgerStore | None = None,
    ) -> None:
        self._gateway = gateway
        self._ledger = ledger

    def execute_probe_set(
        self,
        *,
        probe_set: ProbeSet,
        context: ProbeExecutionContext,
    ) -> ProbeExecutionResult:
        run_id = _clean_required(context.run_id, "run_id")
        cycle_id = _clean_required(context.cycle_id, "cycle_id")
        _validate_probe_set_boundary(probe_set=probe_set, cycle_id=cycle_id)

        signals: list[ExternalSignal] = []
        executed_probe_ids: list[str] = []
        for probe in probe_set.probes:
            probe_signals = self._gateway.execute_probe(probe=probe, context=context)
            executed_probe_ids.append(probe.id)
            signals.extend(
                _normalize_signal(signal=signal, probe=probe, cycle_id=cycle_id)
                for signal in probe_signals
            )

        result = ProbeExecutionResult(
            probe_set=probe_set,
            signals=signals,
            executed_probe_ids=executed_probe_ids,
        )
        self._append_ledger(
            run_id=run_id,
            cycle_id=cycle_id,
            probe_set=probe_set,
            result=result,
        )
        return result

    def _append_ledger(
        self,
        *,
        run_id: str,
        cycle_id: str,
        probe_set: ProbeSet,
        result: ProbeExecutionResult,
    ) -> None:
        if self._ledger is None:
            return
        self._ledger.append(
            "probe_execution",
            {
                "run_id": run_id,
                "cycle_id": cycle_id,
                "probe_set_id": probe_set.probe_set_id,
                "executed_probe_ids": result.executed_probe_ids,
                "signal_ids": [signal.id for signal in result.signals],
            },
        )


def _clean_required(value: str, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must not be empty")
    return cleaned


def _validate_probe_set_boundary(*, probe_set: ProbeSet, cycle_id: str) -> None:
    if probe_set.cycle_id != cycle_id:
        raise ValueError("probe set cycle_id must match execution context cycle_id")
    for probe in probe_set.probes:
        if probe.cycle_id != probe_set.cycle_id:
            raise ValueError("probe cycle_id must match probe set cycle_id")


def _normalize_signal(
    *,
    signal: ExternalSignal,
    probe: ProbeDesign,
    cycle_id: str,
) -> ExternalSignal:
    if signal.signal_kind != SignalKind.ACTIVE:
        raise ValueError("probe execution may return only active external signals")
    return signal.model_copy(
        update={
            "cycle_id": cycle_id,
            "generated_by_probe": probe.id,
            "initial_target_hypotheses": list(probe.target_hypotheses),
        }
    )


def _deterministic_content_cue(method: str) -> str:
    method_lower = method.lower()
    if "anomaly" in method_lower:
        return "ANOMALY"
    if "counterevidence" in method_lower or "refutation" in method_lower or "refute" in method_lower:
        return "REFUTES"
    if "support" in method_lower or "source_tracing" in method_lower:
        return "SUPPORTS"
    return "NEUTRAL"


def _metadata_text(context: ProbeExecutionContext, key: str) -> str:
    value = context.metadata.get(key, "")
    return value.strip() if isinstance(value, str) else ""


def _task_context(context: ProbeExecutionContext) -> str:
    explicit_context = _metadata_text(context, "task_context")
    if explicit_context:
        return explicit_context
    task_frame = context.belief_state.task_frame
    return task_frame.task_context.strip() if task_frame is not None else ""


def _probe_raw_content(payload: dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        raise ModelGatewayValidationError("probe signal payload must be an object")
    raw_content = payload.get("raw_content")
    if not isinstance(raw_content, str) or not raw_content.strip():
        raise ModelGatewayValidationError(
            "probe signal payload raw_content must not be empty"
        )
    return raw_content.strip()


__all__ = [
    "DeterministicProbeToolGateway",
    "ModelBackedProbeToolGateway",
    "ProbeExecutionContext",
    "ProbeExecutionResult",
    "ProbeExecutor",
    "ProbeToolGateway",
]
