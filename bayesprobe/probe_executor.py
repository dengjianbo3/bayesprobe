from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Protocol

from bayesprobe.evidence_memory import (
    derive_deterministic_computation_root,
    derive_model_gateway_signal_source,
    derive_model_provenance_keys,
)
from bayesprobe.ledger import JsonlLedgerStore
from bayesprobe.lifecycle import resolve_belief_lifecycle
from bayesprobe.model_gateway import (
    ModelGateway,
    ModelGatewayValidationError,
    StructuredModelRequest,
    model_gateway_adapter_kind,
    model_gateway_identity,
)
from bayesprobe.schemas import (
    BeliefState,
    EpistemicOrigin,
    ExternalSignal,
    ProbeDesign,
    ProbeSet,
    SignalKind,
    SignalProvenance,
    TaskFrame,
    contains_secret_material,
)


_DETERMINISTIC_PROBE_TOOL_IDENTITY = "deterministic_probe_gateway:v1"
_BLIND_BELIEF_CONTEXT_POLICY = "blind_no_scores_v1"
_FORBIDDEN_EXECUTION_BELIEF_KEYS = frozenset(
    {
        "ad_hoc_penalty",
        "applied_ad_hoc_penalty",
        "applied_complexity_penalty",
        "belief_state",
        "complexity_penalty",
        "correlation_credit",
        "current_best_hypothesis",
        "effective_update_weight",
        "evidence_credit",
        "gap",
        "initial_prior",
        "posterior",
        "posterior_summary",
        "prior",
        "rank",
        "ranking",
        "score",
        "scores",
        "top_hypothesis",
        "uncertainty_summary",
        "unresolved_alternative_mass",
    }
)
_FORBIDDEN_EXECUTION_BELIEF_KEY_PARTS = frozenset(
    {
        "confidence",
        "credit",
        "gap",
        "likelihood",
        "odds",
        "penalty",
        "posterior",
        "prior",
        "probability",
        "rank",
        "ranking",
        "score",
        "scores",
        "uncertainty",
    }
)


@dataclass(frozen=True)
class ProbeExecutionHypothesisView:
    id: str
    statement: str
    scope: str
    predictions: tuple[str, ...]
    falsifiers: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _clean_required(self.id, "hypothesis id"))
        object.__setattr__(
            self,
            "statement",
            _clean_required(self.statement, "hypothesis statement"),
        )
        if not isinstance(self.scope, str):
            raise ValueError("hypothesis scope must be a string")
        object.__setattr__(self, "scope", self.scope.strip())
        object.__setattr__(
            self,
            "predictions",
            _immutable_texts(self.predictions, "hypothesis predictions"),
        )
        object.__setattr__(
            self,
            "falsifiers",
            _immutable_texts(self.falsifiers, "hypothesis falsifiers"),
        )
        if contains_secret_material(
            {
                "id": self.id,
                "statement": self.statement,
                "scope": self.scope,
                "predictions": self.predictions,
                "falsifiers": self.falsifiers,
            }
        ):
            raise ValueError("probe execution brief must not contain secret material")


@dataclass(frozen=True)
class ProbeExecutionBrief:
    run_id: str
    cycle_id: str
    problem: str
    task_context: str
    task_frame: Mapping[str, Any]
    provider_schema_version: Literal["v0.1", "v0.2"]
    hypotheses: tuple[ProbeExecutionHypothesisView, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        run_id = _clean_required(self.run_id, "run_id")
        cycle_id = _clean_required(self.cycle_id, "cycle_id")
        problem = _clean_required(self.problem, "problem")
        task_context = self.task_context.strip()
        if self.provider_schema_version not in {"v0.1", "v0.2"}:
            raise ValueError("provider_schema_version must be v0.1 or v0.2")
        hypotheses = tuple(self.hypotheses)
        if not hypotheses:
            raise ValueError("probe execution brief requires hypotheses")
        if not all(
            isinstance(hypothesis, ProbeExecutionHypothesisView)
            for hypothesis in hypotheses
        ):
            raise ValueError("probe execution brief hypotheses must use blind views")
        if contains_secret_material(self.metadata):
            raise ValueError(
                "probe execution metadata must not contain secret material"
            )
        if _contains_forbidden_belief_key(self.metadata):
            raise ValueError(
                "probe execution metadata must not contain belief scores or ranking"
            )
        if _contains_forbidden_belief_key(self.task_frame):
            raise ValueError("probe execution task frame must be blind to belief scores")
        safe_brief_content = {
            "run_id": run_id,
            "cycle_id": cycle_id,
            "problem": problem,
            "task_context": task_context,
            "task_frame": self.task_frame,
            "hypotheses": [
                {
                    "id": hypothesis.id,
                    "statement": hypothesis.statement,
                    "scope": hypothesis.scope,
                    "predictions": hypothesis.predictions,
                    "falsifiers": hypothesis.falsifiers,
                }
                for hypothesis in hypotheses
            ],
        }
        if contains_secret_material(safe_brief_content):
            raise ValueError("probe execution brief must not contain secret material")
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "cycle_id", cycle_id)
        object.__setattr__(self, "problem", problem)
        object.__setattr__(self, "task_context", task_context)
        object.__setattr__(self, "task_frame", _freeze_mapping(self.task_frame))
        object.__setattr__(self, "hypotheses", hypotheses)
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


def build_probe_execution_brief(
    *,
    run_id: str,
    cycle_id: str,
    belief_state: BeliefState,
    problem: str,
    task_context: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> ProbeExecutionBrief:
    provider_schema_version = resolve_belief_lifecycle(
        belief_state
    ).provider_version
    task_frame = belief_state.task_frame
    if task_frame is None:
        raise ValueError("invalid belief lifecycle: task frame is required")
    resolved_task_context = task_context.strip() or task_frame.task_context.strip()
    safe_task_frame = _blind_task_frame(task_frame)
    return ProbeExecutionBrief(
        run_id=run_id,
        cycle_id=cycle_id,
        problem=problem,
        task_context=resolved_task_context,
        task_frame=safe_task_frame,
        provider_schema_version=provider_schema_version,
        hypotheses=tuple(
            ProbeExecutionHypothesisView(
                id=hypothesis.id,
                statement=hypothesis.statement,
                scope=hypothesis.scope,
                predictions=tuple(hypothesis.predictions),
                falsifiers=tuple(hypothesis.falsifiers),
            )
            for hypothesis in belief_state.hypotheses
        ),
        metadata={} if metadata is None else metadata,
    )


class ProbeToolGateway(Protocol):
    def execute_probe(
        self,
        *,
        probe: ProbeDesign,
        context: ProbeExecutionBrief,
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
        context: ProbeExecutionBrief,
    ) -> list[ExternalSignal]:
        cue = _deterministic_content_cue(probe.method)
        targets = ", ".join(probe.target_hypotheses)
        derivation_root_id = derive_deterministic_computation_root(
            tool_identity=_DETERMINISTIC_PROBE_TOOL_IDENTITY,
            computation_inputs={
                "method": probe.method,
                "inquiry_goal": probe.inquiry_goal,
                "target_hypotheses": sorted(probe.target_hypotheses),
                "support_condition": dict(probe.support_condition),
                "weaken_condition": dict(probe.weaken_condition),
                "reframe_condition": (
                    None
                    if probe.reframe_condition is None
                    else dict(probe.reframe_condition)
                ),
                "expected_probe_behavior": {"probe_type": probe.probe_type},
            },
        )
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
                provenance=SignalProvenance(
                    epistemic_origin=EpistemicOrigin.TOOL_RESULT,
                    source_identity=_DETERMINISTIC_PROBE_TOOL_IDENTITY,
                    provider_model_or_tool_identity=(
                        _DETERMINISTIC_PROBE_TOOL_IDENTITY
                    ),
                    derivation_root_id=derivation_root_id,
                    correlation_group=(
                        f"tool:{_DETERMINISTIC_PROBE_TOOL_IDENTITY}"
                    ),
                    canonical_content_fingerprint="pending-normalization",
                ),
            )
        ]


class ModelBackedProbeToolGateway:
    def __init__(self, model_gateway: ModelGateway) -> None:
        self._model_gateway = model_gateway

    def execute_probe(
        self,
        *,
        probe: ProbeDesign,
        context: ProbeExecutionBrief,
    ) -> list[ExternalSignal]:
        model_identity = model_gateway_identity(self._model_gateway)
        model_keys = derive_model_provenance_keys(
            provider_identity=model_identity,
            session_id=context.run_id,
        )
        adapter_kind = model_gateway_adapter_kind(self._model_gateway)
        signal_source = derive_model_gateway_signal_source(adapter_kind)
        provenance = SignalProvenance(
            epistemic_origin=EpistemicOrigin.MODEL_REASONING,
            source_identity=model_keys.source_identity,
            provider_model_or_tool_identity=model_identity,
            session_id=context.run_id,
            derivation_root_id=(
                f"model-probe:{context.run_id}:{context.cycle_id}:{probe.id}"
            ),
            correlation_group=model_keys.correlation_group,
            canonical_content_fingerprint="pending-normalization",
        )
        request = StructuredModelRequest(
            task="execute_probe",
            input={
                "problem": context.problem,
                "task_context": context.task_context,
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
                        "predictions": list(hypothesis.predictions),
                        "falsifiers": list(hypothesis.falsifiers),
                    }
                    for hypothesis in context.hypotheses
                ],
            },
            prompt_id="probe_execution",
            prompt_version=context.provider_schema_version,
            schema_name="ProbeSignal",
            schema_version=context.provider_schema_version,
            metadata={
                "run_id": context.run_id,
                "cycle_id": context.cycle_id,
                "probe_id": probe.id,
                "belief_context_policy": _BLIND_BELIEF_CONTEXT_POLICY,
            },
        )
        payload = self._model_gateway.complete_structured(request)
        raw_content = _probe_raw_content(payload)
        return [
            ExternalSignal(
                id=f"S_{context.cycle_id}_{probe.id}",
                cycle_id=context.cycle_id,
                signal_kind=SignalKind.ACTIVE,
                source_type="model_probe_gateway",
                source=signal_source,
                raw_content=raw_content,
                generated_by_probe=probe.id,
                initial_target_hypotheses=list(probe.target_hypotheses),
                provenance=provenance,
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
        context: ProbeExecutionBrief,
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
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must not be empty")
    return cleaned


def _immutable_texts(values: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    if not isinstance(values, list | tuple):
        raise ValueError(f"{field_name} must be a sequence")
    result: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError(f"{field_name} must contain only strings")
        result.append(value.strip())
    return tuple(result)


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


def _contains_forbidden_belief_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            _is_forbidden_belief_key(str(key))
            or _contains_forbidden_belief_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list | tuple):
        return any(_contains_forbidden_belief_key(item) for item in value)
    return False


def _without_belief_context(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _without_belief_context(item)
            for key, item in value.items()
            if not _is_forbidden_belief_key(str(key))
        }
    if isinstance(value, list | tuple):
        return [_without_belief_context(item) for item in value]
    return value


def _blind_task_frame(task_frame: TaskFrame) -> dict[str, Any]:
    hypothesis_frame = task_frame.hypothesis_frame
    return _without_belief_context(
        {
            "schema_version": task_frame.schema_version,
            "task_frame_id": task_frame.task_frame_id,
            "admission_decision_id": task_frame.admission_decision_id,
            "task_kind": task_frame.task_kind.value,
            "answer_relationship": (
                None
                if task_frame.answer_relationship is None
                else task_frame.answer_relationship.value
            ),
            "normalized_question": task_frame.normalized_question,
            "task_context": task_frame.task_context,
            "answer_contract": task_frame.answer_contract.model_dump(mode="json"),
            "hypothesis_frame": {
                "frame_id": hypothesis_frame.frame_id,
                "competition": hypothesis_frame.competition.value,
                "coverage": hypothesis_frame.coverage.value,
                "rival_sets": hypothesis_frame.rival_sets,
                "coverage_statement": hypothesis_frame.coverage_statement,
                "coverage_limitation": hypothesis_frame.coverage_limitation,
            },
            "framing_method": task_frame.framing_method.value,
        }
    )


def _is_forbidden_belief_key(value: str) -> bool:
    normalized = value.casefold()
    if normalized in _FORBIDDEN_EXECUTION_BELIEF_KEYS:
        return True
    separated = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    parts = set(re.findall(r"[a-z0-9]+", separated.casefold()))
    if parts.intersection(_FORBIDDEN_EXECUTION_BELIEF_KEY_PARTS):
        return True
    return (
        {"belief", "score"}.issubset(parts)
        or {"current", "best"}.issubset(parts)
        or {"top", "hypothesis"}.issubset(parts)
    )


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("probe execution immutable fields must be mappings")
    frozen: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError("probe execution immutable mapping keys must be strings")
        frozen[key] = _freeze_value(item)
    return MappingProxyType(frozen)


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, list | tuple):
        return tuple(_freeze_value(item) for item in value)
    if value is None or isinstance(value, str | bool):
        return value
    if type(value) in {int, float}:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("probe execution immutable values must be finite")
        return value
    raise ValueError("probe execution immutable values must be JSON-compatible")


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
    "ProbeExecutionBrief",
    "ProbeExecutionHypothesisView",
    "ProbeExecutionResult",
    "ProbeExecutor",
    "ProbeToolGateway",
    "build_probe_execution_brief",
]
