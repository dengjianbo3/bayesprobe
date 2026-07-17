from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from threading import RLock
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from bayesprobe import (
    ExternalSignal,
    ModelGatewayValidationError,
    ProbeDesign,
    ProbeExecutionBrief,
    StructuredModelRequest,
)
from bayesprobe_terminal_bench.actions import (
    ActionObservation,
    ApplyPatchAction,
    ShellAction,
    TerminalAction,
    TerminalProbePlan,
    WriteFileAction,
    action_may_mutate,
)


class RegisteredPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    run_id: str
    cycle_id: str
    probe_id: str
    plan_id: str
    policy_attempt_id: str
    plan: TerminalProbePlan


class CausalActionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    run_id: str
    cycle_id: str
    probe_id: str
    plan_id: str
    policy_attempt_id: str
    action_id: str
    step_index: int
    action_role: Literal["inspect", "intervene", "verify"]
    request_fingerprint: str
    pre_environment_state_id: str
    post_environment_state_id: str
    subject_environment_state_id: str
    intervention_generation: int
    verification_target: str | None
    transition_predictions: dict[str, str]
    observation: ActionObservation


class CausalDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    signal_id: str
    action_id: str
    action_role: Literal["inspect", "intervene", "verify"]
    decision: Literal["admit", "discard"]
    reason_code: Literal[
        "state_scoped_inspection",
        "neutral_mutation_acknowledgement",
        "verified_postcondition",
        "preregistered_causal_transition",
        "unbound_signal",
        "stale_state",
        "nonneutral_mutation_acknowledgement",
        "unexecuted_policy_comparison",
        "missing_transition_predictions",
        "target_mismatch",
    ]
    subject_environment_state_id: str
    judgment_response_sha256: str


class CausalTraceError(ValueError):
    pass


@dataclass(frozen=True)
class _CausalSignalContext:
    record: CausalActionRecord
    plan: RegisteredPlan
    target_hypotheses: tuple[str, ...]
    hypothesis_types: dict[str, str]
    current_action: CausalActionRecord
    current_environment_state_id: str
    current_intervention_generation: int


class _PlanState:
    def __init__(self) -> None:
        self.last_environment_state_id: str | None = None
        self.intervention_count = 0
        self.registered_steps: set[int] = set()


class CausalTraceRegistry:
    """Bind frozen Probe plans, completed actions, and emitted Signals."""

    def __init__(self) -> None:
        self._plans: dict[str, RegisteredPlan] = {}
        self._target_hypotheses_by_plan: dict[str, tuple[str, ...]] = {}
        self._hypothesis_types_by_plan: dict[str, dict[str, str]] = {}
        self._hypothesis_types_by_run: dict[str, dict[str, str]] = {}
        self._policy_attempt_ids: set[str] = set()
        self._plan_states: dict[str, _PlanState] = {}
        self._last_environment_state_by_run: dict[str, str] = {}
        self._intervention_generation_by_run: dict[str, int] = {}
        self._actions: dict[str, CausalActionRecord] = {}
        self._latest_action_id_by_run: dict[str, str] = {}
        self._action_to_signal: dict[str, str] = {}
        self._signal_to_action: dict[str, str] = {}
        self._lock = RLock()

    def register_plan(
        self,
        *,
        probe: ProbeDesign,
        context: ProbeExecutionBrief,
        plan: TerminalProbePlan,
    ) -> RegisteredPlan:
        probe_payload = probe.model_dump(mode="json")
        plan_payload = plan.model_dump(mode="json")
        common = {
            "cycle_id": context.cycle_id,
            "probe": probe_payload,
            "run_id": context.run_id,
        }
        plan_id = "PL_" + canonical_sha256({**common, "plan": plan_payload})
        policy_attempt_id = "PA_" + canonical_sha256(
            {**common, "intervention_plan": plan_payload}
        )
        registered = RegisteredPlan(
            run_id=context.run_id,
            cycle_id=context.cycle_id,
            probe_id=probe.id,
            plan_id=plan_id,
            policy_attempt_id=policy_attempt_id,
            plan=plan,
        )

        with self._lock:
            if plan_id in self._plans:
                raise ValueError(f"duplicate plan_id: {plan_id}")
            if policy_attempt_id in self._policy_attempt_ids:
                raise ValueError(
                    f"duplicate policy_attempt_id: {policy_attempt_id}"
                )
            self._plans[plan_id] = registered
            self._target_hypotheses_by_plan[plan_id] = tuple(
                probe.target_hypotheses
            )
            hypothesis_types = dict(
                self._hypothesis_types_by_run.get(context.run_id, {})
            )
            hypothesis_types.update(
                _task_frame_hypothesis_types(getattr(context, "task_frame", {}))
            )
            self._hypothesis_types_by_plan[plan_id] = hypothesis_types
            self._policy_attempt_ids.add(policy_attempt_id)
            self._plan_states[plan_id] = _PlanState()
        return registered

    def _register_frame_hypothesis_types(
        self,
        *,
        run_id: str,
        hypothesis_types: Mapping[str, str],
    ) -> None:
        with self._lock:
            existing = self._hypothesis_types_by_run.get(run_id)
            incoming = dict(hypothesis_types)
            if existing is not None and existing != incoming:
                raise CausalTraceError("contradictory task-frame hypothesis types")
            self._hypothesis_types_by_run[run_id] = incoming

    def register_action(
        self,
        *,
        plan: RegisteredPlan,
        step_index: int,
        observation: ActionObservation,
    ) -> CausalActionRecord:
        if type(step_index) is not int:
            raise TypeError("step_index must be an integer")
        if step_index < 0 or step_index >= len(plan.plan.steps):
            raise ValueError("step_index is outside the registered plan")

        step = plan.plan.steps[step_index]
        expected_request = executed_request_from_action(step.action)
        observed_request = executed_request_from_action(observation.action)
        if observed_request != expected_request:
            raise ValueError("executed request does not match the registered plan step")
        if (
            not observation.pre_environment_state_id.strip()
            or not observation.post_environment_state_id.strip()
        ):
            raise ValueError("missing environment state")

        request_fingerprint = "sha256:" + canonical_sha256(expected_request)
        action_id = "A_" + canonical_sha256(
            {
                "action_index": observation.action_index,
                "plan_id": plan.plan_id,
                "request_fingerprint": request_fingerprint,
                "step_index": step_index,
            }
        )

        with self._lock:
            stored_plan = self._plans.get(plan.plan_id)
            if stored_plan is None or stored_plan != plan:
                raise ValueError("action references an unregistered plan")
            state = self._plan_states[plan.plan_id]
            if action_id in self._actions:
                raise ValueError(f"duplicate action_id: {action_id}")
            if step_index in state.registered_steps:
                raise ValueError("plan step already has a completed action")
            expected_environment_state = (
                state.last_environment_state_id
                if state.last_environment_state_id is not None
                else self._last_environment_state_by_run.get(plan.run_id)
            )
            if (
                expected_environment_state is not None
                and observation.pre_environment_state_id != expected_environment_state
            ):
                raise ValueError("non-linear environment state")

            intervention_count = state.intervention_count + int(
                step.role == "intervene"
            )
            if intervention_count > 1:
                raise ValueError("second intervention in one plan")
            intervention_generation = self._intervention_generation_by_run.get(
                plan.run_id, 0
            ) + int(step.role == "intervene")
            subject_environment_state_id = (
                observation.pre_environment_state_id
                if step.role == "verify"
                else observation.post_environment_state_id
            )
            record = CausalActionRecord(
                run_id=plan.run_id,
                cycle_id=plan.cycle_id,
                probe_id=plan.probe_id,
                plan_id=plan.plan_id,
                policy_attempt_id=plan.policy_attempt_id,
                action_id=action_id,
                step_index=step_index,
                action_role=step.role,
                request_fingerprint=request_fingerprint,
                pre_environment_state_id=observation.pre_environment_state_id,
                post_environment_state_id=observation.post_environment_state_id,
                subject_environment_state_id=subject_environment_state_id,
                intervention_generation=intervention_generation,
                verification_target=step.verification_target,
                transition_predictions={
                    prediction.hypothesis_id: prediction.expected_transition
                    for prediction in plan.plan.transition_predictions
                },
                observation=observation,
            )

            self._actions[action_id] = record
            self._latest_action_id_by_run[plan.run_id] = action_id
            state.last_environment_state_id = observation.post_environment_state_id
            state.intervention_count = intervention_count
            state.registered_steps.add(step_index)
            self._last_environment_state_by_run[plan.run_id] = (
                observation.post_environment_state_id
            )
            self._intervention_generation_by_run[plan.run_id] = intervention_generation
        return record

    def bind_signal(
        self,
        *,
        action_id: str,
        signal_builder: Callable[[CausalActionRecord], ExternalSignal],
    ) -> ExternalSignal:
        with self._lock:
            if action_id not in self._actions:
                raise KeyError(f"unknown action: {action_id}")
            if action_id in self._action_to_signal:
                raise ValueError(f"action {action_id} already has a Signal")
            signal = signal_builder(self._actions[action_id])
            if not isinstance(signal, ExternalSignal):
                raise TypeError("signal_builder must return an ExternalSignal")
            signal_id = signal.id
            if signal_id in self._signal_to_action:
                raise ValueError(f"Signal ID is already bound: {signal_id}")
            self._action_to_signal[action_id] = signal_id
            self._signal_to_action[signal_id] = action_id
            return signal

    def record_for_signal(self, signal_id: str) -> CausalActionRecord:
        with self._lock:
            try:
                action_id = self._signal_to_action[signal_id]
            except KeyError as error:
                raise KeyError(f"unknown Signal: {signal_id}") from error
            return self._actions[action_id]

    def _admissibility_context_for_signal(
        self,
        signal_id: str,
    ) -> _CausalSignalContext:
        with self._lock:
            try:
                action_id = self._signal_to_action[signal_id]
                record = self._actions[action_id]
                plan = self._plans[record.plan_id]
                target_hypotheses = self._target_hypotheses_by_plan[record.plan_id]
                hypothesis_types = self._hypothesis_types_by_plan[record.plan_id]
                current_action_id = self._latest_action_id_by_run[record.run_id]
                current_action = self._actions[current_action_id]
                current_environment_state_id = self._last_environment_state_by_run[
                    record.run_id
                ]
                current_intervention_generation = (
                    self._intervention_generation_by_run[record.run_id]
                )
            except KeyError as error:
                raise KeyError(f"unknown Signal: {signal_id}") from error
            if self._action_to_signal.get(action_id) != signal_id:
                raise CausalTraceError("ambiguous Signal binding")
            expected_fingerprint = "sha256:" + canonical_sha256(
                executed_request_from_action(record.observation.action)
            )
            if record.request_fingerprint != expected_fingerprint:
                raise CausalTraceError("registered request fingerprint contradiction")
            if (
                plan.plan_id != record.plan_id
                or plan.policy_attempt_id != record.policy_attempt_id
                or plan.probe_id != record.probe_id
            ):
                raise CausalTraceError("registered causal lineage contradiction")
            return _CausalSignalContext(
                record=record,
                plan=plan,
                target_hypotheses=target_hypotheses,
                hypothesis_types=dict(hypothesis_types),
                current_action=current_action,
                current_environment_state_id=current_environment_state_id,
                current_intervention_generation=current_intervention_generation,
            )


_EVIDENCE_TASKS = frozenset({"judge_evidence", "repair_evidence_judgment"})
_CURRENT_STATE_INSPECTION_TYPES = frozenset(
    {"root_cause", "current_behavior", "invariant"}
)
_DIRECT_VERIFICATION_TYPES = frozenset(
    {"current_behavior", "invariant", "postcondition"}
)
_CAUSAL_TRANSITION_TYPES = frozenset({"root_cause", "causal_effect"})
_TERMINAL_HYPOTHESIS_TYPES = (
    _CURRENT_STATE_INSPECTION_TYPES
    | _DIRECT_VERIFICATION_TYPES
    | _CAUSAL_TRANSITION_TYPES
)
_POLICY_HYPOTHESIS_TYPES = frozenset({"implementation_policy", "patch_choice"})


class CausalEvidenceModelGateway:
    """Admit or discard a delegate-owned Evidence judgment's causal route."""

    def __init__(self, *, delegate: Any, registry: CausalTraceRegistry, artifacts: Any) -> None:
        self._delegate = delegate
        self._registry = registry
        self._artifacts = artifacts

    @property
    def adapter_kind(self) -> Any:
        return self._delegate.adapter_kind

    @property
    def model_identity(self) -> Any:
        return self._delegate.model_identity

    @property
    def config(self) -> Any:
        return self._delegate.config

    @property
    def invocation_observer(self) -> Any:
        return self._delegate.invocation_observer

    def complete_structured(self, request: StructuredModelRequest) -> dict[str, Any]:
        response = self._delegate.complete_structured(request)
        if request.task == "frame_open_question":
            run_id = request.metadata.get("run_id")
            hypothesis_types = _framing_response_hypothesis_types(response)
            if isinstance(run_id, str) and hypothesis_types:
                self._registry._register_frame_hypothesis_types(
                    run_id=run_id,
                    hypothesis_types=hypothesis_types,
                )
        if request.task not in _EVIDENCE_TASKS:
            return response

        request_input = _original_evidence_input(request)
        signal_id = _declared_signal_id(request, request_input)
        response_sha256 = _judgment_response_sha256(response)
        context: _CausalSignalContext | None
        try:
            context = self._registry._admissibility_context_for_signal(signal_id)
        except (CausalTraceError, KeyError):
            context = None

        reason_code = self._classify(
            request=request,
            request_input=request_input,
            response=response,
            context=context,
        )
        decision_kind = "admit" if reason_code in {
            "state_scoped_inspection",
            "neutral_mutation_acknowledgement",
            "verified_postcondition",
            "preregistered_causal_transition",
        } else "discard"
        action_id, action_role, subject_environment_state_id = (
            _decision_lineage(request_input, context)
        )
        decision = CausalDecision(
            signal_id=signal_id,
            action_id=action_id,
            action_role=action_role,
            decision=decision_kind,
            reason_code=reason_code,
            subject_environment_state_id=subject_environment_state_id,
            judgment_response_sha256=response_sha256,
        )
        self._artifacts.append_causal_decision(decision)
        if decision.decision == "discard":
            raise ModelGatewayValidationError(
                f"causal_admissibility:{decision.reason_code}"
            )
        return response

    def _classify(
        self,
        *,
        request: StructuredModelRequest,
        request_input: Mapping[str, Any],
        response: Mapping[str, Any],
        context: _CausalSignalContext | None,
    ) -> str:
        # Precedence is part of the public causal contract. Keep these checks ordered.
        parsed_signal = _parsed_signal(request_input)
        if context is None or not _has_exact_request_binding(
            request=request,
            request_input=request_input,
            parsed_signal=parsed_signal,
            context=context,
        ):
            return "unbound_signal"

        target_types = _target_types(request_input, context=context)
        if not _targets_match(
            request_input=request_input,
            response=response,
            context=context,
            target_types=target_types,
        ) or not _action_role_accepts_target_types(
            action_role=context.record.action_role,
            target_types=target_types,
        ):
            return "target_mismatch"

        if _is_stale(
            request_input=request_input,
            parsed_signal=parsed_signal,
            context=context,
        ):
            return "stale_state"

        if any(
            hypothesis_type in _POLICY_HYPOTHESIS_TYPES
            for hypothesis_type in target_types.values()
        ):
            return "unexecuted_policy_comparison"

        if context.record.action_role == "intervene":
            likelihoods = response.get("likelihoods")
            if (
                response.get("evidence_type") != "neutral"
                or not isinstance(likelihoods, Mapping)
                or any(value != "neutral" for value in likelihoods.values())
                or response.get("frame_fit") != "underdetermined"
            ):
                return "nonneutral_mutation_acknowledgement"
            return "neutral_mutation_acknowledgement"

        if context.record.action_role == "verify":
            causal_targets = {
                target
                for target, hypothesis_type in target_types.items()
                if hypothesis_type in _CAUSAL_TRANSITION_TYPES
            }
            if causal_targets and not _has_differentiated_transition_predictions(
                causal_targets=causal_targets,
                predictions=context.record.transition_predictions,
            ):
                return "missing_transition_predictions"
            if causal_targets:
                return "preregistered_causal_transition"
            return "verified_postcondition"

        return "state_scoped_inspection"


@dataclass(frozen=True)
class _ParsedSignal:
    signal: Mapping[str, Any]
    raw: Mapping[str, Any]
    binding: Mapping[str, Any]
    executed_request: Mapping[str, Any]
    provenance: Mapping[str, Any]
    matched_probe: Mapping[str, Any]


def _original_evidence_input(request: StructuredModelRequest) -> Mapping[str, Any]:
    if request.task == "judge_evidence":
        return request.input
    original_request = request.input.get("original_request")
    if not isinstance(original_request, Mapping):
        return {}
    original_input = original_request.get("input")
    if original_request.get("task") != "judge_evidence" or not isinstance(
        original_input, Mapping
    ):
        return {}
    return original_input


def _declared_signal_id(
    request: StructuredModelRequest,
    request_input: Mapping[str, Any],
) -> str:
    signal = request_input.get("signal")
    if isinstance(signal, Mapping) and isinstance(signal.get("id"), str):
        return signal["id"]
    signal_id = request_input.get("signal_id")
    if isinstance(signal_id, str):
        return signal_id
    metadata_signal_id = request.metadata.get("signal_id")
    return metadata_signal_id if isinstance(metadata_signal_id, str) else ""


def _parsed_signal(request_input: Mapping[str, Any]) -> _ParsedSignal | None:
    signal = request_input.get("signal")
    provenance = request_input.get("provenance")
    matched_probe = request_input.get("matched_probe")
    if (
        not isinstance(signal, Mapping)
        or not isinstance(provenance, Mapping)
        or not isinstance(matched_probe, Mapping)
    ):
        return None
    raw_content = signal.get("raw_content")
    if not isinstance(raw_content, str):
        return None
    try:
        raw = json.loads(raw_content)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(raw, Mapping):
        return None
    binding = raw.get("causal_binding")
    executed_request = raw.get("executed_request")
    if not isinstance(binding, Mapping) or not isinstance(executed_request, Mapping):
        return None
    return _ParsedSignal(
        signal=signal,
        raw=raw,
        binding=binding,
        executed_request=executed_request,
        provenance=provenance,
        matched_probe=matched_probe,
    )


def _has_exact_request_binding(
    *,
    request: StructuredModelRequest,
    request_input: Mapping[str, Any],
    parsed_signal: _ParsedSignal | None,
    context: _CausalSignalContext,
) -> bool:
    if parsed_signal is None:
        return False
    record = context.record
    binding = parsed_signal.binding
    raw = parsed_signal.raw
    signal = parsed_signal.signal
    matched_probe = parsed_signal.matched_probe
    expected_request = executed_request_from_action(record.observation.action)
    expected_fingerprint = "sha256:" + canonical_sha256(expected_request)
    metadata_plan_id = request.metadata.get("plan_id")
    if metadata_plan_id is not None and metadata_plan_id != record.plan_id:
        return False
    return (
        signal.get("id") == _declared_signal_id(request, request_input)
        and request.metadata.get("signal_id") == signal.get("id")
        and request.metadata.get("run_id") == record.run_id
        and request.metadata.get("cycle_id") == record.cycle_id
        and signal.get("cycle_id") == record.cycle_id
        and signal.get("generated_by_probe") == record.probe_id
        and matched_probe.get("id") == record.probe_id
        and binding.get("action_id") == record.action_id
        and binding.get("action_role") == record.action_role
        and binding.get("plan_id") == record.plan_id
        and binding.get("request_fingerprint") == expected_fingerprint
        and binding.get("verification_target") == record.verification_target
        and parsed_signal.executed_request == expected_request
        and raw.get("action_index") == record.observation.action_index
        and record.request_fingerprint == expected_fingerprint
    )


def _target_types(
    request_input: Mapping[str, Any],
    *,
    context: _CausalSignalContext,
) -> dict[str, str]:
    hypotheses = request_input.get("hypotheses")
    if not _is_nonstring_sequence(hypotheses):
        return {}
    target_types: dict[str, str] = {}
    for hypothesis in hypotheses:
        if not isinstance(hypothesis, Mapping):
            return {}
        hypothesis_id = hypothesis.get("id")
        hypothesis_type = hypothesis.get("type")
        if not isinstance(hypothesis_id, str) or not isinstance(
            hypothesis_type, str
        ):
            return {}
        if hypothesis_id in target_types:
            return {}
        target_types[hypothesis_id] = hypothesis_type
    registered_types = {
        target: context.hypothesis_types[target]
        for target in context.target_hypotheses
        if target in context.hypothesis_types
    }
    if set(registered_types) == set(context.target_hypotheses):
        return registered_types
    return target_types


def _framing_response_hypothesis_types(
    response: Mapping[str, Any],
) -> dict[str, str]:
    hypotheses = response.get("hypotheses")
    if not _is_nonstring_sequence(hypotheses):
        return {}
    result: dict[str, str] = {}
    for index, hypothesis in enumerate(hypotheses, start=1):
        if not isinstance(hypothesis, Mapping):
            return {}
        hypothesis_type = hypothesis.get("type")
        if not isinstance(hypothesis_type, str):
            return {}
        result[f"H{index}"] = hypothesis_type
    return result


def _task_frame_hypothesis_types(task_frame: object) -> dict[str, str]:
    if not isinstance(task_frame, Mapping):
        return {}
    hypothesis_frame = task_frame.get("hypothesis_frame")
    if not isinstance(hypothesis_frame, Mapping):
        return {}
    hypotheses = hypothesis_frame.get("hypotheses")
    if not _is_nonstring_sequence(hypotheses):
        return {}
    result: dict[str, str] = {}
    for hypothesis in hypotheses:
        if not isinstance(hypothesis, Mapping):
            return {}
        hypothesis_id = hypothesis.get("id")
        hypothesis_type = hypothesis.get("type")
        if not isinstance(hypothesis_id, str) or not isinstance(
            hypothesis_type, str
        ):
            return {}
        if hypothesis_id in result:
            return {}
        result[hypothesis_id] = hypothesis_type
    return result


def _targets_match(
    *,
    request_input: Mapping[str, Any],
    response: Mapping[str, Any],
    context: _CausalSignalContext,
    target_types: Mapping[str, str],
) -> bool:
    targets = _string_sequence(request_input.get("target_hypotheses"))
    likelihoods = response.get("likelihoods")
    signal = request_input.get("signal")
    matched_probe = request_input.get("matched_probe")
    if (
        targets is None
        or len(targets) != len(set(targets))
        or not isinstance(likelihoods, Mapping)
        or not isinstance(signal, Mapping)
        or not isinstance(matched_probe, Mapping)
    ):
        return False
    signal_targets = _string_sequence(signal.get("initial_target_hypotheses"))
    probe_targets = _string_sequence(matched_probe.get("target_hypotheses"))
    expected = set(targets)
    return (
        set(str(key) for key in likelihoods) == expected
        and set(target_types) == expected
        and signal_targets is not None
        and len(signal_targets) == len(set(signal_targets))
        and set(signal_targets) == expected
        and probe_targets is not None
        and len(probe_targets) == len(set(probe_targets))
        and set(probe_targets) == expected
        and len(context.target_hypotheses) == len(set(context.target_hypotheses))
        and set(context.target_hypotheses) == expected
    )


def _action_role_accepts_target_types(
    *,
    action_role: str,
    target_types: Mapping[str, str],
) -> bool:
    hypothesis_types = set(target_types.values())
    if not hypothesis_types or not hypothesis_types.issubset(
        _TERMINAL_HYPOTHESIS_TYPES | _POLICY_HYPOTHESIS_TYPES
    ):
        return False
    if action_role == "inspect":
        return hypothesis_types.issubset(
            _CURRENT_STATE_INSPECTION_TYPES | _POLICY_HYPOTHESIS_TYPES
        )
    if action_role == "verify":
        return hypothesis_types.issubset(
            _DIRECT_VERIFICATION_TYPES
            | _CAUSAL_TRANSITION_TYPES
            | _POLICY_HYPOTHESIS_TYPES
        )
    return action_role == "intervene"


def _is_stale(
    *,
    request_input: Mapping[str, Any],
    parsed_signal: _ParsedSignal | None,
    context: _CausalSignalContext,
) -> bool:
    if parsed_signal is None:
        return True
    record = context.record
    raw = parsed_signal.raw
    if (
        parsed_signal.binding.get("policy_attempt_id") != record.policy_attempt_id
        or parsed_signal.binding.get("subject_environment_state_id")
        != record.subject_environment_state_id
        or parsed_signal.provenance.get("environment_state_id")
        != record.subject_environment_state_id
        or raw.get("pre_environment_state_id")
        != record.pre_environment_state_id
        or raw.get("post_environment_state_id")
        != record.post_environment_state_id
    ):
        return True

    same_plan_preintervention = _is_same_plan_preintervention_inspection(context)
    if same_plan_preintervention:
        return False
    if (
        context.current_action.policy_attempt_id != record.policy_attempt_id
        or context.current_intervention_generation != record.intervention_generation
    ):
        return True
    return (
        record.action_role == "inspect"
        and record.subject_environment_state_id
        != context.current_environment_state_id
    )


def _is_same_plan_preintervention_inspection(
    context: _CausalSignalContext,
) -> bool:
    record = context.record
    if (
        record.action_role != "inspect"
        or context.current_action.plan_id != record.plan_id
        or context.plan.plan.mode != "intervene"
    ):
        return False
    intervention_indexes = [
        index
        for index, step in enumerate(context.plan.plan.steps)
        if step.role == "intervene"
    ]
    return (
        len(intervention_indexes) == 1
        and record.step_index < intervention_indexes[0]
        and context.current_intervention_generation
        == record.intervention_generation + 1
    )


def _has_differentiated_transition_predictions(
    *,
    causal_targets: set[str],
    predictions: Mapping[str, str],
) -> bool:
    if not causal_targets.issubset(predictions):
        return False
    normalized = [
        _normalize_prediction(predictions[target])
        for target in sorted(causal_targets)
        if isinstance(predictions[target], str) and predictions[target].strip()
    ]
    return len(normalized) == len(causal_targets) and len(set(normalized)) == len(
        normalized
    )


def _normalize_prediction(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _decision_lineage(
    request_input: Mapping[str, Any],
    context: _CausalSignalContext | None,
) -> tuple[str, Literal["inspect", "intervene", "verify"], str]:
    if context is not None:
        return (
            context.record.action_id,
            context.record.action_role,
            context.record.subject_environment_state_id,
        )
    parsed_signal = _parsed_signal(request_input)
    binding = parsed_signal.binding if parsed_signal is not None else {}
    action_id = binding.get("action_id")
    action_role = binding.get("action_role")
    subject_environment_state_id = binding.get("subject_environment_state_id")
    if action_role not in {"inspect", "intervene", "verify"}:
        action_role = "inspect"
    return (
        action_id if isinstance(action_id, str) else "unbound",
        action_role,
        (
            subject_environment_state_id
            if isinstance(subject_environment_state_id, str)
            else ""
        ),
    )


def _judgment_response_sha256(response: Mapping[str, Any]) -> str:
    try:
        serialized = canonical_json(response)
    except (TypeError, ValueError):
        serialized = json.dumps(
            response,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=lambda value: type(value).__qualname__,
        )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _is_nonstring_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes)


def _string_sequence(value: object) -> tuple[str, ...] | None:
    if not _is_nonstring_sequence(value) or any(
        not isinstance(item, str) for item in value
    ):
        return None
    return tuple(value)


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def executed_request_from_action(action: TerminalAction) -> dict[str, Any]:
    if isinstance(action, ShellAction):
        return {
            "command": action.command,
            "mutates_environment": action_may_mutate(action),
            "timeout_seconds": action.timeout_seconds,
            "type": action.type,
        }
    if isinstance(action, WriteFileAction):
        content = action.content.encode("utf-8")
        return {
            "content_bytes": len(content),
            "content_sha256": hashlib.sha256(content).hexdigest(),
            "path": action.path,
            "type": action.type,
        }
    if isinstance(action, ApplyPatchAction):
        patch = action.patch.encode("utf-8")
        return {
            "patch_bytes": len(patch),
            "patch_sha256": hashlib.sha256(patch).hexdigest(),
            "strip": action.strip,
            "type": action.type,
        }
    raise TypeError(f"unsupported terminal action: {type(action).__name__}")
