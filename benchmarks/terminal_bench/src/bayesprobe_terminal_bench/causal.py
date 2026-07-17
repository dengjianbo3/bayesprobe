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
    signal: ExternalSignal
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
        self._provisional_frame_type_runs: set[str] = set()
        self._policy_attempt_ids: set[str] = set()
        self._plan_states: dict[str, _PlanState] = {}
        self._last_environment_state_by_run: dict[str, str] = {}
        self._intervention_generation_by_run: dict[str, int] = {}
        self._actions: dict[str, CausalActionRecord] = {}
        self._latest_action_id_by_run: dict[str, str] = {}
        self._action_to_signal: dict[str, str] = {}
        self._signal_to_action: dict[str, str] = {}
        self._signals: dict[str, ExternalSignal] = {}
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
            stored_plan = registered.model_copy(deep=True)
            self._plans[plan_id] = stored_plan
            self._target_hypotheses_by_plan[plan_id] = tuple(
                probe.target_hypotheses
            )
            hypothesis_types = dict(
                self._hypothesis_types_by_run.get(context.run_id, {})
            )
            for hypothesis_id, hypothesis_type in _task_frame_hypothesis_types(
                getattr(context, "task_frame", {})
            ).items():
                hypothesis_types.setdefault(hypothesis_id, hypothesis_type)
            self._hypothesis_types_by_plan[plan_id] = hypothesis_types
            self._provisional_frame_type_runs.discard(context.run_id)
            self._policy_attempt_ids.add(policy_attempt_id)
            self._plan_states[plan_id] = _PlanState()
        return stored_plan.model_copy(deep=True)

    def _register_frame_hypothesis_types(
        self,
        *,
        run_id: str,
        hypothesis_types: Mapping[str, str],
        provisional: bool = False,
    ) -> None:
        with self._lock:
            existing = self._hypothesis_types_by_run.get(run_id)
            incoming = dict(hypothesis_types)
            if existing is not None and existing != incoming:
                raise CausalTraceError("contradictory task-frame hypothesis types")
            self._hypothesis_types_by_run[run_id] = incoming
            if provisional:
                self._provisional_frame_type_runs.add(run_id)
            else:
                self._provisional_frame_type_runs.discard(run_id)

    def _replace_provisional_frame_hypothesis_types(
        self,
        *,
        run_id: str,
        hypothesis_types: Mapping[str, str],
    ) -> None:
        with self._lock:
            if any(plan.run_id == run_id for plan in self._plans.values()):
                raise CausalTraceError("task-frame hypothesis types are already frozen")
            if (
                run_id in self._hypothesis_types_by_run
                and run_id not in self._provisional_frame_type_runs
            ):
                raise CausalTraceError("task-frame hypothesis types are not provisional")
            self._hypothesis_types_by_run[run_id] = dict(hypothesis_types)
            self._provisional_frame_type_runs.discard(run_id)

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

            stored_record = record.model_copy(deep=True)
            self._actions[action_id] = stored_record
            self._latest_action_id_by_run[plan.run_id] = action_id
            state.last_environment_state_id = observation.post_environment_state_id
            state.intervention_count = intervention_count
            state.registered_steps.add(step_index)
            self._last_environment_state_by_run[plan.run_id] = (
                observation.post_environment_state_id
            )
            self._intervention_generation_by_run[plan.run_id] = intervention_generation
        return stored_record.model_copy(deep=True)

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
            signal = signal_builder(self._actions[action_id].model_copy(deep=True))
            if not isinstance(signal, ExternalSignal):
                raise TypeError("signal_builder must return an ExternalSignal")
            signal_id = signal.id
            if signal_id in self._signal_to_action:
                raise ValueError(f"Signal ID is already bound: {signal_id}")
            stored_signal = signal.model_copy(deep=True)
            self._action_to_signal[action_id] = signal_id
            self._signal_to_action[signal_id] = action_id
            self._signals[signal_id] = stored_signal
            return stored_signal.model_copy(deep=True)

    def record_for_signal(self, signal_id: str) -> CausalActionRecord:
        with self._lock:
            try:
                action_id = self._signal_to_action[signal_id]
            except KeyError as error:
                raise KeyError(f"unknown Signal: {signal_id}") from error
            try:
                record = self._actions[action_id]
            except KeyError as error:
                raise CausalTraceError(
                    "incomplete registered causal lineage"
                ) from error
            bound_signal_ids = {
                bound_signal_id
                for bound_signal_id, bound_action_id in self._signal_to_action.items()
                if bound_action_id == action_id
            }
            if (
                self._action_to_signal.get(action_id) != signal_id
                or bound_signal_ids != {signal_id}
            ):
                raise CausalTraceError("ambiguous Signal binding")
            return record.model_copy(deep=True)

    def _admissibility_context_for_signal(
        self,
        signal_id: str,
    ) -> _CausalSignalContext:
        with self._lock:
            if signal_id not in self._signal_to_action:
                raise KeyError(f"unknown Signal: {signal_id}")
            try:
                action_id = self._signal_to_action[signal_id]
                record = self._actions[action_id]
                signal = self._signals[signal_id]
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
                raise CausalTraceError(
                    "incomplete registered causal lineage"
                ) from error
            bound_signal_ids = {
                bound_signal_id
                for bound_signal_id, bound_action_id in self._signal_to_action.items()
                if bound_action_id == action_id
            }
            if (
                self._action_to_signal.get(action_id) != signal_id
                or bound_signal_ids != {signal_id}
            ):
                raise CausalTraceError("ambiguous Signal binding")
            if signal.id != signal_id:
                raise CausalTraceError("registered Signal identity contradiction")
            expected_request = executed_request_from_action(record.observation.action)
            expected_fingerprint = "sha256:" + canonical_sha256(expected_request)
            if record.request_fingerprint != expected_fingerprint:
                raise CausalTraceError("registered request fingerprint contradiction")
            if (
                plan.plan_id != record.plan_id
                or plan.policy_attempt_id != record.policy_attempt_id
                or plan.probe_id != record.probe_id
            ):
                raise CausalTraceError("registered causal lineage contradiction")
            if not _registered_signal_matches_record(
                signal=signal,
                record=record,
                target_hypotheses=target_hypotheses,
                expected_request=expected_request,
            ):
                raise CausalTraceError("registered Signal lineage contradiction")
            return _CausalSignalContext(
                record=record.model_copy(deep=True),
                signal=signal.model_copy(deep=True),
                plan=plan.model_copy(deep=True),
                target_hypotheses=target_hypotheses,
                hypothesis_types=dict(hypothesis_types),
                current_action=current_action.model_copy(deep=True),
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
        if request.task in {"frame_open_question", "repair_task_frame"}:
            _capture_framing_hypothesis_types(
                request=request,
                response=response,
                registry=self._registry,
            )
        if request.task not in _EVIDENCE_TASKS:
            return response

        request_input, declarations_valid = _evidence_request_input(request)
        declared_signal_id = _declared_signal_id(request, request_input)
        signal_id = declared_signal_id or ""
        response_sha256 = _judgment_response_sha256(response)
        context: _CausalSignalContext | None = None
        if declarations_valid and declared_signal_id is not None:
            try:
                context = self._registry._admissibility_context_for_signal(
                    declared_signal_id
                )
            except KeyError:
                pass

        reason_code = self._classify(
            request=request,
            request_input=request_input,
            response=response,
            context=context,
            declarations_valid=declarations_valid,
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
        declarations_valid: bool,
    ) -> str:
        # Precedence is part of the public causal contract. Keep these checks ordered.
        parsed_signal = _parsed_signal(request_input)
        if not declarations_valid or context is None or not _has_exact_request_binding(
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


def _evidence_request_input(
    request: StructuredModelRequest,
) -> tuple[Mapping[str, Any], bool]:
    if request.task == "judge_evidence":
        declarations_valid = (
            request.prompt_id == "evidence_judgment"
            and request.prompt_version == "v0.2"
            and request.schema_name == "EvidenceJudgment"
            and request.schema_version == "v0.2"
            and "repair_attempt_index" not in request.metadata
        )
        return request.input, declarations_valid

    original_request = request.input.get("original_request")
    if not isinstance(original_request, Mapping):
        return {}, False
    original_input = original_request.get("input")
    if original_request.get("task") != "judge_evidence" or not isinstance(
        original_input, Mapping
    ):
        return {}, False

    input_attempt_index = request.input.get("attempt_index")
    metadata_attempt_index = request.metadata.get("repair_attempt_index")
    declarations_valid = (
        request.prompt_id == "evidence_judgment_repair"
        and request.prompt_version == "v0.2"
        and request.schema_name == "EvidenceJudgment"
        and request.schema_version == "v0.2"
        and type(input_attempt_index) is int
        and input_attempt_index > 0
        and metadata_attempt_index == input_attempt_index
        and type(metadata_attempt_index) is int
        and _optional_declaration_matches(
            original_request,
            field_name="prompt_id",
            expected="evidence_judgment",
        )
        and _optional_declaration_matches(
            original_request,
            field_name="prompt_version",
            expected=request.prompt_version,
        )
        and _optional_declaration_matches(
            original_request,
            field_name="schema_name",
            expected=request.schema_name,
        )
        and _optional_declaration_matches(
            original_request,
            field_name="schema_version",
            expected=request.schema_version,
        )
        and _repair_original_metadata_matches(
            original_request=original_request,
            outer_metadata=request.metadata,
        )
    )
    return original_input, declarations_valid


def _optional_declaration_matches(
    declaration: Mapping[str, Any],
    *,
    field_name: str,
    expected: object,
) -> bool:
    return field_name not in declaration or declaration[field_name] == expected


def _repair_original_metadata_matches(
    *,
    original_request: Mapping[str, Any],
    outer_metadata: Mapping[str, Any],
) -> bool:
    if "metadata" not in original_request:
        return True
    original_metadata = original_request["metadata"]
    if not isinstance(original_metadata, Mapping):
        return False
    expected_metadata = dict(outer_metadata)
    expected_metadata.pop("repair_attempt_index", None)
    return dict(original_metadata) == expected_metadata


def _declared_signal_id(
    request: StructuredModelRequest,
    request_input: Mapping[str, Any],
) -> str | None:
    signal = request_input.get("signal")
    if not isinstance(signal, Mapping):
        return None
    signal_id = signal.get("id")
    metadata_signal_id = request.metadata.get("signal_id")
    if (
        not isinstance(signal_id, str)
        or not signal_id.strip()
        or not isinstance(metadata_signal_id, str)
        or not metadata_signal_id.strip()
    ):
        return None
    declarations = [signal_id, metadata_signal_id]
    if "signal_id" in request_input:
        input_signal_id = request_input["signal_id"]
        if not isinstance(input_signal_id, str) or not input_signal_id.strip():
            return None
        declarations.append(input_signal_id)
    if len(set(declarations)) != 1:
        return None
    return signal_id


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
    raw = _json_object(raw_content)
    if raw is None:
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


def _json_object(value: str) -> Mapping[str, Any] | None:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate JSON object key")
            result[key] = item
        return result

    try:
        parsed = json.loads(value, object_pairs_hook=unique_object)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, Mapping) else None


def _public_signal_projection(signal: ExternalSignal) -> dict[str, Any]:
    return {
        "id": signal.id,
        "cycle_id": signal.cycle_id,
        "signal_kind": signal.signal_kind.value,
        "source_type": signal.source_type,
        "source": signal.source,
        "raw_content": signal.raw_content,
        "generated_by_probe": signal.generated_by_probe,
        "inbox_status": signal.inbox_status.value,
        "initial_target_hypotheses": list(signal.initial_target_hypotheses),
    }


def _public_signal_provenance(signal: ExternalSignal) -> dict[str, Any] | None:
    if signal.provenance is None:
        return None
    provenance = signal.provenance.model_dump(mode="json")
    # The unchanged public core preserves the incoming group in this audit field.
    if provenance["supplied_correlation_group"] is None:
        provenance["supplied_correlation_group"] = provenance["correlation_group"]
    return provenance


def _matches_public_signal_snapshot(
    *,
    parsed_signal: _ParsedSignal,
    registered_signal: ExternalSignal,
) -> bool:
    provided_projection = dict(parsed_signal.signal)
    provided_projection.pop("raw_content", None)
    expected_projection = _public_signal_projection(registered_signal)
    expected_raw_content = expected_projection.pop("raw_content")
    expected_raw = _json_object(expected_raw_content)
    return (
        expected_raw is not None
        and provided_projection == expected_projection
        and parsed_signal.raw == expected_raw
        and parsed_signal.provenance == _public_signal_provenance(registered_signal)
    )


def _registered_signal_matches_record(
    *,
    signal: ExternalSignal,
    record: CausalActionRecord,
    target_hypotheses: Sequence[str],
    expected_request: Mapping[str, Any],
) -> bool:
    provenance = signal.provenance
    if provenance is None:
        return False
    raw = _json_object(signal.raw_content)
    if raw is None:
        return False
    binding = raw.get("causal_binding")
    executed_request = raw.get("executed_request")
    if not isinstance(binding, Mapping) or not isinstance(
        executed_request, Mapping
    ):
        return False
    return (
        signal.cycle_id == record.cycle_id
        and signal.generated_by_probe == record.probe_id
        and tuple(signal.initial_target_hypotheses) == tuple(target_hypotheses)
        and provenance.environment_state_id == record.subject_environment_state_id
        and binding.get("action_id") == record.action_id
        and binding.get("action_role") == record.action_role
        and binding.get("plan_id") == record.plan_id
        and binding.get("policy_attempt_id") == record.policy_attempt_id
        and binding.get("request_fingerprint") == record.request_fingerprint
        and binding.get("subject_environment_state_id")
        == record.subject_environment_state_id
        and binding.get("verification_target") == record.verification_target
        and executed_request == expected_request
        and raw.get("action_index") == record.observation.action_index
        and raw.get("pre_environment_state_id")
        == record.pre_environment_state_id
        and raw.get("post_environment_state_id")
        == record.post_environment_state_id
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
        _matches_public_signal_snapshot(
            parsed_signal=parsed_signal,
            registered_signal=context.signal,
        )
        and signal.get("id") == _declared_signal_id(request, request_input)
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
        registered_type = context.hypothesis_types.get(hypothesis_id)
        if registered_type is not None:
            # Initial public-core Hypothesis objects currently project their type as
            # the opaque "claim" placeholder; the frozen frame remains authoritative.
            if hypothesis_type not in {registered_type, "claim"}:
                return {}
            target_types[hypothesis_id] = registered_type
        else:
            target_types[hypothesis_id] = hypothesis_type
    return target_types


def _capture_framing_hypothesis_types(
    *,
    request: StructuredModelRequest,
    response: Mapping[str, Any],
    registry: CausalTraceRegistry,
) -> None:
    run_id = request.metadata.get("run_id")
    hypothesis_types = _framing_response_hypothesis_types(response)
    if not isinstance(run_id, str) or not run_id.strip() or not hypothesis_types:
        return
    if request.task == "frame_open_question":
        registry._register_frame_hypothesis_types(
            run_id=run_id,
            hypothesis_types=hypothesis_types,
            provisional=True,
        )
        return
    if not _is_public_frame_repair_request(request):
        return
    registry._replace_provisional_frame_hypothesis_types(
        run_id=run_id,
        hypothesis_types=hypothesis_types,
    )


def _is_public_frame_repair_request(request: StructuredModelRequest) -> bool:
    input_attempt_index = request.input.get("attempt_index")
    metadata_attempt_index = request.metadata.get("repair_attempt_index")
    return (
        request.task == "repair_task_frame"
        and request.prompt_id == "open_question_task_framing_repair"
        and request.prompt_version == "v0.2"
        and request.schema_name == "OpenQuestionTaskFrame"
        and request.schema_version == "v0.2"
        and type(input_attempt_index) is int
        and input_attempt_index > 0
        and type(metadata_attempt_index) is int
        and metadata_attempt_index == input_attempt_index
        and isinstance(request.input.get("original_request"), Mapping)
    )


def _framing_response_hypothesis_types(
    response: Mapping[str, Any],
) -> dict[str, str]:
    # This is a projection only. TerminalContractModelGateway owns validation.
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
    likelihood_keys = tuple(likelihoods)
    return (
        all(isinstance(key, str) for key in likelihood_keys)
        and set(likelihood_keys) == expected
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
