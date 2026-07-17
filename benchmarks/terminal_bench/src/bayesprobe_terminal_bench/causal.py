from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from threading import RLock
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from bayesprobe import ExternalSignal, ProbeDesign, ProbeExecutionBrief
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


class CausalTraceError(ValueError):
    pass


class _PlanState:
    def __init__(self) -> None:
        self.last_environment_state_id: str | None = None
        self.intervention_count = 0
        self.registered_steps: set[int] = set()


class CausalTraceRegistry:
    """Bind frozen Probe plans, completed actions, and emitted Signals."""

    def __init__(self) -> None:
        self._plans: dict[str, RegisteredPlan] = {}
        self._policy_attempt_ids: set[str] = set()
        self._plan_states: dict[str, _PlanState] = {}
        self._last_environment_state_by_run: dict[str, str] = {}
        self._intervention_generation_by_run: dict[str, int] = {}
        self._actions: dict[str, CausalActionRecord] = {}
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
            self._policy_attempt_ids.add(policy_attempt_id)
            self._plan_states[plan_id] = _PlanState()
        return registered

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
