from __future__ import annotations

from threading import Lock
from typing import Any

from bayesprobe import ExternalSignal, ProbeDesign, ProbeExecutionBrief

from bayesprobe_terminal_bench.actions import ActionObservation
from bayesprobe_terminal_bench.causal import CausalTraceRegistry
from bayesprobe_terminal_bench.config import BudgetExhausted
from bayesprobe_terminal_bench.environment import PolicyViolation
from bayesprobe_terminal_bench.planning import TerminalPlanError
from bayesprobe_terminal_bench.signals import signal_from_observation


_STABLE_PLAN_ERROR_CATEGORIES = frozenset(
    {"plan_error", "provider_contract_error", "provider_error"}
)


class HarborProbeToolGateway:
    """Public ProbeToolGateway adapter for completed Harbor environment actions."""

    def __init__(self, *, planner: Any, bridge: Any, artifacts: Any, budget: Any) -> None:
        self._planner = planner
        self._bridge = bridge
        self._artifacts = artifacts
        self._budget = budget
        self._causal = CausalTraceRegistry()
        self._histories: dict[str, list[ActionObservation]] = {}
        self._execute_lock = Lock()

    def execute_probe(
        self,
        *,
        probe: ProbeDesign,
        context: ProbeExecutionBrief,
    ) -> list[ExternalSignal]:
        # Serialize planning through artifact/history updates so a later probe
        # cannot observe a stale same-run history or overtake an earlier plan.
        with self._execute_lock:
            return self._execute_probe(probe=probe, context=context)

    def _execute_probe(
        self,
        *,
        probe: ProbeDesign,
        context: ProbeExecutionBrief,
    ) -> list[ExternalSignal]:
        history = self._histories.setdefault(context.run_id, [])
        try:
            plan = self._planner.plan(
                probe=probe,
                context=context,
                history=tuple(history[-12:]),
            )
        except BudgetExhausted:
            self._record_decision(
                category="budget_exhausted",
                probe=probe,
                context=context,
                stage="planning",
            )
            raise
        except TerminalPlanError as error:
            self._record_decision(
                category=(
                    error.category
                    if error.category in _STABLE_PLAN_ERROR_CATEGORIES
                    else "plan_error"
                ),
                probe=probe,
                context=context,
                stage="planning",
                error_type=type(error).__name__,
            )
            raise

        registered_plan = self._causal.register_plan(
            probe=probe,
            context=context,
            plan=plan,
        )
        self._artifacts.append_plan(
            {
                "probe_id": probe.id,
                "cycle_id": context.cycle_id,
                "plan_id": registered_plan.plan_id,
                "policy_attempt_id": registered_plan.policy_attempt_id,
                "plan": plan.model_dump(mode="json"),
            }
        )
        signals: list[ExternalSignal] = []
        for step_index, step in enumerate(plan.steps):
            try:
                action_index = self._budget.reserve_action()
            except BudgetExhausted:
                self._record_decision(
                    category="budget_exhausted",
                    probe=probe,
                    context=context,
                    stage="action_budget",
                    plan_id=registered_plan.plan_id,
                    step_index=step_index,
                )
                raise

            try:
                observation = self._bridge.execute(step.action, action_index)
            except PolicyViolation as error:
                self._record_decision(
                    action_index=action_index,
                    category="policy_error",
                    probe=probe,
                    context=context,
                    stage="action_policy",
                    error_type=type(error).__name__,
                    plan_id=registered_plan.plan_id,
                    step_index=step_index,
                )
                continue

            causal_record = self._causal.register_action(
                plan=registered_plan,
                step_index=step_index,
                observation=observation,
            )
            signal = signal_from_observation(
                observation=observation,
                probe=probe,
                context=context,
                causal_record=causal_record,
            )
            self._causal.bind_signal(
                action_id=causal_record.action_id,
                signal_id=signal.id,
            )
            history.append(observation)
            self._artifacts.append_observation(observation)
            self._artifacts.append_causal_action(causal_record)
            signals.append(signal)
        return signals

    def _record_decision(
        self,
        *,
        category: str,
        probe: ProbeDesign,
        context: ProbeExecutionBrief,
        stage: str,
        action_index: int | None = None,
        error_type: str | None = None,
        plan_id: str | None = None,
        step_index: int | None = None,
    ) -> None:
        payload = {
            "category": category,
            "cycle_id": context.cycle_id,
            "probe_id": probe.id,
            "run_id": context.run_id,
            "stage": stage,
        }
        optional = {
            "action_index": action_index,
            "error_type": error_type,
            "plan_id": plan_id,
            "step_index": step_index,
        }
        payload.update(
            {key: value for key, value in optional.items() if value is not None}
        )
        self._artifacts.append_causal_decision(payload)
        self._artifacts.append_error(
            {
                key: value
                for key, value in payload.items()
                if key not in {"cycle_id", "plan_id", "run_id", "stage", "step_index"}
            }
        )
