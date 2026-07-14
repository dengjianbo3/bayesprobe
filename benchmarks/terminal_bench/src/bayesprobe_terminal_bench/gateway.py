from __future__ import annotations

from threading import Lock
from typing import Any

from bayesprobe import ExternalSignal, ProbeDesign, ProbeExecutionBrief

from bayesprobe_terminal_bench.actions import ActionObservation
from bayesprobe_terminal_bench.config import BudgetExhausted
from bayesprobe_terminal_bench.environment import PolicyViolation
from bayesprobe_terminal_bench.planning import TerminalPlanError
from bayesprobe_terminal_bench.signals import signal_from_observation


class HarborProbeToolGateway:
    """Public ProbeToolGateway adapter for completed Harbor environment actions."""

    def __init__(self, *, planner: Any, bridge: Any, artifacts: Any, budget: Any) -> None:
        self._planner = planner
        self._bridge = bridge
        self._artifacts = artifacts
        self._budget = budget
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
            self._artifacts.append_error(
                {"category": "budget_exhausted", "probe_id": probe.id}
            )
            return []
        except TerminalPlanError as error:
            self._artifacts.append_error(
                {
                    "category": "plan_error",
                    "error_type": type(error).__name__,
                    "probe_id": probe.id,
                }
            )
            return []

        self._artifacts.append_plan(
            {
                "probe_id": probe.id,
                "cycle_id": context.cycle_id,
                "plan": plan.model_dump(mode="json"),
            }
        )
        signals: list[ExternalSignal] = []
        for action in plan.actions:
            try:
                action_index = self._budget.reserve_action()
            except BudgetExhausted:
                self._artifacts.append_error(
                    {"category": "budget_exhausted", "probe_id": probe.id}
                )
                break

            try:
                observation = self._bridge.execute(action, action_index)
            except PolicyViolation as error:
                self._artifacts.append_error(
                    {
                        "category": "policy_error",
                        "error_type": type(error).__name__,
                        "probe_id": probe.id,
                    }
                )
                continue

            history.append(observation)
            self._artifacts.append_observation(observation)
            signals.append(
                signal_from_observation(
                    observation=observation,
                    probe=probe,
                    context=context,
                )
            )
        return signals
