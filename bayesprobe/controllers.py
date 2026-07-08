from __future__ import annotations

from dataclasses import dataclass

from bayesprobe.core import BayesProbeCore
from bayesprobe.ledger import JsonlLedgerStore
from bayesprobe.projections import build_answer_projection, build_belief_state_projection
from bayesprobe.schemas import (
    AnswerProjection,
    BeliefState,
    BeliefStateProjection,
    BeliefUpdate,
    CycleRecord,
    CycleSignalShape,
    EvidenceEvent,
    ExternalSignal,
    HypothesisEvolution,
    ProbeSet,
)


@dataclass(frozen=True)
class ControllerResult:
    cycle: CycleRecord
    belief_state: BeliefState
    evidence_events: list[EvidenceEvent]
    belief_updates: list[BeliefUpdate]
    hypothesis_evolutions: list[HypothesisEvolution]
    answer_projection: AnswerProjection | None = None
    belief_state_projection: BeliefStateProjection | None = None


def _next_cycle_id(run_id: str, belief_state: BeliefState) -> str:
    current = belief_state.cycle_id
    scoped_prefix = f"{run_id}_cycle_"
    if current.startswith(scoped_prefix):
        suffix = current[len(scoped_prefix) :]
        if suffix.isdigit():
            return f"{scoped_prefix}{int(suffix) + 1}"
    if current.startswith("cycle_"):
        suffix = current[len("cycle_") :]
        if suffix.isdigit():
            return f"{run_id}_cycle_{int(suffix) + 1}"
    return f"{run_id}_cycle_{belief_state.cycle_index + 1}"


class AutonomousController:
    def __init__(self, core: BayesProbeCore, ledger: JsonlLedgerStore | None = None):
        self.core = core
        self._ledger = core.ledger if ledger is None else ledger

    def run_once(
        self,
        run_id: str,
        belief_state: BeliefState,
        active_signals: list[ExternalSignal],
    ) -> ControllerResult:
        cycle_id = self.core.allocate_cycle_id(_next_cycle_id(run_id, belief_state))
        cycle = CycleRecord(
            cycle_id=cycle_id,
            run_id=run_id,
            cycle_index=belief_state.cycle_index + 1,
            signal_shape=CycleSignalShape.ACTIVE_ONLY,
        )
        probe_set = ProbeSet(
            probe_set_id=f"ps_{cycle_id}",
            cycle_id=cycle_id,
            probes=[],
            selection_reason="Autonomous cycle with provided active signals.",
            may_be_empty=True,
        )
        core_result = self.core.integrate_cycle(
            cycle=cycle,
            belief_state=belief_state,
            probe_set=probe_set,
            signals=active_signals,
        )
        answer_projection = build_answer_projection(cycle_id, belief_state, core_result)
        if self._ledger is not None:
            self._ledger.append("answer_projection", answer_projection)
        return ControllerResult(
            cycle=core_result.cycle,
            belief_state=core_result.belief_state,
            evidence_events=core_result.evidence_events,
            belief_updates=core_result.belief_updates,
            hypothesis_evolutions=core_result.hypothesis_evolutions,
            answer_projection=answer_projection,
        )


class SynchronizedController:
    def __init__(self, core: BayesProbeCore, ledger: JsonlLedgerStore | None = None):
        self.core = core
        self._ledger = core.ledger if ledger is None else ledger

    def process_round(
        self,
        run_id: str,
        round_id: str,
        belief_state: BeliefState,
        passive_signals: list[ExternalSignal],
    ) -> ControllerResult:
        cycle_id = self.core.allocate_cycle_id(_next_cycle_id(run_id, belief_state))
        cycle = CycleRecord(
            cycle_id=cycle_id,
            run_id=run_id,
            round_id=round_id,
            cycle_index=belief_state.cycle_index + 1,
            signal_shape=CycleSignalShape.PASSIVE_ONLY,
        )
        probe_set = ProbeSet(
            probe_set_id=f"ps_{cycle_id}",
            cycle_id=cycle_id,
            probes=[],
            selection_reason="Passive-only synchronized cycle.",
            may_be_empty=True,
        )
        core_result = self.core.integrate_cycle(
            cycle=cycle,
            belief_state=belief_state,
            probe_set=probe_set,
            signals=passive_signals,
        )
        belief_state_projection = build_belief_state_projection(cycle_id, belief_state, core_result)
        if self._ledger is not None:
            self._ledger.append("belief_state_projection", belief_state_projection)
        return ControllerResult(
            cycle=core_result.cycle,
            belief_state=core_result.belief_state,
            evidence_events=core_result.evidence_events,
            belief_updates=core_result.belief_updates,
            hypothesis_evolutions=core_result.hypothesis_evolutions,
            belief_state_projection=belief_state_projection,
        )
