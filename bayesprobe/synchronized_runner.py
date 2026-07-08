from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from bayesprobe.core import BayesProbeCore, CycleResult
from bayesprobe.initialization import (
    BayesProbeInitializer,
    InitializeRunInput,
)
from bayesprobe.probe_executor import (
    DeterministicProbeToolGateway,
    ProbeExecutionContext,
    ProbeExecutionResult,
    ProbeExecutor,
)
from bayesprobe.probe_planner import ProbePlanner, ProbePlanningConfig, ProbePlanningResult
from bayesprobe.projections import build_belief_state_projection
from bayesprobe.schemas import (
    BeliefState,
    BeliefStateProjection,
    BeliefUpdate,
    CycleRecord,
    CycleSignalShape,
    EvidenceEvent,
    ExternalSignal,
    HypothesisEvolution,
    ProbeCandidate,
    ProbeSet,
    RunRecord,
    SignalKind,
)


class SynchronizedRoundShape(StrEnum):
    PASSIVE_ONLY = "passive_only"
    ACTIVE_ONLY = "active_only"
    ACTIVE_PLUS_PASSIVE = "active_plus_passive"


@dataclass(frozen=True)
class SynchronizedRoundInput:
    round_id: str
    shape: SynchronizedRoundShape | str
    passive_signals: list[ExternalSignal] = field(default_factory=list)
    max_probes: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_nonempty(self.round_id, "round_id")
        if self.max_probes < 1:
            raise ValueError("max_probes must be at least 1")
        try:
            shape = SynchronizedRoundShape(self.shape)
        except ValueError as error:
            raise ValueError("shape must be a valid SynchronizedRoundShape") from error
        object.__setattr__(self, "shape", shape)

        if shape == SynchronizedRoundShape.ACTIVE_ONLY and self.passive_signals:
            raise ValueError("active-only synchronized rounds cannot include passive signals")
        if shape in {
            SynchronizedRoundShape.PASSIVE_ONLY,
            SynchronizedRoundShape.ACTIVE_PLUS_PASSIVE,
        } and not self.passive_signals:
            raise ValueError("passive or mixed synchronized rounds require passive signals")
        invalid_signal_ids = [
            signal.id
            for signal in self.passive_signals
            if signal.signal_kind != SignalKind.PASSIVE
        ]
        if invalid_signal_ids:
            raise ValueError("passive_signals must contain only passive external signals")


@dataclass(frozen=True)
class SynchronizedRunInput:
    rounds: list[SynchronizedRoundInput]
    initialize_input: InitializeRunInput | None = None
    run: RunRecord | None = None
    belief_state: BeliefState | None = None
    probe_candidates: list[ProbeCandidate] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.rounds:
            raise ValueError("rounds must not be empty")
        has_initialize_input = self.initialize_input is not None
        has_existing_run_part = self.run is not None or self.belief_state is not None
        if has_initialize_input and has_existing_run_part:
            raise ValueError("provide either initialize_input or existing run and belief_state")
        if not has_initialize_input and not (self.run is not None and self.belief_state is not None):
            raise ValueError("initialize_input or both run and belief_state are required")
        if self.run is not None and self.belief_state is not None:
            if self.run.run_id != self.belief_state.run_id:
                raise ValueError("run and belief_state must have the same run_id")


@dataclass(frozen=True)
class SynchronizedRoundResult:
    round_id: str
    cycle: CycleRecord
    shape: SynchronizedRoundShape
    probe_set: ProbeSet
    signals: list[ExternalSignal]
    active_signal_count: int
    passive_signal_count: int
    belief_state: BeliefState
    evidence_events: list[EvidenceEvent]
    belief_updates: list[BeliefUpdate]
    hypothesis_evolutions: list[HypothesisEvolution]
    belief_state_projection: BeliefStateProjection
    selected_probe_candidates: list[ProbeCandidate]
    remaining_probe_candidates: list[ProbeCandidate]


@dataclass(frozen=True)
class SynchronizedRunResult:
    run: RunRecord
    initial_belief_state: BeliefState
    final_belief_state: BeliefState
    round_results: list[SynchronizedRoundResult]
    final_belief_state_projection: BeliefStateProjection
    remaining_probe_candidates: list[ProbeCandidate]


@dataclass(frozen=True)
class _RoundExecution:
    probe_set: ProbeSet
    signals: list[ExternalSignal]
    active_signal_count: int
    passive_signal_count: int
    selected_probe_candidates: list[ProbeCandidate]


class SynchronizedRoundRunner:
    def __init__(
        self,
        *,
        core: BayesProbeCore,
        initializer: BayesProbeInitializer | None = None,
        planner: ProbePlanner | None = None,
        executor: ProbeExecutor | None = None,
    ) -> None:
        self.core = core
        self.initializer = initializer or BayesProbeInitializer(ledger=core.ledger)
        self.planner = planner or ProbePlanner(ledger=core.ledger)
        self.executor = executor or ProbeExecutor(
            gateway=DeterministicProbeToolGateway(),
            ledger=core.ledger,
        )

    def run_rounds(self, input: SynchronizedRunInput) -> SynchronizedRunResult:
        run, current_belief_state, candidate_pool = self._initial_state(input)
        initial_belief_state = current_belief_state
        round_results: list[SynchronizedRoundResult] = []

        for round_input in input.rounds:
            round_result = self._run_round(
                run=run,
                belief_state=current_belief_state,
                candidate_pool=candidate_pool,
                round_input=round_input,
            )
            round_results.append(round_result)
            current_belief_state = round_result.belief_state
            candidate_pool = list(round_result.remaining_probe_candidates)

        return SynchronizedRunResult(
            run=run,
            initial_belief_state=initial_belief_state,
            final_belief_state=current_belief_state,
            round_results=list(round_results),
            final_belief_state_projection=round_results[-1].belief_state_projection,
            remaining_probe_candidates=list(candidate_pool),
        )

    def _initial_state(
        self,
        input: SynchronizedRunInput,
    ) -> tuple[RunRecord, BeliefState, list[ProbeCandidate]]:
        if input.initialize_input is not None:
            initialization = self.initializer.initialize(input.initialize_input)
            return (
                initialization.run,
                initialization.belief_state,
                list(initialization.probe_candidates),
            )
        assert input.run is not None
        assert input.belief_state is not None
        return input.run, input.belief_state, list(input.probe_candidates)

    def _run_round(
        self,
        *,
        run: RunRecord,
        belief_state: BeliefState,
        candidate_pool: list[ProbeCandidate],
        round_input: SynchronizedRoundInput,
    ) -> SynchronizedRoundResult:
        cycle_id = self.core.allocate_cycle_id(
            f"{run.run_id}_cycle_{belief_state.cycle_index + 1}"
        )
        execution = self._execute_round(
            run=run,
            cycle_id=cycle_id,
            belief_state=belief_state,
            candidate_pool=candidate_pool,
            round_input=round_input,
        )
        cycle = CycleRecord(
            cycle_id=cycle_id,
            run_id=run.run_id,
            round_id=round_input.round_id,
            cycle_index=belief_state.cycle_index + 1,
            signal_shape=_cycle_signal_shape(round_input.shape),
            controller_metadata=dict(round_input.metadata),
        )
        core_result = self.core.integrate_cycle(
            cycle=cycle,
            belief_state=belief_state,
            probe_set=execution.probe_set,
            signals=execution.signals,
        )
        projection = build_belief_state_projection(cycle_id, belief_state, core_result)
        if self.core.ledger is not None:
            self.core.ledger.append("belief_state_projection", projection)
        remaining_candidates = _next_candidate_pool(
            previous_pool=candidate_pool,
            selected_candidates=execution.selected_probe_candidates,
            belief_state_projection=projection,
        )
        return _round_result(
            round_input=round_input,
            core_result=core_result,
            execution=execution,
            projection=projection,
            remaining_candidates=remaining_candidates,
        )

    def _execute_round(
        self,
        *,
        run: RunRecord,
        cycle_id: str,
        belief_state: BeliefState,
        candidate_pool: list[ProbeCandidate],
        round_input: SynchronizedRoundInput,
    ) -> _RoundExecution:
        passive_signals = [
            signal.model_copy(update={"cycle_id": cycle_id})
            for signal in round_input.passive_signals
        ]
        if round_input.shape == SynchronizedRoundShape.PASSIVE_ONLY:
            return _RoundExecution(
                probe_set=ProbeSet(
                    probe_set_id=f"ps_{cycle_id}",
                    cycle_id=cycle_id,
                    probes=[],
                    selection_reason="Passive-only synchronized round.",
                    may_be_empty=True,
                ),
                signals=passive_signals,
                active_signal_count=0,
                passive_signal_count=len(passive_signals),
                selected_probe_candidates=[],
            )

        planning = self.planner.design_probe_set(
            run_id=run.run_id,
            cycle_id=cycle_id,
            belief_state=belief_state,
            candidates=candidate_pool,
            config=ProbePlanningConfig(max_probes=round_input.max_probes),
        )
        execution = self.executor.execute_probe_set(
            probe_set=planning.probe_set,
            context=ProbeExecutionContext(
                run_id=run.run_id,
                cycle_id=cycle_id,
                belief_state=belief_state,
                metadata={"round_id": round_input.round_id},
            ),
        )
        signals = _round_signals(
            round_shape=round_input.shape,
            execution=execution,
            passive_signals=passive_signals,
        )
        return _RoundExecution(
            probe_set=planning.probe_set,
            signals=signals,
            active_signal_count=len(execution.signals),
            passive_signal_count=len(passive_signals),
            selected_probe_candidates=list(planning.selected_candidates),
        )


def _round_signals(
    *,
    round_shape: SynchronizedRoundShape,
    execution: ProbeExecutionResult,
    passive_signals: list[ExternalSignal],
) -> list[ExternalSignal]:
    if round_shape == SynchronizedRoundShape.ACTIVE_ONLY:
        return list(execution.signals)
    return [*execution.signals, *passive_signals]


def _cycle_signal_shape(round_shape: SynchronizedRoundShape) -> CycleSignalShape:
    if round_shape == SynchronizedRoundShape.PASSIVE_ONLY:
        return CycleSignalShape.PASSIVE_ONLY
    if round_shape == SynchronizedRoundShape.ACTIVE_ONLY:
        return CycleSignalShape.ACTIVE_ONLY
    return CycleSignalShape.ACTIVE_PLUS_PASSIVE


def _round_result(
    *,
    round_input: SynchronizedRoundInput,
    core_result: CycleResult,
    execution: _RoundExecution,
    projection: BeliefStateProjection,
    remaining_candidates: list[ProbeCandidate],
) -> SynchronizedRoundResult:
    return SynchronizedRoundResult(
        round_id=round_input.round_id,
        cycle=core_result.cycle,
        shape=round_input.shape,
        probe_set=execution.probe_set,
        signals=list(execution.signals),
        active_signal_count=execution.active_signal_count,
        passive_signal_count=execution.passive_signal_count,
        belief_state=core_result.belief_state,
        evidence_events=core_result.evidence_events,
        belief_updates=core_result.belief_updates,
        hypothesis_evolutions=core_result.hypothesis_evolutions,
        belief_state_projection=projection,
        selected_probe_candidates=list(execution.selected_probe_candidates),
        remaining_probe_candidates=list(remaining_candidates),
    )


def _next_candidate_pool(
    *,
    previous_pool: list[ProbeCandidate],
    selected_candidates: list[ProbeCandidate],
    belief_state_projection: BeliefStateProjection,
) -> list[ProbeCandidate]:
    selected_ids = {candidate.candidate_id for candidate in selected_candidates}
    remaining = [
        candidate
        for candidate in previous_pool
        if candidate.candidate_id not in selected_ids
    ]
    projection_candidates = list(
        belief_state_projection.change_my_mind_condition.structured_probe_candidates
    )
    return [*projection_candidates, *remaining]


def _require_nonempty(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")


__all__ = [
    "SynchronizedRoundInput",
    "SynchronizedRoundResult",
    "SynchronizedRoundRunner",
    "SynchronizedRoundShape",
    "SynchronizedRunInput",
    "SynchronizedRunResult",
]
