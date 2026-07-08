from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from bayesprobe.core import BayesProbeCore
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
from bayesprobe.projections import build_answer_projection
from bayesprobe.schemas import (
    AnswerProjection,
    BeliefState,
    BeliefUpdate,
    CycleRecord,
    CycleSignalShape,
    EvidenceEvent,
    ExternalSignal,
    Hypothesis,
    HypothesisEvolution,
    ProbeCandidate,
    ProbeSet,
    RunRecord,
)


@dataclass(frozen=True)
class AutonomousQuestionRunConfig:
    max_cycles: int = 3
    stop_on_no_probes: bool = True
    confidence_threshold: float | None = None
    posterior_delta_threshold: float | None = None
    max_probes_per_cycle: int = 2

    def __post_init__(self) -> None:
        if self.max_cycles < 1:
            raise ValueError("max_cycles must be at least 1")
        if self.max_probes_per_cycle < 1:
            raise ValueError("max_probes_per_cycle must be at least 1")
        if self.confidence_threshold is not None and not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if self.posterior_delta_threshold is not None and self.posterior_delta_threshold < 0:
            raise ValueError("posterior_delta_threshold must be non-negative")


class AutonomousQuestionStopReason(StrEnum):
    MAX_CYCLES = "max_cycles"
    NO_PROBES = "no_probes"
    CONFIDENCE_REACHED = "confidence_reached"
    POSTERIOR_STABLE = "posterior_stable"


@dataclass(frozen=True)
class AutonomousQuestionCycleResult:
    cycle: CycleRecord
    planning_result: ProbePlanningResult
    execution_result: ProbeExecutionResult
    probe_set: ProbeSet
    signals: list[ExternalSignal]
    belief_state: BeliefState
    evidence_events: list[EvidenceEvent]
    belief_updates: list[BeliefUpdate]
    hypothesis_evolutions: list[HypothesisEvolution]
    answer_projection: AnswerProjection


@dataclass(frozen=True)
class AutonomousQuestionRunResult:
    run: RunRecord
    initial_belief_state: BeliefState
    final_belief_state: BeliefState
    cycle_results: list[AutonomousQuestionCycleResult]
    final_answer_projection: AnswerProjection | None
    stop_reason: AutonomousQuestionStopReason


class AutonomousQuestionRunner:
    def __init__(
        self,
        *,
        core: BayesProbeCore,
        initializer: BayesProbeInitializer | None = None,
        planner: ProbePlanner | None = None,
        executor: ProbeExecutor | None = None,
        config: AutonomousQuestionRunConfig | None = None,
    ) -> None:
        self.core = core
        self.initializer = initializer or BayesProbeInitializer(ledger=core.ledger)
        self.planner = planner or ProbePlanner(ledger=core.ledger)
        self.executor = executor or ProbeExecutor(
            gateway=DeterministicProbeToolGateway(),
            ledger=core.ledger,
        )
        self.config = config or AutonomousQuestionRunConfig()

    def run_question(self, input: InitializeRunInput) -> AutonomousQuestionRunResult:
        initialization = self.initializer.initialize(input)
        run = initialization.run
        initial_belief_state = initialization.belief_state
        current_belief_state = initial_belief_state
        candidate_pool = list(initialization.probe_candidates)
        cycle_results: list[AutonomousQuestionCycleResult] = []
        previous_answer: AnswerProjection | None = None

        for _ in range(self.config.max_cycles):
            previous_belief_state = current_belief_state
            cycle_id = self.core.allocate_cycle_id(
                f"{run.run_id}_cycle_{current_belief_state.cycle_index + 1}"
            )
            planning = self._plan_next_probe_set(
                run=run,
                cycle_id=cycle_id,
                belief_state=current_belief_state,
                candidate_pool=candidate_pool,
            )
            if not planning.probe_set.probes:
                return self._result(
                    run=run,
                    initial_belief_state=initial_belief_state,
                    final_belief_state=current_belief_state,
                    cycle_results=cycle_results,
                    final_answer_projection=previous_answer,
                    stop_reason=AutonomousQuestionStopReason.NO_PROBES,
                )

            execution = self.executor.execute_probe_set(
                probe_set=planning.probe_set,
                context=ProbeExecutionContext(
                    run_id=run.run_id,
                    cycle_id=cycle_id,
                    belief_state=current_belief_state,
                ),
            )
            cycle = CycleRecord(
                cycle_id=cycle_id,
                run_id=run.run_id,
                cycle_index=current_belief_state.cycle_index + 1,
                signal_shape=CycleSignalShape.ACTIVE_ONLY,
            )
            core_result = self.core.integrate_cycle(
                cycle=cycle,
                belief_state=current_belief_state,
                probe_set=planning.probe_set,
                signals=execution.signals,
            )
            answer_projection = build_answer_projection(
                cycle_id,
                previous_belief_state,
                core_result,
            )
            if self.core.ledger is not None:
                self.core.ledger.append("answer_projection", answer_projection)

            cycle_result = AutonomousQuestionCycleResult(
                cycle=core_result.cycle,
                planning_result=planning,
                execution_result=execution,
                probe_set=planning.probe_set,
                signals=execution.signals,
                belief_state=core_result.belief_state,
                evidence_events=core_result.evidence_events,
                belief_updates=core_result.belief_updates,
                hypothesis_evolutions=core_result.hypothesis_evolutions,
                answer_projection=answer_projection,
            )
            cycle_results.append(cycle_result)
            current_belief_state = core_result.belief_state
            previous_answer = answer_projection
            candidate_pool = self._next_candidate_pool(
                previous_pool=candidate_pool,
                selected_candidates=planning.selected_candidates,
                answer_projection=answer_projection,
            )

            if self._confidence_reached(current_belief_state):
                return self._result(
                    run=run,
                    initial_belief_state=initial_belief_state,
                    final_belief_state=current_belief_state,
                    cycle_results=cycle_results,
                    final_answer_projection=previous_answer,
                    stop_reason=AutonomousQuestionStopReason.CONFIDENCE_REACHED,
                )

            if self._posterior_stable(previous=previous_belief_state, current=current_belief_state):
                return self._result(
                    run=run,
                    initial_belief_state=initial_belief_state,
                    final_belief_state=current_belief_state,
                    cycle_results=cycle_results,
                    final_answer_projection=previous_answer,
                    stop_reason=AutonomousQuestionStopReason.POSTERIOR_STABLE,
                )

        return self._result(
            run=run,
            initial_belief_state=initial_belief_state,
            final_belief_state=current_belief_state,
            cycle_results=cycle_results,
            final_answer_projection=previous_answer,
            stop_reason=AutonomousQuestionStopReason.MAX_CYCLES,
        )

    def _plan_next_probe_set(
        self,
        *,
        run: RunRecord,
        cycle_id: str,
        belief_state: BeliefState,
        candidate_pool: list[ProbeCandidate],
    ) -> ProbePlanningResult:
        config = ProbePlanningConfig(
            max_probes=self.config.max_probes_per_cycle,
            allow_empty=self.config.stop_on_no_probes,
        )
        return self.planner.design_probe_set(
            run_id=run.run_id,
            cycle_id=cycle_id,
            belief_state=belief_state,
            candidates=candidate_pool,
            config=config,
        )

    def _next_candidate_pool(
        self,
        *,
        previous_pool: list[ProbeCandidate],
        selected_candidates: list[ProbeCandidate],
        answer_projection: AnswerProjection,
    ) -> list[ProbeCandidate]:
        selected_ids = {candidate.candidate_id for candidate in selected_candidates}
        remaining = [
            candidate
            for candidate in previous_pool
            if candidate.candidate_id not in selected_ids
        ]
        projection_candidates = list(
            answer_projection.change_my_mind_condition.structured_probe_candidates
        )
        return [*projection_candidates, *remaining]

    def _confidence_reached(self, belief_state: BeliefState) -> bool:
        threshold = self.config.confidence_threshold
        if threshold is None:
            return False
        return _top_hypothesis(belief_state).posterior >= threshold

    def _posterior_stable(self, *, previous: BeliefState, current: BeliefState) -> bool:
        threshold = self.config.posterior_delta_threshold
        if threshold is None:
            return False
        return _posterior_delta_is_stable(previous=previous, current=current, threshold=threshold)

    def _result(
        self,
        *,
        run: RunRecord,
        initial_belief_state: BeliefState,
        final_belief_state: BeliefState,
        cycle_results: list[AutonomousQuestionCycleResult],
        final_answer_projection: AnswerProjection | None,
        stop_reason: AutonomousQuestionStopReason,
    ) -> AutonomousQuestionRunResult:
        return AutonomousQuestionRunResult(
            run=run,
            initial_belief_state=initial_belief_state,
            final_belief_state=final_belief_state,
            cycle_results=list(cycle_results),
            final_answer_projection=final_answer_projection,
            stop_reason=stop_reason,
        )


def _top_hypothesis(belief_state: BeliefState) -> Hypothesis:
    return max(belief_state.hypotheses, key=lambda hypothesis: hypothesis.posterior)


def _posterior_delta_is_stable(
    *,
    previous: BeliefState,
    current: BeliefState,
    threshold: float,
) -> bool:
    previous_by_id = previous.hypotheses_by_id()
    current_by_id = current.hypotheses_by_id()
    continuing_ids = set(previous_by_id).intersection(current_by_id)
    if not continuing_ids:
        return False
    return all(
        abs(current_by_id[hypothesis_id].posterior - previous_by_id[hypothesis_id].posterior) <= threshold
        for hypothesis_id in continuing_ids
    )


__all__ = [
    "AutonomousQuestionCycleResult",
    "AutonomousQuestionRunConfig",
    "AutonomousQuestionRunResult",
    "AutonomousQuestionRunner",
    "AutonomousQuestionStopReason",
]
