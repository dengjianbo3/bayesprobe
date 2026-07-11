from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
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
    HypothesisRelation,
    HypothesisEvolution,
    ProbeCandidate,
    ProbeSet,
    RunRecord,
    RunStatus,
    SignalKind,
    TaskFrame,
    utc_now,
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


class AutonomousQuestionProgressKind(StrEnum):
    RUN_STARTED = "run_started"
    TASK_FRAMING_STARTED = "task_framing_started"
    TASK_FRAMING_COMPLETED = "task_framing_completed"
    INITIALIZATION_COMPLETED = "initialization_completed"
    CYCLE_STARTED = "cycle_started"
    PROBE_SET_PLANNED = "probe_set_planned"
    PROBE_EXECUTION_STARTED = "probe_execution_started"
    SIGNALS_COLLECTED = "signals_collected"
    EVIDENCE_INTEGRATION_STARTED = "evidence_integration_started"
    CYCLE_INTEGRATED = "cycle_integrated"
    RUN_COMPLETED = "run_completed"


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
    task_frame: TaskFrame
    initial_belief_state: BeliefState
    final_belief_state: BeliefState
    cycle_results: list[AutonomousQuestionCycleResult]
    final_answer_projection: AnswerProjection | None
    stop_reason: AutonomousQuestionStopReason


@dataclass(frozen=True)
class AutonomousQuestionProgress:
    kind: AutonomousQuestionProgressKind
    run_id: str
    task_frame: TaskFrame | None = None
    cycle_id: str | None = None
    cycle_index: int | None = None
    run: RunRecord | None = None
    belief_state: BeliefState | None = None
    probe_set: ProbeSet | None = None
    signals: tuple[ExternalSignal, ...] = ()
    cycle_result: AutonomousQuestionCycleResult | None = None
    result: AutonomousQuestionRunResult | None = None


AutonomousQuestionProgressObserver = Callable[[AutonomousQuestionProgress], None]


class AutonomousQuestionRunner:
    def __init__(
        self,
        *,
        core: BayesProbeCore,
        initializer: BayesProbeInitializer | None = None,
        planner: ProbePlanner | None = None,
        executor: ProbeExecutor | None = None,
        config: AutonomousQuestionRunConfig | None = None,
        progress_observer: AutonomousQuestionProgressObserver | None = None,
    ) -> None:
        self.core = core
        self.initializer = initializer or BayesProbeInitializer(ledger=core.ledger)
        self.planner = planner or ProbePlanner(ledger=core.ledger)
        self.executor = executor or ProbeExecutor(
            gateway=DeterministicProbeToolGateway(),
            ledger=core.ledger,
        )
        self.config = config or AutonomousQuestionRunConfig()
        self.progress_observer = progress_observer

    def run_question(self, input: InitializeRunInput) -> AutonomousQuestionRunResult:
        self._emit_progress(
            AutonomousQuestionProgressKind.RUN_STARTED,
            run_id=input.run_id,
        )
        self._emit_progress(
            AutonomousQuestionProgressKind.TASK_FRAMING_STARTED,
            run_id=input.run_id,
        )
        initialization = self.initializer.initialize(input)
        run = initialization.run
        task_frame = initialization.task_frame
        initial_belief_state = initialization.belief_state
        self._emit_progress(
            AutonomousQuestionProgressKind.TASK_FRAMING_COMPLETED,
            run_id=run.run_id,
            task_frame=task_frame,
        )
        self._emit_progress(
            AutonomousQuestionProgressKind.INITIALIZATION_COMPLETED,
            run_id=run.run_id,
            run=run,
            belief_state=initial_belief_state,
        )
        current_belief_state = initial_belief_state
        candidate_pool = list(initialization.probe_candidates)
        cycle_results: list[AutonomousQuestionCycleResult] = []
        previous_answer: AnswerProjection | None = None

        for _ in range(self.config.max_cycles):
            previous_belief_state = current_belief_state
            cycle_id = self.core.allocate_cycle_id(
                f"{run.run_id}_cycle_{current_belief_state.cycle_index + 1}"
            )
            self._emit_progress(
                AutonomousQuestionProgressKind.CYCLE_STARTED,
                run_id=run.run_id,
                cycle_id=cycle_id,
                cycle_index=current_belief_state.cycle_index + 1,
                belief_state=current_belief_state,
            )
            planning = self._plan_next_probe_set(
                run=run,
                cycle_id=cycle_id,
                belief_state=current_belief_state,
                candidate_pool=candidate_pool,
            )
            self._emit_progress(
                AutonomousQuestionProgressKind.PROBE_SET_PLANNED,
                run_id=run.run_id,
                cycle_id=cycle_id,
                cycle_index=current_belief_state.cycle_index + 1,
                belief_state=current_belief_state,
                probe_set=planning.probe_set,
            )
            passive_signals = (
                _initial_context_signals(
                    input=input,
                    cycle_id=cycle_id,
                    belief_state=current_belief_state,
                )
                if not cycle_results
                else []
            )
            no_probes_selected = not planning.probe_set.probes
            if no_probes_selected and not passive_signals:
                return self._result(
                    run=run,
                    task_frame=task_frame,
                    initial_belief_state=initial_belief_state,
                    final_belief_state=current_belief_state,
                    cycle_results=cycle_results,
                    final_answer_projection=previous_answer,
                    stop_reason=AutonomousQuestionStopReason.NO_PROBES,
                )

            self._emit_progress(
                AutonomousQuestionProgressKind.PROBE_EXECUTION_STARTED,
                run_id=run.run_id,
                cycle_id=cycle_id,
                cycle_index=current_belief_state.cycle_index + 1,
                belief_state=current_belief_state,
                probe_set=planning.probe_set,
            )
            execution = self.executor.execute_probe_set(
                probe_set=planning.probe_set,
                context=ProbeExecutionContext(
                    run_id=run.run_id,
                    cycle_id=cycle_id,
                    belief_state=current_belief_state,
                    metadata={
                        "problem": run.problem,
                        "task_context": input.task_context.strip(),
                    },
                ),
            )
            signals = [*execution.signals, *passive_signals]
            self._emit_progress(
                AutonomousQuestionProgressKind.SIGNALS_COLLECTED,
                run_id=run.run_id,
                cycle_id=cycle_id,
                cycle_index=current_belief_state.cycle_index + 1,
                belief_state=current_belief_state,
                probe_set=planning.probe_set,
                signals=tuple(signals),
            )
            cycle = CycleRecord(
                cycle_id=cycle_id,
                run_id=run.run_id,
                cycle_index=current_belief_state.cycle_index + 1,
                signal_shape=_cycle_signal_shape(signals),
            )
            self._emit_progress(
                AutonomousQuestionProgressKind.EVIDENCE_INTEGRATION_STARTED,
                run_id=run.run_id,
                cycle_id=cycle_id,
                cycle_index=cycle.cycle_index,
                belief_state=current_belief_state,
                probe_set=planning.probe_set,
                signals=tuple(signals),
            )
            core_result = self.core.integrate_cycle(
                cycle=cycle,
                belief_state=current_belief_state,
                probe_set=planning.probe_set,
                signals=signals,
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
                signals=signals,
                belief_state=core_result.belief_state,
                evidence_events=core_result.evidence_events,
                belief_updates=core_result.belief_updates,
                hypothesis_evolutions=core_result.hypothesis_evolutions,
                answer_projection=answer_projection,
            )
            cycle_results.append(cycle_result)
            self._emit_progress(
                AutonomousQuestionProgressKind.CYCLE_INTEGRATED,
                run_id=run.run_id,
                cycle_id=cycle_id,
                cycle_index=cycle_result.cycle.cycle_index,
                belief_state=cycle_result.belief_state,
                probe_set=cycle_result.probe_set,
                signals=tuple(cycle_result.signals),
                cycle_result=cycle_result,
            )
            current_belief_state = core_result.belief_state
            previous_answer = answer_projection
            candidate_pool = self._next_candidate_pool(
                previous_pool=candidate_pool,
                selected_candidates=planning.selected_candidates,
                answer_projection=answer_projection,
            )

            if no_probes_selected:
                return self._result(
                    run=run,
                    task_frame=task_frame,
                    initial_belief_state=initial_belief_state,
                    final_belief_state=current_belief_state,
                    cycle_results=cycle_results,
                    final_answer_projection=previous_answer,
                    stop_reason=AutonomousQuestionStopReason.NO_PROBES,
                )

            if self._confidence_reached(current_belief_state):
                return self._result(
                    run=run,
                    task_frame=task_frame,
                    initial_belief_state=initial_belief_state,
                    final_belief_state=current_belief_state,
                    cycle_results=cycle_results,
                    final_answer_projection=previous_answer,
                    stop_reason=AutonomousQuestionStopReason.CONFIDENCE_REACHED,
                )

            if self._posterior_stable(previous=previous_belief_state, current=current_belief_state):
                return self._result(
                    run=run,
                    task_frame=task_frame,
                    initial_belief_state=initial_belief_state,
                    final_belief_state=current_belief_state,
                    cycle_results=cycle_results,
                    final_answer_projection=previous_answer,
                    stop_reason=AutonomousQuestionStopReason.POSTERIOR_STABLE,
                )

        return self._result(
            run=run,
            task_frame=task_frame,
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
        if (
            belief_state.task_frame.hypothesis_frame.relation
            == HypothesisRelation.INDEPENDENT
        ):
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
        task_frame: TaskFrame,
        initial_belief_state: BeliefState,
        final_belief_state: BeliefState,
        cycle_results: list[AutonomousQuestionCycleResult],
        final_answer_projection: AnswerProjection | None,
        stop_reason: AutonomousQuestionStopReason,
    ) -> AutonomousQuestionRunResult:
        completed_run = run.model_copy(
            update={
                "status": RunStatus.COMPLETED,
                "current_cycle_id": final_belief_state.cycle_id,
                "updated_at": utc_now(),
                "metadata": {
                    **run.metadata,
                    "stop_reason": stop_reason.value,
                },
            }
        )
        if self.core.ledger is not None:
            self.core.ledger.append("run", completed_run)
        result = AutonomousQuestionRunResult(
            run=completed_run,
            task_frame=task_frame,
            initial_belief_state=initial_belief_state,
            final_belief_state=final_belief_state,
            cycle_results=list(cycle_results),
            final_answer_projection=final_answer_projection,
            stop_reason=stop_reason,
        )
        self._emit_progress(
            AutonomousQuestionProgressKind.RUN_COMPLETED,
            run_id=completed_run.run_id,
            cycle_id=completed_run.current_cycle_id,
            cycle_index=final_belief_state.cycle_index,
            run=completed_run,
            belief_state=final_belief_state,
            result=result,
        )
        return result

    def _emit_progress(
        self,
        kind: AutonomousQuestionProgressKind,
        *,
        run_id: str,
        task_frame: TaskFrame | None = None,
        cycle_id: str | None = None,
        cycle_index: int | None = None,
        run: RunRecord | None = None,
        belief_state: BeliefState | None = None,
        probe_set: ProbeSet | None = None,
        signals: tuple[ExternalSignal, ...] = (),
        cycle_result: AutonomousQuestionCycleResult | None = None,
        result: AutonomousQuestionRunResult | None = None,
    ) -> None:
        if self.progress_observer is None:
            return
        try:
            self.progress_observer(
                deepcopy(
                    AutonomousQuestionProgress(
                        kind=kind,
                        run_id=run_id,
                        task_frame=task_frame,
                        cycle_id=cycle_id,
                        cycle_index=cycle_index,
                        run=run,
                        belief_state=belief_state,
                        probe_set=probe_set,
                        signals=signals,
                        cycle_result=cycle_result,
                        result=result,
                    )
                )
            )
        except Exception:
            return


def _top_hypothesis(belief_state: BeliefState) -> Hypothesis:
    return max(belief_state.hypotheses, key=lambda hypothesis: hypothesis.posterior)


def _initial_context_signals(
    *,
    input: InitializeRunInput,
    cycle_id: str,
    belief_state: BeliefState,
) -> list[ExternalSignal]:
    context = input.context.strip()
    if not context:
        return []
    return [
        ExternalSignal(
            id=f"S_{cycle_id}_initial_context",
            cycle_id=cycle_id,
            signal_kind=SignalKind.PASSIVE,
            source_type="initial_context",
            source="user_context",
            raw_content=context,
            initial_target_hypotheses=[
                hypothesis.id for hypothesis in belief_state.hypotheses
            ],
        )
    ]


def _cycle_signal_shape(signals: list[ExternalSignal]) -> CycleSignalShape:
    has_active = any(signal.signal_kind == SignalKind.ACTIVE for signal in signals)
    has_passive = any(signal.signal_kind == SignalKind.PASSIVE for signal in signals)
    if has_active and has_passive:
        return CycleSignalShape.ACTIVE_PLUS_PASSIVE
    if has_passive:
        return CycleSignalShape.PASSIVE_ONLY
    return CycleSignalShape.ACTIVE_ONLY


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
