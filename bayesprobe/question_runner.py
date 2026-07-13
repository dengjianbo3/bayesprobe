from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from bayesprobe.core import BayesProbeCore, CycleResult
from bayesprobe.frame_policy import FrameAdequacyDecision
from bayesprobe.initialization import (
    BayesProbeInitializer,
    InitializeRunInput,
    validate_initialize_run_input_security,
)
from bayesprobe.probe_design import (
    FrameProbeDesigner,
    ProbeDesignContext,
    ProbeDesignResult,
    ProbeDesigner,
)
from bayesprobe.task_admission import (
    ExplicitTaskAdmitter,
    TaskAdmitter,
    TaskAdmissionInput,
    validate_task_admission_decision,
)
from bayesprobe.task_framing import parse_legacy_answer_choice_frame
from bayesprobe.probe_executor import (
    DeterministicProbeToolGateway,
    ProbeExecutionResult,
    ProbeExecutor,
    build_probe_execution_brief,
)
from bayesprobe.probe_planner import ProbePlanner, ProbePlanningConfig, ProbePlanningResult
from bayesprobe.projections import (
    AnswerProjectionInput,
    AnswerProjector,
    build_answer_projection,
)
from bayesprobe.schemas import (
    AnswerProjection,
    BeliefState,
    BeliefUpdate,
    CapabilityDecision,
    CapabilityDescriptor,
    CycleRecord,
    CycleSignalShape,
    EpistemicProgress,
    EvidenceContributionDelta,
    EvidenceEvent,
    ExternalSignal,
    Hypothesis,
    HypothesisEvolution,
    ProbeCandidate,
    ProbeSet,
    RunRecord,
    RunStatus,
    SignalKind,
    TaskFrame,
    TaskAdmissionDecision,
    TaskAdmissionStatus,
    TaskKind,
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
    EPISTEMIC_STAGNATION = "epistemic_stagnation"


class AutonomousQuestionProgressKind(StrEnum):
    RUN_STARTED = "run_started"
    TASK_FRAMING_STARTED = "task_framing_started"
    TASK_FRAMING_COMPLETED = "task_framing_completed"
    INITIALIZATION_COMPLETED = "initialization_completed"
    PROBE_DESIGN_STARTED = "probe_design_started"
    PROBE_DESIGN_COMPLETED = "probe_design_completed"
    FRAME_ADEQUACY_ASSESSED = "frame_adequacy_assessed"
    HYPOTHESIS_EXPANSION_COMPLETED = "hypothesis_expansion_completed"
    ANSWER_PROJECTION_STARTED = "answer_projection_started"
    ANSWER_PROJECTION_COMPLETED = "answer_projection_completed"
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
    contribution_deltas: list[EvidenceContributionDelta]
    epistemic_progress: EpistemicProgress
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
class NeedsReframingResult:
    admission: TaskAdmissionDecision
    result_type: Literal["needs_reframing"] = "needs_reframing"


@dataclass(frozen=True)
class OutOfScopeResult:
    admission: TaskAdmissionDecision
    result_type: Literal["out_of_scope"] = "out_of_scope"


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
    probe_candidates: tuple[ProbeCandidate, ...] = ()
    capability_decisions: tuple[CapabilityDecision, ...] = ()
    signals: tuple[ExternalSignal, ...] = ()
    frame_adequacy_decision: FrameAdequacyDecision | None = None
    hypothesis_evolutions: tuple[HypothesisEvolution, ...] = ()
    answer_projection: AnswerProjection | None = None
    cycle_result: AutonomousQuestionCycleResult | None = None
    result: AutonomousQuestionRunResult | None = None


AutonomousQuestionProgressObserver = Callable[[AutonomousQuestionProgress], None]


class _AdmissionCapabilityDescriptor(CapabilityDescriptor):
    def model_dump(self, *args, **kwargs):
        kwargs["mode"] = "json"
        return super().model_dump(*args, **kwargs)


class _LegacyAnswerProjector:
    def project(self, input: AnswerProjectionInput) -> AnswerProjection:
        return build_answer_projection(
            input.cycle_id,
            input.previous_belief_state,
            input.cycle_result,
        )


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
        task_admitter: TaskAdmitter | None = None,
        probe_designer: ProbeDesigner | None = None,
        available_capabilities: tuple[CapabilityDescriptor, ...] = (),
        answer_projector: AnswerProjector | None = None,
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
        self.task_admitter = task_admitter or ExplicitTaskAdmitter()
        self.probe_designer = probe_designer or FrameProbeDesigner()
        self.available_capabilities = tuple(available_capabilities)
        self.answer_projector = answer_projector or _LegacyAnswerProjector()

    def run_question(
        self,
        input: InitializeRunInput,
    ) -> AutonomousQuestionRunResult | NeedsReframingResult | OutOfScopeResult:
        validate_initialize_run_input_security(input)
        admission = validate_task_admission_decision(
            self.task_admitter.assess(
                _task_admission_input(
                    input,
                    available_capabilities=self.available_capabilities,
                )
            )
        )
        if admission.status != TaskAdmissionStatus.ADMITTED:
            if self.core.ledger is not None:
                self.core.ledger.append("task_admission", admission)
            if admission.status == TaskAdmissionStatus.NEEDS_REFRAMING:
                return NeedsReframingResult(admission=admission)
            return OutOfScopeResult(admission=admission)
        self._emit_progress(
            AutonomousQuestionProgressKind.RUN_STARTED,
            run_id=input.run_id,
        )
        self._emit_progress(
            AutonomousQuestionProgressKind.TASK_FRAMING_STARTED,
            run_id=input.run_id,
        )
        initialization = self.initializer.initialize(
            input,
            admission_decision=admission,
        )
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
            probe_candidates=tuple(initialization.probe_candidates),
        )
        current_belief_state = initial_belief_state
        if initialization.probe_candidates:
            candidate_pool = list(initialization.probe_candidates)
        else:
            initial_design = self._design_probes(
                run=run,
                cycle_id=initial_belief_state.cycle_id,
                belief_state=initial_belief_state,
            )
            candidate_pool = list(initial_design.candidates)
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
                context=build_probe_execution_brief(
                    run_id=run.run_id,
                    cycle_id=cycle_id,
                    belief_state=current_belief_state,
                    problem=run.problem,
                    task_context=input.task_context,
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
            self._emit_progress(
                AutonomousQuestionProgressKind.FRAME_ADEQUACY_ASSESSED,
                run_id=run.run_id,
                cycle_id=cycle_id,
                cycle_index=core_result.cycle.cycle_index,
                belief_state=core_result.belief_state,
                frame_adequacy_decision=core_result.frame_adequacy_decision,
            )
            if any(
                evolution.operation.value == "spawn"
                for evolution in core_result.hypothesis_evolutions
            ):
                self._emit_progress(
                    AutonomousQuestionProgressKind.HYPOTHESIS_EXPANSION_COMPLETED,
                    run_id=run.run_id,
                    cycle_id=cycle_id,
                    cycle_index=core_result.cycle.cycle_index,
                    belief_state=core_result.belief_state,
                    probe_candidates=tuple(core_result.probe_candidates),
                    hypothesis_evolutions=tuple(core_result.hypothesis_evolutions),
                )
            prospective_stop_reason = self._prospective_stop_reason(
                previous=previous_belief_state,
                current=core_result.belief_state,
                cycle_result=core_result,
                completed_cycle_count=len(cycle_results) + 1,
            )
            self._emit_progress(
                AutonomousQuestionProgressKind.ANSWER_PROJECTION_STARTED,
                run_id=run.run_id,
                cycle_id=cycle_id,
                cycle_index=core_result.cycle.cycle_index,
                belief_state=core_result.belief_state,
            )
            answer_projection = self.answer_projector.project(
                AnswerProjectionInput(
                    cycle_id=cycle_id,
                    previous_belief_state=previous_belief_state,
                    cycle_result=core_result,
                    stop_reason=prospective_stop_reason,
                )
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
                contribution_deltas=core_result.contribution_deltas,
                epistemic_progress=core_result.epistemic_progress,
                answer_projection=answer_projection,
            )
            cycle_results.append(cycle_result)
            self._emit_progress(
                AutonomousQuestionProgressKind.ANSWER_PROJECTION_COMPLETED,
                run_id=run.run_id,
                cycle_id=cycle_id,
                cycle_index=cycle_result.cycle.cycle_index,
                belief_state=cycle_result.belief_state,
                answer_projection=answer_projection,
            )
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
            if prospective_stop_reason is not None:
                return self._result(
                    run=run,
                    task_frame=task_frame,
                    initial_belief_state=initial_belief_state,
                    final_belief_state=current_belief_state,
                    cycle_results=cycle_results,
                    final_answer_projection=previous_answer,
                    stop_reason=AutonomousQuestionStopReason(prospective_stop_reason),
                )
            design_result = self._design_probes(
                run=run,
                cycle_id=cycle_id,
                belief_state=current_belief_state,
            )
            candidate_pool = self._next_candidate_pool(
                previous_pool=candidate_pool,
                selected_candidates=planning.selected_candidates,
                core_candidates=core_result.probe_candidates,
                designed_candidates=design_result.candidates,
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
        core_candidates: list[ProbeCandidate],
        designed_candidates: list[ProbeCandidate],
        answer_projection: AnswerProjection,
    ) -> list[ProbeCandidate]:
        selected_ids = {candidate.candidate_id for candidate in selected_candidates}
        remaining = [
            candidate
            for candidate in previous_pool
            if candidate.candidate_id not in selected_ids
        ]
        ordered = [
            *core_candidates,
            *designed_candidates,
            *answer_projection.change_my_mind_condition.structured_probe_candidates,
            *remaining,
        ]
        return _deduplicate_probe_candidates(ordered)

    def _design_probes(
        self,
        *,
        run: RunRecord,
        cycle_id: str,
        belief_state: BeliefState,
    ) -> ProbeDesignResult:
        self._emit_progress(
            AutonomousQuestionProgressKind.PROBE_DESIGN_STARTED,
            run_id=run.run_id,
            cycle_id=cycle_id,
            cycle_index=belief_state.cycle_index,
            belief_state=belief_state,
        )
        result = self.probe_designer.propose(
            ProbeDesignContext(
                run_id=run.run_id,
                cycle_id=cycle_id,
                task_frame=belief_state.task_frame,
                belief_state=belief_state,
                available_capabilities=self.available_capabilities,
            )
        )
        self._emit_progress(
            AutonomousQuestionProgressKind.PROBE_DESIGN_COMPLETED,
            run_id=run.run_id,
            cycle_id=cycle_id,
            cycle_index=belief_state.cycle_index,
            belief_state=belief_state,
            probe_candidates=tuple(result.candidates),
            capability_decisions=tuple(result.capability_decisions),
        )
        return result

    def _confidence_reached(self, belief_state: BeliefState) -> bool:
        threshold = self.config.confidence_threshold
        if threshold is None:
            return False
        if (
            belief_state.task_frame.hypothesis_frame.competition.value
            == "independent"
        ):
            return False
        return _top_hypothesis(belief_state).posterior >= threshold

    def _posterior_stable(self, *, previous: BeliefState, current: BeliefState) -> bool:
        threshold = self.config.posterior_delta_threshold
        if threshold is None:
            return False
        return _posterior_delta_is_stable(previous=previous, current=current, threshold=threshold)

    def _prospective_stop_reason(
        self,
        *,
        previous: BeliefState,
        current: BeliefState,
        cycle_result: CycleResult,
        completed_cycle_count: int,
    ) -> str | None:
        if _is_epistemically_stagnant(
            previous=previous,
            current=current,
            cycle_result=cycle_result,
        ):
            return AutonomousQuestionStopReason.EPISTEMIC_STAGNATION.value
        if completed_cycle_count >= self.config.max_cycles:
            return AutonomousQuestionStopReason.MAX_CYCLES.value
        if self._confidence_reached(current):
            return AutonomousQuestionStopReason.CONFIDENCE_REACHED.value
        if self._posterior_stable(previous=previous, current=current):
            return AutonomousQuestionStopReason.POSTERIOR_STABLE.value
        return None

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
        probe_candidates: tuple[ProbeCandidate, ...] = (),
        capability_decisions: tuple[CapabilityDecision, ...] = (),
        signals: tuple[ExternalSignal, ...] = (),
        frame_adequacy_decision: FrameAdequacyDecision | None = None,
        hypothesis_evolutions: tuple[HypothesisEvolution, ...] = (),
        answer_projection: AnswerProjection | None = None,
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
                        probe_candidates=probe_candidates,
                        capability_decisions=capability_decisions,
                        signals=signals,
                        frame_adequacy_decision=frame_adequacy_decision,
                        hypothesis_evolutions=hypothesis_evolutions,
                        answer_projection=answer_projection,
                        cycle_result=cycle_result,
                        result=result,
                    )
                )
            )
        except Exception:
            return


def _top_hypothesis(belief_state: BeliefState) -> Hypothesis:
    return max(belief_state.hypotheses, key=lambda hypothesis: hypothesis.posterior)


def _task_admission_input(
    input: InitializeRunInput,
    *,
    available_capabilities: tuple[CapabilityDescriptor, ...],
) -> TaskAdmissionInput:
    choices = list(input.answer_choices)
    if not choices:
        parsed = parse_legacy_answer_choice_frame(input.problem)
        if parsed is not None:
            choices = list(parsed.choices)
    output_shape = input.metadata.get("requested_output_shape")
    return TaskAdmissionInput(
        attempt_id=f"{input.run_id}_admission",
        question=input.problem,
        task_context=input.task_context,
        answer_choices=choices,
        hypothesis_seeds=list(input.hypothesis_seeds),
        available_capabilities=[
            _AdmissionCapabilityDescriptor.model_validate(
                descriptor.model_dump(mode="json")
            )
            for descriptor in available_capabilities
        ],
        requested_output_shape=(
            output_shape
            if isinstance(output_shape, str) and output_shape.strip()
            else None
        ),
        model_metadata={
            "task_kind": input.task_kind.value if input.task_kind is not None else None
        },
    )


def _deduplicate_probe_candidates(
    candidates: list[ProbeCandidate],
) -> list[ProbeCandidate]:
    unique: list[ProbeCandidate] = []
    identities: set[tuple[object, ...]] = set()
    for candidate in candidates:
        probe = candidate.candidate_probe
        identity = (
            probe.purpose,
            tuple(sorted(probe.target_hypotheses)),
            probe.required_capability,
            " ".join(probe.inquiry_goal.casefold().split()),
        )
        if identity in identities:
            continue
        identities.add(identity)
        unique.append(candidate)
    return unique


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


def _is_epistemically_stagnant(
    *,
    previous: BeliefState,
    current: BeliefState,
    cycle_result: CycleResult,
) -> bool:
    progress = cycle_result.epistemic_progress
    has_root_change = (
        progress.new_root_count
        + progress.revised_root_count
        + progress.retracted_root_count
    ) > 0
    return (
        not has_root_change
        and progress.max_absolute_contribution_delta == 0.0
        and not cycle_result.hypothesis_evolutions
        and previous.frame_state == current.frame_state
    )


__all__ = [
    "AutonomousQuestionCycleResult",
    "AutonomousQuestionRunConfig",
    "AutonomousQuestionRunResult",
    "AutonomousQuestionRunner",
    "AutonomousQuestionStopReason",
]
