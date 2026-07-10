from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum

from bayesprobe.controllers import ControllerResult, SynchronizedController
from bayesprobe.core import BayesProbeCore, CycleResult
from bayesprobe.initialization import BayesProbeInitializer, InitializeRunInput
from bayesprobe.ledger import JsonlLedgerStore
from bayesprobe.model_gateway import EvidenceJudgmentRepairPolicy, ModelGateway
from bayesprobe.probe_executor import (
    DeterministicProbeToolGateway,
    ProbeExecutionContext,
    ProbeExecutor,
)
from bayesprobe.probe_planner import ProbePlanner, ProbePlanningConfig
from bayesprobe.projections import build_answer_projection
from bayesprobe.question_runner import (
    AutonomousQuestionRunConfig,
    AutonomousQuestionRunResult,
    AutonomousQuestionRunner,
)
from bayesprobe.schemas import (
    BeliefState,
    BeliefUpdate,
    CycleRecord,
    CycleSignalShape,
    EvidenceEvent,
    ExternalSignal,
    SignalKind,
    UpdateDirection,
)


class BenchmarkSignalShape(StrEnum):
    ACTIVE_ONLY = "active_only"
    PASSIVE_ONLY = "passive_only"
    ACTIVE_PLUS_PASSIVE = "active_plus_passive"


@dataclass(frozen=True)
class BenchmarkSignal:
    signal_id: str
    source_type: str
    source: str
    raw_content: str
    target_hypotheses: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        _require_nonempty(self.signal_id, "signal_id")
        _require_nonempty(self.source_type, "source_type")
        _require_nonempty(self.source, "source")
        _require_nonempty(self.raw_content, "raw_content")

    def to_external_signal(self, *, cycle_id: str, signal_kind: SignalKind) -> ExternalSignal:
        return ExternalSignal(
            id=self.signal_id,
            cycle_id=cycle_id,
            signal_kind=signal_kind,
            source_type=self.source_type,
            source=self.source,
            raw_content=self.raw_content,
            initial_target_hypotheses=list(self.target_hypotheses),
        )


@dataclass(frozen=True)
class BenchmarkSample:
    sample_id: str
    question_or_claim: str
    gold_best_hypothesis: str
    signal_shape: BenchmarkSignalShape | str = BenchmarkSignalShape.ACTIVE_ONLY
    passive_signals: list[BenchmarkSignal] = field(default_factory=list)
    gold_update_directions: dict[str, str] = field(default_factory=dict)
    initial_context: str = ""

    def __post_init__(self) -> None:
        _require_nonempty(self.sample_id, "sample_id")
        _require_nonempty(self.question_or_claim, "question_or_claim")
        _require_nonempty(self.gold_best_hypothesis, "gold_best_hypothesis")
        try:
            signal_shape = BenchmarkSignalShape(self.signal_shape)
        except ValueError as error:
            raise ValueError("signal_shape must be a valid BenchmarkSignalShape") from error
        object.__setattr__(self, "signal_shape", signal_shape)
        if signal_shape == BenchmarkSignalShape.ACTIVE_ONLY and self.passive_signals:
            raise ValueError("active-only samples cannot include passive_signals")
        if signal_shape in {
            BenchmarkSignalShape.PASSIVE_ONLY,
            BenchmarkSignalShape.ACTIVE_PLUS_PASSIVE,
        } and not self.passive_signals:
            raise ValueError("passive or mixed samples require passive_signals")


@dataclass(frozen=True)
class BenchmarkSampleResult:
    sample_id: str
    run_id: str
    signal_shape: BenchmarkSignalShape
    final_best_hypothesis: str
    gold_best_hypothesis: str
    final_correct: bool
    update_direction_accuracy: float | None
    cycle_count: int
    signal_count: int
    active_signal_count: int
    passive_signal_count: int
    evidence_event_count: int
    belief_update_count: int
    discarded_evidence_count: int
    schema_violation_count: int
    dominant_hypothesis_margin: float
    belief_revision_efficiency: float
    projection_kind: str


@dataclass(frozen=True)
class BenchmarkSuiteResult:
    sample_count: int
    results: list[BenchmarkSampleResult]
    final_accuracy: float
    update_direction_accuracy: float | None


class BenchmarkHarness:
    def __init__(
        self,
        *,
        core: BayesProbeCore | None = None,
        ledger: JsonlLedgerStore | None = None,
        model_gateway: ModelGateway | None = None,
        judgment_repair_policy: EvidenceJudgmentRepairPolicy | None = None,
        max_cycles: int = 1,
        max_probes_per_cycle: int = 1,
    ) -> None:
        self.core = core or BayesProbeCore(
            ledger=ledger,
            model_gateway=model_gateway,
            judgment_repair_policy=judgment_repair_policy,
        )
        self.ledger = self.core.ledger
        self.max_cycles = max_cycles
        self.max_probes_per_cycle = max_probes_per_cycle

    def run_sample(self, sample: BenchmarkSample) -> BenchmarkSampleResult:
        if sample.signal_shape == BenchmarkSignalShape.ACTIVE_ONLY:
            result = self._run_active_only(sample)
        elif sample.signal_shape == BenchmarkSignalShape.PASSIVE_ONLY:
            result = self._run_passive_only(sample)
        elif sample.signal_shape == BenchmarkSignalShape.ACTIVE_PLUS_PASSIVE:
            result = self._run_active_plus_passive(sample)
        else:
            raise ValueError("unsupported benchmark signal shape")
        self._append_sample_result(result)
        return result

    def run_suite(self, samples: list[BenchmarkSample]) -> BenchmarkSuiteResult:
        results = [self.run_sample(sample) for sample in samples]
        final_accuracy = _mean(1.0 if result.final_correct else 0.0 for result in results)
        update_direction_scores = [
            result.update_direction_accuracy
            for result in results
            if result.update_direction_accuracy is not None
        ]
        update_direction_accuracy = (
            _mean(update_direction_scores) if update_direction_scores else None
        )
        return BenchmarkSuiteResult(
            sample_count=len(results),
            results=results,
            final_accuracy=final_accuracy,
            update_direction_accuracy=update_direction_accuracy,
        )

    def _run_active_only(self, sample: BenchmarkSample) -> BenchmarkSampleResult:
        runner = AutonomousQuestionRunner(
            core=self.core,
            config=AutonomousQuestionRunConfig(
                max_cycles=self.max_cycles,
                max_probes_per_cycle=self.max_probes_per_cycle,
            ),
        )
        run_result = runner.run_question(_initialize_input(sample))
        belief_updates = [
            update
            for cycle_result in run_result.cycle_results
            for update in cycle_result.belief_updates
        ]
        return _sample_result_from_question_run(
            sample=sample,
            run_result=run_result,
            belief_updates=belief_updates,
            projection_kind="answer_projection",
        )

    def _run_passive_only(self, sample: BenchmarkSample) -> BenchmarkSampleResult:
        initialization = BayesProbeInitializer(ledger=self.ledger).initialize(
            _initialize_input(sample)
        )
        passive_signals = [
            signal.to_external_signal(cycle_id="pending", signal_kind=SignalKind.PASSIVE)
            for signal in sample.passive_signals
        ]
        controller_result = SynchronizedController(core=self.core).process_round(
            run_id=initialization.run.run_id,
            round_id=f"{initialization.run.run_id}_round_1",
            belief_state=initialization.belief_state,
            passive_signals=passive_signals,
        )
        return _sample_result_from_controller_result(
            sample=sample,
            run_id=initialization.run.run_id,
            controller_result=controller_result,
            active_signal_count=0,
            passive_signal_count=len(passive_signals),
            projection_kind="belief_state_projection",
        )

    def _run_active_plus_passive(self, sample: BenchmarkSample) -> BenchmarkSampleResult:
        initialization = BayesProbeInitializer(ledger=self.ledger).initialize(
            _initialize_input(sample)
        )
        cycle_id = self.core.allocate_cycle_id(
            f"{initialization.run.run_id}_cycle_{initialization.belief_state.cycle_index + 1}"
        )
        planner = ProbePlanner(ledger=self.ledger)
        planning = planner.design_probe_set(
            run_id=initialization.run.run_id,
            cycle_id=cycle_id,
            belief_state=initialization.belief_state,
            candidates=list(initialization.probe_candidates),
            config=ProbePlanningConfig(max_probes=self.max_probes_per_cycle),
        )
        execution = ProbeExecutor(
            gateway=DeterministicProbeToolGateway(),
            ledger=self.ledger,
        ).execute_probe_set(
            probe_set=planning.probe_set,
            context=ProbeExecutionContext(
                run_id=initialization.run.run_id,
                cycle_id=cycle_id,
                belief_state=initialization.belief_state,
            ),
        )
        passive_signals = [
            signal.to_external_signal(cycle_id="pending", signal_kind=SignalKind.PASSIVE)
            for signal in sample.passive_signals
        ]
        cycle = CycleRecord(
            cycle_id=cycle_id,
            run_id=initialization.run.run_id,
            cycle_index=initialization.belief_state.cycle_index + 1,
            signal_shape=CycleSignalShape.ACTIVE_PLUS_PASSIVE,
        )
        core_result = self.core.integrate_cycle(
            cycle=cycle,
            belief_state=initialization.belief_state,
            probe_set=planning.probe_set,
            signals=[*execution.signals, *passive_signals],
        )
        answer_projection = build_answer_projection(
            cycle_id,
            initialization.belief_state,
            core_result,
        )
        if self.ledger is not None:
            self.ledger.append("answer_projection", answer_projection)
        return _sample_result_from_cycle_result(
            sample=sample,
            run_id=initialization.run.run_id,
            cycle_result=core_result,
            active_signal_count=len(execution.signals),
            passive_signal_count=len(passive_signals),
            projection_kind="answer_projection",
        )

    def _append_sample_result(self, result: BenchmarkSampleResult) -> None:
        if self.ledger is None:
            return
        payload = asdict(result)
        payload["signal_shape"] = result.signal_shape.value
        self.ledger.append("benchmark_sample_result", payload)


def _initialize_input(sample: BenchmarkSample) -> InitializeRunInput:
    return InitializeRunInput(
        run_id=f"bench_{sample.sample_id}",
        problem=sample.question_or_claim,
        context=sample.initial_context,
    )


def _sample_result_from_question_run(
    *,
    sample: BenchmarkSample,
    run_result: AutonomousQuestionRunResult,
    belief_updates: list[BeliefUpdate],
    projection_kind: str,
) -> BenchmarkSampleResult:
    active_signal_count = sum(len(cycle_result.signals) for cycle_result in run_result.cycle_results)
    evidence_events = [
        event
        for cycle_result in run_result.cycle_results
        for event in cycle_result.evidence_events
    ]
    return _build_sample_result(
        sample=sample,
        run_id=run_result.run.run_id,
        final_belief_state=run_result.final_belief_state,
        belief_updates=belief_updates,
        evidence_events=evidence_events,
        cycle_count=len(run_result.cycle_results),
        active_signal_count=active_signal_count,
        passive_signal_count=0,
        projection_kind=projection_kind,
    )


def _sample_result_from_controller_result(
    *,
    sample: BenchmarkSample,
    run_id: str,
    controller_result: ControllerResult,
    active_signal_count: int,
    passive_signal_count: int,
    projection_kind: str,
) -> BenchmarkSampleResult:
    return _build_sample_result(
        sample=sample,
        run_id=run_id,
        final_belief_state=controller_result.belief_state,
        belief_updates=controller_result.belief_updates,
        evidence_events=controller_result.evidence_events,
        cycle_count=1,
        active_signal_count=active_signal_count,
        passive_signal_count=passive_signal_count,
        projection_kind=projection_kind,
    )


def _sample_result_from_cycle_result(
    *,
    sample: BenchmarkSample,
    run_id: str,
    cycle_result: CycleResult,
    active_signal_count: int,
    passive_signal_count: int,
    projection_kind: str,
) -> BenchmarkSampleResult:
    return _build_sample_result(
        sample=sample,
        run_id=run_id,
        final_belief_state=cycle_result.belief_state,
        belief_updates=cycle_result.belief_updates,
        evidence_events=cycle_result.evidence_events,
        cycle_count=1,
        active_signal_count=active_signal_count,
        passive_signal_count=passive_signal_count,
        projection_kind=projection_kind,
    )


def _build_sample_result(
    *,
    sample: BenchmarkSample,
    run_id: str,
    final_belief_state: BeliefState,
    belief_updates: list[BeliefUpdate],
    evidence_events: list[EvidenceEvent],
    cycle_count: int,
    active_signal_count: int,
    passive_signal_count: int,
    projection_kind: str,
) -> BenchmarkSampleResult:
    final_best_hypothesis = _top_hypothesis_id(final_belief_state)
    return BenchmarkSampleResult(
        sample_id=sample.sample_id,
        run_id=run_id,
        signal_shape=sample.signal_shape,
        final_best_hypothesis=final_best_hypothesis,
        gold_best_hypothesis=sample.gold_best_hypothesis,
        final_correct=final_best_hypothesis == sample.gold_best_hypothesis,
        update_direction_accuracy=_update_direction_accuracy(
            belief_updates=belief_updates,
            gold_update_directions=sample.gold_update_directions,
        ),
        cycle_count=cycle_count,
        signal_count=active_signal_count + passive_signal_count,
        active_signal_count=active_signal_count,
        passive_signal_count=passive_signal_count,
        evidence_event_count=len(evidence_events),
        belief_update_count=len(belief_updates),
        discarded_evidence_count=_discarded_evidence_count(evidence_events),
        schema_violation_count=_schema_violation_count(evidence_events),
        dominant_hypothesis_margin=_dominant_hypothesis_margin(final_belief_state),
        belief_revision_efficiency=_belief_revision_efficiency(
            belief_updates=belief_updates,
            evidence_events=evidence_events,
        ),
        projection_kind=projection_kind,
    )


def _update_direction_accuracy(
    *,
    belief_updates: list[BeliefUpdate],
    gold_update_directions: dict[str, str],
) -> float | None:
    if not gold_update_directions:
        return None
    observed_by_hypothesis: dict[str, list[BeliefUpdate]] = {}
    for update in belief_updates:
        observed_by_hypothesis.setdefault(update.hypothesis_id, []).append(update)

    correct = 0
    for hypothesis_id, expected_direction in gold_update_directions.items():
        hypothesis_updates = observed_by_hypothesis.get(hypothesis_id, [])
        if not hypothesis_updates:
            continue
        initial = hypothesis_updates[0].prior
        final = hypothesis_updates[-1].posterior
        observed_direction = _net_update_direction(initial=initial, final=final)
        if expected_direction == observed_direction.value:
            correct += 1
    return round(correct / len(gold_update_directions), 6)


def _net_update_direction(*, initial: float, final: float) -> UpdateDirection:
    if final > initial + 0.01:
        return UpdateDirection.STRENGTHENED
    if final < initial - 0.01:
        return UpdateDirection.WEAKENED
    return UpdateDirection.NEUTRAL


def _top_hypothesis_id(belief_state: BeliefState) -> str:
    return max(belief_state.hypotheses, key=lambda hypothesis: hypothesis.posterior).id


def _discarded_evidence_count(evidence_events: list[EvidenceEvent]) -> int:
    return sum(1 for event in evidence_events if event.discard_reason is not None)


def _schema_violation_count(evidence_events: list[EvidenceEvent]) -> int:
    return sum(
        1
        for event in evidence_events
        if isinstance(event.discard_reason, str)
        and event.discard_reason.startswith("schema_violation:")
    )


def _dominant_hypothesis_margin(belief_state: BeliefState) -> float:
    posteriors = sorted(
        (hypothesis.posterior for hypothesis in belief_state.hypotheses),
        reverse=True,
    )
    if not posteriors:
        return 0.0
    if len(posteriors) == 1:
        return round(posteriors[0], 6)
    return round(posteriors[0] - posteriors[1], 6)


def _belief_revision_efficiency(
    *,
    belief_updates: list[BeliefUpdate],
    evidence_events: list[EvidenceEvent],
) -> float:
    accepted_evidence_count = sum(
        1 for event in evidence_events if event.discard_reason is None
    )
    if accepted_evidence_count == 0:
        return 0.0
    updates_by_hypothesis: dict[str, list[BeliefUpdate]] = {}
    for update in belief_updates:
        updates_by_hypothesis.setdefault(update.hypothesis_id, []).append(update)
    total_variation = 0.5 * sum(
        abs(updates[-1].posterior - updates[0].prior)
        for updates in updates_by_hypothesis.values()
        if updates
    )
    return round(total_variation / accepted_evidence_count, 6)


def _mean(values) -> float:
    items = list(values)
    if not items:
        return 0.0
    return round(sum(items) / len(items), 6)


def _require_nonempty(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")


__all__ = [
    "BenchmarkHarness",
    "BenchmarkSample",
    "BenchmarkSampleResult",
    "BenchmarkSignal",
    "BenchmarkSignalShape",
    "BenchmarkSuiteResult",
]
