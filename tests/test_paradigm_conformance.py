from __future__ import annotations

from dataclasses import dataclass

from bayesprobe.belief import CoverageAwareBeliefSolver
from bayesprobe.core import BayesProbeCore
from bayesprobe.initialization import HypothesisSeed, InitializeRunInput
from bayesprobe.probe_executor import ProbeExecutionBrief, ProbeExecutor
from bayesprobe.question_runner import (
    AutonomousQuestionRunConfig,
    AutonomousQuestionRunner,
    AutonomousQuestionStopReason,
)
from bayesprobe.schemas import (
    BoundaryStatus,
    EpistemicOrigin,
    EvidenceContributionDelta,
    EvidenceContributionMode,
    EvidenceEvent,
    ExternalSignal,
    HypothesisRelation,
    ProbePurpose,
    SignalKind,
    SignalProvenance,
    TaskKind,
)


@dataclass(frozen=True)
class SignalStep:
    content: str
    origin: EpistemicOrigin
    derivation_root_id: str


class ScriptedEvidenceGateway:
    adapter_kind = "paradigm_conformance_judge"
    model_identity = "paradigm-conformance-judge-v1"

    _LIKELIHOODS = {
        "MODEL_SUPPORT_H1": {
            "H1": "moderately_confirming",
            "H2": "moderately_disconfirming",
        },
        "TOOL_REFUTE_H1": {
            "H1": "strongly_disconfirming",
            "H2": "strongly_confirming",
        },
        "MODEL_WEAK_SUPPORT_H1": {
            "H1": "weakly_confirming",
            "H2": "weakly_disconfirming",
        },
        "MODEL_REVERSE_H1": {
            "H1": "strongly_disconfirming",
            "H2": "strongly_confirming",
        },
    }

    def complete_structured(self, request):
        assert request.task == "judge_evidence"
        content = request.input["signal"]["raw_content"]
        likelihoods = self._LIKELIHOODS[content]
        targets = request.input["target_hypotheses"]
        return {
            "evidence_type": (
                "counterevidence" if "REFUTE" in content or "REVERSE" in content
                else "supporting"
            ),
            "likelihoods": {
                hypothesis_id: likelihoods[hypothesis_id]
                for hypothesis_id in targets
            },
            "unresolved_likelihood": None,
            "frame_fit": "explained_by_named",
            "unexplained_observation": None,
            "interpretation": f"Scripted assessment for {content}.",
            "quality_overrides": {},
        }


class ScriptedProbeGateway:
    def __init__(self, steps: list[SignalStep]) -> None:
        self._steps = list(steps)
        self._index = 0

    def execute_probe(self, *, probe, context: ProbeExecutionBrief):
        if self._index >= len(self._steps):
            raise AssertionError("run executed more probes than the scripted trace")
        step = self._steps[self._index]
        self._index += 1
        is_model = step.origin == EpistemicOrigin.MODEL_REASONING
        source = "scripted-model-v1" if is_model else "scripted-tool-v1"
        return [
            ExternalSignal(
                id=f"S_{context.run_id}_{self._index}",
                cycle_id=probe.cycle_id,
                signal_kind=SignalKind.ACTIVE,
                source_type=(
                    "model_probe_gateway" if is_model else "test_execution"
                ),
                source=source,
                raw_content=step.content,
                generated_by_probe=probe.id,
                initial_target_hypotheses=[
                    hypothesis.id for hypothesis in context.hypotheses
                ],
                provenance=SignalProvenance(
                    epistemic_origin=step.origin,
                    source_identity=source,
                    provider_model_or_tool_identity=source,
                    session_id=context.run_id if is_model else None,
                    derivation_root_id=step.derivation_root_id,
                    correlation_group=f"scripted:{source}",
                    canonical_content_fingerprint="pending-normalization",
                ),
            )
        ]


class RecordingEvidenceGate:
    def __init__(self, delegate) -> None:
        self._delegate = delegate
        self.boundary_statuses = []

    def integrate(self, *, cycle, belief_state, probe_set, signals):
        self.boundary_statuses.append(cycle.boundary_status)
        return self._delegate.integrate(
            cycle=cycle,
            belief_state=belief_state,
            probe_set=probe_set,
            signals=signals,
        )


class RecordingBeliefSolver(CoverageAwareBeliefSolver):
    def __init__(self) -> None:
        super().__init__()
        self.delta_batches = []

    def solve(self, belief_state, contribution_deltas, *, run_id, cycle_id):
        self.delta_batches.append(list(contribution_deltas))
        return super().solve(
            belief_state,
            contribution_deltas,
            run_id=run_id,
            cycle_id=cycle_id,
        )


class ConformanceCore(BayesProbeCore):
    def _create_evidence_integration_gate(self):
        self.recording_gate = RecordingEvidenceGate(
            super()._create_evidence_integration_gate()
        )
        return self.recording_gate

    def _create_belief_solver(self):
        self.recording_solver = RecordingBeliefSolver()
        return self.recording_solver


def _hypothesis_seeds() -> list[HypothesisSeed]:
    return [
        HypothesisSeed(
            id="H1",
            statement="The primary claim is correct.",
            prior=0.5,
            scope="The scripted conformance question.",
            predictions=["An independent check supports the primary claim."],
            falsifiers=["A reliable independent check contradicts the primary claim."],
        ),
        HypothesisSeed(
            id="H2",
            statement="The rival claim is correct.",
            prior=0.5,
            scope="The scripted conformance question.",
            predictions=["An independent check supports the rival claim."],
            falsifiers=["A reliable independent check contradicts the rival claim."],
        ),
    ]


def _run_scripted_question(
    run_id: str,
    steps: list[SignalStep],
    *,
    max_cycles: int,
):
    evidence_gateway = ScriptedEvidenceGateway()
    core = ConformanceCore(model_gateway=evidence_gateway)
    runner = AutonomousQuestionRunner(
        core=core,
        executor=ProbeExecutor(ScriptedProbeGateway(steps)),
        config=AutonomousQuestionRunConfig(
            max_cycles=max_cycles,
            max_probes_per_cycle=1,
        ),
    )
    result = runner.run_question(
        InitializeRunInput(
            run_id=run_id,
            problem="Which of the two rival claims survives independent checking?",
            task_kind=TaskKind.DECISION,
            hypothesis_relation=HypothesisRelation.EXCLUSIVE_EXHAUSTIVE,
            hypothesis_seeds=_hypothesis_seeds(),
        )
    )
    return result, core


def _assert_atomic_loop(result, core: ConformanceCore) -> None:
    assert core.recording_gate.boundary_statuses == [
        BoundaryStatus.CLOSED
    ] * len(result.cycle_results)
    accepted_events = [
        event
        for cycle in result.cycle_results
        for event in cycle.evidence_events
        if event.discard_reason is None
    ]
    assert accepted_events
    assert all(
        event.contribution_root_id is not None
        and event.contribution_root_id.startswith("evidence-root:sha256:")
        for event in accepted_events
    )
    assert len(core.recording_solver.delta_batches) == len(result.cycle_results)
    assert all(core.recording_solver.delta_batches)
    assert all(
        isinstance(delta, EvidenceContributionDelta)
        for batch in core.recording_solver.delta_batches
        for delta in batch
    )
    assert not any(
        isinstance(item, EvidenceEvent)
        for batch in core.recording_solver.delta_batches
        for item in batch
    )


def test_same_model_root_cannot_self_reinforce_across_ten_requested_cycles():
    result, core = _run_scripted_question(
        "conformance_same_model_root",
        [
            SignalStep(
                content="MODEL_SUPPORT_H1",
                origin=EpistemicOrigin.MODEL_REASONING,
                derivation_root_id="model-thought:cycle-1",
            ),
            SignalStep(
                content="MODEL_SUPPORT_H1",
                origin=EpistemicOrigin.MODEL_REASONING,
                derivation_root_id="model-thought:cycle-2",
            ),
        ],
        max_cycles=10,
    )

    first, second = result.cycle_results
    first_posterior = first.belief_state.hypotheses_by_id()["H1"].posterior
    second_posterior = second.belief_state.hypotheses_by_id()["H1"].posterior
    assert len(result.cycle_results) == 2
    assert first.contribution_deltas[0].mode == EvidenceContributionMode.NEW_ROOT
    assert second.contribution_deltas[0].mode == EvidenceContributionMode.NO_CHANGE
    assert second_posterior == first_posterior
    assert result.stop_reason == AutonomousQuestionStopReason.EPISTEMIC_STAGNATION
    assert any(
        probe.purpose == ProbePurpose.HYPOTHESIS_FALSIFICATION
        for probe in second.probe_set.probes
    )
    assert second.epistemic_progress.falsification_probe_executed is True
    _assert_atomic_loop(result, core)


def test_independent_tool_root_can_change_model_reasoning_conclusion():
    result, core = _run_scripted_question(
        "conformance_model_then_tool",
        [
            SignalStep(
                content="MODEL_SUPPORT_H1",
                origin=EpistemicOrigin.MODEL_REASONING,
                derivation_root_id="model-thought:first",
            ),
            SignalStep(
                content="TOOL_REFUTE_H1",
                origin=EpistemicOrigin.TOOL_RESULT,
                derivation_root_id="tool-observation:independent",
            ),
        ],
        max_cycles=2,
    )

    first, second = result.cycle_results
    assert first.contribution_deltas[0].mode == EvidenceContributionMode.NEW_ROOT
    assert second.contribution_deltas[0].mode == EvidenceContributionMode.NEW_ROOT
    assert first.contribution_deltas[0].contribution_root_id != (
        second.contribution_deltas[0].contribution_root_id
    )
    assert second.epistemic_progress.new_root_count == 1
    assert first.belief_state.hypotheses_by_id()["H1"].posterior > 0.5
    assert result.final_belief_state.hypotheses_by_id()["H1"].posterior < 0.5
    assert result.final_answer_projection.current_best_hypothesis == "H2"
    _assert_atomic_loop(result, core)


def test_same_root_revision_can_reverse_without_double_counting():
    result, core = _run_scripted_question(
        "conformance_same_tool_revision",
        [
            SignalStep(
                content="MODEL_WEAK_SUPPORT_H1",
                origin=EpistemicOrigin.MODEL_REASONING,
                derivation_root_id="model-assessment:cycle-1",
            ),
            SignalStep(
                content="MODEL_REVERSE_H1",
                origin=EpistemicOrigin.MODEL_REASONING,
                derivation_root_id="model-assessment:cycle-2",
            ),
        ],
        max_cycles=2,
    )

    first, second = result.cycle_results
    assert first.contribution_deltas[0].mode == EvidenceContributionMode.NEW_ROOT
    assert second.contribution_deltas[0].mode == EvidenceContributionMode.REVISE_ROOT
    assert first.belief_state.hypotheses_by_id()["H1"].posterior > 0.5
    assert second.belief_state.hypotheses_by_id()["H1"].posterior < 0.5
    assert len(second.belief_state.evidence_memory.root_contributions) == 1
    _assert_atomic_loop(result, core)
