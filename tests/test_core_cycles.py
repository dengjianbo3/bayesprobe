from pathlib import Path

import pytest

from bayesprobe.core import BayesProbeCore, EvidenceIntegrationGate
from bayesprobe.evidence import EvidenceIntegrationResult
from bayesprobe.initialization import BayesProbeInitializer, InitializeRunInput
from bayesprobe.inbox import SignalInbox
from bayesprobe.hypothesis_evolution import HypothesisEvolutionEngine
from bayesprobe.ledger import JsonlLedgerStore
from bayesprobe.model_gateway import (
    EvidenceJudgmentRepairPolicy,
    ModelGatewayValidationError,
    ScriptedModelGateway,
)
from bayesprobe.schemas import (
    BeliefState,
    AnswerContractOutline,
    AnswerValueType,
    BoundaryStatus,
    CycleRecord,
    CycleSignalShape,
    EvidenceEvent,
    EvidenceMemorySnapshot,
    EvidenceType,
    ExternalSignal,
    FramingMethod,
    FrameAdequacyStatus,
    FrameFit,
    Hypothesis,
    HypothesisEvolution,
    HypothesisStatus,
    EvolutionOperation,
    EpistemicOrigin,
    LikelihoodBand,
    ProbeDesign,
    ProbeSet,
    SignalKind,
    SignalProvenance,
    TaskAdmissionDecision,
    TaskAdmissionStatus,
    TaskKind,
)
from bayesprobe.task_framing import ModelTaskFramer


class RecordingSignalInbox(SignalInbox):
    def __init__(self, cycle_id: str):
        super().__init__(cycle_id)
        self.added: list[str] = []
        self.closed_called = False

    def add(self, signal: ExternalSignal) -> ExternalSignal:
        self.added.append(signal.id)
        return super().add(signal)

    def close(self) -> list[ExternalSignal]:
        self.closed_called = True
        return super().close()


class RecordingEvidenceIntegrationGate(EvidenceIntegrationGate):
    def __init__(self) -> None:
        self.seen_signal_ids: list[str] = []
        self.seen_cycle_ids: list[str] = []
        self.seen_inbox_statuses: list[str] = []
        self.seen_provenance = []

    def integrate(
        self,
        *,
        cycle: CycleRecord,
        belief_state: BeliefState,
        probe_set: ProbeSet,
        signals: list[ExternalSignal],
    ) -> list:
        self.seen_cycle_ids.append(cycle.cycle_id)
        self.seen_signal_ids.extend(signal.id for signal in signals)
        self.seen_inbox_statuses.extend(signal.inbox_status.value for signal in signals)
        self.seen_provenance.extend(signal.provenance for signal in signals)
        assert all(signal.cycle_id == cycle.cycle_id for signal in signals)
        assert all(signal.inbox_status.value == "accepted" for signal in signals)
        return super().integrate(
            cycle=cycle,
            belief_state=belief_state,
            probe_set=probe_set,
            signals=signals,
        )


class RecordingCore(BayesProbeCore):
    def __init__(self) -> None:
        self.inbox = None
        self.gate = RecordingEvidenceIntegrationGate()
        super().__init__()

    def _create_signal_inbox(self, cycle: CycleRecord) -> RecordingSignalInbox:
        self.inbox = RecordingSignalInbox(cycle.cycle_id)
        return self.inbox

    def _create_evidence_integration_gate(self) -> RecordingEvidenceIntegrationGate:
        return self.gate


def make_belief_state(cycle_id: str = "cycle_1", cycle_index: int = 0) -> BeliefState:
    return BeliefState(
        belief_state_id="bs_1",
        run_id="run_1",
        cycle_id=cycle_id,
        cycle_index=cycle_index,
        hypotheses=[
            Hypothesis(
                id="H1",
                statement="The claim is supported.",
                scope="claim verification",
                prior=0.5,
                posterior=0.5,
                rivals=["H2"],
                falsifiers=["A refuting sentence weakens H1."],
                predictions=["Supporting evidence is likely."],
            ),
            Hypothesis(
                id="H2",
                statement="The claim is refuted.",
                scope="claim verification",
                prior=0.5,
                posterior=0.5,
                rivals=["H1"],
                falsifiers=["A supporting sentence weakens H2."],
                predictions=["Refuting evidence is likely."],
            ),
        ],
    )


def make_exact_belief_state() -> BeliefState:
    initializer = BayesProbeInitializer(
        task_framer=ModelTaskFramer(
            ScriptedModelGateway(
                {
                    "frame_open_question": {
                        "task_kind": "exact_answer",
                        "answer_relationship": "selection",
                        "answer_contract": {
                            "objective": "Return the supported integer value.",
                            "answer_value_type": "integer",
                            "answer_format": "integer",
                            "required_sections": ["answer", "basis", "uncertainty"],
                            "decision_form": "single_value",
                            "permits_synthesis": False,
                        },
                        "competition": "exclusive",
                        "coverage": "open",
                        "hypotheses": [
                            {
                                "statement": "The requested integer is 7.",
                                "type": "answer_candidate",
                                "scope": "The supplied integer constraints.",
                                "falsifiers": ["A constraint excludes 7."],
                                "predictions": ["Substitution verifies every constraint."],
                                "answer_value": 7,
                            },
                            {
                                "statement": "The requested integer is 9.",
                                "type": "answer_candidate",
                                "scope": "The supplied integer constraints.",
                                "falsifiers": ["A constraint excludes 9."],
                                "predictions": ["Substitution verifies every constraint."],
                                "answer_value": 9,
                            },
                        ],
                        "coverage_statement": "The named values are initial candidates.",
                        "coverage_limitation": "Other integer values remain unresolved.",
                    }
                }
            )
        )
    )
    return initializer.initialize(
        InitializeRunInput(
            run_id="run_1",
            problem="Which integer satisfies the constraints?",
        ),
        admission_decision=TaskAdmissionDecision(
            attempt_id="run_1_admission",
            status=TaskAdmissionStatus.ADMITTED,
            epistemic_basis=["The integer answer can be checked."],
            proposed_task_kind=TaskKind.EXACT_ANSWER,
            answer_contract_outline=AnswerContractOutline(
                objective="Return the supported integer value.",
                answer_value_type=AnswerValueType.INTEGER,
                decision_form="single_value",
                permits_synthesis=False,
                required_sections=["answer", "basis", "uncertainty"],
            ),
            reason="The exact-answer task is admissible.",
        ),
    ).belief_state


def make_cycle(cycle_id: str = "cycle_repair") -> CycleRecord:
    return CycleRecord(
        cycle_id=cycle_id,
        run_id="run_1",
        cycle_index=1,
        signal_shape=CycleSignalShape.ACTIVE_ONLY,
    )


def make_empty_probe_set(cycle_id: str = "cycle_repair") -> ProbeSet:
    return ProbeSet(
        probe_set_id=f"ps_{cycle_id}",
        cycle_id=cycle_id,
        probes=[],
        selection_reason="Repair policy fixture.",
        may_be_empty=True,
    )


def make_active_signal(cycle_id: str = "pending") -> ExternalSignal:
    return ExternalSignal(
        id="S_repair",
        cycle_id=cycle_id,
        signal_kind=SignalKind.ACTIVE,
        source_type="benchmark_stream",
        source="fixture",
        raw_content="Malformed judgment fixture.",
        initial_target_hypotheses=["H1", "H2"],
    )


class GatewayValidationErrorOnJudgeGateway:
    adapter_kind = "gateway_validation"

    def complete_structured(self, request):
        raise ModelGatewayValidationError("gateway rejected evidence judgment payload")


class GatewayValidationErrorOnRepairGateway:
    adapter_kind = "gateway_validation"

    def __init__(self) -> None:
        self.requests = []

    def complete_structured(self, request):
        self.requests.append(request)
        if request.task == "judge_evidence":
            return {
                "evidence_type": "not_a_type",
                "likelihoods": {"H1": "neutral", "H2": "neutral"},
                "interpretation": "Initial payload is invalid.",
            }
        raise ModelGatewayValidationError("gateway rejected repair payload")


def test_active_only_signal_updates_belief_through_evidence_gate():
    core = RecordingCore()
    cycle = CycleRecord(
        cycle_id="cycle_1",
        run_id="run_1",
        cycle_index=1,
        signal_shape=CycleSignalShape.ACTIVE_ONLY,
    )
    signal = ExternalSignal(
        id="S1",
        cycle_id="wrong_cycle",
        signal_kind=SignalKind.ACTIVE,
        source_type="benchmark_stream",
        source="fixture",
        raw_content="REFUTES: The cited sentence contradicts the claim.",
    )
    result = core.integrate_cycle(
        cycle=cycle,
        belief_state=make_belief_state(),
        probe_set=ProbeSet(
            probe_set_id="ps_1",
            cycle_id="cycle_1",
            probes=[],
            selection_reason="Fixture active-only cycle.",
            may_be_empty=True,
        ),
        signals=[signal],
    )

    h1 = result.belief_state.hypotheses_by_id()["H1"]
    h2 = result.belief_state.hypotheses_by_id()["H2"]

    assert core.inbox is not None
    assert core.inbox.added == ["S1"]
    assert core.inbox.closed_called is True
    assert core.gate.seen_signal_ids == ["S1"]
    assert core.gate.seen_cycle_ids == ["cycle_1"]
    assert core.gate.seen_provenance == [None]
    assert result.evidence_events[0].evidence_type == EvidenceType.COUNTEREVIDENCE
    assert result.evidence_events[0].likelihoods["H1"] == LikelihoodBand.MODERATELY_DISCONFIRMING
    assert result.evidence_events[0].id == "run_1_cycle_1_E1"
    assert h1.posterior < 0.5
    assert h2.posterior > 0.5
    assert result.belief_updates[0].evidence_id == "run_1_cycle_1_E1"
    assert result.belief_updates[0].update_id.startswith("run_1_cycle_1_U1_")
    assert result.belief_state.ledger_refs["probe_sets"] == ["ps_1"]


def test_passive_projection_is_signal_not_direct_evidence():
    core = RecordingCore()
    cycle = CycleRecord(
        cycle_id="cycle_2",
        run_id="run_1",
        cycle_index=2,
        signal_shape=CycleSignalShape.PASSIVE_ONLY,
    )
    signal = ExternalSignal(
        id="S2",
        cycle_id="wrong_cycle",
        signal_kind=SignalKind.PASSIVE,
        source_type="external_agent_projection",
        source="agent_a",
        raw_content="Agent A believes H2 because Source A refutes the claim.",
    )

    result = core.integrate_cycle(
        cycle=cycle,
        belief_state=make_belief_state(cycle_id="cycle_2"),
        probe_set=ProbeSet(
            probe_set_id="ps_2",
            cycle_id="cycle_2",
            probes=[],
            selection_reason="Passive-only synchronized round.",
            may_be_empty=True,
        ),
        signals=[signal],
    )

    assert result.evidence_events[0].derived_from_signal == "S2"
    assert result.evidence_events[0].evidence_type == EvidenceType.SENDER_JUDGMENT
    assert result.belief_updates


def test_integrated_belief_state_rebuilds_current_summary():
    core = BayesProbeCore()
    cycle = CycleRecord(
        cycle_id="cycle_summary",
        run_id="run_1",
        cycle_index=1,
        signal_shape=CycleSignalShape.ACTIVE_ONLY,
    )
    probe = ProbeDesign(
        id="P_summary",
        cycle_id=cycle.cycle_id,
        target_hypotheses=["H1", "H2"],
        inquiry_goal="Update and summarize the rival distribution.",
        method="source_tracing",
    )

    result = core.integrate_cycle(
        cycle=cycle,
        belief_state=make_belief_state(cycle_id="cycle_0"),
        probe_set=ProbeSet(
            probe_set_id="ps_summary",
            cycle_id=cycle.cycle_id,
            probes=[probe],
            selection_reason="Belief summary fixture.",
        ),
        signals=[
            ExternalSignal(
                id="S_summary",
                cycle_id="pending",
                signal_kind=SignalKind.ACTIVE,
                source_type="benchmark_stream",
                source="fixture",
                raw_content="SUPPORTS: The result supports H1.",
                generated_by_probe=probe.id,
                initial_target_hypotheses=["H1", "H2"],
            )
        ],
    )

    assert result.belief_state.belief_state_id == "run_1_bs_1"
    assert result.belief_state.posterior_summary["top_hypothesis"] == "H1"
    assert result.belief_state.posterior_summary[
        "total_active_posterior"
    ] == pytest.approx(1.0)
    assert result.belief_state.task_frame is not None
    assert result.belief_state.task_frame.framing_method == FramingMethod.LEGACY_MIGRATION
    assert "no external signals" not in result.belief_state.uncertainty_summary


def test_core_returns_integrated_cycle_with_terminal_timestamps():
    core = BayesProbeCore()
    cycle = CycleRecord(
        cycle_id="cycle_lifecycle",
        run_id="run_1",
        cycle_index=1,
        signal_shape=CycleSignalShape.PASSIVE_ONLY,
    )

    result = core.integrate_cycle(
        cycle=cycle,
        belief_state=make_belief_state(cycle_id="cycle_0"),
        probe_set=ProbeSet(
            probe_set_id="ps_lifecycle",
            cycle_id=cycle.cycle_id,
            probes=[],
            selection_reason="Cycle lifecycle fixture.",
            may_be_empty=True,
        ),
        signals=[
            ExternalSignal(
                id="S_lifecycle",
                cycle_id="pending",
                signal_kind=SignalKind.PASSIVE,
                source_type="user_feedback",
                source="user",
                raw_content="SUPPORTS: Lifecycle fixture signal.",
            )
        ],
    )

    assert result.cycle.boundary_status == BoundaryStatus.INTEGRATED
    assert result.cycle.boundary_closed_at is not None
    assert result.cycle.completed_at is not None
    assert (
        result.cycle.started_at
        <= result.cycle.boundary_closed_at
        <= result.cycle.completed_at
    )


def test_active_only_cycle_rejects_passive_signal():
    core = BayesProbeCore()
    cycle = CycleRecord(
        cycle_id="cycle_shape_active",
        run_id="run_1",
        cycle_index=1,
        signal_shape=CycleSignalShape.ACTIVE_ONLY,
    )

    with pytest.raises(ValueError, match="active signals"):
        core.integrate_cycle(
            cycle=cycle,
            belief_state=make_belief_state(cycle_id="cycle_0"),
            probe_set=ProbeSet(
                probe_set_id="ps_shape_active",
                cycle_id="cycle_shape_active",
                probes=[],
                selection_reason="Shape validation.",
                may_be_empty=True,
            ),
            signals=[
                ExternalSignal(
                    id="S_shape_passive",
                    cycle_id="pending",
                    signal_kind=SignalKind.PASSIVE,
                    source_type="external_agent_projection",
                    source="agent_a",
                    raw_content="Agent A reports a passive signal.",
                )
            ],
        )


def test_passive_only_cycle_rejects_active_signal():
    core = BayesProbeCore()
    cycle = CycleRecord(
        cycle_id="cycle_shape_passive",
        run_id="run_1",
        cycle_index=1,
        signal_shape=CycleSignalShape.PASSIVE_ONLY,
    )

    with pytest.raises(ValueError, match="passive signals"):
        core.integrate_cycle(
            cycle=cycle,
            belief_state=make_belief_state(cycle_id="cycle_0"),
            probe_set=ProbeSet(
                probe_set_id="ps_shape_passive",
                cycle_id="cycle_shape_passive",
                probes=[],
                selection_reason="Shape validation.",
                may_be_empty=True,
            ),
            signals=[
                ExternalSignal(
                    id="S_shape_active",
                    cycle_id="pending",
                    signal_kind=SignalKind.ACTIVE,
                    source_type="benchmark_stream",
                    source="fixture",
                    raw_content="SUPPORTS: Active signal.",
                )
            ],
        )


def test_direct_signal_schema_violation_does_not_attempt_repair_by_default():
    gateway = ScriptedModelGateway(
        responses={
            "judge_evidence": {
                "evidence_type": "not_a_type",
                "likelihoods": {"H1": "neutral", "H2": "neutral"},
                "interpretation": "Invalid evidence type.",
            },
            "repair_evidence_judgment": {
                "evidence_type": "supporting",
                "likelihoods": {
                    "H1": "moderately_confirming",
                    "H2": "moderately_disconfirming",
                },
                "interpretation": "This repair should not be called.",
            },
        }
    )
    gate = EvidenceIntegrationGate(model_gateway=gateway)

    result = gate.integrate(
        cycle=make_cycle("cycle_repair_default"),
        belief_state=make_belief_state(cycle_id="cycle_0"),
        probe_set=make_empty_probe_set("cycle_repair_default"),
        signals=[make_active_signal()],
    )

    event = result.evidence_events[0]
    assert [request.task for request in gateway.requests] == ["judge_evidence"]
    assert event.evidence_type == EvidenceType.NEUTRAL
    assert event.discard_reason.startswith("schema_violation:")


def test_direct_signal_repair_success_produces_normal_evidence():
    gateway = ScriptedModelGateway(
        responses={
            "judge_evidence": {
                "evidence_type": "not_a_type",
                "likelihoods": {"H1": "neutral", "H2": "neutral"},
                "interpretation": "Invalid evidence type.",
            },
            "repair_evidence_judgment": {
                "evidence_type": "supporting",
                "likelihoods": {
                    "H1": "moderately_confirming",
                    "H2": "moderately_disconfirming",
                },
                "interpretation": "Repaired supporting judgment.",
                "quality_overrides": {"reliability": 0.91},
            },
        }
    )
    gate = EvidenceIntegrationGate(
        model_gateway=gateway,
        judgment_repair_policy=EvidenceJudgmentRepairPolicy(max_attempts=1),
    )

    result = gate.integrate(
        cycle=make_cycle("cycle_repair_success"),
        belief_state=make_belief_state(cycle_id="cycle_0"),
        probe_set=make_empty_probe_set("cycle_repair_success"),
        signals=[make_active_signal()],
    )

    event = result.evidence_events[0]
    repair_input = gateway.requests[1].input
    assert [request.task for request in gateway.requests] == [
        "judge_evidence",
        "repair_evidence_judgment",
    ]
    assert repair_input["original_request"]["task"] == "judge_evidence"
    assert repair_input["original_request"]["input"]["signal_id"] == "S_repair"
    assert repair_input["invalid_payload"]["evidence_type"] == "not_a_type"
    assert repair_input["validation_error"].startswith("invalid evidence_type")
    assert repair_input["attempt_index"] == 1
    assert "boundary_condition" in repair_input["allowed_evidence_types"]
    assert "moderately_confirming" in repair_input["allowed_likelihood_bands"]
    assert repair_input["required_fields"] == [
        "evidence_type",
        "likelihoods",
        "interpretation",
    ]
    assert event.evidence_type == EvidenceType.SUPPORTING
    assert event.likelihoods["H1"] == LikelihoodBand.MODERATELY_CONFIRMING
    assert event.discard_reason is None
    assert event.interpretation == "Repaired supporting judgment."
    assert event.reliability == 0.8


def test_direct_signal_invalid_repair_becomes_schema_violation():
    gateway = ScriptedModelGateway(
        responses={
            "judge_evidence": {
                "evidence_type": "not_a_type",
                "likelihoods": {"H1": "neutral", "H2": "neutral"},
                "interpretation": "Invalid evidence type.",
            },
            "repair_evidence_judgment": {
                "evidence_type": "still_not_a_type",
                "likelihoods": {"H1": "neutral", "H2": "neutral"},
                "interpretation": "Still invalid.",
            },
        }
    )
    gate = EvidenceIntegrationGate(
        model_gateway=gateway,
        judgment_repair_policy=EvidenceJudgmentRepairPolicy(max_attempts=1),
    )

    result = gate.integrate(
        cycle=make_cycle("cycle_repair_failure"),
        belief_state=make_belief_state(cycle_id="cycle_0"),
        probe_set=make_empty_probe_set("cycle_repair_failure"),
        signals=[make_active_signal()],
    )

    event = result.evidence_events[0]
    assert [request.task for request in gateway.requests] == [
        "judge_evidence",
        "repair_evidence_judgment",
    ]
    assert event.evidence_type == EvidenceType.NEUTRAL
    assert event.discard_reason.startswith(
        "schema_violation: repair failed after 1 attempt(s): invalid evidence_type"
    )
    assert event.reliability == 0.0
    assert event.independence == 0.0
    assert event.relevance == 0.0
    assert event.novelty == 0.0
    assert event.specificity == 0.0
    assert event.verifiability == 0.0


def test_direct_signal_missing_repair_task_raises_when_repair_enabled():
    gateway = ScriptedModelGateway(
        responses={
            "judge_evidence": {
                "evidence_type": "not_a_type",
                "likelihoods": {"H1": "neutral", "H2": "neutral"},
                "interpretation": "Invalid evidence type.",
            }
        }
    )
    gate = EvidenceIntegrationGate(
        model_gateway=gateway,
        judgment_repair_policy=EvidenceJudgmentRepairPolicy(max_attempts=1),
    )

    with pytest.raises(ValueError, match="no scripted response for task: repair_evidence_judgment"):
        gate.integrate(
            cycle=make_cycle("cycle_repair_missing_task"),
            belief_state=make_belief_state(cycle_id="cycle_0"),
            probe_set=make_empty_probe_set("cycle_repair_missing_task"),
            signals=[make_active_signal()],
        )


def test_direct_signal_valid_judgment_records_model_trace():
    gateway = ScriptedModelGateway(
        responses={
            "judge_evidence": {
                "evidence_type": "supporting",
                "likelihoods": {
                    "H1": "moderately_confirming",
                    "H2": "moderately_disconfirming",
                },
                "interpretation": "Scripted supporting judgment.",
            }
        }
    )
    gate = EvidenceIntegrationGate(model_gateway=gateway)

    result = gate.integrate(
        cycle=make_cycle("cycle_model_trace_valid"),
        belief_state=make_belief_state(cycle_id="cycle_0"),
        probe_set=make_empty_probe_set("cycle_model_trace_valid"),
        signals=[make_active_signal()],
    )

    request = gateway.requests[0]
    event = result.evidence_events[0]
    assert request.prompt_id == "evidence_judgment"
    assert request.prompt_version == "v0.1"
    assert request.schema_name == "EvidenceJudgment"
    assert request.schema_version == "v0.1"
    assert request.metadata == {}
    assert event.model_trace == {
        "task": "judge_evidence",
        "adapter_kind": "scripted",
        "prompt_id": "evidence_judgment",
        "prompt_version": "v0.1",
        "schema_name": "EvidenceJudgment",
        "schema_version": "v0.1",
        "metadata": {},
    }


def test_direct_signal_schema_violation_records_judge_model_trace():
    gateway = ScriptedModelGateway(
        responses={
            "judge_evidence": {
                "evidence_type": "not_a_type",
                "likelihoods": {"H1": "neutral", "H2": "neutral"},
                "interpretation": "Invalid evidence type.",
            }
        }
    )
    gate = EvidenceIntegrationGate(model_gateway=gateway)

    result = gate.integrate(
        cycle=make_cycle("cycle_model_trace_violation"),
        belief_state=make_belief_state(cycle_id="cycle_0"),
        probe_set=make_empty_probe_set("cycle_model_trace_violation"),
        signals=[make_active_signal()],
    )

    event = result.evidence_events[0]
    assert event.discard_reason.startswith("schema_violation:")
    assert event.model_trace["task"] == "judge_evidence"
    assert event.model_trace["adapter_kind"] == "scripted"
    assert event.model_trace["prompt_id"] == "evidence_judgment"
    assert event.model_trace["schema_name"] == "EvidenceJudgment"


def test_direct_signal_gateway_validation_error_becomes_schema_violation():
    gate = EvidenceIntegrationGate(model_gateway=GatewayValidationErrorOnJudgeGateway())

    result = gate.integrate(
        cycle=make_cycle("cycle_gateway_validation"),
        belief_state=make_belief_state(cycle_id="cycle_0"),
        probe_set=make_empty_probe_set("cycle_gateway_validation"),
        signals=[make_active_signal()],
    )

    event = result.evidence_events[0]
    assert event.evidence_type == EvidenceType.NEUTRAL
    assert event.discard_reason == "schema_violation: gateway rejected evidence judgment payload"
    assert event.model_trace == {
        "task": "judge_evidence",
        "adapter_kind": "gateway_validation",
        "prompt_id": "evidence_judgment",
        "prompt_version": "v0.1",
        "schema_name": "EvidenceJudgment",
        "schema_version": "v0.1",
        "metadata": {},
    }


def test_direct_signal_repaired_judgment_records_repair_model_trace():
    gateway = ScriptedModelGateway(
        responses={
            "judge_evidence": {
                "evidence_type": "not_a_type",
                "likelihoods": {"H1": "neutral", "H2": "neutral"},
                "interpretation": "Invalid evidence type.",
            },
            "repair_evidence_judgment": {
                "evidence_type": "supporting",
                "likelihoods": {
                    "H1": "moderately_confirming",
                    "H2": "moderately_disconfirming",
                },
                "interpretation": "Repaired supporting judgment.",
            },
        }
    )
    gate = EvidenceIntegrationGate(
        model_gateway=gateway,
        judgment_repair_policy=EvidenceJudgmentRepairPolicy(max_attempts=1),
    )

    result = gate.integrate(
        cycle=make_cycle("cycle_model_trace_repair"),
        belief_state=make_belief_state(cycle_id="cycle_0"),
        probe_set=make_empty_probe_set("cycle_model_trace_repair"),
        signals=[make_active_signal()],
    )

    repair_request = gateway.requests[1]
    event = result.evidence_events[0]
    assert repair_request.prompt_id == "evidence_judgment_repair"
    assert repair_request.prompt_version == "v0.1"
    assert repair_request.schema_name == "EvidenceJudgment"
    assert repair_request.schema_version == "v0.1"
    assert repair_request.metadata == {"repair_attempt_index": 1}
    assert event.discard_reason is None
    assert event.model_trace == {
        "task": "repair_evidence_judgment",
        "adapter_kind": "scripted",
        "prompt_id": "evidence_judgment_repair",
        "prompt_version": "v0.1",
        "schema_name": "EvidenceJudgment",
        "schema_version": "v0.1",
        "repair_attempt_index": 1,
        "metadata": {},
    }


def test_direct_signal_repair_exhaustion_records_latest_repair_trace():
    gateway = GatewayValidationErrorOnRepairGateway()
    gate = EvidenceIntegrationGate(
        model_gateway=gateway,
        judgment_repair_policy=EvidenceJudgmentRepairPolicy(max_attempts=2),
    )

    result = gate.integrate(
        cycle=make_cycle("cycle_model_trace_repair_exhausted"),
        belief_state=make_belief_state(cycle_id="cycle_0"),
        probe_set=make_empty_probe_set("cycle_model_trace_repair_exhausted"),
        signals=[make_active_signal()],
    )

    event = result.evidence_events[0]
    assert [request.task for request in gateway.requests] == [
        "judge_evidence",
        "repair_evidence_judgment",
        "repair_evidence_judgment",
    ]
    assert event.discard_reason == (
        "schema_violation: repair failed after 2 attempt(s): gateway rejected repair payload"
    )
    assert event.model_trace == {
        "task": "repair_evidence_judgment",
        "adapter_kind": "gateway_validation",
        "prompt_id": "evidence_judgment_repair",
        "prompt_version": "v0.1",
        "schema_name": "EvidenceJudgment",
        "schema_version": "v0.1",
        "repair_attempt_index": 2,
        "metadata": {},
    }


def test_projection_decomposition_events_keep_empty_model_trace():
    gate = EvidenceIntegrationGate()
    signal = ExternalSignal(
        id="S_projection_trace",
        cycle_id="pending",
        signal_kind=SignalKind.PASSIVE,
        source_type="external_agent_projection",
        source="agent_a",
        raw_content="Agent A believes H2 because Source A refutes the claim.",
    )

    result = gate.integrate(
        cycle=make_cycle("cycle_projection_trace"),
        belief_state=make_belief_state(cycle_id="cycle_0"),
        probe_set=make_empty_probe_set("cycle_projection_trace"),
        signals=[signal],
    )

    assert [event.evidence_type for event in result.evidence_events] == [
        EvidenceType.SENDER_JUDGMENT,
        EvidenceType.SOURCE_CLAIM,
    ]
    assert [event.model_trace for event in result.evidence_events] == [{}, {}]


def test_core_passes_judgment_repair_policy_to_evidence_gate():
    gateway = ScriptedModelGateway(
        responses={
            "judge_evidence": {
                "evidence_type": "not_a_type",
                "likelihoods": {"H1": "neutral", "H2": "neutral"},
                "interpretation": "Invalid evidence type.",
            },
            "repair_evidence_judgment": {
                "evidence_type": "supporting",
                "likelihoods": {
                    "H1": "moderately_confirming",
                    "H2": "moderately_disconfirming",
                },
                "interpretation": "Core repaired judgment.",
            },
        }
    )
    core = BayesProbeCore(
        model_gateway=gateway,
        judgment_repair_policy=EvidenceJudgmentRepairPolicy(max_attempts=1),
    )
    cycle = make_cycle("cycle_core_repair")

    result = core.integrate_cycle(
        cycle=cycle,
        belief_state=make_belief_state(cycle_id="cycle_0"),
        probe_set=make_empty_probe_set("cycle_core_repair"),
        signals=[make_active_signal()],
    )

    h1 = result.belief_state.hypotheses_by_id()["H1"]
    h2 = result.belief_state.hypotheses_by_id()["H2"]
    assert [request.task for request in gateway.requests] == [
        "judge_evidence",
        "repair_evidence_judgment",
    ]
    assert result.evidence_events[0].discard_reason is None
    assert result.evidence_events[0].evidence_type == EvidenceType.SUPPORTING
    assert len(result.belief_updates) == 2
    assert h1.posterior > 0.5
    assert h2.posterior < 0.5


def test_active_plus_passive_cycle_accepts_mixed_signal_kinds():
    core = BayesProbeCore()
    cycle = CycleRecord(
        cycle_id="cycle_shape_mixed",
        run_id="run_1",
        cycle_index=1,
        signal_shape=CycleSignalShape.ACTIVE_PLUS_PASSIVE,
    )

    result = core.integrate_cycle(
        cycle=cycle,
        belief_state=make_belief_state(cycle_id="cycle_0"),
        probe_set=ProbeSet(
            probe_set_id="ps_shape_mixed",
            cycle_id="cycle_shape_mixed",
            probes=[],
            selection_reason="Shape validation.",
            may_be_empty=True,
        ),
        signals=[
            ExternalSignal(
                id="S_shape_active_ok",
                cycle_id="pending",
                signal_kind=SignalKind.ACTIVE,
                source_type="benchmark_stream",
                source="fixture",
                raw_content="SUPPORTS: Active signal.",
            ),
            ExternalSignal(
                id="S_shape_passive_ok",
                cycle_id="pending",
                signal_kind=SignalKind.PASSIVE,
                source_type="user_feedback",
                source="user",
                raw_content="SUPPORTS: Passive signal.",
            ),
        ],
    )

    assert [signal.derived_from_signal for signal in result.evidence_events] == [
        "S_shape_active_ok",
        "S_shape_passive_ok",
    ]


@pytest.mark.parametrize(
    ("shape", "signals", "message"),
    [
        (CycleSignalShape.ACTIVE_ONLY, [], "at least one active signal"),
        (CycleSignalShape.PASSIVE_ONLY, [], "at least one passive signal"),
        (
            CycleSignalShape.ACTIVE_PLUS_PASSIVE,
            [
                ExternalSignal(
                    id="S_shape_only_active",
                    cycle_id="pending",
                    signal_kind=SignalKind.ACTIVE,
                    source_type="benchmark_stream",
                    source="fixture",
                    raw_content="SUPPORTS: Active signal without passive peer.",
                )
            ],
            "both active and passive signals",
        ),
    ],
)
def test_cycle_signal_shape_requires_its_declared_signal_composition(
    shape,
    signals,
    message,
):
    core = BayesProbeCore()
    cycle = CycleRecord(
        cycle_id=f"cycle_required_{shape.value}",
        run_id="run_1",
        cycle_index=1,
        signal_shape=shape,
    )

    with pytest.raises(ValueError, match=message):
        core.integrate_cycle(
            cycle=cycle,
            belief_state=make_belief_state(cycle_id="cycle_0"),
            probe_set=ProbeSet(
                probe_set_id=f"ps_required_{shape.value}",
                cycle_id=cycle.cycle_id,
                probes=[],
                selection_reason="Exact shape validation fixture.",
                may_be_empty=True,
            ),
            signals=signals,
        )


def test_previous_cycle_belief_state_advances_to_current_cycle():
    core = BayesProbeCore()
    cycle = CycleRecord(
        cycle_id="current_cycle",
        run_id="run_1",
        cycle_index=1,
        signal_shape=CycleSignalShape.PASSIVE_ONLY,
    )

    result = core.integrate_cycle(
        cycle=cycle,
        belief_state=make_belief_state(cycle_id="prior_cycle", cycle_index=0),
        probe_set=ProbeSet(
            probe_set_id="ps_1",
            cycle_id="current_cycle",
            probes=[],
            selection_reason="Advance from previous cycle.",
            may_be_empty=True,
        ),
        signals=[
            ExternalSignal(
                id="S_advance",
                cycle_id="pending",
                signal_kind=SignalKind.PASSIVE,
                source_type="system_log",
                source="fixture",
                raw_content="NEUTRAL: Advance with an auditable passive signal.",
            )
        ],
    )

    assert result.belief_state.cycle_id == "current_cycle"
    assert result.belief_state.cycle_index == 1
    assert result.belief_state.ledger_refs["probe_sets"] == ["ps_1"]


def test_probe_set_cycle_must_match_cycle_boundary():
    core = BayesProbeCore()
    cycle = CycleRecord(
        cycle_id="cycle_3",
        run_id="run_1",
        cycle_index=3,
        signal_shape=CycleSignalShape.ACTIVE_ONLY,
    )

    with pytest.raises(ValueError, match="probe set"):
        core.integrate_cycle(
            cycle=cycle,
            belief_state=make_belief_state(cycle_id="cycle_3"),
            probe_set=ProbeSet(
                probe_set_id="ps_3",
                cycle_id="other_cycle",
                probes=[],
                selection_reason="Mismatched boundary.",
                may_be_empty=True,
            ),
            signals=[],
        )


def test_probe_designs_must_match_frozen_cycle_boundary():
    core = BayesProbeCore()
    cycle = CycleRecord(
        cycle_id="cycle_4",
        run_id="run_1",
        cycle_index=4,
        signal_shape=CycleSignalShape.ACTIVE_ONLY,
    )

    with pytest.raises(ValueError, match="probe design"):
        core.integrate_cycle(
            cycle=cycle,
            belief_state=make_belief_state(cycle_id="cycle_3"),
            probe_set=ProbeSet(
                probe_set_id="ps_4",
                cycle_id="cycle_4",
                probes=[
                    ProbeDesign(
                        id="P1",
                        cycle_id="cycle_3",
                        target_hypotheses=["H1"],
                        inquiry_goal="Check boundary freeze.",
                        method="text_probe",
                    )
                ],
                selection_reason="Mismatched probe cycle.",
                may_be_empty=False,
            ),
            signals=[],
        )


def test_anomaly_triggers_hypothesis_evolution_before_next_probe():
    core = BayesProbeCore()
    cycle = CycleRecord(
        cycle_id="cycle_3",
        run_id="run_1",
        cycle_index=3,
        signal_shape=CycleSignalShape.PASSIVE_ONLY,
    )
    result = core.integrate_cycle(
        cycle=cycle,
        belief_state=make_belief_state(cycle_id="cycle_3", cycle_index=2),
        probe_set=ProbeSet(
            probe_set_id="ps_3",
            cycle_id="cycle_3",
            probes=[],
            selection_reason="Passive-only anomaly fixture.",
            may_be_empty=True,
        ),
        signals=[
            ExternalSignal(
                id="S3",
                cycle_id="cycle_3",
                signal_kind=SignalKind.PASSIVE,
                source_type="system_log",
                source="fixture",
                raw_content="ANOMALY: This signal is poorly explained by current hypotheses.",
            )
        ],
    )

    assert result.evidence_events[0].evidence_type == EvidenceType.ANOMALY
    assert len(result.hypothesis_evolutions) == 1
    evolution = result.hypothesis_evolutions[0]
    assert isinstance(evolution, HypothesisEvolution)
    assert evolution.triggered_by == ["run_1_cycle_3_E1"]
    assert evolution.operation == EvolutionOperation.SPAWN
    assert evolution.to_hypothesis == "H_run_1_cycle_3_E1_spawned"
    assert evolution.evolution_id == "run_1_cycle_3_E1_HE"
    assert evolution.audit_fields["new_hypothesis_prior"] == 0.12
    spawned = result.belief_state.hypotheses_by_id()["H_run_1_cycle_3_E1_spawned"]
    assert spawned.created_by == "spawned"
    assert spawned.prior == 0.12
    assert spawned.posterior != spawned.prior
    assert sum(
        hypothesis.posterior for hypothesis in result.belief_state.hypotheses
    ) == pytest.approx(1.0)
    assert spawned.rivals == ["H1", "H2"]
    assert spawned.why_existing_hypotheses_failed == evolution.reason
    assert result.probe_candidates
    assert result.probe_candidates[0].source == "anomaly"
    assert result.probe_candidates[0].candidate_probe.target_hypotheses == [
        "H_run_1_cycle_3_E1_spawned"
    ]
    assert result.belief_state.ledger_refs["probe_candidates"] == [
        result.probe_candidates[0].candidate_id
    ]


def test_initial_target_hypotheses_limit_evidence_and_updates():
    core = BayesProbeCore()
    cycle = CycleRecord(
        cycle_id="cycle_target_initial",
        run_id="run_1",
        cycle_index=2,
        signal_shape=CycleSignalShape.PASSIVE_ONLY,
    )

    result = core.integrate_cycle(
        cycle=cycle,
        belief_state=make_belief_state(cycle_id="cycle_target_initial", cycle_index=1),
        probe_set=ProbeSet(
            probe_set_id="ps_target_initial",
            cycle_id="cycle_target_initial",
            probes=[],
            selection_reason="Initial targets should constrain integration.",
            may_be_empty=True,
        ),
        signals=[
            ExternalSignal(
                id="S_target_initial",
                cycle_id="pending",
                signal_kind=SignalKind.PASSIVE,
                source_type="user_feedback",
                source="user",
                raw_content="SUPPORTS: Evidence only for H1.",
                initial_target_hypotheses=["H1"],
            )
        ],
    )

    event = result.evidence_events[0]

    assert event.target_hypotheses == ["H1"]
    assert set(event.likelihoods) == {"H1"}
    assert result.belief_state.hypotheses_by_id()["H1"].posterior > 0.5
    assert result.belief_state.hypotheses_by_id()["H2"].posterior < 0.5
    assert sum(
        hypothesis.posterior for hypothesis in result.belief_state.hypotheses
    ) == pytest.approx(1.0)
    assert [update.hypothesis_id for update in result.belief_updates] == ["H1", "H2"]


def test_probe_target_hypotheses_limit_evidence_when_signal_names_probe():
    core = BayesProbeCore()
    cycle = CycleRecord(
        cycle_id="cycle_target_probe",
        run_id="run_1",
        cycle_index=2,
        signal_shape=CycleSignalShape.ACTIVE_ONLY,
    )

    result = core.integrate_cycle(
        cycle=cycle,
        belief_state=make_belief_state(cycle_id="cycle_target_probe", cycle_index=1),
        probe_set=ProbeSet(
            probe_set_id="ps_target_probe",
            cycle_id="cycle_target_probe",
            probes=[
                ProbeDesign(
                    id="P_target_h1",
                    cycle_id="cycle_target_probe",
                    target_hypotheses=["H1"],
                    inquiry_goal="Only test H1.",
                    method="text_probe",
                )
            ],
            selection_reason="Probe targets should constrain integration.",
            may_be_empty=False,
        ),
        signals=[
            ExternalSignal(
                id="S_target_probe",
                cycle_id="pending",
                signal_kind=SignalKind.ACTIVE,
                source_type="benchmark_stream",
                source="fixture",
                raw_content="REFUTES: Evidence returned by the probe.",
                generated_by_probe="P_target_h1",
            )
        ],
    )

    event = result.evidence_events[0]

    assert event.target_hypotheses == ["H1"]
    assert set(event.likelihoods) == {"H1"}
    assert result.belief_state.hypotheses_by_id()["H1"].posterior < 0.5
    assert result.belief_state.hypotheses_by_id()["H2"].posterior > 0.5
    assert sum(
        hypothesis.posterior for hypothesis in result.belief_state.hypotheses
    ) == pytest.approx(1.0)
    assert [update.hypothesis_id for update in result.belief_updates] == ["H1", "H2"]


def test_direct_signal_judgment_uses_model_gateway():
    gateway = ScriptedModelGateway(
        responses={
            "judge_evidence": {
                "evidence_type": "boundary_condition",
                "likelihoods": {"H1": "weakly_disconfirming", "H2": "neutral"},
                "interpretation": "Scripted boundary judgment.",
                "quality_overrides": {"reliability": 0.62},
            }
        }
    )
    gate = EvidenceIntegrationGate(model_gateway=gateway)
    cycle = CycleRecord(
        cycle_id="cycle_model_gateway",
        run_id="run_1",
        cycle_index=2,
        signal_shape=CycleSignalShape.ACTIVE_ONLY,
    )
    signal = ExternalSignal(
        id="S_model_gateway",
        cycle_id="pending",
        signal_kind=SignalKind.ACTIVE,
        source_type="benchmark_stream",
        source="fixture",
        raw_content="This text has no deterministic keyword cue.",
    )

    result = gate.integrate(
        cycle=cycle,
        belief_state=make_belief_state(cycle_id="cycle_model_gateway", cycle_index=1),
        probe_set=ProbeSet(
            probe_set_id="ps_model_gateway",
            cycle_id="cycle_model_gateway",
            probes=[],
            selection_reason="Model gateway seam test.",
            may_be_empty=True,
        ),
        signals=[signal],
    )

    event = result.evidence_events[0]
    request = gateway.requests[0]

    assert event.evidence_type == EvidenceType.BOUNDARY_CONDITION
    assert event.likelihoods["H1"] == LikelihoodBand.WEAKLY_DISCONFIRMING
    assert event.likelihoods["H2"] == LikelihoodBand.NEUTRAL
    assert event.interpretation == "Scripted boundary judgment."
    assert event.reliability == 0.62
    assert request.task == "judge_evidence"
    assert request.input["signal_id"] == "S_model_gateway"
    assert request.input["raw_content"] == signal.raw_content
    assert request.input["target_hypotheses"] == ["H1", "H2"]
    assert request.input["cycle_id"] == "cycle_model_gateway"
    assert request.input["probe_ids"] == []


@pytest.mark.parametrize(
    "likelihoods",
    [
        {"H1": "moderately_confirming"},
        {
            "H1": "moderately_confirming",
            "H2": "moderately_disconfirming",
            "H3": "neutral",
        },
    ],
)
def test_provider_judgment_targets_must_match_requested_hypotheses(likelihoods):
    gateway = ScriptedModelGateway(
        responses={
            "judge_evidence": {
                "evidence_type": "supporting",
                "likelihoods": likelihoods,
                "interpretation": "Target mismatch fixture.",
                "quality_overrides": {},
            }
        }
    )
    core = BayesProbeCore(model_gateway=gateway)
    cycle = make_cycle("cycle_target_contract")

    result = core.integrate_cycle(
        cycle=cycle,
        belief_state=make_belief_state(cycle_id="cycle_0"),
        probe_set=make_empty_probe_set(cycle.cycle_id),
        signals=[make_active_signal()],
    )

    event = result.evidence_events[0]
    assert event.discard_reason is not None
    assert event.discard_reason.startswith("schema_violation:")
    assert result.belief_updates == []
    assert [
        hypothesis.posterior for hypothesis in result.belief_state.hypotheses
    ] == [0.5, 0.5]


def test_target_mismatch_can_be_repaired_against_original_request():
    gateway = ScriptedModelGateway(
        responses={
            "judge_evidence": {
                "evidence_type": "supporting",
                "likelihoods": {"H1": "moderately_confirming"},
                "interpretation": "H2 was omitted.",
            },
            "repair_evidence_judgment": {
                "evidence_type": "supporting",
                "likelihoods": {
                    "H1": "moderately_confirming",
                    "H2": "moderately_disconfirming",
                },
                "interpretation": "Targets repaired.",
                "quality_overrides": {},
            },
        }
    )
    core = BayesProbeCore(
        model_gateway=gateway,
        judgment_repair_policy=EvidenceJudgmentRepairPolicy(max_attempts=1),
    )
    cycle = make_cycle("cycle_target_repair")

    result = core.integrate_cycle(
        cycle=cycle,
        belief_state=make_belief_state(cycle_id="cycle_0"),
        probe_set=make_empty_probe_set(cycle.cycle_id),
        signals=[make_active_signal()],
    )

    event = result.evidence_events[0]
    assert event.discard_reason is None
    assert event.model_trace["task"] == "repair_evidence_judgment"
    assert event.model_trace["repair_attempt_index"] == 1
    assert {update.hypothesis_id for update in result.belief_updates} == {"H1", "H2"}


def test_direct_signal_schema_violation_becomes_discarded_evidence():
    gateway = ScriptedModelGateway(
        responses={
            "judge_evidence": {
                "likelihoods": {"H1": "neutral", "H2": "neutral"},
                "interpretation": "Missing evidence type.",
            }
        }
    )
    gate = EvidenceIntegrationGate(model_gateway=gateway)
    cycle = CycleRecord(
        cycle_id="cycle_schema_violation",
        run_id="run_1",
        cycle_index=1,
        signal_shape=CycleSignalShape.ACTIVE_ONLY,
    )
    signal = ExternalSignal(
        id="S_schema_violation",
        cycle_id="pending",
        signal_kind=SignalKind.ACTIVE,
        source_type="benchmark_stream",
        source="fixture",
        raw_content="Malformed judgment fixture.",
    )

    result = gate.integrate(
        cycle=cycle,
        belief_state=make_belief_state(cycle_id="cycle_schema_violation"),
        probe_set=ProbeSet(
            probe_set_id="ps_schema_violation",
            cycle_id="cycle_schema_violation",
            probes=[],
            selection_reason="Schema violation evidence test.",
            may_be_empty=True,
        ),
        signals=[signal],
    )

    event = result.evidence_events[0]
    assert event.evidence_type == EvidenceType.NEUTRAL
    assert event.likelihoods == {
        "H1": LikelihoodBand.NEUTRAL,
        "H2": LikelihoodBand.NEUTRAL,
    }
    assert event.discard_reason.startswith("schema_violation:")
    assert "evidence judgment missing field: evidence_type" in event.discard_reason
    assert event.interpretation == "Model gateway judgment failed schema validation."
    assert event.reliability == 0.0
    assert event.independence == 0.0
    assert event.relevance == 0.0
    assert event.novelty == 0.0
    assert event.specificity == 0.0
    assert event.verifiability == 0.0


def test_core_schema_violation_does_not_update_belief_state():
    gateway = ScriptedModelGateway(
        responses={
            "judge_evidence": {
                "likelihoods": {"H1": "moderately_confirming", "H2": "moderately_disconfirming"},
                "interpretation": "Missing evidence type.",
            }
        }
    )
    core = BayesProbeCore(model_gateway=gateway)
    cycle = CycleRecord(
        cycle_id="cycle_schema_violation_core",
        run_id="run_1",
        cycle_index=1,
        signal_shape=CycleSignalShape.PASSIVE_ONLY,
    )

    result = core.integrate_cycle(
        cycle=cycle,
        belief_state=make_belief_state(cycle_id="cycle_schema_violation_core"),
        probe_set=ProbeSet(
            probe_set_id="ps_schema_violation_core",
            cycle_id="cycle_schema_violation_core",
            probes=[],
            selection_reason="Schema violation skip update test.",
            may_be_empty=True,
        ),
        signals=[
            ExternalSignal(
                id="S_schema_violation_core",
                cycle_id="pending",
                signal_kind=SignalKind.PASSIVE,
                source_type="user_feedback",
                source="user",
                raw_content="Malformed judgment fixture.",
            )
        ],
    )

    assert result.evidence_events[0].discard_reason.startswith("schema_violation:")
    assert result.belief_updates == []
    assert result.belief_state.hypotheses_by_id()["H1"].posterior == 0.5
    assert result.belief_state.hypotheses_by_id()["H2"].posterior == 0.5


def test_core_accepts_model_gateway_for_evidence_gate():
    gateway = ScriptedModelGateway(
        responses={
            "judge_evidence": {
                "evidence_type": "boundary_condition",
                "likelihoods": {"H1": "weakly_disconfirming", "H2": "neutral"},
                "interpretation": "Core configured scripted judgment.",
            }
        }
    )
    core = BayesProbeCore(model_gateway=gateway)
    cycle = CycleRecord(
        cycle_id="cycle_core_model_gateway",
        run_id="run_1",
        cycle_index=1,
        signal_shape=CycleSignalShape.PASSIVE_ONLY,
    )

    result = core.integrate_cycle(
        cycle=cycle,
        belief_state=make_belief_state(cycle_id="cycle_core_model_gateway"),
        probe_set=ProbeSet(
            probe_set_id="ps_core_model_gateway",
            cycle_id="cycle_core_model_gateway",
            probes=[],
            selection_reason="Core gateway propagation.",
            may_be_empty=True,
        ),
        signals=[
            ExternalSignal(
                id="S_core_model_gateway",
                cycle_id="pending",
                signal_kind=SignalKind.PASSIVE,
                source_type="user_feedback",
                source="user",
                raw_content="No keyword cue.",
            )
        ],
    )

    assert result.evidence_events[0].evidence_type == EvidenceType.BOUNDARY_CONDITION
    assert gateway.requests[0].input["signal_id"] == "S_core_model_gateway"


def test_stale_initial_targets_fall_back_to_all_hypotheses():
    core = BayesProbeCore()
    cycle = CycleRecord(
        cycle_id="cycle_target_stale_initial",
        run_id="run_1",
        cycle_index=2,
        signal_shape=CycleSignalShape.PASSIVE_ONLY,
    )

    result = core.integrate_cycle(
        cycle=cycle,
        belief_state=make_belief_state(cycle_id="cycle_target_stale_initial", cycle_index=1),
        probe_set=ProbeSet(
            probe_set_id="ps_target_stale_initial",
            cycle_id="cycle_target_stale_initial",
            probes=[],
            selection_reason="Stale explicit targets should fall back.",
            may_be_empty=True,
        ),
        signals=[
            ExternalSignal(
                id="S_target_stale_initial",
                cycle_id="pending",
                signal_kind=SignalKind.PASSIVE,
                source_type="user_feedback",
                source="user",
                raw_content="SUPPORTS: Evidence should fall back to all hypotheses.",
                initial_target_hypotheses=["H_old"],
            )
        ],
    )

    event = result.evidence_events[0]

    assert event.target_hypotheses == ["H1", "H2"]
    assert set(event.likelihoods) == {"H1", "H2"}
    assert len(result.belief_updates) == 2


def test_stale_probe_targets_fall_back_to_all_hypotheses():
    core = BayesProbeCore()
    cycle = CycleRecord(
        cycle_id="cycle_target_stale_probe",
        run_id="run_1",
        cycle_index=2,
        signal_shape=CycleSignalShape.ACTIVE_ONLY,
    )

    result = core.integrate_cycle(
        cycle=cycle,
        belief_state=make_belief_state(cycle_id="cycle_target_stale_probe", cycle_index=1),
        probe_set=ProbeSet(
            probe_set_id="ps_target_stale_probe",
            cycle_id="cycle_target_stale_probe",
            probes=[
                ProbeDesign(
                    id="P_target_stale",
                    cycle_id="cycle_target_stale_probe",
                    target_hypotheses=["H_old"],
                    inquiry_goal="Stale target probe.",
                    method="text_probe",
                )
            ],
            selection_reason="Stale probe targets should fall back.",
            may_be_empty=False,
        ),
        signals=[
            ExternalSignal(
                id="S_target_stale_probe",
                cycle_id="pending",
                signal_kind=SignalKind.ACTIVE,
                source_type="benchmark_stream",
                source="fixture",
                raw_content="REFUTES: Evidence should fall back to all hypotheses.",
                generated_by_probe="P_target_stale",
            )
        ],
    )

    event = result.evidence_events[0]

    assert event.target_hypotheses == ["H1", "H2"]
    assert set(event.likelihoods) == {"H1", "H2"}
    assert len(result.belief_updates) == 2


def test_active_and_passive_shapes_use_same_evidence_gate():
    core = BayesProbeCore()
    active_cycle = CycleRecord(
        cycle_id="cycle_4",
        run_id="run_1",
        cycle_index=4,
        signal_shape=CycleSignalShape.ACTIVE_ONLY,
    )
    passive_cycle = CycleRecord(
        cycle_id="cycle_5",
        run_id="run_1",
        cycle_index=5,
        signal_shape=CycleSignalShape.PASSIVE_ONLY,
    )
    active_signal = ExternalSignal(
        id="S4",
        cycle_id="cycle_4",
        signal_kind=SignalKind.ACTIVE,
        source_type="benchmark_stream",
        source="fixture",
        raw_content="SUPPORTS: A sentence supports the claim.",
    )
    passive_signal = ExternalSignal(
        id="S5",
        cycle_id="cycle_5",
        signal_kind=SignalKind.PASSIVE,
        source_type="benchmark_stream",
        source="fixture",
        raw_content="SUPPORTS: A sentence supports the claim.",
    )

    active_result = core.integrate_cycle(
        cycle=active_cycle,
        belief_state=make_belief_state(cycle_id="cycle_4", cycle_index=3),
        probe_set=ProbeSet(
            probe_set_id="ps_4",
            cycle_id="cycle_4",
            probes=[],
            selection_reason="Active fixture.",
            may_be_empty=True,
        ),
        signals=[active_signal],
    )
    passive_result = core.integrate_cycle(
        cycle=passive_cycle,
        belief_state=make_belief_state(cycle_id="cycle_5", cycle_index=4),
        probe_set=ProbeSet(
            probe_set_id="ps_5",
            cycle_id="cycle_5",
            probes=[],
            selection_reason="Passive fixture.",
            may_be_empty=True,
        ),
        signals=[passive_signal],
    )

    assert active_result.evidence_events[0].evidence_type == passive_result.evidence_events[0].evidence_type
    assert active_result.evidence_events[0].likelihoods == passive_result.evidence_events[0].likelihoods


def test_cross_run_belief_state_is_rejected():
    core = BayesProbeCore()
    cycle = CycleRecord(
        cycle_id="cycle_6",
        run_id="run_1",
        cycle_index=6,
        signal_shape=CycleSignalShape.ACTIVE_ONLY,
    )

    with pytest.raises(ValueError, match="current run"):
        core.integrate_cycle(
            cycle=cycle,
            belief_state=make_belief_state(cycle_id="cycle_5").model_copy(update={"run_id": "run_2"}),
            probe_set=ProbeSet(
                probe_set_id="ps_6",
                cycle_id="cycle_6",
                probes=[],
                selection_reason="Cross-run boundary.",
                may_be_empty=True,
            ),
            signals=[],
        )


def test_generated_record_ids_are_unique_across_runs():
    core = BayesProbeCore()
    signal = ExternalSignal(
        id="S_unique",
        cycle_id="pending",
        signal_kind=SignalKind.ACTIVE,
        source_type="benchmark_stream",
        source="fixture",
        raw_content="SUPPORTS: Shared fixture signal.",
    )

    run_1_result = core.integrate_cycle(
        cycle=CycleRecord(
            cycle_id="cycle_1",
            run_id="run_1",
            cycle_index=1,
            signal_shape=CycleSignalShape.ACTIVE_ONLY,
        ),
        belief_state=make_belief_state(cycle_id="cycle_0"),
        probe_set=ProbeSet(
            probe_set_id="ps_run_1",
            cycle_id="cycle_1",
            probes=[],
            selection_reason="Run 1 fixture.",
            may_be_empty=True,
        ),
        signals=[signal],
    )
    run_2_result = core.integrate_cycle(
        cycle=CycleRecord(
            cycle_id="cycle_1",
            run_id="run_2",
            cycle_index=1,
            signal_shape=CycleSignalShape.ACTIVE_ONLY,
        ),
        belief_state=make_belief_state(cycle_id="cycle_0").model_copy(update={"run_id": "run_2"}),
        probe_set=ProbeSet(
            probe_set_id="ps_run_2",
            cycle_id="cycle_1",
            probes=[],
            selection_reason="Run 2 fixture.",
            may_be_empty=True,
        ),
        signals=[signal.model_copy(update={"id": "S_unique_run_2"})],
    )

    assert run_1_result.evidence_events[0].id == "run_1_cycle_1_E1"
    assert run_2_result.evidence_events[0].id == "run_2_cycle_1_E1"
    assert run_1_result.belief_updates[0].update_id != run_2_result.belief_updates[0].update_id


def test_existing_ledger_refs_are_preserved_and_appended():
    core = BayesProbeCore()
    cycle = CycleRecord(
        cycle_id="cycle_7",
        run_id="run_1",
        cycle_index=7,
        signal_shape=CycleSignalShape.PASSIVE_ONLY,
    )
    belief_state = make_belief_state(cycle_id="cycle_6").model_copy(
        update={
            "ledger_refs": {
                "probe_sets": ["ps_prior"],
                "evidence_events": ["E_prior"],
                "belief_updates": ["U_prior"],
                "hypothesis_evolutions": ["HE_prior"],
                "custom_audit": ["keep_me"],
            }
        }
    )

    result = core.integrate_cycle(
        cycle=cycle,
        belief_state=belief_state,
        probe_set=ProbeSet(
            probe_set_id="ps_7",
            cycle_id="cycle_7",
            probes=[],
            selection_reason="Append ledger refs.",
            may_be_empty=True,
        ),
        signals=[
            ExternalSignal(
                id="S_ledger_refs",
                cycle_id="pending",
                signal_kind=SignalKind.PASSIVE,
                source_type="system_log",
                source="fixture",
                raw_content="NEUTRAL: Preserve and append ledger references.",
            )
        ],
    )

    assert result.belief_state.ledger_refs["probe_sets"] == ["ps_prior", "ps_7"]
    assert result.belief_state.ledger_refs["evidence_events"] == [
        "E_prior",
        "run_1_cycle_7_E1",
    ]
    assert result.belief_state.ledger_refs["belief_updates"] == [
        "U_prior",
        "run_1_cycle_7_U1_H1",
        "run_1_cycle_7_U1_H2",
    ]
    assert result.belief_state.ledger_refs["hypothesis_evolutions"] == ["HE_prior"]
    assert result.belief_state.ledger_refs["custom_audit"] == ["keep_me"]


class StaticEventGate(EvidenceIntegrationGate):
    def __init__(self, events: list[EvidenceEvent]) -> None:
        self.events = events
        self.seen_framing_methods: list[FramingMethod] = []

    def integrate(self, *, cycle, belief_state, probe_set, signals):
        assert belief_state.task_frame is not None
        self.seen_framing_methods.append(belief_state.task_frame.framing_method)
        return list(self.events)


class StaticEventCore(BayesProbeCore):
    def __init__(
        self,
        events: list[EvidenceEvent],
        *,
        ledger: JsonlLedgerStore | None = None,
    ) -> None:
        self.static_gate = StaticEventGate(events)
        super().__init__(ledger=ledger)

    def _create_evidence_integration_gate(self):
        return self.static_gate


class InvalidMemoryGate(EvidenceIntegrationGate):
    def integrate(self, *, cycle, belief_state, probe_set, signals):
        return EvidenceIntegrationResult(
            evidence_events=[],
            probe_candidates=[],
            evidence_memory=EvidenceMemorySnapshot.model_construct(memory_version=0),
            normalized_signals=list(signals),
        )


class InvalidMemoryCore(BayesProbeCore):
    def _create_evidence_integration_gate(self):
        return InvalidMemoryGate()


def test_invalid_committed_memory_fails_before_any_cycle_ledger_append(tmp_path: Path):
    ledger = JsonlLedgerStore(tmp_path / "invalid-memory-ledger.jsonl")
    core = InvalidMemoryCore(ledger=ledger)

    with pytest.raises(ValueError, match="final belief state failed recursive validation"):
        core.integrate_cycle(
            cycle=make_cycle("cycle_invalid_memory"),
            belief_state=make_exact_belief_state(),
            probe_set=make_empty_probe_set("cycle_invalid_memory"),
            signals=[make_active_signal()],
        )

    assert ledger.read_all() == []


class TrackingRetirementEngine(HypothesisEvolutionEngine):
    def __init__(self) -> None:
        super().__init__()
        self.public_retirement_calls = 0

    def retire_stale_hypotheses(self, **kwargs):
        self.public_retirement_calls += 1
        return super().retire_stale_hypotheses(**kwargs)

    def _retire_stale_hypotheses(self, **kwargs):
        raise AssertionError("core must not call private retirement helpers")


class TrackingRetirementCore(StaticEventCore):
    def _create_hypothesis_evolution_policy(self):
        self.tracking_retirement_engine = TrackingRetirementEngine()
        return self.tracking_retirement_engine


def test_core_integrates_named_and_unresolved_mass_atomically(tmp_path: Path):
    event = EvidenceEvent(
        id="E_open",
        derived_from_signal="S_open",
        target_hypotheses=["H1", "H2"],
        evidence_type=EvidenceType.COUNTEREVIDENCE,
        content="Both named candidates fail the observed constraint.",
        reliability=1.0,
        independence=1.0,
        relevance=1.0,
        novelty=1.0,
        likelihoods={
            "H1": LikelihoodBand.MODERATELY_DISCONFIRMING,
            "H2": LikelihoodBand.MODERATELY_DISCONFIRMING,
        },
        unresolved_likelihood=LikelihoodBand.MODERATELY_CONFIRMING,
        frame_fit=FrameFit.SUPPORTS_UNRESOLVED,
        effective_update_weight=1.0,
        epistemic_origin="model_reasoning",
        derivation_root_id="model-root",
    )
    ledger = JsonlLedgerStore(tmp_path / "open-core-ledger.jsonl")
    core = StaticEventCore([event], ledger=ledger)

    result = core.integrate_cycle(
        cycle=make_cycle("cycle_open"),
        belief_state=make_exact_belief_state(),
        probe_set=make_empty_probe_set("cycle_open"),
        signals=[make_active_signal("pending")],
    )

    active_mass = sum(
        hypothesis.posterior
        for hypothesis in result.belief_state.hypotheses
        if hypothesis.id in result.belief_state.frame_state.active_hypothesis_ids
    )
    unresolved = result.belief_state.frame_state.unresolved_alternative_mass
    assert active_mass + unresolved == pytest.approx(1.0)
    assert unresolved > 0.50
    assert len(result.frame_mass_updates) == 1
    assert result.frame_mass_updates[0].evidence_id == event.id
    assert result.belief_state.frame_state.adequacy_status == (
        FrameAdequacyStatus.CHALLENGED
    )
    assert result.belief_state.posterior_summary["named_active_mass"] == active_mass
    assert result.belief_state.posterior_summary[
        "unresolved_alternative_mass"
    ] == unresolved
    assert result.belief_state.posterior_summary["frame_adequacy"] == "challenged"

    ordered_types = [
        record["record_type"]
        for record in ledger.read_all()
        if record["record_type"]
        in {
            "external_signal",
            "evidence_event",
            "belief_update",
            "frame_mass_update",
            "frame_adequacy_decision",
            "hypothesis_evolution",
            "probe_candidate",
            "belief_state",
        }
    ]
    assert ordered_types == [
        "external_signal",
        "evidence_event",
        "belief_update",
        "belief_update",
        "frame_mass_update",
        "frame_adequacy_decision",
        "belief_state",
    ]


def test_core_retires_open_hypothesis_with_audited_mass_transfer(tmp_path: Path):
    events = [
        EvidenceEvent(
            id=f"E_open_retire_{index}",
            derived_from_signal=f"S_open_retire_{index}",
            target_hypotheses=["H2"],
            evidence_type=EvidenceType.COUNTEREVIDENCE,
            content=f"Independent constraint {index} excludes H2.",
            reliability=1.0,
            independence=1.0,
            relevance=1.0,
            novelty=1.0,
            likelihoods={"H2": LikelihoodBand.STRONGLY_DISCONFIRMING},
            unresolved_likelihood=LikelihoodBand.NEUTRAL,
            effective_update_weight=1.0,
            epistemic_origin="tool_result",
            derivation_root_id=f"retirement-root-{index}",
        )
        for index in (1, 2)
    ]
    ledger = JsonlLedgerStore(tmp_path / "open-retirement-ledger.jsonl")
    core = StaticEventCore(events, ledger=ledger)

    result = core.integrate_cycle(
        cycle=make_cycle("cycle_open_retirement"),
        belief_state=make_exact_belief_state(),
        probe_set=make_empty_probe_set("cycle_open_retirement"),
        signals=[make_active_signal("pending")],
    )

    hypotheses = result.belief_state.hypotheses_by_id()
    frame_state = result.belief_state.frame_state
    retired = hypotheses["H2"]
    assert retired.status == HypothesisStatus.RETIRED
    assert "H2" not in frame_state.active_hypothesis_ids
    named_active_mass = sum(
        hypotheses[hypothesis_id].posterior
        for hypothesis_id in frame_state.active_hypothesis_ids
    )
    assert named_active_mass + frame_state.unresolved_alternative_mass == 1.0

    retirement_updates = [
        update
        for update in result.frame_mass_updates
        if update.update_id
        == "run_1_cycle_open_retirement_FM_retire_H2"
    ]
    assert len(retirement_updates) == 1
    retirement_update = retirement_updates[0]
    assert retirement_update.evidence_id == events[-1].id
    assert retirement_update.posterior - retirement_update.prior == pytest.approx(
        retired.posterior
    )
    assert "retirement-root-2" in retirement_update.reason
    assert all(update.hypothesis_id != "unresolved" for update in result.belief_updates)
    assert result.frame_adequacy_decision.frame_state == frame_state
    assert [evolution.operation for evolution in result.hypothesis_evolutions] == [
        EvolutionOperation.RETIRE
    ]

    ordered_types = [
        record["record_type"]
        for record in ledger.read_all()
        if record["record_type"]
        in {
            "external_signal",
            "evidence_event",
            "belief_update",
            "frame_mass_update",
            "frame_adequacy_decision",
            "hypothesis_evolution",
            "probe_candidate",
            "belief_state",
        }
    ]
    assert ordered_types == [
        "external_signal",
        "evidence_event",
        "evidence_event",
        "belief_update",
        "belief_update",
        "belief_update",
        "belief_update",
        "frame_mass_update",
        "frame_mass_update",
        "frame_mass_update",
        "frame_adequacy_decision",
        "hypothesis_evolution",
        "belief_state",
    ]


def test_core_retires_every_open_hypothesis_into_a_challenged_valid_frame(
    tmp_path: Path,
):
    events = [
        EvidenceEvent(
            id=f"E_open_retire_all_{index}",
            derived_from_signal=f"S_open_retire_all_{index}",
            target_hypotheses=["H1", "H2"],
            evidence_type=EvidenceType.COUNTEREVIDENCE,
            content=f"Independent constraint {index} excludes every named candidate.",
            reliability=1.0,
            independence=1.0,
            relevance=1.0,
            novelty=1.0,
            likelihoods={
                "H1": LikelihoodBand.STRONGLY_DISCONFIRMING,
                "H2": LikelihoodBand.STRONGLY_DISCONFIRMING,
            },
            unresolved_likelihood=LikelihoodBand.NEUTRAL,
            effective_update_weight=1.0,
            epistemic_origin="tool_result",
            derivation_root_id=f"retire-all-root-{index}",
        )
        for index in (1, 2)
    ]
    ledger = JsonlLedgerStore(tmp_path / "open-retire-all-ledger.jsonl")
    core = TrackingRetirementCore(events, ledger=ledger)

    result = core.integrate_cycle(
        cycle=make_cycle("cycle_open_retire_all"),
        belief_state=make_exact_belief_state(),
        probe_set=make_empty_probe_set("cycle_open_retire_all"),
        signals=[make_active_signal("pending")],
    )

    assert core.tracking_retirement_engine.public_retirement_calls == 1
    assert result.belief_state.frame_state.active_hypothesis_ids == []
    assert result.belief_state.frame_state.unresolved_alternative_mass == 1.0
    assert {
        hypothesis.status for hypothesis in result.belief_state.hypotheses
    } == {HypothesisStatus.RETIRED}
    assert BeliefState.model_validate(
        result.belief_state.model_dump(mode="python")
    ) == result.belief_state
    assert [
        evolution.from_hypothesis
        for evolution in result.hypothesis_evolutions
        if evolution.operation == EvolutionOperation.RETIRE
    ] == ["H1", "H2"]
    retirement_updates = [
        update
        for update in result.frame_mass_updates
        if "_FM_retire_" in update.update_id
    ]
    assert [update.update_id for update in retirement_updates] == [
        "run_1_cycle_open_retire_all_FM_retire_H1",
        "run_1_cycle_open_retire_all_FM_retire_H2",
    ]
    assert sum(
        update.posterior - update.prior for update in retirement_updates
    ) == pytest.approx(sum(h.posterior for h in result.belief_state.hypotheses))
    assert result.frame_adequacy_decision.frame_state.adequacy_status == (
        FrameAdequacyStatus.CHALLENGED
    )
    assert result.frame_adequacy_decision.should_expand is True
    assert result.frame_adequacy_decision.trigger_event_ids == [
        "E_open_retire_all_1",
        "E_open_retire_all_2",
    ]
    assert result.frame_adequacy_decision.reason == (
        "All named hypotheses are retired; unresolved alternatives hold all frame mass."
    )

    ordered_types = [
        record["record_type"]
        for record in ledger.read_all()
        if record["record_type"]
        in {
            "frame_mass_update",
            "frame_adequacy_decision",
            "hypothesis_evolution",
            "belief_state",
        }
    ]
    last_mass_update = max(
        index
        for index, record_type in enumerate(ordered_types)
        if record_type == "frame_mass_update"
    )
    adequacy_decision = ordered_types.index("frame_adequacy_decision")
    first_evolution = ordered_types.index("hypothesis_evolution")
    assert last_mass_update < adequacy_decision < first_evolution
    assert ordered_types[-1] == "belief_state"


def test_core_does_not_retire_an_already_retired_hypothesis_twice(tmp_path: Path):
    first_cycle_events = [
        EvidenceEvent(
            id=f"E_retire_once_{index}",
            derived_from_signal=f"S_retire_once_{index}",
            target_hypotheses=["H2"],
            evidence_type=EvidenceType.COUNTEREVIDENCE,
            content=f"Independent constraint {index} excludes H2.",
            reliability=1.0,
            independence=1.0,
            relevance=1.0,
            novelty=1.0,
            likelihoods={"H2": LikelihoodBand.STRONGLY_DISCONFIRMING},
            unresolved_likelihood=LikelihoodBand.NEUTRAL,
            effective_update_weight=1.0,
            epistemic_origin="tool_result",
            derivation_root_id=f"retire-once-root-{index}",
        )
        for index in (1, 2)
    ]
    ledger = JsonlLedgerStore(tmp_path / "retirement-idempotence-ledger.jsonl")
    core = StaticEventCore(first_cycle_events, ledger=ledger)
    first = core.integrate_cycle(
        cycle=make_cycle("cycle_retire_once"),
        belief_state=make_exact_belief_state(),
        probe_set=make_empty_probe_set("cycle_retire_once"),
        signals=[make_active_signal("pending")],
    )
    unresolved_after_retirement = (
        first.belief_state.frame_state.unresolved_alternative_mass
    )
    frame_mass_record_count = len(ledger.read_all("frame_mass_update"))

    core.static_gate.events = [
        event.model_copy(
            update={
                "id": f"E_retired_target_{index}",
                "derived_from_signal": f"S_retired_target_{index}",
                "derivation_root_id": f"retired-target-root-{index}",
            }
        )
        for index, event in enumerate(first_cycle_events, start=1)
    ]
    second_cycle = make_cycle("cycle_retired_target").model_copy(
        update={"cycle_index": 2}
    )
    second = core.integrate_cycle(
        cycle=second_cycle,
        belief_state=first.belief_state,
        probe_set=make_empty_probe_set("cycle_retired_target"),
        signals=[make_active_signal("pending")],
    )

    assert second.belief_state.hypotheses_by_id()["H2"].status == (
        HypothesisStatus.RETIRED
    )
    assert second.belief_state.frame_state.unresolved_alternative_mass == (
        unresolved_after_retirement
    )
    assert second.frame_mass_updates == []
    assert second.hypothesis_evolutions == []
    assert len(ledger.read_all("hypothesis_evolution")) == 1
    assert len(ledger.read_all("frame_mass_update")) == frame_mass_record_count
    assert [
        record
        for record in ledger.read_all("frame_mass_update")
        if record["payload"]["cycle_id"] == second_cycle.cycle_id
    ] == []
    assert len(
        [
            record
            for record in ledger.read_all("frame_mass_update")
            if "_FM_retire_" in record["payload"]["update_id"]
        ]
    ) == 1


def test_core_rejects_non_open_all_retired_state_before_cycle_ledger_append(
    tmp_path: Path,
):
    state_payload = make_exact_belief_state().model_dump(mode="python")
    state_payload["task_frame"]["task_kind"] = "claim_verification"
    state_payload["task_frame"]["hypothesis_frame"].update(
        {
            "competition": "independent",
            "coverage": "open",
            "rival_sets": {"H1": [], "H2": []},
            "unresolved_alternative_mass": None,
            "coverage_limitation": None,
        }
    )
    state_payload["frame_state"].update(
        {
            "competition": "independent",
            "coverage": "open",
            "unresolved_alternative_mass": None,
        }
    )
    for hypothesis in state_payload["hypotheses"]:
        hypothesis["rivals"] = []
    belief_state = BeliefState.model_validate(state_payload)
    events = [
        EvidenceEvent(
            id=f"E_retire_independent_{index}",
            derived_from_signal=f"S_retire_independent_{index}",
            target_hypotheses=["H1", "H2"],
            evidence_type=EvidenceType.COUNTEREVIDENCE,
            content=f"Independent constraint {index} excludes both claims.",
            reliability=1.0,
            independence=1.0,
            relevance=1.0,
            novelty=1.0,
            likelihoods={
                "H1": LikelihoodBand.STRONGLY_DISCONFIRMING,
                "H2": LikelihoodBand.STRONGLY_DISCONFIRMING,
            },
            effective_update_weight=1.0,
        )
        for index in (1, 2)
    ]
    ledger = JsonlLedgerStore(tmp_path / "invalid-all-retired-ledger.jsonl")
    core = StaticEventCore(events, ledger=ledger)

    with pytest.raises(
        ValueError,
        match="final belief state failed recursive validation",
    ):
        core.integrate_cycle(
            cycle=make_cycle("cycle_invalid_all_retired"),
            belief_state=belief_state,
            probe_set=make_empty_probe_set("cycle_invalid_all_retired"),
            signals=[make_active_signal("pending")],
        )

    assert ledger.read_all() == []


def test_core_open_duplicate_event_moves_frame_mass_once(tmp_path: Path):
    event = EvidenceEvent(
        id="E_open_duplicate",
        derived_from_signal="S_open_duplicate",
        target_hypotheses=["H1"],
        evidence_type=EvidenceType.SUPPORTING,
        content="A repeated open-frame event.",
        likelihoods={"H1": LikelihoodBand.MODERATELY_CONFIRMING},
        unresolved_likelihood=LikelihoodBand.NEUTRAL,
        effective_update_weight=1.0,
    )
    ledger = JsonlLedgerStore(tmp_path / "open-duplicate-ledger.jsonl")
    core = StaticEventCore([event, event], ledger=ledger)

    result = core.integrate_cycle(
        cycle=make_cycle("cycle_open_duplicate"),
        belief_state=make_exact_belief_state(),
        probe_set=make_empty_probe_set("cycle_open_duplicate"),
        signals=[make_active_signal("pending")],
    )

    assert len(result.frame_mass_updates) == 1
    assert {update.evidence_id for update in result.belief_updates} == {event.id}
    assert [
        record["payload"]["id"]
        for record in ledger.read_all("evidence_event")
    ] == [event.id]


def test_core_marks_past_evidence_replay_without_duplicate_canonical_record(
    tmp_path: Path,
):
    replay = EvidenceEvent(
        id="E_replay",
        derived_from_signal="S_replay",
        target_hypotheses=["H1"],
        evidence_type=EvidenceType.SUPPORTING,
        content="Replay fixture.",
        likelihoods={"H1": LikelihoodBand.STRONGLY_CONFIRMING},
    )
    ledger = JsonlLedgerStore(tmp_path / "past-replay-ledger.jsonl")
    ledger.append("evidence_event", replay)
    core = StaticEventCore([replay], ledger=ledger)
    belief_state = make_belief_state(cycle_id="cycle_0").model_copy(
        update={
            "hypotheses": [
                item.model_copy(update={"posterior": 1.0 / 3.0})
                for item in make_belief_state().hypotheses
            ],
            "ledger_refs": {"evidence_events": ["E_replay"]},
        }
    )

    result = core.integrate_cycle(
        cycle=make_cycle("cycle_replay"),
        belief_state=belief_state,
        probe_set=make_empty_probe_set("cycle_replay"),
        signals=[make_active_signal("pending")],
    )

    assert core.static_gate.seen_framing_methods == [FramingMethod.LEGACY_MIGRATION]
    assert result.evidence_events[0].discard_reason == "duplicate evidence event id"
    assert result.belief_state.hypotheses == belief_state.hypotheses
    assert result.belief_updates == []
    assert result.hypothesis_evolutions == []
    assert result.belief_state.ledger_refs["evidence_events"] == ["E_replay"]
    evidence_records = ledger.read_all("evidence_event")
    assert [record["payload"]["id"] for record in evidence_records] == ["E_replay"]
    assert [record["payload"]["discard_reason"] for record in evidence_records] == [
        None
    ]


def test_core_persists_same_cycle_duplicate_identity_once(tmp_path: Path):
    duplicate = EvidenceEvent(
        id="E_same_cycle",
        derived_from_signal="S_same_cycle",
        target_hypotheses=["H1"],
        evidence_type=EvidenceType.SUPPORTING,
        content="Same-cycle duplicate fixture.",
        likelihoods={"H1": LikelihoodBand.MODERATELY_CONFIRMING},
    )
    ledger = JsonlLedgerStore(tmp_path / "same-cycle-replay-ledger.jsonl")
    core = StaticEventCore([duplicate, duplicate], ledger=ledger)

    result = core.integrate_cycle(
        cycle=make_cycle("cycle_same_id"),
        belief_state=make_belief_state(cycle_id="cycle_0"),
        probe_set=make_empty_probe_set("cycle_same_id"),
        signals=[make_active_signal("pending")],
    )

    assert len(result.evidence_events) == 2
    assert result.evidence_events[0].discard_reason is None
    assert result.evidence_events[1].discard_reason == "duplicate evidence event id"
    assert {update.evidence_id for update in result.belief_updates} == {
        "E_same_cycle"
    }
    assert len(result.belief_updates) == 2
    assert result.belief_state.ledger_refs["evidence_events"] == ["E_same_cycle"]
    evidence_records = ledger.read_all("evidence_event")
    assert [record["payload"]["id"] for record in evidence_records] == [
        "E_same_cycle"
    ]
    assert [record["payload"]["discard_reason"] for record in evidence_records] == [
        None
    ]


def test_core_persists_discarded_first_occurrence_as_canonical_audit(
    tmp_path: Path,
):
    discarded = EvidenceEvent(
        id="E_gate_discarded",
        derived_from_signal="S_gate_discarded",
        target_hypotheses=["H1"],
        evidence_type=EvidenceType.SUPPORTING,
        content="Gate-discarded fixture.",
        likelihoods={"H1": LikelihoodBand.STRONGLY_CONFIRMING},
        discard_reason="schema_violation: invalid judgment",
    )
    ledger = JsonlLedgerStore(tmp_path / "discarded-canonical-ledger.jsonl")
    core = StaticEventCore([discarded, discarded], ledger=ledger)

    result = core.integrate_cycle(
        cycle=make_cycle("cycle_gate_discarded"),
        belief_state=make_belief_state(cycle_id="cycle_0"),
        probe_set=make_empty_probe_set("cycle_gate_discarded"),
        signals=[make_active_signal("pending")],
    )

    assert result.evidence_events == [discarded, discarded]
    assert result.belief_updates == []
    assert result.belief_state.ledger_refs["evidence_events"] == [
        "E_gate_discarded"
    ]
    evidence_records = ledger.read_all("evidence_event")
    assert [record["payload"]["id"] for record in evidence_records] == [
        "E_gate_discarded"
    ]
    assert [record["payload"]["discard_reason"] for record in evidence_records] == [
        "schema_violation: invalid judgment"
    ]


def test_future_cycle_belief_state_is_rejected():
    core = BayesProbeCore()
    cycle = CycleRecord(
        cycle_id="current_cycle",
        run_id="run_1",
        cycle_index=5,
        signal_shape=CycleSignalShape.ACTIVE_ONLY,
    )

    with pytest.raises(ValueError, match="future cycle"):
        core.integrate_cycle(
            cycle=cycle,
            belief_state=make_belief_state(cycle_id="future_cycle", cycle_index=6),
            probe_set=ProbeSet(
                probe_set_id="ps_5",
                cycle_id="current_cycle",
                probes=[],
                selection_reason="Future cycle state.",
                may_be_empty=True,
            ),
            signals=[],
        )


def test_external_projection_decomposes_source_claim_and_generates_verification_probe():
    core = BayesProbeCore()
    cycle = CycleRecord(
        cycle_id="cycle_projection",
        run_id="run_1",
        cycle_index=1,
        signal_shape=CycleSignalShape.PASSIVE_ONLY,
    )

    result = core.integrate_cycle(
        cycle=cycle,
        belief_state=make_belief_state(cycle_id="cycle_0"),
        probe_set=ProbeSet(
            probe_set_id="ps_projection",
            cycle_id="cycle_projection",
            probes=[],
            selection_reason="Projection decomposition fixture.",
            may_be_empty=True,
        ),
        signals=[
            ExternalSignal(
                id="S_projection",
                cycle_id="pending",
                signal_kind=SignalKind.PASSIVE,
                source_type="external_agent_projection",
                source="agent_a",
                raw_content="Agent A believes H2 because Source X refutes H1.",
                initial_target_hypotheses=["H1", "H2"],
            )
        ],
    )

    sender_event = result.evidence_events[0]
    source_event = result.evidence_events[1]
    candidate = result.probe_candidates[0]

    assert [event.evidence_type for event in result.evidence_events] == [
        EvidenceType.SENDER_JUDGMENT,
        EvidenceType.SOURCE_CLAIM,
    ]
    assert sender_event.id == "run_1_cycle_projection_E1"
    assert source_event.id == "run_1_cycle_projection_E1_source"
    assert sender_event.likelihoods["H2"] == LikelihoodBand.WEAKLY_CONFIRMING
    assert sender_event.likelihoods["H1"] == LikelihoodBand.NEUTRAL
    assert set(source_event.likelihoods.values()) == {LikelihoodBand.NEUTRAL}
    assert sender_event.reliability == 0.55
    assert sender_event.independence == 0.45
    assert source_event.reliability == 0.5
    assert source_event.verifiability == 0.65
    assert candidate.candidate_id == "pc_run_1_cycle_projection_E1_source_verify_source"
    assert candidate.source == "passive_signal"
    assert candidate.candidate_probe.id == "P_run_1_cycle_projection_E1_source_verify_source"
    assert candidate.candidate_probe.method == "source_tracing"
    assert candidate.candidate_probe.cycle_id == "cycle_projection"
    assert candidate.candidate_probe.target_hypotheses == ["H1", "H2"]


def test_low_reliability_signal_caps_quality_scores():
    core = BayesProbeCore()
    cycle = CycleRecord(
        cycle_id="cycle_low_quality",
        run_id="run_1",
        cycle_index=1,
        signal_shape=CycleSignalShape.ACTIVE_ONLY,
    )

    result = core.integrate_cycle(
        cycle=cycle,
        belief_state=make_belief_state(cycle_id="cycle_0"),
        probe_set=ProbeSet(
            probe_set_id="ps_low_quality",
            cycle_id="cycle_low_quality",
            probes=[],
            selection_reason="Low reliability quality fixture.",
            may_be_empty=True,
        ),
        signals=[
            ExternalSignal(
                id="S_low_quality",
                cycle_id="pending",
                signal_kind=SignalKind.ACTIVE,
                source_type="benchmark_stream",
                source="fixture",
                raw_content="SUPPORTS: This is an unverified rumor, maybe unclear.",
            )
        ],
    )

    event = result.evidence_events[0]
    assert event.evidence_type == EvidenceType.SUPPORTING
    assert event.reliability == 0.35
    assert event.verifiability == 0.4
    assert event.relevance == 0.9


def test_model_probe_signal_uses_conservative_quality_baseline():
    core = BayesProbeCore()
    cycle = CycleRecord(
        cycle_id="cycle_model_probe",
        run_id="run_1",
        cycle_index=1,
        signal_shape=CycleSignalShape.ACTIVE_ONLY,
    )

    result = core.integrate_cycle(
        cycle=cycle,
        belief_state=make_belief_state(cycle_id="cycle_0"),
        probe_set=ProbeSet(
            probe_set_id="ps_model_probe",
            cycle_id="cycle_model_probe",
            probes=[],
            selection_reason="Model probe quality fixture.",
            may_be_empty=True,
        ),
        signals=[
            ExternalSignal(
                id="S_model_probe",
                cycle_id="pending",
                signal_kind=SignalKind.ACTIVE,
                source_type="model_probe_gateway",
                source="model_gateway:scripted",
                raw_content="SUPPORTS: Internal model reasoning favors H1.",
            )
        ],
    )

    event = result.evidence_events[0]
    assert event.reliability == 0.55
    assert event.independence == 0.35
    assert event.relevance == 0.85
    assert event.novelty == 0.55
    assert event.specificity == 0.65
    assert event.verifiability == 0.3


def test_model_probe_quality_overrides_cannot_inflate_its_baseline():
    gateway = ScriptedModelGateway(
        responses={
            "judge_evidence": {
                "evidence_type": "supporting",
                "likelihoods": {
                    "H1": "moderately_confirming",
                    "H2": "moderately_disconfirming",
                },
                "interpretation": "Self-judged model probe fixture.",
                "quality_overrides": {
                    "reliability": 1.0,
                    "independence": 1.0,
                    "relevance": 1.0,
                    "novelty": 1.0,
                    "specificity": 1.0,
                    "verifiability": 1.0,
                },
            }
        }
    )
    core = BayesProbeCore(model_gateway=gateway)
    cycle = make_cycle("cycle_model_probe_caps")

    result = core.integrate_cycle(
        cycle=cycle,
        belief_state=make_belief_state(cycle_id="cycle_0"),
        probe_set=make_empty_probe_set(cycle.cycle_id),
        signals=[
            ExternalSignal(
                id="S_model_probe_caps",
                cycle_id="pending",
                signal_kind=SignalKind.ACTIVE,
                source_type="model_probe_gateway",
                source="model_gateway:scripted",
                raw_content="SUPPORTS: Self-generated model signal.",
                initial_target_hypotheses=["H1", "H2"],
            )
        ],
    )

    event = result.evidence_events[0]
    assert event.reliability == 0.55
    assert event.independence == 0.35
    assert event.relevance == 0.85
    assert event.novelty == 0.55
    assert event.specificity == 0.65
    assert event.verifiability == 0.3


def test_duplicate_signals_downweight_later_event_independence_and_novelty():
    core = BayesProbeCore()
    cycle = CycleRecord(
        cycle_id="cycle_duplicate",
        run_id="run_1",
        cycle_index=1,
        signal_shape=CycleSignalShape.ACTIVE_ONLY,
    )
    duplicate = ExternalSignal(
        id="S_duplicate_1",
        cycle_id="pending",
        signal_kind=SignalKind.ACTIVE,
        source_type="benchmark_stream",
        source="fixture",
        raw_content="SUPPORTS: Same source content.",
    )

    result = core.integrate_cycle(
        cycle=cycle,
        belief_state=make_belief_state(cycle_id="cycle_0"),
        probe_set=ProbeSet(
            probe_set_id="ps_duplicate",
            cycle_id="cycle_duplicate",
            probes=[],
            selection_reason="Duplicate quality fixture.",
            may_be_empty=True,
        ),
        signals=[
            duplicate,
            duplicate.model_copy(update={"id": "S_duplicate_2"}),
        ],
    )

    first_event = result.evidence_events[0]
    second_event = result.evidence_events[1]
    assert first_event.independence == 0.8
    assert first_event.novelty == 0.8
    assert second_event.independence == 0.25
    assert second_event.novelty == 0.25


def _memory_signal(signal_id: str, content: str, *, root: str) -> ExternalSignal:
    return ExternalSignal(
        id=signal_id,
        cycle_id="pending",
        signal_kind=SignalKind.ACTIVE,
        source_type="retrieved_source",
        source="source.example/audit",
        raw_content=content,
        initial_target_hypotheses=["H1", "H2"],
        provenance=SignalProvenance(
            epistemic_origin=EpistemicOrigin.RETRIEVED_SOURCE,
            source_identity="source.example/audit",
            derivation_root_id=root,
            correlation_group="source.example/audit",
            canonical_content_fingerprint="normalize-me",
        ),
    )


def _native_open_judgment():
    return {
        "evidence_type": "supporting",
        "likelihoods": {
            "H1": "moderately_confirming",
            "H2": "moderately_disconfirming",
        },
        "unresolved_likelihood": "neutral",
        "frame_fit": "explained_by_named",
        "unexplained_observation": None,
        "interpretation": "The audit favors H1.",
        "quality_overrides": {},
    }


def test_core_commits_memory_once_and_ledgers_normalized_cross_cycle_duplicate(
    tmp_path: Path,
):
    gateway = ScriptedModelGateway(responses={"judge_evidence": _native_open_judgment()})
    ledger = JsonlLedgerStore(tmp_path / "memory-ledger.jsonl")
    core = BayesProbeCore(ledger=ledger, model_gateway=gateway)
    first_cycle = make_cycle("cycle_memory_1")
    first = core.integrate_cycle(
        cycle=first_cycle,
        belief_state=make_exact_belief_state(),
        probe_set=make_empty_probe_set(first_cycle.cycle_id),
        signals=[_memory_signal("S_memory_1", "The audit supports H1.", root="root-audit")],
    )
    first_credit = dict(first.belief_state.evidence_memory.correlation_credit)
    second_cycle = make_cycle("cycle_memory_2").model_copy(update={"cycle_index": 2})

    second = core.integrate_cycle(
        cycle=second_cycle,
        belief_state=first.belief_state,
        probe_set=make_empty_probe_set(second_cycle.cycle_id),
        signals=[_memory_signal("S_memory_2", "The audit supports H1.", root="root-audit")],
    )

    assert len(gateway.requests) == 1
    assert second.evidence_events[0].discard_reason == "duplicate_exact"
    assert second.evidence_events[0].effective_update_weight == 0.0
    assert second.belief_updates == []
    assert second.frame_mass_updates == []
    assert second.belief_state.evidence_memory.correlation_credit == first_credit
    assert second.belief_state.evidence_memory.accepted_evidence_ids == [
        "run_1_cycle_memory_1_E1"
    ]
    assert second.belief_state.evidence_memory.discard_and_schema_history == [
        "run_1_cycle_memory_2_E1:duplicate_exact"
    ]
    signal_records = ledger.read_all("external_signal")
    assert len(signal_records) == 2
    assert all(record["payload"]["provenance"] for record in signal_records)


def test_replayed_native_evidence_id_does_not_recommit_credit_or_ledger_record(
    tmp_path: Path,
):
    gateway = ScriptedModelGateway(responses={"judge_evidence": _native_open_judgment()})
    ledger = JsonlLedgerStore(tmp_path / "memory-replay-ledger.jsonl")
    core = BayesProbeCore(ledger=ledger, model_gateway=gateway)
    cycle = make_cycle("cycle_memory_replay")
    first = core.integrate_cycle(
        cycle=cycle,
        belief_state=make_exact_belief_state(),
        probe_set=make_empty_probe_set(cycle.cycle_id),
        signals=[_memory_signal("S_replay_1", "First audit result.", root="root-1")],
    )
    prior_memory = first.belief_state.evidence_memory

    replayed = core.integrate_cycle(
        cycle=cycle.model_copy(update={"cycle_index": 2}),
        belief_state=first.belief_state,
        probe_set=make_empty_probe_set(cycle.cycle_id),
        signals=[_memory_signal("S_replay_2", "Second audit result.", root="root-2")],
    )

    assert len(gateway.requests) == 2
    assert replayed.evidence_events[0].discard_reason == "duplicate evidence event id"
    assert replayed.belief_state.evidence_memory == prior_memory
    assert replayed.belief_updates == []
    assert [
        record["payload"]["id"] for record in ledger.read_all("evidence_event")
    ] == ["run_1_cycle_memory_replay_E1"]


def test_saturated_correlation_event_is_ledger_visible_without_mass_update(
    tmp_path: Path,
):
    gateway = ScriptedModelGateway(responses={"judge_evidence": _native_open_judgment()})
    ledger = JsonlLedgerStore(tmp_path / "saturated-memory-ledger.jsonl")
    core = BayesProbeCore(ledger=ledger, model_gateway=gateway)
    state = make_exact_belief_state()
    group = "model:model_gateway:scripted:run_1"
    state = state.model_copy(
        update={
            "evidence_memory": state.evidence_memory.model_copy(
                update={
                    "correlation_credit": {
                        f"{group}|H1|confirming": 1.0,
                        f"{group}|H2|disconfirming": 1.0,
                    }
                }
            )
        }
    )
    cycle = make_cycle("cycle_saturated")

    result = core.integrate_cycle(
        cycle=cycle,
        belief_state=state,
        probe_set=make_empty_probe_set(cycle.cycle_id),
        signals=[
            ExternalSignal(
                id="S_saturated",
                cycle_id="pending",
                signal_kind=SignalKind.ACTIVE,
                source_type="model_probe_gateway",
                source="model_gateway:scripted",
                raw_content="A fresh model restatement favors H1.",
                initial_target_hypotheses=["H1", "H2"],
            )
        ],
    )

    event = result.evidence_events[0]
    assert event.correlation_status == "correlated_novel"
    assert event.discard_reason == "correlation_credit_saturated"
    assert event.effective_update_weight == 0.0
    assert result.belief_updates == []
    assert result.frame_mass_updates == []
    assert [
        record["payload"]["id"] for record in ledger.read_all("evidence_event")
    ] == [event.id]


def test_generated_probe_candidates_are_written_to_ledger_and_belief_refs(tmp_path: Path):
    ledger = JsonlLedgerStore(tmp_path / "core-v02-ledger.jsonl")
    core = BayesProbeCore(ledger=ledger)
    cycle = CycleRecord(
        cycle_id="cycle_projection_ledger",
        run_id="run_1",
        cycle_index=1,
        signal_shape=CycleSignalShape.PASSIVE_ONLY,
    )

    result = core.integrate_cycle(
        cycle=cycle,
        belief_state=make_belief_state(cycle_id="cycle_0"),
        probe_set=ProbeSet(
            probe_set_id="ps_projection_ledger",
            cycle_id="cycle_projection_ledger",
            probes=[],
            selection_reason="Projection ledger fixture.",
            may_be_empty=True,
        ),
        signals=[
            ExternalSignal(
                id="S_projection_ledger",
                cycle_id="pending",
                signal_kind=SignalKind.PASSIVE,
                source_type="external_agent_projection",
                source="agent_a",
                raw_content="Agent A believes H2 because Source X refutes H1.",
                initial_target_hypotheses=["H1", "H2"],
            )
        ],
    )

    candidate_id = result.probe_candidates[0].candidate_id
    record_types = [record["record_type"] for record in ledger.read_all()]

    assert result.belief_state.ledger_refs["probe_candidates"] == [candidate_id]
    assert "probe_candidate" in record_types
    assert record_types.count("cycle") == 1
    assert record_types.count("probe_candidate") == 1


def test_legacy_gate_returning_plain_event_list_still_integrates():
    class PlainListGate(EvidenceIntegrationGate):
        def integrate(self, *, cycle, belief_state, probe_set, signals):
            return [
                EvidenceEvent(
                    id="legacy_E1",
                    derived_from_signal="S_legacy",
                    target_hypotheses=["H1"],
                    evidence_type=EvidenceType.SUPPORTING,
                    content="SUPPORTS: Legacy gate event.",
                    likelihoods={"H1": LikelihoodBand.WEAKLY_CONFIRMING},
                )
            ]

    class PlainListGateCore(BayesProbeCore):
        def _create_evidence_integration_gate(self):
            return PlainListGate()

    core = PlainListGateCore()
    cycle = CycleRecord(
        cycle_id="cycle_legacy",
        run_id="run_1",
        cycle_index=1,
        signal_shape=CycleSignalShape.ACTIVE_ONLY,
    )

    result = core.integrate_cycle(
        cycle=cycle,
        belief_state=make_belief_state(cycle_id="cycle_0"),
        probe_set=ProbeSet(
            probe_set_id="ps_legacy",
            cycle_id="cycle_legacy",
            probes=[],
            selection_reason="Legacy gate fixture.",
            may_be_empty=True,
        ),
        signals=[
            ExternalSignal(
                id="S_legacy",
                cycle_id="pending",
                signal_kind=SignalKind.ACTIVE,
                source_type="benchmark_stream",
                source="fixture",
                raw_content="SUPPORTS: Legacy signal.",
            )
        ],
    )

    assert result.evidence_events[0].id == "legacy_E1"
    assert result.probe_candidates == []
