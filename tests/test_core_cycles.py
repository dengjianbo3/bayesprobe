import json
from pathlib import Path
import unicodedata

import pytest

import bayesprobe.core as core_module
import bayesprobe.evidence_memory as evidence_memory
from bayesprobe.core import BayesProbeCore, EvidenceIntegrationGate
from bayesprobe.evidence import EvidenceIntegrationResult
from bayesprobe.initialization import BayesProbeInitializer, InitializeRunInput
from bayesprobe.inbox import SignalInbox
from bayesprobe.hypothesis_evolution import HypothesisEvolutionEngine
from bayesprobe.kernel_config import CorrelationCreditPolicy
from bayesprobe.ledger import JsonlLedgerStore
from bayesprobe.lifecycle import BeliefLifecycle, resolve_belief_lifecycle
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
    HypothesisCompetition,
    HypothesisCoverage,
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
    decode_discard_history_entry,
    encode_discard_history_entry,
)
from bayesprobe.task_framing import ModelTaskFramer, migrate_legacy_belief_state


_NFKC_SECRET_VALUE = (
    "\uff21\uff55\uff54\uff48\uff4f\uff52\uff49\uff5a\uff41\uff54"
    "\uff49\uff4f\uff4e\uff1a \uff22\uff45\uff41\uff52\uff45\uff52 "
    "provider-secret-value-123"
)


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


def make_explicit_legacy_belief_state(
    cycle_id: str = "cycle_1",
    cycle_index: int = 0,
) -> BeliefState:
    return migrate_legacy_belief_state(
        make_belief_state(cycle_id=cycle_id, cycle_index=cycle_index)
    )


def make_cycle(cycle_id: str = "cycle_repair") -> CycleRecord:
    return CycleRecord(
        cycle_id=cycle_id,
        run_id="run_1",
        cycle_index=1,
        signal_shape=CycleSignalShape.ACTIVE_ONLY,
    )


def legacy_judgment_route_metadata() -> dict[str, str]:
    return {
        "judgment_route": "legacy_v0.1_migration",
        "lifecycle_schema_version": "v0.2",
        "frame_competition": "exclusive",
        "frame_coverage": "exhaustive",
        "framing_method": "legacy_migration",
    }


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
                "quality_overrides": {},
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
        belief_state=make_explicit_legacy_belief_state(cycle_id="cycle_0"),
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
                "quality_overrides": {},
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
        belief_state=make_explicit_legacy_belief_state(cycle_id="cycle_0"),
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
        "quality_overrides",
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
                "quality_overrides": {},
            },
            "repair_evidence_judgment": {
                "evidence_type": "still_not_a_type",
                "likelihoods": {"H1": "neutral", "H2": "neutral"},
                "interpretation": "Still invalid.",
                "quality_overrides": {},
            },
        }
    )
    gate = EvidenceIntegrationGate(
        model_gateway=gateway,
        judgment_repair_policy=EvidenceJudgmentRepairPolicy(max_attempts=1),
    )

    result = gate.integrate(
        cycle=make_cycle("cycle_repair_failure"),
        belief_state=make_explicit_legacy_belief_state(cycle_id="cycle_0"),
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
            belief_state=make_explicit_legacy_belief_state(cycle_id="cycle_0"),
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
                "quality_overrides": {},
            }
        }
    )
    gate = EvidenceIntegrationGate(model_gateway=gateway)

    result = gate.integrate(
        cycle=make_cycle("cycle_model_trace_valid"),
        belief_state=make_explicit_legacy_belief_state(cycle_id="cycle_0"),
        probe_set=make_empty_probe_set("cycle_model_trace_valid"),
        signals=[make_active_signal()],
    )

    request = gateway.requests[0]
    event = result.evidence_events[0]
    assert request.prompt_id == "evidence_judgment"
    assert request.prompt_version == "v0.1"
    assert request.schema_name == "EvidenceJudgment"
    assert request.schema_version == "v0.1"
    assert request.metadata == legacy_judgment_route_metadata()
    assert event.model_trace == {
        "task": "judge_evidence",
        "adapter_kind": "scripted",
        "prompt_id": "evidence_judgment",
        "prompt_version": "v0.1",
        "schema_name": "EvidenceJudgment",
        "schema_version": "v0.1",
        "metadata": legacy_judgment_route_metadata(),
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
        belief_state=make_explicit_legacy_belief_state(cycle_id="cycle_0"),
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
        belief_state=make_explicit_legacy_belief_state(cycle_id="cycle_0"),
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
        "metadata": legacy_judgment_route_metadata(),
    }


def test_direct_signal_repaired_judgment_records_repair_model_trace():
    gateway = ScriptedModelGateway(
        responses={
            "judge_evidence": {
                "evidence_type": "not_a_type",
                "likelihoods": {"H1": "neutral", "H2": "neutral"},
                "interpretation": "Invalid evidence type.",
                "quality_overrides": {},
            },
            "repair_evidence_judgment": {
                "evidence_type": "supporting",
                "likelihoods": {
                    "H1": "moderately_confirming",
                    "H2": "moderately_disconfirming",
                },
                "interpretation": "Repaired supporting judgment.",
                "quality_overrides": {},
            },
        }
    )
    gate = EvidenceIntegrationGate(
        model_gateway=gateway,
        judgment_repair_policy=EvidenceJudgmentRepairPolicy(max_attempts=1),
    )

    result = gate.integrate(
        cycle=make_cycle("cycle_model_trace_repair"),
        belief_state=make_explicit_legacy_belief_state(cycle_id="cycle_0"),
        probe_set=make_empty_probe_set("cycle_model_trace_repair"),
        signals=[make_active_signal()],
    )

    repair_request = gateway.requests[1]
    event = result.evidence_events[0]
    assert repair_request.prompt_id == "evidence_judgment_repair"
    assert repair_request.prompt_version == "v0.1"
    assert repair_request.schema_name == "EvidenceJudgment"
    assert repair_request.schema_version == "v0.1"
    assert repair_request.metadata == {
        **legacy_judgment_route_metadata(),
        "repair_attempt_index": 1,
    }
    assert event.discard_reason is None
    assert event.model_trace == {
        "task": "repair_evidence_judgment",
        "adapter_kind": "scripted",
        "prompt_id": "evidence_judgment_repair",
        "prompt_version": "v0.1",
        "schema_name": "EvidenceJudgment",
        "schema_version": "v0.1",
        "repair_attempt_index": 1,
        "metadata": legacy_judgment_route_metadata(),
    }


def test_direct_signal_repair_exhaustion_records_latest_repair_trace():
    gateway = GatewayValidationErrorOnRepairGateway()
    gate = EvidenceIntegrationGate(
        model_gateway=gateway,
        judgment_repair_policy=EvidenceJudgmentRepairPolicy(max_attempts=2),
    )

    result = gate.integrate(
        cycle=make_cycle("cycle_model_trace_repair_exhausted"),
        belief_state=make_explicit_legacy_belief_state(cycle_id="cycle_0"),
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
        "metadata": legacy_judgment_route_metadata(),
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
        belief_state=make_explicit_legacy_belief_state(cycle_id="cycle_0"),
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
                "quality_overrides": {},
            },
            "repair_evidence_judgment": {
                "evidence_type": "supporting",
                "likelihoods": {
                    "H1": "moderately_confirming",
                    "H2": "moderately_disconfirming",
                },
                "interpretation": "Core repaired judgment.",
                "quality_overrides": {},
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
        belief_state=make_explicit_legacy_belief_state(
            cycle_id="cycle_model_gateway",
            cycle_index=1,
        ),
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
                "quality_overrides": {},
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
        belief_state=make_explicit_legacy_belief_state(
            cycle_id="cycle_schema_violation"
        ),
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
    assert "requires exactly the reviewed four fields" in event.discard_reason
    assert event.interpretation == "Model gateway judgment failed schema validation."
    assert event.reliability == 0.0
    assert event.independence == 0.0
    assert event.relevance == 0.0
    assert event.novelty == 0.0
    assert event.specificity == 0.0
    assert event.verifiability == 0.0


def test_native_exclusive_open_schema_violation_is_neutral_and_underdetermined():
    belief_state = make_exact_belief_state()
    hypothesis_ids = [hypothesis.id for hypothesis in belief_state.hypotheses]
    gateway = ScriptedModelGateway(
        responses={
            "judge_evidence": {
                "likelihoods": {
                    hypothesis_id: "neutral" for hypothesis_id in hypothesis_ids
                },
                "unresolved_likelihood": "neutral",
                "frame_fit": "underdetermined",
                "unexplained_observation": None,
                "interpretation": "Missing evidence type.",
                "quality_overrides": {},
            }
        }
    )
    gate = EvidenceIntegrationGate(model_gateway=gateway)
    cycle = make_cycle("cycle_native_schema_violation")

    result = gate.integrate(
        cycle=cycle,
        belief_state=belief_state,
        probe_set=make_empty_probe_set(cycle.cycle_id),
        signals=[make_active_signal()],
    )

    event = result.evidence_events[0]
    assert event.schema_version == "v0.2"
    assert event.discard_reason.startswith("schema_violation:")
    assert set(event.likelihoods.values()) == {LikelihoodBand.NEUTRAL}
    assert event.unresolved_likelihood == LikelihoodBand.NEUTRAL
    assert event.frame_fit == FrameFit.UNDERDETERMINED


def test_core_schema_violation_does_not_update_belief_state():
    gateway = ScriptedModelGateway(
        responses={
            "judge_evidence": {
                "likelihoods": {"H1": "moderately_confirming", "H2": "moderately_disconfirming"},
                "interpretation": "Missing evidence type.",
                "quality_overrides": {},
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
                "quality_overrides": {},
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
        assert signals
        self.seen_framing_methods.append(belief_state.task_frame.framing_method)
        if belief_state.task_frame.framing_method != FramingMethod.LEGACY_MIGRATION:
            return self._integrate_native(
                cycle=cycle,
                belief_state=belief_state,
                signals=signals,
            )
        normalizer = evidence_memory.SignalProvenanceNormalizer()
        normalized_by_id = {}
        for event in self.events:
            if event.derived_from_signal in normalized_by_id:
                continue
            normalized_by_id[event.derived_from_signal] = normalizer.normalize(
                signals[0].model_copy(
                    update={
                        "id": event.derived_from_signal,
                        "provenance": None,
                    }
                ),
                run_id=cycle.run_id,
            )

        prior_memory = belief_state.evidence_memory or EvidenceMemorySnapshot()
        accepted_ids = list(prior_memory.accepted_evidence_ids)
        discard_history = list(prior_memory.discard_and_schema_history)
        discarded_ids = {
            decode_discard_history_entry(entry)[0]
            for entry in discard_history
        }
        lifecycle_ids = set(accepted_ids) | discarded_ids
        bindings = dict(prior_memory.event_signal_identity_digests)
        for event in self.events:
            if event.id not in lifecycle_ids:
                if event.discard_reason is None:
                    accepted_ids.append(event.id)
                else:
                    discard_history.append(
                        encode_discard_history_entry(
                            event.id,
                            event.discard_reason,
                        )
                    )
                lifecycle_ids.add(event.id)
            bindings[event.id] = evidence_memory.canonical_signal_identity_digest(
                normalized_by_id[event.derived_from_signal]
            )
        committed_memory = EvidenceMemorySnapshot(
            memory_version=2,
            accepted_evidence_ids=accepted_ids,
            content_fingerprints=dict(prior_memory.content_fingerprints),
            source_content_fingerprints=dict(
                prior_memory.source_content_fingerprints
            ),
            derivation_roots=dict(prior_memory.derivation_roots),
            event_signal_identity_digests=bindings,
            correlation_credit=dict(prior_memory.correlation_credit),
            discovery_evidence_ids=list(prior_memory.discovery_evidence_ids),
            counterevidence_ids_by_hypothesis={
                hypothesis_id: list(event_ids)
                for hypothesis_id, event_ids in (
                    prior_memory.counterevidence_ids_by_hypothesis.items()
                )
            },
            discard_and_schema_history=discard_history,
        )
        return EvidenceIntegrationResult(
            evidence_events=list(self.events),
            probe_candidates=[],
            evidence_memory=committed_memory,
            normalized_signals=list(normalized_by_id.values()),
        )

    def _integrate_native(self, *, cycle, belief_state, signals):
        normalizer = evidence_memory.SignalProvenanceNormalizer()
        manager = evidence_memory.EvidenceMemoryManager()
        working_memory = belief_state.evidence_memory
        normalized_by_id = {}
        processed_by_id = {}
        processed_events = []
        seen_signatures: set[tuple[str, str]] = set()
        confirming = {
            LikelihoodBand.WEAKLY_CONFIRMING,
            LikelihoodBand.MODERATELY_CONFIRMING,
            LikelihoodBand.STRONGLY_CONFIRMING,
        }
        for original in self.events:
            if original.id in processed_by_id:
                processed_events.append(processed_by_id[original.id])
                continue
            signal = normalized_by_id.get(original.derived_from_signal)
            if signal is None:
                signal = normalizer.normalize(
                    signals[0].model_copy(
                        update={
                            "id": original.derived_from_signal,
                            "raw_content": original.content,
                            "provenance": _static_event_signal_provenance(original),
                        }
                    ),
                    run_id=cycle.run_id,
                )
                normalized_by_id[signal.id] = signal
            unresolved_likelihood = original.unresolved_likelihood
            requires_unresolved = (
                belief_state.frame_state.competition
                == HypothesisCompetition.EXCLUSIVE
                and belief_state.frame_state.coverage
                == HypothesisCoverage.OPEN
            )
            if requires_unresolved and unresolved_likelihood is None:
                unresolved_likelihood = LikelihoodBand.NEUTRAL
            frame_fit = original.frame_fit
            if unresolved_likelihood in confirming:
                frame_fit = FrameFit.SUPPORTS_UNRESOLVED
            is_cycle_duplicate = (
                evidence_memory.observe_cycle_signal_duplicate(
                    signal,
                    seen_signatures,
                )
            )
            preliminary_decision = manager.classify(
                working_memory,
                signal,
                frame_version=belief_state.frame_state.frame_version,
            )
            quality_cap = evidence_memory.SignalQualityAssessor().assess(
                signal=signal,
                event_type=original.evidence_type,
                is_duplicate=(
                    is_cycle_duplicate
                    or preliminary_decision.correlation_status == "duplicate_exact"
                ),
            )
            quality = {
                metric: min(
                    getattr(original, metric),
                    getattr(quality_cap, metric),
                )
                for metric in evidence_memory.SIGNAL_QUALITY_METRICS
            }
            base_weight = (
                quality["reliability"]
                * quality["independence"]
                * quality["relevance"]
                * quality["novelty"]
            )
            decision = manager.classify(
                working_memory,
                signal,
                likelihoods=original.likelihoods,
                unresolved_likelihood=unresolved_likelihood,
                frame_version=belief_state.frame_state.frame_version,
                base_effective_weight=base_weight,
            )
            if decision.correlation_status == "correlated_restatement":
                quality["independence"] = 0.0
                quality["novelty"] = min(quality["novelty"], 0.25)
            event = EvidenceEvent.model_validate(
                {
                    **original.model_dump(mode="python"),
                    **quality,
                    "schema_version": "v0.2",
                    "epistemic_origin": signal.provenance.epistemic_origin,
                    "derivation_root_id": signal.provenance.derivation_root_id,
                    "unresolved_likelihood": unresolved_likelihood,
                    "frame_fit": frame_fit,
                    "correlation_status": decision.correlation_status,
                    "effective_update_weight": decision.effective_update_weight,
                }
            )
            working_memory = manager.commit(
                working_memory,
                signal=signal,
                event=event,
                decision=decision,
            )
            processed_by_id[event.id] = event
            processed_events.append(event)
        return EvidenceIntegrationResult(
            evidence_events=processed_events,
            probe_candidates=[],
            evidence_memory=working_memory,
            normalized_signals=list(normalized_by_id.values()),
        )


def _static_event_signal_provenance(event: EvidenceEvent) -> SignalProvenance:
    return SignalProvenance(
        epistemic_origin=(
            event.epistemic_origin or EpistemicOrigin.EXTERNAL_OBSERVATION
        ),
        source_identity=f"static-event:{event.derived_from_signal}",
        derivation_root_id=(
            event.derivation_root_id or f"root:{event.derived_from_signal}"
        ),
        correlation_group=f"static-event:{event.derived_from_signal}",
        canonical_content_fingerprint="replace-me",
    )


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

    def integrate_cycle(self, cycle, belief_state, probe_set, signals):
        base_signal = signals[0]
        owned_signals = []
        seen_signal_ids = set()
        for event in self.static_gate.events:
            if event.derived_from_signal in seen_signal_ids:
                continue
            seen_signal_ids.add(event.derived_from_signal)
            owned_signals.append(
                base_signal.model_copy(
                    update={
                        "id": event.derived_from_signal,
                        "raw_content": event.content,
                        "provenance": _static_event_signal_provenance(event),
                    }
                )
            )
        return super().integrate_cycle(
            cycle=cycle,
            belief_state=belief_state,
            probe_set=probe_set,
            signals=owned_signals,
        )


class InvalidMemoryGate(EvidenceIntegrationGate):
    def integrate(self, *, cycle, belief_state, probe_set, signals):
        normalizer = evidence_memory.SignalProvenanceNormalizer()
        return EvidenceIntegrationResult(
            evidence_events=[],
            probe_candidates=[],
            evidence_memory=EvidenceMemorySnapshot.model_construct(memory_version=0),
            normalized_signals=[
                normalizer.normalize(signal, run_id=cycle.run_id)
                for signal in signals
            ],
        )


class InvalidMemoryCore(BayesProbeCore):
    def _create_evidence_integration_gate(self):
        return InvalidMemoryGate()


class SuppliedIntegrationGate(EvidenceIntegrationGate):
    def __init__(self, integration: EvidenceIntegrationResult) -> None:
        self.integration = integration

    def integrate(self, *, cycle, belief_state, probe_set, signals):
        normalizer = evidence_memory.SignalProvenanceNormalizer()
        return EvidenceIntegrationResult(
            evidence_events=list(self.integration.evidence_events),
            probe_candidates=list(self.integration.probe_candidates),
            evidence_memory=self.integration.evidence_memory,
            normalized_signals=[
                normalizer.normalize(signal, run_id=cycle.run_id)
                for signal in signals
            ],
        )


class SuppliedIntegrationCore(BayesProbeCore):
    def __init__(
        self,
        integration: EvidenceIntegrationResult,
        *,
        ledger: JsonlLedgerStore,
        model_gateway=None,
        correlation_credit_policy: CorrelationCreditPolicy | None = None,
    ) -> None:
        self.supplied_gate = SuppliedIntegrationGate(integration)
        super().__init__(
            ledger=ledger,
            model_gateway=model_gateway,
            correlation_credit_policy=correlation_credit_policy,
        )

    def _create_evidence_integration_gate(self):
        return self.supplied_gate

    def integrate_cycle(self, cycle, belief_state, probe_set, signals):
        supplied_signals = self.supplied_gate.integration.normalized_signals
        if supplied_signals:
            signals = list(supplied_signals)
        return super().integrate_cycle(
            cycle=cycle,
            belief_state=belief_state,
            probe_set=probe_set,
            signals=signals,
        )


def _policy_transition_event(
    signal: ExternalSignal,
    *,
    event_id: str,
    correlation_status: str,
    effective_update_weight: float,
    quality: dict[str, float] | None = None,
    discard_reason: str | None = None,
) -> EvidenceEvent:
    quality = quality or {
        "reliability": 0.8,
        "independence": 0.8,
        "relevance": 0.9,
        "novelty": 0.8,
        "specificity": 0.7,
        "verifiability": 0.7,
    }
    return EvidenceEvent(
        schema_version="v0.2",
        id=event_id,
        derived_from_signal=signal.id,
        epistemic_origin=signal.provenance.epistemic_origin,
        derivation_root_id=signal.provenance.derivation_root_id,
        target_hypotheses=["H1", "H2"],
        evidence_type=EvidenceType.SUPPORTING,
        content=signal.raw_content,
        likelihoods={
            "H1": LikelihoodBand.MODERATELY_CONFIRMING,
            "H2": LikelihoodBand.NEUTRAL,
        },
        unresolved_likelihood=LikelihoodBand.NEUTRAL,
        frame_fit=FrameFit.UNDERDETERMINED,
        correlation_status=correlation_status,
        effective_update_weight=effective_update_weight,
        discard_reason=discard_reason,
        **quality,
    )


def _commit_self_declared_transition(
    prior_memory: EvidenceMemorySnapshot,
    *,
    signal: ExternalSignal,
    event: EvidenceEvent,
    resulting_used_credit: float,
) -> EvidenceMemorySnapshot:
    manager = evidence_memory.EvidenceMemoryManager()
    canonical = manager.classify(prior_memory, signal)
    credit_key = (
        f"{canonical.canonical_correlation_group}|H1|confirming"
    )
    declared = evidence_memory.EvidenceMemoryDecision(
        correlation_status=event.correlation_status,
        effective_update_weight=event.effective_update_weight,
        discard_reason=event.discard_reason,
        remaining_credit={credit_key: 1.0 - resulting_used_credit},
        canonical_correlation_group=canonical.canonical_correlation_group,
    )
    return manager.commit(
        prior_memory,
        signal=signal,
        event=event,
        decision=declared,
    )


def _assert_supplied_transition_rejected_atomically(
    tmp_path: Path,
    *,
    name: str,
    state: BeliefState,
    integration: EvidenceIntegrationResult,
    correlation_credit_policy: CorrelationCreditPolicy | None = None,
) -> None:
    ledger_path = tmp_path / f"{name}.jsonl"
    ledger_path.touch()
    core = SuppliedIntegrationCore(
        integration,
        ledger=JsonlLedgerStore(ledger_path),
        correlation_credit_policy=correlation_credit_policy,
    )
    real_solver = core._belief_solver

    class RecordingSolver:
        calls = 0

        def solve(self, *args, **kwargs):
            self.calls += 1
            return real_solver.solve(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(real_solver, name)

    recording_solver = RecordingSolver()
    core._belief_solver = recording_solver
    prior_state = state.model_dump(mode="json")

    with pytest.raises(ValueError, match="native evidence memory transition"):
        core.integrate_cycle(
            cycle=make_cycle(f"cycle_{name}"),
            belief_state=state,
            probe_set=make_empty_probe_set(f"cycle_{name}"),
            signals=[make_active_signal()],
        )

    assert recording_solver.calls == 0
    assert state.model_dump(mode="json") == prior_state
    assert ledger_path.read_bytes() == b""


def _native_transition_fixture():
    state = make_exact_belief_state()
    manager = evidence_memory.EvidenceMemoryManager()
    normalizer = evidence_memory.SignalProvenanceNormalizer()
    accepted_signal = normalizer.normalize(
        ExternalSignal(
            id="S_transition_accepted",
            cycle_id="pending",
            signal_kind=SignalKind.ACTIVE,
            source_type="retrieved_source",
            source="transition-fixture",
            raw_content="A prior accepted observation supports H1.",
            initial_target_hypotheses=["H1", "H2"],
            provenance=SignalProvenance(
                epistemic_origin=EpistemicOrigin.RETRIEVED_SOURCE,
                source_identity="transition-fixture",
                derivation_root_id="root-transition-accepted",
                correlation_group="transition-fixture",
                canonical_content_fingerprint="replace-me",
            ),
        ),
        run_id=state.run_id,
    )
    accepted_decision = manager.classify(
        EvidenceMemorySnapshot(),
        accepted_signal,
        likelihoods={
            "H1": LikelihoodBand.MODERATELY_CONFIRMING,
            "H2": LikelihoodBand.MODERATELY_DISCONFIRMING,
        },
        unresolved_likelihood=LikelihoodBand.NEUTRAL,
        base_effective_weight=0.25,
    )
    accepted_event = EvidenceEvent(
        schema_version="v0.2",
        id="E_transition_accepted",
        derived_from_signal=accepted_signal.id,
        epistemic_origin=accepted_signal.provenance.epistemic_origin,
        derivation_root_id=accepted_signal.provenance.derivation_root_id,
        target_hypotheses=["H1", "H2"],
        evidence_type=EvidenceType.SUPPORTING,
        content=accepted_signal.raw_content,
        likelihoods={
            "H1": LikelihoodBand.MODERATELY_CONFIRMING,
            "H2": LikelihoodBand.MODERATELY_DISCONFIRMING,
        },
        unresolved_likelihood=LikelihoodBand.NEUTRAL,
        frame_fit=FrameFit.UNDERDETERMINED,
        correlation_status=accepted_decision.correlation_status,
        effective_update_weight=accepted_decision.effective_update_weight,
    )
    memory = manager.commit(
        EvidenceMemorySnapshot(),
        signal=accepted_signal,
        event=accepted_event,
        decision=accepted_decision,
    )

    discarded_signal = normalizer.normalize(
        accepted_signal.model_copy(
            update={
                "id": "S_transition_discarded",
                "raw_content": "A prior malformed observation was discarded.",
                "provenance": accepted_signal.provenance.model_copy(
                    update={"derivation_root_id": "root-transition-discarded"}
                ),
            }
        ),
        run_id=state.run_id,
    )
    discarded_decision = manager.classify(
        memory,
        discarded_signal,
        likelihoods={
            "H1": LikelihoodBand.NEUTRAL,
            "H2": LikelihoodBand.NEUTRAL,
        },
        unresolved_likelihood=LikelihoodBand.NEUTRAL,
        base_effective_weight=0.0,
    )
    discarded_event = EvidenceEvent(
        schema_version="v0.2",
        id="E_transition_discarded",
        derived_from_signal=discarded_signal.id,
        epistemic_origin=discarded_signal.provenance.epistemic_origin,
        derivation_root_id=discarded_signal.provenance.derivation_root_id,
        target_hypotheses=["H1", "H2"],
        evidence_type=EvidenceType.NEUTRAL,
        content=discarded_signal.raw_content,
        likelihoods={
            "H1": LikelihoodBand.NEUTRAL,
            "H2": LikelihoodBand.NEUTRAL,
        },
        unresolved_likelihood=LikelihoodBand.NEUTRAL,
        frame_fit=FrameFit.UNDERDETERMINED,
        correlation_status=discarded_decision.correlation_status,
        effective_update_weight=discarded_decision.effective_update_weight,
        discard_reason="schema_violation:fixture",
    )
    memory = manager.commit(
        memory,
        signal=discarded_signal,
        event=discarded_event,
        decision=discarded_decision,
    )
    memory = EvidenceMemorySnapshot.model_validate(
        {
            **memory.model_dump(mode="python"),
            "discovery_evidence_ids": [accepted_event.id],
        }
    )
    payload = state.model_dump(mode="python")
    payload.update(
        {
            "evidence_memory": memory.model_dump(mode="python"),
            "ledger_refs": {
                **state.ledger_refs,
                "evidence_events": [accepted_event.id, discarded_event.id],
            },
        }
    )
    return (
        BeliefState.model_validate(payload),
        memory,
        accepted_event,
        accepted_signal,
    )


def _regressive_native_memory(
    memory: EvidenceMemorySnapshot,
    mutation: str,
) -> EvidenceMemorySnapshot:
    payload = memory.model_dump(mode="python")
    if mutation == "drop_identities":
        payload["content_fingerprints"] = {}
        payload["source_content_fingerprints"] = {}
        payload["derivation_roots"] = {}
    elif mutation == "rebind_identity":
        signal_id = next(iter(payload["content_fingerprints"]))
        changed_fingerprint = "sha256:" + "b" * 64
        identity = json.loads(payload["source_content_fingerprints"][signal_id])
        identity[1] = changed_fingerprint
        payload["content_fingerprints"][signal_id] = changed_fingerprint
        payload["source_content_fingerprints"][signal_id] = json.dumps(
            identity,
            separators=(",", ":"),
        )
    elif mutation == "drop_accepted_history":
        removed = payload["accepted_evidence_ids"].pop(0)
        payload["event_signal_identity_digests"].pop(removed)
        payload["discovery_evidence_ids"] = []
        payload["counterevidence_ids_by_hypothesis"] = {}
    elif mutation == "drop_discard_history":
        removed, _ = decode_discard_history_entry(
            payload["discard_and_schema_history"].pop(0)
        )
        payload["event_signal_identity_digests"].pop(removed)
    elif mutation == "drop_binding":
        payload["event_signal_identity_digests"].pop(
            next(iter(payload["event_signal_identity_digests"]))
        )
    elif mutation == "rebind_binding":
        key = next(iter(payload["event_signal_identity_digests"]))
        payload["event_signal_identity_digests"][key] = "c" * 64
    elif mutation == "rewrite_discard_history":
        event_id, _ = decode_discard_history_entry(
            payload["discard_and_schema_history"][0]
        )
        payload["discard_and_schema_history"][0] = encode_discard_history_entry(
            event_id,
            "schema_violation:changed",
        )
    elif mutation == "drop_discovery":
        payload["discovery_evidence_ids"] = []
    elif mutation == "rewrite_discovery":
        discarded_id, _ = decode_discard_history_entry(
            payload["discard_and_schema_history"][0]
        )
        payload["discovery_evidence_ids"] = [discarded_id]
    elif mutation == "drop_counterevidence":
        payload["counterevidence_ids_by_hypothesis"] = {}
    elif mutation == "rewrite_counterevidence":
        discarded_id, _ = decode_discard_history_entry(
            payload["discard_and_schema_history"][0]
        )
        payload["counterevidence_ids_by_hypothesis"] = {
            "H2": [discarded_id]
        }
    elif mutation == "empty_credit":
        payload["correlation_credit"] = {}
    elif mutation == "decrease_credit":
        key = next(iter(payload["correlation_credit"]))
        payload["correlation_credit"][key] /= 2
    elif mutation == "increase_credit":
        key = next(iter(payload["correlation_credit"]))
        payload["correlation_credit"][key] += 0.1
    else:
        raise AssertionError(f"unknown memory mutation: {mutation}")
    return EvidenceMemorySnapshot.model_validate(payload)


def test_invalid_committed_memory_fails_before_any_cycle_ledger_append(tmp_path: Path):
    ledger = JsonlLedgerStore(tmp_path / "invalid-memory-ledger.jsonl")
    core = InvalidMemoryCore(ledger=ledger)

    with pytest.raises(ValueError, match="native evidence memory transition"):
        core.integrate_cycle(
            cycle=make_cycle("cycle_invalid_memory"),
            belief_state=make_exact_belief_state(),
            probe_set=make_empty_probe_set("cycle_invalid_memory"),
            signals=[make_active_signal()],
        )

    assert ledger.read_all() == []


def test_native_plain_list_gate_fails_before_state_or_ledger_mutation(
    tmp_path: Path,
):
    class PlainListNativeGate(EvidenceIntegrationGate):
        def integrate(self, *, cycle, belief_state, probe_set, signals):
            return [
                EvidenceEvent(
                    schema_version="v0.2",
                    id="E_unowned_native",
                    derived_from_signal="S_unowned_native",
                    epistemic_origin=EpistemicOrigin.MODEL_REASONING,
                    derivation_root_id="root-unowned-native",
                    target_hypotheses=["H1", "H2"],
                    evidence_type=EvidenceType.NEUTRAL,
                    content="An unowned native event must not be applied.",
                    likelihoods={
                        "H1": LikelihoodBand.NEUTRAL,
                        "H2": LikelihoodBand.NEUTRAL,
                    },
                    unresolved_likelihood=LikelihoodBand.NEUTRAL,
                    frame_fit=FrameFit.UNDERDETERMINED,
                    correlation_status="novel",
                    effective_update_weight=0.0,
                )
            ]

    class PlainListNativeCore(BayesProbeCore):
        def _create_evidence_integration_gate(self):
            return PlainListNativeGate()

    ledger_path = tmp_path / "native-plain-list-ledger.jsonl"
    ledger_path.touch()
    state = make_exact_belief_state()
    prior_state = state.model_dump(mode="json")
    prior_memory = state.evidence_memory.model_dump(mode="json")
    core = PlainListNativeCore(ledger=JsonlLedgerStore(ledger_path))
    cycle = make_cycle("cycle_native_plain_list")

    with pytest.raises(
        ValueError,
        match="native closed signal ownership is invalid",
    ):
        core.integrate_cycle(
            cycle=cycle,
            belief_state=state,
            probe_set=make_empty_probe_set(cycle.cycle_id),
            signals=[make_active_signal()],
        )

    assert state.model_dump(mode="json") == prior_state
    assert state.evidence_memory.model_dump(mode="json") == prior_memory
    assert ledger_path.read_bytes() == b""


def test_native_gate_wrong_event_signal_binding_fails_before_ledger(
    tmp_path: Path,
):
    class WrongBindingGate(EvidenceIntegrationGate):
        def integrate(self, *, cycle, belief_state, probe_set, signals):
            normalized = evidence_memory.SignalProvenanceNormalizer().normalize(
                signals[0],
                run_id=cycle.run_id,
            )
            event = EvidenceEvent(
                schema_version="v0.2",
                id="E_wrong_binding",
                derived_from_signal=normalized.id,
                epistemic_origin=EpistemicOrigin.MODEL_REASONING,
                derivation_root_id=normalized.provenance.derivation_root_id,
                target_hypotheses=["H1", "H2"],
                evidence_type=EvidenceType.NEUTRAL,
                content="A syntactically bound native event.",
                likelihoods={
                    "H1": LikelihoodBand.NEUTRAL,
                    "H2": LikelihoodBand.NEUTRAL,
                },
                unresolved_likelihood=LikelihoodBand.NEUTRAL,
                frame_fit=FrameFit.UNDERDETERMINED,
                correlation_status="novel",
                effective_update_weight=0.0,
            )
            return EvidenceIntegrationResult(
                evidence_events=[event],
                probe_candidates=[],
                evidence_memory=EvidenceMemorySnapshot(
                    memory_version=2,
                    accepted_evidence_ids=[event.id],
                    event_signal_identity_digests={event.id: "a" * 64},
                ),
                normalized_signals=[normalized],
            )

    class WrongBindingCore(BayesProbeCore):
        def _create_evidence_integration_gate(self):
            return WrongBindingGate()

    ledger_path = tmp_path / "native-wrong-binding-ledger.jsonl"
    ledger_path.touch()
    state = make_exact_belief_state()
    prior_state = state.model_dump(mode="json")
    core = WrongBindingCore(ledger=JsonlLedgerStore(ledger_path))
    cycle = make_cycle("cycle_native_wrong_binding")

    with pytest.raises(
        ValueError,
        match="native evidence memory transition",
    ):
        core.integrate_cycle(
            cycle=cycle,
            belief_state=state,
            probe_set=make_empty_probe_set(cycle.cycle_id),
            signals=[make_active_signal()],
        )

    assert state.model_dump(mode="json") == prior_state
    assert ledger_path.read_bytes() == b""


@pytest.mark.parametrize(
    "mutation",
    [
        "drop_identities",
        "rebind_identity",
        "drop_accepted_history",
        "drop_discard_history",
        "drop_binding",
        "rebind_binding",
        "rewrite_discard_history",
        "drop_discovery",
        "rewrite_discovery",
        "drop_counterevidence",
        "rewrite_counterevidence",
        "empty_credit",
        "decrease_credit",
        "increase_credit",
    ],
)
@pytest.mark.parametrize(
    "existing_event_only",
    [False, True],
    ids=["no_events", "existing_event"],
)
def test_native_memory_replacement_regressions_fail_before_solver_or_ledger(
    tmp_path: Path,
    mutation: str,
    existing_event_only: bool,
):
    state, prior_memory, event, signal = _native_transition_fixture()
    candidate = _regressive_native_memory(prior_memory, mutation)
    integration = EvidenceIntegrationResult(
        evidence_events=[event] if existing_event_only else [],
        probe_candidates=[],
        evidence_memory=candidate,
        normalized_signals=[signal] if existing_event_only else [],
    )
    gateway = ScriptedModelGateway(
        responses={"judge_evidence": _native_open_judgment()}
    )
    ledger_path = tmp_path / (
        f"native-transition-{mutation}-{existing_event_only}.jsonl"
    )
    ledger_path.touch()
    core = SuppliedIntegrationCore(
        integration,
        ledger=JsonlLedgerStore(ledger_path),
        model_gateway=gateway,
    )
    prior_state = state.model_dump(mode="json")

    with pytest.raises(ValueError, match="native evidence memory transition"):
        core.integrate_cycle(
            cycle=make_cycle(f"cycle_transition_{mutation}"),
            belief_state=state,
            probe_set=make_empty_probe_set(f"cycle_transition_{mutation}"),
            signals=[make_active_signal()],
        )

    assert gateway.requests == []
    assert state.model_dump(mode="json") == prior_state
    assert state.evidence_memory == prior_memory
    assert ledger_path.read_bytes() == b""


def test_existing_event_only_result_cannot_rewrite_directional_credit(
    tmp_path: Path,
):
    state, prior_memory, event, signal = _native_transition_fixture()
    candidate = prior_memory.model_copy(update={"correlation_credit": {}})
    integration = EvidenceIntegrationResult(
        evidence_events=[event],
        probe_candidates=[],
        evidence_memory=candidate,
        normalized_signals=[signal],
    )
    ledger_path = tmp_path / "native-existing-event-credit.jsonl"
    ledger_path.touch()
    core = SuppliedIntegrationCore(
        integration,
        ledger=JsonlLedgerStore(ledger_path),
    )
    prior_state = state.model_dump(mode="json")

    with pytest.raises(ValueError, match="native evidence memory transition"):
        core.integrate_cycle(
            cycle=make_cycle("cycle_existing_event_credit"),
            belief_state=state,
            probe_set=make_empty_probe_set("cycle_existing_event_credit"),
            signals=[make_active_signal()],
        )

    assert state.model_dump(mode="json") == prior_state
    assert ledger_path.read_bytes() == b""


def test_new_directional_event_cannot_skip_its_credit_commit(tmp_path: Path):
    state = make_exact_belief_state()
    gateway = ScriptedModelGateway(
        responses={"judge_evidence": _native_open_judgment()}
    )
    cycle = make_cycle("cycle_missing_new_credit")
    production = EvidenceIntegrationGate(model_gateway=gateway).integrate(
        cycle=cycle,
        belief_state=state,
        probe_set=make_empty_probe_set(cycle.cycle_id),
        signals=[make_active_signal()],
    )
    assert production.evidence_memory.correlation_credit
    invalid_memory = production.evidence_memory.model_copy(
        update={"correlation_credit": {}}
    )
    integration = EvidenceIntegrationResult(
        evidence_events=production.evidence_events,
        probe_candidates=production.probe_candidates,
        evidence_memory=invalid_memory,
        normalized_signals=production.normalized_signals,
    )
    ledger_path = tmp_path / "native-missing-new-credit.jsonl"
    ledger_path.touch()
    core = SuppliedIntegrationCore(
        integration,
        ledger=JsonlLedgerStore(ledger_path),
        model_gateway=gateway,
    )
    prior_state = state.model_dump(mode="json")
    prior_provider_calls = len(gateway.requests)

    with pytest.raises(ValueError, match="native evidence memory transition"):
        core.integrate_cycle(
            cycle=cycle,
            belief_state=state,
            probe_set=make_empty_probe_set(cycle.cycle_id),
            signals=[make_active_signal()],
        )

    assert len(gateway.requests) == prior_provider_calls
    assert state.model_dump(mode="json") == prior_state
    assert ledger_path.read_bytes() == b""


def test_core_constructs_isolated_memory_managers_with_same_policy(monkeypatch):
    instances = []

    class RecordingManager(evidence_memory.EvidenceMemoryManager):
        def __init__(self, policy=None):
            super().__init__(policy)
            instances.append(self)

    monkeypatch.setattr(core_module, "EvidenceMemoryManager", RecordingManager)

    class ManagerFactoryCore(core_module.BayesProbeCore):
        def _create_evidence_integration_gate(self):
            gate = super()._create_evidence_integration_gate()
            self.factory_manager = gate._memory_manager
            return gate

    policy = CorrelationCreditPolicy(
        max_cumulative_effective_weight_per_direction=0.2
    )
    core = ManagerFactoryCore(correlation_credit_policy=policy)

    assert instances == [core._evidence_memory_manager, core.factory_manager]
    assert core._evidence_memory_manager is not core.factory_manager
    assert core._evidence_memory_manager._policy == policy
    assert core.factory_manager._policy == policy
    assert core._evidence_memory_manager._policy is not core.factory_manager._policy


class _GateManagerAuthorityAttack:
    def __init__(self, delegate, attack: str) -> None:
        self._delegate = delegate
        self._attack = attack
        self.event_weight = None

    def integrate(self, *, cycle, belief_state, probe_set, signals):
        gate_manager = self._delegate._memory_manager
        if self._attack == "policy":
            gate_manager._policy = CorrelationCreditPolicy(
                max_cumulative_effective_weight_per_direction=1.0
            )
        integration = self._delegate.integrate(
            cycle=cycle,
            belief_state=belief_state,
            probe_set=probe_set,
            signals=signals,
        )
        event = integration.evidence_events[0]
        if self._attack == "validator":
            event.effective_update_weight = 0.9
            gate_manager.validate_transition = (
                lambda *_args, **_kwargs: integration.evidence_memory
            )
        self.event_weight = event.effective_update_weight
        return integration


class _GateManagerAuthorityAttackCore(BayesProbeCore):
    def __init__(self, attack: str, *, ledger, model_gateway) -> None:
        self._manager_attack = attack
        super().__init__(
            ledger=ledger,
            model_gateway=model_gateway,
            correlation_credit_policy=CorrelationCreditPolicy(
                max_cumulative_effective_weight_per_direction=0.2
            ),
        )

    def _create_evidence_integration_gate(self):
        self.manager_attack_gate = _GateManagerAuthorityAttack(
            super()._create_evidence_integration_gate(),
            self._manager_attack,
        )
        return self.manager_attack_gate


@pytest.mark.parametrize("attack", ["policy", "validator"])
def test_gate_manager_mutation_cannot_redefine_core_transition_authority(
    tmp_path: Path,
    attack: str,
):
    ledger_path = tmp_path / f"gate-manager-{attack}.jsonl"
    ledger_path.touch()
    gateway = ScriptedModelGateway(
        responses={"judge_evidence": _native_open_judgment()}
    )
    core = _GateManagerAuthorityAttackCore(
        attack,
        ledger=JsonlLedgerStore(ledger_path),
        model_gateway=gateway,
    )
    solver = _RecordingSolverProxy(core._belief_solver)
    core._belief_solver = solver
    state = make_exact_belief_state()
    prior_state = state.model_dump(mode="json")
    cycle = make_cycle(f"cycle_gate_manager_{attack}")

    with pytest.raises(
        ValueError,
        match="native evidence memory transition is invalid",
    ):
        core.integrate_cycle(
            cycle=cycle,
            belief_state=state,
            probe_set=make_empty_probe_set(cycle.cycle_id),
            signals=[
                make_active_signal().model_copy(
                    update={"id": f"S_gate_manager_{attack}"}
                )
            ],
        )

    assert core.manager_attack_gate.event_weight > 0.2
    authoritative_policy = core._evidence_memory_manager._policy
    assert (
        authoritative_policy.max_cumulative_effective_weight_per_direction
        == 0.2
    )
    assert solver.calls == 0
    assert state.model_dump(mode="json") == prior_state
    assert ledger_path.read_bytes() == b""


class _EventContentRewriteGate:
    def __init__(self, delegate) -> None:
        self._delegate = delegate
        self.rewritten_content = None

    def integrate(self, *, cycle, belief_state, probe_set, signals):
        integration = self._delegate.integrate(
            cycle=cycle,
            belief_state=belief_state,
            probe_set=probe_set,
            signals=signals,
        )
        event = integration.evidence_events[0]
        self.rewritten_content = f"{event.content} "
        event.content = self.rewritten_content
        return integration


class _EventContentRewriteCore(BayesProbeCore):
    def _create_evidence_integration_gate(self):
        self.content_rewrite_gate = _EventContentRewriteGate(
            super()._create_evidence_integration_gate()
        )
        return self.content_rewrite_gate


def test_native_event_content_rewrite_fails_before_solver_or_ledger(
    tmp_path: Path,
):
    ledger_path = tmp_path / "event-content-rewrite.jsonl"
    ledger_path.touch()
    gateway = ScriptedModelGateway(
        responses={"judge_evidence": _native_open_judgment()}
    )
    core = _EventContentRewriteCore(
        ledger=JsonlLedgerStore(ledger_path),
        model_gateway=gateway,
    )
    solver = _RecordingSolverProxy(core._belief_solver)
    core._belief_solver = solver
    state = make_exact_belief_state()
    prior_state = state.model_dump(mode="json")
    signal = make_active_signal().model_copy(
        update={
            "id": "S_event_content_rewrite",
            "raw_content": "Byte-exact authoritative signal content.",
        }
    )
    cycle = make_cycle("cycle_event_content_rewrite")

    with pytest.raises(
        ValueError,
        match="native evidence memory transition is invalid",
    ):
        core.integrate_cycle(
            cycle=cycle,
            belief_state=state,
            probe_set=make_empty_probe_set(cycle.cycle_id),
            signals=[signal],
        )

    assert core.content_rewrite_gate.rewritten_content == (
        f"{signal.raw_content} "
    )
    assert solver.calls == 0
    assert state.model_dump(mode="json") == prior_state
    assert ledger_path.read_bytes() == b""


def test_core_custom_credit_policy_is_shared_with_production_gate(tmp_path: Path):
    policy = CorrelationCreditPolicy(
        max_cumulative_effective_weight_per_direction=0.2
    )
    gateway = ScriptedModelGateway(
        responses={"judge_evidence": _native_open_judgment()}
    )
    ledger = JsonlLedgerStore(tmp_path / "custom-credit-policy.jsonl")
    core = BayesProbeCore(
        ledger=ledger,
        model_gateway=gateway,
        correlation_credit_policy=policy,
    )
    state = make_exact_belief_state()
    first_cycle = make_cycle("cycle_custom_credit_first")

    first = core.integrate_cycle(
        cycle=first_cycle,
        belief_state=state,
        probe_set=make_empty_probe_set(first_cycle.cycle_id),
        signals=[
            make_active_signal().model_copy(
                update={"id": "S_custom_credit_first"}
            )
        ],
    )

    first_event = first.evidence_events[0]
    assert first_event.effective_update_weight == pytest.approx(0.2)
    first_credit = first.belief_state.evidence_memory.correlation_credit
    assert first_credit
    assert all(value == pytest.approx(0.2) for value in first_credit.values())

    second_cycle = make_cycle("cycle_custom_credit_second").model_copy(
        update={"cycle_index": 2}
    )
    second = core.integrate_cycle(
        cycle=second_cycle,
        belief_state=first.belief_state,
        probe_set=make_empty_probe_set(second_cycle.cycle_id),
        signals=[
            make_active_signal().model_copy(
                update={
                    "id": "S_custom_credit_second",
                    "raw_content": "A second observation spends the same direction.",
                }
            )
        ],
    )

    second_event = second.evidence_events[0]
    assert second_event.correlation_status == "correlated_novel"
    assert second_event.effective_update_weight == 0.0
    assert second_event.discard_reason == "correlation_credit_saturated"
    assert second.belief_state.evidence_memory.correlation_credit == (
        first.belief_state.evidence_memory.correlation_credit
    )
    assert [
        record["payload"]["id"] for record in ledger.read_all("evidence_event")
    ] == [first_event.id, second_event.id]


def test_core_default_credit_policy_behavior_is_unchanged():
    gateway = ScriptedModelGateway(
        responses={"judge_evidence": _native_open_judgment()}
    )
    cycle = make_cycle("cycle_default_credit_policy")

    result = BayesProbeCore(model_gateway=gateway).integrate_cycle(
        cycle=cycle,
        belief_state=make_exact_belief_state(),
        probe_set=make_empty_probe_set(cycle.cycle_id),
        signals=[make_active_signal()],
    )

    assert result.evidence_events[0].effective_update_weight == pytest.approx(
        0.4608
    )


def test_core_rejects_transition_built_under_a_different_credit_policy(
    tmp_path: Path,
):
    state = make_exact_belief_state()
    gateway = ScriptedModelGateway(
        responses={"judge_evidence": _native_open_judgment()}
    )
    cycle = make_cycle("cycle_mismatched_credit_policy")
    production = EvidenceIntegrationGate(model_gateway=gateway).integrate(
        cycle=cycle,
        belief_state=state,
        probe_set=make_empty_probe_set(cycle.cycle_id),
        signals=[make_active_signal()],
    )
    prior_state = state.model_dump(mode="json")
    prior_provider_calls = len(gateway.requests)

    _assert_supplied_transition_rejected_atomically(
        tmp_path,
        name="mismatched_credit_policy",
        state=state,
        integration=production,
        correlation_credit_policy=CorrelationCreditPolicy(
            max_cumulative_effective_weight_per_direction=0.2
        ),
    )

    assert len(gateway.requests) == prior_provider_calls
    assert state.model_dump(mode="json") == prior_state


def test_uncapped_same_batch_duplicate_transition_fails_atomically(tmp_path: Path):
    state = make_exact_belief_state()
    first_raw = _memory_signal(
        "S_uncapped_signature_first",
        "The same source repeats this audited observation.",
        root="root-uncapped-signature-first",
    )
    second_raw = first_raw.model_copy(
        update={
            "id": "S_uncapped_signature_second",
            "provenance": first_raw.provenance.model_copy(
                update={
                    "derivation_root_id": "root-uncapped-signature-second",
                    "correlation_group": "caller-supplied-uncapped-second",
                }
            ),
        }
    )
    gateway = ScriptedModelGateway(
        responses={"judge_evidence": _native_open_judgment()}
    )
    cycle = make_cycle("cycle_uncapped_signature")
    production = EvidenceIntegrationGate(model_gateway=gateway).integrate(
        cycle=cycle,
        belief_state=state,
        probe_set=make_empty_probe_set(cycle.cycle_id),
        signals=[first_raw, second_raw],
    )
    first_signal, second_signal = production.normalized_signals
    first_event, capped_second_event = production.evidence_events
    assert capped_second_event.correlation_status == "correlated_novel"
    assert capped_second_event.effective_update_weight == pytest.approx(0.045)

    manager = evidence_memory.EvidenceMemoryManager()
    first_decision = manager.classify(
        state.evidence_memory,
        first_signal,
        likelihoods=first_event.likelihoods,
        unresolved_likelihood=first_event.unresolved_likelihood,
        frame_version=state.frame_state.frame_version,
        base_effective_weight=(
            first_event.reliability
            * first_event.independence
            * first_event.relevance
            * first_event.novelty
        ),
    )
    forged_memory = manager.commit(
        state.evidence_memory,
        signal=first_signal,
        event=first_event,
        decision=first_decision,
    )
    uncapped_quality = {
        "reliability": 0.8,
        "independence": 0.8,
        "relevance": 0.9,
        "novelty": 0.8,
        "specificity": 0.7,
        "verifiability": 0.7,
    }
    second_decision = manager.classify(
        forged_memory,
        second_signal,
        likelihoods=capped_second_event.likelihoods,
        unresolved_likelihood=capped_second_event.unresolved_likelihood,
        frame_version=state.frame_state.frame_version,
        base_effective_weight=0.8 * 0.8 * 0.9 * 0.8,
    )
    forged_second_event = EvidenceEvent.model_validate(
        {
            **capped_second_event.model_dump(mode="python"),
            **uncapped_quality,
            "correlation_status": second_decision.correlation_status,
            "effective_update_weight": second_decision.effective_update_weight,
        }
    )
    forged_memory = manager.commit(
        forged_memory,
        signal=second_signal,
        event=forged_second_event,
        decision=second_decision,
    )
    assert forged_second_event.effective_update_weight == pytest.approx(0.4608)

    _assert_supplied_transition_rejected_atomically(
        tmp_path,
        name="uncapped_same_batch_duplicate",
        state=state,
        integration=EvidenceIntegrationResult(
            evidence_events=[first_event, forged_second_event],
            probe_candidates=[],
            evidence_memory=forged_memory,
            normalized_signals=[first_signal, second_signal],
        ),
    )


class ClosedSignalOwnershipGate(EvidenceIntegrationGate):
    def __init__(self, case: str) -> None:
        self.case = case

    def integrate(self, *, cycle, belief_state, probe_set, signals):
        if self.case == "none":
            owned = None
        elif self.case == "empty":
            owned = []
        else:
            normalized = [
                evidence_memory.SignalProvenanceNormalizer().normalize(
                    signal,
                    run_id=cycle.run_id,
                )
                for signal in signals
            ]
            if self.case == "valid":
                owned = normalized
            elif self.case == "missing":
                owned = normalized[:-1]
            elif self.case == "extra":
                owned = [
                    *normalized,
                    normalized[-1].model_copy(
                        update={"id": "S_closed_signal_extra"}
                    ),
                ]
            elif self.case == "reordered":
                owned = list(reversed(normalized))
            elif self.case == "changed":
                owned = [
                    normalized[0].model_copy(
                        update={"raw_content": "Changed after inbox closure."}
                    ),
                    *normalized[1:],
                ]
            elif self.case == "noncanonical_provenance":
                owned = [
                    normalized[0].model_copy(
                        update={
                            "provenance": normalized[0].provenance.model_copy(
                                update={
                                    "canonical_content_fingerprint": (
                                        "sha256:" + "0" * 64
                                    )
                                }
                            )
                        }
                    ),
                    *normalized[1:],
                ]
            else:
                raise AssertionError(
                    f"unknown closed signal case: {self.case}"
                )
        return EvidenceIntegrationResult(
            evidence_events=[],
            probe_candidates=[],
            evidence_memory=belief_state.evidence_memory,
            normalized_signals=owned,
        )


class ClosedSignalOwnershipCore(BayesProbeCore):
    def __init__(self, case: str, *, ledger: JsonlLedgerStore) -> None:
        self.closed_signal_case = case
        super().__init__(ledger=ledger)

    def _create_evidence_integration_gate(self):
        return ClosedSignalOwnershipGate(self.closed_signal_case)


def _assert_closed_signal_ownership_rejected(
    tmp_path: Path,
    *,
    case: str,
    signals: list[ExternalSignal],
) -> str:
    ledger_path = tmp_path / f"closed-signal-{case}.jsonl"
    ledger_path.touch()
    core = ClosedSignalOwnershipCore(
        case,
        ledger=JsonlLedgerStore(ledger_path),
    )
    transition_calls = 0
    real_validate_transition = (
        core._evidence_memory_manager.validate_transition
    )

    def recording_validate_transition(*args, **kwargs):
        nonlocal transition_calls
        transition_calls += 1
        return real_validate_transition(*args, **kwargs)

    core._evidence_memory_manager.validate_transition = (
        recording_validate_transition
    )
    real_solver = core._belief_solver

    class RecordingSolver:
        calls = 0

        def solve(self, *args, **kwargs):
            self.calls += 1
            return real_solver.solve(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(real_solver, name)

    solver = RecordingSolver()
    core._belief_solver = solver
    state = make_exact_belief_state()
    prior_state = state.model_dump(mode="json")
    prior_memory = state.evidence_memory.model_dump(mode="json")
    cycle = make_cycle(f"cycle_closed_signal_{case}")

    with pytest.raises(
        ValueError,
        match="native closed signal ownership is invalid",
    ) as exc_info:
        core.integrate_cycle(
            cycle=cycle,
            belief_state=state,
            probe_set=make_empty_probe_set(cycle.cycle_id),
            signals=signals,
        )

    assert transition_calls == 0
    assert solver.calls == 0
    assert state.model_dump(mode="json") == prior_state
    assert state.evidence_memory.model_dump(mode="json") == prior_memory
    assert ledger_path.read_bytes() == b""
    return str(exc_info.value)


@pytest.mark.parametrize("case", ["none", "empty"])
@pytest.mark.parametrize("secret_value", [_NFKC_SECRET_VALUE, "api_key=secret-value-123"])
def test_native_missing_closed_signals_reject_secret_input_atomically(
    tmp_path: Path,
    case: str,
    secret_value: str,
):
    error_text = _assert_closed_signal_ownership_rejected(
        tmp_path,
        case=case,
        signals=[
            make_active_signal().model_copy(
                update={"raw_content": secret_value}
            )
        ],
    )

    assert secret_value not in error_text
    assert unicodedata.normalize("NFKC", secret_value) not in error_text


@pytest.mark.parametrize(
    "case",
    ["missing", "extra", "reordered", "changed", "noncanonical_provenance"],
)
def test_native_closed_signal_mismatch_fails_before_transition(
    tmp_path: Path,
    case: str,
):
    _assert_closed_signal_ownership_rejected(
        tmp_path,
        case=case,
        signals=[
            make_active_signal().model_copy(update={"id": "S_closed_first"}),
            make_active_signal().model_copy(
                update={
                    "id": "S_closed_second",
                    "raw_content": "A distinct second closed signal.",
                }
            ),
        ],
    )


def test_native_owned_signals_with_zero_events_fail_in_transition(tmp_path: Path):
    ledger_path = tmp_path / "owned-signals-zero-events.jsonl"
    ledger_path.touch()
    core = ClosedSignalOwnershipCore(
        "valid",
        ledger=JsonlLedgerStore(ledger_path),
    )
    real_solver = core._belief_solver

    class RecordingSolver:
        calls = 0

        def solve(self, *args, **kwargs):
            self.calls += 1
            return real_solver.solve(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(real_solver, name)

    solver = RecordingSolver()
    core._belief_solver = solver
    state = make_exact_belief_state()
    cycle = make_cycle("cycle_owned_signals_zero_events")

    with pytest.raises(ValueError, match="native evidence memory transition"):
        core.integrate_cycle(
            cycle=cycle,
            belief_state=state,
            probe_set=make_empty_probe_set(cycle.cycle_id),
            signals=[make_active_signal()],
        )

    assert solver.calls == 0
    assert ledger_path.read_bytes() == b""


@pytest.mark.parametrize(
    "secret_value",
    [_NFKC_SECRET_VALUE, "Authorization: Bearer provider-secret-value-123"],
)
def test_explicit_legacy_plain_list_gate_never_ledgers_raw_secrets(
    tmp_path: Path,
    secret_value: str,
):
    class LegacyPlainListGate(EvidenceIntegrationGate):
        def integrate(self, *, cycle, belief_state, probe_set, signals):
            return []

    class LegacyPlainListCore(BayesProbeCore):
        def _create_evidence_integration_gate(self):
            return LegacyPlainListGate()

    ledger_path = tmp_path / "legacy-raw-secret.jsonl"
    ledger_path.touch()
    core = LegacyPlainListCore(ledger=JsonlLedgerStore(ledger_path))
    real_solver = core._belief_solver

    class RecordingSolver:
        calls = 0

        def solve(self, *args, **kwargs):
            self.calls += 1
            return real_solver.solve(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(real_solver, name)

    solver = RecordingSolver()
    core._belief_solver = solver
    state = make_belief_state(cycle_id="cycle_0")
    prior_state = state.model_dump(mode="json")
    cycle = make_cycle("cycle_legacy_raw_secret")

    with pytest.raises(
        ValueError,
        match="legacy closed signals contain secret material",
    ) as exc_info:
        core.integrate_cycle(
            cycle=cycle,
            belief_state=state,
            probe_set=make_empty_probe_set(cycle.cycle_id),
            signals=[
                make_active_signal().model_copy(
                    update={"raw_content": secret_value}
                )
            ],
        )

    error_text = str(exc_info.value)
    assert secret_value not in error_text
    assert unicodedata.normalize("NFKC", secret_value) not in error_text
    assert solver.calls == 0
    assert state.model_dump(mode="json") == prior_state
    assert ledger_path.read_bytes() == b""


@pytest.mark.parametrize(
    ("prior_used_credit", "declared_weight", "resulting_used_credit"),
    [
        pytest.param(0.25, 0.9, 1.15, id="inflated-weight"),
        pytest.param(1.0, 0.2, 1.2, id="saturated-over-cap"),
    ],
)
def test_self_consistent_inflated_credit_transition_fails_atomically(
    tmp_path: Path,
    prior_used_credit: float,
    declared_weight: float,
    resulting_used_credit: float,
):
    state = make_exact_belief_state()
    signal = evidence_memory.SignalProvenanceNormalizer().normalize(
        _memory_signal(
            "S_forged_credit",
            "A fresh audit favors H1.",
            root="root-forged-credit",
        ),
        run_id=state.run_id,
    )
    group = signal.provenance.correlation_group
    prior_memory = state.evidence_memory.model_copy(
        update={
            "correlation_credit": {
                f"{group}|H1|confirming": prior_used_credit
            }
        }
    )
    state = state.model_copy(update={"evidence_memory": prior_memory})
    event = _policy_transition_event(
        signal,
        event_id="E_forged_credit",
        correlation_status="correlated_novel",
        effective_update_weight=declared_weight,
    )
    candidate = _commit_self_declared_transition(
        prior_memory,
        signal=signal,
        event=event,
        resulting_used_credit=resulting_used_credit,
    )
    assert candidate.correlation_credit[
        f"{group}|H1|confirming"
    ] == pytest.approx(resulting_used_credit)

    _assert_supplied_transition_rejected_atomically(
        tmp_path,
        name=f"self_consistent_credit_{prior_used_credit}",
        state=state,
        integration=EvidenceIntegrationResult(
            evidence_events=[event],
            probe_candidates=[],
            evidence_memory=candidate,
            normalized_signals=[signal],
        ),
    )


@pytest.mark.parametrize("repeat_kind", ["exact", "same_root"])
def test_repeat_mislabeled_novel_with_positive_weight_fails_atomically(
    tmp_path: Path,
    repeat_kind: str,
):
    state = make_exact_belief_state()
    normalizer = evidence_memory.SignalProvenanceNormalizer()
    prior_signal = normalizer.normalize(
        _memory_signal(
            "S_repeat_prior",
            "A stable audited observation.",
            root="root-repeat-policy",
        ),
        run_id=state.run_id,
    )
    prior_memory = evidence_memory.EvidenceMemoryManager().remember_signal_identity(
        state.evidence_memory,
        prior_signal,
    )
    current_raw = _memory_signal(
        "S_repeat_current",
        (
            prior_signal.raw_content
            if repeat_kind == "exact"
            else "A differently worded result from the same factual root."
        ),
        root="root-repeat-policy",
    )
    current_signal = normalizer.normalize(current_raw, run_id=state.run_id)
    state = state.model_copy(update={"evidence_memory": prior_memory})
    quality = None
    declared_weight = 0.2
    if repeat_kind == "exact":
        quality = {
            "reliability": 0.8,
            "independence": 0.25,
            "relevance": 0.9,
            "novelty": 0.25,
            "specificity": 0.7,
            "verifiability": 0.7,
        }
        declared_weight = 0.045
    event = _policy_transition_event(
        current_signal,
        event_id=f"E_mislabeled_{repeat_kind}",
        correlation_status="novel",
        effective_update_weight=declared_weight,
        quality=quality,
    )
    candidate = _commit_self_declared_transition(
        prior_memory,
        signal=current_signal,
        event=event,
        resulting_used_credit=declared_weight,
    )

    _assert_supplied_transition_rejected_atomically(
        tmp_path,
        name=f"mislabeled_{repeat_kind}",
        state=state,
        integration=EvidenceIntegrationResult(
            evidence_events=[event],
            probe_candidates=[],
            evidence_memory=candidate,
            normalized_signals=[current_signal],
        ),
    )


def test_inflated_model_origin_quality_and_matching_memory_fail_atomically(
    tmp_path: Path,
):
    state = make_exact_belief_state()
    signal = evidence_memory.SignalProvenanceNormalizer().normalize(
        ExternalSignal(
            id="S_inflated_model_quality",
            cycle_id="pending",
            signal_kind=SignalKind.ACTIVE,
            source_type="model_probe_gateway",
            source="model_gateway:scripted",
            raw_content="A model comparison favors H1.",
            initial_target_hypotheses=["H1", "H2"],
        ),
        run_id=state.run_id,
    )
    inflated_quality = {
        "reliability": 0.9,
        "independence": 0.9,
        "relevance": 0.9,
        "novelty": 0.9,
        "specificity": 0.9,
        "verifiability": 0.9,
    }
    declared_weight = 0.9**4
    event = _policy_transition_event(
        signal,
        event_id="E_inflated_model_quality",
        correlation_status="novel",
        effective_update_weight=declared_weight,
        quality=inflated_quality,
    )
    candidate = _commit_self_declared_transition(
        state.evidence_memory,
        signal=signal,
        event=event,
        resulting_used_credit=declared_weight,
    )

    _assert_supplied_transition_rejected_atomically(
        tmp_path,
        name="inflated_model_quality",
        state=state,
        integration=EvidenceIntegrationResult(
            evidence_events=[event],
            probe_candidates=[],
            evidence_memory=candidate,
            normalized_signals=[signal],
        ),
    )


def test_valid_lower_model_quality_transition_is_accepted(tmp_path: Path):
    state = make_exact_belief_state()
    signal = evidence_memory.SignalProvenanceNormalizer().normalize(
        ExternalSignal(
            id="S_lower_model_quality",
            cycle_id="pending",
            signal_kind=SignalKind.ACTIVE,
            source_type="model_probe_gateway",
            source="model_gateway:scripted",
            raw_content="A cautious model comparison favors H1.",
            initial_target_hypotheses=["H1", "H2"],
        ),
        run_id=state.run_id,
    )
    lower_quality = {
        "reliability": 0.4,
        "independence": 0.2,
        "relevance": 0.7,
        "novelty": 0.3,
        "specificity": 0.5,
        "verifiability": 0.2,
    }
    base_weight = 0.4 * 0.2 * 0.7 * 0.3
    manager = evidence_memory.EvidenceMemoryManager()
    decision = manager.classify(
        state.evidence_memory,
        signal,
        likelihoods={
            "H1": LikelihoodBand.MODERATELY_CONFIRMING,
            "H2": LikelihoodBand.NEUTRAL,
        },
        unresolved_likelihood=LikelihoodBand.NEUTRAL,
        frame_version=state.frame_state.frame_version,
        base_effective_weight=base_weight,
    )
    event = _policy_transition_event(
        signal,
        event_id="E_lower_model_quality",
        correlation_status=decision.correlation_status,
        effective_update_weight=decision.effective_update_weight,
        quality=lower_quality,
    )
    candidate = manager.commit(
        state.evidence_memory,
        signal=signal,
        event=event,
        decision=decision,
    )
    ledger = JsonlLedgerStore(tmp_path / "lower-model-quality.jsonl")
    result = SuppliedIntegrationCore(
        EvidenceIntegrationResult(
            evidence_events=[event],
            probe_candidates=[],
            evidence_memory=candidate,
            normalized_signals=[signal],
        ),
        ledger=ledger,
    ).integrate_cycle(
        cycle=make_cycle("cycle_lower_model_quality"),
        belief_state=state,
        probe_set=make_empty_probe_set("cycle_lower_model_quality"),
        signals=[make_active_signal()],
    )

    assert result.evidence_events == [event]
    assert result.belief_state.evidence_memory == candidate


def _existing_binding_transition_fixture():
    state = make_exact_belief_state()
    signal = evidence_memory.SignalProvenanceNormalizer().normalize(
        _memory_signal(
            "S_binding_prior",
            "The bound historical audit favors H1.",
            root="root-binding-prior",
        ),
        run_id=state.run_id,
    )
    manager = evidence_memory.EvidenceMemoryManager()
    quality = {
        "reliability": 0.8,
        "independence": 0.8,
        "relevance": 0.9,
        "novelty": 0.8,
        "specificity": 0.7,
        "verifiability": 0.7,
    }
    base_weight = 0.8 * 0.8 * 0.9 * 0.8
    decision = manager.classify(
        state.evidence_memory,
        signal,
        likelihoods={
            "H1": LikelihoodBand.MODERATELY_CONFIRMING,
            "H2": LikelihoodBand.NEUTRAL,
        },
        unresolved_likelihood=LikelihoodBand.NEUTRAL,
        frame_version=state.frame_state.frame_version,
        base_effective_weight=base_weight,
    )
    event = _policy_transition_event(
        signal,
        event_id="E_binding_prior",
        correlation_status=decision.correlation_status,
        effective_update_weight=decision.effective_update_weight,
        quality=quality,
    )
    memory = manager.commit(
        state.evidence_memory,
        signal=signal,
        event=event,
        decision=decision,
    )
    payload = state.model_dump(mode="python")
    payload.update(
        {
            "evidence_memory": memory.model_dump(mode="python"),
            "ledger_refs": {
                **state.ledger_refs,
                "evidence_events": [event.id],
            },
        }
    )
    return BeliefState.model_validate(payload), signal, event


@pytest.mark.parametrize("changed_identity", ["content", "root"])
def test_existing_event_changed_signal_binding_fails_atomically(
    tmp_path: Path,
    changed_identity: str,
):
    state, prior_signal, prior_event = _existing_binding_transition_fixture()
    raw_updates = {"id": f"S_binding_changed_{changed_identity}"}
    provenance_updates = {}
    if changed_identity == "content":
        raw_updates["raw_content"] = "A materially changed historical audit."
    else:
        provenance_updates["derivation_root_id"] = "root-binding-changed"
    raw_updates["provenance"] = prior_signal.provenance.model_copy(
        update=provenance_updates
    )
    changed_signal = evidence_memory.SignalProvenanceNormalizer().normalize(
        prior_signal.model_copy(update=raw_updates),
        run_id=state.run_id,
    )
    replay_event = EvidenceEvent.model_validate(
        {
            **prior_event.model_dump(mode="python"),
            "derived_from_signal": changed_signal.id,
            "epistemic_origin": changed_signal.provenance.epistemic_origin,
            "derivation_root_id": changed_signal.provenance.derivation_root_id,
            "evidence_type": EvidenceType.NEUTRAL,
            "likelihoods": {
                "H1": LikelihoodBand.NEUTRAL,
                "H2": LikelihoodBand.NEUTRAL,
            },
            "correlation_status": "duplicate_exact",
            "effective_update_weight": 0.0,
            "discard_reason": "duplicate evidence event id",
        }
    )
    candidate = evidence_memory.EvidenceMemoryManager().remember_signal_identity(
        state.evidence_memory,
        changed_signal,
    )

    _assert_supplied_transition_rejected_atomically(
        tmp_path,
        name=f"changed_existing_binding_{changed_identity}",
        state=state,
        integration=EvidenceIntegrationResult(
            evidence_events=[replay_event],
            probe_candidates=[],
            evidence_memory=candidate,
            normalized_signals=[changed_signal],
        ),
    )


def test_existing_event_missing_historical_binding_fails_atomically(
    tmp_path: Path,
):
    state, signal, prior_event = _existing_binding_transition_fixture()
    memory_without_binding = state.evidence_memory.model_copy(
        update={"event_signal_identity_digests": {}}
    )
    state = BeliefState.model_validate(
        {
            **state.model_dump(mode="python"),
            "evidence_memory": memory_without_binding.model_dump(mode="python"),
        }
    )
    replay_event = EvidenceEvent.model_validate(
        {
            **prior_event.model_dump(mode="python"),
            "evidence_type": EvidenceType.NEUTRAL,
            "likelihoods": {
                "H1": LikelihoodBand.NEUTRAL,
                "H2": LikelihoodBand.NEUTRAL,
            },
            "correlation_status": "duplicate_exact",
            "effective_update_weight": 0.0,
            "discard_reason": "duplicate evidence event id",
        }
    )

    _assert_supplied_transition_rejected_atomically(
        tmp_path,
        name="missing_existing_binding",
        state=state,
        integration=EvidenceIntegrationResult(
            evidence_events=[replay_event],
            probe_candidates=[],
            evidence_memory=memory_without_binding,
            normalized_signals=[signal],
        ),
    )


def _native_event_contract_case(*, replay: bool, malformed: str):
    state = make_exact_belief_state()
    signal = evidence_memory.SignalProvenanceNormalizer().normalize(
        make_active_signal().model_copy(update={"id": "S_native_contract"}),
        run_id=state.run_id,
    )
    manager = evidence_memory.EvidenceMemoryManager()
    decision = manager.classify(
        EvidenceMemorySnapshot(),
        signal,
        likelihoods={
            "H1": LikelihoodBand.NEUTRAL,
            "H2": LikelihoodBand.NEUTRAL,
        },
        unresolved_likelihood=LikelihoodBand.NEUTRAL,
        base_effective_weight=0.0,
    )
    valid_event = EvidenceEvent(
        schema_version="v0.2",
        id="E_native_contract",
        derived_from_signal=signal.id,
        epistemic_origin=signal.provenance.epistemic_origin,
        derivation_root_id=signal.provenance.derivation_root_id,
        target_hypotheses=["H1", "H2"],
        evidence_type=EvidenceType.NEUTRAL,
        content=signal.raw_content,
        likelihoods={
            "H1": LikelihoodBand.NEUTRAL,
            "H2": LikelihoodBand.NEUTRAL,
        },
        unresolved_likelihood=LikelihoodBand.NEUTRAL,
        frame_fit=FrameFit.UNDERDETERMINED,
        correlation_status=decision.correlation_status,
        effective_update_weight=0.0,
    )
    committed = manager.commit(
        EvidenceMemorySnapshot(),
        signal=signal,
        event=valid_event,
        decision=decision,
    )
    if malformed == "legacy_v01":
        event = valid_event.model_copy(update={"schema_version": "v0.1"})
    elif malformed == "missing_effective_weight":
        event = valid_event.model_copy(update={"effective_update_weight": None})
    else:
        raise AssertionError(f"unknown native event defect: {malformed}")

    if replay:
        state_payload = state.model_dump(mode="python")
        state_payload.update(
            {
                "evidence_memory": committed.model_dump(mode="python"),
                "ledger_refs": {
                    **state.ledger_refs,
                    "evidence_events": [valid_event.id],
                },
            }
        )
        state = BeliefState.model_validate(state_payload)
    return state, EvidenceIntegrationResult(
        evidence_events=[event],
        probe_candidates=[],
        evidence_memory=committed,
        normalized_signals=[signal],
    )


@pytest.mark.parametrize("replay", [False, True], ids=["new", "replay"])
@pytest.mark.parametrize(
    "malformed",
    ["legacy_v01", "missing_effective_weight"],
)
def test_native_event_contract_fails_before_solver_or_ledger(
    tmp_path: Path,
    replay: bool,
    malformed: str,
):
    state, integration = _native_event_contract_case(
        replay=replay,
        malformed=malformed,
    )
    ledger_path = tmp_path / f"native-event-{malformed}-{replay}.jsonl"
    ledger_path.touch()
    core = SuppliedIntegrationCore(
        integration,
        ledger=JsonlLedgerStore(ledger_path),
    )
    prior_state = state.model_dump(mode="json")

    with pytest.raises(ValueError, match="native evidence event contract"):
        core.integrate_cycle(
            cycle=make_cycle(f"cycle_native_event_{malformed}_{replay}"),
            belief_state=state,
            probe_set=make_empty_probe_set(
                f"cycle_native_event_{malformed}_{replay}"
            ),
            signals=[make_active_signal()],
        )

    assert state.model_dump(mode="json") == prior_state
    assert ledger_path.read_bytes() == b""


@pytest.mark.parametrize(
    "invalid_envelope",
    [
        "missing_task_frame",
        "tag_only",
        "forged_recognized_marker",
        "transferred_receipt",
        "v01_task_frame",
        "missing_trace",
        "fake_trace",
        "missing_frame_state",
        "missing_evidence_memory",
        "incoherent_frame_state",
    ],
)
def test_invalid_lifecycle_fails_before_provider_or_cycle_ledger_append(
    tmp_path: Path,
    invalid_envelope: str,
):
    gateway = ScriptedModelGateway(responses={"judge_evidence": _native_open_judgment()})
    ledger_path = tmp_path / "invalid-lifecycle-ledger.jsonl"
    ledger_path.touch()
    native = make_exact_belief_state()
    migrated = make_explicit_legacy_belief_state(cycle_id="cycle_0")
    if invalid_envelope == "missing_task_frame":
        state = native.model_copy(update={"task_frame": None})
    elif invalid_envelope == "tag_only":
        state = native.model_copy(
            update={
                "task_frame": native.task_frame.model_copy(
                    update={"framing_method": FramingMethod.LEGACY_MIGRATION}
                )
            }
        )
    elif invalid_envelope == "forged_recognized_marker":
        state = native.model_copy(
            update={
                "task_frame": native.task_frame.model_copy(
                    update={
                        "framing_method": FramingMethod.LEGACY_MIGRATION,
                        "framing_trace": {
                            **native.task_frame.framing_trace,
                            "migration": "belief_state_v0.1_to_v0.2",
                        },
                    }
                )
            }
        )
    elif invalid_envelope == "transferred_receipt":
        forged_native = native.model_copy(
            update={
                "task_frame": native.task_frame.model_copy(
                    update={
                        "framing_method": FramingMethod.LEGACY_MIGRATION,
                        "framing_trace": {
                            "migration": "belief_state_v0.1_to_v0.2"
                        },
                    }
                )
            }
        )
        state = migrated.model_copy(
            update={
                field_name: getattr(forged_native, field_name)
                for field_name in BeliefState.model_fields
            }
        )
    elif invalid_envelope == "v01_task_frame":
        state = migrated.model_copy(
            update={
                "task_frame": migrated.task_frame.model_copy(
                    update={"schema_version": "v0.1"}
                )
            }
        )
    elif invalid_envelope == "missing_trace":
        state = migrated.model_copy(
            update={
                "task_frame": migrated.task_frame.model_copy(
                    update={"framing_trace": {}}
                )
            }
        )
    elif invalid_envelope == "fake_trace":
        state = migrated.model_copy(
            update={
                "task_frame": migrated.task_frame.model_copy(
                    update={"framing_trace": {"migration": "caller_asserted"}}
                )
            }
        )
    elif invalid_envelope == "missing_frame_state":
        state = migrated.model_copy(update={"frame_state": None})
    elif invalid_envelope == "missing_evidence_memory":
        state = migrated.model_copy(update={"evidence_memory": None})
    elif invalid_envelope == "incoherent_frame_state":
        state = migrated.model_copy(
            update={
                "frame_state": migrated.frame_state.model_copy(
                    update={"frame_id": "mismatched_frame"}
                )
            }
        )
    else:
        raise AssertionError(f"unknown invalid lifecycle envelope: {invalid_envelope}")
    prior_state = state.model_dump(mode="json")
    core = BayesProbeCore(ledger=JsonlLedgerStore(ledger_path), model_gateway=gateway)
    cycle = make_cycle("cycle_invalid_lifecycle")

    with pytest.raises(ValueError, match="invalid belief lifecycle"):
        core.integrate_cycle(
            cycle=cycle,
            belief_state=state,
            probe_set=make_empty_probe_set(cycle.cycle_id),
            signals=[make_active_signal()],
        )

    assert gateway.requests == []
    assert state.model_dump(mode="json") == prior_state
    assert ledger_path.read_bytes() == b""


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
        epistemic_origin="tool_result",
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
    projection_bindings = (
        result.belief_state.evidence_memory.event_signal_identity_digests
    )
    assert set(projection_bindings) == {sender_event.id, source_event.id}
    assert len(set(projection_bindings.values())) == 1
    assert sender_event.likelihoods["H2"] == LikelihoodBand.WEAKLY_CONFIRMING
    assert sender_event.likelihoods["H1"] == LikelihoodBand.NEUTRAL
    assert set(source_event.likelihoods.values()) == {LikelihoodBand.NEUTRAL}
    assert sender_event.reliability == 0.55
    assert sender_event.independence == 0.45
    assert source_event.reliability == 0.5
    assert source_event.independence == 0.45
    assert source_event.verifiability == 0.4
    assert sender_event.unresolved_likelihood is None
    assert source_event.unresolved_likelihood is None
    assert candidate.candidate_id == "pc_run_1_cycle_projection_E1_source_verify_source"
    assert candidate.source == "passive_signal"
    assert candidate.candidate_probe.id == "P_run_1_cycle_projection_E1_source_verify_source"
    assert candidate.candidate_probe.method == "source_tracing"
    assert candidate.candidate_probe.cycle_id == "cycle_projection"
    assert candidate.candidate_probe.target_hypotheses == ["H1", "H2"]


def test_native_exclusive_open_projection_events_are_unresolved_neutral():
    core = BayesProbeCore()
    belief_state = make_exact_belief_state()
    hypothesis_ids = [hypothesis.id for hypothesis in belief_state.hypotheses]
    cycle = CycleRecord(
        cycle_id="cycle_native_projection",
        run_id="run_1",
        cycle_index=1,
        signal_shape=CycleSignalShape.PASSIVE_ONLY,
    )

    result = core.integrate_cycle(
        cycle=cycle,
        belief_state=belief_state,
        probe_set=ProbeSet(
            probe_set_id="ps_native_projection",
            cycle_id=cycle.cycle_id,
            probes=[],
            selection_reason="Native projection contract regression.",
            may_be_empty=True,
        ),
        signals=[
            ExternalSignal(
                id="S_native_projection",
                cycle_id="pending",
                signal_kind=SignalKind.PASSIVE,
                source_type="external_agent_projection",
                source="agent_a",
                raw_content="Agent A cites Source X while endorsing one named value.",
                initial_target_hypotheses=hypothesis_ids,
            )
        ],
    )

    assert [event.evidence_type for event in result.evidence_events] == [
        EvidenceType.SENDER_JUDGMENT,
        EvidenceType.SOURCE_CLAIM,
    ]
    assert all(event.schema_version == "v0.2" for event in result.evidence_events)
    assert all(
        event.unresolved_likelihood == LikelihoodBand.NEUTRAL
        for event in result.evidence_events
    )
    assert all(
        event.frame_fit == FrameFit.UNDERDETERMINED
        for event in result.evidence_events
    )
    assert result.evidence_events[0].reliability == 0.55
    assert result.evidence_events[1].verifiability == 0.4
    assert len(result.probe_candidates) == 1
    assert result.probe_candidates[0].candidate_probe.method == "source_tracing"


def test_unicode_normalized_secret_fails_before_provider_memory_or_ledger(tmp_path):
    ledger = JsonlLedgerStore(tmp_path / "unicode-secret-ledger.jsonl")
    gateway = ScriptedModelGateway(
        {
            "judge_evidence": {
                "evidence_type": "neutral",
                "likelihoods": {"H1": "neutral", "H2": "neutral"},
                "interpretation": "This response must never be requested.",
                "quality_overrides": {},
            }
        }
    )
    core = BayesProbeCore(ledger=ledger, model_gateway=gateway)
    belief_state = make_belief_state(cycle_id="cycle_0")
    cycle = CycleRecord(
        cycle_id="cycle_unicode_secret",
        run_id="run_1",
        cycle_index=1,
        signal_shape=CycleSignalShape.ACTIVE_ONLY,
    )
    signal = ExternalSignal(
        id="S_unicode_secret_atomic",
        cycle_id="pending",
        signal_kind=SignalKind.ACTIVE,
        source_type="human_input",
        source="human",
        raw_content=(
            "\uff21\uff55\uff54\uff48\uff4f\uff52\uff49\uff5a\uff41\uff54\uff49\uff4f\uff4e\uff1a "
            "\uff22\uff45\uff41\uff52\uff45\uff52 provider-secret-value-123"
        ),
        initial_target_hypotheses=["H1", "H2"],
    )

    with pytest.raises(ValueError, match="secret"):
        core.integrate_cycle(
            cycle=cycle,
            belief_state=belief_state,
            probe_set=ProbeSet(
                probe_set_id="ps_unicode_secret",
                cycle_id=cycle.cycle_id,
                probes=[],
                selection_reason="Secret atomicity regression.",
                may_be_empty=True,
            ),
            signals=[signal],
        )

    assert signal.provenance is None
    assert belief_state.evidence_memory is None
    assert gateway.requests == []
    assert ledger.read_all() == []


@pytest.mark.parametrize(
    "location",
    [
        "id",
        "cycle_id",
        "generated_by_probe",
        "initial_target_hypotheses",
        "source_type",
        "source",
        "raw_content",
        "provenance.source_identity",
        "provenance.provider_model_or_tool_identity",
        "provenance.session_id",
        "provenance.parent_signal_ids",
        "provenance.derivation_root_id",
        "provenance.correlation_group",
        "provenance.supplied_correlation_group",
        "provenance.canonical_content_fingerprint",
        "provenance.citations",
        "provenance.artifact_refs",
        "provenance.environment_state_id",
    ],
)
def test_nfkc_secret_anywhere_in_signal_fails_atomically(
    tmp_path,
    location,
):
    normalized_secret = unicodedata.normalize("NFKC", _NFKC_SECRET_VALUE)
    gateway = ScriptedModelGateway(
        responses={"judge_evidence": _native_open_judgment()}
    )
    ledger_path = tmp_path / f"recursive-secret-{location.replace('.', '-')}.jsonl"
    ledger_path.touch()
    core = BayesProbeCore(
        ledger=JsonlLedgerStore(ledger_path),
        model_gateway=gateway,
    )
    state = make_exact_belief_state()
    signal = ExternalSignal(
        id="S_recursive_secret",
        cycle_id="pending",
        signal_kind=SignalKind.ACTIVE,
        source_type="retrieved_source",
        source="source.example/audit",
        raw_content="An ordinary audited observation.",
        generated_by_probe="P_recursive_secret",
        initial_target_hypotheses=["H1", "H2"],
        provenance=SignalProvenance(
            epistemic_origin=EpistemicOrigin.RETRIEVED_SOURCE,
            source_identity="source.example/audit",
            provider_model_or_tool_identity="retriever/audit",
            session_id="session-recursive-secret",
            parent_signal_ids=["S_external_parent"],
            derivation_root_id="root-recursive-secret",
            correlation_group="source.example/audit",
            supplied_correlation_group="caller-audit-group",
            canonical_content_fingerprint="normalize-me",
            citations=["source.example/audit#finding"],
            artifact_refs=["artifact-audit-1"],
            environment_state_id="environment-audit-1",
        ),
    )
    cycle_id = "cycle_recursive_secret"
    if location == "cycle_id":
        cycle_id = _NFKC_SECRET_VALUE
    elif location == "initial_target_hypotheses":
        signal = signal.model_copy(
            update={location: ["H1", _NFKC_SECRET_VALUE]}
        )
    elif location.startswith("provenance."):
        provenance_field = location.removeprefix("provenance.")
        provenance_value = (
            [_NFKC_SECRET_VALUE]
            if provenance_field in {"parent_signal_ids", "citations", "artifact_refs"}
            else _NFKC_SECRET_VALUE
        )
        signal = signal.model_copy(
            update={
                "provenance": signal.provenance.model_copy(
                    update={provenance_field: provenance_value}
                )
            }
        )
    else:
        signal = signal.model_copy(update={location: _NFKC_SECRET_VALUE})
    cycle = CycleRecord(
        cycle_id=cycle_id,
        run_id=state.run_id,
        cycle_index=1,
        signal_shape=CycleSignalShape.ACTIVE_ONLY,
    )
    prior_state = state.model_dump(mode="json")
    prior_memory = state.evidence_memory.model_dump(mode="json")
    prior_ledger = ledger_path.read_bytes()

    with pytest.raises(ValueError) as exc_info:
        core.integrate_cycle(
            cycle=cycle,
            belief_state=state,
            probe_set=make_empty_probe_set(cycle.cycle_id),
            signals=[signal],
        )

    error_text = str(exc_info.value)
    assert "secret" in error_text.casefold()
    assert _NFKC_SECRET_VALUE not in error_text
    assert normalized_secret not in error_text
    assert gateway.requests == []
    assert state.model_dump(mode="json") == prior_state
    assert state.evidence_memory.model_dump(mode="json") == prior_memory
    assert ledger_path.read_bytes() == prior_ledger


@pytest.mark.parametrize(
    ("run_id", "cycle_id"),
    [
        (" run_1", "cycle_event_id"),
        ("run_1", "cycle_event_id "),
        (f"run-{_NFKC_SECRET_VALUE}", "cycle_event_id"),
    ],
)
def test_noncanonical_planned_event_namespace_fails_atomically(
    tmp_path,
    run_id,
    cycle_id,
):
    normalized_secret = unicodedata.normalize("NFKC", _NFKC_SECRET_VALUE)
    gateway = ScriptedModelGateway(
        responses={"judge_evidence": _native_open_judgment()}
    )
    ledger_path = tmp_path / "noncanonical-event-id.jsonl"
    ledger_path.touch()
    core = BayesProbeCore(
        ledger=JsonlLedgerStore(ledger_path),
        model_gateway=gateway,
    )
    state = make_exact_belief_state().model_copy(update={"run_id": run_id})
    cycle = CycleRecord(
        cycle_id=cycle_id,
        run_id=run_id,
        cycle_index=1,
        signal_shape=CycleSignalShape.ACTIVE_ONLY,
    )
    prior_state = state.model_dump(mode="json")
    prior_memory = state.evidence_memory.model_dump(mode="json")
    prior_ledger = ledger_path.read_bytes()

    with pytest.raises(ValueError, match="canonical event binding id") as exc_info:
        core.integrate_cycle(
            cycle=cycle,
            belief_state=state,
            probe_set=make_empty_probe_set(cycle.cycle_id),
            signals=[
                ExternalSignal(
                    id="S_noncanonical_event_id",
                    cycle_id="pending",
                    signal_kind=SignalKind.ACTIVE,
                    source_type="retrieved_source",
                    source="source.example/audit",
                    raw_content="An event-id validation observation.",
                    initial_target_hypotheses=["H1", "H2"],
                )
            ],
        )

    error_text = str(exc_info.value)
    assert _NFKC_SECRET_VALUE not in error_text
    assert normalized_secret not in error_text
    assert gateway.requests == []
    assert state.model_dump(mode="json") == prior_state
    assert state.evidence_memory.model_dump(mode="json") == prior_memory
    assert ledger_path.read_bytes() == prior_ledger


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


def _derived_memory_signal(
    signal_id: str,
    content: str,
    *,
    parent_id: str,
    root: str,
) -> ExternalSignal:
    return ExternalSignal(
        id=signal_id,
        cycle_id="pending",
        signal_kind=SignalKind.ACTIVE,
        source_type="derived_summary",
        source="summary-worker",
        raw_content=content,
        initial_target_hypotheses=["H1", "H2"],
        provenance=SignalProvenance(
            epistemic_origin=EpistemicOrigin.DERIVED_SUMMARY,
            source_identity="summary-worker",
            parent_signal_ids=[parent_id],
            derivation_root_id=root,
            correlation_group="summary-worker",
            canonical_content_fingerprint="normalize-me",
        ),
    )


def _model_memory_signal(
    signal_id: str,
    content: str,
    *,
    root: str,
    supplied_group: str,
) -> ExternalSignal:
    return ExternalSignal(
        id=signal_id,
        cycle_id="pending",
        signal_kind=SignalKind.ACTIVE,
        source_type="custom_model_adapter",
        source="provider/model-a",
        raw_content=content,
        initial_target_hypotheses=["H1", "H2"],
        provenance=SignalProvenance(
            epistemic_origin=EpistemicOrigin.MODEL_REASONING,
            source_identity="provider/model-a",
            provider_model_or_tool_identity="provider/model-a",
            session_id="session-1",
            derivation_root_id=root,
            correlation_group=supplied_group,
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


def _legacy_evidence_judgment():
    return {
        "evidence_type": "supporting",
        "likelihoods": {
            "H1": "moderately_confirming",
            "H2": "moderately_disconfirming",
        },
        "interpretation": "The audit favors H1.",
        "quality_overrides": {},
    }


class _InPlaceCoreInputMutationGate:
    def __init__(self, delegate, mutation: str) -> None:
        self._delegate = delegate
        self._mutation = mutation

    def integrate(self, *, cycle, belief_state, probe_set, signals):
        if self._mutation == "signal":
            _mutate_gate_signal(signals[0], "first")
        elif self._mutation == "later_signal":
            _mutate_gate_signal(signals[-1], "later")
        elif self._mutation == "belief_state":
            memory = belief_state.evidence_memory
            assert memory is not None
            memory.accepted_evidence_ids.clear()
            memory.content_fingerprints.clear()
            memory.source_content_fingerprints.clear()
            memory.derivation_roots.clear()
            memory.event_signal_identity_digests.clear()
            memory.correlation_credit.clear()
            memory.discovery_evidence_ids.clear()
            memory.counterevidence_ids_by_hypothesis.clear()
            memory.discard_and_schema_history.clear()
            belief_state.ledger_refs.clear()

        integration = self._delegate.integrate(
            cycle=cycle,
            belief_state=belief_state,
            probe_set=probe_set,
            signals=signals,
        )
        if self._mutation == "cycle_probe":
            cycle.cycle_id = "cycle_gate_mutated"
            cycle.boundary_status = BoundaryStatus.OPEN
            cycle.boundary_closed_at = None
            probe_set.probe_set_id = "ps_gate_mutated"
            probe_set.cycle_id = "cycle_gate_mutated"
            probe_set.probes[0].id = "P_gate_mutated"
            probe_set.probes[0].cycle_id = "cycle_gate_mutated"
            probe_set.probes[0].inquiry_goal = "Gate-mutated inquiry."
        return integration


class _GateInputMutationCore(BayesProbeCore):
    def __init__(self, mutation: str, *, ledger, model_gateway) -> None:
        self._input_mutation = mutation
        super().__init__(ledger=ledger, model_gateway=model_gateway)

    def _create_evidence_integration_gate(self):
        return _InPlaceCoreInputMutationGate(
            super()._create_evidence_integration_gate(),
            self._input_mutation,
        )


class _RecordingSolverProxy:
    def __init__(self, solver) -> None:
        self._solver = solver
        self.calls = 0

    def solve(self, *args, **kwargs):
        self.calls += 1
        return self._solver.solve(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._solver, name)


def _mutate_gate_signal(signal: ExternalSignal, label: str) -> None:
    provenance = signal.provenance
    assert provenance is not None
    signal.raw_content = f"Gate-mutated {label} signal content."
    signal.source = f"gate-mutated-{label}-source"
    provenance.source_identity = f"gate-mutated-{label}-identity"
    provenance.derivation_root_id = f"gate-mutated-{label}-root"
    provenance.correlation_group = f"gate-mutated-{label}-group"
    provenance.supplied_correlation_group = f"gate-mutated-{label}-group"


def _mutating_core(tmp_path: Path, mutation: str):
    ledger_path = tmp_path / f"gate-input-{mutation}.jsonl"
    ledger_path.touch()
    gateway = ScriptedModelGateway(
        responses={"judge_evidence": _native_open_judgment()}
    )
    core = _GateInputMutationCore(
        mutation,
        ledger=JsonlLedgerStore(ledger_path),
        model_gateway=gateway,
    )
    solver = _RecordingSolverProxy(core._belief_solver)
    core._belief_solver = solver
    return core, solver, ledger_path


def test_gate_signal_mutation_cannot_redefine_authoritative_closed_signal(
    tmp_path: Path,
):
    core, solver, ledger_path = _mutating_core(tmp_path, "signal")
    state = make_exact_belief_state()
    signal = _memory_signal(
        "S_gate_mutation",
        "The original signal supports H1.",
        root="root-gate-mutation",
    )
    prior_state = state.model_dump(mode="json")
    prior_signal = signal.model_dump(mode="json")
    cycle = make_cycle("cycle_gate_signal_mutation")

    with pytest.raises(
        ValueError,
        match="native closed signal ownership is invalid",
    ):
        core.integrate_cycle(
            cycle=cycle,
            belief_state=state,
            probe_set=make_empty_probe_set(cycle.cycle_id),
            signals=[signal],
        )

    assert solver.calls == 0
    assert signal.model_dump(mode="json") == prior_signal
    assert state.model_dump(mode="json") == prior_state
    assert ledger_path.read_bytes() == b""


def test_gate_belief_memory_mutation_cannot_erase_authoritative_history(
    tmp_path: Path,
):
    core, solver, ledger_path = _mutating_core(tmp_path, "belief_state")
    state, _, _, _ = _native_transition_fixture()
    prior_memory = state.evidence_memory
    assert prior_memory is not None
    prior_state = state.model_dump(mode="json")
    prior_memory_payload = prior_memory.model_dump(mode="json")
    cycle = make_cycle("cycle_gate_belief_mutation")

    with pytest.raises(
        ValueError,
        match="native evidence memory transition is invalid",
    ):
        core.integrate_cycle(
            cycle=cycle,
            belief_state=state,
            probe_set=make_empty_probe_set(cycle.cycle_id),
            signals=[
                _memory_signal(
                    "S_gate_belief_mutation",
                    "A new observation follows preserved history.",
                    root="root-gate-belief-mutation",
                )
            ],
        )

    assert solver.calls == 0
    assert state.model_dump(mode="json") == prior_state
    assert state.evidence_memory is prior_memory
    assert state.evidence_memory.model_dump(mode="json") == prior_memory_payload
    assert ledger_path.read_bytes() == b""


def test_gate_cycle_and_probe_mutation_never_reaches_result_or_ledger(
    tmp_path: Path,
):
    core, _, ledger_path = _mutating_core(tmp_path, "cycle_probe")
    cycle = make_cycle("cycle_gate_record_mutation")
    probe = ProbeDesign(
        id="P_gate_record_original",
        cycle_id=cycle.cycle_id,
        target_hypotheses=["H1", "H2"],
        inquiry_goal="Preserve the authoritative probe.",
        method="source_tracing",
    )
    probe_set = ProbeSet(
        probe_set_id="ps_gate_record_original",
        cycle_id=cycle.cycle_id,
        probes=[probe],
        selection_reason="Gate isolation regression.",
    )
    signal = _memory_signal(
        "S_gate_record_mutation",
        "The probe result supports H1.",
        root="root-gate-record-mutation",
    ).model_copy(update={"generated_by_probe": probe.id})
    prior_cycle = cycle.model_dump(mode="json")
    prior_probe_set = probe_set.model_dump(mode="json")

    result = core.integrate_cycle(
        cycle=cycle,
        belief_state=make_exact_belief_state(),
        probe_set=probe_set,
        signals=[signal],
    )

    assert cycle.model_dump(mode="json") == prior_cycle
    assert probe_set.model_dump(mode="json") == prior_probe_set
    assert result.cycle.cycle_id == cycle.cycle_id
    assert result.cycle.boundary_status == BoundaryStatus.INTEGRATED
    assert result.cycle.boundary_closed_at is not None
    assert result.belief_state.ledger_refs["probe_sets"] == [
        "ps_gate_record_original"
    ]
    ledger = JsonlLedgerStore(ledger_path)
    assert ledger.read_all("cycle")[0]["payload"]["cycle_id"] == cycle.cycle_id
    ledger_probe_set = ledger.read_all("probe_set")[0]["payload"]
    assert ledger_probe_set["probe_set_id"] == "ps_gate_record_original"
    assert ledger_probe_set["probes"][0]["id"] == "P_gate_record_original"


def test_later_gate_signal_mutation_cannot_change_earlier_authoritative_signal(
    tmp_path: Path,
):
    core, solver, ledger_path = _mutating_core(tmp_path, "later_signal")
    state = make_exact_belief_state()
    first = _memory_signal(
        "S_gate_later_first",
        "The first original observation supports H1.",
        root="root-gate-later-shared",
    )
    second = first.model_copy(
        update={
            "id": "S_gate_later_second",
            "raw_content": "The second original observation supports H1.",
        }
    )
    assert first.provenance is second.provenance
    prior_signals = [
        signal.model_dump(mode="json") for signal in (first, second)
    ]
    prior_state = state.model_dump(mode="json")
    cycle = make_cycle("cycle_gate_later_signal")

    with pytest.raises(
        ValueError,
        match="native closed signal ownership is invalid",
    ):
        core.integrate_cycle(
            cycle=cycle,
            belief_state=state,
            probe_set=make_empty_probe_set(cycle.cycle_id),
            signals=[first, second],
        )

    assert solver.calls == 0
    assert [signal.model_dump(mode="json") for signal in (first, second)] == (
        prior_signals
    )
    assert state.model_dump(mode="json") == prior_state
    assert ledger_path.read_bytes() == b""


def test_core_isolation_preserves_authentic_migration_receipt():
    state = make_explicit_legacy_belief_state(cycle_id="cycle_0")
    prior_state = state.model_dump(mode="json")
    assert resolve_belief_lifecycle(state) == (
        BeliefLifecycle.LEGACY_V01_MIGRATION
    )
    cycle = make_cycle("cycle_isolated_migration")

    result = BayesProbeCore().integrate_cycle(
        cycle=cycle,
        belief_state=state,
        probe_set=make_empty_probe_set(cycle.cycle_id),
        signals=[make_active_signal()],
    )

    assert state.model_dump(mode="json") == prior_state
    assert resolve_belief_lifecycle(result.belief_state) == (
        BeliefLifecycle.LEGACY_V01_MIGRATION
    )


def _memory_lifecycle_ids(state: BeliefState) -> set[str]:
    memory = state.evidence_memory
    assert memory is not None
    return set(memory.accepted_evidence_ids) | {
        json.loads(entry)[0] for entry in memory.discard_and_schema_history
    }


def _migrated_replay_fixture(tmp_path: Path, name: str):
    gateway = ScriptedModelGateway(
        responses={"judge_evidence": _legacy_evidence_judgment()}
    )
    ledger_path = tmp_path / f"{name}.jsonl"
    ledger = JsonlLedgerStore(ledger_path)
    core = BayesProbeCore(ledger=ledger, model_gateway=gateway)
    cycle = make_cycle(f"cycle_{name}")
    first_signal = _memory_signal(
        f"S_{name}_A",
        "The first positional audit favors H1.",
        root=f"root-{name}-A",
    )
    second_signal = _memory_signal(
        f"S_{name}_B",
        "The second positional audit also favors H1.",
        root=f"root-{name}-B",
    )
    first = core.integrate_cycle(
        cycle=cycle,
        belief_state=make_belief_state(cycle_id="cycle_0"),
        probe_set=make_empty_probe_set(cycle.cycle_id),
        signals=[first_signal, second_signal],
    )
    normalized = [
        evidence_memory.SignalProvenanceNormalizer().normalize(
            signal,
            run_id=cycle.run_id,
        )
        for signal in (first_signal, second_signal)
    ]
    expected_bindings = {
        first.evidence_events[index].id:
        evidence_memory.canonical_signal_identity_digest(signal)
        for index, signal in enumerate(normalized)
    }

    assert first.belief_state.evidence_memory.event_signal_identity_digests == (
        expected_bindings
    )
    assert _memory_lifecycle_ids(first.belief_state) <= set(
        first.belief_state.ledger_refs["evidence_events"]
    )
    return (
        gateway,
        ledger_path,
        ledger,
        core,
        cycle,
        first_signal,
        second_signal,
        first,
    )


def test_bypass_migrated_memory_event_without_ledger_ref_fails_atomically(
    tmp_path: Path,
):
    gateway = ScriptedModelGateway(
        responses={"judge_evidence": _legacy_evidence_judgment()}
    )
    ledger_path = tmp_path / "bypass-memory-ledger-invariant.jsonl"
    ledger = JsonlLedgerStore(ledger_path)
    core = BayesProbeCore(ledger=ledger, model_gateway=gateway)
    cycle = make_cycle("cycle_bypass_memory_ledger_invariant")
    first = core.integrate_cycle(
        cycle=cycle,
        belief_state=make_belief_state(cycle_id="cycle_0"),
        probe_set=make_empty_probe_set(cycle.cycle_id),
        signals=[
            _memory_signal(
                "S_bypass_original",
                "The original positional audit favors H1.",
                root="root-bypass-original",
            )
        ],
    )
    event_id = first.evidence_events[0].id
    memory = first.belief_state.evidence_memory
    assert memory is not None
    assert event_id in memory.accepted_evidence_ids
    assert event_id in memory.event_signal_identity_digests
    assert event_id in first.belief_state.ledger_refs["evidence_events"]
    inconsistent_state = first.belief_state.model_copy(
        update={
            "ledger_refs": {
                **first.belief_state.ledger_refs,
                "evidence_events": [],
            }
        }
    )
    prior_state = inconsistent_state.model_dump(mode="json")
    prior_memory = memory.model_dump(mode="json")
    prior_ledger = ledger_path.read_bytes()
    prior_provider_calls = len(gateway.requests)

    with pytest.raises(ValueError, match="invalid belief lifecycle"):
        core.integrate_cycle(
            cycle=cycle.model_copy(update={"cycle_index": 2}),
            belief_state=inconsistent_state,
            probe_set=make_empty_probe_set(cycle.cycle_id),
            signals=[
                _memory_signal(
                    "S_bypass_changed",
                    "Changed content must not rebind the positional event.",
                    root="root-bypass-changed",
                )
            ],
        )

    assert len(gateway.requests) == prior_provider_calls
    assert inconsistent_state.model_dump(mode="json") == prior_state
    assert memory.model_dump(mode="json") == prior_memory
    assert ledger_path.read_bytes() == prior_ledger


@pytest.mark.parametrize(
    "replay_shape",
    [
        "reordered",
        "inserted",
        "deleted",
        "changed_first",
        "changed_later",
    ],
)
def test_migrated_positional_replay_conflicts_fail_atomically(
    tmp_path: Path,
    replay_shape: str,
):
    (
        gateway,
        ledger_path,
        _,
        core,
        cycle,
        first_signal,
        second_signal,
        first,
    ) = _migrated_replay_fixture(tmp_path, f"migrated_{replay_shape}")
    inserted = _memory_signal(
        f"S_{replay_shape}_inserted",
        "An inserted positional observation.",
        root=f"root-{replay_shape}-inserted",
    )
    changed = _memory_signal(
        f"S_{replay_shape}_changed",
        "A materially changed positional observation.",
        root=f"root-{replay_shape}-changed",
    )
    replay_signals = {
        "reordered": [second_signal, first_signal],
        "inserted": [inserted, first_signal, second_signal],
        "deleted": [first_signal],
        "changed_first": [changed, second_signal],
        "changed_later": [first_signal, changed],
    }[replay_shape]
    prior_memory = first.belief_state.evidence_memory.model_dump(mode="json")
    prior_ledger = ledger_path.read_bytes()
    prior_provider_calls = len(gateway.requests)

    with pytest.raises(ValueError, match="evidence event"):
        core.integrate_cycle(
            cycle=cycle.model_copy(update={"cycle_index": 2}),
            belief_state=first.belief_state,
            probe_set=make_empty_probe_set(cycle.cycle_id),
            signals=replay_signals,
        )

    assert len(gateway.requests) == prior_provider_calls
    assert first.belief_state.evidence_memory.model_dump(mode="json") == prior_memory
    assert ledger_path.read_bytes() == prior_ledger


def test_later_positional_conflict_preflights_before_novel_provider(
    tmp_path: Path,
):
    (
        gateway,
        ledger_path,
        _,
        core,
        cycle,
        _,
        _,
        first,
    ) = _migrated_replay_fixture(tmp_path, "migrated_late_preflight")
    first_event_id, second_event_id = [
        event.id for event in first.evidence_events
    ]
    prior_memory = first.belief_state.evidence_memory
    partial_memory = EvidenceMemorySnapshot.model_validate(
        {
            **prior_memory.model_dump(mode="python"),
            "accepted_evidence_ids": [second_event_id],
            "event_signal_identity_digests": {
                second_event_id:
                prior_memory.event_signal_identity_digests[second_event_id]
            },
        }
    )
    partial_state = first.belief_state.model_copy(
        update={
            "evidence_memory": partial_memory,
            "ledger_refs": {"evidence_events": [second_event_id]},
        }
    )
    BeliefState.model_validate(
        partial_state.model_dump(mode="python")
    )
    novel = _memory_signal(
        "S_late_preflight_novel",
        "A novel first event must not reach the provider.",
        root="root-late-preflight-novel",
    )
    conflicting = _memory_signal(
        "S_late_preflight_conflict",
        "A changed second event conflicts with the E2 binding.",
        root="root-late-preflight-conflict",
    )
    prior_state = partial_state.model_dump(mode="json")
    prior_ledger = ledger_path.read_bytes()
    prior_provider_calls = len(gateway.requests)

    with pytest.raises(ValueError, match="invalid belief lifecycle"):
        core.integrate_cycle(
            cycle=cycle.model_copy(update={"cycle_index": 2}),
            belief_state=partial_state,
            probe_set=make_empty_probe_set(cycle.cycle_id),
            signals=[novel, conflicting],
        )

    assert first_event_id not in partial_state.ledger_refs["evidence_events"]
    assert len(gateway.requests) == prior_provider_calls
    assert partial_state.model_dump(mode="json") == prior_state
    assert ledger_path.read_bytes() == prior_ledger


def test_exact_migrated_positional_replay_is_idempotent(tmp_path: Path):
    (
        gateway,
        _,
        ledger,
        core,
        cycle,
        first_signal,
        second_signal,
        first,
    ) = _migrated_replay_fixture(tmp_path, "migrated_exact_replay")
    prior_memory = first.belief_state.evidence_memory
    prior_provider_calls = len(gateway.requests)
    prior_event_records = ledger.read_all("evidence_event")

    replayed = core.integrate_cycle(
        cycle=cycle.model_copy(update={"cycle_index": 2}),
        belief_state=first.belief_state,
        probe_set=make_empty_probe_set(cycle.cycle_id),
        signals=[first_signal, second_signal],
    )

    assert len(gateway.requests) == prior_provider_calls
    assert [event.discard_reason for event in replayed.evidence_events] == [
        "duplicate evidence event id",
        "duplicate evidence event id",
    ]
    assert replayed.belief_state.evidence_memory == prior_memory
    assert _memory_lifecycle_ids(replayed.belief_state) <= set(
        replayed.belief_state.ledger_refs["evidence_events"]
    )
    assert ledger.read_all("evidence_event") == prior_event_records


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
        first.evidence_events[0].id
    ]
    assert [
        json.loads(entry)
        for entry in second.belief_state.evidence_memory.discard_and_schema_history
    ] == [[second.evidence_events[0].id, "duplicate_exact"]]
    signal_records = ledger.read_all("external_signal")
    assert len(signal_records) == 2
    assert all(record["payload"]["provenance"] for record in signal_records)


def test_model_supplied_group_is_audited_and_changed_reuse_fails_atomically(
    tmp_path: Path,
):
    gateway = ScriptedModelGateway(responses={"judge_evidence": _native_open_judgment()})
    ledger_path = tmp_path / "model-supplied-group-ledger.jsonl"
    ledger = JsonlLedgerStore(ledger_path)
    core = BayesProbeCore(ledger=ledger, model_gateway=gateway)
    cycle = make_cycle("cycle_model_supplied_group")
    signal = _model_memory_signal(
        "S_model_supplied",
        "The model favors H1.",
        root="root-model-supplied",
        supplied_group="caller-model-group-1",
    )

    first = core.integrate_cycle(
        cycle=cycle,
        belief_state=make_exact_belief_state(),
        probe_set=make_empty_probe_set(cycle.cycle_id),
        signals=[signal],
    )
    replayed = core.integrate_cycle(
        cycle=cycle.model_copy(update={"cycle_index": 2}),
        belief_state=first.belief_state,
        probe_set=make_empty_probe_set(cycle.cycle_id),
        signals=[signal],
    )
    provenance_payload = ledger.read_all("external_signal")[-1]["payload"]["provenance"]
    persisted_provenance = SignalProvenance.model_validate(provenance_payload)
    canonical_group = persisted_provenance.correlation_group
    identity = json.loads(
        replayed.belief_state.evidence_memory.source_content_fingerprints[signal.id]
    )

    assert len(gateway.requests) == 1
    assert replayed.belief_state.evidence_memory == first.belief_state.evidence_memory
    assert canonical_group.startswith("model:")
    assert persisted_provenance.supplied_correlation_group == "caller-model-group-1"
    assert identity[2:] == [canonical_group, "caller-model-group-1"]
    assert all(
        key.startswith(f"{canonical_group}|")
        for key in first.belief_state.evidence_memory.correlation_credit
    )

    prior_memory_payload = replayed.belief_state.evidence_memory.model_dump(mode="json")
    prior_ledger = ledger_path.read_bytes()
    changed = signal.model_copy(
        update={
            "provenance": signal.provenance.model_copy(
                update={"correlation_group": "caller-model-group-changed"}
            )
        }
    )
    with pytest.raises(ValueError, match="signal id lineage conflict"):
        core.integrate_cycle(
            cycle=cycle.model_copy(update={"cycle_index": 3}),
            belief_state=replayed.belief_state,
            probe_set=make_empty_probe_set(cycle.cycle_id),
            signals=[changed],
        )

    assert len(gateway.requests) == 1
    assert replayed.belief_state.evidence_memory.model_dump(mode="json") == prior_memory_payload
    assert ledger_path.read_bytes() == prior_ledger


def test_prior_known_parent_root_conflict_preflights_before_novel_provider(
    tmp_path: Path,
):
    gateway = ScriptedModelGateway(
        responses={"judge_evidence": _native_open_judgment()}
    )
    ledger_path = tmp_path / "prior-parent-root-preflight-ledger.jsonl"
    ledger = JsonlLedgerStore(ledger_path)
    core = BayesProbeCore(ledger=ledger, model_gateway=gateway)
    parent_cycle = make_cycle("cycle_prior_parent")
    parent = _memory_signal(
        "S_prior_parent",
        "The prior parent observation.",
        root="root-prior-parent",
    )
    first = core.integrate_cycle(
        cycle=parent_cycle,
        belief_state=make_exact_belief_state(),
        probe_set=make_empty_probe_set(parent_cycle.cycle_id),
        signals=[parent],
    )
    next_cycle = make_cycle("cycle_prior_parent_conflict").model_copy(
        update={"cycle_index": 2}
    )
    prior_state = first.belief_state.model_dump(mode="json")
    prior_memory = first.belief_state.evidence_memory.model_dump(mode="json")
    prior_ledger = ledger_path.read_bytes()
    prior_provider_calls = len(gateway.requests)

    with pytest.raises(ValueError, match="preserve parent derivation root"):
        core.integrate_cycle(
            cycle=next_cycle,
            belief_state=first.belief_state,
            probe_set=make_empty_probe_set(next_cycle.cycle_id),
            signals=[
                _memory_signal(
                    "S_novel_before_bad_child",
                    "A novel signal before the invalid child.",
                    root="root-novel-before-child",
                ),
                _derived_memory_signal(
                    "S_bad_prior_child",
                    "A child that changes its known parent's root.",
                    parent_id=parent.id,
                    root="root-changed-child",
                ),
            ],
        )

    assert len(gateway.requests) == prior_provider_calls
    assert first.belief_state.model_dump(mode="json") == prior_state
    assert first.belief_state.evidence_memory.model_dump(mode="json") == prior_memory
    assert ledger_path.read_bytes() == prior_ledger


@pytest.mark.parametrize("order", ["parent_first", "child_first"])
def test_same_batch_parent_root_conflict_preflights_before_provider(
    tmp_path: Path,
    order: str,
):
    gateway = ScriptedModelGateway(
        responses={"judge_evidence": _native_open_judgment()}
    )
    ledger_path = tmp_path / f"same-batch-parent-root-{order}.jsonl"
    ledger_path.touch()
    core = BayesProbeCore(
        ledger=JsonlLedgerStore(ledger_path),
        model_gateway=gateway,
    )
    state = make_exact_belief_state()
    prior_state = state.model_dump(mode="json")
    parent = _memory_signal(
        f"S_batch_parent_{order}",
        "The same-batch parent observation.",
        root="root-same-batch-parent",
    )
    child = _derived_memory_signal(
        f"S_batch_child_{order}",
        "A same-batch child with a conflicting root.",
        parent_id=parent.id,
        root="root-same-batch-child-conflict",
    )
    signals = [parent, child] if order == "parent_first" else [child, parent]
    cycle = make_cycle(f"cycle_same_batch_parent_{order}")

    with pytest.raises(ValueError, match="preserve parent derivation root"):
        core.integrate_cycle(
            cycle=cycle,
            belief_state=state,
            probe_set=make_empty_probe_set(cycle.cycle_id),
            signals=signals,
        )

    assert gateway.requests == []
    assert state.model_dump(mode="json") == prior_state
    assert ledger_path.read_bytes() == b""


@pytest.mark.parametrize("order", ["parent_first", "child_first"])
def test_matching_parent_root_succeeds_with_zero_independence_in_both_orders(
    order: str,
):
    gateway = ScriptedModelGateway(
        responses={"judge_evidence": _native_open_judgment()}
    )
    core = BayesProbeCore(model_gateway=gateway)
    parent = _memory_signal(
        f"S_matching_parent_{order}",
        "The matching parent observation.",
        root="root-matching-parent",
    )
    child = _derived_memory_signal(
        f"S_matching_child_{order}",
        "A child that preserves its parent's root.",
        parent_id=parent.id,
        root="root-matching-parent",
    )
    signals = [parent, child] if order == "parent_first" else [child, parent]
    cycle = make_cycle(f"cycle_matching_parent_{order}")

    result = core.integrate_cycle(
        cycle=cycle,
        belief_state=make_exact_belief_state(),
        probe_set=make_empty_probe_set(cycle.cycle_id),
        signals=signals,
    )

    child_event = next(
        event
        for event in result.evidence_events
        if event.derived_from_signal == child.id
    )
    parent_event = next(
        event
        for event in result.evidence_events
        if event.derived_from_signal == parent.id
    )
    assert child_event.discard_reason is None
    assert child_event.correlation_status == "correlated_restatement"
    assert child_event.independence == 0.0
    assert child_event.effective_update_weight == 0.0
    assert parent_event.correlation_status == (
        "novel" if order == "parent_first" else "correlated_restatement"
    )
    assert len(gateway.requests) == 2


def test_later_cross_cycle_batch_conflict_preflights_before_provider_or_ledger(
    tmp_path: Path,
):
    gateway = ScriptedModelGateway(responses={"judge_evidence": _native_open_judgment()})
    ledger_path = tmp_path / "cross-cycle-batch-preflight-ledger.jsonl"
    ledger = JsonlLedgerStore(ledger_path)
    core = BayesProbeCore(ledger=ledger, model_gateway=gateway)
    first_cycle = make_cycle("cycle_batch_prior")
    prior_signal = _memory_signal(
        "S_batch_prior",
        "Prior batch observation.",
        root="root-batch-prior",
    )
    first = core.integrate_cycle(
        cycle=first_cycle,
        belief_state=make_exact_belief_state(),
        probe_set=make_empty_probe_set(first_cycle.cycle_id),
        signals=[prior_signal],
    )
    prior_memory_payload = first.belief_state.evidence_memory.model_dump(mode="json")
    prior_ledger = ledger_path.read_bytes()
    conflicting = prior_signal.model_copy(
        update={
            "provenance": prior_signal.provenance.model_copy(
                update={"correlation_group": "changed-late-batch-group"}
            )
        }
    )
    second_cycle = make_cycle("cycle_batch_conflict").model_copy(update={"cycle_index": 2})

    with pytest.raises(ValueError, match="signal id lineage conflict"):
        core.integrate_cycle(
            cycle=second_cycle,
            belief_state=first.belief_state,
            probe_set=make_empty_probe_set(second_cycle.cycle_id),
            signals=[
                _memory_signal(
                    "S_batch_novel",
                    "Novel signal before the late conflict.",
                    root="root-batch-novel",
                ),
                conflicting,
            ],
        )

    assert len(gateway.requests) == 1
    assert first.belief_state.evidence_memory.model_dump(mode="json") == prior_memory_payload
    assert ledger_path.read_bytes() == prior_ledger


@pytest.mark.parametrize("conflict", ["source", "content", "root", "group"])
def test_same_batch_reused_signal_conflict_preflights_atomically(
    tmp_path: Path,
    conflict: str,
):
    gateway = ScriptedModelGateway(responses={"judge_evidence": _native_open_judgment()})
    ledger_path = tmp_path / "same-batch-preflight-ledger.jsonl"
    ledger_path.touch()
    ledger = JsonlLedgerStore(ledger_path)
    core = BayesProbeCore(ledger=ledger, model_gateway=gateway)
    state = make_exact_belief_state()
    state_memory_payload = state.evidence_memory.model_dump(mode="json")
    first = _memory_signal(
        "S_same_batch",
        "First same-batch observation.",
        root="root-same-batch",
    )
    signal_updates = {}
    provenance_updates = {}
    if conflict == "source":
        provenance_updates["source_identity"] = "source.example/changed"
    elif conflict == "content":
        signal_updates["raw_content"] = "Changed same-batch observation."
    elif conflict == "root":
        provenance_updates["derivation_root_id"] = "root-same-batch-changed"
    else:
        provenance_updates["correlation_group"] = "same-batch-changed-group"
    signal_updates["provenance"] = first.provenance.model_copy(
        update=provenance_updates
    )
    conflicting = first.model_copy(update=signal_updates)
    cycle = make_cycle("cycle_same_batch_preflight")

    with pytest.raises(ValueError, match="signal id lineage conflict"):
        core.integrate_cycle(
            cycle=cycle,
            belief_state=state,
            probe_set=make_empty_probe_set(cycle.cycle_id),
            signals=[first, conflicting],
        )

    assert gateway.requests == []
    assert state.evidence_memory.model_dump(mode="json") == state_memory_payload
    assert ledger_path.read_bytes() == b""


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
        signals=[_memory_signal("S_replay_2", "First audit result.", root="root-1")],
    )

    assert len(gateway.requests) == 1
    assert replayed.evidence_events[0].discard_reason == "duplicate evidence event id"
    replayed_memory = replayed.belief_state.evidence_memory
    assert set(replayed_memory.content_fingerprints) == {"S_replay_1", "S_replay_2"}
    assert replayed_memory.accepted_evidence_ids == prior_memory.accepted_evidence_ids
    assert replayed_memory.discard_and_schema_history == prior_memory.discard_and_schema_history
    assert replayed_memory.correlation_credit == prior_memory.correlation_credit
    assert replayed_memory.event_signal_identity_digests == (
        prior_memory.event_signal_identity_digests
    )
    assert _memory_lifecycle_ids(replayed.belief_state) <= set(
        replayed.belief_state.ledger_refs["evidence_events"]
    )
    assert replayed.belief_updates == []
    assert [
        record["payload"]["id"] for record in ledger.read_all("evidence_event")
    ] == [first.evidence_events[0].id]


@pytest.mark.parametrize("conflict", ["source", "content", "root", "group"])
def test_replayed_new_signal_identity_is_persisted_then_conflict_fails_atomically(
    tmp_path: Path,
    conflict: str,
):
    gateway = ScriptedModelGateway(responses={"judge_evidence": _native_open_judgment()})
    ledger_path = tmp_path / "lineage-conflict-replay-ledger.jsonl"
    ledger = JsonlLedgerStore(ledger_path)
    core = BayesProbeCore(ledger=ledger, model_gateway=gateway)
    cycle = make_cycle("cycle_lineage_conflict_replay")
    first_signal = _memory_signal(
        "S_replay_original",
        "Stable audit result.",
        root="root-stable",
    )
    first = core.integrate_cycle(
        cycle=cycle,
        belief_state=make_exact_belief_state(),
        probe_set=make_empty_probe_set(cycle.cycle_id),
        signals=[first_signal],
    )
    first_memory = first.belief_state.evidence_memory
    replay_signal = _memory_signal(
        "S_replay_new_identity",
        "Stable audit result.",
        root="root-stable",
    )
    replay_signal = replay_signal.model_copy(
        update={
            "provenance": replay_signal.provenance.model_copy(
                update={"correlation_group": "replay-supplied-group"}
            )
        }
    )
    replayed = core.integrate_cycle(
        cycle=cycle.model_copy(update={"cycle_index": 2}),
        belief_state=first.belief_state,
        probe_set=make_empty_probe_set(cycle.cycle_id),
        signals=[replay_signal],
    )
    replayed_memory = replayed.belief_state.evidence_memory

    assert len(gateway.requests) == 1
    assert replayed.evidence_events[0].discard_reason == "duplicate evidence event id"
    assert set(replayed_memory.content_fingerprints) == {
        first_signal.id,
        replay_signal.id,
    }
    assert json.loads(
        replayed_memory.source_content_fingerprints[replay_signal.id]
    )[2:] == ["source.example/audit", "replay-supplied-group"]
    assert replayed_memory.accepted_evidence_ids == first_memory.accepted_evidence_ids
    assert replayed_memory.discard_and_schema_history == first_memory.discard_and_schema_history
    assert replayed_memory.correlation_credit == first_memory.correlation_credit

    replayed_again = core.integrate_cycle(
        cycle=cycle.model_copy(update={"cycle_index": 3}),
        belief_state=replayed.belief_state,
        probe_set=make_empty_probe_set(cycle.cycle_id),
        signals=[replay_signal],
    )

    assert len(gateway.requests) == 1
    assert replayed_again.belief_state.evidence_memory == replayed_memory

    prior_memory_payload = replayed_again.belief_state.evidence_memory.model_dump(
        mode="json"
    )
    prior_ledger = ledger_path.read_bytes()
    signal_updates = {}
    provenance_updates = {}
    if conflict == "source":
        provenance_updates["source_identity"] = "source.example/changed"
    elif conflict == "content":
        signal_updates["raw_content"] = "Changed audit result."
    elif conflict == "root":
        provenance_updates["derivation_root_id"] = "root-changed"
    else:
        provenance_updates["correlation_group"] = "changed-supplied-group"
    signal_updates["provenance"] = replay_signal.provenance.model_copy(
        update=provenance_updates
    )
    conflicting_signal = replay_signal.model_copy(update=signal_updates)

    with pytest.raises(ValueError, match="signal id lineage conflict"):
        core.integrate_cycle(
            cycle=cycle.model_copy(update={"cycle_index": 4}),
            belief_state=replayed_again.belief_state,
            probe_set=make_empty_probe_set(cycle.cycle_id),
            signals=[conflicting_signal],
        )

    assert len(gateway.requests) == 1
    assert (
        replayed_again.belief_state.evidence_memory.model_dump(mode="json")
        == prior_memory_payload
    )
    assert ledger_path.read_bytes() == prior_ledger


def test_historical_native_event_without_binding_fails_with_other_identity_memory(
    tmp_path: Path,
):
    gateway = ScriptedModelGateway(
        responses={"judge_evidence": _native_open_judgment()}
    )
    ledger_path = tmp_path / "historical-missing-binding-ledger.jsonl"
    ledger = JsonlLedgerStore(ledger_path)
    core = BayesProbeCore(ledger=ledger, model_gateway=gateway)
    other_cycle = make_cycle("cycle_other_identity")
    first = core.integrate_cycle(
        cycle=other_cycle,
        belief_state=make_exact_belief_state(),
        probe_set=make_empty_probe_set(other_cycle.cycle_id),
        signals=[
            _memory_signal(
                "S_other_identity",
                "An unrelated earlier positional event.",
                root="root-other-identity",
            )
        ],
    )
    cycle = make_cycle("cycle_historical_missing_binding").model_copy(
        update={"cycle_index": 2}
    )
    historical_signal = _memory_signal(
        "S_historical_replay",
        "The historical event cannot be proven from unrelated memory.",
        root="root-historical-replay",
    )
    preview = EvidenceIntegrationGate(model_gateway=gateway).integrate(
        cycle=cycle,
        belief_state=first.belief_state,
        probe_set=make_empty_probe_set(cycle.cycle_id),
        signals=[historical_signal],
    )
    prior_event = preview.evidence_events[0]
    event_id = prior_event.id
    ledger.append("evidence_event", prior_event)
    prior_memory = first.belief_state.evidence_memory
    historical_state = first.belief_state.model_copy(
        update={
            "ledger_refs": {
                "evidence_events": [
                    *first.belief_state.ledger_refs["evidence_events"],
                    event_id,
                ]
            }
        }
    )
    BeliefState.model_validate(
        historical_state.model_dump(mode="python")
    )
    prior_state_payload = historical_state.model_dump(mode="json")
    prior_ledger = ledger_path.read_bytes()
    prior_provider_calls = len(gateway.requests)

    assert historical_state.evidence_memory.source_content_fingerprints
    assert event_id not in historical_state.evidence_memory.accepted_evidence_ids
    assert event_id not in (
        historical_state.evidence_memory.event_signal_identity_digests
    )

    with pytest.raises(
        ValueError,
        match="event signal identity binding is missing",
    ):
        core.integrate_cycle(
            cycle=cycle,
            belief_state=historical_state,
            probe_set=make_empty_probe_set(cycle.cycle_id),
            signals=[
                historical_signal,
            ],
        )

    assert len(gateway.requests) == prior_provider_calls
    assert historical_state.model_dump(mode="json") == prior_state_payload
    assert ledger_path.read_bytes() == prior_ledger


def test_saturated_correlation_event_is_ledger_visible_without_mass_update(
    tmp_path: Path,
):
    gateway = ScriptedModelGateway(responses={"judge_evidence": _native_open_judgment()})
    ledger = JsonlLedgerStore(tmp_path / "saturated-memory-ledger.jsonl")
    core = BayesProbeCore(ledger=ledger, model_gateway=gateway)
    state = make_exact_belief_state()
    signal = ExternalSignal(
        id="S_saturated",
        cycle_id="pending",
        signal_kind=SignalKind.ACTIVE,
        source_type="model_probe_gateway",
        source="model_gateway:scripted",
        raw_content="A fresh model restatement favors H1.",
        initial_target_hypotheses=["H1", "H2"],
    )
    group = evidence_memory.SignalProvenanceNormalizer().normalize(
        signal,
        run_id="run_1",
    ).provenance.correlation_group
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
        signals=[signal],
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
