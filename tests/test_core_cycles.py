from pathlib import Path

import pytest

from bayesprobe.core import BayesProbeCore, EvidenceIntegrationGate
from bayesprobe.inbox import SignalInbox
from bayesprobe.ledger import JsonlLedgerStore
from bayesprobe.model_gateway import ScriptedModelGateway
from bayesprobe.schemas import (
    BeliefState,
    CycleRecord,
    CycleSignalShape,
    EvidenceEvent,
    EvidenceType,
    ExternalSignal,
    Hypothesis,
    HypothesisEvolution,
    EvolutionOperation,
    LikelihoodBand,
    ProbeDesign,
    ProbeSet,
    SignalKind,
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


def test_previous_cycle_belief_state_advances_to_current_cycle():
    core = BayesProbeCore()
    cycle = CycleRecord(
        cycle_id="current_cycle",
        run_id="run_1",
        cycle_index=1,
        signal_shape=CycleSignalShape.ACTIVE_ONLY,
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
        signals=[],
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
    assert spawned.posterior == 0.12
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
    assert result.belief_state.hypotheses_by_id()["H2"].posterior == 0.5
    assert [update.hypothesis_id for update in result.belief_updates] == ["H1"]


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
    assert result.belief_state.hypotheses_by_id()["H2"].posterior == 0.5
    assert [update.hypothesis_id for update in result.belief_updates] == ["H1"]


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
        signal_shape=CycleSignalShape.ACTIVE_ONLY,
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
        signals=[],
    )

    assert result.belief_state.ledger_refs["probe_sets"] == ["ps_prior", "ps_7"]
    assert result.belief_state.ledger_refs["evidence_events"] == ["E_prior"]
    assert result.belief_state.ledger_refs["belief_updates"] == ["U_prior"]
    assert result.belief_state.ledger_refs["hypothesis_evolutions"] == ["HE_prior"]
    assert result.belief_state.ledger_refs["custom_audit"] == ["keep_me"]


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
