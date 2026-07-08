from bayesprobe.schemas import (
    BeliefState,
    ChangeMyMindCondition,
    CycleRecord,
    CycleSignalShape,
    ExternalSignal,
    Hypothesis,
    HypothesisStatus,
    ProbeCandidate,
    ProbeDesign,
    ProbeSet,
    RunRecord,
    RunRegime,
    SignalKind,
)


def test_minimal_run_cycle_and_belief_state_round_trip():
    run = RunRecord(run_id="run_1", regime=RunRegime.AUTONOMOUS, problem="Decide X")
    cycle = CycleRecord(
        cycle_id="cycle_1",
        run_id=run.run_id,
        cycle_index=1,
        signal_shape=CycleSignalShape.ACTIVE_ONLY,
    )
    hypothesis = Hypothesis(
        id="H1",
        statement="X is true",
        scope="sample scope",
        prior=0.5,
        posterior=0.5,
        rivals=["H2"],
        falsifiers=["A strong counterexample would weaken H1."],
        predictions=["Evidence A is likely if H1 is true."],
    )
    belief_state = BeliefState(
        belief_state_id="bs_1",
        run_id=run.run_id,
        cycle_id=cycle.cycle_id,
        hypotheses=[hypothesis],
    )

    loaded = BeliefState.model_validate_json(belief_state.model_dump_json())

    assert loaded.hypotheses[0].id == "H1"
    assert loaded.hypotheses[0].status == HypothesisStatus.ACTIVE


def test_probe_set_can_be_empty_for_passive_only_cycle():
    probe_set = ProbeSet(
        probe_set_id="ps_1",
        cycle_id="cycle_1",
        probes=[],
        selection_reason="Passive-only synchronized cycle.",
        may_be_empty=True,
    )

    assert probe_set.probes == []
    assert probe_set.may_be_empty is True


def test_external_signal_kinds_and_change_my_mind_candidates():
    candidate = ProbeCandidate(
        candidate_id="pc_1",
        source="change_my_mind",
        candidate_probe=ProbeDesign(
            id="P1",
            cycle_id="cycle_2",
            target_hypotheses=["H1"],
            inquiry_goal="Check if source A is independent.",
            method="source_tracing",
            support_condition={"H1": "Source A is independent."},
            weaken_condition={"H1": "Source A shares origin with source B."},
        ),
    )
    condition = ChangeMyMindCondition(
        human_readable_condition="I would lower H1 if source A is not independent.",
        structured_probe_candidates=[candidate],
    )
    signal = ExternalSignal(
        id="S1",
        cycle_id="cycle_1",
        signal_kind=SignalKind.PASSIVE,
        source_type="external_agent_projection",
        source="agent_a",
        raw_content="Agent A believes H1 because source A supports it.",
    )

    assert condition.structured_probe_candidates[0].candidate_probe.method == "source_tracing"
    assert signal.signal_kind == SignalKind.PASSIVE
