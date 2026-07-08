from pathlib import Path

from bayesprobe.controllers import AutonomousController, SynchronizedController
from bayesprobe.core import BayesProbeCore
from bayesprobe.ledger import JsonlLedgerStore
from bayesprobe.schemas import BeliefState, ExternalSignal, Hypothesis, SignalKind


def make_belief_state() -> BeliefState:
    return BeliefState(
        belief_state_id="bs_1",
        run_id="run_1",
        cycle_id="cycle_0",
        hypotheses=[
            Hypothesis(
                id="H1",
                statement="The claim is supported.",
                scope="claim verification",
                prior=0.5,
                posterior=0.5,
                rivals=["H2"],
                falsifiers=["Refuting evidence weakens H1."],
                predictions=["Supporting signal is likely."],
            ),
            Hypothesis(
                id="H2",
                statement="The claim is refuted.",
                scope="claim verification",
                prior=0.5,
                posterior=0.5,
                rivals=["H1"],
                falsifiers=["Supporting evidence weakens H2."],
                predictions=["Refuting signal is likely."],
            ),
        ],
    )


def test_autonomous_active_only_run_once_emits_answer_projection():
    controller = AutonomousController(core=BayesProbeCore())
    signal = ExternalSignal(
        id="S1",
        cycle_id="pending",
        signal_kind=SignalKind.ACTIVE,
        source_type="benchmark_stream",
        source="fixture",
        raw_content="REFUTES: The claim is contradicted.",
    )

    result = controller.run_once(
        run_id="run_1",
        belief_state=make_belief_state(),
        active_signals=[signal],
    )

    assert result.cycle.cycle_id == "run_1_cycle_1"
    assert result.cycle.cycle_index == 1
    assert result.cycle.signal_shape == "active_only"
    assert result.answer_projection is not None
    assert result.answer_projection.change_my_mind_condition.human_readable_condition
    assert result.answer_projection.change_my_mind_condition.structured_probe_candidates
    assert result.answer_projection.current_best_hypothesis == "H2"
    assert result.belief_state.hypotheses_by_id()["H2"].posterior > 0.5
    assert result.evidence_events[0].derived_from_signal == "S1"


def test_synchronized_passive_only_round_emits_belief_state_projection():
    controller = SynchronizedController(core=BayesProbeCore())
    signal = ExternalSignal(
        id="S2",
        cycle_id="pending",
        signal_kind=SignalKind.PASSIVE,
        source_type="external_agent_projection",
        source="agent_a",
        raw_content="Agent A believes H2 because source A refutes the claim.",
    )

    result = controller.process_round(
        run_id="run_1",
        round_id="round_1",
        belief_state=make_belief_state(),
        passive_signals=[signal],
    )

    assert result.cycle.cycle_id == "run_1_cycle_1"
    assert result.cycle.round_id == "round_1"
    assert result.cycle.signal_shape == "passive_only"
    assert result.belief_state_projection is not None
    assert result.belief_state_projection.change_my_mind_condition.human_readable_condition
    assert result.belief_state_projection.change_my_mind_condition.structured_probe_candidates
    assert result.belief_state_projection.requested_signal_type == "counterevidence_or_source_challenge"
    assert result.evidence_events[0].derived_from_signal == "S2"
    assert result.belief_state.hypotheses_by_id()["H2"].posterior > 0.5


def test_controller_generated_cycle_ids_are_unique_across_runs():
    controller = AutonomousController(core=BayesProbeCore())
    signal = ExternalSignal(
        id="S_unique_controller",
        cycle_id="pending",
        signal_kind=SignalKind.ACTIVE,
        source_type="benchmark_stream",
        source="fixture",
        raw_content="SUPPORTS: Shared signal.",
    )

    run_1 = controller.run_once(
        run_id="run_1",
        belief_state=make_belief_state(),
        active_signals=[signal],
    )
    run_2 = controller.run_once(
        run_id="run_2",
        belief_state=make_belief_state().model_copy(update={"run_id": "run_2"}),
        active_signals=[signal.model_copy(update={"id": "S_unique_controller_2"})],
    )

    assert run_1.cycle.cycle_id == "run_1_cycle_1"
    assert run_2.cycle.cycle_id == "run_2_cycle_1"
    assert run_1.evidence_events[0].id == "run_1_cycle_1_E1"
    assert run_2.evidence_events[0].id == "run_2_cycle_1_E1"


def test_repeated_autonomous_calls_from_same_prior_state_get_unique_ids():
    controller = AutonomousController(core=BayesProbeCore())
    belief_state = make_belief_state()
    signal = ExternalSignal(
        id="S_repeat",
        cycle_id="pending",
        signal_kind=SignalKind.ACTIVE,
        source_type="benchmark_stream",
        source="fixture",
        raw_content="SUPPORTS: Shared signal.",
    )

    first = controller.run_once(
        run_id="run_1",
        belief_state=belief_state,
        active_signals=[signal],
    )
    second = controller.run_once(
        run_id="run_1",
        belief_state=belief_state,
        active_signals=[signal.model_copy(update={"id": "S_repeat_2"})],
    )

    assert first.cycle.cycle_id == "run_1_cycle_1"
    assert second.cycle.cycle_id == "run_1_cycle_1_r2"
    assert first.evidence_events[0].id == "run_1_cycle_1_E1"
    assert second.evidence_events[0].id == "run_1_cycle_1_r2_E1"


def test_recreated_autonomous_controller_reuses_core_cycle_allocator():
    core = BayesProbeCore()
    belief_state = make_belief_state()
    signal = ExternalSignal(
        id="S_recreated",
        cycle_id="pending",
        signal_kind=SignalKind.ACTIVE,
        source_type="benchmark_stream",
        source="fixture",
        raw_content="SUPPORTS: Shared signal.",
    )

    first = AutonomousController(core=core).run_once(
        run_id="run_1",
        belief_state=belief_state,
        active_signals=[signal],
    )
    second = AutonomousController(core=core).run_once(
        run_id="run_1",
        belief_state=belief_state,
        active_signals=[signal.model_copy(update={"id": "S_recreated_2"})],
    )

    assert first.cycle.cycle_id == "run_1_cycle_1"
    assert second.cycle.cycle_id == "run_1_cycle_1_r2"
    assert first.evidence_events[0].id == "run_1_cycle_1_E1"
    assert second.evidence_events[0].id == "run_1_cycle_1_r2_E1"


def test_recreated_synchronized_controller_reuses_core_cycle_allocator():
    core = BayesProbeCore()
    belief_state = make_belief_state()
    signal = ExternalSignal(
        id="S_recreated_sync",
        cycle_id="pending",
        signal_kind=SignalKind.PASSIVE,
        source_type="external_agent_projection",
        source="agent_a",
        raw_content="Agent A believes H2 because source A refutes the claim.",
    )

    first = SynchronizedController(core=core).process_round(
        run_id="run_1",
        round_id="round_1",
        belief_state=belief_state,
        passive_signals=[signal],
    )
    second = SynchronizedController(core=core).process_round(
        run_id="run_1",
        round_id="round_2",
        belief_state=belief_state,
        passive_signals=[signal.model_copy(update={"id": "S_recreated_sync_2"})],
    )

    assert first.cycle.cycle_id == "run_1_cycle_1"
    assert second.cycle.cycle_id == "run_1_cycle_1_r2"
    assert first.evidence_events[0].id == "run_1_cycle_1_E1"
    assert second.evidence_events[0].id == "run_1_cycle_1_r2_E1"


def test_controllers_append_projection_records_when_ledger_available(tmp_path: Path):
    ledger = JsonlLedgerStore(tmp_path / "controller-ledger.jsonl")
    core = BayesProbeCore(ledger=ledger)
    autonomous = AutonomousController(core=core)
    synchronized = SynchronizedController(core=core)

    autonomous.run_once(
        run_id="run_1",
        belief_state=make_belief_state(),
        active_signals=[
            ExternalSignal(
                id="S_autonomous",
                cycle_id="pending",
                signal_kind=SignalKind.ACTIVE,
                source_type="benchmark_stream",
                source="fixture",
                raw_content="REFUTES: The claim is contradicted.",
            )
        ],
    )
    synchronized.process_round(
        run_id="run_1",
        round_id="round_1",
        belief_state=make_belief_state(),
        passive_signals=[
            ExternalSignal(
                id="S_sync",
                cycle_id="pending",
                signal_kind=SignalKind.PASSIVE,
                source_type="external_agent_projection",
                source="agent_a",
                raw_content="Agent A believes H2 because source A refutes the claim.",
            )
        ],
    )

    record_types = [record["record_type"] for record in ledger.read_all()]

    assert record_types[:8] == [
        "cycle",
        "external_signal",
        "probe_set",
        "evidence_event",
        "belief_update",
        "belief_update",
        "belief_state",
        "answer_projection",
    ]
    assert record_types[8:] == [
        "cycle",
        "external_signal",
        "probe_set",
        "evidence_event",
        "evidence_event",
        "belief_update",
        "belief_update",
        "belief_update",
        "belief_update",
        "probe_candidate",
        "belief_state",
        "belief_state_projection",
    ]
    assert record_types.count("belief_state_projection") == 1
    assert record_types.count("probe_candidate") == 1


def test_every_controller_output_has_change_my_mind_condition():
    autonomous = AutonomousController(core=BayesProbeCore())
    synchronized = SynchronizedController(core=BayesProbeCore())
    active_signal = ExternalSignal(
        id="S6",
        cycle_id="pending",
        signal_kind=SignalKind.ACTIVE,
        source_type="benchmark_stream",
        source="fixture",
        raw_content="SUPPORTS: The claim is supported.",
    )
    passive_signal = ExternalSignal(
        id="S7",
        cycle_id="pending",
        signal_kind=SignalKind.PASSIVE,
        source_type="user_feedback",
        source="user",
        raw_content="The claim may be too broad.",
    )

    autonomous_result = autonomous.run_once(
        run_id="run_1",
        belief_state=make_belief_state(),
        active_signals=[active_signal],
    )
    synchronized_result = synchronized.process_round(
        run_id="run_1",
        round_id="round_2",
        belief_state=make_belief_state(),
        passive_signals=[passive_signal],
    )

    assert autonomous_result.answer_projection is not None
    assert autonomous_result.answer_projection.change_my_mind_condition.structured_probe_candidates
    assert synchronized_result.belief_state_projection is not None
    assert synchronized_result.belief_state_projection.change_my_mind_condition.structured_probe_candidates
