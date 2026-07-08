from pathlib import Path

from bayesprobe.core import BayesProbeCore
from bayesprobe.inbox import SignalInbox
from bayesprobe.ledger import JsonlLedgerStore
from bayesprobe.schemas import (
    BeliefState,
    CycleRecord,
    CycleSignalShape,
    ExternalSignal,
    Hypothesis,
    ProbeSet,
    SignalInboxStatus,
    SignalKind,
)


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


def test_jsonl_ledger_appends_and_reads_records(tmp_path: Path):
    ledger = JsonlLedgerStore(tmp_path / "ledger.jsonl")
    signal = ExternalSignal(
        id="S1",
        cycle_id="cycle_1",
        signal_kind=SignalKind.PASSIVE,
        source_type="user_feedback",
        source="user",
        raw_content="This claim seems too broad.",
    )

    ledger.append("external_signal", signal)
    records = ledger.read_all("external_signal")

    assert len(records) == 1
    assert records[0]["record_type"] == "external_signal"
    assert records[0]["payload"]["id"] == "S1"


def test_signal_inbox_defers_late_signals_after_close():
    inbox = SignalInbox(cycle_id="cycle_1")
    first = ExternalSignal(
        id="S1",
        cycle_id="cycle_1",
        signal_kind=SignalKind.PASSIVE,
        source_type="external_agent_projection",
        source="agent_a",
        raw_content="Agent A reports uncertainty about H1.",
    )
    late = ExternalSignal(
        id="S2",
        cycle_id="cycle_1",
        signal_kind=SignalKind.PASSIVE,
        source_type="system_log",
        source="log",
        raw_content="Late log signal.",
    )

    accepted = inbox.add(first)
    closed_signals = inbox.close()
    deferred = inbox.add(late)

    assert accepted.inbox_status == SignalInboxStatus.ACCEPTED
    assert [signal.id for signal in closed_signals] == ["S1"]
    assert deferred.inbox_status == SignalInboxStatus.DEFERRED
    assert [signal.id for signal in inbox.deferred_signals] == ["S2"]


def test_core_appends_ledger_records_in_stable_order(tmp_path: Path):
    ledger = JsonlLedgerStore(tmp_path / "core-ledger.jsonl")
    core = BayesProbeCore(ledger=ledger)

    result = core.integrate_cycle(
        cycle=CycleRecord(
            cycle_id="cycle_1",
            run_id="run_1",
            cycle_index=1,
            signal_shape=CycleSignalShape.PASSIVE_ONLY,
        ),
        belief_state=make_belief_state(),
        probe_set=ProbeSet(
            probe_set_id="ps_1",
            cycle_id="cycle_1",
            probes=[],
            selection_reason="Ledger fixture.",
            may_be_empty=True,
        ),
        signals=[
            ExternalSignal(
                id="S_ledger",
                cycle_id="pending",
                signal_kind=SignalKind.PASSIVE,
                source_type="system_log",
                source="fixture",
                raw_content="ANOMALY: This signal is poorly explained by current hypotheses.",
            )
        ],
    )

    records = ledger.read_all()

    assert core.ledger is ledger
    assert [record["record_type"] for record in records] == [
        "cycle",
        "external_signal",
        "probe_set",
        "evidence_event",
        "belief_update",
        "belief_update",
        "hypothesis_evolution",
        "probe_candidate",
        "belief_state",
    ]
    assert records[0]["payload"]["cycle_id"] == "cycle_1"
    assert records[1]["payload"]["id"] == "S_ledger"
    assert records[3]["payload"]["id"] == "run_1_cycle_1_E1"
    assert records[6]["payload"]["evolution_id"] == "run_1_cycle_1_E1_HE"
    assert records[7]["payload"]["source"] == "anomaly"
    assert records[-1]["payload"]["hypotheses"][-1]["id"] == "H_run_1_cycle_1_E1_spawned"
    assert result.belief_state.ledger_refs["evidence_events"] == ["run_1_cycle_1_E1"]
