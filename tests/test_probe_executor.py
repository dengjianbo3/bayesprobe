from pathlib import Path

import pytest

from bayesprobe.core import BayesProbeCore
from bayesprobe.initialization import BayesProbeInitializer, InitializeRunInput
from bayesprobe.ledger import JsonlLedgerStore
from bayesprobe.probe_executor import (
    DeterministicProbeToolGateway,
    ProbeExecutionContext,
    ProbeExecutor,
)
from bayesprobe.probe_planner import ProbePlanner, ProbePlanningConfig
from bayesprobe.schemas import (
    BeliefState,
    CycleRecord,
    CycleSignalShape,
    ExternalSignal,
    Hypothesis,
    ProbeDesign,
    ProbeSet,
    SignalKind,
)


class RecordingGateway:
    def __init__(self, signals_by_probe_id: dict[str, list[ExternalSignal]] | None = None):
        self.calls: list[str] = []
        self.signals_by_probe_id = signals_by_probe_id or {}

    def execute_probe(self, *, probe: ProbeDesign, context: ProbeExecutionContext) -> list[ExternalSignal]:
        self.calls.append(probe.id)
        return self.signals_by_probe_id.get(
            probe.id,
            [
                ExternalSignal(
                    id=f"S_gateway_{probe.id}",
                    cycle_id=context.cycle_id,
                    signal_kind=SignalKind.ACTIVE,
                    source_type="recording_gateway",
                    source=probe.method,
                    raw_content=f"SUPPORTS: gateway result for {probe.id}.",
                )
            ],
        )


class PassiveGateway:
    def execute_probe(self, *, probe: ProbeDesign, context: ProbeExecutionContext) -> list[ExternalSignal]:
        return [
            ExternalSignal(
                id="S_passive_bad",
                cycle_id=context.cycle_id,
                signal_kind=SignalKind.PASSIVE,
                source_type="bad_gateway",
                source=probe.method,
                raw_content="This should not be accepted as active execution output.",
            )
        ]


def make_belief_state() -> BeliefState:
    return BeliefState(
        belief_state_id="bs_exec",
        run_id="run_exec",
        cycle_id="cycle_0",
        hypotheses=[
            Hypothesis(
                id="H1",
                statement="The claim is supported.",
                scope="execution fixture",
                prior=0.5,
                posterior=0.5,
                rivals=["H2"],
                falsifiers=["Reliable counterevidence weakens H1."],
                predictions=["Support should be independently observable."],
            ),
            Hypothesis(
                id="H2",
                statement="The claim is refuted.",
                scope="execution fixture",
                prior=0.5,
                posterior=0.5,
                rivals=["H1"],
                falsifiers=["Reliable support weakens H2."],
                predictions=["Counterevidence should be independently observable."],
            ),
        ],
    )


def make_probe(
    probe_id: str,
    target_hypotheses: list[str],
    *,
    cycle_id: str = "run_exec_cycle_1",
    method: str = "source_tracing",
) -> ProbeDesign:
    return ProbeDesign(
        id=probe_id,
        cycle_id=cycle_id,
        target_hypotheses=target_hypotheses,
        inquiry_goal=f"Probe {probe_id}.",
        method=method,
        support_condition={hypothesis_id: "Independent support appears." for hypothesis_id in target_hypotheses},
        weaken_condition={hypothesis_id: "Independent counterevidence appears." for hypothesis_id in target_hypotheses},
    )


def make_probe_set(
    probes: list[ProbeDesign],
    *,
    cycle_id: str = "run_exec_cycle_1",
    may_be_empty: bool = False,
) -> ProbeSet:
    return ProbeSet(
        probe_set_id=f"ps_{cycle_id}",
        cycle_id=cycle_id,
        probes=probes,
        selection_reason="fixture probe set",
        may_be_empty=may_be_empty,
    )


def make_context(cycle_id: str = "run_exec_cycle_1") -> ProbeExecutionContext:
    return ProbeExecutionContext(
        run_id="run_exec",
        cycle_id=cycle_id,
        belief_state=make_belief_state(),
    )


def test_executor_turns_probe_set_into_active_signals():
    probe_set = make_probe_set(
        [
            make_probe("P1", ["H1"]),
            make_probe("P2", ["H2"], method="counterevidence_scan"),
        ]
    )

    result = ProbeExecutor(DeterministicProbeToolGateway()).execute_probe_set(
        probe_set=probe_set,
        context=make_context(),
    )

    assert result.probe_set == probe_set
    assert result.executed_probe_ids == ["P1", "P2"]
    assert [signal.generated_by_probe for signal in result.signals] == ["P1", "P2"]
    assert [signal.initial_target_hypotheses for signal in result.signals] == [["H1"], ["H2"]]
    assert all(signal.signal_kind == SignalKind.ACTIVE for signal in result.signals)
    assert all(signal.cycle_id == "run_exec_cycle_1" for signal in result.signals)
    assert "SUPPORTS" in result.signals[0].raw_content
    assert "REFUTES" in result.signals[1].raw_content


def test_executor_preserves_probe_and_signal_order():
    p1_s1 = ExternalSignal(
        id="S_P1_1",
        cycle_id="placeholder",
        signal_kind=SignalKind.ACTIVE,
        source_type="fixture_gateway",
        source="fixture",
        raw_content="SUPPORTS: first P1 signal.",
    )
    p1_s2 = ExternalSignal(
        id="S_P1_2",
        cycle_id="placeholder",
        signal_kind=SignalKind.ACTIVE,
        source_type="fixture_gateway",
        source="fixture",
        raw_content="SUPPORTS: second P1 signal.",
    )
    p2_s1 = ExternalSignal(
        id="S_P2_1",
        cycle_id="placeholder",
        signal_kind=SignalKind.ACTIVE,
        source_type="fixture_gateway",
        source="fixture",
        raw_content="SUPPORTS: first P2 signal.",
    )
    gateway = RecordingGateway({"P1": [p1_s1, p1_s2], "P2": [p2_s1]})
    probe_set = make_probe_set([make_probe("P1", ["H1"]), make_probe("P2", ["H2"])])

    result = ProbeExecutor(gateway).execute_probe_set(
        probe_set=probe_set,
        context=make_context(),
    )

    assert gateway.calls == ["P1", "P2"]
    assert result.executed_probe_ids == ["P1", "P2"]
    assert [signal.id for signal in result.signals] == ["S_P1_1", "S_P1_2", "S_P2_1"]
    assert [signal.generated_by_probe for signal in result.signals] == ["P1", "P1", "P2"]


def test_executor_returns_empty_result_for_empty_probe_set():
    gateway = RecordingGateway()
    probe_set = make_probe_set([], may_be_empty=True)

    result = ProbeExecutor(gateway).execute_probe_set(
        probe_set=probe_set,
        context=make_context(),
    )

    assert gateway.calls == []
    assert result.signals == []
    assert result.executed_probe_ids == []
    assert result.probe_set == probe_set


def test_executor_rejects_probe_set_cycle_mismatch():
    probe_set = make_probe_set([make_probe("P1", ["H1"])], cycle_id="run_exec_cycle_1")

    with pytest.raises(ValueError):
        ProbeExecutor(RecordingGateway()).execute_probe_set(
            probe_set=probe_set,
            context=make_context(cycle_id="run_exec_cycle_2"),
        )


def test_executor_rejects_passive_gateway_signals():
    probe_set = make_probe_set([make_probe("P1", ["H1"])])

    with pytest.raises(ValueError):
        ProbeExecutor(PassiveGateway()).execute_probe_set(
            probe_set=probe_set,
            context=make_context(),
        )


def test_executor_normalizes_gateway_signals_without_mutating_originals():
    original_signal = ExternalSignal(
        id="S_original",
        cycle_id="placeholder",
        signal_kind=SignalKind.ACTIVE,
        source_type="fixture_gateway",
        source="fixture",
        raw_content="SUPPORTS: raw gateway signal.",
        generated_by_probe=None,
        initial_target_hypotheses=["stale"],
    )
    probe_set = make_probe_set([make_probe("P1", ["H1"])])
    gateway = RecordingGateway({"P1": [original_signal]})

    result = ProbeExecutor(gateway).execute_probe_set(
        probe_set=probe_set,
        context=make_context(),
    )

    normalized = result.signals[0]
    assert normalized is not original_signal
    assert normalized.id == "S_original"
    assert normalized.cycle_id == "run_exec_cycle_1"
    assert normalized.generated_by_probe == "P1"
    assert normalized.initial_target_hypotheses == ["H1"]
    assert original_signal.cycle_id == "placeholder"
    assert original_signal.generated_by_probe is None
    assert original_signal.initial_target_hypotheses == ["stale"]


def test_executor_writes_only_execution_and_signal_records_to_ledger(tmp_path: Path):
    ledger = JsonlLedgerStore(tmp_path / "executor-ledger.jsonl")
    probe_set = make_probe_set([make_probe("P1", ["H1"])])

    ProbeExecutor(DeterministicProbeToolGateway(), ledger=ledger).execute_probe_set(
        probe_set=probe_set,
        context=make_context(),
    )

    record_types = [record["record_type"] for record in ledger.read_all()]
    assert record_types == ["probe_execution", "external_signal"]
    assert "evidence_event" not in record_types
    assert "belief_update" not in record_types
    assert "hypothesis_evolution" not in record_types
    assert "answer_projection" not in record_types


def test_planned_probe_set_executes_and_integrates_through_core():
    initialization = BayesProbeInitializer().initialize(
        InitializeRunInput(
            run_id="run_full_active_path",
            problem="Can the active path produce signals for the core?",
        )
    )
    cycle = CycleRecord(
        cycle_id="run_full_active_path_cycle_1",
        run_id="run_full_active_path",
        cycle_index=1,
        signal_shape=CycleSignalShape.ACTIVE_ONLY,
    )
    planning = ProbePlanner().design_probe_set(
        run_id=initialization.run.run_id,
        cycle_id=cycle.cycle_id,
        belief_state=initialization.belief_state,
        candidates=initialization.probe_candidates,
        config=ProbePlanningConfig(max_probes=1),
    )
    execution = ProbeExecutor(DeterministicProbeToolGateway()).execute_probe_set(
        probe_set=planning.probe_set,
        context=ProbeExecutionContext(
            run_id=initialization.run.run_id,
            cycle_id=cycle.cycle_id,
            belief_state=initialization.belief_state,
        ),
    )

    result = BayesProbeCore().integrate_cycle(
        cycle=cycle,
        belief_state=initialization.belief_state,
        probe_set=planning.probe_set,
        signals=execution.signals,
    )

    assert execution.signals
    assert result.evidence_events
    assert result.belief_updates
    assert result.belief_state.cycle_id == cycle.cycle_id
