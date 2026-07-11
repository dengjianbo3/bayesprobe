from pathlib import Path

import pytest

from bayesprobe.core import BayesProbeCore
from bayesprobe.initialization import BayesProbeInitializer, HypothesisSeed, InitializeRunInput
from bayesprobe.ledger import JsonlLedgerStore
from bayesprobe.runners import AutonomousLoopConfig, AutonomousLoopRunner, AutonomousStopReason
from bayesprobe.schemas import (
    BeliefState,
    ExternalSignal,
    Hypothesis,
    HypothesisRelation,
    SignalKind,
)


class SequenceSignalProvider:
    def __init__(self, batches: list[list[ExternalSignal]]):
        self._batches = list(batches)
        self.calls = []

    def collect_signals(self, *, run_id, cycle_index, belief_state, previous_answer):
        self.calls.append(
            {
                "run_id": run_id,
                "cycle_index": cycle_index,
                "belief_state": belief_state,
                "previous_answer": previous_answer,
            }
        )
        if self._batches:
            return self._batches.pop(0)
        return []


def make_belief_state(run_id: str = "run_1") -> BeliefState:
    return BeliefState(
        belief_state_id=f"bs_{run_id}",
        run_id=run_id,
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


def active_signal(signal_id: str, raw_content: str) -> ExternalSignal:
    return ExternalSignal(
        id=signal_id,
        cycle_id="pending",
        signal_kind=SignalKind.ACTIVE,
        source_type="benchmark_stream",
        source="fixture",
        raw_content=raw_content,
    )


def test_runner_stops_after_max_cycles():
    provider = SequenceSignalProvider(
        [
            [active_signal("S1", "SUPPORTS: The claim is supported.")],
            [active_signal("S2", "REFUTES: The claim is contradicted.")],
            [active_signal("S3", "SUPPORTS: A third signal should not be collected.")],
        ]
    )
    runner = AutonomousLoopRunner(
        core=BayesProbeCore(),
        config=AutonomousLoopConfig(max_cycles=2),
    )

    result = runner.run(
        run_id="run_1",
        initial_belief_state=make_belief_state(),
        signal_provider=provider,
    )

    assert result.stop_reason == AutonomousStopReason.MAX_CYCLES
    assert len(result.cycle_results) == 2
    assert len(provider.calls) == 2
    assert [cycle_result.cycle.cycle_id for cycle_result in result.cycle_results] == [
        "run_1_cycle_1",
        "run_1_cycle_2",
    ]
    assert result.final_answer_projection == result.cycle_results[-1].answer_projection
    assert result.final_belief_state == result.cycle_results[-1].belief_state


def test_runner_stops_before_cycle_when_no_signals():
    initial_belief_state = make_belief_state()
    provider = SequenceSignalProvider([[]])
    runner = AutonomousLoopRunner(core=BayesProbeCore())

    result = runner.run(
        run_id="run_1",
        initial_belief_state=initial_belief_state,
        signal_provider=provider,
    )

    assert result.stop_reason == AutonomousStopReason.NO_SIGNALS
    assert result.cycle_results == []
    assert result.final_answer_projection is None
    assert result.final_belief_state == initial_belief_state
    assert provider.calls[0]["cycle_index"] == 1


def test_runner_feeds_updated_belief_state_into_next_cycle():
    provider = SequenceSignalProvider(
        [
            [active_signal("S1", "REFUTES: The claim is contradicted.")],
            [active_signal("S2", "SUPPORTS: The claim is supported.")],
        ]
    )
    runner = AutonomousLoopRunner(
        core=BayesProbeCore(),
        config=AutonomousLoopConfig(max_cycles=2),
    )

    result = runner.run(
        run_id="run_1",
        initial_belief_state=make_belief_state(),
        signal_provider=provider,
    )

    second_call_belief_state = provider.calls[1]["belief_state"]
    assert result.stop_reason == AutonomousStopReason.MAX_CYCLES
    assert second_call_belief_state.cycle_id == "run_1_cycle_1"
    assert second_call_belief_state.cycle_index == 1
    assert second_call_belief_state.hypotheses_by_id()["H2"].posterior > 0.5
    assert provider.calls[1]["previous_answer"] is not None
    assert provider.calls[1]["previous_answer"].current_best_hypothesis == "H2"


def test_runner_stops_when_confidence_threshold_reached():
    provider = SequenceSignalProvider(
        [
            [active_signal("S1", "SUPPORTS: The claim is supported.")],
            [active_signal("S2", "REFUTES: This later signal should not be collected.")],
        ]
    )
    runner = AutonomousLoopRunner(
        core=BayesProbeCore(),
        config=AutonomousLoopConfig(max_cycles=3, confidence_threshold=0.6),
    )

    result = runner.run(
        run_id="run_1",
        initial_belief_state=make_belief_state(),
        signal_provider=provider,
    )

    assert result.stop_reason == AutonomousStopReason.CONFIDENCE_REACHED
    assert len(result.cycle_results) == 1
    assert len(provider.calls) == 1
    assert result.final_answer_projection is not None
    assert result.final_answer_projection.current_best_hypothesis == "H1"
    assert result.final_belief_state.hypotheses_by_id()["H1"].posterior >= 0.6


def test_runner_does_not_apply_winner_threshold_to_independent_credences():
    initial = BayesProbeInitializer().initialize(
        InitializeRunInput(
            run_id="run_independent_threshold",
            problem="Which independent claims remain credible?",
            hypothesis_relation=HypothesisRelation.INDEPENDENT,
            hypothesis_seeds=[
                HypothesisSeed(id="H1", statement="Claim one remains credible.", prior=0.8),
                HypothesisSeed(id="H2", statement="Claim two remains credible.", prior=0.7),
            ],
        )
    ).belief_state
    provider = SequenceSignalProvider(
        [
            [active_signal("S1", "NEUTRAL: No change to either claim.")],
            [active_signal("S2", "NEUTRAL: Still no change to either claim.")],
        ]
    )
    runner = AutonomousLoopRunner(
        core=BayesProbeCore(),
        config=AutonomousLoopConfig(max_cycles=2, confidence_threshold=0.6),
    )

    result = runner.run(
        run_id="run_independent_threshold",
        initial_belief_state=initial,
        signal_provider=provider,
    )

    assert result.stop_reason == AutonomousStopReason.MAX_CYCLES
    assert len(result.cycle_results) == 2
    assert len(provider.calls) == 2


def test_runner_stops_when_posterior_delta_is_stable():
    provider = SequenceSignalProvider(
        [
            [active_signal("S1", "NEUTRAL: This signal should not move either hypothesis.")],
            [active_signal("S2", "SUPPORTS: This later signal should not be collected.")],
        ]
    )
    runner = AutonomousLoopRunner(
        core=BayesProbeCore(),
        config=AutonomousLoopConfig(max_cycles=3, posterior_delta_threshold=0.0),
    )

    result = runner.run(
        run_id="run_1",
        initial_belief_state=make_belief_state(),
        signal_provider=provider,
    )

    assert result.stop_reason == AutonomousStopReason.POSTERIOR_STABLE
    assert len(result.cycle_results) == 1
    assert len(provider.calls) == 1
    assert result.final_belief_state.hypotheses_by_id()["H1"].posterior == 0.5
    assert result.final_belief_state.hypotheses_by_id()["H2"].posterior == 0.5


def test_runner_materializes_anomaly_spawned_hypothesis_across_cycles():
    provider = SequenceSignalProvider(
        [
            [active_signal("S1", "ANOMALY: Neither current hypothesis predicts this.")],
            [],
        ]
    )
    runner = AutonomousLoopRunner(
        core=BayesProbeCore(),
        config=AutonomousLoopConfig(max_cycles=3),
    )

    result = runner.run(
        run_id="run_1",
        initial_belief_state=make_belief_state(),
        signal_provider=provider,
    )

    spawned_hypothesis_id = result.cycle_results[0].hypothesis_evolutions[0].to_hypothesis
    assert result.stop_reason == AutonomousStopReason.NO_SIGNALS
    assert len(result.cycle_results) == 1
    assert spawned_hypothesis_id is not None
    assert spawned_hypothesis_id in result.final_belief_state.hypotheses_by_id()
    assert spawned_hypothesis_id in provider.calls[1]["belief_state"].hypotheses_by_id()


def test_runner_writes_ledger_records_for_each_executed_cycle(tmp_path: Path):
    ledger = JsonlLedgerStore(tmp_path / "runner-ledger.jsonl")
    provider = SequenceSignalProvider(
        [
            [active_signal("S1", "SUPPORTS: The claim is supported.")],
            [active_signal("S2", "REFUTES: The claim is contradicted.")],
        ]
    )
    runner = AutonomousLoopRunner(
        core=BayesProbeCore(ledger=ledger),
        config=AutonomousLoopConfig(max_cycles=2),
    )

    result = runner.run(
        run_id="run_1",
        initial_belief_state=make_belief_state(),
        signal_provider=provider,
    )

    record_types = [record["record_type"] for record in ledger.read_all()]
    assert len(result.cycle_results) == 2
    assert record_types.count("cycle") == 2
    assert record_types.count("belief_state") == 2
    assert record_types.count("answer_projection") == 2


@pytest.mark.parametrize(
    "config_kwargs",
    [
        {"max_cycles": 0},
        {"confidence_threshold": -0.1},
        {"confidence_threshold": 1.1},
        {"posterior_delta_threshold": -0.1},
    ],
)
def test_invalid_runner_config_is_rejected(config_kwargs):
    with pytest.raises(ValueError):
        AutonomousLoopConfig(**config_kwargs)
