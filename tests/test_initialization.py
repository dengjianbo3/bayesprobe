from pathlib import Path

import pytest

from bayesprobe.core import BayesProbeCore
from bayesprobe.initialization import BayesProbeInitializer, HypothesisSeed, InitializeRunInput
from bayesprobe.ledger import JsonlLedgerStore
from bayesprobe.runners import AutonomousLoopConfig, AutonomousLoopRunner, AutonomousStopReason
from bayesprobe.schemas import ExternalSignal, RunRegime, RunStatus, SignalKind


class OneBatchSignalProvider:
    def __init__(self):
        self.calls = 0

    def collect_signals(self, *, run_id, cycle_index, belief_state, previous_answer):
        self.calls += 1
        if self.calls > 1:
            return []
        return [
            ExternalSignal(
                id="S_init_support",
                cycle_id="pending",
                signal_kind=SignalKind.ACTIVE,
                source_type="benchmark_stream",
                source="fixture",
                raw_content="SUPPORTS: The initialized claim direction is supported.",
            )
        ]


def test_initializer_creates_default_rival_hypotheses_from_problem():
    problem = "Is the new retrieval strategy improving answer accuracy?"

    result = BayesProbeInitializer().initialize(
        InitializeRunInput(
            run_id="run_init_1",
            problem=problem,
            context="Offline evaluation context.",
            metadata={"sample_id": "sample_1"},
        )
    )

    assert result.run.run_id == "run_init_1"
    assert result.run.problem == problem
    assert result.run.regime == RunRegime.AUTONOMOUS
    assert result.run.status == RunStatus.RUNNING
    assert result.run.current_cycle_id == "cycle_0"
    assert result.run.metadata["sample_id"] == "sample_1"
    assert result.run.metadata["initialization_method"] == "deterministic_mvp"

    assert result.belief_state.belief_state_id == "run_init_1_bs_0"
    assert result.belief_state.run_id == "run_init_1"
    assert result.belief_state.cycle_id == "cycle_0"
    assert result.belief_state.cycle_index == 0
    assert result.belief_state.posterior_summary["initialization_method"] == "deterministic_mvp"
    assert result.belief_state.posterior_summary["hypothesis_count"] == 2
    assert result.belief_state.posterior_summary["top_hypothesis"] == "H1"
    assert problem in result.belief_state.uncertainty_summary

    hypotheses = result.belief_state.hypotheses_by_id()
    assert set(hypotheses) == {"H1", "H2"}
    assert hypotheses["H1"].prior == 0.5
    assert hypotheses["H1"].posterior == 0.5
    assert hypotheses["H1"].created_by == "initial"
    assert hypotheses["H1"].rivals == ["H2"]
    assert hypotheses["H1"].falsifiers
    assert hypotheses["H1"].predictions
    assert problem in hypotheses["H1"].statement
    assert hypotheses["H2"].rivals == ["H1"]
    assert hypotheses["H2"].falsifiers
    assert hypotheses["H2"].predictions
    assert problem in hypotheses["H2"].statement

    assert len(result.probe_candidates) == 2
    target_sets = [
        candidate.candidate_probe.target_hypotheses
        for candidate in result.probe_candidates
    ]
    assert target_sets == [["H1"], ["H2"]]
    assert all(candidate.source == "manual" for candidate in result.probe_candidates)
    assert all(candidate.candidate_probe.cycle_id == "cycle_0" for candidate in result.probe_candidates)


def test_initializer_preserves_seeded_hypotheses():
    result = BayesProbeInitializer().initialize(
        InitializeRunInput(
            run_id="run_seeded",
            problem="Which explanation best fits the incident report?",
            regime=RunRegime.SYNCHRONIZED,
            hypothesis_seeds=[
                HypothesisSeed(
                    id="H_network",
                    statement="A network outage caused the incident.",
                    prior=0.7,
                    falsifiers=["Independent network telemetry remains healthy."],
                ),
                HypothesisSeed(
                    statement="A deployment regression caused the incident.",
                    prior=0.3,
                    predictions=["Rollback reduces the error rate."],
                ),
            ],
        )
    )

    hypotheses = result.belief_state.hypotheses_by_id()

    assert result.run.regime == RunRegime.SYNCHRONIZED
    assert set(hypotheses) == {"H_network", "H2"}
    assert hypotheses["H_network"].statement == "A network outage caused the incident."
    assert hypotheses["H_network"].prior == 0.7
    assert hypotheses["H_network"].posterior == 0.7
    assert hypotheses["H_network"].falsifiers == ["Independent network telemetry remains healthy."]
    assert hypotheses["H_network"].predictions
    assert hypotheses["H_network"].rivals == ["H2"]
    assert hypotheses["H2"].statement == "A deployment regression caused the incident."
    assert hypotheses["H2"].prior == 0.3
    assert hypotheses["H2"].posterior == 0.3
    assert hypotheses["H2"].scope
    assert hypotheses["H2"].falsifiers
    assert hypotheses["H2"].predictions == ["Rollback reduces the error rate."]
    assert hypotheses["H2"].rivals == ["H_network"]


@pytest.mark.parametrize(
    "input_value",
    [
        InitializeRunInput(run_id=" ", problem="Valid problem."),
        InitializeRunInput(run_id="run_empty_problem", problem=" "),
        InitializeRunInput(
            run_id="run_one_seed",
            problem="Valid problem.",
            hypothesis_seeds=[HypothesisSeed(statement="Only one hypothesis.")],
        ),
        InitializeRunInput(
            run_id="run_bad_prior",
            problem="Valid problem.",
            hypothesis_seeds=[
                HypothesisSeed(statement="First hypothesis.", prior=1.1),
                HypothesisSeed(statement="Second hypothesis.", prior=0.5),
            ],
        ),
    ],
)
def test_initializer_rejects_invalid_input(input_value):
    with pytest.raises(ValueError):
        BayesProbeInitializer().initialize(input_value)


def test_initializer_writes_ledger_records_without_evidence_or_answers(tmp_path: Path):
    ledger = JsonlLedgerStore(tmp_path / "initialization-ledger.jsonl")

    BayesProbeInitializer(ledger=ledger).initialize(
        InitializeRunInput(
            run_id="run_ledger",
            problem="Should we trust the benchmark result?",
        )
    )

    record_types = [record["record_type"] for record in ledger.read_all()]

    assert record_types == ["run", "belief_state", "probe_candidate", "probe_candidate"]
    assert "evidence_event" not in record_types
    assert "belief_update" not in record_types
    assert "answer_projection" not in record_types


def test_initialized_belief_state_can_run_autonomous_loop():
    initialization = BayesProbeInitializer().initialize(
        InitializeRunInput(
            run_id="run_to_loop",
            problem="Is the deterministic initializer usable by the self-loop?",
        )
    )
    runner = AutonomousLoopRunner(
        core=BayesProbeCore(),
        config=AutonomousLoopConfig(max_cycles=1),
    )

    result = runner.run(
        run_id=initialization.run.run_id,
        initial_belief_state=initialization.belief_state,
        signal_provider=OneBatchSignalProvider(),
    )

    assert result.stop_reason == AutonomousStopReason.MAX_CYCLES
    assert len(result.cycle_results) == 1
    assert result.final_answer_projection is not None
    assert result.final_answer_projection.current_best_hypothesis == "H1"
