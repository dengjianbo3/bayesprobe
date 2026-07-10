from pathlib import Path

import pytest

from bayesprobe.core import BayesProbeCore
from bayesprobe.initialization import InitializeRunInput
from bayesprobe.ledger import JsonlLedgerStore
from bayesprobe.probe_executor import ProbeExecutionResult
from bayesprobe.probe_planner import ProbePlanningResult
from bayesprobe.question_runner import (
    AutonomousQuestionRunConfig,
    AutonomousQuestionRunner,
    AutonomousQuestionStopReason,
)
from bayesprobe.schemas import CycleSignalShape, ProbeSet, SignalKind


class EmptyPlanner:
    def __init__(self):
        self.calls = 0

    def design_probe_set(self, *, run_id, cycle_id, belief_state, candidates, config):
        self.calls += 1
        probe_set = ProbeSet(
            probe_set_id=f"ps_{cycle_id}",
            cycle_id=cycle_id,
            probes=[],
            selection_reason="empty planner fixture",
            may_be_empty=True,
        )
        return ProbePlanningResult(
            probe_set=probe_set,
            selected_candidates=[],
            rejected_candidates=[],
        )


class RecordingExecutor:
    def __init__(self):
        self.calls = 0

    def execute_probe_set(self, *, probe_set, context):
        self.calls += 1
        return ProbeExecutionResult(
            probe_set=probe_set,
            signals=[],
            executed_probe_ids=[],
        )


def test_question_runner_executes_one_end_to_end_cycle():
    runner = AutonomousQuestionRunner(
        core=BayesProbeCore(),
        config=AutonomousQuestionRunConfig(max_cycles=1, max_probes_per_cycle=1),
    )

    result = runner.run_question(
        InitializeRunInput(
            run_id="run_question_1",
            problem="Does the active BayesProbe path work end to end?",
        )
    )

    cycle_result = result.cycle_results[0]
    assert result.stop_reason == AutonomousQuestionStopReason.MAX_CYCLES
    assert result.run.run_id == "run_question_1"
    assert result.initial_belief_state.cycle_id == "cycle_0"
    assert result.final_belief_state == cycle_result.belief_state
    assert result.final_answer_projection == cycle_result.answer_projection
    assert cycle_result.cycle.cycle_id == "run_question_1_cycle_1"
    assert cycle_result.probe_set.probes
    assert cycle_result.signals
    assert cycle_result.evidence_events
    assert cycle_result.belief_updates
    assert cycle_result.answer_projection.current_best_hypothesis
    assert cycle_result.signals[0].generated_by_probe == cycle_result.probe_set.probes[0].id


def test_question_runner_integrates_initial_context_as_passive_signal():
    runner = AutonomousQuestionRunner(
        core=BayesProbeCore(),
        config=AutonomousQuestionRunConfig(max_cycles=1, max_probes_per_cycle=1),
    )

    result = runner.run_question(
        InitializeRunInput(
            run_id="run_question_context",
            problem="Does supplied context enter the evidence path?",
            context="REFUTES: A human reviewer found a counterexample.",
        )
    )

    cycle_result = result.cycle_results[0]
    passive_signals = [
        signal
        for signal in cycle_result.signals
        if signal.signal_kind == SignalKind.PASSIVE
    ]

    assert cycle_result.cycle.signal_shape == CycleSignalShape.ACTIVE_PLUS_PASSIVE
    assert len(passive_signals) == 1
    assert passive_signals[0].source_type == "initial_context"
    assert passive_signals[0].source == "user_context"
    assert passive_signals[0].raw_content == (
        "REFUTES: A human reviewer found a counterexample."
    )
    assert passive_signals[0].generated_by_probe is None
    assert passive_signals[0].initial_target_hypotheses == ["H1", "H2"]
    assert any(
        event.derived_from_signal == passive_signals[0].id
        for event in cycle_result.evidence_events
    )


def test_question_runner_projection_replaces_stale_initial_uncertainty():
    runner = AutonomousQuestionRunner(
        core=BayesProbeCore(),
        config=AutonomousQuestionRunConfig(max_cycles=1, max_probes_per_cycle=1),
    )

    result = runner.run_question(
        InitializeRunInput(
            run_id="run_question_uncertainty",
            problem="Does the projection describe the current posterior uncertainty?",
        )
    )

    uncertainty = result.final_answer_projection.main_uncertainty
    assert "no external signals have been integrated yet" not in uncertainty
    assert "posterior gap between H1 and H2" in uncertainty
    assert "further discriminative evidence" in uncertainty


def test_question_runner_runs_multiple_cycles_with_candidate_pool_from_projection():
    runner = AutonomousQuestionRunner(
        core=BayesProbeCore(),
        config=AutonomousQuestionRunConfig(max_cycles=2, max_probes_per_cycle=1),
    )

    result = runner.run_question(
        InitializeRunInput(
            run_id="run_question_multi",
            problem="Can later cycles use projection-derived probe candidates?",
        )
    )

    first_cycle = result.cycle_results[0]
    second_cycle = result.cycle_results[1]
    first_projection_candidate = (
        first_cycle.answer_projection.change_my_mind_condition.structured_probe_candidates[0]
    )

    assert result.stop_reason == AutonomousQuestionStopReason.MAX_CYCLES
    assert len(result.cycle_results) == 2
    assert first_cycle.cycle.cycle_id == "run_question_multi_cycle_1"
    assert second_cycle.cycle.cycle_id == "run_question_multi_cycle_2"
    assert second_cycle.probe_set.probes[0].id.startswith(
        first_projection_candidate.candidate_probe.id
    )
    assert second_cycle.probe_set.probes[0].cycle_id == "run_question_multi_cycle_2"


def test_question_runner_stops_on_confidence_threshold():
    runner = AutonomousQuestionRunner(
        core=BayesProbeCore(),
        config=AutonomousQuestionRunConfig(
            max_cycles=3,
            max_probes_per_cycle=1,
            confidence_threshold=0.6,
        ),
    )

    result = runner.run_question(
        InitializeRunInput(
            run_id="run_question_confident",
            problem="Does one supportive deterministic probe cross confidence?",
        )
    )

    assert result.stop_reason == AutonomousQuestionStopReason.CONFIDENCE_REACHED
    assert len(result.cycle_results) == 1
    assert result.final_answer_projection is not None
    assert result.final_belief_state.hypotheses_by_id()["H1"].posterior >= 0.6


def test_question_runner_stops_on_no_probes_before_empty_cycle():
    planner = EmptyPlanner()
    executor = RecordingExecutor()
    runner = AutonomousQuestionRunner(
        core=BayesProbeCore(),
        planner=planner,
        executor=executor,
        config=AutonomousQuestionRunConfig(max_cycles=2, stop_on_no_probes=True),
    )

    result = runner.run_question(
        InitializeRunInput(
            run_id="run_question_no_probes",
            problem="What happens when no probes are available?",
        )
    )

    assert result.stop_reason == AutonomousQuestionStopReason.NO_PROBES
    assert result.cycle_results == []
    assert result.final_answer_projection is None
    assert result.final_belief_state == result.initial_belief_state
    assert planner.calls == 1
    assert executor.calls == 0


def test_question_runner_integrates_context_before_stopping_on_no_probes():
    planner = EmptyPlanner()
    executor = RecordingExecutor()
    runner = AutonomousQuestionRunner(
        core=BayesProbeCore(),
        planner=planner,
        executor=executor,
        config=AutonomousQuestionRunConfig(max_cycles=2, stop_on_no_probes=True),
    )

    result = runner.run_question(
        InitializeRunInput(
            run_id="run_question_passive_context",
            problem="Can context form a passive-only autonomous cycle?",
            context="SUPPORTS: A human supplied relevant information.",
        )
    )

    assert result.stop_reason == AutonomousQuestionStopReason.NO_PROBES
    assert len(result.cycle_results) == 1
    assert executor.calls == 1
    assert result.cycle_results[0].cycle.signal_shape == CycleSignalShape.PASSIVE_ONLY
    assert result.cycle_results[0].probe_set.probes == []
    assert [
        signal.signal_kind for signal in result.cycle_results[0].signals
    ] == [SignalKind.PASSIVE]
    assert result.final_answer_projection is not None


@pytest.mark.parametrize(
    "config_kwargs",
    [
        {"max_cycles": 0},
        {"max_probes_per_cycle": 0},
        {"confidence_threshold": -0.1},
        {"confidence_threshold": 1.1},
        {"posterior_delta_threshold": -0.1},
    ],
)
def test_question_runner_rejects_invalid_config(config_kwargs):
    with pytest.raises(ValueError):
        AutonomousQuestionRunConfig(**config_kwargs)


def test_question_runner_writes_end_to_end_ledger_records(tmp_path: Path):
    ledger = JsonlLedgerStore(tmp_path / "question-runner-ledger.jsonl")
    runner = AutonomousQuestionRunner(
        core=BayesProbeCore(ledger=ledger),
        config=AutonomousQuestionRunConfig(max_cycles=1, max_probes_per_cycle=1),
    )

    runner.run_question(
        InitializeRunInput(
            run_id="run_question_ledger",
            problem="Does the orchestrator write a coherent ledger?",
        )
    )

    record_types = [record["record_type"] for record in ledger.read_all()]
    assert "run" in record_types
    assert "probe_candidate" in record_types
    assert "probe_set" in record_types
    assert "probe_execution" in record_types
    assert "external_signal" in record_types
    assert "evidence_event" in record_types
    assert "belief_update" in record_types
    assert "belief_state" in record_types
    assert "answer_projection" in record_types


def test_question_runner_does_not_duplicate_core_integration(tmp_path: Path):
    ledger = JsonlLedgerStore(tmp_path / "question-runner-no-duplicate.jsonl")
    runner = AutonomousQuestionRunner(
        core=BayesProbeCore(ledger=ledger),
        config=AutonomousQuestionRunConfig(max_cycles=1, max_probes_per_cycle=1),
    )

    runner.run_question(
        InitializeRunInput(
            run_id="run_question_single_core",
            problem="Does the orchestrator integrate exactly once per cycle?",
        )
    )

    record_types = [record["record_type"] for record in ledger.read_all()]
    assert record_types.count("cycle") == 1
    assert record_types.count("answer_projection") == 1
