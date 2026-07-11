from pathlib import Path

import pytest

from bayesprobe.core import BayesProbeCore
from bayesprobe.initialization import BayesProbeInitializer, HypothesisSeed, InitializeRunInput
from bayesprobe.ledger import JsonlLedgerStore
from bayesprobe.model_gateway import StructuredModelRequest
from bayesprobe.recorded_gateway import RecordedModelGateway
from bayesprobe.probe_executor import (
    ModelBackedProbeToolGateway,
    ProbeExecutionResult,
    ProbeExecutor,
)
from bayesprobe.probe_planner import ProbePlanningResult
from bayesprobe.question_runner import (
    AutonomousQuestionRunConfig,
    AutonomousQuestionRunner,
    AutonomousQuestionProgressKind,
    AutonomousQuestionStopReason,
)
from bayesprobe.task_framing import ModelTaskFramer
from bayesprobe.schemas import (
    CycleSignalShape,
    HypothesisRelation,
    ProbeSet,
    RunRegime,
    RunStatus,
    SignalKind,
)


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


def explicit_test_hypothesis_seeds() -> list[HypothesisSeed]:
    return [
        HypothesisSeed(id="H1", statement="The fixture's H1 condition holds.", prior=0.5, scope="Deterministic test fixture.", falsifiers=["The fixture emits a reliable H1 refutation."], predictions=["The fixture emits a reliable H1 support cue."]),
        HypothesisSeed(id="H2", statement="The fixture's H2 condition holds instead.", prior=0.5, scope="Deterministic test fixture.", falsifiers=["The fixture emits a reliable H2 refutation."], predictions=["The fixture emits a reliable H2 support cue."]),
    ]


def valid_open_frame_payload() -> dict[str, object]:
    return {
        "task_kind": "claim_verification",
        "answer_contract": {
            "objective": "Design a discriminating validation protocol.",
            "required_sections": ["hypotheses", "controls", "decision_rule"],
            "decision_form": "experimental_protocol",
            "permits_synthesis": True,
        },
        "hypothesis_relation": "independent",
        "hypotheses": [
            {
                "statement": "Scale has an independent effect under matched conditions.",
                "type": "causal_claim",
                "scope": "Matched task and resource conditions.",
                "falsifiers": ["The controlled effect is negligible."],
                "predictions": ["Matched performance rises with size."],
            },
            {
                "statement": "The apparent effect is materially confounded.",
                "type": "confounding_explanation",
                "scope": "Unmatched comparisons.",
                "falsifiers": ["The effect survives matched controls."],
                "predictions": ["The effect shrinks after matching."],
            },
        ],
        "coverage_statement": "Covers the effect and its primary confounder.",
        "coverage_limitation": "Task interactions may remain.",
    }


class RecordingOpenQuestionGateway:
    adapter_kind = "recording_open_question_test"

    def __init__(self) -> None:
        self.requests: list[StructuredModelRequest] = []

    def complete_structured(self, request: StructuredModelRequest) -> dict[str, object]:
        self.requests.append(request)
        if request.task == "frame_open_question":
            return valid_open_frame_payload()
        if request.task == "execute_probe":
            return {"raw_content": "MODEL REASONING: A matched controlled test is required."}
        if request.task == "judge_evidence":
            return {
                "evidence_type": "supporting",
                "likelihoods": {
                    hypothesis_id: "weakly_confirming"
                    for hypothesis_id in request.input["target_hypotheses"]
                },
                "interpretation": "A design suggestion, not an external result.",
                "quality_overrides": {"independence": 0.2, "verifiability": 0.2},
            }
        raise AssertionError(f"unexpected task: {request.task}")


def test_open_question_framing_precedes_belief_initialization():
    gateway = RecordingOpenQuestionGateway()
    observed = []

    def observe(event):
        observed.append((event, [request.task for request in gateway.requests]))

    runner = AutonomousQuestionRunner(
        core=BayesProbeCore(model_gateway=gateway),
        initializer=BayesProbeInitializer(task_framer=ModelTaskFramer(gateway)),
        executor=ProbeExecutor(ModelBackedProbeToolGateway(gateway)),
        config=AutonomousQuestionRunConfig(max_cycles=1, max_probes_per_cycle=1),
        progress_observer=observe,
    )

    result = runner.run_question(
        InitializeRunInput(
            run_id="run_open_progress",
            problem="A team claims that a larger model always improves agent performance. How should it be tested?",
            task_context="Design an experiment for a research audience.",
            context="SUPPORTS: An earlier benchmark showed a scale trend.",
        )
    )

    events = [event for event, _ in observed]
    assert [event.kind for event in events[:4]] == [
        AutonomousQuestionProgressKind.RUN_STARTED,
        AutonomousQuestionProgressKind.TASK_FRAMING_STARTED,
        AutonomousQuestionProgressKind.TASK_FRAMING_COMPLETED,
        AutonomousQuestionProgressKind.INITIALIZATION_COMPLETED,
    ]
    assert events[1].belief_state is None
    assert events[2].belief_state is None
    assert events[2].task_frame is not None
    assert observed[1][1] == []
    assert observed[2][1] == ["frame_open_question"]
    assert gateway.requests[0].task == "frame_open_question"
    assert gateway.requests[0].input["task_context"] == (
        "Design an experiment for a research audience."
    )
    assert "SUPPORTS: An earlier benchmark" not in str(gateway.requests[0].input)
    assert next(
        request for request in gateway.requests if request.task == "execute_probe"
    ).input["task_context"] == "Design an experiment for a research audience."
    assert result.initial_belief_state.task_frame == result.task_frame
    assert result.final_answer_projection.posterior_summary.startswith(
        "Credences (not normalized):"
    )
    assert "independent hypotheses may coexist" in (
        result.final_answer_projection.main_uncertainty
    )


def test_recorded_open_question_frames_before_running_cycle():
    gateway = RecordedModelGateway.from_json(
        Path("tests/fixtures/open_questions/model_scale_validation_v0.1.json")
    )
    runner = AutonomousQuestionRunner(
        core=BayesProbeCore(model_gateway=gateway),
        initializer=BayesProbeInitializer(task_framer=ModelTaskFramer(gateway)),
        executor=ProbeExecutor(ModelBackedProbeToolGateway(gateway)),
        config=AutonomousQuestionRunConfig(max_cycles=1, max_probes_per_cycle=1),
    )

    result = runner.run_question(
        InitializeRunInput(
            run_id="recorded_model_scale",
            problem=(
                "某团队认为‘模型变大一定能提升 agent 的真实任务表现’。"
                "这个命题应该如何验证？"
            ),
        )
    )

    assert [request.task for request in gateway.requests] == [
        "frame_open_question",
        "execute_probe",
        "judge_evidence",
    ]
    statements = [
        item.statement for item in result.task_frame.hypothesis_frame.hypotheses
    ]
    assert len(statements) == len(set(statements)) == 3
    assert all("这个命题应该如何验证" not in statement for statement in statements)
    assert any("独立正向效应" in statement for statement in statements)
    assert any("混杂" in statement for statement in statements)
    assert any("任务和工具条件" in statement for statement in statements)
    assert result.initial_belief_state.task_frame == result.task_frame
    assert result.cycle_results[0].signals[0].source_type == "model_probe_gateway"
    final_by_id = result.final_belief_state.hypotheses_by_id()
    assert final_by_id["H1"].posterior > 0.5
    assert final_by_id["H2"].posterior == 0.5
    assert final_by_id["H3"].posterior == 0.5
    assert sum(item.posterior for item in final_by_id.values()) > 1.0


def test_question_runner_executes_one_end_to_end_cycle():
    runner = AutonomousQuestionRunner(
        core=BayesProbeCore(),
        config=AutonomousQuestionRunConfig(max_cycles=1, max_probes_per_cycle=1),
    )

    result = runner.run_question(
        InitializeRunInput(
            run_id="run_question_1",
            problem="Does the active BayesProbe path work end to end?",
            hypothesis_seeds=explicit_test_hypothesis_seeds(),
        )
    )

    cycle_result = result.cycle_results[0]
    assert result.stop_reason == AutonomousQuestionStopReason.MAX_CYCLES
    assert result.run.run_id == "run_question_1"
    assert result.run.regime == RunRegime.AUTONOMOUS
    assert result.run.status == RunStatus.COMPLETED
    assert result.run.current_cycle_id == result.final_belief_state.cycle_id
    assert result.run.metadata["stop_reason"] == result.stop_reason.value
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
            hypothesis_seeds=explicit_test_hypothesis_seeds(),
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
            hypothesis_seeds=explicit_test_hypothesis_seeds(),
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
            hypothesis_seeds=explicit_test_hypothesis_seeds(),
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
            hypothesis_seeds=explicit_test_hypothesis_seeds(),
        )
    )

    assert result.stop_reason == AutonomousQuestionStopReason.CONFIDENCE_REACHED
    assert len(result.cycle_results) == 1
    assert result.final_answer_projection is not None
    assert result.final_belief_state.hypotheses_by_id()["H1"].posterior >= 0.6


def test_question_runner_does_not_apply_winner_threshold_to_independent_credences():
    runner = AutonomousQuestionRunner(
        core=BayesProbeCore(),
        config=AutonomousQuestionRunConfig(
            max_cycles=2,
            max_probes_per_cycle=1,
            confidence_threshold=0.6,
        ),
    )

    result = runner.run_question(
        InitializeRunInput(
            run_id="run_question_independent_threshold",
            problem="Which independent claims remain credible?",
            hypothesis_relation=HypothesisRelation.INDEPENDENT,
            hypothesis_seeds=[
                HypothesisSeed(id="H1", statement="Claim one remains credible.", prior=0.8),
                HypothesisSeed(id="H2", statement="Claim two remains credible.", prior=0.7),
            ],
        )
    )

    assert result.stop_reason == AutonomousQuestionStopReason.MAX_CYCLES
    assert len(result.cycle_results) == 2


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
            hypothesis_seeds=explicit_test_hypothesis_seeds(),
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
            hypothesis_seeds=explicit_test_hypothesis_seeds(),
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
            hypothesis_seeds=explicit_test_hypothesis_seeds(),
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
            hypothesis_seeds=explicit_test_hypothesis_seeds(),
        )
    )

    record_types = [record["record_type"] for record in ledger.read_all()]
    assert record_types.count("cycle") == 1
    assert record_types.count("answer_projection") == 1
    assert record_types.count("probe_planning") == 1
    assert record_types.count("probe_execution") == 1
    assert record_types.count("probe_set") == 1
    assert record_types.count("external_signal") == 1


def test_question_runner_emits_truthful_progress_for_integrated_cycle():
    events = []
    runner = AutonomousQuestionRunner(
        core=BayesProbeCore(),
        config=AutonomousQuestionRunConfig(max_cycles=1, max_probes_per_cycle=1),
        progress_observer=events.append,
    )

    result = runner.run_question(
        InitializeRunInput(
            run_id="run_question_progress",
            problem="Does progress follow the BayesProbe lifecycle?",
            hypothesis_seeds=explicit_test_hypothesis_seeds(),
        )
    )

    assert [event.kind for event in events] == [
        AutonomousQuestionProgressKind.RUN_STARTED,
        AutonomousQuestionProgressKind.TASK_FRAMING_STARTED,
        AutonomousQuestionProgressKind.TASK_FRAMING_COMPLETED,
        AutonomousQuestionProgressKind.INITIALIZATION_COMPLETED,
        AutonomousQuestionProgressKind.CYCLE_STARTED,
        AutonomousQuestionProgressKind.PROBE_SET_PLANNED,
        AutonomousQuestionProgressKind.PROBE_EXECUTION_STARTED,
        AutonomousQuestionProgressKind.SIGNALS_COLLECTED,
        AutonomousQuestionProgressKind.EVIDENCE_INTEGRATION_STARTED,
        AutonomousQuestionProgressKind.CYCLE_INTEGRATED,
        AutonomousQuestionProgressKind.RUN_COMPLETED,
    ]
    cycle_event = events[-2]
    assert cycle_event.cycle_result == result.cycle_results[0]
    assert cycle_event.cycle_result.cycle.boundary_status.value == "integrated"
    assert sum(
        hypothesis.posterior
        for hypothesis in cycle_event.cycle_result.belief_state.hypotheses
    ) == pytest.approx(1.0)
    assert events[-1].result == result


def test_question_runner_progress_ends_once_when_no_probe_cycle_is_created():
    events = []
    runner = AutonomousQuestionRunner(
        core=BayesProbeCore(),
        planner=EmptyPlanner(),
        executor=RecordingExecutor(),
        config=AutonomousQuestionRunConfig(max_cycles=2, stop_on_no_probes=True),
        progress_observer=events.append,
    )

    result = runner.run_question(
        InitializeRunInput(
            run_id="run_question_progress_no_probes",
            problem="What happens when progress reaches an empty probe set?",
            hypothesis_seeds=explicit_test_hypothesis_seeds(),
        )
    )

    assert result.stop_reason == AutonomousQuestionStopReason.NO_PROBES
    assert [event.kind for event in events][-1] == (
        AutonomousQuestionProgressKind.RUN_COMPLETED
    )
    assert sum(
        event.kind == AutonomousQuestionProgressKind.RUN_COMPLETED
        for event in events
    ) == 1
    assert all(
        event.kind != AutonomousQuestionProgressKind.CYCLE_INTEGRATED
        for event in events
    )


def test_question_runner_repeats_cycle_progress_for_each_integrated_cycle():
    events = []
    runner = AutonomousQuestionRunner(
        core=BayesProbeCore(),
        config=AutonomousQuestionRunConfig(max_cycles=2, max_probes_per_cycle=1),
        progress_observer=events.append,
    )

    result = runner.run_question(
        InitializeRunInput(
            run_id="run_question_progress_multi",
            problem="Can progress report both autonomous cycles?",
            hypothesis_seeds=explicit_test_hypothesis_seeds(),
        )
    )

    integrated = [
        event
        for event in events
        if event.kind == AutonomousQuestionProgressKind.CYCLE_INTEGRATED
    ]
    assert [event.cycle_index for event in integrated] == [1, 2]
    assert [event.cycle_result for event in integrated] == result.cycle_results


def test_question_runner_ignores_progress_observer_exceptions():
    def failing_observer(event):
        raise RuntimeError(f"observer failed for {event.kind}")

    runner = AutonomousQuestionRunner(
        core=BayesProbeCore(),
        config=AutonomousQuestionRunConfig(max_cycles=1, max_probes_per_cycle=1),
        progress_observer=failing_observer,
    )

    result = runner.run_question(
        InitializeRunInput(
            run_id="run_question_progress_observer_failure",
            problem="Does observer failure leave the autonomous run intact?",
            hypothesis_seeds=explicit_test_hypothesis_seeds(),
        )
    )

    assert result.run.status == RunStatus.COMPLETED
    assert result.stop_reason == AutonomousQuestionStopReason.MAX_CYCLES
    assert len(result.cycle_results) == 1
    assert result.cycle_results[0].cycle.boundary_status.value == "integrated"


def test_question_runner_progress_observer_receives_detached_deep_snapshots():
    observed_kinds = []
    mutated_fields = set()

    def hostile_observer(event):
        observed_kinds.append(event.kind)
        if event.run is not None:
            event.run.metadata.clear()
            event.run.metadata["stop_reason"] = "observer_mutation"
            mutated_fields.add("run")
        if event.belief_state is not None:
            event.belief_state.posterior_summary.clear()
            event.belief_state.hypotheses[0].posterior = 0.0
            event.belief_state.hypotheses[0].rivals.clear()
            mutated_fields.add("belief_state")
        if event.probe_set is not None:
            for probe in event.probe_set.probes:
                probe.target_hypotheses.clear()
                probe.support_condition.clear()
            event.probe_set.probes.clear()
            mutated_fields.add("probe_set")
        if event.signals:
            event.signals[0].raw_content = "observer mutation"
            event.signals[0].initial_target_hypotheses.clear()
            mutated_fields.add("signals")
        if event.cycle_result is not None:
            event.cycle_result.signals.clear()
            event.cycle_result.evidence_events.clear()
            event.cycle_result.belief_updates.clear()
            event.cycle_result.hypothesis_evolutions.clear()
            event.cycle_result.answer_projection.main_evidence_events.clear()
            mutated_fields.add("cycle_result")
        if event.result is not None:
            event.result.cycle_results.clear()
            event.result.final_belief_state.hypotheses.clear()
            event.result.final_answer_projection.main_evidence_events.clear()
            mutated_fields.add("result")

    runner = AutonomousQuestionRunner(
        core=BayesProbeCore(),
        config=AutonomousQuestionRunConfig(max_cycles=2, max_probes_per_cycle=1),
        progress_observer=hostile_observer,
    )

    result = runner.run_question(
        InitializeRunInput(
            run_id="run_question_hostile_progress_observer",
            problem="Can observer mutation alter the autonomous lifecycle?",
            hypothesis_seeds=explicit_test_hypothesis_seeds(),
            context="SUPPORTS: Keep a passive signal in the first cycle.",
        )
    )

    per_cycle = [
        AutonomousQuestionProgressKind.CYCLE_STARTED,
        AutonomousQuestionProgressKind.PROBE_SET_PLANNED,
        AutonomousQuestionProgressKind.PROBE_EXECUTION_STARTED,
        AutonomousQuestionProgressKind.SIGNALS_COLLECTED,
        AutonomousQuestionProgressKind.EVIDENCE_INTEGRATION_STARTED,
        AutonomousQuestionProgressKind.CYCLE_INTEGRATED,
    ]
    assert observed_kinds == [
        AutonomousQuestionProgressKind.RUN_STARTED,
        AutonomousQuestionProgressKind.TASK_FRAMING_STARTED,
        AutonomousQuestionProgressKind.TASK_FRAMING_COMPLETED,
        AutonomousQuestionProgressKind.INITIALIZATION_COMPLETED,
        *per_cycle,
        *per_cycle,
        AutonomousQuestionProgressKind.RUN_COMPLETED,
    ]
    assert mutated_fields == {
        "run",
        "belief_state",
        "probe_set",
        "signals",
        "cycle_result",
        "result",
    }
    assert result.stop_reason == AutonomousQuestionStopReason.MAX_CYCLES
    assert result.run.metadata["stop_reason"] == "max_cycles"
    assert len(result.cycle_results) == 2
    assert all(cycle.probe_set.probes for cycle in result.cycle_results)
    assert all(cycle.signals for cycle in result.cycle_results)
    assert all(cycle.evidence_events for cycle in result.cycle_results)
    assert all(cycle.belief_updates for cycle in result.cycle_results)
    assert all(
        signal.raw_content != "observer mutation"
        for cycle in result.cycle_results
        for signal in cycle.signals
    )
    assert sum(
        hypothesis.posterior for hypothesis in result.final_belief_state.hypotheses
    ) == pytest.approx(1.0)
