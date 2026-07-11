from pathlib import Path

import pytest

from bayesprobe.core import BayesProbeCore
from bayesprobe.initialization import BayesProbeInitializer, HypothesisSeed, InitializeRunInput
from bayesprobe.ledger import JsonlLedgerStore
from bayesprobe.runners import AutonomousLoopConfig, AutonomousLoopRunner, AutonomousStopReason
from bayesprobe.schemas import (
    ExternalSignal,
    HypothesisRelation,
    RunRegime,
    RunStatus,
    SignalKind,
)
from bayesprobe.task_framing import TaskFramingError


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


def explicit_test_hypothesis_seeds() -> list[HypothesisSeed]:
    return [
        HypothesisSeed(
            id="H1",
            statement="The fixture's H1 condition holds.",
            prior=0.5,
            scope="Deterministic test fixture.",
            falsifiers=["The fixture emits a reliable H1 refutation."],
            predictions=["The fixture emits a reliable H1 support cue."],
        ),
        HypothesisSeed(
            id="H2",
            statement="The fixture's H2 condition holds instead.",
            prior=0.5,
            scope="Deterministic test fixture.",
            falsifiers=["The fixture emits a reliable H2 refutation."],
            predictions=["The fixture emits a reliable H2 support cue."],
        ),
    ]


def test_initializer_never_creates_generic_binary_hypotheses_for_open_question():
    with pytest.raises(TaskFramingError):
        BayesProbeInitializer().initialize(
            InitializeRunInput(
                run_id="run_open",
                problem="某团队认为模型变大一定提升 agent 表现，应该如何验证？",
            )
        )


def test_initializer_creates_answer_choice_hypotheses_from_multiple_choice_problem():
    problem = """Which graph class is well-behaved?

Answer Choices:
A. The class of all non-bipartite regular graphs
B. The class of all connected cubic graphs
C. The class of all connected graphs
D. The class of all connected non-bipartite graphs
E. The class of all connected bipartite graphs."""

    result = BayesProbeInitializer().initialize(
        InitializeRunInput(run_id="run_mcq", problem=problem)
    )

    hypotheses = result.belief_state.hypotheses_by_id()

    assert list(hypotheses) == ["A", "B", "C", "D", "E"]
    assert result.run.metadata["question_frame"] == "multiple_choice"
    assert result.belief_state.posterior_summary["hypothesis_count"] == 5
    assert hypotheses["D"].statement == (
        "Answer choice D is correct: The class of all connected non-bipartite graphs"
    )
    assert hypotheses["D"].prior == pytest.approx(0.2)
    assert hypotheses["D"].rivals == ["A", "B", "C", "E"]

    first_candidate = result.probe_candidates[0]
    assert first_candidate.source == "manual"
    assert first_candidate.priority_features["probe_role"] == "answer_choice_discriminator"
    assert first_candidate.candidate_probe.target_hypotheses == ["A", "B", "C", "D", "E"]
    assert "which answer choice is best" in first_candidate.candidate_probe.inquiry_goal.lower()


def test_initializer_parses_inline_answer_choices_without_binary_fallback():
    problem = (
        "Which graph class is well-behaved? Answer Choices: "
        "A. Non-bipartite regular graphs "
        "B. Connected cubic graphs "
        "C. Connected graphs "
        "D. Connected non-bipartite graphs "
        "E. Connected bipartite graphs"
    )

    result = BayesProbeInitializer().initialize(
        InitializeRunInput(run_id="run_mcq_inline", problem=problem)
    )

    assert [hypothesis.id for hypothesis in result.belief_state.hypotheses] == [
        "A",
        "B",
        "C",
        "D",
        "E",
    ]
    assert result.run.metadata["question_frame"] == "multiple_choice"


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


def test_initializer_summarizes_independent_values_as_non_normalized_credence():
    result = BayesProbeInitializer().initialize(
        InitializeRunInput(
            run_id="run_independent",
            problem="Which mechanisms remain credible?",
            hypothesis_relation=HypothesisRelation.INDEPENDENT,
            hypothesis_seeds=[
                HypothesisSeed(statement=f"Mechanism {index} remains plausible.")
                for index in range(1, 4)
            ],
        )
    )

    assert [item.posterior for item in result.belief_state.hypotheses] == [
        0.5,
        0.5,
        0.5,
    ]
    summary = result.belief_state.posterior_summary
    assert summary["belief_measure"] == "credence"
    assert summary["total_active_credence"] == 1.5
    assert "total_active_posterior" not in summary
    assert "top_posterior" not in summary


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
            hypothesis_seeds=explicit_test_hypothesis_seeds(),
        )
    )

    record_types = [record["record_type"] for record in ledger.read_all()]

    assert record_types == [
        "task_frame",
        "run",
        "belief_state",
        "probe_candidate",
        "probe_candidate",
    ]
    assert "evidence_event" not in record_types
    assert "belief_update" not in record_types
    assert "answer_projection" not in record_types


def test_initialized_belief_state_can_run_autonomous_loop():
    initialization = BayesProbeInitializer().initialize(
        InitializeRunInput(
            run_id="run_to_loop",
            problem="Is the deterministic initializer usable by the self-loop?",
            hypothesis_seeds=explicit_test_hypothesis_seeds(),
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
