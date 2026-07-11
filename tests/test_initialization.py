from pathlib import Path

import pytest

from bayesprobe.core import BayesProbeCore
from bayesprobe.initialization import BayesProbeInitializer, HypothesisSeed, InitializeRunInput
from bayesprobe.ledger import JsonlLedgerStore
from bayesprobe.model_gateway import ScriptedModelGateway
from bayesprobe.runners import AutonomousLoopConfig, AutonomousLoopRunner, AutonomousStopReason
from bayesprobe.schemas import (
    AnswerContractOutline,
    AnswerValueType,
    ExternalSignal,
    HypothesisRelation,
    RunRegime,
    RunStatus,
    SignalKind,
    TaskAdmissionDecision,
    TaskAdmissionStatus,
    TaskKind,
)
from bayesprobe.task_admission import ExplicitTaskAdmitter
from bayesprobe.task_framing import ModelTaskFramer, TaskFramingError


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


def admitted_seed_decision(attempt_id: str = "admission_seeded") -> TaskAdmissionDecision:
    return TaskAdmissionDecision(
        attempt_id=attempt_id,
        status=TaskAdmissionStatus.ADMITTED,
        epistemic_basis=["The caller supplied explicit hypotheses."],
        proposed_task_kind=TaskKind.DECISION,
        answer_contract_outline=AnswerContractOutline(
            objective="Assess the supplied hypotheses.",
            answer_value_type=AnswerValueType.STRUCTURED_TEXT,
            decision_form="hypothesis_assessment",
            permits_synthesis=True,
            required_sections=["answer", "basis", "uncertainty"],
        ),
        reason="The explicit frame is admissible.",
    )


def admitted_exact_decision(attempt_id: str = "admission_exact") -> TaskAdmissionDecision:
    return TaskAdmissionDecision(
        attempt_id=attempt_id,
        status=TaskAdmissionStatus.ADMITTED,
        epistemic_basis=["The task has testable exact-answer candidates."],
        proposed_task_kind=TaskKind.EXACT_ANSWER,
        answer_contract_outline=AnswerContractOutline(
            objective="Return the supported integer.",
            answer_value_type=AnswerValueType.INTEGER,
            decision_form="single_value",
            permits_synthesis=False,
            required_sections=["answer", "basis", "uncertainty"],
        ),
        reason="The exact answer can be verified.",
    )


class FailingTaskAdmitter:
    def assess(self, input):
        raise AssertionError("runner-supplied admission must not be reassessed")


def test_initializer_uses_supplied_admission_once_and_writes_v02_state_in_order(
    tmp_path: Path,
):
    ledger = JsonlLedgerStore(tmp_path / "initialization-v02.jsonl")
    initializer = BayesProbeInitializer(
        ledger=ledger,
        task_admitter=FailingTaskAdmitter(),
    )

    result = initializer.initialize(
        InitializeRunInput(
            run_id="run_v02_seeded",
            problem="Which explanation best fits?",
            hypothesis_seeds=explicit_test_hypothesis_seeds(),
            task_kind=TaskKind.DECISION,
        ),
        admission_decision=admitted_seed_decision(),
    )

    assert result.task_frame.schema_version == "v0.2"
    assert result.task_frame.admission_decision_id == "admission_seeded"
    assert result.belief_state.schema_version == "v0.2"
    assert result.belief_state.frame_state.frame_id == (
        result.task_frame.hypothesis_frame.frame_id
    )
    assert result.belief_state.evidence_memory.accepted_evidence_ids == []
    assert [record["record_type"] for record in ledger.read_all()[:4]] == [
        "task_admission",
        "task_frame",
        "run",
        "belief_state",
    ]


def test_initializer_preserves_framed_answer_value_in_runtime_hypothesis():
    framed_answer_value = 7
    initializer = BayesProbeInitializer(
        task_framer=ModelTaskFramer(
            ScriptedModelGateway(
                {
                    "frame_open_question": {
                        "task_kind": "exact_answer",
                        "answer_relationship": "selection",
                        "answer_contract": {
                            "objective": "Return the supported integer value.",
                            "answer_value_type": "integer",
                            "answer_format": "integer",
                            "required_sections": ["answer", "basis", "uncertainty"],
                            "decision_form": "single_value",
                            "permits_synthesis": False,
                        },
                        "competition": "exclusive",
                        "coverage": "open",
                        "hypotheses": [
                            {
                                "statement": "The requested integer is 7.",
                                "type": "answer_candidate",
                                "scope": "The supplied integer constraints.",
                                "falsifiers": ["A constraint excludes 7."],
                                "predictions": [
                                    "Substitution verifies every constraint."
                                ],
                                "answer_value": framed_answer_value,
                            }
                        ],
                        "coverage_statement": "The named value is an initial candidate.",
                        "coverage_limitation": "Other integer values remain unresolved.",
                    }
                }
            )
        ),
        task_admitter=FailingTaskAdmitter(),
    )

    result = initializer.initialize(
        InitializeRunInput(
            run_id="run_exact_answer_value",
            problem="Which integer satisfies the constraints?",
        ),
        admission_decision=admitted_exact_decision(),
    )

    assert result.task_frame.hypothesis_frame.hypotheses[0].answer_value == 7
    assert result.belief_state.hypotheses[0].answer_value == 7


def test_initializer_default_admitter_fails_closed_for_unseeded_open_input():
    initializer = BayesProbeInitializer(task_admitter=ExplicitTaskAdmitter())

    with pytest.raises(TaskFramingError, match="requires answer choices or hypothesis seeds"):
        initializer.initialize(
            InitializeRunInput(
                run_id="run_unseeded_closed",
                problem="How should this open question be framed?",
            )
        )


@pytest.mark.parametrize(
    "task_kind",
    [TaskKind.EXACT_ANSWER, TaskKind.MULTIPLE_CHOICE],
)
def test_initializer_creates_no_state_for_answer_valued_seed_task_kind(
    tmp_path,
    task_kind,
):
    ledger = JsonlLedgerStore(tmp_path / f"invalid_seed_{task_kind.value}.jsonl")
    initializer = BayesProbeInitializer(
        ledger=ledger,
        task_admitter=ExplicitTaskAdmitter(),
    )

    with pytest.raises(TaskFramingError, match="hypothesis seeds cannot frame"):
        initializer.initialize(
            InitializeRunInput(
                run_id=f"invalid_seed_{task_kind.value}",
                problem="Which answer is supported?",
                task_kind=task_kind,
                hypothesis_seeds=explicit_test_hypothesis_seeds(),
            )
        )

    for record_type in ("task_admission", "task_frame", "run", "belief_state"):
        assert ledger.read_all(record_type) == []


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
        "task_admission",
        "task_frame",
        "run",
        "belief_state",
        "probe_candidate",
        "probe_candidate",
    ]
    assert "evidence_event" not in record_types
    assert "belief_update" not in record_types
    assert "answer_projection" not in record_types


@pytest.mark.parametrize(
    "context",
    [
        "Source includes sk-abcdefghijklmnop",
        "password = correct-horse-battery-staple",
        "Authorization: Bearer abcdefghijklmnop",
        "-----BEGIN PRIVATE KEY-----",
        "access_key='AKIAEXAMPLEVALUE'",
        "ghp_" + "a" * 36,
        (
            "eyJhbGciOiJIUzI1NiJ9."
            "eyJzdWIiOiIxMjM0NTY3ODkwIn0."
            "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        ),
        "Bearer abcdefghijklmnopqrstuvwx",
    ],
)
def test_initializer_rejects_secret_compatibility_context_before_materialization(
    tmp_path: Path,
    context: str,
):
    ledger_path = tmp_path / "secret-context-initialization.jsonl"
    ledger = JsonlLedgerStore(ledger_path)

    with pytest.raises(
        TaskFramingError,
        match="compatibility context must not contain secret material",
    ) as captured:
        BayesProbeInitializer(ledger=ledger).initialize(
            InitializeRunInput(
                run_id="run_secret_compatibility_context",
                problem="Should this source change the belief state?",
                context=context,
                hypothesis_seeds=explicit_test_hypothesis_seeds(),
            )
        )

    assert not ledger_path.exists()
    assert context not in str(captured.value)
    assert context not in repr(captured.value)


def test_initializer_preserves_ordinary_compatibility_source_text():
    result = BayesProbeInitializer().initialize(
        InitializeRunInput(
            run_id="run_ordinary_compatibility_context",
            problem="Should this source change the belief state?",
            context="The source compares password policies and access key rotation.",
            hypothesis_seeds=explicit_test_hypothesis_seeds(),
        )
    )

    assert result.run.metadata["context_provided"] is True


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
