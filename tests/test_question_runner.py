from dataclasses import replace
from pathlib import Path

import pytest

from bayesprobe.core import BayesProbeCore
from bayesprobe.hypothesis_expansion import (
    HypothesisExpansionService,
    ModelHypothesisExpansionAdapter,
)
from bayesprobe.initialization import BayesProbeInitializer, HypothesisSeed, InitializeRunInput
from bayesprobe.ledger import JsonlLedgerStore
from bayesprobe.model_gateway import ScriptedModelGateway, StructuredModelRequest
from bayesprobe.projections import TaskAwareAnswerProjector, build_answer_projection
from bayesprobe.recorded_gateway import RecordedModelGateway
from bayesprobe.probe_executor import (
    ModelBackedProbeToolGateway,
    ProbeExecutionResult,
    ProbeExecutor,
)
from bayesprobe.probe_design import (
    MODEL_REASONING_CAPABILITY,
    ModelProbeDesigner,
    ProbeDesignResult,
)
from bayesprobe.probe_planner import ProbePlanningResult
from bayesprobe.question_runner import (
    AutonomousQuestionRunConfig,
    AutonomousQuestionRunner,
    AutonomousQuestionProgressKind,
    AutonomousQuestionStopReason,
)
from bayesprobe.task_framing import (
    ExplicitTaskFramer,
    ModelTaskFramer,
    RoutingTaskFramer,
    TaskFramingError,
)
from bayesprobe.task_admission import (
    ExplicitTaskAdmitter,
    ModelTaskAdmitter,
    RoutingTaskAdmitter,
    TaskAdmissionError,
)
from bayesprobe.schemas import (
    AnswerChoice,
    AnswerContractOutline,
    AnswerProjection,
    AnswerValueType,
    BeliefState,
    ChangeMyMindCondition,
    CycleSignalShape,
    EvidenceContributionMode,
    EpistemicProgress,
    EvolutionOperation,
    FramingMethod,
    HypothesisEvolution,
    HypothesisRelation,
    ProbeCandidate,
    ProbeDesign,
    ProbeSet,
    ProjectionMode,
    RunRegime,
    RunStatus,
    SignalKind,
    TaskAdmissionDecision,
    TaskAdmissionStatus,
    TaskKind,
)


def recorded_open_mvp_runtime(
    gateway: RecordedModelGateway,
    *,
    max_cycles: int,
) -> tuple[AutonomousQuestionRunner, InitializeRunInput]:
    runner = AutonomousQuestionRunner(
        core=BayesProbeCore(
            model_gateway=gateway,
            hypothesis_expander=HypothesisExpansionService(
                adapter=ModelHypothesisExpansionAdapter(gateway)
            ),
        ),
        initializer=BayesProbeInitializer(task_framer=ModelTaskFramer(gateway)),
        executor=ProbeExecutor(ModelBackedProbeToolGateway(gateway)),
        config=AutonomousQuestionRunConfig(
            max_cycles=max_cycles,
            max_probes_per_cycle=2,
        ),
        task_admitter=ModelTaskAdmitter(gateway),
        probe_designer=ModelProbeDesigner(gateway),
        available_capabilities=(MODEL_REASONING_CAPABILITY,),
        answer_projector=TaskAwareAnswerProjector(gateway),
    )
    return (
        runner,
        InitializeRunInput(
            run_id="recorded_open_mvp",
            problem="What conclusion follows from the stated open question?",
        ),
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


class ReturningTaskAdmitter:
    def __init__(self, decision):
        self.decision = decision

    def assess(self, input):
        return self.decision


class RecordingProbeDesigner:
    def __init__(self):
        self.contexts = []

    def propose(self, context):
        self.contexts.append(context)
        return ProbeDesignResult(candidates=[], capability_decisions=[])


class RecordingAnswerProjector:
    def __init__(self):
        self.inputs = []

    def project(self, input):
        self.inputs.append(input)
        return build_answer_projection(
            input.cycle_id,
            input.previous_belief_state,
            input.cycle_result,
        )


class SequencedRunnerGateway:
    adapter_kind = "runner_stagnation_test"

    def __init__(
        self,
        bands=("weakly_confirming",),
        *,
        independent_roots=False,
    ):
        self.bands = tuple(bands)
        self.independent_roots = independent_roots
        self.execution_count = 0
        self.judgment_count = 0

    @property
    def model_identity(self):
        root_index = self.execution_count + 1 if self.independent_roots else 1
        return f"runner-stagnation-model-{root_index}"

    def complete_structured(self, request):
        if request.task == "execute_probe":
            self.execution_count += 1
            return {
                "raw_content": "MODEL REASONING: The assessment supports H1."
            }
        if request.task == "judge_evidence":
            band = self.bands[min(self.judgment_count, len(self.bands) - 1)]
            self.judgment_count += 1
            return {
                "evidence_type": "supporting",
                "likelihoods": {
                    hypothesis_id: (
                        band if hypothesis_id == "H1" else "neutral"
                    )
                    for hypothesis_id in request.input["target_hypotheses"]
                },
                "unresolved_likelihood": None,
                "frame_fit": "explained_by_named",
                "unexplained_observation": None,
                "interpretation": "The same model assessment was evaluated.",
                "quality_overrides": {},
            }
        raise AssertionError(f"unexpected task: {request.task}")


class RecordingCycleCore(BayesProbeCore):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.results = []

    def integrate_cycle(self, **kwargs):
        result = super().integrate_cycle(**kwargs)
        self.results.append(result)
        return result


class ProgressExceptionCore(BayesProbeCore):
    def __init__(self, reported_change=None):
        super().__init__()
        self.reported_change = reported_change

    def integrate_cycle(self, *, cycle, belief_state, probe_set, signals):
        result = super().integrate_cycle(
            cycle=cycle,
            belief_state=belief_state,
            probe_set=probe_set,
            signals=signals,
        )
        frame_state = belief_state.frame_state
        assert frame_state is not None
        hypothesis_evolutions = []
        if self.reported_change == "hypothesis":
            hypothesis_evolutions = [
                HypothesisEvolution(
                    evolution_id=f"evolution_{cycle.cycle_id}",
                    cycle_id=cycle.cycle_id,
                    operation=EvolutionOperation.REFRAME,
                    from_hypothesis="H1",
                    to_hypothesis="H1",
                    reason="Core reported a hypothesis reframe.",
                )
            ]
        elif self.reported_change == "frame":
            frame_payload = frame_state.model_dump(mode="python")
            frame_payload.update(
                {
                    "frame_version": frame_state.frame_version + 1,
                    "parent_frame_version": frame_state.frame_version,
                    "revision_count": frame_state.revision_count + 1,
                    "revision_reason": "Core reported a frame revision.",
                }
            )
            frame_state = type(frame_state).model_validate(frame_payload)

        state_payload = result.belief_state.model_dump(mode="python")
        state_payload["frame_state"] = frame_state.model_dump(mode="python")
        current = BeliefState.model_validate(state_payload)
        decision = replace(result.frame_adequacy_decision, frame_state=frame_state)
        return replace(
            result,
            belief_state=current,
            frame_adequacy_decision=decision,
            hypothesis_evolutions=hypothesis_evolutions,
            contribution_deltas=[],
            epistemic_progress=EpistemicProgress(),
        )


def evidence_progress_runner(
    *,
    max_cycles,
    bands=("weakly_confirming",),
    independent_roots=False,
    answer_projector=None,
    ledger=None,
    **config_overrides,
):
    gateway = SequencedRunnerGateway(
        bands,
        independent_roots=independent_roots,
    )
    core = RecordingCycleCore(model_gateway=gateway, ledger=ledger)
    runner = AutonomousQuestionRunner(
        core=core,
        executor=ProbeExecutor(
            ModelBackedProbeToolGateway(gateway),
            ledger=ledger,
        ),
        answer_projector=answer_projector,
        config=AutonomousQuestionRunConfig(
            max_cycles=max_cycles,
            max_probes_per_cycle=1,
            **config_overrides,
        ),
    )
    return runner, core


def epistemic_runner_input(run_id):
    return InitializeRunInput(
        run_id=run_id,
        problem="Does this cycle contribute new information?",
        hypothesis_seeds=explicit_test_hypothesis_seeds(),
    )


def applied_penalty_state(belief_state):
    return {
        hypothesis.id: (
            hypothesis.applied_complexity_penalty,
            hypothesis.applied_ad_hoc_penalty,
        )
        for hypothesis in belief_state.hypotheses
    }


def explicit_test_hypothesis_seeds() -> list[HypothesisSeed]:
    return [
        HypothesisSeed(id="H1", statement="The fixture's H1 condition holds.", prior=0.5, scope="Deterministic test fixture.", falsifiers=["The fixture emits a reliable H1 refutation."], predictions=["The fixture emits a reliable H1 support cue."]),
        HypothesisSeed(id="H2", statement="The fixture's H2 condition holds instead.", prior=0.5, scope="Deterministic test fixture.", falsifiers=["The fixture emits a reliable H2 refutation."], predictions=["The fixture emits a reliable H2 support cue."]),
    ]


def valid_open_frame_payload() -> dict[str, object]:
    return {
        "task_kind": "claim_verification",
        "answer_relationship": "synthesis",
        "answer_contract": {
            "objective": "Design a discriminating validation protocol.",
            "answer_value_type": "structured_text",
            "answer_format": "structured validation protocol",
            "required_sections": ["hypotheses", "controls", "decision_rule"],
            "decision_form": "experimental_protocol",
            "permits_synthesis": True,
        },
        "competition": "independent",
        "coverage": "open",
        "hypotheses": [
            {
                "statement": "Scale has an independent effect under matched conditions.",
                "type": "causal_claim",
                "scope": "Matched task and resource conditions.",
                "falsifiers": ["The controlled effect is negligible."],
                "predictions": ["Matched performance rises with size."],
                "answer_value": None,
            },
            {
                "statement": "The apparent effect is materially confounded.",
                "type": "confounding_explanation",
                "scope": "Unmatched comparisons.",
                "falsifiers": ["The effect survives matched controls."],
                "predictions": ["The effect shrinks after matching."],
                "answer_value": None,
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
        if request.task == "assess_task_admission":
            return {
                "status": "admitted",
                "epistemic_basis": ["The claim can be tested with discriminating hypotheses."],
                "proposed_task_kind": "claim_verification",
                "answer_contract_outline": {
                    "objective": "Return a discriminating validation protocol.",
                    "answer_value_type": "structured_text",
                    "decision_form": "experimental_protocol",
                    "permits_synthesis": True,
                    "required_sections": ["answer", "basis", "uncertainty"],
                },
                "clarification_questions": [],
                "reason": "The task is verifiable.",
            }
        if request.task == "frame_open_question":
            return valid_open_frame_payload()
        if request.task == "design_probes":
            return {
                "proposals": [
                    {
                        "purpose": "hypothesis_discrimination",
                        "target_hypotheses": ["H1", "H2"],
                        "inquiry_goal": "Compare the active explanations under matched conditions.",
                        "expected_observation": "The result favors one explanation over the other.",
                        "support_condition": {
                            "H1": "The matched effect remains.",
                            "H2": "The effect disappears after matching.",
                        },
                        "weaken_condition": {
                            "H1": "Matching removes the effect.",
                            "H2": "The effect survives matching.",
                        },
                        "reframe_condition": None,
                        "required_capability": "model_reasoning",
                    }
                ]
            }
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


@pytest.mark.parametrize("status", ["needs_reframing", "out_of_scope"])
def test_non_admitted_result_creates_no_belief_state(tmp_path: Path, status: str):
    response = {
        "status": status,
        "epistemic_basis": ["The current request cannot enter epistemic framing."],
        "proposed_task_kind": None,
        "answer_contract_outline": None,
        "clarification_questions": (
            ["What concrete answer should BayesProbe evaluate?"]
            if status == "needs_reframing"
            else []
        ),
        "reason": "Admission stopped before framing.",
    }
    gateway = ScriptedModelGateway({"assess_task_admission": response})
    ledger = JsonlLedgerStore(tmp_path / f"{status}.jsonl")
    runner = AutonomousQuestionRunner(
        core=BayesProbeCore(ledger=ledger),
        task_admitter=ModelTaskAdmitter(gateway),
    )

    result = runner.run_question(
        InitializeRunInput(
            run_id=f"run_{status}",
            problem="Please handle this underspecified request.",
        )
    )

    assert result.result_type == status
    assert not hasattr(result, "initial_belief_state")
    assert len(ledger.read_all("task_admission")) == 1
    for record_type in ("task_frame", "run", "cycle", "belief_state"):
        assert ledger.read_all(record_type) == []


def test_secret_bearing_non_admission_never_reaches_ledger(tmp_path: Path):
    secret_response = {
        "status": "out_of_scope",
        "epistemic_basis": ["The request is outside available capabilities."],
        "proposed_task_kind": None,
        "answer_contract_outline": None,
        "clarification_questions": [],
        "reason": "password=provider-value-123",
    }
    gateway = ScriptedModelGateway(
        {
            "assess_task_admission": secret_response,
            "repair_task_admission": secret_response,
        }
    )
    ledger_path = tmp_path / "secret-admission-decision.jsonl"
    runner = AutonomousQuestionRunner(
        core=BayesProbeCore(ledger=JsonlLedgerStore(ledger_path)),
        task_admitter=ModelTaskAdmitter(gateway),
    )

    with pytest.raises(TaskAdmissionError, match="invalid after 1 repair attempt"):
        runner.run_question(
            InitializeRunInput(
                run_id="run_secret_admission_decision",
                problem="Handle this request.",
            )
        )

    assert not ledger_path.exists()


def test_runner_rejects_constructed_secret_bearing_admission_before_progress_or_ledger(
    tmp_path: Path,
):
    secret = "sk-constructedadaptersecret"
    decision = TaskAdmissionDecision.model_construct(
        attempt_id="constructed_admission",
        status=TaskAdmissionStatus.ADMITTED,
        epistemic_basis=["The claim is testable."],
        proposed_task_kind=TaskKind.CLAIM_VERIFICATION,
        answer_contract_outline=AnswerContractOutline(
            objective="Assess the claim.",
            answer_value_type=AnswerValueType.STRUCTURED_TEXT,
            decision_form="claim_assessment",
            permits_synthesis=True,
            required_sections=["answer", "basis", "uncertainty"],
        ),
        clarification_questions=[],
        reason=f"Use {secret} to admit this task.",
        model_trace={},
    )
    ledger_path = tmp_path / "constructed-admission.jsonl"
    progress = []
    runner = AutonomousQuestionRunner(
        core=BayesProbeCore(ledger=JsonlLedgerStore(ledger_path)),
        task_admitter=ReturningTaskAdmitter(decision),
        progress_observer=progress.append,
    )

    with pytest.raises(TaskAdmissionError, match="invalid task admission decision") as captured:
        runner.run_question(
            InitializeRunInput(run_id="run_constructed_admission", problem="Test this claim.")
        )

    assert secret not in str(captured.value)
    assert progress == []
    assert not ledger_path.exists()


def test_runner_rejects_wrong_admitter_return_type_before_progress_or_ledger(tmp_path: Path):
    ledger_path = tmp_path / "wrong-admission-type.jsonl"
    progress = []
    runner = AutonomousQuestionRunner(
        core=BayesProbeCore(ledger=JsonlLedgerStore(ledger_path)),
        task_admitter=ReturningTaskAdmitter({"status": "admitted"}),
        progress_observer=progress.append,
    )

    with pytest.raises(TaskAdmissionError, match="invalid task admission decision"):
        runner.run_question(
            InitializeRunInput(run_id="run_wrong_admission_type", problem="Test this claim.")
        )

    assert progress == []
    assert not ledger_path.exists()


def test_runner_uses_explicit_seeded_initializer_discriminator_before_fresh_design():
    designer = RecordingProbeDesigner()
    runner = AutonomousQuestionRunner(
        core=BayesProbeCore(),
        probe_designer=designer,
        config=AutonomousQuestionRunConfig(max_cycles=1, max_probes_per_cycle=1),
    )

    result = runner.run_question(
        InitializeRunInput(
            run_id="run_explicit_seeded_compatibility",
            problem="Which explicit explanation is better supported?",
            hypothesis_seeds=explicit_test_hypothesis_seeds(),
        )
    )

    assert len(result.cycle_results) == 1
    assert result.cycle_results[0].probe_set.probes[0].method == (
        "frame_discrimination_support"
    )
    assert designer.contexts == []


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
        task_admitter=ModelTaskAdmitter(gateway),
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
    assert observed[1][1] == ["assess_task_admission"]
    assert observed[2][1] == ["assess_task_admission", "frame_open_question"]
    assert gateway.requests[0].task == "assess_task_admission"
    assert gateway.requests[1].task == "frame_open_question"
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


def test_runner_designs_open_probe_after_belief_initialization():
    gateway = RecordingOpenQuestionGateway()
    progress = []
    runner = AutonomousQuestionRunner(
        core=BayesProbeCore(model_gateway=gateway),
        initializer=BayesProbeInitializer(task_framer=ModelTaskFramer(gateway)),
        executor=ProbeExecutor(ModelBackedProbeToolGateway(gateway)),
        config=AutonomousQuestionRunConfig(max_cycles=1, max_probes_per_cycle=1),
        progress_observer=progress.append,
        task_admitter=ModelTaskAdmitter(gateway),
        probe_designer=ModelProbeDesigner(gateway),
        available_capabilities=(MODEL_REASONING_CAPABILITY,),
    )

    result = runner.run_question(
        InitializeRunInput(
            run_id="run_open_probe_design",
            problem="How should a model-scale claim be tested?",
        )
    )

    tasks = [request.task for request in gateway.requests]
    assert tasks.index("frame_open_question") < tasks.index("design_probes")
    assert tasks.index("design_probes") < tasks.index("execute_probe")
    kinds = [event.kind for event in progress]
    assert AutonomousQuestionProgressKind.PROBE_DESIGN_STARTED in kinds
    assert AutonomousQuestionProgressKind.PROBE_DESIGN_COMPLETED in kinds
    completed = next(
        event
        for event in progress
        if event.kind == AutonomousQuestionProgressKind.PROBE_DESIGN_COMPLETED
    )
    assert len(completed.probe_candidates) == 1
    assert completed.capability_decisions[0].available is True
    assert gateway.requests[0].input["available_capabilities"] == [
        MODEL_REASONING_CAPABILITY.model_dump(mode="json")
    ]
    assert len(result.cycle_results[0].probe_set.probes) == 1


def _candidate(candidate_id: str, inquiry_goal: str) -> ProbeCandidate:
    return ProbeCandidate(
        candidate_id=candidate_id,
        source="manual",
        candidate_probe=ProbeDesign(
            id=f"P_{candidate_id}",
            cycle_id="cycle_1",
            target_hypotheses=["H1", "H2"],
            inquiry_goal=inquiry_goal,
            method="model_reasoning",
        ),
    )


def _answer_projection(candidate: ProbeCandidate) -> AnswerProjection:
    return AnswerProjection(
        answer="H1",
        current_best_hypothesis="H1",
        posterior_summary="H1=0.6",
        main_uncertainty="H2 remains plausible.",
        weakest_assumption="The test generalizes.",
        main_evidence_events=[],
        change_my_mind_condition=ChangeMyMindCondition(
            human_readable_condition="A contrary result changes the answer.",
            structured_probe_candidates=[candidate],
        ),
    )


def test_next_pool_keeps_core_candidates_before_fresh_and_remaining():
    runner = AutonomousQuestionRunner(core=BayesProbeCore())
    core = _candidate("core", "Check the core-generated concern.")
    duplicate_core = _candidate("duplicate_core", "  check THE core-generated concern. ")
    designed = _candidate("designed", "Compare the current hypotheses.")
    projection = _candidate("projection", "Challenge the current answer.")
    remaining = _candidate("remaining", "Keep a deferred candidate.")

    pool = runner._next_candidate_pool(
        previous_pool=[remaining],
        selected_candidates=[],
        core_candidates=[core],
        designed_candidates=[duplicate_core, designed],
        answer_projection=_answer_projection(projection),
    )

    assert [item.candidate_id for item in pool] == [
        core.candidate_id,
        designed.candidate_id,
        projection.candidate_id,
        remaining.candidate_id,
    ]


def test_malformed_single_choice_routes_through_model_admission_and_framing():
    gateway = RecordingOpenQuestionGateway()
    runner = AutonomousQuestionRunner(
        core=BayesProbeCore(model_gateway=gateway),
        initializer=BayesProbeInitializer(
            task_framer=RoutingTaskFramer(
                explicit_framer=ExplicitTaskFramer(),
                open_framer=ModelTaskFramer(gateway),
            )
        ),
        executor=ProbeExecutor(ModelBackedProbeToolGateway(gateway)),
        config=AutonomousQuestionRunConfig(max_cycles=1, max_probes_per_cycle=1),
        task_admitter=RoutingTaskAdmitter(
            explicit_admitter=ExplicitTaskAdmitter(),
            open_admitter=ModelTaskAdmitter(gateway),
        ),
    )

    result = runner.run_question(
        InitializeRunInput(
            run_id="run_malformed_single_choice",
            problem="Which result follows?",
            answer_choices=[AnswerChoice(label="A", text="Only result")],
        )
    )

    assert result.task_frame.framing_method == FramingMethod.MODEL
    assert [request.task for request in gateway.requests[:2]] == [
        "assess_task_admission",
        "frame_open_question",
    ]


def test_recorded_open_question_frames_before_running_cycle():
    gateway = RecordedModelGateway.from_json(
        Path("tests/fixtures/open_questions/model_scale_validation_v0.2.json")
    )
    runner = AutonomousQuestionRunner(
        core=BayesProbeCore(model_gateway=gateway),
        initializer=BayesProbeInitializer(task_framer=ModelTaskFramer(gateway)),
        executor=ProbeExecutor(ModelBackedProbeToolGateway(gateway)),
        config=AutonomousQuestionRunConfig(max_cycles=1, max_probes_per_cycle=1),
        task_admitter=ModelTaskAdmitter(gateway),
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
        "assess_task_admission",
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
    assert final_by_id["H1"].posterior == 0.5
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
    assert [delta.mode for delta in cycle_result.contribution_deltas] == [
        EvidenceContributionMode.NEW_ROOT
    ]
    assert cycle_result.epistemic_progress.new_root_count == 1
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


def test_question_runner_projects_with_prospective_stop_reason_and_reuses_candidates():
    projector = RecordingAnswerProjector()
    runner = AutonomousQuestionRunner(
        core=BayesProbeCore(),
        answer_projector=projector,
        config=AutonomousQuestionRunConfig(max_cycles=2, max_probes_per_cycle=1),
    )

    result = runner.run_question(
        InitializeRunInput(
            run_id="run_projector_wiring",
            problem="Can a projection inform the next autonomous cycle?",
            hypothesis_seeds=explicit_test_hypothesis_seeds(),
        )
    )

    assert [input.stop_reason for input in projector.inputs] == [None, "max_cycles"]
    first_candidate = (
        result.cycle_results[0]
        .answer_projection.change_my_mind_condition.structured_probe_candidates[0]
    )
    assert result.cycle_results[1].probe_set.probes[0].id.startswith(
        first_candidate.candidate_probe.id
    )


def test_autonomous_runner_stops_when_same_root_adds_no_information():
    projector = RecordingAnswerProjector()
    runner, core = evidence_progress_runner(
        max_cycles=10,
        answer_projector=projector,
    )

    result = runner.run_question(epistemic_runner_input("run_same_root_stop"))

    first, second = result.cycle_results
    assert result.stop_reason == AutonomousQuestionStopReason.EPISTEMIC_STAGNATION
    assert len(result.cycle_results) == 2
    assert [delta.mode for delta in first.contribution_deltas] == [
        EvidenceContributionMode.NEW_ROOT
    ]
    assert [delta.mode for delta in second.contribution_deltas] == [
        EvidenceContributionMode.NO_CHANGE
    ]
    assert second.epistemic_progress.no_change_count == 1
    assert second.epistemic_progress.max_absolute_contribution_delta == 0.0
    assert second.contribution_deltas is core.results[1].contribution_deltas
    assert second.epistemic_progress is core.results[1].epistemic_progress
    assert second.belief_state.hypotheses == first.belief_state.hypotheses
    assert second.belief_state.frame_state == first.belief_state.frame_state
    assert applied_penalty_state(second.belief_state) == applied_penalty_state(
        first.belief_state
    )
    assert second.belief_updates == []
    assert projector.inputs[-1].stop_reason == "epistemic_stagnation"
    assert result.run.metadata["stop_reason"] == "epistemic_stagnation"


def test_autonomous_runner_does_not_stagnate_on_new_independent_root():
    runner, _ = evidence_progress_runner(
        max_cycles=2,
        independent_roots=True,
    )

    result = runner.run_question(epistemic_runner_input("run_independent_roots"))

    assert result.stop_reason == AutonomousQuestionStopReason.MAX_CYCLES
    assert [
        cycle.epistemic_progress.new_root_count for cycle in result.cycle_results
    ] == [1, 1]
    assert all(
        cycle.contribution_deltas[0].mode == EvidenceContributionMode.NEW_ROOT
        for cycle in result.cycle_results
    )


@pytest.mark.parametrize(
    ("second_band", "expected_mode", "progress_field"),
    [
        (
            "moderately_confirming",
            EvidenceContributionMode.REVISE_ROOT,
            "revised_root_count",
        ),
        (
            "neutral",
            EvidenceContributionMode.RETRACT_ROOT,
            "retracted_root_count",
        ),
    ],
)
def test_autonomous_runner_does_not_stagnate_on_root_progress(
    second_band,
    expected_mode,
    progress_field,
):
    runner, _ = evidence_progress_runner(
        max_cycles=2,
        bands=("weakly_confirming", second_band),
    )

    result = runner.run_question(
        epistemic_runner_input(f"run_{expected_mode.value}")
    )

    terminal_cycle = result.cycle_results[-1]
    assert result.stop_reason == AutonomousQuestionStopReason.MAX_CYCLES
    assert terminal_cycle.contribution_deltas[0].mode == expected_mode
    assert getattr(terminal_cycle.epistemic_progress, progress_field) == 1
    assert terminal_cycle.epistemic_progress.max_absolute_contribution_delta > 0.0


def test_autonomous_runner_does_not_stagnate_on_hypothesis_evolution():
    runner = AutonomousQuestionRunner(
        core=ProgressExceptionCore("hypothesis"),
        config=AutonomousQuestionRunConfig(max_cycles=1, max_probes_per_cycle=1),
    )

    result = runner.run_question(epistemic_runner_input("run_hypothesis_progress"))

    cycle = result.cycle_results[0]
    assert result.stop_reason == AutonomousQuestionStopReason.MAX_CYCLES
    assert cycle.epistemic_progress.max_absolute_contribution_delta == 0.0
    assert cycle.hypothesis_evolutions


def test_autonomous_runner_does_not_stagnate_on_frame_revision():
    runner = AutonomousQuestionRunner(
        core=ProgressExceptionCore("frame"),
        config=AutonomousQuestionRunConfig(max_cycles=1, max_probes_per_cycle=1),
    )

    result = runner.run_question(epistemic_runner_input("run_frame_progress"))

    cycle = result.cycle_results[0]
    assert result.stop_reason == AutonomousQuestionStopReason.MAX_CYCLES
    assert cycle.epistemic_progress.max_absolute_contribution_delta == 0.0
    assert result.initial_belief_state.frame_state != cycle.belief_state.frame_state


def test_epistemic_stagnation_has_priority_after_an_integrated_cycle():
    projector = RecordingAnswerProjector()
    runner = AutonomousQuestionRunner(
        core=ProgressExceptionCore(),
        answer_projector=projector,
        config=AutonomousQuestionRunConfig(
            max_cycles=1,
            max_probes_per_cycle=1,
            confidence_threshold=0.0,
            posterior_delta_threshold=0.0,
        ),
    )

    result = runner.run_question(epistemic_runner_input("run_stagnation_priority"))

    cycle = result.cycle_results[0]
    assert cycle.epistemic_progress.no_change_count == 0
    assert result.stop_reason == AutonomousQuestionStopReason.EPISTEMIC_STAGNATION
    assert projector.inputs[0].stop_reason == "epistemic_stagnation"
    assert result.run.metadata["stop_reason"] == "epistemic_stagnation"


def test_question_runner_stops_on_confidence_threshold():
    runner, _ = evidence_progress_runner(
        max_cycles=3,
        bands=("strongly_confirming",),
        confidence_threshold=0.55,
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
    assert result.final_belief_state.hypotheses_by_id()["H1"].posterior >= 0.55


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
    runner, _ = evidence_progress_runner(
        max_cycles=1,
        bands=("strongly_confirming",),
        ledger=ledger,
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
    assert "evidence_contribution_delta" in record_types
    assert "epistemic_progress" in record_types
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
        AutonomousQuestionProgressKind.FRAME_ADEQUACY_ASSESSED,
        AutonomousQuestionProgressKind.ANSWER_PROJECTION_STARTED,
        AutonomousQuestionProgressKind.ANSWER_PROJECTION_COMPLETED,
        AutonomousQuestionProgressKind.CYCLE_INTEGRATED,
        AutonomousQuestionProgressKind.RUN_COMPLETED,
    ]
    cycle_event = next(
        event
        for event in events
        if event.kind == AutonomousQuestionProgressKind.CYCLE_INTEGRATED
    )
    assert cycle_event.cycle_result == result.cycle_results[0]
    assert cycle_event.cycle_result.cycle.boundary_status.value == "integrated"
    assert sum(
        hypothesis.posterior
        for hypothesis in cycle_event.cycle_result.belief_state.hypotheses
    ) == pytest.approx(1.0)
    assert events[-1].result == result


def test_question_runner_rejects_secret_context_before_progress_or_ledger(tmp_path: Path):
    ledger_path = tmp_path / "secret-context-runner.jsonl"
    ledger = JsonlLedgerStore(ledger_path)
    events = []
    runner = AutonomousQuestionRunner(
        core=BayesProbeCore(ledger=ledger),
        progress_observer=events.append,
    )

    with pytest.raises(
        TaskFramingError,
        match="compatibility context must not contain secret material",
    ):
        runner.run_question(
            InitializeRunInput(
                run_id="run_secret_context_before_progress",
                problem="Does validation happen before progress?",
                context="credential = provider-value-123",
                hypothesis_seeds=explicit_test_hypothesis_seeds(),
            )
        )

    assert events == []
    assert not ledger_path.exists()


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
        AutonomousQuestionProgressKind.FRAME_ADEQUACY_ASSESSED,
        AutonomousQuestionProgressKind.ANSWER_PROJECTION_STARTED,
        AutonomousQuestionProgressKind.ANSWER_PROJECTION_COMPLETED,
        AutonomousQuestionProgressKind.CYCLE_INTEGRATED,
    ]
    non_terminal_cycle = [
        *per_cycle,
        AutonomousQuestionProgressKind.PROBE_DESIGN_STARTED,
        AutonomousQuestionProgressKind.PROBE_DESIGN_COMPLETED,
    ]
    assert observed_kinds == [
        AutonomousQuestionProgressKind.RUN_STARTED,
        AutonomousQuestionProgressKind.TASK_FRAMING_STARTED,
        AutonomousQuestionProgressKind.TASK_FRAMING_COMPLETED,
        AutonomousQuestionProgressKind.INITIALIZATION_COMPLETED,
        *non_terminal_cycle,
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


@pytest.mark.parametrize(
    ("fixture_name", "max_cycles", "mode", "answer_value"),
    [
        ("model_scale_open_mvp_v0.1.json", 1, ProjectionMode.SYNTHESIS, None),
        ("exact_answer_expansion_mvp_v0.1.json", 2, ProjectionMode.SELECTION, 4),
    ],
)
def test_recorded_open_question_mvp_vertical_slice(
    fixture_name, max_cycles, mode, answer_value
):
    gateway = RecordedModelGateway.from_json(
        Path("tests/fixtures/open_questions") / fixture_name
    )
    runner, input = recorded_open_mvp_runtime(gateway, max_cycles=max_cycles)

    result = runner.run_question(input)

    assert result.final_answer_projection.mode == mode
    assert result.final_answer_projection.answer_value == answer_value
    assert len(result.cycle_results) == max_cycles


def test_recorded_open_question_mvp_synthesis_projects_required_sections():
    gateway = RecordedModelGateway.from_json(
        Path("tests/fixtures/open_questions/model_scale_open_mvp_v0.1.json")
    )
    runner, input = recorded_open_mvp_runtime(gateway, max_cycles=1)

    result = runner.run_question(input)

    projection = result.final_answer_projection
    assert projection.mode == ProjectionMode.SYNTHESIS
    assert set(projection.contract_sections) == {
        "hypotheses",
        "controls",
        "metrics",
        "decision_rule",
        "limitations",
    }
    assert not projection.answer.startswith("Current best hypothesis")
    assert [request.task for request in gateway.requests] == [
        "assess_task_admission",
        "frame_open_question",
        "design_probes",
        "execute_probe",
        "judge_evidence",
        "project_answer",
    ]


def test_recorded_open_question_mvp_expands_before_selecting_new_integer_answer():
    gateway = RecordedModelGateway.from_json(
        Path("tests/fixtures/open_questions/exact_answer_expansion_mvp_v0.1.json")
    )
    runner, input = recorded_open_mvp_runtime(gateway, max_cycles=2)

    result = runner.run_question(input)

    first_cycle = result.cycle_results[0]
    spawned_ids = {
        evolution.to_hypothesis
        for evolution in first_cycle.hypothesis_evolutions
        if evolution.operation.value == "spawn"
    }
    discovery_evidence_id = first_cycle.evidence_events[0].id
    assert result.final_answer_projection.mode == ProjectionMode.SELECTION
    assert result.final_answer_projection.answer == "4"
    assert result.final_answer_projection.answer_value == 4
    assert result.final_belief_state.frame_state.frame_version == 2
    assert spawned_ids
    assert {
        hypothesis.answer_value
        for hypothesis in result.final_belief_state.hypotheses
        if hypothesis.id in spawned_ids
    } == {4, 5}
    assert discovery_evidence_id in result.final_belief_state.evidence_memory.discovery_evidence_ids
    assert not {
        update.hypothesis_id for update in first_cycle.belief_updates
    }.intersection(spawned_ids)
    assert [request.task for request in gateway.requests] == [
        "assess_task_admission",
        "frame_open_question",
        "design_probes",
        "execute_probe",
        "judge_evidence",
        "expand_hypotheses",
        "design_probes",
        "execute_probe",
        "execute_probe",
        "judge_evidence",
        "judge_evidence",
    ]
