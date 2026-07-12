import pytest

from bayesprobe.model_gateway import ScriptedModelGateway
from bayesprobe.probe_design import (
    MODEL_REASONING_CAPABILITY,
    FrameProbeDesigner,
    ModelProbeDesigner,
    ProbeDesignError,
    ProbeDesignContext,
)
from bayesprobe.schemas import (
    AnswerContract,
    AnswerRelationship,
    AnswerValueType,
    BeliefState,
    CapabilityKind,
    EvidenceMemorySnapshot,
    FrameAdequacyStatus,
    FrameState,
    FramedHypothesis,
    FramingMethod,
    Hypothesis,
    HypothesisCompetition,
    HypothesisCoverage,
    HypothesisFrame,
    ProbePurpose,
    TaskFrame,
    TaskKind,
)


@pytest.fixture
def open_state() -> BeliefState:
    framed_hypotheses = [
        FramedHypothesis(
            id="H1",
            statement="Model size improves the task outcome.",
            type="claim",
            scope="The stated evaluation suite.",
            initial_prior=0.25,
            falsifiers=["The effect disappears under a matched budget."],
            predictions=["Larger models retain an advantage when budgets match."],
        ),
        FramedHypothesis(
            id="H2",
            statement="Inference compute explains the task outcome.",
            type="claim",
            scope="The stated evaluation suite.",
            initial_prior=0.25,
            falsifiers=["Matching compute leaves no outcome difference."],
            predictions=["Compute matching removes the apparent size effect."],
        ),
    ]
    task_frame = TaskFrame(
        schema_version="v0.2",
        task_frame_id="frame_open",
        admission_decision_id="admission_open",
        task_kind=TaskKind.EXPLANATION,
        answer_relationship=AnswerRelationship.SYNTHESIS,
        normalized_question="What explains the apparent model size effect?",
        answer_contract=AnswerContract(
            objective="Explain the strongest supported causal account.",
            answer_value_type=AnswerValueType.STRUCTURED_TEXT,
            answer_format="A concise explanation.",
            required_sections=["answer", "uncertainty"],
            decision_form="synthesis",
            permits_synthesis=True,
        ),
        hypothesis_frame=HypothesisFrame(
            frame_id="hypothesis_frame_open",
            competition=HypothesisCompetition.EXCLUSIVE,
            coverage=HypothesisCoverage.OPEN,
            hypotheses=framed_hypotheses,
            rival_sets={"H1": ["H2"], "H2": ["H1"]},
            coverage_statement="The named explanations are provisional.",
            unresolved_alternative_mass=0.5,
        ),
        framing_method=FramingMethod.EXPLICIT,
    )
    hypotheses = [
        Hypothesis(
            id=item.id,
            statement=item.statement,
            type=item.type,
            scope=item.scope,
            prior=item.initial_prior,
            posterior=item.initial_prior,
            rivals=[other for other in ("H1", "H2") if other != item.id],
            falsifiers=item.falsifiers,
            predictions=item.predictions,
        )
        for item in framed_hypotheses
    ]
    return BeliefState(
        schema_version="v0.2",
        belief_state_id="belief_open",
        run_id="run_open",
        cycle_id="cycle_0",
        cycle_index=0,
        hypotheses=hypotheses,
        task_frame=task_frame,
        frame_state=FrameState(
            frame_id=task_frame.hypothesis_frame.frame_id,
            competition=HypothesisCompetition.EXCLUSIVE,
            coverage=HypothesisCoverage.OPEN,
            active_hypothesis_ids=["H1", "H2"],
            unresolved_alternative_mass=0.5,
            adequacy_status=FrameAdequacyStatus.PROVISIONAL,
        ),
        evidence_memory=EvidenceMemorySnapshot(memory_version=2),
    )


def proposal(**updates):
    payload = {
        "purpose": "hypothesis_discrimination",
        "target_hypotheses": ["H1", "H2"],
        "inquiry_goal": "Compare model sizes under matched inference budgets.",
        "expected_observation": "The size coefficient survives or collapses after matching.",
        "support_condition": {"H1": "The matched coefficient remains positive."},
        "weaken_condition": {"H1": "The matched coefficient is negligible."},
        "reframe_condition": {
            "frame": "Neither hypothesis explains task interactions."
        },
        "required_capability": "model_reasoning",
    }
    payload.update(updates)
    return payload


def search_proposal():
    return proposal(required_capability="search")


def open_context(open_state):
    return ProbeDesignContext(
        run_id="run_open",
        cycle_id="cycle_1",
        task_frame=open_state.task_frame,
        belief_state=open_state,
        available_capabilities=(MODEL_REASONING_CAPABILITY,),
    )


def test_model_probe_designer_materializes_server_owned_candidate(open_state):
    gateway = ScriptedModelGateway({"design_probes": {"proposals": [proposal()]}})

    result = ModelProbeDesigner(gateway).propose(open_context(open_state))

    assert len(result.candidates) == 1
    probe = result.candidates[0].candidate_probe
    assert probe.id.startswith("P_cycle_1_")
    assert probe.priority == 0.85
    assert probe.required_capability == CapabilityKind.MODEL_REASONING
    response = gateway.responses["design_probes"]["proposals"][0]
    assert "id" not in response
    assert "priority" not in response


def test_model_probe_designer_rejects_unavailable_search(open_state):
    gateway = ScriptedModelGateway({"design_probes": {"proposals": [search_proposal()]}})

    result = ModelProbeDesigner(gateway).propose(open_context(open_state))

    assert result.candidates == []
    assert result.capability_decisions[0].kind == CapabilityKind.SEARCH
    assert result.capability_decisions[0].available is False


def test_model_probe_designer_repairs_one_invalid_response(open_state):
    invalid = proposal(target_hypotheses=["unknown"])
    gateway = ScriptedModelGateway(
        {
            "design_probes": {"proposals": [invalid]},
            "repair_probe_design": {"proposals": [proposal()]},
        }
    )

    result = ModelProbeDesigner(gateway).propose(open_context(open_state))

    assert len(result.candidates) == 1
    assert [request.task for request in gateway.requests] == [
        "design_probes",
        "repair_probe_design",
    ]
    assert gateway.requests[1].metadata["repair_attempt_index"] == 1


def test_model_probe_designer_rejects_unknown_hypotheses_after_repair(open_state):
    invalid = proposal(target_hypotheses=["unknown"])
    gateway = ScriptedModelGateway(
        {
            "design_probes": {"proposals": [invalid]},
            "repair_probe_design": {"proposals": [invalid]},
        }
    )

    with pytest.raises(ProbeDesignError, match="unknown hypothesis"):
        ModelProbeDesigner(gateway).propose(open_context(open_state))


def test_model_probe_designer_removes_semantic_duplicates(open_state):
    duplicate = proposal(
        inquiry_goal="  compare MODEL sizes under matched inference budgets.  "
    )
    gateway = ScriptedModelGateway(
        {"design_probes": {"proposals": [proposal(), duplicate]}}
    )

    result = ModelProbeDesigner(gateway).propose(open_context(open_state))

    assert len(result.candidates) == 1


def test_model_probe_designer_rejects_secret_material(open_state):
    secret = proposal(inquiry_goal="api_key: provider-secret-value-123")
    gateway = ScriptedModelGateway(
        {
            "design_probes": {"proposals": [secret]},
            "repair_probe_design": {"proposals": [secret]},
        }
    )

    with pytest.raises(ProbeDesignError, match="secret"):
        ModelProbeDesigner(gateway).propose(open_context(open_state))


def test_initial_open_design_requires_discriminator_or_frame_coverage(open_state):
    singleton = proposal(
        purpose="source_verification",
        target_hypotheses=["H1"],
    )
    gateway = ScriptedModelGateway(
        {
            "design_probes": {"proposals": [singleton]},
            "repair_probe_design": {"proposals": [singleton]},
        }
    )

    with pytest.raises(ProbeDesignError, match="initial open design"):
        ModelProbeDesigner(gateway).propose(open_context(open_state))


def test_frame_probe_designer_reports_deterministic_model_reasoning(open_state):
    result = FrameProbeDesigner().propose(open_context(open_state))

    assert result.candidates[0].candidate_probe.purpose == (
        ProbePurpose.HYPOTHESIS_DISCRIMINATION
    )
    assert result.capability_decisions[0].descriptor.executor_adapter_id == (
        "deterministic_frame_probe_designer:v1"
    )
