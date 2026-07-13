from __future__ import annotations

import pytest

from bayesprobe.core import CycleResult
from bayesprobe.frame_policy import FrameAdequacyDecision
from bayesprobe.model_gateway import ScriptedModelGateway
from bayesprobe.projections import (
    AnswerProjectionError,
    AnswerProjectionInput,
    TaskAwareAnswerProjector,
)
from bayesprobe.schemas import (
    AnswerContract,
    AnswerRelationship,
    AnswerValueType,
    BeliefState,
    CycleRecord,
    CycleSignalShape,
    EvidenceEvent,
    EvidenceMemorySnapshot,
    EvidenceType,
    FrameAdequacyStatus,
    FrameState,
    FramedHypothesis,
    FramingMethod,
    Hypothesis,
    HypothesisCompetition,
    HypothesisCoverage,
    HypothesisFrame,
    HypothesisStatus,
    ProjectionMode,
    TaskFrame,
    TaskKind,
)


def _frame(*, relationship: AnswerRelationship) -> TaskFrame:
    is_selection = relationship == AnswerRelationship.SELECTION
    contract = AnswerContract(
        objective=(
            "Return the supported integer."
            if is_selection
            else "Design a matched-budget evaluation."
        ),
        answer_value_type=(
            AnswerValueType.INTEGER if is_selection else AnswerValueType.STRUCTURED_TEXT
        ),
        answer_format="integer" if is_selection else "evaluation protocol",
        required_sections=(
            ["answer", "basis", "uncertainty"]
            if is_selection
            else ["hypotheses", "controls", "decision_rule"]
        ),
        decision_form="single_value" if is_selection else "experimental_protocol",
        permits_synthesis=not is_selection,
    )
    framed = [
        FramedHypothesis(
            id="H_exp_f2_1" if is_selection else "H1",
            statement="The supported candidate holds.",
            type="answer_candidate" if is_selection else "causal_claim",
            scope="Test fixture.",
            initial_prior=0.25 if is_selection else 0.5,
            falsifiers=["A controlled result contradicts the candidate."],
            predictions=["The predicted result is observed."],
            answer_value=4 if is_selection else None,
        ),
        FramedHypothesis(
            id="H2",
            statement="The alternative candidate holds.",
            type="answer_candidate" if is_selection else "confounding_explanation",
            scope="Test fixture.",
            initial_prior=0.25 if is_selection else 0.5,
            falsifiers=["A controlled result contradicts the alternative."],
            predictions=["The alternative prediction is observed."],
            answer_value=9 if is_selection else None,
        ),
    ]
    competition = (
        HypothesisCompetition.EXCLUSIVE
        if is_selection
        else HypothesisCompetition.INDEPENDENT
    )
    coverage = HypothesisCoverage.OPEN if is_selection else HypothesisCoverage.OPEN
    return TaskFrame(
        schema_version="v0.2",
        task_frame_id="frame_1",
        admission_decision_id="admission_1",
        task_kind=TaskKind.EXACT_ANSWER if is_selection else TaskKind.DESIGN,
        answer_relationship=relationship,
        normalized_question="What should the answer be?",
        answer_contract=contract,
        hypothesis_frame=HypothesisFrame(
            frame_id="hypothesis_frame_1",
            competition=competition,
            coverage=coverage,
            hypotheses=framed,
            rival_sets={
                item.id: [other.id for other in framed if other.id != item.id]
                if competition == HypothesisCompetition.EXCLUSIVE
                else []
                for item in framed
            },
            coverage_statement="The fixture covers the named hypotheses.",
            unresolved_alternative_mass=0.5 if is_selection else None,
            coverage_limitation="The frame remains open.",
        ),
        framing_method=FramingMethod.MODEL,
    )


def _cycle_result(*, relationship: AnswerRelationship) -> CycleResult:
    frame = _frame(relationship=relationship)
    is_selection = relationship == AnswerRelationship.SELECTION
    hypotheses = [
        Hypothesis(
            id=item.id,
            statement=item.statement,
            scope=item.scope,
            prior=item.initial_prior,
            posterior=0.60 if index == 0 else 0.20 if is_selection else 0.40,
            type=item.type,
            status=HypothesisStatus.ACTIVE,
            rivals=frame.hypothesis_frame.rival_sets[item.id],
            falsifiers=item.falsifiers,
            predictions=item.predictions,
            answer_value=item.answer_value,
        )
        for index, item in enumerate(frame.hypothesis_frame.hypotheses)
    ]
    frame_state = FrameState(
        frame_id=frame.hypothesis_frame.frame_id,
        competition=frame.hypothesis_frame.competition,
        coverage=frame.hypothesis_frame.coverage,
        active_hypothesis_ids=[item.id for item in hypotheses],
        unresolved_alternative_mass=0.20 if is_selection else None,
        adequacy_status=FrameAdequacyStatus.ADEQUATE,
    )
    state = BeliefState(
        schema_version="v0.2",
        belief_state_id="belief_1",
        run_id="run_1",
        cycle_id="cycle_1",
        cycle_index=1,
        hypotheses=hypotheses,
        task_frame=frame,
        frame_state=frame_state,
        evidence_memory=EvidenceMemorySnapshot(),
        uncertainty_summary="A remaining uncertainty needs attention.",
    )
    event = EvidenceEvent(
        id="E_cycle_1",
        derived_from_signal="S_cycle_1",
        target_hypotheses=[hypotheses[0].id],
        evidence_type=EvidenceType.SUPPORTING,
        content="A matched evaluation supports the first hypothesis.",
        interpretation="The result favors the first hypothesis under matched conditions.",
    )
    cycle = CycleRecord(
        cycle_id="cycle_1",
        run_id="run_1",
        cycle_index=1,
        signal_shape=CycleSignalShape.ACTIVE_ONLY,
    )
    return CycleResult(
        cycle=cycle,
        belief_state=state,
        evidence_events=[event],
        belief_updates=[],
        frame_mass_updates=[],
        frame_adequacy_decision=FrameAdequacyDecision(
            frame_state=frame_state,
            should_expand=False,
            trigger_event_ids=[],
            reason="The fixture frame is adequate.",
        ),
        hypothesis_evolutions=[],
    )


def _input(
    result: CycleResult,
    *,
    belief_state: BeliefState | None = None,
    stop_reason: str | None = "cycle limit reached",
) -> AnswerProjectionInput:
    return AnswerProjectionInput(
        cycle_id="cycle_1",
        previous_belief_state=result.belief_state,
        cycle_result=CycleResult(
            cycle=result.cycle,
            belief_state=belief_state or result.belief_state,
            evidence_events=result.evidence_events,
            belief_updates=result.belief_updates,
            frame_mass_updates=result.frame_mass_updates,
            frame_adequacy_decision=result.frame_adequacy_decision,
            hypothesis_evolutions=result.hypothesis_evolutions,
        ),
        stop_reason=stop_reason,
    )


def test_exact_projection_returns_typed_value_from_expanded_hypothesis():
    result = _cycle_result(relationship=AnswerRelationship.SELECTION)

    projection = TaskAwareAnswerProjector().project(_input(result))

    assert projection.mode == ProjectionMode.SELECTION
    assert projection.answer_value == 4
    assert projection.answer == "4"
    assert projection.current_best_hypothesis == "H_exp_f2_1"


def test_exact_projection_abstains_while_unresolved_outranks_named_during_loop():
    result = _cycle_result(relationship=AnswerRelationship.SELECTION)
    state = result.belief_state.model_copy(
        update={
            "hypotheses": [
                item.model_copy(update={"posterior": 0.30 if index == 0 else 0.30})
                for index, item in enumerate(result.belief_state.hypotheses)
            ],
            "frame_state": result.belief_state.frame_state.model_copy(
                update={"unresolved_alternative_mass": 0.40}
            ),
        }
    )

    projection = TaskAwareAnswerProjector().project(
        _input(result, belief_state=state, stop_reason=None)
    )

    assert projection.mode == ProjectionMode.ABSTENTION
    assert projection.answer_value is None
    assert "unresolved" in projection.main_uncertainty.lower()


def test_terminal_exact_projection_selects_best_named_with_unresolved_warning():
    result = _cycle_result(relationship=AnswerRelationship.SELECTION)
    state = result.belief_state.model_copy(
        update={
            "hypotheses": [
                item.model_copy(
                    update={"posterior": 0.35 if index == 0 else 0.25}
                )
                for index, item in enumerate(result.belief_state.hypotheses)
            ],
            "frame_state": result.belief_state.frame_state.model_copy(
                update={"unresolved_alternative_mass": 0.40}
            ),
        }
    )

    projection = TaskAwareAnswerProjector().project(
        _input(result, belief_state=state, stop_reason="max_cycles")
    )

    assert projection.mode == ProjectionMode.SELECTION
    assert projection.answer_value == 4
    assert projection.current_best_hypothesis == "H_exp_f2_1"
    assert "unresolved" in projection.main_uncertainty.lower()
    assert "terminal" in projection.answer_utility_notes.lower()


def test_synthesis_projection_satisfies_every_required_section():
    result = _cycle_result(relationship=AnswerRelationship.SYNTHESIS)
    gateway = ScriptedModelGateway(
        {
            "project_answer": {
                "answer": "Use a preregistered matched-budget factorial evaluation.",
                "contract_sections": {
                    "hypotheses": "Test scale, budget confounding, and task interaction claims.",
                    "controls": "Hold scaffolding, task set, sampling, and inference budget fixed.",
                    "decision_rule": "Accept a scale effect only when the preregistered effect exceeds the practical threshold.",
                },
                "main_uncertainty": "Deployment distributions may differ.",
                "weakest_assumption": "The frozen task set represents deployment.",
                "cited_evidence_ids": ["E_cycle_1"],
            }
        }
    )

    projection = TaskAwareAnswerProjector(gateway).project(_input(result))

    assert projection.mode == ProjectionMode.SYNTHESIS
    assert set(projection.contract_sections) == {
        "hypotheses",
        "controls",
        "decision_rule",
    }
    assert not projection.answer.startswith("Current best hypothesis")
    assert [request.task for request in gateway.requests] == ["project_answer"]


def test_synthesis_projection_abstains_while_expansion_is_pending():
    result = _cycle_result(relationship=AnswerRelationship.SYNTHESIS)
    state = result.belief_state.model_copy(
        update={
            "frame_state": result.belief_state.frame_state.model_copy(
                update={"adequacy_status": FrameAdequacyStatus.EXPANDING}
            )
        }
    )

    projection = TaskAwareAnswerProjector().project(_input(result, belief_state=state))

    assert projection.mode == ProjectionMode.ABSTENTION
    assert "expansion" in projection.main_uncertainty.lower()


@pytest.mark.parametrize(
    "invalid_response",
    [
        {
            "answer": "Use matched controls.",
            "contract_sections": {"hypotheses": "Test the hypotheses."},
            "main_uncertainty": "Generalization remains uncertain.",
            "weakest_assumption": "The benchmark is representative.",
            "cited_evidence_ids": ["E_unknown"],
        },
        {
            "answer": "Use matched controls.",
            "contract_sections": {"hypotheses": "Test the hypotheses."},
            "main_uncertainty": "Generalization remains uncertain.",
            "weakest_assumption": "The benchmark is representative.",
            "cited_evidence_ids": ["E_cycle_1"],
        },
        {
            "answer": "Use matched controls.",
            "contract_sections": {
                "hypotheses": "Test the hypotheses.",
                "controls": "Match all budgets.",
                "decision_rule": "Use a preregistered threshold.",
            },
            "main_uncertainty": "Generalization remains uncertain.",
            "weakest_assumption": "The benchmark is representative.",
            "cited_evidence_ids": ["E_cycle_1"],
            "posterior": 0.99,
        },
    ],
)
def test_synthesis_repairs_unknown_evidence_missing_sections_and_model_beliefs(
    invalid_response: dict[str, object],
):
    result = _cycle_result(relationship=AnswerRelationship.SYNTHESIS)
    valid_response = {
        "answer": "Use a preregistered matched-budget factorial evaluation.",
        "contract_sections": {
            "hypotheses": "Test the hypotheses.",
            "controls": "Match all budgets.",
            "decision_rule": "Use a preregistered threshold.",
        },
        "main_uncertainty": "Generalization remains uncertain.",
        "weakest_assumption": "The benchmark is representative.",
        "cited_evidence_ids": ["E_cycle_1"],
    }
    gateway = ScriptedModelGateway(
        {
            "project_answer": invalid_response,
            "repair_answer_projection": valid_response,
        }
    )

    projection = TaskAwareAnswerProjector(gateway).project(_input(result))

    assert projection.mode == ProjectionMode.SYNTHESIS
    assert [request.task for request in gateway.requests] == [
        "project_answer",
        "repair_answer_projection",
    ]


def test_synthesis_fails_after_one_invalid_repair_attempt():
    result = _cycle_result(relationship=AnswerRelationship.SYNTHESIS)
    invalid_response = {
        "answer": "Use matched controls.",
        "contract_sections": {},
        "main_uncertainty": "Generalization remains uncertain.",
        "weakest_assumption": "The benchmark is representative.",
        "cited_evidence_ids": [],
    }
    gateway = ScriptedModelGateway(
        {
            "project_answer": invalid_response,
            "repair_answer_projection": invalid_response,
        }
    )

    with pytest.raises(
        AnswerProjectionError,
        match="answer projection invalid after 1 repair attempt",
    ):
        TaskAwareAnswerProjector(gateway).project(_input(result))


def test_change_my_mind_candidate_never_uses_generic_source_tracing():
    result = _cycle_result(relationship=AnswerRelationship.SELECTION)

    projection = TaskAwareAnswerProjector().project(_input(result))

    assert all(
        candidate.candidate_probe.method != "source_tracing"
        for candidate in projection.change_my_mind_condition.structured_probe_candidates
    )
