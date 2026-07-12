import math

import pytest

from bayesprobe.frame_policy import FrameAdequacyDecision
from bayesprobe.hypothesis_expansion import (
    HypothesisExpansionError,
    HypothesisExpansionProposal,
    HypothesisExpansionRequest,
    HypothesisExpansionService,
    ModelHypothesisExpansionAdapter,
)
from bayesprobe.kernel_config import ExpansionPolicy, OpenCoveragePolicy
from bayesprobe.model_gateway import ScriptedModelGateway
from bayesprobe.schemas import (
    AnswerContract,
    AnswerRelationship,
    AnswerValueType,
    EvidenceEvent,
    EvidenceType,
    EvolutionOperation,
    FrameAdequacyStatus,
    FrameFit,
    FrameState,
    FramedHypothesis,
    FramingMethod,
    Hypothesis,
    HypothesisCompetition,
    HypothesisCoverage,
    HypothesisFrame,
    HypothesisStatus,
    LikelihoodBand,
    TaskFrame,
    TaskKind,
)


class StaticExpansionAdapter:
    def __init__(self, proposals):
        self.proposals = proposals

    def propose(self, request):
        return self.proposals


def proposal(**updates):
    payload = {
        "statement": "A missing mechanism explains the anomalous result.",
        "type": "claim",
        "scope": "The stated evaluation conditions.",
        "falsifiers": ["The mechanism is absent in a controlled replication."],
        "predictions": ["The effect appears only when the mechanism is present."],
        "answer_value": None,
        "why_current_frame_missed": "The original frame omitted this mechanism.",
        "required_next_probe": "Test the mechanism under matched conditions.",
    }
    payload.update(updates)
    return HypothesisExpansionProposal(**payload)


def raw_proposal(**updates):
    payload = proposal().model_dump(mode="json")
    payload.update(updates)
    return payload


def make_task_frame(
    *,
    competition=HypothesisCompetition.EXCLUSIVE,
    answer_value_type=AnswerValueType.INTEGER,
) -> TaskFrame:
    coverage = HypothesisCoverage.OPEN
    hypotheses = [
        FramedHypothesis(
            id="H1",
            statement="The first named account holds.",
            type="claim",
            scope="The stated evaluation conditions.",
            initial_prior=0.20 if competition == HypothesisCompetition.EXCLUSIVE else 0.5,
            falsifiers=["A controlled result contradicts the first account."],
            predictions=["The first account predicts the observed effect."],
            answer_value=1 if competition == HypothesisCompetition.EXCLUSIVE else None,
        ),
        FramedHypothesis(
            id="H2",
            statement="The second named account holds.",
            type="claim",
            scope="The stated evaluation conditions.",
            initial_prior=0.20 if competition == HypothesisCompetition.EXCLUSIVE else 0.5,
            falsifiers=["A controlled result contradicts the second account."],
            predictions=["The second account predicts the observed effect."],
            answer_value=2 if competition == HypothesisCompetition.EXCLUSIVE else None,
        ),
    ]
    return TaskFrame(
        schema_version="v0.2",
        task_frame_id="task_frame",
        admission_decision_id="admission_1",
        task_kind=(
            TaskKind.EXACT_ANSWER
            if competition == HypothesisCompetition.EXCLUSIVE
            else TaskKind.EXPLANATION
        ),
        answer_relationship=(
            AnswerRelationship.SELECTION
            if competition == HypothesisCompetition.EXCLUSIVE
            else AnswerRelationship.SYNTHESIS
        ),
        normalized_question="Which account best explains the observed effect?",
        answer_contract=AnswerContract(
            objective="Select the supported account.",
            answer_value_type=answer_value_type,
            answer_format="A concise answer.",
            required_sections=["answer", "uncertainty"],
            decision_form=(
                "selection" if competition == HypothesisCompetition.EXCLUSIVE else "synthesis"
            ),
            permits_synthesis=competition == HypothesisCompetition.INDEPENDENT,
        ),
        hypothesis_frame=HypothesisFrame(
            frame_id="hypothesis_frame",
            competition=competition,
            coverage=coverage,
            hypotheses=hypotheses,
            rival_sets={"H1": ["H2"], "H2": ["H1"]}
            if competition == HypothesisCompetition.EXCLUSIVE
            else {"H1": [], "H2": []},
            coverage_statement="The named accounts are provisional.",
            unresolved_alternative_mass=0.60
            if competition == HypothesisCompetition.EXCLUSIVE
            else None,
        ),
        framing_method=FramingMethod.EXPLICIT,
    )


def make_state(
    *,
    competition=HypothesisCompetition.EXCLUSIVE,
    answer_value_type=AnswerValueType.INTEGER,
) -> tuple[TaskFrame, FrameState, tuple[Hypothesis, ...]]:
    task_frame = make_task_frame(
        competition=competition,
        answer_value_type=answer_value_type,
    )
    hypotheses = tuple(
        Hypothesis(
            id=item.id,
            statement=item.statement,
            type=item.type,
            scope=item.scope,
            prior=item.initial_prior,
            posterior=item.initial_prior,
            rivals=["H2"] if item.id == "H1" and competition == HypothesisCompetition.EXCLUSIVE else ["H1"] if competition == HypothesisCompetition.EXCLUSIVE else [],
            falsifiers=item.falsifiers,
            predictions=item.predictions,
            answer_value=item.answer_value,
        )
        for item in task_frame.hypothesis_frame.hypotheses
    )
    return (
        task_frame,
        FrameState(
            frame_id=task_frame.hypothesis_frame.frame_id,
            competition=competition,
            coverage=HypothesisCoverage.OPEN,
            active_hypothesis_ids=["H1", "H2"],
            unresolved_alternative_mass=0.60
            if competition == HypothesisCompetition.EXCLUSIVE
            else None,
            adequacy_status=FrameAdequacyStatus.CHALLENGED,
        ),
        hypotheses,
    )


def event() -> EvidenceEvent:
    return EvidenceEvent(
        schema_version="v0.2",
        id="E1",
        derived_from_signal="signal_1",
        epistemic_origin="external_observation",
        derivation_root_id="root_1",
        target_hypotheses=["H1", "H2"],
        evidence_type=EvidenceType.BOUNDARY_CONDITION,
        content="A controlled observation does not fit either named account.",
        likelihoods={"H1": LikelihoodBand.WEAKLY_DISCONFIRMING},
        unresolved_likelihood=LikelihoodBand.MODERATELY_CONFIRMING,
        frame_fit=FrameFit.SUPPORTS_UNRESOLVED,
        unexplained_observation="The observation contradicts both named accounts.",
        correlation_status="novel",
        effective_update_weight=0.8,
    )


def expansion_request(
    task_frame: TaskFrame,
    frame_state: FrameState,
    hypotheses: tuple[Hypothesis, ...],
) -> HypothesisExpansionRequest:
    return HypothesisExpansionRequest(
        run_id="run_1",
        cycle_id="cycle_1",
        task_frame=task_frame,
        frame_state=frame_state,
        hypotheses=hypotheses,
        triggering_events=(event(),),
        expansion_reason="Accepted evidence supports an unresolved alternative.",
    )


def expansion_decision(frame_state: FrameState, *, should_expand=True):
    return FrameAdequacyDecision(
        frame_state=frame_state,
        should_expand=should_expand,
        trigger_event_ids=["E1"],
        reason="Accepted evidence supports an unresolved alternative.",
    )


def test_exact_expansion_transfers_half_unresolved_mass():
    task_frame, frame_state, hypotheses = make_state()
    adapter = StaticExpansionAdapter(
        [
            proposal(answer_value=4, statement="The supported tendon count is four."),
            proposal(answer_value=5, statement="The supported tendon count is five."),
        ]
    )

    result = HypothesisExpansionService(adapter=adapter).expand(
        request=expansion_request(task_frame, frame_state, hypotheses),
        decision=expansion_decision(frame_state),
    )

    new_items = [item for item in result.hypotheses if item.created_by == "spawned"]
    assert [item.answer_value for item in new_items] == [4, 5]
    assert [item.posterior for item in new_items] == [0.15, 0.15]
    assert result.frame_state.unresolved_alternative_mass == 0.30
    assert sum(item.posterior for item in result.hypotheses if item.status == HypothesisStatus.ACTIVE) + 0.30 == pytest.approx(1.0)
    assert result.frame_state.frame_version == 2
    assert result.frame_state.parent_frame_version == 1
    assert result.frame_state.adequacy_status == FrameAdequacyStatus.PROVISIONAL
    assert result.probe_candidates[0].candidate_probe.target_hypotheses[-2:] == [
        new_items[0].id,
        new_items[1].id,
    ]
    assert result.discovery_evidence_ids == ["E1"]
    assert [item.operation for item in result.evolutions] == [
        EvolutionOperation.SPAWN,
        EvolutionOperation.SPAWN,
    ]
    assert result.frame_mass_updates[0].prior == 0.60
    assert result.frame_mass_updates[0].posterior == 0.30


def test_independent_expansion_adds_claim_without_cross_normalization():
    task_frame, frame_state, hypotheses = make_state(
        competition=HypothesisCompetition.INDEPENDENT,
        answer_value_type=AnswerValueType.STRUCTURED_TEXT,
    )
    service = HypothesisExpansionService(
        adapter=StaticExpansionAdapter(
            [proposal(answer_value=None, statement="Task difficulty moderates the scale effect.")]
        )
    )

    result = service.expand(
        request=expansion_request(task_frame, frame_state, hypotheses),
        decision=expansion_decision(frame_state),
    )

    added = result.hypotheses[-1]
    assert added.prior == 0.5
    assert added.posterior == 0.5
    assert [item.posterior for item in result.hypotheses[:-1]] == [0.5, 0.5]
    assert result.frame_state.unresolved_alternative_mass is None
    assert result.frame_mass_updates == []


def test_expansion_requires_adequacy_decision():
    task_frame, frame_state, hypotheses = make_state()

    with pytest.raises(HypothesisExpansionError, match="requires an expansion decision"):
        HypothesisExpansionService(adapter=StaticExpansionAdapter([proposal(answer_value=4)])).expand(
            request=expansion_request(task_frame, frame_state, hypotheses),
            decision=expansion_decision(frame_state, should_expand=False),
        )


@pytest.mark.parametrize("count", [0, 4])
def test_expansion_requires_one_to_three_proposals(count):
    task_frame, frame_state, hypotheses = make_state()
    proposals = [proposal(answer_value=index + 4) for index in range(count)]

    with pytest.raises(HypothesisExpansionError, match="between one and three"):
        HypothesisExpansionService(adapter=StaticExpansionAdapter(proposals)).expand(
            request=expansion_request(task_frame, frame_state, hypotheses),
            decision=expansion_decision(frame_state),
        )


@pytest.mark.parametrize(
    "statement",
    ["The first named account holds.", "  a retired alternative.  "],
)
def test_expansion_rejects_duplicate_active_or_historical_statement(statement):
    task_frame, frame_state, hypotheses = make_state()
    historical = Hypothesis(
        id="H0",
        statement="A retired alternative.",
        type="claim",
        scope="The stated evaluation conditions.",
        prior=0.1,
        posterior=0.1,
        status=HypothesisStatus.RETIRED,
        falsifiers=["A result refutes it."],
        predictions=["It predicts a result."],
    )
    with pytest.raises(HypothesisExpansionError, match="duplicates an existing hypothesis"):
        HypothesisExpansionService(adapter=StaticExpansionAdapter([proposal(answer_value=4, statement=statement)])).expand(
            request=expansion_request(task_frame, frame_state, (*hypotheses, historical)),
            decision=expansion_decision(frame_state),
        )


@pytest.mark.parametrize(
    ("answer_value_type", "answer_value"),
    [
        (AnswerValueType.INTEGER, 4.5),
        (AnswerValueType.NUMBER, math.inf),
        (AnswerValueType.SHORT_TEXT, 4),
    ],
)
def test_expansion_validates_typed_answer_values(answer_value_type, answer_value):
    task_frame, frame_state, hypotheses = make_state(answer_value_type=answer_value_type)

    with pytest.raises(HypothesisExpansionError, match="answer_value"):
        HypothesisExpansionService(adapter=StaticExpansionAdapter([proposal(answer_value=answer_value)])).expand(
            request=expansion_request(task_frame, frame_state, hypotheses),
            decision=expansion_decision(frame_state),
        )


def test_expansion_enforces_revision_and_active_hypothesis_limits():
    task_frame, frame_state, hypotheses = make_state()
    revised = frame_state.model_copy(update={"revision_count": 3})
    service = HypothesisExpansionService(adapter=StaticExpansionAdapter([proposal(answer_value=4)]))

    with pytest.raises(HypothesisExpansionError, match="revision limit"):
        service.expand(
            request=expansion_request(task_frame, revised, hypotheses),
            decision=expansion_decision(revised),
        )

    active = tuple(
        Hypothesis(
            id=f"H{index}",
            statement=f"Active account {index}.",
            type="claim",
            scope="The stated evaluation conditions.",
            prior=0.01,
            posterior=0.01,
            falsifiers=["A result refutes it."],
            predictions=["It predicts a result."],
            answer_value=index,
        )
        for index in range(1, 9)
    )
    crowded = frame_state.model_copy(update={"active_hypothesis_ids": [item.id for item in active]})
    with pytest.raises(HypothesisExpansionError, match="active hypothesis limit"):
        service.expand(
            request=expansion_request(task_frame, crowded, active),
            decision=expansion_decision(crowded),
        )


def test_expansion_preserves_minimum_unresolved_reserve():
    task_frame, frame_state, hypotheses = make_state()
    frame_state = frame_state.model_copy(update={"unresolved_alternative_mass": 0.06})
    service = HypothesisExpansionService(
        adapter=StaticExpansionAdapter([proposal(answer_value=4)]),
        open_policy=OpenCoveragePolicy(minimum_unresolved_reserve=0.05),
    )

    result = service.expand(
        request=expansion_request(task_frame, frame_state, hypotheses),
        decision=expansion_decision(frame_state),
    )

    assert result.hypotheses[-1].id == "H_exp_f2_1"
    assert result.hypotheses[-1].posterior == pytest.approx(0.01)
    assert result.frame_state.unresolved_alternative_mass == pytest.approx(0.05)


def test_expansion_assigns_server_owned_ids_and_creates_no_belief_updates():
    task_frame, frame_state, hypotheses = make_state()
    result = HypothesisExpansionService(adapter=StaticExpansionAdapter([proposal(answer_value=4)])).expand(
        request=expansion_request(task_frame, frame_state, hypotheses),
        decision=expansion_decision(frame_state),
    )

    added = result.hypotheses[-1]
    assert added.id == "H_exp_f2_1"
    assert result.evolutions[0].to_hypothesis == added.id
    assert not hasattr(result, "belief_updates")


def test_model_adapter_repairs_one_invalid_response_without_forwarding_payload_or_error_text():
    task_frame, frame_state, hypotheses = make_state()
    secret = "sk-abcdefghijklmnopqrstuv"
    invalid = {"candidates": [raw_proposal(api_key=secret, answer_value=4)]}
    valid = {"candidates": [raw_proposal(answer_value=4)]}
    gateway = ScriptedModelGateway(
        {
            "expand_hypotheses": invalid,
            "repair_hypothesis_expansion": valid,
        }
    )

    proposals = ModelHypothesisExpansionAdapter(gateway).propose(
        expansion_request(task_frame, frame_state, hypotheses)
    )

    assert [item.answer_value for item in proposals] == [4]
    assert [request.task for request in gateway.requests] == [
        "expand_hypotheses",
        "repair_hypothesis_expansion",
    ]
    repair = gateway.requests[1]
    assert repair.input["validation_error"] == "hypothesis expansion response invalid"
    assert "invalid_payload" not in repair.input
    assert secret not in repr(repair.input)


def test_model_adapter_raises_fixed_error_after_one_invalid_repair():
    task_frame, frame_state, hypotheses = make_state()
    invalid = {"candidates": [raw_proposal(answer_value=4, id="model-owned")]}
    gateway = ScriptedModelGateway(
        {
            "expand_hypotheses": invalid,
            "repair_hypothesis_expansion": invalid,
        }
    )

    with pytest.raises(
        HypothesisExpansionError,
        match="hypothesis expansion invalid after 1 repair attempt",
    ):
        ModelHypothesisExpansionAdapter(gateway).propose(
            expansion_request(task_frame, frame_state, hypotheses)
        )


def test_model_adapter_repairs_non_object_initial_response():
    task_frame, frame_state, hypotheses = make_state()
    gateway = ScriptedModelGateway(
        {
            "expand_hypotheses": ["not an object"],
            "repair_hypothesis_expansion": {
                "candidates": [raw_proposal(answer_value=4)]
            },
        }
    )

    proposals = ModelHypothesisExpansionAdapter(gateway).propose(
        expansion_request(task_frame, frame_state, hypotheses)
    )

    assert [item.answer_value for item in proposals] == [4]
    assert [request.task for request in gateway.requests] == [
        "expand_hypotheses",
        "repair_hypothesis_expansion",
    ]
    assert gateway.requests[1].input["validation_error"] == "hypothesis expansion response invalid"


def test_model_adapter_raises_fixed_error_after_non_object_repair_response():
    task_frame, frame_state, hypotheses = make_state()
    gateway = ScriptedModelGateway(
        {
            "expand_hypotheses": {"candidates": []},
            "repair_hypothesis_expansion": ["not an object"],
        }
    )

    with pytest.raises(HypothesisExpansionError) as captured:
        ModelHypothesisExpansionAdapter(gateway).propose(
            expansion_request(task_frame, frame_state, hypotheses)
        )

    assert str(captured.value) == "hypothesis expansion invalid after 1 repair attempt"
    assert [request.task for request in gateway.requests] == [
        "expand_hypotheses",
        "repair_hypothesis_expansion",
    ]
