import pytest

from bayesprobe.schemas import (
    BeliefState,
    AnswerContract,
    ChangeMyMindCondition,
    CycleRecord,
    CycleSignalShape,
    EvidenceEvent,
    EvidenceType,
    ExternalSignal,
    FramedHypothesis,
    FramingMethod,
    Hypothesis,
    HypothesisFrame,
    HypothesisRelation,
    HypothesisStatus,
    LikelihoodBand,
    ProbeCandidate,
    ProbeDesign,
    ProbeSet,
    RunRecord,
    RunRegime,
    SignalKind,
    TaskFrame,
    TaskKind,
)


def _open_task_frame() -> TaskFrame:
    return TaskFrame(
        task_frame_id="run_frame_task_frame",
        task_kind=TaskKind.CLAIM_VERIFICATION,
        normalized_question="How should the model-scale claim be tested?",
        task_context="Evaluate on a frozen real-task distribution.",
        answer_contract=AnswerContract(
            objective="Design a discriminating validation protocol.",
            required_sections=["hypotheses", "controls", "decision_rule"],
            decision_form="experimental_protocol",
            permits_synthesis=True,
        ),
        hypothesis_frame=HypothesisFrame(
            frame_id="run_frame_hypothesis_frame",
            relation=HypothesisRelation.INDEPENDENT,
            hypotheses=[
                FramedHypothesis(
                    id="H1",
                    statement="Scale has an independent positive effect.",
                    type="causal_claim",
                    scope="Matched agent and compute conditions.",
                    initial_prior=0.5,
                    falsifiers=["The controlled effect is negligible."],
                    predictions=["Performance rises under matched controls."],
                ),
                FramedHypothesis(
                    id="H2",
                    statement="The apparent effect is caused by confounding.",
                    type="confounding_explanation",
                    scope="Unmatched published comparisons.",
                    initial_prior=0.5,
                    falsifiers=["The effect survives all matched controls."],
                    predictions=["The effect shrinks after matching resources."],
                ),
            ],
            rival_sets={"H1": [], "H2": []},
            coverage_statement="Tests the causal claim and its main confounder.",
            coverage_limitation="Other task-specific interactions may exist.",
        ),
        framing_method=FramingMethod.MODEL,
        framing_trace={"task": "frame_open_question", "schema_version": "v0.1"},
    )


def test_task_frame_accepts_independent_open_hypotheses():
    frame = _open_task_frame()
    assert frame.hypothesis_frame.relation == HypothesisRelation.INDEPENDENT
    assert [item.id for item in frame.hypothesis_frame.hypotheses] == ["H1", "H2"]


@pytest.mark.parametrize(
    "mutator, message",
    [
        (lambda frame: frame.model_copy(update={"answer_contract": frame.answer_contract.model_copy(update={"required_sections": []})}), "required_sections"),
        (lambda frame: frame.model_copy(update={"hypothesis_frame": frame.hypothesis_frame.model_copy(update={"hypotheses": [frame.hypothesis_frame.hypotheses[0], frame.hypothesis_frame.hypotheses[1].model_copy(update={"id": "H1"})]})}), "ids must be unique"),
        (lambda frame: frame.model_copy(update={"hypothesis_frame": frame.hypothesis_frame.model_copy(update={"rival_sets": {"H1": ["missing"], "H2": []}})}), "unknown rival"),
        (lambda frame: frame.model_copy(update={"framing_trace": {"api_key": "forbidden"}}), "secret"),
    ],
)
def test_task_frame_rejects_invalid_contract(mutator, message):
    with pytest.raises(ValueError, match=message):
        TaskFrame.model_validate(mutator(_open_task_frame()).model_dump())


def test_minimal_run_cycle_and_belief_state_round_trip():
    run = RunRecord(run_id="run_1", regime=RunRegime.AUTONOMOUS, problem="Decide X")
    cycle = CycleRecord(
        cycle_id="cycle_1",
        run_id=run.run_id,
        cycle_index=1,
        signal_shape=CycleSignalShape.ACTIVE_ONLY,
    )
    hypothesis = Hypothesis(
        id="H1",
        statement="X is true",
        scope="sample scope",
        prior=0.5,
        posterior=0.5,
        rivals=["H2"],
        falsifiers=["A strong counterexample would weaken H1."],
        predictions=["Evidence A is likely if H1 is true."],
    )
    belief_state = BeliefState(
        belief_state_id="bs_1",
        run_id=run.run_id,
        cycle_id=cycle.cycle_id,
        hypotheses=[hypothesis],
    )

    loaded = BeliefState.model_validate_json(belief_state.model_dump_json())

    assert loaded.hypotheses[0].id == "H1"
    assert loaded.hypotheses[0].status == HypothesisStatus.ACTIVE


def test_probe_set_can_be_empty_for_passive_only_cycle():
    probe_set = ProbeSet(
        probe_set_id="ps_1",
        cycle_id="cycle_1",
        probes=[],
        selection_reason="Passive-only synchronized cycle.",
        may_be_empty=True,
    )

    assert probe_set.probes == []
    assert probe_set.may_be_empty is True


def test_external_signal_kinds_and_change_my_mind_candidates():
    candidate = ProbeCandidate(
        candidate_id="pc_1",
        source="change_my_mind",
        candidate_probe=ProbeDesign(
            id="P1",
            cycle_id="cycle_2",
            target_hypotheses=["H1"],
            inquiry_goal="Check if source A is independent.",
            method="source_tracing",
            support_condition={"H1": "Source A is independent."},
            weaken_condition={"H1": "Source A shares origin with source B."},
        ),
    )
    condition = ChangeMyMindCondition(
        human_readable_condition="I would lower H1 if source A is not independent.",
        structured_probe_candidates=[candidate],
    )
    signal = ExternalSignal(
        id="S1",
        cycle_id="cycle_1",
        signal_kind=SignalKind.PASSIVE,
        source_type="external_agent_projection",
        source="agent_a",
        raw_content="Agent A believes H1 because source A supports it.",
    )

    assert condition.structured_probe_candidates[0].candidate_probe.method == "source_tracing"
    assert signal.signal_kind == SignalKind.PASSIVE


def test_evidence_event_model_trace_defaults_to_empty_dict():
    event = EvidenceEvent(
        id="E1",
        derived_from_signal="S1",
        target_hypotheses=["H1"],
        evidence_type=EvidenceType.SUPPORTING,
        content="SUPPORTS: evidence.",
        likelihoods={"H1": LikelihoodBand.MODERATELY_CONFIRMING},
    )

    assert event.model_trace == {}


def test_evidence_event_model_trace_round_trips_through_json():
    event = EvidenceEvent(
        id="E1",
        derived_from_signal="S1",
        target_hypotheses=["H1"],
        evidence_type=EvidenceType.SUPPORTING,
        content="SUPPORTS: evidence.",
        likelihoods={"H1": LikelihoodBand.MODERATELY_CONFIRMING},
        model_trace={
            "task": "judge_evidence",
            "adapter_kind": "scripted",
            "prompt_id": "evidence_judgment",
            "prompt_version": "v0.1",
            "schema_name": "EvidenceJudgment",
            "schema_version": "v0.1",
            "metadata": {},
        },
    )

    loaded = EvidenceEvent.model_validate_json(event.model_dump_json())

    assert loaded.model_trace == {
        "task": "judge_evidence",
        "adapter_kind": "scripted",
        "prompt_id": "evidence_judgment",
        "prompt_version": "v0.1",
        "schema_name": "EvidenceJudgment",
        "schema_version": "v0.1",
        "metadata": {},
    }
