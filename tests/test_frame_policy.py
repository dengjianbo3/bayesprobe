from dataclasses import FrozenInstanceError

import pytest

from bayesprobe.belief import CoverageAwareBeliefSolver
from bayesprobe.frame_policy import FrameAdequacyPolicy
from bayesprobe.kernel_config import (
    ExpansionPolicy,
    FrameAdequacyPolicyConfig,
    OpenCoveragePolicy,
    ProjectionPolicy,
)
from bayesprobe.schemas import (
    AnswerContract,
    AnswerRelationship,
    AnswerValueType,
    BeliefState,
    EvidenceEvent,
    EvidenceMemorySnapshot,
    EvidenceType,
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


def _exact_state(
    *,
    named: dict[str, float],
    unresolved: float,
    statuses: dict[str, HypothesisStatus] | None = None,
    adequacy: FrameAdequacyStatus = FrameAdequacyStatus.PROVISIONAL,
) -> BeliefState:
    statuses = statuses or {}
    hypotheses = [
        Hypothesis(
            id=hypothesis_id,
            statement=f"The answer is {hypothesis_id}.",
            scope="Exact-answer fixture.",
            prior=posterior,
            posterior=posterior,
            status=statuses.get(hypothesis_id, HypothesisStatus.ACTIVE),
            rivals=[other for other in named if other != hypothesis_id],
            falsifiers=[f"A constraint excludes {hypothesis_id}."],
            predictions=[f"The constraints select {hypothesis_id}."],
            answer_value=hypothesis_id,
        )
        for hypothesis_id, posterior in named.items()
    ]
    frame_id = "run_exact_frame"
    task_frame = TaskFrame(
        schema_version="v0.2",
        task_frame_id="run_exact_task_frame",
        admission_decision_id="run_exact_admission",
        task_kind=TaskKind.EXACT_ANSWER,
        answer_relationship=AnswerRelationship.SELECTION,
        normalized_question="Which exact value satisfies the constraints?",
        answer_contract=AnswerContract(
            objective="Return the supported exact value.",
            answer_value_type=AnswerValueType.SHORT_TEXT,
            answer_format="short text",
            required_sections=["answer", "basis", "uncertainty"],
            decision_form="single_value",
            permits_synthesis=False,
        ),
        hypothesis_frame=HypothesisFrame(
            frame_id=frame_id,
            competition=HypothesisCompetition.EXCLUSIVE,
            coverage=HypothesisCoverage.OPEN,
            hypotheses=[
                FramedHypothesis(
                    id=hypothesis.id,
                    statement=hypothesis.statement,
                    type="answer_candidate",
                    scope=hypothesis.scope,
                    initial_prior=hypothesis.prior,
                    falsifiers=list(hypothesis.falsifiers),
                    predictions=list(hypothesis.predictions),
                    answer_value=hypothesis.answer_value,
                )
                for hypothesis in hypotheses
            ],
            rival_sets={
                hypothesis_id: [other for other in named if other != hypothesis_id]
                for hypothesis_id in named
            },
            coverage_statement="The named values are provisional candidates.",
            unresolved_alternative_mass=unresolved,
            coverage_limitation="Other exact values remain possible.",
        ),
        framing_method=FramingMethod.RECORDED,
    )
    return BeliefState(
        schema_version="v0.2",
        belief_state_id="run_exact_bs_0",
        run_id="run_exact",
        cycle_id="cycle_0",
        hypotheses=hypotheses,
        task_frame=task_frame,
        frame_state=FrameState(
            frame_id=frame_id,
            competition=HypothesisCompetition.EXCLUSIVE,
            coverage=HypothesisCoverage.OPEN,
            active_hypothesis_ids=list(named),
            unresolved_alternative_mass=unresolved,
            adequacy_status=adequacy,
        ),
        evidence_memory=EvidenceMemorySnapshot(),
    )


def _event(
    *,
    likelihoods: dict[str, LikelihoodBand],
    unresolved_likelihood: LikelihoodBand = LikelihoodBand.NEUTRAL,
    frame_fit: FrameFit = FrameFit.UNDERDETERMINED,
    effective_update_weight: float | None = 1.0,
    event_id: str = "E_exact",
    verifiability: float = 0.5,
    origin: str | None = None,
    derivation_root_id: str | None = None,
    discard_reason: str | None = None,
) -> EvidenceEvent:
    return EvidenceEvent(
        id=event_id,
        derived_from_signal=f"S_{event_id}",
        target_hypotheses=list(likelihoods),
        evidence_type=EvidenceType.COUNTEREVIDENCE,
        content="The named exact-answer candidates do not fit the observation.",
        reliability=1.0,
        independence=1.0,
        relevance=1.0,
        novelty=1.0,
        specificity=1.0,
        verifiability=verifiability,
        likelihoods=likelihoods,
        unresolved_likelihood=unresolved_likelihood,
        frame_fit=frame_fit,
        effective_update_weight=effective_update_weight,
        discard_reason=discard_reason,
        epistemic_origin=origin,
        derivation_root_id=derivation_root_id,
    )


def test_exclusive_open_solver_updates_named_and_unresolved_as_one_distribution():
    state = _exact_state(named={"H1": 0.25, "H2": 0.25}, unresolved=0.50)
    event = _event(
        likelihoods={
            "H1": LikelihoodBand.MODERATELY_DISCONFIRMING,
            "H2": LikelihoodBand.MODERATELY_DISCONFIRMING,
        },
        unresolved_likelihood=LikelihoodBand.MODERATELY_CONFIRMING,
        frame_fit=FrameFit.SUPPORTS_UNRESOLVED,
    )

    result = CoverageAwareBeliefSolver().solve(
        state,
        [event],
        run_id="run_exact",
        cycle_id="cycle_1",
    )

    total = (
        sum(hypothesis.posterior for hypothesis in result.hypotheses)
        + result.frame_state.unresolved_alternative_mass
    )
    assert total == pytest.approx(1.0)
    assert result.frame_state.unresolved_alternative_mass > 0.50
    assert all(hypothesis.posterior < 0.25 for hypothesis in result.hypotheses)
    assert {update.hypothesis_id for update in result.belief_updates} == {"H1", "H2"}
    assert len(result.frame_mass_updates) == 1
    assert result.frame_mass_updates[0].evidence_id == event.id
    assert all(
        update.hypothesis_id != "unresolved"
        for update in result.belief_updates
    )


def test_all_named_candidates_can_lose_without_forced_winner():
    state = _exact_state(named={"H1": 0.25, "H2": 0.25}, unresolved=0.50)
    event = _event(
        likelihoods={
            "H1": LikelihoodBand.STRONGLY_DISCONFIRMING,
            "H2": LikelihoodBand.STRONGLY_DISCONFIRMING,
        },
        unresolved_likelihood=LikelihoodBand.STRONGLY_CONFIRMING,
        frame_fit=FrameFit.SUPPORTS_UNRESOLVED,
        origin="model_reasoning",
        derivation_root_id="model-root",
    )

    solve_result = CoverageAwareBeliefSolver().solve(
        state,
        [event],
        run_id="run_exact",
        cycle_id="cycle_1",
    )
    decision = FrameAdequacyPolicy().assess(
        previous=solve_result.frame_state,
        events=[event],
        hypotheses=solve_result.hypotheses,
    )

    assert decision.frame_state.unresolved_alternative_mass > max(
        hypothesis.posterior for hypothesis in solve_result.hypotheses
    )
    assert decision.frame_state.adequacy_status == FrameAdequacyStatus.CHALLENGED
    assert decision.should_expand is True


def test_explicit_zero_effective_weight_remains_zero():
    state = _exact_state(named={"H1": 0.25, "H2": 0.25}, unresolved=0.50)
    event = _event(
        likelihoods={"H1": LikelihoodBand.STRONGLY_CONFIRMING},
        unresolved_likelihood=LikelihoodBand.STRONGLY_DISCONFIRMING,
        effective_update_weight=0.0,
    )

    result = CoverageAwareBeliefSolver().solve(
        state,
        [event],
        run_id="run_exact",
        cycle_id="cycle_1",
    )

    assert [hypothesis.posterior for hypothesis in result.hypotheses] == [0.25, 0.25]
    assert result.frame_state.unresolved_alternative_mass == 0.50
    assert {update.sensitivity["weight"] for update in result.belief_updates} == {0.0}


def test_none_effective_weight_uses_legacy_quality_product():
    state = _exact_state(named={"H1": 0.25, "H2": 0.25}, unresolved=0.50)
    event = _event(
        likelihoods={"H1": LikelihoodBand.STRONGLY_CONFIRMING},
        effective_update_weight=None,
    ).model_copy(update={"reliability": 0.5, "independence": 0.5})

    result = CoverageAwareBeliefSolver().solve(
        state,
        [event],
        run_id="run_exact",
        cycle_id="cycle_1",
    )

    h1_update = next(
        update for update in result.belief_updates if update.hypothesis_id == "H1"
    )
    assert h1_update.sensitivity["weight"] == 0.25
    assert result.hypotheses[0].posterior > 0.25


def test_retirement_returns_named_mass_to_unresolved_before_normalization():
    state = _exact_state(
        named={"H1": 0.37655, "H2": 0.12345},
        unresolved=0.50,
        statuses={"H2": HypothesisStatus.RETIRED},
    )

    result = CoverageAwareBeliefSolver().solve(
        state,
        [],
        run_id="run_exact",
        cycle_id="cycle_1",
    )

    assert result.hypotheses[0].posterior == 0.37655
    assert result.hypotheses[1].posterior == 0.12345
    assert result.frame_state.unresolved_alternative_mass == pytest.approx(0.62345)
    assert (
        result.hypotheses[0].posterior
        + result.frame_state.unresolved_alternative_mass
    ) == 1.0
    assert result.frame_state.active_hypothesis_ids == ["H1"]
    assert result.frame_mass_updates == []


def test_open_solver_preserves_minimum_unresolved_reserve():
    state = _exact_state(named={"H1": 0.475, "H2": 0.475}, unresolved=0.05)
    event = _event(
        likelihoods={
            "H1": LikelihoodBand.STRONGLY_CONFIRMING,
            "H2": LikelihoodBand.STRONGLY_CONFIRMING,
        },
        unresolved_likelihood=LikelihoodBand.STRONGLY_DISCONFIRMING,
    )

    result = CoverageAwareBeliefSolver().solve(
        state,
        [event],
        run_id="run_exact",
        cycle_id="cycle_1",
    )

    assert result.frame_state.unresolved_alternative_mass == 0.05
    assert sum(hypothesis.posterior for hypothesis in result.hypotheses) == 0.95


def test_discarded_event_does_not_change_named_or_unresolved_mass():
    state = _exact_state(named={"H1": 0.25, "H2": 0.25}, unresolved=0.50)
    event = _event(
        likelihoods={"H1": LikelihoodBand.STRONGLY_CONFIRMING},
        unresolved_likelihood=LikelihoodBand.STRONGLY_DISCONFIRMING,
        discard_reason="duplicate_exact",
    )

    result = CoverageAwareBeliefSolver().solve(
        state,
        [event],
        run_id="run_exact",
        cycle_id="cycle_1",
    )

    assert result.hypotheses == state.hypotheses
    assert result.frame_state == state.frame_state
    assert result.belief_updates == []
    assert result.frame_mass_updates == []


def test_native_state_rejects_inconsistent_named_and_unresolved_mass():
    state = _exact_state(named={"H1": 0.25, "H2": 0.25}, unresolved=0.50)
    payload = state.model_dump(mode="python")
    payload["frame_state"]["unresolved_alternative_mass"] = 0.40

    with pytest.raises(ValueError, match="named and unresolved mass must sum to one"):
        BeliefState.model_validate(payload)


def test_open_policy_without_challenge_remains_provisional():
    state = _exact_state(named={"H1": 0.35, "H2": 0.35}, unresolved=0.30)

    decision = FrameAdequacyPolicy().assess(
        previous=state.frame_state,
        events=[],
        hypotheses=state.hypotheses,
    )

    assert decision.frame_state.adequacy_status == FrameAdequacyStatus.PROVISIONAL
    assert decision.should_expand is False
    assert decision.trigger_event_ids == []


def test_model_reasoning_support_for_unresolved_challenges_and_requests_expansion():
    state = _exact_state(named={"H1": 0.25, "H2": 0.25}, unresolved=0.50)
    event = _event(
        likelihoods={"H1": LikelihoodBand.STRONGLY_DISCONFIRMING},
        unresolved_likelihood=LikelihoodBand.STRONGLY_CONFIRMING,
        frame_fit=FrameFit.SUPPORTS_UNRESOLVED,
        verifiability=1.0,
        origin="model_reasoning",
        derivation_root_id="model-root",
    )

    decision = FrameAdequacyPolicy().assess(
        previous=state.frame_state,
        events=[event],
        hypotheses=state.hypotheses,
    )

    assert decision.frame_state.adequacy_status == FrameAdequacyStatus.CHALLENGED
    assert decision.should_expand is True
    assert decision.trigger_event_ids == [event.id]


def test_all_named_disconfirmation_records_the_accepted_trigger_event():
    state = _exact_state(named={"H1": 0.35, "H2": 0.35}, unresolved=0.30)
    event = _event(
        likelihoods={
            "H1": LikelihoodBand.MODERATELY_DISCONFIRMING,
            "H2": LikelihoodBand.MODERATELY_DISCONFIRMING,
        },
        unresolved_likelihood=LikelihoodBand.NEUTRAL,
        frame_fit=FrameFit.UNDERDETERMINED,
    )

    decision = FrameAdequacyPolicy().assess(
        previous=state.frame_state,
        events=[event],
        hypotheses=state.hypotheses,
    )

    assert decision.frame_state.adequacy_status == FrameAdequacyStatus.CHALLENGED
    assert decision.should_expand is True
    assert decision.trigger_event_ids == [event.id]


def test_one_high_verifiability_external_root_marks_open_frame_inadequate():
    state = _exact_state(named={"H1": 0.25, "H2": 0.25}, unresolved=0.50)
    event = _event(
        likelihoods={"H1": LikelihoodBand.STRONGLY_DISCONFIRMING},
        unresolved_likelihood=LikelihoodBand.STRONGLY_CONFIRMING,
        frame_fit=FrameFit.SUPPORTS_UNRESOLVED,
        verifiability=0.75,
        origin="tool_result",
        derivation_root_id="tool-root",
    )

    decision = FrameAdequacyPolicy().assess(
        previous=state.frame_state,
        events=[event],
        hypotheses=state.hypotheses,
    )

    assert decision.frame_state.adequacy_status == FrameAdequacyStatus.INADEQUATE
    assert decision.should_expand is True


def test_two_moderate_external_events_require_distinct_derivation_roots():
    state = _exact_state(named={"H1": 0.25, "H2": 0.25}, unresolved=0.50)
    first = _event(
        event_id="E_first",
        likelihoods={"H1": LikelihoodBand.MODERATELY_DISCONFIRMING},
        unresolved_likelihood=LikelihoodBand.MODERATELY_CONFIRMING,
        frame_fit=FrameFit.SUPPORTS_UNRESOLVED,
        verifiability=0.50,
        origin="retrieved_source",
        derivation_root_id="root-1",
    )
    second = _event(
        event_id="E_second",
        likelihoods={"H2": LikelihoodBand.MODERATELY_DISCONFIRMING},
        unresolved_likelihood=LikelihoodBand.MODERATELY_CONFIRMING,
        frame_fit=FrameFit.SUPPORTS_UNRESOLVED,
        verifiability=0.50,
        origin="external_observation",
        derivation_root_id="root-2",
    )

    distinct = FrameAdequacyPolicy().assess(
        previous=state.frame_state,
        events=[first, second],
        hypotheses=state.hypotheses,
    )
    repeated_root = FrameAdequacyPolicy().assess(
        previous=state.frame_state,
        events=[
            first,
            second.model_copy(update={"derivation_root_id": first.derivation_root_id}),
        ],
        hypotheses=state.hypotheses,
    )

    assert distinct.frame_state.adequacy_status == FrameAdequacyStatus.INADEQUATE
    assert repeated_root.frame_state.adequacy_status == FrameAdequacyStatus.CHALLENGED


def test_inadequate_open_frame_transitions_to_expanding():
    state = _exact_state(
        named={"H1": 0.25, "H2": 0.25},
        unresolved=0.50,
        adequacy=FrameAdequacyStatus.INADEQUATE,
    )

    decision = FrameAdequacyPolicy().assess(
        previous=state.frame_state,
        events=[],
        hypotheses=state.hypotheses,
    )

    assert decision.frame_state.adequacy_status == FrameAdequacyStatus.EXPANDING
    assert decision.should_expand is True


def test_exhaustive_frame_is_adequate():
    previous = FrameState(
        frame_id="mcq-frame",
        competition=HypothesisCompetition.EXCLUSIVE,
        coverage=HypothesisCoverage.EXHAUSTIVE,
        active_hypothesis_ids=["A", "B"],
        unresolved_alternative_mass=0.0,
        adequacy_status=FrameAdequacyStatus.PROVISIONAL,
    )

    decision = FrameAdequacyPolicy().assess(
        previous=previous,
        events=[],
        hypotheses=[],
    )

    assert decision.frame_state.adequacy_status == FrameAdequacyStatus.ADEQUATE
    assert decision.should_expand is False


def test_unresolved_mass_above_every_named_candidate_challenges_open_frame():
    state = _exact_state(named={"H1": 0.20, "H2": 0.20}, unresolved=0.60)

    decision = FrameAdequacyPolicy().assess(
        previous=state.frame_state,
        events=[],
        hypotheses=state.hypotheses,
    )

    assert decision.frame_state.adequacy_status == FrameAdequacyStatus.CHALLENGED
    assert decision.should_expand is True


def test_policy_configuration_is_immutable():
    policy = OpenCoveragePolicy()

    with pytest.raises(FrozenInstanceError):
        policy.initial_unresolved_mass = 0.4


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: OpenCoveragePolicy(initial_unresolved_mass=True), "finite number"),
        (lambda: OpenCoveragePolicy(initial_unresolved_mass=float("nan")), "finite number"),
        (lambda: OpenCoveragePolicy(initial_unresolved_mass=1.1), "between zero and one"),
        (
            lambda: OpenCoveragePolicy(
                initial_unresolved_mass=0.25,
                minimum_unresolved_reserve=0.30,
            ),
            "cannot exceed",
        ),
        (
            lambda: FrameAdequacyPolicyConfig(high_verifiability_threshold=False),
            "finite number",
        ),
        (
            lambda: FrameAdequacyPolicyConfig(required_distinct_moderate_roots=0),
            "positive integer",
        ),
        (lambda: ExpansionPolicy(max_frame_revisions=0), "positive integer"),
        (lambda: ExpansionPolicy(max_active_hypotheses=1.5), "positive integer"),
        (lambda: ProjectionPolicy(exact_margin_threshold=float("inf")), "finite number"),
        (lambda: ProjectionPolicy(max_repair_attempts=False), "positive integer"),
    ],
)
def test_policy_configuration_rejects_invalid_values(factory, match):
    with pytest.raises(ValueError, match=match):
        factory()
