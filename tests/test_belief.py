import pytest

from bayesprobe.belief import solve_updates
from bayesprobe.schemas import (
    AnswerContract,
    BeliefState,
    EvidenceEvent,
    EvidenceType,
    FramedHypothesis,
    FramingMethod,
    Hypothesis,
    HypothesisFrame,
    HypothesisRelation,
    HypothesisStatus,
    LikelihoodBand,
    TaskFrame,
    TaskKind,
)


def _hypothesis(
    hypothesis_id: str,
    posterior: float,
    *,
    complexity_penalty: float = 0.0,
    ad_hoc_penalty: float = 0.0,
    applied_complexity_penalty: float = 0.0,
    applied_ad_hoc_penalty: float = 0.0,
    status: HypothesisStatus = HypothesisStatus.ACTIVE,
) -> Hypothesis:
    return Hypothesis(
        id=hypothesis_id,
        statement=f"{hypothesis_id} is the correct rival.",
        scope="Exclusive categorical fixture.",
        prior=posterior,
        posterior=posterior,
        complexity_penalty=complexity_penalty,
        ad_hoc_penalty=ad_hoc_penalty,
        applied_complexity_penalty=applied_complexity_penalty,
        applied_ad_hoc_penalty=applied_ad_hoc_penalty,
        status=status,
    )


def _belief_state(
    hypotheses: list[Hypothesis],
    *,
    relation: HypothesisRelation = HypothesisRelation.EXCLUSIVE_EXHAUSTIVE,
) -> BeliefState:
    ids = [hypothesis.id for hypothesis in hypotheses]
    return BeliefState(
        belief_state_id="run_belief_bs_0",
        run_id="run_belief",
        cycle_id="cycle_0",
        hypotheses=hypotheses,
        task_frame=TaskFrame(
            task_frame_id="run_belief_task_frame",
            task_kind=TaskKind.DECISION,
            normalized_question="Which hypotheses remain credible?",
            task_context="",
            answer_contract=AnswerContract(
                objective="Report the current beliefs.",
                required_sections=["answer", "uncertainty"],
                decision_form="belief_report",
                permits_synthesis=relation == HypothesisRelation.INDEPENDENT,
            ),
            hypothesis_frame=HypothesisFrame(
                frame_id="run_belief_hypothesis_frame",
                relation=relation,
                hypotheses=[
                    FramedHypothesis(
                        id=hypothesis.id,
                        statement=hypothesis.statement,
                        type=hypothesis.type,
                        scope=hypothesis.scope,
                        initial_prior=hypothesis.prior,
                        falsifiers=list(hypothesis.falsifiers)
                        or [f"A result falsifies {hypothesis.id}."],
                        predictions=list(hypothesis.predictions)
                        or [f"A result supports {hypothesis.id}."],
                    )
                    for hypothesis in hypotheses
                ],
                rival_sets={
                    hypothesis_id: [other for other in ids if other != hypothesis_id]
                    if relation == HypothesisRelation.EXCLUSIVE_EXHAUSTIVE
                    else []
                    for hypothesis_id in ids
                },
                coverage_statement="Controlled relation-aware test frame.",
            ),
            framing_method=FramingMethod.RECORDED,
        ),
    )


def _event(
    likelihoods: dict[str, LikelihoodBand],
    *,
    target_hypotheses: list[str] | None = None,
    event_id: str = "E_belief",
) -> EvidenceEvent:
    targets = target_hypotheses or list(likelihoods)
    return EvidenceEvent(
        id=event_id,
        derived_from_signal="S_belief",
        target_hypotheses=targets,
        evidence_type=EvidenceType.SUPPORTING,
        content="A categorical update fixture.",
        reliability=1.0,
        independence=1.0,
        relevance=1.0,
        novelty=1.0,
        likelihoods=likelihoods,
    )


def test_solve_updates_normalizes_exclusive_rivals_and_audits_rival_movement():
    state = _belief_state(
        [
            _hypothesis("H1", 0.34),
            _hypothesis("H2", 0.33),
            _hypothesis("H3", 0.33),
        ]
    )

    hypotheses, updates = solve_updates(
        run_id="run_belief",
        cycle_id="cycle_1",
        belief_state=state,
        events=[
            _event(
                {"H2": LikelihoodBand.STRONGLY_CONFIRMING},
                target_hypotheses=["H2"],
            )
        ],
    )

    posterior = {hypothesis.id: hypothesis.posterior for hypothesis in hypotheses}
    assert sum(posterior.values()) == pytest.approx(1.0)
    assert posterior["H2"] > posterior["H1"]
    assert posterior["H2"] > posterior["H3"]
    assert posterior["H1"] < 0.34
    assert posterior["H3"] < 0.33
    assert {update.hypothesis_id for update in updates} == {"H1", "H2", "H3"}


def test_solve_updates_applies_complexity_and_ad_hoc_penalties():
    state = _belief_state(
        [
            _hypothesis("H1", 0.5),
            _hypothesis(
                "H2",
                0.5,
                complexity_penalty=0.15,
                ad_hoc_penalty=0.1,
            ),
        ]
    )

    hypotheses, _ = solve_updates(
        run_id="run_belief",
        cycle_id="cycle_1",
        belief_state=state,
        events=[
            _event(
                {
                    "H1": LikelihoodBand.NEUTRAL,
                    "H2": LikelihoodBand.NEUTRAL,
                }
            )
        ],
    )

    posterior = {hypothesis.id: hypothesis.posterior for hypothesis in hypotheses}
    assert posterior["H1"] > posterior["H2"]
    assert sum(posterior.values()) == pytest.approx(1.0)


def test_independent_update_does_not_cross_normalize_untargeted_hypothesis():
    state = _belief_state(
        [_hypothesis("H1", 0.5), _hypothesis("H2", 0.5)],
        relation=HypothesisRelation.INDEPENDENT,
    )

    hypotheses, updates = solve_updates(
        "run_belief",
        "cycle_1",
        state,
        [
            _event(
                {"H1": LikelihoodBand.STRONGLY_CONFIRMING},
                target_hypotheses=["H1"],
            )
        ],
    )

    by_id = {item.id: item for item in hypotheses}
    assert by_id["H1"].posterior > 0.5
    assert by_id["H2"].posterior == 0.5
    assert sum(item.posterior for item in hypotheses) > 1.0
    assert [update.hypothesis_id for update in updates] == ["H1"]


def test_static_penalties_are_not_subtracted_again_on_later_events():
    state = _belief_state(
        [
            _hypothesis(
                "H1",
                0.5,
                complexity_penalty=0.2,
                ad_hoc_penalty=0.1,
            ),
            _hypothesis("H2", 0.5),
        ],
        relation=HypothesisRelation.INDEPENDENT,
    )
    neutral = _event(
        {"H1": LikelihoodBand.NEUTRAL},
        target_hypotheses=["H1"],
    )

    after_first, _ = solve_updates("run_belief", "cycle_1", state, [neutral])
    state_after_first = state.model_copy(update={"hypotheses": after_first})
    after_second, _ = solve_updates(
        "run_belief", "cycle_2", state_after_first, [neutral]
    )

    assert after_second[0].posterior == after_first[0].posterior


def test_discarded_event_applies_neither_evidence_nor_static_penalties():
    state = _belief_state(
        [
            _hypothesis(
                "H1",
                0.5,
                complexity_penalty=0.2,
                ad_hoc_penalty=0.1,
            ),
            _hypothesis("H2", 0.5),
        ],
        relation=HypothesisRelation.INDEPENDENT,
    )
    discarded = _event(
        {"H1": LikelihoodBand.STRONGLY_CONFIRMING},
        target_hypotheses=["H1"],
    ).model_copy(update={"discard_reason": "duplicate"})

    hypotheses, updates = solve_updates(
        "run_belief", "cycle_1", state, [discarded]
    )

    assert hypotheses == state.hypotheses
    assert updates == []


def test_relation_less_direct_solve_fails_with_stable_error():
    state = BeliefState(
        belief_state_id="run_belief_bs_legacy",
        run_id="run_belief",
        cycle_id="cycle_0",
        hypotheses=[_hypothesis("H1", 0.5), _hypothesis("H2", 0.5)],
    )

    with pytest.raises(
        ValueError,
        match="^belief state requires hypothesis relation metadata$",
    ):
        solve_updates("run_belief", "cycle_1", state, [])


def test_past_evidence_id_replay_is_an_exact_no_op():
    state = _belief_state(
        [
            _hypothesis(
                "H1",
                0.5,
                complexity_penalty=0.2,
                ad_hoc_penalty=0.1,
            ),
            _hypothesis("H2", 0.5),
        ],
        relation=HypothesisRelation.INDEPENDENT,
    ).model_copy(update={"ledger_refs": {"evidence_events": ["E_seen"]}})
    replay = _event(
        {"H1": LikelihoodBand.STRONGLY_CONFIRMING},
        target_hypotheses=["H1"],
        event_id="E_seen",
    )

    hypotheses, updates = solve_updates("run_belief", "cycle_1", state, [replay])

    assert hypotheses == state.hypotheses
    assert updates == []


def test_same_cycle_duplicate_evidence_id_applies_at_most_once():
    state = _belief_state(
        [_hypothesis("H1", 0.5), _hypothesis("H2", 0.5)],
        relation=HypothesisRelation.INDEPENDENT,
    )
    duplicate = _event(
        {"H1": LikelihoodBand.MODERATELY_CONFIRMING},
        target_hypotheses=["H1"],
        event_id="E_duplicate",
    )

    hypotheses, updates = solve_updates(
        "run_belief",
        "cycle_1",
        state,
        [duplicate, duplicate],
    )
    once_hypotheses, once_updates = solve_updates(
        "run_belief",
        "cycle_1",
        state,
        [duplicate],
    )

    assert hypotheses == once_hypotheses
    assert updates == once_updates


def test_discarded_exclusive_thirds_are_returned_exactly_unchanged():
    third = 1.0 / 3.0
    state = _belief_state(
        [
            _hypothesis("H1", third),
            _hypothesis("H2", third),
            _hypothesis("H3", third),
        ]
    )
    discarded = _event(
        {"H1": LikelihoodBand.STRONGLY_CONFIRMING},
        target_hypotheses=["H1"],
        event_id="E_discarded_thirds",
    ).model_copy(update={"discard_reason": "inadmissible"})

    hypotheses, updates = solve_updates(
        "run_belief", "cycle_1", state, [discarded]
    )

    assert hypotheses == state.hypotheses
    assert [item.posterior for item in hypotheses] == [third, third, third]
    assert updates == []


def test_penalty_high_water_survives_decrease_and_reincrease_below_peak():
    state = _belief_state(
        [
            _hypothesis(
                "H1",
                0.5,
                complexity_penalty=0.2,
                ad_hoc_penalty=0.1,
                applied_complexity_penalty=0.4,
                applied_ad_hoc_penalty=0.3,
            ),
            _hypothesis("H2", 0.5),
        ],
        relation=HypothesisRelation.INDEPENDENT,
    )
    neutral = _event(
        {"H1": LikelihoodBand.NEUTRAL},
        target_hypotheses=["H1"],
    )

    lowered, _ = solve_updates("run_belief", "cycle_1", state, [neutral])
    lowered_h1 = lowered[0]
    assert lowered_h1.posterior == 0.5
    assert lowered_h1.applied_complexity_penalty == 0.4
    assert lowered_h1.applied_ad_hoc_penalty == 0.3

    below_peak_state = state.model_copy(
        update={
            "hypotheses": [
                lowered_h1.model_copy(
                    update={"complexity_penalty": 0.35, "ad_hoc_penalty": 0.25}
                ),
                lowered[1],
            ]
        }
    )
    below_peak, _ = solve_updates(
        "run_belief", "cycle_2", below_peak_state, [neutral]
    )
    assert below_peak[0].posterior == 0.5
    assert below_peak[0].applied_complexity_penalty == 0.4
    assert below_peak[0].applied_ad_hoc_penalty == 0.3

    above_peak_state = below_peak_state.model_copy(
        update={
            "hypotheses": [
                below_peak[0].model_copy(
                    update={"complexity_penalty": 0.5, "ad_hoc_penalty": 0.4}
                ),
                below_peak[1],
            ]
        }
    )
    above_peak, updates = solve_updates(
        "run_belief", "cycle_3", above_peak_state, [neutral]
    )
    assert above_peak[0].posterior < 0.5
    assert above_peak[0].applied_complexity_penalty == 0.5
    assert above_peak[0].applied_ad_hoc_penalty == 0.4
    assert updates[0].sensitivity["complexity_penalty_delta"] == 0.1
    assert updates[0].sensitivity["ad_hoc_penalty_delta"] == 0.1


def test_independent_multi_event_updates_chain_from_previous_posterior():
    state = _belief_state(
        [_hypothesis("H1", 0.5), _hypothesis("H2", 0.5)],
        relation=HypothesisRelation.INDEPENDENT,
    )
    first = _event(
        {"H1": LikelihoodBand.MODERATELY_CONFIRMING},
        target_hypotheses=["H1"],
        event_id="E_first",
    )
    second = _event(
        {"H1": LikelihoodBand.WEAKLY_DISCONFIRMING},
        target_hypotheses=["H1"],
        event_id="E_second",
    )

    hypotheses, updates = solve_updates(
        "run_belief", "cycle_1", state, [first, second]
    )

    assert len(updates) == 2
    assert updates[1].prior == updates[0].posterior
    assert hypotheses[0].posterior == updates[1].posterior
    assert hypotheses[1] == state.hypotheses[1]


def test_duplicate_targets_create_one_independent_update():
    state = _belief_state(
        [_hypothesis("H1", 0.5), _hypothesis("H2", 0.5)],
        relation=HypothesisRelation.INDEPENDENT,
    )
    event = _event(
        {"H1": LikelihoodBand.WEAKLY_CONFIRMING},
        target_hypotheses=["H1", "H1"],
    )

    _, updates = solve_updates("run_belief", "cycle_1", state, [event])

    assert [update.hypothesis_id for update in updates] == ["H1"]


def test_inactive_independent_target_is_unchanged_and_not_audited():
    state = _belief_state(
        [
            _hypothesis("H1", 0.2, status=HypothesisStatus.RETIRED),
            _hypothesis("H2", 0.6),
        ],
        relation=HypothesisRelation.INDEPENDENT,
    )
    event = _event(
        {"H1": LikelihoodBand.STRONGLY_CONFIRMING},
        target_hypotheses=["H1"],
    )

    hypotheses, updates = solve_updates("run_belief", "cycle_1", state, [event])

    assert hypotheses == state.hypotheses
    assert updates == []


@pytest.mark.parametrize("boundary", [0.0, 1.0])
def test_independent_update_is_numerically_stable_at_probability_boundary(boundary):
    state = _belief_state(
        [_hypothesis("H1", boundary), _hypothesis("H2", 0.5)],
        relation=HypothesisRelation.INDEPENDENT,
    )
    event = _event(
        {"H1": LikelihoodBand.STRONGLY_CONFIRMING},
        target_hypotheses=["H1"],
    )

    hypotheses, updates = solve_updates("run_belief", "cycle_1", state, [event])

    assert 0.0 <= hypotheses[0].posterior <= 1.0
    assert len(updates) == 1
