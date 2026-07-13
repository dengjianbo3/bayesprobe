import json
import math

import pytest

from bayesprobe.belief import (
    CoverageAwareBeliefSolver,
    legacy_event_contribution_deltas,
    solve_updates,
)
from bayesprobe.migrations import migrate_belief_state_v0_1
from bayesprobe.schemas import (
    AnswerContract,
    BeliefState,
    EpistemicOrigin,
    EvidenceContributionDelta,
    EvidenceContributionMode,
    EvidenceEvent,
    EvidenceRootContribution,
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


def _contribution(
    root: str,
    *,
    revision: int,
    hypotheses: dict[str, float],
    unresolved: float | None = None,
    active: bool = True,
    event_id: str | None = None,
) -> EvidenceRootContribution:
    return EvidenceRootContribution(
        contribution_root_id=root,
        revision=revision,
        assessment_event_ids=[event_id or f"E_{root}"],
        epistemic_origin=EpistemicOrigin.MODEL_REASONING,
        per_hypothesis_log_likelihood=hypotheses,
        unresolved_log_likelihood=unresolved,
        active=active,
    )


def _new_delta(
    root: str,
    hypotheses: dict[str, float],
    *,
    unresolved: float | None = None,
    event_id: str | None = None,
) -> EvidenceContributionDelta:
    cause = event_id or f"E_{root}"
    current = _contribution(
        root,
        revision=1,
        hypotheses=hypotheses,
        unresolved=unresolved,
        event_id=cause,
    )
    return EvidenceContributionDelta(
        contribution_root_id=root,
        mode=EvidenceContributionMode.NEW_ROOT,
        current_contribution=current,
        per_hypothesis_delta=hypotheses,
        unresolved_delta=unresolved,
        caused_by_event_ids=[cause],
    )


def _revision_delta(
    root: str,
    previous_hypotheses: dict[str, float],
    current_hypotheses: dict[str, float],
    *,
    event_id: str | None = None,
) -> EvidenceContributionDelta:
    cause = event_id or f"E_{root}_revision"
    hypothesis_ids = sorted(set(previous_hypotheses).union(current_hypotheses))
    hypothesis_delta = {
        hypothesis_id: current_hypotheses.get(hypothesis_id, 0.0)
        - previous_hypotheses.get(hypothesis_id, 0.0)
        for hypothesis_id in hypothesis_ids
    }
    previous = _contribution(
        root,
        revision=1,
        hypotheses=previous_hypotheses,
        event_id=f"E_{root}_previous",
    )
    current = _contribution(
        root,
        revision=2,
        hypotheses=current_hypotheses,
        event_id=cause,
    )
    return EvidenceContributionDelta(
        contribution_root_id=root,
        mode=EvidenceContributionMode.REVISE_ROOT,
        previous_contribution=previous,
        current_contribution=current,
        per_hypothesis_delta=hypothesis_delta,
        caused_by_event_ids=[cause],
    )


def _no_change_delta(
    root: str,
    hypotheses: dict[str, float],
    *,
    unresolved: float | None = None,
) -> EvidenceContributionDelta:
    previous = _contribution(
        root,
        revision=1,
        hypotheses=hypotheses,
        unresolved=unresolved,
        event_id=f"E_{root}_previous",
    )
    current = _contribution(
        root,
        revision=2,
        hypotheses=hypotheses,
        unresolved=unresolved,
        event_id=f"E_{root}_current",
    )
    return EvidenceContributionDelta(
        contribution_root_id=root,
        mode=EvidenceContributionMode.NO_CHANGE,
        previous_contribution=previous,
        current_contribution=current,
        per_hypothesis_delta={hypothesis_id: 0.0 for hypothesis_id in hypotheses},
        unresolved_delta=0.0 if unresolved is not None else None,
        caused_by_event_ids=[f"E_{root}_current"],
    )


def _tolerated_no_change_delta(
    root: str,
    previous_hypotheses: dict[str, float],
    coordinate_changes: dict[str, float],
) -> EvidenceContributionDelta:
    current_hypotheses = {
        hypothesis_id: value + coordinate_changes.get(hypothesis_id, 0.0)
        for hypothesis_id, value in previous_hypotheses.items()
    }
    actual_delta = {
        hypothesis_id: current_hypotheses[hypothesis_id] - value
        for hypothesis_id, value in previous_hypotheses.items()
    }
    previous = _contribution(
        root,
        revision=1,
        hypotheses=previous_hypotheses,
        event_id=f"E_{root}_previous",
    )
    current = _contribution(
        root,
        revision=2,
        hypotheses=current_hypotheses,
        event_id=f"E_{root}_current",
    )
    return EvidenceContributionDelta(
        contribution_root_id=root,
        mode=EvidenceContributionMode.NO_CHANGE,
        previous_contribution=previous,
        current_contribution=current,
        per_hypothesis_delta=actual_delta,
        caused_by_event_ids=[f"E_{root}_current"],
    )


def _native_state(
    hypotheses: list[Hypothesis],
    *,
    relation: HypothesisRelation = HypothesisRelation.EXCLUSIVE_EXHAUSTIVE,
) -> BeliefState:
    return migrate_belief_state_v0_1(
        _belief_state(hypotheses, relation=relation)
    )


def test_no_change_delta_leaves_posterior_and_penalty_high_water_exact():
    state = _native_state(
        [
            _hypothesis(
                "H1",
                0.6,
                complexity_penalty=0.4,
                ad_hoc_penalty=0.2,
                applied_complexity_penalty=0.1,
                applied_ad_hoc_penalty=0.1,
            ),
            _hypothesis("H2", 0.4),
        ]
    )

    result = CoverageAwareBeliefSolver().solve(
        state,
        [_no_change_delta("eroot:model", {"H1": 0.7, "H2": -0.3})],
        run_id=state.run_id,
        cycle_id="cycle_2",
    )

    assert result.hypotheses == state.hypotheses
    assert result.frame_state == state.frame_state
    assert result.belief_updates == []
    assert result.frame_mass_updates == []


def test_tolerance_sized_no_change_is_unconditional_no_op_alone_and_mixed():
    state = _native_state(
        [
            _hypothesis(
                "H1",
                0.4,
                complexity_penalty=0.3,
                ad_hoc_penalty=0.2,
            ),
            _hypothesis("H2", 0.6),
        ],
        relation=HypothesisRelation.INDEPENDENT,
    )
    no_change = _tolerated_no_change_delta(
        "eroot:a-no-change",
        {"H1": 0.25, "H2": -0.25},
        {"H1": 5e-13, "H2": -5e-13},
    )

    alone = CoverageAwareBeliefSolver().solve(
        state,
        [no_change],
        run_id=state.run_id,
        cycle_id="cycle_1",
    )

    assert alone.hypotheses == state.hypotheses
    assert alone.frame_state is state.frame_state
    assert alone.belief_updates == []
    assert alone.frame_mass_updates == []

    effective = _new_delta("eroot:z-effective", {"H2": 0.4})
    expected = CoverageAwareBeliefSolver().solve(
        state,
        [effective],
        run_id=state.run_id,
        cycle_id="cycle_1",
    )
    mixed = CoverageAwareBeliefSolver().solve(
        state,
        [no_change, effective],
        run_id=state.run_id,
        cycle_id="cycle_1",
    )

    assert mixed == expected
    assert {update.evidence_id for update in mixed.belief_updates} == {
        effective.contribution_root_id
    }


def test_empty_and_all_zero_batches_preserve_original_frame_metadata():
    state = _native_state([_hypothesis("H1", 0.5), _hypothesis("H2", 0.5)])
    original_frame = state.frame_state.model_copy(
        update={
            "active_hypothesis_ids": ["H2", "H1"],
            "revision_reason": "Preserve this exact no-op frame metadata.",
            "trigger_event_ids": ["E_prior_frame_revision"],
            "revision_count": 2,
        }
    )
    state = state.model_copy(update={"frame_state": original_frame})
    batches = [
        [],
        [_new_delta("eroot:zero-frame", {"H1": 0.0, "H2": 0.0})],
        [_no_change_delta("eroot:no-change-frame", {"H1": 0.2, "H2": -0.2})],
    ]

    for batch in batches:
        result = CoverageAwareBeliefSolver().solve(
            state,
            batch,
            run_id=state.run_id,
            cycle_id="cycle_1",
        )

        assert result.hypotheses is state.hypotheses
        assert result.frame_state is original_frame
        assert result.frame_state.active_hypothesis_ids == ["H2", "H1"]
        assert result.belief_updates == []
        assert result.frame_mass_updates == []


def test_zero_new_root_is_an_exact_no_op():
    third = 1.0 / 3.0
    state = _native_state(
        [
            _hypothesis("H1", third, complexity_penalty=0.2),
            _hypothesis("H2", third),
            _hypothesis("H3", third),
        ]
    )

    result = CoverageAwareBeliefSolver().solve(
        state,
        [_new_delta("eroot:zero", {"H1": 0.0, "H2": 0.0, "H3": 0.0})],
        run_id=state.run_id,
        cycle_id="cycle_1",
    )

    assert result.hypotheses == state.hypotheses
    assert [item.posterior for item in result.hypotheses] == [third, third, third]
    assert result.belief_updates == []


def test_revision_applies_only_current_minus_previous_contribution():
    state = _native_state([_hypothesis("H1", 0.7), _hypothesis("H2", 0.3)])
    previous_vector = {"H1": 0.9, "H2": -0.4}
    current_vector = {"H1": 0.1, "H2": 0.4}
    delta = _revision_delta(
        "eroot:model",
        previous_vector,
        current_vector,
        event_id="E_counterassessment",
    )
    expected_delta = {
        hypothesis_id: current_vector[hypothesis_id]
        - previous_vector[hypothesis_id]
        for hypothesis_id in previous_vector
    }
    assert expected_delta == {"H1": -0.8, "H2": 0.8}
    assert all(value != 0.0 for value in previous_vector.values())
    assert all(value != 0.0 for value in current_vector.values())
    assert previous_vector != current_vector
    assert delta.per_hypothesis_delta == expected_delta
    assert delta.per_hypothesis_delta not in (previous_vector, current_vector)

    result = CoverageAwareBeliefSolver().solve(
        state,
        [delta],
        run_id=state.run_id,
        cycle_id="cycle_2",
    )

    denominator = sum(
        prior * math.exp(expected_delta[hypothesis_id])
        for hypothesis_id, prior in {"H1": 0.7, "H2": 0.3}.items()
    )
    expected_h1 = round(
        0.7 * math.exp(expected_delta["H1"]) / denominator,
        4,
    )
    by_id = {item.id: item for item in result.hypotheses}
    assert by_id["H1"].posterior == expected_h1
    assert by_id["H2"].posterior == round(1.0 - expected_h1, 4)
    assert {update.evidence_id for update in result.belief_updates} == {
        "eroot:model"
    }
    assert {
        update.sensitivity["contribution_mode"]
        for update in result.belief_updates
    } == {"revise_root"}
    assert {
        tuple(update.sensitivity["caused_by_event_ids"])
        for update in result.belief_updates
    } == {("E_counterassessment",)}


def test_distinct_roots_are_order_invariant_and_round_only_final_distribution():
    state = _native_state(
        [
            _hypothesis("H1", 0.3333),
            _hypothesis("H2", 0.3333),
            _hypothesis("H3", 0.3334),
        ]
    )
    deltas = [
        _new_delta("eroot:z", {"H1": 0.12345, "H2": -0.22222}),
        _new_delta("eroot:a", {"H2": 0.33333, "H3": -0.11111}),
    ]

    forward = CoverageAwareBeliefSolver().solve(
        state,
        deltas,
        run_id=state.run_id,
        cycle_id="cycle_1",
    )
    reverse = CoverageAwareBeliefSolver().solve(
        state,
        list(reversed(deltas)),
        run_id=state.run_id,
        cycle_id="cycle_1",
    )

    scores = {
        "H1": math.log(0.3333) + 0.12345,
        "H2": math.log(0.3333) - 0.22222 + 0.33333,
        "H3": math.log(0.3334) - 0.11111,
    }
    maximum = max(scores.values())
    exponentials = {
        hypothesis_id: math.exp(score - maximum)
        for hypothesis_id, score in scores.items()
    }
    total = sum(exponentials.values())
    expected = {
        hypothesis_id: round(value / total, 4)
        for hypothesis_id, value in exponentials.items()
    }
    expected[max(expected, key=expected.get)] = round(
        expected[max(expected, key=expected.get)]
        + 1.0
        - sum(expected.values()),
        4,
    )
    assert forward == reverse
    assert {item.id: item.posterior for item in forward.hypotheses} == expected
    assert [update.evidence_id for update in forward.belief_updates] == [
        "eroot:a",
        "eroot:a",
        "eroot:a",
        "eroot:z",
        "eroot:z",
        "eroot:z",
    ]


def test_independent_delta_uses_logit_arithmetic_without_cross_normalization():
    state = _native_state(
        [_hypothesis("H1", 0.4), _hypothesis("H2", 0.7)],
        relation=HypothesisRelation.INDEPENDENT,
    )
    delta = _new_delta("eroot:independent", {"H1": math.log(3.0)})

    result = CoverageAwareBeliefSolver().solve(
        state,
        [delta],
        run_id=state.run_id,
        cycle_id="cycle_1",
    )

    by_id = {item.id: item for item in result.hypotheses}
    expected_h1 = round((0.4 * 3.0) / (1.0 - 0.4 + 0.4 * 3.0), 4)
    assert by_id["H1"].posterior == expected_h1
    assert by_id["H2"] == state.hypotheses_by_id()["H2"]
    assert sum(item.posterior for item in result.hypotheses) > 1.0
    assert [update.hypothesis_id for update in result.belief_updates] == ["H1"]


def test_solver_applies_penalty_high_water_only_once_per_delta_batch():
    state = _native_state(
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

    result = CoverageAwareBeliefSolver().solve(
        state,
        [
            _new_delta("eroot:a", {"H1": 0.2}),
            _new_delta("eroot:b", {"H1": 0.2}),
        ],
        run_id=state.run_id,
        cycle_id="cycle_1",
    )

    expected = round(1.0 / (1.0 + math.exp(-0.1)), 4)
    h1 = {item.id: item for item in result.hypotheses}["H1"]
    assert h1.posterior == expected
    assert h1.applied_complexity_penalty == 0.2
    assert h1.applied_ad_hoc_penalty == 0.1
    assert [
        update.sensitivity["complexity_penalty_delta"]
        for update in result.belief_updates
    ] == [0.2, 0.0]
    assert [
        update.sensitivity["ad_hoc_penalty_delta"]
        for update in result.belief_updates
    ] == [0.1, 0.0]


def test_solver_rejects_raw_events_duplicate_roots_and_unknown_coordinates():
    state = _native_state([_hypothesis("H1", 0.5), _hypothesis("H2", 0.5)])
    duplicate = _new_delta("eroot:duplicate", {"H1": 0.1})

    with pytest.raises(TypeError, match="EvidenceContributionDelta"):
        CoverageAwareBeliefSolver().solve(
            state,
            [_event({"H1": LikelihoodBand.WEAKLY_CONFIRMING})],
            run_id=state.run_id,
            cycle_id="cycle_1",
        )
    with pytest.raises(ValueError, match="duplicate contribution root"):
        CoverageAwareBeliefSolver().solve(
            state,
            [duplicate, duplicate],
            run_id=state.run_id,
            cycle_id="cycle_1",
        )
    with pytest.raises(ValueError, match="unknown hypothesis coordinate"):
        CoverageAwareBeliefSolver().solve(
            state,
            [_new_delta("eroot:unknown", {"H_unknown": 0.1})],
            run_id=state.run_id,
            cycle_id="cycle_1",
        )


def test_independent_solver_rejects_unresolved_coordinate():
    state = _native_state(
        [_hypothesis("H1", 0.5), _hypothesis("H2", 0.5)],
        relation=HypothesisRelation.INDEPENDENT,
    )

    with pytest.raises(ValueError, match="unresolved delta"):
        CoverageAwareBeliefSolver().solve(
            state,
            [_new_delta("eroot:invalid-frame", {"H1": 0.1}, unresolved=0.2)],
            run_id=state.run_id,
            cycle_id="cycle_1",
        )


@pytest.mark.parametrize(
    "relation",
    [
        HypothesisRelation.INDEPENDENT,
        HypothesisRelation.EXCLUSIVE_EXHAUSTIVE,
    ],
)
@pytest.mark.parametrize("coordinate_location", ["delta", "current", "previous"])
def test_non_open_solver_rejects_explicit_zero_unresolved_coordinate_anywhere(
    relation: HypothesisRelation,
    coordinate_location: str,
):
    state = _native_state(
        [_hypothesis("H1", 0.5), _hypothesis("H2", 0.5)],
        relation=relation,
    )
    root = f"eroot:unresolved-{coordinate_location}"
    if coordinate_location == "previous":
        previous = _contribution(
            root,
            revision=1,
            hypotheses={"H1": 0.2},
            unresolved=0.0,
        )
        current = _contribution(
            root,
            revision=2,
            hypotheses={"H1": 0.4},
        )
        delta = EvidenceContributionDelta(
            contribution_root_id=root,
            mode=EvidenceContributionMode.REVISE_ROOT,
            previous_contribution=previous,
            current_contribution=current,
            per_hypothesis_delta={"H1": 0.2},
            caused_by_event_ids=[f"E_{root}_current"],
        )
    else:
        current = _contribution(
            root,
            revision=1,
            hypotheses={"H1": 0.2},
            unresolved=0.0 if coordinate_location == "current" else None,
        )
        delta = EvidenceContributionDelta(
            contribution_root_id=root,
            mode=EvidenceContributionMode.NEW_ROOT,
            current_contribution=current,
            per_hypothesis_delta={"H1": 0.2},
            unresolved_delta=0.0 if coordinate_location == "delta" else None,
            caused_by_event_ids=[f"E_{root}"],
        )

    with pytest.raises(ValueError, match="unresolved .*coordinate"):
        CoverageAwareBeliefSolver().solve(
            state,
            [delta],
            run_id=state.run_id,
            cycle_id="cycle_1",
        )


def test_legacy_adapter_matches_historical_weight_and_ignores_discarded_events():
    accepted = _event(
        {"H1": LikelihoodBand.MODERATELY_CONFIRMING},
        event_id="E_legacy_accepted",
    ).model_copy(
        update={
            "reliability": 0.5,
            "independence": 0.4,
            "relevance": 0.75,
            "novelty": 0.8,
        }
    )
    discarded = _event(
        {"H1": LikelihoodBand.STRONGLY_CONFIRMING},
        event_id="E_legacy_discarded",
    ).model_copy(update={"discard_reason": "replayed"})

    deltas = legacy_event_contribution_deltas([discarded, accepted])

    assert len(deltas) == 1
    delta = deltas[0]
    expected_weight = 0.5 * 0.4 * 0.75 * 0.8
    assert delta.per_hypothesis_delta["H1"] == pytest.approx(
        expected_weight * math.log(3.0)
    )
    assert delta.current_contribution.assessment_event_ids == [accepted.id]
    assert delta.caused_by_event_ids == [accepted.id]
    assert delta.contribution_root_id.startswith("legacy-event-root:sha256:")
    assert accepted.id not in delta.contribution_root_id


def test_legacy_adapter_rejects_native_root_bound_event_without_weight():
    event = EvidenceEvent(
        schema_version="v0.2",
        id="E_native_root",
        derived_from_signal="S_native_root",
        epistemic_origin=EpistemicOrigin.MODEL_REASONING,
        derivation_root_id="derivation:model",
        contribution_root_id="eroot:model",
        target_hypotheses=["H1"],
        evidence_type=EvidenceType.SUPPORTING,
        content="A native root-bound event.",
        likelihoods={"H1": LikelihoodBand.WEAKLY_CONFIRMING},
        correlation_status="novel",
        effective_update_weight=None,
    )

    with pytest.raises(ValueError, match="root-bound"):
        legacy_event_contribution_deltas([event])


def test_legacy_adapter_rejects_secret_like_event_id_before_hashing(
    monkeypatch: pytest.MonkeyPatch,
):
    encoding_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class SecretLikeEventId(str):
        def encode(self, *args: object, **kwargs: object) -> bytes:
            encoding_calls.append((args, kwargs))
            raise AssertionError("secret-like ids must be rejected before encoding")

    secret_event_id = SecretLikeEventId("sk-" + "a" * 32)
    event = _event(
        {"H1": LikelihoodBand.WEAKLY_CONFIRMING},
        event_id="E_safe_fixture",
    ).model_copy(update={"id": secret_event_id})
    hash_inputs: list[object] = []

    def unexpected_hash(value: object):
        hash_inputs.append(value)
        raise AssertionError("secret-like ids must be rejected before hashing")

    monkeypatch.setattr("bayesprobe.belief.hashlib.sha256", unexpected_hash)

    with pytest.raises(ValueError, match="secret-like legacy event id") as exc_info:
        legacy_event_contribution_deltas([event])

    assert encoding_calls == []
    assert hash_inputs == []
    rendered_errors = [
        str(exc_info.value),
        repr(exc_info.value),
        json.dumps({"error": str(exc_info.value)}),
    ]
    assert all(secret_event_id not in rendered for rendered in rendered_errors)


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
                    "H1": LikelihoodBand.WEAKLY_CONFIRMING,
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


def test_static_penalties_are_not_subtracted_again_on_later_deltas():
    state = _native_state(
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
    first = CoverageAwareBeliefSolver().solve(
        state,
        [_new_delta("eroot:first", {"H1": 0.5})],
        run_id=state.run_id,
        cycle_id="cycle_1",
    )
    state_after_first = state.model_copy(update={"hypotheses": first.hypotheses})
    second = CoverageAwareBeliefSolver().solve(
        state_after_first,
        [_new_delta("eroot:second", {"H1": 0.5})],
        run_id=state.run_id,
        cycle_id="cycle_2",
    )

    assert first.hypotheses[0].posterior == round(1.0 / (1.0 + math.exp(-0.2)), 4)
    assert second.hypotheses[0].posterior == round(
        1.0 / (1.0 + math.exp(-0.7)),
        4,
    )
    assert second.belief_updates[0].sensitivity["complexity_penalty_delta"] == 0.0
    assert second.belief_updates[0].sensitivity["ad_hoc_penalty_delta"] == 0.0


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
        "run_belief",
        "cycle_3",
        above_peak_state,
        [
            _event(
                {"H1": LikelihoodBand.WEAKLY_DISCONFIRMING},
                target_hypotheses=["H1"],
                event_id="E_above_peak",
            )
        ],
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


@pytest.mark.parametrize(
    "status",
    [HypothesisStatus.RETIRED, HypothesisStatus.ARCHIVED],
)
def test_inactive_independent_target_is_unchanged_and_not_audited(status):
    state = _belief_state(
        [
            _hypothesis("H1", 0.2, status=status),
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

    assert hypotheses[0].posterior == boundary
    assert len(updates) == 1


def test_solve_updates_is_an_explicit_legacy_migration_wrapper():
    state = _belief_state([_hypothesis("H1", 0.5), _hypothesis("H2", 0.5)])
    migrated = migrate_belief_state_v0_1(state)

    with pytest.raises(ValueError, match="v0.1"):
        solve_updates("run_belief", "cycle_1", migrated, [])


def test_deep_solver_requires_native_v02_lifecycle_state():
    state = _belief_state([_hypothesis("H1", 0.5), _hypothesis("H2", 0.5)])

    with pytest.raises(ValueError, match="v0.2"):
        CoverageAwareBeliefSolver().solve(
            state,
            [],
            run_id="run_belief",
            cycle_id="cycle_1",
        )
