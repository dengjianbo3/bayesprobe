import pytest

from bayesprobe.hypothesis_evolution import (
    HypothesisEvolutionConfig,
    HypothesisEvolutionEngine,
    HypothesisEvolutionResult,
)
from bayesprobe.schemas import (
    AnswerContract,
    BeliefState,
    BeliefUpdate,
    CycleRecord,
    CycleSignalShape,
    EvidenceEvent,
    EvidenceType,
    EvolutionOperation,
    FramedHypothesis,
    FramingMethod,
    Hypothesis,
    HypothesisFrame,
    HypothesisRelation,
    HypothesisStatus,
    LikelihoodBand,
    TaskFrame,
    TaskKind,
    UpdateDirection,
)


def make_cycle(cycle_id: str = "cycle_1") -> CycleRecord:
    return CycleRecord(
        cycle_id=cycle_id,
        run_id="run_1",
        cycle_index=1,
        signal_shape=CycleSignalShape.PASSIVE_ONLY,
    )


def make_belief_state(
    h1_posterior: float = 0.5,
    h2_posterior: float = 0.5,
    *,
    relation: HypothesisRelation = HypothesisRelation.EXCLUSIVE_EXHAUSTIVE,
) -> BeliefState:
    rivals = {"H1": ["H2"], "H2": ["H1"]}
    if relation == HypothesisRelation.INDEPENDENT:
        rivals = {"H1": [], "H2": []}
    hypotheses = [
        Hypothesis(
            id="H1",
            statement="The claim is supported in its current scope.",
            scope="claim verification with broad scope",
            prior=0.5,
            posterior=h1_posterior,
            rivals=rivals["H1"],
            falsifiers=["Independent counterevidence weakens H1."],
            predictions=["Supporting evidence should be found."],
        ),
        Hypothesis(
            id="H2",
            statement="The claim is refuted.",
            scope="claim verification",
            prior=0.5,
            posterior=h2_posterior,
            rivals=rivals["H2"],
            falsifiers=["Independent support weakens H2."],
            predictions=["Refuting evidence should be found."],
        ),
    ]
    return BeliefState(
        belief_state_id="bs_1",
        run_id="run_1",
        cycle_id="cycle_0",
        hypotheses=hypotheses,
        task_frame=TaskFrame(
            task_frame_id="run_1_task_frame",
            task_kind=TaskKind.CLAIM_VERIFICATION,
            normalized_question="Is the claim supported?",
            task_context="",
            answer_contract=AnswerContract(
                objective="Select the supported claim state.",
                required_sections=["answer", "uncertainty"],
                decision_form="claim_selection",
            ),
            hypothesis_frame=HypothesisFrame(
                frame_id="run_1_hypothesis_frame",
                relation=relation,
                hypotheses=[
                    FramedHypothesis(
                        id=item.id,
                        statement=item.statement,
                        type=item.type,
                        scope=item.scope,
                        initial_prior=item.prior,
                        falsifiers=list(item.falsifiers),
                        predictions=list(item.predictions),
                    )
                    for item in hypotheses
                ],
                rival_sets=rivals,
                coverage_statement="The claim is either supported or refuted.",
            ),
            framing_method=FramingMethod.RECORDED,
        ),
    )


def anomaly_event(event_id: str = "run_1_cycle_1_E1") -> EvidenceEvent:
    return EvidenceEvent(
        id=event_id,
        derived_from_signal="S_anomaly",
        target_hypotheses=["H1", "H2"],
        evidence_type=EvidenceType.ANOMALY,
        content="ANOMALY: Neither current hypothesis explains this signal.",
        reliability=0.8,
        independence=0.8,
        relevance=0.9,
        novelty=0.9,
        likelihoods={
            "H1": LikelihoodBand.MODERATELY_DISCONFIRMING,
            "H2": LikelihoodBand.MODERATELY_DISCONFIRMING,
        },
    )


def counter_event(event_id: str, *, independence: float = 0.8) -> EvidenceEvent:
    return EvidenceEvent(
        id=event_id,
        derived_from_signal=f"S_{event_id}",
        target_hypotheses=["H1"],
        evidence_type=EvidenceType.COUNTEREVIDENCE,
        content="REFUTES: Independent evidence weakens H1.",
        reliability=0.8,
        independence=independence,
        relevance=0.9,
        novelty=0.8,
        likelihoods={"H1": LikelihoodBand.MODERATELY_DISCONFIRMING},
    )


def update(
    *,
    evidence_id: str,
    prior: float,
    posterior: float,
    direction: UpdateDirection = UpdateDirection.WEAKENED,
) -> BeliefUpdate:
    return BeliefUpdate(
        update_id=f"U_{evidence_id}_H1",
        cycle_id="cycle_1",
        evidence_id=evidence_id,
        hypothesis_id="H1",
        prior=prior,
        posterior=posterior,
        direction=direction,
        reason="counterevidence is disconfirming for H1.",
    )


def updated_hypotheses(
    h1_posterior: float,
    *,
    relation: HypothesisRelation = HypothesisRelation.EXCLUSIVE_EXHAUSTIVE,
) -> list[Hypothesis]:
    return [
        hypothesis.model_copy(update={"posterior": h1_posterior})
        if hypothesis.id == "H1"
        else hypothesis
        for hypothesis in make_belief_state(relation=relation).hypotheses
    ]


def test_evolution_rejects_relation_less_direct_input():
    relation_less = make_belief_state().model_copy(update={"task_frame": None})

    with pytest.raises(
        ValueError,
        match="^belief state requires hypothesis relation metadata$",
    ):
        HypothesisEvolutionEngine().evolve(
            cycle=make_cycle(),
            previous_belief_state=relation_less,
            updated_hypotheses=relation_less.hypotheses,
            evidence_events=[],
            belief_updates=[],
        )


def test_anomaly_spawns_hypothesis_and_probe_candidate():
    result = HypothesisEvolutionEngine().evolve(
        cycle=make_cycle(),
        previous_belief_state=make_belief_state(),
        updated_hypotheses=updated_hypotheses(0.35),
        evidence_events=[anomaly_event()],
        belief_updates=[
            update(evidence_id="run_1_cycle_1_E1", prior=0.5, posterior=0.35),
        ],
    )

    assert isinstance(result, HypothesisEvolutionResult)
    assert result.evolutions[0].operation == EvolutionOperation.SPAWN
    assert result.evolutions[0].to_hypothesis == "H_run_1_cycle_1_E1_spawned"
    spawned = result.hypotheses_by_id()["H_run_1_cycle_1_E1_spawned"]
    assert spawned.created_by == "spawned"
    assert spawned.prior == 0.12
    assert spawned.posterior != spawned.prior
    assert spawned.rivals == ["H1", "H2"]
    assert spawned.why_existing_hypotheses_failed == result.evolutions[0].reason
    assert sum(
        hypothesis.posterior
        for hypothesis in result.hypotheses
        if hypothesis.status != HypothesisStatus.RETIRED
    ) == 1.0
    assert result.probe_candidates[0].source == "anomaly"
    assert result.probe_candidates[0].candidate_probe.target_hypotheses == [
        "H_run_1_cycle_1_E1_spawned"
    ]


def test_low_independence_duplicate_counterevidence_does_not_retire_hypothesis():
    result = HypothesisEvolutionEngine().evolve(
        cycle=make_cycle(),
        previous_belief_state=make_belief_state(),
        updated_hypotheses=updated_hypotheses(0.12),
        evidence_events=[
            counter_event("E_dup_1", independence=0.25),
            counter_event("E_dup_2", independence=0.25),
        ],
        belief_updates=[
            update(evidence_id="E_dup_1", prior=0.5, posterior=0.2),
            update(evidence_id="E_dup_2", prior=0.2, posterior=0.12),
        ],
    )

    h1 = result.hypotheses_by_id()["H1"]
    assert h1.status == HypothesisStatus.ACTIVE
    assert all(evolution.operation != EvolutionOperation.RETIRE for evolution in result.evolutions)


def test_independent_counterevidence_retires_stale_hypothesis():
    result = HypothesisEvolutionEngine().evolve(
        cycle=make_cycle(),
        previous_belief_state=make_belief_state(),
        updated_hypotheses=updated_hypotheses(0.12),
        evidence_events=[
            counter_event("E_independent_1", independence=0.8),
            counter_event("E_independent_2", independence=0.75),
        ],
        belief_updates=[
            update(evidence_id="E_independent_1", prior=0.5, posterior=0.22),
            update(evidence_id="E_independent_2", prior=0.22, posterior=0.12),
        ],
    )

    h1 = result.hypotheses_by_id()["H1"]
    retire = [e for e in result.evolutions if e.operation == EvolutionOperation.RETIRE][0]
    assert h1.status == HypothesisStatus.RETIRED
    assert retire.from_hypothesis == "H1"
    assert retire.audit_fields["independent_counterevidence_count"] == 2
    assert retire.audit_fields["counterevidence_event_ids"] == [
        "E_independent_1",
        "E_independent_2",
    ]


def test_counterevidence_reframes_scoped_top_hypothesis():
    result = HypothesisEvolutionEngine(
        config=HypothesisEvolutionConfig(retire_posterior_threshold=0.05)
    ).evolve(
        cycle=make_cycle(),
        previous_belief_state=make_belief_state(h1_posterior=0.7, h2_posterior=0.3),
        updated_hypotheses=updated_hypotheses(0.55),
        evidence_events=[counter_event("E_reframe", independence=0.8)],
        belief_updates=[
            update(evidence_id="E_reframe", prior=0.7, posterior=0.55),
        ],
    )

    reframe = [e for e in result.evolutions if e.operation == EvolutionOperation.REFRAME][0]
    reframed = result.hypotheses_by_id()["H_H1_cycle_1_reframed"]
    assert reframe.from_hypothesis == "H1"
    assert reframe.to_hypothesis == "H_H1_cycle_1_reframed"
    assert reframed.created_by == "reframed"
    assert reframed.rivals == ["H1", "H2"]
    assert "scope" in reframed.scope.lower()
    assert result.probe_candidates[0].source == "uncertainty"
    assert result.probe_candidates[0].candidate_probe.target_hypotheses == [
        "H1",
        "H_H1_cycle_1_reframed",
    ]


def test_discarded_anomaly_is_evolution_neutral():
    previous = make_belief_state(1.0 / 3.0, 1.0 / 3.0)
    discarded = anomaly_event().model_copy(update={"discard_reason": "inadmissible"})

    result = HypothesisEvolutionEngine().evolve(
        cycle=make_cycle(),
        previous_belief_state=previous,
        updated_hypotheses=previous.hypotheses,
        evidence_events=[discarded],
        belief_updates=[],
    )

    assert result.hypotheses == previous.hypotheses
    assert result.evolutions == []
    assert result.probe_candidates == []


def test_discarded_counterevidence_cannot_retire_or_reframe():
    previous = make_belief_state(h1_posterior=0.7, h2_posterior=0.3)
    discarded_events = [
        counter_event("E_discarded_1").model_copy(
            update={"discard_reason": "inadmissible"}
        ),
        counter_event("E_discarded_2").model_copy(
            update={"discard_reason": "inadmissible"}
        ),
    ]
    current = updated_hypotheses(0.12)

    result = HypothesisEvolutionEngine().evolve(
        cycle=make_cycle(),
        previous_belief_state=previous,
        updated_hypotheses=current,
        evidence_events=discarded_events,
        belief_updates=[
            update(evidence_id="E_discarded_1", prior=0.7, posterior=0.3),
            update(evidence_id="E_discarded_2", prior=0.3, posterior=0.12),
        ],
    )

    assert result.hypotheses == current
    assert result.evolutions == []
    assert result.probe_candidates == []


def test_seen_anomaly_id_is_ignored_by_direct_evolution_call():
    previous = make_belief_state().model_copy(
        update={"ledger_refs": {"evidence_events": ["E_seen_anomaly"]}}
    )

    result = HypothesisEvolutionEngine().evolve(
        cycle=make_cycle(),
        previous_belief_state=previous,
        updated_hypotheses=previous.hypotheses,
        evidence_events=[anomaly_event("E_seen_anomaly")],
        belief_updates=[],
    )

    assert result.hypotheses == previous.hypotheses
    assert result.evolutions == []


def test_same_cycle_duplicate_anomaly_id_spawns_once():
    previous = make_belief_state()
    duplicate = anomaly_event("E_duplicate_anomaly")

    result = HypothesisEvolutionEngine().evolve(
        cycle=make_cycle(),
        previous_belief_state=previous,
        updated_hypotheses=previous.hypotheses,
        evidence_events=[duplicate, duplicate],
        belief_updates=[],
    )

    assert [item.operation for item in result.evolutions] == [EvolutionOperation.SPAWN]
    assert [item.id for item in result.hypotheses].count(
        "H_E_duplicate_anomaly_spawned"
    ) == 1


def test_independent_spawn_has_no_dynamic_rivals():
    previous = make_belief_state(relation=HypothesisRelation.INDEPENDENT)

    result = HypothesisEvolutionEngine().evolve(
        cycle=make_cycle(),
        previous_belief_state=previous,
        updated_hypotheses=previous.hypotheses,
        evidence_events=[anomaly_event("E_independent_spawn")],
        belief_updates=[],
    )

    assert all(not item.rivals for item in result.hypotheses)


def test_exclusive_spawn_reconciles_reciprocal_dynamic_rivals():
    previous = make_belief_state()

    result = HypothesisEvolutionEngine().evolve(
        cycle=make_cycle(),
        previous_belief_state=previous,
        updated_hypotheses=previous.hypotheses,
        evidence_events=[anomaly_event("E_exclusive_spawn")],
        belief_updates=[],
    )

    ids = {item.id for item in result.hypotheses}
    assert {
        item.id: set(item.rivals)
        for item in result.hypotheses
    } == {item.id: ids - {item.id} for item in result.hypotheses}


def test_independent_reframe_has_no_dynamic_rivals():
    previous = make_belief_state(
        h1_posterior=0.7,
        h2_posterior=0.7,
        relation=HypothesisRelation.INDEPENDENT,
    )
    current = updated_hypotheses(0.55, relation=HypothesisRelation.INDEPENDENT)

    result = HypothesisEvolutionEngine(
        config=HypothesisEvolutionConfig(retire_posterior_threshold=0.05)
    ).evolve(
        cycle=make_cycle(),
        previous_belief_state=previous,
        updated_hypotheses=current,
        evidence_events=[counter_event("E_independent_reframe")],
        belief_updates=[
            update(evidence_id="E_independent_reframe", prior=0.7, posterior=0.55)
        ],
    )

    assert any(item.operation == EvolutionOperation.REFRAME for item in result.evolutions)
    assert all(not item.rivals for item in result.hypotheses)


def test_exclusive_reframe_reconciles_reciprocal_dynamic_rivals():
    previous = make_belief_state(h1_posterior=0.7, h2_posterior=0.3)
    current = updated_hypotheses(0.55)

    result = HypothesisEvolutionEngine(
        config=HypothesisEvolutionConfig(retire_posterior_threshold=0.05)
    ).evolve(
        cycle=make_cycle(),
        previous_belief_state=previous,
        updated_hypotheses=current,
        evidence_events=[counter_event("E_exclusive_reframe")],
        belief_updates=[
            update(evidence_id="E_exclusive_reframe", prior=0.7, posterior=0.55)
        ],
    )

    ids = {item.id for item in result.hypotheses}
    assert {
        item.id: set(item.rivals)
        for item in result.hypotheses
    } == {item.id: ids - {item.id} for item in result.hypotheses}
