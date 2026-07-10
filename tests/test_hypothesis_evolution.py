from bayesprobe.hypothesis_evolution import (
    HypothesisEvolutionConfig,
    HypothesisEvolutionEngine,
    HypothesisEvolutionResult,
)
from bayesprobe.schemas import (
    BeliefState,
    BeliefUpdate,
    CycleRecord,
    CycleSignalShape,
    EvidenceEvent,
    EvidenceType,
    EvolutionOperation,
    Hypothesis,
    HypothesisStatus,
    LikelihoodBand,
    UpdateDirection,
)


def make_cycle(cycle_id: str = "cycle_1") -> CycleRecord:
    return CycleRecord(
        cycle_id=cycle_id,
        run_id="run_1",
        cycle_index=1,
        signal_shape=CycleSignalShape.PASSIVE_ONLY,
    )


def make_belief_state(h1_posterior: float = 0.5, h2_posterior: float = 0.5) -> BeliefState:
    return BeliefState(
        belief_state_id="bs_1",
        run_id="run_1",
        cycle_id="cycle_0",
        hypotheses=[
            Hypothesis(
                id="H1",
                statement="The claim is supported in its current scope.",
                scope="claim verification with broad scope",
                prior=0.5,
                posterior=h1_posterior,
                rivals=["H2"],
                falsifiers=["Independent counterevidence weakens H1."],
                predictions=["Supporting evidence should be found."],
            ),
            Hypothesis(
                id="H2",
                statement="The claim is refuted.",
                scope="claim verification",
                prior=0.5,
                posterior=h2_posterior,
                rivals=["H1"],
                falsifiers=["Independent support weakens H2."],
                predictions=["Refuting evidence should be found."],
            ),
        ],
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


def updated_hypotheses(h1_posterior: float) -> list[Hypothesis]:
    return [
        hypothesis.model_copy(update={"posterior": h1_posterior})
        if hypothesis.id == "H1"
        else hypothesis
        for hypothesis in make_belief_state().hypotheses
    ]


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
