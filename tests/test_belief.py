import pytest

from bayesprobe.belief import solve_updates
from bayesprobe.schemas import (
    BeliefState,
    EvidenceEvent,
    EvidenceType,
    Hypothesis,
    LikelihoodBand,
)


def _hypothesis(
    hypothesis_id: str,
    posterior: float,
    *,
    complexity_penalty: float = 0.0,
    ad_hoc_penalty: float = 0.0,
) -> Hypothesis:
    return Hypothesis(
        id=hypothesis_id,
        statement=f"{hypothesis_id} is the correct rival.",
        scope="Exclusive categorical fixture.",
        prior=posterior,
        posterior=posterior,
        complexity_penalty=complexity_penalty,
        ad_hoc_penalty=ad_hoc_penalty,
    )


def _belief_state(hypotheses: list[Hypothesis]) -> BeliefState:
    return BeliefState(
        belief_state_id="run_belief_bs_0",
        run_id="run_belief",
        cycle_id="cycle_0",
        hypotheses=hypotheses,
    )


def _event(
    likelihoods: dict[str, LikelihoodBand],
    *,
    target_hypotheses: list[str] | None = None,
) -> EvidenceEvent:
    targets = target_hypotheses or list(likelihoods)
    return EvidenceEvent(
        id="E_belief",
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
