from __future__ import annotations

import math

from bayesprobe.schemas import (
    BeliefState,
    BeliefUpdate,
    EvidenceEvent,
    Hypothesis,
    LikelihoodBand,
    UpdateDirection,
)


LR_BY_BAND: dict[LikelihoodBand, float] = {
    LikelihoodBand.STRONGLY_DISCONFIRMING: 0.1,
    LikelihoodBand.MODERATELY_DISCONFIRMING: 0.3,
    LikelihoodBand.WEAKLY_DISCONFIRMING: 0.7,
    LikelihoodBand.NEUTRAL: 1.0,
    LikelihoodBand.WEAKLY_CONFIRMING: 1.5,
    LikelihoodBand.MODERATELY_CONFIRMING: 3.0,
    LikelihoodBand.STRONGLY_CONFIRMING: 10.0,
}


def likelihood_band_to_lr(band: LikelihoodBand) -> float:
    return LR_BY_BAND[band]


def _logit(probability: float) -> float:
    clipped = min(max(probability, 0.001), 0.999)
    return math.log(clipped / (1 - clipped))


def _sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-value))


def _direction(prior: float, posterior: float) -> UpdateDirection:
    if posterior > prior + 0.01:
        return UpdateDirection.STRENGTHENED
    if posterior < prior - 0.01:
        return UpdateDirection.WEAKENED
    return UpdateDirection.NEUTRAL


def solve_updates(
    run_id: str,
    cycle_id: str,
    belief_state: BeliefState,
    events: list[EvidenceEvent],
) -> tuple[list[Hypothesis], list[BeliefUpdate]]:
    hypotheses = belief_state.hypotheses_by_id()
    current_posteriors = {hypothesis.id: hypothesis.posterior for hypothesis in belief_state.hypotheses}
    updates: list[BeliefUpdate] = []

    for event_index, event in enumerate(events, start=1):
        if event.discard_reason is not None:
            continue
        for hypothesis_id, band in event.likelihoods.items():
            if hypothesis_id not in hypotheses:
                continue
            prior = current_posteriors[hypothesis_id]
            weight = event.reliability * event.independence * event.relevance * event.novelty
            weighted_log_lr = math.log(likelihood_band_to_lr(band)) * weight
            posterior = _sigmoid(_logit(prior) + weighted_log_lr)
            current_posteriors[hypothesis_id] = posterior
            updates.append(
                BeliefUpdate(
                    update_id=f"{run_id}_{cycle_id}_U{event_index}_{hypothesis_id}",
                    cycle_id=cycle_id,
                    evidence_id=event.id,
                    hypothesis_id=hypothesis_id,
                    prior=round(prior, 4),
                    posterior=round(posterior, 4),
                    direction=_direction(prior, posterior),
                    reason=f"{event.evidence_type.value} is {band.value} for {hypothesis_id}.",
                    sensitivity={
                        "weight": round(weight, 4),
                        "likelihood_band": band.value,
                    },
                )
            )

    updated_hypotheses = [
        hypothesis.model_copy(update={"posterior": round(current_posteriors[hypothesis.id], 4)})
        for hypothesis in belief_state.hypotheses
    ]
    return updated_hypotheses, updates
