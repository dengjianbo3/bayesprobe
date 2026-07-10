from __future__ import annotations

import math
from typing import Any

from bayesprobe.schemas import (
    BeliefState,
    BeliefUpdate,
    EvidenceEvent,
    Hypothesis,
    HypothesisStatus,
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

_MIN_PROBABILITY = 1e-12
_DISTRIBUTION_PRECISION = 4
_NON_PARTICIPATING_STATUSES = {
    HypothesisStatus.RETIRED,
    HypothesisStatus.ARCHIVED,
}


def likelihood_band_to_lr(band: LikelihoodBand) -> float:
    return LR_BY_BAND[band]


def _direction(prior: float, posterior: float) -> UpdateDirection:
    if posterior > prior + 0.01:
        return UpdateDirection.STRENGTHENED
    if posterior < prior - 0.01:
        return UpdateDirection.WEAKENED
    return UpdateDirection.NEUTRAL


def _participates_in_distribution(hypothesis: Hypothesis) -> bool:
    return hypothesis.status not in _NON_PARTICIPATING_STATUSES


def _softmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    maximum = max(scores.values())
    exponentials = {
        hypothesis_id: math.exp(score - maximum)
        for hypothesis_id, score in scores.items()
    }
    total = sum(exponentials.values())
    return {
        hypothesis_id: value / total
        for hypothesis_id, value in exponentials.items()
    }


def _round_distribution(distribution: dict[str, float]) -> dict[str, float]:
    if not distribution:
        return {}
    rounded = {
        hypothesis_id: round(value, _DISTRIBUTION_PRECISION)
        for hypothesis_id, value in distribution.items()
    }
    residual = round(1.0 - sum(rounded.values()), _DISTRIBUTION_PRECISION)
    anchor = max(distribution, key=lambda hypothesis_id: distribution[hypothesis_id])
    rounded[anchor] = round(
        rounded[anchor] + residual,
        _DISTRIBUTION_PRECISION,
    )
    return rounded


def normalize_hypotheses(hypotheses: list[Hypothesis]) -> list[Hypothesis]:
    participants = [
        hypothesis
        for hypothesis in hypotheses
        if _participates_in_distribution(hypothesis)
    ]
    if not participants:
        return list(hypotheses)
    total = sum(
        max(hypothesis.posterior, _MIN_PROBABILITY)
        for hypothesis in participants
    )
    distribution = _round_distribution(
        {
            hypothesis.id: max(hypothesis.posterior, _MIN_PROBABILITY) / total
            for hypothesis in participants
        }
    )
    return [
        hypothesis.model_copy(update={"posterior": distribution[hypothesis.id]})
        if hypothesis.id in distribution
        else hypothesis
        for hypothesis in hypotheses
    ]


def summarize_hypotheses(
    hypotheses: list[Hypothesis],
) -> tuple[dict[str, Any], str]:
    participants = sorted(
        (
            hypothesis
            for hypothesis in hypotheses
            if _participates_in_distribution(hypothesis)
        ),
        key=lambda hypothesis: (-hypothesis.posterior, hypothesis.id),
    )
    if not participants:
        return (
            {
                "top_hypothesis": None,
                "top_posterior": 0.0,
                "runner_up_hypothesis": None,
                "posterior_gap": 0.0,
                "entropy": 0.0,
                "total_active_posterior": 0.0,
            },
            "No active hypotheses remain after the current cycle.",
        )

    top = participants[0]
    runner_up = participants[1] if len(participants) > 1 else None
    posteriors = [hypothesis.posterior for hypothesis in participants]
    posterior_gap = (
        top.posterior - runner_up.posterior
        if runner_up is not None
        else top.posterior
    )
    summary = {
        "top_hypothesis": top.id,
        "top_posterior": top.posterior,
        "runner_up_hypothesis": runner_up.id if runner_up is not None else None,
        "posterior_gap": round(posterior_gap, 6),
        "entropy": round(
            -sum(posterior * math.log(posterior) for posterior in posteriors if posterior > 0),
            6,
        ),
        "total_active_posterior": round(sum(posteriors), 6),
    }
    if runner_up is None:
        uncertainty = (
            f"{top.id} is the only active hypothesis at posterior {top.posterior:.3f}; "
            "independent falsification is still required."
        )
    else:
        uncertainty = (
            f"The posterior gap between {top.id} and {runner_up.id} is "
            f"{posterior_gap:.3f}; further discriminative evidence may change the ranking."
        )
    return summary, uncertainty


def solve_updates(
    run_id: str,
    cycle_id: str,
    belief_state: BeliefState,
    events: list[EvidenceEvent],
) -> tuple[list[Hypothesis], list[BeliefUpdate]]:
    normalized_hypotheses = normalize_hypotheses(belief_state.hypotheses)
    current_posteriors = {
        hypothesis.id: hypothesis.posterior
        for hypothesis in normalized_hypotheses
    }
    updates: list[BeliefUpdate] = []

    for event_index, event in enumerate(events, start=1):
        if event.discard_reason is not None:
            continue
        participants = [
            hypothesis
            for hypothesis in normalized_hypotheses
            if _participates_in_distribution(hypothesis)
        ]
        weight = event.reliability * event.independence * event.relevance * event.novelty
        scores: dict[str, float] = {}
        for hypothesis in participants:
            band = event.likelihoods.get(hypothesis.id, LikelihoodBand.NEUTRAL)
            scores[hypothesis.id] = (
                math.log(max(current_posteriors[hypothesis.id], _MIN_PROBABILITY))
                + math.log(likelihood_band_to_lr(band)) * weight
                - hypothesis.complexity_penalty
                - hypothesis.ad_hoc_penalty
            )
        event_posteriors = _round_distribution(_softmax(scores))
        for hypothesis in participants:
            hypothesis_id = hypothesis.id
            band = event.likelihoods.get(hypothesis_id, LikelihoodBand.NEUTRAL)
            prior = current_posteriors[hypothesis_id]
            posterior = event_posteriors[hypothesis_id]
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
                        "complexity_penalty": hypothesis.complexity_penalty,
                        "ad_hoc_penalty": hypothesis.ad_hoc_penalty,
                    },
                )
            )

    updated_hypotheses = [
        hypothesis.model_copy(update={"posterior": round(current_posteriors[hypothesis.id], 4)})
        for hypothesis in belief_state.hypotheses
    ]
    return updated_hypotheses, updates


__all__ = [
    "likelihood_band_to_lr",
    "normalize_hypotheses",
    "solve_updates",
    "summarize_hypotheses",
]
