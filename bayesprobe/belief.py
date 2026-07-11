from __future__ import annotations

import math
from typing import Any

from bayesprobe.schemas import (
    BeliefState,
    BeliefUpdate,
    EvidenceEvent,
    Hypothesis,
    HypothesisRelation,
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


def _normalize_exclusive_hypotheses(
    hypotheses: list[Hypothesis],
) -> list[Hypothesis]:
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


def normalize_hypotheses(
    hypotheses: list[Hypothesis],
    *,
    relation: HypothesisRelation,
) -> list[Hypothesis]:
    if relation == HypothesisRelation.INDEPENDENT:
        return list(hypotheses)
    return _normalize_exclusive_hypotheses(hypotheses)


def summarize_hypotheses(
    hypotheses: list[Hypothesis],
    *,
    relation: HypothesisRelation,
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
                "hypothesis_relation": relation.value,
                "belief_measure": (
                    "credence"
                    if relation == HypothesisRelation.INDEPENDENT
                    else "posterior_mass"
                ),
                "top_hypothesis": None,
                "runner_up_hypothesis": None,
                **(
                    {"top_credence": 0.0, "credence_gap": 0.0, "total_active_credence": 0.0}
                    if relation == HypothesisRelation.INDEPENDENT
                    else {
                        "top_posterior": 0.0,
                        "posterior_gap": 0.0,
                        "entropy": 0.0,
                        "total_active_posterior": 0.0,
                    }
                ),
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
    if relation == HypothesisRelation.INDEPENDENT:
        summary = {
            "hypothesis_relation": relation.value,
            "belief_measure": "credence",
            "top_hypothesis": top.id,
            "top_credence": top.posterior,
            "runner_up_hypothesis": runner_up.id if runner_up is not None else None,
            "credence_gap": round(posterior_gap, 6),
            "total_active_credence": round(sum(posteriors), 6),
        }
        uncertainty = (
            f"{top.id} has the highest current credence, but independent hypotheses may coexist; "
            "ranking does not by itself select the answer."
        )
        return summary, uncertainty

    summary = {
        "hypothesis_relation": relation.value,
        "belief_measure": "posterior_mass",
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


def _logit(probability: float) -> float:
    bounded = min(max(probability, _MIN_PROBABILITY), 1.0 - _MIN_PROBABILITY)
    return math.log(bounded / (1.0 - bounded))


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _independent_event_posterior(
    prior: float,
    band: LikelihoodBand,
    weight: float,
    complexity_delta: float,
    ad_hoc_delta: float,
) -> float:
    score = (
        _logit(prior)
        + math.log(likelihood_band_to_lr(band)) * weight
        - complexity_delta
        - ad_hoc_delta
    )
    return round(_sigmoid(score), _DISTRIBUTION_PRECISION)


def _penalty_deltas(hypothesis: Hypothesis) -> tuple[float, float]:
    return (
        max(
            hypothesis.complexity_penalty
            - hypothesis.applied_complexity_penalty,
            0.0,
        ),
        max(
            hypothesis.ad_hoc_penalty - hypothesis.applied_ad_hoc_penalty,
            0.0,
        ),
    )


def solve_updates(
    run_id: str,
    cycle_id: str,
    belief_state: BeliefState,
    events: list[EvidenceEvent],
) -> tuple[list[Hypothesis], list[BeliefUpdate]]:
    if belief_state.task_frame is None:
        raise ValueError("belief state requires hypothesis relation metadata")
    relation = belief_state.task_frame.hypothesis_frame.relation
    working_hypotheses = normalize_hypotheses(
        belief_state.hypotheses,
        relation=relation,
    )
    updates: list[BeliefUpdate] = []

    for event_index, event in enumerate(events, start=1):
        if event.discard_reason is not None:
            continue
        active_by_id = {
            hypothesis.id: hypothesis
            for hypothesis in working_hypotheses
            if _participates_in_distribution(hypothesis)
        }
        participants = (
            list(active_by_id.values())
            if relation == HypothesisRelation.EXCLUSIVE_EXHAUSTIVE
            else [
                active_by_id[hypothesis_id]
                for hypothesis_id in dict.fromkeys(event.target_hypotheses)
                if hypothesis_id in active_by_id
            ]
        )
        weight = event.reliability * event.independence * event.relevance * event.novelty
        penalty_deltas = {
            hypothesis.id: _penalty_deltas(hypothesis)
            for hypothesis in participants
        }
        if relation == HypothesisRelation.EXCLUSIVE_EXHAUSTIVE:
            event_posteriors = _round_distribution(
                _softmax(
                    {
                        hypothesis.id: (
                            math.log(max(hypothesis.posterior, _MIN_PROBABILITY))
                            + math.log(
                                likelihood_band_to_lr(
                                    event.likelihoods.get(
                                        hypothesis.id,
                                        LikelihoodBand.NEUTRAL,
                                    )
                                )
                            )
                            * weight
                            - penalty_deltas[hypothesis.id][0]
                            - penalty_deltas[hypothesis.id][1]
                        )
                        for hypothesis in participants
                    }
                )
            )
        else:
            event_posteriors = {
                hypothesis.id: _independent_event_posterior(
                    hypothesis.posterior,
                    event.likelihoods.get(hypothesis.id, LikelihoodBand.NEUTRAL),
                    weight,
                    *penalty_deltas[hypothesis.id],
                )
                for hypothesis in participants
            }
        replacements: dict[str, Hypothesis] = {}
        for hypothesis in participants:
            hypothesis_id = hypothesis.id
            band = event.likelihoods.get(hypothesis_id, LikelihoodBand.NEUTRAL)
            prior = hypothesis.posterior
            posterior = event_posteriors[hypothesis_id]
            complexity_delta, ad_hoc_delta = penalty_deltas[hypothesis_id]
            replacements[hypothesis_id] = hypothesis.model_copy(
                update={
                    "posterior": posterior,
                    "applied_complexity_penalty": hypothesis.complexity_penalty,
                    "applied_ad_hoc_penalty": hypothesis.ad_hoc_penalty,
                }
            )
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
                        "complexity_penalty_delta": round(complexity_delta, 4),
                        "ad_hoc_penalty_delta": round(ad_hoc_delta, 4),
                    },
                )
            )
        working_hypotheses = [
            replacements.get(hypothesis.id, hypothesis)
            for hypothesis in working_hypotheses
        ]

    return working_hypotheses, updates


__all__ = [
    "likelihood_band_to_lr",
    "normalize_hypotheses",
    "solve_updates",
    "summarize_hypotheses",
]
