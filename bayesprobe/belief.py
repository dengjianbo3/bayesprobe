from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from bayesprobe.kernel_config import OpenCoveragePolicy

from bayesprobe.schemas import (
    BeliefState,
    BeliefUpdate,
    EvidenceEvent,
    FrameMassUpdate,
    FrameState,
    Hypothesis,
    HypothesisCompetition,
    HypothesisCoverage,
    HypothesisEvolution,
    HypothesisRelation,
    HypothesisStatus,
    LikelihoodBand,
    EvolutionOperation,
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
_REPLAY_DISCARD_REASON = "duplicate evidence event id"
_UNRESOLVED_SLOT = "__unresolved_alternative_mass__"


@dataclass(frozen=True)
class BeliefSolveResult:
    hypotheses: list[Hypothesis]
    frame_state: FrameState
    belief_updates: list[BeliefUpdate]
    frame_mass_updates: list[FrameMassUpdate]


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


def _round_distribution(
    distribution: dict[str, float],
    *,
    minimums: dict[str, float] | None = None,
) -> dict[str, float]:
    if not distribution:
        return {}
    minimums = minimums or {}
    scale = 10**_DISTRIBUTION_PRECISION
    minimum_units = {
        hypothesis_id: math.ceil(minimum * scale - 1e-12)
        for hypothesis_id, minimum in minimums.items()
    }
    rounded = {
        hypothesis_id: round(value, _DISTRIBUTION_PRECISION)
        for hypothesis_id, value in distribution.items()
    }
    for hypothesis_id, units in minimum_units.items():
        rounded[hypothesis_id] = max(rounded[hypothesis_id], units / scale)
    residual = round(1.0 - sum(rounded.values()), _DISTRIBUTION_PRECISION)
    anchors = [
        hypothesis_id
        for hypothesis_id in distribution
        if rounded[hypothesis_id] + residual
        >= minimum_units.get(hypothesis_id, 0) / scale
    ]
    if not anchors:
        raise ValueError("rounded distribution cannot satisfy configured minimums")
    anchor = max(anchors, key=lambda hypothesis_id: distribution[hypothesis_id])
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
    relation: HypothesisRelation | None = None,
    frame_state: FrameState | None = None,
) -> tuple[dict[str, Any], str]:
    if relation is None and frame_state is None:
        raise ValueError("summary requires relation or frame state")
    independent = (
        frame_state.competition == HypothesisCompetition.INDEPENDENT
        if frame_state is not None
        else relation == HypothesisRelation.INDEPENDENT
    )
    relation_label = _summary_relation_label(
        relation=relation,
        frame_state=frame_state,
        independent=independent,
    )
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
                "hypothesis_relation": relation_label,
                "belief_measure": "credence" if independent else "posterior_mass",
                "top_hypothesis": None,
                "runner_up_hypothesis": None,
                **(
                    {"top_credence": 0.0, "credence_gap": 0.0, "total_active_credence": 0.0}
                    if independent
                    else {
                        "top_posterior": 0.0,
                        "posterior_gap": 0.0,
                        "entropy": 0.0,
                        "total_active_posterior": 0.0,
                    }
                ),
                **_frame_summary_fields(frame_state, named_active_mass=0.0),
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
    if independent:
        summary = {
            "hypothesis_relation": relation_label,
            "belief_measure": "credence",
            "top_hypothesis": top.id,
            "top_credence": top.posterior,
            "runner_up_hypothesis": runner_up.id if runner_up is not None else None,
            "credence_gap": round(posterior_gap, 6),
            "total_active_credence": round(sum(posteriors), 6),
            **_frame_summary_fields(frame_state, named_active_mass=None),
        }
        uncertainty = (
            f"{top.id} has the highest current credence, but independent hypotheses may coexist; "
            "ranking does not by itself select the answer."
        )
        return summary, uncertainty

    summary = {
        "hypothesis_relation": relation_label,
        "belief_measure": "posterior_mass",
        "top_hypothesis": top.id,
        "top_posterior": top.posterior,
        "runner_up_hypothesis": runner_up.id if runner_up is not None else None,
        "posterior_gap": round(posterior_gap, 6),
        "entropy": round(-sum(_entropy_terms(posteriors, frame_state)), 6),
        "total_active_posterior": round(sum(posteriors), 6),
        **_frame_summary_fields(
            frame_state,
            named_active_mass=round(sum(posteriors), 6),
        ),
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


def _summary_relation_label(
    *,
    relation: HypothesisRelation | None,
    frame_state: FrameState | None,
    independent: bool,
) -> str:
    if relation is not None:
        return relation.value
    if independent:
        return "independent"
    if frame_state.coverage == HypothesisCoverage.OPEN:
        return "exclusive_open"
    return "exclusive_exhaustive"


def _frame_summary_fields(
    frame_state: FrameState | None,
    *,
    named_active_mass: float | None,
) -> dict[str, Any]:
    if frame_state is None:
        return {}
    return {
        "hypothesis_competition": frame_state.competition.value,
        "hypothesis_coverage": frame_state.coverage.value,
        "named_active_mass": named_active_mass,
        "unresolved_alternative_mass": frame_state.unresolved_alternative_mass,
        "frame_adequacy": frame_state.adequacy_status.value,
    }


def _entropy_terms(
    posteriors: list[float],
    frame_state: FrameState | None,
):
    values = list(posteriors)
    if (
        frame_state is not None
        and frame_state.unresolved_alternative_mass is not None
    ):
        values.append(frame_state.unresolved_alternative_mass)
    return (posterior * math.log(posterior) for posterior in values if posterior > 0)


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


def mark_replayed_evidence_events(
    belief_state: BeliefState,
    events: list[EvidenceEvent],
) -> list[EvidenceEvent]:
    seen_ids = set(belief_state.ledger_refs.get("evidence_events", []))
    marked: list[EvidenceEvent] = []
    for event in events:
        if event.id in seen_ids and event.discard_reason is None:
            event = event.model_copy(update={"discard_reason": _REPLAY_DISCARD_REASON})
        marked.append(event)
        seen_ids.add(event.id)
    return marked


def solve_updates(
    run_id: str,
    cycle_id: str,
    belief_state: BeliefState,
    events: list[EvidenceEvent],
) -> tuple[list[Hypothesis], list[BeliefUpdate]]:
    """Migrate and solve a legacy v0.1 state; native callers use the solver."""
    if belief_state.schema_version != "v0.1":
        raise ValueError("solve_updates accepts only v0.1 belief states")
    if belief_state.task_frame is None:
        raise ValueError("belief state requires hypothesis relation metadata")
    from bayesprobe.migrations import migrate_belief_state_v0_1

    result = CoverageAwareBeliefSolver().solve(
        migrate_belief_state_v0_1(belief_state),
        events,
        run_id=run_id,
        cycle_id=cycle_id,
    )
    return result.hypotheses, result.belief_updates


class CoverageAwareBeliefSolver:
    def __init__(
        self,
        *,
        open_coverage_policy: OpenCoveragePolicy | None = None,
    ) -> None:
        self.open_coverage_policy = open_coverage_policy or OpenCoveragePolicy()

    def solve(
        self,
        belief_state: BeliefState,
        events: list[EvidenceEvent],
        *,
        run_id: str,
        cycle_id: str,
    ) -> BeliefSolveResult:
        if belief_state.schema_version != "v0.2":
            raise ValueError("coverage-aware solving requires a v0.2 belief state")
        if belief_state.task_frame is None or belief_state.frame_state is None:
            raise ValueError("v0.2 belief state requires task and frame state")
        frame_state = belief_state.frame_state
        events = mark_replayed_evidence_events(belief_state, events)
        working_hypotheses = list(belief_state.hypotheses)
        active_hypotheses = [
            hypothesis
            for hypothesis in working_hypotheses
            if _participates_in_distribution(hypothesis)
        ]
        active_ids = [hypothesis.id for hypothesis in active_hypotheses]
        unresolved_mass = frame_state.unresolved_alternative_mass
        if (
            frame_state.competition == HypothesisCompetition.EXCLUSIVE
            and frame_state.coverage == HypothesisCoverage.OPEN
        ):
            if unresolved_mass is None:
                raise ValueError("exclusive-open frame requires unresolved mass")
            retired_ids = set(frame_state.active_hypothesis_ids).difference(active_ids)
            if retired_ids:
                raise ValueError("retirement transfer requires audit context")
        working_frame_state = frame_state.model_copy(
            update={
                "active_hypothesis_ids": active_ids,
                "unresolved_alternative_mass": unresolved_mass,
            }
        )
        belief_updates: list[BeliefUpdate] = []
        frame_mass_updates: list[FrameMassUpdate] = []

        for event_index, event in enumerate(events, start=1):
            if event.discard_reason is not None:
                continue
            if working_frame_state.competition == HypothesisCompetition.INDEPENDENT:
                working_hypotheses, event_updates = self._solve_independent_event(
                    working_hypotheses,
                    event,
                    run_id=run_id,
                    cycle_id=cycle_id,
                    event_index=event_index,
                )
                belief_updates.extend(event_updates)
                continue
            (
                working_hypotheses,
                working_frame_state,
                event_updates,
                frame_mass_update,
            ) = self._solve_exclusive_event(
                working_hypotheses,
                working_frame_state,
                event,
                run_id=run_id,
                cycle_id=cycle_id,
                event_index=event_index,
            )
            belief_updates.extend(event_updates)
            if frame_mass_update is not None:
                frame_mass_updates.append(frame_mass_update)

        return BeliefSolveResult(
            hypotheses=working_hypotheses,
            frame_state=working_frame_state,
            belief_updates=belief_updates,
            frame_mass_updates=frame_mass_updates,
        )

    def reconcile_retirements(
        self,
        solve_result: BeliefSolveResult,
        *,
        evolved_hypotheses: list[Hypothesis],
        evolutions: list[HypothesisEvolution],
        events: list[EvidenceEvent],
        run_id: str,
        cycle_id: str,
    ) -> BeliefSolveResult:
        frame_state = solve_result.frame_state
        if not (
            frame_state.competition == HypothesisCompetition.EXCLUSIVE
            and frame_state.coverage == HypothesisCoverage.OPEN
        ):
            raise ValueError("retirement reconciliation requires an exclusive-open frame")
        unresolved_mass = frame_state.unresolved_alternative_mass
        if unresolved_mass is None:
            raise ValueError("exclusive-open frame requires unresolved mass")

        solved_by_id = {
            hypothesis.id: hypothesis for hypothesis in solve_result.hypotheses
        }
        events_by_id = {
            event.id: event for event in events if event.discard_reason is None
        }
        retirement_evolutions = {
            evolution.from_hypothesis: evolution
            for evolution in evolutions
            if evolution.operation == EvolutionOperation.RETIRE
            and evolution.from_hypothesis is not None
        }
        newly_retired = [
            hypothesis
            for hypothesis in evolved_hypotheses
            if hypothesis.status == HypothesisStatus.RETIRED
            and hypothesis.id in frame_state.active_hypothesis_ids
        ]
        retirement_updates: list[FrameMassUpdate] = []
        for hypothesis in newly_retired:
            solved = solved_by_id.get(hypothesis.id)
            evolution = retirement_evolutions.get(hypothesis.id)
            if solved is None or evolution is None:
                raise ValueError(
                    f"retirement transfer for {hypothesis.id} requires an evolution audit"
                )
            trigger_event = next(
                (
                    events_by_id[event_id]
                    for event_id in reversed(evolution.triggered_by)
                    if event_id in events_by_id
                ),
                None,
            )
            if trigger_event is None:
                raise ValueError(
                    f"retirement transfer for {hypothesis.id} requires triggering evidence"
                )
            prior = unresolved_mass
            unresolved_mass = prior + solved.posterior
            root_context = (
                f" with derivation root {trigger_event.derivation_root_id}"
                if trigger_event.derivation_root_id is not None
                else ""
            )
            retirement_updates.append(
                FrameMassUpdate(
                    update_id=(
                        f"{run_id}_{cycle_id}_FM_retire_{hypothesis.id}"
                    ),
                    cycle_id=cycle_id,
                    evidence_id=trigger_event.id,
                    prior=prior,
                    posterior=unresolved_mass,
                    direction=_direction(prior, unresolved_mass),
                    reason=(
                        f"Retiring {hypothesis.id} transfers its posterior mass to "
                        f"unresolved alternatives after {trigger_event.id}{root_context}."
                    ),
                )
            )

        active_ids = [
            hypothesis.id
            for hypothesis in evolved_hypotheses
            if _participates_in_distribution(hypothesis)
        ]
        return BeliefSolveResult(
            hypotheses=evolved_hypotheses,
            frame_state=frame_state.model_copy(
                update={
                    "active_hypothesis_ids": active_ids,
                    "unresolved_alternative_mass": unresolved_mass,
                }
            ),
            belief_updates=solve_result.belief_updates,
            frame_mass_updates=[
                *solve_result.frame_mass_updates,
                *retirement_updates,
            ],
        )

    def _solve_exclusive_event(
        self,
        hypotheses: list[Hypothesis],
        frame_state: FrameState,
        event: EvidenceEvent,
        *,
        run_id: str,
        cycle_id: str,
        event_index: int,
    ) -> tuple[list[Hypothesis], FrameState, list[BeliefUpdate], FrameMassUpdate | None]:
        participants = [
            hypothesis
            for hypothesis in hypotheses
            if _participates_in_distribution(hypothesis)
        ]
        if not participants:
            return hypotheses, frame_state, [], None
        weight = _effective_update_weight(event)
        penalty_deltas = {
            hypothesis.id: _penalty_deltas(hypothesis)
            for hypothesis in participants
        }
        scores = {
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
        unresolved_prior = frame_state.unresolved_alternative_mass
        if frame_state.coverage == HypothesisCoverage.OPEN:
            if unresolved_prior is None:
                raise ValueError("exclusive-open frame requires unresolved mass")
            unresolved_band = event.unresolved_likelihood or LikelihoodBand.NEUTRAL
            scores[_UNRESOLVED_SLOT] = (
                math.log(max(unresolved_prior, _MIN_PROBABILITY))
                + math.log(likelihood_band_to_lr(unresolved_band)) * weight
            )
        distribution = _softmax(scores)
        if frame_state.coverage == HypothesisCoverage.OPEN:
            distribution = _preserve_unresolved_reserve(
                distribution,
                reserve=self.open_coverage_policy.minimum_unresolved_reserve,
            )
        event_posteriors = _round_distribution(
            distribution,
            minimums=(
                {
                    _UNRESOLVED_SLOT:
                    self.open_coverage_policy.minimum_unresolved_reserve,
                }
                if frame_state.coverage == HypothesisCoverage.OPEN
                else None
            ),
        )
        replacements: dict[str, Hypothesis] = {}
        updates: list[BeliefUpdate] = []
        for hypothesis in participants:
            prior = hypothesis.posterior
            posterior = event_posteriors[hypothesis.id]
            band = event.likelihoods.get(hypothesis.id, LikelihoodBand.NEUTRAL)
            complexity_delta, ad_hoc_delta = penalty_deltas[hypothesis.id]
            replacements[hypothesis.id] = _updated_hypothesis(hypothesis, posterior)
            updates.append(
                _belief_update(
                    hypothesis=hypothesis,
                    prior=prior,
                    posterior=posterior,
                    band=band,
                    event=event,
                    weight=weight,
                    complexity_delta=complexity_delta,
                    ad_hoc_delta=ad_hoc_delta,
                    update_id=(
                        f"{run_id}_{cycle_id}_U{event_index}_{hypothesis.id}"
                    ),
                    cycle_id=cycle_id,
                )
            )
        updated_hypotheses = [
            replacements.get(hypothesis.id, hypothesis)
            for hypothesis in hypotheses
        ]
        frame_mass_update = None
        if frame_state.coverage == HypothesisCoverage.OPEN:
            unresolved_posterior = event_posteriors[_UNRESOLVED_SLOT]
            unresolved_band = event.unresolved_likelihood or LikelihoodBand.NEUTRAL
            frame_state = frame_state.model_copy(
                update={"unresolved_alternative_mass": unresolved_posterior}
            )
            rounded_unresolved_prior = round(
                unresolved_prior,
                _DISTRIBUTION_PRECISION,
            )
            if unresolved_posterior != rounded_unresolved_prior:
                frame_mass_update = FrameMassUpdate(
                    update_id=f"{run_id}_{cycle_id}_FM{event_index}",
                    cycle_id=cycle_id,
                    evidence_id=event.id,
                    prior=rounded_unresolved_prior,
                    posterior=unresolved_posterior,
                    direction=_direction(unresolved_prior, unresolved_posterior),
                    reason=(
                        f"{event.evidence_type.value} is {unresolved_band.value} for "
                        "unresolved alternative mass."
                    ),
                )
        return updated_hypotheses, frame_state, updates, frame_mass_update

    @staticmethod
    def _solve_independent_event(
        hypotheses: list[Hypothesis],
        event: EvidenceEvent,
        *,
        run_id: str,
        cycle_id: str,
        event_index: int,
    ) -> tuple[list[Hypothesis], list[BeliefUpdate]]:
        active_by_id = {
            hypothesis.id: hypothesis
            for hypothesis in hypotheses
            if _participates_in_distribution(hypothesis)
        }
        participants = [
            active_by_id[hypothesis_id]
            for hypothesis_id in dict.fromkeys(event.target_hypotheses)
            if hypothesis_id in active_by_id
        ]
        weight = _effective_update_weight(event)
        replacements: dict[str, Hypothesis] = {}
        updates: list[BeliefUpdate] = []
        for hypothesis in participants:
            band = event.likelihoods.get(hypothesis.id, LikelihoodBand.NEUTRAL)
            complexity_delta, ad_hoc_delta = _penalty_deltas(hypothesis)
            posterior = _independent_event_posterior(
                hypothesis.posterior,
                band,
                weight,
                complexity_delta,
                ad_hoc_delta,
            )
            replacements[hypothesis.id] = _updated_hypothesis(hypothesis, posterior)
            updates.append(
                _belief_update(
                    hypothesis=hypothesis,
                    prior=hypothesis.posterior,
                    posterior=posterior,
                    band=band,
                    event=event,
                    weight=weight,
                    complexity_delta=complexity_delta,
                    ad_hoc_delta=ad_hoc_delta,
                    update_id=(
                        f"{run_id}_{cycle_id}_U{event_index}_{hypothesis.id}"
                    ),
                    cycle_id=cycle_id,
                )
            )
        return (
            [replacements.get(hypothesis.id, hypothesis) for hypothesis in hypotheses],
            updates,
        )


def _effective_update_weight(event: EvidenceEvent) -> float:
    if event.schema_version == "v0.2":
        if event.effective_update_weight is None:
            raise ValueError("v0.2 evidence requires effective_update_weight")
        return event.effective_update_weight
    if event.effective_update_weight is not None:
        return event.effective_update_weight
    return event.reliability * event.independence * event.relevance * event.novelty


def _preserve_unresolved_reserve(
    distribution: dict[str, float],
    *,
    reserve: float,
) -> dict[str, float]:
    unresolved = distribution[_UNRESOLVED_SLOT]
    if unresolved >= reserve:
        return distribution
    named_total = 1.0 - unresolved
    if named_total <= 0:
        return {_UNRESOLVED_SLOT: 1.0}
    named_scale = (1.0 - reserve) / named_total
    return {
        key: reserve if key == _UNRESOLVED_SLOT else value * named_scale
        for key, value in distribution.items()
    }


def _updated_hypothesis(hypothesis: Hypothesis, posterior: float) -> Hypothesis:
    return hypothesis.model_copy(
        update={
            "posterior": posterior,
            "applied_complexity_penalty": max(
                hypothesis.applied_complexity_penalty,
                hypothesis.complexity_penalty,
            ),
            "applied_ad_hoc_penalty": max(
                hypothesis.applied_ad_hoc_penalty,
                hypothesis.ad_hoc_penalty,
            ),
        }
    )


def _belief_update(
    *,
    hypothesis: Hypothesis,
    prior: float,
    posterior: float,
    band: LikelihoodBand,
    event: EvidenceEvent,
    weight: float,
    complexity_delta: float,
    ad_hoc_delta: float,
    update_id: str,
    cycle_id: str,
) -> BeliefUpdate:
    return BeliefUpdate(
        update_id=update_id,
        cycle_id=cycle_id,
        evidence_id=event.id,
        hypothesis_id=hypothesis.id,
        prior=round(prior, _DISTRIBUTION_PRECISION),
        posterior=round(posterior, _DISTRIBUTION_PRECISION),
        direction=_direction(prior, posterior),
        reason=(
            f"{event.evidence_type.value} is {band.value} for {hypothesis.id}."
        ),
        sensitivity={
            "weight": round(weight, _DISTRIBUTION_PRECISION),
            "likelihood_band": band.value,
            "complexity_penalty": hypothesis.complexity_penalty,
            "ad_hoc_penalty": hypothesis.ad_hoc_penalty,
            "complexity_penalty_delta": round(
                complexity_delta,
                _DISTRIBUTION_PRECISION,
            ),
            "ad_hoc_penalty_delta": round(ad_hoc_delta, _DISTRIBUTION_PRECISION),
        },
    )


__all__ = [
    "BeliefSolveResult",
    "CoverageAwareBeliefSolver",
    "likelihood_band_to_lr",
    "mark_replayed_evidence_events",
    "normalize_hypotheses",
    "solve_updates",
    "summarize_hypotheses",
]
