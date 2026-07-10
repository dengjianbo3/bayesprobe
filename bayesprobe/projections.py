from __future__ import annotations

from bayesprobe.core import CycleResult
from bayesprobe.schemas import (
    AnswerProjection,
    BeliefState,
    BeliefStateProjection,
    ChangeMyMindCondition,
    Hypothesis,
    ProbeCandidate,
    ProbeDesign,
)


def build_answer_projection(
    cycle_id: str,
    previous_belief_state: BeliefState,
    cycle_result: CycleResult,
) -> AnswerProjection:
    top = _top_hypothesis(cycle_result.belief_state)
    return AnswerProjection(
        answer=_answer_text(top),
        current_best_hypothesis=top.id,
        posterior_summary=_posterior_summary_text(cycle_result.belief_state),
        main_uncertainty=_main_uncertainty_text(
            previous_belief_state=previous_belief_state,
            cycle_result=cycle_result,
        ),
        weakest_assumption=top.falsifiers[0] if top.falsifiers else "Independent refutation may still be missing.",
        main_evidence_events=[event.id for event in cycle_result.evidence_events],
        change_my_mind_condition=_change_my_mind_condition(cycle_id, top),
        answer_utility_notes=f"Generated after integrating BayesProbe cycle {cycle_id}.",
    )


def build_belief_state_projection(
    cycle_id: str,
    previous_belief_state: BeliefState,
    cycle_result: CycleResult,
) -> BeliefStateProjection:
    top = _top_hypothesis(cycle_result.belief_state)
    uncertainty = _main_uncertainty_text(
        previous_belief_state=previous_belief_state,
        cycle_result=cycle_result,
    )
    return BeliefStateProjection(
        current_best_hypothesis=top.id,
        posterior_or_confidence_interval=_posterior_summary_text(cycle_result.belief_state),
        main_evidence_events=[event.id for event in cycle_result.evidence_events],
        main_uncertainties=[uncertainty],
        questions_for_others=[
            f"Can someone verify whether the strongest remaining challenge to {top.id} is independent?"
        ],
        change_my_mind_condition=_change_my_mind_condition(cycle_id, top),
        requested_signal_type="counterevidence_or_source_challenge",
        cited_sources=[],
        projection_metadata={"cycle_id": cycle_id},
    )


def _top_hypothesis(belief_state: BeliefState) -> Hypothesis:
    return max(belief_state.hypotheses, key=lambda hypothesis: hypothesis.posterior)


def _answer_text(hypothesis: Hypothesis) -> str:
    prefix = f"Answer choice {hypothesis.id} is correct: "
    if hypothesis.statement.startswith(prefix):
        return f"Current best answer is {hypothesis.id}: {hypothesis.statement[len(prefix):]}"
    return f"Current best hypothesis is {hypothesis.id}: {hypothesis.statement}"


def _posterior_summary_text(belief_state: BeliefState) -> str:
    ranked = sorted(
        belief_state.hypotheses,
        key=lambda hypothesis: (-hypothesis.posterior, hypothesis.id),
    )
    parts = [f"{hypothesis.id}={hypothesis.posterior:.3f}" for hypothesis in ranked]
    return ", ".join(parts)


def _main_uncertainty_text(
    *,
    previous_belief_state: BeliefState,
    cycle_result: CycleResult,
) -> str:
    if cycle_result.evidence_events:
        ranked = sorted(
            cycle_result.belief_state.hypotheses,
            key=lambda hypothesis: (-hypothesis.posterior, hypothesis.id),
        )
        if len(ranked) >= 2:
            gap = ranked[0].posterior - ranked[1].posterior
            return (
                f"The current posterior gap between {ranked[0].id} and {ranked[1].id} "
                f"is {gap:.3f}; further discriminative evidence may change the ranking."
            )
        return (
            f"Evidence was integrated for {ranked[0].id}, but independent verification "
            "may still change its posterior."
        )
    return (
        previous_belief_state.uncertainty_summary
        or "The remaining rival mass still needs sharper evidence."
    )


def _change_my_mind_condition(cycle_id: str, hypothesis: Hypothesis) -> ChangeMyMindCondition:
    support_text = hypothesis.predictions[0] if hypothesis.predictions else "Independent support appears."
    weaken_text = hypothesis.falsifiers[0] if hypothesis.falsifiers else "A reliable counterevidence source appears."
    probe = ProbeCandidate(
        candidate_id=f"pc_{cycle_id}_{hypothesis.id}",
        source="change_my_mind",
        candidate_probe=ProbeDesign(
            id=f"P_{cycle_id}_{hypothesis.id}",
            cycle_id=cycle_id,
            target_hypotheses=[hypothesis.id],
            inquiry_goal=f"Check whether {hypothesis.id} still holds up.",
            method="source_tracing",
            support_condition={hypothesis.id: support_text},
            weaken_condition={hypothesis.id: weaken_text},
            expected_information_gain=0.8,
            decision_relevance=0.9,
            cost_estimate=0.4,
            priority=0.85,
        ),
        priority_features={
            "projection_role": "change_my_mind",
            "target_hypothesis": hypothesis.id,
        },
    )
    return ChangeMyMindCondition(
        human_readable_condition=(
            f"I would lower confidence in {hypothesis.id} if a reliable independent signal "
            f"shows {weaken_text.lower()}"
        ),
        structured_probe_candidates=[probe],
    )


__all__ = [
    "build_answer_projection",
    "build_belief_state_projection",
]
