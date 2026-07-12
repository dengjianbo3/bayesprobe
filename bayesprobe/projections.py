from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from bayesprobe.core import CycleResult
from bayesprobe.model_gateway import ModelGateway, StructuredModelRequest
from bayesprobe.schemas import (
    AnswerProjection,
    AnswerRelationship,
    AnswerValueType,
    BeliefState,
    BeliefStateProjection,
    CapabilityKind,
    ChangeMyMindCondition,
    FrameAdequacyStatus,
    Hypothesis,
    HypothesisCompetition,
    HypothesisStatus,
    ProbeCandidate,
    ProbeDesign,
    ProbePurpose,
    ProjectionMode,
    TaskFrame,
    redact_secret_material,
)


class AnswerProjectionError(ValueError):
    pass


@dataclass(frozen=True)
class AnswerProjectionInput:
    cycle_id: str
    previous_belief_state: BeliefState
    cycle_result: CycleResult
    stop_reason: str | None = None


class AnswerProjector(Protocol):
    def project(self, input: AnswerProjectionInput) -> AnswerProjection: ...


class TaskAwareAnswerProjector:
    def __init__(self, model_gateway: ModelGateway | None = None) -> None:
        self._model_gateway = model_gateway

    def project(self, input: AnswerProjectionInput) -> AnswerProjection:
        belief_state = input.cycle_result.belief_state
        task_frame = belief_state.task_frame
        if task_frame is None or task_frame.answer_relationship is None:
            raise AnswerProjectionError("answer projection requires a task frame")
        if task_frame.answer_relationship == AnswerRelationship.SELECTION:
            return self._project_selection(input, task_frame)
        if task_frame.answer_relationship == AnswerRelationship.SYNTHESIS:
            return self._project_synthesis(input, task_frame)
        raise AnswerProjectionError("answer projection has an unsupported answer relationship")

    def _project_selection(
        self,
        input: AnswerProjectionInput,
        task_frame: TaskFrame,
    ) -> AnswerProjection:
        belief_state = input.cycle_result.belief_state
        top = _top_active_hypothesis(belief_state)
        blocked_reason = _frame_block_reason(belief_state)
        if blocked_reason is not None:
            return _abstention_projection(
                input=input,
                top=top,
                reason=blocked_reason,
            )
        if top is None:
            return _abstention_projection(
                input=input,
                top=None,
                reason="No active named answer candidate is available.",
            )
        unresolved_mass = (
            belief_state.frame_state.unresolved_alternative_mass
            if belief_state.frame_state is not None
            else None
        )
        if unresolved_mass is not None and unresolved_mass > top.posterior:
            return _abstention_projection(
                input=input,
                top=top,
                reason=(
                    "Unresolved alternative mass outranks the best named answer "
                    "candidate."
                ),
            )
        if not _answer_value_matches_contract(
            top.answer_value,
            task_frame.answer_contract.answer_value_type,
        ):
            return _abstention_projection(
                input=input,
                top=top,
                reason=(
                    "The best named candidate does not provide an answer value "
                    "compatible with the task contract."
                ),
            )
        return AnswerProjection(
            mode=ProjectionMode.SELECTION,
            answer=str(top.answer_value),
            answer_value=top.answer_value,
            current_best_hypothesis=top.id,
            posterior_summary=_posterior_summary_text(belief_state),
            main_uncertainty=_main_uncertainty_text(
                previous_belief_state=input.previous_belief_state,
                cycle_result=input.cycle_result,
            ),
            weakest_assumption=_weakest_assumption(top, task_frame),
            main_evidence_events=_admitted_evidence_ids(input.cycle_result),
            change_my_mind_condition=_task_aware_change_my_mind_condition(
                input.cycle_id,
                top,
            ),
            answer_utility_notes=(
                "Selected deterministically from the active hypothesis with a "
                "contract-compatible answer value."
            ),
        )

    def _project_synthesis(
        self,
        input: AnswerProjectionInput,
        task_frame: TaskFrame,
    ) -> AnswerProjection:
        belief_state = input.cycle_result.belief_state
        top = _top_active_hypothesis(belief_state)
        blocked_reason = _frame_block_reason(belief_state)
        if blocked_reason is not None:
            return _abstention_projection(
                input=input,
                top=top,
                reason=blocked_reason,
            )
        request = self._request(
            task="project_answer",
            input=_synthesis_request_input(input, task_frame),
            prompt_id="answer_projection",
            metadata={"cycle_id": input.cycle_id},
        )
        response = self._complete(request)
        try:
            validated = _validate_synthesis_response(
                response,
                task_frame=task_frame,
                admitted_evidence_ids=set(_admitted_evidence_ids(input.cycle_result)),
            )
        except AnswerProjectionError:
            repair_request = self._request(
                task="repair_answer_projection",
                input={
                    "original_request": redact_secret_material(request.input),
                    "validation_error": "answer projection response invalid",
                    "attempt_index": 1,
                },
                prompt_id="answer_projection_repair",
                metadata={"cycle_id": input.cycle_id, "repair_attempt_index": 1},
            )
            repaired = self._complete(repair_request)
            try:
                validated = _validate_synthesis_response(
                    repaired,
                    task_frame=task_frame,
                    admitted_evidence_ids=set(
                        _admitted_evidence_ids(input.cycle_result)
                    ),
                )
            except AnswerProjectionError:
                raise AnswerProjectionError(
                    "answer projection invalid after 1 repair attempt"
                ) from None
        return AnswerProjection(
            mode=ProjectionMode.SYNTHESIS,
            answer=validated["answer"],
            contract_sections=validated["contract_sections"],
            current_best_hypothesis=None if top is None else top.id,
            posterior_summary=_posterior_summary_text(belief_state),
            main_uncertainty=validated["main_uncertainty"],
            weakest_assumption=validated["weakest_assumption"],
            main_evidence_events=validated["cited_evidence_ids"],
            change_my_mind_condition=_task_aware_change_my_mind_condition(
                input.cycle_id,
                top,
            ),
            answer_utility_notes=(
                "Synthesized from the task contract and server-admitted evidence."
            ),
        )

    @staticmethod
    def _request(
        *,
        task: str,
        input: dict[str, Any],
        prompt_id: str,
        metadata: dict[str, Any],
    ) -> StructuredModelRequest:
        try:
            return StructuredModelRequest(
                task=task,
                input=input,
                prompt_id=prompt_id,
                prompt_version="v0.2",
                schema_name="AnswerProjection",
                schema_version="v0.2",
                metadata=metadata,
            )
        except (TypeError, ValueError):
            raise AnswerProjectionError(
                "answer projection model gateway call failed"
            ) from None

    def _complete(self, request: StructuredModelRequest) -> Any:
        if self._model_gateway is None:
            raise AnswerProjectionError("answer projection model gateway call failed")
        try:
            return self._model_gateway.complete_structured(request)
        except Exception:
            raise AnswerProjectionError(
                "answer projection model gateway call failed"
            ) from None


def build_answer_projection(
    cycle_id: str,
    previous_belief_state: BeliefState,
    cycle_result: CycleResult,
) -> AnswerProjection:
    """Keep the pre-v0.2 projection shape stable for synchronized callers."""
    top = _top_hypothesis(cycle_result.belief_state)
    return AnswerProjection(
        answer=_answer_text(top),
        current_best_hypothesis=top.id,
        posterior_summary=_posterior_summary_text(cycle_result.belief_state),
        main_uncertainty=_main_uncertainty_text(
            previous_belief_state=previous_belief_state,
            cycle_result=cycle_result,
        ),
        weakest_assumption=_weakest_assumption(
            top,
            cycle_result.belief_state.task_frame,
        ),
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


def _synthesis_request_input(
    input: AnswerProjectionInput,
    task_frame: TaskFrame,
) -> dict[str, Any]:
    belief_state = input.cycle_result.belief_state
    evidence_summaries = [
        {
            "id": event.id,
            "target_hypotheses": list(event.target_hypotheses),
            "evidence_type": event.evidence_type.value,
            "content": event.content,
            "interpretation": event.interpretation,
            "frame_fit": event.frame_fit.value,
            "quality": {
                "reliability": event.reliability,
                "independence": event.independence,
                "relevance": event.relevance,
                "verifiability": event.verifiability,
            },
        }
        for event in input.cycle_result.evidence_events
        if event.discard_reason is None
    ]
    return {
        "task_frame": task_frame.model_dump(mode="json"),
        "hypothesis_summaries": [
            {
                "id": hypothesis.id,
                "statement": hypothesis.statement,
                "scope": hypothesis.scope,
                "status": hypothesis.status.value,
                "posterior": hypothesis.posterior,
                "predictions": list(hypothesis.predictions),
                "falsifiers": list(hypothesis.falsifiers),
            }
            for hypothesis in _active_hypotheses(belief_state)
        ],
        "belief_summary": {
            "posterior_summary": dict(belief_state.posterior_summary),
            "uncertainty_summary": belief_state.uncertainty_summary,
            "frame_state": (
                None
                if belief_state.frame_state is None
                else belief_state.frame_state.model_dump(mode="json")
            ),
        },
        "admitted_evidence": redact_secret_material(evidence_summaries),
        "stop_reason": input.stop_reason,
    }


def _validate_synthesis_response(
    response: Any,
    *,
    task_frame: TaskFrame,
    admitted_evidence_ids: set[str],
) -> dict[str, Any]:
    fields = {
        "answer",
        "contract_sections",
        "main_uncertainty",
        "weakest_assumption",
        "cited_evidence_ids",
    }
    if not isinstance(response, Mapping) or set(response) != fields:
        raise AnswerProjectionError("answer projection response invalid")
    answer = _required_projection_text(response["answer"])
    main_uncertainty = _required_projection_text(response["main_uncertainty"])
    weakest_assumption = _required_projection_text(response["weakest_assumption"])
    sections = response["contract_sections"]
    if not isinstance(sections, Mapping) or set(sections) != set(
        task_frame.answer_contract.required_sections
    ):
        raise AnswerProjectionError("answer projection response invalid")
    clean_sections = {
        _required_projection_text(key): _required_projection_text(value)
        for key, value in sections.items()
    }
    cited_ids = response["cited_evidence_ids"]
    if (
        not isinstance(cited_ids, list)
        or any(not isinstance(item, str) or not item.strip() for item in cited_ids)
        or len(cited_ids) != len(set(cited_ids))
        or not set(cited_ids).issubset(admitted_evidence_ids)
    ):
        raise AnswerProjectionError("answer projection response invalid")
    return {
        "answer": answer,
        "contract_sections": clean_sections,
        "main_uncertainty": main_uncertainty,
        "weakest_assumption": weakest_assumption,
        "cited_evidence_ids": list(cited_ids),
    }


def _required_projection_text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AnswerProjectionError("answer projection response invalid")
    return value.strip()


def _answer_value_matches_contract(
    value: str | int | float | None,
    answer_value_type: AnswerValueType,
) -> bool:
    if value is None:
        return False
    if answer_value_type == AnswerValueType.INTEGER:
        return type(value) is int
    if answer_value_type == AnswerValueType.NUMBER:
        return type(value) in {int, float} and math.isfinite(value)
    return type(value) is str and bool(value.strip())


def _frame_block_reason(belief_state: BeliefState) -> str | None:
    if belief_state.frame_state is None:
        return "The answer frame has no adequacy state."
    status = belief_state.frame_state.adequacy_status
    if status == FrameAdequacyStatus.INADEQUATE:
        return "The answer frame is inadequate and needs expansion."
    if status == FrameAdequacyStatus.EXPANDING:
        return "Answer projection is deferred while frame expansion is pending."
    return None


def _abstention_projection(
    *,
    input: AnswerProjectionInput,
    top: Hypothesis | None,
    reason: str,
) -> AnswerProjection:
    return AnswerProjection(
        mode=ProjectionMode.ABSTENTION,
        answer="Abstain until the answer contract is adequately supported.",
        current_best_hypothesis=None if top is None else top.id,
        posterior_summary=_posterior_summary_text(input.cycle_result.belief_state),
        main_uncertainty=reason,
        weakest_assumption=(
            _weakest_assumption(top, input.cycle_result.belief_state.task_frame)
            if top is not None
            else "A named candidate is needed before an answer can be selected."
        ),
        main_evidence_events=_admitted_evidence_ids(input.cycle_result),
        change_my_mind_condition=_task_aware_change_my_mind_condition(
            input.cycle_id,
            top,
        ),
        answer_utility_notes="The projector withheld an answer because the frame is unresolved.",
    )


def _admitted_evidence_ids(cycle_result: CycleResult) -> list[str]:
    return [
        event.id
        for event in cycle_result.evidence_events
        if event.discard_reason is None
    ]


def _active_hypotheses(belief_state: BeliefState) -> list[Hypothesis]:
    active_ids = (
        set(belief_state.frame_state.active_hypothesis_ids)
        if belief_state.frame_state is not None
        else {hypothesis.id for hypothesis in belief_state.hypotheses}
    )
    return [
        hypothesis
        for hypothesis in belief_state.hypotheses
        if hypothesis.id in active_ids and hypothesis.status == HypothesisStatus.ACTIVE
    ]


def _top_active_hypothesis(belief_state: BeliefState) -> Hypothesis | None:
    active = _active_hypotheses(belief_state)
    return (
        None
        if not active
        else min(active, key=lambda hypothesis: (-hypothesis.posterior, hypothesis.id))
    )


def _task_aware_change_my_mind_condition(
    cycle_id: str,
    hypothesis: Hypothesis | None,
) -> ChangeMyMindCondition:
    if hypothesis is None:
        return ChangeMyMindCondition(
            human_readable_condition=(
                "A discriminating result is needed before a named answer can be revised."
            )
        )
    weaken_text = (
        hypothesis.falsifiers[0]
        if hypothesis.falsifiers
        else "A reliable counterevidence result is observed."
    )
    probe = ProbeCandidate(
        candidate_id=f"pc_{cycle_id}_{hypothesis.id}",
        source="change_my_mind",
        candidate_probe=ProbeDesign(
            id=f"P_{cycle_id}_{hypothesis.id}",
            cycle_id=cycle_id,
            target_hypotheses=[hypothesis.id],
            inquiry_goal=f"Test the strongest falsifier for {hypothesis.id}.",
            method=CapabilityKind.MODEL_REASONING.value,
            purpose=ProbePurpose.HYPOTHESIS_FALSIFICATION,
            required_capability=CapabilityKind.MODEL_REASONING,
            support_condition={
                hypothesis.id: (
                    hypothesis.predictions[0]
                    if hypothesis.predictions
                    else "The candidate's prediction is observed."
                )
            },
            weaken_condition={hypothesis.id: weaken_text},
            priority=0.8,
        ),
        priority_features={
            "projection_role": "change_my_mind",
            "target_hypothesis": hypothesis.id,
            "server_owned_priority": 0.8,
        },
    )
    return ChangeMyMindCondition(
        human_readable_condition=(
            f"I would lower confidence in {hypothesis.id} if a reliable signal shows "
            f"{weaken_text.lower()}"
        ),
        structured_probe_candidates=[probe],
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
    task_frame = belief_state.task_frame
    independent = (
        task_frame is not None
        and task_frame.hypothesis_frame.competition == HypothesisCompetition.INDEPENDENT
    )
    prefix = "Credences (not normalized):" if independent else "Posterior mass:"
    return f"{prefix} {', '.join(parts)}"


def _main_uncertainty_text(
    *,
    previous_belief_state: BeliefState,
    cycle_result: CycleResult,
) -> str:
    task_frame = cycle_result.belief_state.task_frame
    if (
        task_frame is not None
        and task_frame.hypothesis_frame.competition == HypothesisCompetition.INDEPENDENT
    ):
        return cycle_result.belief_state.uncertainty_summary
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


def _weakest_assumption(hypothesis: Hypothesis, task_frame: TaskFrame | None) -> str:
    if hypothesis.falsifiers:
        return hypothesis.falsifiers[0]
    if (
        task_frame is not None
        and task_frame.hypothesis_frame.coverage_limitation is not None
    ):
        return task_frame.hypothesis_frame.coverage_limitation
    return "Independent refutation may still be missing."


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
    "AnswerProjectionError",
    "AnswerProjectionInput",
    "AnswerProjector",
    "TaskAwareAnswerProjector",
    "build_answer_projection",
    "build_belief_state_projection",
]
