from __future__ import annotations

import math
from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator, model_validator

from bayesprobe.frame_policy import FrameAdequacyDecision
from bayesprobe.kernel_config import ExpansionPolicy, OpenCoveragePolicy
from bayesprobe.model_gateway import ModelGateway, StructuredModelRequest
from bayesprobe.schemas import (
    AnswerValueType,
    EvidenceEvent,
    EvolutionOperation,
    FrameAdequacyStatus,
    FrameMassUpdate,
    FrameState,
    Hypothesis,
    HypothesisCompetition,
    HypothesisEvolution,
    HypothesisStatus,
    ProbeCandidate,
    ProbeDesign,
    ProbePurpose,
    TaskFrame,
    UpdateDirection,
    is_forbidden_secret_key_name,
    is_secret_like_value,
    redact_secret_material,
)


class HypothesisExpansionError(ValueError):
    pass


class HypothesisExpansionProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    statement: str
    type: str
    scope: str
    falsifiers: list[str]
    predictions: list[str]
    answer_value: str | int | float | None
    why_current_frame_missed: str
    required_next_probe: str

    @field_validator(
        "statement",
        "type",
        "scope",
        "why_current_frame_missed",
        "required_next_probe",
    )
    @classmethod
    def clean_text(cls, value: str, info: ValidationInfo) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{info.field_name} must not be empty")
        if is_secret_like_value(value):
            raise ValueError("hypothesis expansion proposal must not contain secret material")
        return value.strip()

    @field_validator("falsifiers", "predictions")
    @classmethod
    def clean_semantic_list(
        cls,
        value: list[str],
        info: ValidationInfo,
    ) -> list[str]:
        if not value:
            raise ValueError(f"{info.field_name} must not be empty")
        cleaned: list[str] = []
        for item in value:
            if (
                not isinstance(item, str)
                or not item.strip()
                or is_secret_like_value(item)
            ):
                raise ValueError(
                    "hypothesis expansion proposal must not contain secret material"
                )
            normalized = item.strip()
            if normalized not in cleaned:
                cleaned.append(normalized)
        return cleaned

    @field_validator("answer_value", mode="before")
    @classmethod
    def validate_answer_value_scalar(cls, value: Any) -> str | int | float | None:
        if value is not None and type(value) not in {str, int, float}:
            raise ValueError("answer_value must be a scalar or null")
        if isinstance(value, str):
            if not value.strip() or is_secret_like_value(value):
                raise ValueError("hypothesis expansion proposal must not contain secret material")
            return value.strip()
        return value

    @model_validator(mode="after")
    def reject_secret_material(self) -> "HypothesisExpansionProposal":
        if any(
            is_forbidden_secret_key_name(key)
            for key in self.model_dump(mode="python")
        ):
            raise ValueError("hypothesis expansion proposal must not contain secret material")
        return self


class _HypothesisExpansionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[HypothesisExpansionProposal]

    @field_validator("candidates")
    @classmethod
    def require_one_to_three(
        cls,
        value: list[HypothesisExpansionProposal],
    ) -> list[HypothesisExpansionProposal]:
        if not 1 <= len(value) <= 3:
            raise ValueError("hypothesis expansion requires between one and three candidates")
        return value


@dataclass(frozen=True)
class HypothesisExpansionRequest:
    run_id: str
    cycle_id: str
    task_frame: TaskFrame
    frame_state: FrameState
    hypotheses: tuple[Hypothesis, ...]
    triggering_events: tuple[EvidenceEvent, ...]
    expansion_reason: str


@dataclass(frozen=True)
class HypothesisExpansionResult:
    hypotheses: list[Hypothesis]
    frame_state: FrameState
    evolutions: list[HypothesisEvolution]
    probe_candidates: list[ProbeCandidate]
    frame_mass_updates: list[FrameMassUpdate]
    discovery_evidence_ids: list[str]


class HypothesisExpansionAdapter(Protocol):
    def propose(
        self,
        request: HypothesisExpansionRequest,
    ) -> list[HypothesisExpansionProposal]:
        ...


class ModelHypothesisExpansionAdapter:
    def __init__(self, model_gateway: ModelGateway) -> None:
        self._model_gateway = model_gateway

    def propose(
        self,
        request: HypothesisExpansionRequest,
    ) -> list[HypothesisExpansionProposal]:
        model_request = self._request(
            task="expand_hypotheses",
            input=_model_input(request),
            prompt_id="hypothesis_expansion",
            metadata={"run_id": request.run_id, "cycle_id": request.cycle_id},
        )
        response = self._complete(model_request, stage="hypothesis expansion")
        try:
            return self._validate_response(response)
        except HypothesisExpansionError:
            repair_request = self._request(
                task="repair_hypothesis_expansion",
                input={
                    "original_request": redact_secret_material(model_request.input),
                    "validation_error": "hypothesis expansion response invalid",
                    "attempt_index": 1,
                },
                prompt_id="hypothesis_expansion_repair",
                metadata={
                    "run_id": request.run_id,
                    "cycle_id": request.cycle_id,
                    "repair_attempt_index": 1,
                },
            )
            repaired = self._complete(
                repair_request,
                stage="hypothesis expansion repair",
            )
            try:
                return self._validate_response(repaired)
            except HypothesisExpansionError as error:
                raise HypothesisExpansionError(
                    "hypothesis expansion invalid after 1 repair attempt"
                ) from None

    @staticmethod
    def _request(
        *,
        task: str,
        input: dict[str, Any],
        prompt_id: str,
        metadata: dict[str, Any],
    ) -> StructuredModelRequest:
        try:
            request = StructuredModelRequest(
                task=task,
                input=input,
                prompt_id=prompt_id,
                prompt_version="v0.2",
                schema_name="HypothesisExpansion",
                schema_version="v0.2",
                metadata=metadata,
            )
        except (TypeError, ValueError):
            request = None
        if request is None:
            raise HypothesisExpansionError("hypothesis expansion request construction failed")
        return request

    def _complete(
        self,
        request: StructuredModelRequest,
        *,
        stage: str,
    ) -> dict[str, Any]:
        try:
            response = self._model_gateway.complete_structured(request)
        except Exception:
            response = None
        if not isinstance(response, dict):
            raise HypothesisExpansionError(f"{stage} model gateway call failed")
        return response

    @staticmethod
    def _validate_response(
        response: dict[str, Any],
    ) -> list[HypothesisExpansionProposal]:
        try:
            parsed = _HypothesisExpansionResponse.model_validate(response)
        except Exception:
            parsed = None
        if parsed is None:
            raise HypothesisExpansionError("hypothesis expansion response invalid")
        return parsed.candidates


class HypothesisExpansionService:
    def __init__(
        self,
        *,
        adapter: HypothesisExpansionAdapter,
        expansion_policy: ExpansionPolicy | None = None,
        open_policy: OpenCoveragePolicy | None = None,
    ) -> None:
        self._adapter = adapter
        self._expansion_policy = expansion_policy or ExpansionPolicy()
        self._open_policy = open_policy or OpenCoveragePolicy()

    def expand(
        self,
        *,
        request: HypothesisExpansionRequest,
        decision: FrameAdequacyDecision,
    ) -> HypothesisExpansionResult:
        self._validate_expansion_request(request, decision)
        proposals = self._validate_proposals(request, self._adapter.propose(request))
        next_frame_version = request.frame_state.frame_version + 1
        discovery_evidence_ids = _unique_ids(event.id for event in request.triggering_events)
        prior_by_id = {hypothesis.id: hypothesis for hypothesis in request.hypotheses}
        active_ids = [
            hypothesis_id
            for hypothesis_id in request.frame_state.active_hypothesis_ids
            if hypothesis_id in prior_by_id
            and prior_by_id[hypothesis_id].status == HypothesisStatus.ACTIVE
        ]
        new_hypotheses, frame_mass_updates, next_unresolved = self._materialize_hypotheses(
            request=request,
            proposals=proposals,
            active_ids=active_ids,
            next_frame_version=next_frame_version,
            discovery_evidence_ids=discovery_evidence_ids,
        )
        all_active_ids = [*active_ids, *(item.id for item in new_hypotheses)]
        hypotheses = _reconcile_rivals(
            hypotheses=list(request.hypotheses),
            active_ids=all_active_ids,
            competition=request.frame_state.competition,
        )
        hypotheses.extend(new_hypotheses)
        frame_state = request.frame_state.model_copy(
            update={
                "frame_version": next_frame_version,
                "parent_frame_version": request.frame_state.frame_version,
                "active_hypothesis_ids": all_active_ids,
                "unresolved_alternative_mass": next_unresolved,
                "adequacy_status": FrameAdequacyStatus.PROVISIONAL,
                "revision_reason": decision.reason,
                "trigger_event_ids": discovery_evidence_ids,
                "revision_count": request.frame_state.revision_count + 1,
            }
        )
        evolutions = [
            HypothesisEvolution(
                evolution_id=f"{request.cycle_id}_{hypothesis.id}_spawn_HE",
                cycle_id=request.cycle_id,
                operation=EvolutionOperation.SPAWN,
                from_hypothesis=None,
                to_hypothesis=hypothesis.id,
                triggered_by=discovery_evidence_ids,
                reason=proposal.why_current_frame_missed,
                audit_fields={
                    "required_next_probe": proposal.required_next_probe,
                    "new_hypothesis_prior": hypothesis.prior,
                    "frame_version": next_frame_version,
                },
            )
            for proposal, hypothesis in zip(proposals, new_hypotheses, strict=True)
        ]
        return HypothesisExpansionResult(
            hypotheses=hypotheses,
            frame_state=frame_state,
            evolutions=evolutions,
            probe_candidates=[
                _follow_up_probe(
                    request=request,
                    active_ids=all_active_ids,
                    new_hypotheses=new_hypotheses,
                    proposals=proposals,
                )
            ],
            frame_mass_updates=frame_mass_updates,
            discovery_evidence_ids=discovery_evidence_ids,
        )

    def _validate_expansion_request(
        self,
        request: HypothesisExpansionRequest,
        decision: FrameAdequacyDecision,
    ) -> None:
        if not decision.should_expand:
            raise HypothesisExpansionError("hypothesis expansion requires an expansion decision")
        if request.frame_state.coverage.value != "open":
            raise HypothesisExpansionError("hypothesis expansion requires an open frame")
        if request.frame_state.revision_count >= self._expansion_policy.max_frame_revisions:
            raise HypothesisExpansionError("hypothesis expansion revision limit reached")
        if request.frame_state.frame_id != request.task_frame.hypothesis_frame.frame_id:
            raise HypothesisExpansionError("hypothesis expansion frame state does not match task frame")
        if not request.expansion_reason.strip():
            raise HypothesisExpansionError("hypothesis expansion requires a reason")

    def _validate_proposals(
        self,
        request: HypothesisExpansionRequest,
        proposals: list[HypothesisExpansionProposal],
    ) -> list[HypothesisExpansionProposal]:
        if not isinstance(proposals, list) or not 1 <= len(proposals) <= 3:
            raise HypothesisExpansionError(
                "hypothesis expansion requires between one and three proposals"
            )
        if any(not isinstance(item, HypothesisExpansionProposal) for item in proposals):
            raise HypothesisExpansionError("hypothesis expansion proposals are invalid")
        active_count = sum(
            hypothesis.status == HypothesisStatus.ACTIVE
            and hypothesis.id in request.frame_state.active_hypothesis_ids
            for hypothesis in request.hypotheses
        )
        if active_count + len(proposals) > self._expansion_policy.max_active_hypotheses:
            raise HypothesisExpansionError("hypothesis expansion active hypothesis limit reached")
        known_statements = {
            _semantic_text(hypothesis.statement) for hypothesis in request.hypotheses
        }
        proposal_statements: set[str] = set()
        known_answer_values = {
            _answer_value_key(hypothesis.answer_value, request.task_frame.answer_contract.answer_value_type)
            for hypothesis in request.hypotheses
            if hypothesis.answer_value is not None
        }
        proposal_answer_values: set[tuple[str, str | int | float]] = set()
        for proposal in proposals:
            statement = _semantic_text(proposal.statement)
            if statement in known_statements or statement in proposal_statements:
                raise HypothesisExpansionError(
                    "hypothesis expansion duplicates an existing hypothesis statement"
                )
            proposal_statements.add(statement)
            self._validate_answer_value(request, proposal)
            if proposal.answer_value is not None:
                key = _answer_value_key(
                    proposal.answer_value,
                    request.task_frame.answer_contract.answer_value_type,
                )
                if key in known_answer_values or key in proposal_answer_values:
                    raise HypothesisExpansionError(
                        "hypothesis expansion duplicates an existing answer_value"
                    )
                proposal_answer_values.add(key)
        return proposals

    @staticmethod
    def _validate_answer_value(
        request: HypothesisExpansionRequest,
        proposal: HypothesisExpansionProposal,
    ) -> None:
        contract = request.task_frame.answer_contract
        is_exact_answer = request.task_frame.task_kind.value == "exact_answer"
        value = proposal.answer_value
        if not is_exact_answer:
            if value is not None:
                raise HypothesisExpansionError(
                    "hypothesis expansion non-answer candidates require answer_value null"
                )
            return
        if value is None:
            raise HypothesisExpansionError(
                "hypothesis expansion exact-answer candidates require answer_value"
            )
        if not _answer_value_matches_type(value, contract.answer_value_type):
            raise HypothesisExpansionError(
                "hypothesis expansion answer_value must match answer_value_type"
            )

    def _materialize_hypotheses(
        self,
        *,
        request: HypothesisExpansionRequest,
        proposals: list[HypothesisExpansionProposal],
        active_ids: list[str],
        next_frame_version: int,
        discovery_evidence_ids: list[str],
    ) -> tuple[list[Hypothesis], list[FrameMassUpdate], float | None]:
        if request.frame_state.competition == HypothesisCompetition.INDEPENDENT:
            prior = 0.5
            return (
                _spawned_hypotheses(
                    proposals=proposals,
                    prior=prior,
                    active_ids=active_ids,
                    next_frame_version=next_frame_version,
                    independent=True,
                ),
                [],
                None,
            )
        current_unresolved = request.frame_state.unresolved_alternative_mass
        if current_unresolved is None:
            raise HypothesisExpansionError(
                "exclusive-open hypothesis expansion requires unresolved mass"
            )
        available = max(
            current_unresolved - self._open_policy.minimum_unresolved_reserve,
            0.0,
        )
        transfer = min(current_unresolved * 0.5, available)
        per_candidate = transfer / len(proposals)
        next_unresolved = current_unresolved - transfer
        evidence_id = discovery_evidence_ids[0] if discovery_evidence_ids else request.cycle_id
        return (
            _spawned_hypotheses(
                proposals=proposals,
                prior=per_candidate,
                active_ids=active_ids,
                next_frame_version=next_frame_version,
                independent=False,
            ),
            [
                FrameMassUpdate(
                    update_id=f"{request.cycle_id}_frame_mass_expansion_f{next_frame_version}",
                    cycle_id=request.cycle_id,
                    evidence_id=evidence_id,
                    prior=current_unresolved,
                    posterior=next_unresolved,
                    direction=UpdateDirection.WEAKENED,
                    reason="Expansion transferred unresolved mass to new hypotheses.",
                )
            ],
            next_unresolved,
        )


def _model_input(request: HypothesisExpansionRequest) -> dict[str, Any]:
    return {
        "task_frame": request.task_frame.model_dump(mode="json"),
        "frame_state": request.frame_state.model_dump(mode="json"),
        "hypotheses": [item.model_dump(mode="json") for item in request.hypotheses],
        "triggering_evidence": [
            _safe_event_summary(item) for item in request.triggering_events
        ],
        "expansion_reason": request.expansion_reason,
        "answer_value_type": request.task_frame.answer_contract.answer_value_type.value,
        "proposal_count": {"minimum": 1, "maximum": 3},
    }


def _safe_event_summary(event: EvidenceEvent) -> dict[str, Any]:
    return {
        "id": event.id,
        "evidence_type": event.evidence_type.value,
        "target_hypotheses": list(event.target_hypotheses),
        "frame_fit": event.frame_fit.value,
        "unresolved_likelihood": (
            None if event.unresolved_likelihood is None else event.unresolved_likelihood.value
        ),
        "unexplained_observation": event.unexplained_observation,
        "interpretation": event.interpretation,
        "verifiability": event.verifiability,
        "independence": event.independence,
    }


def _spawned_hypotheses(
    *,
    proposals: list[HypothesisExpansionProposal],
    prior: float,
    active_ids: list[str],
    next_frame_version: int,
    independent: bool,
) -> list[Hypothesis]:
    proposal_ids = [
        f"H_exp_f{next_frame_version}_{index}"
        for index in range(1, len(proposals) + 1)
    ]
    all_active_ids = [*active_ids, *proposal_ids]
    return [
        Hypothesis(
            id=hypothesis_id,
            statement=proposal.statement,
            type=proposal.type,
            scope=proposal.scope,
            prior=prior,
            posterior=prior,
            rivals=[] if independent else [item for item in all_active_ids if item != hypothesis_id],
            falsifiers=list(proposal.falsifiers),
            predictions=list(proposal.predictions),
            created_by="spawned",
            why_existing_hypotheses_failed=proposal.why_current_frame_missed,
            answer_value=proposal.answer_value,
        )
        for proposal, hypothesis_id in zip(proposals, proposal_ids, strict=True)
    ]


def _reconcile_rivals(
    *,
    hypotheses: list[Hypothesis],
    active_ids: list[str],
    competition: HypothesisCompetition,
) -> list[Hypothesis]:
    if competition == HypothesisCompetition.INDEPENDENT:
        return hypotheses
    active_id_set = set(active_ids)
    return [
        hypothesis.model_copy(
            update={"rivals": [item for item in active_ids if item != hypothesis.id]}
        )
        if hypothesis.id in active_id_set
        else hypothesis
        for hypothesis in hypotheses
    ]


def _follow_up_probe(
    *,
    request: HypothesisExpansionRequest,
    active_ids: list[str],
    new_hypotheses: list[Hypothesis],
    proposals: list[HypothesisExpansionProposal],
) -> ProbeCandidate:
    identity = json.dumps(
        {
            "cycle_id": request.cycle_id,
            "targets": active_ids,
            "new_hypotheses": [item.id for item in new_hypotheses],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = sha256(identity.encode("utf-8")).hexdigest()[:12]
    required_next_probe = " ".join(item.required_next_probe for item in proposals)
    probe = ProbeDesign(
        id=f"P_{request.cycle_id}_expansion_{digest}",
        cycle_id=request.cycle_id,
        target_hypotheses=active_ids,
        inquiry_goal="Test the newly proposed hypotheses against the active rivals.",
        method="hypothesis_expansion_follow_up",
        purpose=ProbePurpose.FRAME_COVERAGE,
        expected_observation=required_next_probe,
        support_condition={
            item.id: "The required follow-up observation supports this new hypothesis."
            for item in new_hypotheses
        },
        weaken_condition={
            item.id: "The required follow-up observation fails for this new hypothesis."
            for item in new_hypotheses
        },
        reframe_condition={
            "frame": "Neither the active rivals nor the new hypotheses explain the observation."
        },
        expected_information_gain=0.8,
        decision_relevance=0.8,
        cost_estimate=0.5,
        priority=0.85,
    )
    return ProbeCandidate(
        candidate_id=f"C_{request.cycle_id}_expansion_{digest}",
        source="uncertainty",
        candidate_probe=probe,
        priority_features={
            "server_owned_priority": probe.priority,
            "expansion_hypothesis_ids": [item.id for item in new_hypotheses],
        },
    )


def _answer_value_matches_type(
    value: str | int | float,
    answer_value_type: AnswerValueType,
) -> bool:
    if answer_value_type == AnswerValueType.INTEGER:
        return type(value) is int
    if answer_value_type == AnswerValueType.NUMBER:
        return type(value) in {int, float} and math.isfinite(value)
    if answer_value_type in {
        AnswerValueType.CHOICE_LABEL,
        AnswerValueType.SHORT_TEXT,
        AnswerValueType.STRUCTURED_TEXT,
    }:
        return type(value) is str
    return False


def _answer_value_key(
    value: str | int | float,
    answer_value_type: AnswerValueType,
) -> tuple[str, str | int | float]:
    if answer_value_type == AnswerValueType.NUMBER:
        return "number", value
    return type(value).__name__, value


def _semantic_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _unique_ids(values: Any) -> list[str]:
    return list(dict.fromkeys(values))


__all__ = [
    "HypothesisExpansionAdapter",
    "HypothesisExpansionError",
    "HypothesisExpansionProposal",
    "HypothesisExpansionRequest",
    "HypothesisExpansionResult",
    "HypothesisExpansionService",
    "ModelHypothesisExpansionAdapter",
]
