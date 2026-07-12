from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator

from bayesprobe.model_gateway import ModelGateway, StructuredModelRequest
from bayesprobe.schemas import (
    BeliefState,
    CapabilityDecision,
    CapabilityDescriptor,
    CapabilityKind,
    EpistemicOrigin,
    ProbeCandidate,
    ProbeDesign,
    ProbePurpose,
    TaskFrame,
    is_forbidden_secret_key_name,
    is_secret_like_value,
    redact_secret_material,
)


class ProbeDesignError(ValueError):
    pass


@dataclass(frozen=True)
class ProbeDesignContext:
    run_id: str
    cycle_id: str
    task_frame: TaskFrame
    belief_state: BeliefState
    available_capabilities: tuple[CapabilityDescriptor, ...]


@dataclass(frozen=True)
class ProbeDesignResult:
    candidates: list[ProbeCandidate]
    capability_decisions: list[CapabilityDecision]


class ProbeDesigner(Protocol):
    def propose(self, context: ProbeDesignContext) -> ProbeDesignResult: ...


MODEL_REASONING_CAPABILITY = CapabilityDescriptor(
    kind=CapabilityKind.MODEL_REASONING,
    available=True,
    cost_class="bounded",
    latency_class="interactive",
    epistemic_origin=EpistemicOrigin.MODEL_REASONING,
    quality_caps={"verifiability": 0.45, "independence": 0.25},
    executor_adapter_id="model_probe_gateway:v1",
)


_PRIORITY_BY_PURPOSE = {
    ProbePurpose.HYPOTHESIS_DISCRIMINATION: 0.85,
    ProbePurpose.HYPOTHESIS_FALSIFICATION: 0.80,
    ProbePurpose.FRAME_COVERAGE: 0.82,
    ProbePurpose.SOURCE_VERIFICATION: 0.70,
    ProbePurpose.ANOMALY_CLARIFICATION: 0.78,
    ProbePurpose.ANSWER_CONTRACT_GAP: 0.75,
}


class _ProbeProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    purpose: ProbePurpose
    target_hypotheses: list[str]
    inquiry_goal: str
    expected_observation: str
    support_condition: dict[str, str]
    weaken_condition: dict[str, str]
    reframe_condition: dict[str, str] | None
    required_capability: CapabilityKind

    @field_validator("inquiry_goal", "expected_observation")
    @classmethod
    def clean_text(cls, value: str, info: ValidationInfo) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{info.field_name} must not be empty")
        if is_secret_like_value(value):
            raise ValueError("probe proposal must not contain secret material")
        return value.strip()

    @field_validator("target_hypotheses")
    @classmethod
    def clean_targets(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("target_hypotheses must not be empty")
        cleaned: list[str] = []
        for target in value:
            if not isinstance(target, str) or not target.strip():
                raise ValueError("target_hypotheses must contain non-empty strings")
            if is_secret_like_value(target):
                raise ValueError("probe proposal must not contain secret material")
            clean_target = target.strip()
            if clean_target not in cleaned:
                cleaned.append(clean_target)
        return cleaned

    @field_validator("support_condition", "weaken_condition", "reframe_condition")
    @classmethod
    def clean_conditions(
        cls,
        value: dict[str, str] | None,
        info: ValidationInfo,
    ) -> dict[str, str] | None:
        if value is None:
            return None
        cleaned: dict[str, str] = {}
        for key, item in value.items():
            if (
                not isinstance(key, str)
                or not key.strip()
                or is_forbidden_secret_key_name(key)
                or is_secret_like_value(key)
                or not isinstance(item, str)
                or not item.strip()
                or is_secret_like_value(item)
            ):
                raise ValueError("probe proposal must not contain secret material")
            cleaned[key.strip()] = item.strip()
        return cleaned


class _ProbeProposalResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposals: list[_ProbeProposal]

    @field_validator("proposals")
    @classmethod
    def require_proposals(cls, value: list[_ProbeProposal]) -> list[_ProbeProposal]:
        if not 1 <= len(value) <= 3:
            raise ValueError("probe design requires between one and three proposals")
        return value


class FrameProbeDesigner:
    def propose(self, context: ProbeDesignContext) -> ProbeDesignResult:
        targets = _active_hypothesis_ids(context)
        if not targets:
            raise ProbeDesignError("probe design requires active hypotheses")
        proposal = _ProbeProposal(
            purpose=ProbePurpose.HYPOTHESIS_DISCRIMINATION,
            target_hypotheses=targets,
            inquiry_goal="Distinguish the active hypotheses using their stated predictions.",
            expected_observation=(
                "An observation favors one active hypothesis over its alternatives."
            ),
            support_condition={
                hypothesis_id: "Its stated prediction is observed."
                for hypothesis_id in targets
            },
            weaken_condition={
                hypothesis_id: "Its stated falsifier is observed."
                for hypothesis_id in targets
            },
            reframe_condition=None,
            required_capability=CapabilityKind.MODEL_REASONING,
        )
        descriptor = MODEL_REASONING_CAPABILITY.model_copy(
            update={
                "executor_adapter_id": "deterministic_frame_probe_designer:v1",
            }
        )
        return ProbeDesignResult(
            candidates=[_materialize_candidate(context, proposal)],
            capability_decisions=[
                CapabilityDecision(
                    kind=CapabilityKind.MODEL_REASONING,
                    available=True,
                    descriptor=descriptor,
                    reason="deterministic compatibility probe designer is available",
                )
            ],
        )


class ModelProbeDesigner:
    def __init__(self, model_gateway: ModelGateway) -> None:
        self._model_gateway = model_gateway

    def propose(self, context: ProbeDesignContext) -> ProbeDesignResult:
        request = self._request(
            stage="probe design",
            task="design_probes",
            input=_design_request_input(context),
            prompt_id="probe_design",
            metadata={"run_id": context.run_id, "cycle_id": context.cycle_id},
        )
        response = self._complete(request, stage="probe design")
        try:
            proposals = self._validate_proposals(
                response,
                context,
                stage="probe design",
            )
            return _result_for_proposals(context, proposals)
        except ProbeDesignError as error:
            if str(error) == "probe design response contains secret material":
                raise
            repair_request = self._request(
                stage="probe design repair",
                task="repair_probe_design",
                input={
                    "original_request": redact_secret_material(request.input),
                    "validation_error": "probe design response invalid",
                    "attempt_index": 1,
                },
                prompt_id="probe_design_repair",
                metadata={
                    "run_id": context.run_id,
                    "cycle_id": context.cycle_id,
                    "repair_attempt_index": 1,
                },
            )
            repaired = self._complete(repair_request, stage="probe design repair")
            proposals = self._validate_proposals(
                repaired,
                context,
                stage="probe design repair",
            )
            return _result_for_proposals(context, proposals)

    def _request(
        self,
        *,
        stage: str,
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
                schema_name="ProbeDesign",
                schema_version="v0.2",
                metadata=metadata,
            )
        except (TypeError, ValueError):
            request = None
        if request is None:
            raise ProbeDesignError(f"{stage} request construction failed")
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
        if response is None:
            raise ProbeDesignError(f"{stage} model gateway call failed")
        return response

    def _validate_proposals(
        self,
        response: dict[str, Any],
        context: ProbeDesignContext,
        *,
        stage: str,
    ) -> list[_ProbeProposal]:
        try:
            parsed = _ProbeProposalResponse.model_validate(response)
        except Exception as error:
            parsed = None
            validation_error = _validation_message(error, stage=stage)
        else:
            validation_error = None
        if parsed is None:
            raise ProbeDesignError(validation_error)
        known_hypotheses = set(_active_hypothesis_ids(context))
        for proposal in parsed.proposals:
            unknown = set(proposal.target_hypotheses).difference(known_hypotheses)
            if unknown:
                raise ProbeDesignError(f"{stage} response invalid")
        return parsed.proposals


def _result_for_proposals(
    context: ProbeDesignContext,
    proposals: list[_ProbeProposal],
) -> ProbeDesignResult:
    decisions: list[CapabilityDecision] = []
    candidates: list[ProbeCandidate] = []
    seen_identities: set[str] = set()
    descriptors = {
        descriptor.kind: descriptor
        for descriptor in context.available_capabilities
    }
    for proposal in proposals:
        descriptor = descriptors.get(proposal.required_capability)
        available = descriptor is not None and descriptor.available
        decisions.append(
            CapabilityDecision(
                kind=proposal.required_capability,
                available=available,
                descriptor=descriptor,
                reason=(
                    "required capability is available"
                    if available
                    else "required capability is unavailable"
                ),
            )
        )
        if not available:
            continue
        identity = _semantic_identity(context, proposal)
        if identity in seen_identities:
            continue
        seen_identities.add(identity)
        candidates.append(_materialize_candidate(context, proposal))
    if _requires_initial_open_coverage(context) and not any(
        proposal.purpose
        in {
            ProbePurpose.HYPOTHESIS_DISCRIMINATION,
            ProbePurpose.FRAME_COVERAGE,
        }
        and len(proposal.target_hypotheses) >= 2
        for proposal in proposals
    ):
        raise ProbeDesignError(
            "initial open design requires a multi-hypothesis discriminator or frame-coverage proposal"
        )
    return ProbeDesignResult(candidates=candidates, capability_decisions=decisions)


def _materialize_candidate(
    context: ProbeDesignContext,
    proposal: _ProbeProposal,
) -> ProbeCandidate:
    identity = _semantic_identity(context, proposal)
    digest = sha256(identity.encode("utf-8")).hexdigest()[:12]
    probe = ProbeDesign(
        id=f"P_{context.cycle_id}_{digest}",
        cycle_id=context.cycle_id,
        target_hypotheses=list(proposal.target_hypotheses),
        inquiry_goal=proposal.inquiry_goal,
        method=proposal.required_capability.value,
        purpose=proposal.purpose,
        expected_observation=proposal.expected_observation,
        required_capability=proposal.required_capability,
        support_condition=dict(proposal.support_condition),
        weaken_condition=dict(proposal.weaken_condition),
        reframe_condition=(
            None
            if proposal.reframe_condition is None
            else dict(proposal.reframe_condition)
        ),
        priority=_PRIORITY_BY_PURPOSE[proposal.purpose],
    )
    return ProbeCandidate(
        candidate_id=f"C_{context.cycle_id}_{digest}",
        source="uncertainty",
        candidate_probe=probe,
        priority_features={"server_owned_priority": probe.priority},
    )


def _semantic_identity(context: ProbeDesignContext, proposal: _ProbeProposal) -> str:
    identity = {
        "cycle_id": context.cycle_id,
        "purpose": proposal.purpose.value,
        "targets": sorted(proposal.target_hypotheses),
        "goal": " ".join(proposal.inquiry_goal.casefold().split()),
        "capability": proposal.required_capability.value,
    }
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


def _active_hypothesis_ids(context: ProbeDesignContext) -> list[str]:
    active_ids = set(
        context.belief_state.frame_state.active_hypothesis_ids
        if context.belief_state.frame_state is not None
        else context.belief_state.hypotheses_by_id()
    )
    return [
        hypothesis.id
        for hypothesis in context.belief_state.hypotheses
        if hypothesis.id in active_ids
    ]


def _requires_initial_open_coverage(context: ProbeDesignContext) -> bool:
    return (
        context.belief_state.cycle_index == 0
        and context.task_frame.hypothesis_frame.coverage.value == "open"
    )


def _design_request_input(context: ProbeDesignContext) -> dict[str, Any]:
    return {
        "run_id": context.run_id,
        "cycle_id": context.cycle_id,
        "task_frame": {
            "task_kind": context.task_frame.task_kind.value,
            "normalized_question": context.task_frame.normalized_question,
            "task_context": context.task_frame.task_context,
            "answer_contract": context.task_frame.answer_contract.model_dump(
                mode="json"
            ),
            "competition": context.task_frame.hypothesis_frame.competition.value,
            "coverage": context.task_frame.hypothesis_frame.coverage.value,
            "coverage_statement": context.task_frame.hypothesis_frame.coverage_statement,
        },
        "hypotheses": [
            {
                "id": hypothesis.id,
                "statement": hypothesis.statement,
                "scope": hypothesis.scope,
                "posterior": hypothesis.posterior,
                "predictions": list(hypothesis.predictions),
                "falsifiers": list(hypothesis.falsifiers),
            }
            for hypothesis in context.belief_state.hypotheses
            if hypothesis.id in set(_active_hypothesis_ids(context))
        ],
        "available_capabilities": [
            descriptor.model_dump(mode="json")
            for descriptor in context.available_capabilities
        ],
    }


def _validation_message(error: Exception, *, stage: str) -> str:
    message = str(error)
    if "probe proposal must not contain secret material" in message:
        return f"{stage} response contains secret material"
    return f"{stage} response invalid"


__all__ = [
    "MODEL_REASONING_CAPABILITY",
    "FrameProbeDesigner",
    "ModelProbeDesigner",
    "ProbeDesignContext",
    "ProbeDesignError",
    "ProbeDesignResult",
    "ProbeDesigner",
]
