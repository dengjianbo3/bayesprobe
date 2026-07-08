from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RunRegime(StrEnum):
    AUTONOMOUS = "autonomous"
    SYNCHRONIZED = "synchronized"


class RunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class CycleSignalShape(StrEnum):
    ACTIVE_ONLY = "active_only"
    PASSIVE_ONLY = "passive_only"
    ACTIVE_PLUS_PASSIVE = "active_plus_passive"


class BoundaryStatus(StrEnum):
    OPEN = "open"
    CLOSED = "closed"
    INTEGRATED = "integrated"


class HypothesisStatus(StrEnum):
    ACTIVE = "active"
    WEAKENED = "weakened"
    REFRAMED = "reframed"
    SPLIT = "split"
    RETIRED = "retired"
    ARCHIVED = "archived"


class SignalKind(StrEnum):
    ACTIVE = "active"
    PASSIVE = "passive"


class SignalInboxStatus(StrEnum):
    ACCEPTED = "accepted"
    DEFERRED = "deferred"


class EvidenceType(StrEnum):
    SUPPORTING = "supporting"
    COUNTEREVIDENCE = "counterevidence"
    BOUNDARY_CONDITION = "boundary_condition"
    ANOMALY = "anomaly"
    NEUTRAL = "neutral"
    SOURCE_CLAIM = "source_claim"
    SENDER_JUDGMENT = "sender_judgment"


class LikelihoodBand(StrEnum):
    STRONGLY_DISCONFIRMING = "strongly_disconfirming"
    MODERATELY_DISCONFIRMING = "moderately_disconfirming"
    WEAKLY_DISCONFIRMING = "weakly_disconfirming"
    NEUTRAL = "neutral"
    WEAKLY_CONFIRMING = "weakly_confirming"
    MODERATELY_CONFIRMING = "moderately_confirming"
    STRONGLY_CONFIRMING = "strongly_confirming"


class UpdateDirection(StrEnum):
    STRENGTHENED = "strengthened"
    WEAKENED = "weakened"
    NEUTRAL = "neutral"


class EvolutionOperation(StrEnum):
    SPAWN = "spawn"
    SPLIT = "split"
    MERGE = "merge"
    REFRAME = "reframe"
    REJECT = "reject"
    RETIRE = "retire"
    REACTIVATE = "reactivate"


class RunBudget(BaseModel):
    max_cycles: int = 5
    max_tool_calls: int = 20
    max_tokens: int | None = None
    max_cost: float | None = None


class RunRecord(BaseModel):
    run_id: str
    regime: RunRegime
    problem: str
    status: RunStatus = RunStatus.RUNNING
    current_cycle_id: str | None = None
    budget: RunBudget = Field(default_factory=RunBudget)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CycleRecord(BaseModel):
    cycle_id: str
    run_id: str
    cycle_index: int
    signal_shape: CycleSignalShape
    round_id: str | None = None
    boundary_status: BoundaryStatus = BoundaryStatus.OPEN
    started_at: datetime = Field(default_factory=utc_now)
    boundary_closed_at: datetime | None = None
    completed_at: datetime | None = None
    controller_metadata: dict[str, Any] = Field(default_factory=dict)


class Hypothesis(BaseModel):
    id: str
    statement: str
    scope: str
    prior: float
    posterior: float
    type: str = "claim"
    status: HypothesisStatus = HypothesisStatus.ACTIVE
    rivals: list[str] = Field(default_factory=list)
    falsifiers: list[str] = Field(default_factory=list)
    predictions: list[str] = Field(default_factory=list)
    complexity_penalty: float = 0.0
    ad_hoc_penalty: float = 0.0
    created_by: Literal["initial", "spawned", "split", "reframed"] = "initial"
    why_existing_hypotheses_failed: str | None = None

    @field_validator("prior", "posterior", "complexity_penalty", "ad_hoc_penalty")
    @classmethod
    def probability_like(cls, value: float) -> float:
        if value < 0 or value > 1:
            raise ValueError("value must be between 0 and 1")
        return value


class BeliefState(BaseModel):
    belief_state_id: str
    run_id: str
    cycle_id: str
    cycle_index: int = 0
    hypotheses: list[Hypothesis]
    posterior_summary: dict[str, Any] = Field(default_factory=dict)
    uncertainty_summary: str = ""
    ledger_refs: dict[str, list[str]] = Field(default_factory=dict)

    def hypotheses_by_id(self) -> dict[str, Hypothesis]:
        return {hypothesis.id: hypothesis for hypothesis in self.hypotheses}


class ProbeDesign(BaseModel):
    id: str
    cycle_id: str
    target_hypotheses: list[str]
    inquiry_goal: str
    method: str
    probe_type: str = "discriminative_test"
    support_condition: dict[str, str] = Field(default_factory=dict)
    weaken_condition: dict[str, str] = Field(default_factory=dict)
    reframe_condition: dict[str, str] | None = None
    expected_information_gain: float = 0.5
    decision_relevance: float = 0.5
    cost_estimate: float = 0.5
    priority: float = 0.5
    status: str = "candidate"

    @field_validator("expected_information_gain", "decision_relevance", "cost_estimate", "priority")
    @classmethod
    def score_between_zero_and_one(cls, value: float) -> float:
        if value < 0 or value > 1:
            raise ValueError("score must be between 0 and 1")
        return value


class ProbeSet(BaseModel):
    probe_set_id: str
    cycle_id: str
    probes: list[ProbeDesign] = Field(default_factory=list)
    boundary_id: str | None = None
    selection_reason: str
    budget_allocated: dict[str, int | float] = Field(default_factory=dict)
    may_be_empty: bool = False


class ProbeCandidate(BaseModel):
    candidate_id: str
    source: Literal["change_my_mind", "uncertainty", "anomaly", "passive_signal", "manual"]
    candidate_probe: ProbeDesign
    priority_features: dict[str, Any] = Field(default_factory=dict)
    selected_in_cycle: str | None = None


class ChangeMyMindCondition(BaseModel):
    human_readable_condition: str
    structured_probe_candidates: list[ProbeCandidate] = Field(default_factory=list)


class ExternalSignal(BaseModel):
    id: str
    cycle_id: str
    signal_kind: SignalKind
    source_type: str
    source: str
    raw_content: str
    generated_by_probe: str | None = None
    received_at: datetime = Field(default_factory=utc_now)
    inbox_status: SignalInboxStatus = SignalInboxStatus.ACCEPTED
    initial_target_hypotheses: list[str] = Field(default_factory=list)


class EvidenceEvent(BaseModel):
    id: str
    derived_from_signal: str
    target_hypotheses: list[str]
    evidence_type: EvidenceType
    content: str
    reliability: float = 0.5
    independence: float = 0.5
    relevance: float = 0.5
    novelty: float = 0.5
    specificity: float = 0.5
    verifiability: float = 0.5
    likelihoods: dict[str, LikelihoodBand] = Field(default_factory=dict)
    interpretation: str = ""
    discard_reason: str | None = None

    @field_validator("reliability", "independence", "relevance", "novelty", "specificity", "verifiability")
    @classmethod
    def score_between_zero_and_one(cls, value: float) -> float:
        if value < 0 or value > 1:
            raise ValueError("quality score must be between 0 and 1")
        return value


class BeliefUpdate(BaseModel):
    update_id: str
    cycle_id: str
    evidence_id: str
    hypothesis_id: str
    prior: float
    posterior: float
    direction: UpdateDirection
    reason: str
    sensitivity: dict[str, Any] = Field(default_factory=dict)


class HypothesisEvolution(BaseModel):
    evolution_id: str
    cycle_id: str
    operation: EvolutionOperation
    from_hypothesis: str | None = None
    to_hypothesis: str | None = None
    triggered_by: list[str] = Field(default_factory=list)
    reason: str
    audit_fields: dict[str, Any] = Field(default_factory=dict)


class AnswerProjection(BaseModel):
    answer: str
    current_best_hypothesis: str
    posterior_summary: str
    main_uncertainty: str
    weakest_assumption: str
    main_evidence_events: list[str]
    change_my_mind_condition: ChangeMyMindCondition
    answer_utility_notes: str = ""


class BeliefStateProjection(BaseModel):
    current_best_hypothesis: str
    posterior_or_confidence_interval: str
    main_evidence_events: list[str]
    main_uncertainties: list[str]
    questions_for_others: list[str]
    change_my_mind_condition: ChangeMyMindCondition
    requested_signal_type: str
    cited_sources: list[str] = Field(default_factory=list)
    projection_metadata: dict[str, Any] = Field(default_factory=dict)
