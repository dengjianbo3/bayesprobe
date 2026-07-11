from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
import math
import re
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)


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


class TaskKind(StrEnum):
    MULTIPLE_CHOICE = "multiple_choice"
    CLAIM_VERIFICATION = "claim_verification"
    EXPLANATION = "explanation"
    DIAGNOSIS = "diagnosis"
    DESIGN = "design"
    DECISION = "decision"


class HypothesisRelation(StrEnum):
    EXCLUSIVE_EXHAUSTIVE = "exclusive_exhaustive"
    INDEPENDENT = "independent"


class FramingMethod(StrEnum):
    EXPLICIT = "explicit"
    MODEL = "model"
    RECORDED = "recorded"
    LEGACY_MIGRATION = "legacy_migration"


def _required_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value.strip()


def _normalized_semantic_text(value: str) -> str:
    return " ".join(value.casefold().split())


_SECRET_VALUE_PATTERN = re.compile(r"sk-[A-Za-z0-9_-]{12,}")
_FORBIDDEN_SECRET_KEY_PARTS = ("apikey", "authorization", "token", "secret")


def is_secret_like_value(value: str) -> bool:
    return isinstance(value, str) and _SECRET_VALUE_PATTERN.search(value) is not None


def is_forbidden_secret_key_name(value: str) -> bool:
    if not isinstance(value, str):
        return False
    normalized_key = re.sub(r"[^a-z0-9]", "", value.casefold())
    return any(part in normalized_key for part in _FORBIDDEN_SECRET_KEY_PARTS)


def _reject_secret_string(value: str) -> None:
    if is_secret_like_value(value):
        raise ValueError("TaskFrame must not contain secret values")


def _reject_secret_material(value: Any) -> None:
    if isinstance(value, BaseModel):
        _reject_secret_material(value.__dict__)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("TaskFrame must contain only JSON-compatible values")
            _reject_secret_string(key)
            if is_forbidden_secret_key_name(key):
                raise ValueError("TaskFrame must not contain secret fields")
            _reject_secret_material(item)
    elif isinstance(value, list):
        for item in value:
            _reject_secret_material(item)
    elif isinstance(value, str):
        _reject_secret_string(value)
    elif value is not None and not isinstance(value, (bool, int, float, str)):
        raise ValueError("TaskFrame must contain only JSON-compatible values")


class StrictTaskModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AnswerChoice(StrictTaskModel):
    label: str
    text: str

    @field_validator("label")
    @classmethod
    def clean_label(cls, value: str) -> str:
        return _required_text(value, "answer choice label").upper()

    @field_validator("text")
    @classmethod
    def clean_text(cls, value: str) -> str:
        return _required_text(value, "answer choice text")


class AnswerContract(StrictTaskModel):
    objective: str
    required_sections: list[str]
    decision_form: str
    permits_synthesis: bool = False

    @field_validator("objective", "decision_form")
    @classmethod
    def clean_required_text(cls, value: str, info: ValidationInfo) -> str:
        return _required_text(value, info.field_name)

    @field_validator("required_sections")
    @classmethod
    def clean_required_sections(cls, value: list[str]) -> list[str]:
        sections = [_required_text(item, "required section") for item in value]
        if not sections or len(sections) != len(set(sections)):
            raise ValueError("required_sections must be non-empty and unique")
        return sections


class FramedHypothesis(StrictTaskModel):
    id: str
    statement: str
    type: str
    scope: str
    initial_prior: float
    falsifiers: list[str]
    predictions: list[str]

    @field_validator("id", "statement", "type", "scope")
    @classmethod
    def clean_required_text(cls, value: str, info: ValidationInfo) -> str:
        return _required_text(value, info.field_name)

    @field_validator("initial_prior")
    @classmethod
    def validate_initial_prior(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("initial_prior must be between zero and one")
        return value

    @field_validator("falsifiers", "predictions")
    @classmethod
    def clean_semantic_lists(cls, value: list[str], info: ValidationInfo) -> list[str]:
        items = [_required_text(item, info.field_name) for item in value]
        if not items:
            raise ValueError(f"{info.field_name} must not be empty")
        return items


class HypothesisFrame(StrictTaskModel):
    frame_id: str
    relation: HypothesisRelation
    hypotheses: list[FramedHypothesis]
    rival_sets: dict[str, list[str]]
    coverage_statement: str
    unresolved_alternative_mass: float | None = None
    coverage_limitation: str | None = None

    @field_validator("frame_id", "coverage_statement")
    @classmethod
    def clean_required_text(cls, value: str, info: ValidationInfo) -> str:
        return _required_text(value, info.field_name)

    @field_validator("coverage_limitation")
    @classmethod
    def clean_optional_text(cls, value: str | None) -> str | None:
        return None if value is None else _required_text(value, "coverage_limitation")

    @field_validator("unresolved_alternative_mass")
    @classmethod
    def validate_alternative_mass(cls, value: float | None) -> float | None:
        if value is not None and not 0 <= value <= 1:
            raise ValueError("unresolved_alternative_mass must be between zero and one")
        return value

    @model_validator(mode="after")
    def validate_frame(self) -> "HypothesisFrame":
        if not 1 <= len(self.hypotheses) <= 6:
            raise ValueError("hypothesis frame must contain between 1 and 6 hypotheses")
        ids = [item.id for item in self.hypotheses]
        if len(ids) != len(set(ids)):
            raise ValueError("hypothesis ids must be unique")
        statements = [_normalized_semantic_text(item.statement) for item in self.hypotheses]
        if len(statements) != len(set(statements)):
            raise ValueError("hypothesis statements must be semantically distinct")
        if set(self.rival_sets) != set(ids):
            raise ValueError("rival_sets must contain every hypothesis id exactly once")
        for hypothesis_id, rivals in self.rival_sets.items():
            unknown = set(rivals).difference(ids)
            if unknown:
                raise ValueError(f"unknown rival ids for {hypothesis_id}: {sorted(unknown)}")
            if hypothesis_id in rivals:
                raise ValueError("a hypothesis cannot rival itself")
        if self.relation == HypothesisRelation.EXCLUSIVE_EXHAUSTIVE:
            for hypothesis_id, rivals in self.rival_sets.items():
                if set(rivals) != set(ids).difference({hypothesis_id}):
                    raise ValueError("exclusive frames require all-to-all rival sets")
            if not math.isclose(
                sum(item.initial_prior for item in self.hypotheses),
                1.0,
                abs_tol=1e-6,
            ):
                raise ValueError("exclusive frame initial priors must sum to one")
        return self


class TaskFrame(StrictTaskModel):
    task_frame_id: str
    task_kind: TaskKind
    normalized_question: str
    task_context: str = ""
    answer_contract: AnswerContract
    hypothesis_frame: HypothesisFrame
    framing_method: FramingMethod
    framing_trace: dict[str, Any] = Field(default_factory=dict)

    @field_validator("task_frame_id", "normalized_question")
    @classmethod
    def clean_required_text(cls, value: str, info: ValidationInfo) -> str:
        return _required_text(value, info.field_name)

    @field_validator("task_context")
    @classmethod
    def clean_task_context(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("task_context must be a string")
        return value.strip()

    @model_validator(mode="after")
    def validate_frame(self) -> "TaskFrame":
        _reject_secret_material(self)
        if (
            self.framing_method != FramingMethod.LEGACY_MIGRATION
            and len(self.hypothesis_frame.hypotheses) < 2
        ):
            raise ValueError("new task frames require at least two hypotheses")
        if (
            self.task_kind == TaskKind.MULTIPLE_CHOICE
            and self.hypothesis_frame.relation != HypothesisRelation.EXCLUSIVE_EXHAUSTIVE
        ):
            raise ValueError("multiple-choice tasks require an exclusive frame")
        return self


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
    task_frame: TaskFrame | None = None

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
    model_trace: dict[str, Any] = Field(default_factory=dict)

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
