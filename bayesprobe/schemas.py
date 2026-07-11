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
    EXACT_ANSWER = "exact_answer"
    CLAIM_VERIFICATION = "claim_verification"
    EXPLANATION = "explanation"
    DIAGNOSIS = "diagnosis"
    DESIGN = "design"
    DECISION = "decision"


class HypothesisRelation(StrEnum):
    EXCLUSIVE_EXHAUSTIVE = "exclusive_exhaustive"
    INDEPENDENT = "independent"


class TaskAdmissionStatus(StrEnum):
    ADMITTED = "admitted"
    NEEDS_REFRAMING = "needs_reframing"
    OUT_OF_SCOPE = "out_of_scope"


class AnswerRelationship(StrEnum):
    SELECTION = "selection"
    SYNTHESIS = "synthesis"


class AnswerValueType(StrEnum):
    CHOICE_LABEL = "choice_label"
    INTEGER = "integer"
    NUMBER = "number"
    SHORT_TEXT = "short_text"
    STRUCTURED_TEXT = "structured_text"


class HypothesisCompetition(StrEnum):
    EXCLUSIVE = "exclusive"
    INDEPENDENT = "independent"


class HypothesisCoverage(StrEnum):
    EXHAUSTIVE = "exhaustive"
    OPEN = "open"


class FrameAdequacyStatus(StrEnum):
    PROVISIONAL = "provisional"
    ADEQUATE = "adequate"
    CHALLENGED = "challenged"
    INADEQUATE = "inadequate"
    EXPANDING = "expanding"


class FrameFit(StrEnum):
    EXPLAINED_BY_NAMED = "explained_by_named"
    UNDERDETERMINED = "underdetermined"
    SUPPORTS_UNRESOLVED = "supports_unresolved"


class ProjectionMode(StrEnum):
    SELECTION = "selection"
    SYNTHESIS = "synthesis"
    ABSTENTION = "abstention"


class EpistemicOrigin(StrEnum):
    EXTERNAL_OBSERVATION = "external_observation"
    RETRIEVED_SOURCE = "retrieved_source"
    TOOL_RESULT = "tool_result"
    MODEL_REASONING = "model_reasoning"
    HUMAN_INPUT = "human_input"
    AGENT_MESSAGE = "agent_message"
    DERIVED_SUMMARY = "derived_summary"


class ProbePurpose(StrEnum):
    HYPOTHESIS_DISCRIMINATION = "hypothesis_discrimination"
    HYPOTHESIS_FALSIFICATION = "hypothesis_falsification"
    FRAME_COVERAGE = "frame_coverage"
    SOURCE_VERIFICATION = "source_verification"
    ANOMALY_CLARIFICATION = "anomaly_clarification"
    ANSWER_CONTRACT_GAP = "answer_contract_gap"


class CapabilityKind(StrEnum):
    MODEL_REASONING = "model_reasoning"
    PYTHON_COMPUTATION = "python_computation"
    SEARCH = "search"
    DOCUMENT_RETRIEVAL = "document_retrieval"
    REPOSITORY_READ = "repository_read"
    TEST_EXECUTION = "test_execution"
    EXTERNAL_AGENT_REQUEST = "external_agent_request"
    HUMAN_REQUEST = "human_request"


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


_SECRET_VALUE_PATTERNS = (
    re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{12,}", re.IGNORECASE),
    re.compile(
        r"(?<![A-Za-z0-9_])(?:gh[pousr]_[A-Za-z0-9]{20,}|"
        r"github_pat_[A-Za-z0-9_]{20,})(?![A-Za-z0-9_])"
    ),
    re.compile(
        r"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{10,}\."
        r"[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}(?![A-Za-z0-9_-])"
    ),
    re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])"),
    re.compile(
        r"(?<![A-Za-z0-9])xox[a-z]-[A-Za-z0-9-]{10,}"
        r"(?![A-Za-z0-9-])",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:api[ _-]?key|access[ _-]?key|private[ _-]?key|password|passwd|"
        r"credential(?:s)?|cookie|secret|token)\b\s*(?:=|:)\s*[\"']?"
        r"(?:Bearer\s+)?[A-Za-z0-9._~+/=-]{6,}",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bauthorization\b\s*(?:=|:)\s*[\"']?Bearer\s+"
        r"[A-Za-z0-9._~+/=-]{8,}",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bBearer\s+(?=[A-Za-z0-9._~+/=-]{12,}(?:\s|$))"
        r"(?=[A-Za-z0-9._~+/=-]*[0-9._~+/=-])[A-Za-z0-9._~+/=-]{12,}",
        re.IGNORECASE,
    ),
    re.compile(r"\bBearer\s+[A-Za-z]{16,}\b", re.IGNORECASE),
    re.compile(
        r"-----BEGIN(?: [A-Z0-9]+)* PRIVATE KEY-----",
        re.IGNORECASE,
    ),
)
_FORBIDDEN_SECRET_KEY_COMPOUNDS = {
    "apikey",
    "accesskey",
    "privatekey",
    "proxyauthorization",
}
_FORBIDDEN_SECRET_KEY_WORDS = {
    "authorization",
    "token",
    "secret",
    "password",
    "passwd",
    "credential",
    "credentials",
    "cookie",
}
_BENIGN_SECRET_KEY_FOLLOWERS = {
    "token": {"count"},
    "password": {"policy"},
    "credential": {"score"},
    "credentials": {"score"},
    "cookie": {"policy"},
}
_FORBIDDEN_SECRET_KEY_SEQUENCES = {
    ("api", "key"),
    ("access", "key"),
    ("private", "key"),
}


def is_secret_like_value(value: str) -> bool:
    return isinstance(value, str) and any(
        pattern.search(value) is not None for pattern in _SECRET_VALUE_PATTERNS
    )


def is_forbidden_secret_key_name(value: str) -> bool:
    if not isinstance(value, str):
        return False
    separated = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    parts = re.findall(r"[a-z0-9]+", separated.casefold())
    if not parts:
        return False
    compact = "".join(parts)
    if compact in _FORBIDDEN_SECRET_KEY_COMPOUNDS:
        return True
    if any(
        tuple(parts[index:index + 2]) in _FORBIDDEN_SECRET_KEY_SEQUENCES
        for index in range(len(parts) - 1)
    ):
        return True
    for index, part in enumerate(parts):
        if part not in _FORBIDDEN_SECRET_KEY_WORDS:
            continue
        follower = parts[index + 1] if index + 1 < len(parts) else None
        if follower in _BENIGN_SECRET_KEY_FOLLOWERS.get(part, set()):
            continue
        return True
    return False


def redact_secret_material(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return redact_secret_material(value.model_dump(mode="python"))
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if is_forbidden_secret_key_name(key_text) or is_secret_like_value(key_text):
                continue
            sanitized[key_text] = redact_secret_material(item)
        return sanitized
    if isinstance(value, list | tuple):
        return [redact_secret_material(item) for item in value]
    if isinstance(value, str):
        return "[REDACTED]" if is_secret_like_value(value) else value
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return "[UNSUPPORTED]"


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


def _unique_semantic_texts(value: list[str], field_name: str) -> list[str]:
    items = [_required_text(item, field_name) for item in value]
    normalized = [_normalized_semantic_text(item) for item in items]
    if not items or len(normalized) != len(set(normalized)):
        raise ValueError(f"{field_name} must be non-empty and unique")
    return items


def _unique_optional_semantic_texts(
    value: list[str],
    field_name: str,
) -> list[str]:
    if not value:
        return []
    return _unique_semantic_texts(value, field_name)


class AnswerContractOutline(StrictTaskModel):
    objective: str
    answer_value_type: AnswerValueType
    decision_form: str
    permits_synthesis: bool
    required_sections: list[str]

    @field_validator("objective", "decision_form")
    @classmethod
    def clean_required_text(cls, value: str, info: ValidationInfo) -> str:
        return _required_text(value, info.field_name)

    @field_validator("required_sections")
    @classmethod
    def clean_required_sections(cls, value: list[str]) -> list[str]:
        return _unique_semantic_texts(value, "required_sections")


class TaskAdmissionDecision(StrictTaskModel):
    attempt_id: str
    status: TaskAdmissionStatus
    epistemic_basis: list[str]
    proposed_task_kind: TaskKind | None = None
    answer_contract_outline: AnswerContractOutline | None = None
    clarification_questions: list[str] = Field(default_factory=list)
    reason: str
    model_trace: dict[str, Any] = Field(default_factory=dict)

    @field_validator("attempt_id", "reason")
    @classmethod
    def clean_required_text(cls, value: str, info: ValidationInfo) -> str:
        return _required_text(value, info.field_name)

    @field_validator("epistemic_basis")
    @classmethod
    def clean_epistemic_basis(cls, value: list[str]) -> list[str]:
        return _unique_semantic_texts(value, "epistemic_basis")

    @field_validator("clarification_questions")
    @classmethod
    def clean_clarification_questions(cls, value: list[str]) -> list[str]:
        if not value:
            return []
        return _unique_semantic_texts(value, "clarification_questions")

    @model_validator(mode="after")
    def validate_status_contract(self) -> "TaskAdmissionDecision":
        _reject_secret_material(self)
        if self.status == TaskAdmissionStatus.ADMITTED:
            if self.proposed_task_kind is None:
                raise ValueError("admitted decisions require proposed_task_kind")
            if self.answer_contract_outline is None:
                raise ValueError("admitted decisions require answer_contract_outline")
        elif self.status == TaskAdmissionStatus.NEEDS_REFRAMING:
            if not self.clarification_questions:
                raise ValueError(
                    "needs_reframing decisions require clarification_questions"
                )
        elif self.proposed_task_kind is not None:
            raise ValueError("out_of_scope decisions must not propose a task kind")
        elif self.answer_contract_outline is not None:
            raise ValueError("out_of_scope decisions must not include an answer contract")
        return self


class FrameState(StrictTaskModel):
    frame_id: str
    frame_version: int = 1
    parent_frame_version: int | None = None
    competition: HypothesisCompetition
    coverage: HypothesisCoverage
    active_hypothesis_ids: list[str]
    unresolved_alternative_mass: float | None = None
    adequacy_status: FrameAdequacyStatus
    revision_reason: str | None = None
    trigger_event_ids: list[str] = Field(default_factory=list)
    revision_count: int = 0

    @field_validator("frame_id")
    @classmethod
    def clean_frame_id(cls, value: str) -> str:
        return _required_text(value, "frame_id")

    @field_validator("active_hypothesis_ids")
    @classmethod
    def clean_active_ids(cls, value: list[str]) -> list[str]:
        return _unique_semantic_texts(value, "active_hypothesis_ids")

    @field_validator("trigger_event_ids")
    @classmethod
    def clean_trigger_ids(cls, value: list[str]) -> list[str]:
        if not value:
            return []
        return _unique_semantic_texts(value, "trigger_event_ids")

    @field_validator("revision_reason")
    @classmethod
    def clean_revision_reason(cls, value: str | None) -> str | None:
        return None if value is None else _required_text(value, "revision_reason")

    @field_validator("unresolved_alternative_mass")
    @classmethod
    def validate_unresolved_mass(cls, value: float | None) -> float | None:
        if value is not None and not 0 <= value <= 1:
            raise ValueError("unresolved_alternative_mass must be between zero and one")
        return value

    @model_validator(mode="after")
    def validate_versions_and_mass(self) -> "FrameState":
        if self.frame_version < 1 or self.revision_count < 0:
            raise ValueError("frame versions and revision count must be non-negative")
        if self.competition == HypothesisCompetition.INDEPENDENT:
            if self.unresolved_alternative_mass is not None:
                raise ValueError("independent frames do not use shared unresolved mass")
        elif (
            self.coverage == HypothesisCoverage.EXHAUSTIVE
            and self.unresolved_alternative_mass not in {None, 0.0}
        ):
            raise ValueError("unresolved mass is legal only for exclusive-open frames")
        return self


class EvidenceMemorySnapshot(StrictTaskModel):
    memory_version: int = 1
    accepted_evidence_ids: list[str] = Field(default_factory=list)
    content_fingerprints: dict[str, str] = Field(default_factory=dict)
    source_content_fingerprints: dict[str, str] = Field(default_factory=dict)
    derivation_roots: dict[str, str] = Field(default_factory=dict)
    correlation_credit: dict[str, float] = Field(default_factory=dict)
    discovery_evidence_ids: list[str] = Field(default_factory=list)
    counterevidence_ids_by_hypothesis: dict[str, list[str]] = Field(default_factory=dict)
    discard_and_schema_history: list[str] = Field(default_factory=list)

    @field_validator("memory_version")
    @classmethod
    def validate_memory_version(cls, value: int) -> int:
        if value < 1:
            raise ValueError("memory_version must be at least one")
        return value

    @field_validator(
        "accepted_evidence_ids",
        "discovery_evidence_ids",
        "discard_and_schema_history",
    )
    @classmethod
    def clean_identity_lists(cls, value: list[str], info: ValidationInfo) -> list[str]:
        return _unique_optional_semantic_texts(value, info.field_name)

    @field_validator("counterevidence_ids_by_hypothesis")
    @classmethod
    def clean_counterevidence_ids(
        cls,
        value: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        return {
            _required_text(hypothesis_id, "counterevidence hypothesis id"):
            _unique_optional_semantic_texts(
                evidence_ids,
                "counterevidence_ids_by_hypothesis",
            )
            for hypothesis_id, evidence_ids in value.items()
        }


class SignalProvenance(StrictTaskModel):
    epistemic_origin: EpistemicOrigin
    source_identity: str
    provider_model_or_tool_identity: str | None = None
    session_id: str | None = None
    parent_signal_ids: list[str] = Field(default_factory=list)
    derivation_root_id: str
    correlation_group: str
    canonical_content_fingerprint: str
    citations: list[str] = Field(default_factory=list)
    artifact_refs: list[str] = Field(default_factory=list)
    environment_state_id: str | None = None

    @field_validator(
        "source_identity",
        "derivation_root_id",
        "correlation_group",
        "canonical_content_fingerprint",
    )
    @classmethod
    def clean_required_text(cls, value: str, info: ValidationInfo) -> str:
        return _required_text(value, info.field_name)

    @field_validator("parent_signal_ids", "citations", "artifact_refs")
    @classmethod
    def clean_identity_lists(cls, value: list[str], info: ValidationInfo) -> list[str]:
        return _unique_optional_semantic_texts(value, info.field_name)


class FrameMassUpdate(StrictTaskModel):
    update_id: str
    cycle_id: str
    evidence_id: str
    prior: float
    posterior: float
    direction: UpdateDirection
    reason: str

    @field_validator("update_id", "cycle_id", "evidence_id", "reason")
    @classmethod
    def clean_required_text(cls, value: str, info: ValidationInfo) -> str:
        return _required_text(value, info.field_name)

    @field_validator("prior", "posterior")
    @classmethod
    def validate_mass(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("frame mass must be between zero and one")
        return value


class CapabilityDescriptor(StrictTaskModel):
    kind: CapabilityKind
    available: bool
    cost_class: str = "bounded"
    latency_class: str = "interactive"
    epistemic_origin: EpistemicOrigin = EpistemicOrigin.MODEL_REASONING
    quality_caps: dict[str, float] = Field(default_factory=dict)
    executor_adapter_id: str

    @field_validator("cost_class", "latency_class", "executor_adapter_id")
    @classmethod
    def clean_required_text(cls, value: str, info: ValidationInfo) -> str:
        return _required_text(value, info.field_name)


class CapabilityDecision(StrictTaskModel):
    kind: CapabilityKind
    available: bool
    descriptor: CapabilityDescriptor | None
    reason: str

    @field_validator("reason")
    @classmethod
    def clean_reason(cls, value: str) -> str:
        return _required_text(value, "reason")


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
    answer_value_type: AnswerValueType = AnswerValueType.STRUCTURED_TEXT
    answer_format: str = "structured text"
    required_sections: list[str]
    decision_form: str
    permits_synthesis: bool = False

    @field_validator("objective", "answer_format", "decision_form")
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
    answer_value: str | int | float | None = None

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
        return _unique_semantic_texts(value, info.field_name)


def _relation_mapping(
    relation: HypothesisRelation,
) -> tuple[HypothesisCompetition, HypothesisCoverage]:
    if relation == HypothesisRelation.EXCLUSIVE_EXHAUSTIVE:
        return HypothesisCompetition.EXCLUSIVE, HypothesisCoverage.EXHAUSTIVE
    if relation == HypothesisRelation.INDEPENDENT:
        return HypothesisCompetition.INDEPENDENT, HypothesisCoverage.OPEN
    raise ValueError(f"unsupported legacy relation: {relation}")


class HypothesisFrame(StrictTaskModel):
    frame_id: str
    competition: HypothesisCompetition
    coverage: HypothesisCoverage
    hypotheses: list[FramedHypothesis]
    rival_sets: dict[str, list[str]]
    coverage_statement: str
    unresolved_alternative_mass: float | None = None
    coverage_limitation: str | None = None

    @model_validator(mode="before")
    @classmethod
    def consume_legacy_relation(cls, value: Any) -> Any:
        if not isinstance(value, Mapping) or "relation" not in value:
            return value
        payload = dict(value)
        if "competition" in payload or "coverage" in payload:
            raise ValueError(
                "legacy relation cannot be combined with competition or coverage"
            )
        relation = HypothesisRelation(payload.pop("relation"))
        competition, coverage = _relation_mapping(relation)
        payload["competition"] = competition
        payload["coverage"] = coverage
        return payload

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
        if self.competition == HypothesisCompetition.EXCLUSIVE:
            for hypothesis_id, rivals in self.rival_sets.items():
                if set(rivals) != set(ids).difference({hypothesis_id}):
                    raise ValueError("exclusive frames require all-to-all rival sets")
            unresolved = self.unresolved_alternative_mass or 0.0
            if not math.isclose(
                sum(item.initial_prior for item in self.hypotheses) + unresolved,
                1.0,
                abs_tol=1e-6,
            ):
                raise ValueError("exclusive named and unresolved mass must sum to one")
            if (
                self.coverage == HypothesisCoverage.EXHAUSTIVE
                and unresolved != 0.0
            ):
                raise ValueError("unresolved mass is legal only for exclusive-open frames")
        elif self.unresolved_alternative_mass is not None:
            raise ValueError("independent frames do not use shared unresolved mass")
        return self

    @property
    def relation(self) -> HypothesisRelation:
        if self.competition == HypothesisCompetition.INDEPENDENT:
            return HypothesisRelation.INDEPENDENT
        if self.coverage == HypothesisCoverage.EXHAUSTIVE:
            return HypothesisRelation.EXCLUSIVE_EXHAUSTIVE
        raise ValueError("exclusive-open frames have no legacy hypothesis relation")


class TaskFrame(StrictTaskModel):
    schema_version: Literal["v0.1", "v0.2"] = "v0.1"
    task_frame_id: str
    admission_decision_id: str | None = None
    task_kind: TaskKind
    answer_relationship: AnswerRelationship | None = None
    normalized_question: str
    task_context: str = ""
    answer_contract: AnswerContract
    hypothesis_frame: HypothesisFrame
    framing_method: FramingMethod
    framing_trace: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def require_explicit_v02_fields(cls, value: Any) -> Any:
        if not isinstance(value, Mapping) or value.get("schema_version") != "v0.2":
            return value
        contract = value.get("answer_contract")
        if isinstance(contract, AnswerContract):
            contract_data = contract.model_dump(mode="python")
            explicit_fields = contract.model_fields_set
        else:
            contract_data = contract
            explicit_fields = set(contract) if isinstance(contract, Mapping) else set()
        if not isinstance(contract_data, Mapping):
            return value
        if "answer_value_type" not in explicit_fields:
            raise ValueError("v0.2 answer contract requires answer_value_type")
        if "answer_format" not in explicit_fields:
            raise ValueError("v0.2 answer contract requires answer_format")
        return value

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
            self.schema_version == "v0.1"
            and
            self.framing_method != FramingMethod.LEGACY_MIGRATION
            and len(self.hypothesis_frame.hypotheses) < 2
        ):
            raise ValueError("new task frames require at least two hypotheses")
        if (
            self.task_kind == TaskKind.MULTIPLE_CHOICE
            and (
                self.hypothesis_frame.competition != HypothesisCompetition.EXCLUSIVE
                or self.hypothesis_frame.coverage != HypothesisCoverage.EXHAUSTIVE
            )
        ):
            raise ValueError("multiple-choice tasks require an exclusive frame")
        if self.schema_version == "v0.2":
            if self.admission_decision_id is None:
                raise ValueError("v0.2 task frame requires admission_decision_id")
            self.admission_decision_id = _required_text(
                self.admission_decision_id, "admission_decision_id"
            )
            if self.answer_relationship is None:
                raise ValueError("v0.2 task frame requires answer_relationship")
            if (
                self.task_kind
                in {TaskKind.MULTIPLE_CHOICE, TaskKind.EXACT_ANSWER}
                and self.answer_relationship != AnswerRelationship.SELECTION
            ):
                raise ValueError("answer candidate tasks require selection")
            if self.task_kind == TaskKind.EXACT_ANSWER:
                if (
                    self.hypothesis_frame.competition
                    != HypothesisCompetition.EXCLUSIVE
                    or self.hypothesis_frame.coverage != HypothesisCoverage.OPEN
                ):
                    raise ValueError(
                        "exact-answer tasks require an exclusive-open frame"
                    )
            elif (
                self.framing_method != FramingMethod.LEGACY_MIGRATION
                and len(self.hypothesis_frame.hypotheses) < 2
            ):
                raise ValueError("new task frames require at least two hypotheses")
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
    applied_complexity_penalty: float = 0.0
    applied_ad_hoc_penalty: float = 0.0
    created_by: Literal["initial", "spawned", "split", "reframed"] = "initial"
    why_existing_hypotheses_failed: str | None = None
    answer_value: str | int | float | None = None

    @field_validator(
        "prior",
        "posterior",
        "complexity_penalty",
        "ad_hoc_penalty",
        "applied_complexity_penalty",
        "applied_ad_hoc_penalty",
    )
    @classmethod
    def probability_like(cls, value: float) -> float:
        if value < 0 or value > 1:
            raise ValueError("value must be between 0 and 1")
        return value


class BeliefState(BaseModel):
    schema_version: Literal["v0.1", "v0.2"] = "v0.1"
    belief_state_id: str
    run_id: str
    cycle_id: str
    cycle_index: int = 0
    hypotheses: list[Hypothesis]
    posterior_summary: dict[str, Any] = Field(default_factory=dict)
    uncertainty_summary: str = ""
    ledger_refs: dict[str, list[str]] = Field(default_factory=dict)
    task_frame: TaskFrame | None = None
    frame_state: FrameState | None = None
    evidence_memory: EvidenceMemorySnapshot | None = None

    @model_validator(mode="after")
    def validate_v02_lifecycle_state(self) -> "BeliefState":
        if self.schema_version == "v0.2" and self.frame_state is None:
            raise ValueError("v0.2 belief state requires frame_state")
        if self.schema_version == "v0.2" and self.evidence_memory is None:
            raise ValueError("v0.2 belief state requires evidence_memory")
        if self.schema_version == "v0.2" and self.task_frame is None:
            raise ValueError("v0.2 belief state requires task_frame")
        if (
            self.schema_version == "v0.2"
            and self.task_frame is not None
            and self.task_frame.schema_version != "v0.2"
        ):
            raise ValueError("v0.2 belief state requires a v0.2 task_frame")
        return self

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
    provenance: SignalProvenance | None = None


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
    unresolved_likelihood: LikelihoodBand | None = None
    frame_fit: FrameFit = FrameFit.UNDERDETERMINED
    unexplained_observation: str | None = None
    correlation_status: str = "unassessed"
    effective_update_weight: float | None = None
    interpretation: str = ""
    discard_reason: str | None = None
    model_trace: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "reliability",
        "independence",
        "relevance",
        "novelty",
        "specificity",
        "verifiability",
        "effective_update_weight",
    )
    @classmethod
    def score_between_zero_and_one(cls, value: float | None) -> float | None:
        if value is None:
            return None
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
