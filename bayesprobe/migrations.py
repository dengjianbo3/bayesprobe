from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import hashlib
import hmac
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from bayesprobe.schemas import (
    AnswerContract,
    AnswerRelationship,
    AnswerValueType,
    BeliefState,
    EvidenceMemorySnapshot,
    FramedHypothesis,
    FramingMethod,
    FrameAdequacyStatus,
    FrameState,
    Hypothesis,
    HypothesisCompetition,
    HypothesisCoverage,
    HypothesisFrame,
    HypothesisRelation,
    HypothesisStatus,
    TaskFrame,
    TaskKind,
)


BELIEF_STATE_V01_TO_V02_MIGRATION_MARKER = "belief_state_v0.1_to_v0.2"
TASK_FRAME_V01_TO_V02_MIGRATION_MARKER = "task_frame_v0.1_to_v0.2"
RECOGNIZED_V01_TO_V02_MIGRATION_MARKERS = frozenset(
    {
        BELIEF_STATE_V01_TO_V02_MIGRATION_MARKER,
        TASK_FRAME_V01_TO_V02_MIGRATION_MARKER,
    }
)


@dataclass(frozen=True)
class _V01MigrationReceipt:
    envelope_digest: str

    def __deepcopy__(self, memo: dict[int, Any]) -> "_V01MigrationReceipt":
        return self


def _migration_envelope_digest(state: BeliefState) -> str:
    payload = json.dumps(
        state.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _mark_v01_migration(state: BeliefState) -> BeliefState:
    state._v01_migration_receipt = _V01MigrationReceipt(
        envelope_digest=_migration_envelope_digest(state)
    )
    return state


def _has_v01_migration_receipt(state: BeliefState) -> bool:
    receipt = state._v01_migration_receipt
    if not isinstance(receipt, _V01MigrationReceipt):
        return False
    try:
        current_digest = _migration_envelope_digest(state)
    except (TypeError, ValueError):
        return False
    return hmac.compare_digest(receipt.envelope_digest, current_digest)


def _carry_v01_migration_receipt(
    source: BeliefState,
    target: BeliefState,
) -> BeliefState:
    source_frame = source.task_frame
    target_frame = target.task_frame
    if (
        _has_v01_migration_receipt(source)
        and source_frame is not None
        and target_frame is not None
        and source_frame.framing_method == FramingMethod.LEGACY_MIGRATION
        and target_frame.framing_method == FramingMethod.LEGACY_MIGRATION
        and source_frame.framing_trace.get("migration")
        == target_frame.framing_trace.get("migration")
    ):
        _mark_v01_migration(target)
    return target


class _V01Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _V01TaskKind(StrEnum):
    MULTIPLE_CHOICE = "multiple_choice"
    CLAIM_VERIFICATION = "claim_verification"
    EXPLANATION = "explanation"
    DIAGNOSIS = "diagnosis"
    DESIGN = "design"
    DECISION = "decision"


class _V01AnswerContract(_V01Model):
    objective: str
    required_sections: list[str]
    decision_form: str
    permits_synthesis: bool = False


class _V01FramedHypothesis(_V01Model):
    id: str
    statement: str
    type: str
    scope: str
    initial_prior: float
    falsifiers: list[str]
    predictions: list[str]


class _V01HypothesisFrame(_V01Model):
    frame_id: str
    relation: HypothesisRelation
    hypotheses: list[_V01FramedHypothesis]
    rival_sets: dict[str, list[str]]
    coverage_statement: str
    unresolved_alternative_mass: float | None = None
    coverage_limitation: str | None = None


class _V01TaskFrame(_V01Model):
    task_frame_id: str
    task_kind: _V01TaskKind
    normalized_question: str
    task_context: str = ""
    answer_contract: _V01AnswerContract
    hypothesis_frame: _V01HypothesisFrame
    framing_method: FramingMethod
    framing_trace: dict[str, Any] = Field(default_factory=dict)


class _V01Hypothesis(_V01Model):
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


class _V01BeliefState(_V01Model):
    belief_state_id: str
    run_id: str
    cycle_id: str
    cycle_index: int = 0
    hypotheses: list[_V01Hypothesis]
    posterior_summary: dict[str, Any] = Field(default_factory=dict)
    uncertainty_summary: str = ""
    ledger_refs: dict[str, list[str]] = Field(default_factory=dict)
    task_frame: _V01TaskFrame | None = None


def _relation_mapping(
    relation: HypothesisRelation,
) -> tuple[HypothesisCompetition, HypothesisCoverage]:
    if relation == HypothesisRelation.EXCLUSIVE_EXHAUSTIVE:
        return HypothesisCompetition.EXCLUSIVE, HypothesisCoverage.EXHAUSTIVE
    if relation == HypothesisRelation.INDEPENDENT:
        return HypothesisCompetition.INDEPENDENT, HypothesisCoverage.OPEN
    raise ValueError(f"unsupported legacy relation: {relation}")


def _legacy_task_frame_payload(frame: TaskFrame) -> dict[str, Any]:
    if frame.schema_version != "v0.1":
        raise ValueError("explicit v0.1 migration requires a v0.1 task frame")
    return {
        "task_frame_id": frame.task_frame_id,
        "task_kind": frame.task_kind,
        "normalized_question": frame.normalized_question,
        "task_context": frame.task_context,
        "answer_contract": {
            "objective": frame.answer_contract.objective,
            "required_sections": list(frame.answer_contract.required_sections),
            "decision_form": frame.answer_contract.decision_form,
            "permits_synthesis": frame.answer_contract.permits_synthesis,
        },
        "hypothesis_frame": {
            "frame_id": frame.hypothesis_frame.frame_id,
            "relation": frame.hypothesis_frame.relation,
            "hypotheses": [
                {
                    "id": item.id,
                    "statement": item.statement,
                    "type": item.type,
                    "scope": item.scope,
                    "initial_prior": item.initial_prior,
                    "falsifiers": list(item.falsifiers),
                    "predictions": list(item.predictions),
                }
                for item in frame.hypothesis_frame.hypotheses
            ],
            "rival_sets": {
                key: list(value)
                for key, value in frame.hypothesis_frame.rival_sets.items()
            },
            "coverage_statement": frame.hypothesis_frame.coverage_statement,
            "unresolved_alternative_mass": (
                frame.hypothesis_frame.unresolved_alternative_mass
            ),
            "coverage_limitation": frame.hypothesis_frame.coverage_limitation,
        },
        "framing_method": frame.framing_method,
        "framing_trace": dict(frame.framing_trace),
    }


def _legacy_belief_state_payload(state: BeliefState) -> dict[str, Any]:
    if state.schema_version != "v0.1":
        raise ValueError("explicit v0.1 migration requires a v0.1 belief state")
    return {
        "belief_state_id": state.belief_state_id,
        "run_id": state.run_id,
        "cycle_id": state.cycle_id,
        "cycle_index": state.cycle_index,
        "hypotheses": [
            item.model_dump(mode="python", exclude={"answer_value"})
            for item in state.hypotheses
        ],
        "posterior_summary": dict(state.posterior_summary),
        "uncertainty_summary": state.uncertainty_summary,
        "ledger_refs": {
            key: list(value) for key, value in state.ledger_refs.items()
        },
        "task_frame": (
            None
            if state.task_frame is None
            else _legacy_task_frame_payload(state.task_frame)
        ),
    }


def migrate_task_frame_v0_1(payload: Any) -> TaskFrame:
    raw_payload = (
        _legacy_task_frame_payload(payload)
        if isinstance(payload, TaskFrame)
        else payload
    )
    legacy = _V01TaskFrame.model_validate(raw_payload)
    competition, coverage = _relation_mapping(legacy.hypothesis_frame.relation)
    unresolved_mass = (
        0.0 if competition == HypothesisCompetition.EXCLUSIVE else None
    )
    answer_value_type = (
        AnswerValueType.CHOICE_LABEL
        if legacy.task_kind == TaskKind.MULTIPLE_CHOICE
        else AnswerValueType.STRUCTURED_TEXT
    )
    answer_format = (
        "choice label"
        if answer_value_type == AnswerValueType.CHOICE_LABEL
        else "structured text"
    )
    answer_relationship = (
        AnswerRelationship.SYNTHESIS
        if legacy.answer_contract.permits_synthesis
        else AnswerRelationship.SELECTION
    )
    return TaskFrame(
        schema_version="v0.2",
        task_frame_id=legacy.task_frame_id,
        admission_decision_id=f"{legacy.task_frame_id}_migration_admission",
        task_kind=TaskKind(legacy.task_kind.value),
        answer_relationship=answer_relationship,
        normalized_question=legacy.normalized_question,
        task_context=legacy.task_context,
        answer_contract=AnswerContract(
            objective=legacy.answer_contract.objective,
            answer_value_type=answer_value_type,
            answer_format=answer_format,
            required_sections=legacy.answer_contract.required_sections,
            decision_form=legacy.answer_contract.decision_form,
            permits_synthesis=legacy.answer_contract.permits_synthesis,
        ),
        hypothesis_frame=HypothesisFrame(
            frame_id=legacy.hypothesis_frame.frame_id,
            competition=competition,
            coverage=coverage,
            hypotheses=[
                FramedHypothesis(
                    **item.model_dump(),
                    answer_value=(
                        item.id
                        if answer_value_type == AnswerValueType.CHOICE_LABEL
                        else None
                    ),
                )
                for item in legacy.hypothesis_frame.hypotheses
            ],
            rival_sets=legacy.hypothesis_frame.rival_sets,
            coverage_statement=legacy.hypothesis_frame.coverage_statement,
            unresolved_alternative_mass=unresolved_mass,
            coverage_limitation=legacy.hypothesis_frame.coverage_limitation,
        ),
        framing_method=FramingMethod.LEGACY_MIGRATION,
        framing_trace={
            **legacy.framing_trace,
            "migration": TASK_FRAME_V01_TO_V02_MIGRATION_MARKER,
        },
    )


def _categorical_task_frame(legacy: _V01BeliefState) -> TaskFrame:
    if not legacy.hypotheses:
        raise ValueError("legacy belief state requires at least one hypothesis")
    prior_total = sum(max(item.prior, 0.0) for item in legacy.hypotheses)
    priors = (
        [max(item.prior, 0.0) / prior_total for item in legacy.hypotheses]
        if prior_total > 0
        else [1.0 / len(legacy.hypotheses)] * len(legacy.hypotheses)
    )
    ids = [item.id for item in legacy.hypotheses]
    return TaskFrame(
        schema_version="v0.2",
        task_frame_id=f"{legacy.run_id}_legacy_task_frame",
        admission_decision_id=f"{legacy.run_id}_migration_admission",
        task_kind=TaskKind.DECISION,
        answer_relationship=AnswerRelationship.SELECTION,
        normalized_question="Legacy categorical BayesProbe state.",
        answer_contract=AnswerContract(
            objective="Preserve legacy categorical belief behavior.",
            answer_value_type=AnswerValueType.SHORT_TEXT,
            answer_format="legacy hypothesis id",
            required_sections=["answer", "uncertainty"],
            decision_form="legacy_selection",
            permits_synthesis=False,
        ),
        hypothesis_frame=HypothesisFrame(
            frame_id=f"{legacy.run_id}_legacy_hypothesis_frame",
            competition=HypothesisCompetition.EXCLUSIVE,
            coverage=HypothesisCoverage.EXHAUSTIVE,
            hypotheses=[
                FramedHypothesis(
                    id=item.id,
                    statement=item.statement,
                    type=item.type,
                    scope=item.scope,
                    initial_prior=priors[index],
                    falsifiers=list(item.falsifiers)
                    or [f"A reliable result falsifies legacy hypothesis {item.id}."],
                    predictions=list(item.predictions)
                    or [f"A reliable result supports legacy hypothesis {item.id}."],
                    answer_value=item.id,
                )
                for index, item in enumerate(legacy.hypotheses)
            ],
            rival_sets={
                item: [other for other in ids if other != item] for item in ids
            },
            coverage_statement="Migrated legacy categorical hypothesis set.",
            unresolved_alternative_mass=0.0,
            coverage_limitation=(
                "Competition and coverage were assigned by explicit v0.1 migration."
            ),
        ),
        framing_method=FramingMethod.LEGACY_MIGRATION,
        framing_trace={
            "migration": BELIEF_STATE_V01_TO_V02_MIGRATION_MARKER
        },
    )


def migrate_belief_state_v0_1(payload: Any) -> BeliefState:
    raw_payload = (
        _legacy_belief_state_payload(payload)
        if isinstance(payload, BeliefState)
        else payload
    )
    legacy = _V01BeliefState.model_validate(raw_payload)
    task_frame = (
        _categorical_task_frame(legacy)
        if legacy.task_frame is None
        else migrate_task_frame_v0_1(legacy.task_frame.model_dump())
    )
    hypothesis_frame = task_frame.hypothesis_frame
    adequacy = (
        FrameAdequacyStatus.ADEQUATE
        if hypothesis_frame.coverage == HypothesisCoverage.EXHAUSTIVE
        else FrameAdequacyStatus.PROVISIONAL
    )
    migrated = BeliefState(
        schema_version="v0.2",
        belief_state_id=legacy.belief_state_id,
        run_id=legacy.run_id,
        cycle_id=legacy.cycle_id,
        cycle_index=legacy.cycle_index,
        hypotheses=[
            Hypothesis(**item.model_dump())
            for item in legacy.hypotheses
        ],
        posterior_summary=legacy.posterior_summary,
        uncertainty_summary=legacy.uncertainty_summary,
        ledger_refs=legacy.ledger_refs,
        task_frame=task_frame,
        frame_state=FrameState(
            frame_id=hypothesis_frame.frame_id,
            competition=hypothesis_frame.competition,
            coverage=hypothesis_frame.coverage,
            active_hypothesis_ids=[item.id for item in hypothesis_frame.hypotheses],
            unresolved_alternative_mass=(
                hypothesis_frame.unresolved_alternative_mass
            ),
            adequacy_status=adequacy,
        ),
        evidence_memory=EvidenceMemorySnapshot(),
    )
    return _mark_v01_migration(migrated)


__all__ = [
    "BELIEF_STATE_V01_TO_V02_MIGRATION_MARKER",
    "RECOGNIZED_V01_TO_V02_MIGRATION_MARKERS",
    "TASK_FRAME_V01_TO_V02_MIGRATION_MARKER",
    "migrate_belief_state_v0_1",
    "migrate_task_frame_v0_1",
]
