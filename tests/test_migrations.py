import pytest

from bayesprobe.migrations import (
    _has_v01_migration_receipt,
    migrate_belief_state_v0_1,
    migrate_task_frame_v0_1,
)
from bayesprobe.lifecycle import BeliefLifecycle, resolve_belief_lifecycle
from bayesprobe.schemas import (
    AnswerRelationship,
    BeliefState,
    FrameAdequacyStatus,
    FramingMethod,
    HypothesisCompetition,
    HypothesisCoverage,
)


def legacy_mcq_frame_payload() -> dict:
    return {
        "task_frame_id": "legacy_task_frame",
        "task_kind": "multiple_choice",
        "normalized_question": "Which option is correct?",
        "task_context": "",
        "answer_contract": {
            "objective": "Select the correct option.",
            "required_sections": ["answer", "uncertainty"],
            "decision_form": "choice_selection",
            "permits_synthesis": False,
        },
        "hypothesis_frame": {
            "frame_id": "legacy_hypothesis_frame",
            "relation": "exclusive_exhaustive",
            "hypotheses": [
                {
                    "id": "A",
                    "statement": "Option A is correct.",
                    "type": "answer_candidate",
                    "scope": "The stated question.",
                    "initial_prior": 0.5,
                    "falsifiers": ["Option A conflicts with the evidence."],
                    "predictions": ["The evidence entails option A."],
                },
                {
                    "id": "B",
                    "statement": "Option B is correct.",
                    "type": "answer_candidate",
                    "scope": "The stated question.",
                    "initial_prior": 0.5,
                    "falsifiers": ["Option B conflicts with the evidence."],
                    "predictions": ["The evidence entails option B."],
                },
            ],
            "rival_sets": {"A": ["B"], "B": ["A"]},
            "coverage_statement": "The supplied choices are exhaustive.",
            "unresolved_alternative_mass": None,
            "coverage_limitation": None,
        },
        "framing_method": "explicit",
        "framing_trace": {"schema_version": "v0.1"},
    }


def legacy_independent_frame_payload() -> dict:
    payload = legacy_mcq_frame_payload()
    payload["task_kind"] = "explanation"
    payload["answer_contract"]["permits_synthesis"] = True
    payload["hypothesis_frame"]["relation"] = "independent"
    payload["hypothesis_frame"]["rival_sets"] = {"A": [], "B": []}
    payload["hypothesis_frame"]["coverage_statement"] = (
        "The named explanations are not exhaustive."
    )
    return payload


def legacy_belief_state_payload(*, include_task_frame: bool = True) -> dict:
    return {
        "belief_state_id": "legacy_belief",
        "run_id": "run_1",
        "cycle_id": "cycle_0",
        "cycle_index": 0,
        "hypotheses": [
            {
                "id": "A",
                "statement": "Option A is correct.",
                "scope": "The stated question.",
                "prior": 0.5,
                "posterior": 0.6,
            },
            {
                "id": "B",
                "statement": "Option B is correct.",
                "scope": "The stated question.",
                "prior": 0.5,
                "posterior": 0.4,
            },
        ],
        "posterior_summary": {},
        "uncertainty_summary": "",
        "ledger_refs": {},
        "task_frame": legacy_mcq_frame_payload() if include_task_frame else None,
    }


def test_migrates_legacy_exclusive_frame_to_exclusive_exhaustive():
    migrated = migrate_task_frame_v0_1(legacy_mcq_frame_payload())

    assert migrated.schema_version == "v0.2"
    assert migrated.answer_relationship == AnswerRelationship.SELECTION
    assert migrated.hypothesis_frame.competition == HypothesisCompetition.EXCLUSIVE
    assert migrated.hypothesis_frame.coverage == HypothesisCoverage.EXHAUSTIVE
    assert migrated.hypothesis_frame.unresolved_alternative_mass == 0.0
    assert "relation" not in migrated.hypothesis_frame.model_dump()


def test_explicit_migration_keeps_v01_fixture_compatibility():
    payload = legacy_mcq_frame_payload()

    migrated = migrate_task_frame_v0_1(payload)

    assert payload["framing_trace"] == {"schema_version": "v0.1"}
    assert migrated.schema_version == "v0.2"
    assert migrated.task_frame_id == "legacy_task_frame"


def test_migrates_legacy_independent_frame_to_independent_open():
    migrated = migrate_task_frame_v0_1(legacy_independent_frame_payload())

    assert migrated.hypothesis_frame.competition == HypothesisCompetition.INDEPENDENT
    assert migrated.hypothesis_frame.coverage == HypothesisCoverage.OPEN
    assert migrated.hypothesis_frame.unresolved_alternative_mass is None


def test_migration_rejects_unknown_legacy_fields():
    payload = legacy_mcq_frame_payload()
    payload["unknown"] = "not part of v0.1"

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        migrate_task_frame_v0_1(payload)


def test_migration_rejects_task_kind_absent_from_v01_vocabulary():
    payload = legacy_mcq_frame_payload()
    payload["task_kind"] = "exact_answer"

    with pytest.raises(ValueError, match="task_kind"):
        migrate_task_frame_v0_1(payload)


def test_migrates_belief_state_with_frame_and_empty_memory():
    payload = legacy_belief_state_payload()

    migrated = migrate_belief_state_v0_1(payload)

    assert migrated.schema_version == "v0.2"
    assert migrated.frame_state is not None
    assert migrated.frame_state.frame_id == "legacy_hypothesis_frame"
    assert migrated.frame_state.active_hypothesis_ids == ["A", "B"]
    assert migrated.frame_state.adequacy_status == FrameAdequacyStatus.ADEQUATE
    assert migrated.evidence_memory is not None
    assert migrated.evidence_memory.memory_version == 1
    assert migrated.evidence_memory.accepted_evidence_ids == []
    assert [item.answer_value for item in migrated.hypotheses] == [None, None]


@pytest.mark.parametrize(
    ("field", "raw_value"),
    [
        ("memory_version", True),
        (
            "correlation_credit",
            {"legacy-group|A|confirming": True},
        ),
    ],
    ids=["boolean-version", "boolean-credit"],
)
def test_migrated_belief_state_restore_rejects_coercive_memory_scalars(
    field,
    raw_value,
):
    migrated = migrate_belief_state_v0_1(legacy_belief_state_payload())
    payload = migrated.model_dump(mode="python")
    payload["evidence_memory"][field] = raw_value

    with pytest.raises(ValueError):
        BeliefState.model_validate(payload)


@pytest.mark.parametrize("include_task_frame", [False, True])
def test_explicit_migration_receipt_survives_copy_but_not_public_round_trip(
    include_task_frame,
):
    migrated = migrate_belief_state_v0_1(
        legacy_belief_state_payload(include_task_frame=include_task_frame)
    )

    assert resolve_belief_lifecycle(migrated) == (
        BeliefLifecycle.LEGACY_V01_MIGRATION
    )
    for copied in (migrated.model_copy(), migrated.model_copy(deep=True)):
        assert resolve_belief_lifecycle(copied) == (
            BeliefLifecycle.LEGACY_V01_MIGRATION
        )

    round_tripped = BeliefState.model_validate(
        migrated.model_dump(mode="python")
    )
    with pytest.raises(ValueError, match="invalid belief lifecycle"):
        resolve_belief_lifecycle(round_tripped)


def test_native_public_fields_cannot_forge_legacy_migration_authority():
    migrated = migrate_belief_state_v0_1(legacy_belief_state_payload())
    native_payload = migrated.model_dump(mode="python")
    native_payload["task_frame"]["framing_method"] = FramingMethod.EXPLICIT
    native_payload["task_frame"]["framing_trace"] = {"source": "native_fixture"}
    native = BeliefState.model_validate(native_payload)
    forged = native.model_copy(
        update={
            "task_frame": native.task_frame.model_copy(
                update={
                    "framing_method": FramingMethod.LEGACY_MIGRATION,
                    "framing_trace": {
                        "migration": "belief_state_v0.1_to_v0.2"
                    },
                }
            )
        }
    )

    with pytest.raises(ValueError, match="invalid belief lifecycle"):
        resolve_belief_lifecycle(forged)


def test_migration_receipt_is_bound_to_the_exact_public_envelope():
    migrated = migrate_belief_state_v0_1(legacy_belief_state_payload())

    changed_copy = migrated.model_copy(update={"cycle_id": "changed_cycle"})
    mutated = migrated.model_copy(deep=True)
    mutated.cycle_id = "mutated_cycle"

    for changed in (changed_copy, mutated):
        assert _has_v01_migration_receipt(changed) is False
        with pytest.raises(ValueError, match="invalid belief lifecycle"):
            resolve_belief_lifecycle(changed)


def test_authentic_receipt_cannot_be_transferred_to_another_public_envelope():
    migrated = migrate_belief_state_v0_1(legacy_belief_state_payload())
    native_payload = migrated.model_dump(mode="python")
    native_payload.update(
        {
            "belief_state_id": "native_replacement",
            "cycle_id": "native_cycle",
            "cycle_index": 7,
        }
    )
    replacement = BeliefState.model_validate(native_payload)

    transferred = migrated.model_copy(
        update={
            field_name: getattr(replacement, field_name)
            for field_name in BeliefState.model_fields
        }
    )

    assert transferred.model_dump(mode="python") == replacement.model_dump(
        mode="python"
    )
    assert _has_v01_migration_receipt(transferred) is False
    with pytest.raises(ValueError, match="invalid belief lifecycle"):
        resolve_belief_lifecycle(transferred)
