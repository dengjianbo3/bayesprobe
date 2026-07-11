import pytest

from bayesprobe.migrations import (
    migrate_belief_state_v0_1,
    migrate_task_frame_v0_1,
)
from bayesprobe.schemas import (
    AnswerRelationship,
    FrameAdequacyStatus,
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
    payload = {
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
        "task_frame": legacy_mcq_frame_payload(),
    }

    migrated = migrate_belief_state_v0_1(payload)

    assert migrated.schema_version == "v0.2"
    assert migrated.frame_state is not None
    assert migrated.frame_state.frame_id == "legacy_hypothesis_frame"
    assert migrated.frame_state.active_hypothesis_ids == ["A", "B"]
    assert migrated.frame_state.adequacy_status == FrameAdequacyStatus.ADEQUATE
    assert migrated.evidence_memory is not None
    assert migrated.evidence_memory.accepted_evidence_ids == []
    assert [item.answer_value for item in migrated.hypotheses] == [None, None]
