from __future__ import annotations

from bayesprobe.evaluation.contracts import ArmCaseResult
from bayesprobe.evaluation.search_matrix import build_search_matrix_report


def _result(sample_id: str, arm: str, answer: str) -> ArmCaseResult:
    return ArmCaseResult(
        sample_id=sample_id,
        arm=arm,
        state="completed",
        answer_label=answer,
        probabilities={"A": 0.1, "B": 0.9} if answer == "B" else {"A": 0.9, "B": 0.1},
        answer_summary="Restricted answer text must not appear in the report.",
    )


def test_search_matrix_report_builds_four_arm_accuracy_and_transitions():
    report = build_search_matrix_report(
        gold_labels={"s1": "B", "s2": "A"},
        baseline_correctness={
            "s1": {"direct_no_web": False, "bayesprobe_no_web": True},
            "s2": {"direct_no_web": True, "bayesprobe_no_web": False},
        },
        search_results=[
            _result("s1", "direct_search", "B"),
            _result("s1", "bayesprobe_search", "A"),
            _result("s2", "direct_search", "A"),
            _result("s2", "bayesprobe_search", "A"),
        ],
        source_binding={"source_checkpoint_id": "checkpoint-1"},
        search_policy={"max_search_calls": 2},
    )

    assert report["sample_count"] == 2
    assert report["accuracy"] == {
        "direct_no_web": 0.5,
        "direct_search": 1.0,
        "bayesprobe_no_web": 0.5,
        "bayesprobe_search": 0.5,
    }
    assert report["no_web_to_search_transitions"]["direct"]["wrong_to_correct"] == 1
    assert report["no_web_to_search_transitions"]["bayesprobe"]["correct_to_wrong"] == 1
    assert "s1" not in str(report)
    assert "Restricted answer text" not in str(report)
