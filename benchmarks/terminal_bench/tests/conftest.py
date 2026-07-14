from __future__ import annotations

import pytest

from bayesprobe import ProbeDesign, ProbeExecutionBrief, ProbeExecutionHypothesisView


@pytest.fixture
def probe() -> ProbeDesign:
    return ProbeDesign(
        id="P_cycle_1_inspect",
        cycle_id="cycle_1",
        target_hypotheses=["H_workspace"],
        inquiry_goal="Inspect the workspace to identify the failing component.",
        method="terminal inspection",
        expected_observation="A concrete error linked to the workspace state.",
        support_condition={"H_workspace": "The workspace shows the suspected failure."},
        weaken_condition={"H_workspace": "The workspace contradicts the suspected failure."},
        reframe_condition=None,
    )


@pytest.fixture
def execution_context() -> ProbeExecutionBrief:
    return ProbeExecutionBrief(
        run_id="run_1",
        cycle_id="cycle_1",
        problem="Repair the task workspace.",
        task_context="Use the provided task workspace only.",
        task_frame={
            "schema_version": "v0.2",
            "task_frame_id": "TF_run_1",
            "admission_decision_id": "TA_run_1",
            "task_kind": "diagnosis",
            "answer_relationship": "open_ended",
            "normalized_question": "Repair the task workspace.",
            "task_context": "Use the provided task workspace only.",
            "answer_contract": {
                "objective": "Repair the task workspace.",
                "answer_value_type": "structured_text",
                "answer_format": "plain_text",
                "required_sections": ["result"],
                "decision_form": "implementation",
                "permits_synthesis": True,
            },
            "hypothesis_frame": {
                "frame_id": "HF_run_1",
                "competition": "open",
                "coverage": "open",
                "rival_sets": {"H_workspace": []},
                "coverage_statement": "The current hypothesis is incomplete.",
                "coverage_limitation": "Additional causes may exist.",
            },
            "framing_method": "explicit",
        },
        provider_schema_version="v0.2",
        hypotheses=(
            ProbeExecutionHypothesisView(
                id="H_workspace",
                statement="A workspace defect blocks task completion.",
                scope="task workspace",
                predictions=("Inspection exposes a concrete defect.",),
                falsifiers=("The workspace is already valid.",),
            ),
        ),
    )
