from __future__ import annotations

import json
from pathlib import Path

import pytest

from bayesprobe import ProbeDesign, ProbeExecutionBrief, ProbeExecutionHypothesisView


FIXED_DATASET = "terminal-bench/terminal-bench-2"
FIXED_TASK = "terminal-bench/break-filter-js-from-html"
FIXED_DATASET_REVISION = "sha256:" + "1" * 64
FIXED_TASK_CHECKSUM = "sha256:" + "2" * 64
FIXED_CONTAINER_IMAGE = (
    "ghcr.io/laude-institute/break-filter-js-from-html@sha256:" + "3" * 64
)


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


@pytest.fixture
def synthetic_oracle_job(tmp_path: Path) -> Path:
    job_dir = tmp_path / "oracle-job"
    trial_dir = job_dir / "break-filter-js-from-html__oracle"
    trial_dir.mkdir(parents=True)

    job_config = {
        "job_name": "bayesprobe-terminal-bench-oracle-smoke",
        "jobs_dir": ".runs/harbor/oracle",
        "n_attempts": 1,
        "n_concurrent_trials": 1,
        "datasets": [
            {
                "name": FIXED_DATASET,
                "ref": FIXED_DATASET_REVISION,
                "task_names": [FIXED_TASK],
            }
        ],
        "agents": [{"name": "oracle"}],
        "environment": {"type": "docker", "delete": True},
    }
    trial_config = {
        "task": {
            "name": FIXED_TASK,
            "ref": FIXED_DATASET_REVISION,
            "source": FIXED_DATASET,
        },
        "trial_name": trial_dir.name,
        "trials_dir": str(job_dir),
        "agent": {"name": "oracle"},
        "environment": {"type": "docker", "delete": True},
        # Harbor's downloaded task environment supplies this value. The
        # fixture keeps it explicit so lock parsing remains fully offline.
        "task_environment": {"docker_image": FIXED_CONTAINER_IMAGE},
    }
    trial_result = {
        "task_name": "break-filter-js-from-html",
        "trial_name": trial_dir.name,
        "task_id": {
            "org": "terminal-bench",
            "name": "break-filter-js-from-html",
            "ref": FIXED_DATASET_REVISION,
        },
        "source": FIXED_DATASET,
        "task_checksum": FIXED_TASK_CHECKSUM,
        "config": trial_config,
        "agent_info": {"name": "oracle", "version": "0.18.0"},
        "verifier_result": {"rewards": {"reward": 1.0}},
        "started_at": "2026-07-14T00:00:00Z",
        "finished_at": "2026-07-14T00:01:00Z",
    }

    (job_dir / "config.json").write_text(
        json.dumps(job_config), encoding="utf-8"
    )
    (trial_dir / "config.json").write_text(
        json.dumps(trial_config), encoding="utf-8"
    )
    (trial_dir / "result.json").write_text(
        json.dumps(trial_result), encoding="utf-8"
    )
    return job_dir
