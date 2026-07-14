from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import pytest

from bayesprobe import ProbeDesign, ProbeExecutionBrief, ProbeExecutionHypothesisView
from harbor.models.job.config import DatasetConfig, JobConfig, RetryConfig
from harbor.models.job.lock import HarborLockInfo, JobLock, TaskLock, TrialLock
from harbor.models.task.id import PackageTaskId
from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
    VerifierConfig,
)
from harbor.models.trial.result import (
    AgentInfo,
    ExceptionInfo,
    TrialResult,
    VerifierResult,
)


FIXED_DATASET = "terminal-bench/terminal-bench-2"
FIXED_TASK = "terminal-bench/break-filter-js-from-html"
FIXED_DATASET_REVISION = "sha256:" + "1" * 64
FIXED_TASK_CHECKSUM = "sha256:" + "2" * 64
FIXED_CONTAINER_IMAGE = (
    "ghcr.io/laude-institute/break-filter-js-from-html@sha256:" + "3" * 64
)
FIXED_TIME = datetime(2026, 7, 14, tzinfo=timezone.utc)


def write_harbor_job_artifacts(
    job_dir: Path,
    trial_dir: Path,
    *,
    agent_name: str,
    reward: float | None,
    exception_type: str | None = None,
    exception_message: str | None = None,
) -> None:
    trial_dir.mkdir(parents=True, exist_ok=True)
    agent = AgentConfig(name=agent_name)
    environment = EnvironmentConfig(type="docker", delete=True)
    verifier = VerifierConfig()
    task = TaskConfig(
        name=FIXED_TASK,
        ref=FIXED_TASK_CHECKSUM,
        source=FIXED_DATASET,
    )
    trial_config = TrialConfig(
        task=task,
        trial_name=trial_dir.name,
        trials_dir=job_dir,
        agent=agent,
        environment=environment,
        verifier=verifier,
    )
    trial_lock = TrialLock(
        task=TaskLock(
            name=FIXED_TASK,
            type="package",
            digest=FIXED_TASK_CHECKSUM,
            source=FIXED_DATASET,
        ),
        agent=agent,
        environment=environment,
        verifier=verifier,
    )
    job_config = JobConfig(
        job_name="bayesprobe-terminal-bench-smoke",
        jobs_dir=job_dir.parent,
        n_attempts=1,
        n_concurrent_trials=1,
        agents=[agent],
        datasets=[
            DatasetConfig(
                name=FIXED_DATASET,
                ref=FIXED_DATASET_REVISION,
                task_names=[FIXED_TASK],
            )
        ],
        environment=environment,
        verifier=verifier,
    )
    job_lock = JobLock(
        created_at=FIXED_TIME,
        harbor=HarborLockInfo(version="0.18.0"),
        n_concurrent_trials=1,
        retry=RetryConfig(),
        trials=[trial_lock],
    )
    exception = None
    if exception_type is not None:
        exception = ExceptionInfo(
            exception_type=exception_type,
            exception_message=exception_message or "trial failed",
            exception_traceback="synthetic offline traceback",
            occurred_at=FIXED_TIME,
        )
    result = TrialResult(
        id=UUID("00000000-0000-0000-0000-000000000009"),
        task_name="break-filter-js-from-html",
        trial_name=trial_dir.name,
        trial_uri=trial_dir.resolve().as_uri(),
        task_id=PackageTaskId(
            org="terminal-bench",
            name="break-filter-js-from-html",
            ref=FIXED_TASK_CHECKSUM,
        ),
        source=FIXED_DATASET,
        task_checksum=FIXED_TASK_CHECKSUM,
        config=trial_config,
        agent_info=AgentInfo(name=agent_name, version="0.18.0"),
        verifier_result=(
            VerifierResult(rewards={"reward": reward})
            if reward is not None
            else None
        ),
        exception_info=exception,
        started_at=FIXED_TIME,
        finished_at=FIXED_TIME,
    )

    artifacts = {
        job_dir / "config.json": job_config.model_dump(
            mode="json", exclude_none=True
        ),
        job_dir / "lock.json": job_lock.model_dump(mode="json", exclude_none=True),
        trial_dir / "config.json": trial_config.model_dump(
            mode="json", exclude_none=True
        ),
        trial_dir / "lock.json": trial_lock.model_dump(
            mode="json", exclude_none=True
        ),
        trial_dir / "result.json": result.model_dump(mode="json"),
    }
    for path, payload in artifacts.items():
        path.write_text(json.dumps(payload), encoding="utf-8")


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
    write_harbor_job_artifacts(
        job_dir,
        trial_dir,
        agent_name="oracle",
        reward=1.0,
    )
    return job_dir
