from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from bayesprobe.evaluation.artifacts import CapabilityArtifactStore
from bayesprobe.evaluation.config import (
    CapabilityExperimentConfig,
    validate_pricing_snapshot,
)
from bayesprobe.evaluation.contracts import ArmCaseResult, EvaluationCase, ExperimentArm
from bayesprobe.evaluation.python_probe import (
    DockerPythonSandbox,
    ResolvedSandboxImage,
)
from bayesprobe.provider_telemetry import provider_error_category


_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class ExperimentIdentity:
    experiment_id: str
    experiment_name: str
    code_git_sha: str
    dataset_revision_sha: str
    selection_manifest_sha256: str
    config_sha256: str
    prompt_registry_sha256: str
    python_image_digest: str


@dataclass(frozen=True)
class ScheduledArmTask:
    sequence_index: int
    sample_id: str
    arm: str


@dataclass(frozen=True)
class CapabilityRunSummary:
    experiment_id: str
    task_count: int
    executed_count: int
    terminal_count: int
    completed_count: int
    terminal_failed_count: int


@dataclass(frozen=True)
class CapabilityPreflightResult:
    identity: ExperimentIdentity
    code_git_sha: str
    image: ResolvedSandboxImage


def build_experiment_identity(
    *,
    experiment_name: str,
    code_git_sha: str,
    dataset_revision_sha: str,
    selection_manifest_sha256: str,
    config_sha256: str,
    prompt_registry_sha256: str,
    python_image_digest: str,
) -> ExperimentIdentity:
    if not isinstance(experiment_name, str) or not experiment_name.strip():
        raise ValueError("experiment_name must not be empty")
    for name, value in (
        ("code_git_sha", code_git_sha),
        ("dataset_revision_sha", dataset_revision_sha),
    ):
        if not isinstance(value, str) or not _GIT_SHA.fullmatch(value.lower()):
            raise ValueError(f"{name} must be a full 40-character SHA")
    for name, value in (
        ("selection_manifest_sha256", selection_manifest_sha256),
        ("config_sha256", config_sha256),
        ("prompt_registry_sha256", prompt_registry_sha256),
    ):
        if not isinstance(value, str) or not _SHA256.fullmatch(value.lower()):
            raise ValueError(f"{name} must be a SHA-256 hex digest")
    if not isinstance(python_image_digest, str) or not _IMAGE_DIGEST.fullmatch(
        python_image_digest.lower()
    ):
        raise ValueError("python_image_digest must be an immutable sha256 digest")
    components = {
        "experiment_name": experiment_name.strip(),
        "code_git_sha": code_git_sha.lower(),
        "dataset_revision_sha": dataset_revision_sha.lower(),
        "selection_manifest_sha256": selection_manifest_sha256.lower(),
        "config_sha256": config_sha256.lower(),
        "prompt_registry_sha256": prompt_registry_sha256.lower(),
        "python_image_digest": python_image_digest.lower(),
    }
    digest = _canonical_sha256(components)
    experiment_id = f"{_slug(experiment_name)}-{digest[:16]}"
    return ExperimentIdentity(experiment_id=experiment_id, **components)


def deterministic_task_schedule(
    experiment_id: str,
    cases: list[EvaluationCase],
) -> list[ScheduledArmTask]:
    tasks: list[ScheduledArmTask] = []
    for case in sorted(cases, key=lambda item: item.sample_id):
        digest = hashlib.sha256(
            f"{experiment_id}:{case.sample_id}".encode("utf-8")
        ).digest()
        arm_order = (
            ("direct_flash", "bayesprobe_python")
            if digest[-1] & 1 == 0
            else ("bayesprobe_python", "direct_flash")
        )
        for arm in arm_order:
            tasks.append(
                ScheduledArmTask(
                    sequence_index=len(tasks),
                    sample_id=case.sample_id,
                    arm=arm,
                )
            )
    return tasks


def run_capability_preflight(
    config: CapabilityExperimentConfig,
    prepared: Any,
    sandbox: DockerPythonSandbox,
    *,
    environ: Mapping[str, str] | None = None,
    run_command: Callable[..., Any] = subprocess.run,
    repo_root: str | Path = ".",
) -> CapabilityPreflightResult:
    environment = os.environ if environ is None else environ
    if not environment.get(config.model_gateway.api_key_env):
        raise ValueError(
            f"provider API key environment variable {config.model_gateway.api_key_env} is not set"
        )
    if prepared.dataset_revision != config.selection.revision:
        raise ValueError("prepared dataset revision does not match frozen config")
    if prepared.requested_sample_count != config.selection.sample_count:
        raise ValueError("prepared sample count does not match frozen config")
    if not _SHA256.fullmatch(str(prepared.manifest_sha256).lower()):
        raise ValueError("prepared selection manifest hash is invalid")
    validate_pricing_snapshot(config.pricing_snapshot)
    prompts = config.prompt_registry.get("prompts")
    if not isinstance(prompts, Mapping) or not prompts:
        raise ValueError("prompt registry must contain frozen prompts")

    repository = Path(repo_root)
    code_git_sha = _git_output(
        run_command,
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
    ).strip().lower()
    if not _GIT_SHA.fullmatch(code_git_sha):
        raise ValueError("Git HEAD did not resolve to a full commit SHA")
    status = _git_output(
        run_command,
        ["git", "status", "--porcelain"],
        cwd=repository,
    )
    if status.strip():
        raise ValueError("Git worktree must be clean before capability run")
    ignored = run_command(
        ["git", "check-ignore", "--quiet", str(config.restricted_root)],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    if ignored.returncode != 0:
        raise ValueError("restricted capability path must be ignored by Git")

    image = sandbox.preflight()
    identity = build_experiment_identity(
        experiment_name=config.experiment_name,
        code_git_sha=code_git_sha,
        dataset_revision_sha=config.selection.revision,
        selection_manifest_sha256=prepared.manifest_sha256,
        config_sha256=config.config_sha256,
        prompt_registry_sha256=config.prompt_registry_sha256,
        python_image_digest=image.digest,
    )
    return CapabilityPreflightResult(
        identity=identity,
        code_git_sha=code_git_sha,
        image=image,
    )


class CapabilityExperimentRunner:
    def __init__(
        self,
        *,
        identity: ExperimentIdentity,
        cases: list[EvaluationCase],
        arms: dict[str, ExperimentArm],
        artifact_store: CapabilityArtifactStore,
        direct_concurrency: int = 8,
        bayesprobe_concurrency: int = 4,
    ) -> None:
        if set(arms) != {"direct_flash", "bayesprobe_python"}:
            raise ValueError("capability runner requires both frozen experiment arms")
        for name, value in (
            ("direct_concurrency", direct_concurrency),
            ("bayesprobe_concurrency", bayesprobe_concurrency),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be positive")
        sample_ids = [case.sample_id for case in cases]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("capability runner sample ids must be unique")
        self.identity = identity
        self.cases = list(cases)
        self._cases_by_id = {case.sample_id: case for case in cases}
        self.arms = dict(arms)
        self.artifact_store = artifact_store
        self._semaphores = {
            "direct_flash": threading.BoundedSemaphore(direct_concurrency),
            "bayesprobe_python": threading.BoundedSemaphore(
                bayesprobe_concurrency
            ),
        }
        self._max_workers = direct_concurrency + bayesprobe_concurrency

    def run(self) -> CapabilityRunSummary:
        for case in self.cases:
            for arm in CapabilityArtifactStore.arm_names:
                self.artifact_store.initialize_case(arm, case.sample_id)
        schedule = deterministic_task_schedule(
            self.identity.experiment_id,
            self.cases,
        )
        runnable: list[ScheduledArmTask] = []
        for task in schedule:
            if self.artifact_store.should_run(task.arm, task.sample_id):
                self.artifact_store.mark_running(task.arm, task.sample_id)
                runnable.append(task)

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = [executor.submit(self._execute_task, task) for task in runnable]
            for future in as_completed(futures):
                future.result()

        states = [
            self.artifact_store.status(arm, case.sample_id)["state"]
            for case in self.cases
            for arm in CapabilityArtifactStore.arm_names
        ]
        terminal_count = sum(
            state in {"completed", "terminal_failed"} for state in states
        )
        return CapabilityRunSummary(
            experiment_id=self.identity.experiment_id,
            task_count=len(schedule),
            executed_count=len(runnable),
            terminal_count=terminal_count,
            completed_count=states.count("completed"),
            terminal_failed_count=states.count("terminal_failed"),
        )

    def _execute_task(self, task: ScheduledArmTask) -> None:
        case = self._cases_by_id[task.sample_id]
        with self._semaphores[task.arm]:
            try:
                result = self.arms[task.arm].run_case(case)
                if result.sample_id != case.sample_id or result.arm != task.arm:
                    raise ValueError("arm result identity does not match scheduled task")
            except Exception as error:
                result = ArmCaseResult(
                    sample_id=case.sample_id,
                    arm=task.arm,
                    state="terminal_failed",
                    answer_label=None,
                    probabilities=None,
                    error_category=provider_error_category(error),
                )
            self.artifact_store.write_terminal_result(result)


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_output(
    run_command: Callable[..., Any],
    command: list[str],
    *,
    cwd: Path,
) -> str:
    try:
        completed = run_command(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as error:
        raise ValueError("Git is unavailable for capability preflight") from error
    if completed.returncode != 0:
        raise ValueError(f"Git preflight command failed: {' '.join(command[:2])}")
    return str(completed.stdout)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:48] or "experiment"


__all__ = [
    "CapabilityExperimentRunner",
    "CapabilityPreflightResult",
    "CapabilityRunSummary",
    "ExperimentIdentity",
    "ScheduledArmTask",
    "build_experiment_identity",
    "deterministic_task_schedule",
    "run_capability_preflight",
]
