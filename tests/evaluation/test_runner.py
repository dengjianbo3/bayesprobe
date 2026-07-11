import json
import inspect
import stat
import hashlib
from types import SimpleNamespace
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from bayesprobe.evaluation.artifacts import CapabilityArtifactStore
from bayesprobe.evaluation.contracts import ArmCaseResult, EvaluationCase
from bayesprobe.evaluation.runner import (
    CapabilityExperimentRunner,
    ExperimentIdentity,
    build_experiment_identity,
    deterministic_task_schedule,
    run_capability_preflight,
)
from bayesprobe.evaluation.config import capability_config_from_mapping
from bayesprobe.evaluation.python_probe import ResolvedSandboxImage


def identity():
    return build_experiment_identity(
        experiment_name="synthetic pilot",
        code_git_sha="a" * 40,
        dataset_revision_sha="b" * 40,
        selection_manifest_sha256="c" * 64,
        config_sha256="d" * 64,
        prompt_registry_sha256="e" * 64,
        python_image_digest="sha256:" + "f" * 64,
    )


def cases(count=4):
    return [
        EvaluationCase(
            sample_id=f"synthetic_{index:03d}",
            question=f"Synthetic question {index}? Answer Choices: A. yes B. no",
            choices={"A": "yes", "B": "no"},
        )
        for index in range(count)
    ]


class RecordingArm:
    def __init__(self, name, answer):
        self.name = name
        self.answer = answer
        self.calls = []

    def run_case(self, case):
        self.calls.append(case.sample_id)
        return ArmCaseResult(
            sample_id=case.sample_id,
            arm=self.name,
            state="completed",
            answer_label=self.answer,
            probabilities={"A": 0.75, "B": 0.25}
            if self.answer == "A"
            else {"A": 0.25, "B": 0.75},
            answer_summary="Synthetic result.",
        )


def test_experiment_identity_is_content_addressed_and_validated():
    first = identity()
    second = identity()
    changed = build_experiment_identity(
        experiment_name="synthetic pilot",
        code_git_sha="a" * 40,
        dataset_revision_sha="b" * 40,
        selection_manifest_sha256="c" * 64,
        config_sha256="0" * 64,
        prompt_registry_sha256="e" * 64,
        python_image_digest="sha256:" + "f" * 64,
    )

    assert isinstance(first, ExperimentIdentity)
    assert first == second
    assert first.experiment_id != changed.experiment_id
    assert first.experiment_id.startswith("synthetic-pilot-")


def test_schedule_is_deterministic_and_balances_which_arm_goes_first():
    sample_cases = cases(100)

    first = deterministic_task_schedule(identity().experiment_id, sample_cases)
    second = deterministic_task_schedule(identity().experiment_id, list(reversed(sample_cases)))

    assert first == second
    assert len(first) == 200
    first_by_sample = {}
    for task in first:
        first_by_sample.setdefault(task.sample_id, task.arm)
    direct_first = sum(arm == "direct_flash" for arm in first_by_sample.values())
    assert 30 <= direct_first <= 70
    assert all(
        {task.arm for task in first if task.sample_id == case.sample_id}
        == {"direct_flash", "bayesprobe_python"}
        for case in sample_cases
    )


def test_artifact_store_uses_hmac_paths_and_atomic_terminal_state(tmp_path: Path):
    store = CapabilityArtifactStore(
        tmp_path / "restricted",
        identity(),
        secret=b"fixed-test-secret" * 2,
    )
    case = cases(1)[0]

    paths = store.initialize_case("direct_flash", case.sample_id)
    assert case.sample_id not in str(paths.root)
    assert paths.status_path.exists()
    assert json.loads(paths.status_path.read_text(encoding="utf-8"))["state"] == "pending"
    for audit_path in (
        paths.ledger_path,
        paths.provider_invocations_path,
        paths.python_executions_path,
    ):
        assert audit_path.exists()
        assert stat.S_IMODE(audit_path.stat().st_mode) == 0o600

    store.mark_running("direct_flash", case.sample_id)
    result = RecordingArm("direct_flash", "A").run_case(case)
    store.write_terminal_result(result)

    assert json.loads(paths.status_path.read_text(encoding="utf-8"))["state"] == "completed"
    assert json.loads(paths.result_path.read_text(encoding="utf-8"))["answer_label"] == "A"
    with pytest.raises(ValueError, match="terminal case is immutable"):
        store.write_terminal_result(result)


def test_sample_pseudonyms_are_secret_keyed_not_plain_hashes(tmp_path: Path):
    first = CapabilityArtifactStore(
        tmp_path / "first",
        identity(),
        secret=b"first-fixed-secret" * 2,
    )
    second = CapabilityArtifactStore(
        tmp_path / "second",
        identity(),
        secret=b"second-fixed-secret" * 2,
    )
    sample_id = "private_sample_1"

    assert first.pseudonym_for(sample_id) == first.pseudonym_for(sample_id)
    assert first.pseudonym_for(sample_id) != second.pseudonym_for(sample_id)
    assert first.pseudonym_for(sample_id) != hashlib.sha256(
        sample_id.encode("utf-8")
    ).hexdigest()


def test_stale_running_case_is_resumable_but_fresh_running_is_not(tmp_path: Path):
    store = CapabilityArtifactStore(
        tmp_path / "restricted",
        identity(),
        secret=b"fixed-test-secret" * 2,
    )
    case = cases(1)[0]
    store.initialize_case("direct_flash", case.sample_id)
    store.mark_running("direct_flash", case.sample_id)
    now = datetime.now(UTC)

    assert store.should_run(
        "direct_flash",
        case.sample_id,
        now=now,
        stale_after=timedelta(hours=1),
    ) is False

    status = json.loads(
        store.paths_for("direct_flash", case.sample_id).status_path.read_text(
            encoding="utf-8"
        )
    )
    status["started_at"] = (now - timedelta(hours=2)).isoformat()
    store._write_status_for_test("direct_flash", case.sample_id, status)
    assert store.should_run(
        "direct_flash",
        case.sample_id,
        now=now,
        stale_after=timedelta(hours=1),
    ) is True


def test_runner_executes_both_arms_and_resume_skips_completed_cases(tmp_path: Path):
    direct = RecordingArm("direct_flash", "A")
    bayesprobe = RecordingArm("bayesprobe_python", "B")
    store = CapabilityArtifactStore(
        tmp_path / "restricted",
        identity(),
        secret=b"fixed-test-secret" * 2,
    )
    sample_cases = cases(3)
    completed_case = sample_cases[0]
    store.initialize_case("direct_flash", completed_case.sample_id)
    store.mark_running("direct_flash", completed_case.sample_id)
    store.write_terminal_result(
        RecordingArm("direct_flash", "A").run_case(completed_case)
    )
    runner = CapabilityExperimentRunner(
        identity=identity(),
        cases=sample_cases,
        arms={"direct_flash": direct, "bayesprobe_python": bayesprobe},
        artifact_store=store,
        direct_concurrency=2,
        bayesprobe_concurrency=2,
    )

    summary = runner.run()

    assert summary.terminal_count == 6
    assert summary.completed_count == 6
    assert completed_case.sample_id not in direct.calls
    assert len(direct.calls) == 2
    assert len(bayesprobe.calls) == 3
    assert store.all_terminal(sample_cases) is True


def test_runner_constructor_has_no_gold_store_parameter():
    assert "gold" not in str(inspect.signature(CapabilityExperimentRunner.__init__))


def preflight_config(tmp_path: Path):
    return capability_config_from_mapping(
        {
            "experiment_name": "synthetic pilot",
            "dataset": {"revision": "b" * 40},
            "paths": {
                "restricted_root": str(tmp_path / "artifacts/restricted"),
                "report_root": str(tmp_path / "reports"),
            },
            "python": {"image": "bayesprobe-hle-python:v0.1"},
            "prompt_registry": {"version": "v0.1", "prompts": {"p": "v0.1"}},
            "pricing_snapshot": {
                "as_of": "2026-07-11",
                "currency": "USD",
                "rates": {"input": 1},
                "status": "frozen",
            },
        }
    )


class PreflightSandbox:
    def preflight(self):
        return ResolvedSandboxImage(
            requested_reference="bayesprobe-hle-python:v0.1",
            digest="sha256:" + "f" * 64,
        )


def clean_git_command(command, **kwargs):
    if command[:3] == ["git", "rev-parse", "HEAD"]:
        return SimpleNamespace(returncode=0, stdout="a" * 40 + "\n", stderr="")
    if command[:3] == ["git", "status", "--porcelain"]:
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    if command[:2] == ["git", "check-ignore"]:
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    raise AssertionError(command)


def test_preflight_builds_identity_from_clean_git_manifest_config_and_image(tmp_path: Path):
    config = preflight_config(tmp_path)
    prepared = SimpleNamespace(
        dataset_revision="b" * 40,
        requested_sample_count=100,
        manifest_sha256="c" * 64,
    )

    result = run_capability_preflight(
        config,
        prepared,
        PreflightSandbox(),
        environ={"DEEPSEEK_API_KEY": "sk-runtime-only"},
        run_command=clean_git_command,
        repo_root=tmp_path,
    )

    assert result.code_git_sha == "a" * 40
    assert result.image.digest == "sha256:" + "f" * 64
    assert result.identity.dataset_revision_sha == "b" * 40
    assert result.identity.selection_manifest_sha256 == "c" * 64


def test_preflight_rejects_missing_provider_key_before_run(tmp_path: Path):
    config = preflight_config(tmp_path)
    prepared = SimpleNamespace(
        dataset_revision="b" * 40,
        requested_sample_count=100,
        manifest_sha256="c" * 64,
    )

    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY is not set"):
        run_capability_preflight(
            config,
            prepared,
            PreflightSandbox(),
            environ={},
            run_command=clean_git_command,
            repo_root=tmp_path,
        )


def test_preflight_rejects_dirty_git_or_unignored_restricted_path(tmp_path: Path):
    config = preflight_config(tmp_path)
    prepared = SimpleNamespace(
        dataset_revision="b" * 40,
        requested_sample_count=100,
        manifest_sha256="c" * 64,
    )

    def dirty_git(command, **kwargs):
        result = clean_git_command(command, **kwargs)
        if command[:3] == ["git", "status", "--porcelain"]:
            result.stdout = " M bayesprobe/file.py\n"
        return result

    with pytest.raises(ValueError, match="Git worktree must be clean"):
        run_capability_preflight(
            config,
            prepared,
            PreflightSandbox(),
            environ={"DEEPSEEK_API_KEY": "sk-runtime-only"},
            run_command=dirty_git,
            repo_root=tmp_path,
        )
