from __future__ import annotations

import asyncio
import json
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from bayesprobe import (
    AutonomousQuestionRunner,
    BayesProbeCore,
    CapabilityKind,
    DeterministicModelGateway,
    TaskAwareAnswerProjector,
)
from bayesprobe_terminal_bench.artifacts import TrialArtifactStore
from bayesprobe_terminal_bench.config import (
    BudgetExhausted,
    RunBudget,
    TerminalBenchConfig,
)
from bayesprobe_terminal_bench.runner_factory import (
    ArtifactInvocationObserver,
    BudgetedModelGateway,
    RepositoryGitIdentity,
    build_live_session,
    build_runner,
    collect_repository_git_identity,
    load_and_validate_lock,
    safe_run_id,
)
from write_benchmark_lock import RuntimeIdentity, build_lock


_CONFIG_LOCKED_KEYS = (
    "schema_version",
    "harbor_version",
    "dataset_name",
    "task_id",
    "model",
    "base_url",
    "provider_protocol",
    "api_key_env",
    "temperature",
    "max_cycles",
    "max_probes_per_cycle",
    "max_actions_per_probe",
    "max_total_actions",
    "max_model_calls",
    "command_timeout_seconds",
    "provider_timeout_seconds",
    "max_output_tokens",
    "signal_output_bytes",
    "terminal_plan_version",
)
_IDENTITY_KEYS = (
    "dataset_revision",
    "task_checksum",
    "container_image",
    "image_digest",
    "root_git_sha",
    "adapter_tree_sha",
    "n_attempts",
)
_LOCKED_KEYS = _CONFIG_LOCKED_KEYS + _IDENTITY_KEYS


class NoSignalGateway:
    def execute_probe(self, *, probe: object, context: object) -> list[object]:
        return []


class RecordingModelGateway:
    adapter_kind = "recording"
    model_identity = "recording:model"

    def __init__(self) -> None:
        self.calls: list[object] = []
        self.config = SimpleNamespace(model="recording-model")
        self.invocation_observer = object()
        self.api_key = "must-not-appear-in-wrapper-repr"

    def complete_structured(self, request: object) -> dict[str, object]:
        self.calls.append(request)
        return {"request": request}


class ProviderRecord:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def to_dict(self) -> dict[str, Any]:
        return dict(self._payload)


class NeverExecutedEnvironment:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def exec(self, *args: object, **kwargs: object) -> object:
        self.calls.append("exec")
        raise AssertionError("live-session construction must not execute commands")

    async def upload_file(self, *args: object, **kwargs: object) -> None:
        self.calls.append("upload_file")
        raise AssertionError("live-session construction must not upload files")


def _lock_payload(config: TerminalBenchConfig) -> dict[str, object]:
    return {
        "schema_version": "terminal_bench_lock:v0.1",
        "harbor_version": "0.18.0",
        "dataset_name": "terminal-bench/terminal-bench-2",
        "dataset_revision": "sha256:" + "1" * 64,
        "task_id": "terminal-bench/break-filter-js-from-html",
        "task_checksum": "sha256:" + "2" * 64,
        "container_image": "registry.example/terminal-bench/task:locked",
        "image_digest": "sha256:" + "3" * 64,
        "root_git_sha": "4" * 40,
        "adapter_tree_sha": "5" * 40,
        "n_attempts": 1,
        "model": config.model,
        "base_url": config.base_url,
        "provider_protocol": "openai_chat_completions",
        "api_key_env": config.api_key_env,
        "temperature": 0,
        "max_cycles": config.max_cycles,
        "max_probes_per_cycle": config.max_probes_per_cycle,
        "max_actions_per_probe": config.max_actions_per_probe,
        "max_total_actions": config.max_total_actions,
        "max_model_calls": config.max_model_calls,
        "command_timeout_seconds": config.command_timeout_seconds,
        "provider_timeout_seconds": config.provider_timeout_seconds,
        "max_output_tokens": config.max_output_tokens,
        "signal_output_bytes": config.signal_output_bytes,
        "terminal_plan_version": "terminal_probe_plan:v0.1",
    }


def _write_lock(path: Path, config: TerminalBenchConfig) -> dict[str, object]:
    payload = _lock_payload(config)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _runtime_git_identity(
    payload: dict[str, object],
    *,
    dirty: bool = False,
) -> RepositoryGitIdentity:
    return RepositoryGitIdentity(
        root_git_sha=str(payload["root_git_sha"]),
        adapter_tree_sha=str(payload["adapter_tree_sha"]),
        adapter_dirty=dirty,
    )


def _different(value: object) -> object:
    if value is None:
        return "unexpected"
    if type(value) is int:
        return value + 1
    return f"{value}-unexpected"


def test_factory_returns_real_public_runner_and_core(tmp_path: Path) -> None:
    model_gateway = DeterministicModelGateway()
    probe_gateway = NoSignalGateway()

    runner = build_runner(
        model_gateway=model_gateway,
        probe_gateway=probe_gateway,
        ledger_path=tmp_path / "ledger.jsonl",
        config=TerminalBenchConfig(model="deterministic", max_cycles=1),
    )

    assert type(runner) is AutonomousQuestionRunner
    assert type(runner.core) is BayesProbeCore
    assert runner.core._model_gateway is model_gateway
    assert runner.executor._gateway is probe_gateway
    assert {item.kind for item in runner.available_capabilities} == {
        CapabilityKind.REPOSITORY_READ,
        CapabilityKind.TEST_EXECUTION,
    }


def test_factory_isolates_capability_descriptors_between_runners(tmp_path: Path) -> None:
    first = build_runner(
        model_gateway=DeterministicModelGateway(),
        probe_gateway=NoSignalGateway(),
        ledger_path=tmp_path / "first.jsonl",
        config=TerminalBenchConfig(model="deterministic"),
    )
    second = build_runner(
        model_gateway=DeterministicModelGateway(),
        probe_gateway=NoSignalGateway(),
        ledger_path=tmp_path / "second.jsonl",
        config=TerminalBenchConfig(model="deterministic"),
    )

    first.available_capabilities[0].quality_caps["verifiability"] = 0.1

    assert second.available_capabilities[0].quality_caps["verifiability"] == 0.95


def test_budgeted_model_gateway_reserves_once_and_preserves_public_properties() -> None:
    delegate = RecordingModelGateway()
    budget = RunBudget(max_model_calls=1)
    gateway = BudgetedModelGateway(delegate, budget)
    request = object()

    assert gateway.complete_structured(request) == {"request": request}
    assert budget.model_calls_used == 1
    assert delegate.calls == [request]
    assert gateway.adapter_kind == delegate.adapter_kind
    assert gateway.model_identity == delegate.model_identity
    assert gateway.config is delegate.config
    assert gateway.invocation_observer is delegate.invocation_observer
    assert delegate.api_key not in repr(gateway)

    with pytest.raises(BudgetExhausted, match="model call budget exhausted"):
        gateway.complete_structured(object())
    assert delegate.calls == [request]


def test_artifact_invocation_observer_accepts_provider_record_and_redacts(
    tmp_path: Path,
) -> None:
    secret = "provider-secret-value"
    artifacts = TrialArtifactStore(tmp_path, restricted_values=(secret,))
    observer = ArtifactInvocationObserver(artifacts)

    observer.observe(
        ProviderRecord(
            {
                "task": "judge_evidence",
                "model": "provider-model",
                "message": f"do not persist {secret}",
            }
        )
    )
    observer.observe(object())

    telemetry = (tmp_path / "provider_telemetry.jsonl").read_text(encoding="utf-8")
    assert secret not in telemetry
    assert "[REDACTED]" in telemetry
    assert len(telemetry.splitlines()) == 1


def test_safe_run_id_normalizes_prefixes_and_caps_harbor_ids() -> None:
    assert safe_run_id(" /job:alpha? 42 ") == "tb_job_alpha_42"
    assert safe_run_id("***") == "tb_harbor"
    capped = safe_run_id("x" * 200)
    assert len(capped) == 96
    assert re.fullmatch(r"[A-Za-z0-9_.-]+", capped)


def test_repository_identity_ignores_runs_but_detects_adapter_edits(
    tmp_path: Path,
) -> None:
    adapter = tmp_path / "benchmarks" / "terminal_bench"
    adapter.mkdir(parents=True)
    (adapter / ".gitignore").write_text(".runs/\n", encoding="utf-8")
    tracked = adapter / "tracked.txt"
    tracked.write_text("committed\n", encoding="utf-8")
    commands = [
        ["git", "init", "-q"],
        ["git", "config", "user.name", "Terminal Bench Test"],
        ["git", "config", "user.email", "terminal-bench@example.invalid"],
        ["git", "add", "benchmarks/terminal_bench"],
        ["git", "commit", "-qm", "fixture"],
    ]
    for command in commands:
        subprocess.run(command, cwd=tmp_path, check=True, capture_output=True)
    generated = adapter / ".runs" / "harbor" / "result.json"
    generated.parent.mkdir(parents=True)
    generated.write_text("{}\n", encoding="utf-8")

    clean = collect_repository_git_identity(tmp_path)
    tracked.write_text("dirty\n", encoding="utf-8")
    dirty = collect_repository_git_identity(tmp_path)

    assert clean.adapter_dirty is False
    assert dirty.adapter_dirty is True
    assert clean.root_git_sha == dirty.root_git_sha
    assert clean.adapter_tree_sha == dirty.adapter_tree_sha


def test_lock_is_required_and_must_contain_an_object(tmp_path: Path) -> None:
    config = TerminalBenchConfig(model="locked-model")
    missing = tmp_path / "missing.json"

    with pytest.raises(ValueError, match="Terminal-Bench lock is required"):
        load_and_validate_lock(missing, config)

    non_object = tmp_path / "non-object.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="Terminal-Bench lock must be an object"):
        load_and_validate_lock(non_object, config)

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError):
        load_and_validate_lock(invalid, config)


@pytest.mark.parametrize("key", _CONFIG_LOCKED_KEYS)
def test_lock_rejects_every_locked_setting_mismatch(tmp_path: Path, key: str) -> None:
    config = TerminalBenchConfig(
        model="locked-model",
        base_url="https://provider.example/v1",
    )
    payload = _lock_payload(config)
    payload[key] = _different(payload[key])
    path = tmp_path / f"mismatch-{key}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Terminal-Bench lock mismatch") as error:
        load_and_validate_lock(path, config)

    assert key in str(error.value)


@pytest.mark.parametrize(
    ("key", "wrong_type_value"),
    [("temperature", False), ("max_cycles", True), ("max_model_calls", 40.0)],
)
def test_lock_rejects_equal_numeric_values_with_different_json_types(
    tmp_path: Path,
    key: str,
    wrong_type_value: object,
) -> None:
    config = TerminalBenchConfig(model="locked-model", max_cycles=1)
    payload = _lock_payload(config)
    payload[key] = wrong_type_value
    path = tmp_path / f"wrong-type-{key}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Terminal-Bench lock mismatch") as error:
        load_and_validate_lock(path, config)

    assert key in str(error.value)


@pytest.mark.parametrize("key", _LOCKED_KEYS)
def test_lock_rejects_missing_locked_settings(tmp_path: Path, key: str) -> None:
    config = TerminalBenchConfig(model="locked-model")
    payload = _lock_payload(config)
    payload.pop(key)
    path = tmp_path / f"missing-{key}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Terminal-Bench lock mismatch") as error:
        load_and_validate_lock(path, config)

    assert key in str(error.value)


def test_valid_lock_is_returned_without_rewriting_it(tmp_path: Path) -> None:
    config = TerminalBenchConfig(model="locked-model")
    path = tmp_path / "benchmark.lock.json"
    payload = _write_lock(path, config)
    original = path.read_bytes()

    assert load_and_validate_lock(
        path,
        config,
        runtime_git_identity=_runtime_git_identity(payload),
    ) == payload
    assert path.read_bytes() == original


@pytest.mark.parametrize("stale_field", ["root_git_sha", "adapter_tree_sha"])
def test_live_loader_rejects_a_well_formed_stale_git_lock(
    tmp_path: Path,
    stale_field: str,
) -> None:
    config = TerminalBenchConfig(model="locked-model")
    path = tmp_path / "benchmark.lock.json"
    payload = _write_lock(path, config)
    runtime_identity = RepositoryGitIdentity(
        root_git_sha=str(payload["root_git_sha"]),
        adapter_tree_sha=str(payload["adapter_tree_sha"]),
        adapter_dirty=False,
    )
    runtime_identity = RepositoryGitIdentity(
        root_git_sha=(
            "9" * 40
            if stale_field == "root_git_sha"
            else runtime_identity.root_git_sha
        ),
        adapter_tree_sha=(
            "8" * 40
            if stale_field == "adapter_tree_sha"
            else runtime_identity.adapter_tree_sha
        ),
        adapter_dirty=False,
    )

    with pytest.raises(ValueError, match=stale_field):
        load_and_validate_lock(
            path,
            config,
            runtime_git_identity=runtime_identity,
        )


def test_live_loader_rejects_dirty_adapter_even_when_committed_ids_match(
    tmp_path: Path,
) -> None:
    config = TerminalBenchConfig(model="locked-model")
    path = tmp_path / "benchmark.lock.json"
    payload = _write_lock(path, config)

    with pytest.raises(ValueError, match="dirty"):
        load_and_validate_lock(
            path,
            config,
            runtime_git_identity=RepositoryGitIdentity(
                root_git_sha=str(payload["root_git_sha"]),
                adapter_tree_sha=str(payload["adapter_tree_sha"]),
                adapter_dirty=True,
            ),
        )


def test_partial_pre_task9_lock_is_rejected_before_agent_construction(
    tmp_path: Path,
) -> None:
    config = TerminalBenchConfig(model="locked-model")
    payload = _lock_payload(config)
    for key in _IDENTITY_KEYS:
        payload.pop(key)
    path = tmp_path / "old-partial.lock.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Terminal-Bench lock mismatch") as error:
        load_and_validate_lock(path, config)

    assert set(_IDENTITY_KEYS) <= set(re.findall(r"[a-z_]+", str(error.value)))


@pytest.mark.parametrize(
    ("key", "invalid_value"),
    [
        ("dataset_revision", "latest"),
        ("task_checksum", "sha256:not-a-digest"),
        ("container_image", ""),
        ("image_digest", "sha256:" + "A" * 64),
        ("root_git_sha", "not-a-git-object-id"),
        ("adapter_tree_sha", "f" * 39),
        ("n_attempts", 2),
    ],
)
def test_lock_rejects_malformed_task9_identity_before_agent_construction(
    tmp_path: Path,
    key: str,
    invalid_value: object,
) -> None:
    config = TerminalBenchConfig(model="locked-model")
    payload = _lock_payload(config)
    payload[key] = invalid_value
    path = tmp_path / f"malformed-{key}.lock.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Terminal-Bench lock mismatch") as error:
        load_and_validate_lock(path, config)

    assert key in str(error.value)


def test_loader_accepts_complete_build_lock_output(
    tmp_path: Path,
    synthetic_oracle_job: Path,
) -> None:
    config = TerminalBenchConfig(model="locked-model")
    lock = build_lock(
        job_dir=synthetic_oracle_job,
        config=config,
        runtime_identity=RuntimeIdentity(
            harbor_version="0.18.0",
            root_git_sha="a" * 40,
            adapter_tree_sha="b" * 40,
            container_image="registry.example/terminal-bench/task:locked",
            image_digest="sha256:" + "c" * 64,
        ),
    )
    path = tmp_path / "complete.lock.json"
    path.write_text(json.dumps(lock), encoding="utf-8")

    assert load_and_validate_lock(
        path,
        config,
        runtime_git_identity=_runtime_git_identity(lock),
    ) == lock


@pytest.mark.asyncio
async def test_build_live_session_composes_shared_budget_and_active_loop_without_running(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "benchmark.lock.json"
    config = TerminalBenchConfig(
        model="live-model",
        base_url="https://provider.example/v1",
        max_cycles=1,
        lock_path=lock_path,
    )
    lock = _write_lock(lock_path, config)
    environment = NeverExecutedEnvironment()
    active_loop = asyncio.get_running_loop()
    api_key = "one-time-live-provider-secret"

    session = build_live_session(
        config=config,
        api_key=api_key,
        instruction="Repair the supplied task workspace.",
        environment=environment,
        event_loop=active_loop,
        logs_dir=tmp_path / "logs",
        session_id="session/id",
        context_id="context:id",
        runtime_git_identity=_runtime_git_identity(lock),
    )

    assert type(session.runner) is AutonomousQuestionRunner
    assert session.input.run_id == "tb_context_id"
    assert session.input.problem == "Repair the supplied task workspace."
    assert "official verifier" in session.input.task_context
    assert session.artifacts.root == (tmp_path / "logs" / "bayesprobe").resolve()
    assert session.budget.actions_used == 0
    assert session.budget.model_calls_used == 0
    assert environment.calls == []
    assert type(session.runner.answer_projector) is TaskAwareAnswerProjector

    model_gateway = session.runner.core._model_gateway
    probe_gateway = session.runner.executor._gateway
    assert type(model_gateway) is BudgetedModelGateway
    assert model_gateway._budget is session.budget
    assert probe_gateway._budget is session.budget
    assert probe_gateway._planner._budget is session.budget
    assert probe_gateway._bridge._loop is active_loop
    assert api_key not in repr(session)
    assert list(session.artifacts.root.iterdir()) == []
