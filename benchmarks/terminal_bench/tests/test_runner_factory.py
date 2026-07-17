from __future__ import annotations

import asyncio
import hashlib
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
    EvidenceJudgmentRepairPolicy,
    OpenAIChatCompletionsModelGateway,
    RecordedTaskAdmitter,
    TaskAwareAnswerProjector,
)
from bayesprobe_terminal_bench.artifacts import TrialArtifactStore
from bayesprobe_terminal_bench.causal import (
    CausalEvidenceModelGateway,
    CausalTraceRegistry,
)
from bayesprobe_terminal_bench.config import (
    BudgetExhausted,
    RunBudget,
    TerminalBenchConfig,
)
from bayesprobe_terminal_bench.experiment_lock import (
    FROZEN_GATE_TASK_IDS,
    FROZEN_GATE_TASK_REFS,
    PAIRED_GATE_ARMS,
)
from bayesprobe_terminal_bench.gateway import HarborProbeToolGateway
from bayesprobe_terminal_bench.planning import TerminalPlanError
from bayesprobe_terminal_bench.provider_contract import TerminalContractModelGateway
from bayesprobe_terminal_bench.runner_factory import (
    ArtifactInvocationObserver,
    BudgetedModelGateway,
    RepositoryGitIdentity,
    build_live_session,
    build_runner,
    collect_repository_git_identity,
    load_and_validate_lock,
    safe_run_id,
    validate_runtime_lock,
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


class RecordingOpenAIClient:
    def __init__(self, responses: list[object]) -> None:
        self._responses = iter(responses)
        self.options: list[dict[str, object]] = []
        self.requests: list[dict[str, object]] = []

    def with_options(self, **kwargs: object) -> object:
        self.options.append(dict(kwargs))
        parent = self

        class Completions:
            def create(self, **request: object) -> object:
                parent.requests.append(dict(request))
                return next(parent._responses)

        return SimpleNamespace(chat=SimpleNamespace(completions=Completions()))


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


def _active_lock_payload(config: TerminalBenchConfig) -> dict[str, object]:
    payload = _lock_payload(config)
    payload.update(
        {
            "schema_version": "terminal_bench_lock:v1",
            "terminal_plan_version": "terminal_probe_plan:v1",
            "max_provider_tokens": config.max_provider_tokens,
            "agent_timeout_seconds": config.task_timeout_seconds,
            "expected_provider_model": config.model,
            "expected_system_fingerprint": "fp-locked",
        }
    )
    return payload


def _write_active_lock(path: Path, config: TerminalBenchConfig) -> dict[str, object]:
    payload = _active_lock_payload(config)
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
    assert type(runner.initializer._task_admitter) is RecordedTaskAdmitter
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
    budget = RunBudget(max_provider_tokens=100)
    observer = ArtifactInvocationObserver(
        artifacts,
        budget=budget,
        expected_model="provider-model",
        expected_system_fingerprint="fp-1",
    )

    observer.observe(
        ProviderRecord(
            {
                "task": "judge_evidence",
                "model": "provider-model",
                "system_fingerprint": "fp-1",
                "usage": {"total_tokens": 12},
                "outcome": "success",
                "message": f"do not persist {secret}",
            }
        )
    )
    observer.observe(object())

    telemetry = (tmp_path / "provider_telemetry.jsonl").read_text(encoding="utf-8")
    assert secret not in telemetry
    assert "[REDACTED]" in telemetry
    assert len(telemetry.splitlines()) == 1
    assert budget.provider_tokens_used == 12


def test_shared_provider_accounting_accumulates_core_terminal_and_react_calls(
    tmp_path: Path,
) -> None:
    budget = RunBudget(max_provider_tokens=100)
    observer = ArtifactInvocationObserver(
        TrialArtifactStore(tmp_path, restricted_values=()),
        budget=budget,
        expected_model="provider-model",
        expected_system_fingerprint="fp-1",
    )

    for task, tokens in (
        ("judge_evidence", 11),
        ("terminal_probe_plan", 13),
        ("react_step", 17),
    ):
        observer.observe(
            ProviderRecord(
                {
                    "task": task,
                    "model": "provider-model",
                    "system_fingerprint": "fp-1",
                    "usage": {"total_tokens": tokens},
                    "outcome": "success",
                }
            )
        )

    assert budget.provider_tokens_used == 41


def test_planner_sdk_observation_records_identity_usage_without_raw_content(
    tmp_path: Path,
) -> None:
    secret = "raw-provider-secret"
    budget = RunBudget(max_provider_tokens=100)
    observer = ArtifactInvocationObserver(
        TrialArtifactStore(tmp_path, restricted_values=(secret,)),
        budget=budget,
        expected_model="provider-model",
        expected_system_fingerprint="fp-1",
    )
    response = SimpleNamespace(
        id="response-1",
        model="provider-model",
        system_fingerprint="fp-1",
        usage=SimpleNamespace(
            prompt_tokens=7,
            completion_tokens=5,
            total_tokens=12,
        ),
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=secret),
            )
        ],
    )

    observer.observe_sdk_response(response, task="terminal_probe_plan")

    assert budget.provider_tokens_used == 12
    telemetry = (tmp_path / "provider_telemetry.jsonl").read_text(encoding="utf-8")
    assert secret not in telemetry
    assert "terminal_probe_plan" in telemetry


def test_public_observer_uses_raw_response_identity_capture(tmp_path: Path) -> None:
    observer = ArtifactInvocationObserver(
        TrialArtifactStore(tmp_path, restricted_values=()),
        budget=RunBudget(max_provider_tokens=100),
        expected_model="provider-model",
        expected_system_fingerprint="fp-1",
    )
    observer.capture_provider_response(
        SimpleNamespace(
            model="drifted-model",
            system_fingerprint="fp-1",
        )
    )

    with pytest.raises(BudgetExhausted) as failure:
        observer.observe(
            ProviderRecord(
                {
                    "model": "provider-model",
                    "system_fingerprint": "fp-1",
                    "usage": {"total_tokens": 1},
                    "outcome": "success",
                }
            )
        )

    assert failure.value.category == "provider_identity_error"


@pytest.mark.parametrize(
    "payload",
    [
        {
            "model": "provider-model",
            "system_fingerprint": "fp-1",
            "usage": {"total_tokens": None},
            "outcome": "success",
        },
        {
            "model": "provider-model",
            "system_fingerprint": "fp-1",
            "usage": {"total_tokens": True},
            "outcome": "success",
        },
        {
            "model": "provider-model",
            "system_fingerprint": "fp-1",
            "usage": {"total_tokens": 1.5},
            "outcome": "success",
        },
        {
            "model": "provider-model",
            "system_fingerprint": "fp-1",
            "usage": {"total_tokens": -1},
            "outcome": "success",
        },
        {
            "model": "drifted-model",
            "system_fingerprint": "fp-1",
            "usage": {"total_tokens": 1},
            "outcome": "success",
        },
        {
            "model": "provider-model",
            "system_fingerprint": "fp-2",
            "usage": {"total_tokens": 1},
            "outcome": "success",
        },
    ],
)
def test_artifact_observer_fails_closed_on_usage_or_identity_drift(
    tmp_path: Path,
    payload: dict[str, object],
) -> None:
    observer = ArtifactInvocationObserver(
        TrialArtifactStore(tmp_path, restricted_values=()),
        budget=RunBudget(max_provider_tokens=100),
        expected_model="provider-model",
        expected_system_fingerprint="fp-1",
    )

    with pytest.raises(BudgetExhausted) as failure:
        observer.observe(ProviderRecord(payload))

    assert failure.value.category == "provider_identity_error"


def test_artifact_observer_detects_fingerprint_availability_drift(
    tmp_path: Path,
) -> None:
    observer = ArtifactInvocationObserver(
        TrialArtifactStore(tmp_path, restricted_values=()),
        budget=RunBudget(max_provider_tokens=100),
        expected_model="provider-model",
        expected_system_fingerprint=None,
    )

    with pytest.raises(BudgetExhausted) as failure:
        observer.observe(
            ProviderRecord(
                {
                    "model": "provider-model",
                    "system_fingerprint": "became-available",
                    "usage": {"total_tokens": 1},
                    "outcome": "success",
                }
            )
        )

    assert failure.value.category == "provider_identity_error"


def test_budgeted_gateway_surfaces_identity_failure_swallowed_by_delegate(
    tmp_path: Path,
) -> None:
    observer = ArtifactInvocationObserver(
        TrialArtifactStore(tmp_path, restricted_values=()),
        budget=RunBudget(max_provider_tokens=100),
        expected_model="provider-model",
        expected_system_fingerprint="fp-1",
    )

    class Delegate(RecordingModelGateway):
        def __init__(self) -> None:
            super().__init__()
            self.invocation_observer = observer

        def complete_structured(self, request: object) -> dict[str, object]:
            self.calls.append(request)
            try:
                observer.observe(
                    ProviderRecord(
                        {
                            "model": "drifted-model",
                            "system_fingerprint": "fp-1",
                            "usage": {"total_tokens": 1},
                            "outcome": "success",
                        }
                    )
                )
            except BudgetExhausted:
                pass
            return {"ok": True}

    budget = RunBudget(max_model_calls=2, max_provider_tokens=100)
    gateway = BudgetedModelGateway(Delegate(), budget)

    with pytest.raises(BudgetExhausted) as failure:
        gateway.complete_structured(object())

    assert failure.value.category == "provider_identity_error"
    assert budget.model_calls_used == 1


def test_budgeted_gateway_surfaces_artifact_failure_swallowed_by_delegate() -> None:
    class FailingArtifacts:
        def append_provider_call(self, payload: object) -> None:
            raise RuntimeError("artifact write failed")

    observer = ArtifactInvocationObserver(
        FailingArtifacts(),
        budget=RunBudget(max_provider_tokens=100),
        expected_model="provider-model",
        expected_system_fingerprint="fp-1",
    )

    class Delegate(RecordingModelGateway):
        def __init__(self) -> None:
            super().__init__()
            self.invocation_observer = observer

        def complete_structured(self, request: object) -> dict[str, object]:
            self.calls.append(request)
            try:
                observer.observe(
                    ProviderRecord(
                        {
                            "model": "provider-model",
                            "system_fingerprint": "fp-1",
                            "usage": {"total_tokens": 1},
                            "outcome": "success",
                        }
                    )
                )
            except RuntimeError:
                pass
            return {"ok": True}

    gateway = BudgetedModelGateway(
        Delegate(),
        RunBudget(max_model_calls=2, max_provider_tokens=100),
    )

    with pytest.raises(RuntimeError, match="artifact write failed"):
        gateway.complete_structured(object())


def test_openai_deadline_proxy_recomputes_timeout_and_disables_sdk_retries() -> None:
    from bayesprobe_terminal_bench.deadline import DeadlineOpenAIClient, TrialDeadline

    now = [10.0]
    base_client = RecordingOpenAIClient([{"ok": 1}, {"ok": 2}])
    observed_responses: list[object] = []
    observed_errors: list[Exception] = []
    client = DeadlineOpenAIClient(
        base_client=base_client,
        deadline=TrialDeadline(timeout_seconds=100, monotonic=lambda: now[0]),
        configured_timeout_seconds=360,
        response_observer=observed_responses.append,
        error_observer=observed_errors.append,
    )

    assert client.chat.completions.create(model="test") == {"ok": 1}
    now[0] = 20.2
    assert client.chat.completions.create(model="test") == {"ok": 2}
    assert base_client.options == [
        {"timeout": 95, "max_retries": 0},
        {"timeout": 84, "max_retries": 0},
    ]
    assert observed_responses == [{"ok": 1}, {"ok": 2}]

    now[0] = 105.0
    with pytest.raises(BudgetExhausted) as failure:
        client.chat.completions.create(model="test")
    assert failure.value.category == "budget_error"
    assert observed_errors == [failure.value]
    assert len(base_client.requests) == 2


def test_expired_deadline_rejects_terminal_plan_before_model_reservation(
    probe: object,
    execution_context: object,
) -> None:
    from bayesprobe_terminal_bench.deadline import DeadlineOpenAIClient, TrialDeadline
    from bayesprobe_terminal_bench.planning import (
        OpenAICompatibleTerminalProbePlanner,
    )

    deadline = TrialDeadline(timeout_seconds=5, monotonic=lambda: 0.0)
    budget = RunBudget(
        max_model_calls=3,
        reservation_guard=deadline.require_active,
    )
    base_client = RecordingOpenAIClient([])
    planner = OpenAICompatibleTerminalProbePlanner(
        config=TerminalBenchConfig(model="test-model"),
        budget=budget,
        client=DeadlineOpenAIClient(
            base_client=base_client,
            deadline=deadline,
            configured_timeout_seconds=360,
        ),
    )

    with pytest.raises(BudgetExhausted) as failure:
        planner.plan(probe=probe, context=execution_context, history=())

    assert failure.value.category == "budget_error"
    assert budget.model_calls_used == 0
    assert base_client.options == []
    assert base_client.requests == []


def test_environment_deadline_proxy_clamps_shell_timeout_and_stops_after_expiry() -> None:
    from bayesprobe_terminal_bench.actions import ShellAction
    from bayesprobe_terminal_bench.deadline import (
        DeadlineEnvironmentBridge,
        TrialDeadline,
    )

    class Bridge:
        def __init__(self) -> None:
            self.calls: list[tuple[object, int]] = []

        def execute(self, action: object, action_index: int) -> object:
            self.calls.append((action, action_index))
            return "observed"

    now = [0.0]
    bridge = Bridge()
    guarded = DeadlineEnvironmentBridge(
        delegate=bridge,
        deadline=TrialDeadline(timeout_seconds=100, monotonic=lambda: now[0]),
        configured_timeout_seconds=120,
    )

    assert guarded.execute(ShellAction(command="pwd", timeout_seconds=120), 1) == (
        "observed"
    )
    assert bridge.calls[0][0].timeout_seconds == 95

    now[0] = 95.0
    with pytest.raises(BudgetExhausted) as failure:
        guarded.execute(ShellAction(command="ls", timeout_seconds=120), 2)
    assert failure.value.category == "budget_error"
    assert len(bridge.calls) == 1


def test_environment_deadline_proxy_clamps_non_shell_bridge_timeout() -> None:
    from bayesprobe_terminal_bench.actions import WriteFileAction
    from bayesprobe_terminal_bench.deadline import (
        DeadlineEnvironmentBridge,
        TrialDeadline,
    )

    class Bridge:
        _NON_SHELL_TIMEOUT_SECONDS = 120

        def __init__(self) -> None:
            self.timeouts: list[int] = []

        def execute(self, action: object, action_index: int) -> object:
            self.timeouts.append(self._NON_SHELL_TIMEOUT_SECONDS)
            return "observed"

    bridge = Bridge()
    guarded = DeadlineEnvironmentBridge(
        delegate=bridge,
        deadline=TrialDeadline(timeout_seconds=20, monotonic=lambda: 0.0),
        configured_timeout_seconds=120,
    )

    assert guarded.execute(WriteFileAction(path="/tmp/result", content="ok"), 1) == (
        "observed"
    )
    assert bridge.timeouts == [15]
    assert bridge._NON_SHELL_TIMEOUT_SECONDS == 120


def test_expired_deadline_rejects_terminal_action_before_action_reservation(
    tmp_path: Path,
    probe: object,
    execution_context: object,
) -> None:
    from bayesprobe_terminal_bench.actions import (
        ShellAction,
        TerminalPlanStep,
        TerminalProbePlan,
    )
    from bayesprobe_terminal_bench.deadline import (
        DeadlineEnvironmentBridge,
        TrialDeadline,
    )

    plan = TerminalProbePlan(
        mode="inspect",
        steps=(
            TerminalPlanStep(
                role="inspect",
                action=ShellAction(command="pwd"),
            ),
        ),
        expected_observation="current workspace path",
    )

    class Planner:
        def plan(self, **_: object) -> TerminalProbePlan:
            return plan

    class Bridge:
        def __init__(self) -> None:
            self.calls: list[tuple[object, int]] = []

        def execute(self, action: object, action_index: int) -> object:
            self.calls.append((action, action_index))
            raise AssertionError("expired action must not reach the delegate")

    deadline = TrialDeadline(timeout_seconds=5, monotonic=lambda: 0.0)
    budget = RunBudget(
        max_actions=1,
        reservation_guard=deadline.require_active,
    )
    delegate = Bridge()
    gateway = HarborProbeToolGateway(
        planner=Planner(),
        bridge=DeadlineEnvironmentBridge(
            delegate=delegate,
            deadline=deadline,
            configured_timeout_seconds=120,
        ),
        artifacts=TrialArtifactStore(tmp_path, restricted_values=()),
        budget=budget,
    )

    with pytest.raises(BudgetExhausted) as failure:
        gateway.execute_probe(probe=probe, context=execution_context)

    assert failure.value.category == "budget_error"
    assert budget.actions_used == 0
    assert delegate.calls == []


def test_terminal_plan_failure_propagates_without_starting_an_action(
    tmp_path: Path,
    probe: object,
    execution_context: object,
) -> None:
    class FailingPlanner:
        def plan(self, **_: object) -> object:
            raise TerminalPlanError(
                category="provider_contract_error",
                attempts=3,
            )

    class NeverBridge:
        def execute(self, *args: object, **kwargs: object) -> object:
            raise AssertionError("an invalid plan must not execute an action")

    artifacts = TrialArtifactStore(tmp_path, restricted_values=())
    gateway = HarborProbeToolGateway(
        planner=FailingPlanner(),
        bridge=NeverBridge(),
        artifacts=artifacts,
        budget=RunBudget(max_actions=3, max_model_calls=3),
    )

    with pytest.raises(TerminalPlanError) as failure:
        gateway.execute_probe(probe=probe, context=execution_context)

    assert failure.value.category == "provider_contract_error"
    errors = [
        json.loads(line)
        for line in (tmp_path / "errors.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert errors == [
        {
            "category": "provider_contract_error",
            "error_type": "TerminalPlanError",
            "probe_id": probe.id,
        }
    ]


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


def test_active_runtime_rejects_v01_lock_and_accepts_v1_identity(
    tmp_path: Path,
) -> None:
    config = TerminalBenchConfig(
        model="locked-model",
        task_timeout_seconds=900,
    )
    path = tmp_path / "benchmark.lock.json"
    historical = _write_lock(path, config)

    with pytest.raises(ValueError, match="schema_version"):
        validate_runtime_lock(
            path,
            config,
            arm="bayesprobe",
            session_id="break-filter-js-from-html__run__agent",
            runtime_git_identity=_runtime_git_identity(historical),
        )

    active = _write_active_lock(path, config)
    validated = validate_runtime_lock(
        path,
        config,
        arm="bayesprobe",
        session_id="break-filter-js-from-html__run__agent",
        runtime_git_identity=_runtime_git_identity(active),
    )

    assert validated["schema_version"] == "terminal_bench_lock:v1"
    assert validated["terminal_plan_version"] == "terminal_probe_plan:v1"


def test_active_runtime_rejects_task_timeout_mismatch(tmp_path: Path) -> None:
    config = TerminalBenchConfig(model="locked-model", task_timeout_seconds=900)
    path = tmp_path / "benchmark.lock.json"
    payload = _active_lock_payload(config)
    payload["agent_timeout_seconds"] = 901
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="agent_timeout_seconds"):
        validate_runtime_lock(
            path,
            config,
            arm="bayesprobe",
            session_id="break-filter-js-from-html__run__agent",
            runtime_git_identity=_runtime_git_identity(payload),
        )


def test_active_runtime_requires_locked_fingerprint_availability(
    tmp_path: Path,
) -> None:
    config = TerminalBenchConfig(model="locked-model", task_timeout_seconds=900)
    path = tmp_path / "benchmark.lock.json"
    payload = _active_lock_payload(config)
    payload.pop("expected_system_fingerprint")
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="expected_system_fingerprint"):
        validate_runtime_lock(
            path,
            config,
            arm="bayesprobe",
            session_id="break-filter-js-from-html__run__agent",
            runtime_git_identity=_runtime_git_identity(payload),
        )


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
        task_timeout_seconds=900,
        lock_path=lock_path,
    )
    lock = _write_active_lock(lock_path, config)
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
    expected_lock_bytes = json.dumps(
        lock,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert session.runtime_lock_sha256 == (
        f"sha256:{hashlib.sha256(expected_lock_bytes).hexdigest()}"
    )
    assert environment.calls == []
    assert type(session.runner.answer_projector) is TaskAwareAnswerProjector

    model_gateway = session.runner.core._model_gateway
    probe_gateway = session.runner.executor._gateway
    assert type(model_gateway) is CausalEvidenceModelGateway
    assert model_gateway._delegate._delegate._budget is session.budget
    assert probe_gateway._budget is session.budget
    assert probe_gateway._planner._budget is session.budget
    assert probe_gateway._bridge._loop is active_loop
    assert api_key not in repr(session)
    assert list(session.artifacts.root.iterdir()) == []


@pytest.mark.asyncio
async def test_live_session_uses_exact_causal_composition_and_shared_registry(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "benchmark.lock.json"
    config = TerminalBenchConfig(
        model="live-model",
        lock_path=lock_path,
        task_timeout_seconds=900,
    )
    lock = _write_active_lock(lock_path, config)

    session = build_live_session(
        config=config,
        api_key="one-time-live-provider-secret",
        instruction="Repair the supplied task workspace.",
        environment=NeverExecutedEnvironment(),
        event_loop=asyncio.get_running_loop(),
        logs_dir=tmp_path / "logs",
        session_id="session/id",
        context_id="context:id",
        runtime_git_identity=_runtime_git_identity(lock),
    )

    guarded = session.runner.core._model_gateway
    assert type(guarded) is CausalEvidenceModelGateway
    assert type(guarded._registry) is CausalTraceRegistry
    assert type(guarded._delegate) is TerminalContractModelGateway
    assert type(guarded._delegate._delegate) is BudgetedModelGateway
    assert type(guarded._delegate._delegate._delegate) is OpenAIChatCompletionsModelGateway
    assert guarded._delegate._delegate._budget is session.budget

    probe_gateway = session.runner.executor._gateway
    assert probe_gateway._causal is guarded._registry
    assert session.runner.initializer._task_framer._open_framer._model_gateway is guarded
    assert session.runner.probe_designer._model_gateway is guarded
    assert session.runner.core._hypothesis_expander._adapter._model_gateway is guarded
    assert session.runner.answer_projector._model_gateway is guarded
    assert session.runner.core._judgment_repair_policy == EvidenceJudgmentRepairPolicy(
        max_attempts=2
    )


@pytest.mark.asyncio
async def test_live_session_injects_one_deadline_into_every_request_and_action_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = TerminalBenchConfig(
        model="live-model",
        lock_path=tmp_path / "unused.lock.json",
        task_timeout_seconds=100,
    )
    monkeypatch.setattr(
        "bayesprobe_terminal_bench.runner_factory.validate_runtime_lock",
        lambda *args, **kwargs: {
            "expected_provider_model": "live-model",
            "expected_system_fingerprint": "fp-locked",
            "agent_timeout_seconds": 100,
        },
    )

    session = build_live_session(
        config=config,
        api_key="one-time-live-provider-secret",
        instruction="Repair the supplied task workspace.",
        environment=NeverExecutedEnvironment(),
        event_loop=asyncio.get_running_loop(),
        logs_dir=tmp_path / "logs",
        session_id="session/id",
        context_id="context:id",
    )

    guarded = session.runner.core._model_gateway
    provider = guarded._delegate._delegate._delegate
    probe_gateway = session.runner.executor._gateway
    assert provider._client._deadline is session.deadline
    assert probe_gateway._planner._deadline is session.deadline
    assert probe_gateway._bridge._deadline is session.deadline
    assert session.budget._reservation_guard.__self__ is session.deadline


@pytest.mark.asyncio
async def test_build_live_session_accepts_paired_gate_lock_for_locked_task(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "paired.lock.json"
    config = TerminalBenchConfig(
        model="live-model",
        base_url="https://provider.example/v1",
        task_timeout_seconds=900,
        lock_path=lock_path,
    )
    payload = {
        "schema_version": "terminal_bench_paired_gate:v1",
        "harbor_version": "0.18.0",
        "dataset_name": "terminal-bench/terminal-bench-2",
        "dataset_revision": "sha256:" + "1" * 64,
        "tasks": [
            {
                "task_id": task_id,
                "task_ref": FROZEN_GATE_TASK_REFS[task_id],
                "image_digest": "sha256:" + str(index) * 64,
                "agent_timeout_seconds": 900,
            }
            for index, task_id in enumerate(FROZEN_GATE_TASK_IDS, start=2)
        ],
        "root_git_sha": "a" * 40,
        "adapter_tree_sha": "b" * 40,
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
        "max_provider_tokens": config.max_provider_tokens,
        "command_timeout_seconds": config.command_timeout_seconds,
        "provider_timeout_seconds": config.provider_timeout_seconds,
        "max_output_tokens": config.max_output_tokens,
        "signal_output_bytes": config.signal_output_bytes,
        "terminal_plan_version": "terminal_probe_plan:v1",
        "expected_provider_model": config.model,
        "expected_system_fingerprint": "fp-locked",
        "arms": PAIRED_GATE_ARMS,
    }
    tasks = payload["tasks"]
    assert isinstance(tasks, list)
    first_task = tasks[0]
    assert isinstance(first_task, dict)
    locked_digest = first_task["image_digest"]
    first_task["image_digest"] = "not-a-digest"
    lock_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="tasks"):
        validate_runtime_lock(
            lock_path,
            config,
            arm="bayesprobe",
            session_id="cancel-async-tasks__AbCd123__agent",
            runtime_git_identity=RepositoryGitIdentity(
                root_git_sha="a" * 40,
                adapter_tree_sha="b" * 40,
                adapter_dirty=False,
            ),
        )

    first_task["image_digest"] = locked_digest
    lock_path.write_text(json.dumps(payload), encoding="utf-8")

    session = build_live_session(
        config=config,
        api_key="one-time-live-provider-secret",
        instruction="Repair the supplied task workspace.",
        environment=NeverExecutedEnvironment(),
        event_loop=asyncio.get_running_loop(),
        logs_dir=tmp_path / "logs",
        session_id="cancel-async-tasks__AbCd123__agent",
        context_id="context:id",
        runtime_git_identity=RepositoryGitIdentity(
            root_git_sha="a" * 40,
            adapter_tree_sha="b" * 40,
            adapter_dirty=False,
        ),
    )

    assert type(session.runner) is AutonomousQuestionRunner
