from __future__ import annotations

import asyncio
import os
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
from harbor.models.agent.context import AgentContext

from bayesprobe_terminal_bench.agent import (
    BayesProbeHarborAgent,
    BayesProbeHarborAgentError,
)
from bayesprobe_terminal_bench.provider_contract import ProviderContractError


_API_KEY = "one-time-agent-secret"
_EXTRA_ENV = {
    "BAYESPROBE_BENCH_API_KEY": _API_KEY,
    "BAYESPROBE_BENCH_MODEL": "deepseek-v4-flash",
    "BAYESPROBE_BENCH_TASK_TIMEOUT_SECONDS": "900",
}
_BAYESPROBE_METADATA = {
    "bayesprobe_run_id",
    "bayesprobe_stop_reason",
    "bayesprobe_cycles",
    "terminal_actions",
    "model_calls",
    "runtime_lock_sha256",
    "runtime_budgets",
}
WORKTREE_DIR = Path(__file__).resolve().parents[3]


class SpyArtifacts:
    def __init__(self) -> None:
        self.summaries: list[dict[str, object]] = []
        self.errors: list[dict[str, object]] = []

    def write_summary(self, payload: dict[str, object]) -> None:
        self.summaries.append(dict(payload))

    def append_error(self, payload: dict[str, object]) -> None:
        self.errors.append(dict(payload))


class SpyRunner:
    def __init__(self) -> None:
        self.calls = 0
        self.thread_id: int | None = None

    def run_question(self, input: object) -> object:
        self.calls += 1
        self.thread_id = threading.get_ident()
        return SimpleNamespace(
            stop_reason=SimpleNamespace(value="max_cycles"),
            cycle_results=(object(), object()),
        )


class HostileFailure(RuntimeError):
    def __init__(self, secret: str) -> None:
        super().__init__(f"hostile failure with {secret}")
        self.secret = secret
        self.payload = {"api_key": secret}

    def __repr__(self) -> str:
        return f"HostileFailure(secret={self.secret!r})"


def _session(*, runner: object, artifacts: SpyArtifacts) -> object:
    return SimpleNamespace(
        runner=runner,
        input=SimpleNamespace(run_id="tb_harbor-run"),
        artifacts=artifacts,
        runtime_lock_sha256="sha256:" + "9" * 64,
        budget=SimpleNamespace(
            provider_tokens_used=0,
            actions_used=3,
            model_calls_used=4,
        ),
    )


def _agent(tmp_path: Path) -> BayesProbeHarborAgent:
    return BayesProbeHarborAgent(
        logs_dir=tmp_path,
        model_name="deepseek-v4-flash",
        extra_env=_EXTRA_ENV,
    )


def _raise_hostile_failure() -> None:
    try:
        raise RuntimeError(f"hostile cause with {_API_KEY}")
    except RuntimeError as cause:
        failure = HostileFailure(_API_KEY)
        failure.linked_exception = cause
        raise failure from cause


def _assert_stable_failure(
    failure: BayesProbeHarborAgentError,
    expected_message: str,
    expected_category: str,
) -> None:
    assert type(failure) is BayesProbeHarborAgentError
    assert str(failure) == expected_message
    assert failure.category == expected_category
    assert failure.__cause__ is None
    assert failure.__context__ is None
    for value in (
        repr(failure),
        repr(vars(failure)),
        repr(failure.__cause__),
        repr(failure.__context__),
    ):
        assert _API_KEY not in value


def test_exact_uv_project_command_imports_the_agent() -> None:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [
            "uv",
            "run",
            "--project",
            "benchmarks/terminal_bench",
            "python",
            "-c",
            (
                "from bayesprobe_terminal_bench.agent import BayesProbeHarborAgent; "
                "print(BayesProbeHarborAgent.import_path())"
            ),
        ],
        capture_output=True,
        check=False,
        cwd=WORKTREE_DIR,
        env=environment,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "bayesprobe_terminal_bench.agent:BayesProbeHarborAgent"


@pytest.mark.asyncio
async def test_agent_runs_the_real_session_once_in_a_worker_and_records_safe_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = SpyRunner()
    artifacts = SpyArtifacts()
    session = _session(runner=runner, artifacts=artifacts)
    build_arguments: dict[str, object] = {}

    def build_session(**kwargs: object) -> object:
        build_arguments.update(kwargs)
        return session

    monkeypatch.setattr(
        "bayesprobe_terminal_bench.agent.build_live_session",
        build_session,
    )
    original_environment = dict(os.environ)
    context = AgentContext(metadata={"harbor_owned": "preserved"})
    event_loop_thread = threading.get_ident()
    active_loop = asyncio.get_running_loop()

    await _agent(tmp_path).run("solve the task", object(), context)

    assert runner.calls == 1
    assert runner.thread_id is not None
    assert runner.thread_id != event_loop_thread
    assert build_arguments["event_loop"] is active_loop
    assert build_arguments["instruction"] == "solve the task"
    assert build_arguments["environment"] is not None
    assert dict(os.environ) == original_environment
    assert context.metadata == {
        "harbor_owned": "preserved",
        "bayesprobe_run_id": "tb_harbor-run",
        "bayesprobe_stop_reason": "max_cycles",
        "bayesprobe_cycles": 2,
        "terminal_actions": 3,
        "model_calls": 4,
        "runtime_lock_sha256": "sha256:" + "9" * 64,
        "runtime_budgets": {
            "max_total_actions": 24,
            "max_model_calls": 72,
            "max_provider_tokens": 160000,
            "max_output_tokens": 8192,
            "command_timeout_seconds": 120,
            "provider_timeout_seconds": 360,
            "signal_output_bytes": 32768,
            "provider_tokens_used": 0,
        },
    }
    assert set(context.metadata) - {"harbor_owned"} == _BAYESPROBE_METADATA
    assert artifacts.summaries == [
        {
            "bayesprobe_run_id": "tb_harbor-run",
            "bayesprobe_stop_reason": "max_cycles",
            "bayesprobe_cycles": 2,
            "terminal_actions": 3,
            "model_calls": 4,
            "runtime_lock_sha256": "sha256:" + "9" * 64,
            "runtime_budgets": {
                "max_total_actions": 24,
                "max_model_calls": 72,
                "max_provider_tokens": 160000,
                "max_output_tokens": 8192,
                "command_timeout_seconds": 120,
                "provider_timeout_seconds": 360,
                "signal_output_bytes": 32768,
                "provider_tokens_used": 0,
            },
        }
    ]
    assert _API_KEY not in repr(_agent(tmp_path))
    assert _API_KEY not in repr(context.metadata)
    assert _API_KEY not in repr(artifacts.summaries)


@pytest.mark.asyncio
async def test_agent_records_actual_provider_tokens_in_runtime_budget_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = SpyArtifacts()
    session = _session(runner=SpyRunner(), artifacts=artifacts)
    session.budget.provider_tokens_used = 160001
    monkeypatch.setattr(
        "bayesprobe_terminal_bench.agent.build_live_session",
        lambda **kwargs: session,
    )

    await _agent(tmp_path).run("solve the task", object(), AgentContext())

    assert artifacts.summaries[0]["runtime_budgets"]["provider_tokens_used"] == 160001


@pytest.mark.asyncio
async def test_agent_setup_does_not_mutate_the_harbor_environment(tmp_path: Path) -> None:
    class UnusedEnvironment:
        def __getattr__(self, name: str) -> object:
            raise AssertionError(f"setup must not access environment.{name}")

    await _agent(tmp_path).setup(UnusedEnvironment())


@pytest.mark.asyncio
async def test_agent_serializes_harbor_uuid_context_id_for_the_session_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = SpyArtifacts()
    session = _session(runner=SpyRunner(), artifacts=artifacts)
    build_arguments: dict[str, object] = {}

    def build_session(**kwargs: object) -> object:
        build_arguments.update(kwargs)
        return session

    monkeypatch.setattr(
        "bayesprobe_terminal_bench.agent.build_live_session",
        build_session,
    )
    agent = _agent(tmp_path)
    agent.context_id = UUID("12345678-1234-5678-1234-567812345678")

    await agent.run("solve the task", object(), AgentContext())

    assert build_arguments["context_id"] == "12345678-1234-5678-1234-567812345678"


@pytest.mark.asyncio
async def test_agent_replaces_invalid_metadata_with_its_bounded_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = SpyArtifacts()
    session = _session(runner=SpyRunner(), artifacts=artifacts)
    monkeypatch.setattr(
        "bayesprobe_terminal_bench.agent.build_live_session",
        lambda **kwargs: session,
    )
    context = AgentContext.model_construct(metadata="not-a-mapping")

    await _agent(tmp_path).run("solve the task", object(), context)

    assert context.metadata == artifacts.summaries[0]
    assert set(context.metadata) == _BAYESPROBE_METADATA
    assert all(
        isinstance(value, (str, int, dict)) for value in context.metadata.values()
    )


@pytest.mark.asyncio
async def test_agent_propagates_runner_failure_without_writing_or_leaking_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingRunner:
        def run_question(self, input: object) -> object:
            _raise_hostile_failure()

    artifacts = SpyArtifacts()
    monkeypatch.setattr(
        "bayesprobe_terminal_bench.agent.build_live_session",
        lambda **kwargs: _session(runner=FailingRunner(), artifacts=artifacts),
    )
    context = AgentContext(metadata={"harbor_owned": "preserved"})

    with pytest.raises(BayesProbeHarborAgentError) as failure:
        await _agent(tmp_path).run("solve the task", object(), context)

    _assert_stable_failure(
        failure.value,
        "BayesProbe Harbor agent failed: adapter_error",
        "adapter_error",
    )
    assert context.metadata == {"harbor_owned": "preserved"}
    assert artifacts.summaries == []
    assert artifacts.errors == [
        {"category": "adapter_error", "error_type": "HostileFailure"}
    ]


@pytest.mark.asyncio
async def test_agent_persists_provider_contract_category_before_rethrow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingRunner:
        def run_question(self, input: object) -> object:
            raise ProviderContractError(stage="terminal_task_frame", attempts=3)

    artifacts = SpyArtifacts()
    monkeypatch.setattr(
        "bayesprobe_terminal_bench.agent.build_live_session",
        lambda **kwargs: _session(runner=FailingRunner(), artifacts=artifacts),
    )

    with pytest.raises(BayesProbeHarborAgentError) as failure:
        await _agent(tmp_path).run("solve the task", object(), AgentContext())

    _assert_stable_failure(
        failure.value,
        "BayesProbe Harbor agent failed: provider_contract_error",
        "provider_contract_error",
    )
    assert artifacts.errors == [
        {
            "category": "provider_contract_error",
            "error_type": "ProviderContractError",
        }
    ]


@pytest.mark.asyncio
async def test_agent_persists_classified_post_artifact_construction_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = SpyArtifacts()
    monkeypatch.setattr(
        "bayesprobe_terminal_bench.agent.TrialArtifactStore",
        lambda *args, **kwargs: artifacts,
        raising=False,
    )

    def fail_after_artifacts(**kwargs: object) -> object:
        assert kwargs["artifacts"] is artifacts
        raise ProviderContractError(stage="terminal_probe_design", attempts=3)

    monkeypatch.setattr(
        "bayesprobe_terminal_bench.agent.build_live_session",
        fail_after_artifacts,
    )

    with pytest.raises(BayesProbeHarborAgentError) as failure:
        await _agent(tmp_path).run("solve the task", object(), AgentContext())

    assert failure.value.category == "adapter_error"
    assert artifacts.errors == [
        {
            "category": "provider_contract_error",
            "error_type": "ProviderContractError",
        },
        {
            "category": "adapter_error",
            "error_type": "TrajectoryExportError",
            "stage": "trajectory_export",
        },
    ]
    assert not (tmp_path / "trajectory.json").exists()


@pytest.mark.asyncio
async def test_agent_raises_stable_error_for_hostile_config_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = False

    def build_session(**kwargs: object) -> object:
        nonlocal started
        started = True
        raise AssertionError("config validation must happen before session construction")

    monkeypatch.setattr(
        "bayesprobe_terminal_bench.agent.build_live_session",
        build_session,
    )
    monkeypatch.setattr(
        "bayesprobe_terminal_bench.agent.TerminalBenchConfig",
        SimpleNamespace(from_sources=lambda extra_env: _raise_hostile_failure()),
    )
    agent = _agent(tmp_path)
    context = AgentContext(metadata={"harbor_owned": "preserved"})

    with pytest.raises(BayesProbeHarborAgentError) as failure:
        await agent.run("solve the task", object(), context)

    _assert_stable_failure(
        failure.value,
        "BayesProbe Harbor agent configuration failed: adapter_error",
        "adapter_error",
    )
    assert not started
    assert context.metadata == {"harbor_owned": "preserved"}


@pytest.mark.asyncio
async def test_agent_raises_stable_error_for_hostile_session_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def build_session(**kwargs: object) -> object:
        _raise_hostile_failure()

    monkeypatch.setattr(
        "bayesprobe_terminal_bench.agent.build_live_session",
        build_session,
    )
    context = AgentContext(metadata={"harbor_owned": "preserved"})

    with pytest.raises(BayesProbeHarborAgentError) as failure:
        await _agent(tmp_path).run("solve the task", object(), context)

    _assert_stable_failure(
        failure.value,
        "BayesProbe Harbor agent failed: adapter_error",
        "adapter_error",
    )
    assert context.metadata == {"harbor_owned": "preserved"}


@pytest.mark.asyncio
async def test_agent_cancellation_does_not_repeat_runner_or_update_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    released = threading.Event()
    finished = threading.Event()

    class BlockingRunner:
        def __init__(self) -> None:
            self.calls = 0

        def run_question(self, input: object) -> object:
            self.calls += 1
            started.set()
            try:
                assert released.wait(timeout=2)
            finally:
                finished.set()
            return SimpleNamespace(
                stop_reason=SimpleNamespace(value="max_cycles"),
                cycle_results=(),
            )

    runner = BlockingRunner()
    artifacts = SpyArtifacts()
    monkeypatch.setattr(
        "bayesprobe_terminal_bench.agent.build_live_session",
        lambda **kwargs: _session(runner=runner, artifacts=artifacts),
    )
    context = AgentContext(metadata={"harbor_owned": "preserved"})
    task = asyncio.create_task(_agent(tmp_path).run("solve the task", object(), context))

    assert await asyncio.to_thread(started.wait, 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    released.set()
    assert await asyncio.to_thread(finished.wait, 1)

    assert runner.calls == 1
    assert context.metadata == {"harbor_owned": "preserved"}
    assert artifacts.summaries == []
