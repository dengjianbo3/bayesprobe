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


_API_KEY = "one-time-agent-secret"
_EXTRA_ENV = {
    "BAYESPROBE_BENCH_API_KEY": _API_KEY,
    "BAYESPROBE_BENCH_MODEL": "deepseek-v4-flash",
}
_BAYESPROBE_METADATA = {
    "bayesprobe_run_id",
    "bayesprobe_stop_reason",
    "bayesprobe_cycles",
    "terminal_actions",
    "model_calls",
}
WORKTREE_DIR = Path(__file__).resolve().parents[3]


class SpyArtifacts:
    def __init__(self) -> None:
        self.summaries: list[dict[str, object]] = []

    def write_summary(self, payload: dict[str, object]) -> None:
        self.summaries.append(dict(payload))


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
        budget=SimpleNamespace(actions_used=3, model_calls_used=4),
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
) -> None:
    assert type(failure) is BayesProbeHarborAgentError
    assert str(failure) == expected_message
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
    }
    assert set(context.metadata) - {"harbor_owned"} == _BAYESPROBE_METADATA
    assert artifacts.summaries == [
        {
            "bayesprobe_run_id": "tb_harbor-run",
            "bayesprobe_stop_reason": "max_cycles",
            "bayesprobe_cycles": 2,
            "terminal_actions": 3,
            "model_calls": 4,
        }
    ]
    assert _API_KEY not in repr(_agent(tmp_path))
    assert _API_KEY not in repr(context.metadata)
    assert _API_KEY not in repr(artifacts.summaries)


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
    assert all(isinstance(value, (str, int)) for value in context.metadata.values())


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

    _assert_stable_failure(failure.value, "BayesProbe Harbor agent execution failed")
    assert context.metadata == {"harbor_owned": "preserved"}
    assert artifacts.summaries == []


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

    _assert_stable_failure(failure.value, "BayesProbe Harbor agent configuration failed")
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

    _assert_stable_failure(failure.value, "BayesProbe Harbor agent execution failed")
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
