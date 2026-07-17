from __future__ import annotations

import asyncio
import os
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from harbor.models.agent.context import AgentContext

from bayesprobe_terminal_bench.direct_agent import (
    DirectHarborAgent,
    DirectHarborAgentError,
    build_direct_session,
)
from bayesprobe_terminal_bench.config import TerminalBenchConfig
from bayesprobe_terminal_bench.provider_contract import ProviderContractError


_API_KEY = "one-time-direct-secret"
_EXTRA_ENV = {
    "BAYESPROBE_BENCH_API_KEY": _API_KEY,
    "BAYESPROBE_BENCH_MODEL": "test-model",
    "BAYESPROBE_BENCH_TASK_TIMEOUT_SECONDS": "900",
}
WORKTREE_DIR = Path(__file__).resolve().parents[3]


class _Artifacts:
    def __init__(self) -> None:
        self.summaries: list[dict[str, object]] = []
        self.errors: list[dict[str, object]] = []

    def write_summary(self, payload: dict[str, object]) -> None:
        self.summaries.append(dict(payload))

    def append_error(self, payload: dict[str, object]) -> None:
        self.errors.append(dict(payload))


class _Controller:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[str] = []
        self.thread_id: int | None = None

    def run(self, instruction: str) -> object:
        self.calls.append(instruction)
        self.thread_id = threading.get_ident()
        return self.result


def _agent(tmp_path: Path) -> DirectHarborAgent:
    return DirectHarborAgent(
        logs_dir=tmp_path,
        model_name="test-model",
        extra_env=_EXTRA_ENV,
    )


def test_exact_uv_project_command_imports_direct_agent() -> None:
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
                "from bayesprobe_terminal_bench.direct_agent import DirectHarborAgent; "
                "print(DirectHarborAgent.import_path())"
            ),
        ],
        capture_output=True,
        check=False,
        cwd=WORKTREE_DIR,
        env=environment,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == (
        "bayesprobe_terminal_bench.direct_agent:DirectHarborAgent"
    )


@pytest.mark.asyncio
async def test_direct_session_shares_budget_deadline_and_provider_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "bayesprobe_terminal_bench.direct_agent.validate_runtime_lock",
        lambda *args, **kwargs: {
            "expected_provider_model": "test-model",
            "expected_system_fingerprint": "fp-locked",
            "agent_timeout_seconds": 100,
        },
    )
    config = TerminalBenchConfig(
        model="test-model",
        lock_path=tmp_path / "unused.lock.json",
        task_timeout_seconds=100,
    )

    session = build_direct_session(
        config=config,
        api_key=_API_KEY,
        environment=object(),
        event_loop=asyncio.get_running_loop(),
        logs_dir=tmp_path / "logs",
        session_id="session/id",
    )

    assert session.controller._budget is session.budget
    assert session.controller._planner._budget is session.budget
    assert session.budget.max_provider_tokens == 160_000
    assert session.controller._planner._client._deadline is session.deadline
    assert session.controller._bridge._deadline is session.deadline
    assert session.budget._reservation_guard.__self__ is session.deadline


@pytest.mark.asyncio
async def test_direct_agent_runs_once_in_worker_and_records_safe_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = _Artifacts()
    controller = _Controller(
        SimpleNamespace(
            stop_reason="completed",
            completion_summary="verified",
            steps=3,
            observations=2,
        )
    )
    session = SimpleNamespace(
        controller=controller,
        artifacts=artifacts,
        budget=SimpleNamespace(actions_used=2, model_calls_used=3),
    )
    arguments: dict[str, object] = {}

    def build_session(**kwargs: object) -> object:
        arguments.update(kwargs)
        return session

    monkeypatch.setattr(
        "bayesprobe_terminal_bench.direct_agent.build_direct_session",
        build_session,
    )
    context = AgentContext(metadata={"harbor_owned": "preserved"})
    event_thread = threading.get_ident()
    active_loop = asyncio.get_running_loop()

    await _agent(tmp_path).run("solve it", object(), context)

    assert controller.calls == ["solve it"]
    assert controller.thread_id is not None
    assert controller.thread_id != event_thread
    assert arguments["event_loop"] is active_loop
    assert context.metadata == {
        "harbor_owned": "preserved",
        "experiment_arm": "direct",
        "stop_reason": "completed",
        "react_steps": 3,
        "observations": 2,
        "terminal_actions": 2,
        "model_calls": 3,
    }
    assert artifacts.summaries == [
        {
            "experiment_arm": "direct",
            "stop_reason": "completed",
            "react_steps": 3,
            "observations": 2,
            "terminal_actions": 2,
            "model_calls": 3,
        }
    ]
    assert _API_KEY not in repr(context.metadata)
    assert _API_KEY not in repr(artifacts.summaries)


@pytest.mark.asyncio
async def test_direct_agent_setup_does_not_touch_environment(tmp_path: Path) -> None:
    class _UnusedEnvironment:
        def __getattr__(self, name: str) -> object:
            raise AssertionError(f"setup must not access environment.{name}")

    await _agent(tmp_path).setup(_UnusedEnvironment())


@pytest.mark.asyncio
async def test_direct_agent_exposes_stable_error_without_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(**kwargs: object) -> object:
        raise RuntimeError(f"provider failed with {_API_KEY}")

    monkeypatch.setattr(
        "bayesprobe_terminal_bench.direct_agent.build_direct_session",
        fail,
    )
    context = AgentContext(metadata={"harbor_owned": "preserved"})

    with pytest.raises(DirectHarborAgentError) as failure:
        await _agent(tmp_path).run("solve it", object(), context)

    assert str(failure.value) == "Direct Harbor agent failed: adapter_error"
    assert failure.value.category == "adapter_error"
    assert failure.value.__cause__ is None
    assert failure.value.__context__ is None
    assert _API_KEY not in repr(failure.value)
    assert context.metadata == {"harbor_owned": "preserved"}


@pytest.mark.asyncio
async def test_direct_agent_persists_classified_post_artifact_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = _Artifacts()
    monkeypatch.setattr(
        "bayesprobe_terminal_bench.direct_agent.TrialArtifactStore",
        lambda *args, **kwargs: artifacts,
    )

    def fail_after_artifacts(**kwargs: object) -> object:
        assert kwargs["artifacts"] is artifacts
        raise ProviderContractError(stage="react_step", attempts=3)

    monkeypatch.setattr(
        "bayesprobe_terminal_bench.direct_agent.build_direct_session",
        fail_after_artifacts,
    )

    with pytest.raises(DirectHarborAgentError) as failure:
        await _agent(tmp_path).run("solve it", object(), AgentContext())

    assert str(failure.value) == (
        "Direct Harbor agent failed: provider_contract_error"
    )
    assert failure.value.category == "provider_contract_error"
    assert artifacts.errors == [
        {
            "category": "provider_contract_error",
            "error_type": "ProviderContractError",
        }
    ]


@pytest.mark.asyncio
async def test_direct_agent_propagates_cancellation_without_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    released = threading.Event()

    class _BlockingController:
        def run(self, instruction: str) -> object:
            started.set()
            released.wait(timeout=2)
            return SimpleNamespace(
                stop_reason="completed",
                steps=1,
                observations=0,
            )

    artifacts = _Artifacts()
    monkeypatch.setattr(
        "bayesprobe_terminal_bench.direct_agent.build_direct_session",
        lambda **kwargs: SimpleNamespace(
            controller=_BlockingController(),
            artifacts=artifacts,
            budget=SimpleNamespace(actions_used=0, model_calls_used=1),
        ),
    )
    context = AgentContext(metadata={"harbor_owned": "preserved"})
    task = asyncio.create_task(_agent(tmp_path).run("solve it", object(), context))
    await asyncio.to_thread(started.wait, 2)
    task.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        released.set()

    assert context.metadata == {"harbor_owned": "preserved"}
    assert artifacts.summaries == []
