from __future__ import annotations

import asyncio
import gc
import hashlib
import json
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path

import pytest

from bayesprobe_terminal_bench.actions import ApplyPatchAction, ShellAction, WriteFileAction
from bayesprobe_terminal_bench.environment import (
    ActionPolicy,
    HarborEnvironmentBridge,
    PolicyViolation,
)


@dataclass(frozen=True)
class FakeExecResult:
    stdout: str | None = "observed"
    stderr: str | None = ""
    return_code: int = 0


class RecordingEnvironment:
    def __init__(self, result: FakeExecResult | None = None) -> None:
        self.result = result or FakeExecResult()
        self.commands: list[tuple[str, int | None]] = []
        self.uploads: list[tuple[str, str]] = []
        self.exec_loop: asyncio.AbstractEventLoop | None = None
        self.exec_thread_id: int | None = None

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | None = None,
    ) -> FakeExecResult:
        self.commands.append((command, timeout_sec))
        self.exec_loop = asyncio.get_running_loop()
        self.exec_thread_id = threading.get_ident()
        return self.result

    async def upload_file(self, source_path: str, target_path: str) -> None:
        self.uploads.append((str(source_path), target_path))


def make_bridge(
    loop: asyncio.AbstractEventLoop,
    environment: RecordingEnvironment,
    *,
    output_limit_bytes: int = 32_768,
) -> HarborEnvironmentBridge:
    return HarborEnvironmentBridge(
        loop=loop,
        environment=environment,
        policy=ActionPolicy(),
        output_limit_bytes=output_limit_bytes,
    )


@pytest.mark.asyncio
async def test_bridge_submits_harbor_coroutine_from_worker_to_captured_loop() -> None:
    loop = asyncio.get_running_loop()
    environment = RecordingEnvironment()
    bridge = make_bridge(loop, environment)

    observation = await asyncio.to_thread(bridge.execute, ShellAction(command="pwd"), 1)

    assert observation.stdout == "observed"
    assert observation.pre_environment_state_id == "env:0"
    assert observation.post_environment_state_id == "env:0"
    assert environment.exec_loop is loop
    assert environment.exec_thread_id == threading.get_ident()
    assert environment.commands == [("pwd", 120)]


@pytest.mark.asyncio
async def test_only_provably_read_only_shell_actions_preserve_environment_lineage() -> None:
    environment = RecordingEnvironment(FakeExecResult(return_code=7))
    bridge = make_bridge(asyncio.get_running_loop(), environment)

    read_observation = await asyncio.to_thread(bridge.execute, ShellAction(command="pwd"), 1)
    mutate_observation = await asyncio.to_thread(
        bridge.execute,
        ShellAction(command="touch /app/value.txt"),
        2,
    )

    assert read_observation.return_code == 7
    assert read_observation.error_category is None
    assert read_observation.pre_environment_state_id == "env:0"
    assert read_observation.post_environment_state_id == "env:0"
    assert mutate_observation.return_code == 7
    assert mutate_observation.pre_environment_state_id == "env:0"
    assert mutate_observation.post_environment_state_id == "env:1"


class UploadFailureEnvironment(RecordingEnvironment):
    async def upload_file(self, source_path: str, target_path: str) -> None:
        raise RuntimeError("BAYESPROBE_BENCH_API_KEY=provider-secret")


@pytest.mark.asyncio
async def test_transport_failure_advances_mutating_state_without_leaking_error_secrets() -> None:
    environment = UploadFailureEnvironment()
    bridge = make_bridge(asyncio.get_running_loop(), environment)

    observation = await asyncio.to_thread(
        bridge.execute,
        WriteFileAction(path="/app/value.txt", content="value"),
        1,
    )

    assert observation.return_code is None
    assert observation.error_category == "transport"
    assert observation.stderr == "Harbor action failed."
    assert "provider-secret" not in observation.stderr
    assert "provider-secret" not in observation.model_facing_output
    assert observation.pre_environment_state_id == "env:0"
    assert observation.post_environment_state_id == "env:1"


@pytest.mark.asyncio
async def test_closed_captured_loop_is_a_clean_transport_failure_with_mutating_lineage() -> None:
    closed_loop = asyncio.new_event_loop()
    closed_loop.close()
    bridge = make_bridge(closed_loop, RecordingEnvironment())

    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        observation = await asyncio.to_thread(
            bridge.execute,
            ShellAction(command="touch /app/value.txt"),
            1,
        )
        gc.collect()

    assert observation.error_category == "transport"
    assert observation.stderr == "Harbor action failed."
    assert observation.pre_environment_state_id == "env:0"
    assert observation.post_environment_state_id == "env:1"
    assert not [
        warning
        for warning in captured_warnings
        if issubclass(warning.category, RuntimeWarning)
        and "was never awaited" in str(warning.message)
    ]


class HangingEnvironment(RecordingEnvironment):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | None = None,
    ) -> FakeExecResult:
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        raise AssertionError("the blocked Harbor action unexpectedly completed")


@pytest.mark.asyncio
async def test_timeout_is_bounded_cancels_harbor_work_and_advances_mutating_state() -> None:
    environment = HangingEnvironment()
    bridge = make_bridge(asyncio.get_running_loop(), environment)

    worker = asyncio.create_task(
        asyncio.to_thread(
            bridge.execute,
            ShellAction(command="touch /app/value.txt", timeout_seconds=1),
            1,
        )
    )
    await asyncio.wait_for(environment.started.wait(), timeout=1)
    observation = await asyncio.wait_for(worker, timeout=7)
    await asyncio.wait_for(environment.cancelled.wait(), timeout=1)

    assert observation.timed_out is True
    assert observation.error_category == "timeout"
    assert observation.return_code is None
    assert observation.stderr == "Harbor action exceeded the configured timeout."
    assert observation.pre_environment_state_id == "env:0"
    assert observation.post_environment_state_id == "env:1"


class PatchCleanupFailureEnvironment(RecordingEnvironment):
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | None = None,
    ) -> FakeExecResult:
        self.commands.append((command, timeout_sec))
        if command.startswith("rm -f "):
            raise RuntimeError("cleanup-secret")
        return FakeExecResult(stdout="patch output", stderr="patch error", return_code=7)


@pytest.mark.asyncio
async def test_patch_cleanup_failure_never_masks_nonzero_execution_result() -> None:
    environment = PatchCleanupFailureEnvironment()
    bridge = make_bridge(asyncio.get_running_loop(), environment)

    observation = await asyncio.to_thread(
        bridge.execute,
        ApplyPatchAction(patch="*** Begin Patch\n*** End Patch\n"),
        1,
    )

    assert observation.return_code == 7
    assert observation.stdout == "patch output"
    assert observation.stderr == "patch error"
    assert observation.error_category is None
    assert observation.post_environment_state_id == "env:1"
    assert "cleanup-secret" not in observation.model_facing_output
    assert len(environment.uploads) == 1
    assert any(command.startswith("rm -f ") for command, _ in environment.commands)


@pytest.mark.asyncio
async def test_local_cleanup_failure_never_masks_patch_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = RecordingEnvironment(FakeExecResult(return_code=7))
    bridge = make_bridge(asyncio.get_running_loop(), environment)

    def fail_unlink(self: Path, missing_ok: bool = False) -> None:
        raise RuntimeError("local-cleanup-secret")

    monkeypatch.setattr(Path, "unlink", fail_unlink)
    observation = await asyncio.to_thread(
        bridge.execute,
        ApplyPatchAction(patch="*** Begin Patch\n*** End Patch\n"),
        1,
    )

    assert observation.return_code == 7
    assert observation.error_category is None
    assert "local-cleanup-secret" not in observation.model_facing_output


class PatchPrimaryAndCleanupFailureEnvironment(PatchCleanupFailureEnvironment):
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | None = None,
    ) -> FakeExecResult:
        self.commands.append((command, timeout_sec))
        if command.startswith("rm -f "):
            raise RuntimeError("cleanup-secret")
        raise RuntimeError("primary-secret")


@pytest.mark.asyncio
async def test_patch_cleanup_failure_never_masks_primary_transport_failure() -> None:
    environment = PatchPrimaryAndCleanupFailureEnvironment()
    bridge = make_bridge(asyncio.get_running_loop(), environment)

    observation = await asyncio.to_thread(
        bridge.execute,
        ApplyPatchAction(patch="*** Begin Patch\n*** End Patch\n"),
        1,
    )

    assert observation.error_category == "transport"
    assert observation.stderr == "Harbor action failed."
    assert observation.post_environment_state_id == "env:1"
    assert "primary-secret" not in observation.model_facing_output
    assert "cleanup-secret" not in observation.model_facing_output


@pytest.mark.asyncio
async def test_output_hash_covers_complete_raw_observation_before_model_truncation() -> None:
    environment = RecordingEnvironment(
        FakeExecResult(stdout="a" * 80, stderr="b" * 40, return_code=3)
    )
    bridge = make_bridge(asyncio.get_running_loop(), environment, output_limit_bytes=32)

    observation = await asyncio.to_thread(bridge.execute, ShellAction(command="pwd"), 1)

    full_output = json.dumps(
        {
            "return_code": 3,
            "stderr": "b" * 40,
            "stdout": "a" * 80,
            "timed_out": False,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    encoded = full_output.encode("utf-8")
    assert observation.full_output_sha256 == hashlib.sha256(encoded).hexdigest()
    assert observation.model_facing_output == encoded[:32].decode("utf-8", errors="replace")
    assert observation.output_truncated is True


@pytest.mark.parametrize(
    "action",
    [
        ShellAction(command="cat /logs/verifier/reward.txt"),
        ShellAction(command=r"cat /workspace/../tests/hidden.py"),
        ShellAction(command=r"cat /te\\sts/hidden.py"),
        ShellAction(command="cat %2Fsolution%2Fanswer.txt"),
        ShellAction(command=r"cat $'\x2fvar\x2frun\x2fdocker.sock'"),
        WriteFileAction(path="/tmp/../solution/answer.txt", content="x"),
        WriteFileAction(path=r"tests\\hidden.py", content="x"),
        ApplyPatchAction(
            patch="*** Begin Patch\n*** Update File: a/../tests/hidden.py\n*** End Patch\n"
        ),
        ApplyPatchAction(
            patch="*** Begin Patch\n*** Update File: /var/run/../run/docker.sock\n*** End Patch\n"
        ),
    ],
)
def test_policy_blocks_protected_paths_after_normalization_or_obfuscation(action: object) -> None:
    with pytest.raises(PolicyViolation, match="terminal action targets a protected path"):
        ActionPolicy().validate(action)  # type: ignore[arg-type]


def test_policy_rejection_has_no_transport_or_environment_side_effect() -> None:
    policy = ActionPolicy()

    with pytest.raises(PolicyViolation):
        policy.validate(WriteFileAction(path="/tests/hidden.py", content="x"))
