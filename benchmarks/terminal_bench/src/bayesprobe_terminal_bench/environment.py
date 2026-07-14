from __future__ import annotations

import asyncio
import hashlib
import json
import posixpath
import re
import shlex
import tempfile
import time
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from threading import Lock
from typing import Protocol
from urllib.parse import unquote
from uuid import uuid4

from bayesprobe_terminal_bench.actions import (
    ActionObservation,
    ApplyPatchAction,
    ShellAction,
    TerminalAction,
    WriteFileAction,
    action_may_mutate,
)


class ExecResultLike(Protocol):
    stdout: str | None
    stderr: str | None
    return_code: int


class HarborEnvironmentLike(Protocol):
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | None = None,
    ) -> ExecResultLike: ...

    async def upload_file(self, source_path: Path | str, target_path: str) -> None: ...


@dataclass(frozen=True)
class LocalExecResult:
    stdout: str | None
    stderr: str | None
    return_code: int


class PolicyViolation(ValueError):
    pass


class ActionPolicy:
    """Reject direct access to evaluator-only files before Harbor receives it."""

    _PROTECTED = (
        "/logs/verifier",
        "/solution",
        "/tests",
        "/var/run/docker.sock",
        "/run/docker.sock",
        "docker.sock",
    )
    _PROTECTED_RELATIVE_PATHS = ("logs/verifier", "solution", "tests")
    _PROTECTED_GLOB_TARGETS = (
        "/logs/verifier/reward.txt",
        "/solution/answer.txt",
        "/tests/hidden.py",
        "/var/run/docker.sock",
        "/run/docker.sock",
        "logs/verifier/reward.txt",
        "solution/answer.txt",
        "tests/hidden.py",
    )
    _TOKEN_SPLIT = re.compile(r"[\s'\"`|;&<>]+")
    _SHELL_ESCAPE = re.compile(r"\\(.)")
    _PATCH_TARGET_PREFIXES = (
        "--- ",
        "+++ ",
        "*** Update File: ",
        "*** Add File: ",
        "*** Delete File: ",
    )

    def validate(self, action: TerminalAction) -> None:
        if isinstance(action, ShellAction):
            candidates = self._shell_candidates(action.command)
        elif isinstance(action, WriteFileAction):
            if not PurePosixPath(action.path).is_absolute():
                raise PolicyViolation("write_file path must be an absolute POSIX path")
            candidates = self._path_candidates(action.path)
        else:
            candidates = self._patch_candidates(action.patch, action.strip)

        if any(self._is_protected_path(candidate) for candidate in candidates):
            raise PolicyViolation("terminal action targets a protected path")

    @classmethod
    def _shell_candidates(cls, command: str) -> tuple[str, ...]:
        candidates = [command, *cls._TOKEN_SPLIT.split(command)]
        try:
            candidates.extend(shlex.split(command, posix=True))
        except ValueError:
            # Malformed quoting is potentially unsafe and will be executed by
            # neither the planner nor this policy.
            candidates.append("/tests")
        return cls._expand_candidates(candidates)

    @classmethod
    def _patch_candidates(cls, patch: str, strip: int) -> tuple[str, ...]:
        targets = cls._patch_target_paths(patch)
        stripped_targets = [cls._strip_patch_path(target, strip) for target in targets]
        return cls._expand_candidates([*targets, *stripped_targets])

    @classmethod
    def _path_candidates(cls, path: str) -> tuple[str, ...]:
        return cls._expand_candidates((path,))

    @classmethod
    def _patch_target_paths(cls, patch: str) -> tuple[str, ...]:
        targets: list[str] = []
        for line in patch.splitlines():
            for prefix in cls._PATCH_TARGET_PREFIXES:
                if line.startswith(prefix):
                    target = line.removeprefix(prefix).split("\t", 1)[0].strip()
                    if target:
                        targets.append(target)
                    break
            else:
                if not line.startswith("diff --git "):
                    continue
                try:
                    fields = shlex.split(line, posix=True)
                except ValueError:
                    continue
                targets.extend(fields[2:4])
        return tuple(targets)

    @staticmethod
    def _strip_patch_path(path: str, strip: int) -> str:
        parts = [
            part
            for part in re.sub(r"/+", "/", path.replace("\\", "/")).split("/")
            if part and part != "."
        ]
        return "/".join(parts[strip:])

    @classmethod
    def _expand_candidates(cls, candidates: list[str] | tuple[str, ...]) -> tuple[str, ...]:
        expanded: list[str] = []
        for candidate in candidates:
            if not candidate:
                continue
            expanded.append(candidate)

            percent_decoded = unquote(candidate)
            if percent_decoded != candidate:
                expanded.append(percent_decoded)

            unescaped = cls._SHELL_ESCAPE.sub(r"\1", candidate)
            if unescaped != candidate:
                expanded.append(unescaped)

            ansi_candidate = candidate[1:] if candidate.startswith("$") else candidate
            if re.search(r"\\(?:[0-7]{1,3}|x[0-9a-fA-F]{2}|u[0-9a-fA-F]{4})", ansi_candidate):
                try:
                    expanded.append(bytes(ansi_candidate, "utf-8").decode("unicode_escape"))
                except UnicodeDecodeError:
                    pass
        return tuple(expanded)

    @classmethod
    def _is_protected_path(cls, candidate: str) -> bool:
        normalized = posixpath.normpath(re.sub(r"/+", "/", candidate.replace("\\", "/")))
        if normalized in (".", ""):
            return False
        if any(marker in normalized for marker in ("*", "?", "[")):
            literal_prefix = re.split(r"[*?[]", normalized, maxsplit=1)[0]
            if any(
                fnmatchcase(protected, normalized) or protected.startswith(literal_prefix)
                for protected in cls._PROTECTED_GLOB_TARGETS
            ):
                return True
        if normalized == "docker.sock" or normalized.endswith("/docker.sock"):
            return True
        if normalized.startswith("/"):
            return any(
                normalized == protected or normalized.startswith(f"{protected}/")
                for protected in cls._PROTECTED
                if protected.startswith("/")
            )

        relative = normalized
        while relative == ".." or relative.startswith("../"):
            relative = relative[3:] if relative.startswith("../") else ""
        return any(
            relative == protected or relative.startswith(f"{protected}/")
            for protected in cls._PROTECTED_RELATIVE_PATHS
        )


class EnvironmentState:
    def __init__(self) -> None:
        self._version = 0
        self._lock = Lock()

    def current(self) -> str:
        with self._lock:
            return f"env:{self._version}"

    def advance(self) -> str:
        with self._lock:
            self._version += 1
            return f"env:{self._version}"


class HarborEnvironmentBridge:
    _NON_SHELL_TIMEOUT_SECONDS = 120
    _WAIT_GRACE_SECONDS = 5
    _HARBOR_DOCKER_TIMEOUT = re.compile(r"Command timed out after [1-9]\d* seconds")

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        environment: HarborEnvironmentLike,
        policy: ActionPolicy,
        output_limit_bytes: int,
    ) -> None:
        if isinstance(output_limit_bytes, bool) or output_limit_bytes < 1:
            raise ValueError("output_limit_bytes must be a positive integer")
        self._loop = loop
        self._environment = environment
        self._policy = policy
        self._output_limit_bytes = output_limit_bytes
        self._state = EnvironmentState()
        self._execution_lock = Lock()

    def execute(self, action: TerminalAction, action_index: int) -> ActionObservation:
        self._policy.validate(action)
        with self._execution_lock:
            return self._execute_locked(action, action_index)

    def _execute_locked(self, action: TerminalAction, action_index: int) -> ActionObservation:
        before = self._state.current()
        timeout_seconds = (
            action.timeout_seconds
            if isinstance(action, ShellAction)
            else self._NON_SHELL_TIMEOUT_SECONDS
        )
        started = time.monotonic()
        stdout = ""
        stderr = ""
        return_code: int | None = None
        timed_out = False
        error_category: str | None = None
        future: Future[ExecResultLike] | None = None

        if not self._loop.is_running():
            error_category = "transport"
            stderr = "Harbor action failed."
        else:
            try:
                coroutine = self._execute_async(action)
                try:
                    future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
                except Exception:
                    coroutine.close()
                    raise
                result = future.result(timeout=timeout_seconds + self._WAIT_GRACE_SECONDS)
                stdout = result.stdout or ""
                stderr = result.stderr or ""
                return_code = result.return_code
            except FutureTimeoutError:
                if future is not None:
                    future.cancel()
                timed_out = True
                error_category = "timeout"
                stderr = "Harbor action exceeded the configured timeout."
            except Exception as error:
                if self._is_harbor_docker_timeout(error):
                    timed_out = True
                    error_category = "timeout"
                    stderr = "Harbor action exceeded the configured timeout."
                else:
                    error_category = "transport"
                    stderr = "Harbor action failed."

        after = self._state.advance() if action_may_mutate(action) else before
        raw_output = json.dumps(
            {
                "stdout": stdout,
                "stderr": stderr,
                "return_code": return_code,
                "timed_out": timed_out,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        encoded_output = raw_output.encode("utf-8")
        model_facing_output = _bounded_utf8(encoded_output, self._output_limit_bytes)

        return ActionObservation(
            action_index=action_index,
            action=action,
            stdout=stdout,
            stderr=stderr,
            return_code=return_code,
            timed_out=timed_out,
            error_category=error_category,
            duration_ms=int((time.monotonic() - started) * 1000),
            pre_environment_state_id=before,
            post_environment_state_id=after,
            full_output_sha256=hashlib.sha256(encoded_output).hexdigest(),
            model_facing_output=model_facing_output,
            output_truncated=len(encoded_output) > self._output_limit_bytes,
        )

    async def _execute_async(self, action: TerminalAction) -> ExecResultLike:
        if isinstance(action, ShellAction):
            return await self._environment.exec(
                action.command,
                timeout_sec=action.timeout_seconds,
            )
        if isinstance(action, WriteFileAction):
            return await self._write_file(action)
        return await self._apply_patch(action)

    async def _write_file(self, action: WriteFileAction) -> LocalExecResult:
        local_path = self._write_temporary_file(action.content)
        try:
            await self._environment.upload_file(local_path, action.path)
            return LocalExecResult(
                stdout=f"wrote {action.path}",
                stderr="",
                return_code=0,
            )
        finally:
            self._remove_local_file(local_path)

    async def _apply_patch(self, action: ApplyPatchAction) -> ExecResultLike:
        remote_patch = f"/tmp/.bayesprobe-{uuid4().hex}.patch"
        local_patch = self._write_temporary_file(action.patch)
        try:
            await self._environment.upload_file(local_patch, remote_patch)
            command = f"patch --batch --forward -p{action.strip} < {shlex.quote(remote_patch)}"
            return await self._environment.exec(
                command,
                timeout_sec=self._NON_SHELL_TIMEOUT_SECONDS,
            )
        finally:
            self._remove_local_file(local_patch)
            self._schedule_remote_patch_cleanup(remote_patch)

    @classmethod
    def _is_harbor_docker_timeout(cls, error: Exception) -> bool:
        return isinstance(error, RuntimeError) and bool(
            cls._HARBOR_DOCKER_TIMEOUT.fullmatch(str(error))
        )

    @staticmethod
    def _write_temporary_file(content: str) -> Path:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
            handle.write(content)
            return Path(handle.name)

    @staticmethod
    def _remove_local_file(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass

    def _schedule_remote_patch_cleanup(self, remote_patch: str) -> None:
        coroutine = self._remove_remote_patch(remote_patch)
        try:
            task = asyncio.get_running_loop().create_task(coroutine)
        except Exception:
            coroutine.close()
            return
        task.add_done_callback(self._consume_cleanup_task)

    @staticmethod
    def _consume_cleanup_task(task: asyncio.Task[None]) -> None:
        try:
            task.result()
        except BaseException:
            pass

    async def _remove_remote_patch(self, remote_patch: str) -> None:
        try:
            await self._environment.exec(
                f"rm -f {shlex.quote(remote_patch)}",
                timeout_sec=30,
            )
        except BaseException:
            # A cleanup result is never an observation and cannot replace the
            # patch result that triggered it.
            pass


def _bounded_utf8(value: bytes, limit: int) -> str:
    if len(value) <= limit:
        return value.decode("utf-8")
    return value[:limit].decode("utf-8", errors="ignore")
