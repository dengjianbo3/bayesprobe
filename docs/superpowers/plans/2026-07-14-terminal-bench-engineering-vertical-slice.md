# Terminal-Bench Engineering Vertical Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated Harbor 0.18.0 adapter that runs one real Terminal-Bench 2.0 task through the existing public BayesProbe kernel and proves the integration with deterministic tests and an official verifier smoke run.

**Architecture:** The benchmark project is a downstream consumer of `bayesprobe`, importing only its top-level public interface. A Harbor `BaseAgent` runs the existing synchronous `AutonomousQuestionRunner` in a worker thread; a benchmark-owned `ProbeToolGateway` submits terminal actions to Harbor's async `BaseEnvironment` and converts only observed action results into `ExternalSignal` records. No benchmark-owned autonomous loop, Evidence gate, posterior updater, or completion controller is introduced.

**Tech Stack:** Python 3.12+, uv nested project, Harbor 0.18.0, Pydantic 2, OpenAI-compatible Chat Completions, pytest, pytest-asyncio, Docker.

## Global Constraints

- Do not modify `bayesprobe/`, the root `pyproject.toml`, or root test semantics.
- Production source under `benchmarks/terminal_bench/src/` may import BayesProbe only with `from bayesprobe import ...`.
- Do not create `loop.py`, `core.py`, `evidence.py`, or `updater.py` in the benchmark package.
- Use the real public `AutonomousQuestionRunner`, `BayesProbeCore`, `ProbeExecutor`, task framing, Evidence path, updater, and ledger.
- The benchmark Adapter may implement only public extension interfaces and Harbor integration behavior.
- Model-produced terminal plans are operational plans, never Signals or Evidence.
- Only Harbor action observations use `EpistemicOrigin.TOOL_RESULT` and enter the Evidence path.
- An action-planning or policy failure returns no Signal and cannot directionally update the posterior.
- Use `max_cycles=8`, `max_probes_per_cycle=2`, at most three actions per Probe, 24 terminal actions, and 40 logical model calls. A terminal-plan repair consumes a call; transport retries remain inside that logical call, and telemetry records the attempts exposed by the Adapter.
- Use a 120-second command timeout, 360-second provider timeout, and 32 KB model-facing output cap per action.
- Do not add a benchmark-owned `finish` action. The existing runner stop reason ends the agent phase, after which Harbor invokes the verifier.
- Provider secrets exist only in `BAYESPROBE_BENCH_API_KEY`; never write the value to config, traces, metadata, or Git.
- Pin Harbor to `0.18.0`; do not float the harness version during the smoke milestone.
- Use test-first red-green-refactor for every production module and commit after each independently testable task.
- This plan delivers only the BayesProbe engineering vertical slice. The ReAct control and paired experiment runner are a separate follow-up plan after this slice passes.

## File Map

| File | Responsibility |
| --- | --- |
| `benchmarks/terminal_bench/pyproject.toml` | Nested dependency and test boundary |
| `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/actions.py` | Terminal action and observation contracts |
| `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/config.py` | Secret-free provider, budget, and lock configuration |
| `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/artifacts.py` | Append-only plans, observations, telemetry, and summaries |
| `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/environment.py` | Harbor async environment bridge and action policy |
| `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/planning.py` | OpenAI-compatible terminal-plan requests |
| `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/signals.py` | Action-observation to `ExternalSignal` conversion |
| `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/gateway.py` | Public `ProbeToolGateway` Adapter composition |
| `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/runner_factory.py` | Construction of the real public BayesProbe runner |
| `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/agent.py` | Harbor `BaseAgent` entry point and worker-thread handoff |
| `benchmarks/terminal_bench/scripts/write_benchmark_lock.py` | Oracle validation and immutable run lock generation |
| `benchmarks/terminal_bench/scripts/validate_smoke_run.py` | Verifier and BayesProbe trace acceptance checks |

---

### Task 1: Scaffold the isolated project and enforce public reuse

**Files:**

- Create: `benchmarks/terminal_bench/pyproject.toml`
- Create: `benchmarks/terminal_bench/.gitignore`
- Create: `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/__init__.py`
- Create: `benchmarks/terminal_bench/tests/test_public_reuse.py`
- Generate: `benchmarks/terminal_bench/uv.lock`

**Interfaces:**

- The nested project installs the repository root as editable `bayesprobe[openai]`.
- `test_public_reuse.py` is a permanent AST gate against private imports and copied kernel modules.

- [ ] **Step 1: Create project metadata and the failing reuse test**

```toml
[project]
name = "bayesprobe-terminal-bench"
version = "0.1.0"
description = "Harbor adapter for evaluating BayesProbe on Terminal-Bench 2.0"
requires-python = ">=3.12"
dependencies = [
  "bayesprobe[openai]",
  "harbor==0.18.0",
  "pydantic>=2.7,<3",
]

[dependency-groups]
dev = ["pytest>=8,<9", "pytest-asyncio>=0.23,<2"]

[tool.uv.sources]
bayesprobe = { path = "../..", editable = true }

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src", "scripts"]
asyncio_mode = "auto"
```

```python
# tests/test_public_reuse.py
from __future__ import annotations

import ast
from pathlib import Path

import bayesprobe


PROJECT_DIR = Path(__file__).resolve().parents[1]
PACKAGE_DIR = PROJECT_DIR / "src" / "bayesprobe_terminal_bench"
FORBIDDEN_FILES = {"loop.py", "core.py", "evidence.py", "updater.py"}


def test_nested_project_is_materialized() -> None:
    assert (PACKAGE_DIR / "__init__.py").is_file()
    assert (PROJECT_DIR / "uv.lock").is_file()


def test_required_runtime_types_are_public_root_exports() -> None:
    required = {
        "AutonomousQuestionRunner",
        "BayesProbeCore",
        "ExternalSignal",
        "ProbeExecutor",
        "ProbeToolGateway",
    }
    assert required.issubset(set(bayesprobe.__all__))


def test_benchmark_has_no_shadow_kernel_modules() -> None:
    assert {path.name for path in PACKAGE_DIR.glob("*.py")}.isdisjoint(FORBIDDEN_FILES)


def test_production_source_uses_only_bayesprobe_root_imports() -> None:
    violations: list[str] = []
    for source_path in sorted(PACKAGE_DIR.glob("*.py")):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "bayesprobe" or alias.name.startswith("bayesprobe."):
                        violations.append(f"{source_path.name}:{node.lineno}: import {alias.name}")
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.startswith("bayesprobe."):
                    violations.append(f"{source_path.name}:{node.lineno}: from {module}")
    assert violations == []
```

- [ ] **Step 2: Verify RED**

Run: `uv run --project benchmarks/terminal_bench pytest benchmarks/terminal_bench/tests/test_public_reuse.py -q`

Expected: FAIL because the package marker and nested lock do not exist.

- [ ] **Step 3: Add the package marker and resolve dependencies**

```python
"""Terminal-Bench adapter for the public BayesProbe package."""

__version__ = "0.1.0"
```

Run:

```bash
uv lock --project benchmarks/terminal_bench
uv sync --project benchmarks/terminal_bench --group dev
```

Expected: `uv.lock` resolves Harbor exactly `0.18.0` and the editable root package.

- [ ] **Step 4: Verify GREEN and root isolation**

Run:

```bash
uv run --project benchmarks/terminal_bench pytest benchmarks/terminal_bench/tests/test_public_reuse.py -q
git diff --exit-code -- bayesprobe pyproject.toml
git diff --exit-code 068b414..HEAD -- bayesprobe pyproject.toml
```

Expected: tests PASS and the second command prints nothing.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/terminal_bench/pyproject.toml benchmarks/terminal_bench/uv.lock benchmarks/terminal_bench/.gitignore benchmarks/terminal_bench/src/bayesprobe_terminal_bench/__init__.py benchmarks/terminal_bench/tests/test_public_reuse.py
git commit -m "build: scaffold Terminal-Bench adapter project"
```

### Task 2: Define action, observation, configuration, and budget contracts

**Files:**

- Create: `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/actions.py`
- Create: `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/config.py`
- Create: `benchmarks/terminal_bench/tests/test_actions.py`
- Create: `benchmarks/terminal_bench/tests/test_config.py`

**Interfaces:**

- `TerminalProbePlan` accepts one to three `ShellAction`, `WriteFileAction`, or `ApplyPatchAction` values.
- `action_may_mutate` distrusts model labels: only a conservative allowlist of simple shell reads is treated as non-mutating.
- `ActionObservation` is the only input accepted by Signal conversion.
- `RunBudget` provides thread-safe hard counters for actions and model calls.
- `TerminalBenchConfig.from_sources(extra_env)` merges host environment and Harbor `extra_env`, returns the API key separately, and never stores its value.

- [ ] **Step 1: Write failing contract tests**

```python
import json

import pytest
from pydantic import ValidationError

from bayesprobe_terminal_bench.actions import ShellAction, TerminalProbePlan, WriteFileAction
from bayesprobe_terminal_bench.config import BudgetExhausted, RunBudget, TerminalBenchConfig


def test_inspect_plan_rejects_mutation() -> None:
    with pytest.raises(ValidationError, match="inspect plans require provably read-only actions"):
        TerminalProbePlan(
            mode="inspect",
            actions=[ShellAction(command="touch /tmp/x", mutates_environment=True)],
            expected_observation="The filesystem state is visible.",
        )


def test_model_cannot_mislabel_a_mutating_shell_command() -> None:
    with pytest.raises(ValidationError, match="inspect plans require provably read-only actions"):
        TerminalProbePlan(
            mode="inspect",
            actions=[ShellAction(command="rm -f output.txt", mutates_environment=False)],
            expected_observation="The output is absent.",
        )


def test_verify_allows_shell_but_not_direct_file_writes() -> None:
    plan = TerminalProbePlan(
        mode="verify",
        actions=[ShellAction(command="pytest -q", mutates_environment=True)],
        expected_observation="The test result is observed.",
    )
    assert plan.mode == "verify"
    with pytest.raises(ValidationError, match="verify plans accept shell actions only"):
        TerminalProbePlan(
            mode="verify",
            actions=[WriteFileAction(path="/app/result.txt", content="x")],
            expected_observation="A file is written.",
        )


def test_shared_budget_is_hard() -> None:
    budget = RunBudget(max_actions=1, max_model_calls=1)
    assert budget.reserve_action() == 1
    assert budget.reserve_model_call() == 1
    with pytest.raises(BudgetExhausted, match="terminal action budget exhausted"):
        budget.reserve_action()
    with pytest.raises(BudgetExhausted, match="model call budget exhausted"):
        budget.reserve_model_call()


def test_extra_env_wins_and_config_never_serializes_key_value(monkeypatch) -> None:
    monkeypatch.setenv("BAYESPROBE_BENCH_API_KEY", "host-secret")
    config, api_key = TerminalBenchConfig.from_sources({
        "BAYESPROBE_BENCH_API_KEY": "one-time-provider-secret",
        "BAYESPROBE_BENCH_MODEL": "deepseek-v4-flash",
    })
    assert api_key == "one-time-provider-secret"
    assert config.api_key_env == "BAYESPROBE_BENCH_API_KEY"
    assert "one-time-provider-secret" not in json.dumps(config.model_dump(mode="json"))
    assert "host-secret" not in json.dumps(config.model_dump(mode="json"))
```

- [ ] **Step 2: Verify RED**

Run: `uv run --project benchmarks/terminal_bench pytest benchmarks/terminal_bench/tests/test_actions.py benchmarks/terminal_bench/tests/test_config.py -q`

Expected: import failures because the contract modules do not exist.

- [ ] **Step 3: Implement strict action and observation models**

```python
from __future__ import annotations

import shlex
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShellAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    type: Literal["shell"] = "shell"
    command: str = Field(min_length=1, max_length=32_768)
    timeout_seconds: int = Field(default=120, ge=1, le=120)
    mutates_environment: bool = False


class WriteFileAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    type: Literal["write_file"] = "write_file"
    path: str = Field(min_length=1, max_length=4_096)
    content: str = Field(max_length=1_000_000)


class ApplyPatchAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    type: Literal["apply_patch"] = "apply_patch"
    patch: str = Field(min_length=1, max_length=1_000_000)
    strip: int = Field(default=0, ge=0, le=3)


TerminalAction = Annotated[
    ShellAction | WriteFileAction | ApplyPatchAction,
    Field(discriminator="type"),
]


_READ_ONLY_COMMANDS = frozenset({
    "cat", "file", "grep", "head", "ls", "md5sum", "pwd", "rg", "sha256sum",
    "stat", "tail", "test", "wc", "which",
})
_READ_ONLY_GIT_SUBCOMMANDS = frozenset({
    "diff", "log", "ls-files", "rev-parse", "show", "status",
})
_SHELL_COMPOSITION_MARKERS = ("\n", ";", "&&", "||", "|", ">", "<", "`", "$(")


def shell_command_is_provably_read_only(command: str) -> bool:
    if any(marker in command for marker in _SHELL_COMPOSITION_MARKERS):
        return False
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return False
    if not tokens:
        return False
    executable = Path(tokens[0]).name
    if executable == "git":
        return len(tokens) >= 2 and tokens[1] in _READ_ONLY_GIT_SUBCOMMANDS
    return executable in _READ_ONLY_COMMANDS


def action_may_mutate(action: TerminalAction) -> bool:
    if not isinstance(action, ShellAction):
        return True
    return action.mutates_environment or not shell_command_is_provably_read_only(
        action.command
    )


class TerminalProbePlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    mode: Literal["inspect", "intervene", "verify"]
    actions: list[TerminalAction] = Field(min_length=1, max_length=3)
    expected_observation: str = Field(min_length=1, max_length=4_096)

    @model_validator(mode="after")
    def validate_mode(self) -> "TerminalProbePlan":
        mutating = [action for action in self.actions if action_may_mutate(action)]
        if self.mode == "inspect" and mutating:
            raise ValueError("inspect plans require provably read-only actions")
        if self.mode == "verify" and any(
            not isinstance(action, ShellAction) for action in self.actions
        ):
            raise ValueError("verify plans accept shell actions only")
        if self.mode == "intervene" and not mutating:
            raise ValueError("intervene plans require a mutating action")
        return self


class ActionObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    action_index: int = Field(ge=1)
    action: TerminalAction
    stdout: str = ""
    stderr: str = ""
    return_code: int | None = None
    timed_out: bool = False
    error_category: str | None = None
    duration_ms: int = Field(ge=0)
    pre_environment_state_id: str
    post_environment_state_id: str
    full_output_sha256: str
    model_facing_output: str
    output_truncated: bool = False
```

- [ ] **Step 4: Implement config and counters**

```python
from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from threading import Lock
from typing import Self

from pydantic import BaseModel, ConfigDict, Field


class BudgetExhausted(RuntimeError):
    pass


class RunBudget:
    def __init__(self, *, max_actions: int = 24, max_model_calls: int = 40) -> None:
        self.max_actions = max_actions
        self.max_model_calls = max_model_calls
        self._actions = 0
        self._model_calls = 0
        self._lock = Lock()

    def reserve_action(self) -> int:
        with self._lock:
            if self._actions >= self.max_actions:
                raise BudgetExhausted("terminal action budget exhausted")
            self._actions += 1
            return self._actions

    def reserve_model_call(self) -> int:
        with self._lock:
            if self._model_calls >= self.max_model_calls:
                raise BudgetExhausted("model call budget exhausted")
            self._model_calls += 1
            return self._model_calls

    @property
    def actions_used(self) -> int:
        return self._actions

    @property
    def model_calls_used(self) -> int:
        return self._model_calls


class TerminalBenchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    model: str
    api_key_env: str = "BAYESPROBE_BENCH_API_KEY"
    base_url: str | None = None
    provider_timeout_seconds: int = Field(default=360, ge=1)
    command_timeout_seconds: int = Field(default=120, ge=1, le=120)
    max_output_tokens: int = Field(default=8_192, ge=256)
    max_cycles: int = Field(default=8, ge=1)
    max_probes_per_cycle: int = Field(default=2, ge=1)
    max_actions_per_probe: int = Field(default=3, ge=1, le=3)
    max_total_actions: int = Field(default=24, ge=1)
    max_model_calls: int = Field(default=40, ge=1)
    signal_output_bytes: int = Field(default=32_768, ge=1)
    lock_path: Path = Path(".runs/benchmark.lock.json")

    @classmethod
    def from_sources(
        cls,
        extra_env: Mapping[str, str] | None = None,
    ) -> tuple[Self, str]:
        source = {**os.environ, **dict(extra_env or {})}
        model = source.get("BAYESPROBE_BENCH_MODEL", "").strip()
        if not model:
            raise ValueError("BAYESPROBE_BENCH_MODEL is required")
        api_key = source.get("BAYESPROBE_BENCH_API_KEY", "").strip()
        if not api_key:
            raise ValueError("BAYESPROBE_BENCH_API_KEY is required")
        config = cls(
            model=model,
            base_url=source.get("BAYESPROBE_BENCH_BASE_URL", "").strip() or None,
            lock_path=Path(source.get("BAYESPROBE_BENCH_LOCK_PATH", ".runs/benchmark.lock.json")),
        )
        return config, api_key
```

- [ ] **Step 5: Verify GREEN**

Run: `uv run --project benchmarks/terminal_bench pytest benchmarks/terminal_bench/tests/test_actions.py benchmarks/terminal_bench/tests/test_config.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/terminal_bench/src/bayesprobe_terminal_bench/actions.py benchmarks/terminal_bench/src/bayesprobe_terminal_bench/config.py benchmarks/terminal_bench/tests/test_actions.py benchmarks/terminal_bench/tests/test_config.py
git commit -m "feat: define Terminal-Bench action contracts"
```

### Task 3: Add restricted append-only artifacts

**Files:**

- Create: `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/artifacts.py`
- Create: `benchmarks/terminal_bench/tests/test_artifacts.py`

**Interfaces:**

- `TrialArtifactStore` writes plans, observations, provider calls, errors, and summaries to separate files.
- The exact provider secret is redacted recursively before serialization.
- A model plan is never written to a Signal or Evidence stream.

- [ ] **Step 1: Write failing artifact tests**

```python
from bayesprobe_terminal_bench.artifacts import TrialArtifactStore


def test_store_redacts_exact_provider_secret(tmp_path) -> None:
    store = TrialArtifactStore(tmp_path, restricted_values=("provider-secret",))
    store.append_plan({"message": "provider-secret must not survive"})
    text = (tmp_path / "plans.jsonl").read_text()
    assert "provider-secret" not in text
    assert "[REDACTED]" in text


def test_plan_does_not_create_signal_or_evidence_stream(tmp_path) -> None:
    store = TrialArtifactStore(tmp_path, restricted_values=())
    store.append_plan({"probe_id": "P1", "actions": []})
    assert (tmp_path / "plans.jsonl").exists()
    assert not (tmp_path / "signals.jsonl").exists()
    assert not (tmp_path / "evidence.jsonl").exists()
```

- [ ] **Step 2: Verify RED**

Run: `uv run --project benchmarks/terminal_bench pytest benchmarks/terminal_bench/tests/test_artifacts.py -q`

Expected: import failure because `artifacts.py` does not exist.

- [ ] **Step 3: Implement the artifact store**

```python
from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any

from pydantic import BaseModel


class TrialArtifactStore:
    def __init__(self, root: Path, *, restricted_values: tuple[str, ...]) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._restricted_values = tuple(value for value in restricted_values if value)
        self._lock = Lock()

    def append_plan(self, payload) -> None:
        self._append("plans.jsonl", payload)

    def append_observation(self, payload) -> None:
        self._append("environment_actions.jsonl", payload)

    def append_provider_call(self, payload) -> None:
        self._append("provider_telemetry.jsonl", payload)

    def append_error(self, payload) -> None:
        self._append("errors.jsonl", payload)

    def write_summary(self, payload: dict[str, Any]) -> None:
        safe = self._redact(payload)
        (self.root / "summary.json").write_text(
            json.dumps(safe, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    def _append(self, name: str, payload) -> None:
        raw = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
        line = json.dumps(self._redact(raw), ensure_ascii=False, sort_keys=True) + "\n"
        with self._lock:
            with (self.root / name).open("a", encoding="utf-8") as handle:
                handle.write(line)

    def _redact(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self._redact(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._redact(item) for item in value]
        if isinstance(value, str):
            for restricted in self._restricted_values:
                value = value.replace(restricted, "[REDACTED]")
        return value
```

- [ ] **Step 4: Verify GREEN**

Run: `uv run --project benchmarks/terminal_bench pytest benchmarks/terminal_bench/tests/test_artifacts.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/terminal_bench/src/bayesprobe_terminal_bench/artifacts.py benchmarks/terminal_bench/tests/test_artifacts.py
git commit -m "feat: record restricted Terminal-Bench artifacts"
```

### Task 4: Bridge synchronous Probe execution to Harbor's async environment

**Files:**

- Create: `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/environment.py`
- Create: `benchmarks/terminal_bench/tests/test_environment.py`

**Interfaces:**

- `HarborEnvironmentLike` is the narrow structural interface shared by fake and real environments.
- `ActionPolicy.validate(action)` rejects protected evaluator and Docker-control paths before execution.
- `HarborEnvironmentBridge.execute(action, action_index) -> ActionObservation` is synchronous and is called from the BayesProbe worker thread.
- Every attempted mutating action advances `environment_state_id`, including timeout and non-zero return, because partial mutation cannot be excluded.

- [ ] **Step 1: Write failing bridge, lineage, and policy tests**

```python
from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from bayesprobe_terminal_bench.actions import ShellAction, WriteFileAction
from bayesprobe_terminal_bench.environment import ActionPolicy, HarborEnvironmentBridge, PolicyViolation


@dataclass
class FakeExecResult:
    stdout: str | None
    stderr: str | None
    return_code: int


class FakeEnvironment:
    def __init__(self) -> None:
        self.commands: list[str] = []
        self.uploads: list[tuple[str, str]] = []

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        self.commands.append(command)
        return FakeExecResult(stdout="observed", stderr="", return_code=0)

    async def upload_file(self, source_path, target_path) -> None:
        self.uploads.append((str(source_path), target_path))


@pytest.mark.asyncio
async def test_bridge_calls_harbor_loop_from_worker_thread() -> None:
    bridge = HarborEnvironmentBridge(
        loop=asyncio.get_running_loop(),
        environment=FakeEnvironment(),
        policy=ActionPolicy(),
        output_limit_bytes=32_768,
    )
    observation = await asyncio.to_thread(bridge.execute, ShellAction(command="pwd"), 1)
    assert observation.stdout == "observed"
    assert observation.pre_environment_state_id == "env:0"
    assert observation.post_environment_state_id == "env:0"


@pytest.mark.asyncio
async def test_write_advances_environment_lineage() -> None:
    bridge = HarborEnvironmentBridge(
        loop=asyncio.get_running_loop(),
        environment=FakeEnvironment(),
        policy=ActionPolicy(),
        output_limit_bytes=32_768,
    )
    observation = await asyncio.to_thread(
        bridge.execute,
        WriteFileAction(path="/app/value.txt", content="value"),
        1,
    )
    assert observation.post_environment_state_id == "env:1"


def test_policy_blocks_verifier_and_docker_paths() -> None:
    policy = ActionPolicy()
    with pytest.raises(PolicyViolation):
        policy.validate(ShellAction(command="cat /logs/verifier/reward.txt"))
    with pytest.raises(PolicyViolation):
        policy.validate(WriteFileAction(path="/var/run/docker.sock", content="x"))
    with pytest.raises(PolicyViolation):
        policy.validate(WriteFileAction(path="/tests/hidden.py", content="x"))
```

- [ ] **Step 2: Verify RED**

Run: `uv run --project benchmarks/terminal_bench pytest benchmarks/terminal_bench/tests/test_environment.py -q`

Expected: import failure because `environment.py` does not exist.

- [ ] **Step 3: Implement the narrow Harbor seam, policy, and lineage**

```python
from __future__ import annotations

import asyncio
import hashlib
import json
import shlex
import tempfile
import time
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Protocol
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
    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None) -> ExecResultLike:
        raise NotImplementedError

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        raise NotImplementedError


@dataclass(frozen=True)
class LocalExecResult:
    stdout: str | None
    stderr: str | None
    return_code: int


class PolicyViolation(ValueError):
    pass


class ActionPolicy:
    _PROTECTED = (
        "/logs/verifier",
        "/solution",
        "/tests",
        "/var/run/docker.sock",
        "docker.sock",
    )

    def validate(self, action: TerminalAction) -> None:
        text = action.command if isinstance(action, ShellAction) else (
            action.path if isinstance(action, WriteFileAction) else action.patch
        )
        if any(item in text for item in self._PROTECTED):
            raise PolicyViolation("terminal action targets a protected path")


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
```

Implement the execution method exactly around Harbor's event loop:

```python
class HarborEnvironmentBridge:
    def __init__(self, *, loop, environment, policy, output_limit_bytes) -> None:
        self._loop = loop
        self._environment = environment
        self._policy = policy
        self._output_limit_bytes = output_limit_bytes
        self._state = EnvironmentState()

    def execute(self, action: TerminalAction, action_index: int) -> ActionObservation:
        self._policy.validate(action)
        before = self._state.current()
        timeout = action.timeout_seconds if isinstance(action, ShellAction) else 120
        started = time.monotonic()
        future = asyncio.run_coroutine_threadsafe(self._execute_async(action), self._loop)
        stdout, stderr, return_code = "", "", None
        timed_out, error_category = False, None
        try:
            result = future.result(timeout=timeout + 5)
            stdout, stderr = result.stdout or "", result.stderr or ""
            return_code = result.return_code
        except FutureTimeoutError:
            future.cancel()
            timed_out, error_category = True, "timeout"
            stderr = "Harbor action exceeded the configured timeout."
        except Exception as error:
            error_category = "transport"
            stderr = f"{type(error).__name__}: Harbor action failed"
        after = self._state.advance() if action_may_mutate(action) else before
        full = json.dumps(
            {"stdout": stdout, "stderr": stderr, "return_code": return_code, "timed_out": timed_out},
            ensure_ascii=False,
            sort_keys=True,
        )
        encoded = full.encode("utf-8")
        visible = encoded[: self._output_limit_bytes].decode("utf-8", errors="replace")
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
            full_output_sha256=hashlib.sha256(encoded).hexdigest(),
            model_facing_output=visible,
            output_truncated=len(encoded) > self._output_limit_bytes,
        )

    async def _execute_async(self, action: TerminalAction) -> ExecResultLike:
        if isinstance(action, ShellAction):
            return await self._environment.exec(action.command, timeout_sec=action.timeout_seconds)
        if isinstance(action, WriteFileAction):
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
                handle.write(action.content)
                local_path = Path(handle.name)
            try:
                await self._environment.upload_file(local_path, action.path)
                return LocalExecResult(stdout=f"wrote {action.path}", stderr="", return_code=0)
            finally:
                local_path.unlink(missing_ok=True)
        remote_patch = f"/tmp/.bayesprobe-{uuid4().hex}.patch"
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
            handle.write(action.patch)
            local_patch = Path(handle.name)
        try:
            await self._environment.upload_file(local_patch, remote_patch)
            command = f"patch --batch --forward -p{action.strip} < {shlex.quote(remote_patch)}"
            return await self._environment.exec(command, timeout_sec=120)
        finally:
            local_patch.unlink(missing_ok=True)
            try:
                await self._environment.exec(
                    f"rm -f {shlex.quote(remote_patch)}",
                    timeout_sec=30,
                )
            except Exception:
                # Cleanup failure must not hide the actual patch result or error.
                pass
```

- [ ] **Step 4: Verify GREEN and no deadlock**

Run: `uv run --project benchmarks/terminal_bench pytest benchmarks/terminal_bench/tests/test_environment.py -q`

Expected: PASS without hanging.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/terminal_bench/src/bayesprobe_terminal_bench/environment.py benchmarks/terminal_bench/tests/test_environment.py
git commit -m "feat: bridge BayesProbe probes to Harbor"
```

### Task 5: Add the benchmark-local terminal action planner

**Files:**

- Create: `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/planning.py`
- Create: `benchmarks/terminal_bench/tests/conftest.py`
- Create: `benchmarks/terminal_bench/tests/test_planning.py`

**Interfaces:**

- `TerminalProbePlanner.plan(probe, context, history) -> TerminalProbePlan` is the narrow seam consumed by the gateway.
- `OpenAICompatibleTerminalProbePlanner` uses the same model configuration and shared `RunBudget` as the core gateway.
- Exactly one schema-repair request is allowed.
- Validation or provider failure raises `TerminalPlanError`; callers create no Signal.
- Every physical provider attempt emits secret-free terminal-planner telemetry; each initial or repair request consumes one logical model-call budget unit.

- [ ] **Step 1: Write failing parsing, repair, and budget tests**

```python
from types import SimpleNamespace

import pytest

from bayesprobe_terminal_bench.config import RunBudget, TerminalBenchConfig
from bayesprobe_terminal_bench.planning import OpenAICompatibleTerminalProbePlanner, TerminalPlanError


class FakeCompletions:
    def __init__(self, contents) -> None:
        self.contents = list(contents)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.contents.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class FakeClient:
    def __init__(self, contents) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions(contents))


def test_planner_repairs_invalid_json_once(probe, execution_context) -> None:
    telemetry = []
    client = FakeClient([
        "not-json",
        '{"mode":"inspect","actions":[{"type":"shell","command":"pwd","timeout_seconds":30,"mutates_environment":false}],"expected_observation":"working directory"}',
    ])
    planner = OpenAICompatibleTerminalProbePlanner(
        config=TerminalBenchConfig(model="test-model"),
        budget=RunBudget(max_actions=24, max_model_calls=2),
        client=client,
        invocation_observer=telemetry.append,
    )
    assert planner.plan(probe=probe, context=execution_context, history=()).actions[0].command == "pwd"
    assert len(client.chat.completions.calls) == 2
    assert [item["outcome"] for item in telemetry] == ["success", "success"]
    assert [item["repair"] for item in telemetry] == [False, True]


def test_planner_never_falls_back_to_imagined_action(probe, execution_context) -> None:
    planner = OpenAICompatibleTerminalProbePlanner(
        config=TerminalBenchConfig(model="test-model"),
        budget=RunBudget(max_actions=24, max_model_calls=2),
        client=FakeClient(["bad", "still bad"]),
    )
    with pytest.raises(TerminalPlanError, match="terminal plan validation failed"):
        planner.plan(probe=probe, context=execution_context, history=())
```

- [ ] **Step 2: Verify RED**

Run: `uv run --project benchmarks/terminal_bench pytest benchmarks/terminal_bench/tests/test_planning.py -q`

Expected: import failure because `planning.py` does not exist.

- [ ] **Step 3: Implement the planner Protocol and OpenAI-compatible Adapter**

```python
from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any, Protocol

from openai import OpenAI

from bayesprobe import ProbeDesign, ProbeExecutionBrief
from bayesprobe_terminal_bench.actions import ActionObservation, TerminalProbePlan
from bayesprobe_terminal_bench.config import RunBudget, TerminalBenchConfig


class TerminalPlanError(ValueError):
    pass


class TerminalProbePlanner(Protocol):
    def plan(
        self,
        *,
        probe: ProbeDesign,
        context: ProbeExecutionBrief,
        history: tuple[ActionObservation, ...],
    ) -> TerminalProbePlan:
        raise NotImplementedError


def terminal_plan_input(
    *,
    probe: ProbeDesign,
    context: ProbeExecutionBrief,
    history: tuple[ActionObservation, ...],
) -> dict[str, Any]:
    return {
        "task": {
            "problem": context.problem,
            "task_context": context.task_context,
            "task_frame": dict(context.task_frame),
        },
        "hypotheses": [
            {
                "id": item.id,
                "statement": item.statement,
                "scope": item.scope,
                "predictions": list(item.predictions),
                "falsifiers": list(item.falsifiers),
            }
            for item in context.hypotheses
        ],
        "probe": {
            "id": probe.id,
            "inquiry_goal": probe.inquiry_goal,
            "method": probe.method,
            "expected_observation": probe.expected_observation,
            "target_hypotheses": list(probe.target_hypotheses),
            "support_condition": dict(probe.support_condition),
            "weaken_condition": dict(probe.weaken_condition),
            "reframe_condition": probe.reframe_condition,
        },
        "recent_observations": [
            {
                "action": item.action.model_dump(mode="json"),
                "observation": item.model_facing_output,
                "return_code": item.return_code,
                "timed_out": item.timed_out,
                "environment_state_id": item.post_environment_state_id,
            }
            for item in history[-12:]
        ],
    }


class OpenAICompatibleTerminalProbePlanner:
    def __init__(
        self,
        *,
        config: TerminalBenchConfig,
        budget: RunBudget,
        api_key: str | None = None,
        client: Any | None = None,
        invocation_observer: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._config = config
        self._budget = budget
        self._invocation_observer = invocation_observer
        if client is None:
            if not api_key:
                raise ValueError("terminal planner requires an explicit API key")
            client = OpenAI(
                api_key=api_key,
                base_url=config.base_url,
                timeout=config.provider_timeout_seconds,
                max_retries=0,
            )
        self._client = client

    def plan(
        self,
        *,
        probe: ProbeDesign,
        context: ProbeExecutionBrief,
        history: tuple[ActionObservation, ...],
    ) -> TerminalProbePlan:
        payload = terminal_plan_input(probe=probe, context=context, history=history)
        first = self._complete(payload=payload, repair=False)
        try:
            return TerminalProbePlan.model_validate_json(first)
        except Exception:
            repaired = self._complete(
                payload={"original_input": payload, "validation_error": "invalid terminal plan"},
                repair=True,
            )
        try:
            return TerminalProbePlan.model_validate_json(repaired)
        except Exception as error:
            raise TerminalPlanError("terminal plan validation failed") from error

    def _complete(self, *, payload: dict[str, Any], repair: bool) -> str:
        logical_call_index = self._budget.reserve_model_call()
        instruction = (
            "Repair one terminal action plan and return JSON only."
            if repair
            else "Plan one bounded terminal Probe. Return JSON only. Do not claim any command ran."
        )
        started = time.monotonic()
        try:
            response = self._client.chat.completions.create(
                model=self._config.model,
                messages=[
                    {
                        "role": "system",
                        "content": instruction + " Schema: " + json.dumps(TerminalProbePlan.model_json_schema(), sort_keys=True),
                    },
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=self._config.max_output_tokens,
            )
        except Exception as error:
            self._observe({
                "task": "terminal_probe_plan",
                "model": self._config.model,
                "repair": repair,
                "logical_call_index": logical_call_index,
                "latency_seconds": time.monotonic() - started,
                "outcome": "error",
                "error_type": type(error).__name__,
            })
            raise TerminalPlanError("terminal planner provider request failed") from error
        self._observe({
            "task": "terminal_probe_plan",
            "model": self._config.model,
            "repair": repair,
            "logical_call_index": logical_call_index,
            "latency_seconds": time.monotonic() - started,
            "outcome": "success",
            "response_id": getattr(response, "id", None),
            "finish_reason": getattr(response.choices[0], "finish_reason", None),
            "usage": {
                "input_tokens": getattr(getattr(response, "usage", None), "prompt_tokens", None),
                "output_tokens": getattr(getattr(response, "usage", None), "completion_tokens", None),
                "total_tokens": getattr(getattr(response, "usage", None), "total_tokens", None),
            },
        })
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise TerminalPlanError("terminal planner returned no content")
        return content

    def _observe(self, payload: dict[str, Any]) -> None:
        if self._invocation_observer is not None:
            self._invocation_observer(payload)
```

The input function above deliberately excludes priors, posteriors, credentials,
verifier paths, and reasoning fields. Add a focused test that serializes its
result and proves those fields are absent.

- [ ] **Step 4: Verify GREEN**

Run: `uv run --project benchmarks/terminal_bench pytest benchmarks/terminal_bench/tests/test_planning.py -q`

Expected: PASS and exactly two calls in the repair test.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/terminal_bench/src/bayesprobe_terminal_bench/planning.py benchmarks/terminal_bench/tests/conftest.py benchmarks/terminal_bench/tests/test_planning.py
git commit -m "feat: plan bounded terminal probes"
```

### Task 6: Convert Harbor observations into BayesProbe Signals

**Files:**

- Create: `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/signals.py`
- Create: `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/gateway.py`
- Create: `benchmarks/terminal_bench/tests/test_gateway.py`

**Interfaces:**

- `signal_from_observation` always returns a public `ExternalSignal` with `TOOL_RESULT` provenance.
- `HarborProbeToolGateway.execute_probe` structurally satisfies the public `ProbeToolGateway` interface.
- Plans are recorded separately; one Signal is returned per observed action.
- Plan, policy, and exhausted-budget failures return no Signal.

- [ ] **Step 1: Write failing provenance and no-fabrication tests**

```python
from bayesprobe import EpistemicOrigin
from bayesprobe_terminal_bench.actions import ShellAction, TerminalProbePlan
from bayesprobe_terminal_bench.gateway import HarborProbeToolGateway


class ScriptedPlanner:
    def __init__(self, plan=None, error=None) -> None:
        self.plan, self.error = plan, error

    def plan(self, *, probe, context, history):
        if self.error is not None:
            raise self.error
        return self.plan


def test_gateway_emits_only_tool_result_signals(probe, execution_context, bridge, artifact_store, budget) -> None:
    planner = ScriptedPlanner(TerminalProbePlan(
        mode="inspect",
        actions=[ShellAction(command="pwd")],
        expected_observation="working directory",
    ))
    gateway = HarborProbeToolGateway(planner=planner, bridge=bridge, artifacts=artifact_store, budget=budget)
    signals = gateway.execute_probe(probe=probe, context=execution_context)
    assert len(signals) == 1
    assert signals[0].provenance.epistemic_origin is EpistemicOrigin.TOOL_RESULT


def test_planner_failure_creates_no_signal(probe, execution_context, bridge, artifact_store, budget) -> None:
    gateway = HarborProbeToolGateway(
        planner=ScriptedPlanner(error=ValueError("invalid plan")),
        bridge=bridge,
        artifacts=artifact_store,
        budget=budget,
    )
    assert gateway.execute_probe(probe=probe, context=execution_context) == []
```

- [ ] **Step 2: Verify RED**

Run: `uv run --project benchmarks/terminal_bench pytest benchmarks/terminal_bench/tests/test_gateway.py -q`

Expected: import failures because Signal and gateway modules do not exist.

- [ ] **Step 3: Implement canonical Signal conversion**

```python
from __future__ import annotations

import json

from bayesprobe import EpistemicOrigin, ExternalSignal, SignalKind, SignalProvenance


def signal_from_observation(*, observation, probe, context) -> ExternalSignal:
    return ExternalSignal(
        id=f"S_{context.cycle_id}_{probe.id}_{observation.action_index}",
        cycle_id=context.cycle_id,
        signal_kind=SignalKind.ACTIVE,
        source_type="harbor_terminal",
        source="harbor:environment",
        raw_content=json.dumps(
            {
                "action": observation.action.model_dump(mode="json"),
                "observation": observation.model_facing_output,
                "return_code": observation.return_code,
                "timed_out": observation.timed_out,
                "error_category": observation.error_category,
                "output_truncated": observation.output_truncated,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        generated_by_probe=probe.id,
        initial_target_hypotheses=list(probe.target_hypotheses),
        provenance=SignalProvenance(
            epistemic_origin=EpistemicOrigin.TOOL_RESULT,
            source_identity="harbor_terminal:v1",
            provider_model_or_tool_identity="harbor:0.18.0",
            derivation_root_id=f"harbor-action:sha256:{observation.full_output_sha256}",
            correlation_group=f"harbor-env:{context.run_id}:{observation.post_environment_state_id}",
            canonical_content_fingerprint="pending-normalization",
            artifact_refs=[f"environment_actions.jsonl#{observation.action_index}"],
            environment_state_id=observation.post_environment_state_id,
        ),
    )
```

- [ ] **Step 4: Implement the Probe gateway Adapter**

```python
from bayesprobe_terminal_bench.config import BudgetExhausted
from bayesprobe_terminal_bench.environment import PolicyViolation
from bayesprobe_terminal_bench.signals import signal_from_observation


class HarborProbeToolGateway:
    def __init__(self, *, planner, bridge, artifacts, budget) -> None:
        self._planner, self._bridge = planner, bridge
        self._artifacts, self._budget = artifacts, budget
        self._history = []

    def execute_probe(self, *, probe, context):
        try:
            plan = self._planner.plan(probe=probe, context=context, history=tuple(self._history[-12:]))
        except Exception as error:
            self._artifacts.append_error(
                {"category": "plan_error", "probe_id": probe.id, "error_type": type(error).__name__}
            )
            return []
        self._artifacts.append_plan(
            {"probe_id": probe.id, "cycle_id": context.cycle_id, "plan": plan.model_dump(mode="json")}
        )
        signals = []
        for action in plan.actions:
            try:
                action_index = self._budget.reserve_action()
                observation = self._bridge.execute(action, action_index)
            except BudgetExhausted:
                self._artifacts.append_error({"category": "budget_exhausted", "probe_id": probe.id})
                break
            except PolicyViolation as error:
                self._artifacts.append_error(
                    {"category": "policy_error", "probe_id": probe.id, "error_type": type(error).__name__}
                )
                continue
            self._history.append(observation)
            self._artifacts.append_observation(observation)
            signals.append(signal_from_observation(observation=observation, probe=probe, context=context))
        return signals
```

- [ ] **Step 5: Verify GREEN and public imports**

Run: `uv run --project benchmarks/terminal_bench pytest benchmarks/terminal_bench/tests/test_gateway.py benchmarks/terminal_bench/tests/test_public_reuse.py -q`

Expected: PASS; plan failures produce no Signal.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/terminal_bench/src/bayesprobe_terminal_bench/signals.py benchmarks/terminal_bench/src/bayesprobe_terminal_bench/gateway.py benchmarks/terminal_bench/tests/test_gateway.py
git commit -m "feat: emit Harbor observations as BayesProbe signals"
```

### Task 7: Construct and exercise the real public BayesProbe runner

**Files:**

- Create: `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/runner_factory.py`
- Create: `benchmarks/terminal_bench/tests/test_runner_factory.py`
- Create: `benchmarks/terminal_bench/tests/test_conformance.py`

**Interfaces:**

- `build_runner(model_gateway, probe_gateway, ledger_path, config, progress_observer=None) -> AutonomousQuestionRunner` returns the installed public class.
- `build_live_session(...) -> LiveSession` composes the public provider gateway, terminal Adapter, real runner, and `InitializeRunInput` without executing the run.
- `BudgetedModelGateway` delegates every root model request after reserving one call from the same `RunBudget` used by terminal planning.

- [ ] **Step 1: Write failing public-type and real-cycle tests**

```python
from bayesprobe import (
    AutonomousQuestionRunner,
    BayesProbeCore,
    DeterministicModelGateway,
    EpistemicOrigin,
    HypothesisRelation,
    HypothesisSeed,
    InitializeRunInput,
    TaskKind,
)

from bayesprobe_terminal_bench.config import TerminalBenchConfig
from bayesprobe_terminal_bench.runner_factory import build_runner


def test_factory_returns_real_public_runner(tmp_path, tool_signal_gateway) -> None:
    runner = build_runner(
        model_gateway=DeterministicModelGateway(),
        probe_gateway=tool_signal_gateway,
        ledger_path=tmp_path / "ledger.jsonl",
        config=TerminalBenchConfig(model="deterministic", max_cycles=1),
    )
    assert type(runner) is AutonomousQuestionRunner
    assert type(runner.core) is BayesProbeCore


def test_real_runner_integrates_terminal_tool_signal(tmp_path, tool_signal_gateway) -> None:
    runner = build_runner(
        model_gateway=DeterministicModelGateway(),
        probe_gateway=tool_signal_gateway,
        ledger_path=tmp_path / "ledger.jsonl",
        config=TerminalBenchConfig(model="deterministic", max_cycles=1),
    )
    result = runner.run_question(InitializeRunInput(
        run_id="terminal_conformance",
        problem="Determine which implementation diagnosis matches the observed test.",
        task_kind=TaskKind.DECISION,
        hypothesis_relation=HypothesisRelation.EXCLUSIVE_EXHAUSTIVE,
        hypothesis_seeds=[
            HypothesisSeed(
                id="H1",
                statement="The file is missing.",
                prior=0.5,
                predictions=["ls reports missing"],
                falsifiers=["ls reports present"],
            ),
            HypothesisSeed(
                id="H2",
                statement="The file exists but is invalid.",
                prior=0.5,
                predictions=["parser rejects content"],
                falsifiers=["parser accepts content"],
            ),
        ],
    ))
    assert len(result.cycle_results) == 1
    event = result.cycle_results[0].evidence_events[0]
    assert event.epistemic_origin is EpistemicOrigin.TOOL_RESULT
    assert event.derived_from_signal == result.cycle_results[0].signals[0].id
```

- [ ] **Step 2: Verify RED**

Run: `uv run --project benchmarks/terminal_bench pytest benchmarks/terminal_bench/tests/test_runner_factory.py benchmarks/terminal_bench/tests/test_conformance.py -q`

Expected: import failure because `runner_factory.py` does not exist.

- [ ] **Step 3: Implement the runner factory with top-level imports only**

```python
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from bayesprobe import (
    AutonomousQuestionRunConfig,
    AutonomousQuestionRunner,
    BayesProbeCore,
    BayesProbeInitializer,
    CapabilityDescriptor,
    CapabilityKind,
    EpistemicOrigin,
    ExplicitTaskAdmitter,
    ExplicitTaskFramer,
    HypothesisExpansionService,
    InitializeRunInput,
    JsonlLedgerStore,
    ModelHypothesisExpansionAdapter,
    ModelProbeDesigner,
    ModelTaskAdmitter,
    ModelTaskFramer,
    ProbeExecutor,
    RoutingTaskAdmitter,
    RoutingTaskFramer,
    TaskAwareAnswerProjector,
)

from bayesprobe_terminal_bench.artifacts import TrialArtifactStore
from bayesprobe_terminal_bench.config import RunBudget, TerminalBenchConfig
from bayesprobe_terminal_bench.environment import ActionPolicy, HarborEnvironmentBridge
from bayesprobe_terminal_bench.gateway import HarborProbeToolGateway
from bayesprobe_terminal_bench.planning import OpenAICompatibleTerminalProbePlanner


def build_runner(*, model_gateway, probe_gateway, ledger_path, config, progress_observer=None):
    ledger = JsonlLedgerStore(ledger_path)
    task_admitter = RoutingTaskAdmitter(
        explicit_admitter=ExplicitTaskAdmitter(),
        open_admitter=ModelTaskAdmitter(model_gateway),
    )
    task_framer = RoutingTaskFramer(
        explicit_framer=ExplicitTaskFramer(),
        open_framer=ModelTaskFramer(model_gateway),
    )
    core = BayesProbeCore(
        ledger=ledger,
        model_gateway=model_gateway,
        hypothesis_expander=HypothesisExpansionService(
            adapter=ModelHypothesisExpansionAdapter(model_gateway)
        ),
    )
    capabilities = (
        CapabilityDescriptor(
            kind=CapabilityKind.REPOSITORY_READ,
            available=True,
            epistemic_origin=EpistemicOrigin.TOOL_RESULT,
            executor_adapter_id="harbor_terminal:v1",
            quality_caps={"verifiability": 0.95, "independence": 0.8},
        ),
        CapabilityDescriptor(
            kind=CapabilityKind.TEST_EXECUTION,
            available=True,
            epistemic_origin=EpistemicOrigin.TOOL_RESULT,
            executor_adapter_id="harbor_terminal:v1",
            quality_caps={"verifiability": 1.0, "independence": 0.9},
        ),
    )
    return AutonomousQuestionRunner(
        core=core,
        initializer=BayesProbeInitializer(
            ledger=ledger,
            task_framer=task_framer,
            task_admitter=task_admitter,
        ),
        executor=ProbeExecutor(gateway=probe_gateway, ledger=ledger),
        config=AutonomousQuestionRunConfig(
            max_cycles=config.max_cycles,
            max_probes_per_cycle=config.max_probes_per_cycle,
            stop_on_no_probes=True,
            confidence_threshold=None,
            posterior_delta_threshold=None,
        ),
        progress_observer=progress_observer,
        task_admitter=task_admitter,
        probe_designer=ModelProbeDesigner(model_gateway),
        available_capabilities=capabilities,
        answer_projector=TaskAwareAnswerProjector(model_gateway),
    )
```

The public `CapabilityKind` has no repository-write member. Treat writes and
patches as interventions inside a repository-read or test-execution Probe; the
benchmark action policy owns side-effect permission. Do not add a core enum.

- [ ] **Step 4: Implement the shared model-call wrapper and live composition**

```python
class BudgetedModelGateway:
    def __init__(self, delegate, budget) -> None:
        self._delegate, self._budget = delegate, budget

    @property
    def adapter_kind(self):
        return self._delegate.adapter_kind

    @property
    def model_identity(self):
        return self._delegate.model_identity

    def complete_structured(self, request):
        self._budget.reserve_model_call()
        return self._delegate.complete_structured(request)


@dataclass(frozen=True)
class LiveSession:
    runner: AutonomousQuestionRunner
    input: InitializeRunInput
    artifacts: TrialArtifactStore
    budget: RunBudget


class ArtifactInvocationObserver:
    def __init__(self, artifacts: TrialArtifactStore) -> None:
        self._artifacts = artifacts

    def observe(self, record) -> None:
        self._artifacts.append_provider_call(record.to_dict())
```

Add `load_and_validate_lock(path, config) -> dict[str, object]`. It must reject a
missing lock, a non-object payload, a mismatched Harbor/dataset/task identity,
or any model, provider, budget, timeout, prompt, or schema setting that differs
from the locked value.

Use this exact public-provider composition inside `build_live_session`:

```python
from bayesprobe import (
    InitializeRunInput,
    OpenAIChatCompletionsModelGateway,
    OpenAIModelGatewayConfig,
    ProviderRequestControls,
)


def safe_run_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.-") or "harbor"
    return f"tb_{normalized}"[:96]


def load_and_validate_lock(
    path: Path,
    config: TerminalBenchConfig,
) -> dict[str, object]:
    if not path.is_file():
        raise ValueError("Terminal-Bench lock is required")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Terminal-Bench lock must be an object")
    expected = {
        "schema_version": "terminal_bench_lock:v0.1",
        "harbor_version": "0.18.0",
        "dataset_name": "terminal-bench/terminal-bench-2",
        "task_id": "terminal-bench/break-filter-js-from-html",
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
    mismatches = {
        key: {"expected": expected_value, "actual": payload.get(key)}
        for key, expected_value in expected.items()
        if payload.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(f"Terminal-Bench lock mismatch: {sorted(mismatches)}")
    return payload


def build_live_session(
    *,
    config: TerminalBenchConfig,
    api_key: str,
    instruction: str,
    environment,
    event_loop,
    logs_dir,
    session_id: str | None,
    context_id: str | None,
) -> LiveSession:
    load_and_validate_lock(config.lock_path, config)
    artifacts = TrialArtifactStore(
        Path(logs_dir) / "bayesprobe",
        restricted_values=(api_key,),
    )
    budget = RunBudget(
        max_actions=config.max_total_actions,
        max_model_calls=config.max_model_calls,
    )
    provider = OpenAIChatCompletionsModelGateway(
        config=OpenAIModelGatewayConfig(
            model=config.model,
            api_key_env=config.api_key_env,
            timeout_seconds=config.provider_timeout_seconds,
            max_output_tokens=config.max_output_tokens,
            base_url=config.base_url,
            request_controls=ProviderRequestControls(temperature=0),
        ),
        api_key=api_key,
        invocation_observer=ArtifactInvocationObserver(artifacts),
    )
    model_gateway = BudgetedModelGateway(provider, budget)
    bridge = HarborEnvironmentBridge(
        loop=event_loop,
        environment=environment,
        policy=ActionPolicy(),
        output_limit_bytes=config.signal_output_bytes,
    )
    planner = OpenAICompatibleTerminalProbePlanner(
        config=config,
        budget=budget,
        api_key=api_key,
        invocation_observer=artifacts.append_provider_call,
    )
    probe_gateway = HarborProbeToolGateway(
        planner=planner,
        bridge=bridge,
        artifacts=artifacts,
        budget=budget,
    )
    run_id = safe_run_id(context_id or session_id or "harbor")
    runner = build_runner(
        model_gateway=model_gateway,
        probe_gateway=probe_gateway,
        ledger_path=artifacts.root / "bayesprobe_ledger.jsonl",
        config=config,
    )
    input = InitializeRunInput(
        run_id=run_id,
        problem=instruction,
        task_context=(
            "Work only in the provided Terminal-Bench task environment. "
            "Use observable repository and test results to diagnose, modify, "
            "and verify the task; Harbor runs the official verifier afterward."
        ),
        metadata={
            "experiment_id": "terminal_bench_engineering_v0.1",
            "arm": "bayesprobe",
            "sample_id": context_id or session_id or run_id,
        },
    )
    return LiveSession(runner=runner, input=input, artifacts=artifacts, budget=budget)
```

`safe_run_id` must map arbitrary Harbor IDs to `[A-Za-z0-9_.-]`, prefix the
result with `tb_`, and cap it at 96 characters. `build_live_session` must perform
these operations in order:

1. require and parse `config.lock_path`;
2. construct `TrialArtifactStore` under `logs_dir / "bayesprobe"`;
3. construct the public `OpenAIChatCompletionsModelGateway` with public
   `OpenAIModelGatewayConfig`, the explicitly supplied API key, and an invocation
   observer that writes secret-free telemetry;
4. wrap that gateway in `BudgetedModelGateway`;
5. construct `HarborEnvironmentBridge`, `OpenAICompatibleTerminalProbePlanner`,
   and `HarborProbeToolGateway` with one shared `RunBudget`;
6. call `build_runner` and create `InitializeRunInput` using the Harbor
   instruction as `problem` and a fixed terminal-workspace contract as
   `task_context`;
7. return `LiveSession` without calling `run_question`.

The `tool_signal_gateway` fixture used above must be the real
`HarborProbeToolGateway` with a scripted `TerminalProbePlan`, a bridge that
returns an `ActionObservation` whose output begins with `SUPPORTS:`, a real
`TrialArtifactStore`, and a real `RunBudget`. It must not bypass the Adapter by
constructing `ExternalSignal` directly.

- [ ] **Step 5: Add no-signal conformance coverage**

Run the real runner for one cycle with a gateway returning `[]` and assert:

```python
assert [item.posterior for item in result.final_belief_state.hypotheses] == [0.5, 0.5]
assert result.cycle_results[0].evidence_events == []
assert result.cycle_results[0].belief_updates == []
```

- [ ] **Step 6: Verify GREEN and import guard**

Run: `uv run --project benchmarks/terminal_bench pytest benchmarks/terminal_bench/tests/test_runner_factory.py benchmarks/terminal_bench/tests/test_conformance.py benchmarks/terminal_bench/tests/test_public_reuse.py -q`

Expected: PASS; source uses no private BayesProbe imports.

- [ ] **Step 7: Commit**

```bash
git add benchmarks/terminal_bench/src/bayesprobe_terminal_bench/runner_factory.py benchmarks/terminal_bench/tests/test_runner_factory.py benchmarks/terminal_bench/tests/test_conformance.py benchmarks/terminal_bench/tests/conftest.py
git commit -m "feat: run Harbor signals through public BayesProbe"
```

### Task 8: Implement the Harbor BaseAgent entry point

**Files:**

- Create: `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/agent.py`
- Create: `benchmarks/terminal_bench/tests/test_agent.py`

**Interfaces:**

- `BayesProbeHarborAgent` implements Harbor 0.18.0 `BaseAgent`.
- `setup` performs no in-container installation.
- `run` resolves provider settings from `BaseAgent.extra_env`, captures the Harbor loop, and calls the real runner exactly once through `asyncio.to_thread`.
- `AgentContext.metadata` contains only bounded counts, run identity, and stop reason.

- [ ] **Step 1: Write the failing one-call and worker-thread test**

```python
import threading
from types import SimpleNamespace

import pytest
from harbor.models.agent.context import AgentContext

from bayesprobe_terminal_bench.agent import BayesProbeHarborAgent


class SpyRunner:
    def __init__(self) -> None:
        self.calls, self.thread_id = 0, None

    def run_question(self, input):
        self.calls += 1
        self.thread_id = threading.get_ident()
        return SimpleNamespace(
            run=SimpleNamespace(run_id=input.run_id),
            stop_reason=SimpleNamespace(value="max_cycles"),
            cycle_results=[],
        )


@pytest.mark.asyncio
async def test_agent_calls_runner_once_in_worker_thread(tmp_path, monkeypatch) -> None:
    runner = SpyRunner()
    session = SimpleNamespace(
        runner=runner,
        input=SimpleNamespace(run_id="harbor-run"),
        artifacts=SimpleNamespace(write_summary=lambda payload: None),
        budget=SimpleNamespace(actions_used=0, model_calls_used=0),
    )
    monkeypatch.setattr(
        "bayesprobe_terminal_bench.agent.build_live_session",
        lambda **kwargs: session,
    )
    agent = BayesProbeHarborAgent(
        logs_dir=tmp_path,
        model_name="deepseek-v4-flash",
        extra_env={
            "BAYESPROBE_BENCH_API_KEY": "test-secret",
            "BAYESPROBE_BENCH_MODEL": "deepseek-v4-flash",
        },
    )
    context = AgentContext()
    context.metadata = {"harbor_owned": "preserved"}
    event_loop_thread = threading.get_ident()
    await agent.run("solve the task", object(), context)
    assert runner.calls == 1
    assert runner.thread_id != event_loop_thread
    assert context.metadata["bayesprobe_stop_reason"] == "max_cycles"
    assert context.metadata["harbor_owned"] == "preserved"
```

- [ ] **Step 2: Verify RED**

Run: `uv run --project benchmarks/terminal_bench pytest benchmarks/terminal_bench/tests/test_agent.py -q`

Expected: import failure because `agent.py` does not exist.

- [ ] **Step 3: Implement the Harbor agent without a local cycle**

```python
from __future__ import annotations

import asyncio

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from bayesprobe_terminal_bench import __version__
from bayesprobe_terminal_bench.config import TerminalBenchConfig
from bayesprobe_terminal_bench.runner_factory import build_live_session


class BayesProbeHarborAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return "bayesprobe"

    def version(self) -> str | None:
        return __version__

    async def setup(self, environment: BaseEnvironment) -> None:
        return None

    async def run(self, instruction, environment, context) -> None:
        config, api_key = TerminalBenchConfig.from_sources(self.extra_env)
        session = build_live_session(
            config=config,
            api_key=api_key,
            instruction=instruction,
            environment=environment,
            event_loop=asyncio.get_running_loop(),
            logs_dir=self.logs_dir,
            session_id=self.session_id,
            context_id=self.context_id,
        )
        result = await asyncio.to_thread(session.runner.run_question, session.input)
        stop_reason = getattr(getattr(result, "stop_reason", None), "value", "not_admitted")
        metadata = {
            "bayesprobe_run_id": session.input.run_id,
            "bayesprobe_stop_reason": stop_reason,
            "bayesprobe_cycles": len(getattr(result, "cycle_results", [])),
            "terminal_actions": session.budget.actions_used,
            "model_calls": session.budget.model_calls_used,
        }
        context.metadata = {**dict(context.metadata or {}), **metadata}
        session.artifacts.write_summary(metadata)
```

Use the `TerminalBenchConfig.from_sources` contract implemented in Task 2. Do
not mutate global environment variables or copy the API key into Harbor
metadata.

- [ ] **Step 4: Verify GREEN and Harbor importability**

Run:

```bash
uv run --project benchmarks/terminal_bench pytest benchmarks/terminal_bench/tests/test_agent.py benchmarks/terminal_bench/tests/test_public_reuse.py -q
uv run --project benchmarks/terminal_bench python -c 'from bayesprobe_terminal_bench.agent import BayesProbeHarborAgent; print(BayesProbeHarborAgent.import_path())'
```

Expected: tests PASS and the import path is
`bayesprobe_terminal_bench.agent:BayesProbeHarborAgent`.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/terminal_bench/src/bayesprobe_terminal_bench/agent.py benchmarks/terminal_bench/src/bayesprobe_terminal_bench/config.py benchmarks/terminal_bench/tests/test_agent.py benchmarks/terminal_bench/tests/test_config.py
git commit -m "feat: expose BayesProbe as a Harbor agent"
```

### Task 9: Lock and run the official one-task engineering smoke

**Files:**

- Create: `benchmarks/terminal_bench/configs/oracle-smoke.yaml`
- Create: `benchmarks/terminal_bench/configs/bayesprobe-smoke.yaml`
- Create: `benchmarks/terminal_bench/scripts/write_benchmark_lock.py`
- Create: `benchmarks/terminal_bench/scripts/validate_smoke_run.py`
- Create: `benchmarks/terminal_bench/tests/test_benchmark_lock.py`
- Create: `benchmarks/terminal_bench/README.md`

**Interfaces:**

- The fixed engineering task is `terminal-bench/break-filter-js-from-html`.
- Oracle reward 1 is required before `.runs/benchmark.lock.json` is written.
- The lock records Harbor version, dataset, task identity, checksum, image digest, Git identities, prompt/schema versions, and every budget.
- The BayesProbe smoke refuses to start without a matching lock.
- `validate_smoke_run.py` requires a completed verifier and complete BayesProbe trace; reward 0 remains a task result.

- [ ] **Step 1: Write failing lock and secret-scan tests**

```python
import json

import pytest

from bayesprobe_terminal_bench.config import TerminalBenchConfig
from write_benchmark_lock import RuntimeIdentity, build_lock


RUNTIME_IDENTITY = RuntimeIdentity(
    harbor_version="0.18.0",
    root_git_sha="root-sha",
    adapter_tree_sha="adapter-sha",
    image_digest="sha256:image",
)


def test_lock_requires_oracle_reward_one(tmp_path) -> None:
    job = tmp_path / "oracle-job"
    job.mkdir()
    (job / "result.json").write_text(json.dumps({"reward": 0.0}))
    with pytest.raises(ValueError, match="oracle reward must be 1"):
        build_lock(
            job_dir=job,
            config=TerminalBenchConfig(model="test-model"),
            runtime_identity=RUNTIME_IDENTITY,
        )


def test_serialized_lock_excludes_provider_key(synthetic_oracle_job) -> None:
    lock = build_lock(
        job_dir=synthetic_oracle_job,
        config=TerminalBenchConfig(model="test-model"),
        runtime_identity=RUNTIME_IDENTITY,
        restricted_values=("provider-secret",),
    )
    assert "provider-secret" not in json.dumps(lock, sort_keys=True)
```

Extend `tests/conftest.py` with `synthetic_oracle_job`. It must create exactly
one completed trial containing `config.json` and `result.json` with reward
`1.0`, dataset `terminal-bench/terminal-bench-2`, task
`terminal-bench/break-filter-js-from-html`, a fixed dataset revision and task
checksum, and a fixed container image. This fixture defines the offline parser
contract; a captured real Oracle layout is added after the first bootstrap if
Harbor carries equivalent fields at different nested paths.

- [ ] **Step 2: Verify RED**

Run: `uv run --project benchmarks/terminal_bench pytest benchmarks/terminal_bench/tests/test_benchmark_lock.py -q`

Expected: import failure because the lock script does not exist.

- [ ] **Step 3: Add exact Harbor smoke configurations**

```yaml
# configs/oracle-smoke.yaml
job_name: bayesprobe-terminal-bench-oracle-smoke
jobs_dir: .runs/harbor/oracle
n_attempts: 1
datasets:
  - name: terminal-bench/terminal-bench-2
    task_names:
      - terminal-bench/break-filter-js-from-html
agents:
  - name: oracle
environment:
  type: docker
  delete: true
orchestrator:
  type: local
  n_concurrent_trials: 1
```

```yaml
# configs/bayesprobe-smoke.yaml
job_name: bayesprobe-terminal-bench-agent-smoke
jobs_dir: .runs/harbor/bayesprobe
n_attempts: 1
datasets:
  - name: terminal-bench/terminal-bench-2
    task_names:
      - terminal-bench/break-filter-js-from-html
agents:
  - import_path: bayesprobe_terminal_bench.agent:BayesProbeHarborAgent
    model_name: ${BAYESPROBE_BENCH_MODEL}
    env:
      BAYESPROBE_BENCH_API_KEY: ${BAYESPROBE_BENCH_API_KEY}
      BAYESPROBE_BENCH_BASE_URL: ${BAYESPROBE_BENCH_BASE_URL}
      BAYESPROBE_BENCH_MODEL: ${BAYESPROBE_BENCH_MODEL}
      BAYESPROBE_BENCH_LOCK_PATH: .runs/benchmark.lock.json
environment:
  type: docker
  delete: true
orchestrator:
  type: local
  n_concurrent_trials: 1
```

- [ ] **Step 4: Implement deterministic lock extraction**

Separate pure lock construction from live process discovery. Define this value
object:

```python
@dataclass(frozen=True)
class RuntimeIdentity:
    harbor_version: str
    root_git_sha: str
    adapter_tree_sha: str
    image_digest: str
```

Implement `build_lock(*, job_dir: Path, config: TerminalBenchConfig,
runtime_identity: RuntimeIdentity, restricted_values: tuple[str, ...] = ()) ->
dict[str, object]` and `collect_runtime_identity(*, repository_root: Path,
container_image: str) -> RuntimeIdentity`.

`build_lock` is pure apart from reading the supplied Oracle job directory; unit
tests pass fixed `RuntimeIdentity` values and never invoke Git, Docker, or
network access. `collect_runtime_identity` is called only by the CLI and runs
the pinned Harbor-version, Git, and Docker-image checks.

Together they must perform all of these checks:

1. locate exactly one completed trial below the supplied Oracle job directory;
2. recursively extract the official reward and require `1.0`;
3. extract task ID, ref/checksum, source dataset, and Docker image from Harbor's
   `config.json` and `result.json`;
4. resolve the image digest with `docker image inspect`;
5. require `importlib.metadata.version("harbor") == "0.18.0"`;
6. record `git rev-parse HEAD` and
   `git rev-parse HEAD:benchmarks/terminal_bench`;
7. record the fixed budgets and prompt/schema version `terminal_probe_plan:v0.1`;
8. reject serialized content containing the resolved provider key;
9. write the result atomically to `.runs/benchmark.lock.json`.

The serialized lock uses the flat keys consumed by `load_and_validate_lock` in
Task 7, including provider protocol, key-variable name, and temperature, plus
`dataset_revision`, `task_checksum`, `container_image`,
`image_digest`, `root_git_sha`, and `adapter_tree_sha`. This is one shared schema,
not a second lock representation.

- [ ] **Step 5: Implement smoke-result classification**

`validate_smoke_run.py` emits exactly one classification:

```text
engineering_pass       verifier completed and BayesProbe trace is complete
task_failure           verifier completed with reward 0
infrastructure_error   Harbor failed before the first agent action
provider_error         provider failed after agent startup
conformance_error      a completed cycle lacks provenance-linked stages
```

It exits zero for `engineering_pass` and `task_failure`, because both prove the
engineering path reached the verifier. It exits non-zero for the other three.

The README must contain these concrete sections: prerequisites, nested `uv`
setup, Oracle bootstrap, lock creation, BayesProbe smoke, artifact locations,
result classifications, secret handling, and the explicit statement that this
slice is an engineering test rather than an accuracy claim. Commands must be
the same commands used below, not paraphrased alternatives.

- [ ] **Step 6: Verify all offline benchmark tests**

Run: `uv run --project benchmarks/terminal_bench pytest -q`

Expected: all nested tests PASS without Docker or live provider access.

- [ ] **Step 7: Commit the tested smoke harness before generating identities**

```bash
git add benchmarks/terminal_bench/configs benchmarks/terminal_bench/scripts benchmarks/terminal_bench/tests/conftest.py benchmarks/terminal_bench/tests/test_benchmark_lock.py benchmarks/terminal_bench/README.md
git commit -m "test: add Terminal-Bench engineering smoke"
```

Expected: the adapter tree referenced by `HEAD:benchmarks/terminal_bench` now
includes every file that the smoke run will execute.

- [ ] **Step 8: Run Oracle and create the lock**

Run from `benchmarks/terminal_bench`:

Before invoking the lock writer, export `BAYESPROBE_BENCH_MODEL`,
`BAYESPROBE_BENCH_BASE_URL`, and `BAYESPROBE_BENCH_API_KEY`. The CLI uses the
Task 2 source resolver, records only non-secret provider settings, and passes
the returned key only as a restricted value for the leak scan.

```bash
HARBOR_TELEMETRY=off uv run harbor run -c configs/oracle-smoke.yaml
ORACLE_JOB_DIR="$(find .runs/harbor/oracle -mindepth 1 -maxdepth 1 -type d | sort | tail -1)"
uv run python scripts/write_benchmark_lock.py --oracle-job "$ORACLE_JOB_DIR" --output .runs/benchmark.lock.json
```

Expected: Oracle reward is `1.0` and the lock is created.

- [ ] **Step 9: Run one real BayesProbe task and validate it**

After exporting the three provider variables, run:

```bash
HARBOR_TELEMETRY=off uv run harbor run -c configs/bayesprobe-smoke.yaml
BAYESPROBE_JOB_DIR="$(find .runs/harbor/bayesprobe -mindepth 1 -maxdepth 1 -type d | sort | tail -1)"
uv run python scripts/validate_smoke_run.py --job "$BAYESPROBE_JOB_DIR" --lock .runs/benchmark.lock.json
```

Expected: the custom agent imports, at least one terminal action executes, the
real runner emits a complete cycle trace, and Harbor reaches the official
verifier. Reward 0 is `task_failure`, not an integration failure.

- [ ] **Step 10: Run final root and nested regression gates**

Run from the repository root:

```bash
uv run pytest -q
uv run --project benchmarks/terminal_bench pytest -q
git diff --exit-code -- bayesprobe pyproject.toml
git diff --exit-code 068b414..HEAD -- bayesprobe pyproject.toml
git diff --check
```

Expected: both suites PASS and both Git checks print nothing.

## Completion Gate

This plan is complete only when:

1. all nested offline tests pass;
2. all existing root tests pass;
3. production source has no private BayesProbe import or shadow kernel module;
4. the public `AutonomousQuestionRunner.run_question` is called exactly once per trial;
5. every directional posterior change traces to Harbor `TOOL_RESULT` Signal IDs;
6. a no-signal cycle leaves posterior values unchanged;
7. Oracle receives reward 1 on the fixed task;
8. the BayesProbe trial reaches the official verifier;
9. no provider key appears in committed or generated artifacts;
10. no file under `bayesprobe/` or the root `pyproject.toml` changes.

After this gate passes, write a separate implementation plan for the minimal
ReAct shell control and paired three-task capability smoke. Do not add that work
to this vertical slice.
