from __future__ import annotations

import json
import re
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from bayesprobe_terminal_bench.config import TerminalBenchConfig


FROZEN_GATE_TASK_IDS = (
    "terminal-bench/break-filter-js-from-html",
    "terminal-bench/cancel-async-tasks",
    "terminal-bench/log-summary-date-ranges",
)
FROZEN_GATE_TASK_REFS = {
    "terminal-bench/break-filter-js-from-html": (
        "sha256:59a2641df9bca789642ad4ab3f5790de5ffed6eb4a594ca7846d26422a55c4a8"
    ),
    "terminal-bench/cancel-async-tasks": (
        "sha256:7c230a29f27c49c2fff88f4721165f4241e456bd87a94cd525be05ae98c6cbbb"
    ),
    "terminal-bench/log-summary-date-ranges": (
        "sha256:bd0eb5e8434840a46c623c8d29c71b4a6d0fc5c7bcbf637b6d1aef36b98f5cc5"
    ),
}
FROZEN_GATE_TASK_TIMEOUTS = {
    "terminal-bench/break-filter-js-from-html": 1200,
    "terminal-bench/cancel-async-tasks": 900,
    "terminal-bench/log-summary-date-ranges": 900,
}
PAIRED_GATE_ARMS = {
    "direct": "bayesprobe_terminal_bench.direct_agent:DirectHarborAgent",
    "bayesprobe": "bayesprobe_terminal_bench.agent:BayesProbeHarborAgent",
}
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_OBJECT_ID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


def experiment_lock_sha256(payload: Mapping[str, object]) -> str:
    serialized = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return f"sha256:{sha256(serialized.encode('utf-8')).hexdigest()}"


class GateTask(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    task_id: str
    task_ref: str
    image_digest: str
    agent_timeout_seconds: int | None = Field(
        default=None,
        ge=1,
        exclude_if=lambda value: value is None,
    )

    @field_validator("task_ref", "image_digest")
    @classmethod
    def require_sha256(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("digest must be sha256")
        return value


class LockedBudgets(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    max_total_actions: int = Field(ge=1)
    max_model_calls: int = Field(ge=1)
    max_provider_tokens: int = Field(ge=1)
    max_output_tokens: int = Field(ge=256)
    command_timeout_seconds: int = Field(ge=1, le=120)
    provider_timeout_seconds: int = Field(ge=1)
    signal_output_bytes: int = Field(ge=1)


_STAGE0_BUDGETS = LockedBudgets(
    max_total_actions=24,
    max_model_calls=72,
    max_provider_tokens=160_000,
    max_output_tokens=8_192,
    command_timeout_seconds=120,
    provider_timeout_seconds=360,
    signal_output_bytes=32_768,
)


class CausalQualificationLock(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["terminal_bench_causal_qualification:v1"]
    harbor_version: Literal["0.18.0"]
    dataset_name: Literal["terminal-bench/terminal-bench-2"]
    dataset_revision: str
    tasks: tuple[GateTask, GateTask, GateTask]
    root_git_sha: str
    adapter_tree_sha: str
    model: str
    base_url: str | None
    provider_protocol: Literal["openai_chat_completions"]
    temperature: Literal[0]
    budgets: LockedBudgets
    prompt_schema_hashes: dict[str, str]
    expected_provider_model: str
    provider_identity_sha256: str
    expected_system_fingerprint_available: bool
    expected_system_fingerprint: str | None

    @model_validator(mode="before")
    @classmethod
    def normalize_tasks(cls, value: object) -> object:
        if isinstance(value, Mapping) and isinstance(value.get("tasks"), list):
            return {**value, "tasks": tuple(value["tasks"])}
        return value

    @field_validator("dataset_revision")
    @classmethod
    def require_dataset_digest(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("dataset revision must be sha256")
        return value

    @field_validator("root_git_sha", "adapter_tree_sha")
    @classmethod
    def require_git_object_id(cls, value: str) -> str:
        if not _GIT_OBJECT_ID.fullmatch(value):
            raise ValueError("Git identity must be an object id")
        return value

    @field_validator("model", "expected_provider_model")
    @classmethod
    def require_nonempty_identity(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model identity must be non-empty")
        return value

    @field_validator("base_url")
    @classmethod
    def require_nonempty_base_url(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("base URL must be non-empty")
        return value

    @field_validator("expected_system_fingerprint")
    @classmethod
    def require_valid_fingerprint(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("provider fingerprint must be non-empty")
        return value

    @field_validator("provider_identity_sha256")
    @classmethod
    def require_provider_identity_digest(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("provider identity artifact hash must be sha256")
        return value

    @model_validator(mode="after")
    def require_frozen_contract(self) -> CausalQualificationLock:
        if tuple(task.task_id for task in self.tasks) != FROZEN_GATE_TASK_IDS:
            raise ValueError("causal qualification requires the frozen task order")
        if any(
            task.task_ref != FROZEN_GATE_TASK_REFS[task.task_id]
            for task in self.tasks
        ):
            raise ValueError("causal qualification requires the frozen task refs")
        if any(
            task.agent_timeout_seconds is None
            or task.agent_timeout_seconds
            != FROZEN_GATE_TASK_TIMEOUTS[task.task_id]
            for task in self.tasks
        ):
            raise ValueError("causal qualification requires the official agent timeout")
        if self.budgets != _STAGE0_BUDGETS:
            raise ValueError("causal qualification requires the Stage 0 budgets")
        if self.expected_system_fingerprint_available != (
            self.expected_system_fingerprint is not None
        ):
            raise ValueError("provider fingerprint availability disagrees with value")
        from bayesprobe_terminal_bench.planning import plan_contract_identity
        from bayesprobe_terminal_bench.provider_contract import contract_identity

        expected_identities = {
            **contract_identity(),
            **plan_contract_identity(),
        }
        if self.prompt_schema_hashes != expected_identities:
            raise ValueError("causal qualification prompt/schema identity drift")
        return self


class PairedGateLock(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: str
    harbor_version: str
    dataset_name: str
    dataset_revision: str
    tasks: tuple[GateTask, ...] = Field(min_length=3, max_length=3)
    root_git_sha: str
    adapter_tree_sha: str
    n_attempts: int = Field(ge=1)
    model: str
    base_url: str | None
    provider_protocol: str
    api_key_env: str
    temperature: int
    max_cycles: int = Field(ge=1)
    max_probes_per_cycle: int = Field(ge=1)
    max_actions_per_probe: int = Field(ge=1)
    max_total_actions: int = Field(ge=1)
    max_model_calls: int = Field(ge=1)
    command_timeout_seconds: int = Field(ge=1)
    provider_timeout_seconds: int = Field(ge=1)
    max_output_tokens: int = Field(ge=1)
    signal_output_bytes: int = Field(ge=1)
    arms: dict[str, str]

    @model_validator(mode="before")
    @classmethod
    def normalize_tasks(cls, value: object) -> object:
        if isinstance(value, Mapping) and isinstance(value.get("tasks"), list):
            return {**value, "tasks": tuple(value["tasks"])}
        return value

    @field_validator("dataset_revision")
    @classmethod
    def require_dataset_digest(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("dataset revision must be sha256")
        return value

    @field_validator("root_git_sha", "adapter_tree_sha")
    @classmethod
    def require_git_object_id(cls, value: str) -> str:
        if not _GIT_OBJECT_ID.fullmatch(value):
            raise ValueError("Git identity must be an object id")
        return value

    @model_validator(mode="after")
    def require_frozen_contract(self) -> PairedGateLock:
        if tuple(task.task_id for task in self.tasks) != FROZEN_GATE_TASK_IDS:
            raise ValueError("paired gate requires the frozen task order")
        if any(
            task.task_ref != FROZEN_GATE_TASK_REFS[task.task_id]
            for task in self.tasks
        ):
            raise ValueError("paired gate requires the frozen task refs")
        if self.arms != PAIRED_GATE_ARMS:
            raise ValueError("paired gate arms do not match")
        return self


def load_paired_gate_lock(
    path: Path,
    config: TerminalBenchConfig,
    *,
    arm: str,
    session_id: str | None,
    runtime_git_identity: Any | None = None,
) -> PairedGateLock:
    if arm not in PAIRED_GATE_ARMS:
        raise ValueError("unknown paired gate arm")
    lock_path = Path(path)
    try:
        lock = PairedGateLock.model_validate_json(lock_path.read_text(encoding="utf-8"))
    except Exception as error:
        detail = str(error)
        if "frozen task order" in detail:
            raise ValueError("paired gate requires the frozen task order") from None
        if "paired gate arms do not match" in detail:
            raise ValueError("paired gate arms do not match") from None
        if "frozen task refs" in detail:
            raise ValueError("paired gate requires the frozen task refs") from None
        raise ValueError("paired gate lock must contain valid paired gate JSON") from None

    expected: dict[str, object] = {
        "schema_version": "terminal_bench_paired_gate:v0.1",
        "harbor_version": "0.18.0",
        "dataset_name": "terminal-bench/terminal-bench-2",
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
        "arms": PAIRED_GATE_ARMS,
    }
    mismatches = {
        name
        for name, expected_value in expected.items()
        if getattr(lock, name) != expected_value
    }
    if mismatches:
        raise ValueError(f"paired gate lock mismatch: {sorted(mismatches)}")

    if not _session_matches_locked_task(session_id, lock.tasks):
        raise ValueError("session task is not locked")

    runtime = runtime_git_identity
    if runtime is None:
        from bayesprobe_terminal_bench.runner_factory import (
            collect_repository_git_identity,
        )

        runtime = collect_repository_git_identity(
            Path(__file__).resolve().parents[4]
        )
    runtime_mismatches = {
        name
        for name, actual in {
            "root_git_sha": runtime.root_git_sha,
            "adapter_tree_sha": runtime.adapter_tree_sha,
        }.items()
        if getattr(lock, name) != actual
    }
    if runtime.adapter_dirty:
        runtime_mismatches.add("dirty_adapter_worktree")
    if runtime_mismatches:
        raise ValueError(
            f"paired gate runtime mismatch: {sorted(runtime_mismatches)}"
        )
    return lock


def paired_gate_schema_version(path: Path) -> str | None:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, Mapping):
        return None
    schema = value.get("schema_version")
    return schema if isinstance(schema, str) else None


def _session_matches_locked_task(
    session_id: str | None,
    tasks: tuple[GateTask, ...],
) -> bool:
    if not isinstance(session_id, str) or not session_id:
        return False
    return any(
        session_id.startswith(f"{task.task_id.split('/', 1)[1]}__")
        for task in tasks
    )
