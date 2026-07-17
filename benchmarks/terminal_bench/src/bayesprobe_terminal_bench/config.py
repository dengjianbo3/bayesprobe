from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from threading import Lock
from typing import Self

from pydantic import BaseModel, ConfigDict, Field


class BudgetExhausted(RuntimeError):
    category = "budget_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)


class ProviderIdentityError(BudgetExhausted):
    # Existing contract repair wrappers already rethrow BudgetExhausted unchanged.
    category = "provider_identity_error"


_STABLE_TRIAL_CATEGORIES = frozenset(
    {
        "provider_contract_error",
        "provider_transport_error",
        "provider_identity_error",
        "budget_error",
        "adapter_error",
        "causal_conformance_error",
        "policy_error",
    }
)
_CATEGORY_ALIASES = {
    "provider_error": "provider_transport_error",
    "plan_error": "adapter_error",
    "budget_exhausted": "budget_error",
    "model_budget_exhausted": "budget_error",
    "action_budget_exhausted": "budget_error",
    "causal_adapter_error": "causal_conformance_error",
}


def classify_trial_error(error: BaseException) -> str:
    if isinstance(error, ProviderIdentityError):
        return "provider_identity_error"
    if isinstance(error, BudgetExhausted):
        return "budget_error"

    category = getattr(error, "category", None)
    if category in _STABLE_TRIAL_CATEGORIES:
        return category
    if isinstance(category, str) and category in _CATEGORY_ALIASES:
        return _CATEGORY_ALIASES[category]

    error_name = type(error).__name__
    if error_name == "ProviderContractError":
        return "provider_contract_error"
    if error_name == "CausalTraceError":
        return "causal_conformance_error"
    if error_name == "PolicyViolation":
        return "policy_error"
    if type(error).__module__.split(".", 1)[0] in {"httpx", "openai"}:
        return "provider_transport_error"
    return "adapter_error"


class RunBudget:
    def __init__(
        self,
        *,
        max_actions: int = 24,
        max_model_calls: int = 40,
        max_provider_tokens: int = 160_000,
        reservation_guard: Callable[[], None] | None = None,
    ) -> None:
        self.max_actions = max_actions
        self.max_model_calls = max_model_calls
        self.max_provider_tokens = max_provider_tokens
        self._reservation_guard = reservation_guard
        self._actions = 0
        self._model_calls = 0
        self._provider_tokens = 0
        self._lock = Lock()

    def reserve_action(self) -> int:
        self._require_reservation_allowed()
        with self._lock:
            if self._actions >= self.max_actions:
                raise BudgetExhausted("terminal action budget exhausted")
            self._actions += 1
            return self._actions

    def reserve_model_call(self) -> int:
        self._require_reservation_allowed()
        with self._lock:
            if self._model_calls >= self.max_model_calls:
                raise BudgetExhausted("model call budget exhausted")
            self._model_calls += 1
            return self._model_calls

    def record_provider_usage(self, total_tokens: object) -> int:
        with self._lock:
            if type(total_tokens) is not int or total_tokens < 0:
                raise ProviderIdentityError(
                    "provider total token usage must be a non-negative integer"
                )
            self._provider_tokens += total_tokens
            if self._provider_tokens > self.max_provider_tokens:
                raise BudgetExhausted("provider token budget exhausted")
            return self._provider_tokens

    def _require_reservation_allowed(self) -> None:
        if self._reservation_guard is not None:
            self._reservation_guard()

    @property
    def actions_used(self) -> int:
        with self._lock:
            return self._actions

    @property
    def model_calls_used(self) -> int:
        with self._lock:
            return self._model_calls

    @property
    def provider_tokens_used(self) -> int:
        with self._lock:
            return self._provider_tokens


class TerminalBenchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    model: str
    api_key_env: str = "BAYESPROBE_BENCH_API_KEY"
    base_url: str | None = None
    provider_timeout_seconds: int = Field(default=360, ge=1)
    command_timeout_seconds: int = Field(default=120, ge=1, le=120)
    max_output_tokens: int = Field(default=8_192, ge=256)
    max_cycles: int = Field(default=3, ge=1)
    max_probes_per_cycle: int = Field(default=2, ge=1)
    max_actions_per_probe: int = Field(default=3, ge=1, le=3)
    max_total_actions: int = Field(default=24, ge=1)
    max_model_calls: int = Field(default=72, ge=1)
    max_provider_tokens: int = Field(default=160_000, ge=1)
    signal_output_bytes: int = Field(default=32_768, ge=1)
    task_timeout_seconds: int | None = Field(default=None, ge=1)
    lock_path: Path = Path(".runs/benchmark.lock.json")

    @classmethod
    def from_sources(
        cls,
        extra_env: Mapping[str, str] | None = None,
    ) -> tuple[Self, str]:
        if extra_env is not None and (
            not isinstance(extra_env, Mapping)
            or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in extra_env.items()
            )
        ):
            raise ValueError("extra_env must be a mapping of strings to strings")
        source = {**os.environ, **dict(extra_env or {})}
        model = source.get("BAYESPROBE_BENCH_MODEL", "").strip()
        if not model:
            raise ValueError("BAYESPROBE_BENCH_MODEL is required")
        api_key = source.get("BAYESPROBE_BENCH_API_KEY", "").strip()
        if not api_key:
            raise ValueError("BAYESPROBE_BENCH_API_KEY is required")
        task_timeout = source.get(
            "BAYESPROBE_BENCH_TASK_TIMEOUT_SECONDS", ""
        ).strip()
        if not task_timeout:
            raise ValueError("BAYESPROBE_BENCH_TASK_TIMEOUT_SECONDS is required")
        if not re.fullmatch(r"[1-9]\d*", task_timeout):
            raise ValueError(
                "BAYESPROBE_BENCH_TASK_TIMEOUT_SECONDS must be a positive integer"
            )
        config = cls(
            model=model,
            base_url=source.get("BAYESPROBE_BENCH_BASE_URL", "").strip() or None,
            task_timeout_seconds=int(task_timeout),
            lock_path=Path(
                source.get("BAYESPROBE_BENCH_LOCK_PATH", ".runs/benchmark.lock.json")
            ),
        )
        return config, api_key
