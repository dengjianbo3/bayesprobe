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
        with self._lock:
            return self._actions

    @property
    def model_calls_used(self) -> int:
        with self._lock:
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
            lock_path=Path(
                source.get("BAYESPROBE_BENCH_LOCK_PATH", ".runs/benchmark.lock.json")
            ),
        )
        return config, api_key
