from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import ValidationError

from bayesprobe_terminal_bench.config import BudgetExhausted, RunBudget, TerminalBenchConfig


def test_shared_budget_is_hard() -> None:
    budget = RunBudget(max_actions=1, max_model_calls=1)
    assert budget.reserve_action() == 1
    assert budget.reserve_model_call() == 1
    with pytest.raises(BudgetExhausted, match="terminal action budget exhausted"):
        budget.reserve_action()
    with pytest.raises(BudgetExhausted, match="model call budget exhausted"):
        budget.reserve_model_call()


def test_shared_budget_serializes_concurrent_reservations_and_reads() -> None:
    budget = RunBudget(max_actions=40, max_model_calls=40)

    def reserve_both() -> tuple[int, int]:
        return budget.reserve_action(), budget.reserve_model_call()

    with ThreadPoolExecutor(max_workers=16) as executor:
        reservations = list(executor.map(lambda _: reserve_both(), range(40)))

    assert sorted(action for action, _ in reservations) == list(range(1, 41))
    assert sorted(call for _, call in reservations) == list(range(1, 41))
    assert budget.actions_used == 40
    assert budget.model_calls_used == 40


def test_extra_env_wins_and_config_never_serializes_key_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BAYESPROBE_BENCH_API_KEY", "host-secret")
    config, api_key = TerminalBenchConfig.from_sources({
        "BAYESPROBE_BENCH_API_KEY": "one-time-provider-secret",
        "BAYESPROBE_BENCH_MODEL": "deepseek-v4-flash",
    })
    assert api_key == "one-time-provider-secret"
    assert config.api_key_env == "BAYESPROBE_BENCH_API_KEY"
    assert "one-time-provider-secret" not in json.dumps(config.model_dump(mode="json"))
    assert "host-secret" not in json.dumps(config.model_dump(mode="json"))


def test_config_uses_exact_defaults_and_source_overrides() -> None:
    config, _ = TerminalBenchConfig.from_sources({
        "BAYESPROBE_BENCH_API_KEY": "secret",
        "BAYESPROBE_BENCH_MODEL": "model",
        "BAYESPROBE_BENCH_BASE_URL": "https://provider.example/v1 ",
        "BAYESPROBE_BENCH_LOCK_PATH": ".runs/custom.lock.json",
    })

    assert config.base_url == "https://provider.example/v1"
    assert str(config.lock_path) == ".runs/custom.lock.json"
    assert config.provider_timeout_seconds == 360
    assert config.command_timeout_seconds == 120
    assert config.max_output_tokens == 8_192
    assert config.max_cycles == 8
    assert config.max_probes_per_cycle == 2
    assert config.max_actions_per_probe == 3
    assert config.max_total_actions == 24
    assert config.max_model_calls == 40
    assert config.signal_output_bytes == 32_768


@pytest.mark.parametrize("name", ["BAYESPROBE_BENCH_MODEL", "BAYESPROBE_BENCH_API_KEY"])
def test_config_requires_model_and_api_key(name: str) -> None:
    source = {
        "BAYESPROBE_BENCH_API_KEY": "secret",
        "BAYESPROBE_BENCH_MODEL": "model",
    }
    source[name] = "  "

    with pytest.raises(ValueError, match=f"{name} is required"):
        TerminalBenchConfig.from_sources(source)


def test_config_enforces_brief_bounds() -> None:
    with pytest.raises(ValidationError):
        TerminalBenchConfig(model="model", command_timeout_seconds=121)
    with pytest.raises(ValidationError):
        TerminalBenchConfig(model="model", max_actions_per_probe=4)
