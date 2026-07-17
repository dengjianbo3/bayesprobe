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


def test_shared_budget_rejects_concurrent_attempts_above_hard_limit() -> None:
    budget = RunBudget(max_actions=8, max_model_calls=8)

    def reserve_action_with_interleaved_read() -> tuple[int | None, BaseException | None, int]:
        try:
            return budget.reserve_action(), None, budget.actions_used
        except BudgetExhausted as error:
            return None, error, budget.actions_used

    with ThreadPoolExecutor(max_workers=24) as executor:
        results = list(executor.map(lambda _: reserve_action_with_interleaved_read(), range(24)))

    successful = [reservation for reservation, error, _ in results if error is None]
    exhausted = [error for _, error, _ in results if error is not None]
    observed_counts = [count for _, _, count in results]

    assert sorted(successful) == list(range(1, 9))
    assert len(exhausted) == 16
    assert all(isinstance(error, BudgetExhausted) for error in exhausted)
    assert all(count <= 8 for count in observed_counts)
    assert budget.actions_used == 8


def test_shared_budget_accumulates_provider_tokens_thread_safely() -> None:
    budget = RunBudget(
        max_actions=1,
        max_model_calls=1,
        max_provider_tokens=1_000,
    )

    with ThreadPoolExecutor(max_workers=16) as executor:
        totals = list(executor.map(budget.record_provider_usage, [10] * 40))

    assert sorted(totals) == list(range(10, 401, 10))
    assert budget.provider_tokens_used == 400


@pytest.mark.parametrize("usage", [None, True, 1.5, "1", -1])
def test_shared_budget_rejects_invalid_provider_usage_without_coercion(
    usage: object,
) -> None:
    budget = RunBudget(max_provider_tokens=100)

    with pytest.raises(BudgetExhausted) as failure:
        budget.record_provider_usage(usage)

    assert failure.value.category == "provider_identity_error"
    assert budget.provider_tokens_used == 0


def test_shared_budget_fails_immediately_after_provider_token_overflow() -> None:
    budget = RunBudget(max_provider_tokens=10)
    assert budget.record_provider_usage(6) == 6

    with pytest.raises(BudgetExhausted) as failure:
        budget.record_provider_usage(5)

    assert failure.value.category == "budget_error"
    assert budget.provider_tokens_used == 11


def test_trial_deadline_applies_margin_and_recomputes_fresh_timeouts() -> None:
    from bayesprobe_terminal_bench.deadline import TrialDeadline

    now = [100.0]
    deadline = TrialDeadline(timeout_seconds=100, monotonic=lambda: now[0])

    assert deadline.timeout_for(configured_timeout_seconds=360) == 95
    now[0] = 101.1
    assert deadline.timeout_for(configured_timeout_seconds=360) == 93
    now[0] = 110.0
    assert deadline.timeout_for(configured_timeout_seconds=60) == 60

    now[0] = 195.0
    with pytest.raises(BudgetExhausted) as failure:
        deadline.timeout_for(configured_timeout_seconds=360)
    assert failure.value.category == "budget_error"


def test_trial_deadline_keeps_configured_operation_caps_independent() -> None:
    from bayesprobe_terminal_bench.deadline import TrialDeadline

    now = [0.0]
    deadline = TrialDeadline(timeout_seconds=500, monotonic=lambda: now[0])

    assert deadline.timeout_for(configured_timeout_seconds=120) == 120
    now[0] = 1.0
    assert deadline.timeout_for(configured_timeout_seconds=360) == 360


def test_trial_error_classification_uses_only_stable_public_categories() -> None:
    from bayesprobe_terminal_bench.causal import CausalTraceError
    from bayesprobe_terminal_bench.config import (
        ProviderIdentityError,
        classify_trial_error,
    )
    from bayesprobe_terminal_bench.environment import PolicyViolation
    from bayesprobe_terminal_bench.planning import TerminalPlanError
    from bayesprobe_terminal_bench.provider_contract import ProviderContractError

    cases = [
        (
            ProviderContractError(stage="terminal_task_frame", attempts=3),
            "provider_contract_error",
        ),
        (
            TerminalPlanError(category="provider_error", attempts=1),
            "provider_transport_error",
        ),
        (ProviderIdentityError("provider model drift"), "provider_identity_error"),
        (BudgetExhausted("deadline exhausted"), "budget_error"),
        (RuntimeError("adapter failed"), "adapter_error"),
        (CausalTraceError("lineage failed"), "causal_conformance_error"),
        (PolicyViolation("denied"), "policy_error"),
    ]

    assert [classify_trial_error(error) for error, _ in cases] == [
        category for _, category in cases
    ]


def test_extra_env_wins_and_config_never_serializes_key_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BAYESPROBE_BENCH_API_KEY", "host-secret")
    config, api_key = TerminalBenchConfig.from_sources({
        "BAYESPROBE_BENCH_API_KEY": "one-time-provider-secret",
        "BAYESPROBE_BENCH_MODEL": "deepseek-v4-flash",
        "BAYESPROBE_BENCH_TASK_TIMEOUT_SECONDS": "900",
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
        "BAYESPROBE_BENCH_TASK_TIMEOUT_SECONDS": "900",
    })

    assert config.base_url == "https://provider.example/v1"
    assert str(config.lock_path) == ".runs/custom.lock.json"
    assert config.provider_timeout_seconds == 360
    assert config.command_timeout_seconds == 120
    assert config.max_output_tokens == 8_192
    assert config.max_cycles == 3
    assert config.max_probes_per_cycle == 2
    assert config.max_actions_per_probe == 3
    assert config.max_total_actions == 24
    assert config.max_model_calls == 72
    assert config.max_provider_tokens == 160_000
    assert config.signal_output_bytes == 32_768
    assert config.task_timeout_seconds == 900


def test_default_smoke_model_budget_covers_all_cycle_repair_paths() -> None:
    config = TerminalBenchConfig(model="test-model")
    initialization_calls = 6
    calls_per_cycle = (
        2 * config.max_probes_per_cycle
        + 2 * config.max_probes_per_cycle * config.max_actions_per_probe
        + 2
        + 2
    )
    intercycle_probe_design_calls = 2 * (config.max_cycles - 1)

    assert config.max_model_calls >= (
        initialization_calls
        + config.max_cycles * calls_per_cycle
        + intercycle_probe_design_calls
    )


@pytest.mark.parametrize(
    "name",
    [
        "BAYESPROBE_BENCH_MODEL",
        "BAYESPROBE_BENCH_API_KEY",
        "BAYESPROBE_BENCH_TASK_TIMEOUT_SECONDS",
    ],
)
def test_config_requires_model_and_api_key(name: str) -> None:
    source = {
        "BAYESPROBE_BENCH_API_KEY": "secret",
        "BAYESPROBE_BENCH_MODEL": "model",
        "BAYESPROBE_BENCH_TASK_TIMEOUT_SECONDS": "900",
    }
    source[name] = "  "

    with pytest.raises(ValueError, match=f"{name} is required"):
        TerminalBenchConfig.from_sources(source)


@pytest.mark.parametrize(
    "extra_env",
    [
        ["not-a-mapping"],
        {"BAYESPROBE_BENCH_MODEL": 1},
        {1: "model"},
    ],
)
def test_config_rejects_non_string_extra_environment(extra_env: object) -> None:
    with pytest.raises(ValueError, match="extra_env must be a mapping of strings to strings"):
        TerminalBenchConfig.from_sources(extra_env)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_cycles": "8"},
        {"max_cycles": True},
        {"command_timeout_seconds": "120"},
    ],
)
def test_config_rejects_coerced_and_boolean_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        TerminalBenchConfig(model="model", **kwargs)


def test_config_enforces_brief_bounds() -> None:
    with pytest.raises(ValidationError):
        TerminalBenchConfig(model="model", command_timeout_seconds=121)
    with pytest.raises(ValidationError):
        TerminalBenchConfig(model="model", max_actions_per_probe=4)
