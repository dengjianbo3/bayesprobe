from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from pydantic import ValidationError

from bayesprobe_terminal_bench.config import TerminalBenchConfig
from bayesprobe_terminal_bench.experiment_lock import (
    CausalQualificationLock,
    FROZEN_GATE_TASK_IDS,
    FROZEN_GATE_TASK_REFS,
    LockedBudgets,
    load_paired_gate_lock,
)
from bayesprobe_terminal_bench.planning import plan_contract_identity
from bayesprobe_terminal_bench.provider_contract import contract_identity
from bayesprobe_terminal_bench.runner_factory import (
    RepositoryGitIdentity,
    validate_runtime_lock,
)


def _config() -> TerminalBenchConfig:
    return TerminalBenchConfig(
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
    )


def _payload() -> dict[str, object]:
    config = _config()
    return {
        "schema_version": "terminal_bench_paired_gate:v0.1",
        "harbor_version": "0.18.0",
        "dataset_name": "terminal-bench/terminal-bench-2",
        "dataset_revision": "sha256:" + "1" * 64,
        "tasks": [
            {
                "task_id": task_id,
                "task_ref": FROZEN_GATE_TASK_REFS[task_id],
                "image_digest": "sha256:" + str(index + 3) * 64,
            }
            for index, task_id in enumerate(FROZEN_GATE_TASK_IDS, start=2)
        ],
        "root_git_sha": "a" * 40,
        "adapter_tree_sha": "b" * 40,
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
        "arms": {
            "direct": (
                "bayesprobe_terminal_bench.direct_agent:DirectHarborAgent"
            ),
            "bayesprobe": (
                "bayesprobe_terminal_bench.agent:BayesProbeHarborAgent"
            ),
        },
    }


def _write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _runtime(*, dirty: bool = False) -> object:
    return SimpleNamespace(
        root_git_sha="a" * 40,
        adapter_tree_sha="b" * 40,
        adapter_dirty=dirty,
    )


def _causal_payload() -> dict[str, object]:
    return {
        "schema_version": "terminal_bench_causal_qualification:v1",
        "harbor_version": "0.18.0",
        "dataset_name": "terminal-bench/terminal-bench-2",
        "dataset_revision": "sha256:" + "1" * 64,
        "tasks": [
            {
                "task_id": task_id,
                "task_ref": FROZEN_GATE_TASK_REFS[task_id],
                "image_digest": "sha256:" + str(index + 2) * 64,
                "agent_timeout_seconds": timeout,
            }
            for index, (task_id, timeout) in enumerate(
                zip(FROZEN_GATE_TASK_IDS, (1200, 900, 900), strict=True)
            )
        ],
        "root_git_sha": "a" * 40,
        "adapter_tree_sha": "b" * 40,
        "model": "deepseek-v4-flash",
        "base_url": "https://api.deepseek.com",
        "provider_protocol": "openai_chat_completions",
        "temperature": 0,
        "budgets": {
            "max_total_actions": 24,
            "max_model_calls": 72,
            "max_provider_tokens": 160000,
            "max_output_tokens": 8192,
            "command_timeout_seconds": 120,
            "provider_timeout_seconds": 360,
            "signal_output_bytes": 32768,
        },
        "prompt_schema_hashes": {
            **contract_identity(),
            **plan_contract_identity(),
        },
        "expected_provider_model": "fixture-model-v1",
        "provider_identity_sha256": "sha256:" + "c" * 64,
        "expected_system_fingerprint_available": True,
        "expected_system_fingerprint": "fixture-fingerprint-v1",
    }


def test_paired_gate_lock_accepts_exact_controls_arm_and_session_task(
    tmp_path: Path,
) -> None:
    path = tmp_path / "gate.lock.json"
    _write(path, _payload())

    lock = load_paired_gate_lock(
        path,
        _config(),
        arm="direct",
        session_id="cancel-async-tasks__AbCd123__agent",
        runtime_git_identity=_runtime(),
    )

    assert tuple(task.task_id for task in lock.tasks) == FROZEN_GATE_TASK_IDS
    assert lock.arms["direct"].endswith(":DirectHarborAgent")
    assert all(task.agent_timeout_seconds is None for task in lock.tasks)
    assert all(
        "agent_timeout_seconds" not in task
        for task in lock.model_dump(mode="json")["tasks"]
    )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda payload: payload["tasks"].reverse(),
            "frozen task order",
        ),
        (
            lambda payload: payload.update(max_model_calls=71),
            "max_model_calls",
        ),
        (
            lambda payload: payload["arms"].update(direct="other:Agent"),
            "arms",
        ),
    ],
)
def test_paired_gate_lock_rejects_changed_tasks_or_fairness_controls(
    tmp_path: Path,
    mutation: object,
    match: str,
) -> None:
    payload = _payload()
    mutation(payload)
    path = tmp_path / "gate.lock.json"
    _write(path, payload)

    with pytest.raises(ValueError, match=match):
        load_paired_gate_lock(
            path,
            _config(),
            arm="direct",
            session_id="cancel-async-tasks__AbCd123__agent",
            runtime_git_identity=_runtime(),
        )


def test_paired_gate_lock_rejects_unknown_arm_or_unlocked_task(tmp_path: Path) -> None:
    path = tmp_path / "gate.lock.json"
    _write(path, _payload())

    with pytest.raises(ValueError, match="unknown paired gate arm"):
        load_paired_gate_lock(
            path,
            _config(),
            arm="oracle",
            session_id="cancel-async-tasks__AbCd123__agent",
            runtime_git_identity=_runtime(),
        )
    with pytest.raises(ValueError, match="session task is not locked"):
        load_paired_gate_lock(
            path,
            _config(),
            arm="direct",
            session_id="substituted-task__AbCd123__agent",
            runtime_git_identity=_runtime(),
        )


def test_paired_gate_lock_rejects_dirty_or_different_runtime(tmp_path: Path) -> None:
    path = tmp_path / "gate.lock.json"
    _write(path, _payload())

    with pytest.raises(ValueError, match="dirty_adapter_worktree"):
        load_paired_gate_lock(
            path,
            _config(),
            arm="bayesprobe",
            session_id="log-summary-date-ranges__AbCd123__agent",
            runtime_git_identity=_runtime(dirty=True),
        )
    with pytest.raises(ValueError, match="root_git_sha"):
        load_paired_gate_lock(
            path,
            _config(),
            arm="bayesprobe",
            session_id="log-summary-date-ranges__AbCd123__agent",
            runtime_git_identity=SimpleNamespace(
                root_git_sha="f" * 40,
                adapter_tree_sha="b" * 40,
                adapter_dirty=False,
            ),
        )


def test_paired_gate_lock_rejects_secret_shaped_fields(tmp_path: Path) -> None:
    payload = _payload()
    payload["api_key"] = "must-not-be-written"
    path = tmp_path / "gate.lock.json"
    _write(path, payload)

    with pytest.raises(ValueError, match="valid paired gate JSON"):
        load_paired_gate_lock(
            path,
            _config(),
            arm="direct",
            session_id="cancel-async-tasks__AbCd123__agent",
            runtime_git_identity=_runtime(),
        )


def test_gate_configs_freeze_same_tasks_and_single_attempt() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    configs = {
        name: yaml.safe_load((project_dir / "configs" / name).read_text(encoding="utf-8"))
        for name in ("oracle-gate.yaml", "direct-gate.yaml", "bayesprobe-gate.yaml")
    }

    assert {
        tuple(config["datasets"][0]["task_names"])
        for config in configs.values()
    } == {FROZEN_GATE_TASK_IDS}
    assert {config["n_attempts"] for config in configs.values()} == {1}
    assert {
        config["orchestrator"]["n_concurrent_trials"]
        for config in configs.values()
    } == {1}
    assert configs["direct-gate.yaml"]["agents"][0]["env"] == configs[
        "bayesprobe-gate.yaml"
    ]["agents"][0]["env"]


def test_causal_qualification_lock_is_strict_frozen_and_requires_task_timeouts() -> None:
    payload = _causal_payload()
    lock = CausalQualificationLock.model_validate(payload)

    assert isinstance(lock.budgets, LockedBudgets)
    assert tuple(task.agent_timeout_seconds for task in lock.tasks) == (
        1200,
        900,
        900,
    )
    with pytest.raises(ValidationError):
        lock.model = "changed"  # type: ignore[misc]

    missing_timeout = _causal_payload()
    tasks = missing_timeout["tasks"]
    assert isinstance(tasks, list)
    first = tasks[0]
    assert isinstance(first, dict)
    first.pop("agent_timeout_seconds")
    with pytest.raises(ValidationError, match="agent timeout"):
        CausalQualificationLock.model_validate(missing_timeout)

    extra = _causal_payload()
    extra["api_key"] = "forbidden"
    with pytest.raises(ValidationError):
        CausalQualificationLock.model_validate(extra)


def test_causal_qualification_lock_rejects_budget_task_and_contract_drift() -> None:
    mutations: list[tuple[str, object]] = []

    budget = _causal_payload()
    budgets = budget["budgets"]
    assert isinstance(budgets, dict)
    budgets["max_model_calls"] = 71
    mutations.append(("Stage 0 budgets", budget))

    timeout = _causal_payload()
    tasks = timeout["tasks"]
    assert isinstance(tasks, list)
    second = tasks[1]
    assert isinstance(second, dict)
    second["agent_timeout_seconds"] = 901
    mutations.append(("agent timeout", timeout))

    task_order = _causal_payload()
    ordered_tasks = task_order["tasks"]
    assert isinstance(ordered_tasks, list)
    ordered_tasks.reverse()
    mutations.append(("frozen task order", task_order))

    stale_contract = _causal_payload()
    hashes = stale_contract["prompt_schema_hashes"]
    assert isinstance(hashes, dict)
    hashes["terminal_task_frame:v1:schema"] = "sha256:" + "f" * 64
    mutations.append(("prompt/schema", stale_contract))

    old_plan = _causal_payload()
    hashes = old_plan["prompt_schema_hashes"]
    assert isinstance(hashes, dict)
    hashes["terminal_probe_plan:v0.1:schema"] = hashes.pop(
        "terminal_probe_plan:v1:schema"
    )
    mutations.append(("prompt/schema", old_plan))

    old_signal = _causal_payload()
    hashes = old_signal["prompt_schema_hashes"]
    assert isinstance(hashes, dict)
    hashes["harbor-observation:v2:schema"] = hashes.pop(
        "harbor-observation:v3:schema"
    )
    mutations.append(("prompt/schema", old_signal))

    missing_provider_artifact = _causal_payload()
    missing_provider_artifact.pop("provider_identity_sha256")
    mutations.append(("provider_identity_sha256", missing_provider_artifact))

    unavailable_fingerprint_value = _causal_payload()
    unavailable_fingerprint_value["expected_system_fingerprint_available"] = False
    mutations.append(("fingerprint availability", unavailable_fingerprint_value))

    available_missing_fingerprint = _causal_payload()
    available_missing_fingerprint["expected_system_fingerprint"] = None
    mutations.append(("fingerprint availability", available_missing_fingerprint))

    for match, candidate in mutations:
        with pytest.raises(ValidationError, match=match):
            CausalQualificationLock.model_validate(candidate)


def test_active_runtime_accepts_causal_lock_for_exact_session_task(
    tmp_path: Path,
) -> None:
    payload = _causal_payload()
    path = tmp_path / "causal.lock.json"
    _write(path, payload)
    config = TerminalBenchConfig(
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
        task_timeout_seconds=1200,
    )

    validated = validate_runtime_lock(
        path,
        config,
        arm="bayesprobe",
        session_id="break-filter-js-from-html__run__agent",
        runtime_git_identity=RepositoryGitIdentity(
            root_git_sha="a" * 40,
            adapter_tree_sha="b" * 40,
            adapter_dirty=False,
        ),
    )

    assert validated["budgets"]["max_provider_tokens"] == 160000
    assert validated["expected_provider_model"] == "fixture-model-v1"
    assert validated["expected_system_fingerprint"] == "fixture-fingerprint-v1"


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("arm", "arm"),
        ("session", "session_id"),
        ("timeout", "agent_timeout_seconds"),
        ("model", "model"),
        ("budget", "max_model_calls"),
        ("git", "root_git_sha"),
        ("dirty", "dirty_adapter_worktree"),
    ],
)
def test_active_runtime_rejects_causal_lock_drift(
    tmp_path: Path,
    case: str,
    match: str,
) -> None:
    payload = _causal_payload()
    path = tmp_path / f"{case}.lock.json"
    _write(path, payload)
    config = TerminalBenchConfig(
        model=("other-model" if case == "model" else "deepseek-v4-flash"),
        base_url="https://api.deepseek.com",
        task_timeout_seconds=(900 if case == "timeout" else 1200),
        max_model_calls=(71 if case == "budget" else 72),
    )
    runtime = RepositoryGitIdentity(
        root_git_sha=("c" * 40 if case == "git" else "a" * 40),
        adapter_tree_sha="b" * 40,
        adapter_dirty=case == "dirty",
    )

    with pytest.raises(ValueError, match=match):
        validate_runtime_lock(
            path,
            config,
            arm="direct" if case == "arm" else "bayesprobe",
            session_id=(
                "unknown-task__run__agent"
                if case == "session"
                else "break-filter-js-from-html__run__agent"
            ),
            runtime_git_identity=runtime,
        )
