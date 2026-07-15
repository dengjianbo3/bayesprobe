from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from bayesprobe_terminal_bench.config import TerminalBenchConfig
from bayesprobe_terminal_bench.experiment_lock import (
    FROZEN_GATE_TASK_IDS,
    FROZEN_GATE_TASK_REFS,
    load_paired_gate_lock,
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
            session_id="build-cython-ext__AbCd123__agent",
            runtime_git_identity=_runtime(dirty=True),
        )
    with pytest.raises(ValueError, match="root_git_sha"):
        load_paired_gate_lock(
            path,
            _config(),
            arm="bayesprobe",
            session_id="build-cython-ext__AbCd123__agent",
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
