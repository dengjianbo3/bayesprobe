from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bayesprobe_terminal_bench.config import TerminalBenchConfig
from bayesprobe_terminal_bench.experiment_lock import (
    FROZEN_GATE_TASK_IDS,
    FROZEN_GATE_TASK_REFS,
)
from validate_paired_gate import validate_paired_gate_jobs
from write_paired_gate_lock import build_paired_gate_lock


def _config() -> TerminalBenchConfig:
    return TerminalBenchConfig(
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _oracle_job(root: Path, *, failed_task: str | None = None) -> Path:
    _write_json(
        root / "config.json",
        {
            "n_attempts": 1,
            "datasets": [
                {
                    "name": "terminal-bench/terminal-bench-2",
                    "ref": "sha256:" + "1" * 64,
                    "task_names": list(FROZEN_GATE_TASK_IDS),
                }
            ],
        },
    )
    _write_json(
        root / "lock.json",
        {
            "harbor": {"version": "0.18.0"},
            "trials": [
                {
                    "task": {
                        "name": task_id,
                        "digest": FROZEN_GATE_TASK_REFS[task_id],
                        "source": "terminal-bench/terminal-bench-2",
                    }
                }
                for task_id in FROZEN_GATE_TASK_IDS
            ],
        },
    )
    for task_id in FROZEN_GATE_TASK_IDS:
        slug = task_id.split("/", 1)[1]
        reward = 0.0 if task_id == failed_task else 1.0
        _write_json(
            root / f"{slug}__oracle" / "result.json",
            {
                "task_name": task_id,
                "task_id": {
                    "org": "terminal-bench",
                    "name": slug,
                    "ref": FROZEN_GATE_TASK_REFS[task_id],
                },
                "verifier_result": {"rewards": {"reward": reward}},
                "exception_info": None,
                "finished_at": "2026-07-15T00:00:00Z",
            },
        )
    return root


def _arm_job(
    root: Path,
    *,
    arm: str,
    rewards: tuple[float, float, float],
    missing_verifier_task: str | None = None,
) -> Path:
    import_path = (
        "bayesprobe_terminal_bench.direct_agent:DirectHarborAgent"
        if arm == "direct"
        else "bayesprobe_terminal_bench.agent:BayesProbeHarborAgent"
    )
    for index, task_id in enumerate(FROZEN_GATE_TASK_IDS):
        slug = task_id.split("/", 1)[1]
        trial = root / f"{slug}__{arm}"
        metadata = {
            "terminal_actions": index + 1,
            "model_calls": index + 2,
        }
        result = {
            "task_name": task_id,
            "task_id": {
                "org": "terminal-bench",
                "name": slug,
                "ref": FROZEN_GATE_TASK_REFS[task_id],
            },
            "config": {"agent": {"import_path": import_path}},
            "agent_result": {"metadata": metadata},
            "verifier_result": (
                None
                if task_id == missing_verifier_task
                else {"rewards": {"reward": rewards[index]}}
            ),
            "exception_info": None,
            "started_at": "2026-07-15T00:00:00Z",
            "finished_at": "2026-07-15T00:01:00Z",
        }
        _write_json(trial / "result.json", result)
        artifact_name = "direct" if arm == "direct" else "bayesprobe"
        _write_json(trial / "agent" / artifact_name / "summary.json", metadata)
    return root


def test_build_paired_gate_lock_requires_three_successful_oracle_trials(
    tmp_path: Path,
) -> None:
    job = _oracle_job(tmp_path / "oracle")
    runtime = SimpleNamespace(
        harbor_version="0.18.0",
        root_git_sha="a" * 40,
        adapter_tree_sha="b" * 40,
    )

    lock = build_paired_gate_lock(
        job_dir=job,
        config=_config(),
        runtime_identity=runtime,
        image_digest_resolver=lambda task_id, task_ref: (
            "sha256:" + str(FROZEN_GATE_TASK_IDS.index(task_id) + 4) * 64
        ),
    )

    assert lock["schema_version"] == "terminal_bench_paired_gate:v0.1"
    assert [task["task_id"] for task in lock["tasks"]] == list(
        FROZEN_GATE_TASK_IDS
    )
    assert lock["max_model_calls"] == 72


def test_build_paired_gate_lock_rejects_oracle_zero(tmp_path: Path) -> None:
    failed = FROZEN_GATE_TASK_IDS[1]
    job = _oracle_job(tmp_path / "oracle", failed_task=failed)
    with pytest.raises(ValueError, match="Oracle reward must be 1"):
        build_paired_gate_lock(
            job_dir=job,
            config=_config(),
            runtime_identity=SimpleNamespace(
                harbor_version="0.18.0",
                root_git_sha="a" * 40,
                adapter_tree_sha="b" * 40,
            ),
            image_digest_resolver=lambda task_id, task_ref: "sha256:" + "4" * 64,
        )


def test_validate_paired_gate_reports_official_rewards_and_gate_pass(
    tmp_path: Path,
) -> None:
    lock = build_paired_gate_lock(
        job_dir=_oracle_job(tmp_path / "oracle"),
        config=_config(),
        runtime_identity=SimpleNamespace(
            harbor_version="0.18.0",
            root_git_sha="a" * 40,
            adapter_tree_sha="b" * 40,
        ),
        image_digest_resolver=lambda task_id, task_ref: "sha256:" + "4" * 64,
    )
    lock_path = tmp_path / "gate.lock.json"
    _write_json(lock_path, lock)
    direct = _arm_job(
        tmp_path / "direct",
        arm="direct",
        rewards=(1.0, 0.0, 0.0),
    )
    bayesprobe = _arm_job(
        tmp_path / "bayesprobe",
        arm="bayesprobe",
        rewards=(0.0, 1.0, 0.0),
    )

    report = validate_paired_gate_jobs(
        lock_path=lock_path,
        direct_job=direct,
        bayesprobe_job=bayesprobe,
        trace_validator=lambda path: True,
    )

    assert report["gate_passed"] is True
    assert report["arms"]["direct"]["reward_total"] == 1.0
    assert report["arms"]["bayesprobe"]["reward_total"] == 1.0
    assert report["arms"]["bayesprobe"]["tasks"][1]["reward"] == 1.0
    assert report["arms"]["bayesprobe"]["tasks"][1]["terminal_actions"] == 2


def test_validate_paired_gate_fails_when_bayesprobe_is_zero_of_three(
    tmp_path: Path,
) -> None:
    lock = build_paired_gate_lock(
        job_dir=_oracle_job(tmp_path / "oracle"),
        config=_config(),
        runtime_identity=SimpleNamespace(
            harbor_version="0.18.0",
            root_git_sha="a" * 40,
            adapter_tree_sha="b" * 40,
        ),
        image_digest_resolver=lambda task_id, task_ref: "sha256:" + "4" * 64,
    )
    lock_path = tmp_path / "gate.lock.json"
    _write_json(lock_path, lock)

    report = validate_paired_gate_jobs(
        lock_path=lock_path,
        direct_job=_arm_job(
            tmp_path / "direct", arm="direct", rewards=(0.0, 0.0, 0.0)
        ),
        bayesprobe_job=_arm_job(
            tmp_path / "bayesprobe",
            arm="bayesprobe",
            rewards=(0.0, 0.0, 0.0),
        ),
        trace_validator=lambda path: True,
    )

    assert report["gate_passed"] is False
    assert "bayesprobe_zero_of_three" in report["gate_failures"]


def test_validate_paired_gate_rejects_missing_verifier_or_secret(
    tmp_path: Path,
) -> None:
    lock = build_paired_gate_lock(
        job_dir=_oracle_job(tmp_path / "oracle"),
        config=_config(),
        runtime_identity=SimpleNamespace(
            harbor_version="0.18.0",
            root_git_sha="a" * 40,
            adapter_tree_sha="b" * 40,
        ),
        image_digest_resolver=lambda task_id, task_ref: "sha256:" + "4" * 64,
    )
    lock_path = tmp_path / "gate.lock.json"
    _write_json(lock_path, lock)
    direct = _arm_job(
        tmp_path / "direct",
        arm="direct",
        rewards=(0.0, 0.0, 0.0),
        missing_verifier_task=FROZEN_GATE_TASK_IDS[0],
    )
    bayesprobe = _arm_job(
        tmp_path / "bayesprobe",
        arm="bayesprobe",
        rewards=(1.0, 0.0, 0.0),
    )

    report = validate_paired_gate_jobs(
        lock_path=lock_path,
        direct_job=direct,
        bayesprobe_job=bayesprobe,
        trace_validator=lambda path: True,
    )
    assert report["gate_passed"] is False
    assert "direct_incomplete_verifier" in report["gate_failures"]

    secret_file = next(direct.rglob("summary.json"))
    secret_file.write_text('sk-abcdefghijklmnop1234567890', encoding="utf-8")
    with pytest.raises(ValueError, match="secret-like content"):
        validate_paired_gate_jobs(
            lock_path=lock_path,
            direct_job=direct,
            bayesprobe_job=bayesprobe,
            trace_validator=lambda path: True,
        )
