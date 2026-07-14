from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bayesprobe_terminal_bench.config import TerminalBenchConfig
from bayesprobe_terminal_bench.runner_factory import load_and_validate_lock
from conftest import (
    FIXED_CONTAINER_IMAGE,
    FIXED_DATASET,
    FIXED_DATASET_REVISION,
    FIXED_TASK,
    FIXED_TASK_CHECKSUM,
)
from validate_smoke_run import classify_smoke_run, main as validate_main
import write_benchmark_lock
from write_benchmark_lock import (
    RuntimeIdentity,
    build_lock,
    collect_runtime_identity,
    write_lock_atomic,
)


RUNTIME_IDENTITY = RuntimeIdentity(
    harbor_version="0.18.0",
    root_git_sha="root-sha",
    adapter_tree_sha="adapter-sha",
    image_digest="sha256:" + "4" * 64,
)


def test_lock_requires_oracle_reward_one(tmp_path: Path) -> None:
    job = tmp_path / "oracle-job"
    job.mkdir()
    (job / "result.json").write_text(json.dumps({"reward": 0.0}))

    with pytest.raises(ValueError, match="oracle reward must be 1"):
        build_lock(
            job_dir=job,
            config=TerminalBenchConfig(model="test-model"),
            runtime_identity=RUNTIME_IDENTITY,
        )


def test_lock_extracts_official_identity_and_matches_runtime_validator(
    synthetic_oracle_job: Path,
    tmp_path: Path,
) -> None:
    config = TerminalBenchConfig(
        model="test-model",
        base_url="https://provider.example/v1",
    )

    lock = build_lock(
        job_dir=synthetic_oracle_job,
        config=config,
        runtime_identity=RUNTIME_IDENTITY,
    )

    assert lock == {
        "schema_version": "terminal_bench_lock:v0.1",
        "harbor_version": "0.18.0",
        "dataset_name": FIXED_DATASET,
        "dataset_revision": FIXED_DATASET_REVISION,
        "task_id": FIXED_TASK,
        "task_checksum": FIXED_TASK_CHECKSUM,
        "container_image": FIXED_CONTAINER_IMAGE,
        "image_digest": "sha256:" + "4" * 64,
        "root_git_sha": "root-sha",
        "adapter_tree_sha": "adapter-sha",
        "n_attempts": 1,
        "model": "test-model",
        "base_url": "https://provider.example/v1",
        "provider_protocol": "openai_chat_completions",
        "api_key_env": "BAYESPROBE_BENCH_API_KEY",
        "temperature": 0,
        "max_cycles": 8,
        "max_probes_per_cycle": 2,
        "max_actions_per_probe": 3,
        "max_total_actions": 24,
        "max_model_calls": 40,
        "command_timeout_seconds": 120,
        "provider_timeout_seconds": 360,
        "max_output_tokens": 8_192,
        "signal_output_bytes": 32_768,
        "terminal_plan_version": "terminal_probe_plan:v0.1",
    }
    output = tmp_path / "benchmark.lock.json"
    write_lock_atomic(output, lock)
    assert load_and_validate_lock(output, config) == lock


def test_lock_rejects_multiple_completed_trials(
    synthetic_oracle_job: Path,
) -> None:
    first_trial = next(
        path for path in synthetic_oracle_job.iterdir() if path.is_dir()
    )
    second_trial = synthetic_oracle_job / "second-completed-trial"
    second_trial.mkdir()
    for filename in ("config.json", "result.json"):
        (second_trial / filename).write_bytes((first_trial / filename).read_bytes())

    with pytest.raises(ValueError, match="exactly one completed Oracle trial"):
        build_lock(
            job_dir=synthetic_oracle_job,
            config=TerminalBenchConfig(model="test-model"),
            runtime_identity=RUNTIME_IDENTITY,
        )


def test_lock_rejects_conflicting_trial_task_id(
    synthetic_oracle_job: Path,
) -> None:
    trial_dir = next(path for path in synthetic_oracle_job.iterdir() if path.is_dir())
    config_path = trial_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["task"]["name"] = "terminal-bench/a-different-task"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="conflicting task identities"):
        build_lock(
            job_dir=synthetic_oracle_job,
            config=TerminalBenchConfig(model="test-model"),
            runtime_identity=RUNTIME_IDENTITY,
        )


def test_serialized_lock_excludes_provider_key(
    synthetic_oracle_job: Path,
) -> None:
    lock = build_lock(
        job_dir=synthetic_oracle_job,
        config=TerminalBenchConfig(model="test-model"),
        runtime_identity=RUNTIME_IDENTITY,
        restricted_values=("provider-secret",),
    )
    assert "provider-secret" not in json.dumps(lock, sort_keys=True)


def test_atomic_writer_rejects_restricted_value_without_replacing_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "benchmark.lock.json"
    output.write_text('{"old": true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="restricted value"):
        write_lock_atomic(
            output,
            {"message": "provider-secret"},
            restricted_values=("provider-secret",),
        )

    assert output.read_text(encoding="utf-8") == '{"old": true}\n'
    assert list(tmp_path.iterdir()) == [output]


def test_runtime_discovery_selects_digest_for_the_locked_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], Path | None]] = []
    matching_digest = "sha256:" + "a" * 64
    unrelated_digest = "sha256:" + "b" * 64

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append((command, kwargs.get("cwd")))
        if command == ["git", "rev-parse", "HEAD"]:
            stdout = "root-sha\n"
        elif command == ["git", "rev-parse", "HEAD:benchmarks/terminal_bench"]:
            stdout = "adapter-sha\n"
        else:
            stdout = json.dumps(
                [
                    {
                        "Id": "sha256:" + "c" * 64,
                        "RepoDigests": [
                            f"registry.example/unrelated@{unrelated_digest}",
                            f"registry.example/task@{matching_digest}",
                        ],
                    }
                ]
            )
        return SimpleNamespace(stdout=stdout)

    monkeypatch.setattr(write_benchmark_lock, "version", lambda name: "0.18.0")
    monkeypatch.setattr(write_benchmark_lock.subprocess, "run", fake_run)

    identity = collect_runtime_identity(
        repository_root=tmp_path,
        container_image="registry.example/task:locked",
    )

    assert identity == RuntimeIdentity(
        harbor_version="0.18.0",
        root_git_sha="root-sha",
        adapter_tree_sha="adapter-sha",
        image_digest=matching_digest,
    )
    assert calls == [
        (["git", "rev-parse", "HEAD"], tmp_path),
        (["git", "rev-parse", "HEAD:benchmarks/terminal_bench"], tmp_path),
        (["docker", "image", "inspect", "registry.example/task:locked"], None),
    ]


def _write_smoke_job(
    root: Path,
    *,
    reward: float | None,
    trace: str,
) -> Path:
    job_dir = root / f"job-{trace}"
    trial_dir = job_dir / "trial"
    bayesprobe_dir = trial_dir / "agent" / "bayesprobe"
    bayesprobe_dir.mkdir(parents=True)
    (trial_dir / "config.json").write_text("{}", encoding="utf-8")

    exception_info = None
    verifier_result = None
    finished_at = "2026-07-14T00:01:00Z"
    if reward is not None:
        verifier_result = {"rewards": {"reward": reward}}
    elif trace == "infrastructure":
        exception_info = {
            "exception_type": "DockerImageBuildError",
            "exception_message": "environment failed",
        }
    elif trace == "provider":
        exception_info = {
            "exception_type": "BayesProbeHarborAgentError",
            "exception_message": "agent execution failed",
        }
    elif trace == "agent_failure":
        exception_info = {
            "exception_type": "BayesProbeHarborAgentError",
            "exception_message": "agent execution failed",
        }

    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "finished_at": finished_at,
                "verifier_result": verifier_result,
                "exception_info": exception_info,
            }
        ),
        encoding="utf-8",
    )

    if trace == "infrastructure":
        return job_dir

    if trace == "provider":
        (bayesprobe_dir / "provider_telemetry.jsonl").write_text(
            json.dumps(
                {
                    "task": "terminal_probe_plan",
                    "outcome": "error",
                    "error_type": "AuthenticationError",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return job_dir

    (bayesprobe_dir / "summary.json").write_text(
        json.dumps(
            {
                "bayesprobe_cycles": 1,
                "terminal_actions": 1,
                "model_calls": 4,
            }
        ),
        encoding="utf-8",
    )
    (bayesprobe_dir / "environment_actions.jsonl").write_text(
        json.dumps(
            {
                "action_index": 1,
                "post_environment_state_id": "env:0",
                "return_code": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    records = [
        {
            "record_type": "belief_state",
            "payload": {"cycle_id": "cycle_0", "belief_state_id": "B0"},
        },
        {
            "record_type": "cycle",
            "payload": {
                "cycle_id": "cycle_1",
                "boundary_status": "integrated",
                "completed_at": "2026-07-14T00:00:30Z",
            },
        },
        {
            "record_type": "probe_set",
            "payload": {
                "cycle_id": "cycle_1",
                "probe_set_id": "PS1",
                "probes": [{"id": "P1", "cycle_id": "cycle_1"}],
            },
        },
        {
            "record_type": "external_signal",
            "payload": {
                "id": "S1",
                "cycle_id": "cycle_1",
                "generated_by_probe": "P1",
                "provenance": {
                    "epistemic_origin": "tool_result",
                    "derivation_root_id": "harbor-action:sha256:abc",
                },
            },
        },
        {
            "record_type": "evidence_event",
            "payload": {
                "id": "E1",
                "derived_from_signal": "S1",
                "epistemic_origin": "tool_result",
                "discard_reason": None,
            },
        },
        {
            "record_type": "belief_update",
            "payload": {
                "cycle_id": "cycle_1",
                "evidence_id": "E1",
                "prior": 0.5,
                "posterior": 0.7,
                "direction": "strengthened",
                "sensitivity": {"caused_by_event_ids": ["E1"]},
            },
        },
        {
            "record_type": "belief_state",
            "payload": {"cycle_id": "cycle_1", "belief_state_id": "B1"},
        },
        {
            "record_type": "run",
            "payload": {"status": "completed", "current_cycle_id": "cycle_1"},
        },
    ]
    if trace == "incomplete":
        records = [record for record in records if record["record_type"] != "evidence_event"]
    (bayesprobe_dir / "bayesprobe_ledger.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return job_dir


@pytest.mark.parametrize(
    ("reward", "trace", "classification", "exit_code"),
    [
        (1.0, "complete", "engineering_pass", 0),
        (0.0, "complete", "task_failure", 0),
        (None, "infrastructure", "infrastructure_error", 1),
        (None, "provider", "provider_error", 1),
        (None, "agent_failure", "conformance_error", 1),
        (1.0, "incomplete", "conformance_error", 1),
    ],
)
def test_smoke_classifier_and_cli_exit_codes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    reward: float | None,
    trace: str,
    classification: str,
    exit_code: int,
) -> None:
    job_dir = _write_smoke_job(tmp_path, reward=reward, trace=trace)
    lock_path = tmp_path / "benchmark.lock.json"
    lock_path.write_text(
        json.dumps({"task_id": FIXED_TASK, "task_checksum": FIXED_TASK_CHECKSUM}),
        encoding="utf-8",
    )

    assert classify_smoke_run(job_dir=job_dir, lock_path=lock_path) == classification
    assert validate_main(["--job", str(job_dir), "--lock", str(lock_path)]) == exit_code
    assert capsys.readouterr().out == classification + "\n"
