from __future__ import annotations

import hashlib
import json
import unicodedata
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from bayesprobe import ExternalSignal, ProbeDesign, ProbeExecutionBrief
from bayesprobe.evidence_memory import SignalProvenanceNormalizer
from bayesprobe_terminal_bench.actions import ActionObservation, ShellAction
from bayesprobe_terminal_bench.config import TerminalBenchConfig
from bayesprobe_terminal_bench.runner_factory import (
    RepositoryGitIdentity,
    load_and_validate_lock,
)
from bayesprobe_terminal_bench.signals import signal_from_observation
from conftest import (
    FIXED_CONTAINER_IMAGE,
    FIXED_DATASET,
    FIXED_DATASET_REVISION,
    FIXED_TASK,
    FIXED_TASK_CHECKSUM,
    write_harbor_job_artifacts,
)
from harbor.environments.definition import environment_content_hash
from harbor.models.task.config import (
    EnvironmentConfig as TaskEnvironmentConfig,
    TaskConfig as TaskDefinitionConfig,
)
from harbor.models.trial.config import TrialConfig
from harbor.models.trial.result import TrialResult
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
    root_git_sha="a" * 40,
    adapter_tree_sha="b" * 40,
    container_image=FIXED_CONTAINER_IMAGE,
    image_digest="sha256:" + "4" * 64,
)


def _classify(
    *,
    job_dir: Path,
    lock_path: Path,
    runtime_identity: RuntimeIdentity = RUNTIME_IDENTITY,
) -> str:
    return classify_smoke_run(
        job_dir=job_dir,
        lock_path=lock_path,
        runtime_identity=runtime_identity,
    )


def test_runtime_identity_owns_container_image() -> None:
    assert "container_image" in RuntimeIdentity.__dataclass_fields__


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
        "root_git_sha": "a" * 40,
        "adapter_tree_sha": "b" * 40,
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
    assert load_and_validate_lock(
        output,
        config,
        runtime_git_identity=RepositoryGitIdentity(
            root_git_sha=RUNTIME_IDENTITY.root_git_sha,
            adapter_tree_sha=RUNTIME_IDENTITY.adapter_tree_sha,
            adapter_dirty=False,
        ),
    ) == lock


def test_real_harbor_model_dumps_build_lock_without_persisted_image_fields(
    synthetic_oracle_job: Path,
) -> None:
    trial_dir = next(path for path in synthetic_oracle_job.iterdir() if path.is_dir())
    trial_config_payload = json.loads(
        (trial_dir / "config.json").read_text(encoding="utf-8")
    )
    trial_result_payload = json.loads(
        (trial_dir / "result.json").read_text(encoding="utf-8")
    )

    TrialConfig.model_validate(trial_config_payload)
    TrialResult.model_validate(trial_result_payload)
    serialized = json.dumps(
        {"config": trial_config_payload, "result": trial_result_payload}
    )
    assert "docker_image" not in serialized
    assert "container_image" not in serialized

    lock = build_lock(
        job_dir=synthetic_oracle_job,
        config=TerminalBenchConfig(model="test-model"),
        runtime_identity=RUNTIME_IDENTITY,
    )

    assert lock["container_image"] == FIXED_CONTAINER_IMAGE


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


def test_lock_accepts_real_harbor_package_identity_shape(
    synthetic_oracle_job: Path,
) -> None:
    trial_dir = next(path for path in synthetic_oracle_job.iterdir() if path.is_dir())
    result_path = trial_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["task_name"] = FIXED_TASK
    result["task_checksum"] = "4" * 64
    result_path.write_text(json.dumps(result), encoding="utf-8")

    lock = build_lock(
        job_dir=synthetic_oracle_job,
        config=TerminalBenchConfig(model="test-model"),
        runtime_identity=RUNTIME_IDENTITY,
    )

    assert lock["task_checksum"] == "sha256:" + "4" * 64


@pytest.mark.parametrize(
    ("task_names", "remove_field"),
    [
        (None, True),
        ([], False),
        ([FIXED_TASK, "terminal-bench/other-task"], False),
        (["terminal-bench/other-task"], False),
    ],
)
def test_lock_requires_dataset_task_names_to_select_exactly_the_fixed_task(
    synthetic_oracle_job: Path,
    task_names: list[str] | None,
    remove_field: bool,
) -> None:
    config_path = synthetic_oracle_job / "config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if remove_field:
        payload["datasets"][0].pop("task_names")
    else:
        payload["datasets"][0]["task_names"] = task_names
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="task_names"):
        build_lock(
            job_dir=synthetic_oracle_job,
            config=TerminalBenchConfig(model="test-model"),
            runtime_identity=RUNTIME_IDENTITY,
        )


@pytest.mark.parametrize(
    "conflict",
    [
        "job-lock-source",
        "result-source",
        "result-config-ref",
        "job-lock-digest",
    ],
)
def test_lock_rejects_cross_artifact_identity_conflicts_without_precedence_masking(
    synthetic_oracle_job: Path,
    conflict: str,
) -> None:
    trial_dir = next(path for path in synthetic_oracle_job.iterdir() if path.is_dir())
    if conflict == "job-lock-source":
        path = synthetic_oracle_job / "lock.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["trials"][0]["task"]["source"] = "terminal-bench/other-dataset"
    elif conflict == "result-source":
        path = trial_dir / "result.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["source"] = "terminal-bench/other-dataset"
    elif conflict == "result-config-ref":
        path = trial_dir / "result.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["config"]["task"]["ref"] = "sha256:" + "8" * 64
    else:
        path = synthetic_oracle_job / "lock.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["trials"][0]["task"]["digest"] = "sha256:" + "9" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="conflicting"):
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


def _write_cached_task(
    package_cache_dir: Path,
    *,
    docker_image: str | None,
    dockerfile: str | None = None,
) -> Path:
    task_dir = (
        package_cache_dir
        / "terminal-bench"
        / "break-filter-js-from-html"
        / FIXED_TASK_CHECKSUM.removeprefix("sha256:")
    )
    environment_dir = task_dir / "environment"
    environment_dir.mkdir(parents=True)
    config = TaskDefinitionConfig(
        environment=TaskEnvironmentConfig(docker_image=docker_image)
    )
    (task_dir / "task.toml").write_text(
        config.model_dump_toml(), encoding="utf-8"
    )
    if dockerfile is not None:
        (environment_dir / "Dockerfile").write_text(dockerfile, encoding="utf-8")
    return task_dir


def test_runtime_discovery_resolves_prebuilt_image_from_cached_package_task(
    synthetic_oracle_job: Path,
    tmp_path: Path,
) -> None:
    package_cache = tmp_path / "packages"
    _write_cached_task(package_cache, docker_image=FIXED_CONTAINER_IMAGE)

    assert write_benchmark_lock.discover_container_image(
        job_dir=synthetic_oracle_job,
        package_cache_dir=package_cache,
    ) == FIXED_CONTAINER_IMAGE


def test_runtime_discovery_derives_harbor_content_addressed_build_tag(
    synthetic_oracle_job: Path,
    tmp_path: Path,
) -> None:
    package_cache = tmp_path / "packages"
    task_dir = _write_cached_task(
        package_cache,
        docker_image=None,
        dockerfile="FROM alpine:3.20\nWORKDIR /app\n",
    )

    expected_hash = environment_content_hash(task_dir / "environment")
    assert write_benchmark_lock.discover_container_image(
        job_dir=synthetic_oracle_job,
        package_cache_dir=package_cache,
    ) == f"hb__{expected_hash}"


def test_runtime_discovery_selects_digest_for_the_exact_cached_task_image(
    synthetic_oracle_job: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], Path | None]] = []
    matching_digest = "sha256:" + "a" * 64
    unrelated_digest = "sha256:" + "b" * 64
    package_cache = tmp_path / "packages"
    _write_cached_task(package_cache, docker_image="registry.example/task:locked")

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
    monkeypatch.setattr(
        write_benchmark_lock,
        "collect_repository_git_identity",
        lambda root: RepositoryGitIdentity(
            root_git_sha="root-sha",
            adapter_tree_sha="adapter-sha",
            adapter_dirty=False,
        ),
    )
    monkeypatch.setattr(write_benchmark_lock.subprocess, "run", fake_run)

    identity = collect_runtime_identity(
        repository_root=tmp_path,
        job_dir=synthetic_oracle_job,
        package_cache_dir=package_cache,
    )

    assert identity == RuntimeIdentity(
        harbor_version="0.18.0",
        root_git_sha="root-sha",
        adapter_tree_sha="adapter-sha",
        container_image="registry.example/task:locked",
        image_digest=matching_digest,
    )
    assert calls == [
        (["docker", "image", "inspect", "registry.example/task:locked"], None),
    ]


def test_lock_writer_runtime_identity_rejects_dirty_adapter_source(
    synthetic_oracle_job: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(write_benchmark_lock, "version", lambda name: "0.18.0")
    monkeypatch.setattr(
        write_benchmark_lock,
        "collect_repository_git_identity",
        lambda root: RepositoryGitIdentity(
            root_git_sha="root-sha",
            adapter_tree_sha="adapter-sha",
            adapter_dirty=True,
        ),
    )
    monkeypatch.setattr(
        write_benchmark_lock,
        "discover_container_image",
        lambda **kwargs: pytest.fail("dirty source must fail before image discovery"),
    )
    monkeypatch.setattr(
        write_benchmark_lock,
        "_docker_image_digest",
        lambda image: pytest.fail("dirty source must fail before Docker inspection"),
    )

    with pytest.raises(ValueError, match="dirty"):
        collect_runtime_identity(
            repository_root=tmp_path,
            job_dir=synthetic_oracle_job,
        )


@pytest.mark.parametrize("image_id_matches", [False, True])
def test_digest_pinned_image_rejects_docker_identity_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    image_id_matches: bool,
) -> None:
    requested = "sha256:" + "a" * 64
    different = "sha256:" + "b" * 64
    image_id = requested if image_id_matches else "sha256:" + "c" * 64

    monkeypatch.setattr(
        write_benchmark_lock,
        "_run",
        lambda *args, **kwargs: json.dumps(
            [
                {
                    "Id": image_id,
                    "RepoDigests": [f"registry.example/task@{different}"],
                }
            ]
        ),
    )

    with pytest.raises(ValueError, match="requested sha256 digest"):
        write_benchmark_lock._docker_image_digest(
            f"registry.example/task@{requested}"
        )


def test_docker_digest_never_selects_an_unrelated_repository_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unrelated = "sha256:" + "b" * 64
    image_id = "sha256:" + "c" * 64
    monkeypatch.setattr(
        write_benchmark_lock,
        "_run",
        lambda *args, **kwargs: json.dumps(
            [
                {
                    "Id": image_id,
                    "RepoDigests": [f"registry.example/unrelated@{unrelated}"],
                }
            ]
        ),
    )

    assert (
        write_benchmark_lock._docker_image_digest("registry.example/task:locked")
        == image_id
    )


def _write_smoke_job(
    root: Path,
    *,
    reward: float | None,
    trace: str,
    probe: ProbeDesign,
    execution_context: ProbeExecutionBrief,
) -> Path:
    job_dir = root / f"job-{trace}"
    trial_dir = job_dir / "break-filter-js-from-html__bayesprobe"
    exception_type = None
    exception_message = None
    if trace == "infrastructure":
        exception_type = "DockerImageBuildError"
        exception_message = "environment failed"
    elif trace in {"provider", "agent_failure"}:
        exception_type = "BayesProbeHarborAgentError"
        exception_message = "agent execution failed"
    write_harbor_job_artifacts(
        job_dir,
        trial_dir,
        agent_name="bayesprobe-terminal-bench",
        reward=reward,
        exception_type=exception_type,
        exception_message=exception_message,
    )
    bayesprobe_dir = trial_dir / "agent" / "bayesprobe"
    bayesprobe_dir.mkdir(parents=True)

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

    output = "/workspace\n"
    observation = ActionObservation(
        action_index=1,
        action=ShellAction(command="pwd"),
        stdout=output,
        return_code=0,
        duration_ms=4,
        pre_environment_state_id="env:0",
        post_environment_state_id="env:0",
        full_output_sha256=hashlib.sha256(output.encode("utf-8")).hexdigest(),
        model_facing_output=output,
    )
    signal = signal_from_observation(
        observation=observation,
        probe=probe,
        context=execution_context,
    )
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
        json.dumps(observation.model_dump(mode="json"))
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
                "run_id": execution_context.run_id,
                "cycle_index": 1,
                "boundary_status": "integrated",
                "completed_at": "2026-07-14T00:00:30Z",
            },
        },
        {
            "record_type": "probe_set",
            "payload": {
                "cycle_id": "cycle_1",
                "probe_set_id": "PS1",
                "probes": [probe.model_dump(mode="json")],
            },
        },
        {
            "record_type": "external_signal",
            "payload": signal.model_dump(mode="json"),
        },
        {
            "record_type": "evidence_event",
            "payload": {
                "id": "E1",
                "derived_from_signal": signal.id,
                "epistemic_origin": "tool_result",
                "derivation_root_id": signal.provenance.derivation_root_id,
                "discard_reason": None,
            },
        },
        {
            "record_type": "evidence_contribution_delta",
            "payload": {
                "contribution_root_id": "evidence-root:sha256:" + "1" * 64,
                "caused_by_event_ids": ["E1"],
            },
        },
        {
            "record_type": "belief_update",
            "payload": {
                "cycle_id": "cycle_1",
                "evidence_id": "evidence-root:sha256:" + "1" * 64,
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
            "payload": {
                "run_id": execution_context.run_id,
                "status": "completed",
                "current_cycle_id": "cycle_1",
            },
        },
    ]
    if trace == "incomplete":
        records = [record for record in records if record["record_type"] != "evidence_event"]
    (bayesprobe_dir / "bayesprobe_ledger.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return job_dir


def _write_complete_benchmark_lock(
    path: Path,
    *,
    oracle_job: Path,
) -> dict[str, object]:
    lock = build_lock(
        job_dir=oracle_job,
        config=TerminalBenchConfig(model="test-model"),
        runtime_identity=RUNTIME_IDENTITY,
    )
    write_lock_atomic(path, lock)
    return lock


def _trace_dir(job_dir: Path) -> Path:
    trial_dir = next(path for path in job_dir.iterdir() if path.is_dir())
    return trial_dir / "agent" / "bayesprobe"


def _read_trace_records(bayesprobe_dir: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (bayesprobe_dir / "bayesprobe_ledger.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]


def _write_trace_records(
    bayesprobe_dir: Path,
    records: list[dict[str, object]],
) -> None:
    (bayesprobe_dir / "bayesprobe_ledger.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


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
    synthetic_oracle_job: Path,
    probe: ProbeDesign,
    execution_context: ProbeExecutionBrief,
    reward: float | None,
    trace: str,
    classification: str,
    exit_code: int,
) -> None:
    job_dir = _write_smoke_job(
        tmp_path,
        reward=reward,
        trace=trace,
        probe=probe,
        execution_context=execution_context,
    )
    lock_path = tmp_path / "benchmark.lock.json"
    _write_complete_benchmark_lock(
        lock_path,
        oracle_job=synthetic_oracle_job,
    )

    assert _classify(job_dir=job_dir, lock_path=lock_path) == classification
    assert validate_main(
        ["--job", str(job_dir), "--lock", str(lock_path)],
        runtime_identity_loader=lambda **kwargs: RUNTIME_IDENTITY,
    ) == exit_code
    assert capsys.readouterr().out == classification + "\n"


def test_smoke_classifier_rejects_partial_lock(
    tmp_path: Path,
    probe: ProbeDesign,
    execution_context: ProbeExecutionBrief,
) -> None:
    job_dir = _write_smoke_job(
        tmp_path,
        reward=1.0,
        trace="complete",
        probe=probe,
        execution_context=execution_context,
    )
    lock_path = tmp_path / "partial.lock.json"
    lock_path.write_text(
        json.dumps({"task_id": FIXED_TASK, "task_checksum": FIXED_TASK_CHECKSUM}),
        encoding="utf-8",
    )

    assert (
        _classify(job_dir=job_dir, lock_path=lock_path)
        == "conformance_error"
    )


def test_smoke_classifier_accepts_public_core_normalized_adapter_signal(
    tmp_path: Path,
    synthetic_oracle_job: Path,
    probe: ProbeDesign,
    execution_context: ProbeExecutionBrief,
) -> None:
    job_dir = _write_smoke_job(
        tmp_path,
        reward=1.0,
        trace="complete",
        probe=probe,
        execution_context=execution_context,
    )
    bayesprobe_dir = _trace_dir(job_dir)
    records = _read_trace_records(bayesprobe_dir)
    signal_record = next(
        record
        for record in records
        if record["record_type"] == "external_signal"
    )
    adapter_signal = ExternalSignal.model_validate(signal_record["payload"])
    normalized_signal = SignalProvenanceNormalizer().normalize(
        adapter_signal,
        run_id=execution_context.run_id,
    )
    assert adapter_signal.provenance is not None
    assert normalized_signal.provenance is not None
    canonical_content = " ".join(
        unicodedata.normalize("NFKC", normalized_signal.raw_content).split()
    )
    expected_fingerprint = "sha256:" + hashlib.sha256(
        (
            f"{normalized_signal.provenance.source_identity}\n"
            f"{canonical_content}"
        ).encode("utf-8")
    ).hexdigest()
    assert (
        normalized_signal.provenance.canonical_content_fingerprint
        == expected_fingerprint
    )
    assert normalized_signal.id == adapter_signal.id
    assert (
        normalized_signal.provenance.derivation_root_id
        == adapter_signal.provenance.derivation_root_id
    )
    signal_record["payload"] = normalized_signal.model_dump(mode="json")
    _write_trace_records(bayesprobe_dir, records)
    lock_path = tmp_path / "benchmark.lock.json"
    _write_complete_benchmark_lock(lock_path, oracle_job=synthetic_oracle_job)

    assert (
        _classify(job_dir=job_dir, lock_path=lock_path)
        == "engineering_pass"
    )


@pytest.mark.parametrize("evidence_outcome", ["admitted", "discarded", "neutral"])
def test_smoke_classifier_allows_linked_evidence_without_directional_update(
    tmp_path: Path,
    synthetic_oracle_job: Path,
    probe: ProbeDesign,
    execution_context: ProbeExecutionBrief,
    evidence_outcome: str,
) -> None:
    job_dir = _write_smoke_job(
        tmp_path,
        reward=1.0,
        trace="complete",
        probe=probe,
        execution_context=execution_context,
    )
    bayesprobe_dir = _trace_dir(job_dir)
    records = _read_trace_records(bayesprobe_dir)
    evidence = next(
        record["payload"]
        for record in records
        if record["record_type"] == "evidence_event"
    )
    if evidence_outcome == "discarded":
        evidence["discard_reason"] = "duplicate_exact"
    if evidence_outcome == "neutral":
        update = next(
            record["payload"]
            for record in records
            if record["record_type"] == "belief_update"
        )
        update["direction"] = "neutral"
        update["posterior"] = update["prior"]
    else:
        records = [
            record
            for record in records
            if record["record_type"]
            not in {"belief_update", "evidence_contribution_delta"}
        ]
    _write_trace_records(bayesprobe_dir, records)
    lock_path = tmp_path / "benchmark.lock.json"
    _write_complete_benchmark_lock(lock_path, oracle_job=synthetic_oracle_job)

    assert (
        _classify(job_dir=job_dir, lock_path=lock_path)
        == "engineering_pass"
    )


def test_smoke_classifier_allows_a_completed_no_signal_no_update_cycle(
    tmp_path: Path,
    synthetic_oracle_job: Path,
    probe: ProbeDesign,
    execution_context: ProbeExecutionBrief,
) -> None:
    job_dir = _write_smoke_job(
        tmp_path,
        reward=1.0,
        trace="complete",
        probe=probe,
        execution_context=execution_context,
    )
    bayesprobe_dir = _trace_dir(job_dir)
    summary_path = bayesprobe_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["bayesprobe_cycles"] = 2
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    records = _read_trace_records(bayesprobe_dir)
    run = next(
        record["payload"] for record in records if record["record_type"] == "run"
    )
    run["current_cycle_id"] = "cycle_2"
    records[-1:-1] = [
        {
            "record_type": "cycle",
            "payload": {
                "cycle_id": "cycle_2",
                "run_id": execution_context.run_id,
                "cycle_index": 2,
                "boundary_status": "integrated",
                "completed_at": "2026-07-14T00:00:31Z",
            },
        },
        {
            "record_type": "probe_set",
            "payload": {
                "cycle_id": "cycle_2",
                "probe_set_id": "PS2",
                "probes": [],
                "may_be_empty": True,
            },
        },
        {
            "record_type": "belief_state",
            "payload": {"cycle_id": "cycle_2", "belief_state_id": "B2"},
        },
    ]
    _write_trace_records(bayesprobe_dir, records)
    lock_path = tmp_path / "benchmark.lock.json"
    _write_complete_benchmark_lock(lock_path, oracle_job=synthetic_oracle_job)

    assert (
        _classify(job_dir=job_dir, lock_path=lock_path)
        == "engineering_pass"
    )


def test_policy_denied_reserved_action_is_reconciled_without_an_observation(
    tmp_path: Path,
    synthetic_oracle_job: Path,
    probe: ProbeDesign,
    execution_context: ProbeExecutionBrief,
) -> None:
    job_dir = _write_smoke_job(
        tmp_path,
        reward=1.0,
        trace="complete",
        probe=probe,
        execution_context=execution_context,
    )
    bayesprobe_dir = _trace_dir(job_dir)
    summary_path = bayesprobe_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["terminal_actions"] = 2
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    (bayesprobe_dir / "errors.jsonl").write_text(
        json.dumps(
            {
                "action_index": 2,
                "category": "policy_error",
                "error_type": "PolicyViolation",
                "probe_id": probe.id,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    lock_path = tmp_path / "benchmark.lock.json"
    _write_complete_benchmark_lock(lock_path, oracle_job=synthetic_oracle_job)

    assert (
        _classify(job_dir=job_dir, lock_path=lock_path)
        == "engineering_pass"
    )


def test_all_policy_denied_cycle_needs_no_observation_signal_or_update(
    tmp_path: Path,
    synthetic_oracle_job: Path,
    probe: ProbeDesign,
    execution_context: ProbeExecutionBrief,
) -> None:
    job_dir = _write_smoke_job(
        tmp_path,
        reward=1.0,
        trace="complete",
        probe=probe,
        execution_context=execution_context,
    )
    bayesprobe_dir = _trace_dir(job_dir)
    (bayesprobe_dir / "environment_actions.jsonl").unlink()
    records = [
        record
        for record in _read_trace_records(bayesprobe_dir)
        if record["record_type"]
        not in {
            "external_signal",
            "evidence_event",
            "evidence_contribution_delta",
            "belief_update",
        }
    ]
    _write_trace_records(bayesprobe_dir, records)
    errors_path = bayesprobe_dir / "errors.jsonl"
    errors_path.write_text(
        json.dumps(
            {
                "action_index": 1,
                "category": "policy_error",
                "error_type": "PolicyViolation",
                "probe_id": probe.id,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    lock_path = tmp_path / "benchmark.lock.json"
    _write_complete_benchmark_lock(lock_path, oracle_job=synthetic_oracle_job)

    assert (
        _classify(job_dir=job_dir, lock_path=lock_path)
        == "engineering_pass"
    )

    errors_path.unlink()
    assert (
        _classify(job_dir=job_dir, lock_path=lock_path)
        == "conformance_error"
    )


@pytest.mark.parametrize("orphan", ["signal", "evidence", "update"])
def test_smoke_classifier_rejects_orphan_epistemic_rows(
    tmp_path: Path,
    synthetic_oracle_job: Path,
    probe: ProbeDesign,
    execution_context: ProbeExecutionBrief,
    orphan: str,
) -> None:
    job_dir = _write_smoke_job(
        tmp_path,
        reward=1.0,
        trace="complete",
        probe=probe,
        execution_context=execution_context,
    )
    bayesprobe_dir = _trace_dir(job_dir)
    records = _read_trace_records(bayesprobe_dir)
    if orphan == "signal":
        signal = deepcopy(
            next(
                record["payload"]
                for record in records
                if record["record_type"] == "external_signal"
            )
        )
        signal["id"] = "S_orphan"
        signal["cycle_id"] = "cycle_orphan"
        records.append({"record_type": "external_signal", "payload": signal})
    elif orphan == "evidence":
        evidence = deepcopy(
            next(
                record["payload"]
                for record in records
                if record["record_type"] == "evidence_event"
            )
        )
        evidence["id"] = "E_orphan"
        evidence["derived_from_signal"] = "S_missing"
        records.append({"record_type": "evidence_event", "payload": evidence})
    else:
        update = deepcopy(
            next(
                record["payload"]
                for record in records
                if record["record_type"] == "belief_update"
            )
        )
        update["update_id"] = "U_orphan"
        update["evidence_id"] = "E_missing"
        update["sensitivity"] = {"caused_by_event_ids": ["E_missing"]}
        records.append({"record_type": "belief_update", "payload": update})
    _write_trace_records(bayesprobe_dir, records)
    lock_path = tmp_path / "benchmark.lock.json"
    _write_complete_benchmark_lock(lock_path, oracle_job=synthetic_oracle_job)

    assert (
        _classify(job_dir=job_dir, lock_path=lock_path)
        == "conformance_error"
    )


@pytest.mark.parametrize(
    "conflict",
    [
        "missing-task-name",
        "task-name",
        "task-id-ref",
        "source",
        "task-checksum",
        "result-config-ref",
    ],
)
def test_smoke_classifier_rejects_missing_or_conflicting_result_identity(
    tmp_path: Path,
    synthetic_oracle_job: Path,
    probe: ProbeDesign,
    execution_context: ProbeExecutionBrief,
    conflict: str,
) -> None:
    job_dir = _write_smoke_job(
        tmp_path,
        reward=1.0,
        trace="complete",
        probe=probe,
        execution_context=execution_context,
    )
    trial_dir = next(path for path in job_dir.iterdir() if path.is_dir())
    result_path = trial_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    other_digest = "sha256:" + "9" * 64
    if conflict == "missing-task-name":
        result.pop("task_name")
    elif conflict == "task-name":
        result["task_name"] = "a-different-task"
    elif conflict == "task-id-ref":
        result["task_id"]["ref"] = other_digest
    elif conflict == "source":
        result["source"] = "terminal-bench/other-dataset"
    elif conflict == "task-checksum":
        result["task_checksum"] = other_digest
    else:
        result["config"]["task"]["ref"] = other_digest
    result_path.write_text(json.dumps(result), encoding="utf-8")
    lock_path = tmp_path / "benchmark.lock.json"
    _write_complete_benchmark_lock(lock_path, oracle_job=synthetic_oracle_job)

    assert (
        _classify(job_dir=job_dir, lock_path=lock_path)
        == "conformance_error"
    )


@pytest.mark.parametrize("stale_field", ["task_checksum", "dataset_revision"])
def test_smoke_classifier_rejects_stale_complete_lock_identity(
    tmp_path: Path,
    synthetic_oracle_job: Path,
    probe: ProbeDesign,
    execution_context: ProbeExecutionBrief,
    stale_field: str,
) -> None:
    job_dir = _write_smoke_job(
        tmp_path,
        reward=1.0,
        trace="complete",
        probe=probe,
        execution_context=execution_context,
    )
    lock_path = tmp_path / "benchmark.lock.json"
    lock = _write_complete_benchmark_lock(
        lock_path,
        oracle_job=synthetic_oracle_job,
    )
    lock[stale_field] = "sha256:" + "9" * 64
    write_lock_atomic(lock_path, lock)

    assert (
        _classify(job_dir=job_dir, lock_path=lock_path)
        == "conformance_error"
    )


@pytest.mark.parametrize(
    ("stale_field", "stale_value"),
    [
        ("root_git_sha", "9" * 40),
        ("adapter_tree_sha", "8" * 40),
        ("container_image", "registry.example/terminal-bench/stale:locked"),
        ("image_digest", "sha256:" + "7" * 64),
    ],
)
def test_smoke_classifier_rejects_well_formed_stale_runtime_lock(
    tmp_path: Path,
    synthetic_oracle_job: Path,
    probe: ProbeDesign,
    execution_context: ProbeExecutionBrief,
    stale_field: str,
    stale_value: str,
) -> None:
    job_dir = _write_smoke_job(
        tmp_path,
        reward=1.0,
        trace="complete",
        probe=probe,
        execution_context=execution_context,
    )
    lock_path = tmp_path / "benchmark.lock.json"
    lock = _write_complete_benchmark_lock(
        lock_path,
        oracle_job=synthetic_oracle_job,
    )
    lock[stale_field] = stale_value
    write_lock_atomic(lock_path, lock)

    assert (
        _classify(job_dir=job_dir, lock_path=lock_path)
        == "conformance_error"
    )


@pytest.mark.parametrize(
    "broken_link",
    [
        "arbitrary-root",
        "fingerprint",
        "no-matching-action",
        "request-mismatch",
        "environment-state",
        "action-index",
        "cycle-run",
        "missing-evidence-root",
        "missing-direction",
        "neutral-direction",
        "unknown-direction",
        "inconsistent-direction",
    ],
)
def test_smoke_classifier_rejects_false_action_provenance_links(
    tmp_path: Path,
    synthetic_oracle_job: Path,
    probe: ProbeDesign,
    execution_context: ProbeExecutionBrief,
    broken_link: str,
) -> None:
    job_dir = _write_smoke_job(
        tmp_path,
        reward=1.0,
        trace="complete",
        probe=probe,
        execution_context=execution_context,
    )
    trial_dir = next(path for path in job_dir.iterdir() if path.is_dir())
    bayesprobe_dir = trial_dir / "agent" / "bayesprobe"
    actions_path = bayesprobe_dir / "environment_actions.jsonl"
    ledger_path = bayesprobe_dir / "bayesprobe_ledger.jsonl"
    actions = [
        json.loads(line)
        for line in actions_path.read_text(encoding="utf-8").splitlines()
    ]
    records = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
    ]
    signal = next(
        record["payload"]
        for record in records
        if record["record_type"] == "external_signal"
    )
    if broken_link == "arbitrary-root":
        signal["provenance"]["derivation_root_id"] = (
            "harbor-action:sha256:" + "f" * 64
        )
    elif broken_link == "fingerprint":
        signal["provenance"]["canonical_content_fingerprint"] = (
            "sha256:" + "f" * 64
        )
    elif broken_link == "no-matching-action":
        actions[0]["action_index"] = 2
    elif broken_link == "request-mismatch":
        actions[0]["action"]["command"] = "printf different request"
    elif broken_link == "environment-state":
        signal["provenance"]["environment_state_id"] = "env:other"
    elif broken_link == "action-index":
        signal["provenance"]["artifact_refs"] = [
            "environment_actions.jsonl#2"
        ]
    elif broken_link == "cycle-run":
        cycle = next(
            record["payload"]
            for record in records
            if record["record_type"] == "cycle"
        )
        cycle["run_id"] = "run_other"
    else:
        evidence = next(
            record["payload"]
            for record in records
            if record["record_type"] == "evidence_event"
        )
        update = next(
            record["payload"]
            for record in records
            if record["record_type"] == "belief_update"
        )
        if broken_link == "missing-evidence-root":
            evidence.pop("derivation_root_id")
        elif broken_link == "missing-direction":
            update.pop("direction")
        elif broken_link == "neutral-direction":
            update["direction"] = "neutral"
        elif broken_link == "unknown-direction":
            update["direction"] = "sideways"
        else:
            update["direction"] = "weakened"
    actions_path.write_text(
        "".join(json.dumps(action) + "\n" for action in actions),
        encoding="utf-8",
    )
    ledger_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    lock_path = tmp_path / "benchmark.lock.json"
    _write_complete_benchmark_lock(lock_path, oracle_job=synthetic_oracle_job)

    assert (
        _classify(job_dir=job_dir, lock_path=lock_path)
        == "conformance_error"
    )
