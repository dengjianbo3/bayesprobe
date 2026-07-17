from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import validate_causal_qualification
from bayesprobe_terminal_bench.config import TerminalBenchConfig
from bayesprobe_terminal_bench.experiment_lock import (
    FROZEN_GATE_TASK_IDS,
    FROZEN_GATE_TASK_REFS,
)
from bayesprobe_terminal_bench.planning import plan_contract_identity
from bayesprobe_terminal_bench.provider_contract import contract_identity
from capture_provider_identity import (
    capture_provider_identity,
    load_provider_identity_artifact,
    write_provider_identity_artifact,
)
from validate_causal_qualification import (
    replay_offline_gate,
    retry_eligible,
    validate_causal_qualification_job,
)
from write_causal_qualification_lock import (
    CachedQualificationTask,
    CausalQualificationRuntimeIdentity,
    build_causal_qualification_lock,
)


FIXTURES = Path(__file__).parent / "fixtures"
HISTORICAL_FIXTURES = FIXTURES / "historical_traces"
CAUSAL_FIXTURES = FIXTURES / "causal_traces"
CONFORMANT_FIXTURE = CAUSAL_FIXTURES / "conformant-inspect-intervene-verify"
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_TASK_TIMEOUTS = {
    FROZEN_GATE_TASK_IDS[0]: 1200,
    FROZEN_GATE_TASK_IDS[1]: 900,
    FROZEN_GATE_TASK_IDS[2]: 900,
}


def _config() -> TerminalBenchConfig:
    return TerminalBenchConfig(
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _canonical_sha256(payload: object) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return f"sha256:{hashlib.sha256(serialized.encode('utf-8')).hexdigest()}"


def _oracle_job(root: Path, *, failed_task: str | None = None) -> Path:
    _write_json(
        root / "config.json",
        {
            "n_attempts": 1,
            "agents": [{"name": "oracle"}],
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
                    },
                    "agent": {"name": "oracle"},
                }
                for task_id in reversed(FROZEN_GATE_TASK_IDS)
            ],
        },
    )
    for task_id in FROZEN_GATE_TASK_IDS:
        slug = task_id.split("/", 1)[1]
        _write_json(
            root / f"{slug}__oracle" / "result.json",
            {
                "task_name": slug,
                "task_id": {
                    "org": "terminal-bench",
                    "name": slug,
                    "ref": FROZEN_GATE_TASK_REFS[task_id],
                },
                "config": {"agent": {"name": "oracle"}},
                "agent_info": {"name": "oracle"},
                "verifier_result": {
                    "rewards": {
                        "reward": 0.0 if task_id == failed_task else 1.0,
                    }
                },
                "exception_info": None,
                "finished_at": "2026-07-17T00:00:00Z",
            },
        )
    return root


class _FakeCompletions:
    def __init__(self, response: object) -> None:
        self.response = response
        self.requests: list[dict[str, object]] = []

    def create(self, **request: object) -> object:
        self.requests.append(dict(request))
        return self.response


class _FakeClient:
    def __init__(self, response: object) -> None:
        self.completions = _FakeCompletions(response)
        self.chat = SimpleNamespace(completions=self.completions)


def _identity_response(*, fingerprint: str | None = "fixture-fingerprint-v1") -> object:
    payload: dict[str, object] = {
        "model": "fixture-model-v1",
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 1,
            "total_tokens": 6,
        },
    }
    if fingerprint is not None:
        payload["system_fingerprint"] = fingerprint
    return payload


def _provider_identity_path(tmp_path: Path) -> Path:
    artifact = capture_provider_identity(
        client=_FakeClient(_identity_response()),
        model=_config().model,
        base_url=_config().base_url,
    )
    return write_provider_identity_artifact(tmp_path / "identity", artifact)


def _runtime(*, dirty: bool = False) -> CausalQualificationRuntimeIdentity:
    return CausalQualificationRuntimeIdentity(
        harbor_version="0.18.0",
        root_git_sha="a" * 40,
        adapter_tree_sha="b" * 40,
        adapter_dirty=dirty,
    )


def _cached_task(task_id: str, task_ref: str) -> CachedQualificationTask:
    del task_ref
    index = FROZEN_GATE_TASK_IDS.index(task_id) + 2
    return CachedQualificationTask(
        image_digest="sha256:" + str(index) * 64,
        agent_timeout_seconds=_TASK_TIMEOUTS[task_id],
    )


def _lock_payload(tmp_path: Path) -> dict[str, object]:
    return build_causal_qualification_lock(
        job_dir=_oracle_job(tmp_path / "oracle"),
        config=_config(),
        runtime_identity=_runtime(),
        provider_identity_path=_provider_identity_path(tmp_path),
        task_identity_resolver=_cached_task,
    )


def _write_lock(tmp_path: Path) -> Path:
    path = tmp_path / "causal-qualification.lock.json"
    _write_json(path, _lock_payload(tmp_path))
    return path


def _live_jobs(
    root: Path,
    *,
    lock_path: Path,
    rewards: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[Path, Path, Path]:
    runtime_lock = json.loads(lock_path.read_text(encoding="utf-8"))
    runtime_lock_sha256 = _canonical_sha256(runtime_lock)
    jobs: list[Path] = []
    for index, task_id in enumerate(FROZEN_GATE_TASK_IDS):
        slug = task_id.split("/", 1)[1]
        job = root / f"{slug}-job"
        jobs.append(job)
        agent_timeout = _TASK_TIMEOUTS[task_id]
        agent = {
            "import_path": "bayesprobe_terminal_bench.agent:BayesProbeHarborAgent",
            "model_name": "deepseek-v4-flash",
            "env": {
                "BAYESPROBE_BENCH_BASE_URL": "https://api.deepseek.com",
                "BAYESPROBE_BENCH_LOCK_PATH": ".runs/causal-qualification.lock.json",
                "BAYESPROBE_BENCH_MODEL": "deepseek-v4-flash",
                "BAYESPROBE_BENCH_TASK_TIMEOUT_SECONDS": str(agent_timeout),
            },
        }
        _write_json(
            job / "config.json",
            {
                "job_name": f"bayesprobe-causal-qualification-{slug}",
                "n_attempts": 1,
                "n_concurrent_trials": 1,
                "retry": {"max_retries": 0},
                "agents": [agent],
                "datasets": [
                    {
                        "name": "terminal-bench/terminal-bench-2",
                        "ref": "sha256:" + "1" * 64,
                        "task_names": [task_id],
                    }
                ],
            },
        )
        _write_json(
            job / "lock.json",
            {
                "harbor": {"version": "0.18.0"},
                "n_concurrent_trials": 1,
                "retry": {"max_retries": 0},
                "trials": [
                    {
                        "task": {
                            "name": task_id,
                            "digest": FROZEN_GATE_TASK_REFS[task_id],
                            "source": "terminal-bench/terminal-bench-2",
                        },
                        "agent": agent,
                    }
                ],
            },
        )
        trial = job / f"{slug}__qualification"
        shutil.copytree(CONFORMANT_FIXTURE, trial / "agent" / "bayesprobe")
        summary_path = trial / "agent" / "bayesprobe" / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["runtime_lock_sha256"] = runtime_lock_sha256
        summary["runtime_budgets"] = {
            "max_total_actions": 24,
            "max_model_calls": 72,
            "max_provider_tokens": 160000,
            "max_output_tokens": 8192,
            "command_timeout_seconds": 120,
            "provider_timeout_seconds": 360,
            "signal_output_bytes": 32768,
            "provider_tokens_used": 60,
        }
        _write_json(summary_path, summary)
        _write_json(
            trial / "result.json",
            {
                "task_name": slug,
                "task_id": {
                    "org": "terminal-bench",
                    "name": slug,
                    "ref": FROZEN_GATE_TASK_REFS[task_id],
                },
                "config": {
                    "agent": {
                        "import_path": (
                            "bayesprobe_terminal_bench.agent:BayesProbeHarborAgent"
                        )
                    }
                },
                "verifier_result": {"rewards": {"reward": rewards[index]}},
                "exception_info": None,
                "started_at": "2026-07-17T00:00:00Z",
                "finished_at": "2026-07-17T00:01:00Z",
            },
        )
    return jobs[0], jobs[1], jobs[2]


def _replace_task_artifacts(job: Path, task_id: str, fixture: Path) -> Path:
    slug = task_id.split("/", 1)[1]
    artifact_dir = job / f"{slug}__qualification" / "agent" / "bayesprobe"
    shutil.rmtree(artifact_dir)
    shutil.copytree(fixture, artifact_dir)
    return artifact_dir


def test_contract_identities_are_versioned_canonical_hashes() -> None:
    identities = {**contract_identity(), **plan_contract_identity()}

    assert set(identities) == {
        "terminal_task_frame:v1:prompt",
        "terminal_task_frame:v1:schema",
        "terminal_probe_design:v1:prompt",
        "terminal_probe_design:v1:schema",
        "terminal_probe_plan:v1:prompt",
        "terminal_probe_plan:v1:repair_prompt",
        "terminal_probe_plan:v1:schema",
        "harbor-observation:v3:schema",
    }
    assert all(_SHA256.fullmatch(value) for value in identities.values())
    assert (
        identities["terminal_probe_plan:v1:prompt"]
        != identities["terminal_probe_plan:v1:repair_prompt"]
    )
    assert contract_identity() == contract_identity()
    assert plan_contract_identity() == plan_contract_identity()


@pytest.mark.parametrize(
    ("fingerprint", "available"),
    [("fixture-fingerprint-v1", True), (None, False)],
)
def test_capture_provider_identity_makes_one_minimal_structured_request(
    tmp_path: Path,
    fingerprint: str | None,
    available: bool,
) -> None:
    client = _FakeClient(_identity_response(fingerprint=fingerprint))

    artifact = capture_provider_identity(
        client=client,
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
    )
    path = write_provider_identity_artifact(
        tmp_path,
        artifact,
        restricted_values=("one-time-provider-secret",),
    )

    assert len(client.completions.requests) == 1
    request = client.completions.requests[0]
    assert request["response_format"] == {"type": "json_object"}
    assert request["temperature"] == 0
    assert request["max_tokens"] == 8
    assert artifact.returned_model == "fixture-model-v1"
    assert artifact.system_fingerprint_available is available
    assert artifact.system_fingerprint == fingerprint
    assert artifact.usage.total_tokens == 6
    assert path.name == f"{artifact.content_sha256.removeprefix('sha256:')}.json"
    assert load_provider_identity_artifact(path) == artifact
    serialized = path.read_text(encoding="utf-8").casefold()
    for forbidden in (
        "one-time-provider-secret",
        "authorization",
        "reasoning",
        "response_body",
        "messages",
    ):
        assert forbidden not in serialized


def test_provider_identity_artifact_detects_tampering(tmp_path: Path) -> None:
    path = _provider_identity_path(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["returned_model"] = "substituted-model"
    _write_json(path, payload)

    with pytest.raises(ValueError, match="content hash"):
        load_provider_identity_artifact(path)


def test_capture_provider_identity_rejects_malformed_fingerprint() -> None:
    client = _FakeClient(_identity_response(fingerprint="   "))

    with pytest.raises(ValueError, match="fingerprint"):
        capture_provider_identity(
            client=client,
            model="deepseek-v4-flash",
            base_url="https://api.deepseek.com",
        )


def test_build_causal_lock_freezes_oracle_tasks_budgets_and_provider_identity(
    tmp_path: Path,
) -> None:
    lock = _lock_payload(tmp_path)

    assert lock["schema_version"] == "terminal_bench_causal_qualification:v1"
    assert [item["task_id"] for item in lock["tasks"]] == list(
        FROZEN_GATE_TASK_IDS
    )
    assert [item["agent_timeout_seconds"] for item in lock["tasks"]] == [
        1200,
        900,
        900,
    ]
    assert lock["budgets"] == {
        "max_total_actions": 24,
        "max_model_calls": 72,
        "max_provider_tokens": 160000,
        "max_output_tokens": 8192,
        "command_timeout_seconds": 120,
        "provider_timeout_seconds": 360,
        "signal_output_bytes": 32768,
    }
    assert lock["expected_provider_model"] == "fixture-model-v1"
    assert _SHA256.fullmatch(str(lock["provider_identity_sha256"]))
    assert lock["expected_system_fingerprint_available"] is True
    assert lock["expected_system_fingerprint"] == "fixture-fingerprint-v1"
    assert lock["prompt_schema_hashes"] == {
        **contract_identity(),
        **plan_contract_identity(),
    }


def test_causal_lock_writer_rejects_oracle_failure_dirty_tree_or_missing_canary(
    tmp_path: Path,
) -> None:
    provider_identity = _provider_identity_path(tmp_path)
    kwargs = {
        "config": _config(),
        "provider_identity_path": provider_identity,
        "task_identity_resolver": _cached_task,
    }

    with pytest.raises(ValueError, match="Oracle reward must be 1"):
        build_causal_qualification_lock(
            job_dir=_oracle_job(
                tmp_path / "failed-oracle",
                failed_task=FROZEN_GATE_TASK_IDS[1],
            ),
            runtime_identity=_runtime(),
            **kwargs,
        )
    with pytest.raises(ValueError, match="dirty"):
        build_causal_qualification_lock(
            job_dir=_oracle_job(tmp_path / "clean-oracle"),
            runtime_identity=_runtime(dirty=True),
            **kwargs,
        )
    with pytest.raises(ValueError, match="provider identity artifact"):
        build_causal_qualification_lock(
            job_dir=_oracle_job(tmp_path / "missing-canary-oracle"),
            runtime_identity=_runtime(),
            config=_config(),
            provider_identity_path=tmp_path / "missing.json",
            task_identity_resolver=_cached_task,
        )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("config.json", {"agents": [{"name": "oracle"}, {"name": "other"}]}),
        ("lock.json", {"agent": {"name": "other"}}),
        (
            "break-filter-js-from-html__oracle/result.json",
            {"config": {"agent": {"name": "other"}}},
        ),
        (
            "break-filter-js-from-html__oracle/result.json",
            {"agent_info": {"name": "other"}},
        ),
    ],
)
def test_causal_lock_writer_rejects_non_oracle_provenance(
    tmp_path: Path,
    path: str,
    value: dict[str, object],
) -> None:
    job = _oracle_job(tmp_path / "oracle")
    target = job / path
    payload = json.loads(target.read_text(encoding="utf-8"))
    if path == "lock.json":
        payload["trials"][0].update(value)
    else:
        for key, replacement in value.items():
            payload[key] = replacement
    _write_json(target, payload)

    with pytest.raises(ValueError, match="Oracle agent"):
        build_causal_qualification_lock(
            job_dir=job,
            config=_config(),
            runtime_identity=_runtime(),
            provider_identity_path=_provider_identity_path(tmp_path),
            task_identity_resolver=_cached_task,
        )


@pytest.mark.parametrize(
    "finished_at",
    ["", "not-a-timestamp", "2026-07-17T00:00:00"],
)
def test_causal_lock_writer_rejects_invalid_oracle_completion_time(
    tmp_path: Path,
    finished_at: str,
) -> None:
    job = _oracle_job(tmp_path / "oracle")
    result_path = job / "break-filter-js-from-html__oracle" / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["finished_at"] = finished_at
    _write_json(result_path, result)

    with pytest.raises(ValueError, match="completion"):
        build_causal_qualification_lock(
            job_dir=job,
            config=_config(),
            runtime_identity=_runtime(),
            provider_identity_path=_provider_identity_path(tmp_path),
            task_identity_resolver=_cached_task,
        )


def test_causal_qualification_configs_freeze_identity_and_per_task_invocation() -> None:
    project = Path(__file__).resolve().parents[1]
    oracle = yaml.safe_load(
        (project / "configs" / "oracle-causal-qualification.yaml").read_text(
            encoding="utf-8"
        )
    )
    bayesprobe = yaml.safe_load(
        (project / "configs" / "bayesprobe-causal-qualification.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert tuple(oracle["datasets"][0]["task_names"]) == FROZEN_GATE_TASK_IDS
    assert tuple(bayesprobe["datasets"][0]["task_names"]) == FROZEN_GATE_TASK_IDS
    assert oracle["n_attempts"] == bayesprobe["n_attempts"] == 1
    assert oracle["orchestrator"]["n_concurrent_trials"] == 1
    assert bayesprobe["orchestrator"]["n_concurrent_trials"] == 1
    agent = bayesprobe["agents"][0]
    assert agent["model_name"] == "deepseek-v4-flash"
    assert agent["env"] == {
        "BAYESPROBE_BENCH_API_KEY": "${BAYESPROBE_BENCH_API_KEY}",
        "BAYESPROBE_BENCH_BASE_URL": "https://api.deepseek.com",
        "BAYESPROBE_BENCH_LOCK_PATH": ".runs/causal-qualification.lock.json",
        "BAYESPROBE_BENCH_MODEL": "deepseek-v4-flash",
        "BAYESPROBE_BENCH_TASK_TIMEOUT_SECONDS": (
            "${BAYESPROBE_BENCH_TASK_TIMEOUT_SECONDS}"
        ),
    }


def test_offline_gate_replays_preregistered_counts_and_conformant_fixture() -> None:
    report = replay_offline_gate(historical_fixtures=HISTORICAL_FIXTURES)

    assert report["historical_replay_passed"] is True
    assert report["historical_classification_counts"] == {
        "provider_contract_error": 2,
        "causal_conformance_error": 1,
    }
    assert report["synthetic_conformant_passed"] is True
    assert report["synthetic_classification"] == "conformant"
    assert report["offline_gate_passed"] is True
    assert "qualification_passed" not in report


def test_offline_cli_never_enters_live_job_validation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        validate_causal_qualification,
        "validate_causal_qualification_job",
        lambda **kwargs: pytest.fail("offline-only must not validate a live job"),
    )

    exit_code = validate_causal_qualification.main(
        [
            "--historical-fixtures",
            str(HISTORICAL_FIXTURES),
            "--offline-only",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["offline_gate_passed"] is True
    assert "qualification_passed" not in payload


def test_reward_zero_conformant_trials_pass_live_qualification(tmp_path: Path) -> None:
    lock_path = _write_lock(tmp_path)
    report = validate_causal_qualification_job(
        lock_path=lock_path,
        job_dirs=_live_jobs(tmp_path / "live", lock_path=lock_path),
        provider_identity_path=_provider_identity_path(tmp_path),
    )

    assert report["qualification_passed"] is True
    assert [item["reward"] for item in report["tasks"]] == [0.0, 0.0, 0.0]
    assert all(item["classification"] == "conformant" for item in report["tasks"])
    assert all(item["complete_cycles"] >= 1 for item in report["tasks"])


@pytest.mark.parametrize(
    ("case", "expected_classification", "expected_failure"),
    [
        (
            "reward_one_causal",
            "causal_conformance_error",
            "causal_conformance_error",
        ),
        (
            "provider_contract",
            "provider_contract_error",
            "provider_contract_error",
        ),
        ("missing_atif", "adapter_error", "missing_atif"),
        ("missing_verifier", "conformant", "missing_verifier"),
        ("unfinished_verifier", "conformant", "incomplete_verifier"),
        ("budget", "budget_error", "budget_exceeded"),
        (
            "provider_identity",
            "provider_contract_error",
            "provider_identity_drift",
        ),
    ],
)
def test_live_qualification_fails_independently_of_reward(
    tmp_path: Path,
    case: str,
    expected_classification: str,
    expected_failure: str,
) -> None:
    lock_path = _write_lock(tmp_path)
    jobs = _live_jobs(
        tmp_path / "live",
        lock_path=lock_path,
        rewards=(1.0, 1.0, 1.0),
    )
    task_id = FROZEN_GATE_TASK_IDS[0]
    slug = task_id.split("/", 1)[1]
    job = jobs[0]
    if case == "reward_one_causal":
        _replace_task_artifacts(
            job,
            task_id,
            CAUSAL_FIXTURES / "broken-bindings" / "discarded-update",
        )
    elif case == "provider_contract":
        _replace_task_artifacts(
            job,
            task_id,
            HISTORICAL_FIXTURES / "break-filter-js-from-html",
        )
    elif case == "missing_atif":
        (job / f"{slug}__qualification" / "agent" / "bayesprobe" / "trajectory.json").unlink()
    elif case == "missing_verifier":
        result_path = job / f"{slug}__qualification" / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["verifier_result"] = None
        _write_json(result_path, result)
    elif case == "unfinished_verifier":
        result_path = job / f"{slug}__qualification" / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["finished_at"] = None
        _write_json(result_path, result)
    elif case == "budget":
        summary_path = (
            job
            / f"{slug}__qualification"
            / "agent"
            / "bayesprobe"
            / "summary.json"
        )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["terminal_actions"] = 25
        _write_json(summary_path, summary)
    else:
        telemetry_path = (
            job
            / f"{slug}__qualification"
            / "agent"
            / "bayesprobe"
            / "provider_telemetry.jsonl"
        )
        records = [
            json.loads(line)
            for line in telemetry_path.read_text(encoding="utf-8").splitlines()
        ]
        for record in records:
            record["model"] = "substituted-provider-model"
        telemetry_path.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )

    report = validate_causal_qualification_job(
        lock_path=lock_path,
        job_dirs=jobs,
        provider_identity_path=_provider_identity_path(tmp_path),
    )

    first = report["tasks"][0]
    assert report["qualification_passed"] is False
    assert first["reward"] == (None if case == "missing_verifier" else 1.0)
    assert first["classification"] == expected_classification
    assert expected_failure in first["failures"]


def test_live_qualification_rejects_noncanonical_job_shape(tmp_path: Path) -> None:
    lock_path = _write_lock(tmp_path)
    jobs = _live_jobs(tmp_path / "live", lock_path=lock_path)
    provider_identity = _provider_identity_path(tmp_path)

    with pytest.raises(ValueError, match="exactly three"):
        validate_causal_qualification_job(
            lock_path=lock_path,
            job_dirs=jobs[:2],
            provider_identity_path=provider_identity,
        )

    duplicate = jobs[0]
    with pytest.raises(ValueError, match="duplicate"):
        validate_causal_qualification_job(
            lock_path=lock_path,
            job_dirs=(jobs[0], duplicate, jobs[2]),
            provider_identity_path=provider_identity,
        )

    missing = _live_jobs(tmp_path / "missing", lock_path=lock_path)
    shutil.rmtree(missing[1] / "cancel-async-tasks__qualification")
    with pytest.raises(ValueError, match="exactly one result"):
        validate_causal_qualification_job(
            lock_path=lock_path,
            job_dirs=missing,
            provider_identity_path=provider_identity,
        )

    multi = _live_jobs(tmp_path / "multi", lock_path=lock_path)
    shutil.copytree(
        multi[0] / "break-filter-js-from-html__qualification",
        multi[0] / "second-result",
    )
    with pytest.raises(ValueError, match="exactly one result"):
        validate_causal_qualification_job(
            lock_path=lock_path,
            job_dirs=multi,
            provider_identity_path=provider_identity,
        )

    unknown_result = json.loads(
        (jobs[1] / "cancel-async-tasks__qualification" / "result.json").read_text(
            encoding="utf-8"
        )
    )
    unknown_result["task_name"] = "terminal-bench/unknown-task"
    unknown_result["task_id"]["name"] = "unknown-task"
    _write_json(jobs[1] / "cancel-async-tasks__qualification" / "result.json", unknown_result)
    with pytest.raises(ValueError, match="unknown"):
        validate_causal_qualification_job(
            lock_path=lock_path,
            job_dirs=jobs,
            provider_identity_path=provider_identity,
        )


@pytest.mark.parametrize(
    "case",
    [
        "missing_config",
        "job_name",
        "dataset_revision",
        "harbor_version",
        "task_ref",
        "task_timeout",
        "hidden_retry",
    ],
)
def test_live_qualification_binds_harbor_job_provenance(
    tmp_path: Path,
    case: str,
) -> None:
    lock_path = _write_lock(tmp_path)
    jobs = _live_jobs(tmp_path / "live", lock_path=lock_path)
    job = jobs[0]

    if case == "missing_config":
        (job / "config.json").unlink()
    elif case in {"job_name", "dataset_revision", "task_timeout", "hidden_retry"}:
        config_path = job / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if case == "job_name":
            config["job_name"] = "unregistered-qualification-job"
        elif case == "dataset_revision":
            config["datasets"][0]["ref"] = "sha256:" + "9" * 64
        elif case == "task_timeout":
            config["agents"][0]["env"][
                "BAYESPROBE_BENCH_TASK_TIMEOUT_SECONDS"
            ] = "1199"
        else:
            config["retry"]["max_retries"] = 1
        _write_json(config_path, config)
    else:
        job_lock_path = job / "lock.json"
        job_lock = json.loads(job_lock_path.read_text(encoding="utf-8"))
        if case == "harbor_version":
            job_lock["harbor"]["version"] = "0.17.0"
        else:
            job_lock["trials"][0]["task"]["digest"] = "sha256:" + "9" * 64
        _write_json(job_lock_path, job_lock)

    with pytest.raises(ValueError, match="job provenance"):
        validate_causal_qualification_job(
            lock_path=lock_path,
            job_dirs=jobs,
            provider_identity_path=_provider_identity_path(tmp_path),
        )


def test_live_qualification_binds_runtime_lock_hash(tmp_path: Path) -> None:
    lock_path = _write_lock(tmp_path)
    jobs = _live_jobs(tmp_path / "live", lock_path=lock_path)
    summary_path = (
        jobs[0]
        / "break-filter-js-from-html__qualification"
        / "agent"
        / "bayesprobe"
        / "summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["runtime_lock_sha256"] = "sha256:" + "9" * 64
    _write_json(summary_path, summary)

    report = validate_causal_qualification_job(
        lock_path=lock_path,
        job_dirs=jobs,
        provider_identity_path=_provider_identity_path(tmp_path),
    )

    assert report["qualification_passed"] is False
    assert "runtime_lock_drift" in report["tasks"][0]["failures"]


def test_live_qualification_rejects_provider_artifact_and_budget_drift(
    tmp_path: Path,
) -> None:
    lock_path = _write_lock(tmp_path)
    jobs = _live_jobs(tmp_path / "live", lock_path=lock_path)
    provider_identity = _provider_identity_path(tmp_path)
    summary_path = jobs[0] / "break-filter-js-from-html__qualification" / "agent" / "bayesprobe" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["runtime_budgets"]["command_timeout_seconds"] = 119
    _write_json(summary_path, summary)

    report = validate_causal_qualification_job(
        lock_path=lock_path,
        job_dirs=jobs,
        provider_identity_path=provider_identity,
    )
    assert report["qualification_passed"] is False
    assert "runtime_budget_drift" in report["tasks"][0]["failures"]

    summary["runtime_budgets"]["command_timeout_seconds"] = 120
    summary["runtime_budgets"]["provider_tokens_used"] = 59
    _write_json(summary_path, summary)
    report = validate_causal_qualification_job(
        lock_path=lock_path,
        job_dirs=jobs,
        provider_identity_path=provider_identity,
    )
    assert "runtime_budget_drift" in report["tasks"][0]["failures"]

    summary["runtime_budgets"]["provider_tokens_used"] = 60
    _write_json(summary_path, summary)
    artifact = json.loads(provider_identity.read_text(encoding="utf-8"))
    artifact["returned_model"] = "tampered-model"
    _write_json(provider_identity, artifact)
    with pytest.raises(ValueError, match="provider identity artifact"):
        validate_causal_qualification_job(
            lock_path=lock_path,
            job_dirs=jobs,
            provider_identity_path=provider_identity,
        )

    with pytest.raises(ValueError, match="provider identity artifact"):
        validate_causal_qualification_job(
            lock_path=lock_path,
            job_dirs=jobs,
            provider_identity_path=tmp_path / "missing-provider-identity.json",
        )

    drifted_artifact = capture_provider_identity(
        client=_FakeClient({**_identity_response(), "model": "different-model"}),
        model=_config().model,
        base_url=_config().base_url,
    )
    drifted_path = write_provider_identity_artifact(tmp_path / "drifted", drifted_artifact)
    with pytest.raises(ValueError, match="provider identity artifact drift"):
        validate_causal_qualification_job(
            lock_path=lock_path,
            job_dirs=jobs,
            provider_identity_path=drifted_path,
        )


@pytest.mark.parametrize(
    ("configured_model", "base_url"),
    [
        ("different-configured-model", "https://api.deepseek.com"),
        ("deepseek-v4-flash", "https://alternate.example.test"),
    ],
)
def test_live_qualification_binds_provider_artifact_configuration(
    tmp_path: Path,
    configured_model: str,
    base_url: str,
) -> None:
    artifact = capture_provider_identity(
        client=_FakeClient(_identity_response()),
        model=configured_model,
        base_url=base_url,
    )
    artifact_path = write_provider_identity_artifact(
        tmp_path / "drifted-configuration",
        artifact,
    )
    lock = _lock_payload(tmp_path / "lock-fixture")
    lock["provider_identity_sha256"] = artifact.content_sha256
    lock_path = tmp_path / "causal-qualification.lock.json"
    _write_json(lock_path, lock)

    with pytest.raises(ValueError, match="provider identity artifact drift"):
        validate_causal_qualification_job(
            lock_path=lock_path,
            job_dirs=_live_jobs(tmp_path / "live", lock_path=lock_path),
            provider_identity_path=artifact_path,
        )


@pytest.mark.parametrize(
    "exception",
    [
        {"category": "provider_transport_error", "status_code": 429},
        {"category": "provider_transport_error", "status_code": 503},
        {
            "category": "provider_transport_error",
            "exception_type": "NetworkTransportError",
        },
        {"exception_type": "NetworkTransportError"},
        {"exception_type": "DockerImageBuildError"},
        {"exception_type": "ImagePullError"},
        {"exception_type": "VerifierTimeoutError"},
        {"exception_type": "HarborInfrastructureError"},
    ],
)
def test_external_failure_is_retryable_exactly_once(
    exception: dict[str, object],
) -> None:
    result = {"exception_info": exception}

    assert retry_eligible(result, retries_used=0) is True
    assert retry_eligible(result, retries_used=1) is False


def test_live_qualification_uses_prior_job_as_retry_history(tmp_path: Path) -> None:
    lock_path = _write_lock(tmp_path)
    current_jobs = _live_jobs(tmp_path / "current", lock_path=lock_path)
    prior_jobs = _live_jobs(tmp_path / "prior", lock_path=lock_path)
    task_slug = FROZEN_GATE_TASK_IDS[0].split("/", 1)[1]
    external_failure = {
        "category": "network_transport_error",
        "exception_type": "NetworkTransportError",
    }
    for job in (prior_jobs[0], current_jobs[0]):
        result_path = job / f"{task_slug}__qualification" / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["exception_info"] = external_failure
        result["verifier_result"] = None
        _write_json(result_path, result)
    current_config_path = current_jobs[0] / "config.json"
    current_config = json.loads(current_config_path.read_text(encoding="utf-8"))
    current_config["job_name"] += "-retry-1"
    _write_json(current_config_path, current_config)

    report = validate_causal_qualification_job(
        lock_path=lock_path,
        job_dirs=current_jobs,
        prior_job_dirs=(prior_jobs[0],),
        provider_identity_path=_provider_identity_path(tmp_path),
    )

    first = report["tasks"][0]
    assert first["retries_used"] == 1
    assert first["retry_eligible"] is False


def test_live_qualification_rejects_retry_after_nonretryable_prior_job(
    tmp_path: Path,
) -> None:
    lock_path = _write_lock(tmp_path)
    current_jobs = _live_jobs(tmp_path / "current", lock_path=lock_path)
    prior_jobs = _live_jobs(tmp_path / "prior", lock_path=lock_path)

    with pytest.raises(ValueError, match="prior qualification result"):
        validate_causal_qualification_job(
            lock_path=lock_path,
            job_dirs=current_jobs,
            prior_job_dirs=(prior_jobs[0],),
            provider_identity_path=_provider_identity_path(tmp_path),
        )


def test_live_qualification_rejects_retry_job_without_prior_history(
    tmp_path: Path,
) -> None:
    lock_path = _write_lock(tmp_path)
    current_jobs = _live_jobs(tmp_path / "current", lock_path=lock_path)
    config_path = current_jobs[0] / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["job_name"] += "-retry-1"
    _write_json(config_path, config)

    with pytest.raises(ValueError, match="job provenance"):
        validate_causal_qualification_job(
            lock_path=lock_path,
            job_dirs=current_jobs,
            provider_identity_path=_provider_identity_path(tmp_path),
        )


@pytest.mark.parametrize(
    "exception",
    [
        {"category": "provider_transport_error", "status_code": 400},
        {"category": "provider_contract_error"},
        {"category": "provider_identity_error"},
        {"category": "budget_error"},
        {"category": "adapter_error"},
        {"category": "agent_error"},
        {"category": "policy_error"},
        {"category": "causal_conformance_error"},
        {"exception_type": "AgentTimeoutError"},
        {"exception_type": "DockerAgentPolicyError"},
    ],
)
def test_internal_or_policy_failure_is_never_retryable(
    exception: dict[str, object],
) -> None:
    assert retry_eligible({"exception_info": exception}, retries_used=0) is False
