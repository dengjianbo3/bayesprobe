from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from bayesprobe_terminal_bench.conformance import (
    TraceClassification,
    validate_trial_trace,
)
from bayesprobe_terminal_bench.runner_factory import (
    terminal_bench_lock_schema_mismatches,
)
from write_benchmark_lock import (
    RuntimeIdentity,
    collect_runtime_identity,
    extract_job_identity,
)


SmokeClassification = Literal[
    "engineering_pass",
    "task_failure",
    "infrastructure_error",
    "provider_error",
    "conformance_error",
]
_SUCCESS = frozenset({"engineering_pass", "task_failure"})


def classify_smoke_run(
    *,
    job_dir: Path,
    lock_path: Path,
    runtime_identity: RuntimeIdentity,
) -> SmokeClassification:
    lock = _read_object(Path(lock_path))
    if not _valid_lock_identity(lock):
        return "conformance_error"
    if not _runtime_matches_lock(runtime_identity, lock):
        return "conformance_error"

    trial_dirs = _trial_dirs(Path(job_dir))
    if len(trial_dirs) != 1:
        return "infrastructure_error"
    trial_dir = trial_dirs[0]
    result_path = trial_dir / "result.json"
    if not result_path.is_file():
        return "infrastructure_error"
    try:
        result = _read_object(result_path)
    except ValueError:
        return "infrastructure_error"
    if not _result_matches_lock(Path(job_dir), lock):
        return "conformance_error"

    bayesprobe_dir = trial_dir / "agent" / "bayesprobe"
    verifier_completed, reward = _verifier_result(result)
    if not verifier_completed:
        if _provider_failed(bayesprobe_dir, result):
            return "provider_error"
        return (
            "infrastructure_error"
            if not _agent_started(bayesprobe_dir)
            else "conformance_error"
        )

    if not _complete_trace(bayesprobe_dir):
        return "conformance_error"
    if reward == 0.0:
        return "task_failure"
    return "engineering_pass"


def main(
    argv: Sequence[str] | None = None,
    *,
    runtime_identity_loader: Any = collect_runtime_identity,
) -> int:
    parser = argparse.ArgumentParser(
        description="Classify one locked Terminal-Bench engineering smoke run."
    )
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        runtime_identity = runtime_identity_loader(
            repository_root=Path(__file__).resolve().parents[3],
            job_dir=args.job,
        )
        classification = classify_smoke_run(
            job_dir=args.job,
            lock_path=args.lock,
            runtime_identity=runtime_identity,
        )
    except Exception:
        classification = "infrastructure_error"
    print(classification)
    return 0 if classification in _SUCCESS else 1


def _trial_dirs(job_dir: Path) -> list[Path]:
    if not job_dir.is_dir():
        return []
    return sorted(
        path
        for path in job_dir.iterdir()
        if path.is_dir() and (path / "config.json").is_file()
    )


def _valid_lock_identity(lock: Mapping[str, Any]) -> bool:
    return not terminal_bench_lock_schema_mismatches(lock)


def _runtime_matches_lock(
    runtime_identity: RuntimeIdentity,
    lock: Mapping[str, Any],
) -> bool:
    return all(
        lock.get(key) == value
        for key, value in {
            "harbor_version": runtime_identity.harbor_version,
            "root_git_sha": runtime_identity.root_git_sha,
            "adapter_tree_sha": runtime_identity.adapter_tree_sha,
            "container_image": runtime_identity.container_image,
            "image_digest": runtime_identity.image_digest,
        }.items()
    )


def _result_matches_lock(job_dir: Path, lock: Mapping[str, Any]) -> bool:
    try:
        identity = extract_job_identity(job_dir, required_reward=None)
    except ValueError:
        return False
    return (
        identity.dataset_name == lock.get("dataset_name")
        and identity.dataset_revision == lock.get("dataset_revision")
        and identity.task_id == lock.get("task_id")
        and identity.task_checksum == lock.get("task_checksum")
        and identity.n_attempts == lock.get("n_attempts")
    )


def _verifier_result(result: Mapping[str, Any]) -> tuple[bool, float | None]:
    if result.get("finished_at") is None:
        return False, None
    verifier = result.get("verifier_result")
    if not isinstance(verifier, Mapping):
        return False, None
    rewards = verifier.get("rewards")
    if not isinstance(rewards, Mapping):
        return False, None
    values = [
        float(value)
        for value in rewards.values()
        if type(value) in (int, float)
    ]
    if "reward" in rewards and type(rewards["reward"]) in (int, float):
        return True, float(rewards["reward"])
    return (True, values[0]) if len(values) == 1 else (False, None)


def _provider_failed(bayesprobe_dir: Path, result: Mapping[str, Any]) -> bool:
    report = validate_trial_trace(bayesprobe_dir)
    if report.classification is TraceClassification.PROVIDER_CONTRACT_ERROR:
        return True
    exception = result.get("exception_info")
    if not isinstance(exception, Mapping):
        return False
    text = " ".join(
        str(exception.get(key, ""))
        for key in ("exception_type", "exception_message")
    ).casefold()
    return any(
        marker in text
        for marker in ("provider", "authentication", "openai", "rate limit")
    )


def _agent_started(bayesprobe_dir: Path) -> bool:
    if not bayesprobe_dir.is_dir():
        return False
    return any(
        path.is_file() and path.stat().st_size > 0
        for path in bayesprobe_dir.iterdir()
    )


def _complete_trace(bayesprobe_dir: Path) -> bool:
    return (
        validate_trial_trace(bayesprobe_dir).classification
        is TraceClassification.CONFORMANT
    )


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError(f"could not read JSON artifact: {path}") from None
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
