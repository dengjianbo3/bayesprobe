from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path

from bayesprobe_terminal_bench.conformance import (
    TraceClassification,
    validate_trial_trace,
)
from bayesprobe_terminal_bench.experiment_lock import (
    FROZEN_GATE_TASK_IDS,
    PAIRED_GATE_ARMS,
    PairedGateLock,
)


_SECRET_PATTERN = re.compile(
    rb"(?:sk-[A-Za-z0-9_-]{12,}|tvly-[A-Za-z0-9_-]{12,}|"
    rb"github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16})",
    re.IGNORECASE,
)


def _trace_is_conformant(artifact_root: Path) -> bool:
    return (
        validate_trial_trace(artifact_root).classification
        is TraceClassification.CONFORMANT
    )


def validate_paired_gate_jobs(
    *,
    lock_path: Path,
    direct_job: Path,
    bayesprobe_job: Path,
    trace_validator: Callable[[Path], bool] = _trace_is_conformant,
) -> dict[str, object]:
    try:
        lock = PairedGateLock.model_validate_json(
            Path(lock_path).read_text(encoding="utf-8")
        )
    except Exception:
        raise ValueError("paired gate lock is invalid") from None
    _scan_for_secrets(Path(lock_path))
    _scan_for_secrets(Path(direct_job))
    _scan_for_secrets(Path(bayesprobe_job))

    arms = {
        "direct": _validate_arm(
            job_dir=Path(direct_job),
            arm="direct",
            lock=lock,
            trace_validator=trace_validator,
        ),
        "bayesprobe": _validate_arm(
            job_dir=Path(bayesprobe_job),
            arm="bayesprobe",
            lock=lock,
            trace_validator=trace_validator,
        ),
    }
    failures: list[str] = []
    for arm, report in arms.items():
        if report["completed_verifiers"] != 3:
            failures.append(f"{arm}_incomplete_verifier")
    if arms["bayesprobe"]["complete_traces"] != 3:
        failures.append("bayesprobe_incomplete_trace")
    if arms["bayesprobe"]["reward_total"] < 1.0:
        failures.append("bayesprobe_zero_of_three")
    return {
        "schema_version": "terminal_bench_paired_gate_report:v0.1",
        "tasks": list(FROZEN_GATE_TASK_IDS),
        "arms": arms,
        "gate_passed": not failures,
        "gate_failures": failures,
    }


def _validate_arm(
    *,
    job_dir: Path,
    arm: str,
    lock: PairedGateLock,
    trace_validator: Callable[[Path], bool],
) -> dict[str, object]:
    results = _trial_results(job_dir)
    task_reports: list[dict[str, object]] = []
    completed_verifiers = 0
    complete_traces = 0
    for task in lock.tasks:
        pair = results.get(task.task_id)
        if pair is None:
            task_reports.append(
                {"task_id": task.task_id, "status": "missing", "reward": None}
            )
            continue
        trial_dir, result = pair
        _validate_task_identity(result, task.task_id, task.task_ref)
        config = result.get("config")
        agent = config.get("agent") if isinstance(config, Mapping) else None
        if not isinstance(agent, Mapping) or agent.get("import_path") != PAIRED_GATE_ARMS[arm]:
            raise ValueError(f"{arm} trial uses the wrong agent import path")
        reward = _optional_reward(result)
        verifier_complete = reward is not None and result.get("finished_at") is not None
        if verifier_complete:
            completed_verifiers += 1
        artifact_name = "direct" if arm == "direct" else "bayesprobe"
        artifact_dir = trial_dir / "agent" / artifact_name
        trace_complete = arm == "direct" or trace_validator(artifact_dir)
        if trace_complete:
            complete_traces += 1
        metadata = _metadata(result)
        usage = _provider_usage(artifact_dir / "provider_telemetry.jsonl")
        task_reports.append(
            {
                "task_id": task.task_id,
                "status": (
                    "completed" if verifier_complete else _failure_status(result)
                ),
                "reward": reward,
                "trace_complete": trace_complete,
                "terminal_actions": _nonnegative_int(metadata.get("terminal_actions")),
                "model_calls": _nonnegative_int(metadata.get("model_calls")),
                "input_tokens": usage[0],
                "output_tokens": usage[1],
                "latency_seconds": _duration_seconds(result),
            }
        )
    return {
        "tasks": task_reports,
        "completed_verifiers": completed_verifiers,
        "complete_traces": complete_traces,
        "reward_total": sum(
            float(item["reward"])
            for item in task_reports
            if type(item.get("reward")) in (int, float)
        ),
        "terminal_actions": sum(
            int(item.get("terminal_actions") or 0) for item in task_reports
        ),
        "model_calls": sum(int(item.get("model_calls") or 0) for item in task_reports),
        "input_tokens": sum(int(item.get("input_tokens") or 0) for item in task_reports),
        "output_tokens": sum(int(item.get("output_tokens") or 0) for item in task_reports),
    }


def _trial_results(job_dir: Path) -> dict[str, tuple[Path, dict[str, object]]]:
    results: dict[str, tuple[Path, dict[str, object]]] = {}
    for path in sorted(job_dir.glob("*/result.json")):
        result = _read_object(path)
        task_name = result.get("task_name")
        if not isinstance(task_name, str):
            continue
        if task_name in results:
            raise ValueError(f"duplicate result for {task_name}")
        results[task_name] = (path.parent, result)
    return results


def _validate_task_identity(
    result: Mapping[str, object], task_id: str, task_ref: str
) -> None:
    identity = result.get("task_id")
    if not isinstance(identity, Mapping):
        raise ValueError("trial task identity is missing")
    resolved = f"{identity.get('org')}/{identity.get('name')}"
    if result.get("task_name") != task_id or resolved != task_id:
        raise ValueError("trial task was substituted")
    if identity.get("ref") != task_ref:
        raise ValueError("trial task ref was substituted")


def _optional_reward(result: Mapping[str, object]) -> float | None:
    verifier = result.get("verifier_result")
    rewards = verifier.get("rewards") if isinstance(verifier, Mapping) else None
    reward = rewards.get("reward") if isinstance(rewards, Mapping) else None
    return float(reward) if type(reward) in (int, float) else None


def _metadata(result: Mapping[str, object]) -> Mapping[str, object]:
    agent_result = result.get("agent_result")
    metadata = agent_result.get("metadata") if isinstance(agent_result, Mapping) else None
    return metadata if isinstance(metadata, Mapping) else {}


def _failure_status(result: Mapping[str, object]) -> str:
    exception = result.get("exception_info")
    if isinstance(exception, Mapping):
        name = exception.get("exception_type") or exception.get("type")
        return f"error:{name}" if isinstance(name, str) else "error"
    return "incomplete_verifier"


def _provider_usage(path: Path) -> tuple[int, int]:
    input_tokens = 0
    output_tokens = 0
    if not path.is_file():
        return input_tokens, output_tokens
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        usage = record.get("usage") if isinstance(record, Mapping) else None
        if isinstance(usage, Mapping):
            input_tokens += _nonnegative_int(usage.get("input_tokens")) or 0
            output_tokens += _nonnegative_int(usage.get("output_tokens")) or 0
    return input_tokens, output_tokens


def _duration_seconds(result: Mapping[str, object]) -> float | None:
    try:
        started = datetime.fromisoformat(str(result["started_at"]).replace("Z", "+00:00"))
        finished = datetime.fromisoformat(str(result["finished_at"]).replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError):
        return None
    return max(0.0, (finished - started).total_seconds())


def _nonnegative_int(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _scan_for_secrets(path: Path) -> None:
    if path.is_file():
        files = [path]
    else:
        candidates = {
            path / "config.json",
            path / "lock.json",
            path / "result.json",
            *path.glob("*/config.json"),
            *path.glob("*/lock.json"),
            *path.glob("*/result.json"),
            *path.glob("*/agent/**/*"),
        }
        files = sorted(item for item in candidates if item.is_file())
    for file_path in files:
        try:
            data = file_path.read_bytes()
        except OSError:
            continue
        if _SECRET_PATTERN.search(data):
            raise ValueError(f"secret-like content found in {file_path.name}")


def _read_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise ValueError(f"invalid JSON artifact: {path}") from None
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--direct-job", required=True, type=Path)
    parser.add_argument("--bayesprobe-job", required=True, type=Path)
    args = parser.parse_args(argv)
    report = validate_paired_gate_jobs(
        lock_path=args.lock,
        direct_job=args.direct_job,
        bayesprobe_job=args.bayesprobe_job,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if report["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
