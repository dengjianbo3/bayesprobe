from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

from bayesprobe_terminal_bench.conformance import (
    TraceClassification,
    validate_trial_trace,
)
from bayesprobe_terminal_bench.experiment_lock import CausalQualificationLock
from capture_provider_identity import load_provider_identity_artifact
from freeze_historical_traces import FROZEN_TASKS, HistoricalTraceManifest


_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_BAYESPROBE_AGENT = "bayesprobe_terminal_bench.agent:BayesProbeHarborAgent"
_NONRETRYABLE_CATEGORIES = frozenset(
    {
        "adapter_error",
        "agent_error",
        "budget_error",
        "causal_conformance_error",
        "provider_contract_error",
        "provider_identity_error",
        "policy_error",
    }
)
_RETRYABLE_CATEGORIES = frozenset(
    {
        "docker_infrastructure_error",
        "harbor_infrastructure_error",
        "image_pull_error",
        "network_transport_error",
        "verifier_infrastructure_error",
    }
)


def replay_offline_gate(
    *,
    historical_fixtures: Path,
    synthetic_fixture: Path | None = None,
) -> dict[str, object]:
    historical_root = Path(historical_fixtures)
    manifest = _verified_historical_manifest(historical_root)
    expected_tasks = tuple(task_id for task_id, _ in FROZEN_TASKS)
    expected_classes = {task_id: classification for task_id, classification in FROZEN_TASKS}
    traces: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    for trace in manifest.traces:
        slug = trace.task_id.split("/", 1)[1]
        report = validate_trial_trace(historical_root / slug)
        classification = report.classification.value
        counts[classification] += 1
        traces.append(
            {
                "task_id": trace.task_id,
                "expected_classification": trace.expected_classification,
                "actual_classification": classification,
                "passed": classification == trace.expected_classification,
            }
        )
    historical_passed = (
        tuple(trace.task_id for trace in manifest.traces) == expected_tasks
        and all(item["passed"] is True for item in traces)
        and dict(counts)
        == {
            "provider_contract_error": 2,
            "causal_conformance_error": 1,
        }
        and all(
            trace.expected_classification == expected_classes[trace.task_id]
            for trace in manifest.traces
        )
    )

    synthetic = Path(synthetic_fixture) if synthetic_fixture is not None else (
        historical_root.parent
        / "causal_traces"
        / "conformant-inspect-intervene-verify"
    )
    _verify_synthetic_fixture(synthetic)
    synthetic_report = validate_trial_trace(synthetic)
    synthetic_passed = (
        synthetic_report.classification is TraceClassification.CONFORMANT
        and synthetic_report.complete_cycles >= 1
    )
    return {
        "schema_version": "terminal_bench_causal_offline_gate:v1",
        "offline_only": True,
        "historical_replay_passed": historical_passed,
        "historical_classification_counts": {
            "provider_contract_error": counts["provider_contract_error"],
            "causal_conformance_error": counts["causal_conformance_error"],
        },
        "historical_traces": traces,
        "synthetic_fixture": synthetic.name,
        "synthetic_classification": synthetic_report.classification.value,
        "synthetic_complete_cycles": synthetic_report.complete_cycles,
        "synthetic_conformant_passed": synthetic_passed,
        "offline_gate_passed": historical_passed and synthetic_passed,
    }


def validate_causal_qualification_job(
    *,
    lock_path: Path,
    job_dirs: Sequence[Path],
    provider_identity_path: Path,
) -> dict[str, object]:
    try:
        lock = CausalQualificationLock.model_validate_json(
            Path(lock_path).read_text(encoding="utf-8")
        )
    except Exception:
        raise ValueError("causal qualification lock is invalid") from None
    _validate_provider_identity_artifact(
        lock=lock,
        provider_identity_path=provider_identity_path,
    )
    results = _trial_results(
        job_dirs,
        locked_task_ids=tuple(task.task_id for task in lock.tasks),
    )
    task_reports = [
        _validate_live_task(
            lock=lock,
            task=task,
            trial=results.get(task.task_id),
        )
        for task in lock.tasks
    ]
    return {
        "schema_version": "terminal_bench_causal_qualification_report:v1",
        "model": lock.model,
        "base_url": lock.base_url,
        "expected_provider_model": lock.expected_provider_model,
        "expected_system_fingerprint": lock.expected_system_fingerprint,
        "budgets": lock.budgets.model_dump(mode="json"),
        "tasks": task_reports,
        "qualification_passed": all(
            report["passed"] is True for report in task_reports
        ),
    }


def _validate_live_task(
    *,
    lock: CausalQualificationLock,
    task: object,
    trial: tuple[Path, Mapping[str, object]] | None,
) -> dict[str, object]:
    task_id = getattr(task, "task_id")
    failures: list[str] = []
    if trial is None:
        return {
            "task_id": task_id,
            "reward": None,
            "classification": TraceClassification.ADAPTER_ERROR.value,
            "complete_cycles": 0,
            "actions": 0,
            "model_calls": 0,
            "provider_tokens": 0,
            "nonneutral_updates": 0,
            "discarded_evidence": 0,
            "retry_eligible": False,
            "failures": ["missing_result"],
            "passed": False,
        }

    trial_dir, result = trial
    if not _task_identity_matches(result, task_id, getattr(task, "task_ref")):
        failures.append("task_identity_drift")
    config = result.get("config")
    agent = config.get("agent") if isinstance(config, Mapping) else None
    if not isinstance(agent, Mapping) or agent.get("import_path") != _BAYESPROBE_AGENT:
        failures.append("agent_identity_drift")

    reward = _official_reward(result)
    if reward is None:
        failures.append("missing_verifier")
    elif not isinstance(result.get("finished_at"), str) or not str(
        result["finished_at"]
    ).strip():
        failures.append("incomplete_verifier")
    if result.get("exception_info") is not None:
        failures.append("trial_exception")

    artifact_root = trial_dir / "agent" / "bayesprobe"
    conformance = validate_trial_trace(artifact_root)
    if not _trajectory_exists(artifact_root):
        failures.append("missing_atif")
    if conformance.classification is not TraceClassification.CONFORMANT:
        failures.append(conformance.classification.value)
    if conformance.complete_cycles < 1:
        failures.append("incomplete_cycle")

    model_calls, provider_tokens, identity_matches = _provider_accounting(
        artifact_root / "provider_telemetry.jsonl",
        expected_model=lock.expected_provider_model,
        expected_fingerprint_available=lock.expected_system_fingerprint_available,
        expected_fingerprint=lock.expected_system_fingerprint,
    )
    if not identity_matches:
        failures.append("provider_identity_drift")
    actions = _declared_counter(
        artifact_root / "summary.json",
        "terminal_actions",
        fallback=conformance.actions,
    )
    if not _runtime_budgets_match(
        artifact_root / "summary.json",
        locked_budgets=lock.budgets.model_dump(mode="json"),
        provider_tokens=provider_tokens,
    ):
        failures.append("runtime_budget_drift")
    if (
        actions > lock.budgets.max_total_actions
        or model_calls > lock.budgets.max_model_calls
        or provider_tokens > lock.budgets.max_provider_tokens
    ):
        failures.append("budget_exceeded")

    return {
        "task_id": task_id,
        "reward": reward,
        "classification": conformance.classification.value,
        "complete_cycles": conformance.complete_cycles,
        "actions": actions,
        "model_calls": model_calls,
        "provider_tokens": provider_tokens,
        "nonneutral_updates": conformance.nonneutral_updates,
        "discarded_evidence": conformance.discarded_evidence,
        "retry_eligible": retry_eligible(result, retries_used=0),
        "failures": list(dict.fromkeys(failures)),
        "passed": not failures,
    }


def retry_eligible(
    result: Mapping[str, object],
    *,
    retries_used: int,
) -> bool:
    if retries_used != 0:
        return False
    exception = result.get("exception_info")
    if not isinstance(exception, Mapping):
        return False
    category = exception.get("category")
    if category in _NONRETRYABLE_CATEGORIES:
        return False
    if category in _RETRYABLE_CATEGORIES:
        return True
    if category == "provider_transport_error":
        status = exception.get("status_code")
        if type(status) is int and (status == 429 or 500 <= status <= 599):
            return True

    exception_type = exception.get("exception_type")
    if not isinstance(exception_type, str):
        return False
    normalized = exception_type.casefold()
    if "agenttimeout" in normalized:
        return False
    return any(
        marker in normalized
        for marker in (
            "docker",
            "harborinfrastructure",
            "imagepull",
            "networktransport",
            "verifiertimeout",
            "verifierinfrastructure",
        )
    )


def _verified_historical_manifest(root: Path) -> HistoricalTraceManifest:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("historical fixtures must be a real directory")
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("historical fixture symlink is forbidden")
    try:
        manifest = HistoricalTraceManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except Exception:
        raise ValueError("historical fixture manifest is invalid") from None
    expected_files = {"manifest.json"}
    for trace in manifest.traces:
        task_root = root / trace.task_id.split("/", 1)[1]
        for relative, expected_digest in trace.files.items():
            normalized = PurePosixPath(relative)
            if normalized.is_absolute() or ".." in normalized.parts:
                raise ValueError("historical fixture path is unsafe")
            if not _SHA256.fullmatch(expected_digest):
                raise ValueError("historical fixture digest is invalid")
            path = task_root.joinpath(*normalized.parts)
            if path.is_symlink() or not path.is_file():
                raise ValueError("historical fixture file is invalid")
            actual = f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"
            if actual != expected_digest:
                raise ValueError("historical fixture digest mismatch")
            expected_files.add(path.relative_to(root).as_posix())
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise ValueError("historical fixture inventory mismatch")
    return manifest


def _verify_synthetic_fixture(fixture: Path) -> None:
    if fixture.is_symlink() or not fixture.is_dir():
        raise ValueError("synthetic fixture must be a real directory")
    fixture_root = fixture.parent
    if fixture_root.name == "broken-bindings":
        fixture_root = fixture_root.parent
    manifest_path = fixture_root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise ValueError("synthetic fixture manifest is invalid") from None
    files = manifest.get("files") if isinstance(manifest, Mapping) else None
    if not isinstance(files, Mapping):
        raise ValueError("synthetic fixture manifest is invalid")
    prefix = f"{fixture.relative_to(fixture_root).as_posix()}/"
    expected = {
        relative.removeprefix(prefix): digest
        for relative, digest in files.items()
        if isinstance(relative, str) and relative.startswith(prefix)
    }
    actual = {
        path.relative_to(fixture).as_posix(): path
        for path in fixture.rglob("*")
        if path.is_file()
    }
    if set(actual) != set(expected):
        raise ValueError("synthetic fixture inventory mismatch")
    for relative, path in actual.items():
        digest = expected[relative]
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError("synthetic fixture digest is invalid")
        if path.is_symlink() or (
            f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}" != digest
        ):
            raise ValueError("synthetic fixture digest mismatch")


def _trial_results(
    job_dirs: Sequence[Path],
    *,
    locked_task_ids: tuple[str, str, str],
) -> dict[str, tuple[Path, Mapping[str, object]]]:
    if len(job_dirs) != 3:
        raise ValueError(
            "causal qualification requires exactly three --job directories"
        )
    results: dict[str, tuple[Path, Mapping[str, object]]] = {}
    for job_dir in job_dirs:
        if job_dir.is_symlink() or not job_dir.is_dir():
            raise ValueError("causal qualification job must be a real directory")
        paths = sorted(job_dir.glob("*/result.json"))
        if len(paths) != 1:
            raise ValueError("each qualification job must contain exactly one result")
        path = paths[0]
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raise ValueError("causal qualification result is invalid") from None
        if not isinstance(payload, Mapping):
            raise ValueError("causal qualification result is invalid")
        task_id = payload.get("task_name")
        if not isinstance(task_id, str):
            raise ValueError("qualification result task is invalid")
        if task_id not in locked_task_ids:
            raise ValueError("qualification result task is unknown")
        if task_id in results:
            raise ValueError(f"duplicate qualification result for {task_id}")
        results[task_id] = (path.parent, payload)
    missing = set(locked_task_ids) - set(results)
    if missing:
        raise ValueError(f"missing qualification result for {sorted(missing)}")
    return results


def _task_identity_matches(
    result: Mapping[str, object],
    task_id: str,
    task_ref: str,
) -> bool:
    identity = result.get("task_id")
    if not isinstance(identity, Mapping):
        return False
    return (
        result.get("task_name") == task_id
        and f"{identity.get('org')}/{identity.get('name')}" == task_id
        and identity.get("ref") == task_ref
    )


def _official_reward(result: Mapping[str, object]) -> float | None:
    verifier = result.get("verifier_result")
    rewards = verifier.get("rewards") if isinstance(verifier, Mapping) else None
    reward = rewards.get("reward") if isinstance(rewards, Mapping) else None
    if type(reward) not in (int, float):
        return None
    value = float(reward)
    return value if math.isfinite(value) else None


def _trajectory_exists(artifact_root: Path) -> bool:
    return (artifact_root / "trajectory.json").is_file() or (
        artifact_root.parent / "trajectory.json"
    ).is_file()


def _provider_accounting(
    path: Path,
    *,
    expected_model: str,
    expected_fingerprint_available: bool,
    expected_fingerprint: str | None,
) -> tuple[int, int, bool]:
    calls = 0
    tokens = 0
    identity_matches = True
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return 0, 0, False
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            return calls, tokens, False
        if not isinstance(record, Mapping) or record.get("outcome") != "success":
            identity_matches = False
            continue
        calls += 1
        usage = record.get("usage")
        total = usage.get("total_tokens") if isinstance(usage, Mapping) else None
        if type(total) is not int or total < 0:
            identity_matches = False
        else:
            tokens += total
        fingerprint = record.get("system_fingerprint")
        if (
            record.get("model") != expected_model
            or (fingerprint is not None) != expected_fingerprint_available
            or fingerprint != expected_fingerprint
        ):
            identity_matches = False
    if calls == 0:
        identity_matches = False
    return calls, tokens, identity_matches


def _declared_counter(path: Path, name: str, *, fallback: int) -> int:
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback
    value = summary.get(name) if isinstance(summary, Mapping) else None
    return value if type(value) is int and value >= 0 else fallback


def _runtime_budgets_match(
    path: Path,
    *,
    locked_budgets: Mapping[str, object],
    provider_tokens: int,
) -> bool:
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    runtime_budgets = (
        summary.get("runtime_budgets") if isinstance(summary, Mapping) else None
    )
    expected = {**locked_budgets, "provider_tokens_used": provider_tokens}
    return (
        isinstance(runtime_budgets, Mapping)
        and set(runtime_budgets) == set(expected)
        and all(
            type(runtime_budgets.get(name)) is int and runtime_budgets[name] == value
            for name, value in expected.items()
        )
    )


def _validate_provider_identity_artifact(
    *,
    lock: CausalQualificationLock,
    provider_identity_path: Path,
) -> None:
    try:
        artifact = load_provider_identity_artifact(provider_identity_path)
    except Exception:
        raise ValueError("provider identity artifact is invalid") from None
    if (
        artifact.content_sha256 != lock.provider_identity_sha256
        or artifact.returned_model != lock.expected_provider_model
        or artifact.system_fingerprint_available
        != lock.expected_system_fingerprint_available
        or artifact.system_fingerprint != lock.expected_system_fingerprint
    ):
        raise ValueError("provider identity artifact drift")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--historical-fixtures", required=True, type=Path)
    parser.add_argument("--offline-only", action="store_true")
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--job", action="append", type=Path)
    parser.add_argument("--provider-identity", type=Path)
    args = parser.parse_args(argv)
    offline = replay_offline_gate(historical_fixtures=args.historical_fixtures)
    if args.offline_only:
        if (
            args.lock is not None
            or args.job is not None
            or args.provider_identity is not None
        ):
            parser.error("--offline-only cannot be combined with --lock or --job")
        report = offline
        passed = bool(report["offline_gate_passed"])
    else:
        if (
            args.lock is None
            or args.job is None
            or len(args.job) != 3
            or args.provider_identity is None
        ):
            parser.error(
                "live validation requires --lock, --provider-identity, and exactly three --job values"
            )
        live = validate_causal_qualification_job(
            lock_path=args.lock,
            job_dirs=args.job,
            provider_identity_path=args.provider_identity,
        )
        report = {
            **offline,
            "offline_only": False,
            **live,
            "qualification_passed": (
                bool(offline["offline_gate_passed"])
                and bool(live["qualification_passed"])
            ),
        }
        passed = bool(report["qualification_passed"])
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
