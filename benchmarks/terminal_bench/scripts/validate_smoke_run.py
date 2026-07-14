from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal


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
) -> SmokeClassification:
    lock = _read_object(Path(lock_path))
    if not _valid_lock_identity(lock):
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Classify one locked Terminal-Bench engineering smoke run."
    )
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        classification = classify_smoke_run(
            job_dir=args.job,
            lock_path=args.lock,
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
    task_id = lock.get("task_id")
    task_checksum = lock.get("task_checksum")
    return (
        task_id == "terminal-bench/break-filter-js-from-html"
        and isinstance(task_checksum, str)
        and bool(task_checksum.strip())
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
    for record in _read_jsonl(bayesprobe_dir / "provider_telemetry.jsonl") or []:
        if record.get("outcome") in {"error", "empty_content"}:
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
        for marker in (
            "provider",
            "authentication",
            "openai",
            "rate limit",
        )
    )


def _agent_started(bayesprobe_dir: Path) -> bool:
    if not bayesprobe_dir.is_dir():
        return False
    return any(
        path.is_file() and path.stat().st_size > 0
        for path in bayesprobe_dir.iterdir()
    )


def _complete_trace(bayesprobe_dir: Path) -> bool:
    summary = _try_read_object(bayesprobe_dir / "summary.json")
    if summary is None:
        return False
    if not _positive_int(summary.get("terminal_actions")):
        return False
    if not _positive_int(summary.get("bayesprobe_cycles")):
        return False
    actions = _read_jsonl(bayesprobe_dir / "environment_actions.jsonl")
    records = _read_jsonl(bayesprobe_dir / "bayesprobe_ledger.jsonl")
    if not actions or not records:
        return False

    by_type: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        record_type = record.get("record_type")
        payload = record.get("payload")
        if not isinstance(record_type, str) or not isinstance(payload, Mapping):
            return False
        by_type.setdefault(record_type, []).append(payload)

    runs = by_type.get("run", [])
    if not any(run.get("status") == "completed" for run in runs):
        return False
    cycles = [
        cycle
        for cycle in by_type.get("cycle", [])
        if cycle.get("boundary_status") == "integrated"
        and cycle.get("completed_at") is not None
    ]
    if not cycles:
        return False
    if len(cycles) != summary.get("bayesprobe_cycles"):
        return False
    if not by_type.get("belief_state"):
        return False

    for cycle in cycles:
        cycle_id = cycle.get("cycle_id")
        if not isinstance(cycle_id, str):
            return False
        if not _cycle_is_linked(cycle_id, by_type):
            return False
    return True


def _cycle_is_linked(
    cycle_id: str,
    by_type: Mapping[str, list[Mapping[str, Any]]],
) -> bool:
    probe_sets = [
        item for item in by_type.get("probe_set", []) if item.get("cycle_id") == cycle_id
    ]
    if len(probe_sets) != 1:
        return False
    probes = probe_sets[0].get("probes")
    if not isinstance(probes, list):
        return False
    probe_ids = {
        item.get("id")
        for item in probes
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }
    if not probe_ids:
        return False

    signals = [
        item for item in by_type.get("external_signal", []) if item.get("cycle_id") == cycle_id
    ]
    signal_ids: set[str] = set()
    for signal in signals:
        provenance = signal.get("provenance")
        signal_id = signal.get("id")
        if (
            not isinstance(signal_id, str)
            or signal.get("generated_by_probe") not in probe_ids
            or not isinstance(provenance, Mapping)
            or provenance.get("epistemic_origin") != "tool_result"
            or not isinstance(provenance.get("derivation_root_id"), str)
        ):
            return False
        signal_ids.add(signal_id)
    if not signal_ids:
        return False

    evidence_by_id: dict[str, Mapping[str, Any]] = {}
    for evidence in by_type.get("evidence_event", []):
        evidence_id = evidence.get("id")
        if (
            isinstance(evidence_id, str)
            and evidence.get("derived_from_signal") in signal_ids
            and evidence.get("epistemic_origin") == "tool_result"
            and evidence.get("discard_reason") is None
        ):
            evidence_by_id[evidence_id] = evidence
    if not evidence_by_id:
        return False

    updates = [
        item for item in by_type.get("belief_update", []) if item.get("cycle_id") == cycle_id
    ]
    linked_updates = 0
    for update in updates:
        evidence_id = update.get("evidence_id")
        if evidence_id not in evidence_by_id:
            return False
        sensitivity = update.get("sensitivity")
        caused_by = (
            sensitivity.get("caused_by_event_ids")
            if isinstance(sensitivity, Mapping)
            else None
        )
        if not isinstance(caused_by, list) or evidence_id not in caused_by:
            return False
        if update.get("prior") != update.get("posterior"):
            linked_updates += 1
    return linked_updates > 0


def _positive_int(value: object) -> bool:
    return type(value) is int and value > 0


def _read_jsonl(path: Path) -> list[dict[str, Any]] | None:
    if not path.is_file():
        return None
    records: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                return None
            records.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return records


def _try_read_object(path: Path) -> dict[str, Any] | None:
    try:
        return _read_object(path)
    except ValueError:
        return None


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
