from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from bayesprobe_terminal_bench.runner_factory import (
    terminal_bench_lock_schema_mismatches,
)
from write_benchmark_lock import (
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
_HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$")


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
    return not terminal_bench_lock_schema_mismatches(lock)


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
    actions_by_index = _actions_by_index(actions)
    if actions_by_index is None:
        return False
    if len(actions_by_index) != summary.get("terminal_actions"):
        return False

    by_type: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        record_type = record.get("record_type")
        payload = record.get("payload")
        if not isinstance(record_type, str) or not isinstance(payload, Mapping):
            return False
        by_type.setdefault(record_type, []).append(payload)

    runs = by_type.get("run", [])
    run_ids = {
        run.get("run_id")
        for run in runs
        if isinstance(run.get("run_id"), str) and run.get("run_id")
    }
    if len(run_ids) != 1 or not any(
        run.get("status") == "completed" for run in runs
    ):
        return False
    run_id = run_ids.pop()
    if any(run.get("run_id") != run_id for run in runs):
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
    cycle_indices = {cycle.get("cycle_index") for cycle in cycles}
    if cycle_indices != set(range(1, len(cycles) + 1)):
        return False
    if not by_type.get("belief_state"):
        return False

    used_action_indices: set[int] = set()
    for cycle in cycles:
        cycle_id = cycle.get("cycle_id")
        if not isinstance(cycle_id, str) or cycle.get("run_id") != run_id:
            return False
        linked_actions = _cycle_linked_actions(
            cycle_id=cycle_id,
            run_id=run_id,
            actions_by_index=actions_by_index,
            by_type=by_type,
        )
        if linked_actions is None or used_action_indices & linked_actions:
            return False
        used_action_indices.update(linked_actions)
    return used_action_indices == set(actions_by_index)


def _actions_by_index(
    actions: Sequence[Mapping[str, Any]],
) -> dict[int, Mapping[str, Any]] | None:
    by_index: dict[int, Mapping[str, Any]] = {}
    for action in actions:
        action_index = action.get("action_index")
        nested_action = action.get("action")
        if (
            type(action_index) is not int
            or action_index < 1
            or action_index in by_index
            or not isinstance(nested_action, Mapping)
            or not isinstance(nested_action.get("type"), str)
            or not isinstance(action.get("pre_environment_state_id"), str)
            or not isinstance(action.get("post_environment_state_id"), str)
            or not isinstance(action.get("full_output_sha256"), str)
            or not _HEX_DIGEST.fullmatch(action["full_output_sha256"])
        ):
            return None
        by_index[action_index] = action
    return by_index


def _cycle_linked_actions(
    *,
    cycle_id: str,
    run_id: str,
    actions_by_index: Mapping[int, Mapping[str, Any]],
    by_type: Mapping[str, list[Mapping[str, Any]]],
) -> set[int] | None:
    probe_sets = [
        item for item in by_type.get("probe_set", []) if item.get("cycle_id") == cycle_id
    ]
    if len(probe_sets) != 1:
        return None
    probes = probe_sets[0].get("probes")
    if not isinstance(probes, list):
        return None
    probe_ids = {
        item.get("id")
        for item in probes
        if isinstance(item, Mapping)
        and item.get("cycle_id") == cycle_id
        and isinstance(item.get("id"), str)
    }
    if not probe_ids:
        return None

    signals = [
        item
        for item in by_type.get("external_signal", [])
        if item.get("cycle_id") == cycle_id
        and item.get("inbox_status", "accepted") == "accepted"
    ]
    if not signals:
        return None
    used_action_indices: set[int] = set()
    for signal in signals:
        action_index = _signal_action_index(
            signal=signal,
            cycle_id=cycle_id,
            run_id=run_id,
            probe_ids=probe_ids,
            actions_by_index=actions_by_index,
        )
        if action_index is None or action_index in used_action_indices:
            return None
        if not _signal_has_directional_evidence(signal, cycle_id, by_type):
            return None
        used_action_indices.add(action_index)
    return used_action_indices


def _signal_action_index(
    *,
    signal: Mapping[str, Any],
    cycle_id: str,
    run_id: str,
    probe_ids: set[object],
    actions_by_index: Mapping[int, Mapping[str, Any]],
) -> int | None:
    signal_id = signal.get("id")
    probe_id = signal.get("generated_by_probe")
    provenance = signal.get("provenance")
    if (
        not isinstance(signal_id, str)
        or probe_id not in probe_ids
        or not isinstance(probe_id, str)
        or not isinstance(provenance, Mapping)
        or provenance.get("epistemic_origin") != "tool_result"
    ):
        return None
    artifact_refs = provenance.get("artifact_refs")
    if not isinstance(artifact_refs, list) or len(artifact_refs) != 1:
        return None
    prefix = "environment_actions.jsonl#"
    artifact_ref = artifact_refs[0]
    if not isinstance(artifact_ref, str) or not artifact_ref.startswith(prefix):
        return None
    index_text = artifact_ref.removeprefix(prefix)
    if not index_text.isdigit():
        return None
    action_index = int(index_text)
    action = actions_by_index.get(action_index)
    if action is None:
        return None
    if provenance.get("environment_state_id") != action.get(
        "post_environment_state_id"
    ):
        return None
    if not _signal_content_matches_action(signal, action):
        return None

    full_output_sha256 = action["full_output_sha256"]
    direct_root = f"harbor-action:sha256:{full_output_sha256}"
    composite_root = "harbor-action:sha256:" + _canonical_digest(
        {
            "action": action["action"],
            "action_index": action_index,
            "cycle_id": cycle_id,
            "full_output_sha256": full_output_sha256,
            "post_environment_state_id": action["post_environment_state_id"],
            "pre_environment_state_id": action["pre_environment_state_id"],
            "probe_id": probe_id,
            "run_id": run_id,
            "schema_version": "harbor-observation:v1",
        }
    )
    if provenance.get("derivation_root_id") not in {direct_root, composite_root}:
        return None
    return action_index


def _signal_content_matches_action(
    signal: Mapping[str, Any],
    action: Mapping[str, Any],
) -> bool:
    raw_content = signal.get("raw_content")
    if not isinstance(raw_content, str):
        return False
    try:
        payload = json.loads(raw_content)
    except json.JSONDecodeError:
        return False
    nested_action = action.get("action")
    if not isinstance(payload, Mapping) or not isinstance(nested_action, Mapping):
        return False
    expected = {
        "action_index": action.get("action_index"),
        "action_type": nested_action.get("type"),
        "error_category": action.get("error_category"),
        "model_facing_output": action.get("model_facing_output"),
        "output_truncated": action.get("output_truncated"),
        "post_environment_state_id": action.get("post_environment_state_id"),
        "pre_environment_state_id": action.get("pre_environment_state_id"),
        "return_code": action.get("return_code"),
        "timed_out": action.get("timed_out"),
    }
    return dict(payload) == expected


def _signal_has_directional_evidence(
    signal: Mapping[str, Any],
    cycle_id: str,
    by_type: Mapping[str, list[Mapping[str, Any]]],
) -> bool:
    signal_id = signal.get("id")
    provenance = signal.get("provenance")
    if not isinstance(signal_id, str) or not isinstance(provenance, Mapping):
        return False
    derivation_root = provenance.get("derivation_root_id")
    if not isinstance(derivation_root, str) or not derivation_root:
        return False
    signal_evidence = [
        evidence
        for evidence in by_type.get("evidence_event", [])
        if evidence.get("derived_from_signal") == signal_id
    ]
    if not signal_evidence:
        return False
    evidence_ids: set[str] = set()
    for evidence in signal_evidence:
        evidence_id = evidence.get("id")
        if (
            not isinstance(evidence_id, str)
            or not evidence_id
            or evidence_id in evidence_ids
            or evidence.get("epistemic_origin") != "tool_result"
            or evidence.get("discard_reason") is not None
            or evidence.get("derivation_root_id") != derivation_root
        ):
            return False
        evidence_ids.add(evidence_id)
    linked_updates = [
        update
        for update in by_type.get("belief_update", [])
        if update.get("cycle_id") == cycle_id
        and update.get("evidence_id") in evidence_ids
    ]
    if not linked_updates:
        return False
    for update in linked_updates:
        evidence_id = update.get("evidence_id")
        sensitivity = update.get("sensitivity")
        caused_by = (
            sensitivity.get("caused_by_event_ids")
            if isinstance(sensitivity, Mapping)
            else None
        )
        prior = update.get("prior")
        posterior = update.get("posterior")
        direction = update.get("direction")
        if (
            not isinstance(caused_by, list)
            or evidence_id not in caused_by
            or type(prior) not in (int, float)
            or type(posterior) not in (int, float)
            or direction not in {"strengthened", "weakened"}
            or (direction == "strengthened" and posterior <= prior)
            or (direction == "weakened" and posterior >= prior)
        ):
            return False
    return True


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


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
