from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import TypeAdapter, ValidationError

from bayesprobe_terminal_bench.actions import TerminalAction
from bayesprobe_terminal_bench.runner_factory import (
    terminal_bench_lock_schema_mismatches,
)
from bayesprobe_terminal_bench.signals import executed_request_from_action
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
_HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_TERMINAL_ACTION_ADAPTER = TypeAdapter(TerminalAction)
_SIGNAL_SCHEMA_VERSION = "harbor-observation:v2"
_MAX_OBSERVATION_BYTES = 32_768


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
    records = _read_jsonl(bayesprobe_dir / "bayesprobe_ledger.jsonl")
    if not records:
        return False
    errors_path = bayesprobe_dir / "errors.jsonl"
    errors = _read_jsonl(errors_path) if errors_path.is_file() else []
    if errors is None:
        return False
    reserved_without_observation = _reserved_no_observation_indices(errors)
    if reserved_without_observation is None:
        return False
    actions_path = bayesprobe_dir / "environment_actions.jsonl"
    actions = _read_jsonl(actions_path) if actions_path.is_file() else []
    if actions is None:
        return False
    actions_by_index = _actions_by_index(actions)
    if (
        actions_by_index is None
        or set(actions_by_index) & reserved_without_observation
        or set(actions_by_index) | reserved_without_observation
        != set(range(1, int(summary["terminal_actions"]) + 1))
    ):
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
    cycle_ids = {cycle.get("cycle_id") for cycle in cycles}
    if any(not isinstance(cycle_id, str) for cycle_id in cycle_ids):
        return False
    belief_state_cycles = {
        state.get("cycle_id") for state in by_type.get("belief_state", [])
    }
    if not cycle_ids.issubset(belief_state_cycles):
        return False
    if not _epistemic_links_are_valid(by_type=by_type, cycle_ids=cycle_ids):
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


def _reserved_no_observation_indices(
    errors: Sequence[Mapping[str, Any]],
) -> set[int] | None:
    reserved: set[int] = set()
    for error in errors:
        if error.get("category") != "policy_error":
            continue
        action_index = error.get("action_index")
        if (
            type(action_index) is not int
            or action_index < 1
            or action_index in reserved
            or error.get("error_type") != "PolicyViolation"
            or not isinstance(error.get("probe_id"), str)
        ):
            return None
        reserved.add(action_index)
    return reserved


def _epistemic_links_are_valid(
    *,
    by_type: Mapping[str, list[Mapping[str, Any]]],
    cycle_ids: set[object],
) -> bool:
    signals_by_id: dict[str, Mapping[str, Any]] = {}
    for signal in by_type.get("external_signal", []):
        signal_id = signal.get("id")
        if (
            not isinstance(signal_id, str)
            or not signal_id
            or signal_id in signals_by_id
            or signal.get("cycle_id") not in cycle_ids
            or signal.get("inbox_status", "accepted") != "accepted"
        ):
            return False
        signals_by_id[signal_id] = signal

    evidence_by_id: dict[str, Mapping[str, Any]] = {}
    evidence_by_signal: dict[str, list[Mapping[str, Any]]] = {}
    for evidence in by_type.get("evidence_event", []):
        evidence_id = evidence.get("id")
        signal_id = evidence.get("derived_from_signal")
        signal = signals_by_id.get(signal_id) if isinstance(signal_id, str) else None
        provenance = signal.get("provenance") if isinstance(signal, Mapping) else None
        if (
            not isinstance(evidence_id, str)
            or not evidence_id
            or evidence_id in evidence_by_id
            or signal is None
            or not isinstance(provenance, Mapping)
            or evidence.get("epistemic_origin") != "tool_result"
            or evidence.get("derivation_root_id")
            != provenance.get("derivation_root_id")
        ):
            return False
        evidence_by_id[evidence_id] = evidence
        evidence_by_signal.setdefault(signal_id, []).append(evidence)

    if set(evidence_by_signal) != set(signals_by_id):
        return False

    for update in by_type.get("belief_update", []):
        evidence_id = update.get("evidence_id")
        evidence = evidence_by_id.get(evidence_id) if isinstance(evidence_id, str) else None
        if evidence is None or evidence.get("discard_reason") is not None:
            return False
        signal = signals_by_id.get(evidence.get("derived_from_signal"))
        if (
            signal is None
            or update.get("cycle_id") != signal.get("cycle_id")
            or not _update_is_consistent(update, evidence_id)
        ):
            return False
    return True


def _update_is_consistent(update: Mapping[str, Any], evidence_id: str) -> bool:
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
        or direction not in {"strengthened", "weakened", "neutral"}
    ):
        return False
    return (
        (direction == "strengthened" and posterior > prior)
        or (direction == "weakened" and posterior < prior)
        or (direction == "neutral" and posterior == prior)
    )


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
    signals = [
        item
        for item in by_type.get("external_signal", [])
        if item.get("cycle_id") == cycle_id
        and item.get("inbox_status", "accepted") == "accepted"
    ]
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
    try:
        terminal_action = _TERMINAL_ACTION_ADAPTER.validate_python(action["action"])
    except ValidationError:
        return None
    executed_request = executed_request_from_action(terminal_action)
    environment_digest = _canonical_digest(
        {
            "run_id": run_id,
            "schema_version": _SIGNAL_SCHEMA_VERSION,
            "post_environment_state_id": action["post_environment_state_id"],
        }
    )
    source_identity = f"harbor-terminal:sha256:{environment_digest}"
    raw_content = signal.get("raw_content")
    if (
        provenance.get("source_identity") != source_identity
        or provenance.get("correlation_group")
        != f"harbor-env:sha256:{environment_digest}"
        or not isinstance(raw_content, str)
        or provenance.get("canonical_content_fingerprint")
        != _canonical_content_fingerprint(source_identity, raw_content)
    ):
        return None
    composite_root = "harbor-action:sha256:" + _canonical_digest(
        {
            "action_index": action_index,
            "cycle_id": cycle_id,
            "executed_request": executed_request,
            "full_output_sha256": full_output_sha256,
            "post_environment_state_id": action["post_environment_state_id"],
            "pre_environment_state_id": action["pre_environment_state_id"],
            "probe_id": probe_id,
            "run_id": run_id,
            "schema_version": _SIGNAL_SCHEMA_VERSION,
        }
    )
    if provenance.get("derivation_root_id") != composite_root:
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
    try:
        terminal_action = _TERMINAL_ACTION_ADAPTER.validate_python(nested_action)
    except ValidationError:
        return False
    observation = action.get("model_facing_output")
    if (
        not isinstance(observation, str)
        or len(observation.encode("utf-8")) > _MAX_OBSERVATION_BYTES
    ):
        return False
    expected = {
        "action_index": action.get("action_index"),
        "error_category": action.get("error_category"),
        "executed_request": executed_request_from_action(terminal_action),
        "observation": observation,
        "output_truncated": action.get("output_truncated"),
        "post_environment_state_id": action.get("post_environment_state_id"),
        "pre_environment_state_id": action.get("pre_environment_state_id"),
        "return_code": action.get("return_code"),
        "timed_out": action.get("timed_out"),
    }
    return dict(payload) == expected


def _canonical_content_fingerprint(source_identity: str, raw_content: str) -> str:
    canonical_content = " ".join(
        unicodedata.normalize("NFKC", raw_content).split()
    )
    digest = hashlib.sha256(
        f"{source_identity}\n{canonical_content}".encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


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
