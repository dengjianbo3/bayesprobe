from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from harbor.models.trajectories import (
    Agent,
    FinalMetrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)
from harbor.utils.trajectory_validator import TrajectoryValidator
from pydantic import TypeAdapter

from bayesprobe_terminal_bench.actions import TerminalAction
from bayesprobe_terminal_bench.environment import ActionPolicy, PolicyViolation
from bayesprobe_terminal_bench.planning import _redact_sensitive_text


Arm = Literal["bayesprobe", "direct"]
_ACTION_ADAPTER = TypeAdapter(TerminalAction)
_SIGNAL_SCHEMA_VERSION = "harbor-observation:v3"


class TrajectoryExportError(RuntimeError):
    category = "adapter_error"


def write_atif_trajectory(
    *,
    logs_dir: str | Path,
    artifact_root: str | Path,
    arm: Arm,
    instruction: str,
    run_id: str,
    session_id: str | None,
    model_name: str | None,
    adapter_version: str,
    stop_reason: str,
    budget: object | None,
    restricted_values: tuple[str, ...] = (),
) -> Path:
    """Build, validate, and atomically publish one Harbor ATIF-v1.7 file."""
    try:
        trajectory = _build_trajectory(
            artifact_root=Path(artifact_root),
            arm=arm,
            instruction=instruction,
            run_id=run_id,
            session_id=session_id,
            model_name=model_name,
            adapter_version=adapter_version,
            stop_reason=stop_reason,
            budget=budget,
        )
        payload = _redact(
            trajectory.to_json_dict(),
            restricted_values=restricted_values,
        )
        validator = TrajectoryValidator()
        if not validator.validate(payload):
            raise TrajectoryExportError("Harbor ATIF validation failed")
        destination = Path(logs_dir) / "trajectory.json"
        _atomic_write(destination, payload)
        return destination
    except TrajectoryExportError:
        raise
    except PolicyViolation as error:
        raise TrajectoryExportError("trajectory action targets a protected path") from None
    except Exception:
        raise TrajectoryExportError("trajectory export failed") from None


def _build_trajectory(
    *,
    artifact_root: Path,
    arm: Arm,
    instruction: str,
    run_id: str,
    session_id: str | None,
    model_name: str | None,
    adapter_version: str,
    stop_reason: str,
    budget: object | None,
) -> Trajectory:
    if arm not in {"bayesprobe", "direct"}:
        raise ValueError("unsupported trajectory arm")

    steps = [Step(step_id=1, source="user", message=instruction)]
    if arm == "bayesprobe":
        steps.extend(_bayesprobe_steps(artifact_root, starting_at=2))
    else:
        steps.extend(_direct_steps(artifact_root, starting_at=2))

    trajectory_id = f"trajectory:{run_id}"
    steps.append(
        Step(
            step_id=len(steps) + 1,
            source="system",
            message="Trial terminated.",
            extra={
                "kind": "termination",
                "stop_reason": stop_reason,
                "artifact_id": trajectory_id,
            },
        )
    )
    prompt_tokens, completion_tokens, recorded_provider_tokens = _provider_totals(
        artifact_root
    )
    provider_tokens = _nonnegative_int(
        getattr(budget, "provider_tokens_used", None),
        fallback=recorded_provider_tokens,
    )
    final_metrics = FinalMetrics(
        total_prompt_tokens=prompt_tokens,
        total_completion_tokens=completion_tokens,
        total_steps=len(steps),
        extra={
            "provider_tokens_used": provider_tokens,
            "model_calls_used": _nonnegative_int(
                getattr(budget, "model_calls_used", None)
            ),
            "terminal_actions_used": _nonnegative_int(
                getattr(budget, "actions_used", None)
            ),
        },
    )
    return Trajectory(
        schema_version="ATIF-v1.7",
        session_id=session_id,
        trajectory_id=trajectory_id,
        agent=Agent(
            name="bayesprobe" if arm == "bayesprobe" else "bayesprobe-direct",
            version=adapter_version,
            model_name=model_name,
        ),
        steps=steps,
        final_metrics=final_metrics,
        extra={
            "experiment_id": "terminal_bench_causal:v1",
            "artifact_schema": "terminal:v1",
            "arm": arm,
        },
    )


def _bayesprobe_steps(artifact_root: Path, *, starting_at: int) -> list[Step]:
    causal_actions = _read_jsonl(artifact_root / "causal_actions.jsonl")
    ledger = _read_jsonl(artifact_root / "bayesprobe_ledger.jsonl")
    signal_by_action = _signal_ids_by_action(ledger)
    action_lineage: dict[str, dict[str, str]] = {}
    lineage_by_action: dict[str, dict[str, str]] = {}
    steps: list[Step] = []

    for record in causal_actions:
        observation = _mapping(record.get("observation"), "causal action observation")
        action_id = _required_text(record, "action_id")
        signal_id = signal_by_action.get(action_id) or _derived_signal_id(
            action_id=action_id,
            observation=observation,
        )
        lineage = {
            "plan_id": _required_text(record, "plan_id"),
            "policy_attempt_id": _required_text(record, "policy_attempt_id"),
            "probe_id": _required_text(record, "probe_id"),
            "action_id": action_id,
            "signal_id": signal_id,
            "request_fingerprint": _required_text(record, "request_fingerprint"),
        }
        action_lineage[signal_id] = lineage
        lineage_by_action[action_id] = lineage
        steps.append(
            _action_step(
                step_id=starting_at + len(steps),
                observation=observation,
                lineage=lineage,
            )
        )

    for decision in _read_jsonl(artifact_root / "causal_decisions.jsonl"):
        decision_kind = decision.get("decision")
        if decision_kind not in {"admit", "discard"}:
            continue
        signal_id = _optional_text(decision.get("signal_id"))
        action_id = _optional_text(decision.get("action_id"))
        lineage = (
            action_lineage.get(signal_id, {}) if signal_id is not None else {}
        ) or (
            lineage_by_action.get(action_id, {}) if action_id is not None else {}
        )
        extra = {
            "kind": "causal_evidence_decision",
            "causal_decision": decision_kind,
            "reason_code": _required_text(decision, "reason_code"),
        }
        if signal_id is not None:
            extra["signal_id"] = signal_id
        if action_id is not None:
            extra["action_id"] = action_id
        if "probe_id" in lineage:
            extra["probe_id"] = lineage["probe_id"]
        steps.append(
            Step(
                step_id=starting_at + len(steps),
                source="agent",
                message="Causal evidence route evaluated.",
                llm_call_count=0,
                extra=extra,
            )
        )

    updates_by_evidence = _updates_by_evidence(ledger)
    for record in ledger:
        if record.get("record_type") != "evidence_event":
            continue
        payload = _mapping(record.get("payload"), "evidence payload")
        evidence_id = _required_text(payload, "id")
        signal_id = _required_text(payload, "derived_from_signal")
        lineage = action_lineage.get(signal_id, {})
        extra: dict[str, Any] = {
            "kind": "bayesprobe_transition",
            "evidence_id": evidence_id,
            "evidence_status": (
                "discarded" if payload.get("discard_reason") is not None else "accepted"
            ),
            "signal_id": signal_id,
            "update_ids": updates_by_evidence.get(evidence_id, []),
        }
        if "probe_id" in lineage:
            extra["probe_id"] = lineage["probe_id"]
        steps.append(
            Step(
                step_id=starting_at + len(steps),
                timestamp=_optional_text(record.get("recorded_at")),
                source="agent",
                message="BayesProbe evidence state recorded.",
                llm_call_count=0,
                extra=extra,
            )
        )
    return steps


def _direct_steps(artifact_root: Path, *, starting_at: int) -> list[Step]:
    plans = _read_jsonl(artifact_root / "plans.jsonl")
    planned_actions: list[tuple[int, Mapping[str, Any]]] = []
    for record in plans:
        step_number = _positive_int(record.get("step"), fallback=len(planned_actions) + 1)
        plan = record.get("plan")
        if not isinstance(plan, Mapping):
            continue
        actions = plan.get("actions")
        if not isinstance(actions, Sequence) or isinstance(actions, str | bytes):
            continue
        planned_actions.extend(
            (step_number, action) for action in actions if isinstance(action, Mapping)
        )

    cursor = 0
    steps: list[Step] = []
    for observation in _read_jsonl(artifact_root / "environment_actions.jsonl"):
        action = _mapping(observation.get("action"), "direct action")
        plan_step: int | None = None
        for index in range(cursor, len(planned_actions)):
            candidate_step, candidate_action = planned_actions[index]
            if _actions_match(candidate_action, action):
                plan_step = candidate_step
                cursor = index + 1
                break
        action_index = _positive_int(observation.get("action_index"), fallback=len(steps) + 1)
        request_fingerprint = "sha256:" + _canonical_sha256(action)
        lineage = {
            "plan_id": f"react-plan:{plan_step or 'unmatched'}",
            "policy_attempt_id": f"react-policy:{plan_step or 'unmatched'}",
            "probe_id": f"react-step:{plan_step or 'unmatched'}",
            "action_id": f"react-action:{action_index}",
            "signal_id": f"react-signal:{action_index}",
            "request_fingerprint": request_fingerprint,
        }
        steps.append(
            _action_step(
                step_id=starting_at + len(steps),
                observation=observation,
                lineage=lineage,
            )
        )
    return steps


def _action_step(
    *,
    step_id: int,
    observation: Mapping[str, Any],
    lineage: Mapping[str, str],
) -> Step:
    action = _mapping(observation.get("action"), "terminal action")
    parsed_action = _ACTION_ADAPTER.validate_python(action)
    ActionPolicy().validate(parsed_action)
    arguments = parsed_action.model_dump(mode="json")
    action_id = lineage["action_id"]
    tool_call_id = f"tool:{action_id}"
    result_content = observation.get("model_facing_output")
    if not isinstance(result_content, str):
        result_content = json.dumps(
            {
                "return_code": observation.get("return_code"),
                "stderr": observation.get("stderr", ""),
                "stdout": observation.get("stdout", ""),
                "timed_out": observation.get("timed_out", False),
            },
            sort_keys=True,
        )
    return Step(
        step_id=step_id,
        source="agent",
        message="Execute terminal action.",
        llm_call_count=0,
        tool_calls=[
            ToolCall(
                tool_call_id=tool_call_id,
                function_name=f"terminal.{arguments['type']}",
                arguments=arguments,
            )
        ],
        observation=Observation(
            results=[
                ObservationResult(
                    source_call_id=tool_call_id,
                    content=result_content,
                    extra={
                        "action_index": _positive_int(
                            observation.get("action_index"), fallback=1
                        ),
                        "return_code": observation.get("return_code"),
                        "timed_out": bool(observation.get("timed_out", False)),
                        "pre_environment_state_id": observation.get(
                            "pre_environment_state_id"
                        ),
                        "post_environment_state_id": observation.get(
                            "post_environment_state_id"
                        ),
                    },
                )
            ]
        ),
        extra={"kind": "terminal_action", **dict(lineage)},
    )


def _signal_ids_by_action(
    ledger: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    result: dict[str, str] = {}
    prefix = "causal_actions.jsonl#"
    for record in ledger:
        if record.get("record_type") != "external_signal":
            continue
        payload = record.get("payload")
        if not isinstance(payload, Mapping):
            continue
        provenance = payload.get("provenance")
        if not isinstance(provenance, Mapping):
            continue
        refs = provenance.get("artifact_refs")
        if not isinstance(refs, Sequence) or isinstance(refs, str | bytes):
            continue
        signal_id = payload.get("id")
        if not isinstance(signal_id, str) or not signal_id:
            continue
        for ref in refs:
            if isinstance(ref, str) and ref.startswith(prefix):
                result[ref.removeprefix(prefix)] = signal_id
    return result


def _updates_by_evidence(
    ledger: Sequence[Mapping[str, Any]],
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for record in ledger:
        if record.get("record_type") != "belief_update":
            continue
        payload = record.get("payload")
        if not isinstance(payload, Mapping):
            continue
        update_id = payload.get("update_id")
        sensitivity = payload.get("sensitivity")
        if not isinstance(update_id, str) or not isinstance(sensitivity, Mapping):
            continue
        causes = sensitivity.get("caused_by_event_ids")
        if not isinstance(causes, Sequence) or isinstance(causes, str | bytes):
            continue
        for evidence_id in causes:
            if isinstance(evidence_id, str):
                result.setdefault(evidence_id, []).append(update_id)
    return result


def _derived_signal_id(
    *,
    action_id: str,
    observation: Mapping[str, Any],
) -> str:
    return "S_harbor_" + _canonical_sha256(
        {
            "action_id": action_id,
            "full_output_sha256": _required_text(observation, "full_output_sha256"),
            "schema_version": _SIGNAL_SCHEMA_VERSION,
        }
    )


def _provider_totals(artifact_root: Path) -> tuple[int | None, int | None, int]:
    prompt = 0
    completion = 0
    provider = 0
    saw_prompt = False
    saw_completion = False
    for record in _read_jsonl(artifact_root / "provider_telemetry.jsonl"):
        usage = record.get("usage")
        if not isinstance(usage, Mapping):
            continue
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        total_tokens = usage.get("total_tokens")
        if type(input_tokens) is int and input_tokens >= 0:
            prompt += input_tokens
            saw_prompt = True
        if type(output_tokens) is int and output_tokens >= 0:
            completion += output_tokens
            saw_completion = True
        if type(total_tokens) is int and total_tokens >= 0:
            provider += total_tokens
    return (
        prompt if saw_prompt else None,
        completion if saw_completion else None,
        provider,
    )


def _actions_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    action_type = left.get("type")
    if action_type != right.get("type"):
        return False
    if action_type == "shell":
        return left.get("command") == right.get("command")
    if action_type == "write_file":
        return left.get("path") == right.get("path")
    if action_type == "apply_patch":
        return left.get("strip", 0) == right.get("strip", 0)
    return False


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    if not path.is_file():
        return []
    records: list[Mapping[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError("JSONL record must be an object")
        records.append(payload)
    return records


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _required_text(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ValueError(f"{key} must be non-empty text")
    return item


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _positive_int(value: object, *, fallback: int) -> int:
    return value if type(value) is int and value > 0 else fallback


def _nonnegative_int(value: object, *, fallback: int = 0) -> int:
    return value if type(value) is int and value >= 0 else fallback


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _redact(value: Any, *, restricted_values: tuple[str, ...]) -> Any:
    restricted = tuple(
        sorted({item for item in restricted_values if item}, key=lambda item: -len(item))
    )
    if isinstance(value, str):
        for item in restricted:
            value = value.replace(item, "[REDACTED]")
        return _redact_sensitive_text(value)
    if isinstance(value, Mapping):
        return {
            _redact(str(key), restricted_values=restricted): _redact(
                item, restricted_values=restricted
            )
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_redact(item, restricted_values=restricted) for item in value]
    return value


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".trajectory-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
