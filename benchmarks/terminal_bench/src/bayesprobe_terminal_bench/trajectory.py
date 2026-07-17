from __future__ import annotations

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

from bayesprobe_terminal_bench.actions import ActionObservation, TerminalAction
from bayesprobe_terminal_bench.causal import (
    CausalActionRecord,
    canonical_sha256,
    executed_request_from_action,
)
from bayesprobe_terminal_bench.environment import ActionPolicy, PolicyViolation
from bayesprobe_terminal_bench.planning import _redact_sensitive_text


Arm = Literal["bayesprobe", "direct"]
_SIGNAL_SCHEMA_VERSION = "harbor-observation:v3"
_REDACTION_MARKER = "[REDACTED]"


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
    provider_tokens, model_calls, terminal_actions = _budget_counters(budget)

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
    prompt_tokens, completion_tokens = _provider_totals(artifact_root)
    final_metrics = FinalMetrics(
        total_prompt_tokens=prompt_tokens,
        total_completion_tokens=completion_tokens,
        total_steps=len(steps),
        extra={
            "provider_tokens_used": provider_tokens,
            "model_calls_used": model_calls,
            "terminal_actions_used": terminal_actions,
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
    observations = [
        ActionObservation.model_validate(record)
        for record in _read_jsonl(artifact_root / "environment_actions.jsonl")
    ]
    observation_indexes: set[int] = set()
    for observation in observations:
        if observation.action_index in observation_indexes:
            raise ValueError("duplicate executed action_index")
        observation_indexes.add(observation.action_index)

    causal_actions = [
        CausalActionRecord.model_validate(record)
        for record in _read_jsonl(artifact_root / "causal_actions.jsonl")
    ]
    if len(causal_actions) != len(observations):
        raise ValueError("executed and causal action cardinality differs")

    causal_by_index: dict[int, CausalActionRecord] = {}
    action_ids: set[str] = set()
    for record in causal_actions:
        action_index = record.observation.action_index
        if action_index in causal_by_index:
            raise ValueError("duplicate causal action_index")
        action_id = _nonempty_text(record.action_id, "action_id")
        if action_id in action_ids:
            raise ValueError("duplicate causal action_id")
        if record.step_index < 0:
            raise ValueError("causal step_index must be nonnegative")
        expected_fingerprint = _request_fingerprint(record.observation.action)
        if record.request_fingerprint != expected_fingerprint:
            raise ValueError("causal request fingerprint mismatch")
        if (
            record.pre_environment_state_id
            != record.observation.pre_environment_state_id
            or record.post_environment_state_id
            != record.observation.post_environment_state_id
        ):
            raise ValueError("causal environment state mismatch")
        causal_by_index[action_index] = record
        action_ids.add(action_id)

    if set(causal_by_index) != observation_indexes:
        raise ValueError("causal action lineage is unmatched")

    ledger = _read_jsonl(artifact_root / "bayesprobe_ledger.jsonl")
    signal_by_action = _signal_ids_by_action(
        ledger,
        known_action_ids=action_ids,
    )
    action_lineage: dict[str, dict[str, str]] = {}
    lineage_by_action: dict[str, dict[str, str]] = {}
    signal_ids: set[str] = set()
    steps: list[Step] = []

    for observation in observations:
        record = causal_by_index[observation.action_index]
        if record.observation != observation:
            raise ValueError("causal observation does not match executed action")
        action_id = _nonempty_text(record.action_id, "action_id")
        signal_id = signal_by_action.get(action_id) or _derived_signal_id(
            action_id=action_id,
            observation=observation,
        )
        if signal_id in signal_ids:
            raise ValueError("duplicate causal signal_id")
        signal_ids.add(signal_id)
        request_fingerprint = _request_fingerprint(observation.action)
        lineage = {
            "plan_id": _nonempty_text(record.plan_id, "plan_id"),
            "policy_attempt_id": _nonempty_text(
                record.policy_attempt_id,
                "policy_attempt_id",
            ),
            "probe_id": _nonempty_text(record.probe_id, "probe_id"),
            "action_id": action_id,
            "signal_id": signal_id,
            "request_fingerprint": request_fingerprint,
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
    planned_actions: dict[
        tuple[str, str],
        tuple[int, int, Mapping[str, Any]],
    ] = {}
    react_step_ids: set[str] = set()
    plan_position = 0
    for record in plans:
        step_number = _required_positive_int(record.get("step"), "react step")
        react_step_id = _required_text(record, "react_step_id")
        if react_step_id != f"react-step:{step_number}":
            raise ValueError("react_step_id does not match plan step")
        if react_step_id in react_step_ids:
            raise ValueError("duplicate react_step_id")
        react_step_ids.add(react_step_id)
        plan = _mapping(record.get("plan"), "direct plan")
        actions = plan.get("actions")
        if not isinstance(actions, Sequence) or isinstance(actions, str | bytes):
            raise ValueError("direct plan actions must be a sequence")
        for action in actions:
            safe_action = _mapping(action, "direct planned action")
            request_fingerprint = _required_fingerprint(safe_action)
            _validate_direct_plan_action(safe_action)
            key = (react_step_id, request_fingerprint)
            if key in planned_actions:
                raise ValueError("ambiguous direct plan action lineage")
            planned_actions[key] = (plan_position, step_number, safe_action)
            plan_position += 1

    cursor = -1
    action_indexes: set[int] = set()
    executed_lineage: set[tuple[str, str]] = set()
    steps: list[Step] = []
    for raw_observation in _read_jsonl(
        artifact_root / "environment_actions.jsonl"
    ):
        observation_payload = dict(raw_observation)
        react_step_id = _required_text(observation_payload, "react_step_id")
        request_fingerprint = _required_fingerprint(observation_payload)
        observation_payload.pop("react_step_id")
        observation_payload.pop("request_fingerprint")
        observation = ActionObservation.model_validate(observation_payload)
        action_index = observation.action_index
        if action_index in action_indexes:
            raise ValueError("duplicate direct action_index")
        action_indexes.add(action_index)

        key = (react_step_id, request_fingerprint)
        if key in executed_lineage:
            raise ValueError("duplicate direct executed lineage")
        executed_lineage.add(key)
        planned = planned_actions.get(key)
        if planned is None:
            raise ValueError("unmatched direct plan/action lineage")
        position, plan_step, safe_action = planned
        if position <= cursor:
            raise ValueError("direct action lineage is out of sequence")
        cursor = position
        _validate_direct_observation(
            safe_action=safe_action,
            observation=observation,
        )
        lineage = {
            "plan_id": f"react-plan:{plan_step}",
            "react_step_id": react_step_id,
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
    observation: ActionObservation,
    lineage: Mapping[str, str],
) -> Step:
    parsed_action = observation.action
    ActionPolicy().validate(parsed_action)
    arguments = parsed_action.model_dump(mode="json")
    action_id = lineage["action_id"]
    tool_call_id = f"tool:{action_id}"
    result_content = observation.model_facing_output
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
                        "action_index": observation.action_index,
                        "return_code": observation.return_code,
                        "timed_out": observation.timed_out,
                        "pre_environment_state_id": (
                            observation.pre_environment_state_id
                        ),
                        "post_environment_state_id": (
                            observation.post_environment_state_id
                        ),
                    },
                )
            ]
        ),
        extra={"kind": "terminal_action", **dict(lineage)},
    )


def _signal_ids_by_action(
    ledger: Sequence[Mapping[str, Any]],
    *,
    known_action_ids: set[str],
) -> dict[str, str]:
    result: dict[str, str] = {}
    action_by_signal: dict[str, str] = {}
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
        for ref in refs:
            if isinstance(ref, str) and ref.startswith(prefix):
                signal_id = _required_text(payload, "id")
                action_id = ref.removeprefix(prefix)
                if not action_id or action_id not in known_action_ids:
                    raise ValueError("external signal references an unknown action")
                if action_id in result:
                    raise ValueError("duplicate external signal action lineage")
                if signal_id in action_by_signal:
                    raise ValueError("duplicate external signal identity")
                result[action_id] = signal_id
                action_by_signal[signal_id] = action_id
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
    observation: ActionObservation,
) -> str:
    return "S_harbor_" + canonical_sha256(
        {
            "action_id": action_id,
            "full_output_sha256": observation.full_output_sha256,
            "schema_version": _SIGNAL_SCHEMA_VERSION,
        }
    )


def _provider_totals(artifact_root: Path) -> tuple[int | None, int | None]:
    prompt = 0
    completion = 0
    saw_prompt = False
    saw_completion = False
    for record in _read_jsonl(artifact_root / "provider_telemetry.jsonl"):
        usage = record.get("usage")
        if not isinstance(usage, Mapping):
            continue
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        if type(input_tokens) is int and input_tokens >= 0:
            prompt += input_tokens
            saw_prompt = True
        if type(output_tokens) is int and output_tokens >= 0:
            completion += output_tokens
            saw_completion = True
    return (
        prompt if saw_prompt else None,
        completion if saw_completion else None,
    )


def _budget_counters(budget: object | None) -> tuple[int, int, int]:
    if budget is None:
        raise ValueError("trajectory export requires the shared run budget")
    counters: list[int] = []
    for field in ("provider_tokens_used", "model_calls_used", "actions_used"):
        value = getattr(budget, field, None)
        if type(value) is not int or value < 0:
            raise ValueError(f"shared run budget has invalid {field}")
        counters.append(value)
    return counters[0], counters[1], counters[2]


def _request_fingerprint(action: TerminalAction) -> str:
    return "sha256:" + canonical_sha256(executed_request_from_action(action))


def _required_fingerprint(value: Mapping[str, Any]) -> str:
    fingerprint = _required_text(value, "request_fingerprint")
    digest = fingerprint.removeprefix("sha256:")
    if (
        not fingerprint.startswith("sha256:")
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("request_fingerprint must be canonical sha256 text")
    return fingerprint


def _validate_direct_plan_action(action: Mapping[str, Any]) -> None:
    action_type = action.get("type")
    expected_keys: set[str]
    if action_type == "shell":
        expected_keys = {
            "type",
            "command",
            "timeout_seconds",
            "mutates_environment",
            "request_fingerprint",
        }
        if (
            not isinstance(action.get("command"), str)
            or not action["command"]
            or type(action.get("timeout_seconds")) is not int
            or type(action.get("mutates_environment")) is not bool
        ):
            raise ValueError("invalid safe shell plan action")
    elif action_type == "write_file":
        expected_keys = {"type", "path", "request_fingerprint"}
        if not isinstance(action.get("path"), str) or not action["path"]:
            raise ValueError("invalid safe write plan action")
    elif action_type == "apply_patch":
        expected_keys = {"type", "strip", "request_fingerprint"}
        if type(action.get("strip")) is not int:
            raise ValueError("invalid safe patch plan action")
    else:
        raise ValueError("unsupported direct plan action")
    if set(action) != expected_keys:
        raise ValueError("direct plan action has unsafe or missing fields")


def _validate_direct_observation(
    *,
    safe_action: Mapping[str, Any],
    observation: ActionObservation,
) -> None:
    action = observation.action.model_dump(mode="json")
    action_type = safe_action["type"]
    if action.get("type") != action_type:
        raise ValueError("direct action type does not match its plan")
    if action_type == "shell":
        expected = {
            key: value
            for key, value in safe_action.items()
            if key != "request_fingerprint"
        }
        if action != expected:
            raise ValueError("direct shell action does not match its plan")
    elif action_type == "write_file":
        if (
            action.get("path") != safe_action.get("path")
            or action.get("content") != _REDACTION_MARKER
        ):
            raise ValueError("direct write action does not match safe telemetry")
    elif action_type == "apply_patch" and (
        action.get("strip") != safe_action.get("strip")
        or action.get("patch") != _REDACTION_MARKER
    ):
        raise ValueError("direct patch action does not match safe telemetry")


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


def _nonempty_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be non-empty text")
    return value


def _required_positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


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
