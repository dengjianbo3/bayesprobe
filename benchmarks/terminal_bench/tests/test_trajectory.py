from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from harbor.models.agent.context import AgentContext
from harbor.utils.trajectory_validator import TrajectoryValidator
from pydantic import TypeAdapter

from bayesprobe_terminal_bench.actions import TerminalAction
from bayesprobe_terminal_bench.agent import BayesProbeHarborAgent, BayesProbeHarborAgentError
from bayesprobe_terminal_bench.artifacts import TrialArtifactStore
from bayesprobe_terminal_bench.causal import (
    canonical_sha256,
    executed_request_from_action,
)
from bayesprobe_terminal_bench.direct_agent import DirectHarborAgent, DirectHarborAgentError
from bayesprobe_terminal_bench.provider_contract import ProviderContractError
from bayesprobe_terminal_bench.trajectory import (
    TrajectoryExportError,
    write_atif_trajectory,
)


_SECRET = "trajectory-secret-value"
_ACTION_ADAPTER = TypeAdapter(TerminalAction)
_DEFAULT_BUDGET = object()
_EXTRA_ENV = {
    "BAYESPROBE_BENCH_API_KEY": _SECRET,
    "BAYESPROBE_BENCH_MODEL": "test-model",
    "BAYESPROBE_BENCH_TASK_TIMEOUT_SECONDS": "900",
}


def _append(path: Path, filename: str, *payloads: object) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with (path / filename).open("a", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _read_records(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _overwrite_records(path: Path, payloads: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(payload, sort_keys=True) + "\n" for payload in payloads),
        encoding="utf-8",
    )


def _request_fingerprint(action: object) -> str:
    parsed = _ACTION_ADAPTER.validate_python(action)
    return "sha256:" + canonical_sha256(executed_request_from_action(parsed))


def _observation(
    *,
    command: str = "pytest -q",
    action_index: int = 1,
    action: dict[str, object] | None = None,
) -> dict[str, object]:
    if action is None:
        action = {
            "type": "shell",
            "command": command,
            "timeout_seconds": 120,
            "mutates_environment": False,
        }
    return {
        "action_index": action_index,
        "action": action,
        "stdout": "1 passed",
        "stderr": "",
        "return_code": 0,
        "timed_out": False,
        "error_category": None,
        "duration_ms": 12,
        "pre_environment_state_id": "env:0",
        "post_environment_state_id": "env:0",
        "full_output_sha256": "a" * 64,
        "model_facing_output": "1 passed",
        "output_truncated": False,
    }


def _causal_action(
    observation: dict[str, object],
    *,
    action_id: str = "action-1",
) -> dict[str, object]:
    return {
        "run_id": "run-1",
        "cycle_id": "cycle-1",
        "probe_id": "probe-1",
        "plan_id": "plan-1",
        "policy_attempt_id": "policy-1",
        "action_id": action_id,
        "step_index": 0,
        "action_role": "verify",
        "request_fingerprint": _request_fingerprint(observation["action"]),
        "pre_environment_state_id": "env:0",
        "post_environment_state_id": "env:0",
        "subject_environment_state_id": "env:0",
        "intervention_generation": 0,
        "verification_target": "public tests pass",
        "transition_predictions": {},
        "observation": observation,
    }


def _direct_action_artifact(action: dict[str, object]) -> dict[str, object]:
    fingerprint = _request_fingerprint(action)
    action_type = action["type"]
    if action_type == "shell":
        return {**action, "request_fingerprint": fingerprint}
    if action_type == "write_file":
        return {
            "type": "write_file",
            "path": action["path"],
            "request_fingerprint": fingerprint,
        }
    if action_type == "apply_patch":
        return {
            "type": "apply_patch",
            "strip": action["strip"],
            "request_fingerprint": fingerprint,
        }
    raise AssertionError("unsupported test action")


def _direct_observation(
    action: dict[str, object],
    *,
    action_index: int,
    react_step_id: str,
) -> dict[str, object]:
    observation = _observation(action_index=action_index, action=copy.deepcopy(action))
    if action["type"] == "write_file":
        observation["action"] = {
            "type": "write_file",
            "path": action["path"],
            "content": "[REDACTED]",
        }
    elif action["type"] == "apply_patch":
        observation["action"] = {
            "type": "apply_patch",
            "patch": "[REDACTED]",
            "strip": action["strip"],
        }
    observation["react_step_id"] = react_step_id
    observation["request_fingerprint"] = _request_fingerprint(action)
    return observation


def _seed_bayesprobe_artifacts(root: Path) -> None:
    observation = _observation()
    _append(
        root,
        "plans.jsonl",
        {
            "cycle_id": "cycle-1",
            "plan_id": "plan-1",
            "policy_attempt_id": "policy-1",
            "probe_id": "probe-1",
            "plan": {
                "mode": "verify",
                "steps": [
                    {
                        "role": "verify",
                        "action": observation["action"],
                        "verification_target": "public tests pass",
                    }
                ],
                "expected_observation": "tests pass",
                "transition_predictions": [],
            },
        },
    )
    _append(root, "environment_actions.jsonl", observation)
    _append(root, "causal_actions.jsonl", _causal_action(observation))
    _append(
        root,
        "causal_decisions.jsonl",
        {
            "signal_id": "signal-1",
            "action_id": "action-1",
            "action_role": "verify",
            "decision": "admit",
            "reason_code": "verified_postcondition",
            "subject_environment_state_id": "env:0",
            "judgment_response_sha256": "sha256:" + "c" * 64,
        },
        {
            "signal_id": "signal-discarded",
            "action_id": "action-1",
            "action_role": "verify",
            "decision": "discard",
            "reason_code": "stale_state",
            "subject_environment_state_id": "env:0",
            "judgment_response_sha256": "sha256:" + "d" * 64,
        },
    )
    _append(
        root,
        "bayesprobe_ledger.jsonl",
        {
            "record_type": "external_signal",
            "recorded_at": "2026-07-17T00:00:00+00:00",
            "payload": {
                "id": "signal-1",
                "generated_by_probe": "probe-1",
                "provenance": {
                    "artifact_refs": ["causal_actions.jsonl#action-1"]
                },
            },
        },
        {
            "record_type": "evidence_event",
            "recorded_at": "2026-07-17T00:00:01+00:00",
            "payload": {
                "id": "evidence-1",
                "derived_from_signal": "signal-1",
                "discard_reason": None,
                "target_hypotheses": ["H1"],
            },
        },
        {
            "record_type": "belief_update",
            "recorded_at": "2026-07-17T00:00:02+00:00",
            "payload": {
                "cycle_id": "cycle-1",
                "evidence_id": "evidence-root-1",
                "hypothesis_id": "H1",
                "update_id": "update-1",
                "sensitivity": {"caused_by_event_ids": ["evidence-1"]},
            },
        },
    )
    _append(
        root,
        "provider_telemetry.jsonl",
        {
            "task": "terminal_probe_plan",
            "outcome": "success",
            "usage": {"input_tokens": 7, "output_tokens": 5, "total_tokens": 12},
        },
        {
            "task": "judge_evidence",
            "outcome": "success",
            "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        },
    )


def _seed_direct_artifacts(root: Path) -> None:
    action = {
        "type": "shell",
        "command": "python -m pytest -q",
        "timeout_seconds": 120,
        "mutates_environment": False,
    }
    observation = _direct_observation(
        action,
        action_index=1,
        react_step_id="react-step:1",
    )
    _append(
        root,
        "plans.jsonl",
        {
            "step": 1,
            "react_step_id": "react-step:1",
            "plan": {
                "actions": [_direct_action_artifact(action)],
                "done": False,
                "completion_summary": None,
            },
        },
        {
            "step": 2,
            "react_step_id": "react-step:2",
            "plan": {
                "actions": [],
                "done": True,
                "completion_summary": "verified",
            },
        },
    )
    _append(root, "environment_actions.jsonl", observation)
    _append(
        root,
        "provider_telemetry.jsonl",
        {
            "task": "react_step",
            "outcome": "success",
            "usage": {"input_tokens": 11, "output_tokens": 6, "total_tokens": 17},
        },
    )


def _seed_direct_action_sequence(
    root: Path,
    *,
    planned_actions: list[dict[str, object]],
    executed_actions: list[dict[str, object]] | None = None,
) -> None:
    react_step_id = "react-step:1"
    _append(
        root,
        "plans.jsonl",
        {
            "step": 1,
            "react_step_id": react_step_id,
            "plan": {
                "actions": [
                    _direct_action_artifact(action) for action in planned_actions
                ],
                "done": False,
                "completion_summary": None,
            },
        },
    )
    for action_index, action in enumerate(
        executed_actions if executed_actions is not None else planned_actions,
        start=1,
    ):
        _append(
            root,
            "environment_actions.jsonl",
            _direct_observation(
                action,
                action_index=action_index,
                react_step_id=react_step_id,
            ),
        )


def _write(
    tmp_path: Path,
    *,
    arm: str,
    stop_reason: str = "completed",
    budget: object = _DEFAULT_BUDGET,
) -> Path:
    selected_budget = (
        SimpleNamespace(
            provider_tokens_used=17,
            model_calls_used=2,
            actions_used=1,
        )
        if budget is _DEFAULT_BUDGET
        else budget
    )
    return write_atif_trajectory(
        logs_dir=tmp_path,
        artifact_root=tmp_path / arm,
        arm=arm,
        instruction=f"solve the task without exposing {_SECRET}",
        run_id="run-1",
        session_id="session-1",
        model_name="test-model",
        adapter_version="0.1.0",
        stop_reason=stop_reason,
        budget=selected_budget,
        restricted_values=(_SECRET,),
    )


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_bayesprobe_success_is_complete_valid_atif_with_causal_linkage(
    tmp_path: Path,
) -> None:
    _seed_bayesprobe_artifacts(tmp_path / "bayesprobe")

    path = _write(tmp_path, arm="bayesprobe")
    payload = _load(path)

    validator = TrajectoryValidator()
    assert validator.validate(path), validator.get_errors()
    assert path == tmp_path / "trajectory.json"
    assert payload["schema_version"] == "ATIF-v1.7"
    assert payload["steps"][0] == {
        "step_id": 1,
        "source": "user",
        "message": "solve the task without exposing [REDACTED]",
    }
    assert [step["step_id"] for step in payload["steps"]] == list(
        range(1, len(payload["steps"]) + 1)
    )

    action_steps = [
        step for step in payload["steps"] if step.get("extra", {}).get("kind") == "terminal_action"
    ]
    assert len(action_steps) == 1
    action_step = action_steps[0]
    assert len(action_step["tool_calls"]) == 1
    assert len(action_step["observation"]["results"]) == 1
    assert (
        action_step["observation"]["results"][0]["source_call_id"]
        == action_step["tool_calls"][0]["tool_call_id"]
    )
    expected_fingerprint = _request_fingerprint(_observation()["action"])
    assert action_step["extra"] == {
        "kind": "terminal_action",
        "plan_id": "plan-1",
        "policy_attempt_id": "policy-1",
        "probe_id": "probe-1",
        "action_id": "action-1",
        "signal_id": "signal-1",
        "request_fingerprint": expected_fingerprint,
    }

    evidence_step = next(
        step for step in payload["steps"] if step.get("extra", {}).get("evidence_id") == "evidence-1"
    )
    assert evidence_step["source"] == "agent"
    assert evidence_step["llm_call_count"] == 0
    assert "reasoning_content" not in evidence_step
    assert evidence_step["extra"]["probe_id"] == "probe-1"
    assert evidence_step["extra"]["signal_id"] == "signal-1"
    assert evidence_step["extra"]["update_ids"] == ["update-1"]

    discarded = next(
        step
        for step in payload["steps"]
        if step.get("extra", {}).get("causal_decision") == "discard"
    )
    assert discarded["extra"]["signal_id"] == "signal-discarded"
    assert discarded["extra"]["reason_code"] == "stale_state"
    assert discarded["extra"]["probe_id"] == "probe-1"
    assert discarded["llm_call_count"] == 0

    assert payload["final_metrics"]["extra"]["provider_tokens_used"] == 17
    assert payload["final_metrics"]["total_prompt_tokens"] == 10
    assert payload["final_metrics"]["total_completion_tokens"] == 7
    final_step = payload["steps"][-1]
    assert final_step["source"] == "system"
    assert final_step["extra"]["stop_reason"] == "completed"
    assert final_step["extra"]["artifact_id"] == "trajectory:run-1"
    serialized = json.dumps(payload)
    assert _SECRET not in serialized
    assert "reward" not in serialized.casefold()


def test_provider_contract_failure_has_terminal_system_step(tmp_path: Path) -> None:
    _append(
        tmp_path / "bayesprobe",
        "errors.jsonl",
        {"category": "provider_contract_error", "error_type": "ProviderContractError"},
    )

    path = _write(
        tmp_path,
        arm="bayesprobe",
        stop_reason="provider_contract_error",
        budget=SimpleNamespace(
            provider_tokens_used=0,
            model_calls_used=3,
            actions_used=0,
        ),
    )
    payload = _load(path)

    assert TrajectoryValidator().validate(path)
    assert payload["steps"][-1]["source"] == "system"
    assert payload["steps"][-1]["extra"]["stop_reason"] == "provider_contract_error"
    assert payload["final_metrics"]["extra"]["provider_tokens_used"] == 0


def test_unbound_causal_discard_is_recorded_without_inventing_identities(
    tmp_path: Path,
) -> None:
    _append(
        tmp_path / "bayesprobe",
        "causal_decisions.jsonl",
        {
            "signal_id": "",
            "action_id": "",
            "action_role": "inspect",
            "decision": "discard",
            "reason_code": "unbound_signal",
            "subject_environment_state_id": "",
            "judgment_response_sha256": "sha256:" + "e" * 64,
        },
    )

    payload = _load(_write(tmp_path, arm="bayesprobe"))

    discard = next(
        step
        for step in payload["steps"]
        if step.get("extra", {}).get("causal_decision") == "discard"
    )
    assert discard["extra"] == {
        "kind": "causal_evidence_decision",
        "causal_decision": "discard",
        "reason_code": "unbound_signal",
    }
    assert TrajectoryValidator().validate(tmp_path / "trajectory.json")


def test_reactive_success_has_request_bound_action_and_react_lineage(
    tmp_path: Path,
) -> None:
    _seed_direct_artifacts(tmp_path / "direct")

    payload = _load(_write(tmp_path, arm="direct"))

    action = next(
        step for step in payload["steps"] if step.get("extra", {}).get("kind") == "terminal_action"
    )
    assert action["extra"]["plan_id"] == "react-plan:1"
    assert action["extra"]["react_step_id"] == "react-step:1"
    assert action["extra"]["action_id"] == "react-action:1"
    assert action["extra"]["signal_id"] == "react-signal:1"
    assert action["extra"]["request_fingerprint"] == _request_fingerprint(
        {
            "type": "shell",
            "command": "python -m pytest -q",
            "timeout_seconds": 120,
            "mutates_environment": False,
        }
    )
    assert "probe_id" not in action["extra"]
    assert "policy_attempt_id" not in action["extra"]
    assert action["observation"]["results"][0]["source_call_id"] == action["tool_calls"][0]["tool_call_id"]
    assert payload["agent"]["name"] == "bayesprobe-direct"
    assert payload["final_metrics"]["extra"]["provider_tokens_used"] == 17


def test_bayesprobe_rejects_interrupted_causal_append(tmp_path: Path) -> None:
    root = tmp_path / "bayesprobe"
    _seed_bayesprobe_artifacts(root)
    _append(
        root,
        "environment_actions.jsonl",
        _observation(command="pwd", action_index=2),
    )

    with pytest.raises(TrajectoryExportError):
        _write(tmp_path, arm="bayesprobe")

    assert not (tmp_path / "trajectory.json").exists()


def test_bayesprobe_rejects_duplicate_causal_record(tmp_path: Path) -> None:
    root = tmp_path / "bayesprobe"
    _seed_bayesprobe_artifacts(root)
    record = _read_records(root / "causal_actions.jsonl")[0]
    _append(root, "causal_actions.jsonl", record)

    with pytest.raises(TrajectoryExportError):
        _write(tmp_path, arm="bayesprobe")


def test_bayesprobe_rejects_extra_unmatched_causal_record(tmp_path: Path) -> None:
    root = tmp_path / "bayesprobe"
    _seed_bayesprobe_artifacts(root)
    extra_observation = _observation(command="pwd", action_index=2)
    _append(
        root,
        "causal_actions.jsonl",
        _causal_action(extra_observation, action_id="action-extra"),
    )

    with pytest.raises(TrajectoryExportError):
        _write(tmp_path, arm="bayesprobe")


def test_bayesprobe_preserves_environment_action_order(tmp_path: Path) -> None:
    root = tmp_path / "bayesprobe"
    _seed_bayesprobe_artifacts(root)
    second = _observation(command="pwd", action_index=2)
    _append(root, "environment_actions.jsonl", second)
    records = _read_records(root / "causal_actions.jsonl")
    records.append(_causal_action(second, action_id="action-2"))
    _overwrite_records(root / "causal_actions.jsonl", list(reversed(records)))

    payload = _load(
        _write(
            tmp_path,
            arm="bayesprobe",
            budget=SimpleNamespace(
                provider_tokens_used=17,
                model_calls_used=2,
                actions_used=2,
            ),
        )
    )

    action_steps = [
        step
        for step in payload["steps"]
        if step.get("extra", {}).get("kind") == "terminal_action"
    ]
    assert [step["extra"]["action_id"] for step in action_steps] == [
        "action-1",
        "action-2",
    ]


def test_bayesprobe_rejects_nested_observation_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "bayesprobe"
    _seed_bayesprobe_artifacts(root)
    record = _read_records(root / "causal_actions.jsonl")[0]
    nested = _observation(command="pwd", action_index=1)
    record["observation"] = nested
    record["request_fingerprint"] = _request_fingerprint(nested["action"])
    _overwrite_records(root / "causal_actions.jsonl", [record])

    with pytest.raises(TrajectoryExportError):
        _write(tmp_path, arm="bayesprobe")


def test_bayesprobe_rejects_request_fingerprint_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "bayesprobe"
    _seed_bayesprobe_artifacts(root)
    record = _read_records(root / "causal_actions.jsonl")[0]
    record["request_fingerprint"] = "sha256:" + "f" * 64
    _overwrite_records(root / "causal_actions.jsonl", [record])

    with pytest.raises(TrajectoryExportError):
        _write(tmp_path, arm="bayesprobe")


def test_bayesprobe_rejects_duplicate_action_identity(tmp_path: Path) -> None:
    root = tmp_path / "bayesprobe"
    _seed_bayesprobe_artifacts(root)
    second = _observation(command="pwd", action_index=2)
    _append(root, "environment_actions.jsonl", second)
    _append(root, "causal_actions.jsonl", _causal_action(second, action_id="action-1"))

    with pytest.raises(TrajectoryExportError):
        _write(tmp_path, arm="bayesprobe")


def test_bayesprobe_rejects_duplicate_signal_identity(tmp_path: Path) -> None:
    root = tmp_path / "bayesprobe"
    _seed_bayesprobe_artifacts(root)
    second = _observation(command="pwd", action_index=2)
    _append(root, "environment_actions.jsonl", second)
    _append(
        root,
        "causal_actions.jsonl",
        _causal_action(second, action_id="action-2"),
    )
    _append(
        root,
        "bayesprobe_ledger.jsonl",
        {
            "record_type": "external_signal",
            "recorded_at": "2026-07-17T00:00:03+00:00",
            "payload": {
                "id": "signal-1",
                "generated_by_probe": "probe-1",
                "provenance": {
                    "artifact_refs": ["causal_actions.jsonl#action-2"]
                },
            },
        },
    )

    with pytest.raises(TrajectoryExportError):
        _write(tmp_path, arm="bayesprobe")


def test_bayesprobe_strictly_rejects_malformed_causal_record(tmp_path: Path) -> None:
    root = tmp_path / "bayesprobe"
    _seed_bayesprobe_artifacts(root)
    record = _read_records(root / "causal_actions.jsonl")[0]
    record["unexpected"] = "ignored by the old exporter"
    _overwrite_records(root / "causal_actions.jsonl", [record])

    with pytest.raises(TrajectoryExportError):
        _write(tmp_path, arm="bayesprobe")


@pytest.mark.parametrize(
    "actions",
    [
        [
            {
                "type": "write_file",
                "path": "/app/result.txt",
                "content": "first body",
            },
            {
                "type": "write_file",
                "path": "/app/result.txt",
                "content": "second body",
            },
        ],
        [
            {
                "type": "apply_patch",
                "patch": "*** Begin Patch\n*** Add File: /app/a\n+one\n*** End Patch",
                "strip": 0,
            },
            {
                "type": "apply_patch",
                "patch": "*** Begin Patch\n*** Add File: /app/b\n+two\n*** End Patch",
                "strip": 0,
            },
        ],
    ],
    ids=("same-path-writes", "same-strip-patches"),
)
def test_direct_rejects_out_of_sequence_same_target_actions(
    tmp_path: Path,
    actions: list[dict[str, object]],
) -> None:
    _seed_direct_action_sequence(
        tmp_path / "direct",
        planned_actions=actions,
        executed_actions=list(reversed(actions)),
    )

    with pytest.raises(TrajectoryExportError):
        _write(tmp_path, arm="direct")


def test_direct_uses_exact_fingerprints_in_execution_order(tmp_path: Path) -> None:
    actions = [
        {
            "type": "write_file",
            "path": "/app/result.txt",
            "content": "first body",
        },
        {
            "type": "write_file",
            "path": "/app/result.txt",
            "content": "second body",
        },
    ]
    _seed_direct_action_sequence(
        tmp_path / "direct",
        planned_actions=actions,
    )

    payload = _load(_write(tmp_path, arm="direct"))

    action_steps = [
        step
        for step in payload["steps"]
        if step.get("extra", {}).get("kind") == "terminal_action"
    ]
    assert [step["extra"]["request_fingerprint"] for step in action_steps] == [
        _request_fingerprint(action) for action in actions
    ]
    assert all(step["extra"]["react_step_id"] == "react-step:1" for step in action_steps)
    serialized = json.dumps(payload)
    assert "first body" not in serialized
    assert "second body" not in serialized
    assert "probe_id" not in serialized
    assert "belief" not in serialized.casefold()
    assert "evidence" not in serialized.casefold()


def test_direct_rejects_ambiguous_duplicate_plan_fingerprint(tmp_path: Path) -> None:
    action = {
        "type": "write_file",
        "path": "/app/result.txt",
        "content": "same body",
    }
    _seed_direct_action_sequence(
        tmp_path / "direct",
        planned_actions=[action, action],
        executed_actions=[action],
    )

    with pytest.raises(TrajectoryExportError):
        _write(tmp_path, arm="direct")


@pytest.mark.parametrize("missing_from", ["plan", "observation"])
def test_direct_rejects_missing_request_lineage(
    tmp_path: Path,
    missing_from: str,
) -> None:
    root = tmp_path / "direct"
    _seed_direct_artifacts(root)
    filename = "plans.jsonl" if missing_from == "plan" else "environment_actions.jsonl"
    path = root / filename
    records = _read_records(path)
    if missing_from == "plan":
        del records[0]["plan"]["actions"][0]["request_fingerprint"]
    else:
        del records[0]["request_fingerprint"]
    _overwrite_records(path, records)

    with pytest.raises(TrajectoryExportError):
        _write(tmp_path, arm="direct")


def test_direct_rejects_duplicate_executed_lineage(tmp_path: Path) -> None:
    root = tmp_path / "direct"
    _seed_direct_artifacts(root)
    observation = _read_records(root / "environment_actions.jsonl")[0]
    duplicate = copy.deepcopy(observation)
    duplicate["action_index"] = 2
    _append(root, "environment_actions.jsonl", duplicate)

    with pytest.raises(TrajectoryExportError):
        _write(tmp_path, arm="direct")


def test_direct_rejects_unmatched_request_fingerprint(tmp_path: Path) -> None:
    root = tmp_path / "direct"
    _seed_direct_artifacts(root)
    observations = _read_records(root / "environment_actions.jsonl")
    observations[0]["request_fingerprint"] = "sha256:" + "f" * 64
    _overwrite_records(root / "environment_actions.jsonl", observations)

    with pytest.raises(TrajectoryExportError):
        _write(tmp_path, arm="direct")


@pytest.mark.parametrize("arm", ["bayesprobe", "direct"])
def test_export_rejects_missing_shared_budget(tmp_path: Path, arm: str) -> None:
    if arm == "bayesprobe":
        _seed_bayesprobe_artifacts(tmp_path / arm)
    else:
        _seed_direct_artifacts(tmp_path / arm)

    with pytest.raises(TrajectoryExportError):
        _write(tmp_path, arm=arm, budget=None)

    assert not (tmp_path / "trajectory.json").exists()


@pytest.mark.parametrize("arm", ["bayesprobe", "direct"])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("provider_tokens_used", True),
        ("model_calls_used", -1),
        ("actions_used", 1.0),
    ],
)
def test_export_rejects_malformed_shared_budget_counter(
    tmp_path: Path,
    arm: str,
    field: str,
    value: object,
) -> None:
    if arm == "bayesprobe":
        _seed_bayesprobe_artifacts(tmp_path / arm)
    else:
        _seed_direct_artifacts(tmp_path / arm)
    counters = {
        "provider_tokens_used": 17,
        "model_calls_used": 2,
        "actions_used": 1,
    }
    counters[field] = value

    with pytest.raises(TrajectoryExportError):
        _write(tmp_path, arm=arm, budget=SimpleNamespace(**counters))


def test_replace_failure_preserves_destination_and_cleans_only_current_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_direct_artifacts(tmp_path / "direct")
    destination = tmp_path / "trajectory.json"
    destination.write_text("existing\n", encoding="utf-8")
    stale = tmp_path / ".trajectory-stale.tmp"
    stale.write_text("stale\n", encoding="utf-8")
    current_temps: list[Path] = []
    fsync_calls: list[int] = []

    monkeypatch.setattr(
        "bayesprobe_terminal_bench.trajectory.os.fsync",
        fsync_calls.append,
    )

    def fail_replace(source: object, target: object) -> None:
        source_path = Path(source)
        current_temps.append(source_path)
        assert source_path.parent == destination.parent
        assert Path(target) == destination
        raise OSError("forced replace failure")

    monkeypatch.setattr(
        "bayesprobe_terminal_bench.trajectory.os.replace",
        fail_replace,
    )

    with pytest.raises(TrajectoryExportError):
        _write(tmp_path, arm="direct")

    assert fsync_calls
    assert destination.read_text(encoding="utf-8") == "existing\n"
    assert len(current_temps) == 1
    assert not current_temps[0].exists()
    assert stale.read_text(encoding="utf-8") == "stale\n"


def test_protected_evaluator_path_fails_without_publication(tmp_path: Path) -> None:
    root = tmp_path / "direct"
    _seed_direct_artifacts(root)
    unsafe_action = {
        "type": "shell",
        "command": "cat /tests/hidden.py",
        "timeout_seconds": 120,
        "mutates_environment": False,
    }
    plans = _read_records(root / "plans.jsonl")
    plans[0]["plan"]["actions"] = [_direct_action_artifact(unsafe_action)]
    _overwrite_records(root / "plans.jsonl", plans)
    _overwrite_records(
        root / "environment_actions.jsonl",
        [
            _direct_observation(
                unsafe_action,
                action_index=1,
                react_step_id="react-step:1",
            )
        ],
    )

    with pytest.raises(TrajectoryExportError, match="protected path"):
        _write(tmp_path, arm="direct")

    assert not (tmp_path / "trajectory.json").exists()


def test_validator_rejection_does_not_replace_existing_trajectory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_direct_artifacts(tmp_path / "direct")
    destination = tmp_path / "trajectory.json"
    destination.write_text("existing\n", encoding="utf-8")
    monkeypatch.setattr(
        "bayesprobe_terminal_bench.trajectory.TrajectoryValidator.validate",
        lambda self, payload: False,
    )
    monkeypatch.setattr(
        "bayesprobe_terminal_bench.trajectory.TrajectoryValidator.get_errors",
        lambda self: ["forced rejection"],
    )

    with pytest.raises(TrajectoryExportError, match="Harbor ATIF validation failed"):
        _write(tmp_path, arm="direct")

    assert destination.read_text(encoding="utf-8") == "existing\n"


class _Runner:
    def __init__(self, result: object | None = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error

    def run_question(self, input: object) -> object:
        if self.error is not None:
            raise self.error
        return self.result


class _Controller:
    def run(self, instruction: str) -> object:
        return SimpleNamespace(stop_reason="completed", steps=2, observations=1)


class _FailingController:
    def run(self, instruction: str) -> object:
        raise ProviderContractError(stage="react_step", attempts=3)


@pytest.mark.asyncio
async def test_bayesprobe_agent_emits_success_trajectory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = TrialArtifactStore(
        tmp_path / "bayesprobe", restricted_values=(_SECRET,)
    )
    _seed_bayesprobe_artifacts(artifacts.root)
    budget = SimpleNamespace(
        provider_tokens_used=17,
        model_calls_used=2,
        actions_used=1,
    )
    session = SimpleNamespace(
        runner=_Runner(
            result=SimpleNamespace(
                stop_reason=SimpleNamespace(value="max_cycles"),
                cycle_results=(object(),),
            )
        ),
        input=SimpleNamespace(run_id="run-agent"),
        artifacts=artifacts,
        budget=budget,
    )
    monkeypatch.setattr(
        "bayesprobe_terminal_bench.agent.build_live_session", lambda **kwargs: session
    )
    agent = BayesProbeHarborAgent(
        logs_dir=tmp_path,
        model_name="test-model",
        extra_env=_EXTRA_ENV,
    )

    await agent.run("solve it", object(), AgentContext())

    payload = _load(tmp_path / "trajectory.json")
    assert payload["steps"][-1]["extra"]["stop_reason"] == "max_cycles"
    assert TrajectoryValidator().validate(tmp_path / "trajectory.json")


@pytest.mark.asyncio
async def test_bayesprobe_agent_emits_failure_trajectory_and_supports_atif(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = TrialArtifactStore(
        tmp_path / "bayesprobe", restricted_values=(_SECRET,)
    )
    budget = SimpleNamespace(
        provider_tokens_used=0,
        model_calls_used=3,
        actions_used=0,
    )
    session = SimpleNamespace(
        runner=_Runner(
            error=ProviderContractError(stage="terminal_task_frame", attempts=3)
        ),
        input=SimpleNamespace(run_id="run-agent"),
        artifacts=artifacts,
        budget=budget,
    )
    monkeypatch.setattr(
        "bayesprobe_terminal_bench.agent.build_live_session", lambda **kwargs: session
    )
    agent = BayesProbeHarborAgent(
        logs_dir=tmp_path,
        model_name="test-model",
        extra_env=_EXTRA_ENV,
    )

    with pytest.raises(BayesProbeHarborAgentError) as failure:
        await agent.run("solve it", object(), AgentContext())

    assert failure.value.category == "provider_contract_error"
    assert BayesProbeHarborAgent.SUPPORTS_ATIF is True
    payload = _load(tmp_path / "trajectory.json")
    assert payload["steps"][-1]["extra"]["stop_reason"] == "provider_contract_error"
    assert TrajectoryValidator().validate(tmp_path / "trajectory.json")


@pytest.mark.asyncio
async def test_direct_agent_emits_success_trajectory_and_supports_atif(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = TrialArtifactStore(tmp_path / "direct", restricted_values=(_SECRET,))
    _seed_direct_artifacts(artifacts.root)
    budget = SimpleNamespace(
        provider_tokens_used=17,
        model_calls_used=2,
        actions_used=1,
    )
    monkeypatch.setattr(
        "bayesprobe_terminal_bench.direct_agent.build_direct_session",
        lambda **kwargs: SimpleNamespace(
            controller=_Controller(), artifacts=artifacts, budget=budget
        ),
    )
    agent = DirectHarborAgent(
        logs_dir=tmp_path,
        model_name="test-model",
        extra_env=_EXTRA_ENV,
    )

    await agent.run("solve it", object(), AgentContext())

    assert DirectHarborAgent.SUPPORTS_ATIF is True
    assert TrajectoryValidator().validate(tmp_path / "trajectory.json")
    assert _load(tmp_path / "trajectory.json")["steps"][-1]["extra"]["stop_reason"] == "completed"


@pytest.mark.asyncio
async def test_direct_agent_emits_classified_failure_trajectory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = TrialArtifactStore(tmp_path / "direct", restricted_values=(_SECRET,))
    budget = SimpleNamespace(
        provider_tokens_used=0,
        model_calls_used=3,
        actions_used=0,
    )
    monkeypatch.setattr(
        "bayesprobe_terminal_bench.direct_agent.build_direct_session",
        lambda **kwargs: SimpleNamespace(
            controller=_FailingController(), artifacts=artifacts, budget=budget
        ),
    )
    agent = DirectHarborAgent(
        logs_dir=tmp_path,
        model_name="test-model",
        extra_env=_EXTRA_ENV,
    )

    with pytest.raises(DirectHarborAgentError) as failure:
        await agent.run("solve it", object(), AgentContext())

    assert failure.value.category == "provider_contract_error"
    payload = _load(tmp_path / "trajectory.json")
    assert payload["steps"][-1]["extra"]["stop_reason"] == "provider_contract_error"
    assert TrajectoryValidator().validate(tmp_path / "trajectory.json")


@pytest.mark.asyncio
async def test_bayesprobe_agent_malformed_budget_has_adapter_error_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = TrialArtifactStore(
        tmp_path / "bayesprobe",
        restricted_values=(_SECRET,),
    )
    session = SimpleNamespace(
        runner=_Runner(
            error=ProviderContractError(stage="terminal_task_frame", attempts=3)
        ),
        input=SimpleNamespace(run_id="run-agent"),
        artifacts=artifacts,
        budget=SimpleNamespace(
            provider_tokens_used="0",
            model_calls_used=3,
            actions_used=0,
        ),
    )
    monkeypatch.setattr(
        "bayesprobe_terminal_bench.agent.build_live_session",
        lambda **kwargs: session,
    )
    agent = BayesProbeHarborAgent(
        logs_dir=tmp_path,
        model_name="test-model",
        extra_env=_EXTRA_ENV,
    )

    with pytest.raises(BayesProbeHarborAgentError) as failure:
        await agent.run("solve it", object(), AgentContext())

    assert failure.value.category == "adapter_error"
    assert not (tmp_path / "trajectory.json").exists()
    assert _read_records(artifacts.root / "errors.jsonl")[-1] == {
        "category": "adapter_error",
        "error_type": "TrajectoryExportError",
        "stage": "trajectory_export",
    }


@pytest.mark.asyncio
async def test_direct_agent_malformed_budget_has_adapter_error_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = TrialArtifactStore(
        tmp_path / "direct",
        restricted_values=(_SECRET,),
    )
    session = SimpleNamespace(
        controller=_FailingController(),
        artifacts=artifacts,
        budget=SimpleNamespace(
            provider_tokens_used=0,
            model_calls_used=True,
            actions_used=0,
        ),
    )
    monkeypatch.setattr(
        "bayesprobe_terminal_bench.direct_agent.build_direct_session",
        lambda **kwargs: session,
    )
    agent = DirectHarborAgent(
        logs_dir=tmp_path,
        model_name="test-model",
        extra_env=_EXTRA_ENV,
    )

    with pytest.raises(DirectHarborAgentError) as failure:
        await agent.run("solve it", object(), AgentContext())

    assert failure.value.category == "adapter_error"
    assert not (tmp_path / "trajectory.json").exists()
    assert _read_records(artifacts.root / "errors.jsonl")[-1] == {
        "category": "adapter_error",
        "error_type": "TrajectoryExportError",
        "stage": "trajectory_export",
    }


@pytest.mark.asyncio
async def test_trajectory_write_failure_becomes_adapter_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = TrialArtifactStore(tmp_path / "direct", restricted_values=(_SECRET,))
    session = SimpleNamespace(
        controller=_Controller(),
        artifacts=artifacts,
        budget=SimpleNamespace(
            provider_tokens_used=0,
            model_calls_used=1,
            actions_used=0,
        ),
    )
    monkeypatch.setattr(
        "bayesprobe_terminal_bench.direct_agent.build_direct_session",
        lambda **kwargs: session,
    )
    monkeypatch.setattr(
        "bayesprobe_terminal_bench.direct_agent.write_atif_trajectory",
        lambda **kwargs: (_ for _ in ()).throw(TrajectoryExportError("invalid")),
    )
    agent = DirectHarborAgent(
        logs_dir=tmp_path,
        model_name="test-model",
        extra_env=_EXTRA_ENV,
    )

    with pytest.raises(DirectHarborAgentError) as failure:
        await agent.run("solve it", object(), AgentContext())

    assert failure.value.category == "adapter_error"
    assert artifacts.root.joinpath("errors.jsonl").is_file()
