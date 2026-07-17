from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from harbor.models.agent.context import AgentContext
from harbor.utils.trajectory_validator import TrajectoryValidator

from bayesprobe_terminal_bench.agent import BayesProbeHarborAgent, BayesProbeHarborAgentError
from bayesprobe_terminal_bench.artifacts import TrialArtifactStore
from bayesprobe_terminal_bench.direct_agent import DirectHarborAgent, DirectHarborAgentError
from bayesprobe_terminal_bench.provider_contract import ProviderContractError
from bayesprobe_terminal_bench.trajectory import (
    TrajectoryExportError,
    write_atif_trajectory,
)


_SECRET = "trajectory-secret-value"
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


def _observation(*, command: str = "pytest -q", action_index: int = 1) -> dict[str, object]:
    return {
        "action_index": action_index,
        "action": {
            "type": "shell",
            "command": command,
            "timeout_seconds": 120,
            "mutates_environment": False,
        },
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
    _append(
        root,
        "causal_actions.jsonl",
        {
            "run_id": "run-1",
            "cycle_id": "cycle-1",
            "probe_id": "probe-1",
            "plan_id": "plan-1",
            "policy_attempt_id": "policy-1",
            "action_id": "action-1",
            "step_index": 0,
            "action_role": "verify",
            "request_fingerprint": "sha256:" + "b" * 64,
            "pre_environment_state_id": "env:0",
            "post_environment_state_id": "env:0",
            "subject_environment_state_id": "env:0",
            "intervention_generation": 0,
            "verification_target": "public tests pass",
            "transition_predictions": {},
            "observation": observation,
        },
    )
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
    observation = _observation(command="python -m pytest -q")
    _append(
        root,
        "plans.jsonl",
        {
            "step": 1,
            "plan": {
                "actions": [observation["action"]],
                "done": False,
                "completion_summary": None,
            },
        },
        {
            "step": 2,
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


def _write(
    tmp_path: Path,
    *,
    arm: str,
    stop_reason: str = "completed",
    budget: object | None = None,
) -> Path:
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
        budget=budget
        or SimpleNamespace(
            provider_tokens_used=17,
            model_calls_used=2,
            actions_used=1,
        ),
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
    assert action_step["extra"] == {
        "kind": "terminal_action",
        "plan_id": "plan-1",
        "policy_attempt_id": "policy-1",
        "probe_id": "probe-1",
        "action_id": "action-1",
        "signal_id": "signal-1",
        "request_fingerprint": "sha256:" + "b" * 64,
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


def test_reactive_success_has_request_bound_action_and_synthetic_lineage(
    tmp_path: Path,
) -> None:
    _seed_direct_artifacts(tmp_path / "direct")

    payload = _load(_write(tmp_path, arm="direct"))

    action = next(
        step for step in payload["steps"] if step.get("extra", {}).get("kind") == "terminal_action"
    )
    assert action["extra"]["plan_id"] == "react-plan:1"
    assert action["extra"]["action_id"] == "react-action:1"
    assert action["extra"]["signal_id"] == "react-signal:1"
    assert action["observation"]["results"][0]["source_call_id"] == action["tool_calls"][0]["tool_call_id"]
    assert payload["agent"]["name"] == "bayesprobe-direct"
    assert payload["final_metrics"]["extra"]["provider_tokens_used"] == 17


def test_protected_evaluator_path_fails_without_publication(tmp_path: Path) -> None:
    _seed_direct_artifacts(tmp_path / "direct")
    observations = tmp_path / "direct" / "environment_actions.jsonl"
    unsafe = _observation(command="cat /tests/hidden.py")
    observations.write_text(json.dumps(unsafe) + "\n", encoding="utf-8")

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
