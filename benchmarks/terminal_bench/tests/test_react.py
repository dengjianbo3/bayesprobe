from __future__ import annotations

import json
from collections import deque
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from bayesprobe_terminal_bench.actions import (
    ActionObservation,
    ShellAction,
    WriteFileAction,
)
from bayesprobe_terminal_bench.config import (
    BudgetExhausted,
    ProviderIdentityError,
    RunBudget,
    TerminalBenchConfig,
)
from bayesprobe_terminal_bench.environment import PolicyViolation
from bayesprobe_terminal_bench.react import (
    OpenAICompatibleReActPlanner,
    ReActController,
    ReActPlanError,
    ReActStep,
    react_step_input,
)


def _response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        id="response-1",
        model="test-model",
        system_fingerprint="fp-test",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        ),
    )


class _Completions:
    def __init__(self, responses: list[object]) -> None:
        self.responses = deque(responses)
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.requests.append(kwargs)
        response = self.responses.popleft()
        if isinstance(response, Exception):
            raise response
        return response


class _Client:
    def __init__(self, responses: list[object]) -> None:
        self.chat = SimpleNamespace(completions=_Completions(responses))


class _Planner:
    def __init__(self, steps: list[ReActStep]) -> None:
        self.steps = deque(steps)
        self.histories: list[tuple[ActionObservation, ...]] = []

    def next_step(
        self,
        *,
        instruction: str,
        history: tuple[ActionObservation, ...],
    ) -> ReActStep:
        assert instruction == "repair the task"
        self.histories.append(history)
        return self.steps.popleft()


class _Bridge:
    def __init__(self, *, reject_first: bool = False) -> None:
        self.reject_first = reject_first
        self.actions: list[tuple[ShellAction, int]] = []

    def execute(self, action: ShellAction, action_index: int) -> ActionObservation:
        self.actions.append((action, action_index))
        if self.reject_first:
            self.reject_first = False
            raise PolicyViolation("denied")
        return ActionObservation(
            action_index=action_index,
            action=action,
            stdout="ok",
            stderr="",
            return_code=0,
            duration_ms=1,
            pre_environment_state_id="env:0",
            post_environment_state_id="env:0",
            full_output_sha256="a" * 64,
            model_facing_output='{"stdout":"ok"}',
        )


class _Artifacts:
    def __init__(self) -> None:
        self.plans: list[object] = []
        self.observations: list[object] = []
        self.errors: list[object] = []
        self.provider_calls: list[object] = []

    def append_plan(self, value: object) -> None:
        self.plans.append(value)

    def append_observation(self, value: object) -> None:
        self.observations.append(value)

    def append_error(self, value: object) -> None:
        self.errors.append(value)

    def append_provider_call(self, value: object) -> None:
        self.provider_calls.append(value)


def _config() -> TerminalBenchConfig:
    return TerminalBenchConfig(
        model="test-model",
        base_url="https://provider.invalid",
        max_model_calls=4,
        max_total_actions=3,
    )


def test_react_step_requires_actions_until_done() -> None:
    with pytest.raises(ValidationError, match="unfinished steps require actions"):
        ReActStep(thought_summary="inspect", actions=(), done=False)


def test_react_step_rejects_actions_after_done() -> None:
    with pytest.raises(ValidationError, match="completed steps cannot contain actions"):
        ReActStep(
            thought_summary="complete",
            actions=(ShellAction(command="pwd"),),
            done=True,
            completion_summary="verified",
        )


def test_planner_repairs_invalid_json_once_and_charges_both_calls() -> None:
    client = _Client(
        [
            _response("not-json"),
            _response(
                json.dumps(
                    {
                        "thought_summary": "inspect files",
                        "actions": [
                            {
                                "type": "shell",
                                "command": "pwd",
                                "timeout_seconds": 120,
                                "mutates_environment": False,
                            }
                        ],
                        "done": False,
                        "completion_summary": None,
                    }
                )
            ),
        ]
    )
    budget = RunBudget(max_actions=3, max_model_calls=4)
    telemetry: list[dict[str, object]] = []
    planner = OpenAICompatibleReActPlanner(
        config=_config(),
        budget=budget,
        client=client,
        invocation_observer=telemetry.append,
    )

    step = planner.next_step(instruction="repair the task", history=())

    assert step.actions[0].command == "pwd"
    assert budget.model_calls_used == 2
    assert [item["repair"] for item in telemetry] == [False, True]
    assert len(client.chat.completions.requests) == 2


def test_planner_uses_initial_plus_two_repairs_with_safe_field_telemetry() -> None:
    secret = "sk-react-secret-1234567890"
    client = _Client(
        [
            _response("not-json"),
            _response(json.dumps({"thought_summary": secret, "actions": []})),
            _response(json.dumps({"done": True, "completion_summary": None})),
        ]
    )
    budget = RunBudget(max_model_calls=3, max_provider_tokens=100)
    telemetry: list[dict[str, object]] = []
    planner = OpenAICompatibleReActPlanner(
        config=_config(),
        budget=budget,
        client=client,
        invocation_observer=telemetry.append,
        expected_provider_model="test-model",
        expected_system_fingerprint="fp-test",
    )

    with pytest.raises(ReActPlanError) as failure:
        planner.next_step(instruction="repair the task", history=())

    assert failure.value.category == "provider_contract_error"
    assert failure.value.attempts == 3
    assert budget.model_calls_used == 3
    assert budget.provider_tokens_used == 45
    assert len(client.chat.completions.requests) == 3
    assert [item["attempt_index"] for item in telemetry] == [0, 1, 2]
    assert all(item["field_errors"] for item in telemetry)
    assert all(item["response_sha256"] for item in telemetry)
    assert secret not in json.dumps(telemetry)

    repair_payloads = [
        json.loads(request["messages"][1]["content"])
        for request in client.chat.completions.requests[1:]
    ]
    assert all(secret not in json.dumps(payload) for payload in repair_payloads)


def test_planner_does_not_retry_provider_errors() -> None:
    client = _Client([RuntimeError("secret provider detail")])
    budget = RunBudget(max_actions=3, max_model_calls=4)
    planner = OpenAICompatibleReActPlanner(
        config=_config(),
        budget=budget,
        client=client,
    )

    with pytest.raises(ReActPlanError, match="provider request failed"):
        planner.next_step(instruction="repair the task", history=())

    assert budget.model_calls_used == 1
    assert len(client.chat.completions.requests) == 1


def test_planner_records_usage_before_return_and_stops_on_token_overflow() -> None:
    client = _Client(
        [
            _response(
                json.dumps(
                    {
                        "thought_summary": "inspect files",
                        "actions": [
                            {
                                "type": "shell",
                                "command": "pwd",
                                "timeout_seconds": 120,
                                "mutates_environment": False,
                            }
                        ],
                        "done": False,
                        "completion_summary": None,
                    }
                )
            )
        ]
    )
    budget = RunBudget(max_model_calls=4, max_provider_tokens=10)

    def fail_observer(payload: object) -> None:
        raise RuntimeError("telemetry write failed")

    planner = OpenAICompatibleReActPlanner(
        config=_config(),
        budget=budget,
        client=client,
        invocation_observer=fail_observer,
        expected_provider_model="test-model",
        expected_system_fingerprint="fp-test",
    )

    with pytest.raises(BudgetExhausted) as failure:
        planner.next_step(instruction="repair the task", history=())

    assert failure.value.category == "budget_error"
    assert budget.model_calls_used == 1
    assert budget.provider_tokens_used == 15
    assert len(client.chat.completions.requests) == 1


def test_expired_deadline_rejects_react_plan_before_model_reservation() -> None:
    from bayesprobe_terminal_bench.deadline import DeadlineOpenAIClient, TrialDeadline

    deadline = TrialDeadline(timeout_seconds=5, monotonic=lambda: 0.0)
    budget = RunBudget(
        max_model_calls=3,
        reservation_guard=deadline.require_active,
    )
    base_client = _Client([])
    planner = OpenAICompatibleReActPlanner(
        config=_config(),
        budget=budget,
        client=DeadlineOpenAIClient(
            base_client=base_client,
            deadline=deadline,
            configured_timeout_seconds=360,
        ),
        expected_provider_model="test-model",
        expected_system_fingerprint="fp-test",
    )

    with pytest.raises(BudgetExhausted) as failure:
        planner.next_step(instruction="repair the task", history=())

    assert failure.value.category == "budget_error"
    assert budget.model_calls_used == 0
    assert base_client.chat.completions.requests == []


def test_react_history_is_bounded_redacted_and_omits_written_content() -> None:
    secret = "sk-abcdefghijklmnop1234567890"
    action = WriteFileAction(path="/app/result.txt", content="private" * 10_000)
    observation = ActionObservation(
        action_index=1,
        action=action,
        stdout="",
        stderr="",
        return_code=0,
        duration_ms=1,
        pre_environment_state_id="env:0",
        post_environment_state_id="env:1",
        full_output_sha256="a" * 64,
        model_facing_output=secret + ("x" * 10_000),
        output_truncated=True,
    )

    payload = react_step_input(
        instruction="repair the task",
        history=(observation,),
    )
    serialized = json.dumps(payload)
    projected = payload["recent_observations"][0]

    assert secret not in serialized
    assert "privateprivate" not in serialized
    assert projected["action"] == {
        "type": "write_file",
        "path": "/app/result.txt",
    }
    assert len(projected["observation"].encode("utf-8")) <= 4_096


def test_controller_executes_shared_actions_and_passes_real_history() -> None:
    planner = _Planner(
        [
            ReActStep(
                thought_summary="inspect",
                actions=(ShellAction(command="pwd"),),
                done=False,
            ),
            ReActStep(
                thought_summary="verified",
                actions=(),
                done=True,
                completion_summary="task complete",
            ),
        ]
    )
    bridge = _Bridge()
    artifacts = _Artifacts()
    budget = RunBudget(max_actions=3, max_model_calls=4)
    controller = ReActController(
        planner=planner,
        bridge=bridge,
        artifacts=artifacts,
        budget=budget,
    )

    result = controller.run("repair the task")

    assert result.stop_reason == "completed"
    assert result.completion_summary == "task complete"
    assert result.steps == 2
    assert result.observations == 1
    assert budget.actions_used == 1
    assert planner.histories[0] == ()
    assert planner.histories[1][0].stdout == "ok"
    assert len(artifacts.plans) == 2
    assert len(artifacts.observations) == 1
    assert "thought_summary" not in json.dumps(artifacts.plans)


def test_controller_plan_artifacts_omit_action_payloads_and_redact_completion() -> None:
    secret = "sk-react-artifact-1234567890"
    planner = _Planner(
        [
            ReActStep(
                thought_summary="write the repair",
                actions=(
                    WriteFileAction(
                        path="/app/result.txt",
                        content=f"{secret}\nread /solution/answer.py",
                    ),
                ),
                done=False,
            ),
            ReActStep(
                thought_summary="verified",
                actions=(),
                done=True,
                completion_summary=f"done with {secret} at /solution/answer.py",
            ),
        ]
    )
    artifacts = _Artifacts()
    controller = ReActController(
        planner=planner,
        bridge=_Bridge(),
        artifacts=artifacts,
        budget=RunBudget(max_actions=2, max_model_calls=2),
    )

    controller.run("repair the task")

    persisted = json.dumps(artifacts.plans)
    assert '"type": "write_file"' in persisted
    assert '"path": "/app/result.txt"' in persisted
    assert '"content"' not in persisted
    assert "thought_summary" not in persisted
    assert secret not in persisted
    assert "/solution/answer.py" not in persisted
    observations = json.dumps(
        artifacts.observations,
        default=lambda value: value.model_dump(mode="json"),
    )
    assert secret not in observations
    assert "/solution/answer.py" not in observations


def test_controller_propagates_contract_failure_without_executing_actions() -> None:
    class FailingPlanner:
        def next_step(self, **_: object) -> ReActStep:
            raise ReActPlanError(
                category="provider_contract_error",
                attempts=3,
            )

    artifacts = _Artifacts()
    bridge = _Bridge()
    controller = ReActController(
        planner=FailingPlanner(),
        bridge=bridge,
        artifacts=artifacts,
        budget=RunBudget(max_actions=3, max_model_calls=4),
    )

    with pytest.raises(ReActPlanError) as failure:
        controller.run("repair the task")

    assert failure.value.category == "provider_contract_error"
    assert bridge.actions == []
    assert artifacts.errors == [
        {
            "category": "provider_contract_error",
            "error_type": "ReActPlanError",
            "step": 1,
        }
    ]


def test_controller_preserves_provider_identity_error_category() -> None:
    class FailingPlanner:
        def next_step(self, **_: object) -> ReActStep:
            raise ProviderIdentityError("provider model identity drift")

    artifacts = _Artifacts()
    controller = ReActController(
        planner=FailingPlanner(),
        bridge=_Bridge(),
        artifacts=artifacts,
        budget=RunBudget(max_actions=1, max_model_calls=1),
    )

    with pytest.raises(ProviderIdentityError):
        controller.run("repair the task")

    assert artifacts.errors == [{"category": "provider_identity_error"}]


def test_controller_records_policy_rejection_without_observation() -> None:
    planner = _Planner(
        [
            ReActStep(
                thought_summary="attempt",
                actions=(ShellAction(command="pwd"),),
                done=False,
            ),
            ReActStep(
                thought_summary="stop",
                actions=(),
                done=True,
                completion_summary="no change",
            ),
        ]
    )
    artifacts = _Artifacts()
    controller = ReActController(
        planner=planner,
        bridge=_Bridge(reject_first=True),
        artifacts=artifacts,
        budget=RunBudget(max_actions=3, max_model_calls=4),
    )

    result = controller.run("repair the task")

    assert result.observations == 0
    assert planner.histories[1] == ()
    assert artifacts.errors == [
        {
            "action_index": 1,
            "category": "policy_error",
            "error_type": "PolicyViolation",
            "step": 1,
        }
    ]


def test_controller_stops_cleanly_when_action_budget_is_exhausted() -> None:
    planner = _Planner(
        [
            ReActStep(
                thought_summary="one",
                actions=(ShellAction(command="pwd"),),
                done=False,
            ),
            ReActStep(
                thought_summary="two",
                actions=(ShellAction(command="ls"),),
                done=False,
            ),
        ]
    )
    artifacts = _Artifacts()
    controller = ReActController(
        planner=planner,
        bridge=_Bridge(),
        artifacts=artifacts,
        budget=RunBudget(max_actions=1, max_model_calls=4),
    )

    result = controller.run("repair the task")

    assert result.stop_reason == "action_budget_exhausted"
    assert result.steps == 2
    assert result.observations == 1
    assert artifacts.errors == [{"category": "budget_error", "step": 2}]


def test_expired_deadline_rejects_react_action_before_action_reservation() -> None:
    from bayesprobe_terminal_bench.deadline import (
        DeadlineEnvironmentBridge,
        TrialDeadline,
    )

    deadline = TrialDeadline(timeout_seconds=5, monotonic=lambda: 0.0)
    budget = RunBudget(
        max_actions=1,
        reservation_guard=deadline.require_active,
    )
    planner = _Planner(
        [
            ReActStep(
                thought_summary="inspect",
                actions=(ShellAction(command="pwd"),),
                done=False,
            )
        ]
    )
    delegate = _Bridge()
    artifacts = _Artifacts()
    controller = ReActController(
        planner=planner,
        bridge=DeadlineEnvironmentBridge(
            delegate=delegate,
            deadline=deadline,
            configured_timeout_seconds=120,
        ),
        artifacts=artifacts,
        budget=budget,
    )

    result = controller.run("repair the task")

    assert result.stop_reason == "action_budget_exhausted"
    assert artifacts.errors == [{"category": "budget_error", "step": 1}]
    assert budget.actions_used == 0
    assert delegate.actions == []
