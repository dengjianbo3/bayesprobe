from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from bayesprobe_terminal_bench.actions import ActionObservation, ShellAction
from bayesprobe_terminal_bench.config import BudgetExhausted, RunBudget, TerminalBenchConfig
from bayesprobe_terminal_bench.planning import (
    OpenAICompatibleTerminalProbePlanner,
    TerminalPlanError,
    terminal_plan_input,
)


VALID_PLAN = json.dumps(
    {
        "mode": "inspect",
        "actions": [
            {
                "type": "shell",
                "command": "pwd",
                "timeout_seconds": 30,
                "mutates_environment": False,
            }
        ],
        "expected_observation": "The working directory is visible.",
    }
)


class FakeCompletions:
    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        if isinstance(response, str):
            return SimpleNamespace(
                id="response_1",
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content=response),
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=12,
                    completion_tokens=8,
                    total_tokens=20,
                ),
            )
        return response


class FakeClient:
    def __init__(self, responses: list[object]) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions(responses))


def _planner(
    responses: list[object],
    *,
    budget: RunBudget | None = None,
    telemetry: list[dict[str, object]] | None = None,
) -> tuple[OpenAICompatibleTerminalProbePlanner, FakeClient]:
    client = FakeClient(responses)
    return (
        OpenAICompatibleTerminalProbePlanner(
            config=TerminalBenchConfig(model="test-model"),
            budget=budget or RunBudget(max_actions=24, max_model_calls=2),
            client=client,
            invocation_observer=None if telemetry is None else telemetry.append,
        ),
        client,
    )


def test_planner_repairs_invalid_json_once(probe, execution_context) -> None:
    telemetry: list[dict[str, object]] = []
    planner, client = _planner(["not-json", VALID_PLAN], telemetry=telemetry)

    plan = planner.plan(probe=probe, context=execution_context, history=())

    assert plan.actions[0].command == "pwd"
    assert len(client.chat.completions.calls) == 2
    assert [item["outcome"] for item in telemetry] == ["success", "success"]
    assert [item["repair"] for item in telemetry] == [False, True]
    assert [item["plan_validation"] for item in telemetry] == ["invalid", "valid"]


def test_planner_never_falls_back_to_an_imagined_action(probe, execution_context) -> None:
    planner, client = _planner(["bad", "still bad", VALID_PLAN])

    with pytest.raises(TerminalPlanError, match="^terminal plan validation failed$"):
        planner.plan(probe=probe, context=execution_context, history=())

    assert len(client.chat.completions.calls) == 2


def test_empty_or_missing_choices_is_repaired_without_indexing_the_response(probe, execution_context) -> None:
    telemetry: list[dict[str, object]] = []
    missing_choices = SimpleNamespace(id="response_missing", choices=[], usage=None)
    planner, client = _planner([missing_choices, VALID_PLAN], telemetry=telemetry)

    plan = planner.plan(probe=probe, context=execution_context, history=())

    assert plan.actions[0].command == "pwd"
    assert len(client.chat.completions.calls) == 2
    assert [item["outcome"] for item in telemetry] == ["empty_content", "success"]
    assert [item["plan_validation"] for item in telemetry] == ["invalid", "valid"]


def test_provider_failure_is_stable_and_telemetry_does_not_expose_error_text(probe, execution_context) -> None:
    telemetry: list[dict[str, object]] = []
    planner, _ = _planner(
        [RuntimeError("provider included provider-secret in an error")],
        telemetry=telemetry,
    )

    with pytest.raises(TerminalPlanError, match="^terminal planner provider request failed$"):
        planner.plan(probe=probe, context=execution_context, history=())

    assert telemetry == [
        {
            "task": "terminal_probe_plan",
            "model": "test-model",
            "repair": False,
            "logical_call_index": 1,
            "outcome": "error",
            "error_type": "RuntimeError",
            "plan_validation": "not_attempted",
            "latency_seconds": pytest.approx(telemetry[0]["latency_seconds"]),
        }
    ]
    assert "provider-secret" not in json.dumps(telemetry)


def test_budget_exhaustion_is_preserved_before_the_provider_is_called(probe, execution_context) -> None:
    planner, client = _planner([VALID_PLAN], budget=RunBudget(max_actions=24, max_model_calls=0))

    with pytest.raises(BudgetExhausted, match="^model call budget exhausted$"):
        planner.plan(probe=probe, context=execution_context, history=())

    assert client.chat.completions.calls == []


def test_repair_consumes_the_same_shared_budget(probe, execution_context) -> None:
    budget = RunBudget(max_actions=24, max_model_calls=1)
    planner, client = _planner(["bad", VALID_PLAN], budget=budget)

    with pytest.raises(BudgetExhausted, match="^model call budget exhausted$"):
        planner.plan(probe=probe, context=execution_context, history=())

    assert len(client.chat.completions.calls) == 1
    assert budget.model_calls_used == 1


def test_planner_request_uses_the_existing_chat_completion_token_parameter(probe, execution_context) -> None:
    planner, client = _planner([VALID_PLAN])

    planner.plan(probe=probe, context=execution_context, history=())

    request = client.chat.completions.calls[0]
    assert request["model"] == "test-model"
    assert request["max_tokens"] == 8_192
    assert "max_completion_tokens" not in request
    assert request["temperature"] == 0


def test_terminal_plan_input_excludes_private_state_and_uses_only_capped_history_output(probe) -> None:
    context = SimpleNamespace(
        problem="Repair the task workspace.",
        task_context="Use the provided task workspace only.",
        task_frame={
            "safe": "included",
            "priors": "prior-value",
            "posterior": "posterior-value",
            "score": "score-value",
            "credentials": "credential-value",
            "verifier_path": "/logs/verifier/reward.txt",
            "verifierPath": "/solution/answer.txt",
            "chain_of_thought": "reasoning-value",
            "nested": {"reasoning": "nested-reasoning", "safe": "included"},
        },
        hypotheses=(
            SimpleNamespace(
                id="H_workspace",
                statement="A workspace defect blocks task completion.",
                scope="task workspace",
                predictions=("Inspection exposes a concrete defect.",),
                falsifiers=("The workspace is already valid.",),
            ),
        ),
    )
    history = (
        ActionObservation(
            action_index=1,
            action=ShellAction(command="pwd"),
            stdout="uncapped-stdout-provider-secret",
            stderr="uncapped-stderr-provider-secret",
            return_code=0,
            duration_ms=1,
            pre_environment_state_id="env:0",
            post_environment_state_id="env:0",
            full_output_sha256="a" * 64,
            model_facing_output="capped model-facing output",
        ),
    )

    serialized = json.dumps(
        terminal_plan_input(probe=probe, context=context, history=history),
        sort_keys=True,
    )

    for forbidden in (
        "prior-value",
        "posterior-value",
        "score-value",
        "credential-value",
        "/logs/verifier/reward.txt",
        "/solution/answer.txt",
        "reasoning-value",
        "nested-reasoning",
        "uncapped-stdout-provider-secret",
        "uncapped-stderr-provider-secret",
    ):
        assert forbidden not in serialized
    assert "capped model-facing output" in serialized
    assert '"expected_information_gain"' not in serialized
    assert '"decision_relevance"' not in serialized
    assert '"cost_estimate"' not in serialized
    assert '"priority"' not in serialized
