from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from bayesprobe_terminal_bench.actions import (
    ActionObservation,
    ApplyPatchAction,
    ShellAction,
    WriteFileAction,
)
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


class RetryControlledClient:
    def __init__(self, responses: list[object]) -> None:
        self.with_options_calls: list[dict[str, object]] = []
        self.derived_client = FakeClient(responses)

    def with_options(self, **kwargs: object) -> FakeClient:
        self.with_options_calls.append(kwargs)
        return self.derived_client


class ExplodingContentResponse:
    @property
    def choices(self) -> object:
        raise RuntimeError("response-access-secret")

    @property
    def id(self) -> object:
        raise RuntimeError("response-access-secret")

    @property
    def usage(self) -> object:
        raise RuntimeError("response-access-secret")


class ExplodingChoiceSequence(list[object]):
    def __bool__(self) -> bool:
        raise RuntimeError("choice-sequence-secret")


class MalformedChoiceSequenceResponse:
    def __init__(self) -> None:
        self.choices = ExplodingChoiceSequence()


class ExplodingMetadataResponse:
    def __init__(self, content: str) -> None:
        self.choices = [
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=content),
            )
        ]

    @property
    def id(self) -> object:
        raise RuntimeError("metadata-access-secret")

    @property
    def usage(self) -> object:
        raise RuntimeError("metadata-access-secret")


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


def test_terminal_plan_input_recursively_sanitizes_every_outbound_value(probe) -> None:
    context = SimpleNamespace(
        problem="ordinary task text remains visible; never read /solution/answer.txt",
        task_context="ordinary task context remains visible; do not inspect //logs/verifier",
        task_frame={
            "safe": "included",
            "priors": "prior-value",
            "posterior": "posterior-value",
            "score": "score-value",
            "credentials": "credential-value",
            "verifier_path": "/logs/verifier/reward.txt",
            "verifierPath": "/solution/answer.txt",
            "chain_of_thought": "reasoning-value",
            "nested": {
                "reasoning": "nested-reasoning",
                "safe": "included",
                "non_recognized": (
                    "posterior=0.9 confidence=0.8 "
                    "database_password=frame-secret credential=frame-credential"
                ),
            },
        },
        hypotheses=(
            SimpleNamespace(
                id="H_workspace",
                statement="A workspace defect has database_password=hypothesis-secret.",
                scope="task workspace with confidence=0.7",
                predictions=("Inspection exposes credential=hypothesis-credential.",),
                falsifiers=("The workspace is already valid.",),
            ),
        ),
    )
    history = (
        ActionObservation(
            action_index=1,
            action=WriteFileAction(
                path="/solution/answer.txt",
                content="write-content-secret database_password=write-secret",
            ),
            stdout="uncapped-stdout-provider-secret",
            stderr="uncapped-stderr-provider-secret",
            return_code=0,
            duration_ms=1,
            pre_environment_state_id="env:0",
            post_environment_state_id="env:0",
            full_output_sha256="a" * 64,
            model_facing_output="secret=history-secret credential=history-credential" + "x" * 5_000,
        ),
        ActionObservation(
            action_index=2,
            action=ApplyPatchAction(
                patch="*** Begin Patch\npatch-content-secret token=patch-token\n*** End Patch",
            ),
            return_code=0,
            duration_ms=1,
            pre_environment_state_id="env:0",
            post_environment_state_id="env:1",
            full_output_sha256="b" * 64,
            model_facing_output="patch history output",
        ),
        ActionObservation(
            action_index=3,
            action=ShellAction(command="cat //logs/verifier"),
            return_code=0,
            duration_ms=1,
            pre_environment_state_id="env:1",
            post_environment_state_id="env:1",
            full_output_sha256="c" * 64,
            model_facing_output="ordinary history output",
        ),
    )
    unsafe_probe = probe.model_copy(
        update={
            "inquiry_goal": "Inspect /solution/answer.txt only after credential=probe-credential.",
            "method": "Inspect tokenization after database_password=probe-secret.",
            "expected_observation": "confidence=0.99 must not cross the boundary.",
            "support_condition": {"reasoning": "reasoning-value", "safe": "token=probe-token"},
            "weaken_condition": {"safe": "//logs/verifier must remain hidden"},
            "reframe_condition": {"safe": "hidden tests are unavailable"},
        }
    )

    payload = terminal_plan_input(probe=unsafe_probe, context=context, history=history)
    serialized = json.dumps(payload, sort_keys=True)

    for forbidden in (
        "prior-value",
        "posterior-value",
        "score-value",
        "credential-value",
        "/logs/verifier/reward.txt",
        "/solution/answer.txt",
        "//logs/verifier",
        "posterior",
        "confidence",
        "database_password",
        "credential",
        "reasoning-value",
        "nested-reasoning",
        "frame-secret",
        "frame-credential",
        "hypothesis-secret",
        "hypothesis-credential",
        "probe-secret",
        "probe-token",
        "write-content-secret",
        "patch-content-secret",
        "patch-token",
        "history-secret",
        "history-credential",
        "uncapped-stdout-provider-secret",
        "uncapped-stderr-provider-secret",
    ):
        assert forbidden not in serialized
    assert "ordinary task text remains visible" in serialized
    assert "ordinary task context remains visible" in serialized
    assert "[REDACTED]" in serialized
    assert "content" not in payload["recent_observations"][0]["action"]
    assert "patch" not in payload["recent_observations"][1]["action"]
    assert payload["recent_observations"][2]["action"]["type"] == "shell"
    assert len(payload["recent_observations"][0]["observation"]) <= 4_096
    assert '"expected_information_gain"' not in serialized
    assert '"decision_relevance"' not in serialized
    assert '"cost_estimate"' not in serialized
    assert '"priority"' not in serialized


def test_terminal_plan_input_preserves_benign_security_related_identifiers(probe) -> None:
    context = SimpleNamespace(
        problem="Run rg token src and follow the password-policy task text.",
        task_context="Tokenization remains an ordinary implementation concern.",
        task_frame={
            "token_count": 3,
            "password_policy": "required",
            "credential_score": 0.8,
        },
        hypotheses=(
            SimpleNamespace(
                id="H_tokenization",
                statement="Tokenization may be incomplete.",
                scope="source tree",
                predictions=("rg token src finds the relevant code.",),
                falsifiers=("No tokenization code exists.",),
            ),
        ),
    )
    benign_probe = probe.model_copy(
        update={
            "target_hypotheses": ["H_tokenization"],
            "method": "rg token src",
            "inquiry_goal": "Inspect the password-policy implementation.",
        }
    )

    payload = terminal_plan_input(probe=benign_probe, context=context, history=())
    serialized = json.dumps(payload, sort_keys=True)

    assert "rg token src" in serialized
    assert "password-policy task text" in serialized
    assert "H_tokenization" in serialized
    assert payload["task"]["task_frame"] == {
        "token_count": 3,
        "password_policy": "required",
        "credential_score": 0.8,
    }


def test_terminal_plan_input_redacts_relative_and_absolute_evaluator_paths(probe) -> None:
    protected_paths = (
        "solution/answer.txt",
        "./solution/answer.txt",
        "../solution/answer.txt",
        "//solution//answer.txt",
        "tests/hidden.py",
        "./tests/hidden.py",
        "../../tests/hidden.py",
        "logs/verifier/reward.txt",
        "./logs/verifier/reward.txt",
        "../logs/verifier/reward.txt",
        "//logs//verifier//reward.txt",
        "/var/run/docker.sock",
        "//run//docker.sock",
        "./docker.sock",
        "../docker.sock",
        "var/run/docker.sock",
    )
    context = SimpleNamespace(
        problem="The solution is ordinary prose; tests are ordinary prose too.",
        task_context="The docker socket phrase is ordinary documentation.",
        task_frame={"paths": list(protected_paths)},
        hypotheses=(
            SimpleNamespace(
                id="H_workspace",
                statement="The workspace needs inspection.",
                scope="workspace",
                predictions=("A file reveals the issue.",),
                falsifiers=("The workspace is empty.",),
            ),
        ),
    )

    payload = terminal_plan_input(probe=probe, context=context, history=())
    serialized = json.dumps(payload, sort_keys=True)

    assert all(path not in serialized for path in protected_paths)
    assert payload["task"]["task_frame"]["paths"] == ["[REDACTED]"] * len(protected_paths)
    assert "The solution is ordinary prose" in serialized
    assert "tests are ordinary prose too" in serialized
    assert "docker socket phrase is ordinary documentation" in serialized


def test_terminal_plan_input_redacts_evaluator_directories_and_windows_paths(probe) -> None:
    protected_paths = (
        "solution",
        "./solution",
        "../solution",
        "//solution",
        "tests",
        "./tests",
        "../tests",
        "//tests",
        "logs/verifier",
        "./logs/verifier",
        "../logs/verifier",
        "//logs//verifier",
        r"\solution\answer.txt",
        r".\solution",
        r"..\tests",
        r"logs\verifier",
    )
    context = SimpleNamespace(
        problem="The solution is ordinary prose; tests are ordinary prose too.",
        task_context="The logs verifier words are ordinary documentation.",
        task_frame={"paths": list(protected_paths)},
        hypotheses=(),
    )

    payload = terminal_plan_input(probe=probe, context=context, history=())
    serialized = json.dumps(payload, sort_keys=True)

    assert payload["task"]["task_frame"]["paths"] == ["[REDACTED]"] * len(protected_paths)
    assert "The solution is ordinary prose" in serialized
    assert "tests are ordinary prose too" in serialized
    assert "logs verifier words are ordinary documentation" in serialized


def test_history_output_is_byte_capped_after_redaction_expands_it(probe, execution_context) -> None:
    history = (
        ActionObservation(
            action_index=1,
            action=ShellAction(command="pwd"),
            duration_ms=0,
            pre_environment_state_id="env:0",
            post_environment_state_id="env:0",
            full_output_sha256="e" * 64,
            model_facing_output="a" * 4_089 + " tests/",
        ),
    )

    output = terminal_plan_input(probe=probe, context=execution_context, history=history)[
        "recent_observations"
    ][0]["observation"]

    assert "tests/" not in output
    assert len(output.encode("utf-8")) <= 4_096


def test_history_output_truncation_is_utf8_byte_bounded(probe, execution_context) -> None:
    history = (
        ActionObservation(
            action_index=1,
            action=ShellAction(command="pwd"),
            duration_ms=0,
            pre_environment_state_id="env:0",
            post_environment_state_id="env:0",
            full_output_sha256="d" * 64,
            model_facing_output="a" * 4_095 + "é" + "ignored",
        ),
    )

    payload = terminal_plan_input(probe=probe, context=execution_context, history=history)
    output = payload["recent_observations"][0]["observation"]

    assert output == "a" * 4_095
    assert len(output.encode("utf-8")) <= 4_096


def test_successful_response_with_exploding_metadata_still_emits_one_record(probe, execution_context) -> None:
    telemetry: list[dict[str, object]] = []
    planner, client = _planner(
        [ExplodingMetadataResponse(VALID_PLAN)],
        telemetry=telemetry,
    )

    plan = planner.plan(probe=probe, context=execution_context, history=())

    assert plan.actions[0].command == "pwd"
    assert len(client.chat.completions.calls) == 1
    assert len(telemetry) == 1
    assert telemetry[0]["outcome"] == "success"
    assert "metadata-access-secret" not in json.dumps(telemetry)


def test_telemetry_redacts_authorization_and_openai_style_response_identifiers(
    probe,
    execution_context,
) -> None:
    telemetry: list[dict[str, object]] = []
    response = SimpleNamespace(
        id="sk-abcdefghijklmno123456789",
        choices=[
            SimpleNamespace(
                finish_reason="Authorization: Bearer abcdefghijklmnop",
                message=SimpleNamespace(content=VALID_PLAN),
            )
        ],
        usage=None,
    )
    planner, _ = _planner([response], telemetry=telemetry)

    planner.plan(probe=probe, context=execution_context, history=())

    serialized = json.dumps(telemetry)
    assert "sk-abcdefghijklmno123456789" not in serialized
    assert "Authorization: Bearer abcdefghijklmnop" not in serialized
    assert "[REDACTED]" in serialized


def test_telemetry_redacts_secret_like_provider_error_class_names(probe, execution_context) -> None:
    secret_error_type = type("ProviderError_sk-abcdefghijklmno123456789", (RuntimeError,), {})
    telemetry: list[dict[str, object]] = []
    planner, _ = _planner([secret_error_type()], telemetry=telemetry)

    with pytest.raises(TerminalPlanError, match="^terminal planner provider request failed$"):
        planner.plan(probe=probe, context=execution_context, history=())

    serialized = json.dumps(telemetry)
    assert "sk-abcdefghijklmno123456789" not in serialized
    assert "[REDACTED]" in serialized


def test_injected_sdk_client_is_derived_with_no_retries_and_used_once(probe, execution_context) -> None:
    telemetry: list[dict[str, object]] = []
    budget = RunBudget(max_actions=24, max_model_calls=2)
    client = RetryControlledClient([VALID_PLAN])
    planner = OpenAICompatibleTerminalProbePlanner(
        config=TerminalBenchConfig(model="test-model"),
        budget=budget,
        client=client,
        invocation_observer=telemetry.append,
    )

    plan = planner.plan(probe=probe, context=execution_context, history=())

    assert plan.actions[0].command == "pwd"
    assert client.with_options_calls == [{"max_retries": 0}]
    assert len(client.derived_client.chat.completions.calls) == 1
    assert budget.model_calls_used == 1
    assert len(telemetry) == 1


def test_exploding_response_accessors_use_one_repair_and_emit_one_record_per_return(
    probe,
    execution_context,
) -> None:
    telemetry: list[dict[str, object]] = []
    budget = RunBudget(max_actions=24, max_model_calls=2)
    planner, client = _planner(
        [ExplodingContentResponse(), ExplodingContentResponse()],
        budget=budget,
        telemetry=telemetry,
    )

    with pytest.raises(TerminalPlanError, match="^terminal plan validation failed$") as error:
        planner.plan(probe=probe, context=execution_context, history=())

    assert "response-access-secret" not in str(error.value)
    assert len(client.chat.completions.calls) == 2
    assert budget.model_calls_used == 2
    assert len(telemetry) == 2
    assert [item["outcome"] for item in telemetry] == ["empty_content", "empty_content"]
    assert [item["repair"] for item in telemetry] == [False, True]
    assert "response-access-secret" not in json.dumps(telemetry)


def test_malformed_choice_sequences_cannot_escape_the_repair_path(probe, execution_context) -> None:
    telemetry: list[dict[str, object]] = []
    planner, client = _planner(
        [MalformedChoiceSequenceResponse(), MalformedChoiceSequenceResponse()],
        telemetry=telemetry,
    )

    with pytest.raises(TerminalPlanError, match="^terminal plan validation failed$"):
        planner.plan(probe=probe, context=execution_context, history=())

    assert len(client.chat.completions.calls) == 2
    assert len(telemetry) == 2
    assert "choice-sequence-secret" not in json.dumps(telemetry)


def test_observer_failure_does_not_change_planner_result(probe, execution_context) -> None:
    client = FakeClient([VALID_PLAN])

    def fail_observer(_: dict[str, object]) -> None:
        raise RuntimeError("observer failure")

    planner = OpenAICompatibleTerminalProbePlanner(
        config=TerminalBenchConfig(model="test-model"),
        budget=RunBudget(max_actions=24, max_model_calls=2),
        client=client,
        invocation_observer=fail_observer,
    )

    assert planner.plan(probe=probe, context=execution_context, history=()).actions[0].command == "pwd"
