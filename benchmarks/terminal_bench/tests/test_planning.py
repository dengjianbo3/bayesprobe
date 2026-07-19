from __future__ import annotations

import json
from hashlib import sha256
from types import SimpleNamespace

import pytest

from bayesprobe import CapabilityKind, ProbePurpose
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
        "steps": [
            {
                "role": "inspect",
                "action": {
                    "type": "shell",
                    "command": "pwd",
                    "timeout_seconds": 30,
                    "mutates_environment": False,
                },
            }
        ],
        "expected_observation": "The working directory is visible.",
    }
)

INTERVENTION_PLAN = json.dumps(
    {
        "mode": "intervene",
        "steps": [
            {
                "role": "intervene",
                "action": {
                    "type": "write_file",
                    "path": "/app/result",
                    "content": "done",
                },
            },
            {
                "role": "verify",
                "action": {"type": "shell", "command": "cat /app/result"},
                "verification_target": "The result contains done.",
            },
        ],
        "expected_observation": "The result changes.",
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
            budget=budget or RunBudget(max_actions=24, max_model_calls=3),
            client=client,
            invocation_observer=None if telemetry is None else telemetry.append,
        ),
        client,
    )


def test_planner_uses_initial_attempt_and_two_targeted_repairs(probe, execution_context) -> None:
    telemetry: list[dict[str, object]] = []
    planner, client = _planner(["not-json", "still-not-json", VALID_PLAN], telemetry=telemetry)

    plan = planner.plan(probe=probe, context=execution_context, history=())

    assert plan.steps[0].action.command == "pwd"
    assert len(client.chat.completions.calls) == 3
    assert [item["outcome"] for item in telemetry] == ["success"] * 3
    assert [item["repair"] for item in telemetry] == [False, True, True]
    assert [item["plan_validation"] for item in telemetry] == ["invalid", "invalid", "valid"]
    assert [item["response_sha256"] for item in telemetry] == [
        sha256(content.encode()).hexdigest()
        for content in ("not-json", "still-not-json", VALID_PLAN)
    ]

    first_repair = json.loads(client.chat.completions.calls[1]["messages"][1]["content"])
    second_repair = json.loads(client.chat.completions.calls[2]["messages"][1]["content"])
    assert first_repair["schema_version"] == "terminal_probe_plan:v1"
    assert first_repair["attempt_index"] == 1
    assert second_repair["attempt_index"] == 2
    assert first_repair["validation_error"]
    assert all(":" in item for item in first_repair["validation_error"])
    assert first_repair["invalid_response_sha256"] == sha256(b"not-json").hexdigest()
    assert "not-json" not in json.dumps(first_repair["invalid_payload"])


def test_planner_repair_exposes_safe_semantic_error_code(
    probe,
    execution_context,
) -> None:
    invalid_inspect_plan = json.dumps(
        {
            "mode": "inspect",
            "steps": [
                {
                    "role": "inspect",
                    "action": {
                        "type": "shell",
                        "command": "python -V",
                        "timeout_seconds": 30,
                        "mutates_environment": False,
                    },
                }
            ],
            "expected_observation": "The Python version is visible.",
        }
    )
    planner, client = _planner([invalid_inspect_plan, VALID_PLAN])

    planner.plan(probe=probe, context=execution_context, history=())

    repair = json.loads(client.chat.completions.calls[1]["messages"][1]["content"])
    assert repair["validation_error"] == ["plan:inspect_read_only_actions"]


def test_planner_never_falls_back_to_an_imagined_action(probe, execution_context) -> None:
    planner, client = _planner(["bad", "still bad", "bad a third time"])

    with pytest.raises(TerminalPlanError, match="provider contract failed after 3 attempts") as raised:
        planner.plan(probe=probe, context=execution_context, history=())

    assert raised.value.category == "provider_contract_error"
    assert raised.value.attempts == 3
    assert len(client.chat.completions.calls) == 3


def test_provider_controlled_field_locations_are_sanitized_everywhere(
    probe,
    execution_context,
) -> None:
    secret_field = "sk-abcdefghijklmnop1234"
    invalid_payload = json.loads(VALID_PLAN)
    invalid_payload["steps"][0][secret_field] = "provider-controlled-value"
    invalid_plan = json.dumps(invalid_payload)
    telemetry: list[dict[str, object]] = []
    planner, client = _planner(
        [invalid_plan, invalid_plan, invalid_plan],
        telemetry=telemetry,
    )

    with pytest.raises(TerminalPlanError) as raised:
        planner.plan(probe=probe, context=execution_context, history=())

    repair_payloads = [
        json.loads(call["messages"][1]["content"])
        for call in client.chat.completions.calls[1:]
    ]
    planner_artifacts = json.dumps(
        {
            "exception": str(raised.value),
            "repair_requests": repair_payloads,
            "observer_telemetry": telemetry,
        },
        sort_keys=True,
    )
    assert secret_field not in planner_artifacts
    expected_errors = [
        "steps.0.<field>:extra_forbidden",
        "steps:too_short",
    ]
    assert [item["field_errors"] for item in telemetry] == [expected_errors] * 3
    assert [item["validation_error"] for item in repair_payloads] == [
        expected_errors
    ] * 2


def test_empty_or_missing_choices_is_repaired_without_indexing_the_response(probe, execution_context) -> None:
    telemetry: list[dict[str, object]] = []
    missing_choices = SimpleNamespace(id="response_missing", choices=[], usage=None)
    planner, client = _planner([missing_choices, VALID_PLAN], telemetry=telemetry)

    plan = planner.plan(probe=probe, context=execution_context, history=())

    assert plan.steps[0].action.command == "pwd"
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
            "attempt_index": 0,
            "logical_call_index": 1,
            "outcome": "error",
            "error_type": "RuntimeError",
            "plan_validation": "not_attempted",
            "field_errors": [],
            "response_sha256": None,
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
    budget = RunBudget(max_actions=24, max_model_calls=2)
    planner, client = _planner(["bad", "still bad", VALID_PLAN], budget=budget)

    with pytest.raises(BudgetExhausted, match="^model call budget exhausted$"):
        planner.plan(probe=probe, context=execution_context, history=())

    assert len(client.chat.completions.calls) == 2
    assert budget.model_calls_used == 2


def test_transition_predictions_must_cover_probe_targets(probe, execution_context) -> None:
    incomplete_plan = json.dumps(
        {
            "mode": "intervene",
            "steps": [
                {
                    "role": "intervene",
                    "action": {"type": "write_file", "path": "/app/x", "content": "x"},
                },
                {
                    "role": "verify",
                    "action": {"type": "shell", "command": "cat /app/x"},
                    "verification_target": "The file contains x.",
                },
            ],
            "expected_observation": "The file changes.",
            "transition_predictions": [
                {"hypothesis_id": "H_other", "expected_transition": "The file changes."}
            ],
        }
    )
    planner, client = _planner([incomplete_plan, VALID_PLAN])

    plan = planner.plan(probe=probe, context=execution_context, history=())

    assert plan.mode == "inspect"
    assert len(client.chat.completions.calls) == 2
    repair = json.loads(client.chat.completions.calls[1]["messages"][1]["content"])
    assert any("transition_predictions" in item for item in repair["validation_error"])


def test_frame_coverage_probe_declares_required_inspect_mode(
    probe,
    execution_context,
) -> None:
    coverage_probe = probe.model_copy(
        update={
            "purpose": ProbePurpose.FRAME_COVERAGE,
            "required_capability": CapabilityKind.REPOSITORY_READ,
        }
    )

    payload = terminal_plan_input(
        probe=coverage_probe,
        context=execution_context,
        history=(),
    )

    assert payload["probe"]["purpose"] == "frame_coverage"
    assert payload["probe"]["required_capability"] == "repository_read"
    assert payload["probe"]["required_plan_mode"] == "inspect"


def test_frame_coverage_probe_repairs_intervention_until_inspect(
    probe,
    execution_context,
) -> None:
    coverage_probe = probe.model_copy(
        update={
            "purpose": ProbePurpose.FRAME_COVERAGE,
            "required_capability": CapabilityKind.REPOSITORY_READ,
        }
    )
    planner, client = _planner(
        [INTERVENTION_PLAN, INTERVENTION_PLAN, VALID_PLAN]
    )

    plan = planner.plan(
        probe=coverage_probe,
        context=execution_context,
        history=(),
    )

    assert plan.mode == "inspect"
    assert len(client.chat.completions.calls) == 3
    for call in client.chat.completions.calls[1:]:
        repair = json.loads(call["messages"][1]["content"])
        assert "plan:required_probe_mode" in repair["validation_error"]


def test_noncoverage_probe_retains_intervention_mode(probe, execution_context) -> None:
    planner, client = _planner([INTERVENTION_PLAN])

    plan = planner.plan(probe=probe, context=execution_context, history=())

    assert plan.mode == "intervene"
    assert len(client.chat.completions.calls) == 1


def test_planner_instruction_states_causal_execution_semantics(probe, execution_context) -> None:
    planner, client = _planner([VALID_PLAN])

    planner.plan(probe=probe, context=execution_context, history=())

    instruction = client.chat.completions.calls[0]["messages"][0]["content"]
    assert "terminal_probe_plan:v1" in instruction
    assert "Writes and patches are interventions" in instruction
    assert "Successful mutation output is acknowledgement, not verification" in instruction
    assert "Verification must follow the mutation" in instruction
    assert "task text names a public test or validation command" in instruction
    assert "weaker proxy observation is not equivalent" in instruction
    assert "Transition predictions are optional; when provided" in instruction
    assert "they must be declared before execution" in instruction
    assert "cover every Probe target hypothesis" in instruction
    assert "differentiated expected transitions" in instruction
    assert "Transition predictions must be declared" not in instruction
    assert '"inspect"' in instruction
    assert '"provably_read_only_shell_only"' in instruction
    assert '"verify"' in instruction
    assert '"non_empty_verification_target"' in instruction
    assert '"task_supplied_check"' in instruction
    assert '"proxy_observation"' in instruction
    assert '"intervene"' in instruction
    assert '"optional_inspect_one_intervene_one_or_more_verify"' in instruction
    assert '"required_plan_mode"' in instruction


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

    assert plan.steps[0].action.command == "pwd"
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

    assert plan.steps[0].action.command == "pwd"
    assert client.with_options_calls == [{"max_retries": 0}]
    assert len(client.derived_client.chat.completions.calls) == 1
    assert budget.model_calls_used == 1
    assert len(telemetry) == 1


def test_exploding_response_accessors_use_two_repairs_and_emit_one_record_per_return(
    probe,
    execution_context,
) -> None:
    telemetry: list[dict[str, object]] = []
    budget = RunBudget(max_actions=24, max_model_calls=3)
    planner, client = _planner(
        [ExplodingContentResponse(), ExplodingContentResponse(), ExplodingContentResponse()],
        budget=budget,
        telemetry=telemetry,
    )

    with pytest.raises(TerminalPlanError, match="provider contract failed after 3 attempts") as error:
        planner.plan(probe=probe, context=execution_context, history=())

    assert "response-access-secret" not in str(error.value)
    assert len(client.chat.completions.calls) == 3
    assert budget.model_calls_used == 3
    assert len(telemetry) == 3
    assert [item["outcome"] for item in telemetry] == ["empty_content"] * 3
    assert [item["repair"] for item in telemetry] == [False, True, True]
    assert "response-access-secret" not in json.dumps(telemetry)


def test_malformed_choice_sequences_cannot_escape_the_repair_path(probe, execution_context) -> None:
    telemetry: list[dict[str, object]] = []
    planner, client = _planner(
        [
            MalformedChoiceSequenceResponse(),
            MalformedChoiceSequenceResponse(),
            MalformedChoiceSequenceResponse(),
        ],
        telemetry=telemetry,
    )

    with pytest.raises(TerminalPlanError, match="provider contract failed after 3 attempts"):
        planner.plan(probe=probe, context=execution_context, history=())

    assert len(client.chat.completions.calls) == 3
    assert len(telemetry) == 3
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

    assert planner.plan(probe=probe, context=execution_context, history=()).steps[0].action.command == "pwd"
