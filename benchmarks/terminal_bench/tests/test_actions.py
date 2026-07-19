from __future__ import annotations

import pytest
from pydantic import ValidationError

from bayesprobe_terminal_bench.actions import (
    ActionObservation,
    ApplyPatchAction,
    ShellAction,
    TerminalPlanStep,
    TerminalProbePlan,
    TransitionPrediction,
    WriteFileAction,
    action_may_mutate,
    shell_command_is_provably_read_only,
)


def _step(
    role: str,
    action: ShellAction | WriteFileAction | ApplyPatchAction,
    verification_target: str | None = None,
) -> TerminalPlanStep:
    return TerminalPlanStep(
        role=role,
        action=action,
        verification_target=verification_target,
    )


def test_inspect_plan_requires_read_only_inspect_steps() -> None:
    plan = TerminalProbePlan(
        mode="inspect",
        steps=[_step("inspect", ShellAction(command="git status"))],
        expected_observation="The repository state is visible.",
    )

    assert plan.steps[0].role == "inspect"

    with pytest.raises(ValidationError, match="inspect plans require provably read-only actions"):
        TerminalProbePlan(
            mode="inspect",
            steps=[_step("inspect", ShellAction(command="touch /tmp/x"))],
            expected_observation="The filesystem state is visible.",
        )


def test_model_cannot_mislabel_a_mutating_shell_command() -> None:
    with pytest.raises(ValidationError, match="inspect plans require provably read-only actions"):
        TerminalProbePlan(
            mode="inspect",
            steps=[_step("inspect", ShellAction(command="rm -f output.txt"))],
            expected_observation="The output is absent.",
        )


def test_model_label_does_not_override_read_only_allowlist() -> None:
    action = ShellAction(command="ls -la", mutates_environment=True)

    assert not action_may_mutate(action)


@pytest.mark.parametrize(
    "command",
    [
        "./ls --all",
        "/tmp/ls --all",
        "ls & touch /tmp/x",
        "rg --pre=rm needle target.txt",
        "rg --pre cat needle target.txt",
        "rg --pre-glob='*.txt' needle target.txt",
        "git status --ext-diff",
        "git diff --output=result.patch",
        "file --compile magic-file",
        "file -C",
        "file -Ckm custom.magic",
        "file -kCm custom.magic",
    ],
)
def test_inspect_plan_rejects_allowlisted_executable_bypasses(command: str) -> None:
    with pytest.raises(ValidationError, match="inspect plans require provably read-only actions"):
        TerminalProbePlan(
            mode="inspect",
            steps=[_step("inspect", ShellAction(command=command))],
            expected_observation="The command is rejected as potentially mutating.",
        )


@pytest.mark.parametrize("command", ["ls -la", "rg needle README.md", "git status"])
def test_inspect_plan_accepts_simple_read_only_commands(command: str) -> None:
    plan = TerminalProbePlan(
        mode="inspect",
        steps=[_step("inspect", ShellAction(command=command))],
        expected_observation="The command is read-only.",
    )

    assert plan.steps[0].action.command == command


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("git status", True),
        ("rg needle README.md", True),
        ("git checkout main", False),
        ("ls && pwd", False),
        ("echo $(pwd)", False),
        ("unclosed 'quote", False),
    ],
)
def test_read_only_commands_require_a_simple_allowlisted_command(
    command: str, expected: bool
) -> None:
    assert shell_command_is_provably_read_only(command) is expected


@pytest.mark.parametrize("command", ["ls & touch /tmp/x", "file -Ckm", "file -kCm"])
def test_classifier_rejects_composition_and_clustered_compile_options(command: str) -> None:
    assert not shell_command_is_provably_read_only(command)


def test_verify_requires_shell_steps_with_non_empty_targets() -> None:
    plan = TerminalProbePlan(
        mode="verify",
        steps=[_step("verify", ShellAction(command="pytest -q"), "The focused tests pass.")],
        expected_observation="The test result is observed.",
    )
    assert plan.mode == "verify"

    with pytest.raises(ValidationError, match="verification target"):
        TerminalProbePlan(
            mode="verify",
            steps=[_step("verify", ShellAction(command="pytest -q"))],
            expected_observation="A test result is observed.",
        )

    with pytest.raises(ValidationError, match="verification actions must be shell commands"):
        TerminalProbePlan(
            mode="verify",
            steps=[
                _step(
                    "verify",
                    WriteFileAction(path="/app/result.txt", content="x"),
                    "The file contains x.",
                )
            ],
            expected_observation="A file is written.",
        )


@pytest.mark.parametrize(
    "path",
    ["result.txt", "workspace/result.txt", "./result.txt", "../result.txt", r"C:\\result.txt"],
)
def test_write_file_requires_an_absolute_posix_path(path: str) -> None:
    with pytest.raises(ValidationError, match="absolute POSIX path"):
        WriteFileAction(path=path, content="result")


def test_intervene_accepts_inspect_then_one_mutation_then_verify() -> None:
    plan = TerminalProbePlan(
        mode="intervene",
        steps=[
            _step("inspect", ShellAction(command="cat /app/config.json")),
            _step("intervene", WriteFileAction(path="/app/config.json", content="{}")),
            _step(
                "verify",
                ShellAction(command="cat /app/config.json"),
                "The configuration contains {}.",
            ),
        ],
        expected_observation="The configuration changes and the read-back confirms it.",
    )

    assert [step.role for step in plan.steps] == ["inspect", "intervene", "verify"]


def test_intervene_accepts_test_execution_as_trailing_verification() -> None:
    plan = TerminalProbePlan(
        mode="intervene",
        steps=[
            _step(
                "intervene",
                WriteFileAction(path="/app/out.html", content="<p>candidate</p>"),
            ),
            _step(
                "verify",
                ShellAction(command="python /app/test_outputs.py"),
                "The public task check passes for /app/out.html.",
            ),
        ],
        expected_observation="The candidate file passes the task check.",
    )

    assert [step.role for step in plan.steps] == ["intervene", "verify"]


def test_required_probe_plan_mode_rejects_only_mismatched_modes() -> None:
    intervention = {
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

    with pytest.raises(ValidationError, match="required Probe plan mode"):
        TerminalProbePlan.model_validate(
            intervention,
            context={"required_plan_mode": "inspect"},
        )

    assert TerminalProbePlan.model_validate(intervention).mode == "intervene"
    inspect = TerminalProbePlan.model_validate(
        {
            "mode": "inspect",
            "steps": [
                {
                    "role": "inspect",
                    "action": {"type": "shell", "command": "git status"},
                }
            ],
            "expected_observation": "The repository state is visible.",
        },
        context={"required_plan_mode": "inspect"},
    )
    assert inspect.mode == "inspect"


def test_intervene_requires_the_declared_intervention_to_mutate() -> None:
    with pytest.raises(ValidationError, match="exactly one intended mutation"):
        TerminalProbePlan(
            mode="intervene",
            steps=[
                _step("intervene", ShellAction(command="cat /app/result")),
                _step(
                    "verify",
                    ShellAction(command="cat /app/result"),
                    "The result is visible.",
                ),
            ],
            expected_observation="The declared intervention changes the workspace.",
        )


def test_intervene_requires_trailing_verification() -> None:
    with pytest.raises(ValidationError, match="trailing verify"):
        TerminalProbePlan(
            mode="intervene",
            steps=[_step("intervene", WriteFileAction(path="/app/result", content="done"))],
            expected_observation="The result is written.",
        )


def test_intervene_rejects_verify_before_intervention() -> None:
    with pytest.raises(ValidationError, match="optional inspect, one intervene, then verify"):
        TerminalProbePlan(
            mode="intervene",
            steps=[
                _step("verify", ShellAction(command="cat /app/result"), "The result is absent."),
                _step("intervene", WriteFileAction(path="/app/result", content="done")),
            ],
            expected_observation="Verification cannot precede mutation.",
        )


def test_transition_predictions_are_forbidden_without_intervention() -> None:
    with pytest.raises(ValidationError, match="transition predictions require intervene mode"):
        TerminalProbePlan(
            mode="inspect",
            steps=[_step("inspect", ShellAction(command="pwd"))],
            expected_observation="The directory is visible.",
            transition_predictions=[
                TransitionPrediction(hypothesis_id="H1", expected_transition="Tests pass.")
            ],
        )


def test_transition_prediction_texts_must_be_normalized_and_distinct() -> None:
    with pytest.raises(ValidationError, match="distinct normalized texts"):
        TerminalProbePlan(
            mode="intervene",
            steps=[
                _step("intervene", WriteFileAction(path="/app/result", content="done")),
                _step("verify", ShellAction(command="cat /app/result"), "The result is done."),
            ],
            expected_observation="The result changes.",
            transition_predictions=[
                TransitionPrediction(hypothesis_id="H1", expected_transition=" Tests   pass "),
                TransitionPrediction(hypothesis_id="H2", expected_transition="tests pass"),
            ],
        )


def test_old_actions_field_is_rejected() -> None:
    with pytest.raises(ValidationError, match="actions"):
        TerminalProbePlan.model_validate(
            {
                "mode": "inspect",
                "actions": [{"type": "shell", "command": "pwd"}],
                "expected_observation": "The directory is visible.",
            }
        )


def test_plan_collections_are_immutable_after_json_array_validation() -> None:
    plan = TerminalProbePlan.model_validate_json(
        """{
            "mode": "intervene",
            "steps": [
                {"role": "intervene", "action": {"type": "write_file", "path": "/app/x", "content": "x"}},
                {"role": "verify", "action": {"type": "shell", "command": "cat /app/x"}, "verification_target": "The file contains x."}
            ],
            "expected_observation": "The file changes.",
            "transition_predictions": [
                {"hypothesis_id": "H1", "expected_transition": "The file now contains x."}
            ]
        }"""
    )

    assert isinstance(plan.steps, tuple)
    assert isinstance(plan.transition_predictions, tuple)
    with pytest.raises(AttributeError):
        plan.steps.append(_step("inspect", ShellAction(command="pwd")))
    with pytest.raises(AttributeError):
        plan.transition_predictions.append(
            TransitionPrediction(hypothesis_id="H2", expected_transition="Another transition.")
        )
    with pytest.raises(ValidationError):
        plan.steps = (_step("inspect", ShellAction(command="pwd")),)
    with pytest.raises(ValidationError):
        plan.steps[0].action = ShellAction(command="pwd")


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ShellAction(command="ls", timeout_seconds="120"),
        lambda: ShellAction(command="ls", mutates_environment="false"),
        lambda: ShellAction(command="ls", timeout_seconds=True),
        lambda: ActionObservation(
            action_index=True,
            action=ShellAction(command="ls"),
            duration_ms=True,
            pre_environment_state_id="before",
            post_environment_state_id="after",
            full_output_sha256="a" * 64,
            model_facing_output="Directory contents.",
        ),
    ],
)
def test_action_models_reject_coerced_and_boolean_values(factory: object) -> None:
    with pytest.raises(ValidationError):
        factory()


def test_observation_preserves_discriminated_action_details() -> None:
    observation = ActionObservation(
        action_index=1,
        action=ApplyPatchAction(patch="*** Begin Patch\n*** End Patch"),
        duration_ms=0,
        pre_environment_state_id="before",
        post_environment_state_id="after",
        full_output_sha256="a" * 64,
        model_facing_output="Applied patch.",
    )

    assert observation.action.type == "apply_patch"
    assert observation.return_code is None
