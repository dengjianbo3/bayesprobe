from __future__ import annotations

import pytest
from pydantic import ValidationError

from bayesprobe_terminal_bench.actions import (
    ActionObservation,
    ApplyPatchAction,
    ShellAction,
    TerminalProbePlan,
    WriteFileAction,
    action_may_mutate,
    shell_command_is_provably_read_only,
)


def test_inspect_plan_rejects_mutation() -> None:
    with pytest.raises(ValidationError, match="inspect plans require provably read-only actions"):
        TerminalProbePlan(
            mode="inspect",
            actions=[ShellAction(command="touch /tmp/x", mutates_environment=True)],
            expected_observation="The filesystem state is visible.",
        )


def test_model_cannot_mislabel_a_mutating_shell_command() -> None:
    with pytest.raises(ValidationError, match="inspect plans require provably read-only actions"):
        TerminalProbePlan(
            mode="inspect",
            actions=[ShellAction(command="rm -f output.txt", mutates_environment=False)],
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
        "rg --pre=rm needle target.txt",
        "rg --pre cat needle target.txt",
        "rg --pre-glob='*.txt' needle target.txt",
        "git status --ext-diff",
        "git diff --output=result.patch",
        "file --compile magic-file",
        "file -C",
    ],
)
def test_inspect_plan_rejects_allowlisted_executable_bypasses(command: str) -> None:
    with pytest.raises(ValidationError, match="inspect plans require provably read-only actions"):
        TerminalProbePlan(
            mode="inspect",
            actions=[ShellAction(command=command)],
            expected_observation="The command is rejected as potentially mutating.",
        )


@pytest.mark.parametrize("command", ["ls -la", "rg needle README.md", "git status"])
def test_inspect_plan_accepts_simple_read_only_commands(command: str) -> None:
    plan = TerminalProbePlan(
        mode="inspect",
        actions=[ShellAction(command=command)],
        expected_observation="The command is read-only.",
    )

    assert plan.actions[0].command == command


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


def test_verify_allows_shell_but_not_direct_file_writes() -> None:
    plan = TerminalProbePlan(
        mode="verify",
        actions=[ShellAction(command="pytest -q", mutates_environment=True)],
        expected_observation="The test result is observed.",
    )
    assert plan.mode == "verify"
    with pytest.raises(ValidationError, match="verify plans accept shell actions only"):
        TerminalProbePlan(
            mode="verify",
            actions=[WriteFileAction(path="/app/result.txt", content="x")],
            expected_observation="A file is written.",
        )


def test_intervene_requires_a_potentially_mutating_action() -> None:
    with pytest.raises(ValidationError, match="intervene plans require a mutating action"):
        TerminalProbePlan(
            mode="intervene",
            actions=[ShellAction(command="pwd")],
            expected_observation="The working directory is shown.",
        )


def test_plan_actions_are_immutable_after_json_array_validation() -> None:
    plan = TerminalProbePlan.model_validate_json(
        """{
            "mode": "inspect",
            "actions": [{"type": "shell", "command": "ls -la"}],
            "expected_observation": "Directory contents are visible."
        }"""
    )

    assert isinstance(plan.actions, tuple)
    with pytest.raises(AttributeError):
        plan.actions.append(ShellAction(command="touch output.txt"))
    with pytest.raises(ValidationError):
        plan.actions = (ShellAction(command="touch output.txt"),)
    with pytest.raises(ValidationError):
        plan.actions[0].command = "touch output.txt"
    assert plan.actions == (ShellAction(command="ls -la"),)


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
