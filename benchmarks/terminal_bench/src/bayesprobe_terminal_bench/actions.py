from __future__ import annotations

import shlex
from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShellAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    type: Literal["shell"] = "shell"
    command: str = Field(min_length=1, max_length=32_768)
    timeout_seconds: int = Field(default=120, ge=1, le=120)
    mutates_environment: bool = False


class WriteFileAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    type: Literal["write_file"] = "write_file"
    path: str = Field(min_length=1, max_length=4_096)
    content: str = Field(max_length=1_000_000)


class ApplyPatchAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    type: Literal["apply_patch"] = "apply_patch"
    patch: str = Field(min_length=1, max_length=1_000_000)
    strip: int = Field(default=0, ge=0, le=3)


TerminalAction = Annotated[
    ShellAction | WriteFileAction | ApplyPatchAction,
    Field(discriminator="type"),
]


_READ_ONLY_COMMANDS = frozenset({
    "cat",
    "file",
    "grep",
    "head",
    "ls",
    "md5sum",
    "pwd",
    "rg",
    "sha256sum",
    "stat",
    "tail",
    "test",
    "wc",
    "which",
})
_READ_ONLY_GIT_SUBCOMMANDS = frozenset({
    "diff",
    "log",
    "ls-files",
    "rev-parse",
    "show",
    "status",
})
_SHELL_COMPOSITION_MARKERS = ("\r", "\n", ";", "&", "|", ">", "<", "`", "$(")
_RG_PREPROCESSOR_OPTIONS = frozenset({"--pre", "--pre-glob"})
_GIT_UNSAFE_OPTIONS = frozenset({"--ext-diff", "--textconv", "--output"})
_FILE_UNSAFE_OPTIONS = frozenset({"-C", "--compile"})


def _arguments_are_provably_read_only(executable: str, arguments: list[str]) -> bool:
    if executable == "rg":
        return not any(
            argument in _RG_PREPROCESSOR_OPTIONS
            or any(argument.startswith(f"{option}=") for option in _RG_PREPROCESSOR_OPTIONS)
            for argument in arguments
        )
    if executable == "git":
        return not any(
            argument in _GIT_UNSAFE_OPTIONS
            or any(argument.startswith(f"{option}=") for option in _GIT_UNSAFE_OPTIONS)
            for argument in arguments
        )
    if executable == "file":
        return not any(
            argument in _FILE_UNSAFE_OPTIONS
            or argument.startswith("--compile=")
            or (argument.startswith("-") and not argument.startswith("--") and "C" in argument[1:])
            for argument in arguments
        )
    return True


def shell_command_is_provably_read_only(command: str) -> bool:
    if any(marker in command for marker in _SHELL_COMPOSITION_MARKERS):
        return False
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return False
    if not tokens:
        return False
    executable = tokens[0]
    if "/" in executable or "\\" in executable:
        return False
    if executable == "git":
        return (
            len(tokens) >= 2
            and tokens[1] in _READ_ONLY_GIT_SUBCOMMANDS
            and _arguments_are_provably_read_only(executable, tokens[2:])
        )
    return executable in _READ_ONLY_COMMANDS and _arguments_are_provably_read_only(
        executable, tokens[1:]
    )


def action_may_mutate(action: TerminalAction) -> bool:
    """Treat every action as mutating unless its shell command is allowlisted."""
    return not isinstance(action, ShellAction) or not shell_command_is_provably_read_only(
        action.command
    )


class TerminalProbePlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    mode: Literal["inspect", "intervene", "verify"]
    actions: tuple[TerminalAction, ...] = Field(min_length=1, max_length=3)
    expected_observation: str = Field(min_length=1, max_length=4_096)

    @model_validator(mode="before")
    @classmethod
    def normalize_actions(cls, value: object) -> object:
        if isinstance(value, Mapping) and isinstance(value.get("actions"), list):
            return {**value, "actions": tuple(value["actions"])}
        return value

    @model_validator(mode="after")
    def validate_mode(self) -> TerminalProbePlan:
        mutating = [action for action in self.actions if action_may_mutate(action)]
        if self.mode == "inspect" and mutating:
            raise ValueError("inspect plans require provably read-only actions")
        if self.mode == "verify" and any(
            not isinstance(action, ShellAction) for action in self.actions
        ):
            raise ValueError("verify plans accept shell actions only")
        if self.mode == "intervene" and not mutating:
            raise ValueError("intervene plans require a mutating action")
        return self


class ActionObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    action_index: int = Field(ge=1)
    action: TerminalAction
    stdout: str = ""
    stderr: str = ""
    return_code: int | None = None
    timed_out: bool = False
    error_category: str | None = None
    duration_ms: int = Field(ge=0)
    pre_environment_state_id: str
    post_environment_state_id: str
    full_output_sha256: str
    model_facing_output: str
    output_truncated: bool = False
