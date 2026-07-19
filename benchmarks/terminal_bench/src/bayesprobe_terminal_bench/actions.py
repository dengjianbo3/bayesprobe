from __future__ import annotations

import shlex
import unicodedata
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)


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

    @field_validator("path")
    @classmethod
    def require_absolute_posix_path(cls, value: str) -> str:
        if not PurePosixPath(value).is_absolute():
            raise ValueError("write_file path must be an absolute POSIX path")
        return value


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


class TerminalPlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    role: Literal["inspect", "intervene", "verify"]
    action: TerminalAction
    verification_target: str | None = Field(default=None, max_length=4_096)


class TransitionPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    hypothesis_id: str
    expected_transition: str = Field(min_length=1, max_length=4_096)


class TerminalProbePlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    mode: Literal["inspect", "intervene", "verify"]
    steps: tuple[TerminalPlanStep, ...] = Field(min_length=1, max_length=3)
    expected_observation: str = Field(min_length=1, max_length=4_096)
    transition_predictions: tuple[TransitionPrediction, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def normalize_collections(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        normalized = dict(value)
        if isinstance(normalized.get("steps"), list):
            normalized["steps"] = tuple(normalized["steps"])
        if isinstance(normalized.get("transition_predictions"), list):
            normalized["transition_predictions"] = tuple(
                normalized["transition_predictions"]
            )
        return normalized

    @field_validator("transition_predictions")
    @classmethod
    def validate_transition_predictions(
        cls,
        value: tuple[TransitionPrediction, ...],
        info: ValidationInfo,
    ) -> tuple[TransitionPrediction, ...]:
        if not value:
            return value
        if info.data.get("mode") != "intervene":
            raise ValueError("transition predictions require intervene mode")
        normalized = [_normalized_prediction(item.expected_transition) for item in value]
        if len(set(normalized)) != len(normalized):
            raise ValueError("transition predictions require distinct normalized texts")
        context = info.context if isinstance(info.context, Mapping) else {}
        targets = context.get("target_hypotheses")
        if targets is not None:
            target_ids = tuple(targets)
            prediction_ids = tuple(item.hypothesis_id for item in value)
            if len(prediction_ids) != len(target_ids) or set(prediction_ids) != set(
                target_ids
            ):
                raise ValueError("transition prediction IDs must equal Probe targets")
        return value

    @model_validator(mode="after")
    def validate_mode(self, info: ValidationInfo) -> TerminalProbePlan:
        context = info.context if isinstance(info.context, Mapping) else {}
        required_mode = context.get("required_plan_mode")
        if required_mode is not None and self.mode != required_mode:
            raise ValueError("plan mode must equal the required Probe plan mode")

        if self.mode == "inspect":
            if any(step.role != "inspect" for step in self.steps):
                raise ValueError("inspect plans require inspect roles")
            if any(action_may_mutate(step.action) for step in self.steps):
                raise ValueError("inspect plans require provably read-only actions")
            return self

        if self.mode == "verify":
            if any(step.role != "verify" for step in self.steps):
                raise ValueError("verify plans require verify roles")
            _validate_verification_steps(self.steps)
            return self

        roles = tuple(step.role for step in self.steps)
        intervention_indexes = [
            index for index, role in enumerate(roles) if role == "intervene"
        ]
        if len(intervention_indexes) != 1:
            raise ValueError(
                "intervene role order must be optional inspect, one intervene, then verify"
            )
        intervention_index = intervention_indexes[0]
        if (
            intervention_index not in (0, 1)
            or any(role != "inspect" for role in roles[:intervention_index])
            or any(role != "verify" for role in roles[intervention_index + 1 :])
        ):
            raise ValueError(
                "intervene role order must be optional inspect, one intervene, then verify"
            )
        verification_steps = self.steps[intervention_index + 1 :]
        if not verification_steps:
            raise ValueError("intervene plans require one or more trailing verify steps")
        if any(action_may_mutate(step.action) for step in self.steps[:intervention_index]):
            raise ValueError("inspect steps require provably read-only actions")
        if not action_may_mutate(self.steps[intervention_index].action):
            raise ValueError("intervene plans require exactly one intended mutation")
        _validate_verification_steps(verification_steps)
        return self


def _validate_verification_steps(steps: tuple[TerminalPlanStep, ...]) -> None:
    if any(not isinstance(step.action, ShellAction) for step in steps):
        raise ValueError("verification actions must be shell commands")
    if any(
        not isinstance(step.verification_target, str)
        or not step.verification_target.strip()
        for step in steps
    ):
        raise ValueError("verification steps require a non-empty verification target")


def _normalized_prediction(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


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
