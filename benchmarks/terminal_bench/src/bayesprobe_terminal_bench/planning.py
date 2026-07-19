from __future__ import annotations

import json
import posixpath
import re
import time
import unicodedata
from collections.abc import Callable, Mapping
from hashlib import sha256
from typing import Any, Literal, Protocol

from openai import OpenAI
from pydantic import ValidationError

from bayesprobe import (
    CapabilityKind,
    ExternalSignal,
    ProbeDesign,
    ProbeExecutionBrief,
    ProbePurpose,
)
from bayesprobe_terminal_bench.actions import (
    ActionObservation,
    ApplyPatchAction,
    ShellAction,
    TerminalProbePlan,
    WriteFileAction,
)
from bayesprobe_terminal_bench.config import RunBudget, TerminalBenchConfig


class TerminalPlanError(ValueError):
    def __init__(
        self,
        message: str | None = None,
        *,
        category: str | None = None,
        attempts: int | None = None,
    ) -> None:
        self.category = category
        self.attempts = attempts
        if message is None:
            if category == "provider_contract_error":
                message = f"terminal plan provider contract failed after {attempts} attempts"
            elif category == "provider_error":
                message = "terminal planner provider request failed"
            else:
                message = "terminal planner failed"
        super().__init__(message)


class TerminalProbePlanner(Protocol):
    def plan(
        self,
        *,
        probe: ProbeDesign,
        context: ProbeExecutionBrief,
        history: tuple[ActionObservation, ...],
    ) -> TerminalProbePlan: ...


_REDACTION_MARKER = "[REDACTED]"
_HISTORY_MODEL_FACING_OUTPUT_LIMIT = 4_096
_HISTORY_ACTION_TEXT_LIMIT = 4_096
_TERMINAL_PLAN_SCHEMA_VERSION = "terminal_probe_plan:v1"
_SAFE_PLAN_FIELD_LOCATION_COMPONENTS = frozenset(
    {
        "action",
        "command",
        "content",
        "expected_observation",
        "expected_transition",
        "hypothesis_id",
        "mode",
        "mutates_environment",
        "patch",
        "path",
        "role",
        "steps",
        "strip",
        "timeout_seconds",
        "transition_predictions",
        "type",
        "verification_target",
    }
)
_UNKNOWN_PLAN_FIELD_LOCATION = "<field>"
_SAFE_PLAN_SEMANTIC_ERROR_CODES = {
    "transition predictions require intervene mode": (
        "transition_predictions:requires_intervene_mode"
    ),
    "transition predictions require distinct normalized texts": (
        "transition_predictions:distinct_expected_transitions"
    ),
    "transition prediction IDs must equal Probe targets": (
        "transition_predictions:target_ids_must_equal_probe_targets"
    ),
    "plan mode must equal the required Probe plan mode": (
        "plan:required_probe_mode"
    ),
    "inspect plans require inspect roles": "plan:inspect_roles_only",
    "inspect plans require provably read-only actions": (
        "plan:inspect_read_only_actions"
    ),
    "verify plans require verify roles": "plan:verify_roles_only",
    "intervene role order must be optional inspect, one intervene, then verify": (
        "plan:intervene_role_order"
    ),
    "intervene plans require one or more trailing verify steps": (
        "plan:intervene_requires_trailing_verify"
    ),
    "inspect steps require provably read-only actions": (
        "plan:intervene_inspect_read_only"
    ),
    "intervene plans require exactly one intended mutation": (
        "plan:intervene_exactly_one_mutation"
    ),
    "verification actions must be shell commands": (
        "plan:verification_shell_only"
    ),
    "verification steps require a non-empty verification target": (
        "plan:verification_target_required"
    ),
}
_SECRET_VALUE_PATTERNS = (
    re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{12,}", re.IGNORECASE),
    re.compile(
        r"(?<![A-Za-z0-9_])(?:gh[pousr]_[A-Za-z0-9]{20,}|"
        r"github_pat_[A-Za-z0-9_]{20,})(?![A-Za-z0-9_])"
    ),
    re.compile(
        r"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{10,}\."
        r"[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}(?![A-Za-z0-9_-])"
    ),
    re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])"),
    re.compile(
        r"(?<![A-Za-z0-9])xox[a-z]-[A-Za-z0-9-]{10,}"
        r"(?![A-Za-z0-9-])",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:api[ _-]?key|access[ _-]?key|private[ _-]?key|password|passwd|"
        r"credential(?:s)?|cookie|secret|token)\b\s*(?:=|:)\s*[\"']?"
        r"(?:Bearer\s+)?[A-Za-z0-9._~+/=-]{6,}",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bauthorization\b\s*(?:=|:)\s*[\"']?Bearer\s+"
        r"[A-Za-z0-9._~+/=-]{8,}",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bBearer\s+(?=[A-Za-z0-9._~+/=-]{12,}(?:\s|$))"
        r"(?=[A-Za-z0-9._~+/=-]*[0-9._~+/=-])[A-Za-z0-9._~+/=-]{12,}",
        re.IGNORECASE,
    ),
    re.compile(r"\bBearer\s+[A-Za-z]{16,}\b", re.IGNORECASE),
    re.compile(r"-----BEGIN(?: [A-Z0-9]+)* PRIVATE KEY-----", re.IGNORECASE),
)
_SECRET_KEY_COMPOUNDS = {"apikey", "accesskey", "privatekey", "proxyauthorization"}
_SECRET_KEY_WORDS = {
    "authorization",
    "token",
    "secret",
    "password",
    "passwd",
    "credential",
    "credentials",
    "cookie",
}
_BENIGN_SECRET_KEY_FOLLOWERS = {
    "token": {"count"},
    "password": {"policy"},
    "credential": {"score"},
    "credentials": {"score"},
    "cookie": {"policy"},
}
_SECRET_KEY_SEQUENCES = {("api", "key"), ("access", "key"), ("private", "key")}
_EPISTEMIC_FIELD_WORDS = {
    "prior",
    "priors",
    "posterior",
    "posteriors",
    "score",
    "scores",
    "confidence",
    "probability",
    "reasoning",
    "cot",
    "thought",
}
_BENIGN_EPISTEMIC_FIELD_SEQUENCES = {("credential", "score"), ("credentials", "score")}
_EVALUATOR_PATH_SUFFIXES = {"path", "dir", "directory", "file", "files", "root"}
_PROTECTED_ABSOLUTE_EVALUATOR_PATHS = (
    "/logs/verifier",
    "/solution",
    "/tests",
    "/var/run/docker.sock",
    "/run/docker.sock",
)
_PROTECTED_RELATIVE_EVALUATOR_PATHS = ("logs/verifier", "solution", "tests")
_EVALUATOR_PATH_PATTERN = re.compile(
    r"""(?ix)
    (?<![A-Za-z0-9_])
    (?:
        [/\\]+(?:logs[/\\]+verifier|solution|tests)(?:[/\\]+[^\s'\"<>]*)?
        |(?:(?:\.[/\\]+|\.\.[/\\]+))+(?:logs[/\\]+verifier|solution|tests)(?:[/\\]+[^\s'\"<>]*)?
        |logs[/\\]+verifier(?:[/\\]+[^\s'\"<>]*)?
        |(?:solution|tests)[/\\]+[^\s'\"<>]*
        |(?:[/\\]+|(?:(?:\.[/\\]+|\.\.[/\\]+)*))?(?:[^\s/'\"<>\\]+[/\\]+)*docker\.sock\b
    )
    """,
)
_ASSIGNMENT_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?P<key>[A-Za-z][A-Za-z0-9_-]*)\s*(?:=|:)\s*"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)


def _required_plan_mode(probe: ProbeDesign) -> Literal["inspect"] | None:
    if (
        probe.purpose is ProbePurpose.FRAME_COVERAGE
        and probe.required_capability is CapabilityKind.REPOSITORY_READ
    ):
        return "inspect"
    return None


def terminal_plan_input(
    *,
    probe: ProbeDesign,
    context: ProbeExecutionBrief,
    history: tuple[ActionObservation, ...],
) -> dict[str, Any]:
    """Build the blind, bounded context the terminal planner may receive."""
    payload = {
        "task": {
            "problem": context.problem,
            "task_context": context.task_context,
            "task_frame": context.task_frame,
        },
        "hypotheses": [
            {
                "id": item.id,
                "statement": item.statement,
                "scope": item.scope,
                "predictions": list(item.predictions),
                "falsifiers": list(item.falsifiers),
            }
            for item in context.hypotheses
        ],
        "probe": {
            "id": probe.id,
            "purpose": probe.purpose.value,
            "inquiry_goal": probe.inquiry_goal,
            "method": probe.method,
            "expected_observation": probe.expected_observation,
            "target_hypotheses": list(probe.target_hypotheses),
            "required_capability": probe.required_capability.value,
            "required_plan_mode": _required_plan_mode(probe),
            "support_condition": dict(probe.support_condition),
            "weaken_condition": dict(probe.weaken_condition),
            "reframe_condition": (
                None
                if probe.reframe_condition is None
                else dict(probe.reframe_condition)
            ),
        },
        "recent_observations": [
            {
                "action": _history_action_input(item),
                "observation": _bounded_text(
                    item.model_facing_output,
                    _HISTORY_MODEL_FACING_OUTPUT_LIMIT,
                ),
                "return_code": item.return_code,
                "timed_out": item.timed_out,
                "environment_state_id": item.post_environment_state_id,
            }
            for item in history[-12:]
        ],
    }
    sanitized = _sanitize_outbound(payload)
    for observation in sanitized["recent_observations"]:
        observation["observation"] = _bounded_text(
            observation["observation"],
            _HISTORY_MODEL_FACING_OUTPUT_LIMIT,
        )
    return sanitized


class OpenAICompatibleTerminalProbePlanner:
    def __init__(
        self,
        *,
        config: TerminalBenchConfig,
        budget: RunBudget,
        api_key: str | None = None,
        client: Any | None = None,
        invocation_observer: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._config = config
        self._budget = budget
        self._invocation_observer = invocation_observer
        if client is None:
            if not isinstance(api_key, str) or not api_key.strip():
                raise ValueError("terminal planner requires an explicit API key")
            client = OpenAI(
                api_key=api_key.strip(),
                base_url=config.base_url,
                timeout=config.provider_timeout_seconds,
                max_retries=0,
            )
        else:
            client = _single_attempt_client(client)
        self._client = client

    def plan(
        self,
        *,
        probe: ProbeDesign,
        context: ProbeExecutionBrief,
        history: tuple[ActionObservation, ...],
    ) -> TerminalProbePlan:
        original_input = terminal_plan_input(
            probe=probe,
            context=context,
            history=history,
        )
        payload = original_input
        for attempt_index in range(3):
            plan, content, field_errors = self._complete(
                payload=payload,
                probe=probe,
                repair=attempt_index > 0,
                attempt_index=attempt_index,
            )
            if plan is not None:
                return plan
            if attempt_index < 2:
                payload = _repair_payload(
                    original_input=original_input,
                    invalid_content=content,
                    field_errors=field_errors,
                    attempt_index=attempt_index + 1,
                )
        raise TerminalPlanError(
            category="provider_contract_error",
            attempts=3,
        ) from None

    def _complete(
        self,
        *,
        payload: dict[str, Any],
        probe: ProbeDesign,
        repair: bool,
        attempt_index: int,
    ) -> tuple[TerminalProbePlan | None, str | None, tuple[str, ...]]:
        logical_call_index = self._budget.reserve_model_call()
        started = time.monotonic()
        response: Any = None
        try:
            response = self._client.chat.completions.create(
                model=self._config.model,
                messages=[
                    {
                        "role": "system",
                        "content": _planner_instruction(repair=repair),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            payload,
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=self._config.max_output_tokens,
            )
        except Exception as error:
            self._observe_attempt(
                repair=repair,
                logical_call_index=logical_call_index,
                started=started,
                outcome="error",
                plan_validation="not_attempted",
                attempt_index=attempt_index,
                field_errors=(),
                response_sha256=None,
                error_type=type(error).__name__,
            )
            raise TerminalPlanError(
                category="provider_error",
                attempts=attempt_index + 1,
            ) from None

        content = _chat_completion_content(response)
        if not _has_text_content(content):
            field_errors = ("response:missing",)
            self._observe_attempt(
                repair=repair,
                logical_call_index=logical_call_index,
                started=started,
                outcome="empty_content",
                plan_validation="invalid",
                attempt_index=attempt_index,
                field_errors=field_errors,
                response_sha256=None,
                response=response,
            )
            return None, None, field_errors

        try:
            plan = TerminalProbePlan.model_validate_json(
                content,
                context={
                    "target_hypotheses": tuple(probe.target_hypotheses),
                    "required_plan_mode": _required_plan_mode(probe),
                },
            )
        except ValidationError as error:
            field_errors = _safe_field_errors(error)
            self._observe_attempt(
                repair=repair,
                logical_call_index=logical_call_index,
                started=started,
                outcome="success",
                plan_validation="invalid",
                attempt_index=attempt_index,
                field_errors=field_errors,
                response_sha256=_content_sha256(content),
                response=response,
            )
            return None, content, field_errors

        self._observe_attempt(
            repair=repair,
            logical_call_index=logical_call_index,
            started=started,
            outcome="success",
            plan_validation="valid",
            attempt_index=attempt_index,
            field_errors=(),
            response_sha256=_content_sha256(content),
            response=response,
        )
        return plan, content, ()

    def _observe_attempt(
        self,
        *,
        repair: bool,
        logical_call_index: int,
        started: float,
        outcome: str,
        plan_validation: str,
        attempt_index: int,
        field_errors: tuple[str, ...],
        response_sha256: str | None,
        response: Any = None,
        error_type: str | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "task": "terminal_probe_plan",
            "model": _redact_sensitive_text(self._config.model),
            "repair": repair,
            "attempt_index": attempt_index,
            "logical_call_index": logical_call_index,
            "outcome": outcome,
            "plan_validation": plan_validation,
            "field_errors": list(field_errors),
            "response_sha256": response_sha256,
            "latency_seconds": max(0.0, time.monotonic() - started),
        }
        if response is not None:
            try:
                record.update(_response_telemetry(response))
            except Exception:
                record["response_metadata"] = "unavailable"
        if error_type is not None:
            record["error_type"] = _redact_sensitive_text(error_type)
        if self._invocation_observer is None:
            return
        try:
            self._invocation_observer(record)
        except Exception:
            return


def _planner_instruction(*, repair: bool) -> str:
    request = (
        "Repair one terminal action plan and return JSON only."
        if repair
        else "Plan one bounded terminal Probe. Return JSON only. Do not claim any command ran."
    )
    return (
        request
        + f" Schema version: {_TERMINAL_PLAN_SCHEMA_VERSION}."
        + " Writes and patches are interventions."
        + " Successful mutation output is acknowledgement, not verification."
        + " Verification must follow the mutation."
        + " Transition predictions are optional; when provided, they must be declared"
        + " before execution, cover every Probe target hypothesis, and have"
        + " differentiated expected transitions."
        + " Semantic contract: "
        + json.dumps(
            {
                "inspect": {
                    "actions": "provably_read_only_shell_only",
                    "forbidden_shell_composition": [
                        "semicolon",
                        "ampersand",
                        "pipe",
                        "redirect",
                        "command_substitution",
                        "executable_path",
                    ],
                    "roles": ["inspect"],
                    "safe_command_examples": [
                        "pwd",
                        "ls",
                        "cat",
                        "rg",
                        "grep",
                        "head",
                        "tail",
                        "stat",
                        "wc",
                        "git status",
                        "git diff",
                    ],
                },
                "verify": {
                    "actions": "shell_only",
                    "requirement": "non_empty_verification_target",
                    "roles": ["verify"],
                },
                "intervene": {
                    "mutation_count": 1,
                    "mutation_role": "intervene",
                    "role_order": (
                        "optional_inspect_one_intervene_one_or_more_verify"
                    ),
                    "verification": "required_after_mutation",
                },
                "transition_predictions": {
                    "allowed_mode": "intervene_only",
                    "coverage": "exactly_probe_target_hypotheses",
                    "expected_transitions": "distinct_non_empty_text",
                    "required": False,
                },
                "required_plan_mode": {
                    "source": "probe.required_plan_mode",
                    "rule": "plan_mode_must_equal_when_present",
                },
            },
            sort_keys=True,
        )
        + " Schema: "
        + json.dumps(
            TerminalProbePlan.model_json_schema(),
            sort_keys=True,
        )
    )


def _identity_sha256(value: Any) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return f"sha256:{sha256(serialized.encode('utf-8')).hexdigest()}"


def plan_contract_identity() -> dict[str, str]:
    """Return canonical hashes for terminal planning and Signal schemas."""
    return {
        "terminal_probe_plan:v1:prompt": _identity_sha256(
            _planner_instruction(repair=False)
        ),
        "terminal_probe_plan:v1:repair_prompt": _identity_sha256(
            _planner_instruction(repair=True)
        ),
        "terminal_probe_plan:v1:schema": _identity_sha256(
            {
                "schema": TerminalProbePlan.model_json_schema(),
                "schema_version": _TERMINAL_PLAN_SCHEMA_VERSION,
            }
        ),
        "harbor-observation:v3:schema": _identity_sha256(
            {
                "causal_binding_fields": [
                    "action_id",
                    "action_role",
                    "plan_id",
                    "policy_attempt_id",
                    "request_fingerprint",
                    "subject_environment_state_id",
                    "verification_target",
                ],
                "max_observation_bytes": 32_768,
                "schema": ExternalSignal.model_json_schema(),
                "schema_version": "harbor-observation:v3",
            }
        ),
    }


def _repair_payload(
    *,
    original_input: dict[str, Any],
    invalid_content: str | None,
    field_errors: tuple[str, ...],
    attempt_index: int,
) -> dict[str, Any]:
    return {
        "schema_version": _TERMINAL_PLAN_SCHEMA_VERSION,
        "original_input": original_input,
        "invalid_payload": _redacted_content_shape(invalid_content),
        "invalid_response_sha256": (
            None if invalid_content is None else _content_sha256(invalid_content)
        ),
        "validation_error": list(field_errors),
        "attempt_index": attempt_index,
    }


def _safe_field_errors(error: ValidationError) -> tuple[str, ...]:
    errors: set[str] = set()
    for item in error.errors(include_url=False, include_input=False):
        semantic_code = _safe_semantic_error_code(item)
        if semantic_code is not None:
            errors.add(semantic_code)
            continue
        location = ".".join(
            _safe_field_location_component(part) for part in item["loc"]
        )
        errors.add(f"{location}:{item['type']}")
    return tuple(sorted(errors))[:32]


def _safe_semantic_error_code(item: Mapping[str, Any]) -> str | None:
    context = item.get("ctx")
    error = context.get("error") if isinstance(context, Mapping) else None
    if not isinstance(error, ValueError):
        return None
    return _SAFE_PLAN_SEMANTIC_ERROR_CODES.get(str(error))


def _safe_field_location_component(value: object) -> str:
    if type(value) is int:
        return str(value)
    if isinstance(value, str) and value in _SAFE_PLAN_FIELD_LOCATION_COMPONENTS:
        return value
    return _UNKNOWN_PLAN_FIELD_LOCATION


def _content_sha256(content: str) -> str:
    return sha256(content.encode("utf-8")).hexdigest()


def _redacted_content_shape(content: str | None) -> Any:
    if content is None:
        return None
    try:
        value = json.loads(content)
    except (TypeError, ValueError):
        return "[REDACTED]"
    return _redacted_payload_shape(value)


def _redacted_payload_shape(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            f"field_{index}": _redacted_payload_shape(item)
            for index, item in enumerate(value.values())
        }
    if isinstance(value, list | tuple):
        return [_redacted_payload_shape(item) for item in value]
    return "[REDACTED]"


def _history_action_input(observation: ActionObservation) -> dict[str, Any]:
    action = observation.action
    if isinstance(action, ShellAction):
        return {
            "type": action.type,
            "command": _bounded_text(action.command, _HISTORY_ACTION_TEXT_LIMIT),
            "timeout_seconds": action.timeout_seconds,
            "mutates_environment": action.mutates_environment,
        }
    if isinstance(action, WriteFileAction):
        return {
            "type": action.type,
            "path": _bounded_text(action.path, _HISTORY_ACTION_TEXT_LIMIT),
        }
    if isinstance(action, ApplyPatchAction):
        return {"type": action.type, "strip": action.strip}
    raise TypeError("unsupported terminal action")


def _bounded_text(value: str, limit: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= limit:
        return value
    pieces: list[str] = []
    used = 0
    index = 0
    while index < len(value):
        piece = (
            _REDACTION_MARKER
            if value.startswith(_REDACTION_MARKER, index)
            else value[index]
        )
        piece_size = len(piece.encode("utf-8"))
        if used + piece_size > limit:
            break
        pieces.append(piece)
        used += piece_size
        index += len(piece)
    return "".join(pieces)


def _single_attempt_client(client: Any) -> Any:
    """Disable SDK retries; clients without controls are single-attempt transports."""
    with_options = getattr(client, "with_options", None)
    if not callable(with_options):
        return client
    return with_options(max_retries=0)


def _sanitize_outbound(value: Any) -> Any:
    """Drop forbidden fields and redact sensitive string fragments recursively."""
    if isinstance(value, str):
        return _redact_sensitive_text(value)
    if isinstance(value, Mapping):
        return {
            key: _sanitize_outbound(item)
            for key, item in value.items()
            if isinstance(key, str) and not _is_private_field_name(key)
        }
    if isinstance(value, list | tuple):
        return [_sanitize_outbound(item) for item in value]
    return value


def _is_private_field_name(value: str) -> bool:
    return (
        _is_secret_field_name(value)
        or _is_epistemic_field_name(value)
        or _is_evaluator_path_field_name(value)
    )


def _semantic_key_parts(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", value)
    separated = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", normalized)
    return tuple(re.findall(r"[a-z0-9]+", separated.casefold()))


def _is_secret_field_name(value: str) -> bool:
    parts = _semantic_key_parts(value)
    if not parts:
        return False
    compact = "".join(parts)
    if compact in _SECRET_KEY_COMPOUNDS:
        return True
    if any(
        tuple(parts[index:index + 2]) in _SECRET_KEY_SEQUENCES
        for index in range(len(parts) - 1)
    ):
        return True
    for index, part in enumerate(parts):
        if part not in _SECRET_KEY_WORDS:
            continue
        follower = parts[index + 1] if index + 1 < len(parts) else None
        if follower in _BENIGN_SECRET_KEY_FOLLOWERS.get(part, set()):
            continue
        return True
    return False


def _is_epistemic_field_name(value: str) -> bool:
    parts = _semantic_key_parts(value)
    if any(
        tuple(parts[index:index + 2]) in _BENIGN_EPISTEMIC_FIELD_SEQUENCES
        for index in range(len(parts) - 1)
    ):
        return False
    return any(part in _EPISTEMIC_FIELD_WORDS for part in parts) or any(
        tuple(parts[index:index + 3]) == ("chain", "of", "thought")
        for index in range(len(parts) - 2)
    )


def _is_evaluator_path_field_name(value: str) -> bool:
    parts = _semantic_key_parts(value)
    if parts in {
        ("verifier",),
        ("solution",),
        ("hidden", "test"),
        ("hidden", "tests"),
    }:
        return True
    has_hidden_tests = any(
        tuple(parts[index:index + 2]) in {("hidden", "test"), ("hidden", "tests")}
        for index in range(len(parts) - 1)
    )
    return (
        bool(set(parts).intersection(_EVALUATOR_PATH_SUFFIXES))
        and ("verifier" in parts or "solution" in parts or has_hidden_tests)
    )


def _redact_sensitive_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value)
    for pattern in _SECRET_VALUE_PATTERNS:
        text = pattern.sub(_REDACTION_MARKER, text)
    text = _ASSIGNMENT_PATTERN.sub(_redact_sensitive_assignment, text)
    if _is_protected_evaluator_path(text):
        return _REDACTION_MARKER
    return _EVALUATOR_PATH_PATTERN.sub(_REDACTION_MARKER, text)


def _is_protected_evaluator_path(value: str) -> bool:
    normalized = posixpath.normpath(re.sub(r"[/\\]+", "/", value))
    if normalized in (".", ""):
        return False
    if normalized == "docker.sock" or normalized.endswith("/docker.sock"):
        return True
    if normalized.startswith("/"):
        return any(
            normalized == path or normalized.startswith(f"{path}/")
            for path in _PROTECTED_ABSOLUTE_EVALUATOR_PATHS
        )
    relative = normalized
    while relative == ".." or relative.startswith("../"):
        relative = relative[3:] if relative.startswith("../") else ""
    return any(
        relative == path or relative.startswith(f"{path}/")
        for path in _PROTECTED_RELATIVE_EVALUATOR_PATHS
    )


def _redact_sensitive_assignment(match: re.Match[str]) -> str:
    key = match.group("key")
    return (
        _REDACTION_MARKER
        if _is_secret_field_name(key) or _is_epistemic_field_name(key)
        else match.group(0)
    )


def _chat_completion_content(response: Any) -> str | None:
    choice = _first_choice(response)
    if choice is None:
        return None
    message = _response_value(choice, "message")
    if message is None:
        return None
    content = _response_value(message, "content")
    return content if isinstance(content, str) else None


def _has_text_content(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return bool(str.strip(value))
    except Exception:
        return False


def _response_telemetry(response: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "usage": _usage_telemetry(_response_value(response, "usage")),
    }
    response_id = _response_value(response, "id")
    if isinstance(response_id, str):
        record["response_id"] = _redact_sensitive_text(response_id)
    choice = _first_choice(response)
    finish_reason = None if choice is None else _response_value(choice, "finish_reason")
    if isinstance(finish_reason, str):
        record["finish_reason"] = _redact_sensitive_text(finish_reason)
    return record


def _usage_telemetry(usage: Any) -> dict[str, int | None]:
    return {
        "input_tokens": _integer_response_value(usage, "prompt_tokens"),
        "output_tokens": _integer_response_value(usage, "completion_tokens"),
        "total_tokens": _integer_response_value(usage, "total_tokens"),
    }


def _integer_response_value(value: Any, name: str) -> int | None:
    result = _response_value(value, name)
    return result if type(result) is int else None


def _first_choice(response: Any) -> Any | None:
    choices = _response_value(response, "choices")
    if not isinstance(choices, list | tuple):
        return None
    try:
        if len(choices) < 1:
            return None
        return choices[0]
    except Exception:
        return None


def _response_value(value: Any, name: str) -> Any:
    try:
        if isinstance(value, Mapping):
            return value.get(name)
        return getattr(value, name, None)
    except Exception:
        return None
