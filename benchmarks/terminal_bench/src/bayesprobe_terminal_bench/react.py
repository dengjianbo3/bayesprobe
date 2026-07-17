from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from bayesprobe_terminal_bench.actions import (
    ActionObservation,
    ApplyPatchAction,
    ShellAction,
    TerminalAction,
    WriteFileAction,
)
from bayesprobe_terminal_bench.config import (
    BudgetExhausted,
    DeadlineExhausted,
    ProviderIdentityError,
    RunBudget,
    TerminalBenchConfig,
)
from bayesprobe_terminal_bench.environment import PolicyViolation
from bayesprobe_terminal_bench.planning import (
    _bounded_text,
    _chat_completion_content,
    _content_sha256,
    _has_text_content,
    _history_action_input,
    _redact_sensitive_text,
    _response_telemetry,
    _response_value,
    _sanitize_outbound,
    _single_attempt_client,
)
from bayesprobe_terminal_bench.provider_contract import safe_field_errors


_REDACTION_MARKER = "[REDACTED]"


class ReActPlanError(ValueError):
    def __init__(
        self,
        message: str | None = None,
        *,
        category: str = "adapter_error",
        attempts: int | None = None,
    ) -> None:
        self.category = category
        self.attempts = attempts
        if message is None:
            if category == "provider_contract_error":
                message = f"ReAct provider contract failed after {attempts} attempts"
            elif category == "provider_transport_error":
                message = "ReAct planner provider request failed"
            else:
                message = "ReAct planner failed"
        super().__init__(message)


class ReActStep(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    thought_summary: str = Field(min_length=1, max_length=2_048)
    actions: tuple[TerminalAction, ...] = Field(default=(), max_length=3)
    done: bool = False
    completion_summary: str | None = Field(default=None, max_length=4_096)

    @model_validator(mode="before")
    @classmethod
    def normalize_actions(cls, value: object) -> object:
        if isinstance(value, Mapping) and isinstance(value.get("actions"), list):
            return {**value, "actions": tuple(value["actions"])}
        return value

    @model_validator(mode="after")
    def validate_completion(self) -> ReActStep:
        if self.done and self.actions:
            raise ValueError("completed steps cannot contain actions")
        if self.done and not self.completion_summary:
            raise ValueError("completed steps require a completion summary")
        if not self.done and not self.actions:
            raise ValueError("unfinished steps require actions")
        if not self.done and self.completion_summary is not None:
            raise ValueError("unfinished steps cannot contain a completion summary")
        return self


def react_step_input(
    *,
    instruction: str,
    history: tuple[ActionObservation, ...],
) -> dict[str, Any]:
    payload = {
        "task": instruction,
        "recent_observations": [
            {
                "action": _history_action_input(item),
                "observation": item.model_facing_output,
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
            4_096,
        )
    return sanitized


class OpenAICompatibleReActPlanner:
    def __init__(
        self,
        *,
        config: TerminalBenchConfig,
        budget: RunBudget,
        api_key: str | None = None,
        client: Any | None = None,
        invocation_observer: Callable[[dict[str, Any]], None] | None = None,
        expected_provider_model: str | None = None,
        expected_system_fingerprint: str | None | object = ...,
    ) -> None:
        self._config = config
        self._budget = budget
        self._invocation_observer = invocation_observer
        self._expected_provider_model = expected_provider_model
        self._expected_system_fingerprint = expected_system_fingerprint
        if client is None:
            if not isinstance(api_key, str) or not api_key.strip():
                raise ValueError("ReAct planner requires an explicit API key")
            client = OpenAI(
                api_key=api_key.strip(),
                base_url=config.base_url,
                timeout=config.provider_timeout_seconds,
                max_retries=0,
            )
        else:
            client = _single_attempt_client(client)
        self._client = client

    def next_step(
        self,
        *,
        instruction: str,
        history: tuple[ActionObservation, ...],
    ) -> ReActStep:
        original_input = react_step_input(instruction=instruction, history=history)
        payload = original_input
        for attempt_index in range(3):
            step, response_sha256, field_errors = self._complete(
                payload=payload,
                repair=attempt_index > 0,
                attempt_index=attempt_index,
            )
            if step is not None:
                return step
            if attempt_index < 2:
                payload = {
                    "schema_version": "react_step:v1",
                    "original_input": original_input,
                    "invalid_response_sha256": response_sha256,
                    "field_errors": list(field_errors),
                    "attempt_index": attempt_index + 1,
                }
        raise ReActPlanError(
            category="provider_contract_error",
            attempts=3,
        ) from None

    def _complete(
        self,
        *,
        payload: dict[str, Any],
        repair: bool,
        attempt_index: int,
    ) -> tuple[ReActStep | None, str | None, tuple[str, ...]]:
        logical_call_index = self._budget.reserve_model_call()
        started = time.monotonic()
        response: Any = None
        try:
            response = self._client.chat.completions.create(
                model=self._config.model,
                messages=[
                    {
                        "role": "system",
                        "content": _react_instruction(repair=repair),
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
        except BudgetExhausted:
            raise
        except Exception as error:
            self._observe_attempt(
                repair=repair,
                logical_call_index=logical_call_index,
                started=started,
                outcome="error",
                validation="not_attempted",
                attempt_index=attempt_index,
                field_errors=(),
                response_sha256=None,
                error_type=type(error).__name__,
            )
            raise ReActPlanError(
                category="provider_transport_error",
                attempts=attempt_index + 1,
            ) from None

        try:
            self._record_response_accounting(response)
        except BudgetExhausted as error:
            try:
                self._observe_attempt(
                    repair=repair,
                    logical_call_index=logical_call_index,
                    started=started,
                    outcome="success",
                    validation="not_attempted",
                    attempt_index=attempt_index,
                    field_errors=(),
                    response_sha256=None,
                    response=response,
                )
            except Exception as observer_error:
                raise error from observer_error
            raise

        content = _chat_completion_content(response)
        if not _has_text_content(content):
            field_errors = ("response:missing",)
            self._observe_attempt(
                repair=repair,
                logical_call_index=logical_call_index,
                started=started,
                outcome="empty_content",
                validation="invalid",
                attempt_index=attempt_index,
                field_errors=field_errors,
                response_sha256=None,
                response=response,
            )
            return None, None, field_errors
        try:
            step = ReActStep.model_validate_json(content)
        except ValidationError as error:
            field_errors = safe_field_errors(error)
            response_sha256 = _content_sha256(content)
            self._observe_attempt(
                repair=repair,
                logical_call_index=logical_call_index,
                started=started,
                outcome="success",
                validation="invalid",
                attempt_index=attempt_index,
                field_errors=field_errors,
                response_sha256=response_sha256,
                response=response,
            )
            return None, response_sha256, field_errors
        self._observe_attempt(
            repair=repair,
            logical_call_index=logical_call_index,
            started=started,
            outcome="success",
            validation="valid",
            attempt_index=attempt_index,
            field_errors=(),
            response_sha256=_content_sha256(content),
            response=response,
        )
        return step, _content_sha256(content), ()

    def _record_response_accounting(self, response: object) -> None:
        model = _response_value(response, "model")
        if (
            self._expected_provider_model is not None
            and model != self._expected_provider_model
        ):
            raise ProviderIdentityError("provider model identity drift")
        fingerprint = _response_value(response, "system_fingerprint")
        if (
            self._expected_system_fingerprint is not ...
            and fingerprint != self._expected_system_fingerprint
        ):
            raise ProviderIdentityError("provider system fingerprint drift")
        usage = _response_value(response, "usage")
        self._budget.record_provider_usage(
            _response_value(usage, "total_tokens")
        )

    def _observe_attempt(
        self,
        *,
        repair: bool,
        logical_call_index: int,
        started: float,
        outcome: str,
        validation: str,
        attempt_index: int,
        field_errors: tuple[str, ...],
        response_sha256: str | None,
        response: Any = None,
        error_type: str | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "task": "react_step",
            "model": _redact_sensitive_text(self._config.model),
            "repair": repair,
            "attempt_index": attempt_index,
            "logical_call_index": logical_call_index,
            "outcome": outcome,
            "step_validation": validation,
            "field_errors": list(field_errors),
            "response_sha256": response_sha256,
            "latency_seconds": max(0.0, time.monotonic() - started),
        }
        if response is not None:
            record.update(_response_telemetry(response))
            record["provider_model"] = _redact_sensitive_text(
                str(_response_value(response, "model") or "")
            )
            fingerprint = _response_value(response, "system_fingerprint")
            record["system_fingerprint_available"] = fingerprint is not None
            record["system_fingerprint"] = (
                _redact_sensitive_text(fingerprint)
                if isinstance(fingerprint, str)
                else None
            )
        if error_type is not None:
            record["error_type"] = error_type
        if self._invocation_observer is not None:
            self._invocation_observer(record)


def _react_instruction(*, repair: bool) -> str:
    request = (
        "Repair one Direct/ReAct terminal step and return JSON only."
        if repair
        else (
            "Take one bounded Direct/ReAct step toward completing the task. "
            "Use only observed command results as facts. Return JSON only. "
            "Set done=true only after the environment deliverable has been verified."
        )
    )
    return request + " Schema: " + json.dumps(
        ReActStep.model_json_schema(),
        sort_keys=True,
    )


@dataclass(frozen=True)
class ReActRunResult:
    stop_reason: str
    completion_summary: str | None
    steps: int
    observations: int


def _safe_plan_action(action: TerminalAction) -> dict[str, object]:
    if isinstance(action, ShellAction):
        return {
            "type": action.type,
            "command": _bounded_text(_redact_sensitive_text(action.command), 4_096),
            "timeout_seconds": action.timeout_seconds,
            "mutates_environment": action.mutates_environment,
        }
    if isinstance(action, WriteFileAction):
        return {
            "type": action.type,
            "path": _redact_sensitive_text(action.path),
        }
    if isinstance(action, ApplyPatchAction):
        return {
            "type": action.type,
            "strip": action.strip,
        }
    raise TypeError(f"unsupported ReAct action: {type(action).__name__}")


def _safe_observation_artifact(observation: ActionObservation) -> dict[str, Any]:
    payload = _sanitize_outbound(observation.model_dump(mode="json"))
    action = observation.action
    if isinstance(action, WriteFileAction):
        payload["action"] = {
            **_safe_plan_action(action),
            "content": _REDACTION_MARKER,
        }
    elif isinstance(action, ApplyPatchAction):
        payload["action"] = {
            **_safe_plan_action(action),
            "patch": _REDACTION_MARKER,
        }
    else:
        payload["action"] = _safe_plan_action(action)
    for key in ("stdout", "stderr", "model_facing_output"):
        payload[key] = _bounded_text(payload[key], 32_768)
    return payload


class ReActController:
    def __init__(self, *, planner: Any, bridge: Any, artifacts: Any, budget: RunBudget) -> None:
        self._planner = planner
        self._bridge = bridge
        self._artifacts = artifacts
        self._budget = budget

    def run(self, instruction: str) -> ReActRunResult:
        history: list[ActionObservation] = []
        step_index = 0
        while True:
            step_index += 1
            try:
                step = self._planner.next_step(
                    instruction=instruction,
                    history=tuple(history[-12:]),
                )
            except BudgetExhausted as error:
                self._artifacts.append_error({"category": error.category})
                raise
            except ReActPlanError as error:
                self._artifacts.append_error(
                    {
                        "category": error.category,
                        "error_type": type(error).__name__,
                        "step": step_index,
                    }
                )
                raise

            self._artifacts.append_plan(
                {
                    "step": step_index,
                    "plan": {
                        "actions": [_safe_plan_action(action) for action in step.actions],
                        "done": step.done,
                        "completion_summary": (
                            _bounded_text(
                                _redact_sensitive_text(step.completion_summary),
                                4_096,
                            )
                            if step.completion_summary is not None
                            else None
                        ),
                    },
                }
            )
            if step.done:
                return ReActRunResult(
                    stop_reason="completed",
                    completion_summary=step.completion_summary,
                    steps=step_index,
                    observations=len(history),
                )

            for action in step.actions:
                try:
                    action_index = self._budget.reserve_action()
                except DeadlineExhausted as error:
                    self._artifacts.append_error(
                        {"category": error.category, "step": step_index}
                    )
                    raise
                except BudgetExhausted:
                    self._artifacts.append_error(
                        {"category": "budget_error", "step": step_index}
                    )
                    return ReActRunResult(
                        stop_reason="action_budget_exhausted",
                        completion_summary=None,
                        steps=step_index,
                        observations=len(history),
                    )
                try:
                    observation = self._bridge.execute(action, action_index)
                except PolicyViolation as error:
                    self._artifacts.append_error(
                        {
                            "action_index": action_index,
                            "category": "policy_error",
                            "error_type": type(error).__name__,
                            "step": step_index,
                        }
                    )
                    continue
                history.append(observation)
                self._artifacts.append_observation(
                    _safe_observation_artifact(observation)
                )
