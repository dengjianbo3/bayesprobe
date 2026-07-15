from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, model_validator

from bayesprobe_terminal_bench.actions import ActionObservation, TerminalAction
from bayesprobe_terminal_bench.config import (
    BudgetExhausted,
    RunBudget,
    TerminalBenchConfig,
)
from bayesprobe_terminal_bench.environment import PolicyViolation
from bayesprobe_terminal_bench.planning import (
    _bounded_text,
    _chat_completion_content,
    _has_text_content,
    _history_action_input,
    _redact_sensitive_text,
    _response_telemetry,
    _sanitize_outbound,
    _single_attempt_client,
)


class ReActPlanError(ValueError):
    pass


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
    ) -> None:
        self._config = config
        self._budget = budget
        self._invocation_observer = invocation_observer
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
        payload = react_step_input(instruction=instruction, history=history)
        initial = self._complete(payload=payload, repair=False)
        if initial is not None:
            return initial
        repaired = self._complete(
            payload={
                "original_input": payload,
                "validation_error": "invalid ReAct step",
            },
            repair=True,
        )
        if repaired is not None:
            return repaired
        raise ReActPlanError("ReAct step validation failed") from None

    def _complete(
        self,
        *,
        payload: dict[str, Any],
        repair: bool,
    ) -> ReActStep | None:
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
        except Exception as error:
            self._observe_attempt(
                repair=repair,
                logical_call_index=logical_call_index,
                started=started,
                outcome="error",
                validation="not_attempted",
                error_type=type(error).__name__,
            )
            raise ReActPlanError("ReAct planner provider request failed") from None

        content = _chat_completion_content(response)
        if not _has_text_content(content):
            self._observe_attempt(
                repair=repair,
                logical_call_index=logical_call_index,
                started=started,
                outcome="empty_content",
                validation="invalid",
                response=response,
            )
            return None
        try:
            step = ReActStep.model_validate_json(content)
        except Exception:
            self._observe_attempt(
                repair=repair,
                logical_call_index=logical_call_index,
                started=started,
                outcome="success",
                validation="invalid",
                response=response,
            )
            return None
        self._observe_attempt(
            repair=repair,
            logical_call_index=logical_call_index,
            started=started,
            outcome="success",
            validation="valid",
            response=response,
        )
        return step

    def _observe_attempt(
        self,
        *,
        repair: bool,
        logical_call_index: int,
        started: float,
        outcome: str,
        validation: str,
        response: Any = None,
        error_type: str | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "task": "react_step",
            "model": _redact_sensitive_text(self._config.model),
            "repair": repair,
            "logical_call_index": logical_call_index,
            "outcome": outcome,
            "step_validation": validation,
            "latency_seconds": max(0.0, time.monotonic() - started),
        }
        if response is not None:
            record.update(_response_telemetry(response))
        if error_type is not None:
            record["error_type"] = error_type
        if self._invocation_observer is not None:
            try:
                self._invocation_observer(record)
            except Exception:
                return


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
            except BudgetExhausted:
                self._artifacts.append_error({"category": "model_budget_exhausted"})
                return ReActRunResult(
                    stop_reason="model_budget_exhausted",
                    completion_summary=None,
                    steps=step_index - 1,
                    observations=len(history),
                )
            except ReActPlanError as error:
                self._artifacts.append_error(
                    {
                        "category": "plan_error",
                        "error_type": type(error).__name__,
                        "step": step_index,
                    }
                )
                return ReActRunResult(
                    stop_reason="plan_error",
                    completion_summary=None,
                    steps=step_index,
                    observations=len(history),
                )

            self._artifacts.append_plan(
                {"step": step_index, "plan": step.model_dump(mode="json")}
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
                except BudgetExhausted:
                    self._artifacts.append_error(
                        {"category": "action_budget_exhausted", "step": step_index}
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
                self._artifacts.append_observation(observation)
