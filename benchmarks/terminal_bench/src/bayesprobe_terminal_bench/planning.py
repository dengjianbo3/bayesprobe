from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from openai import OpenAI

from bayesprobe import ProbeDesign, ProbeExecutionBrief
from bayesprobe_terminal_bench.actions import ActionObservation, TerminalProbePlan
from bayesprobe_terminal_bench.config import RunBudget, TerminalBenchConfig


class TerminalPlanError(ValueError):
    pass


class TerminalProbePlanner(Protocol):
    def plan(
        self,
        *,
        probe: ProbeDesign,
        context: ProbeExecutionBrief,
        history: tuple[ActionObservation, ...],
    ) -> TerminalProbePlan: ...


_EXCLUDED_INPUT_FIELDS = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "analysis",
        "authorization",
        "chain_of_thought",
        "chainofthought",
        "credential",
        "credentials",
        "password",
        "posterior",
        "posteriors",
        "prior",
        "priors",
        "rationale",
        "reasoning",
        "reasoning_trace",
        "score",
        "scores",
        "secret",
        "thought_process",
        "token",
        "verifier_path",
        "verifier_paths",
    }
)
_EXCLUDED_INPUT_FIELD_PREFIXES = (
    "posterior_",
    "prior_",
    "score_",
    "verifier_path_",
)
_COMPACT_EXCLUDED_INPUT_FIELDS = frozenset(
    field.replace("_", "") for field in _EXCLUDED_INPUT_FIELDS
)


def terminal_plan_input(
    *,
    probe: ProbeDesign,
    context: ProbeExecutionBrief,
    history: tuple[ActionObservation, ...],
) -> dict[str, Any]:
    """Build the blind, bounded context the terminal planner may receive."""
    return {
        "task": {
            "problem": context.problem,
            "task_context": context.task_context,
            "task_frame": _without_private_fields(context.task_frame),
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
            "inquiry_goal": probe.inquiry_goal,
            "method": probe.method,
            "expected_observation": probe.expected_observation,
            "target_hypotheses": list(probe.target_hypotheses),
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
                "action": item.action.model_dump(mode="json"),
                "observation": item.model_facing_output,
                "return_code": item.return_code,
                "timed_out": item.timed_out,
                "environment_state_id": item.post_environment_state_id,
            }
            for item in history[-12:]
        ],
    }


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
        self._client = client

    def plan(
        self,
        *,
        probe: ProbeDesign,
        context: ProbeExecutionBrief,
        history: tuple[ActionObservation, ...],
    ) -> TerminalProbePlan:
        payload = terminal_plan_input(probe=probe, context=context, history=history)
        initial = self._complete(payload=payload, repair=False)
        if initial is not None:
            return initial

        repaired = self._complete(
            payload={
                "original_input": payload,
                "validation_error": "invalid terminal plan",
            },
            repair=True,
        )
        if repaired is not None:
            return repaired
        raise TerminalPlanError("terminal plan validation failed") from None

    def _complete(
        self,
        *,
        payload: dict[str, Any],
        repair: bool,
    ) -> TerminalProbePlan | None:
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
                error_type=type(error).__name__,
            )
            raise TerminalPlanError("terminal planner provider request failed") from None

        content = _chat_completion_content(response)
        if not isinstance(content, str) or not content.strip():
            self._observe_attempt(
                repair=repair,
                logical_call_index=logical_call_index,
                started=started,
                outcome="empty_content",
                plan_validation="invalid",
                response=response,
            )
            return None

        try:
            plan = TerminalProbePlan.model_validate_json(content)
        except Exception:
            self._observe_attempt(
                repair=repair,
                logical_call_index=logical_call_index,
                started=started,
                outcome="success",
                plan_validation="invalid",
                response=response,
            )
            return None

        self._observe_attempt(
            repair=repair,
            logical_call_index=logical_call_index,
            started=started,
            outcome="success",
            plan_validation="valid",
            response=response,
        )
        return plan

    def _observe_attempt(
        self,
        *,
        repair: bool,
        logical_call_index: int,
        started: float,
        outcome: str,
        plan_validation: str,
        response: Any = None,
        error_type: str | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "task": "terminal_probe_plan",
            "model": self._config.model,
            "repair": repair,
            "logical_call_index": logical_call_index,
            "outcome": outcome,
            "plan_validation": plan_validation,
            "latency_seconds": max(0.0, time.monotonic() - started),
        }
        if response is not None:
            record.update(_response_telemetry(response))
        if error_type is not None:
            record["error_type"] = error_type
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
    return request + " Schema: " + json.dumps(
        TerminalProbePlan.model_json_schema(),
        sort_keys=True,
    )


def _without_private_fields(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _without_private_fields(item)
            for key, item in value.items()
            if isinstance(key, str) and not _is_private_field_name(key)
        }
    if isinstance(value, list | tuple):
        return [_without_private_fields(item) for item in value]
    return value


def _is_private_field_name(value: str) -> bool:
    normalized = "".join(
        character if character.isalnum() else "_"
        for character in value.casefold()
    ).strip("_")
    compact = normalized.replace("_", "")
    return (
        normalized in _EXCLUDED_INPUT_FIELDS
        or compact in _COMPACT_EXCLUDED_INPUT_FIELDS
        or normalized.startswith(_EXCLUDED_INPUT_FIELD_PREFIXES)
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


def _response_telemetry(response: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "usage": _usage_telemetry(_response_value(response, "usage")),
    }
    response_id = _response_value(response, "id")
    if isinstance(response_id, str):
        record["response_id"] = response_id
    choice = _first_choice(response)
    finish_reason = None if choice is None else _response_value(choice, "finish_reason")
    if isinstance(finish_reason, str):
        record["finish_reason"] = finish_reason
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
    if not isinstance(choices, list | tuple) or not choices:
        return None
    return choices[0]


def _response_value(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)
