from __future__ import annotations

import json
import math
import os
import random as random_module
import re
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from types import SimpleNamespace
from typing import Any, Callable
from urllib.parse import urlparse

from bayesprobe.model_gateway import (
    ModelGatewayValidationError,
    ProviderRequestControls,
    StructuredModelRequest,
)
from bayesprobe.schemas import EvidenceType, LikelihoodBand
from bayesprobe.provider_telemetry import (
    ProviderInvocationContext,
    ProviderInvocationObserver,
    ProviderInvocationRecord,
    ProviderUsage,
    extract_provider_response_metadata,
    provider_error_category,
    provider_usage_from_response,
    sanitized_request_sha256,
)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


EVIDENCE_JUDGMENT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "evidence_type",
        "likelihoods",
        "interpretation",
        "quality_overrides",
    ],
    "properties": {
        "evidence_type": {
            "type": "string",
            "enum": [evidence_type.value for evidence_type in EvidenceType],
        },
        "likelihoods": {
            "type": "object",
            "additionalProperties": {
                "type": "string",
                "enum": [band.value for band in LikelihoodBand],
            },
        },
        "interpretation": {"type": "string"},
        "quality_overrides": {
            "type": "object",
            "additionalProperties": {"type": "number"},
        },
    },
}

PROBE_SIGNAL_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["raw_content"],
    "properties": {
        "raw_content": {"type": "string"},
    },
}

MULTIPLE_CHOICE_ANSWER_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer_label", "choice_probabilities", "answer_summary"],
    "properties": {
        "answer_label": {"type": "string"},
        "choice_probabilities": {
            "type": "object",
            "additionalProperties": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "answer_summary": {"type": "string", "minLength": 1},
    },
}

ENVIRONMENT_VARIABLE_NAME_PATTERN = re.compile(r"^[A-Z_][A-Z0-9_]*$")
EVIDENCE_JUDGMENT_SCHEMA_KEYS = frozenset(
    {"evidence_type", "likelihoods", "interpretation", "quality_overrides"}
)
MAX_PROVIDER_ATTEMPTS = 3


class ProviderHTTPError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        headers: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response = SimpleNamespace(headers=dict(headers or {}))


@dataclass(frozen=True)
class OpenAIModelGatewayConfig:
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    timeout_seconds: float = 30.0
    max_output_tokens: int | None = None
    base_url: str | None = None
    request_controls: ProviderRequestControls = ProviderRequestControls()

    def __post_init__(self) -> None:
        if not isinstance(self.model, str):
            raise ValueError("openai model gateway model must be a string")
        if not self.model.strip():
            raise ValueError("openai model gateway model must not be empty")
        if not isinstance(self.api_key_env, str):
            raise ValueError("openai model gateway api_key_env must be a string")
        if not self.api_key_env.strip():
            raise ValueError("openai model gateway api_key_env must not be empty")
        if not ENVIRONMENT_VARIABLE_NAME_PATTERN.fullmatch(self.api_key_env.strip()):
            raise ValueError(
                "openai model gateway api_key_env must be an environment variable name"
            )
        if type(self.timeout_seconds) not in (int, float):
            raise ValueError("openai model gateway timeout_seconds must be a number")
        if not math.isfinite(self.timeout_seconds):
            raise ValueError(
                "openai model gateway timeout_seconds must be finite and positive"
            )
        if self.timeout_seconds <= 0:
            raise ValueError("openai model gateway timeout_seconds must be positive")
        if self.max_output_tokens is not None:
            if type(self.max_output_tokens) is not int:
                raise ValueError(
                    "openai model gateway max_output_tokens must be an integer"
                )
            if self.max_output_tokens < 1:
                raise ValueError(
                    "openai model gateway max_output_tokens must be positive"
                )
        if self.base_url is not None:
            if not isinstance(self.base_url, str):
                raise ValueError("openai model gateway base_url must be a string")
            if not self.base_url.strip():
                raise ValueError("openai model gateway base_url must not be empty")
            object.__setattr__(self, "base_url", self.base_url.strip())
        if not isinstance(self.request_controls, ProviderRequestControls):
            raise ValueError(
                "openai model gateway request_controls must be ProviderRequestControls"
            )
        object.__setattr__(self, "model", self.model.strip())
        object.__setattr__(self, "api_key_env", self.api_key_env.strip())


class OpenAIResponsesModelGateway:
    adapter_kind = "openai"

    def __init__(
        self,
        *,
        config: OpenAIModelGatewayConfig,
        client: Any | None = None,
        api_key: str | None = None,
        invocation_observer: ProviderInvocationObserver | None = None,
        sleep: Callable[[float], None] = time.sleep,
        random_value: Callable[[], float] = random_module.random,
    ) -> None:
        self.config = config
        self._client = client
        self._api_key = _optional_request_api_key(api_key)
        self._invocation_observer = invocation_observer
        self._sleep = sleep
        self._random_value = random_value

    def complete_structured(self, request: StructuredModelRequest) -> dict[str, Any]:
        payload = build_openai_request_payload(
            request,
            model=self.config.model,
            max_output_tokens=self.config.max_output_tokens,
        )
        return _complete_with_observation(
            request=request,
            payload=payload,
            config=self.config,
            adapter_kind=self.adapter_kind,
            invoke=lambda: self._client_for_request().responses.create(**payload),
            parse=parse_openai_structured_response,
            observer=self._invocation_observer,
            sleep=self._sleep,
            random_value=self._random_value,
        )

    @property
    def invocation_observer(self) -> ProviderInvocationObserver | None:
        return self._invocation_observer

    def _client_for_request(self) -> Any:
        if self._client is None:
            self._client = _build_default_openai_client(
                self.config,
                api_key=self._api_key,
            )
        return self._client


class OpenAIChatCompletionsModelGateway:
    adapter_kind = "openai_chat_completions"

    def __init__(
        self,
        *,
        config: OpenAIModelGatewayConfig,
        client: Any | None = None,
        api_key: str | None = None,
        invocation_observer: ProviderInvocationObserver | None = None,
        sleep: Callable[[float], None] = time.sleep,
        random_value: Callable[[], float] = random_module.random,
    ) -> None:
        self.config = config
        self._client = client
        self._api_key = _optional_request_api_key(api_key)
        self._invocation_observer = invocation_observer
        self._sleep = sleep
        self._random_value = random_value

    def complete_structured(self, request: StructuredModelRequest) -> dict[str, Any]:
        payload = build_openai_chat_completions_payload(
            request,
            model=self.config.model,
            max_output_tokens=self.config.max_output_tokens,
            controls=self.config.request_controls,
        )
        return _complete_with_observation(
            request=request,
            payload=payload,
            config=self.config,
            adapter_kind=self.adapter_kind,
            invoke=lambda: self._client_for_request().chat.completions.create(
                **payload
            ),
            parse=parse_openai_chat_completions_response,
            observer=self._invocation_observer,
            sleep=self._sleep,
            random_value=self._random_value,
        )

    @property
    def invocation_observer(self) -> ProviderInvocationObserver | None:
        return self._invocation_observer

    def _client_for_request(self) -> Any:
        if self._client is None:
            self._client = _build_default_openai_chat_client(
                self.config,
                api_key=self._api_key,
            )
        return self._client


def build_openai_request_payload(
    request: StructuredModelRequest,
    *,
    model: str,
    max_output_tokens: int | None = None,
) -> dict[str, Any]:
    schema_name, schema = _structured_output_for_task(request.task)
    payload: dict[str, Any] = {
        "model": model,
        "input": [
            {
                "role": "developer",
                "content": _instruction_for_task(request.task),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"task": request.task, "input": request.input},
                    sort_keys=True,
                ),
            },
        ],
        "metadata": _metadata_for_request(request, model=model),
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": schema,
            }
        },
    }
    if max_output_tokens is not None:
        payload["max_output_tokens"] = max_output_tokens
    return payload


def build_openai_chat_completions_payload(
    request: StructuredModelRequest,
    *,
    model: str,
    max_output_tokens: int | None = None,
    controls: ProviderRequestControls | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": _chat_instruction_for_task(request.task),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": request.task,
                        "input": request.input,
                        "required_output": _required_output_for_task(request.task),
                    },
                    sort_keys=True,
                ),
            },
        ],
        "response_format": {"type": "json_object"},
        "stream": False,
    }
    if max_output_tokens is not None:
        payload["max_tokens"] = max_output_tokens
    request_controls = controls or ProviderRequestControls()
    if request_controls.temperature is not None:
        payload["temperature"] = request_controls.temperature
    if request_controls.top_p is not None:
        payload["top_p"] = request_controls.top_p
    if request_controls.thinking is not None:
        payload["thinking"] = {"type": request_controls.thinking}
    if request_controls.reasoning_effort is not None:
        payload["reasoning_effort"] = request_controls.reasoning_effort
    return payload


def _complete_with_observation(
    *,
    request: StructuredModelRequest,
    payload: Mapping[str, Any],
    config: OpenAIModelGatewayConfig,
    adapter_kind: str,
    invoke: Callable[[], Any],
    parse: Callable[[Any], dict[str, Any]],
    observer: ProviderInvocationObserver | None,
    sleep: Callable[[float], None],
    random_value: Callable[[], float],
) -> dict[str, Any]:
    for attempt_index in range(1, MAX_PROVIDER_ATTEMPTS + 1):
        started_at = datetime.now(UTC)
        started_monotonic = time.monotonic()
        response = None
        try:
            response = invoke()
            result = parse(response)
        except Exception as error:
            _observe_provider_attempt(
                observer=observer,
                request=request,
                payload=payload,
                config=config,
                adapter_kind=adapter_kind,
                attempt_index=attempt_index,
                started_at=started_at,
                started_monotonic=started_monotonic,
                response=response,
                error=error,
            )
            if attempt_index >= MAX_PROVIDER_ATTEMPTS or not _is_retryable_error(
                error
            ):
                raise
            sleep(_retry_delay_seconds(error, attempt_index, random_value))
            continue

        _observe_provider_attempt(
            observer=observer,
            request=request,
            payload=payload,
            config=config,
            adapter_kind=adapter_kind,
            attempt_index=attempt_index,
            started_at=started_at,
            started_monotonic=started_monotonic,
            response=response,
            error=None,
        )
        return result
    raise RuntimeError("provider attempt loop exited unexpectedly")


def _observe_provider_attempt(
    *,
    observer: ProviderInvocationObserver | None,
    request: StructuredModelRequest,
    payload: Mapping[str, Any],
    config: OpenAIModelGatewayConfig,
    adapter_kind: str,
    attempt_index: int,
    started_at: datetime,
    started_monotonic: float,
    response: Any,
    error: Exception | None,
) -> None:
    if observer is None:
        return
    completed_at = datetime.now(UTC)
    finish_reason, response_id, system_fingerprint = (
        extract_provider_response_metadata(response)
        if response is not None
        else (None, None, None)
    )
    record = ProviderInvocationRecord(
        task=request.task,
        adapter_kind=adapter_kind,
        model=config.model,
        base_host=_provider_base_host(config, adapter_kind=adapter_kind),
        prompt_id=request.prompt_id,
        prompt_version=request.prompt_version,
        schema_name=request.schema_name,
        schema_version=request.schema_version,
        request_sha256=sanitized_request_sha256(payload),
        started_at=_utc_text(started_at),
        completed_at=_utc_text(completed_at),
        latency_seconds=max(0.0, time.monotonic() - started_monotonic),
        usage=(
            provider_usage_from_response(response)
            if response is not None
            else ProviderUsage()
        ),
        finish_reason=finish_reason,
        response_id=response_id,
        system_fingerprint=system_fingerprint,
        outcome="success" if error is None else "error",
        error_category=None if error is None else provider_error_category(error),
        context=ProviderInvocationContext(
            experiment_id=_request_metadata_text(request, "experiment_id"),
            arm=_request_metadata_text(request, "arm"),
            sample_id=_request_metadata_text(request, "sample_id"),
            run_id=_request_metadata_text(request, "run_id"),
            cycle_id=_request_metadata_text(request, "cycle_id"),
            probe_id=_request_metadata_text(request, "probe_id"),
            attempt_index=attempt_index,
        ),
    )
    try:
        observer.observe(record)
    except Exception:
        return


def _provider_base_host(
    config: OpenAIModelGatewayConfig, *, adapter_kind: str
) -> str | None:
    if config.base_url is not None:
        base_url = config.base_url
    elif adapter_kind == "openai_chat_completions":
        base_url = "https://api.openai.com/v1"
    else:
        base_url = "https://api.openai.com/v1"
    return urlparse(base_url).hostname


def _request_metadata_text(
    request: StructuredModelRequest, key: str
) -> str | None:
    value = request.metadata.get(key)
    return value if isinstance(value, str) and value else None


def _utc_text(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _is_retryable_error(error: Exception) -> bool:
    return provider_error_category(error) in {
        "connection",
        "provider_server_error",
        "rate_limited",
        "timeout",
    }


def _retry_delay_seconds(
    error: Exception,
    attempt_index: int,
    random_value: Callable[[], float],
) -> float:
    retry_after = _retry_after_seconds(error)
    if retry_after is not None:
        return min(60.0, max(0.0, retry_after))
    exponential = min(8.0, 0.5 * (2 ** (attempt_index - 1)))
    jitter = min(1.0, max(0.0, float(random_value()))) * 0.25
    return exponential + jitter


def _retry_after_seconds(error: Exception) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None) or getattr(error, "headers", None)
    if not isinstance(headers, Mapping):
        return None
    value = headers.get("Retry-After") or headers.get("retry-after")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(str(value))
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        return max(0.0, (retry_at - datetime.now(UTC)).total_seconds())


def parse_openai_structured_response(response: Any) -> dict[str, Any]:
    if isinstance(response, Mapping):
        if "output_text" in response:
            return _parse_json_object(response["output_text"])
        if "text" in response:
            return _parse_json_object(response["text"])
        text = _extract_text_from_output(response)
        if text is not None:
            return _parse_json_object(text)
        if _looks_like_structured_task_payload(response):
            return dict(response)
        raise ModelGatewayValidationError("openai structured response text was missing")
    if isinstance(response, str):
        return _parse_json_object(response)
    if isinstance(response, list):
        raise ModelGatewayValidationError("openai structured response must be an object")

    output_text = getattr(response, "output_text", None)
    if output_text is not None:
        return _parse_json_object(output_text)

    text = _extract_text_from_output(response)
    if text is not None:
        return _parse_json_object(text)

    raise ModelGatewayValidationError("openai structured response text was missing")


def parse_openai_chat_completions_response(response: Any) -> dict[str, Any]:
    content = _chat_message_content(response)
    content_missing = content is None or not content.strip()
    if content_missing and _chat_finish_reason(response) == "length":
        raise ModelGatewayValidationError(
            "openai chat completion exhausted max_tokens before producing "
            "structured content"
        )
    if content is None:
        raise ModelGatewayValidationError("openai chat completion content was missing")
    return _parse_json_object(content)


def _instruction_for_task(task: str) -> str:
    if task == "execute_probe":
        return (
            "You are the active probe executor inside BayesProbe. Perform the supplied "
            "inquiry against the problem, current hypotheses, and initial context. "
            "Return a raw informational signal only; do not assign posterior probabilities "
            "or claim access to external sources that were not supplied."
        )
    if task == "judge_evidence":
        return (
            "You are the evidence judgment component inside BayesProbe. "
            "Convert the provided signal context into one EvidenceJudgment JSON object. "
            "Use only the supplied hypotheses and likelihood bands."
        )
    if task == "repair_evidence_judgment":
        return (
            "Repair the malformed BayesProbe evidence judgment. "
            "Return exactly one valid EvidenceJudgment JSON object. "
            "Preserve the intended evidence meaning when it can be inferred."
        )
    if task == "answer_multiple_choice":
        return (
            "Answer the supplied multiple-choice problem using only the question and "
            "choices. Return a complete probability distribution over every supplied "
            "choice and a concise final justification."
        )
    if task == "repair_multiple_choice_answer":
        return (
            "Repair the malformed multiple-choice answer using the supplied question, "
            "choices, invalid payload, and validation error. Return one valid answer "
            "without adding or removing choice labels."
        )
    raise ValueError(f"unsupported openai model task: {task}")


def _structured_output_for_task(task: str) -> tuple[str, dict[str, Any]]:
    if task == "execute_probe":
        return "ProbeSignal", PROBE_SIGNAL_JSON_SCHEMA
    if task in {"judge_evidence", "repair_evidence_judgment"}:
        return "EvidenceJudgment", EVIDENCE_JUDGMENT_JSON_SCHEMA
    if task in {"answer_multiple_choice", "repair_multiple_choice_answer"}:
        return "MultipleChoiceAnswer", MULTIPLE_CHOICE_ANSWER_JSON_SCHEMA
    raise ValueError(f"unsupported openai model task: {task}")


def _chat_instruction_for_task(task: str) -> str:
    base_instruction = _instruction_for_task(task)
    if task == "execute_probe":
        return (
            f"{base_instruction} Return only one JSON object with exactly one "
            "top-level key: raw_content. Do not include markdown."
        )
    if task == "judge_evidence":
        return (
            f"{base_instruction} Return only one JSON object with exactly these "
            "top-level keys: evidence_type, likelihoods, interpretation, "
            "quality_overrides. Do not copy input fields such as signal_id, "
            "source, source_type, target_hypotheses, likelihood_bands, or "
            "evidence into the output. Do not include markdown."
        )
    if task == "repair_evidence_judgment":
        return (
            f"{base_instruction} Return only one JSON object with exactly these "
            "top-level keys: evidence_type, likelihoods, interpretation, "
            "quality_overrides. Do not include markdown."
        )
    if task in {"answer_multiple_choice", "repair_multiple_choice_answer"}:
        return (
            f"{base_instruction} Return only one JSON object with exactly these "
            "top-level keys: answer_label, choice_probabilities, answer_summary. "
            "Do not include markdown."
        )
    raise ValueError(f"unsupported openai model task: {task}")


def _required_output_for_task(task: str) -> dict[str, Any]:
    if task == "execute_probe":
        return {
            "type": "ProbeSignal",
            "required_keys": ["raw_content"],
            "json_schema": PROBE_SIGNAL_JSON_SCHEMA,
            "notes": [
                "raw_content must report the inquiry result without posterior updates",
                "do not claim external retrieval or verification unless supplied in the input",
            ],
        }
    if task in {"judge_evidence", "repair_evidence_judgment"}:
        return {
            "type": "EvidenceJudgment",
            "required_keys": [
                "evidence_type",
                "likelihoods",
                "interpretation",
                "quality_overrides",
            ],
            "json_schema": EVIDENCE_JUDGMENT_JSON_SCHEMA,
            "notes": [
                "likelihoods must be an object keyed only by supplied hypothesis ids",
                "quality_overrides may be an empty object",
            ],
        }
    if task in {"answer_multiple_choice", "repair_multiple_choice_answer"}:
        return {
            "type": "MultipleChoiceAnswer",
            "required_keys": [
                "answer_label",
                "choice_probabilities",
                "answer_summary",
            ],
            "json_schema": MULTIPLE_CHOICE_ANSWER_JSON_SCHEMA,
            "notes": [
                "choice_probabilities keys must exactly match the supplied labels",
                "choice_probabilities must sum to one within 1e-3",
            ],
        }
    raise ValueError(f"unsupported openai model task: {task}")


def _metadata_for_request(
    request: StructuredModelRequest, *, model: str
) -> dict[str, str]:
    metadata: dict[str, str] = {
        "provider": "openai",
        "model": model,
        "task": request.task,
    }
    optional_fields = {
        "prompt_id": request.prompt_id,
        "prompt_version": request.prompt_version,
        "schema_name": request.schema_name,
        "schema_version": request.schema_version,
    }
    for key, value in optional_fields.items():
        if value is not None:
            metadata[key] = value
    for key, value in request.metadata.items():
        if isinstance(value, str):
            metadata[str(key)] = value
        elif isinstance(value, bool | int | float):
            metadata[str(key)] = str(value)
    return metadata


def _parse_json_object(text: Any) -> dict[str, Any]:
    if not isinstance(text, str):
        raise ModelGatewayValidationError("openai structured response text was missing")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        raise ModelGatewayValidationError(
            "openai structured response was not valid JSON"
        ) from error
    if not isinstance(parsed, Mapping):
        raise ModelGatewayValidationError("openai structured response must be an object")
    return dict(parsed)


def _extract_text_from_output(response: Any) -> str | None:
    output = getattr(response, "output", None)
    if output is None and isinstance(response, Mapping):
        output = response.get("output")
    if output is None:
        return None
    for item in output:
        content = getattr(item, "content", None)
        if content is None and isinstance(item, Mapping):
            content = item.get("content")
        if content is None:
            continue
        for part in content:
            text = getattr(part, "text", None)
            if text is None and isinstance(part, Mapping):
                text = part.get("text")
            if isinstance(text, str):
                return text
    return None


def _chat_message_content(response: Any) -> str | None:
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, Mapping):
        choices = response.get("choices")
    if not isinstance(choices, list | tuple) or not choices:
        return None
    choice = choices[0]
    message = getattr(choice, "message", None)
    if message is None and isinstance(choice, Mapping):
        message = choice.get("message")
    if message is None:
        return None
    content = getattr(message, "content", None)
    if content is None and isinstance(message, Mapping):
        content = message.get("content")
    if isinstance(content, str):
        return content
    return None


def _chat_finish_reason(response: Any) -> str | None:
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, Mapping):
        choices = response.get("choices")
    if not isinstance(choices, list | tuple) or not choices:
        return None
    choice = choices[0]
    finish_reason = getattr(choice, "finish_reason", None)
    if finish_reason is None and isinstance(choice, Mapping):
        finish_reason = choice.get("finish_reason")
    if isinstance(finish_reason, str):
        return finish_reason
    return None


def _looks_like_structured_task_payload(response: Mapping[str, Any]) -> bool:
    return "raw_content" in response or any(
        key in response for key in EVIDENCE_JUDGMENT_SCHEMA_KEYS
    )


def _optional_request_api_key(api_key: str | None) -> str | None:
    if api_key is None:
        return None
    if not isinstance(api_key, str):
        raise ValueError("openai request api_key must be a string")
    if not api_key.strip():
        raise ValueError("openai request api_key must not be empty")
    return api_key.strip()


def _build_default_openai_client(
    config: OpenAIModelGatewayConfig, *, api_key: str | None = None
) -> Any:
    resolved_api_key = api_key or os.environ.get(config.api_key_env)
    if not resolved_api_key:
        raise RuntimeError(
            f"OpenAI API key environment variable {config.api_key_env} is not set"
        )
    if OpenAI is None:
        raise RuntimeError(
            "OpenAI Python package is required for OpenAIResponsesModelGateway. "
            "Install bayesprobe[openai] or install openai."
        )
    kwargs: dict[str, Any] = {
        "api_key": resolved_api_key,
        "timeout": config.timeout_seconds,
    }
    if config.base_url is not None:
        kwargs["base_url"] = config.base_url
    return OpenAI(**kwargs)


def _build_default_openai_chat_client(
    config: OpenAIModelGatewayConfig, *, api_key: str | None = None
) -> Any:
    if OpenAI is not None:
        return _build_default_openai_client(config, api_key=api_key)
    resolved_api_key = api_key or os.environ.get(config.api_key_env)
    if not resolved_api_key:
        raise RuntimeError(
            f"OpenAI API key environment variable {config.api_key_env} is not set"
        )
    return _StdlibOpenAIChatClient(
        api_key=resolved_api_key,
        base_url=config.base_url or "https://api.openai.com/v1",
        timeout_seconds=config.timeout_seconds,
    )


class _StdlibOpenAIChatClient:
    def __init__(self, *, api_key: str, base_url: str, timeout_seconds: float) -> None:
        self.chat = SimpleNamespace(
            completions=_StdlibOpenAIChatCompletions(
                api_key=api_key,
                base_url=base_url,
                timeout_seconds=timeout_seconds,
            )
        )


class _StdlibOpenAIChatCompletions:
    def __init__(self, *, api_key: str, base_url: str, timeout_seconds: float) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._timeout_seconds = timeout_seconds

    def create(self, **payload: Any) -> dict[str, Any]:
        return _post_json(
            _chat_completions_url(self._base_url),
            payload,
            api_key=self._api_key,
            timeout_seconds=self._timeout_seconds,
        )


def _chat_completions_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/chat/completions"


def _post_json(
    url: str,
    payload: Mapping[str, Any],
    *,
    api_key: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise ProviderHTTPError(
            (
                "OpenAI-compatible chat completion request failed with HTTP "
                f"{error.code}: {_sanitize_provider_error(body, api_key)}"
            ),
            status_code=error.code,
            headers=error.headers,
        ) from error
    except urllib.error.URLError as error:
        message = _sanitize_provider_error(str(error.reason), api_key)
        if isinstance(error.reason, TimeoutError):
            raise TimeoutError(message) from error
        raise ConnectionError(message) from error
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as error:
        raise ModelGatewayValidationError(
            "openai chat completion response was not valid JSON"
        ) from error
    if not isinstance(parsed, Mapping):
        raise ModelGatewayValidationError(
            "openai chat completion response must be an object"
        )
    return dict(parsed)


def _sanitize_provider_error(message: str, api_key: str) -> str:
    return message.replace(api_key, "sk-redacted")


__all__ = [
    "EVIDENCE_JUDGMENT_JSON_SCHEMA",
    "MULTIPLE_CHOICE_ANSWER_JSON_SCHEMA",
    "PROBE_SIGNAL_JSON_SCHEMA",
    "OpenAIChatCompletionsModelGateway",
    "OpenAIModelGatewayConfig",
    "OpenAIResponsesModelGateway",
    "build_openai_chat_completions_payload",
    "build_openai_request_payload",
    "parse_openai_chat_completions_response",
    "parse_openai_structured_response",
]
