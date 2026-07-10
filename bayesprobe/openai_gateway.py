from __future__ import annotations

import json
import math
import os
import re
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from bayesprobe.model_gateway import ModelGatewayValidationError, StructuredModelRequest
from bayesprobe.schemas import EvidenceType, LikelihoodBand

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

ENVIRONMENT_VARIABLE_NAME_PATTERN = re.compile(r"^[A-Z_][A-Z0-9_]*$")
EVIDENCE_JUDGMENT_SCHEMA_KEYS = frozenset(
    {"evidence_type", "likelihoods", "interpretation", "quality_overrides"}
)


@dataclass(frozen=True)
class OpenAIModelGatewayConfig:
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    timeout_seconds: float = 30.0
    max_output_tokens: int | None = None
    base_url: str | None = None

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
    ) -> None:
        self.config = config
        self._client = client
        self._api_key = _optional_request_api_key(api_key)

    def complete_structured(self, request: StructuredModelRequest) -> dict[str, Any]:
        payload = build_openai_request_payload(
            request,
            model=self.config.model,
            max_output_tokens=self.config.max_output_tokens,
        )
        response = self._client_for_request().responses.create(**payload)
        return parse_openai_structured_response(response)

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
    ) -> None:
        self.config = config
        self._client = client
        self._api_key = _optional_request_api_key(api_key)

    def complete_structured(self, request: StructuredModelRequest) -> dict[str, Any]:
        payload = build_openai_chat_completions_payload(
            request,
            model=self.config.model,
            max_output_tokens=self.config.max_output_tokens,
        )
        response = self._client_for_request().chat.completions.create(**payload)
        return parse_openai_chat_completions_response(response)

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
    return payload


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
    raise ValueError(f"unsupported openai model task: {task}")


def _structured_output_for_task(task: str) -> tuple[str, dict[str, Any]]:
    if task == "execute_probe":
        return "ProbeSignal", PROBE_SIGNAL_JSON_SCHEMA
    if task in {"judge_evidence", "repair_evidence_judgment"}:
        return "EvidenceJudgment", EVIDENCE_JUDGMENT_JSON_SCHEMA
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
        raise RuntimeError(
            f"OpenAI-compatible chat completion request failed with HTTP {error.code}: "
            f"{_sanitize_provider_error(body, api_key)}"
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"OpenAI-compatible chat completion request failed: {error.reason}"
        ) from error
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
    "PROBE_SIGNAL_JSON_SCHEMA",
    "OpenAIChatCompletionsModelGateway",
    "OpenAIModelGatewayConfig",
    "OpenAIResponsesModelGateway",
    "build_openai_chat_completions_payload",
    "build_openai_request_payload",
    "parse_openai_chat_completions_response",
    "parse_openai_structured_response",
]
