from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
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


def build_openai_request_payload(
    request: StructuredModelRequest,
    *,
    model: str,
    max_output_tokens: int | None = None,
) -> dict[str, Any]:
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
                "name": "EvidenceJudgment",
                "strict": True,
                "schema": EVIDENCE_JUDGMENT_JSON_SCHEMA,
            }
        },
    }
    if max_output_tokens is not None:
        payload["max_output_tokens"] = max_output_tokens
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
        if _looks_like_evidence_judgment_payload(response):
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


def _instruction_for_task(task: str) -> str:
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


def _looks_like_evidence_judgment_payload(response: Mapping[str, Any]) -> bool:
    return any(key in response for key in EVIDENCE_JUDGMENT_SCHEMA_KEYS)


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


__all__ = [
    "EVIDENCE_JUDGMENT_JSON_SCHEMA",
    "OpenAIModelGatewayConfig",
    "OpenAIResponsesModelGateway",
    "build_openai_request_payload",
    "parse_openai_structured_response",
]
