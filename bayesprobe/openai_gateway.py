from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from bayesprobe.model_gateway import ModelGatewayValidationError, StructuredModelRequest
from bayesprobe.schemas import EvidenceType, LikelihoodBand


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


@dataclass(frozen=True)
class OpenAIModelGatewayConfig:
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    timeout_seconds: float = 30.0
    max_output_tokens: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model, str):
            raise ValueError("openai model gateway model must be a string")
        if not self.model.strip():
            raise ValueError("openai model gateway model must not be empty")
        if not isinstance(self.api_key_env, str):
            raise ValueError("openai model gateway api_key_env must be a string")
        if not self.api_key_env.strip():
            raise ValueError("openai model gateway api_key_env must not be empty")
        if type(self.timeout_seconds) not in (int, float):
            raise ValueError("openai model gateway timeout_seconds must be a number")
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
        object.__setattr__(self, "model", self.model.strip())
        object.__setattr__(self, "api_key_env", self.api_key_env.strip())


class OpenAIResponsesModelGateway:
    adapter_kind = "openai"

    def __init__(
        self,
        *,
        config: OpenAIModelGatewayConfig,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self._client = client

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
            self._client = _build_default_openai_client(self.config)
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
        return dict(response)
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


def _build_default_openai_client(config: OpenAIModelGatewayConfig) -> Any:
    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"OpenAI API key environment variable {config.api_key_env} is not set"
        )
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError(
            "OpenAI Python package is required for OpenAIResponsesModelGateway. "
            "Install bayesprobe[openai] or install openai."
        ) from error
    return OpenAI(api_key=api_key, timeout=config.timeout_seconds)


__all__ = [
    "EVIDENCE_JUDGMENT_JSON_SCHEMA",
    "OpenAIModelGatewayConfig",
    "OpenAIResponsesModelGateway",
    "build_openai_request_payload",
    "parse_openai_structured_response",
]
