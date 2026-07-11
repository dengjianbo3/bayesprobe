from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from bayesprobe.schemas import EvidenceType, LikelihoodBand


class ModelGatewayValidationError(ValueError):
    pass


def _validate_optional_string(
    value: str | None,
    *,
    owner: str,
    field_name: str,
) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise ValueError(f"{owner} {field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{owner} {field_name} must not be empty")


@dataclass(frozen=True)
class StructuredModelRequest:
    task: str
    input: dict[str, Any]
    prompt_id: str | None = None
    prompt_version: str | None = None
    schema_name: str | None = None
    schema_version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.task, str):
            raise ValueError("structured model request task must be a string")
        if not self.task.strip():
            raise ValueError("structured model request task must not be empty")
        if not isinstance(self.input, Mapping):
            raise ValueError("structured model request input must be an object")
        _validate_optional_string(
            self.prompt_id,
            owner="structured model request",
            field_name="prompt_id",
        )
        _validate_optional_string(
            self.prompt_version,
            owner="structured model request",
            field_name="prompt_version",
        )
        _validate_optional_string(
            self.schema_name,
            owner="structured model request",
            field_name="schema_name",
        )
        _validate_optional_string(
            self.schema_version,
            owner="structured model request",
            field_name="schema_version",
        )
        if not isinstance(self.metadata, Mapping):
            raise ValueError("structured model request metadata must be an object")
        object.__setattr__(self, "input", dict(self.input))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class ProviderRequestControls:
    temperature: float | None = None
    top_p: float | None = None
    thinking: str | None = None
    reasoning_effort: str | None = None

    def __post_init__(self) -> None:
        if self.temperature is not None:
            if (
                type(self.temperature) not in (int, float)
                or not math.isfinite(self.temperature)
                or self.temperature < 0
            ):
                raise ValueError(
                    "provider request temperature must be finite and non-negative"
                )
        if self.top_p is not None:
            if (
                type(self.top_p) not in (int, float)
                or not math.isfinite(self.top_p)
                or not 0 < self.top_p <= 1
            ):
                raise ValueError(
                    "provider request top_p must be finite and in the interval (0, 1]"
                )
        for field_name in ("thinking", "reasoning_effort"):
            value = getattr(self, field_name)
            if value is None:
                continue
            if not isinstance(value, str):
                raise ValueError(f"provider request {field_name} must be a string")
            if not value.strip():
                raise ValueError(f"provider request {field_name} must not be empty")
            object.__setattr__(self, field_name, value.strip())


@dataclass(frozen=True)
class ModelGatewayConfig:
    kind: str = "deterministic"
    responses: dict[str, dict[str, Any]] | None = None
    model: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    timeout_seconds: float = 30.0
    max_output_tokens: int | None = None
    base_url: str | None = None
    fixture_path: str | Path | None = None
    request_controls: ProviderRequestControls = field(
        default_factory=ProviderRequestControls
    )


@dataclass(frozen=True)
class ModelInvocationTrace:
    task: str
    adapter_kind: str
    prompt_id: str | None = None
    prompt_version: str | None = None
    schema_name: str | None = None
    schema_version: str | None = None
    repair_attempt_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.task, str):
            raise ValueError("model invocation task must be a string")
        if not self.task.strip():
            raise ValueError("model invocation task must not be empty")
        if not isinstance(self.adapter_kind, str):
            raise ValueError("model invocation adapter_kind must be a string")
        if not self.adapter_kind.strip():
            raise ValueError("model invocation adapter_kind must not be empty")
        _validate_optional_string(
            self.prompt_id,
            owner="model invocation",
            field_name="prompt_id",
        )
        _validate_optional_string(
            self.prompt_version,
            owner="model invocation",
            field_name="prompt_version",
        )
        _validate_optional_string(
            self.schema_name,
            owner="model invocation",
            field_name="schema_name",
        )
        _validate_optional_string(
            self.schema_version,
            owner="model invocation",
            field_name="schema_version",
        )
        if self.repair_attempt_index is not None and (
            type(self.repair_attempt_index) is not int or self.repair_attempt_index < 1
        ):
            raise ValueError(
                "model invocation repair_attempt_index must be a positive integer"
            )
        if not isinstance(self.metadata, Mapping):
            raise ValueError("model invocation metadata must be an object")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_request(
        cls, request: StructuredModelRequest, *, adapter_kind: str
    ) -> "ModelInvocationTrace":
        metadata = dict(request.metadata)
        repair_attempt_index = metadata.pop("repair_attempt_index", None)
        return cls(
            task=request.task,
            adapter_kind=adapter_kind,
            prompt_id=request.prompt_id,
            prompt_version=request.prompt_version,
            schema_name=request.schema_name,
            schema_version=request.schema_version,
            repair_attempt_index=repair_attempt_index,
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "task": self.task,
            "adapter_kind": self.adapter_kind,
            "metadata": dict(self.metadata),
        }
        if self.prompt_id is not None:
            payload["prompt_id"] = self.prompt_id
        if self.prompt_version is not None:
            payload["prompt_version"] = self.prompt_version
        if self.schema_name is not None:
            payload["schema_name"] = self.schema_name
        if self.schema_version is not None:
            payload["schema_version"] = self.schema_version
        if self.repair_attempt_index is not None:
            payload["repair_attempt_index"] = self.repair_attempt_index
        return payload


@dataclass(frozen=True)
class EvidenceJudgmentRepairPolicy:
    max_attempts: int = 0
    repair_task: str = "repair_evidence_judgment"

    def __post_init__(self) -> None:
        if type(self.max_attempts) is not int:
            raise ValueError("judgment repair max_attempts must be an integer")
        if self.max_attempts < 0:
            raise ValueError("judgment repair max_attempts must be non-negative")
        if not isinstance(self.repair_task, str):
            raise ValueError("judgment repair task must be a string")
        if not self.repair_task.strip():
            raise ValueError("judgment repair task must not be empty")

    @classmethod
    def from_config(
        cls,
        config: "EvidenceJudgmentRepairPolicy | Mapping[str, Any] | None" = None,
    ) -> "EvidenceJudgmentRepairPolicy":
        if config is None:
            return cls()
        if isinstance(config, cls):
            return config
        if not isinstance(config, Mapping):
            raise ValueError("judgment repair policy config must be an object")
        max_attempts = config.get("max_attempts", 0)
        repair_task = config.get("repair_task", "repair_evidence_judgment")
        return cls(max_attempts=max_attempts, repair_task=repair_task)


class ModelGateway(Protocol):
    def complete_structured(self, request: StructuredModelRequest) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class EvidenceJudgment:
    evidence_type: EvidenceType
    likelihoods: dict[str, LikelihoodBand]
    interpretation: str
    quality_overrides: dict[str, float] = field(default_factory=dict)


_QUALITY_OVERRIDE_METRICS = {
    "reliability",
    "independence",
    "relevance",
    "novelty",
    "specificity",
    "verifiability",
}


def evidence_judgment_from_mapping(payload: dict[str, Any]) -> EvidenceJudgment:
    if not isinstance(payload, Mapping):
        raise ModelGatewayValidationError("evidence judgment payload must be an object")
    if "evidence_type" not in payload:
        raise ModelGatewayValidationError("evidence judgment missing field: evidence_type")

    raw_evidence_type = payload["evidence_type"]
    try:
        evidence_type = EvidenceType(raw_evidence_type)
    except (TypeError, ValueError) as error:
        raise ModelGatewayValidationError(f"invalid evidence_type: {raw_evidence_type}") from error

    likelihoods_payload = payload.get("likelihoods", {})
    if not isinstance(likelihoods_payload, Mapping):
        raise ModelGatewayValidationError("evidence judgment likelihoods must be an object")

    likelihoods: dict[str, LikelihoodBand] = {}
    for hypothesis_id, likelihood in likelihoods_payload.items():
        try:
            likelihoods[str(hypothesis_id)] = LikelihoodBand(likelihood)
        except (TypeError, ValueError) as error:
            raise ModelGatewayValidationError(
                f"invalid likelihood band for {hypothesis_id}: {likelihood}"
            ) from error

    quality_overrides_payload = payload.get("quality_overrides", {})
    if quality_overrides_payload is None:
        quality_overrides_payload = {}
    if not isinstance(quality_overrides_payload, Mapping):
        raise ModelGatewayValidationError("evidence judgment quality_overrides must be an object")

    quality_overrides: dict[str, float] = {}
    for metric, value in quality_overrides_payload.items():
        metric_name = str(metric)
        if metric_name not in _QUALITY_OVERRIDE_METRICS:
            raise ModelGatewayValidationError(
                f"unsupported quality override metric: {metric_name}"
            )
        try:
            parsed_value = float(value)
        except (TypeError, ValueError) as error:
            raise ModelGatewayValidationError(
                f"invalid quality override for {metric}: {value}"
            ) from error
        if not math.isfinite(parsed_value) or not 0 <= parsed_value <= 1:
            raise ModelGatewayValidationError(
                f"quality override {metric_name} must be finite and between 0 and 1"
            )
        quality_overrides[metric_name] = parsed_value

    return EvidenceJudgment(
        evidence_type=evidence_type,
        likelihoods=likelihoods,
        interpretation=str(payload.get("interpretation", "")),
        quality_overrides=quality_overrides,
    )


class DeterministicModelGateway:
    adapter_kind = "deterministic"

    def complete_structured(self, request: StructuredModelRequest) -> dict[str, Any]:
        if request.task != "judge_evidence":
            raise ValueError(f"unsupported deterministic model task: {request.task}")

        content_upper = str(request.input.get("raw_content", "")).upper()
        source_type = str(request.input.get("source_type", "unknown"))
        hypothesis_ids = [
            str(hypothesis_id)
            for hypothesis_id in request.input.get("target_hypotheses", [])
        ]
        likelihoods = {
            hypothesis_id: LikelihoodBand.NEUTRAL.value
            for hypothesis_id in hypothesis_ids
        }
        evidence_type = EvidenceType.NEUTRAL
        explicit_target = _deterministic_target_hypothesis(
            raw_content=content_upper,
            hypothesis_ids=hypothesis_ids,
        )

        if "REFUTES" in content_upper or "CONTRADICTS" in content_upper:
            evidence_type = EvidenceType.COUNTEREVIDENCE
            for hypothesis_id in hypothesis_ids:
                likelihoods[hypothesis_id] = (
                    LikelihoodBand.MODERATELY_DISCONFIRMING.value
                    if hypothesis_id == explicit_target
                    else LikelihoodBand.MODERATELY_CONFIRMING.value
                )
        elif "SUPPORTS" in content_upper:
            evidence_type = EvidenceType.SUPPORTING
            for hypothesis_id in hypothesis_ids:
                likelihoods[hypothesis_id] = (
                    LikelihoodBand.MODERATELY_CONFIRMING.value
                    if hypothesis_id == explicit_target
                    else LikelihoodBand.MODERATELY_DISCONFIRMING.value
                )
        elif "ANOMALY" in content_upper:
            evidence_type = EvidenceType.ANOMALY
            likelihoods = {
                hypothesis_id: LikelihoodBand.MODERATELY_DISCONFIRMING.value
                for hypothesis_id in hypothesis_ids
            }

        return {
            "evidence_type": evidence_type.value,
            "likelihoods": likelihoods,
            "interpretation": f"Deterministic v0.2 interpretation for {source_type}.",
            "quality_overrides": {},
        }


def _deterministic_target_hypothesis(
    *,
    raw_content: str,
    hypothesis_ids: list[str],
) -> str | None:
    if not hypothesis_ids:
        return None
    by_upper = {hypothesis_id.upper(): hypothesis_id for hypothesis_id in hypothesis_ids}
    match = re.search(
        r"\b(?:SUPPORTS|REFUTES|CONTRADICTS)\s+"
        r"(?:ANSWER\s+CHOICE\s+)?([A-Z][A-Z0-9_]*)\b",
        raw_content,
    )
    if match is not None and match.group(1) in by_upper:
        return by_upper[match.group(1)]
    return hypothesis_ids[0]


class ScriptedModelGateway:
    adapter_kind = "scripted"

    def __init__(self, responses: dict[str, dict[str, Any]]) -> None:
        self.responses = responses
        self.requests: list[StructuredModelRequest] = []

    def complete_structured(self, request: StructuredModelRequest) -> dict[str, Any]:
        self.requests.append(request)
        if request.task not in self.responses:
            raise ValueError(f"no scripted response for task: {request.task}")
        return self.responses[request.task]


def build_model_gateway(
    config: ModelGatewayConfig | Mapping[str, Any] | None = None,
) -> ModelGateway:
    gateway_config = _model_gateway_config_from_input(config)
    if gateway_config.kind == "deterministic":
        return DeterministicModelGateway()
    if gateway_config.kind == "scripted":
        if gateway_config.responses is None:
            raise ValueError("scripted model gateway requires responses")
        return ScriptedModelGateway(responses=gateway_config.responses)
    if gateway_config.kind == "openai":
        if gateway_config.model is None:
            raise ValueError("openai model gateway requires model")
        from bayesprobe.openai_gateway import (
            OpenAIModelGatewayConfig,
            OpenAIResponsesModelGateway,
        )

        return OpenAIResponsesModelGateway(
            config=OpenAIModelGatewayConfig(
                model=gateway_config.model,
                api_key_env=gateway_config.api_key_env,
                timeout_seconds=gateway_config.timeout_seconds,
                max_output_tokens=gateway_config.max_output_tokens,
                base_url=gateway_config.base_url,
                request_controls=gateway_config.request_controls,
            )
        )
    if gateway_config.kind == "openai_chat_completions":
        if gateway_config.model is None:
            raise ValueError("openai chat completions model gateway requires model")
        from bayesprobe.openai_gateway import (
            OpenAIChatCompletionsModelGateway,
            OpenAIModelGatewayConfig,
        )

        return OpenAIChatCompletionsModelGateway(
            config=OpenAIModelGatewayConfig(
                model=gateway_config.model,
                api_key_env=gateway_config.api_key_env,
                timeout_seconds=gateway_config.timeout_seconds,
                max_output_tokens=gateway_config.max_output_tokens,
                base_url=gateway_config.base_url,
                request_controls=gateway_config.request_controls,
            )
        )
    if gateway_config.kind == "recorded":
        if gateway_config.fixture_path is None:
            raise ValueError("recorded model gateway requires fixture_path")
        from bayesprobe.recorded_gateway import RecordedModelGateway

        return RecordedModelGateway.from_json(gateway_config.fixture_path)
    raise ValueError(f"unsupported model gateway kind: {gateway_config.kind}")


def model_gateway_adapter_kind(gateway: object) -> str:
    adapter_kind = getattr(gateway, "adapter_kind", None)
    if isinstance(adapter_kind, str) and adapter_kind.strip():
        return adapter_kind
    return gateway.__class__.__name__


def _model_gateway_config_from_input(
    config: ModelGatewayConfig | Mapping[str, Any] | None,
) -> ModelGatewayConfig:
    if config is None:
        return ModelGatewayConfig()
    if isinstance(config, ModelGatewayConfig):
        return config
    if not isinstance(config, Mapping):
        raise ValueError("model gateway config must be an object")

    kind = str(config.get("kind", "deterministic"))
    responses = config.get("responses")
    if responses is not None and not isinstance(responses, Mapping):
        raise ValueError("model gateway responses must be an object")
    model = config.get("model")
    if model is not None and not isinstance(model, str):
        raise ValueError("openai model gateway model must be a string")
    api_key_env = config.get("api_key_env", "OPENAI_API_KEY")
    timeout_seconds = config.get("timeout_seconds", 30.0)
    max_output_tokens = config.get("max_output_tokens")
    base_url = config.get("base_url")
    if base_url is not None and not isinstance(base_url, str):
        raise ValueError("openai model gateway base_url must be a string")
    fixture_path = config.get("fixture_path")
    if fixture_path is not None and not isinstance(fixture_path, (str, Path)):
        raise ValueError("recorded model gateway fixture_path must be a path")

    return ModelGatewayConfig(
        kind=kind,
        responses=dict(responses) if responses is not None else None,
        model=model,
        api_key_env=api_key_env,
        timeout_seconds=timeout_seconds,
        max_output_tokens=max_output_tokens,
        base_url=base_url,
        fixture_path=fixture_path,
        request_controls=ProviderRequestControls(
            temperature=config.get("temperature"),
            top_p=config.get("top_p"),
            thinking=config.get("thinking"),
            reasoning_effort=config.get("reasoning_effort"),
        ),
    )


__all__ = [
    "DeterministicModelGateway",
    "EvidenceJudgment",
    "EvidenceJudgmentRepairPolicy",
    "ModelGateway",
    "ModelGatewayConfig",
    "ModelGatewayValidationError",
    "ModelInvocationTrace",
    "ProviderRequestControls",
    "ScriptedModelGateway",
    "StructuredModelRequest",
    "build_model_gateway",
    "evidence_judgment_from_mapping",
    "model_gateway_adapter_kind",
]
