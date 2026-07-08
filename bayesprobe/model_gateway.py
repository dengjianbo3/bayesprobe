from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from bayesprobe.schemas import EvidenceType, LikelihoodBand


class ModelGatewayValidationError(ValueError):
    pass


@dataclass(frozen=True)
class StructuredModelRequest:
    task: str
    input: dict[str, Any]


@dataclass(frozen=True)
class ModelGatewayConfig:
    kind: str = "deterministic"
    responses: dict[str, dict[str, Any]] | None = None


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
        try:
            quality_overrides[str(metric)] = float(value)
        except (TypeError, ValueError) as error:
            raise ModelGatewayValidationError(
                f"invalid quality override for {metric}: {value}"
            ) from error

    return EvidenceJudgment(
        evidence_type=evidence_type,
        likelihoods=likelihoods,
        interpretation=str(payload.get("interpretation", "")),
        quality_overrides=quality_overrides,
    )


class DeterministicModelGateway:
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

        if "REFUTES" in content_upper or "CONTRADICTS" in content_upper:
            evidence_type = EvidenceType.COUNTEREVIDENCE
            if "H1" in likelihoods:
                likelihoods["H1"] = LikelihoodBand.MODERATELY_DISCONFIRMING.value
            if "H2" in likelihoods:
                likelihoods["H2"] = LikelihoodBand.MODERATELY_CONFIRMING.value
        elif "SUPPORTS" in content_upper:
            evidence_type = EvidenceType.SUPPORTING
            if "H1" in likelihoods:
                likelihoods["H1"] = LikelihoodBand.MODERATELY_CONFIRMING.value
            if "H2" in likelihoods:
                likelihoods["H2"] = LikelihoodBand.MODERATELY_DISCONFIRMING.value
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


class ScriptedModelGateway:
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
    raise ValueError(f"unsupported model gateway kind: {gateway_config.kind}")


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

    return ModelGatewayConfig(
        kind=kind,
        responses=dict(responses) if responses is not None else None,
    )


__all__ = [
    "DeterministicModelGateway",
    "EvidenceJudgment",
    "EvidenceJudgmentRepairPolicy",
    "ModelGateway",
    "ModelGatewayConfig",
    "ModelGatewayValidationError",
    "ScriptedModelGateway",
    "StructuredModelRequest",
    "build_model_gateway",
    "evidence_judgment_from_mapping",
]
