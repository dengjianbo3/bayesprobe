from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from hashlib import sha256
import json
from typing import Any, Literal

from bayesprobe import StructuredModelRequest
from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, field_validator, model_validator

from bayesprobe_terminal_bench.artifacts import TrialArtifactStore
from bayesprobe_terminal_bench.config import BudgetExhausted


TERMINAL_HYPOTHESIS_TYPES = frozenset(
    {
        "root_cause",
        "current_behavior",
        "invariant",
        "postcondition",
        "causal_effect",
    }
)

_TERMINAL_CAPABILITIES = frozenset({"repository_read", "test_execution"})
_FRAME_REQUIRED_KEYS = frozenset(
    {
        "task_kind",
        "answer_relationship",
        "answer_contract",
        "competition",
        "coverage",
        "hypotheses",
        "coverage_statement",
        "coverage_limitation",
    }
)
_PROBE_REQUIRED_KEYS = frozenset({"proposals"})


class ProviderContractError(RuntimeError):
    def __init__(self, *, stage: str, attempts: int) -> None:
        self.stage = stage
        self.attempts = attempts
        super().__init__(f"{stage} provider contract failed after {attempts} attempts")


class ContractAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    stage: Literal["terminal_task_frame", "terminal_probe_design"]
    attempt_index: int
    request_task: str
    response_sha256: str | None
    required_keys_present: tuple[str, ...]
    validation: Literal["valid", "invalid", "provider_error", "empty"]
    field_errors: tuple[str, ...]


def safe_field_errors(error: ValidationError) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                f"{'.'.join(str(part) for part in item['loc'])}:{item['type']}"
                for item in error.errors(include_url=False, include_input=False)
            }
        )
    )[:32]


class _TerminalAnswerContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    objective: str
    answer_value_type: Literal["structured_text"]
    answer_format: str
    required_sections: list[str] = Field(min_length=1)
    decision_form: str
    permits_synthesis: Literal[True]

    @field_validator("objective", "answer_format", "decision_form")
    @classmethod
    def require_text(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("value must be non-empty text")
        return value.strip()

    @field_validator("required_sections")
    @classmethod
    def require_sections(cls, value: list[str]) -> list[str]:
        if any(not isinstance(item, str) or not item.strip() for item in value):
            raise ValueError("required sections must be non-empty text")
        normalized = [" ".join(item.casefold().split()) for item in value]
        if len(normalized) != len(set(normalized)):
            raise ValueError("required sections must be semantically distinct")
        return [item.strip() for item in value]


class _TerminalHypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    statement: str
    type: Literal[
        "root_cause",
        "current_behavior",
        "invariant",
        "postcondition",
        "causal_effect",
    ]
    scope: str
    falsifiers: list[str] = Field(min_length=1)
    predictions: list[str] = Field(min_length=1)
    answer_value: None

    @field_validator("statement", "scope")
    @classmethod
    def require_text(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("value must be non-empty text")
        return value.strip()

    @field_validator("falsifiers", "predictions")
    @classmethod
    def require_text_list(cls, value: list[str]) -> list[str]:
        if any(not isinstance(item, str) or not item.strip() for item in value):
            raise ValueError("items must be non-empty text")
        normalized = [" ".join(item.casefold().split()) for item in value]
        if len(normalized) != len(set(normalized)):
            raise ValueError("items must be semantically distinct")
        return [item.strip() for item in value]


class _TerminalTaskFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_kind: Literal["design"]
    answer_relationship: Literal["synthesis"]
    answer_contract: _TerminalAnswerContract
    competition: Literal["exclusive", "independent"]
    coverage: Literal["open"]
    hypotheses: list[_TerminalHypothesis] = Field(min_length=2, max_length=6)
    coverage_statement: str
    coverage_limitation: str | None

    @field_validator("coverage_statement")
    @classmethod
    def require_coverage_text(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("coverage statement must be non-empty text")
        return value.strip()

    @field_validator("coverage_limitation")
    @classmethod
    def clean_coverage_limitation(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError("coverage limitation must be non-empty text")
        return value.strip()

    @model_validator(mode="after")
    def require_distinct_hypotheses(self) -> "_TerminalTaskFrame":
        statements = [" ".join(item.statement.casefold().split()) for item in self.hypotheses]
        if len(statements) != len(set(statements)):
            raise ValueError("hypotheses must be semantically distinct")
        return self


class _TerminalProbeProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    purpose: Literal[
        "hypothesis_discrimination",
        "hypothesis_falsification",
        "frame_coverage",
        "source_verification",
        "anomaly_clarification",
        "answer_contract_gap",
    ]
    target_hypotheses: list[str] = Field(min_length=1)
    inquiry_goal: str
    expected_observation: str
    support_condition: dict[str, str]
    weaken_condition: dict[str, str]
    reframe_condition: dict[str, str] | None
    required_capability: Literal["repository_read", "test_execution"]

    @field_validator("target_hypotheses")
    @classmethod
    def require_known_distinct_targets(
        cls,
        value: list[str],
        info: ValidationInfo,
    ) -> list[str]:
        if any(not isinstance(item, str) or not item.strip() for item in value):
            raise ValueError("targets must be non-empty text")
        targets = [item.strip() for item in value]
        if len(targets) != len(set(targets)):
            raise ValueError("targets must be distinct")
        known_targets = set((info.context or {}).get("known_targets", ()))
        if set(targets).difference(known_targets):
            raise ValueError("targets must be known to the terminal frame")
        return targets

    @field_validator("inquiry_goal", "expected_observation")
    @classmethod
    def require_text(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("value must be non-empty text")
        return value.strip()

    @field_validator("support_condition", "weaken_condition")
    @classmethod
    def require_exact_target_conditions(
        cls,
        value: dict[str, str],
        info: ValidationInfo,
    ) -> dict[str, str]:
        targets = info.data.get("target_hypotheses")
        if not isinstance(targets, list) or set(value) != set(targets):
            raise ValueError("conditions must be keyed by exactly the targets")
        if any(not isinstance(item, str) or not item.strip() for item in value.values()):
            raise ValueError("condition text must be non-empty")
        return {key: item.strip() for key, item in value.items()}

    @field_validator("reframe_condition")
    @classmethod
    def require_optional_condition_text(
        cls,
        value: dict[str, str] | None,
    ) -> dict[str, str] | None:
        if value is None:
            return None
        if any(
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(item, str)
            or not item.strip()
            for key, item in value.items()
        ):
            raise ValueError("reframe condition must contain non-empty text")
        return {key.strip(): item.strip() for key, item in value.items()}

    @field_validator("required_capability")
    @classmethod
    def require_available_terminal_capability(
        cls,
        value: str,
        info: ValidationInfo,
    ) -> str:
        available = set((info.context or {}).get("available_capabilities", ()))
        if value not in available:
            raise ValueError("required terminal capability is unavailable")
        return value


class _TerminalProbeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposals: list[_TerminalProbeProposal] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def require_initial_open_coverage(self, info: ValidationInfo) -> "_TerminalProbeResponse":
        if not (info.context or {}).get("requires_initial_open_coverage", False):
            return self
        if not any(
            proposal.purpose in {"hypothesis_discrimination", "frame_coverage"}
            and len(proposal.target_hypotheses) >= 2
            for proposal in self.proposals
        ):
            raise ValueError(
                "initial open design requires a multi-hypothesis discriminator or frame-coverage proposal"
            )
        return self


class TerminalContractModelGateway:
    def __init__(
        self,
        delegate: Any,
        *,
        artifacts: TrialArtifactStore,
    ) -> None:
        self._delegate = delegate
        self._artifacts = artifacts

    @property
    def adapter_kind(self) -> str:
        value = getattr(self._delegate, "adapter_kind", None)
        return value if isinstance(value, str) and value.strip() else type(self._delegate).__name__

    @property
    def model_identity(self) -> str:
        value = getattr(self._delegate, "model_identity", None)
        return value if isinstance(value, str) and value.strip() else self.adapter_kind

    @property
    def config(self) -> Any:
        return getattr(self._delegate, "config", None)

    @property
    def invocation_observer(self) -> object | None:
        return getattr(self._delegate, "invocation_observer", None)

    def complete_structured(self, request: StructuredModelRequest) -> dict[str, Any]:
        if request.task == "frame_open_question":
            return self._complete_terminal(
                request,
                stage="terminal_task_frame",
                repair_task="repair_task_frame",
            )
        if request.task == "design_probes":
            return self._complete_terminal(
                request,
                stage="terminal_probe_design",
                repair_task="repair_probe_design",
            )
        return self._delegate.complete_structured(request)

    def _complete_terminal(
        self,
        request: StructuredModelRequest,
        *,
        stage: Literal["terminal_task_frame", "terminal_probe_design"],
        repair_task: Literal["repair_task_frame", "repair_probe_design"],
    ) -> dict[str, Any]:
        initial_request = self._with_terminal_policy(request, stage=stage)
        active_request = initial_request
        invalid_payload: Any = None
        field_errors: tuple[str, ...] = ()

        for attempt_index in range(3):
            try:
                response = self._delegate.complete_structured(active_request)
            except BudgetExhausted:
                raise
            except Exception:
                self._record_attempt(
                    stage=stage,
                    attempt_index=attempt_index,
                    request_task=active_request.task,
                    response=None,
                    validation="provider_error",
                    field_errors=(),
                )
            else:
                if _is_empty_response(response):
                    self._record_attempt(
                        stage=stage,
                        attempt_index=attempt_index,
                        request_task=active_request.task,
                        response=None,
                        validation="empty",
                        field_errors=(),
                    )
                    invalid_payload = response
                    field_errors = ()
                else:
                    try:
                        self._validate_response(stage, response, initial_request.input)
                    except ValidationError as error:
                        field_errors = safe_field_errors(error)
                        invalid_payload = response
                        self._record_attempt(
                            stage=stage,
                            attempt_index=attempt_index,
                            request_task=active_request.task,
                            response=response,
                            validation="invalid",
                            field_errors=field_errors,
                        )
                    else:
                        self._record_attempt(
                            stage=stage,
                            attempt_index=attempt_index,
                            request_task=active_request.task,
                            response=response,
                            validation="valid",
                            field_errors=(),
                        )
                        if not isinstance(response, dict):
                            raise ProviderContractError(stage=stage, attempts=attempt_index + 1)
                        return response

            if attempt_index == 2:
                raise ProviderContractError(stage=stage, attempts=3)
            active_request = self._repair_request(
                original_request=initial_request,
                invalid_payload=invalid_payload,
                field_errors=field_errors,
                attempt_index=attempt_index + 1,
                repair_task=repair_task,
                stage=stage,
            )

        raise AssertionError("terminal contract retry loop must return or raise")

    def _validate_response(
        self,
        stage: Literal["terminal_task_frame", "terminal_probe_design"],
        response: Any,
        request_input: Mapping[str, Any],
    ) -> None:
        if stage == "terminal_task_frame":
            _TerminalTaskFrame.model_validate(response)
            return
        _TerminalProbeResponse.model_validate(
            response,
            context=_probe_validation_context(request_input),
        )

    def _with_terminal_policy(
        self,
        request: StructuredModelRequest,
        *,
        stage: Literal["terminal_task_frame", "terminal_probe_design"],
    ) -> StructuredModelRequest:
        payload = dict(request.input)
        payload["terminal_policy"] = _terminal_policy(stage, payload)
        return replace(request, input=payload)

    def _repair_request(
        self,
        *,
        original_request: StructuredModelRequest,
        invalid_payload: Any,
        field_errors: tuple[str, ...],
        attempt_index: int,
        repair_task: Literal["repair_task_frame", "repair_probe_design"],
        stage: Literal["terminal_task_frame", "terminal_probe_design"],
    ) -> StructuredModelRequest:
        policy = original_request.input["terminal_policy"]
        repair_input: dict[str, Any] = {
            "original_request": dict(original_request.input),
            "invalid_payload": _redacted_payload_shape(invalid_payload),
            "validation_error": list(field_errors),
            "attempt_index": attempt_index,
            "terminal_policy": policy,
        }
        if stage == "terminal_task_frame":
            repair_input["required_fields"] = sorted(_FRAME_REQUIRED_KEYS)
        metadata = dict(original_request.metadata)
        metadata["repair_attempt_index"] = attempt_index
        return StructuredModelRequest(
            task=repair_task,
            input=repair_input,
            prompt_id=(
                "open_question_task_framing_repair"
                if stage == "terminal_task_frame"
                else "probe_design_repair"
            ),
            prompt_version=original_request.prompt_version,
            schema_name=original_request.schema_name,
            schema_version=original_request.schema_version,
            metadata=metadata,
        )

    def _record_attempt(
        self,
        *,
        stage: Literal["terminal_task_frame", "terminal_probe_design"],
        attempt_index: int,
        request_task: str,
        response: Any,
        validation: Literal["valid", "invalid", "provider_error", "empty"],
        field_errors: tuple[str, ...],
    ) -> None:
        required_keys = _FRAME_REQUIRED_KEYS if stage == "terminal_task_frame" else _PROBE_REQUIRED_KEYS
        present = (
            tuple(sorted(set(response).intersection(required_keys)))
            if isinstance(response, Mapping)
            else ()
        )
        attempt = ContractAttempt(
            stage=stage,
            attempt_index=attempt_index,
            request_task=request_task,
            response_sha256=(None if response is None else _response_sha256(response)),
            required_keys_present=present,
            validation=validation,
            field_errors=field_errors,
        )
        self._artifacts.append_contract_attempt(attempt)


def _terminal_policy(
    stage: Literal["terminal_task_frame", "terminal_probe_design"],
    request_input: Mapping[str, Any],
) -> dict[str, Any]:
    if stage == "terminal_task_frame":
        return {
            "stage": stage,
            "task_kind": "design",
            "answer_relationship": "synthesis",
            "coverage": "open",
            "hypothesis_count": {"minimum": 2, "maximum": 6},
            "hypothesis_types": sorted(TERMINAL_HYPOTHESIS_TYPES),
            "forbidden_hypothesis_types": ["implementation_policy", "patch_choice"],
            "answer_value": None,
        }
    context = _probe_validation_context(request_input)
    return {
        "stage": stage,
        "proposal_count": {"minimum": 1, "maximum": 3},
        "known_target_hypotheses": sorted(context["known_targets"]),
        "available_terminal_capabilities": sorted(context["available_capabilities"]),
        "requires_initial_open_coverage": context["requires_initial_open_coverage"],
        "condition_maps": {
            "support_condition": {
                "keys": "exactly_target_hypotheses",
                "values": "non_empty_text",
            },
            "weaken_condition": {
                "keys": "exactly_target_hypotheses",
                "values": "non_empty_text",
            },
        },
    }


def _probe_validation_context(request_input: Mapping[str, Any]) -> dict[str, Any]:
    hypotheses = request_input.get("hypotheses")
    hypothesis_items = hypotheses if isinstance(hypotheses, list) else []
    known_targets = {
        item.get("id")
        for item in hypothesis_items
        if isinstance(item, Mapping)
        and isinstance(item.get("id"), str)
        and item["id"].strip()
    }
    capabilities = request_input.get("available_capabilities")
    capability_items = capabilities if isinstance(capabilities, list) else []
    available_capabilities = {
        item.get("kind")
        for item in capability_items
        if isinstance(item, Mapping)
        and item.get("available") is True
        and item.get("kind") in _TERMINAL_CAPABILITIES
    }
    task_frame = request_input.get("task_frame")
    coverage = task_frame.get("coverage") if isinstance(task_frame, Mapping) else None
    return {
        "known_targets": tuple(sorted(known_targets)),
        "available_capabilities": tuple(sorted(available_capabilities)),
        "requires_initial_open_coverage": (
            coverage == "open" and request_input.get("cycle_id") == "cycle_0"
        ),
    }


def _is_empty_response(response: Any) -> bool:
    if response is None:
        return True
    if isinstance(response, str):
        return not response.strip()
    if isinstance(response, Mapping | Sequence):
        return not response
    return False


def _response_sha256(response: Any) -> str:
    serialized = json.dumps(
        response,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=lambda value: type(value).__qualname__,
    )
    return sha256(serialized.encode("utf-8")).hexdigest()


def _identity_sha256(value: Any) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return f"sha256:{sha256(serialized.encode('utf-8')).hexdigest()}"


def contract_identity() -> dict[str, str]:
    """Return canonical hashes for the adapter-owned provider contracts."""
    frame_prompt = {
        "max_attempts": 3,
        "policy": _terminal_policy("terminal_task_frame", {}),
        "repair_task": "repair_task_frame",
        "required_fields": sorted(_FRAME_REQUIRED_KEYS),
    }
    probe_prompt = {
        "max_attempts": 3,
        "policy": _terminal_policy(
            "terminal_probe_design",
            {
                "available_capabilities": [
                    {"available": True, "kind": "repository_read"},
                    {"available": True, "kind": "test_execution"},
                ],
                "cycle_id": "cycle_0",
                "hypotheses": [{"id": "<target_hypothesis>"}],
                "task_frame": {"coverage": "open"},
            },
        ),
        "repair_task": "repair_probe_design",
        "required_fields": sorted(_PROBE_REQUIRED_KEYS),
    }
    return {
        "terminal_task_frame:v1:prompt": _identity_sha256(frame_prompt),
        "terminal_task_frame:v1:schema": _identity_sha256(
            _TerminalTaskFrame.model_json_schema()
        ),
        "terminal_probe_design:v1:prompt": _identity_sha256(probe_prompt),
        "terminal_probe_design:v1:schema": _identity_sha256(
            _TerminalProbeResponse.model_json_schema()
        ),
    }


def _redacted_payload_shape(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            f"field_{index}": _redacted_payload_shape(item)
            for index, item in enumerate(value.values())
        }
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [_redacted_payload_shape(item) for item in value]
    return "[REDACTED]"


__all__ = [
    "ContractAttempt",
    "ProviderContractError",
    "TERMINAL_HYPOTHESIS_TYPES",
    "TerminalContractModelGateway",
    "contract_identity",
    "safe_field_errors",
]
