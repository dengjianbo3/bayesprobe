from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bayesprobe.model_gateway import (
    ModelGatewayValidationError,
    StructuredModelRequest,
    evidence_judgment_from_mapping,
)

_SECRET_KEY_PARTS = ("api_key", "apikey", "authorization", "token", "secret")


class RecordedModelGateway:
    adapter_kind = "recorded"

    def __init__(
        self,
        *,
        fixture_name: str,
        responses: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
        fixture_path: str | Path | None = None,
    ) -> None:
        if not isinstance(fixture_name, str) or not fixture_name.strip():
            raise ValueError("recorded model fixture_name must not be empty")
        self.fixture_name = fixture_name.strip()
        self.responses = list(responses)
        self.metadata = dict(metadata or {})
        self.fixture_path = Path(fixture_path) if fixture_path is not None else None
        self.requests: list[StructuredModelRequest] = []

    @classmethod
    def from_json(cls, path: str | Path) -> "RecordedModelGateway":
        fixture_path = Path(path)
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("recorded model fixture must be an object")
        _reject_secrets(payload)
        responses = payload.get("responses")
        if not isinstance(responses, list):
            raise ValueError("recorded model fixture responses must be an array")
        for entry in responses:
            _validate_entry(entry)
        fixture_name = payload.get("fixture_name", fixture_path.stem)
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("recorded model fixture metadata must be an object")
        return cls(
            fixture_name=str(fixture_name),
            responses=[dict(entry) for entry in responses],
            metadata=dict(metadata),
            fixture_path=fixture_path,
        )

    def complete_structured(self, request: StructuredModelRequest) -> dict[str, Any]:
        self.requests.append(request)
        signal_id = str(request.input.get("signal_id", ""))
        for entry in self.responses:
            match = entry["match"]
            if _matches_request(match, request):
                return dict(entry["response"])
        raise ModelGatewayValidationError(
            f"no recorded model response for task={request.task} signal_id={signal_id}"
        )


def _matches_request(match: Mapping[str, Any], request: StructuredModelRequest) -> bool:
    task = match.get("task")
    if task is not None and task != request.task:
        return False
    signal_id = match.get("signal_id")
    if signal_id is not None and signal_id != request.input.get("signal_id"):
        return False
    return True


def _validate_entry(entry: Any) -> None:
    if not isinstance(entry, Mapping):
        raise ValueError("recorded model response entry must be an object")
    match = entry.get("match")
    if not isinstance(match, Mapping):
        raise ValueError("recorded model response entry match must be an object")
    if "task" not in match:
        raise ValueError("recorded model response match must include task")
    response = entry.get("response")
    if not isinstance(response, Mapping):
        raise ValueError("recorded model response must be an object")
    evidence_judgment_from_mapping(dict(response))


def _reject_secrets(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).replace("_", "").replace("-", "").lower()
            if any(secret_part in key_text for secret_part in _SECRET_KEY_PARTS):
                raise ValueError("recorded model fixture must not contain secrets")
            _reject_secrets(item)
    elif isinstance(value, list):
        for item in value:
            _reject_secrets(item)


__all__ = ["RecordedModelGateway"]
