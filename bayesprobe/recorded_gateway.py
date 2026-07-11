from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bayesprobe.model_gateway import (
    ModelGatewayValidationError,
    StructuredModelRequest,
)
from bayesprobe.schemas import is_forbidden_secret_key_name, is_secret_like_value


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
        clean_fixture_name = fixture_name.strip()
        copied_responses = list(responses)
        copied_metadata = dict(metadata or {})
        _reject_secrets(
            {
                "fixture_name": clean_fixture_name,
                "responses": copied_responses,
                "metadata": copied_metadata,
            }
        )
        self.fixture_name = clean_fixture_name
        self.responses = copied_responses
        self.metadata = copied_metadata
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


def _reject_secrets(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if (
                is_forbidden_secret_key_name(key_text)
                or is_secret_like_value(key_text)
            ):
                raise ValueError("recorded model fixture must not contain secrets")
            _reject_secrets(item)
    elif isinstance(value, list | tuple):
        for item in value:
            _reject_secrets(item)
    elif isinstance(value, str) and is_secret_like_value(value):
        raise ValueError("recorded model fixture must not contain secrets")


__all__ = ["RecordedModelGateway"]
