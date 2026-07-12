from __future__ import annotations

import hashlib
import json
import unicodedata
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
        for entry in copied_responses:
            _validate_entry(entry)
        copied_metadata = dict(metadata or {})
        _reject_secrets(
            {
                "fixture_name": clean_fixture_name,
                "responses": copied_responses,
                "metadata": copied_metadata,
            }
        )
        model_identity = _recorded_model_identity(
            fixture_name=clean_fixture_name,
            responses=copied_responses,
            metadata=copied_metadata,
        )
        self.fixture_name = clean_fixture_name
        self.responses = copied_responses
        self.metadata = copied_metadata
        self.fixture_path = Path(fixture_path) if fixture_path is not None else None
        self.model_identity = model_identity
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
    for key in ("cycle_id", "probe_id"):
        expected = match.get(key)
        if expected is not None and expected != request.metadata.get(key):
            return False
    return True


def _validate_entry(entry: Any) -> None:
    if not isinstance(entry, Mapping):
        raise ValueError("recorded model response entry must be an object")
    match = entry.get("match")
    if not isinstance(match, Mapping):
        raise ValueError("recorded model response entry match must be an object")
    unsupported_keys = set(match).difference(
        {"task", "signal_id", "cycle_id", "probe_id"}
    )
    if unsupported_keys:
        raise ValueError(
            "recorded model response match contains unsupported match key"
        )
    if "task" not in match:
        raise ValueError("recorded model response match must include task")
    response = entry.get("response")
    if not isinstance(response, Mapping):
        raise ValueError("recorded model response must be an object")


def _reject_secrets(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            normalized_key = unicodedata.normalize("NFKC", key_text)
            if (
                is_forbidden_secret_key_name(key_text)
                or is_forbidden_secret_key_name(normalized_key)
                or is_secret_like_value(key_text)
                or is_secret_like_value(normalized_key)
            ):
                raise ValueError("recorded model fixture must not contain secrets")
            _reject_secrets(item)
    elif isinstance(value, list | tuple):
        for item in value:
            _reject_secrets(item)
    elif isinstance(value, str):
        normalized = unicodedata.normalize("NFKC", value)
        if is_secret_like_value(value) or is_secret_like_value(normalized):
            raise ValueError("recorded model fixture must not contain secrets")


_EXPLICIT_IDENTITY_METADATA_KEYS = (
    "fixture_identity",
    "provider_kind",
    "provider",
    "model",
)
_EXCLUDED_IDENTITY_KEYS = {
    "apiurl",
    "baseurl",
    "endpoint",
    "filepath",
    "fixturepath",
    "headers",
    "localpath",
    "problem",
    "prompt",
    "question",
    "questiontext",
    "requestheaders",
}


def _recorded_model_identity(
    *,
    fixture_name: str,
    responses: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> str:
    explicit_metadata: dict[str, str] = {}
    for key in _EXPLICIT_IDENTITY_METADATA_KEYS:
        if key not in metadata:
            continue
        value = metadata[key]
        if not isinstance(value, str) or not value.strip() or "://" in value:
            raise ValueError("recorded model identity metadata must be a safe string")
        _reject_secrets(value)
        explicit_metadata[key] = value.strip()
    if explicit_metadata:
        identity_payload: Any = {
            "fixture_name": fixture_name,
            "identity_metadata": explicit_metadata,
        }
    else:
        identity_payload = _identity_fixture_content(
            {
                "fixture_name": fixture_name,
                "metadata": metadata,
                "responses": responses,
            }
        )
    try:
        canonical = json.dumps(
            identity_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("recorded model identity must be canonical JSON") from error
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"recorded:sha256:{digest}"


def _identity_fixture_content(value: Any) -> Any:
    if isinstance(value, Mapping):
        filtered: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            compact_key = "".join(
                character
                for character in unicodedata.normalize("NFKC", key_text).casefold()
                if character.isalnum()
            )
            if compact_key in _EXCLUDED_IDENTITY_KEYS:
                continue
            filtered[key_text] = _identity_fixture_content(item)
        return filtered
    if isinstance(value, list | tuple):
        return [_identity_fixture_content(item) for item in value]
    return value


__all__ = ["RecordedModelGateway"]
