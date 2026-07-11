from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol


_SECRET_KEYS = {
    "apikey",
    "authorization",
    "cookie",
    "password",
    "proxyauthorization",
    "secret",
    "token",
}


@dataclass(frozen=True)
class ProviderUsage:
    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    reasoning_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class ProviderInvocationContext:
    experiment_id: str | None = None
    arm: str | None = None
    sample_id: str | None = None
    run_id: str | None = None
    cycle_id: str | None = None
    probe_id: str | None = None
    attempt_index: int = 1

    def __post_init__(self) -> None:
        if type(self.attempt_index) is not int or self.attempt_index < 1:
            raise ValueError("provider invocation attempt_index must be positive")


@dataclass(frozen=True)
class ProviderInvocationRecord:
    task: str
    adapter_kind: str
    model: str
    base_host: str | None
    prompt_id: str | None
    prompt_version: str | None
    schema_name: str | None
    schema_version: str | None
    request_sha256: str
    started_at: str
    completed_at: str
    latency_seconds: float
    usage: ProviderUsage
    finish_reason: str | None
    response_id: str | None
    system_fingerprint: str | None
    outcome: str
    error_category: str | None
    context: ProviderInvocationContext

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProviderInvocationObserver(Protocol):
    def observe(self, record: ProviderInvocationRecord) -> None:
        ...


class JsonlProviderInvocationObserver:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def observe(self, record: ProviderInvocationRecord) -> None:
        payload = (
            json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8")
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            descriptor = os.open(
                self.path,
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                0o600,
            )
            try:
                _write_all(descriptor, payload)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)


def provider_usage_from_response(response: Any) -> ProviderUsage:
    usage = _read(response, "usage")
    if usage is None:
        return ProviderUsage()
    input_details = _read(usage, "prompt_tokens_details") or _read(
        usage, "input_tokens_details"
    )
    output_details = _read(usage, "completion_tokens_details") or _read(
        usage, "output_tokens_details"
    )
    return ProviderUsage(
        input_tokens=_token_count(usage, "prompt_tokens", "input_tokens"),
        cached_input_tokens=(
            _token_count(input_details, "cached_tokens")
            if input_details is not None
            else _token_count(
                usage,
                "prompt_cache_hit_tokens",
                "cached_input_tokens",
            )
        ),
        reasoning_tokens=_token_count(output_details, "reasoning_tokens"),
        output_tokens=_token_count(usage, "completion_tokens", "output_tokens"),
        total_tokens=_token_count(usage, "total_tokens"),
    )


def extract_provider_response_metadata(
    response: Any,
) -> tuple[str | None, str | None, str | None]:
    finish_reason = None
    choices = _read(response, "choices")
    if isinstance(choices, list | tuple) and choices:
        finish_reason = _string_or_none(_read(choices[0], "finish_reason"))
    if finish_reason is None:
        finish_reason = _string_or_none(_read(response, "status"))
    return (
        finish_reason,
        _string_or_none(_read(response, "id")),
        _string_or_none(_read(response, "system_fingerprint")),
    )


def sanitized_request_sha256(payload: Mapping[str, Any]) -> str:
    sanitized = _sanitize(payload)
    serialized = json.dumps(
        sanitized,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def provider_error_category(error: Any) -> str:
    status_code = _read(error, "status_code")
    if status_code == 429:
        return "rate_limited"
    if isinstance(status_code, int) and 500 <= status_code <= 599:
        return "provider_server_error"
    if isinstance(error, TimeoutError):
        return "timeout"
    if isinstance(error, ConnectionError):
        return "connection"
    error_type_name = type(error).__name__.lower()
    if "timeout" in error_type_name:
        return "timeout"
    if "connection" in error_type_name:
        return "connection"
    if isinstance(error, (ValueError, json.JSONDecodeError)):
        return "invalid_response"
    return "provider_error"


def _read(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _token_count(value: Any, *keys: str) -> int | None:
    if value is None:
        return None
    for key in keys:
        candidate = _read(value, key)
        if type(candidate) is int and candidate >= 0:
            return candidate
    return None


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            collapsed = "".join(character for character in key_text.lower() if character.isalnum())
            sanitized[key_text] = (
                "<redacted>" if collapsed in _SECRET_KEYS else _sanitize(item)
            )
        return sanitized
    if isinstance(value, list | tuple):
        return [_sanitize(item) for item in value]
    return value


def _write_all(descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written < 1:
            raise OSError("provider telemetry write made no progress")
        remaining = remaining[written:]


__all__ = [
    "JsonlProviderInvocationObserver",
    "ProviderInvocationContext",
    "ProviderInvocationObserver",
    "ProviderInvocationRecord",
    "ProviderUsage",
    "extract_provider_response_metadata",
    "provider_error_category",
    "provider_usage_from_response",
    "sanitized_request_sha256",
]
