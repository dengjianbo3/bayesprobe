from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from write_benchmark_lock import write_lock_atomic


STAGE0_MODEL = "deepseek-v4-flash"
STAGE0_BASE_URL = "https://api.deepseek.com"
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_SECRET_PATTERN = re.compile(
    rb"(?:sk-[A-Za-z0-9_-]{12,}|tvly-[A-Za-z0-9_-]{12,}|"
    rb"github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16})",
    re.IGNORECASE,
)


class ProviderIdentityUsage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def require_consistent_total(self) -> ProviderIdentityUsage:
        if self.input_tokens + self.output_tokens != self.total_tokens:
            raise ValueError("provider identity token total disagrees")
        return self


class ProviderIdentityArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["terminal_bench_provider_identity:v1"]
    configured_model: str
    base_url: str | None
    provider_protocol: Literal["openai_chat_completions"]
    temperature: Literal[0]
    returned_model: str
    system_fingerprint_available: bool
    system_fingerprint: str | None
    usage: ProviderIdentityUsage
    content_sha256: str

    @field_validator("configured_model", "returned_model")
    @classmethod
    def require_nonempty_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("provider identity model must be non-empty")
        return value

    @field_validator("base_url")
    @classmethod
    def require_nonempty_base_url(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("provider identity base URL must be non-empty")
        return value

    @field_validator("system_fingerprint")
    @classmethod
    def require_nonempty_fingerprint(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("provider identity fingerprint must be non-empty")
        return value

    @field_validator("content_sha256")
    @classmethod
    def require_content_digest(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("provider identity content hash must be sha256")
        return value

    @model_validator(mode="after")
    def require_sealed_identity(self) -> ProviderIdentityArtifact:
        if self.system_fingerprint_available != (self.system_fingerprint is not None):
            raise ValueError("provider fingerprint availability disagrees")
        expected = provider_identity_content_sha256(
            self.model_dump(mode="json", exclude={"content_sha256"})
        )
        if self.content_sha256 != expected:
            raise ValueError("provider identity content hash mismatch")
        return self


def provider_identity_content_sha256(payload: Mapping[str, object]) -> str:
    serialized = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return f"sha256:{sha256(serialized.encode('utf-8')).hexdigest()}"


def capture_provider_identity(
    *,
    client: object,
    model: str,
    base_url: str | None,
) -> ProviderIdentityArtifact:
    completions = _required_attr(_required_attr(client, "chat"), "completions")
    create = _required_attr(completions, "create")
    if not callable(create):
        raise ValueError("provider identity client is invalid")
    response = create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Return exactly one empty JSON object.",
            }
        ],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=8,
    )
    returned_model = _required_text(
        _response_value(response, "model"),
        "provider returned model",
    )
    fingerprint_value = _response_value(response, "system_fingerprint")
    if fingerprint_value is None:
        fingerprint = None
    elif isinstance(fingerprint_value, str) and fingerprint_value.strip():
        fingerprint = fingerprint_value.strip()
    else:
        raise ValueError("provider system fingerprint is invalid")
    usage_value = _response_value(response, "usage")
    usage = ProviderIdentityUsage(
        input_tokens=_required_nonnegative_int(
            _response_value(usage_value, "prompt_tokens"),
            "provider input tokens",
        ),
        output_tokens=_required_nonnegative_int(
            _response_value(usage_value, "completion_tokens"),
            "provider output tokens",
        ),
        total_tokens=_required_nonnegative_int(
            _response_value(usage_value, "total_tokens"),
            "provider total tokens",
        ),
        cached_input_tokens=_optional_cached_tokens(usage_value),
    )
    payload: dict[str, object] = {
        "schema_version": "terminal_bench_provider_identity:v1",
        "configured_model": _required_text(model, "configured model"),
        "base_url": base_url,
        "provider_protocol": "openai_chat_completions",
        "temperature": 0,
        "returned_model": returned_model,
        "system_fingerprint_available": fingerprint is not None,
        "system_fingerprint": fingerprint,
        "usage": usage.model_dump(mode="json"),
    }
    return ProviderIdentityArtifact(
        **payload,
        content_sha256=provider_identity_content_sha256(payload),
    )


def write_provider_identity_artifact(
    output_directory: Path,
    artifact: ProviderIdentityArtifact,
    *,
    restricted_values: tuple[str, ...] = (),
) -> Path:
    validated = ProviderIdentityArtifact.model_validate(artifact)
    payload = validated.model_dump(mode="json")
    _require_secret_free(payload)
    output = Path(output_directory) / (
        f"{validated.content_sha256.removeprefix('sha256:')}.json"
    )
    write_lock_atomic(
        output,
        payload,
        restricted_values=restricted_values,
    )
    return output


def load_provider_identity_artifact(path: Path) -> ProviderIdentityArtifact:
    artifact_path = Path(path)
    try:
        artifact = ProviderIdentityArtifact.model_validate_json(
            artifact_path.read_text(encoding="utf-8")
        )
    except Exception:
        raise ValueError(
            "provider identity artifact content hash or schema is invalid"
        ) from None
    expected_name = f"{artifact.content_sha256.removeprefix('sha256:')}.json"
    if artifact_path.name != expected_name:
        raise ValueError("provider identity artifact is not content-addressed")
    _require_secret_free(artifact.model_dump(mode="json"))
    return artifact


def _require_secret_free(payload: Mapping[str, object]) -> None:
    serialized = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if _SECRET_PATTERN.search(serialized):
        raise ValueError("provider identity artifact contains secret-shaped content")


def _required_attr(value: object, name: str) -> object:
    try:
        result = getattr(value, name)
    except Exception:
        raise ValueError("provider identity client is invalid") from None
    return result


def _response_value(value: object, name: str) -> object:
    try:
        if isinstance(value, Mapping):
            return value.get(name)
        return getattr(value, name, None)
    except Exception:
        return None


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is required")
    return value.strip()


def _required_nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _optional_cached_tokens(usage: object) -> int | None:
    details = _response_value(usage, "prompt_tokens_details")
    cached = _response_value(details, "cached_tokens")
    if cached is None:
        return None
    return _required_nonnegative_int(cached, "provider cached input tokens")


def main(
    argv: Sequence[str] | None = None,
    *,
    client_factory: Any = OpenAI,
) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output-directory", required=True, type=Path)
    args = parser.parse_args(argv)
    api_key = os.environ.get("BAYESPROBE_BENCH_API_KEY", "").strip()
    if not api_key:
        raise ValueError("BAYESPROBE_BENCH_API_KEY is required")
    client = client_factory(
        api_key=api_key,
        base_url=STAGE0_BASE_URL,
        timeout=360,
        max_retries=0,
    )
    artifact = capture_provider_identity(
        client=client,
        model=STAGE0_MODEL,
        base_url=STAGE0_BASE_URL,
    )
    path = write_provider_identity_artifact(
        args.output_directory,
        artifact,
        restricted_values=(api_key,),
    )
    print(
        json.dumps(
            {
                "artifact": path.name,
                "content_sha256": artifact.content_sha256,
                "returned_model": artifact.returned_model,
                "system_fingerprint": artifact.system_fingerprint,
                "system_fingerprint_available": (
                    artifact.system_fingerprint_available
                ),
                "usage": artifact.usage.model_dump(mode="json"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
