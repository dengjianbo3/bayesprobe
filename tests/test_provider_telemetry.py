import json
import stat
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from bayesprobe.provider_telemetry import (
    JsonlProviderInvocationObserver,
    ProviderInvocationContext,
    ProviderInvocationRecord,
    ProviderUsage,
    extract_provider_response_metadata,
    provider_error_category,
    provider_usage_from_response,
    sanitized_request_sha256,
)


def make_record() -> ProviderInvocationRecord:
    return ProviderInvocationRecord(
        task="judge_evidence",
        adapter_kind="openai_chat_completions",
        model="provider-model",
        base_host="provider.example",
        prompt_id="evidence_judgment",
        prompt_version="v0.1",
        schema_name="EvidenceJudgment",
        schema_version="v0.1",
        request_sha256="a" * 64,
        started_at="2026-07-11T00:00:00Z",
        completed_at="2026-07-11T00:00:01Z",
        latency_seconds=1.0,
        usage=ProviderUsage(
            input_tokens=10,
            cached_input_tokens=4,
            reasoning_tokens=3,
            output_tokens=8,
            total_tokens=18,
        ),
        finish_reason="stop",
        response_id="resp_1",
        system_fingerprint="fp_1",
        outcome="success",
        error_category=None,
        context=ProviderInvocationContext(
            experiment_id="experiment_1",
            arm="direct_flash",
            sample_id="sample_pseudonym",
            run_id="run_1",
            cycle_id=None,
            probe_id=None,
            attempt_index=1,
        ),
    )


def test_provider_usage_from_chat_completion_mapping_reads_detail_tokens():
    response = {
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18,
            "prompt_tokens_details": {"cached_tokens": 4},
            "completion_tokens_details": {"reasoning_tokens": 3},
        }
    }

    assert provider_usage_from_response(response) == ProviderUsage(
        input_tokens=10,
        cached_input_tokens=4,
        reasoning_tokens=3,
        output_tokens=8,
        total_tokens=18,
    )


def test_provider_usage_from_responses_object_reads_detail_tokens():
    response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=12,
            output_tokens=7,
            total_tokens=19,
            input_tokens_details=SimpleNamespace(cached_tokens=5),
            output_tokens_details=SimpleNamespace(reasoning_tokens=6),
        )
    )

    assert provider_usage_from_response(response) == ProviderUsage(
        input_tokens=12,
        cached_input_tokens=5,
        reasoning_tokens=6,
        output_tokens=7,
        total_tokens=19,
    )


def test_provider_usage_from_deepseek_mapping_reads_cache_hit_tokens():
    response = {
        "usage": {
            "prompt_tokens": 20,
            "prompt_cache_hit_tokens": 9,
            "completion_tokens": 11,
            "completion_tokens_details": {"reasoning_tokens": 8},
            "total_tokens": 31,
        }
    }

    assert provider_usage_from_response(response).cached_input_tokens == 9


def test_extract_provider_response_metadata_supports_mapping_and_object():
    mapping = {
        "id": "chatcmpl_1",
        "system_fingerprint": "fp_map",
        "choices": [{"finish_reason": "stop"}],
    }
    object_response = SimpleNamespace(
        id="resp_2",
        system_fingerprint="fp_object",
        status="completed",
    )

    assert extract_provider_response_metadata(mapping) == (
        "stop",
        "chatcmpl_1",
        "fp_map",
    )
    assert extract_provider_response_metadata(object_response) == (
        "completed",
        "resp_2",
        "fp_object",
    )


def test_sanitized_request_hash_is_stable_and_secret_value_independent():
    first = {
        "model": "provider-model",
        "messages": [{"role": "user", "content": "same prompt"}],
        "api_key": "sk-first-secret",
        "headers": {"Authorization": "Bearer first"},
    }
    second = {
        "headers": {"Authorization": "Bearer second"},
        "api_key": "sk-second-secret",
        "messages": [{"content": "same prompt", "role": "user"}],
        "model": "provider-model",
    }

    assert sanitized_request_sha256(first) == sanitized_request_sha256(second)
    assert len(sanitized_request_sha256(first)) == 64


def test_sanitized_request_hash_ignores_secret_field_variants_and_secret_text():
    first = {
        "model": "provider-model",
        "messages": [{"role": "user", "content": "same prompt"}],
        "private_key": "first-private-value",
        "nested": {
            "credential": "first-credential-value",
            "header": "Authorization: Bearer abcdefghijklmnop",
        },
    }
    second = {
        "model": "provider-model",
        "messages": [{"role": "user", "content": "same prompt"}],
        "password": "second-password-value",
        "nested": {
            "access_key": "second-access-value",
            "header": "Authorization: Bearer zyxwvutsrqponmlk",
        },
    }

    assert sanitized_request_sha256(first) == sanitized_request_sha256(second)


@pytest.mark.parametrize(
    "first_secret,second_secret",
    [
        ("ghp_" + "a" * 36, "gho_" + "b" * 36),
        (
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0."
            "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
            "eyJ0eXAiOiJKV1QifQ.eyJpc3MiOiJwcm92aWRlciJ9."
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMN",
        ),
        (
            "Bearer abcdefghijklmnopqrstuvwx",
            "Bearer zyxwvutsrqponmlkjihgfedc",
        ),
        ("AKIAIOSFODNN7EXAMPLE", "ASIAQWERTYUIOPASDFGH"),
        (
            "xox" + "b-123456789012-1234567890123-abcdefghijklmnopqrstuvwx",
            "xox" + "p-987654321098-9876543210987-zyxwvutsrqponmlkjihgfedc",
        ),
    ],
)
def test_sanitized_request_hash_is_generic_credential_value_independent(
    first_secret,
    second_secret,
):
    first = {"model": "provider-model", "metadata": {"value": first_secret}}
    second = {"model": "provider-model", "metadata": {"value": second_secret}}

    assert sanitized_request_sha256(first) == sanitized_request_sha256(second)


def test_sanitized_request_hash_preserves_benign_secret_vocabulary_fields():
    first = {
        "token_count": 10,
        "password_policy": "rotate",
        "credential_score": 0.5,
        "cookie_policy": "strict",
    }
    second = {**first, "token_count": 11}

    assert sanitized_request_sha256(first) != sanitized_request_sha256(second)


def test_provider_error_category_normalizes_common_transport_failures():
    assert provider_error_category(TimeoutError("slow")) == "timeout"
    assert provider_error_category(ConnectionError("reset")) == "connection"
    assert (
        provider_error_category(SimpleNamespace(status_code=429))
        == "rate_limited"
    )
    assert (
        provider_error_category(SimpleNamespace(status_code=503))
        == "provider_server_error"
    )
    assert provider_error_category(ValueError("bad schema")) == "invalid_response"


def test_provider_error_category_recognizes_openai_style_exception_names():
    class APITimeoutError(RuntimeError):
        pass

    class APIConnectionError(RuntimeError):
        pass

    assert provider_error_category(APITimeoutError("slow")) == "timeout"
    assert provider_error_category(APIConnectionError("reset")) == "connection"


def test_jsonl_observer_appends_mode_0600_records(tmp_path: Path):
    path = tmp_path / "provider_invocations.jsonl"
    observer = JsonlProviderInvocationObserver(path)
    observer.observe(make_record())
    observer.observe(replace(make_record(), response_id="resp_2"))

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["usage"] == {
        "cached_input_tokens": 4,
        "input_tokens": 10,
        "output_tokens": 8,
        "reasoning_tokens": 3,
        "total_tokens": 18,
    }
    assert json.loads(lines[1])["response_id"] == "resp_2"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert "same prompt" not in path.read_text(encoding="utf-8")
