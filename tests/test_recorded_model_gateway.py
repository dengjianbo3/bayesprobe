import json
from pathlib import Path
from typing import Any

import pytest

from bayesprobe.model_gateway import ModelGatewayValidationError, StructuredModelRequest
from bayesprobe.recorded_gateway import RecordedModelGateway


def write_fixture(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_request(signal_id: str = "S_chem_constant_volume") -> StructuredModelRequest:
    return StructuredModelRequest(
        task="judge_evidence",
        input={
            "signal_id": signal_id,
            "raw_content": "Constant-volume inert gas evidence.",
            "target_hypotheses": ["H1", "H2"],
        },
        prompt_id="evidence_judgment",
        prompt_version="v0.1",
        schema_name="EvidenceJudgment",
        schema_version="v0.1",
    )


def recorded_fixture_payload() -> dict:
    return {
        "fixture_name": "deepseek_chat_evidence_v0_1",
        "metadata": {
            "provider_kind": "openai_chat_completions",
            "model": "deepseek-v4-flash",
            "recorded_at": "2026-07-10",
        },
        "responses": [
            {
                "match": {
                    "task": "judge_evidence",
                    "signal_id": "S_chem_constant_volume",
                },
                "response": {
                    "evidence_type": "supporting",
                    "likelihoods": {
                        "H1": "moderately_confirming",
                        "H2": "moderately_disconfirming",
                    },
                    "interpretation": "Recorded provider judgment.",
                    "quality_overrides": {},
                },
            }
        ],
    }


def test_recorded_model_gateway_replays_response_by_task_and_signal_id(tmp_path: Path):
    path = tmp_path / "recorded.json"
    write_fixture(path, recorded_fixture_payload())
    gateway = RecordedModelGateway.from_json(path)

    result = gateway.complete_structured(make_request())

    assert gateway.adapter_kind == "recorded"
    assert gateway.fixture_name == "deepseek_chat_evidence_v0_1"
    assert gateway.metadata["model"] == "deepseek-v4-flash"
    assert result["evidence_type"] == "supporting"
    assert result["likelihoods"]["H1"] == "moderately_confirming"
    assert gateway.requests[0].input["signal_id"] == "S_chem_constant_volume"


def test_recorded_model_gateway_raises_clear_error_when_no_entry_matches(
    tmp_path: Path,
):
    path = tmp_path / "recorded.json"
    write_fixture(path, recorded_fixture_payload())
    gateway = RecordedModelGateway.from_json(path)

    with pytest.raises(
        ModelGatewayValidationError,
        match="no recorded model response for task=judge_evidence signal_id=S_unknown",
    ):
        gateway.complete_structured(make_request("S_unknown"))


def test_recorded_model_gateway_rejects_fixture_with_api_key(tmp_path: Path):
    path = tmp_path / "unsafe.json"
    payload = recorded_fixture_payload()
    payload["metadata"]["api_key"] = "sk-unsafe"
    write_fixture(path, payload)

    with pytest.raises(
        ValueError,
        match="recorded model fixture must not contain secrets",
    ):
        RecordedModelGateway.from_json(path)


def test_recorded_model_gateway_replays_malformed_response_for_gate_validation(
    tmp_path: Path,
):
    path = tmp_path / "invalid.json"
    payload = recorded_fixture_payload()
    payload["responses"][0]["response"] = {"likelihoods": {}}
    write_fixture(path, payload)

    gateway = RecordedModelGateway.from_json(path)

    assert gateway.complete_structured(make_request()) == {"likelihoods": {}}


def test_open_question_fixture_is_recursively_secret_free_and_task_only_matched():
    payload = json.loads(
        Path(
            "tests/fixtures/open_questions/model_scale_validation_v0.1.json"
        ).read_text(encoding="utf-8")
    )

    assert _secret_like_entries(payload) == []
    assert [entry["match"] for entry in payload["responses"]] == [
        {"task": "frame_open_question"},
        {"task": "execute_probe"},
        {"task": "judge_evidence"},
    ]


def _secret_like_entries(value: Any, path: str = "$") -> list[str]:
    findings: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_path = f"{path}.{key}"
            normalized_key = str(key).replace("_", "").replace("-", "").lower()
            if any(part in normalized_key for part in ("apikey", "authorization", "token", "secret")):
                findings.append(key_path)
            findings.extend(_secret_like_entries(item, key_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            findings.extend(_secret_like_entries(item, f"{path}[{index}]"))
    elif isinstance(value, str) and value.lower().startswith("sk-"):
        findings.append(path)
    return findings
