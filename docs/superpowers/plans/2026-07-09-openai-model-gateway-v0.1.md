# OpenAI ModelGateway v0.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an OpenAI-first provider-backed `ModelGateway` adapter that connects real model calls to BayesProbe's existing structured request/response seam without changing core control flow.

**Architecture:** Add a focused `bayesprobe/openai_gateway.py` module that implements `ModelGateway` through an `OpenAIResponsesModelGateway`, request-payload assembly, and response parsing helpers. Keep OpenAI optional and lazy-loaded so normal imports and tests do not require network, API keys, or the OpenAI package. Wire `kind="openai"` through existing config/factory/public SDK seams only after the adapter is test-covered.

**Tech Stack:** Python 3.11+, dataclasses, `collections.abc.Mapping`, optional `openai` Python package, pytest, existing BayesProbe `ModelGateway`, `StructuredModelRequest`, `ModelGatewayValidationError`, `EvidenceType`, and `LikelihoodBand`.

## Global Constraints

- No generic provider registry.
- No multi-provider abstraction.
- No prompt template registry.
- No provider token, cost, latency, or rate-limit accounting.
- No transport retry policy.
- No default live-network test.
- No API key in JSON config, ledger records, or model trace.
- No direct OpenAI response object stored as evidence.
- No bypass of `evidence_judgment_from_mapping(...)`.
- No changes to posterior update math.
- No changes to projection decomposition.
- No changes to probe planning or probe execution.
- No hidden chain-of-thought storage.
- `model` is required for OpenAI config and must be explicit.
- Default `pytest` must not make network calls.
- Live smoke is explicit opt-in.

---

## File Structure

- Create `bayesprobe/openai_gateway.py`: OpenAI-specific config, payload assembly, response parsing, lazy client construction, and `OpenAIResponsesModelGateway`.
- Modify `bayesprobe/model_gateway.py`: extend `ModelGatewayConfig` with OpenAI config fields and wire `kind="openai"` to the OpenAI adapter through a lazy import.
- Modify `bayesprobe/config.py`: parse OpenAI model gateway fields from JSON experiment config.
- Modify `bayesprobe/__init__.py`: export `OpenAIModelGatewayConfig`, `OpenAIResponsesModelGateway`, `build_openai_request_payload`, and `parse_openai_structured_response`.
- Modify `pyproject.toml`: add an optional `openai` extra without making OpenAI a required dependency.
- Create `tests/test_openai_gateway.py`: offline adapter, payload, response parsing, and validation tests.
- Modify `tests/test_model_gateway.py`: factory support for `kind="openai"`.
- Modify `tests/test_public_api_and_config.py`: JSON config parsing and public SDK exports.
- Modify `tests/test_experiment_runner.py`: experiment runner can construct an OpenAI gateway from config without network when only construction is needed.
- Create `tests/test_openai_live.py`: opt-in live smoke test skipped unless explicitly enabled.
- Modify `docs/ARCHITECTURE.md`: mark OpenAI adapter MVP status after implementation while provider observability remains future work.

### Task 1: OpenAI Adapter Module And Offline Unit Tests

**Files:**
- Create: `tests/test_openai_gateway.py`
- Create: `bayesprobe/openai_gateway.py`

**Interfaces:**
- Consumes:
  - `StructuredModelRequest`
  - `ModelGatewayValidationError`
  - `EvidenceType`
  - `LikelihoodBand`
- Produces:
  - `OpenAIModelGatewayConfig(model: str, api_key_env: str = "OPENAI_API_KEY", timeout_seconds: float = 30.0, max_output_tokens: int | None = None)`
  - `build_openai_request_payload(request: StructuredModelRequest, *, model: str, max_output_tokens: int | None = None) -> dict[str, Any]`
  - `parse_openai_structured_response(response: Any) -> dict[str, Any]`
  - `OpenAIResponsesModelGateway(config: OpenAIModelGatewayConfig, client: Any | None = None)`
  - `OpenAIResponsesModelGateway.complete_structured(request: StructuredModelRequest) -> dict[str, Any]`

- [ ] **Step 1: Write failing OpenAI config validation tests**

Create `tests/test_openai_gateway.py` with these imports and helpers:

```python
import json
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from bayesprobe.model_gateway import ModelGatewayValidationError, StructuredModelRequest
from bayesprobe.openai_gateway import (
    EVIDENCE_JUDGMENT_JSON_SCHEMA,
    OpenAIModelGatewayConfig,
    OpenAIResponsesModelGateway,
    build_openai_request_payload,
    parse_openai_structured_response,
)
```

Add:

```python
def make_judge_request() -> StructuredModelRequest:
    return StructuredModelRequest(
        task="judge_evidence",
        input={
            "signal_id": "S1",
            "source_type": "benchmark_stream",
            "source": "fixture",
            "raw_content": "SUPPORTS: source supports H1.",
            "target_hypotheses": ["H1", "H2"],
        },
        prompt_id="evidence_judgment",
        prompt_version="v0.1",
        schema_name="EvidenceJudgment",
        schema_version="v0.1",
        metadata={"run_id": "run_1"},
    )
```

Add:

```python
def test_openai_model_gateway_config_requires_explicit_model():
    config = OpenAIModelGatewayConfig(model="gpt-5.5")

    assert config.model == "gpt-5.5"
    assert config.api_key_env == "OPENAI_API_KEY"
    assert config.timeout_seconds == 30.0
    assert config.max_output_tokens is None
    with pytest.raises(FrozenInstanceError):
        config.model = "other"
```

Add:

```python
@pytest.mark.parametrize(
    ("kwargs", "expected_message"),
    [
        ({"model": ""}, "openai model gateway model must not be empty"),
        ({"model": "   "}, "openai model gateway model must not be empty"),
        ({"model": 1}, "openai model gateway model must be a string"),
        (
            {"model": "gpt-5.5", "api_key_env": ""},
            "openai model gateway api_key_env must not be empty",
        ),
        (
            {"model": "gpt-5.5", "api_key_env": 1},
            "openai model gateway api_key_env must be a string",
        ),
        (
            {"model": "gpt-5.5", "timeout_seconds": 0},
            "openai model gateway timeout_seconds must be positive",
        ),
        (
            {"model": "gpt-5.5", "timeout_seconds": "30"},
            "openai model gateway timeout_seconds must be a number",
        ),
        (
            {"model": "gpt-5.5", "max_output_tokens": 0},
            "openai model gateway max_output_tokens must be positive",
        ),
        (
            {"model": "gpt-5.5", "max_output_tokens": "128"},
            "openai model gateway max_output_tokens must be an integer",
        ),
    ],
)
def test_openai_model_gateway_config_rejects_invalid_values(kwargs, expected_message):
    with pytest.raises(ValueError, match=expected_message):
        OpenAIModelGatewayConfig(**kwargs)
```

- [ ] **Step 2: Write failing payload assembly tests**

Add:

```python
def test_build_openai_request_payload_for_judge_evidence():
    payload = build_openai_request_payload(
        make_judge_request(),
        model="gpt-5.5",
        max_output_tokens=256,
    )

    assert payload["model"] == "gpt-5.5"
    assert payload["max_output_tokens"] == 256
    assert payload["metadata"] == {
        "provider": "openai",
        "model": "gpt-5.5",
        "task": "judge_evidence",
        "prompt_id": "evidence_judgment",
        "prompt_version": "v0.1",
        "schema_name": "EvidenceJudgment",
        "schema_version": "v0.1",
        "run_id": "run_1",
    }
    assert payload["text"]["format"]["type"] == "json_schema"
    assert payload["text"]["format"]["name"] == "EvidenceJudgment"
    assert payload["text"]["format"]["strict"] is True
    assert payload["text"]["format"]["schema"] == EVIDENCE_JUDGMENT_JSON_SCHEMA
    assert payload["input"][0]["role"] == "developer"
    assert "BayesProbe" in payload["input"][0]["content"]
    assert payload["input"][1]["role"] == "user"
    user_payload = json.loads(payload["input"][1]["content"])
    assert user_payload["task"] == "judge_evidence"
    assert user_payload["input"]["signal_id"] == "S1"
```

Add:

```python
def test_build_openai_request_payload_for_repair_evidence_judgment():
    request = StructuredModelRequest(
        task="repair_evidence_judgment",
        input={
            "original_request": {"task": "judge_evidence", "input": {"signal_id": "S1"}},
            "invalid_payload": {"evidence_type": "not_a_type"},
            "validation_error": "invalid evidence_type: not_a_type",
            "attempt_index": 2,
        },
        prompt_id="evidence_judgment_repair",
        prompt_version="v0.1",
        schema_name="EvidenceJudgment",
        schema_version="v0.1",
        metadata={"repair_attempt_index": 2},
    )

    payload = build_openai_request_payload(request, model="gpt-5.5")

    assert "max_output_tokens" not in payload
    assert payload["metadata"]["task"] == "repair_evidence_judgment"
    assert payload["metadata"]["prompt_id"] == "evidence_judgment_repair"
    assert payload["metadata"]["repair_attempt_index"] == "2"
    assert "Repair" in payload["input"][0]["content"]
    assert json.loads(payload["input"][1]["content"])["input"]["attempt_index"] == 2
```

Add:

```python
def test_build_openai_request_payload_rejects_unknown_task():
    request = StructuredModelRequest(task="other_task", input={})

    with pytest.raises(ValueError, match="unsupported openai model task: other_task"):
        build_openai_request_payload(request, model="gpt-5.5")
```

- [ ] **Step 3: Write failing response parsing tests**

Add:

```python
def valid_payload() -> dict[str, object]:
    return {
        "evidence_type": "supporting",
        "likelihoods": {"H1": "moderately_confirming", "H2": "neutral"},
        "interpretation": "OpenAI fixture judgment.",
        "quality_overrides": {},
    }
```

Add:

```python
def test_parse_openai_structured_response_accepts_direct_dict():
    assert parse_openai_structured_response(valid_payload()) == valid_payload()
```

Add:

```python
def test_parse_openai_structured_response_accepts_json_string():
    assert parse_openai_structured_response(json.dumps(valid_payload())) == valid_payload()
```

Add:

```python
def test_parse_openai_structured_response_accepts_output_text_object():
    response = SimpleNamespace(output_text=json.dumps(valid_payload()))

    assert parse_openai_structured_response(response) == valid_payload()
```

Add:

```python
def test_parse_openai_structured_response_accepts_mapping_output_text():
    response = {"output_text": json.dumps(valid_payload())}

    assert parse_openai_structured_response(response) == valid_payload()
```

Add:

```python
@pytest.mark.parametrize(
    ("response", "expected_message"),
    [
        ("{", "openai structured response was not valid JSON"),
        ("[]", "openai structured response must be an object"),
        (["not", "valid"], "openai structured response must be an object"),
        (SimpleNamespace(output_text=None), "openai structured response text was missing"),
    ],
)
def test_parse_openai_structured_response_rejects_malformed_output(
    response,
    expected_message,
):
    with pytest.raises(ModelGatewayValidationError, match=expected_message):
        parse_openai_structured_response(response)
```

- [ ] **Step 4: Write failing gateway fake-client tests**

Add:

```python
class FakeResponses:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **payload):
        self.calls.append(payload)
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


class FakeOpenAIClient:
    def __init__(self, response):
        self.responses = FakeResponses(response)
```

Add:

```python
def test_openai_responses_model_gateway_calls_fake_client_and_returns_dict():
    client = FakeOpenAIClient(response=json.dumps(valid_payload()))
    gateway = OpenAIResponsesModelGateway(
        config=OpenAIModelGatewayConfig(model="gpt-5.5", max_output_tokens=128),
        client=client,
    )

    result = gateway.complete_structured(make_judge_request())

    assert result == valid_payload()
    assert len(client.responses.calls) == 1
    assert client.responses.calls[0]["model"] == "gpt-5.5"
    assert client.responses.calls[0]["max_output_tokens"] == 128
```

Add:

```python
def test_openai_responses_model_gateway_propagates_provider_exceptions():
    error = RuntimeError("provider outage")
    gateway = OpenAIResponsesModelGateway(
        config=OpenAIModelGatewayConfig(model="gpt-5.5"),
        client=FakeOpenAIClient(response=error),
    )

    with pytest.raises(RuntimeError, match="provider outage"):
        gateway.complete_structured(make_judge_request())
```

Add:

```python
def test_openai_responses_model_gateway_raises_clear_error_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    gateway = OpenAIResponsesModelGateway(
        config=OpenAIModelGatewayConfig(model="gpt-5.5"),
    )

    with pytest.raises(RuntimeError, match="OpenAI API key environment variable OPENAI_API_KEY is not set"):
        gateway.complete_structured(make_judge_request())
```

- [ ] **Step 5: Run tests to verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_openai_gateway.py -q -p no:cacheprovider
```

Expected: failure because `bayesprobe.openai_gateway` does not exist.

- [ ] **Step 6: Implement `bayesprobe/openai_gateway.py`**

Create `bayesprobe/openai_gateway.py`:

```python
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from bayesprobe.model_gateway import ModelGatewayValidationError, StructuredModelRequest
from bayesprobe.schemas import EvidenceType, LikelihoodBand


EVIDENCE_JUDGMENT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "evidence_type",
        "likelihoods",
        "interpretation",
        "quality_overrides",
    ],
    "properties": {
        "evidence_type": {
            "type": "string",
            "enum": [evidence_type.value for evidence_type in EvidenceType],
        },
        "likelihoods": {
            "type": "object",
            "additionalProperties": {
                "type": "string",
                "enum": [band.value for band in LikelihoodBand],
            },
        },
        "interpretation": {"type": "string"},
        "quality_overrides": {
            "type": "object",
            "additionalProperties": {"type": "number"},
        },
    },
}


@dataclass(frozen=True)
class OpenAIModelGatewayConfig:
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    timeout_seconds: float = 30.0
    max_output_tokens: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model, str):
            raise ValueError("openai model gateway model must be a string")
        if not self.model.strip():
            raise ValueError("openai model gateway model must not be empty")
        if not isinstance(self.api_key_env, str):
            raise ValueError("openai model gateway api_key_env must be a string")
        if not self.api_key_env.strip():
            raise ValueError("openai model gateway api_key_env must not be empty")
        if type(self.timeout_seconds) not in (int, float):
            raise ValueError("openai model gateway timeout_seconds must be a number")
        if self.timeout_seconds <= 0:
            raise ValueError("openai model gateway timeout_seconds must be positive")
        if self.max_output_tokens is not None:
            if type(self.max_output_tokens) is not int:
                raise ValueError("openai model gateway max_output_tokens must be an integer")
            if self.max_output_tokens < 1:
                raise ValueError("openai model gateway max_output_tokens must be positive")
        object.__setattr__(self, "model", self.model.strip())
        object.__setattr__(self, "api_key_env", self.api_key_env.strip())


class OpenAIResponsesModelGateway:
    adapter_kind = "openai"

    def __init__(
        self,
        *,
        config: OpenAIModelGatewayConfig,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self._client = client

    def complete_structured(self, request: StructuredModelRequest) -> dict[str, Any]:
        payload = build_openai_request_payload(
            request,
            model=self.config.model,
            max_output_tokens=self.config.max_output_tokens,
        )
        response = self._client_for_request().responses.create(**payload)
        return parse_openai_structured_response(response)

    def _client_for_request(self) -> Any:
        if self._client is None:
            self._client = _build_default_openai_client(self.config)
        return self._client


def build_openai_request_payload(
    request: StructuredModelRequest,
    *,
    model: str,
    max_output_tokens: int | None = None,
) -> dict[str, Any]:
    instruction = _instruction_for_task(request.task)
    payload: dict[str, Any] = {
        "model": model,
        "input": [
            {
                "role": "developer",
                "content": instruction,
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": request.task,
                        "input": request.input,
                    },
                    sort_keys=True,
                ),
            },
        ],
        "metadata": _metadata_for_request(request, model=model),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "EvidenceJudgment",
                "strict": True,
                "schema": EVIDENCE_JUDGMENT_JSON_SCHEMA,
            }
        },
    }
    if max_output_tokens is not None:
        payload["max_output_tokens"] = max_output_tokens
    return payload


def parse_openai_structured_response(response: Any) -> dict[str, Any]:
    if isinstance(response, Mapping):
        if "output_text" in response:
            return _parse_json_object(response["output_text"])
        if "text" in response:
            return _parse_json_object(response["text"])
        return dict(response)
    if isinstance(response, str):
        return _parse_json_object(response)

    output_text = getattr(response, "output_text", None)
    if output_text is not None:
        return _parse_json_object(output_text)

    text = _extract_text_from_output(response)
    if text is not None:
        return _parse_json_object(text)

    raise ModelGatewayValidationError("openai structured response text was missing")


def _instruction_for_task(task: str) -> str:
    if task == "judge_evidence":
        return (
            "You are the evidence judgment component inside BayesProbe. "
            "Convert the provided signal context into one EvidenceJudgment JSON object. "
            "Use only the supplied hypotheses and likelihood bands."
        )
    if task == "repair_evidence_judgment":
        return (
            "Repair the malformed BayesProbe evidence judgment. "
            "Return exactly one valid EvidenceJudgment JSON object. "
            "Preserve the intended evidence meaning when it can be inferred."
        )
    raise ValueError(f"unsupported openai model task: {task}")


def _metadata_for_request(request: StructuredModelRequest, *, model: str) -> dict[str, str]:
    metadata: dict[str, str] = {
        "provider": "openai",
        "model": model,
        "task": request.task,
    }
    optional_fields = {
        "prompt_id": request.prompt_id,
        "prompt_version": request.prompt_version,
        "schema_name": request.schema_name,
        "schema_version": request.schema_version,
    }
    for key, value in optional_fields.items():
        if value is not None:
            metadata[key] = value
    for key, value in request.metadata.items():
        if isinstance(value, str):
            metadata[str(key)] = value
        elif isinstance(value, bool | int | float):
            metadata[str(key)] = str(value)
    return metadata


def _parse_json_object(text: Any) -> dict[str, Any]:
    if not isinstance(text, str):
        raise ModelGatewayValidationError("openai structured response text was missing")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        raise ModelGatewayValidationError(
            "openai structured response was not valid JSON"
        ) from error
    if not isinstance(parsed, Mapping):
        raise ModelGatewayValidationError("openai structured response must be an object")
    return dict(parsed)


def _extract_text_from_output(response: Any) -> str | None:
    output = getattr(response, "output", None)
    if output is None:
        return None
    for item in output:
        content = getattr(item, "content", None)
        if content is None and isinstance(item, Mapping):
            content = item.get("content")
        if content is None:
            continue
        for part in content:
            text = getattr(part, "text", None)
            if text is None and isinstance(part, Mapping):
                text = part.get("text")
            if isinstance(text, str):
                return text
    return None


def _build_default_openai_client(config: OpenAIModelGatewayConfig) -> Any:
    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"OpenAI API key environment variable {config.api_key_env} is not set"
        )
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError(
            "OpenAI Python package is required for OpenAIResponsesModelGateway. "
            "Install bayesprobe[openai] or install openai."
        ) from error
    return OpenAI(api_key=api_key, timeout=config.timeout_seconds)


__all__ = [
    "EVIDENCE_JUDGMENT_JSON_SCHEMA",
    "OpenAIModelGatewayConfig",
    "OpenAIResponsesModelGateway",
    "build_openai_request_payload",
    "parse_openai_structured_response",
]
```

- [ ] **Step 7: Run OpenAI gateway unit tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_openai_gateway.py -q -p no:cacheprovider
```

Expected: all tests in `tests/test_openai_gateway.py` pass.

- [ ] **Step 8: Commit Task 1**

Run:

```bash
git add bayesprobe/openai_gateway.py tests/test_openai_gateway.py
git commit -m "feat: add openai model gateway adapter"
```

### Task 2: Gateway Factory And Experiment Config Wiring

**Files:**
- Modify: `bayesprobe/model_gateway.py`
- Modify: `bayesprobe/config.py`
- Modify: `tests/test_model_gateway.py`
- Modify: `tests/test_public_api_and_config.py`
- Modify: `tests/test_experiment_runner.py`

**Interfaces:**
- Consumes:
  - `OpenAIModelGatewayConfig`
  - `OpenAIResponsesModelGateway`
- Produces:
  - `ModelGatewayConfig(kind="openai", model=..., api_key_env=..., timeout_seconds=..., max_output_tokens=...)`
  - `build_model_gateway({"kind": "openai", "model": "gpt-5.5"}) -> OpenAIResponsesModelGateway`
  - `experiment_config_from_mapping(...)` support for OpenAI fields

- [ ] **Step 1: Write failing factory tests**

Update imports in `tests/test_model_gateway.py`:

```python
from bayesprobe.openai_gateway import OpenAIResponsesModelGateway
```

Add after scripted config tests:

```python
def test_build_model_gateway_accepts_openai_mapping_without_network():
    gateway = build_model_gateway(
        {
            "kind": "openai",
            "model": "gpt-5.5",
            "api_key_env": "BAYESPROBE_TEST_OPENAI_KEY",
            "timeout_seconds": 12.5,
            "max_output_tokens": 256,
        }
    )

    assert isinstance(gateway, OpenAIResponsesModelGateway)
    assert gateway.config.model == "gpt-5.5"
    assert gateway.config.api_key_env == "BAYESPROBE_TEST_OPENAI_KEY"
    assert gateway.config.timeout_seconds == 12.5
    assert gateway.config.max_output_tokens == 256
```

Add:

```python
def test_build_model_gateway_rejects_openai_without_model():
    with pytest.raises(ValueError, match="openai model gateway requires model"):
        build_model_gateway({"kind": "openai"})
```

Add:

```python
def test_build_model_gateway_rejects_invalid_openai_timeout():
    with pytest.raises(ValueError, match="openai model gateway timeout_seconds must be positive"):
        build_model_gateway({"kind": "openai", "model": "gpt-5.5", "timeout_seconds": 0})
```

- [ ] **Step 2: Write failing experiment config parsing test**

Add to `tests/test_public_api_and_config.py`:

```python
def test_experiment_config_from_mapping_parses_openai_model_gateway(tmp_path: Path):
    config = experiment_config_from_mapping(
        {
            "dataset_path": "datasets/toy.json",
            "report_path": "outputs/toy-report.json",
            "model_gateway": {
                "kind": "openai",
                "model": "gpt-5.5",
                "api_key_env": "BAYESPROBE_TEST_OPENAI_KEY",
                "timeout_seconds": 12.5,
                "max_output_tokens": 256,
            },
        },
        base_dir=tmp_path,
    )

    assert isinstance(config.model_gateway, ModelGatewayConfig)
    assert config.model_gateway.kind == "openai"
    assert config.model_gateway.model == "gpt-5.5"
    assert config.model_gateway.api_key_env == "BAYESPROBE_TEST_OPENAI_KEY"
    assert config.model_gateway.timeout_seconds == 12.5
    assert config.model_gateway.max_output_tokens == 256
```

Add an invalid config case to the existing parametrized invalid config list:

```python
(
    "openai_missing_model.json",
    json.dumps(
        {
            "dataset_path": "dataset.json",
            "report_path": "report.json",
            "model_gateway": {"kind": "openai"},
        }
    ),
    "openai model gateway requires model",
),
```

- [ ] **Step 3: Write failing experiment runner construction test**

Add to `tests/test_experiment_runner.py`:

```python
def test_run_benchmark_experiment_constructs_openai_gateway_without_network(
    tmp_path: Path,
    monkeypatch,
):
    import bayesprobe.experiment_runner as experiment_runner
    from bayesprobe.openai_gateway import OpenAIModelGatewayConfig

    captured = {}

    class CapturingGateway:
        adapter_kind = "capturing_openai"

        def __init__(self, *, config):
            captured["config"] = config

        def complete_structured(self, request):
            return {
                "evidence_type": "supporting",
                "likelihoods": {
                    "H1": "moderately_confirming",
                    "H2": "moderately_disconfirming",
                },
                "interpretation": "Captured OpenAI fixture.",
                "quality_overrides": {},
            }

    def fake_build_model_gateway(config):
        assert config["kind"] == "openai"
        return CapturingGateway(
            config=OpenAIModelGatewayConfig(model=config["model"])
        )

    monkeypatch.setattr(experiment_runner, "build_model_gateway", fake_build_model_gateway)
    report_path = tmp_path / "toy-report.json"

    result = run_benchmark_experiment(
        ExperimentRunConfig(
            dataset_path=FIXTURE_PATH,
            report_path=report_path,
            model_gateway={"kind": "openai", "model": "gpt-5.5"},
        )
    )

    assert captured["config"].model == "gpt-5.5"
    assert result.suite_result.sample_count == 3
```

- [ ] **Step 4: Run tests to verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_model_gateway.py::test_build_model_gateway_accepts_openai_mapping_without_network tests/test_public_api_and_config.py::test_experiment_config_from_mapping_parses_openai_model_gateway tests/test_experiment_runner.py::test_run_benchmark_experiment_constructs_openai_gateway_without_network -q -p no:cacheprovider
```

Expected: failures because `ModelGatewayConfig` and config parsing do not yet support OpenAI fields.

- [ ] **Step 5: Extend `ModelGatewayConfig`**

Update `bayesprobe/model_gateway.py` dataclass:

```python
@dataclass(frozen=True)
class ModelGatewayConfig:
    kind: str = "deterministic"
    responses: dict[str, dict[str, Any]] | None = None
    model: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    timeout_seconds: float = 30.0
    max_output_tokens: int | None = None
```

Update `build_model_gateway(...)`:

```python
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
            )
        )
    raise ValueError(f"unsupported model gateway kind: {gateway_config.kind}")
```

Update `_model_gateway_config_from_input(...)`:

```python
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

    return ModelGatewayConfig(
        kind=kind,
        responses=dict(responses) if responses is not None else None,
        model=model,
        api_key_env=api_key_env,
        timeout_seconds=timeout_seconds,
        max_output_tokens=max_output_tokens,
    )
```

- [ ] **Step 6: Extend experiment config parsing**

Update `_optional_model_gateway_config(...)` in `bayesprobe/config.py`:

```python
def _optional_model_gateway_config(data: Mapping[str, Any]) -> ModelGatewayConfig | None:
    if "model_gateway" not in data or data["model_gateway"] is None:
        return None
    value = data["model_gateway"]
    if not isinstance(value, Mapping):
        raise ValueError("experiment config field model_gateway must be an object")

    kind = str(value.get("kind", "deterministic"))
    responses = value.get("responses")
    if responses is not None and not isinstance(responses, Mapping):
        raise ValueError("model gateway responses must be an object")

    model = value.get("model")
    if kind == "openai" and model is None:
        raise ValueError("openai model gateway requires model")
    api_key_env = value.get("api_key_env", "OPENAI_API_KEY")
    timeout_seconds = value.get("timeout_seconds", 30.0)
    max_output_tokens = value.get("max_output_tokens")

    return ModelGatewayConfig(
        kind=kind,
        responses=dict(responses) if responses is not None else None,
        model=model,
        api_key_env=api_key_env,
        timeout_seconds=timeout_seconds,
        max_output_tokens=max_output_tokens,
    )
```

- [ ] **Step 7: Run focused config/factory tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_model_gateway.py tests/test_public_api_and_config.py tests/test_experiment_runner.py -q -p no:cacheprovider
```

Expected: all selected tests pass.

- [ ] **Step 8: Commit Task 2**

Run:

```bash
git add bayesprobe/model_gateway.py bayesprobe/config.py tests/test_model_gateway.py tests/test_public_api_and_config.py tests/test_experiment_runner.py
git commit -m "feat: wire openai gateway config"
```

### Task 3: Public SDK, Optional Dependency, And Live Smoke

**Files:**
- Modify: `bayesprobe/__init__.py`
- Modify: `pyproject.toml`
- Create: `tests/test_openai_live.py`
- Modify: `tests/test_public_api_and_config.py`

**Interfaces:**
- Consumes:
  - `OpenAIModelGatewayConfig`
  - `OpenAIResponsesModelGateway`
  - `build_openai_request_payload`
  - `parse_openai_structured_response`
- Produces:
  - package root exports for OpenAI adapter/config/helpers
  - optional dependency extra `openai = ["openai>=1.0,<3"]`
  - opt-in live smoke test skipped by default

- [ ] **Step 1: Write failing SDK export test**

Update imports in `tests/test_public_api_and_config.py`:

```python
from bayesprobe import (
    BenchmarkDataset,
    BenchmarkHarness,
    BenchmarkSample,
    BenchmarkSampleResult,
    BenchmarkSignal,
    BenchmarkSignalShape,
    BenchmarkSuiteResult,
    DeterministicModelGateway,
    EvidenceJudgment,
    EvidenceJudgmentRepairPolicy,
    ExperimentRunConfig,
    ExperimentRunResult,
    ModelGateway,
    ModelGatewayConfig,
    ModelGatewayValidationError,
    ModelInvocationTrace,
    OpenAIModelGatewayConfig,
    OpenAIResponsesModelGateway,
    ScriptedModelGateway,
    StructuredModelRequest,
    build_model_gateway,
    build_openai_request_payload,
    evidence_judgment_from_mapping,
    load_benchmark_dataset,
    load_experiment_config,
    parse_openai_structured_response,
    run_benchmark_experiment,
    write_benchmark_report,
)
```

Add these names to `expected_names`:

```python
"OpenAIModelGatewayConfig",
"OpenAIResponsesModelGateway",
"build_openai_request_payload",
"parse_openai_structured_response",
```

Add assertions:

```python
assert OpenAIModelGatewayConfig is not None
assert OpenAIResponsesModelGateway is not None
assert build_openai_request_payload is not None
assert parse_openai_structured_response is not None
```

- [ ] **Step 2: Write opt-in live smoke test**

Create `tests/test_openai_live.py`:

```python
import os

import pytest

from bayesprobe.model_gateway import StructuredModelRequest, evidence_judgment_from_mapping
from bayesprobe.openai_gateway import OpenAIModelGatewayConfig, OpenAIResponsesModelGateway


def test_openai_live_smoke_judges_evidence_when_explicitly_enabled():
    if os.environ.get("BAYESPROBE_RUN_OPENAI_LIVE") != "1":
        pytest.skip("set BAYESPROBE_RUN_OPENAI_LIVE=1 to run OpenAI live smoke")
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("set OPENAI_API_KEY to run OpenAI live smoke")

    gateway = OpenAIResponsesModelGateway(
        config=OpenAIModelGatewayConfig(model="gpt-5.5", max_output_tokens=256)
    )
    payload = gateway.complete_structured(
        StructuredModelRequest(
            task="judge_evidence",
            input={
                "signal_id": "S_live_openai",
                "source_type": "live_smoke",
                "source": "pytest",
                "raw_content": "SUPPORTS: this fixture supports H1 more than H2.",
                "target_hypotheses": ["H1", "H2"],
            },
            prompt_id="evidence_judgment",
            prompt_version="v0.1",
            schema_name="EvidenceJudgment",
            schema_version="v0.1",
        )
    )

    judgment = evidence_judgment_from_mapping(payload)

    assert judgment.interpretation
```

- [ ] **Step 3: Run tests to verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_public_api_and_config.py::test_public_sdk_exports_supported_names tests/test_openai_live.py -q -p no:cacheprovider
```

Expected: public export test fails because package root does not export OpenAI names; live smoke test is skipped by default.

- [ ] **Step 4: Export OpenAI names from package root**

Update `bayesprobe/__init__.py` imports:

```python
from bayesprobe.openai_gateway import (
    OpenAIModelGatewayConfig,
    OpenAIResponsesModelGateway,
    build_openai_request_payload,
    parse_openai_structured_response,
)
```

Update `__all__`:

```python
"OpenAIModelGatewayConfig",
"OpenAIResponsesModelGateway",
"build_openai_request_payload",
"parse_openai_structured_response",
```

- [ ] **Step 5: Add optional OpenAI dependency extra**

Update `pyproject.toml`:

```toml
[project.optional-dependencies]
dev = [
  "pytest>=8,<9",
]
openai = [
  "openai>=1.0,<3",
]
```

- [ ] **Step 6: Run SDK and live smoke default-skip tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_public_api_and_config.py::test_public_sdk_exports_supported_names tests/test_openai_live.py -q -p no:cacheprovider
```

Expected: public export test passes; live smoke is skipped unless environment variables are set.

- [ ] **Step 7: Commit Task 3**

Run:

```bash
git add bayesprobe/__init__.py pyproject.toml tests/test_public_api_and_config.py tests/test_openai_live.py
git commit -m "feat: expose openai gateway sdk surface"
```

### Task 4: Architecture Docs And Final Regression

**Files:**
- Modify: `docs/ARCHITECTURE.md`

**Interfaces:**
- Consumes all previous tasks.
- Produces docs and final verification that OpenAI adapter v0.1 is implemented without changing core BayesProbe semantics.

- [ ] **Step 1: Update architecture status**

In `docs/ARCHITECTURE.md`, change:

```markdown
| Model gateway | Partial | Structured seam and scripted/deterministic adapters exist; no real provider. |
```

to:

```markdown
| Model gateway | Good MVP | Structured seam plus deterministic, scripted, and OpenAI Responses adapters exist. Provider observability remains future work. |
```

In Phase 2, change:

```markdown
Status: prompt/response metadata contract implemented as MVP; provider adapter remains future work.
```

to:

```markdown
Status: OpenAI Responses adapter implemented as v0.1; broader provider registry and provider observability remain future work.
```

Add a short bullet under `ModelGateway`:

```markdown
- `OpenAIResponsesModelGateway` provides the first real provider-backed adapter while preserving the same structured output validation path.
```

- [ ] **Step 2: Run full regression**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
```

Expected: all tests pass; live OpenAI smoke is skipped by default.

- [ ] **Step 3: Check no network defaults**

Run:

```bash
BAYESPROBE_RUN_OPENAI_LIVE=0 PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_openai_live.py -q -p no:cacheprovider
```

Expected: one skipped test, no network call.

- [ ] **Step 4: Check whitespace and OpenAI references**

Run:

```bash
git diff --check
rg -n "OpenAIResponsesModelGateway|OpenAIModelGatewayConfig|build_openai_request_payload|parse_openai_structured_response|kind\": \"openai\"|BAYESPROBE_RUN_OPENAI_LIVE" bayesprobe tests docs pyproject.toml
```

Expected:

- no `git diff --check` output;
- OpenAI symbols appear in adapter, SDK exports, config/factory tests, live smoke, and architecture docs;
- no API key literal other than environment variable name `OPENAI_API_KEY`.

- [ ] **Step 5: Commit docs and final verification**

Run:

```bash
git add docs/ARCHITECTURE.md
git commit -m "docs: mark openai gateway implemented"
```

- [ ] **Step 6: Review branch status**

Run:

```bash
git status --short
git log --oneline -8
```

Expected:

- working tree is clean;
- recent commits include OpenAI adapter, config wiring, SDK surface, and architecture status.

## Self-Review Checklist

- Spec coverage: tasks cover OpenAI config, required model, no default model in gateway config, payload assembly, Structured Outputs schema, fake client tests, response parsing, provider exception propagation, JSON config, public SDK exports, optional dependency, opt-in live smoke, architecture docs, and full regression.
- Placeholder scan: this plan contains no unresolved task bodies or vague implementation instructions.
- Type consistency: `OpenAIModelGatewayConfig`, `OpenAIResponsesModelGateway`, `build_openai_request_payload`, `parse_openai_structured_response`, `ModelGatewayConfig.model`, `api_key_env`, `timeout_seconds`, and `max_output_tokens` are used consistently.
- Scope check: the plan does not add a provider registry, provider retries, cost/latency accounting, prompt registry, model-call ledger record, posterior changes, projection changes, or default network tests.
