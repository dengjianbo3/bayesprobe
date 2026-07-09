# Autonomous WebUI v0.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local WebUI and JSON API that runs BayesProbe autonomous questions, supports deterministic and OpenAI Responses provider configuration, and displays final answer plus belief-revision trace.

**Architecture:** The WebUI is an observation and execution surface above `AutonomousQuestionRunner`; it must not change `BayesProbeCore`, evidence integration, posterior update rules, or probe control flow. The backend owns provider calls and request-scoped secrets, serializes existing domain results into UI JSON, and serves plain static assets. OpenAI Responses remains a `ModelGateway` adapter; Chat Completions is reserved but returns an explicit unsupported-provider error in v0.1.

**Tech Stack:** Python 3.11+, stdlib `http.server`, stdlib `json`, existing Pydantic models, existing dataclasses, plain HTML/CSS/JavaScript, optional `openai>=1.0,<3` extra.

## Global Constraints

- WebUI/API is local-only and must not be treated as a hosted multi-user service.
- API keys are request-scoped only and must not be written to JSON config, artifacts, logs, ledger records, static assets, browser local storage, or JSON error responses.
- Deterministic mode must run without network access or an API key.
- `openai_responses` is the only provider-backed protocol implemented in v0.1.
- `openai_chat_completions` must return a clear unsupported-provider error in v0.1.
- Do not modify `BayesProbeCore`, evidence integration rules, posterior update rules, or probe control flow.
- Do not add a web framework dependency.
- Static UI assets must be actual app screens, not a landing page.
- UI must not use browser local storage for secrets.
- Full repository tests must pass before completion.

---

## File Structure

- Modify `bayesprobe/openai_gateway.py`: add `base_url` validation and request-scoped API key support to the existing Responses adapter.
- Modify `bayesprobe/model_gateway.py`: add `base_url` to `ModelGatewayConfig` and thread it into `OpenAIModelGatewayConfig`.
- Modify `bayesprobe/config.py`: parse and validate optional `base_url` for persisted `kind="openai"` configs while continuing to reject raw API key config.
- Modify `bayesprobe/experiment_artifacts.py`: include sanitized `base_url` in model gateway snapshots and continue excluding raw keys.
- Modify `bayesprobe/__init__.py`: no new public exports expected unless implementation introduces a public helper.
- Create `bayesprobe/webui.py`: local HTTP server, request validation, autonomous run execution, trace serialization, static serving, CLI entrypoint through `python -m bayesprobe.webui`.
- Create `bayesprobe/webui_static/index.html`: operational workbench screen.
- Create `bayesprobe/webui_static/styles.css`: responsive app styling.
- Create `bayesprobe/webui_static/app.js`: form handling, API call, trace rendering.
- Create `tests/test_webui.py`: backend API, serialization, static serving, secret redaction tests.
- Modify `tests/test_openai_gateway.py`: base URL and request-scoped key tests.
- Modify `tests/test_public_api_and_config.py`: persisted config parsing and raw key rejection tests.
- Modify `tests/test_experiment_artifacts.py`: sanitized `base_url` snapshot coverage.
- Modify `docs/ARCHITECTURE.md`: record WebUI as observation surface.

---

### Task 1: OpenAI Responses Adapter Base URL and Request-Scoped Key

**Files:**
- Modify: `bayesprobe/openai_gateway.py`
- Modify: `bayesprobe/model_gateway.py`
- Modify: `bayesprobe/config.py`
- Modify: `bayesprobe/experiment_artifacts.py`
- Test: `tests/test_openai_gateway.py`
- Test: `tests/test_public_api_and_config.py`
- Test: `tests/test_experiment_artifacts.py`

**Interfaces:**
- Consumes: `OpenAIModelGatewayConfig(model, api_key_env, timeout_seconds, max_output_tokens)` and `OpenAIResponsesModelGateway(config, client=None)`.
- Produces:
  - `OpenAIModelGatewayConfig(..., base_url: str | None = None)`.
  - `OpenAIResponsesModelGateway(config, client=None, api_key: str | None = None)`.
  - `ModelGatewayConfig.base_url: str | None`.
  - persisted experiment configs still use `api_key_env`, never raw `api_key`.

- [ ] **Step 1: Write failing OpenAI adapter tests**

Add to `tests/test_openai_gateway.py`:

```python
def test_openai_model_gateway_config_accepts_base_url():
    config = OpenAIModelGatewayConfig(
        model="gpt-5.5",
        base_url="https://provider.example/v1",
    )

    assert config.base_url == "https://provider.example/v1"


@pytest.mark.parametrize(
    ("base_url", "expected_message"),
    [
        ("", "openai model gateway base_url must not be empty"),
        ("   ", "openai model gateway base_url must not be empty"),
        (1, "openai model gateway base_url must be a string"),
    ],
)
def test_openai_model_gateway_config_rejects_invalid_base_url(
    base_url,
    expected_message,
):
    with pytest.raises(ValueError, match=expected_message):
        OpenAIModelGatewayConfig(model="gpt-5.5", base_url=base_url)
```

Append this fake OpenAI class near the existing fake client tests:

```python
class RecordingOpenAI:
    created_with = []

    def __init__(self, **kwargs):
        self.__class__.created_with.append(kwargs)
        self.responses = FakeResponses(response=json.dumps(valid_payload()))
```

Add:

```python
def test_openai_responses_model_gateway_uses_request_scoped_key_and_base_url(
    monkeypatch,
):
    import bayesprobe.openai_gateway as openai_gateway

    RecordingOpenAI.created_with = []
    monkeypatch.setattr(openai_gateway, "OpenAI", RecordingOpenAI, raising=False)
    gateway = OpenAIResponsesModelGateway(
        config=OpenAIModelGatewayConfig(
            model="gpt-5.5",
            base_url="https://provider.example/v1",
            timeout_seconds=12.5,
        ),
        api_key="sk-request-scoped",
    )

    result = gateway.complete_structured(make_judge_request())

    assert result == valid_payload()
    assert RecordingOpenAI.created_with == [
        {
            "api_key": "sk-request-scoped",
            "timeout": 12.5,
            "base_url": "https://provider.example/v1",
        }
    ]
```

- [ ] **Step 2: Write failing persisted config and artifact tests**

Add to `tests/test_public_api_and_config.py` near the OpenAI config parser tests:

```python
def test_experiment_config_from_mapping_parses_openai_base_url(tmp_path: Path):
    dataset_path = tmp_path / "dataset.json"
    report_path = tmp_path / "report.json"

    config = experiment_config_from_mapping(
        {
            "dataset_path": str(dataset_path),
            "report_path": str(report_path),
            "model_gateway": {
                "kind": "openai",
                "model": "gpt-5.5",
                "api_key_env": "BAYESPROBE_TEST_OPENAI_KEY",
                "base_url": "https://provider.example/v1",
            },
        }
    )

    assert config.model_gateway is not None
    assert config.model_gateway.base_url == "https://provider.example/v1"
```

Add to the invalid config parameter list in `tests/test_public_api_and_config.py`:

```python
(
    "openai_raw_api_key_rejected.json",
    {
        "dataset_path": "dataset.json",
        "report_path": "report.json",
        "model_gateway": {
            "kind": "openai",
            "model": "gpt-5.5",
            "api_key": "sk-not-allowed",
        },
    },
    "openai model gateway api_key is not allowed in experiment config",
),
(
    "openai_invalid_base_url.json",
    {
        "dataset_path": "dataset.json",
        "report_path": "report.json",
        "model_gateway": {
            "kind": "openai",
            "model": "gpt-5.5",
            "base_url": "",
        },
    },
    "openai model gateway base_url must not be empty",
),
```

Add to `tests/test_experiment_artifacts.py`:

```python
def test_artifact_snapshot_includes_openai_base_url_without_raw_api_key(tmp_path: Path):
    dataset = load_benchmark_dataset(FIXTURE_PATH)
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps({"results": [], "sample_count": 0}),
        encoding="utf-8",
    )
    config = ExperimentRunConfig(
        dataset_path=FIXTURE_PATH,
        report_path=report_path,
        artifact_dir=tmp_path / "artifact",
        model_gateway={
            "kind": "openai",
            "model": "gpt-5.5",
            "api_key_env": "OPENAI_API_KEY",
            "base_url": "https://provider.example/v1",
        },
        metadata={"api_key": "sk-secret"},
    )

    bundle = write_experiment_artifact_bundle(
        artifact_dir=tmp_path / "artifact",
        config=config,
        dataset=dataset,
        report_path=report_path,
        ledger_path=None,
        sample_count=0,
    )

    manifest_text = bundle.manifest_path.read_text(encoding="utf-8")
    snapshot = json.loads(bundle.config_snapshot_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_text)
    assert snapshot["model_gateway"]["base_url"] == "https://provider.example/v1"
    assert manifest["model_gateway"]["base_url"] == "https://provider.example/v1"
    assert "sk-secret" not in manifest_text
    assert "sk-secret" not in bundle.config_snapshot_path.read_text(encoding="utf-8")
```

- [ ] **Step 3: Verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tests/test_openai_gateway.py::test_openai_model_gateway_config_accepts_base_url \
  tests/test_openai_gateway.py::test_openai_responses_model_gateway_uses_request_scoped_key_and_base_url \
  tests/test_public_api_and_config.py::test_experiment_config_from_mapping_parses_openai_base_url \
  tests/test_experiment_artifacts.py::test_artifact_snapshot_includes_openai_base_url_without_raw_api_key \
  -q -p no:cacheprovider
```

Expected: failures because `base_url`, request-scoped key, and raw-key config rejection are not implemented.

- [ ] **Step 4: Implement adapter and config support**

In `bayesprobe/openai_gateway.py`, add a module-level fallback binding before `_build_default_openai_client`:

```python
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
```

Update `OpenAIModelGatewayConfig`:

```python
@dataclass(frozen=True)
class OpenAIModelGatewayConfig:
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    timeout_seconds: float = 30.0
    max_output_tokens: int | None = None
    base_url: str | None = None
```

Add validation inside `__post_init__`:

```python
if self.base_url is not None:
    if not isinstance(self.base_url, str):
        raise ValueError("openai model gateway base_url must be a string")
    if not self.base_url.strip():
        raise ValueError("openai model gateway base_url must not be empty")
    object.__setattr__(self, "base_url", self.base_url.strip())
```

Update `OpenAIResponsesModelGateway.__init__`:

```python
def __init__(
    self,
    *,
    config: OpenAIModelGatewayConfig,
    client: Any | None = None,
    api_key: str | None = None,
) -> None:
    self.config = config
    self._client = client
    self._api_key = _optional_request_api_key(api_key)
```

Add:

```python
def _optional_request_api_key(api_key: str | None) -> str | None:
    if api_key is None:
        return None
    if not isinstance(api_key, str):
        raise ValueError("openai request api_key must be a string")
    if not api_key.strip():
        raise ValueError("openai request api_key must not be empty")
    return api_key.strip()
```

Update `_client_for_request`:

```python
self._client = _build_default_openai_client(self.config, api_key=self._api_key)
```

Update `_build_default_openai_client`:

```python
def _build_default_openai_client(
    config: OpenAIModelGatewayConfig, *, api_key: str | None = None
) -> Any:
    resolved_api_key = api_key or os.environ.get(config.api_key_env)
    if not resolved_api_key:
        raise RuntimeError(
            f"OpenAI API key environment variable {config.api_key_env} is not set"
        )
    if OpenAI is None:
        raise RuntimeError(
            "OpenAI Python package is required for OpenAIResponsesModelGateway. "
            "Install bayesprobe[openai] or install openai."
        )
    kwargs: dict[str, Any] = {
        "api_key": resolved_api_key,
        "timeout": config.timeout_seconds,
    }
    if config.base_url is not None:
        kwargs["base_url"] = config.base_url
    return OpenAI(**kwargs)
```

In `bayesprobe/model_gateway.py`, update `ModelGatewayConfig`:

```python
base_url: str | None = None
```

Thread `base_url` through `build_model_gateway(...)` and `_model_gateway_config_from_input(...)`:

```python
base_url = config.get("base_url")
if base_url is not None and not isinstance(base_url, str):
    raise ValueError("openai model gateway base_url must be a string")
...
base_url=base_url,
```

```python
OpenAIModelGatewayConfig(
    model=gateway_config.model,
    api_key_env=gateway_config.api_key_env,
    timeout_seconds=gateway_config.timeout_seconds,
    max_output_tokens=gateway_config.max_output_tokens,
    base_url=gateway_config.base_url,
)
```

In `bayesprobe/config.py`, reject raw keys and validate base URL in `_optional_model_gateway_config(...)`:

```python
if kind == "openai" and "api_key" in value:
    raise ValueError("openai model gateway api_key is not allowed in experiment config")
base_url = value.get("base_url")
```

After `validated_openai_config`, assign:

```python
base_url = validated_openai_config.base_url
```

Return `ModelGatewayConfig(..., base_url=base_url)`.

In `bayesprobe/experiment_artifacts.py`, include sanitized `base_url` in both `ModelGatewayConfig` and mapping snapshots:

```python
"base_url": config.base_url,
```

and:

```python
"base_url": config.get("base_url"),
```

- [ ] **Step 5: Verify GREEN for focused adapter/config tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tests/test_openai_gateway.py \
  tests/test_public_api_and_config.py \
  tests/test_experiment_artifacts.py \
  -q -p no:cacheprovider
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```bash
git add bayesprobe/openai_gateway.py bayesprobe/model_gateway.py bayesprobe/config.py bayesprobe/experiment_artifacts.py tests/test_openai_gateway.py tests/test_public_api_and_config.py tests/test_experiment_artifacts.py
git commit -m "feat: support openai responses base url"
```

---

### Task 2: WebUI Backend Request Validation and Deterministic Run API

**Files:**
- Create: `bayesprobe/webui.py`
- Create: `tests/test_webui.py`

**Interfaces:**
- Consumes:
  - `BayesProbeCore`
  - `AutonomousQuestionRunConfig`
  - `AutonomousQuestionRunner`
  - `InitializeRunInput`
  - deterministic default model gateway/probe gateway.
- Produces:
  - `handle_autonomous_run_request(payload: Mapping[str, Any], *, client_factory: Callable[..., Any] | None = None) -> tuple[int, dict[str, Any]]`
  - `serialize_autonomous_run_result(result: AutonomousQuestionRunResult) -> dict[str, Any]`
  - `create_handler_class() -> type[BaseHTTPRequestHandler]`
  - `main(argv: list[str] | None = None) -> int`

- [ ] **Step 1: Write failing deterministic API tests**

Create `tests/test_webui.py`:

```python
import json
from http.client import HTTPConnection
from threading import Thread

import pytest

from bayesprobe.webui import (
    create_handler_class,
    handle_autonomous_run_request,
)


def test_webui_deterministic_autonomous_run_returns_trace():
    status, payload = handle_autonomous_run_request(
        {
            "question": "Does the autonomous WebUI path expose trace state?",
            "context": "SUPPORTS: local deterministic run should favor H1.",
            "provider": {"kind": "deterministic"},
            "runner": {"max_cycles": 1, "max_probes_per_cycle": 1},
        }
    )

    assert status == 200
    assert payload["run_id"].startswith("webui_")
    assert payload["stop_reason"] == "max_cycles"
    assert payload["final_answer"]["current_best_hypothesis"] == "H1"
    assert payload["initial_belief_state"]["cycle_id"] == "cycle_0"
    assert payload["final_belief_state"]["cycle_index"] == 1
    assert len(payload["cycles"]) == 1
    cycle = payload["cycles"][0]
    assert cycle["signal_shape"] == "active_only"
    assert cycle["probes"]
    assert cycle["signals"]
    assert cycle["evidence_events"]
    assert cycle["belief_updates"]
    assert cycle["answer_projection"]["current_best_hypothesis"] == "H1"
```

Add validation tests:

```python
@pytest.mark.parametrize(
    ("payload", "expected_message"),
    [
        ({}, "question must not be empty"),
        ({"question": ""}, "question must not be empty"),
        (
            {"question": "Q", "runner": {"max_cycles": 0}},
            "max_cycles must be at least 1",
        ),
        (
            {"question": "Q", "provider": {"kind": "openai_chat_completions"}},
            "provider kind openai_chat_completions is not supported in v0.1",
        ),
    ],
)
def test_webui_autonomous_run_rejects_invalid_payloads(payload, expected_message):
    status, response = handle_autonomous_run_request(payload)

    assert status == 400
    assert response["error"]["message"] == expected_message
```

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_webui.py -q -p no:cacheprovider
```

Expected: import failure because `bayesprobe.webui` does not exist.

- [ ] **Step 3: Implement backend validation, runner execution, and serialization**

Create `bayesprobe/webui.py` with:

```python
from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, is_dataclass
from enum import Enum
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from bayesprobe.core import BayesProbeCore
from bayesprobe.initialization import InitializeRunInput
from bayesprobe.model_gateway import DeterministicModelGateway, ModelGateway
from bayesprobe.question_runner import (
    AutonomousQuestionRunConfig,
    AutonomousQuestionRunner,
    AutonomousQuestionRunResult,
)

STATIC_DIR = Path(__file__).with_name("webui_static")
SUPPORTED_PROVIDER_KINDS = {"deterministic", "openai_responses"}
RESERVED_PROVIDER_KINDS = {"openai_chat_completions"}


class WebUIError(Exception):
    status_code = HTTPStatus.BAD_REQUEST
    error_type = "validation_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class UnsupportedProviderError(WebUIError):
    error_type = "unsupported_provider"


class ProviderError(WebUIError):
    status_code = HTTPStatus.BAD_GATEWAY
    error_type = "provider_error"


def handle_autonomous_run_request(
    payload: Mapping[str, Any],
    *,
    client_factory: Callable[..., Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    try:
        request = _parse_autonomous_request(payload)
        gateway = _build_webui_model_gateway(
            request["provider"], client_factory=client_factory
        )
        core = BayesProbeCore(model_gateway=gateway)
        runner = AutonomousQuestionRunner(
            core=core,
            config=request["runner_config"],
        )
        run_id = _webui_run_id()
        result = runner.run_question(
            InitializeRunInput(
                run_id=run_id,
                problem=request["question"],
                context=request["context"],
            )
        )
        return HTTPStatus.OK, serialize_autonomous_run_result(result)
    except WebUIError as error:
        return int(error.status_code), _error_payload(error.error_type, error.message)
    except (RuntimeError, OSError) as error:
        return int(HTTPStatus.BAD_GATEWAY), _error_payload(
            "provider_error", _sanitize_error_message(str(error))
        )
```

Add request parsing:

```python
def _parse_autonomous_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise WebUIError("request payload must be an object")
    question = _required_nonempty_string(payload.get("question"), "question")
    context = _optional_string(payload.get("context"), "context", default="")
    provider = payload.get("provider", {"kind": "deterministic"})
    if provider is None:
        provider = {"kind": "deterministic"}
    if not isinstance(provider, Mapping):
        raise WebUIError("provider must be an object")
    runner_payload = payload.get("runner", {})
    if runner_payload is None:
        runner_payload = {}
    if not isinstance(runner_payload, Mapping):
        raise WebUIError("runner must be an object")
    return {
        "question": question,
        "context": context,
        "provider": dict(provider),
        "runner_config": _runner_config_from_payload(runner_payload),
    }
```

Add runner parsing:

```python
def _runner_config_from_payload(payload: Mapping[str, Any]) -> AutonomousQuestionRunConfig:
    return AutonomousQuestionRunConfig(
        max_cycles=_optional_int(payload, "max_cycles", default=3),
        max_probes_per_cycle=_optional_int(payload, "max_probes_per_cycle", default=2),
        stop_on_no_probes=_optional_bool(payload, "stop_on_no_probes", default=True),
        confidence_threshold=_optional_float(payload, "confidence_threshold"),
        posterior_delta_threshold=_optional_float(payload, "posterior_delta_threshold"),
    )
```

Add deterministic provider builder for now:

```python
def _build_webui_model_gateway(
    provider: Mapping[str, Any],
    *,
    client_factory: Callable[..., Any] | None,
) -> ModelGateway:
    kind = _optional_string(provider.get("kind"), "provider.kind", default="deterministic")
    if kind in RESERVED_PROVIDER_KINDS:
        raise UnsupportedProviderError(
            f"provider kind {kind} is not supported in v0.1"
        )
    if kind == "deterministic":
        return DeterministicModelGateway()
    if kind == "openai_responses":
        raise UnsupportedProviderError("provider kind openai_responses is not wired yet")
    raise UnsupportedProviderError(f"unsupported provider kind: {kind}")
```

Add serialization:

```python
def serialize_autonomous_run_result(result: AutonomousQuestionRunResult) -> dict[str, Any]:
    return {
        "run_id": result.run.run_id,
        "stop_reason": result.stop_reason.value,
        "final_answer": _dump_domain(result.final_answer_projection),
        "initial_belief_state": _dump_domain(result.initial_belief_state),
        "final_belief_state": _dump_domain(result.final_belief_state),
        "cycles": [
            {
                "cycle_id": cycle.cycle.cycle_id,
                "signal_shape": cycle.cycle.signal_shape.value,
                "cycle": _dump_domain(cycle.cycle),
                "probes": _dump_domain(cycle.probe_set.probes),
                "signals": _dump_domain(cycle.signals),
                "evidence_events": _dump_domain(cycle.evidence_events),
                "belief_updates": _dump_domain(cycle.belief_updates),
                "hypothesis_evolutions": _dump_domain(cycle.hypothesis_evolutions),
                "answer_projection": _dump_domain(cycle.answer_projection),
            }
            for cycle in result.cycle_results
        ],
    }
```

Add `_dump_domain`, scalar validators, `_webui_run_id`, `_error_payload`, and `_sanitize_error_message`:

```python
def _dump_domain(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value):
        return _dump_domain(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _dump_domain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_dump_domain(item) for item in value]
    return value
```

```python
def _webui_run_id() -> str:
    return f"webui_{int(time.time() * 1000)}"


def _error_payload(error_type: str, message: str) -> dict[str, Any]:
    return {"error": {"type": error_type, "message": _sanitize_error_message(message)}}


def _sanitize_error_message(message: str) -> str:
    return re.sub(r"sk-[A-Za-z0-9_\\-]+", "sk-redacted", message)
```

Implement `_required_nonempty_string`, `_optional_string`, `_optional_int`, `_optional_bool`, and `_optional_float` with exact messages used in tests.

- [ ] **Step 4: Verify GREEN for deterministic backend**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_webui.py -q -p no:cacheprovider
```

Expected: tests pass.

- [ ] **Step 5: Commit**

```bash
git add bayesprobe/webui.py tests/test_webui.py
git commit -m "feat: add autonomous webui backend"
```

---

### Task 3: Local HTTP Server, Static Serving, and OpenAI Responses Provider Wiring

**Files:**
- Modify: `bayesprobe/webui.py`
- Modify: `tests/test_webui.py`

**Interfaces:**
- Consumes:
  - `OpenAIModelGatewayConfig`
  - `OpenAIResponsesModelGateway(config, client=None, api_key=None)`
  - `handle_autonomous_run_request(...)` from Task 2.
- Produces:
  - working `GET /`, `GET /styles.css`, `GET /app.js`.
  - working `POST /api/runs/autonomous`.
  - request-scoped `openai_responses` provider path.
  - sanitized HTTP errors.

- [ ] **Step 1: Write failing HTTP/static and OpenAI provider tests**

Append the missing server import to `tests/test_webui.py`:

```python
from http.server import ThreadingHTTPServer
```

Then append:

```python
def serve_test_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler_class())
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def request_json(server, payload):
    conn = HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
    conn.request(
        "POST",
        "/api/runs/autonomous",
        body=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )
    response = conn.getresponse()
    data = json.loads(response.read().decode("utf-8"))
    conn.close()
    return response.status, data


def test_webui_http_server_serves_static_index():
    server, thread = serve_test_server()
    try:
        conn = HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        conn.request("GET", "/")
        response = conn.getresponse()
        body = response.read().decode("utf-8")
        conn.close()
    finally:
        server.shutdown()
        server.server_close()

    assert response.status == 200
    assert "BayesProbe" in body
```

Add:

```python
class FakeWebUIResponses:
    def __init__(self):
        self.calls = []

    def create(self, **payload):
        self.calls.append(payload)
        return json.dumps(
            {
                "evidence_type": "supporting",
                "likelihoods": {
                    "H1": "moderately_confirming",
                    "H2": "moderately_disconfirming",
                },
                "interpretation": "WebUI fake OpenAI response.",
                "quality_overrides": {},
            }
        )


class FakeWebUIOpenAI:
    created_with = []

    def __init__(self, **kwargs):
        self.__class__.created_with.append(kwargs)
        self.responses = FakeWebUIResponses()
```

Add:

```python
def test_webui_openai_responses_provider_uses_request_key_and_redacts_response():
    FakeWebUIOpenAI.created_with = []

    status, payload = handle_autonomous_run_request(
        {
            "question": "Can the WebUI use a provider-backed evidence judgment?",
            "provider": {
                "kind": "openai_responses",
                "api_key": "sk-webui-secret",
                "base_url": "https://provider.example/v1",
                "model": "gpt-5.5",
                "timeout_seconds": 11,
                "max_output_tokens": 128,
            },
            "runner": {"max_cycles": 1, "max_probes_per_cycle": 1},
        },
        client_factory=FakeWebUIOpenAI,
    )

    assert status == 200
    assert FakeWebUIOpenAI.created_with == [
        {
            "api_key": "sk-webui-secret",
            "timeout": 11,
            "base_url": "https://provider.example/v1",
        }
    ]
    assert "sk-webui-secret" not in json.dumps(payload)
    assert payload["cycles"][0]["evidence_events"][0]["model_trace"]["adapter_kind"] == "openai"
```

Add provider error test:

```python
class FailingWebUIOpenAI:
    def __init__(self, **kwargs):
        self.responses = self

    def create(self, **payload):
        raise RuntimeError("provider rejected key sk-webui-secret")


def test_webui_provider_errors_are_sanitized():
    status, payload = handle_autonomous_run_request(
        {
            "question": "Will provider errors leak secrets?",
            "provider": {
                "kind": "openai_responses",
                "api_key": "sk-webui-secret",
                "model": "gpt-5.5",
            },
        },
        client_factory=FailingWebUIOpenAI,
    )

    assert status == 502
    assert payload["error"]["type"] == "provider_error"
    assert "sk-webui-secret" not in json.dumps(payload)
```

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_webui.py -q -p no:cacheprovider
```

Expected: failures because static files and `openai_responses` provider path are not implemented.

- [ ] **Step 3: Implement OpenAI Responses provider builder**

In `bayesprobe/webui.py`, import:

```python
from bayesprobe.openai_gateway import OpenAIModelGatewayConfig, OpenAIResponsesModelGateway
```

Update `_build_webui_model_gateway(...)`:

```python
if kind == "openai_responses":
    model = _required_nonempty_string(provider.get("model"), "provider.model")
    api_key = _required_nonempty_string(provider.get("api_key"), "provider.api_key")
    config = OpenAIModelGatewayConfig(
        model=model,
        base_url=_optional_string(provider.get("base_url"), "provider.base_url"),
        timeout_seconds=_optional_number_from_mapping(
            provider, "timeout_seconds", default=30.0
        ),
        max_output_tokens=_optional_int_or_none(provider, "max_output_tokens"),
    )
    return OpenAIResponsesModelGateway(
        config=config,
        api_key=api_key,
        client=client_factory(**_openai_client_kwargs(config, api_key))
        if client_factory is not None
        else None,
    )
```

Add helpers:

```python
def _openai_client_kwargs(config: OpenAIModelGatewayConfig, api_key: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"api_key": api_key, "timeout": config.timeout_seconds}
    if config.base_url is not None:
        kwargs["base_url"] = config.base_url
    return kwargs
```

```python
def _optional_int_or_none(payload: Mapping[str, Any], field_name: str) -> int | None:
    if field_name not in payload or payload[field_name] is None:
        return None
    value = payload[field_name]
    if type(value) is not int:
        raise WebUIError(f"{field_name} must be an integer")
    if value < 1:
        raise WebUIError(f"{field_name} must be positive")
    return value


def _optional_number_from_mapping(
    payload: Mapping[str, Any], field_name: str, *, default: float
) -> float:
    if field_name not in payload or payload[field_name] is None:
        return default
    value = payload[field_name]
    if type(value) not in (int, float):
        raise WebUIError(f"{field_name} must be a number")
    if value <= 0:
        raise WebUIError(f"{field_name} must be positive")
    return float(value)
```

- [ ] **Step 4: Implement HTTP handler and static serving**

In `bayesprobe/webui.py`, add:

```python
def create_handler_class() -> type[BaseHTTPRequestHandler]:
    class BayesProbeWebUIHandler(BaseHTTPRequestHandler):
        server_version = "BayesProbeWebUI/0.1"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            if self.path in {"/", "/index.html"}:
                self._send_static("index.html", "text/html; charset=utf-8")
                return
            if self.path == "/styles.css":
                self._send_static("styles.css", "text/css; charset=utf-8")
                return
            if self.path == "/app.js":
                self._send_static("app.js", "text/javascript; charset=utf-8")
                return
            self._send_json(HTTPStatus.NOT_FOUND, _error_payload("not_found", "not found"))

        def do_POST(self) -> None:
            if self.path != "/api/runs/autonomous":
                self._send_json(HTTPStatus.NOT_FOUND, _error_payload("not_found", "not found"))
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(length).decode("utf-8")
                payload = json.loads(raw_body or "{}")
            except json.JSONDecodeError:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    _error_payload("invalid_json", "request body must be valid JSON"),
                )
                return
            status, response = handle_autonomous_run_request(payload)
            self._send_json(status, response)

        def _send_static(self, filename: str, content_type: str) -> None:
            path = STATIC_DIR / filename
            if not path.exists():
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    _error_payload("not_found", "static asset not found"),
                )
                return
            body = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: int, payload: Mapping[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return BayesProbeWebUIHandler
```

Add a minimal `main(...)`:

```python
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the BayesProbe local WebUI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), create_handler_class())
    host, port = server.server_address
    print(f"BayesProbe WebUI running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Verify GREEN for backend HTTP/provider tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_webui.py tests/test_openai_gateway.py -q -p no:cacheprovider
```

Expected: tests pass.

- [ ] **Step 6: Commit**

```bash
git add bayesprobe/webui.py tests/test_webui.py
git commit -m "feat: wire webui openai responses provider"
```

---

### Task 4: Static WebUI Workbench

**Files:**
- Create: `bayesprobe/webui_static/index.html`
- Create: `bayesprobe/webui_static/styles.css`
- Create: `bayesprobe/webui_static/app.js`
- Modify: `tests/test_webui.py`

**Interfaces:**
- Consumes: `POST /api/runs/autonomous` JSON API.
- Produces: a responsive local operational workbench with:
  - provider controls;
  - runner controls;
  - question/context form;
  - final answer summary;
  - belief state table;
  - cycle trace details.

- [ ] **Step 1: Write failing static asset content tests**

Add to `tests/test_webui.py`:

```python
from pathlib import Path


STATIC_DIR = Path("bayesprobe/webui_static")


def test_webui_static_assets_define_operational_workbench():
    index = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    styles = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")
    script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert "BayesProbe" in index
    assert "provider-kind" in index
    assert "api-key" in index
    assert "base-url" in index
    assert "model-name" in index
    assert "max-cycles" in index
    assert "trace-pane" in index
    assert "localStorage" not in script
    assert "fetch('/api/runs/autonomous'" in script
    assert ".trace-item" in styles
    assert "@media" in styles
```

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_webui.py::test_webui_static_assets_define_operational_workbench -q -p no:cacheprovider
```

Expected: failure because static assets do not exist.

- [ ] **Step 3: Create `index.html`**

Create `bayesprobe/webui_static/index.html`:

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>BayesProbe Autonomous Workbench</title>
    <link rel="stylesheet" href="/styles.css" />
  </head>
  <body>
    <main class="shell">
      <aside class="control-rail" aria-label="Run configuration">
        <div class="brand-block">
          <span class="mark">BP</span>
          <div>
            <h1>BayesProbe</h1>
            <p>Autonomous workbench</p>
          </div>
        </div>

        <section class="panel">
          <h2>Provider</h2>
          <label>
            Protocol
            <select id="provider-kind">
              <option value="deterministic">Deterministic</option>
              <option value="openai_responses">OpenAI Responses</option>
              <option value="openai_chat_completions">Chat Completions</option>
            </select>
          </label>
          <label>
            API key
            <input id="api-key" type="password" autocomplete="off" spellcheck="false" />
          </label>
          <label>
            Base URL
            <input id="base-url" type="url" placeholder="https://api.openai.com/v1" spellcheck="false" />
          </label>
          <label>
            Model
            <input id="model-name" type="text" placeholder="gpt-5.5" spellcheck="false" />
          </label>
          <label>
            Timeout seconds
            <input id="timeout-seconds" type="number" min="1" value="30" />
          </label>
          <label>
            Max output tokens
            <input id="max-output-tokens" type="number" min="1" value="512" />
          </label>
        </section>

        <section class="panel">
          <h2>Autonomy</h2>
          <label>
            Max cycles
            <input id="max-cycles" type="number" min="1" value="2" />
          </label>
          <label>
            Max probes per cycle
            <input id="max-probes" type="number" min="1" value="2" />
          </label>
          <label>
            Confidence threshold
            <input id="confidence-threshold" type="number" min="0" max="1" step="0.01" placeholder="optional" />
          </label>
          <label>
            Posterior stability
            <input id="posterior-delta-threshold" type="number" min="0" step="0.01" placeholder="optional" />
          </label>
          <label class="toggle-row">
            <input id="stop-on-no-probes" type="checkbox" checked />
            <span>Stop when no probes remain</span>
          </label>
        </section>
      </aside>

      <section class="workspace">
        <form id="run-form" class="question-panel">
          <label>
            Question
            <textarea id="question" required>Does the current evidence support H1 or H2?</textarea>
          </label>
          <label>
            Context
            <textarea id="context">SUPPORTS: The local deterministic signal supports H1.</textarea>
          </label>
          <button id="run-button" type="submit">Run autonomous loop</button>
        </form>

        <section id="status-banner" class="status-banner" aria-live="polite"></section>

        <section class="result-grid">
          <article class="result-panel">
            <h2>Answer Projection</h2>
            <div id="answer-panel" class="empty-state">No run yet.</div>
          </article>
          <article class="result-panel">
            <h2>Belief State</h2>
            <div id="belief-panel" class="empty-state">No belief state yet.</div>
          </article>
        </section>

        <section class="trace-section">
          <div class="section-head">
            <h2>Cycle Trace</h2>
            <span id="run-id"></span>
          </div>
          <div id="trace-pane" class="trace-pane"></div>
        </section>
      </section>
    </main>
    <script src="/app.js"></script>
  </body>
</html>
```

- [ ] **Step 4: Create `styles.css`**

Use a restrained analytical workbench style. Create `bayesprobe/webui_static/styles.css` with stable dimensions, no nested cards, no gradient orbs, and no text overflow:

```css
:root {
  color-scheme: dark;
  --bg: #101214;
  --panel: #181b1f;
  --panel-2: #20242a;
  --line: #3a4149;
  --text: #f3efe6;
  --muted: #a9b0b8;
  --accent: #e6c15a;
  --good: #78d191;
  --bad: #e87878;
  --focus: #7db7ff;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  min-height: 100vh;
  background: var(--bg);
  color: var(--text);
  font-family: ui-serif, Georgia, "Times New Roman", serif;
}

button,
input,
select,
textarea {
  font: inherit;
}

.shell {
  display: grid;
  grid-template-columns: minmax(280px, 360px) minmax(0, 1fr);
  min-height: 100vh;
}

.control-rail {
  border-right: 1px solid var(--line);
  background: #14171a;
  padding: 20px;
  overflow-y: auto;
}

.brand-block {
  display: flex;
  gap: 12px;
  align-items: center;
  margin-bottom: 20px;
}

.mark {
  display: inline-grid;
  place-items: center;
  width: 44px;
  height: 44px;
  border: 1px solid var(--accent);
  color: var(--accent);
  font-weight: 700;
}

h1,
h2,
p {
  margin: 0;
}

h1 {
  font-size: 1.45rem;
}

h2 {
  font-size: 0.9rem;
  text-transform: uppercase;
  color: var(--accent);
}

.panel,
.question-panel,
.result-panel,
.trace-section {
  border-top: 1px solid var(--line);
  padding-top: 16px;
}

.panel {
  display: grid;
  gap: 12px;
  margin-top: 18px;
}

label {
  display: grid;
  gap: 6px;
  color: var(--muted);
  font-size: 0.88rem;
}

input,
select,
textarea {
  width: 100%;
  border: 1px solid var(--line);
  background: var(--panel);
  color: var(--text);
  padding: 10px 11px;
  border-radius: 6px;
  min-width: 0;
}

textarea {
  min-height: 104px;
  resize: vertical;
}

input:focus,
select:focus,
textarea:focus {
  outline: 2px solid var(--focus);
  outline-offset: 1px;
}

.toggle-row {
  grid-template-columns: 18px minmax(0, 1fr);
  align-items: center;
}

.workspace {
  padding: 22px;
  overflow: auto;
}

.question-panel {
  display: grid;
  gap: 14px;
}

button {
  justify-self: start;
  border: 1px solid var(--accent);
  background: var(--accent);
  color: #17130a;
  border-radius: 6px;
  padding: 10px 14px;
  cursor: pointer;
  font-weight: 700;
}

button:disabled {
  cursor: wait;
  opacity: 0.65;
}

.status-banner {
  min-height: 32px;
  margin: 16px 0;
  color: var(--muted);
}

.status-banner.error {
  color: var(--bad);
}

.result-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 18px;
}

.result-panel {
  min-height: 180px;
}

.empty-state {
  color: var(--muted);
  padding-top: 12px;
}

.belief-row,
.kv-row {
  display: grid;
  grid-template-columns: minmax(100px, 160px) minmax(0, 1fr);
  gap: 10px;
  padding: 8px 0;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}

.trace-section {
  margin-top: 24px;
}

.section-head {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: baseline;
}

.trace-pane {
  display: grid;
  gap: 14px;
  margin-top: 14px;
}

.trace-item {
  border: 1px solid var(--line);
  background: var(--panel);
  border-radius: 8px;
  padding: 14px;
}

.trace-item summary {
  cursor: pointer;
  color: var(--accent);
  font-weight: 700;
}

pre {
  overflow: auto;
  max-width: 100%;
  background: var(--panel-2);
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 12px;
  color: var(--text);
}

@media (max-width: 860px) {
  .shell {
    grid-template-columns: 1fr;
  }

  .control-rail {
    border-right: 0;
    border-bottom: 1px solid var(--line);
  }

  .result-grid {
    grid-template-columns: 1fr;
  }
}
```

- [ ] **Step 5: Create `app.js`**

Create `bayesprobe/webui_static/app.js`:

```javascript
const form = document.querySelector("#run-form");
const statusBanner = document.querySelector("#status-banner");
const answerPanel = document.querySelector("#answer-panel");
const beliefPanel = document.querySelector("#belief-panel");
const tracePane = document.querySelector("#trace-pane");
const runId = document.querySelector("#run-id");
const runButton = document.querySelector("#run-button");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setStatus("Running autonomous loop...", false);
  runButton.disabled = true;
  try {
    const response = await fetch('/api/runs/autonomous', {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildPayload()),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error?.message || "Run failed");
    }
    renderRun(payload);
    setStatus(`Stopped: ${payload.stop_reason}`, false);
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    runButton.disabled = false;
  }
});

function buildPayload() {
  const providerKind = valueOf("provider-kind");
  const provider = { kind: providerKind };
  if (providerKind === "openai_responses") {
    provider.api_key = valueOf("api-key");
    provider.base_url = valueOf("base-url") || null;
    provider.model = valueOf("model-name");
    provider.timeout_seconds = numberOrNull("timeout-seconds");
    provider.max_output_tokens = numberOrNull("max-output-tokens");
  }
  return {
    question: valueOf("question"),
    context: valueOf("context"),
    provider,
    runner: {
      max_cycles: numberOrNull("max-cycles"),
      max_probes_per_cycle: numberOrNull("max-probes"),
      stop_on_no_probes: document.querySelector("#stop-on-no-probes").checked,
      confidence_threshold: numberOrNull("confidence-threshold"),
      posterior_delta_threshold: numberOrNull("posterior-delta-threshold"),
    },
  };
}

function renderRun(payload) {
  runId.textContent = payload.run_id;
  renderAnswer(payload.final_answer);
  renderBeliefs(payload.final_belief_state);
  tracePane.innerHTML = "";
  for (const cycle of payload.cycles) {
    tracePane.appendChild(renderCycle(cycle));
  }
}

function renderAnswer(answer) {
  if (!answer) {
    answerPanel.textContent = "No answer projection.";
    return;
  }
  answerPanel.innerHTML = "";
  answerPanel.appendChild(kv("Best hypothesis", answer.current_best_hypothesis));
  answerPanel.appendChild(kv("Answer", answer.answer));
  answerPanel.appendChild(kv("Posterior summary", answer.posterior_summary));
  answerPanel.appendChild(kv("Main uncertainty", answer.main_uncertainty));
  answerPanel.appendChild(kv("Weakest assumption", answer.weakest_assumption));
}

function renderBeliefs(beliefState) {
  beliefPanel.innerHTML = "";
  for (const hypothesis of beliefState.hypotheses || []) {
    beliefPanel.appendChild(
      kv(
        hypothesis.id,
        `${formatNumber(hypothesis.posterior)} | ${hypothesis.statement}`
      )
    );
  }
}

function renderCycle(cycle) {
  const details = document.createElement("details");
  details.className = "trace-item";
  details.open = true;
  const summary = document.createElement("summary");
  summary.textContent = `${cycle.cycle_id} (${cycle.signal_shape})`;
  details.appendChild(summary);
  details.appendChild(block("Probes", cycle.probes));
  details.appendChild(block("Signals", cycle.signals));
  details.appendChild(block("Evidence", cycle.evidence_events));
  details.appendChild(block("Belief updates", cycle.belief_updates));
  details.appendChild(block("Hypothesis evolution", cycle.hypothesis_evolutions));
  return details;
}

function block(title, value) {
  const wrapper = document.createElement("section");
  const heading = document.createElement("h3");
  const pre = document.createElement("pre");
  heading.textContent = title;
  pre.textContent = JSON.stringify(value, null, 2);
  wrapper.appendChild(heading);
  wrapper.appendChild(pre);
  return wrapper;
}

function kv(label, value) {
  const row = document.createElement("div");
  row.className = "kv-row";
  const key = document.createElement("strong");
  const val = document.createElement("span");
  key.textContent = label;
  val.textContent = value ?? "";
  row.appendChild(key);
  row.appendChild(val);
  return row;
}

function valueOf(id) {
  return document.querySelector(`#${id}`).value.trim();
}

function numberOrNull(id) {
  const raw = valueOf(id);
  return raw === "" ? null : Number(raw);
}

function formatNumber(value) {
  return typeof value === "number" ? value.toFixed(3) : value;
}

function setStatus(message, isError) {
  statusBanner.textContent = message;
  statusBanner.classList.toggle("error", isError);
}
```

- [ ] **Step 6: Verify GREEN for static tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_webui.py -q -p no:cacheprovider
```

Expected: tests pass.

- [ ] **Step 7: Commit**

```bash
git add bayesprobe/webui_static/index.html bayesprobe/webui_static/styles.css bayesprobe/webui_static/app.js tests/test_webui.py
git commit -m "feat: add autonomous webui static app"
```

---

### Task 5: Docs, Architecture Alignment, and End-to-End Verification

**Files:**
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/superpowers/specs/2026-07-09-autonomous-webui-v0.1-design.md` if status needs updating after implementation
- Test: full repository

**Interfaces:**
- Consumes: completed backend, OpenAI adapter, and static WebUI.
- Produces: documented implementation status and verified local startup command.

- [ ] **Step 1: Update architecture docs**

In `docs/ARCHITECTURE.md`, add a subsection after `4.14 Public SDK`:

```markdown
### 4.15 Autonomous WebUI

Current file: `bayesprobe/webui.py`

Responsibilities:

- serve the local autonomous workbench;
- validate local WebUI requests;
- build request-scoped provider gateways;
- run `AutonomousQuestionRunner`;
- serialize final answer, belief state, cycle, signal, evidence, update, and
  evolution traces.

Architectural rule:

The WebUI is an observation and execution surface. It must not convert signals
to evidence, update posterior values, evolve hypotheses, or bypass
`BayesProbeCore`.

Current limitations:

- local-only;
- no streaming UI;
- no multi-user auth;
- `openai_chat_completions` protocol is reserved but not implemented.
```

Update the capability matrix with:

```markdown
| Autonomous WebUI | MVP | Local deterministic/OpenAI Responses workbench for autonomous runs and trace inspection. |
```

Update Phase 3 or roadmap wording to mention methodology benchmark remains next after WebUI tracer bullet.

- [ ] **Step 2: Mark spec implemented**

In `docs/superpowers/specs/2026-07-09-autonomous-webui-v0.1-design.md`, change:

```markdown
Status: Proposed for user review
```

to:

```markdown
Status: Implemented as v0.1
```

- [ ] **Step 3: Run focused verification**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tests/test_webui.py \
  tests/test_openai_gateway.py \
  tests/test_public_api_and_config.py \
  tests/test_experiment_artifacts.py \
  -q -p no:cacheprovider
```

Expected: all selected tests pass.

- [ ] **Step 4: Run full verification**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
git diff --check
```

Expected: all offline tests pass, default-skipped live OpenAI tests remain skipped unless explicitly enabled, and `git diff --check` emits no output.

- [ ] **Step 5: Manually start local WebUI**

Run:

```bash
python3 -m bayesprobe.webui --host 127.0.0.1 --port 8765
```

Expected output:

```text
BayesProbe WebUI running at http://127.0.0.1:8765
```

Open `http://127.0.0.1:8765`, run deterministic mode, and verify:

- answer projection renders;
- belief state renders;
- cycle trace renders;
- provider key field is not stored in local storage;
- no layout overlap at desktop width and at a narrow mobile-sized viewport.

- [ ] **Step 6: Commit**

```bash
git add docs/ARCHITECTURE.md docs/superpowers/specs/2026-07-09-autonomous-webui-v0.1-design.md
git commit -m "docs: record autonomous webui"
```

---

## Final Review and Merge Checklist

- [ ] Run a final branch review against the spec:

```bash
git diff --stat 0aa4c1f..HEAD
rg -n "api_key|sk-|localStorage|OpenAIResponsesModelGateway|base_url|AutonomousQuestionRunner|BayesProbeCore" bayesprobe tests docs
```

- [ ] Verify no raw API key is accepted in persisted experiment config.
- [ ] Verify WebUI provider errors sanitize key-like strings.
- [ ] Verify `openai_chat_completions` returns unsupported-provider error.
- [ ] Verify no core/evidence/posterior/probe control-flow files changed except imports if absolutely necessary. Expected unchanged files:
  - `bayesprobe/core.py`
  - `bayesprobe/evidence.py`
  - `bayesprobe/belief.py`
  - `bayesprobe/probe_planner.py`
  - `bayesprobe/probe_executor.py`
- [ ] Run final commands:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
git diff --check
git status --short --branch
```

Expected:

- full tests pass with only default-skipped live OpenAI tests skipped;
- no whitespace errors;
- working tree clean after final commit.

## Self-Review

- Spec coverage: tasks cover local WebUI, deterministic mode, OpenAI Responses key/base URL/model/timeout/max-output settings, runner config, trace serialization, static UI, secret handling, docs, and verification.
- Scope check: plan does not implement Chat Completions, benchmark comparison UI, streaming UI, search/tool gateway, multi-user auth, or core control-flow changes.
- Type consistency: provider kind names are consistently `deterministic`, `openai_responses`, and `openai_chat_completions`; runner field names match `AutonomousQuestionRunConfig`; OpenAI adapter uses `base_url`, `api_key`, `api_key_env`, `timeout_seconds`, and `max_output_tokens`.
- Placeholder scan: no TBD/TODO/fill-in placeholders are intended; all implementation steps name concrete files, functions, and verification commands.
