# OpenAI-Compatible Chat Completions Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a provider-agnostic OpenAI-compatible Chat Completions `ModelGateway` and enable it in WebUI/config without changing BayesProbe core control flow.

**Architecture:** The new `OpenAIChatCompletionsModelGateway` lives beside `OpenAIResponsesModelGateway` and implements the same `complete_structured(...)` interface. WebUI/config only select and configure the adapter; all model output still goes through the existing Evidence Integration Gate and `EvidenceJudgment` validation.

**Tech Stack:** Python 3.11+, existing optional `openai` package extra, Pydantic/domain models, stdlib WebUI server, plain HTML/CSS/JS tests.

## Global Constraints

- Support OpenAI-compatible Chat Completions providers using the common Chat Completions request/response shape.
- Treat DeepSeek as one smoke-testable provider example, not a special-case adapter.
- Do not add provider-specific adapter branches for individual OpenAI-compatible providers.
- Preserve deterministic mode and OpenAI Responses mode unchanged.
- API keys are request-scoped in WebUI and must not be written to JSON config, artifacts, logs, ledger records, static assets, browser local storage, or JSON error responses.
- Keep local WebUI loopback-only.
- No changes to `BayesProbeCore`, the Evidence Integration Gate, posterior update rules, hypothesis evolution, probe planning, or probe execution.
- No streaming UI or token streaming.
- Existing `kind="openai"` remains the Responses adapter for backward compatibility.
- Full offline tests must pass before completion.

---

## File Structure

- Modify `bayesprobe/openai_gateway.py`: add Chat Completions request payload, response parsing, and `OpenAIChatCompletionsModelGateway`.
- Modify `bayesprobe/model_gateway.py`: route `ModelGatewayConfig.kind == "openai_chat_completions"` to the new adapter.
- Modify `bayesprobe/config.py`: parse persisted `openai_chat_completions` config with `api_key_env` only, rejecting raw `api_key`.
- Modify `bayesprobe/experiment_artifacts.py`: preserve sanitized kind/model/api_key_env/base_url snapshots for the new kind.
- Modify `bayesprobe/__init__.py`: export `OpenAIChatCompletionsModelGateway`.
- Modify `bayesprobe/webui.py`: build Chat Completions gateway from request-scoped provider config.
- Modify `bayesprobe/webui_static/index.html`: rename option to `Chat Completions`.
- Modify `bayesprobe/webui_static/app.js`: treat Chat Completions as a runnable OpenAI-compatible provider.
- Modify `docs/ARCHITECTURE.md`: record Chat Completions adapter as implemented.
- Modify `docs/superpowers/specs/2026-07-10-openai-chat-completions-provider-design.md`: mark implemented after code lands.
- Test `tests/test_openai_gateway.py`, `tests/test_model_gateway.py`, `tests/test_public_api_and_config.py`, `tests/test_experiment_artifacts.py`, and `tests/test_webui.py`.

---

### Task 1: Chat Completions Gateway Adapter

**Files:**
- Modify: `bayesprobe/openai_gateway.py`
- Test: `tests/test_openai_gateway.py`

**Interfaces:**
- Consumes: `OpenAIModelGatewayConfig`, `StructuredModelRequest`, existing `_instruction_for_task(...)`, `_metadata_for_request(...)`, `_optional_request_api_key(...)`, `_build_default_openai_client(...)`, and `_parse_json_object(...)`.
- Produces:
  - `build_openai_chat_completions_payload(request, model, max_output_tokens=None) -> dict[str, Any]`
  - `parse_openai_chat_completions_response(response: Any) -> dict[str, Any]`
  - `OpenAIChatCompletionsModelGateway(config, client=None, api_key=None)`

- [ ] **Step 1: Write failing payload and parser tests**

Add to `tests/test_openai_gateway.py`:

```python
def test_build_openai_chat_completions_payload_uses_common_json_object_shape():
    payload = build_openai_chat_completions_payload(
        make_judge_request(),
        model="provider-model",
        max_output_tokens=256,
    )

    assert payload["model"] == "provider-model"
    assert payload["stream"] is False
    assert payload["max_tokens"] == 256
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["messages"][0]["role"] == "system"
    assert "evidence judgment component" in payload["messages"][0]["content"]
    assert payload["messages"][1]["role"] == "user"
    assert json.loads(payload["messages"][1]["content"])["task"] == "judge_evidence"


def test_parse_openai_chat_completions_response_extracts_mapping_message_content():
    response = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(valid_payload()),
                }
            }
        ]
    }

    assert parse_openai_chat_completions_response(response) == valid_payload()
```

Add an object-shaped parser test:

```python
class FakeChatMessage:
    content = json.dumps(valid_payload())


class FakeChatChoice:
    message = FakeChatMessage()


class FakeChatResponse:
    choices = [FakeChatChoice()]


def test_parse_openai_chat_completions_response_extracts_object_message_content():
    assert parse_openai_chat_completions_response(FakeChatResponse()) == valid_payload()
```

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tests/test_openai_gateway.py::test_build_openai_chat_completions_payload_uses_common_json_object_shape \
  tests/test_openai_gateway.py::test_parse_openai_chat_completions_response_extracts_mapping_message_content \
  tests/test_openai_gateway.py::test_parse_openai_chat_completions_response_extracts_object_message_content \
  -q -p no:cacheprovider
```

Expected: import/name failures because functions are not implemented.

- [ ] **Step 3: Implement payload and parser**

In `bayesprobe/openai_gateway.py`, add:

```python
def build_openai_chat_completions_payload(
    request: StructuredModelRequest,
    *,
    model: str,
    max_output_tokens: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": _instruction_for_task(request.task)},
            {
                "role": "user",
                "content": json.dumps(
                    {"task": request.task, "input": request.input},
                    sort_keys=True,
                ),
            },
        ],
        "response_format": {"type": "json_object"},
        "stream": False,
    }
    if max_output_tokens is not None:
        payload["max_tokens"] = max_output_tokens
    return payload
```

Add parser helpers:

```python
def parse_openai_chat_completions_response(response: Any) -> dict[str, Any]:
    if isinstance(response, Mapping):
        return _parse_json_object(_chat_message_content_from_mapping(response))
    content = _chat_message_content_from_object(response)
    if content is None:
        raise ModelGatewayValidationError("openai chat completion content was missing")
    return _parse_json_object(content)
```

Implement `_chat_message_content_from_mapping(...)` and `_chat_message_content_from_object(...)` to read `choices[0].message.content`, raising `ModelGatewayValidationError("openai chat completion content was missing")` for missing content.

- [ ] **Step 4: Verify GREEN**

Run the same focused command from Step 2.

Expected: all three tests pass.

- [ ] **Step 5: Add gateway class test**

Add to `tests/test_openai_gateway.py`:

```python
class FakeChatCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **payload):
        self.calls.append(payload)
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(valid_payload()),
                    }
                }
            ]
        }


class FakeChatClient:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": FakeChatCompletions()})()


def test_openai_chat_completions_model_gateway_calls_fake_client_and_returns_dict():
    client = FakeChatClient()
    gateway = OpenAIChatCompletionsModelGateway(
        config=OpenAIModelGatewayConfig(model="provider-model", max_output_tokens=128),
        client=client,
    )

    result = gateway.complete_structured(make_judge_request())

    assert result == valid_payload()
    assert client.chat.completions.calls[0]["model"] == "provider-model"
    assert client.chat.completions.calls[0]["max_tokens"] == 128
```

- [ ] **Step 6: Verify RED/GREEN and commit**

Run focused test, implement `OpenAIChatCompletionsModelGateway`, export names in `__all__`, then run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_openai_gateway.py -q -p no:cacheprovider
git add bayesprobe/openai_gateway.py tests/test_openai_gateway.py
git commit -m "feat: add chat completions model gateway"
```

---

### Task 2: Config, SDK, and Artifact Wiring

**Files:**
- Modify: `bayesprobe/model_gateway.py`
- Modify: `bayesprobe/config.py`
- Modify: `bayesprobe/experiment_artifacts.py`
- Modify: `bayesprobe/__init__.py`
- Test: `tests/test_model_gateway.py`
- Test: `tests/test_public_api_and_config.py`
- Test: `tests/test_experiment_artifacts.py`

**Interfaces:**
- Consumes: `OpenAIChatCompletionsModelGateway` from Task 1.
- Produces: `ModelGatewayConfig(kind="openai_chat_completions", ...)` support in config/factory/public SDK/artifact snapshots.

- [ ] **Step 1: Write failing factory and public export tests**

Add to `tests/test_model_gateway.py`:

```python
def test_build_model_gateway_creates_openai_chat_completions_gateway():
    gateway = build_model_gateway(
        {
            "kind": "openai_chat_completions",
            "model": "provider-model",
            "api_key_env": "PROVIDER_API_KEY",
            "base_url": "https://provider.example/v1",
        }
    )

    assert isinstance(gateway, OpenAIChatCompletionsModelGateway)
    assert gateway.config.model == "provider-model"
    assert gateway.config.api_key_env == "PROVIDER_API_KEY"
    assert gateway.config.base_url == "https://provider.example/v1"
```

Update `tests/test_public_api_and_config.py` public export expectations to include `OpenAIChatCompletionsModelGateway`.

- [ ] **Step 2: Write failing persisted config test**

Add to `tests/test_public_api_and_config.py`:

```python
def test_experiment_config_from_mapping_parses_openai_chat_completions(tmp_path: Path):
    dataset_path = tmp_path / "dataset.json"
    report_path = tmp_path / "report.json"

    config = experiment_config_from_mapping(
        {
            "dataset_path": str(dataset_path),
            "report_path": str(report_path),
            "model_gateway": {
                "kind": "openai_chat_completions",
                "model": "provider-model",
                "api_key_env": "PROVIDER_API_KEY",
                "base_url": "https://provider.example/v1",
            },
        }
    )

    assert config.model_gateway is not None
    assert config.model_gateway.kind == "openai_chat_completions"
    assert config.model_gateway.model == "provider-model"
    assert config.model_gateway.api_key_env == "PROVIDER_API_KEY"
    assert config.model_gateway.base_url == "https://provider.example/v1"
```

- [ ] **Step 3: Verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tests/test_model_gateway.py::test_build_model_gateway_creates_openai_chat_completions_gateway \
  tests/test_public_api_and_config.py::test_experiment_config_from_mapping_parses_openai_chat_completions \
  -q -p no:cacheprovider
```

Expected: import/factory/config failures.

- [ ] **Step 4: Implement factory/config/public export**

In `bayesprobe/model_gateway.py`, add a branch:

```python
if gateway_config.kind == "openai_chat_completions":
    if gateway_config.model is None:
        raise ValueError("openai chat completions model gateway requires model")
    from bayesprobe.openai_gateway import (
        OpenAIChatCompletionsModelGateway,
        OpenAIModelGatewayConfig,
    )
    return OpenAIChatCompletionsModelGateway(...)
```

In `bayesprobe/config.py`, allow `kind in {"openai", "openai_chat_completions"}` for OpenAI-style config validation while continuing to reject raw `api_key`.

In `bayesprobe/__init__.py`, export `OpenAIChatCompletionsModelGateway`.

- [ ] **Step 5: Verify GREEN and artifact coverage**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tests/test_model_gateway.py \
  tests/test_public_api_and_config.py \
  tests/test_experiment_artifacts.py \
  -q -p no:cacheprovider
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```bash
git add bayesprobe/model_gateway.py bayesprobe/config.py bayesprobe/experiment_artifacts.py bayesprobe/__init__.py tests/test_model_gateway.py tests/test_public_api_and_config.py tests/test_experiment_artifacts.py
git commit -m "feat: wire chat completions config"
```

---

### Task 3: WebUI Chat Completions Provider

**Files:**
- Modify: `bayesprobe/webui.py`
- Modify: `bayesprobe/webui_static/index.html`
- Modify: `bayesprobe/webui_static/app.js`
- Test: `tests/test_webui.py`

**Interfaces:**
- Consumes: `OpenAIChatCompletionsModelGateway`.
- Produces: runnable WebUI `provider.kind == "openai_chat_completions"`.

- [ ] **Step 1: Write failing backend WebUI test**

Add fake Chat Completions client and test to `tests/test_webui.py`:

```python
class FakeWebUIChatCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **payload):
        self.calls.append(payload)
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "evidence_type": "supporting",
                                "likelihoods": {
                                    "H1": "moderately_confirming",
                                    "H2": "moderately_disconfirming",
                                },
                                "interpretation": "WebUI fake chat response.",
                                "quality_overrides": {},
                            }
                        )
                    }
                }
            ]
        }


class FakeWebUIChatOpenAI:
    created_with = []

    def __init__(self, **kwargs):
        self.__class__.created_with.append(kwargs)
        self.chat = type("Chat", (), {"completions": FakeWebUIChatCompletions()})()


def test_webui_openai_chat_completions_provider_uses_request_key_and_redacts_response():
    FakeWebUIChatOpenAI.created_with = []

    status, payload = handle_autonomous_run_request(
        {
            "question": "Can the WebUI use a chat completions evidence judgment?",
            "provider": {
                "kind": "openai_chat_completions",
                "api_key": "provider-secret-123",
                "base_url": "https://provider.example/v1",
                "model": "provider-model",
                "timeout_seconds": 11,
                "max_output_tokens": 128,
            },
            "runner": {"max_cycles": 1, "max_probes_per_cycle": 1},
        },
        client_factory=FakeWebUIChatOpenAI,
    )

    assert status == 200
    assert FakeWebUIChatOpenAI.created_with == [
        {
            "api_key": "provider-secret-123",
            "timeout": 11.0,
            "base_url": "https://provider.example/v1",
        }
    ]
    assert "provider-secret-123" not in json.dumps(payload)
    assert payload["cycles"][0]["evidence_events"][0]["model_trace"]["adapter_kind"] == "openai_chat_completions"
```

- [ ] **Step 2: Write failing static UI test**

Update static asset tests to assert:

```python
assert "Chat Completions (unsupported)" not in index
assert "Chat Completions" in index
assert "openai_chat_completions" in script
assert 'provider.kind === "openai_responses" || provider.kind === "openai_chat_completions"' in script
assert "Chat Completions stays visible in v0.1 but is not supported" not in script
```

- [ ] **Step 3: Verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tests/test_webui.py::test_webui_openai_chat_completions_provider_uses_request_key_and_redacts_response \
  tests/test_webui.py::test_webui_static_assets_define_operational_workbench \
  -q -p no:cacheprovider
```

Expected: unsupported-provider/static-copy failures.

- [ ] **Step 4: Implement backend and frontend**

In `bayesprobe/webui.py`:

- remove `openai_chat_completions` from reserved providers;
- import `OpenAIChatCompletionsModelGateway`;
- branch both `openai_responses` and `openai_chat_completions` through shared OpenAI-compatible provider config parsing;
- return `OpenAIChatCompletionsModelGateway` for chat kind.

In `app.js`:

- make `usesRemoteProvider` true for both provider kinds;
- remove run-button disable for chat completions;
- include provider settings for chat completions in `buildPayload`.

In `index.html`, remove `(unsupported)`.

- [ ] **Step 5: Verify GREEN and commit**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_webui.py -q -p no:cacheprovider
git add bayesprobe/webui.py bayesprobe/webui_static/index.html bayesprobe/webui_static/app.js tests/test_webui.py
git commit -m "feat: enable webui chat completions provider"
```

---

### Task 4: Docs, Verification, and Final Review

**Files:**
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/superpowers/specs/2026-07-10-openai-chat-completions-provider-design.md`
- Test: full repository

**Interfaces:**
- Consumes: completed adapter/config/WebUI work.
- Produces: documented implemented status and final verification evidence.

- [ ] **Step 1: Update docs**

In `docs/ARCHITECTURE.md`:

- add `OpenAIChatCompletionsModelGateway` to current Model Gateway adapters;
- update WebUI limitation from "`openai_chat_completions` protocol is reserved but not implemented" to "OpenAI-compatible Chat Completions is implemented as a generic provider adapter";
- update capability matrix WebUI note to mention deterministic/OpenAI Responses/Chat Completions.

In the spec, change:

```markdown
Status: Proposed for implementation
```

to:

```markdown
Status: Implemented
```

- [ ] **Step 2: Run focused verification**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tests/test_openai_gateway.py \
  tests/test_model_gateway.py \
  tests/test_public_api_and_config.py \
  tests/test_experiment_artifacts.py \
  tests/test_webui.py \
  -q -p no:cacheprovider
```

Expected: all focused tests pass.

- [ ] **Step 3: Run full verification and diff hygiene**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
git diff --check
```

Expected: full offline tests pass with default live tests skipped, and diff check has no output.

- [ ] **Step 4: Manual local smoke**

Start:

```bash
python3 -m bayesprobe.webui --host 127.0.0.1 --port 8766
```

Verify deterministic mode still runs. With a user-provided compatible key, verify Chat Completions can be selected and posts provider settings. Do not commit or log raw keys.

- [ ] **Step 5: Commit docs**

```bash
git add docs/ARCHITECTURE.md docs/superpowers/specs/2026-07-10-openai-chat-completions-provider-design.md
git commit -m "docs: record chat completions provider"
```

- [ ] **Step 6: Final review and push**

Run final review against the spec, fix Critical/Important findings, then:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
git diff --check
git status --short --branch
git push origin main
```

Expected: branch is clean and pushed to `origin/main`.
