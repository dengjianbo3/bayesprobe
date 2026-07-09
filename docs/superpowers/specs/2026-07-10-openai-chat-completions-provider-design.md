# OpenAI-Compatible Chat Completions Provider Design

Date: 2026-07-10
Status: Implemented

## Context

BayesProbe WebUI v0.1 can already run deterministic autonomous questions and
OpenAI Responses-backed evidence judgment. The next practical gap is broader
OpenAI-compatible provider support. Many providers expose an OpenAI-shaped
`/chat/completions` API rather than the newer Responses API. DeepSeek is an
immediate validation example, not the design boundary: its quick-start
documentation lists
`https://api.deepseek.com` as the OpenAI `base_url`, names current models such
as `deepseek-v4-flash` and `deepseek-v4-pro`, and demonstrates
`client.chat.completions.create(...)`.

The feature must not make Chat Completions a new BayesProbe control-flow layer.
It is only another `ModelGateway` adapter. Model output still becomes evidence
only through the existing Evidence Integration Gate and structured
`EvidenceJudgment` validation path.

## Goals

- Add a first-class `OpenAIChatCompletionsModelGateway`.
- Support OpenAI-compatible Chat Completions providers in the local WebUI using
  the common Chat Completions request/response shape:
  - `kind="openai_chat_completions"`;
  - request-scoped `api_key`;
  - `base_url`;
  - `model`;
  - `timeout_seconds`;
  - `max_output_tokens`.
- Treat DeepSeek as one smoke-testable provider example, not a special-case
  adapter.
- Preserve deterministic mode and OpenAI Responses mode unchanged.
- Preserve all WebUI secret rules:
  - no raw API key in JSON config;
  - no raw API key in artifacts, logs, ledger records, static assets, browser
    local storage, or JSON error responses;
  - request-scoped API keys only for WebUI provider calls.
- Keep local WebUI loopback-only.
- Keep provider-backed evidence judgment behind the shared `ModelGateway`
  interface and `EvidenceJudgment` validation.

## Non-Goals

- No provider registry in this slice.
- No provider-specific adapter branches for individual OpenAI-compatible
  providers.
- No streaming UI or token streaming.
- No provider-specific extension parameters such as custom reasoning controls,
  cache controls, beta flags, or provider-specific tool formats in v0.1.
- No benchmark comparison UI.
- No changes to `BayesProbeCore`, the Evidence Integration Gate, posterior
  update rules, hypothesis evolution, probe planning, or probe execution.
- No persistent raw API key support in experiment config.

## External References

- DeepSeek quick start:
  `https://api-docs.deepseek.com/`
- DeepSeek Python chat sample:
  `https://api-docs.deepseek.com/api_samples/chat_python/`
- OpenAI Chat API reference:
  `https://developers.openai.com/api/reference/resources/chat`

## Architecture

### Provider Adapter

Add `OpenAIChatCompletionsModelGateway` in `bayesprobe/openai_gateway.py`, next
to `OpenAIResponsesModelGateway`.

Both adapters should share the existing:

- `OpenAIModelGatewayConfig`;
- request-scoped API key validation helper;
- lazy OpenAI client creation;
- structured response parsing helpers where possible;
- task instructions for `judge_evidence` and `repair_evidence_judgment`.

The Chat Completions adapter implements:

```python
class OpenAIChatCompletionsModelGateway:
    adapter_kind = "openai_chat_completions"

    def complete_structured(self, request: StructuredModelRequest) -> dict[str, Any]:
        ...
```

The request payload should call:

```python
client.chat.completions.create(
    model=config.model,
    messages=[
        {"role": "system", "content": instruction},
        {"role": "user", "content": json_payload},
    ],
    response_format={"type": "json_object"},
    stream=False,
    max_tokens=config.max_output_tokens,
)
```

Notes:

- `max_output_tokens` maps to Chat Completions `max_tokens`.
- `response_format={"type": "json_object"}` is the broad compatible baseline.
  It does not rely on provider support for OpenAI Responses JSON schema mode.
- The returned assistant message content must be parsed as JSON and validated by
  the existing downstream schema path.
- Provider exceptions may propagate from the adapter, but WebUI must convert
  them to generic provider errors that do not echo secrets.

### WebUI

`bayesprobe/webui.py` should stop treating `openai_chat_completions` as a
reserved unsupported provider. Instead, it should build
`OpenAIChatCompletionsModelGateway` with the same request-scoped key, base URL,
model, timeout, and max-output settings already used for Responses.

`bayesprobe/webui_static/index.html` should rename the option from
`Chat Completions (unsupported)` to `Chat Completions`.

`bayesprobe/webui_static/app.js` should:

- show provider auth/settings for both OpenAI Responses and Chat Completions;
- include `api_key`, `base_url`, `model`, `timeout_seconds`, and
  `max_output_tokens` for both provider kinds;
- keep clearing the API key field after submission;
- keep avoiding `localStorage`;
- no longer disable the run button for `openai_chat_completions`.

### Config and SDK

Persisted experiment config should gain a provider kind that can select the Chat
Completions adapter without allowing raw API keys:

```json
{
  "model_gateway": {
    "kind": "openai_chat_completions",
    "model": "provider-model-name",
    "api_key_env": "PROVIDER_API_KEY",
    "base_url": "https://provider.example/v1"
  }
}
```

`ModelGatewayConfig.kind` should accept `openai_chat_completions`, and
`build_model_gateway(...)` should instantiate
`OpenAIChatCompletionsModelGateway` for that kind.

The existing `kind="openai"` path should remain the Responses adapter for
backward compatibility.

Public exports should include `OpenAIChatCompletionsModelGateway` so external
code can construct it directly.

### Artifacts

Experiment artifact snapshots should continue to include only sanitized provider
configuration:

- kind;
- model;
- api_key_env name;
- timeout;
- max output tokens;
- base URL.

They must never include raw API key values.

## Error Handling

- Missing WebUI Chat Completions `api_key` returns HTTP 400
  `validation_error`.
- Provider request or initialization failures return HTTP 502
  `provider_error` with a generic message.
- Raw provider exception text must not appear in JSON error responses.
- Malformed provider JSON output follows the existing schema failure behavior:
  it is converted by the Evidence Integration Gate into discarded neutral
  evidence rather than directly updating belief.

## Testing Strategy

Tests are written before implementation.

Focused tests:

- Chat Completions request payload uses `messages`, `response_format`, `stream`
  false, model, and `max_tokens`.
- Chat Completions response parser extracts
  `choices[0].message.content` from mapping and object-shaped responses.
- Chat Completions adapter uses request-scoped key and base URL.
- `build_model_gateway({"kind": "openai_chat_completions", ...})` returns
  `OpenAIChatCompletionsModelGateway`.
- persisted experiment config parses `kind="openai_chat_completions"` with
  `api_key_env` and rejects raw `api_key`;
- WebUI builds Chat Completions provider with request-scoped key and redacts
  errors;
- WebUI static assets no longer mark Chat Completions as unsupported and submit
  provider settings for it;
- deterministic and Responses-focused tests remain green.

Verification:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tests/test_openai_gateway.py \
  tests/test_model_gateway.py \
  tests/test_public_api_and_config.py \
  tests/test_experiment_artifacts.py \
  tests/test_webui.py \
  -q -p no:cacheprovider

PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
git diff --check
```

Optional live smoke with a user-provided key should be manual and explicit. It
must not run by default in CI or offline tests. DeepSeek can be the first live
smoke target because it motivated this work, but the implementation should not
hard-code DeepSeek-specific branches.

## Definition of Done

- OpenAI-compatible Chat Completions WebUI configuration can use:
  - Protocol: Chat Completions;
  - provider base URL such as `https://api.deepseek.com` or another compatible
    endpoint;
  - provider model name such as `deepseek-v4-flash` or another compatible model;
  - request-scoped API key.
- Chat Completions model output enters BayesProbe only through
  `ModelGateway.complete_structured(...)` and existing evidence validation.
- No core/evidence/belief/probe control-flow files are modified.
- API keys are not persisted or echoed.
- Focused and full offline tests pass.
- Architecture docs record Chat Completions as implemented provider support.
