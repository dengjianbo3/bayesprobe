# Autonomous WebUI v0.1 Design

Date: 2026-07-09
Status: Proposed for user review

## Context

BayesProbe now has an autonomous question runner, an evidence-gated belief
revision core, OpenAI Responses adapter support, prompt/provider provenance
artifacts, and deterministic benchmark fixtures. The next practical gap is
observability: a human should be able to ask the agent a concrete question and
inspect how BayesProbe moves from hypotheses to probes, signals, evidence,
posterior updates, and an answer projection.

This WebUI is an engineering observation surface, not a new agent paradigm. It
must preserve the existing BayesProbe control flow:

```text
WebUI / local API
  -> AutonomousQuestionRunner
  -> BayesProbeCore
  -> Evidence Integration Gate
  -> Belief Solver
  -> Hypothesis Evolution
  -> Answer Projection
```

The UI also needs provider configuration for OpenAI-compatible services:
API key, base URL, model name, timeout, and output-token budget. The first
functional provider protocol is OpenAI Responses. Chat Completions-compatible
providers are designed as the next adapter, because many third-party
OpenAI-compatible endpoints still primarily support `/chat/completions`.

## Goals

- Add a local WebUI for running BayesProbe in autonomous mode.
- Allow the user to configure provider protocol, API key, base URL, model, and
  runner limits from the page.
- Keep secrets ephemeral: request-scoped only, never written to JSON config,
  artifacts, logs, ledger records, or browser local storage.
- Show a readable run trace:
  - final answer projection;
  - stop reason;
  - initial and final belief state;
  - cycles;
  - probe set and selected candidates;
  - external signals;
  - evidence events;
  - belief updates;
  - hypothesis evolution records.
- Preserve deterministic local mode so the WebUI can be tested and demoed
  without network access or an API key.
- Keep the backend API stable enough that a future React/Vite frontend can
  replace the static v0.1 frontend without changing BayesProbe internals.

## Non-Goals

- No multi-user auth, deployment hardening, or hosted service mode.
- No persistent secret storage.
- No benchmark comparison UI in v0.1.
- No streaming token UI in v0.1.
- No real web search or external tool gateway in v0.1.
- No direct modification of `BayesProbeCore`, evidence integration rules,
  posterior update rules, or probe control flow.
- No requirement that every OpenAI-compatible provider work through Responses;
  Chat Completions is a planned adapter after the WebUI tracer bullet.

## External References

- [OpenAI API overview](https://platform.openai.com/docs/overview) recommends
  choosing an API surface and explicitly warns that API keys should not be
  exposed in browser/client-side code.
- [OpenAI text generation guide](https://platform.openai.com/docs/guides/text)
  recommends Responses for new text-generation work, especially with reasoning
  models.
- [OpenAI migration guide](https://platform.openai.com/docs/guides/responses-vs-chat-completions)
  states that Chat Completions remains supported while Responses is the newer
  primitive.
- [OpenAI Python SDK](https://github.com/openai/openai-python) supports
  `base_url` and `OPENAI_BASE_URL`, which makes a Responses-compatible custom
  endpoint possible without changing the adapter contract.

## Proposed Architecture

### Backend

Create a small local server module:

```text
bayesprobe/webui.py
```

Responsibilities:

- serve static assets;
- expose `POST /api/runs/autonomous`;
- validate request payloads;
- build an ephemeral `ModelGateway`;
- run `AutonomousQuestionRunner`;
- serialize the run result into UI-friendly JSON;
- map validation errors and provider errors into sanitized JSON error payloads.

The server should use Python standard-library HTTP serving for v0.1 unless tests
show it becomes awkward. This avoids adding a new web framework dependency while
the API surface is still small.

### Frontend

Create static assets:

```text
bayesprobe/webui_static/index.html
bayesprobe/webui_static/styles.css
bayesprobe/webui_static/app.js
```

The UI is an operational workbench, not a landing page. The first viewport
should show the actual runner:

- left rail: provider and runner configuration;
- main pane: question/context input and run button;
- result pane: final answer and belief summary;
- trace pane: cycle-by-cycle signals, evidence events, updates, and evolution.

The v0.1 frontend should use plain HTML/CSS/JS. A later React implementation can
consume the same JSON API if the interface grows.

## API Shape

### `POST /api/runs/autonomous`

Request:

```json
{
  "question": "Which explanation best fits this case?",
  "context": "Optional background context.",
  "provider": {
    "kind": "deterministic"
  },
  "runner": {
    "max_cycles": 2,
    "max_probes_per_cycle": 2,
    "stop_on_no_probes": true,
    "confidence_threshold": null,
    "posterior_delta_threshold": null
  }
}
```

Responses-compatible provider request:

```json
{
  "question": "Which hypothesis should I believe?",
  "context": "",
  "provider": {
    "kind": "openai_responses",
    "api_key": "request scoped secret",
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-5.5",
    "timeout_seconds": 30,
    "max_output_tokens": 512
  },
  "runner": {
    "max_cycles": 2,
    "max_probes_per_cycle": 2
  }
}
```

Response:

```json
{
  "run_id": "webui_...",
  "stop_reason": "max_cycles",
  "final_answer": {
    "current_best_hypothesis": "H1",
    "summary": "...",
    "confidence": 0.72,
    "change_my_mind_condition": {}
  },
  "initial_belief_state": {},
  "final_belief_state": {},
  "cycles": [
    {
      "cycle_id": "...",
      "signal_shape": "active_only",
      "probes": [],
      "signals": [],
      "evidence_events": [],
      "belief_updates": [],
      "hypothesis_evolutions": [],
      "answer_projection": {}
    }
  ]
}
```

Error response:

```json
{
  "error": {
    "type": "validation_error",
    "message": "provider.model must not be empty"
  }
}
```

Errors must not echo API keys.

## Provider Configuration

### Deterministic

`kind="deterministic"` uses the existing deterministic model gateway and
deterministic probe tool gateway. This is the default and is always available.

### OpenAI Responses

`kind="openai_responses"` builds an ephemeral OpenAI client from request fields
and passes that client into `OpenAIResponsesModelGateway`.

The existing provider adapter should be deepened to support:

- `base_url: str | None`;
- request-scoped `api_key` supplied by the WebUI backend;
- existing `api_key_env` for config-file and CLI usage;
- sanitized snapshots that never include API key values.

The persisted experiment config path continues to allow only `api_key_env`, not
raw API key values.

### Chat Completions

The UI request schema reserves `kind="openai_chat_completions"`, but v0.1 must
return a clear unsupported-provider error until the next adapter lands.

The later adapter should still implement `ModelGateway.complete_structured(...)`
and return an `EvidenceJudgment`-compatible JSON object after validation. It
must not bypass the Evidence Integration Gate.

## Runner Configuration

The WebUI maps runner settings to `AutonomousQuestionRunConfig`:

- `max_cycles`;
- `max_probes_per_cycle`;
- `stop_on_no_probes`;
- `confidence_threshold`;
- `posterior_delta_threshold`.

Validation should mirror the dataclass rules:

- `max_cycles >= 1`;
- `max_probes_per_cycle >= 1`;
- `confidence_threshold` is `null` or between `0` and `1`;
- `posterior_delta_threshold` is `null` or non-negative.

## Trace Serialization

The WebUI must serialize domain objects without mutating them. Pydantic models
should use their structured dump method; dataclasses should use structured
serialization. Enums should become their `.value` strings.

The trace should prefer explicit UI fields over raw ledger replay. The ledger is
still useful for experiments, but the WebUI should not force users to inspect
JSONL to understand a run.

## Secret Handling

API keys are allowed only in the WebUI request body and in the in-memory OpenAI
client created for that request. The implementation must not:

- write API keys to config snapshots;
- write API keys to artifact bundles;
- append API keys to the ledger;
- print API keys in server logs;
- include API keys in JSON error responses;
- store API keys in browser local storage.

The UI may keep the key in the current form field while the page is open.
Because this is a local-only workbench, the browser form may temporarily contain
the user's key. The backend must perform provider calls; this UI must not be
deployed as a hosted page that calls provider APIs directly from browser code.

## Error Handling

- Invalid JSON returns HTTP 400 with `type="invalid_json"`.
- Invalid request fields return HTTP 400 with `type="validation_error"`.
- Unsupported provider protocol returns HTTP 400 with
  `type="unsupported_provider"`.
- Provider package/key/base-url/network errors return HTTP 502 with
  `type="provider_error"` and a sanitized message.
- Unexpected server errors return HTTP 500 with `type="server_error"` and no
  secret-bearing details.

## Testing Strategy

Tests are written before implementation.

Focused tests:

- deterministic API request returns final answer, stop reason, belief states,
  and cycle trace;
- invalid request payloads return sanitized 400 responses;
- `openai_responses` request builds a client with request-scoped key and
  optional base URL without writing the key to the response;
- OpenAI gateway config accepts and validates `base_url`;
- experiment config snapshots continue to exclude raw API keys;
- static UI assets are served by the local server.

Manual verification:

- start the local WebUI;
- run a deterministic question without network;
- run an OpenAI Responses question only when the user provides a key;
- inspect that final answer and trace panes render without overlap on desktop
  and narrow mobile widths.

## Implementation Order

1. Deepen the OpenAI Responses adapter for `base_url` and request-scoped client
   creation while preserving config-file secret rules.
2. Add a small WebUI runner service that validates requests and serializes
   autonomous results.
3. Add static WebUI assets with deterministic mode first.
4. Wire `openai_responses` mode into the local API.
5. Add docs and verification instructions.

## Relationship to Methodology Validation Benchmark

This WebUI does not replace benchmark validation. It is the interactive
observation layer for the same method:

- WebUI answers: "What does BayesProbe do on this concrete question?"
- methodology benchmark answers: "Does BayesProbe systematically improve
  answer utility and belief-revision quality across controlled cases?"

After the WebUI tracer bullet is implemented, the next benchmark slice should
add `methodology_validation_v0.1` fixtures and report breakdowns for passive
signals, projection-as-signal behavior, counterevidence response, schema
violation neutrality, and repair recovery.

## Definition of Done

- `python -m bayesprobe.webui` starts a local WebUI and prints the URL.
- Deterministic mode runs with no network or API key.
- OpenAI Responses mode accepts API key, base URL, model, timeout, and
  max-output-token settings from the UI.
- The response displays final answer, belief state summary, and cycle trace.
- API key values are not persisted or echoed.
- Focused WebUI/backend tests pass.
- Full repository tests pass.
- Architecture docs identify the WebUI as an observation surface, not a core
  control-flow layer.
