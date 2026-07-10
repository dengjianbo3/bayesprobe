# WebUI Streaming Progress and Session-Only Credential Design

Date: 2026-07-10
Status: Implemented and verified

## Context

The autonomous WebUI currently sends one JSON request to
`/api/runs/autonomous` and waits for the complete run result. During provider
calls the only visible state is a generic running message, even though the
runner is progressing through initialization, probe planning, probe execution,
signal collection, evidence integration, and posterior revision.

The frontend also clears the API-key input immediately after building a request
and again when the request finishes. This prevents consecutive runs in one page
session and provides no additional protection against an already running page.

This change adds truthful phase and per-cycle progress without changing the
BayesProbe control flow. It also keeps the API key in the current DOM until the
page is refreshed or the user edits it, while continuing to prohibit browser
storage, server persistence, logs, ledgers, and response events from containing
the credential.

## Goals

1. Preserve the API-key input across successful and failed runs in the same
   loaded page.
2. Preserve the key when switching between provider modes in the same page.
3. Clear the key naturally on page refresh by never writing it to persistent or
   session browser storage.
4. Display real runner phases as they occur.
5. Update the answer, belief state, and cycle trace immediately after each
   integrated cycle.
6. Keep the existing synchronous API and final response contract compatible.
7. Keep all signal-to-evidence and belief updates inside the existing runner and
   `BayesProbeCore` path.

## Non-Goals

- Token-level model output or hidden reasoning display.
- WebSocket infrastructure.
- A durable background job queue or resumable runs.
- Cooperative cancellation of an in-flight provider request.
- Persisting credentials in `localStorage`, `sessionStorage`, cookies, config,
  logs, ledgers, or artifacts.
- Streaming synchronized multi-agent rounds in this milestone.

## Considered Transport Approaches

### A. POST With NDJSON Response

The browser sends the existing request body once and reads newline-delimited
JSON events from the response stream. This keeps credentials in the request
body, needs no job registry, works with the current threaded HTTP server, and
allows each event to be flushed as soon as the runner emits it.

### B. Create-Run POST Plus Server-Sent Events

SSE provides a familiar event API but requires a POST to create a run, a
server-side job registry, and a second GET connection to consume events. It
introduces lifecycle, cleanup, and concurrency state that the local MVP does
not otherwise need.

### C. Background Job Plus Polling

Polling is simple for clients but requires the same job registry, adds repeated
requests and latency, and makes exact phase ordering less direct.

### Decision

Use POST with an `application/x-ndjson` response. Preserve the existing JSON
endpoint for compatibility.

## Architecture

The implementation has three layers:

1. `AutonomousQuestionRunner` publishes optional typed progress observations.
2. The WebUI streaming adapter serializes those observations into NDJSON.
3. The frontend consumes events and updates a compact progress list and the
   existing result panels.

The observer is optional. Existing SDK callers and benchmarks that do not
provide one retain current behavior.

### Runner Observation Contract

Add an `AutonomousQuestionProgressKind` enum and immutable
`AutonomousQuestionProgress` value. The progress value carries the typed domain
objects relevant to its phase rather than HTTP-shaped dictionaries.

The runner accepts an optional callback:

```python
Callable[[AutonomousQuestionProgress], None]
```

The runner emits these phases in order:

1. `run_started`
2. `initialization_completed`
3. `cycle_started`
4. `probe_set_planned`
5. `probe_execution_started`
6. `signals_collected`
7. `evidence_integration_started`
8. `cycle_integrated`
9. Repeat phases 3-8 for later cycles
10. `run_completed`

`cycle_integrated` contains the completed
`AutonomousQuestionCycleResult`. `run_completed` contains the final
`AutonomousQuestionRunResult`. Early stop conditions still end with
`run_completed` and the actual stop reason.

Progress publication is observational. It must not select probes, mutate
signals, construct evidence, change posterior values, or decide stop
conditions.

## Streaming API

Add:

```text
POST /api/runs/autonomous/stream
Content-Type: application/json
Accept: application/x-ndjson
```

The request schema is identical to `/api/runs/autonomous`.

Each response line has this envelope:

```json
{
  "event": "cycle_integrated",
  "sequence": 8,
  "timestamp": "2026-07-10T00:00:00Z",
  "run_id": "webui_...",
  "cycle_id": "webui_..._cycle_1",
  "cycle_index": 1,
  "data": {}
}
```

The sequence starts at one and is strictly increasing within the connection.
The adapter flushes after every complete line.

Event data is deliberately bounded:

- `run_started`: run id only;
- `initialization_completed`: initial run and belief state;
- `cycle_started`: cycle identity and current belief summary;
- `probe_set_planned`: selected probe set;
- `probe_execution_started`: probe count and cycle identity;
- `signals_collected`: signals with their normal domain serialization;
- `evidence_integration_started`: signal count and cycle identity;
- `cycle_integrated`: serialized cycle result, including that cycle's belief
  state and answer projection;
- `run_completed`: the same complete result shape used by the synchronous API;
- `run_failed`: sanitized error type and message.

Request validation and provider configuration errors detected before response
streaming retain their normal HTTP status and JSON error body. Once a `200`
NDJSON stream has started, provider or runtime failures are represented by one
terminal `run_failed` event because the HTTP status can no longer change.

The existing `/api/runs/autonomous` endpoint and
`handle_autonomous_run_request` remain supported.

### Serialization Reuse

Introduce one cycle-result serializer and use it for both `cycle_integrated`
events and the final `cycles` array. Add the cycle's `belief_state` to this
additive response shape so the frontend can render an intermediate posterior
without reconstructing state from updates.

The final event uses `serialize_autonomous_run_result` directly. Streaming and
non-streaming paths therefore cannot drift into different final semantics.

## Frontend State and Rendering

Add an unframed `Run Progress` region between the status banner and result
panels. It contains a compact ordered list of real events. Each row shows the
phase label, cycle number when applicable, and state. It does not display raw
model reasoning.

Frontend behavior by event:

- `run_started`: reset prior output and start the progress list;
- `initialization_completed`: render the initial belief state;
- phase events: append or update the current progress row;
- `cycle_integrated`: render the cycle answer and belief state immediately and
  append the cycle trace;
- `run_completed`: render the complete final payload and terminal run status;
- `run_failed`: retain completed progress and cycle output, mark the current
  phase failed, and show the sanitized error.

The frontend reads `response.body` with a `TextDecoder`, retains an incomplete
line buffer between chunks, and parses only complete newline-delimited records.
Malformed lines or a stream ending without a terminal event produce a visible
stream error while preserving already rendered cycles.

For a non-2xx response received before streaming begins, the frontend parses
the existing JSON error envelope instead of attempting NDJSON consumption.

The run button remains disabled while the connection is active. Cancellation
is not shown because this milestone cannot truthfully cancel an in-flight
provider operation.

## API-Key Lifecycle and Security

Remove every automatic `clearApiKey()` call from submit completion and provider
switching.

The key remains only in:

- the password input element;
- the transient request payload and browser fetch implementation while a run is
  active;
- the request-scoped server gateway object for that run.

The implementation must not:

- write the key to any browser storage API;
- include the key or provider request object in progress events;
- include the key in exception messages, server logs, ledgers, or artifacts;
- add a default key to HTML, JavaScript, Python, fixtures, or tests.

A full page refresh recreates the empty password field and therefore removes
the page-held key.

## Disconnect and Error Handling

The streaming writer treats `BrokenPipeError` and connection reset as client
disconnects. Once disconnected, later observer emissions become no-ops so a
closed browser connection does not alter runner semantics or create secondary
errors. The current synchronous provider operation may finish in its handler
thread; durable cancellation is outside scope.

All externally visible errors use the existing sanitized provider and server
messages. Partial evidence is never fabricated to make a failed cycle appear
integrated.

## Testing

Runner tests verify:

- exact event order for one- and multi-cycle runs;
- early stop paths still emit one terminal `run_completed` event;
- `cycle_integrated` contains an integrated cycle and normalized belief state;
- observing progress does not change the final result.

WebUI tests verify:

- the streaming route accepts the existing request schema;
- NDJSON sequence numbers are monotonic and every record is valid JSON;
- every event is flushed as a complete line;
- `cycle_integrated` precedes `run_completed`;
- provider failures after stream start produce `run_failed`;
- request validation before stream start preserves normal HTTP errors;
- no event or response contains the supplied API key;
- the synchronous endpoint remains compatible.

Frontend and browser tests verify:

- the key survives success, failure, and provider switching in one loaded page;
- no browser storage API is used;
- progress appears before a deliberately delayed provider run completes;
- each integrated cycle updates answer, belief, and trace panels;
- a failure preserves already completed cycle output;
- desktop and 390 px mobile layouts have no global overflow or incoherent
  overlap.

The full pytest suite must pass after focused tests.

## Verification Record

Verification completed on 2026-07-10 after implementation.

- `pytest tests/test_question_runner.py tests/test_webui.py tests/test_public_api_and_config.py -q`:
  `92 passed in 5.40s`.
- `node --test tests/test_webui_stream.js`: `6 pass`, `0 fail`.
- `pytest -q`: `377 passed, 2 skipped in 5.67s`.
- Restarted `python3 -m bayesprobe.webui --host 127.0.0.1 --port 8766` and
  confirmed `GET /` returns `200 OK`.
- Browser verification on the local WebUI confirmed that a non-secret test
  value remains in the API-key field across a deterministic one-cycle run and
  a switch back to Chat Completions, then is empty after reload. The A-E graph
  run rendered initialization, probe planning, probe execution, evidence
  integration, posterior update, and completion; it finished as `completed`,
  its cycle was `integrated`, its best answer was `D`, and its posterior mass
  was `1.000`.
- A test-only delayed Chat Completions provider rendered `Executing probes`
  before the request completed, then rendered the integrated cycle and
  terminal completion on the same page.
- At the default desktop viewport and at `390 x 844`, document scroll width
  did not exceed client width and progress labels/statuses did not overlap.
  The populated trace has a remaining layout concern: its JSON blocks expand
  beyond the trace pane rather than fitting inside a constrained scrollable
  block. This concern is recorded for follow-up and does not change the
  streaming contract above.

## Acceptance Criteria

1. A user can run multiple real-provider questions without re-entering the API
   key until the page is refreshed.
2. The key disappears after refresh and is absent from source, storage, logs,
   ledger records, responses, and progress events.
3. A real provider run visibly advances through truthful runner phases.
4. A multi-cycle run updates the answer and normalized belief state after every
   integrated cycle, before final completion.
5. Provider failure preserves prior progress and completed cycle output.
6. The old JSON endpoint remains usable and returns the same final domain
   result shape.
7. Streaming does not bypass or duplicate `BayesProbeCore` integration.
