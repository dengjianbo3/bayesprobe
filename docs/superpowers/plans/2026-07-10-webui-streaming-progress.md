# WebUI Streaming Progress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep request-scoped API credentials available for consecutive runs in one loaded page and stream truthful BayesProbe runner phases plus per-cycle results into the WebUI.

**Architecture:** Add an optional typed progress observer to `AutonomousQuestionRunner`, adapt those observations to POST-delivered NDJSON in the local WebUI server, and consume the stream incrementally in the existing vanilla JavaScript frontend. The current synchronous JSON endpoint and `BayesProbeCore` integration path remain authoritative and compatible.

**Tech Stack:** Python 3.11+, Pydantic 2, `http.server.ThreadingHTTPServer`, newline-delimited JSON, browser Fetch streams, vanilla JavaScript, HTML/CSS, pytest.

## Global Constraints

- Do not add a runtime dependency.
- Keep `/api/runs/autonomous` operational and backward compatible.
- Add `/api/runs/autonomous/stream` with `application/x-ndjson` output.
- Never include an API key in source, browser storage, events, responses, logs, ledgers, fixtures, or artifacts.
- Keep the API key only in the loaded page DOM and request-scoped gateway; refresh must clear it.
- Do not display token-level output or hidden model reasoning.
- Progress is observational and must not influence probe selection, evidence construction, posterior updates, or stop decisions.
- A `cycle_integrated` event must carry the real integrated cycle and normalized belief state.
- Preserve completed cycles and their output when a later provider operation fails.
- Keep the existing restrained WebUI visual language and responsive behavior.

## File Map

- Modify `bayesprobe/question_runner.py`: typed progress contract and phase emission.
- Modify `bayesprobe/__init__.py`: export the supported progress types.
- Modify `bayesprobe/webui.py`: shared run preparation, cycle serialization, NDJSON event adapter, and streaming route.
- Modify `bayesprobe/webui_static/index.html`: progress region.
- Modify `bayesprobe/webui_static/app.js`: Fetch stream consumer, event state, incremental rendering, and session-only key retention.
- Modify `bayesprobe/webui_static/styles.css`: compact progress layout and responsive states.
- Modify `tests/test_question_runner.py`: runner event ordering and semantic neutrality.
- Modify `tests/test_webui.py`: event serialization, HTTP flushing, secret redaction, compatibility, and frontend contracts.
- Modify `docs/ARCHITECTURE.md`: record the streaming observation surface and its non-goals.

---

### Task 1: Typed Runner Progress Observations

**Files:**
- Modify: `bayesprobe/question_runner.py`
- Modify: `bayesprobe/__init__.py`
- Test: `tests/test_question_runner.py`

**Interfaces:**
- Produces: `AutonomousQuestionProgressKind`
- Produces: `AutonomousQuestionProgress`
- Produces: `AutonomousQuestionProgressObserver`
- Changes: `AutonomousQuestionRunner(..., progress_observer: AutonomousQuestionProgressObserver | None = None)`
- Preserves: `AutonomousQuestionRunner.run_question(...) -> AutonomousQuestionRunResult`

- [ ] **Step 1: Add a failing one-cycle event-order test**

Add imports for the two progress types and append this test to
`tests/test_question_runner.py`:

```python
def test_question_runner_emits_truthful_progress_for_integrated_cycle():
    events = []
    runner = AutonomousQuestionRunner(
        core=BayesProbeCore(),
        config=AutonomousQuestionRunConfig(max_cycles=1, max_probes_per_cycle=1),
        progress_observer=events.append,
    )

    result = runner.run_question(
        InitializeRunInput(
            run_id="run_question_progress",
            problem="Does progress follow the BayesProbe lifecycle?",
        )
    )

    assert [event.kind for event in events] == [
        AutonomousQuestionProgressKind.RUN_STARTED,
        AutonomousQuestionProgressKind.INITIALIZATION_COMPLETED,
        AutonomousQuestionProgressKind.CYCLE_STARTED,
        AutonomousQuestionProgressKind.PROBE_SET_PLANNED,
        AutonomousQuestionProgressKind.PROBE_EXECUTION_STARTED,
        AutonomousQuestionProgressKind.SIGNALS_COLLECTED,
        AutonomousQuestionProgressKind.EVIDENCE_INTEGRATION_STARTED,
        AutonomousQuestionProgressKind.CYCLE_INTEGRATED,
        AutonomousQuestionProgressKind.RUN_COMPLETED,
    ]
    cycle_event = events[-2]
    assert cycle_event.cycle_result == result.cycle_results[0]
    assert cycle_event.cycle_result.cycle.boundary_status.value == "integrated"
    assert sum(
        hypothesis.posterior
        for hypothesis in cycle_event.cycle_result.belief_state.hypotheses
    ) == pytest.approx(1.0)
    assert events[-1].result == result
```

- [ ] **Step 2: Add failing early-stop and multi-cycle tests**

```python
def test_question_runner_progress_ends_once_when_no_probe_cycle_is_created():
    events = []
    runner = AutonomousQuestionRunner(
        core=BayesProbeCore(),
        planner=EmptyPlanner(),
        executor=RecordingExecutor(),
        config=AutonomousQuestionRunConfig(max_cycles=2, stop_on_no_probes=True),
        progress_observer=events.append,
    )

    result = runner.run_question(
        InitializeRunInput(
            run_id="run_question_progress_no_probes",
            problem="What happens when progress reaches an empty probe set?",
        )
    )

    assert result.stop_reason == AutonomousQuestionStopReason.NO_PROBES
    assert [event.kind for event in events][-1] == (
        AutonomousQuestionProgressKind.RUN_COMPLETED
    )
    assert sum(
        event.kind == AutonomousQuestionProgressKind.RUN_COMPLETED
        for event in events
    ) == 1
    assert all(
        event.kind != AutonomousQuestionProgressKind.CYCLE_INTEGRATED
        for event in events
    )


def test_question_runner_repeats_cycle_progress_for_each_integrated_cycle():
    events = []
    runner = AutonomousQuestionRunner(
        core=BayesProbeCore(),
        config=AutonomousQuestionRunConfig(max_cycles=2, max_probes_per_cycle=1),
        progress_observer=events.append,
    )

    result = runner.run_question(
        InitializeRunInput(
            run_id="run_question_progress_multi",
            problem="Can progress report both autonomous cycles?",
        )
    )

    integrated = [
        event
        for event in events
        if event.kind == AutonomousQuestionProgressKind.CYCLE_INTEGRATED
    ]
    assert [event.cycle_index for event in integrated] == [1, 2]
    assert [event.cycle_result for event in integrated] == result.cycle_results
```

- [ ] **Step 3: Run the focused tests and confirm the contract is missing**

Run:

```bash
pytest tests/test_question_runner.py -q
```

Expected: collection or constructor failures because the progress types and
`progress_observer` argument do not exist.

- [ ] **Step 4: Add the progress enum and immutable value**

In `bayesprobe/question_runner.py`, import `Callable`, define the enum near the
stop-reason enum, and define the progress value after the cycle/run result
dataclasses so its fields use concrete types:

```python
from collections.abc import Callable


class AutonomousQuestionProgressKind(StrEnum):
    RUN_STARTED = "run_started"
    INITIALIZATION_COMPLETED = "initialization_completed"
    CYCLE_STARTED = "cycle_started"
    PROBE_SET_PLANNED = "probe_set_planned"
    PROBE_EXECUTION_STARTED = "probe_execution_started"
    SIGNALS_COLLECTED = "signals_collected"
    EVIDENCE_INTEGRATION_STARTED = "evidence_integration_started"
    CYCLE_INTEGRATED = "cycle_integrated"
    RUN_COMPLETED = "run_completed"


@dataclass(frozen=True)
class AutonomousQuestionProgress:
    kind: AutonomousQuestionProgressKind
    run_id: str
    cycle_id: str | None = None
    cycle_index: int | None = None
    run: RunRecord | None = None
    belief_state: BeliefState | None = None
    probe_set: ProbeSet | None = None
    signals: tuple[ExternalSignal, ...] = ()
    cycle_result: AutonomousQuestionCycleResult | None = None
    result: AutonomousQuestionRunResult | None = None


AutonomousQuestionProgressObserver = Callable[[AutonomousQuestionProgress], None]
```

- [ ] **Step 5: Store the observer and add one emission helper**

Extend the runner constructor and add this private method:

```python
def __init__(
    self,
    *,
    core: BayesProbeCore,
    initializer: BayesProbeInitializer | None = None,
    planner: ProbePlanner | None = None,
    executor: ProbeExecutor | None = None,
    config: AutonomousQuestionRunConfig | None = None,
    progress_observer: AutonomousQuestionProgressObserver | None = None,
) -> None:
    # Preserve the existing assignments.
    self.progress_observer = progress_observer


def _emit_progress(
    self,
    kind: AutonomousQuestionProgressKind,
    *,
    run_id: str,
    cycle_id: str | None = None,
    cycle_index: int | None = None,
    run: RunRecord | None = None,
    belief_state: BeliefState | None = None,
    probe_set: ProbeSet | None = None,
    signals: tuple[ExternalSignal, ...] = (),
    cycle_result: AutonomousQuestionCycleResult | None = None,
    result: AutonomousQuestionRunResult | None = None,
) -> None:
    if self.progress_observer is None:
        return
    self.progress_observer(
        AutonomousQuestionProgress(
            kind=kind,
            run_id=run_id,
            cycle_id=cycle_id,
            cycle_index=cycle_index,
            run=run,
            belief_state=belief_state,
            probe_set=probe_set,
            signals=signals,
            cycle_result=cycle_result,
            result=result,
        )
    )
```

- [ ] **Step 6: Emit phases at the existing control-flow boundaries**

Add emissions without moving any existing planner, executor, core, or stop
logic:

```python
self._emit_progress(
    AutonomousQuestionProgressKind.RUN_STARTED,
    run_id=input.run_id,
)
initialization = self.initializer.initialize(input)
self._emit_progress(
    AutonomousQuestionProgressKind.INITIALIZATION_COMPLETED,
    run_id=run.run_id,
    run=run,
    belief_state=initial_belief_state,
)
```

For each cycle emit `CYCLE_STARTED` after allocating the cycle id,
`PROBE_SET_PLANNED` after `_plan_next_probe_set`,
`PROBE_EXECUTION_STARTED` immediately before `execute_probe_set`,
`SIGNALS_COLLECTED` after combining active and passive signals,
`EVIDENCE_INTEGRATION_STARTED` immediately before `core.integrate_cycle`, and
`CYCLE_INTEGRATED` after constructing and appending the cycle result. Populate
`cycle_id`, `cycle_index`, `belief_state`, `probe_set`, `signals`, and
`cycle_result` only where those values already exist.

In `_result`, create the `AutonomousQuestionRunResult` in a local variable,
emit `RUN_COMPLETED` exactly once, then return it:

```python
result = AutonomousQuestionRunResult(
    run=completed_run,
    initial_belief_state=initial_belief_state,
    final_belief_state=final_belief_state,
    cycle_results=list(cycle_results),
    final_answer_projection=final_answer_projection,
    stop_reason=stop_reason,
)
self._emit_progress(
    AutonomousQuestionProgressKind.RUN_COMPLETED,
    run_id=completed_run.run_id,
    cycle_id=completed_run.current_cycle_id,
    cycle_index=final_belief_state.cycle_index,
    run=completed_run,
    belief_state=final_belief_state,
    result=result,
)
return result
```

- [ ] **Step 7: Export the progress types from the package root**

Add the three names to the `bayesprobe.question_runner` import and `__all__` in
`bayesprobe/__init__.py`.

- [ ] **Step 8: Run runner and public API tests**

Run:

```bash
pytest tests/test_question_runner.py tests/test_public_api_and_config.py -q
```

Expected: all selected tests pass.

- [ ] **Step 9: Commit the runner observation contract**

```bash
git add bayesprobe/question_runner.py bayesprobe/__init__.py tests/test_question_runner.py tests/test_public_api_and_config.py
git commit -m "feat: expose autonomous run progress"
```

---

### Task 2: NDJSON Streaming Adapter and Route

**Files:**
- Modify: `bayesprobe/webui.py`
- Test: `tests/test_webui.py`

**Interfaces:**
- Consumes: `AutonomousQuestionProgress` and `AutonomousQuestionProgressKind`
- Produces: `serialize_autonomous_cycle_result(...) -> dict[str, Any]`
- Produces: `handle_autonomous_stream_request(payload, *, event_sink, client_factory=None) -> tuple[int, dict[str, Any] | None]`
- Produces: `POST /api/runs/autonomous/stream`
- Preserves: `handle_autonomous_run_request(...)` and `POST /api/runs/autonomous`

- [ ] **Step 1: Add failing direct stream-event tests**

Import `handle_autonomous_stream_request` and add:

```python
def test_webui_stream_emits_ordered_cycle_and_terminal_events():
    events = []
    status, error = handle_autonomous_stream_request(
        {
            "question": "Does the stream expose a completed BayesProbe cycle?",
            "context": "SUPPORTS: deterministic context favors H1.",
            "provider": {"kind": "deterministic"},
            "runner": {"max_cycles": 1, "max_probes_per_cycle": 1},
        },
        event_sink=events.append,
    )

    assert status == 200
    assert error is None
    assert [event["sequence"] for event in events] == list(
        range(1, len(events) + 1)
    )
    assert events[0]["event"] == "run_started"
    assert events[-2]["event"] == "cycle_integrated"
    assert events[-1]["event"] == "run_completed"
    cycle = events[-2]["data"]
    assert cycle["cycle"]["boundary_status"] == "integrated"
    assert cycle["belief_state"]["posterior_summary"][
        "total_active_posterior"
    ] == pytest.approx(1.0)
```

Add a provider-backed redaction test using `FakeWebUIChatOpenAI` and the literal
`provider-secret-123`; assert that literal is absent from `json.dumps(events)`.

- [ ] **Step 2: Add failing stream-error tests**

```python
def test_webui_stream_returns_preflight_validation_as_http_error_payload():
    events = []
    status, error = handle_autonomous_stream_request(
        {"question": ""},
        event_sink=events.append,
    )

    assert status == 400
    assert error["error"]["type"] == "validation_error"
    assert events == []


def test_webui_stream_emits_terminal_sanitized_provider_failure():
    events = []
    status, error = handle_autonomous_stream_request(
        {
            "question": "Will a streaming provider failure stay sanitized?",
            "provider": {
                "kind": "openai_responses",
                "api_key": "sk-webui-secret",
                "model": "gpt-5.5",
            },
        },
        event_sink=events.append,
        client_factory=FailingWebUIOpenAI,
    )

    assert status == 200
    assert error is None
    assert events[-1]["event"] == "run_failed"
    assert events[-1]["data"]["error"]["type"] == "provider_error"
    assert "sk-webui-secret" not in json.dumps(events)
```

- [ ] **Step 3: Add failing HTTP NDJSON tests**

Extend `create_handler_class` test setup to accept an optional client factory,
then test:

```python
def test_webui_http_server_streams_valid_ndjson():
    status, content_type, payload = request_http(
        "POST",
        "/api/runs/autonomous/stream",
        body=json.dumps(
            {
                "question": "Does HTTP expose progress?",
                "provider": {"kind": "deterministic"},
                "runner": {"max_cycles": 1, "max_probes_per_cycle": 1},
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    events = [json.loads(line) for line in payload.splitlines()]
    assert status == 200
    assert content_type == "application/x-ndjson; charset=utf-8"
    assert events[0]["event"] == "run_started"
    assert events[-1]["event"] == "run_completed"
```

Also send malformed JSON to the stream route and assert the existing `400`
JSON error envelope and `application/json; charset=utf-8` content type.

- [ ] **Step 4: Run WebUI tests and verify the stream API is absent**

Run:

```bash
pytest tests/test_webui.py -q
```

Expected: failures for missing stream handler, route, and per-cycle belief
serialization.

- [ ] **Step 5: Extract shared run preparation and cycle serialization**

Create a private immutable prepared-run value containing `runner`,
`InitializeRunInput`, and `provider_kind`. Refactor the synchronous handler to
use one `_prepare_autonomous_run(...)` helper without changing its returned
status or payload.

Extract:

```python
def serialize_autonomous_cycle_result(
    cycle: AutonomousQuestionCycleResult,
) -> dict[str, Any]:
    return {
        "cycle_id": cycle.cycle.cycle_id,
        "signal_shape": cycle.cycle.signal_shape.value,
        "cycle": _dump_domain(cycle.cycle),
        "probes": _dump_domain(cycle.probe_set.probes),
        "signals": _dump_domain(cycle.signals),
        "belief_state": _dump_domain(cycle.belief_state),
        "evidence_events": _dump_domain(cycle.evidence_events),
        "belief_updates": _dump_domain(cycle.belief_updates),
        "hypothesis_evolutions": _dump_domain(cycle.hypothesis_evolutions),
        "answer_projection": _dump_domain(cycle.answer_projection),
    }
```

Use this helper in `serialize_autonomous_run_result`.

- [ ] **Step 6: Implement the sequence-owning event adapter**

Add a small private adapter whose only responsibility is converting typed
progress into transport envelopes:

```python
class _AutonomousProgressEventEmitter:
    def __init__(self, sink: Callable[[Mapping[str, Any]], None]) -> None:
        self._sink = sink
        self.sequence = 0
        self.run_id: str | None = None

    @property
    def started(self) -> bool:
        return self.sequence > 0

    def emit(self, progress: AutonomousQuestionProgress) -> None:
        self.run_id = progress.run_id
        self.sequence += 1
        self._sink(
            {
                "event": progress.kind.value,
                "sequence": self.sequence,
                "timestamp": _webui_timestamp(),
                "run_id": progress.run_id,
                "cycle_id": progress.cycle_id,
                "cycle_index": progress.cycle_index,
                "data": _serialize_progress_data(progress),
            }
        )

    def emit_failure(self, error_type: str, message: str) -> None:
        self.sequence += 1
        self._sink(
            {
                "event": "run_failed",
                "sequence": self.sequence,
                "timestamp": _webui_timestamp(),
                "run_id": self.run_id,
                "cycle_id": None,
                "cycle_index": None,
                "data": {"error": {"type": error_type, "message": message}},
            }
        )
```

Implement `_serialize_progress_data` with an explicit branch for every progress
kind. Never serialize the request provider mapping.

- [ ] **Step 7: Implement the stream request function**

The function must return preflight errors before any event and convert later
failures into a terminal event:

```python
def handle_autonomous_stream_request(
    payload: Mapping[str, Any],
    *,
    event_sink: Callable[[Mapping[str, Any]], None],
    client_factory: Callable[..., Any] | None = None,
) -> tuple[int, dict[str, Any] | None]:
    emitter = _AutonomousProgressEventEmitter(event_sink)
    try:
        prepared = _prepare_autonomous_run(
            payload,
            client_factory=client_factory,
            progress_observer=emitter.emit,
        )
    except WebUIError as error:
        return int(error.status_code), _error_payload(error.error_type, error.message)

    try:
        prepared.runner.run_question(prepared.input)
    except Exception:
        if prepared.provider_kind in OPENAI_COMPATIBLE_PROVIDER_KINDS:
            emitter.emit_failure(
                "provider_error",
                _provider_error_message(prepared.provider_kind),
            )
        else:
            emitter.emit_failure("server_error", _generic_server_error_message())
    return int(HTTPStatus.OK), None
```

Use `server_error`, not `validation_error`, for unexpected deterministic
runtime failures.

- [ ] **Step 8: Add the lazy-flushing HTTP route**

Change `create_handler_class` to accept `client_factory=None`. In `do_POST`,
dispatch the stream path before the synchronous path. Use a closure that sends
headers only when its first event arrives, writes exactly one JSON object plus
`b"\n"`, and calls `self.wfile.flush()` after each event.

Catch `BrokenPipeError` and `ConnectionResetError` inside that closure, set a
`disconnected` flag, and make later writes no-ops. If
`handle_autonomous_stream_request` returns a non-null error before the stream
started, send it with `_send_json` and its original status.

- [ ] **Step 9: Prove the first events flush before provider completion**

Add a blocking fake Chat Completions client using two `threading.Event`
instances. Its first `create` call sets `provider_entered`, waits on
`release_provider`, then returns the normal fake response. Start a handler with
that client factory, issue a stream request, read the first NDJSON line, and
assert `run_started` is available while `provider_entered` is set and before
`release_provider` is set. Release in a `finally` block and consume the terminal
event so the server thread exits cleanly.

- [ ] **Step 10: Run all WebUI and runner tests**

Run:

```bash
pytest tests/test_question_runner.py tests/test_webui.py -q
```

Expected: all selected tests pass.

- [ ] **Step 11: Commit the streaming backend**

```bash
git add bayesprobe/webui.py tests/test_webui.py
git commit -m "feat: stream autonomous run progress"
```

---

### Task 3: Incremental Frontend and Session-Only Key Retention

**Files:**
- Modify: `bayesprobe/webui_static/index.html`
- Modify: `bayesprobe/webui_static/app.js`
- Modify: `bayesprobe/webui_static/styles.css`
- Test: `tests/test_webui.py`

**Interfaces:**
- Consumes: `/api/runs/autonomous/stream` NDJSON events from Task 2
- Produces: `#progress-panel` and `#progress-list`
- Produces: `consumeRunStream(response)` and `handleProgressEvent(event)`
- Preserves: existing answer, belief, and trace renderers

- [ ] **Step 1: Replace stale static-source assertions with failing stream contracts**

Update `test_webui_static_assets_define_operational_workbench` and replace the
old clear-output test with assertions that require:

```python
assert 'id="progress-panel"' in index
assert 'id="progress-list"' in index
assert "fetch('/api/runs/autonomous/stream'" in script
assert "response.body.getReader()" in script
assert "new TextDecoder()" in script
assert "function handleProgressEvent(" in script
assert "cycle_integrated" in script
assert "run_completed" in script
assert "run_failed" in script
assert "clearApiKey" not in script
assert 'apiKeyField.value = ""' not in script
assert "localStorage" not in script
assert "sessionStorage" not in script
assert "document.cookie" not in script
assert ".progress-list" in styles
assert ".progress-item" in styles
```

Keep assertions for the existing provider controls, trace, posterior mass, and
responsive media queries.

- [ ] **Step 2: Run the static frontend contract test and verify failure**

Run:

```bash
pytest tests/test_webui.py::test_webui_static_assets_define_operational_workbench -q
```

Expected: failure because the progress region and stream consumer do not exist
and key clearing still exists.

- [ ] **Step 3: Add the progress region to the existing workspace**

Insert this unframed region after `#status-banner` and before `.result-grid`:

```html
<section
  id="progress-panel"
  class="progress-panel"
  aria-labelledby="progress-heading"
>
  <div class="section-head">
    <h2 id="progress-heading">Run Progress</h2>
    <span id="progress-state" class="eyebrow">Idle</span>
  </div>
  <ol id="progress-list" class="progress-list" aria-live="polite">
    <li class="empty-state">No active run.</li>
  </ol>
</section>
```

- [ ] **Step 4: Stop clearing the credential**

Remove both `clearApiKey()` calls from `handleSubmit`, remove the call from
`syncProviderControls`, and delete `clearApiKey`. Do not replace them with any
storage API. Keep the password input and `autocomplete="off"` unchanged.

- [ ] **Step 5: Replace whole-response fetch handling with NDJSON consumption**

In `handleSubmit`, call the stream endpoint. For non-2xx responses, reuse
`parseJsonResponse`. For successful responses require `response.body` and call:

```javascript
async function consumeRunStream(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let terminalEventSeen = false;

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (!line.trim()) continue;
      let event;
      try {
        event = JSON.parse(line);
      } catch (error) {
        throw new Error("Server returned an invalid progress event");
      }
      handleProgressEvent(event);
      terminalEventSeen ||= ["run_completed", "run_failed"].includes(event.event);
    }

    if (done) break;
  }

  if (buffer.trim()) {
    throw new Error("Progress stream ended with an incomplete event");
  }
  if (!terminalEventSeen) {
    throw new Error("Progress stream ended before the run completed");
  }
}
```

- [ ] **Step 6: Add deterministic event state and incremental rendering**

Maintain a module-local `streamedCycles` array and reset it only at the start of
a new run. `handleProgressEvent` must:

- set the run id on `run_started`;
- render initial belief state on `initialization_completed`;
- append/update the compact progress list for every event;
- upsert a cycle by `cycle_id`, render its `answer_projection` and
  `belief_state`, and render all received cycles on `cycle_integrated`;
- call existing `renderRun(event.data)` and set terminal success status on
  `run_completed`;
- mark progress failed and throw an `Error` with the sanitized server message
  on `run_failed`.

Use an explicit label map:

```javascript
const PROGRESS_LABEL_BY_EVENT = {
  run_started: "Run started",
  initialization_completed: "Belief initialized",
  cycle_started: "Cycle started",
  probe_set_planned: "Probes planned",
  probe_execution_started: "Executing probes",
  signals_collected: "Signals collected",
  evidence_integration_started: "Integrating evidence",
  cycle_integrated: "Posterior updated",
  run_completed: "Run completed",
  run_failed: "Run failed",
};
```

Mark the previous active row complete when a new event arrives. Mark the
terminal row complete or failed. Include `Cycle N` only when
`event.cycle_index` is non-null.

- [ ] **Step 7: Preserve completed output on failure**

Replace the unconditional `clearRunOutput("failed")` catch behavior. A failure
after at least one `cycle_integrated` event must leave answer, belief, and trace
unchanged. A failure before any cycle may show the existing failure states
but must retain the progress list and API-key input.

- [ ] **Step 8: Style the progress surface without introducing nested cards**

Add `.progress-panel` to the same full-width section treatment as
`.trace-section`. Use a stable list layout:

```css
.progress-panel {
  border-top: 1px solid var(--line);
  padding-top: 1rem;
  margin: 1rem 0;
}

.progress-list {
  display: grid;
  gap: 0;
  margin: 0.75rem 0 0;
  padding: 0;
  list-style: none;
}

.progress-item {
  display: grid;
  grid-template-columns: 2.5rem minmax(0, 1fr) auto;
  gap: 0.75rem;
  align-items: baseline;
  min-width: 0;
  padding: 0.55rem 0;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}

.progress-sequence,
.progress-meta,
.progress-status {
  color: var(--muted);
  font-size: 0.82rem;
}

.progress-item[data-state="active"] .progress-status {
  color: var(--accent);
}

.progress-item[data-state="complete"] .progress-status {
  color: var(--good);
}

.progress-item[data-state="failed"] .progress-status {
  color: var(--bad);
}
```

At the existing mobile breakpoint, use two columns and place status on a new
row if required. Do not add decorative gradients, floating sections, or nested
cards.

- [ ] **Step 9: Run frontend and HTTP regression tests**

Run:

```bash
pytest tests/test_webui.py -q
```

Expected: all WebUI tests pass, including legacy synchronous endpoint tests.

- [ ] **Step 10: Commit the incremental frontend**

```bash
git add bayesprobe/webui_static/index.html bayesprobe/webui_static/app.js bayesprobe/webui_static/styles.css tests/test_webui.py
git commit -m "feat: render live BayesProbe progress"
```

---

### Task 4: Architecture Record and End-to-End Verification

**Files:**
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/superpowers/specs/2026-07-10-webui-streaming-progress-design.md`
- Test: complete repository and browser behavior

**Interfaces:**
- Documents: runner observation contract and NDJSON route
- Verifies: deterministic, fake provider, API-key lifecycle, desktop/mobile

- [ ] **Step 1: Update the architecture document**

Record that the WebUI now has a streaming observation adapter, while the core
remains transport-agnostic. Add these explicit limitations:

- progress is phase/cycle streaming, not token streaming;
- an HTTP disconnect does not cooperatively cancel an in-flight provider call;
- only autonomous WebUI runs stream in M0.10;
- credentials remain request-scoped and page-memory-only.

Update the capability matrix without changing the overall paradigm claims.

- [ ] **Step 2: Mark the design implemented only after verification**

Change the spec status to `Implemented and verified` and add the actual pytest
count plus browser verification record after those commands have run. Do not
write expected counts in advance.

- [ ] **Step 3: Run the focused suites**

```bash
pytest tests/test_question_runner.py tests/test_webui.py tests/test_public_api_and_config.py -q
```

Expected: all selected tests pass.

- [ ] **Step 4: Run the complete suite**

```bash
pytest -q
```

Expected: zero failures; only explicitly gated live-provider tests may skip.

- [ ] **Step 5: Restart the local WebUI on port 8766**

Stop the existing BayesProbe server cleanly, then run:

```bash
python3 -m bayesprobe.webui --host 127.0.0.1 --port 8766
```

Verify `GET http://127.0.0.1:8766/` returns `200`.

- [ ] **Step 6: Verify key retention and event rendering in a real browser**

Using the browser DOM and stable locators:

1. Select Chat Completions and enter a non-secret test value in the API-key
   input.
2. Switch to Deterministic, run the existing A-E graph question with one cycle,
   then switch back to Chat Completions.
3. Verify the API-key field still contains the test value.
4. Verify progress rows include initialization, probe planning/execution,
   evidence integration, posterior update, and completion.
5. Verify best answer D, posterior mass `1.000`, terminal run status
   `completed`, and cycle status `integrated`.
6. Reload and verify the API-key field is empty.

- [ ] **Step 7: Verify incremental delivery with a delayed fake provider**

Start a test-only WebUI server using
`create_handler_class(client_factory=DelayedFakeChatOpenAI)` where the fake
provider sleeps briefly before returning valid structured responses. In the
browser, confirm `Executing probes` is visible before the request completes and
that the same page later renders the integrated cycle. Do not add this fake
provider to production configuration.

- [ ] **Step 8: Check desktop and mobile layout**

Capture and inspect the default desktop viewport and a temporary 390 by 844
viewport. Assert:

- document scroll width does not exceed client width;
- progress labels and status do not overlap;
- result panels remain readable;
- long cycle JSON stays inside its scrollable trace block.

Reset the temporary viewport before finishing.

- [ ] **Step 9: Run repository hygiene checks**

```bash
git diff --check
git status --short
```

Search tracked files for the exact test credential literals used during manual
verification and fail the check if any are present.

- [ ] **Step 10: Commit documentation and verification record**

```bash
git add docs/ARCHITECTURE.md docs/superpowers/specs/2026-07-10-webui-streaming-progress-design.md
git commit -m "docs: record WebUI streaming milestone"
```

- [ ] **Step 11: Push the verified main branch**

```bash
git push origin main
```

Confirm `git rev-parse HEAD origin/main` prints the same commit twice and leave
the WebUI server running at `http://127.0.0.1:8766` for manual testing.
