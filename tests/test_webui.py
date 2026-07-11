from contextlib import contextmanager
import json
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
import shutil
import subprocess
from threading import Event, Thread

import pytest

import bayesprobe.webui as webui
from bayesprobe.webui import (
    create_handler_class,
    handle_autonomous_run_request,
    handle_autonomous_stream_request,
)


STATIC_DIR = Path(__file__).resolve().parents[1] / "bayesprobe" / "webui_static"
STREAM_BEHAVIOR_TEST = Path(__file__).with_name("test_webui_stream.js")


@contextmanager
def serve_webui(client_factory=None):
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), create_handler_class(client_factory=client_factory)
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def request_http(method, path, body=None, headers=None, client_factory=None):
    with serve_webui(client_factory=client_factory) as address:
        connection = HTTPConnection(*address)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            payload = response.read()
        finally:
            connection.close()
    return response.status, response.getheader("Content-Type"), payload


def serve_test_server(client_factory=None):
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), create_handler_class(client_factory=client_factory)
    )
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


def deterministic_answer_choices() -> list[dict[str, str]]:
    return [
        {"label": "H1", "text": "The deterministic fixture supports H1."},
        {"label": "H2", "text": "The deterministic fixture supports H2."},
    ]


def test_webui_deterministic_autonomous_run_returns_trace():
    status, payload = handle_autonomous_run_request(
        {
            "question": "Does the autonomous WebUI path expose trace state?",
            "answer_choices": deterministic_answer_choices(),
            "context": "SUPPORTS: local deterministic run should favor H1.",
            "provider": {"kind": "deterministic"},
            "runner": {"max_cycles": 1, "max_probes_per_cycle": 1},
        }
    )

    assert status == 200
    assert payload["run_id"].startswith("webui_")
    assert payload["stop_reason"] == "max_cycles"
    assert payload["run"]["status"] == "completed"
    assert payload["run"]["regime"] == "autonomous"
    assert payload["run"]["current_cycle_id"] == payload["cycles"][0]["cycle_id"]
    assert payload["final_answer"]["current_best_hypothesis"] == "H1"
    assert payload["initial_belief_state"]["cycle_id"] == "cycle_0"
    assert payload["final_belief_state"]["cycle_index"] == 1
    assert payload["final_belief_state"]["posterior_summary"][
        "total_active_posterior"
    ] == pytest.approx(1.0)
    assert "no external signals" not in payload["final_belief_state"][
        "uncertainty_summary"
    ]
    assert len(payload["cycles"]) == 1
    cycle = payload["cycles"][0]
    assert cycle["signal_shape"] == "active_plus_passive"
    assert cycle["cycle"]["boundary_status"] == "integrated"
    assert cycle["cycle"]["boundary_closed_at"] is not None
    assert cycle["cycle"]["completed_at"] is not None
    assert [signal["signal_kind"] for signal in cycle["signals"]] == [
        "active",
        "passive",
    ]
    assert cycle["probes"]
    assert cycle["signals"]
    assert cycle["evidence_events"]
    assert cycle["belief_updates"]
    assert cycle["answer_projection"]["current_best_hypothesis"] == "H1"


def test_webui_stream_emits_ordered_cycle_and_terminal_events():
    events = []
    status, error = handle_autonomous_stream_request(
        {
            "question": "Does the stream expose a completed BayesProbe cycle?",
            "answer_choices": deterministic_answer_choices(),
            "context": "SUPPORTS: deterministic context favors H1.",
            "provider": {"kind": "deterministic"},
            "runner": {"max_cycles": 1, "max_probes_per_cycle": 1},
        },
        event_sink=events.append,
    )

    assert status == 200
    assert error is None
    assert [event["sequence"] for event in events] == list(range(1, len(events) + 1))
    assert events[0]["event"] == "run_started"
    assert events[-2]["event"] == "cycle_integrated"
    assert events[-1]["event"] == "run_completed"
    cycle = events[-2]["data"]
    assert cycle["cycle"]["boundary_status"] == "integrated"
    assert cycle["belief_state"]["posterior_summary"][
        "total_active_posterior"
    ] == pytest.approx(1.0)


def test_webui_stream_phase_payloads_match_bounded_contract():
    events = []
    status, error = handle_autonomous_stream_request(
        {
            "question": "Does each phase expose only its bounded payload?",
            "answer_choices": deterministic_answer_choices(),
            "context": "SUPPORTS: Include one passive signal.",
            "provider": {"kind": "deterministic"},
            "runner": {"max_cycles": 1, "max_probes_per_cycle": 1},
        },
        event_sink=events.append,
    )

    assert status == 200
    assert error is None
    assert all(
        set(event)
        == {
            "event",
            "sequence",
            "timestamp",
            "run_id",
            "cycle_id",
            "cycle_index",
            "data",
        }
        for event in events
    )
    event_by_name = {event["event"]: event for event in events}
    assert set(event_by_name["run_started"]["data"]) == set()
    assert set(event_by_name["initialization_completed"]["data"]) == {
        "run",
        "belief_state",
    }
    assert set(event_by_name["cycle_started"]["data"]) == {"belief_summary"}
    assert set(event_by_name["cycle_started"]["data"]["belief_summary"]) == {
        "posterior_summary",
        "uncertainty_summary",
    }
    assert set(event_by_name["probe_set_planned"]["data"]) == {"probe_set"}
    assert set(event_by_name["probe_execution_started"]["data"]) == {
        "probe_count"
    }
    assert event_by_name["probe_execution_started"]["data"]["probe_count"] == 1
    assert set(event_by_name["signals_collected"]["data"]) == {"signals"}
    assert len(event_by_name["signals_collected"]["data"]["signals"]) == 2
    assert set(event_by_name["evidence_integration_started"]["data"]) == {
        "signal_count"
    }
    assert event_by_name["evidence_integration_started"]["data"][
        "signal_count"
    ] == 2
    assert set(event_by_name["cycle_integrated"]["data"]) == {
        "cycle_id",
        "signal_shape",
        "cycle",
        "probes",
        "signals",
        "belief_state",
        "evidence_events",
        "belief_updates",
        "hypothesis_evolutions",
        "answer_projection",
    }
    assert set(event_by_name["run_completed"]["data"]) == {
        "run_id",
        "run",
        "stop_reason",
        "final_answer",
        "initial_belief_state",
        "final_belief_state",
        "cycles",
    }


def test_webui_unseeded_open_question_is_framing_validation_before_provider_execution():
    FakeWebUIChatOpenAI.created_with = []
    events = []
    request = {
        "question": "Can streaming use a Chat Completions provider?",
        "provider": {
            "kind": "openai_chat_completions",
            "api_key": "provider-secret-123",
            "model": "provider-model",
        },
        "runner": {"max_cycles": 1, "max_probes_per_cycle": 1},
    }
    status, response = handle_autonomous_run_request(
        request,
        client_factory=FakeWebUIChatOpenAI,
    )
    stream_status, error = handle_autonomous_stream_request(
        request,
        event_sink=events.append,
        client_factory=FakeWebUIChatOpenAI,
    )

    expected_error = {
        "error": {
            "type": "validation_error",
            "message": "task framing requires explicit answer choices or hypothesis seeds",
        }
    }
    assert status == 400
    assert response == expected_error
    assert stream_status == 400
    assert error == expected_error
    assert events == []
    assert FakeWebUIChatOpenAI.created_with == []


def test_webui_invalid_explicit_frame_is_validation_error_before_provider_execution():
    FakeWebUIChatOpenAI.created_with = []

    status, response = handle_autonomous_run_request(
        {
            "question": "Does one choice form a valid task frame?",
            "answer_choices": [{"label": "A", "text": "Only choice"}],
            "provider": {
                "kind": "openai_chat_completions",
                "api_key": "provider-secret-123",
                "model": "provider-model",
            },
        },
        client_factory=FakeWebUIChatOpenAI,
    )

    assert status == 400
    assert response["error"]["type"] == "validation_error"
    assert FakeWebUIChatOpenAI.created_with == []


def test_webui_materializes_explicit_frames_once_and_unseeded_frames_never(monkeypatch):
    materializations = 0
    original_frame = webui.ExplicitTaskFramer.frame

    def count_materializations(self, input):
        nonlocal materializations
        materializations += 1
        return original_frame(self, input)

    monkeypatch.setattr(webui.ExplicitTaskFramer, "frame", count_materializations)

    status, _ = handle_autonomous_run_request(
        {
            "question": "Does the explicit WebUI path materialize once?",
            "answer_choices": deterministic_answer_choices(),
            "runner": {"max_cycles": 1, "max_probes_per_cycle": 1},
        }
    )

    assert status == 200
    assert materializations == 1

    materializations = 0
    status, _ = handle_autonomous_run_request(
        {"question": "How should this claim be tested?"}
    )

    assert status == 400
    assert materializations == 0


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
    FailingWebUIOpenAI.calls = 0
    events = []
    status, error = handle_autonomous_stream_request(
        {
            "question": "Will a streaming provider failure stay sanitized?",
            "answer_choices": deterministic_answer_choices(),
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
    assert FailingWebUIOpenAI.calls == 1


def test_webui_deterministic_structured_choices_select_explicit_choice():
    status, payload = handle_autonomous_run_request(
        {
            "question": "Which graph class is well-behaved?",
            "answer_choices": [
                {"label": "A", "text": "Non-bipartite regular graphs"},
                {"label": "B", "text": "Connected cubic graphs"},
                {"label": "C", "text": "Connected graphs"},
                {"label": "D", "text": "Connected non-bipartite graphs"},
                {"label": "E", "text": "Connected bipartite graphs"},
            ],
            "context": (
                "SUPPORTS D: The chain is irreducible exactly when the connected "
                "graph is non-bipartite, while self-loops make it aperiodic."
            ),
            "provider": {"kind": "deterministic"},
            "runner": {"max_cycles": 1, "max_probes_per_cycle": 1},
        }
    )

    assert status == 200
    assert [
        hypothesis["id"]
        for hypothesis in payload["final_belief_state"]["hypotheses"]
    ] == ["A", "B", "C", "D", "E"]
    assert payload["final_answer"]["current_best_hypothesis"] == "D"
    assert sum(
        hypothesis["posterior"]
        for hypothesis in payload["final_belief_state"]["hypotheses"]
    ) == pytest.approx(1.0)


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
            {
                "question": "Q",
                "answer_choices": deterministic_answer_choices(),
                "provider": {"kind": "openai_chat_completions", "model": "gpt-5.5"},
            },
            "provider.api_key must not be empty",
        ),
    ],
)
def test_webui_autonomous_run_rejects_invalid_payloads(payload, expected_message):
    status, response = handle_autonomous_run_request(payload)

    assert status == 400
    assert response["error"]["message"] == expected_message


def test_webui_http_server_serves_static_index():
    server, thread = serve_test_server()
    try:
        conn = HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        conn.request("GET", "/")
        response = conn.getresponse()
        body = response.read().decode("utf-8")
        cache_control = response.getheader("Cache-Control")
        conn.close()
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert response.status == 200
    assert "BayesProbe" in body
    assert cache_control == "no-store"


def test_webui_static_assets_define_operational_workbench():
    index = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    styles = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")
    script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert "BayesProbe" in index
    assert "provider-kind" in index
    assert "Chat Completions (unsupported)" not in index
    assert '<option value="openai_chat_completions" selected>Chat Completions</option>' in index
    assert "api-key" in index
    assert "base-url" in index
    assert 'value="https://api.deepseek.com"' in index
    assert "model-name" in index
    assert 'value="deepseek-v4-flash"' in index
    assert 'id="timeout-seconds" type="number" min="360" value="360"' in index
    assert 'id="max-output-tokens" type="number" min="1" value="32768"' in index
    assert "max-cycles" in index
    assert "trace-pane" in index
    assert 'id="progress-panel"' in index
    assert 'id="progress-list"' in index
    assert 'id="answer-projection-state"' in index
    assert 'id="context" placeholder="Optional external information"' in index
    assert "SUPPORTS: The local deterministic signal supports H1." not in index
    assert "localStorage" not in script
    assert "sessionStorage" not in script
    assert "document.cookie" not in script
    assert "clearApiKey" not in script
    assert 'apiKeyField.value = ""' not in script
    assert "fetch('/api/runs/autonomous/stream'" in script
    assert "response.body.getReader()" in script
    assert "new TextDecoder()" in script
    assert "function handleProgressEvent(" in script
    assert "cycle_integrated" in script
    assert "run_completed" in script
    assert "run_failed" in script
    assert "Responses-compatible providers only." in script
    assert "Check base URL, model, API key, and max output tokens." in script
    assert "Best answer / hypothesis" in script
    assert "Posterior mass" in script
    assert "Cycle lifecycle" in script
    assert "boundary_status" in script
    assert "Chat Completions stays visible in v0.1 but is not supported." not in script
    assert 'provider.kind === "openai_chat_completions"' in script
    assert 'providerKind.value === "openai_chat_completions"' not in script
    assert ".trace-item" in styles
    assert ".progress-list" in styles
    assert ".progress-item" in styles
    assert "@media" in styles


def test_webui_trace_css_contains_long_json_within_each_pre():
    styles = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")

    assert (
        ".trace-pane,\n"
        ".trace-item {\n"
        "  min-width: 0;\n"
        "  max-width: 100%;\n"
        "}" in styles
    )
    assert (
        ".trace-stack,\n"
        ".trace-stack > section,\n"
        "pre {\n"
        "  min-width: 0;\n"
        "  max-width: 100%;\n"
        "}" in styles
    )
    pre_styles = styles[styles.index("pre {") :]
    assert "  width: 100%;\n" in pre_styles
    assert "  overflow: auto;\n" in pre_styles


def test_webui_static_index_declares_inline_favicon_to_avoid_browser_404():
    index = (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    assert 'rel="icon"' in index
    assert 'href="data:,"' in index


def test_webui_static_script_preserves_streamed_output_after_a_late_failure():
    script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert "function clearRunOutput(" in script
    assert 'clearRunOutput("running");' in script
    assert "streamedCycles.length === 0" in script
    assert 'clearRunOutput("failed");' in script
    assert script.index("streamedCycles.length === 0") < script.index(
        'clearRunOutput("failed");'
    )
    assert 'runId.textContent = failed ? "Last run failed." : "Run pending.";' in script
    assert "Run failed. No answer projection." in script
    assert "Run failed. No belief state." in script
    assert "Run failed. Cycle trace unavailable." in script


def test_webui_frontend_stream_behavior():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not available for frontend stream behavior tests")

    result = subprocess.run(
        [node, "--test", str(STREAM_BEHAVIOR_TEST)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("path", "content_type", "expected_body"),
        [
            ("/styles.css", "text/css; charset=utf-8", ".trace-item"),
            (
                "/app.js",
                "text/javascript; charset=utf-8",
                "fetch('/api/runs/autonomous/stream'",
            ),
        ],
)
def test_webui_http_server_serves_static_assets(path, content_type, expected_body):
    status, response_content_type, payload = request_http("GET", path)

    assert status == 200
    assert response_content_type == content_type
    assert expected_body in payload.decode("utf-8")


def test_webui_http_server_handles_autonomous_run_post():
    status, content_type, payload = request_http(
        "POST",
        "/api/runs/autonomous",
        body=json.dumps(
            {
                "question": "Does the HTTP handler complete a deterministic run?",
                "answer_choices": deterministic_answer_choices(),
                "context": "SUPPORTS: local deterministic run should favor H1.",
                "provider": {"kind": "deterministic"},
                "runner": {"max_cycles": 1, "max_probes_per_cycle": 1},
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    response = json.loads(payload)

    assert status == 200
    assert content_type == "application/json; charset=utf-8"
    assert response["run_id"].startswith("webui_")
    assert response["stop_reason"] == "max_cycles"
    assert response["cycles"][0]["evidence_events"]


def test_webui_http_server_streams_valid_ndjson():
    status, content_type, payload = request_http(
        "POST",
        "/api/runs/autonomous/stream",
        body=json.dumps(
            {
                "question": "Does HTTP expose progress?",
                "answer_choices": deterministic_answer_choices(),
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


@pytest.mark.parametrize("disconnect_error", [BrokenPipeError, ConnectionResetError])
def test_webui_ndjson_writer_contains_client_disconnects(disconnect_error):
    class DisconnectingStream:
        def __init__(self):
            self.write_calls = 0
            self.flush_calls = 0

        def write(self, payload):
            del payload
            self.write_calls += 1
            raise disconnect_error("client disconnected")

        def flush(self):
            self.flush_calls += 1

    class RecordingHandler:
        def __init__(self):
            self.wfile = DisconnectingStream()
            self.responses = []
            self.headers = []
            self.headers_ended = 0

        def send_response(self, status):
            self.responses.append(status)

        def send_header(self, name, value):
            self.headers.append((name, value))

        def end_headers(self):
            self.headers_ended += 1

    handler = RecordingHandler()
    writer = webui._NDJSONEventWriter(handler)

    status, error = handle_autonomous_stream_request(
        {
            "question": "Does a disconnected client leave the runner intact?",
            "answer_choices": deterministic_answer_choices(),
            "provider": {"kind": "deterministic"},
            "runner": {"max_cycles": 1, "max_probes_per_cycle": 1},
        },
        event_sink=writer.emit,
    )

    assert status == 200
    assert error is None
    assert writer.started is True
    assert writer.disconnected is True
    assert handler.responses == [200]
    assert handler.headers_ended == 1
    assert handler.wfile.write_calls == 1
    assert handler.wfile.flush_calls == 0


def test_webui_handler_returns_invalid_json_for_malformed_request_body():
    status, content_type, payload = request_http(
        "POST",
        "/api/runs/autonomous",
        body=b"{",
        headers={"Content-Length": "1", "Content-Type": "application/json"},
    )

    assert status == 400
    assert content_type == "application/json; charset=utf-8"
    assert json.loads(payload) == {
        "error": {
            "type": "invalid_json",
            "message": "request body must be valid JSON",
        }
    }


def test_webui_stream_handler_returns_invalid_json_for_malformed_request_body():
    status, content_type, payload = request_http(
        "POST",
        "/api/runs/autonomous/stream",
        body=b"{",
        headers={"Content-Length": "1", "Content-Type": "application/json"},
    )

    assert status == 400
    assert content_type == "application/json; charset=utf-8"
    assert json.loads(payload) == {
        "error": {
            "type": "invalid_json",
            "message": "request body must be valid JSON",
        }
    }


def test_webui_handler_returns_server_error_for_unexpected_post_failure(monkeypatch):
    def boom(payload, *, client_factory=None):
        del payload, client_factory
        raise RuntimeError("traceback-worthy secret at /tmp/private sk-test123")

    monkeypatch.setattr(webui, "handle_autonomous_run_request", boom)

    status, content_type, payload = request_http(
        "POST",
        "/api/runs/autonomous",
        body=json.dumps({"question": "Q"}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    assert status == 500
    assert content_type == "application/json; charset=utf-8"
    assert json.loads(payload) == {
        "error": {
            "type": "server_error",
            "message": "internal server error",
        }
    }


def test_webui_handler_sanitizes_static_file_read_failures(monkeypatch):
    original_is_file = webui.Path.is_file
    original_read_bytes = webui.Path.read_bytes

    def fake_is_file(path):
        if path == webui.STATIC_DIR / "index.html":
            return True
        return original_is_file(path)

    def failing_read_bytes(path):
        if path == webui.STATIC_DIR / "index.html":
            raise OSError(
                "permission denied reading /Users/dengjianbo/Documents/BayesProbe/secret.txt"
            )
        return original_read_bytes(path)

    monkeypatch.setattr(webui.Path, "is_file", fake_is_file)
    monkeypatch.setattr(webui.Path, "read_bytes", failing_read_bytes)

    status, content_type, payload = request_http("GET", "/")

    assert status == 500
    assert content_type == "application/json; charset=utf-8"
    assert json.loads(payload) == {
        "error": {
            "type": "server_error",
            "message": "internal server error",
        }
    }


class FakeWebUIResponses:
    def __init__(self):
        self.calls = []

    def create(self, **payload):
        self.calls.append(payload)
        request = json.loads(payload["input"][1]["content"])
        if request["task"] == "execute_probe":
            response = {
                "raw_content": "The model executed the requested active probe."
            }
        else:
            response = {
                "evidence_type": "supporting",
                "likelihoods": {
                    "H1": "moderately_confirming",
                    "H2": "moderately_disconfirming",
                },
                "interpretation": "WebUI fake OpenAI response.",
                "quality_overrides": {},
            }
        return json.dumps(response)


class FakeWebUIOpenAI:
    created_with = []
    responses = None

    def __init__(self, **kwargs):
        self.__class__.created_with.append(kwargs)
        responses = FakeWebUIResponses()
        self.__class__.responses = responses
        self.responses = responses


class FakeWebUIChatCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **payload):
        self.calls.append(payload)
        request = json.loads(payload["messages"][1]["content"])
        if request["task"] == "execute_probe":
            content = {
                "raw_content": "The chat model executed the requested active probe."
            }
        else:
            content = {
                "evidence_type": "supporting",
                "likelihoods": {
                    "H1": "moderately_confirming",
                    "H2": "moderately_disconfirming",
                },
                "interpretation": "WebUI fake chat response.",
                "quality_overrides": {},
            }
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(content)
                    }
                }
            ]
        }


class FakeWebUIChatOpenAI:
    created_with = []
    completions = None

    def __init__(self, **kwargs):
        self.__class__.created_with.append(kwargs)
        completions = FakeWebUIChatCompletions()
        self.__class__.completions = completions
        self.chat = type(
            "FakeChat",
            (),
            {"completions": completions},
        )()


class FailDuringSecondCycleWebUIChatCompletions(FakeWebUIChatCompletions):
    def __init__(self):
        super().__init__()
        self.tasks = []
        self.execute_calls = 0

    def create(self, **payload):
        request = json.loads(payload["messages"][1]["content"])
        self.tasks.append(request["task"])
        if request["task"] == "execute_probe":
            self.execute_calls += 1
            if self.execute_calls == 2:
                raise RuntimeError("cycle two provider failure")
        return super().create(**payload)


class FailDuringSecondCycleWebUIChatOpenAI:
    completions = None

    def __init__(self, **kwargs):
        del kwargs
        completions = FailDuringSecondCycleWebUIChatCompletions()
        self.__class__.completions = completions
        self.chat = type("FakeChat", (), {"completions": completions})()


def test_webui_stream_preserves_real_first_cycle_before_second_cycle_failure():
    events = []
    status, error = handle_autonomous_stream_request(
        {
            "question": "Can a genuine first cycle survive a cycle two failure?",
            "answer_choices": deterministic_answer_choices(),
            "provider": {
                "kind": "openai_chat_completions",
                "api_key": "provider-secret-123",
                "model": "provider-model",
            },
            "runner": {"max_cycles": 2, "max_probes_per_cycle": 1},
        },
        event_sink=events.append,
        client_factory=FailDuringSecondCycleWebUIChatOpenAI,
    )

    integrated = [event for event in events if event["event"] == "cycle_integrated"]
    failures = [event for event in events if event["event"] == "run_failed"]
    assert status == 200
    assert error is None
    assert FailDuringSecondCycleWebUIChatOpenAI.completions.tasks == [
        "execute_probe",
        "judge_evidence",
        "execute_probe",
    ]
    assert len(integrated) == 1
    assert integrated[0]["cycle_index"] == 1
    assert integrated[0]["data"]["cycle"]["boundary_status"] == "integrated"
    assert len(failures) == 1
    assert failures[0]["cycle_index"] == 2
    assert failures[0]["cycle_id"].endswith("_cycle_2")
    assert set(failures[0]["data"]) == {"error"}
    assert set(failures[0]["data"]["error"]) == {"type", "message"}
    assert events[-1] == failures[0]
    assert all(event["event"] != "run_completed" for event in events)


class BlockingWebUIChatCompletions:
    provider_entered = Event()
    release_provider = Event()

    def create(self, **payload):
        request = json.loads(payload["messages"][1]["content"])
        if request["task"] == "execute_probe":
            self.provider_entered.set()
            assert self.release_provider.wait(timeout=5)
            content = {"raw_content": "The provider completed the active probe."}
        else:
            content = {
                "evidence_type": "supporting",
                "likelihoods": {
                    "H1": "moderately_confirming",
                    "H2": "moderately_disconfirming",
                },
                "interpretation": "The provider judged the active probe.",
                "quality_overrides": {},
            }
        return {"choices": [{"message": {"content": json.dumps(content)}}]}


class BlockingWebUIChatOpenAI:
    def __init__(self, **kwargs):
        del kwargs
        self.chat = type(
            "FakeChat",
            (),
            {"completions": BlockingWebUIChatCompletions()},
        )()


def test_webui_stream_flushes_first_event_before_provider_completion():
    BlockingWebUIChatCompletions.provider_entered.clear()
    BlockingWebUIChatCompletions.release_provider.clear()
    server, thread = serve_test_server(client_factory=BlockingWebUIChatOpenAI)
    connection = HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
    try:
        connection.request(
            "POST",
            "/api/runs/autonomous/stream",
            body=json.dumps(
                {
                    "question": "Does streaming flush before provider completion?",
                    "answer_choices": deterministic_answer_choices(),
                    "provider": {
                        "kind": "openai_chat_completions",
                        "api_key": "provider-secret-123",
                        "model": "provider-model",
                    },
                    "runner": {"max_cycles": 1, "max_probes_per_cycle": 1},
                }
            ),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        assert BlockingWebUIChatCompletions.provider_entered.wait(timeout=5)
        first_event = json.loads(response.readline())
        assert response.status == 200
        assert first_event["event"] == "run_started"
        assert not BlockingWebUIChatCompletions.release_provider.is_set()

        BlockingWebUIChatCompletions.release_provider.set()
        remaining_events = [json.loads(line) for line in response.read().splitlines()]
        assert remaining_events[-1]["event"] == "run_completed"
    finally:
        BlockingWebUIChatCompletions.release_provider.set()
        connection.close()
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


class FakeChoiceAwareChatCompletions:
    def create(self, **payload):
        user_content = payload["messages"][1]["content"]
        request = json.loads(user_content)
        if request["task"] == "execute_probe":
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "raw_content": (
                                        "Connected non-bipartite graphs make the chain "
                                        "irreducible and aperiodic, so D is correct."
                                    )
                                }
                            )
                        }
                    }
                ]
            }
        targets = request["input"]["target_hypotheses"]
        likelihoods = {
            target: (
                "strongly_confirming"
                if target == "D"
                else "moderately_disconfirming"
            )
            for target in targets
        }
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "evidence_type": "supporting",
                                "likelihoods": likelihoods,
                                "interpretation": "Choice D is the well-behaved graph class.",
                                "quality_overrides": {},
                            }
                        )
                    }
                }
            ]
        }


class FakeChoiceAwareChatOpenAI:
    def __init__(self, **kwargs):
        self.chat = type(
            "FakeChat",
            (),
            {"completions": FakeChoiceAwareChatCompletions()},
        )()


class FakeProbeAwareChatCompletions:
    def __init__(self):
        self.tasks = []

    def create(self, **payload):
        request = json.loads(payload["messages"][1]["content"])
        task = request["task"]
        self.tasks.append(task)
        if task == "execute_probe":
            content = {
                "raw_content": (
                    "Comparing irreducibility and aperiodicity across the choices "
                    "supports D and rules out the bipartite classes."
                )
            }
        else:
            targets = request["input"]["target_hypotheses"]
            content = {
                "evidence_type": "supporting",
                "likelihoods": {
                    target: (
                        "strongly_confirming"
                        if target == "D"
                        else "moderately_disconfirming"
                    )
                    for target in targets
                },
                "interpretation": "The model-generated probe signal favors D.",
                "quality_overrides": {},
            }
        return {"choices": [{"message": {"content": json.dumps(content)}}]}


class FakeProbeAwareChatOpenAI:
    completions = None

    def __init__(self, **kwargs):
        del kwargs
        completions = FakeProbeAwareChatCompletions()
        self.__class__.completions = completions
        self.chat = type("FakeChat", (), {"completions": completions})()


def test_webui_provider_executes_active_probe_before_judging_evidence():
    question = """Which option follows from the graph-chain conditions?

Answer Choices:
A. Regularity alone
B. Connectedness alone
C. Bipartiteness
D. Connectedness and non-bipartiteness
E. Cubicity alone"""

    status, payload = handle_autonomous_run_request(
        {
            "question": question,
            "provider": {
                "kind": "openai_chat_completions",
                "api_key": "provider-secret-123",
                "base_url": "https://provider.example/v1",
                "model": "provider-model",
            },
            "runner": {"max_cycles": 1, "max_probes_per_cycle": 1},
        },
        client_factory=FakeProbeAwareChatOpenAI,
    )

    assert status == 200
    assert FakeProbeAwareChatOpenAI.completions.tasks == [
        "execute_probe",
        "judge_evidence",
    ]
    assert payload["cycles"][0]["signals"][0]["source_type"] == (
        "model_probe_gateway"
    )
    assert payload["cycles"][0]["signals"][0]["raw_content"].startswith(
        "Comparing irreducibility"
    )
    assert payload["final_answer"]["current_best_hypothesis"] == "D"


def test_webui_openai_responses_provider_uses_request_key_and_redacts_response():
    FakeWebUIOpenAI.created_with = []

    status, payload = handle_autonomous_run_request(
        {
            "question": "Can the WebUI use a provider-backed evidence judgment?",
            "answer_choices": deterministic_answer_choices(),
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
            "timeout": 360.0,
            "base_url": "https://provider.example/v1",
        }
    ]
    assert "sk-webui-secret" not in json.dumps(payload)
    assert len(FakeWebUIOpenAI.responses.calls) == 2
    assert payload["cycles"][0]["evidence_events"][0]["model_trace"]["adapter_kind"] == "openai"


def test_webui_openai_chat_completions_provider_uses_request_key_and_redacts_response():
    FakeWebUIChatOpenAI.created_with = []

    status, payload = handle_autonomous_run_request(
        {
            "question": "Can the WebUI use a Chat Completions-compatible provider?",
            "answer_choices": deterministic_answer_choices(),
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
            "timeout": 360.0,
            "base_url": "https://provider.example/v1",
        }
    ]
    assert "provider-secret-123" not in json.dumps(payload)
    assert len(FakeWebUIChatOpenAI.completions.calls) == 2
    assert (
        payload["cycles"][0]["evidence_events"][0]["model_trace"]["adapter_kind"]
        == "openai_chat_completions"
    )


@pytest.mark.parametrize(
    ("kind", "client_factory"),
    [
        ("openai_responses", FakeWebUIOpenAI),
        ("openai_chat_completions", FakeWebUIChatOpenAI),
    ],
)
@pytest.mark.parametrize(
    ("requested_timeout", "expected_timeout"),
    [(None, 360.0), (30, 360.0), (720, 720.0)],
)
def test_webui_provider_timeout_has_360_second_floor(
    kind,
    client_factory,
    requested_timeout,
    expected_timeout,
):
    client_factory.created_with = []
    provider = {
        "kind": kind,
        "api_key": "provider-secret-123",
        "base_url": "https://provider.example/v1",
        "model": "provider-model",
    }
    if requested_timeout is not None:
        provider["timeout_seconds"] = requested_timeout

    webui._build_webui_model_gateway(provider, client_factory=client_factory)

    assert client_factory.created_with == [
        {
            "api_key": "provider-secret-123",
            "timeout": expected_timeout,
            "base_url": "https://provider.example/v1",
        }
    ]


@pytest.mark.parametrize(
    ("requested_tokens", "expected_tokens"),
    [(None, 32768), (8196, 32768), (65536, 65536)],
)
def test_webui_official_deepseek_v4_has_32768_output_token_floor(
    requested_tokens,
    expected_tokens,
):
    provider = {
        "kind": "openai_chat_completions",
        "api_key": "provider-secret-123",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-flash",
    }
    if requested_tokens is not None:
        provider["max_output_tokens"] = requested_tokens

    gateway = webui._build_webui_model_gateway(
        provider,
        client_factory=FakeWebUIChatOpenAI,
    )

    assert gateway.config.max_output_tokens == expected_tokens


def test_webui_generic_chat_provider_preserves_lower_output_token_budget():
    gateway = webui._build_webui_model_gateway(
        {
            "kind": "openai_chat_completions",
            "api_key": "provider-secret-123",
            "base_url": "https://provider.example/v1",
            "model": "provider-model",
            "max_output_tokens": 8196,
        },
        client_factory=FakeWebUIChatOpenAI,
    )

    assert gateway.config.max_output_tokens == 8196


def test_webui_multiple_choice_question_returns_answer_choice_projection():
    question = """Which graph class is well-behaved?

Answer Choices:
A. The class of all non-bipartite regular graphs
B. The class of all connected cubic graphs
C. The class of all connected graphs
D. The class of all connected non-bipartite graphs
E. The class of all connected bipartite graphs."""

    status, payload = handle_autonomous_run_request(
        {
            "question": question,
            "provider": {
                "kind": "openai_chat_completions",
                "api_key": "provider-secret-123",
                "base_url": "https://provider.example/v1",
                "model": "provider-model",
            },
            "runner": {"max_cycles": 1, "max_probes_per_cycle": 1},
        },
        client_factory=FakeChoiceAwareChatOpenAI,
    )

    assert status == 200
    assert payload["final_answer"]["current_best_hypothesis"] == "D"
    assert payload["final_answer"]["answer"].startswith("Current best answer is D:")
    assert "connected non-bipartite graphs" in payload["final_answer"]["answer"]
    assert payload["cycles"][0]["probes"][0]["target_hypotheses"] == [
        "A",
        "B",
        "C",
        "D",
        "E",
    ]


class FailingWebUIOpenAI:
    calls = 0
    def __init__(self, **kwargs):
        self.responses = self

    def create(self, **payload):
        self.__class__.calls += 1
        raise RuntimeError("provider rejected key sk-webui-secret")


class FailingInitWebUIOpenAI:
    attempts = 0
    def __init__(self, **kwargs):
        self.__class__.attempts += 1
        raise RuntimeError("provider rejected key sk-webui-secret during init")


class FailingNonOpenAIShapedSecretWebUIOpenAI:
    calls = 0
    def __init__(self, **kwargs):
        self.responses = self

    def create(self, **payload):
        self.__class__.calls += 1
        raise RuntimeError("provider rejected key provider-secret-123")


class FailingNonOpenAIShapedSecretInitWebUIOpenAI:
    attempts = 0
    def __init__(self, **kwargs):
        self.__class__.attempts += 1
        raise RuntimeError("provider rejected key provider-secret-123 during init")


def test_webui_provider_errors_are_sanitized():
    FailingWebUIOpenAI.calls = 0
    status, payload = handle_autonomous_run_request(
        {
            "question": "Will provider errors leak secrets?",
            "answer_choices": deterministic_answer_choices(),
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
    assert payload["error"]["message"] == (
        "provider request failed for openai_responses. "
        "Use Chat Completions for /chat/completions-compatible providers."
    )
    assert "sk-webui-secret" not in json.dumps(payload)
    assert FailingWebUIOpenAI.calls == 1


def test_webui_provider_request_failures_return_safe_diagnostic_for_non_sk_keys():
    FailingNonOpenAIShapedSecretWebUIOpenAI.calls = 0
    status, payload = handle_autonomous_run_request(
        {
            "question": "Will provider request failures leak non-sk API keys?",
            "answer_choices": deterministic_answer_choices(),
            "provider": {
                "kind": "openai_responses",
                "api_key": "provider-secret-123",
                "model": "gpt-5.5",
            },
        },
        client_factory=FailingNonOpenAIShapedSecretWebUIOpenAI,
    )

    assert status == 502
    assert payload == {
        "error": {
            "type": "provider_error",
            "message": (
                "provider request failed for openai_responses. "
                "Use Chat Completions for /chat/completions-compatible providers."
            ),
        }
    }
    assert "provider-secret-123" not in json.dumps(payload)
    assert FailingNonOpenAIShapedSecretWebUIOpenAI.calls == 1


def test_webui_openai_responses_invalid_timeout_is_validation_error():
    status, payload = handle_autonomous_run_request(
        {
            "question": "Does invalid provider config stay a validation error?",
            "answer_choices": deterministic_answer_choices(),
            "provider": {
                "kind": "openai_responses",
                "api_key": "sk-webui-secret",
                "model": "gpt-5.5",
                "timeout_seconds": float("nan"),
            },
        },
        client_factory=FakeWebUIOpenAI,
    )

    assert status == 400
    assert payload == {
        "error": {
            "type": "validation_error",
            "message": "openai model gateway timeout_seconds must be finite and positive",
        }
    }
    assert "sk-webui-secret" not in json.dumps(payload)


def test_webui_provider_initialization_errors_are_sanitized():
    FailingInitWebUIOpenAI.attempts = 0
    status, payload = handle_autonomous_run_request(
        {
            "question": "Will provider init errors leak secrets?",
            "answer_choices": deterministic_answer_choices(),
            "provider": {
                "kind": "openai_responses",
                "api_key": "sk-webui-secret",
                "model": "gpt-5.5",
            },
        },
        client_factory=FailingInitWebUIOpenAI,
    )

    assert status == 502
    assert payload["error"]["type"] == "provider_error"
    assert payload["error"]["message"] == (
        "provider request failed for openai_responses. "
        "Use Chat Completions for /chat/completions-compatible providers."
    )
    assert "sk-webui-secret" not in json.dumps(payload)
    assert FailingInitWebUIOpenAI.attempts == 1


def test_webui_provider_initialization_failures_return_safe_diagnostic_for_non_sk_keys():
    FailingNonOpenAIShapedSecretInitWebUIOpenAI.attempts = 0
    status, payload = handle_autonomous_run_request(
        {
            "question": "Will provider init failures leak non-sk API keys?",
            "answer_choices": deterministic_answer_choices(),
            "provider": {
                "kind": "openai_responses",
                "api_key": "provider-secret-123",
                "model": "gpt-5.5",
            },
        },
        client_factory=FailingNonOpenAIShapedSecretInitWebUIOpenAI,
    )

    assert status == 502
    assert payload == {
        "error": {
            "type": "provider_error",
            "message": (
                "provider request failed for openai_responses. "
                "Use Chat Completions for /chat/completions-compatible providers."
            ),
        }
    }
    assert "provider-secret-123" not in json.dumps(payload)
    assert FailingNonOpenAIShapedSecretInitWebUIOpenAI.attempts == 1


class FailingChatWebUIOpenAI:
    calls = 0
    def __init__(self, **kwargs):
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, **payload):
        self.__class__.calls += 1
        raise RuntimeError("provider rejected max token value provider-secret-123")


class LengthExhaustedChatWebUIOpenAI:
    calls = 0
    def __init__(self, **kwargs):
        del kwargs
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, **payload):
        self.__class__.calls += 1
        del payload
        return {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {
                        "content": "",
                        "reasoning_content": "reasoning used the output budget",
                    },
                }
            ]
        }


def test_webui_reports_exhausted_output_budget_without_leaking_provider_data():
    LengthExhaustedChatWebUIOpenAI.calls = 0
    request = {
        "question": "Will an exhausted reasoning budget be diagnosed?",
        "answer_choices": deterministic_answer_choices(),
        "provider": {
            "kind": "openai_chat_completions",
            "api_key": "provider-secret-123",
            "model": "deepseek-v4-flash",
            "base_url": "https://api.deepseek.com",
            "max_output_tokens": 8196,
        },
        "runner": {"max_cycles": 1, "max_probes_per_cycle": 1},
    }

    status, payload = handle_autonomous_run_request(
        request,
        client_factory=LengthExhaustedChatWebUIOpenAI,
    )
    events = []
    stream_status, stream_error = handle_autonomous_stream_request(
        request,
        event_sink=events.append,
        client_factory=LengthExhaustedChatWebUIOpenAI,
    )

    expected_message = (
        "provider exhausted max output tokens before producing structured "
        "content. Increase max output tokens and retry."
    )
    assert status == 502
    assert payload == {
        "error": {"type": "provider_error", "message": expected_message}
    }
    assert stream_status == 200
    assert stream_error is None
    assert events[-1]["event"] == "run_failed"
    assert events[-1]["data"]["error"] == {
        "type": "provider_error",
        "message": expected_message,
    }
    assert "provider-secret-123" not in json.dumps([payload, events])
    assert LengthExhaustedChatWebUIOpenAI.calls == 2


def test_webui_chat_completions_provider_failures_return_safe_diagnostic_hint():
    FailingChatWebUIOpenAI.calls = 0
    status, payload = handle_autonomous_run_request(
        {
            "question": "Will chat provider failures be diagnosable without leaking keys?",
            "answer_choices": deterministic_answer_choices(),
            "provider": {
                "kind": "openai_chat_completions",
                "api_key": "provider-secret-123",
                "model": "deepseek-v4-flash",
                "base_url": "https://api.deepseek.com",
                "max_output_tokens": 102400,
            },
        },
        client_factory=FailingChatWebUIOpenAI,
    )

    assert status == 502
    assert payload == {
        "error": {
            "type": "provider_error",
            "message": (
                "provider request failed for openai_chat_completions. "
                "Check base URL, model, API key, and max output tokens."
            ),
        }
    }
    assert "provider-secret-123" not in json.dumps(payload)
    assert FailingChatWebUIOpenAI.calls == 1


def test_webui_main_rejects_non_loopback_host_before_binding(monkeypatch):
    server_started = False

    def fail_if_called(*args, **kwargs):
        nonlocal server_started
        server_started = True
        raise AssertionError("server should not bind for non-loopback hosts")

    monkeypatch.setattr(webui, "ThreadingHTTPServer", fail_if_called)

    with pytest.raises(SystemExit) as excinfo:
        webui.main(["--host", "0.0.0.0"])

    assert excinfo.value.code == 2
    assert server_started is False
