from contextlib import contextmanager
import json
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

import bayesprobe.webui as webui
from bayesprobe.webui import (
    create_handler_class,
    handle_autonomous_run_request,
)


STATIC_DIR = Path(__file__).resolve().parents[1] / "bayesprobe" / "webui_static"


@contextmanager
def serve_webui():
    server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler_class())
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def request_http(method, path, body=None, headers=None):
    with serve_webui() as address:
        connection = HTTPConnection(*address)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            payload = response.read()
        finally:
            connection.close()
    return response.status, response.getheader("Content-Type"), payload


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
        thread.join(timeout=5)
        server.server_close()

    assert response.status == 200
    assert "BayesProbe" in body


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


def test_webui_static_index_declares_inline_favicon_to_avoid_browser_404():
    index = (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    assert 'rel="icon"' in index
    assert 'href="data:,"' in index


def test_webui_static_script_clears_stale_run_output_on_submit_and_failure():
    script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert "function clearRunOutput(" in script
    assert 'clearRunOutput("running");' in script
    assert 'clearRunOutput("failed");' in script
    assert script.index('clearRunOutput("running");') < script.index(
        "const response = await fetch('/api/runs/autonomous',"
    )
    assert script.index('clearRunOutput("failed");') < script.index(
        'setStatus(error.message || "Run failed", "error");'
    )
    assert 'runId.textContent = failed ? "Last run failed." : "Run pending.";' in script
    assert "Run failed. No answer projection." in script
    assert "Run failed. No belief state." in script
    assert "Run failed. Cycle trace unavailable." in script


@pytest.mark.parametrize(
    ("path", "content_type", "expected_body"),
    [
        ("/styles.css", "text/css; charset=utf-8", ".trace-item"),
        ("/app.js", "text/javascript; charset=utf-8", "fetch('/api/runs/autonomous'"),
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
            "timeout": 11.0,
            "base_url": "https://provider.example/v1",
        }
    ]
    assert "sk-webui-secret" not in json.dumps(payload)
    assert payload["cycles"][0]["evidence_events"][0]["model_trace"]["adapter_kind"] == "openai"


class FailingWebUIOpenAI:
    def __init__(self, **kwargs):
        self.responses = self

    def create(self, **payload):
        raise RuntimeError("provider rejected key sk-webui-secret")


class FailingInitWebUIOpenAI:
    def __init__(self, **kwargs):
        raise RuntimeError("provider rejected key sk-webui-secret during init")


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


def test_webui_openai_responses_invalid_timeout_is_validation_error():
    status, payload = handle_autonomous_run_request(
        {
            "question": "Does invalid provider config stay a validation error?",
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
    status, payload = handle_autonomous_run_request(
        {
            "question": "Will provider init errors leak secrets?",
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
    assert "sk-webui-secret" not in json.dumps(payload)
