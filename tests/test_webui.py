from contextlib import contextmanager
import json
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from threading import Thread

import pytest

import bayesprobe.webui as webui
from bayesprobe.webui import (
    create_handler_class,
    handle_autonomous_run_request,
)


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


def request_json(method, path, body=None, headers=None):
    with serve_webui() as address:
        connection = HTTPConnection(*address)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            payload = response.read()
        finally:
            connection.close()
    return response.status, response.getheader("Content-Type"), payload


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


def test_webui_autonomous_run_rejects_openai_responses_until_wired():
    status, response = handle_autonomous_run_request(
        {"question": "Q", "provider": {"kind": "openai_responses"}}
    )

    assert status == 400
    assert response == {
        "error": {
            "type": "unsupported_provider",
            "message": "provider kind openai_responses is not wired yet",
        }
    }


def test_webui_handler_returns_invalid_json_for_malformed_request_body():
    status, content_type, payload = request_json(
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

    status, content_type, payload = request_json(
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

    status, content_type, payload = request_json("GET", "/")

    assert status == 500
    assert content_type == "application/json; charset=utf-8"
    assert json.loads(payload) == {
        "error": {
            "type": "server_error",
            "message": "internal server error",
        }
    }
