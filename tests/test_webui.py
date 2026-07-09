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
