from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel

from bayesprobe_terminal_bench.artifacts import TrialArtifactStore


class ProviderEnvelope(BaseModel):
    detail: str
    nested: list[object]


def test_store_redacts_exact_provider_secret_recursively_before_serialization(tmp_path) -> None:
    store = TrialArtifactStore(tmp_path, restricted_values=("provider-secret", ""))

    store.append_plan(
        {
            "message": "provider-secret must not survive",
            "items": [
                ProviderEnvelope(
                    detail="provider-secret",
                    nested=[{"token": "prefix-provider-secret-suffix"}],
                )
            ],
        }
    )

    text = (tmp_path / "plans.jsonl").read_text(encoding="utf-8")
    assert "provider-secret" not in text
    assert text.count("[REDACTED]") == 3
    assert json.loads(text) == {
        "items": [{"detail": "[REDACTED]", "nested": [{"token": "prefix-[REDACTED]-suffix"}]}],
        "message": "[REDACTED] must not survive",
    }


def test_plan_does_not_create_signal_or_evidence_stream(tmp_path) -> None:
    store = TrialArtifactStore(tmp_path, restricted_values=())

    store.append_plan({"probe_id": "P1", "actions": []})

    assert (tmp_path / "plans.jsonl").exists()
    assert not (tmp_path / "signals.jsonl").exists()
    assert not (tmp_path / "evidence.jsonl").exists()


def test_store_writes_each_explicit_artifact_type_to_its_own_line_delimited_stream(tmp_path) -> None:
    store = TrialArtifactStore(tmp_path, restricted_values=("provider-secret",))

    store.append_observation({"stdout": "provider-secret"})
    store.append_provider_call({"request": "provider-secret"})
    store.append_error({"message": "provider-secret"})

    assert (tmp_path / "environment_actions.jsonl").read_text(encoding="utf-8") == (
        '{"stdout": "[REDACTED]"}\n'
    )
    assert (tmp_path / "provider_telemetry.jsonl").read_text(encoding="utf-8") == (
        '{"request": "[REDACTED]"}\n'
    )
    assert (tmp_path / "errors.jsonl").read_text(encoding="utf-8") == (
        '{"message": "[REDACTED]"}\n'
    )


def test_concurrent_appends_are_complete_json_lines(tmp_path) -> None:
    store = TrialArtifactStore(tmp_path, restricted_values=())

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda index: store.append_observation({"index": index}), range(64)))

    lines = (tmp_path / "environment_actions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 64
    assert {json.loads(line)["index"] for line in lines} == set(range(64))


def test_summary_is_redacted_and_deterministic_json(tmp_path) -> None:
    store = TrialArtifactStore(tmp_path, restricted_values=("provider-secret",))

    store.write_summary({"zeta": "provider-secret", "alpha": {"secret": "provider-secret"}})

    assert (tmp_path / "summary.json").read_text(encoding="utf-8") == (
        "{\n"
        '  "alpha": {\n'
        '    "secret": "[REDACTED]"\n'
        "  },\n"
        '  "zeta": "[REDACTED]"\n'
        "}\n"
    )
