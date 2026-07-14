from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor

import pytest
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


def test_redaction_uses_deletion_when_a_restricted_value_overlaps_the_marker(tmp_path) -> None:
    store = TrialArtifactStore(tmp_path, restricted_values=("E",))

    store.append_plan({"value": "E"})
    store.write_summary({"value": "E"})

    assert json.loads((tmp_path / "plans.jsonl").read_text(encoding="utf-8")) == {"value": ""}
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8")) == {"value": ""}


def test_redaction_prefers_longest_overlapping_values_in_plans_and_summaries(tmp_path) -> None:
    store = TrialArtifactStore(tmp_path, restricted_values=("provider", "provider-secret"))

    store.append_plan({"value": "provider-secret"})
    store.write_summary({"value": "provider-secret"})

    assert json.loads((tmp_path / "plans.jsonl").read_text(encoding="utf-8")) == {
        "value": "[REDACTED]"
    }
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8")) == {
        "value": "[REDACTED]"
    }


def test_redaction_reaches_a_fixed_point_in_plans_and_summaries(tmp_path) -> None:
    store = TrialArtifactStore(tmp_path, restricted_values=("[REDACTED]y", "x"))

    store.append_plan({"value": "xy"})
    store.write_summary({"value": "xy"})

    assert json.loads((tmp_path / "plans.jsonl").read_text(encoding="utf-8")) == {
        "value": "[REDACTED]"
    }
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8")) == {
        "value": "[REDACTED]"
    }


def test_redacted_mapping_key_collision_fails_before_writing_an_artifact(tmp_path) -> None:
    store = TrialArtifactStore(tmp_path, restricted_values=("provider-secret",))

    with pytest.raises(ValueError, match="redacted mapping keys collide"):
        store.append_plan({"[REDACTED]": "existing", "provider-secret": "restricted"})

    assert not (tmp_path / "plans.jsonl").exists()


def test_same_root_stores_share_a_lock_and_append_complete_json_lines(tmp_path) -> None:
    first = TrialArtifactStore(tmp_path, restricted_values=())
    second = TrialArtifactStore(tmp_path, restricted_values=())

    assert first._lock is second._lock
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(
            executor.map(
                lambda index: (first if index % 2 else second).append_observation({"index": index}),
                range(128),
            )
        )

    lines = (tmp_path / "environment_actions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 128
    assert {json.loads(line)["index"] for line in lines} == set(range(128))


def test_same_root_stores_write_complete_atomic_summaries(tmp_path) -> None:
    first = TrialArtifactStore(tmp_path, restricted_values=())
    second = TrialArtifactStore(tmp_path, restricted_values=())
    summaries = (
        {"writer": "first", "payload": "a" * 100_000},
        {"writer": "second", "payload": "b" * 100_000},
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(lambda item: (first if item["writer"] == "first" else second).write_summary(item), summaries))

    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8")) in summaries


def test_summary_cleans_up_temporary_file_when_atomic_replace_fails(tmp_path, monkeypatch) -> None:
    store = TrialArtifactStore(tmp_path, restricted_values=())

    def fail_replace(source: str, destination: str) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        store.write_summary({"status": "complete"})

    assert not (tmp_path / "summary.json").exists()
    assert not list(tmp_path.glob(".summary-*.tmp"))
