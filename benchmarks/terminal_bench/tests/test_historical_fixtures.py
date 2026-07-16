from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path, PurePosixPath

import pytest
from pydantic import ValidationError

from freeze_historical_traces import (
    HistoricalTraceManifest,
    HistoricalTraceRef,
    freeze_historical_traces,
    main,
)


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "historical_traces"
EXPECTED_SOURCE_COMMIT = "12288ad29d162fd9fc8afa296f5f7ec930da9cd0"
EXPECTED_TASK_IDS = (
    "terminal-bench/break-filter-js-from-html",
    "terminal-bench/cancel-async-tasks",
    "terminal-bench/log-summary-date-ranges",
)
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_SECRET_PATTERN = re.compile(
    r"(?:sk-[A-Za-z0-9_-]{12,}|tvly-[A-Za-z0-9_-]{12,}|"
    r"github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16})",
    re.IGNORECASE,
)


def test_historical_trace_manifest_is_strict_and_immutable() -> None:
    trace = HistoricalTraceRef(
        task_id=EXPECTED_TASK_IDS[0],
        expected_classification="provider_contract_error",
        files={"provider_telemetry.jsonl": "sha256:" + "a" * 64},
    )
    manifest = HistoricalTraceManifest(
        schema_version="terminal_historical_trace:v1",
        source_commit=EXPECTED_SOURCE_COMMIT,
        traces=(trace,),
    )

    with pytest.raises(ValidationError):
        HistoricalTraceManifest.model_validate(
            {**manifest.model_dump(), "unexpected": "value"}
        )
    with pytest.raises(ValidationError):
        manifest.source_commit = "f" * 40  # type: ignore[misc]


def test_historical_trace_fixtures_are_complete_redacted_and_immutable() -> None:
    manifest_path = FIXTURE_ROOT / "manifest.json"
    manifest = HistoricalTraceManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )

    assert manifest.schema_version == "terminal_historical_trace:v1"
    assert manifest.source_commit == EXPECTED_SOURCE_COMMIT
    assert tuple(trace.task_id for trace in manifest.traces) == EXPECTED_TASK_IDS
    assert Counter(trace.expected_classification for trace in manifest.traces) == {
        "provider_contract_error": 2,
        "causal_conformance_error": 1,
    }

    for path in FIXTURE_ROOT.rglob("*"):
        assert not path.is_symlink(), f"fixture symlink is not allowed: {path}"

    expected_files = {"manifest.json"}
    for trace in manifest.traces:
        task_directory = FIXTURE_ROOT / trace.task_id.split("/", maxsplit=1)[1]
        actual_files = {
            path.relative_to(task_directory).as_posix()
            for path in task_directory.rglob("*")
            if path.is_file()
        }
        assert actual_files == set(trace.files)
        expected_files.update(
            f"{task_directory.name}/{relative_path}"
            for relative_path in trace.files
        )

        for relative_path, expected_digest in trace.files.items():
            normalized_path = PurePosixPath(relative_path)
            assert not normalized_path.is_absolute()
            assert ".." not in normalized_path.parts
            assert _SHA256.fullmatch(expected_digest)

            path = task_directory.joinpath(*normalized_path.parts)
            contents = path.read_bytes()
            decoded = contents.decode("utf-8")
            assert not _SECRET_PATTERN.search(decoded), (
                f"secret-shaped value found in {path.relative_to(FIXTURE_ROOT)}"
            )
            assert f"sha256:{hashlib.sha256(contents).hexdigest()}" == expected_digest

    actual_files = {
        path.relative_to(FIXTURE_ROOT).as_posix()
        for path in FIXTURE_ROOT.rglob("*")
        if path.is_file()
    }
    assert actual_files == expected_files

    manifest_text = manifest_path.read_text(encoding="utf-8")
    assert not _SECRET_PATTERN.search(manifest_text)


def test_freezer_copies_only_allowed_normalized_artifacts(tmp_path: Path) -> None:
    source_job = tmp_path / "source-job"
    for task_id in EXPECTED_TASK_IDS:
        artifact_directory = (
            source_job
            / f"{task_id.split('/', maxsplit=1)[1]}__historical"
            / "agent"
            / "bayesprobe"
        )
        artifact_directory.mkdir(parents=True)
        (artifact_directory / "bayesprobe_ledger.jsonl").write_text(
            '{"z": 2, "a": 1}\n', encoding="utf-8"
        )
        (artifact_directory / "ignored.log").write_text("ignored", encoding="utf-8")
    (source_job / "cancel-async-tasks__historical" / "agent" / "bayesprobe" / "summary.json").write_text(
        '{"z": 2, "a": 1}', encoding="utf-8"
    )

    output = tmp_path / "fixtures"
    freeze_historical_traces(
        source_job=source_job,
        output=output,
        source_commit=EXPECTED_SOURCE_COMMIT,
    )

    manifest = HistoricalTraceManifest.model_validate_json(
        (output / "manifest.json").read_text(encoding="utf-8")
    )
    assert tuple(trace.task_id for trace in manifest.traces) == EXPECTED_TASK_IDS
    assert (output / "break-filter-js-from-html" / "bayesprobe_ledger.jsonl").read_text(
        encoding="utf-8"
    ) == '{"a":1,"z":2}\n'
    assert (output / "cancel-async-tasks" / "summary.json").read_text(
        encoding="utf-8"
    ) == '{"a":1,"z":2}\n'
    assert not (output / "log-summary-date-ranges" / "ignored.log").exists()


@pytest.mark.parametrize(
    ("source_value", "restricted_value_envs", "error_message"),
    [
        ("provider-secret", ("FIXTURE_RESTRICTED_VALUE",), "restricted value"),
        ("sk-1234567890ab", (), "secret-shaped content"),
    ],
)
def test_freezer_rejects_unsafe_source_content_without_replacing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_value: str,
    restricted_value_envs: tuple[str, ...],
    error_message: str,
) -> None:
    source_job = tmp_path / "source-job"
    for task_id in EXPECTED_TASK_IDS:
        artifact_directory = (
            source_job
            / f"{task_id.split('/', maxsplit=1)[1]}__historical"
            / "agent"
            / "bayesprobe"
        )
        artifact_directory.mkdir(parents=True)
        (artifact_directory / "provider_telemetry.jsonl").write_text(
            f'{{"value": "{source_value}"}}\n', encoding="utf-8"
        )

    output = tmp_path / "fixtures"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("existing fixture", encoding="utf-8")
    monkeypatch.setenv("FIXTURE_RESTRICTED_VALUE", "provider-secret")

    with pytest.raises(ValueError, match=error_message):
        freeze_historical_traces(
            source_job=source_job,
            output=output,
            source_commit=EXPECTED_SOURCE_COMMIT,
            restricted_value_envs=restricted_value_envs,
        )

    assert sentinel.read_text(encoding="utf-8") == "existing fixture"


@pytest.mark.parametrize(
    ("environment", "error_message"),
    [
        ({}, "restricted value environment variable is missing"),
        ({"FIXTURE_RESTRICTED_VALUE": ""}, "restricted value environment variable is empty"),
    ],
)
def test_freezer_requires_nonempty_named_restricted_value_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    environment: dict[str, str],
    error_message: str,
) -> None:
    source_job = _safe_source_job(tmp_path)
    monkeypatch.delenv("FIXTURE_RESTRICTED_VALUE", raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=error_message):
        freeze_historical_traces(
            source_job=source_job,
            output=tmp_path / "fixtures",
            source_commit=EXPECTED_SOURCE_COMMIT,
            restricted_value_envs=("FIXTURE_RESTRICTED_VALUE",),
        )


@pytest.mark.parametrize("environment_name", ("", "1INVALID", "INVALID-NAME"))
def test_freezer_rejects_invalid_restricted_value_environment_name(
    tmp_path: Path,
    environment_name: str,
) -> None:
    with pytest.raises(ValueError, match="restricted value environment name is invalid"):
        freeze_historical_traces(
            source_job=_safe_source_job(tmp_path),
            output=tmp_path / "fixtures",
            source_commit=EXPECTED_SOURCE_COMMIT,
            restricted_value_envs=(environment_name,),
        )


def test_freezer_cli_rejects_raw_restricted_values(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(
            [
                "--source-job",
                "source-job",
                "--output",
                "fixtures",
                "--source-commit",
                EXPECTED_SOURCE_COMMIT,
                "--restricted-value",
                "provider-secret",
            ]
        )

    assert error.value.code == 2
    assert "unrecognized arguments: --restricted-value provider-secret" in capsys.readouterr().err


def _safe_source_job(tmp_path: Path) -> Path:
    source_job = tmp_path / "source-job"
    for task_id in EXPECTED_TASK_IDS:
        artifact_directory = (
            source_job
            / f"{task_id.split('/', maxsplit=1)[1]}__historical"
            / "agent"
            / "bayesprobe"
        )
        artifact_directory.mkdir(parents=True)
        (artifact_directory / "provider_telemetry.jsonl").write_text(
            '{"value":"safe"}\n', encoding="utf-8"
        )
    return source_job
