from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict


FROZEN_TASKS = (
    (
        "terminal-bench/break-filter-js-from-html",
        "provider_contract_error",
    ),
    (
        "terminal-bench/cancel-async-tasks",
        "causal_conformance_error",
    ),
    (
        "terminal-bench/log-summary-date-ranges",
        "provider_contract_error",
    ),
)
ARTIFACT_FILENAMES = (
    "bayesprobe_ledger.jsonl",
    "provider_telemetry.jsonl",
    "plans.jsonl",
    "environment_actions.jsonl",
    "errors.jsonl",
    "summary.json",
)
_SECRET_PATTERN = re.compile(
    rb"(?:sk-[A-Za-z0-9_-]{12,}|tvly-[A-Za-z0-9_-]{12,}|"
    rb"github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16})",
    re.IGNORECASE,
)


class HistoricalTraceRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    task_id: str
    expected_classification: Literal[
        "provider_contract_error", "causal_conformance_error"
    ]
    files: dict[str, str]


class HistoricalTraceManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["terminal_historical_trace:v1"]
    source_commit: str
    traces: tuple[HistoricalTraceRef, ...]


def freeze_historical_traces(
    *,
    source_job: Path,
    output: Path,
    source_commit: str,
    restricted_values: tuple[str, ...] = (),
) -> HistoricalTraceManifest:
    source_job = Path(source_job)
    output = Path(output)
    if source_job.is_symlink() or not source_job.is_dir():
        raise ValueError("source job must be a real directory")

    values = tuple(value for value in restricted_values if value)
    if any(not isinstance(value, str) for value in values):
        raise TypeError("restricted values must be strings")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = Path(
        tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent)
    )
    try:
        traces: list[HistoricalTraceRef] = []
        for task_id, classification in FROZEN_TASKS:
            source_artifacts = _source_artifacts(source_job, task_id)
            task_directory = temporary_output / task_id.split("/", maxsplit=1)[1]
            files: dict[str, str] = {}
            for filename in ARTIFACT_FILENAMES:
                source_file = source_artifacts / filename
                if not source_file.exists():
                    continue
                contents = _normalized_source_contents(
                    source_file, restricted_values=values
                )
                destination = task_directory / filename
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(contents)
                files[filename] = f"sha256:{hashlib.sha256(contents).hexdigest()}"
            if not files:
                raise ValueError(f"no allowed artifacts found for {task_id}")
            traces.append(
                HistoricalTraceRef(
                    task_id=task_id,
                    expected_classification=classification,
                    files=files,
                )
            )

        manifest = HistoricalTraceManifest(
            schema_version="terminal_historical_trace:v1",
            source_commit=source_commit,
            traces=tuple(traces),
        )
        (temporary_output / "manifest.json").write_bytes(
            _dump_json(manifest.model_dump(mode="json"))
        )
        if output.exists() or output.is_symlink():
            raise ValueError("output fixture directory already exists")
        os.replace(temporary_output, output)
        return manifest
    except Exception:
        shutil.rmtree(temporary_output, ignore_errors=True)
        raise


def _source_artifacts(source_job: Path, task_id: str) -> Path:
    task_name = task_id.split("/", maxsplit=1)[1]
    matches = sorted(source_job.glob(f"{task_name}__*"))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one historical trial for {task_id}")
    trial_directory = matches[0]
    if trial_directory.is_symlink() or not trial_directory.is_dir():
        raise ValueError(f"historical trial must be a real directory: {task_id}")
    artifacts = trial_directory / "agent" / "bayesprobe"
    if artifacts.is_symlink() or not artifacts.is_dir():
        raise ValueError(f"BayesProbe artifacts are missing for {task_id}")
    return artifacts


def _normalized_source_contents(
    source_file: Path, *, restricted_values: tuple[str, ...]
) -> bytes:
    if source_file.is_symlink() or not source_file.is_file():
        raise ValueError(f"source artifact must be a real file: {source_file.name}")
    contents = source_file.read_bytes()
    if _SECRET_PATTERN.search(contents):
        raise ValueError(f"secret-shaped content found in {source_file.name}")
    for restricted_value in restricted_values:
        if restricted_value.encode("utf-8") in contents:
            raise ValueError(f"restricted value found in {source_file.name}")

    try:
        text = contents.decode("utf-8")
        if source_file.suffix == ".jsonl":
            records = [json.loads(line) for line in text.splitlines() if line]
            return b"".join(_dump_json(record) for record in records)
        return _dump_json(json.loads(text))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"source artifact is not valid JSON: {source_file.name}") from error


def _dump_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-job", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--restricted-value", action="append", default=[])
    args = parser.parse_args(argv)
    manifest = freeze_historical_traces(
        source_job=args.source_job,
        output=args.output,
        source_commit=args.source_commit,
        restricted_values=tuple(args.restricted_value),
    )
    print(json.dumps(manifest.model_dump(mode="json"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
