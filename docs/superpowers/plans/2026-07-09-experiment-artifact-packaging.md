# Experiment Artifact Packaging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reproducible experiment artifact packaging for BayesProbe benchmark runs.

**Architecture:** Keep BayesProbe core untouched. Add optional run packaging at the experiment layer: config parses `artifact_dir`, the runner writes normal report/ledger outputs, and a focused `experiment_artifacts` module snapshots report, ledger, config, dataset, and manifest into a stable directory.

**Tech Stack:** Python dataclasses, pathlib, JSON files, pytest, existing BayesProbe benchmark/config/CLI modules.

## Global Constraints

- Do not change `BayesProbeCore`, evidence integration, posterior updates, or probe control flow.
- `artifact_dir` is optional; existing configs without it keep current behavior.
- `artifact_dir` is the exact run directory for v0.1.
- If `artifact_dir` is configured and `ledger_path` is omitted, use `artifact_dir / "ledger.jsonl"` as the effective ledger path.
- Artifact files are `manifest.json`, `report.json`, `ledger.jsonl`, `config_snapshot.json`, and `dataset_snapshot.json`.
- Provider secrets must not be written to artifact files.
- Tests must use deterministic/offline fixtures only.

---

## File Structure

- Modify `bayesprobe/experiment_runner.py`: add config/result fields, choose effective ledger path, call artifact writer after the normal report is written.
- Modify `bayesprobe/config.py`: parse `artifact_dir`, `run_name`, and `metadata` from JSON config mappings.
- Create `bayesprobe/experiment_artifacts.py`: own artifact serialization and sanitized manifest/config snapshots.
- Modify `bayesprobe/cli.py`: append `artifact=<path>` only when artifact packaging is enabled.
- Modify `bayesprobe/__init__.py`: export artifact result type only if the implementation creates one as a stable public type.
- Modify `tests/test_public_api_and_config.py`: cover config parsing and validation.
- Modify `tests/test_experiment_runner.py`: cover artifact bundle writing, default ledger path, and secret redaction.
- Modify `tests/test_cli.py`: cover artifact summary output.
- Modify `docs/ARCHITECTURE.md`: mark Phase 5 artifact directory as implemented v0.1 and leave SQLite/prompt registry as future work.

---

### Task 1: Experiment Config Surface

**Files:**
- Modify: `tests/test_public_api_and_config.py`
- Modify: `bayesprobe/experiment_runner.py`
- Modify: `bayesprobe/config.py`

**Interfaces:**
- Consumes: existing `ExperimentRunConfig` and `experiment_config_from_mapping`.
- Produces:
  - `ExperimentRunConfig.artifact_dir: str | Path | None`
  - `ExperimentRunConfig.run_name: str | None`
  - `ExperimentRunConfig.metadata: dict[str, Any]`
  - config loader support for JSON fields `artifact_dir`, `run_name`, `metadata`

- [ ] **Step 1: Write failing config parsing test**

Append this test near the path-resolution tests in `tests/test_public_api_and_config.py`:

```python
def test_experiment_config_from_mapping_parses_artifact_metadata(tmp_path: Path):
    config = experiment_config_from_mapping(
        {
            "dataset_path": "datasets/toy.json",
            "report_path": "outputs/toy-report.json",
            "artifact_dir": "runs/toy",
            "run_name": "toy-offline-smoke",
            "metadata": {"suite": "offline", "prompt_registry": "none"},
        },
        base_dir=tmp_path,
    )

    assert config.dataset_path == tmp_path / "datasets" / "toy.json"
    assert config.report_path == tmp_path / "outputs" / "toy-report.json"
    assert config.artifact_dir == tmp_path / "runs" / "toy"
    assert config.run_name == "toy-offline-smoke"
    assert config.metadata == {"suite": "offline", "prompt_registry": "none"}
```

- [ ] **Step 2: Run the new test to verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_public_api_and_config.py::test_experiment_config_from_mapping_parses_artifact_metadata -q -p no:cacheprovider
```

Expected: FAIL because `ExperimentRunConfig` has no `artifact_dir`, `run_name`, or `metadata` fields.

- [ ] **Step 3: Implement minimal config fields**

In `bayesprobe/experiment_runner.py`, update imports and `ExperimentRunConfig`:

```python
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
```

```python
@dataclass(frozen=True)
class ExperimentRunConfig:
    dataset_path: str | Path
    report_path: str | Path
    ledger_path: str | Path | None = None
    artifact_dir: str | Path | None = None
    run_name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    max_cycles: int = 1
    max_probes_per_cycle: int = 1
    model_gateway: ModelGatewayConfig | Mapping[str, Any] | None = None
    judgment_repair_policy: EvidenceJudgmentRepairPolicy | Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.max_cycles < 1:
            raise ValueError("max_cycles must be at least 1")
        if self.max_probes_per_cycle < 1:
            raise ValueError("max_probes_per_cycle must be at least 1")
        if self.artifact_dir is not None and not isinstance(
            self.artifact_dir, (str, Path)
        ):
            raise ValueError("artifact_dir must be a path")
        if self.run_name is not None:
            if not isinstance(self.run_name, str):
                raise ValueError("run_name must be a string")
            if not self.run_name.strip():
                raise ValueError("run_name must not be empty")
            object.__setattr__(self, "run_name", self.run_name.strip())
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be an object")
        object.__setattr__(self, "metadata", dict(self.metadata))
```

In `bayesprobe/config.py`, pass the new fields:

```python
return ExperimentRunConfig(
    dataset_path=_required_path(data, "dataset_path", base_dir=base_dir),
    report_path=_required_path(data, "report_path", base_dir=base_dir),
    ledger_path=_optional_path(data, "ledger_path", base_dir=base_dir),
    artifact_dir=_optional_path(data, "artifact_dir", base_dir=base_dir),
    run_name=_optional_string(data, "run_name"),
    metadata=_optional_mapping(data, "metadata"),
    max_cycles=_optional_int(data, "max_cycles", default=1),
    max_probes_per_cycle=_optional_int(data, "max_probes_per_cycle", default=1),
    model_gateway=_optional_model_gateway_config(data),
    judgment_repair_policy=_optional_judgment_repair_policy(data),
)
```

Add helpers to `bayesprobe/config.py`:

```python
def _optional_string(data: Mapping[str, Any], field_name: str) -> str | None:
    if field_name not in data or data[field_name] is None:
        return None
    value = data[field_name]
    if not isinstance(value, str):
        raise ValueError(f"experiment config field {field_name} must be a string")
    if not value.strip():
        raise ValueError(f"experiment config field {field_name} must not be empty")
    return value.strip()


def _optional_mapping(data: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    if field_name not in data or data[field_name] is None:
        return {}
    value = data[field_name]
    if not isinstance(value, Mapping):
        raise ValueError(f"experiment config field {field_name} must be an object")
    return dict(value)
```

- [ ] **Step 4: Verify config parsing test is GREEN**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_public_api_and_config.py::test_experiment_config_from_mapping_parses_artifact_metadata -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 5: Add validation tests**

Add these parameter cases to `test_load_experiment_config_rejects_invalid_config_files` in `tests/test_public_api_and_config.py`:

```python
(
    "non_string_artifact_dir.json",
    json.dumps({"dataset_path": "dataset.json", "report_path": "report.json", "artifact_dir": 1}),
    "experiment config field artifact_dir must be a string",
),
(
    "non_string_run_name.json",
    json.dumps({"dataset_path": "dataset.json", "report_path": "report.json", "run_name": 1}),
    "experiment config field run_name must be a string",
),
(
    "empty_run_name.json",
    json.dumps({"dataset_path": "dataset.json", "report_path": "report.json", "run_name": "   "}),
    "experiment config field run_name must not be empty",
),
(
    "non_object_metadata.json",
    json.dumps({"dataset_path": "dataset.json", "report_path": "report.json", "metadata": []}),
    "experiment config field metadata must be an object",
),
```

- [ ] **Step 6: Run validation tests to verify GREEN**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_public_api_and_config.py::test_load_experiment_config_rejects_invalid_config_files -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 7: Commit Task 1**

```bash
git add bayesprobe/experiment_runner.py bayesprobe/config.py tests/test_public_api_and_config.py
git commit -m "feat: add experiment artifact config fields"
```

---

### Task 2: Artifact Bundle Writer

**Files:**
- Create: `bayesprobe/experiment_artifacts.py`
- Modify: `tests/test_experiment_runner.py`

**Interfaces:**
- Consumes:
  - `ExperimentRunConfig`-like object with `dataset_path`, `report_path`, `ledger_path`, `artifact_dir`, `run_name`, `metadata`, `max_cycles`, `max_probes_per_cycle`, `model_gateway`, and `judgment_repair_policy` attributes.
  - `BenchmarkDataset`
  - benchmark report file already written to `report_path`
  - ledger file path if one was used
- Produces:
  - `ExperimentArtifactBundle`
  - `write_experiment_artifact_bundle(...) -> ExperimentArtifactBundle`

- [ ] **Step 1: Write failing artifact writer test through the runner API**

Append to `tests/test_experiment_runner.py`:

```python
def test_run_benchmark_experiment_writes_artifact_bundle(tmp_path: Path):
    report_path = tmp_path / "reports" / "toy-report.json"
    ledger_path = tmp_path / "ledgers" / "toy-ledger.jsonl"
    artifact_dir = tmp_path / "artifacts" / "toy-run"

    result = run_benchmark_experiment(
        ExperimentRunConfig(
            dataset_path=FIXTURE_PATH,
            report_path=report_path,
            ledger_path=ledger_path,
            artifact_dir=artifact_dir,
            run_name="toy-artifact-run",
            metadata={"suite": "offline"},
            model_gateway={
                "kind": "scripted",
                "api_key": "sk-proj-secret-value",
                "responses": {
                    "judge_evidence": {
                        "evidence_type": "supporting",
                        "likelihoods": {
                            "H1": "moderately_confirming",
                            "H2": "moderately_disconfirming",
                        },
                        "interpretation": "Scripted artifact judgment.",
                    }
                },
            },
            judgment_repair_policy={"max_attempts": 1},
        )
    )

    manifest_path = artifact_dir / "manifest.json"
    config_snapshot_path = artifact_dir / "config_snapshot.json"
    dataset_snapshot_path = artifact_dir / "dataset_snapshot.json"
    artifact_report_path = artifact_dir / "report.json"
    artifact_ledger_path = artifact_dir / "ledger.jsonl"

    assert result.artifact_dir == artifact_dir
    assert result.artifact_manifest_path == manifest_path
    assert manifest_path.exists()
    assert config_snapshot_path.exists()
    assert dataset_snapshot_path.exists()
    assert artifact_report_path.exists()
    assert artifact_ledger_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_snapshot = json.loads(config_snapshot_path.read_text(encoding="utf-8"))
    dataset_snapshot = json.loads(dataset_snapshot_path.read_text(encoding="utf-8"))
    artifact_report = json.loads(artifact_report_path.read_text(encoding="utf-8"))
    artifact_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            manifest_path,
            config_snapshot_path,
            dataset_snapshot_path,
            artifact_report_path,
            artifact_ledger_path,
        ]
    )

    assert manifest["artifact_version"] == "0.1"
    assert manifest["run_name"] == "toy-artifact-run"
    assert manifest["dataset_name"] == "toy_belief_revision"
    assert manifest["sample_count"] == 3
    assert manifest["metadata"] == {"suite": "offline"}
    assert manifest["model_gateway"]["kind"] == "scripted"
    assert manifest["model_gateway"]["scripted_response_tasks"] == ["judge_evidence"]
    assert "responses" not in manifest["model_gateway"]
    assert config_snapshot["artifact_dir"] == str(artifact_dir)
    assert config_snapshot["ledger_path"] == str(ledger_path)
    assert dataset_snapshot["dataset_name"] == "toy_belief_revision"
    assert len(dataset_snapshot["samples"]) == 3
    assert artifact_report["sample_count"] == 3
    assert "sk-proj-secret-value" not in artifact_text
    assert '"api_key"' not in artifact_text
```

- [ ] **Step 2: Run artifact writer test to verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_experiment_runner.py::test_run_benchmark_experiment_writes_artifact_bundle -q -p no:cacheprovider
```

Expected: FAIL because `ExperimentRunResult` has no artifact fields and no artifact bundle is written.

- [ ] **Step 3: Implement artifact writer module**

Create `bayesprobe/experiment_artifacts.py`:

```python
from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bayesprobe.benchmark_io import BenchmarkDataset
from bayesprobe.model_gateway import EvidenceJudgmentRepairPolicy, ModelGatewayConfig


@dataclass(frozen=True)
class ExperimentArtifactBundle:
    artifact_dir: Path
    manifest_path: Path
    report_path: Path
    ledger_path: Path | None
    config_snapshot_path: Path
    dataset_snapshot_path: Path


def write_experiment_artifact_bundle(
    *,
    artifact_dir: str | Path,
    config: Any,
    dataset: BenchmarkDataset,
    report_path: str | Path,
    ledger_path: str | Path | None,
    sample_count: int,
    created_at_utc: datetime | None = None,
) -> ExperimentArtifactBundle:
    target_dir = Path(artifact_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    artifact_report_path = target_dir / "report.json"
    artifact_ledger_path = target_dir / "ledger.jsonl" if ledger_path is not None else None
    config_snapshot_path = target_dir / "config_snapshot.json"
    dataset_snapshot_path = target_dir / "dataset_snapshot.json"
    manifest_path = target_dir / "manifest.json"

    _copy_json_file(Path(report_path), artifact_report_path)
    if ledger_path is not None:
        _copy_text_file(Path(ledger_path), artifact_ledger_path)
    _write_json(config_snapshot_path, _config_snapshot(config, ledger_path=ledger_path))
    _write_json(dataset_snapshot_path, _dataset_snapshot(dataset))
    _write_json(
        manifest_path,
        _manifest_payload(
            config=config,
            dataset=dataset,
            sample_count=sample_count,
            artifact_dir=target_dir,
            report_path=artifact_report_path,
            ledger_path=artifact_ledger_path,
            config_snapshot_path=config_snapshot_path,
            dataset_snapshot_path=dataset_snapshot_path,
            created_at_utc=created_at_utc,
        ),
    )

    return ExperimentArtifactBundle(
        artifact_dir=target_dir,
        manifest_path=manifest_path,
        report_path=artifact_report_path,
        ledger_path=artifact_ledger_path,
        config_snapshot_path=config_snapshot_path,
        dataset_snapshot_path=dataset_snapshot_path,
    )


def _copy_json_file(source: Path, destination: Path) -> None:
    payload = json.loads(source.read_text(encoding="utf-8"))
    _write_json(destination, payload)


def _copy_text_file(source: Path, destination: Path | None) -> None:
    if destination is None:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == destination.resolve():
        return
    shutil.copyfile(source, destination)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _dataset_snapshot(dataset: BenchmarkDataset) -> dict[str, Any]:
    samples = []
    for sample in dataset.samples:
        sample_payload = asdict(sample)
        sample_payload["signal_shape"] = sample.signal_shape.value
        samples.append(sample_payload)
    return {
        "dataset_name": dataset.dataset_name,
        "metadata": dict(dataset.metadata),
        "samples": samples,
    }


def _config_snapshot(config: Any, *, ledger_path: str | Path | None) -> dict[str, Any]:
    return {
        "dataset_path": str(Path(config.dataset_path)),
        "report_path": str(Path(config.report_path)),
        "ledger_path": str(Path(ledger_path)) if ledger_path is not None else None,
        "artifact_dir": str(Path(config.artifact_dir)) if config.artifact_dir is not None else None,
        "run_name": config.run_name,
        "metadata": dict(config.metadata),
        "max_cycles": config.max_cycles,
        "max_probes_per_cycle": config.max_probes_per_cycle,
        "model_gateway": _model_gateway_snapshot(config.model_gateway),
        "judgment_repair_policy": _repair_policy_snapshot(config.judgment_repair_policy),
    }


def _manifest_payload(
    *,
    config: Any,
    dataset: BenchmarkDataset,
    sample_count: int,
    artifact_dir: Path,
    report_path: Path,
    ledger_path: Path | None,
    config_snapshot_path: Path,
    dataset_snapshot_path: Path,
    created_at_utc: datetime | None,
) -> dict[str, Any]:
    created_at = created_at_utc or datetime.now(UTC)
    return {
        "artifact_version": "0.1",
        "created_at_utc": created_at.isoformat().replace("+00:00", "Z"),
        "run_name": config.run_name,
        "artifact_dir": str(artifact_dir),
        "dataset_name": dataset.dataset_name,
        "sample_count": sample_count,
        "report_path": str(report_path),
        "ledger_path": str(ledger_path) if ledger_path is not None else None,
        "config_snapshot_path": str(config_snapshot_path),
        "dataset_snapshot_path": str(dataset_snapshot_path),
        "model_gateway": _model_gateway_snapshot(config.model_gateway),
        "judgment_repair_policy": _repair_policy_snapshot(config.judgment_repair_policy),
        "metadata": dict(config.metadata),
    }


def _model_gateway_snapshot(config: Any) -> dict[str, Any]:
    if config is None:
        return {"kind": "deterministic"}
    if isinstance(config, ModelGatewayConfig):
        payload = {
            "kind": config.kind,
            "model": config.model,
            "api_key_env": config.api_key_env,
            "timeout_seconds": config.timeout_seconds,
            "max_output_tokens": config.max_output_tokens,
        }
        if config.responses is not None:
            payload["scripted_response_tasks"] = sorted(config.responses)
        return {key: value for key, value in payload.items() if value is not None}
    if isinstance(config, Mapping):
        payload = {
            "kind": str(config.get("kind", "deterministic")),
            "model": config.get("model"),
            "api_key_env": config.get("api_key_env"),
            "timeout_seconds": config.get("timeout_seconds"),
            "max_output_tokens": config.get("max_output_tokens"),
        }
        responses = config.get("responses")
        if isinstance(responses, Mapping):
            payload["scripted_response_tasks"] = sorted(str(task) for task in responses)
        return {key: value for key, value in payload.items() if value is not None}
    return {"kind": type(config).__name__}


def _repair_policy_snapshot(config: Any) -> dict[str, Any]:
    if config is None:
        policy = EvidenceJudgmentRepairPolicy()
    elif isinstance(config, EvidenceJudgmentRepairPolicy):
        policy = config
    elif isinstance(config, Mapping):
        policy = EvidenceJudgmentRepairPolicy.from_config(config)
    else:
        return {"kind": type(config).__name__}
    return {
        "max_attempts": policy.max_attempts,
        "repair_task": policy.repair_task,
    }


__all__ = [
    "ExperimentArtifactBundle",
    "write_experiment_artifact_bundle",
]
```

- [ ] **Step 4: Add artifact fields to result and integrate runner**

In `bayesprobe/experiment_runner.py`, import the writer:

```python
from bayesprobe.experiment_artifacts import write_experiment_artifact_bundle
```

Update `ExperimentRunResult`:

```python
@dataclass(frozen=True)
class ExperimentRunResult:
    dataset: BenchmarkDataset
    suite_result: BenchmarkSuiteResult
    report_path: Path
    ledger_path: Path | None = None
    artifact_dir: Path | None = None
    artifact_manifest_path: Path | None = None
```

Update `run_benchmark_experiment` after `write_benchmark_report(...)`:

```python
artifact_dir = Path(config.artifact_dir) if config.artifact_dir is not None else None
artifact_manifest_path = None
if artifact_dir is not None:
    artifact_bundle = write_experiment_artifact_bundle(
        artifact_dir=artifact_dir,
        config=config,
        dataset=dataset,
        report_path=report_path,
        ledger_path=ledger_path,
        sample_count=suite_result.sample_count,
    )
    artifact_manifest_path = artifact_bundle.manifest_path
```

Return the new fields:

```python
return ExperimentRunResult(
    dataset=dataset,
    suite_result=suite_result,
    report_path=report_path,
    ledger_path=ledger_path,
    artifact_dir=artifact_dir,
    artifact_manifest_path=artifact_manifest_path,
)
```

- [ ] **Step 5: Run artifact bundle test to verify GREEN**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_experiment_runner.py::test_run_benchmark_experiment_writes_artifact_bundle -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 6: Commit Task 2**

```bash
git add bayesprobe/experiment_artifacts.py bayesprobe/experiment_runner.py tests/test_experiment_runner.py
git commit -m "feat: write experiment artifact bundles"
```

---

### Task 3: Default Artifact Ledger and CLI Summary

**Files:**
- Modify: `tests/test_experiment_runner.py`
- Modify: `tests/test_cli.py`
- Modify: `bayesprobe/experiment_runner.py`
- Modify: `bayesprobe/cli.py`

**Interfaces:**
- Consumes: artifact writer from Task 2.
- Produces:
  - artifact-enabled runs without explicit `ledger_path` write `artifact_dir / "ledger.jsonl"`.
  - CLI summary appends `artifact=<artifact_dir>` only when artifacts are enabled.

- [ ] **Step 1: Write failing default ledger test**

Append to `tests/test_experiment_runner.py`:

```python
def test_run_benchmark_experiment_uses_artifact_ledger_when_ledger_path_is_omitted(
    tmp_path: Path,
):
    report_path = tmp_path / "reports" / "toy-report.json"
    artifact_dir = tmp_path / "artifacts" / "toy-run"

    result = run_benchmark_experiment(
        ExperimentRunConfig(
            dataset_path=FIXTURE_PATH,
            report_path=report_path,
            artifact_dir=artifact_dir,
        )
    )

    assert result.ledger_path == artifact_dir / "ledger.jsonl"
    assert (artifact_dir / "ledger.jsonl").exists()
    record_types = [
        record["record_type"]
        for record in JsonlLedgerStore(artifact_dir / "ledger.jsonl").read_all()
    ]
    assert "benchmark_sample_result" in record_types
```

- [ ] **Step 2: Run default ledger test to verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_experiment_runner.py::test_run_benchmark_experiment_uses_artifact_ledger_when_ledger_path_is_omitted -q -p no:cacheprovider
```

Expected: FAIL because artifact runs without `ledger_path` still have no ledger.

- [ ] **Step 3: Implement effective ledger path**

In `run_benchmark_experiment`, replace ledger-path setup with:

```python
artifact_dir = Path(config.artifact_dir) if config.artifact_dir is not None else None
if config.ledger_path is not None:
    ledger_path = Path(config.ledger_path)
elif artifact_dir is not None:
    ledger_path = artifact_dir / "ledger.jsonl"
else:
    ledger_path = None
ledger = JsonlLedgerStore(ledger_path) if ledger_path is not None else None
```

Remove the later duplicate `artifact_dir = ...` assignment from Task 2.

- [ ] **Step 4: Verify default ledger test is GREEN**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_experiment_runner.py::test_run_benchmark_experiment_uses_artifact_ledger_when_ledger_path_is_omitted -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 5: Write failing CLI artifact summary test**

Add a helper variation in `tests/test_cli.py` or inline JSON in a new test:

```python
def test_cli_run_prints_artifact_summary_when_enabled(tmp_path: Path, capsys):
    config_path = tmp_path / "experiment.json"
    report_path = config_path.parent / "outputs" / "report.json"
    artifact_dir = config_path.parent / "artifacts" / "toy-run"
    write_json(
        config_path,
        {
            "dataset_path": str(FIXTURE_PATH.resolve()),
            "report_path": "outputs/report.json",
            "artifact_dir": "artifacts/toy-run",
        },
    )

    exit_code = main(["run", "--config", str(config_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert f"report={report_path}" in captured.out
    assert f"ledger={artifact_dir / 'ledger.jsonl'}" in captured.out
    assert f"artifact={artifact_dir}" in captured.out
    assert (artifact_dir / "manifest.json").exists()
```

- [ ] **Step 6: Run CLI artifact summary test to verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_cli.py::test_cli_run_prints_artifact_summary_when_enabled -q -p no:cacheprovider
```

Expected: FAIL because `_format_summary` does not print `artifact=...`.

- [ ] **Step 7: Implement CLI summary field**

In `bayesprobe/cli.py`, replace `_format_summary` with:

```python
def _format_summary(result: ExperimentRunResult) -> str:
    suite = result.suite_result
    parts = [
        "BayesProbe experiment complete:",
        f"dataset={result.dataset.dataset_name}",
        f"samples={suite.sample_count}",
        f"final_accuracy={suite.final_accuracy}",
        f"update_direction_accuracy={suite.update_direction_accuracy}",
        f"report={result.report_path}",
        f"ledger={result.ledger_path}",
    ]
    if result.artifact_dir is not None:
        parts.append(f"artifact={result.artifact_dir}")
    return " ".join(parts)
```

- [ ] **Step 8: Verify Task 3 tests are GREEN**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_experiment_runner.py::test_run_benchmark_experiment_uses_artifact_ledger_when_ledger_path_is_omitted tests/test_cli.py::test_cli_run_prints_artifact_summary_when_enabled -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 9: Commit Task 3**

```bash
git add bayesprobe/experiment_runner.py bayesprobe/cli.py tests/test_experiment_runner.py tests/test_cli.py
git commit -m "feat: default artifact runs to ledger bundle"
```

---

### Task 4: Public Surface, Architecture Docs, and Full Verification

**Files:**
- Modify: `tests/test_public_api_and_config.py`
- Modify: `bayesprobe/__init__.py`
- Modify: `docs/ARCHITECTURE.md`

**Interfaces:**
- Consumes: `ExperimentArtifactBundle` from Task 2.
- Produces: stable public export if included, updated architecture status, all tests green.

- [ ] **Step 1: Decide public export and write failing public API assertion**

If `ExperimentArtifactBundle` is intended as public SDK surface, update imports and assertions in `tests/test_public_api_and_config.py`:

```python
from bayesprobe import (
    ExperimentArtifactBundle,
    ...
)
```

Add `"ExperimentArtifactBundle"` to `expected_names`, and add:

```python
assert ExperimentArtifactBundle is not None
```

- [ ] **Step 2: Run public API assertion to verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_public_api_and_config.py::test_public_sdk_exports_supported_names -q -p no:cacheprovider
```

Expected: FAIL because `ExperimentArtifactBundle` is not exported.

- [ ] **Step 3: Export artifact bundle**

In `bayesprobe/__init__.py`, add:

```python
from bayesprobe.experiment_artifacts import ExperimentArtifactBundle
```

Add `"ExperimentArtifactBundle"` to `__all__`.

- [ ] **Step 4: Verify public API assertion is GREEN**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_public_api_and_config.py::test_public_sdk_exports_supported_names -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 5: Update architecture docs**

In `docs/ARCHITECTURE.md`, update Phase 5 status:

```markdown
### Phase 5: Persistence and Experiment Packaging

Status: stable artifact directory implemented as v0.1; SQLite persistence,
dataset split filters, and prompt registry snapshots remain future work.
```

Keep the existing goal and shape bullets, and add one sentence explaining that
artifact v0.1 writes manifest, report, ledger, config snapshot, and dataset
snapshot without changing BayesProbe core control flow.

- [ ] **Step 6: Run focused test set**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_public_api_and_config.py tests/test_experiment_runner.py tests/test_cli.py -q -p no:cacheprovider
```

Expected: all selected tests pass.

- [ ] **Step 7: Run full verification**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
git diff --check
```

Expected: all tests pass and `git diff --check` emits no errors.

- [ ] **Step 8: Commit Task 4**

```bash
git add bayesprobe/__init__.py tests/test_public_api_and_config.py docs/ARCHITECTURE.md
git commit -m "docs: record experiment artifact packaging"
```

---

## Self-Review

- Spec coverage: all configured deliverables are mapped to tasks. `artifact_dir`, `run_name`, `metadata`, manifest, report snapshot, ledger snapshot, config snapshot, dataset snapshot, secret exclusion, CLI output, and architecture docs each have a test or task.
- Placeholder scan: the plan contains no deferred implementation markers. Follow-up work is not part of this plan.
- Type consistency: `ExperimentRunConfig.artifact_dir`, `ExperimentRunResult.artifact_dir`, `ExperimentRunResult.artifact_manifest_path`, and `ExperimentArtifactBundle` names are consistent across tasks.
- Scope check: this is one subsystem at the experiment layer. Provider registry, prompt registry, benchmark scoring, and SQLite persistence remain future work.
