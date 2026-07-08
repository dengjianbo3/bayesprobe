# Public SDK and Config Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose a stable BayesProbe public Python API and add a JSON experiment config loader so external code can import, configure, and run benchmark experiments.

**Architecture:** Add a focused `bayesprobe/config.py` module that converts JSON or mapping data into `ExperimentRunConfig`, resolving relative paths against a config base directory. Update `bayesprobe/__init__.py` to re-export the supported MVP SDK surface without changing core, evidence, benchmark, or experiment-runner behavior.

**Tech Stack:** Python 3.11+, dataclasses already in use, pathlib, json, existing BayesProbe benchmark and experiment runner modules, pytest.

## Global Constraints

- No CLI in this slice.
- No YAML/TOML config support.
- No environment variable interpolation.
- No plugin system or dynamic component registry.
- No changes to BayesProbe core control flow, evidence integration, benchmark scoring, or experiment runner behavior.
- `load_experiment_config(path)` must resolve relative path fields against the config file's parent directory.
- `experiment_config_from_mapping(data, base_dir=None)` must leave relative paths relative when `base_dir` is `None`.
- Do not attempt git commits because this workspace is not currently a git repository.

---

## File Structure

- Create `tests/test_public_api_and_config.py`: public import and JSON config behavior tests.
- Create `bayesprobe/config.py`: config parsing, path resolution, and validation.
- Modify `bayesprobe/__init__.py`: supported public API exports.

### Task 1: Public API and Config Tests

**Files:**
- Create: `tests/test_public_api_and_config.py`

**Interfaces:**
- Consumes planned public names from `bayesprobe`.
- Consumes planned `load_experiment_config` and `experiment_config_from_mapping`.
- Produces failing tests for public SDK exports and config loading behavior.

- [x] **Step 1: Write failing tests**

Create tests equivalent to:

```python
import json
from pathlib import Path

import pytest

import bayesprobe
from bayesprobe import (
    BenchmarkDataset,
    BenchmarkHarness,
    BenchmarkSample,
    BenchmarkSampleResult,
    BenchmarkSignal,
    BenchmarkSignalShape,
    BenchmarkSuiteResult,
    ExperimentRunConfig,
    ExperimentRunResult,
    load_benchmark_dataset,
    load_experiment_config,
    run_benchmark_experiment,
    write_benchmark_report,
)
from bayesprobe.config import experiment_config_from_mapping


def test_public_sdk_exports_supported_names():
    expected_names = {
        "BenchmarkDataset",
        "BenchmarkHarness",
        "BenchmarkSample",
        "BenchmarkSampleResult",
        "BenchmarkSignal",
        "BenchmarkSignalShape",
        "BenchmarkSuiteResult",
        "ExperimentRunConfig",
        "ExperimentRunResult",
        "load_benchmark_dataset",
        "load_experiment_config",
        "run_benchmark_experiment",
        "write_benchmark_report",
    }

    assert expected_names.issubset(set(bayesprobe.__all__))
    assert BenchmarkHarness is not None
    assert ExperimentRunConfig is not None
    assert load_experiment_config is not None
```

Also add tests for:

- JSON config path resolution relative to the config file.
- Mapping config path resolution with `base_dir`.
- Mapping config leaving relative paths relative without `base_dir`.
- Running loaded config through `run_benchmark_experiment`.
- Invalid extension, malformed JSON, missing path fields, non-string path fields, and non-integer numeric fields.

- [x] **Step 2: Verify RED**

Run:

```bash
python3 -m pytest tests/test_public_api_and_config.py -q
```

Expected: failure because `bayesprobe.config` or public exports do not exist yet.

### Task 2: Config Loader

**Files:**
- Create: `bayesprobe/config.py`
- Test: `tests/test_public_api_and_config.py`

**Interfaces:**
- Produces:
  - `load_experiment_config(path: str | Path) -> ExperimentRunConfig`
  - `experiment_config_from_mapping(data: Mapping[str, Any], *, base_dir: str | Path | None = None) -> ExperimentRunConfig`

- [x] **Step 1: Implement JSON loading and mapping conversion**

Implement:

```python
from __future__ import annotations

import json
from collections.abc import Mapping
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from bayesprobe.experiment_runner import ExperimentRunConfig


def load_experiment_config(path: str | Path) -> ExperimentRunConfig:
    config_path = Path(path)
    if config_path.suffix.lower() != ".json":
        raise ValueError("experiment config path must end with .json")
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except JSONDecodeError as error:
        raise ValueError("could not parse experiment config JSON") from error
    if not isinstance(payload, Mapping):
        raise ValueError("experiment config must be a JSON object")
    return experiment_config_from_mapping(payload, base_dir=config_path.parent)
```

- [x] **Step 2: Implement validation and path resolution**

Implement helpers:

```python
def experiment_config_from_mapping(
    data: Mapping[str, Any],
    *,
    base_dir: str | Path | None = None,
) -> ExperimentRunConfig:
    dataset_path = _required_path(data, "dataset_path", base_dir=base_dir)
    report_path = _required_path(data, "report_path", base_dir=base_dir)
    ledger_path = _optional_path(data, "ledger_path", base_dir=base_dir)
    return ExperimentRunConfig(
        dataset_path=dataset_path,
        report_path=report_path,
        ledger_path=ledger_path,
        max_cycles=_optional_int(data, "max_cycles", default=1),
        max_probes_per_cycle=_optional_int(data, "max_probes_per_cycle", default=1),
    )
```

- [x] **Step 3: Run focused config tests**

Run:

```bash
python3 -m pytest tests/test_public_api_and_config.py -q
```

Expected: config tests pass except public facade exports may still fail until Task 3.

### Task 3: Public Package Facade

**Files:**
- Modify: `bayesprobe/__init__.py`
- Test: `tests/test_public_api_and_config.py`

**Interfaces:**
- Produces supported package-level exports listed in the design spec.

- [x] **Step 1: Re-export supported MVP API**

Update `bayesprobe/__init__.py` to import and expose:

```python
from bayesprobe.benchmark import (
    BenchmarkHarness,
    BenchmarkSample,
    BenchmarkSampleResult,
    BenchmarkSignal,
    BenchmarkSignalShape,
    BenchmarkSuiteResult,
)
from bayesprobe.benchmark_io import (
    BenchmarkDataset,
    load_benchmark_dataset,
    write_benchmark_report,
)
from bayesprobe.config import load_experiment_config
from bayesprobe.experiment_runner import (
    ExperimentRunConfig,
    ExperimentRunResult,
    run_benchmark_experiment,
)
```

Set `__all__` exactly to the supported public names.

- [x] **Step 2: Run focused public API tests**

Run:

```bash
python3 -m pytest tests/test_public_api_and_config.py tests/test_experiment_runner.py -q
```

Expected: all public API, config, and experiment runner tests pass.

### Task 4: Regression Verification

**Files:**
- Test: all pytest files

**Interfaces:**
- Confirms public facade and config loader do not alter existing behavior.

- [x] **Step 1: Run full test suite**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
```

Expected: all tests pass with no failures.

- [x] **Step 2: Remove generated caches**

Run:

```bash
find . -type d \( -name __pycache__ -o -name .pytest_cache \) -prune -exec rm -rf {} +
```

Expected: no generated cache directories remain.

## Self-Review

- Spec coverage: The plan covers public facade exports, JSON config loading, mapping config loading, path resolution, validation errors, experiment execution from loaded config, focused tests, and full regression verification.
- Placeholder scan: No unspecified implementation placeholders remain.
- Type consistency: Public names and function signatures match the design spec.
