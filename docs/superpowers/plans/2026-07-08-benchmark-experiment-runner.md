# Benchmark Experiment Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a repeatable benchmark experiment runner that loads the toy dataset fixture, runs the existing harness, writes a JSON report, and optionally records a ledger.

**Architecture:** Add a thin `bayesprobe/experiment_runner.py` orchestration module on top of `benchmark_io`, `BenchmarkHarness`, and `JsonlLedgerStore`. Add one checked-in toy dataset fixture under `fixtures/benchmarks`; the runner remains an I/O and orchestration layer and does not change BayesProbe core control flow or scoring.

**Tech Stack:** Python 3.11+, dataclasses, pathlib, json, existing BayesProbe benchmark modules, pytest.

## Global Constraints

- No changes to `BayesProbeCore`, `EvidenceIntegrationGate`, controllers, planner, executor, or benchmark scoring.
- No CLI in this slice.
- No FEVER, PubMedQA, web retrieval, model calls, dataset downloaders, baselines, LLM evaluator, or statistical significance testing.
- The runner must not inspect or mutate hypotheses, evidence events, projections, probes, or belief updates.
- The fixture must include `active_only`, `passive_only`, and `active_plus_passive` samples.
- Do not attempt git commits because this workspace is not currently a git repository.

---

## File Structure

- Create `fixtures/benchmarks/toy_belief_revision.json`: deterministic three-sample benchmark fixture.
- Create `tests/test_experiment_runner.py`: TDD tests for end-to-end experiment execution, optional ledger output, and config validation.
- Create `bayesprobe/experiment_runner.py`: public config/result dataclasses and `run_benchmark_experiment`.

### Task 1: Experiment Runner Tests

**Files:**
- Create: `tests/test_experiment_runner.py`

**Interfaces:**
- Consumes planned API:
  - `ExperimentRunConfig`
  - `run_benchmark_experiment(config: ExperimentRunConfig)`
- Consumes existing fixture path:
  - `fixtures/benchmarks/toy_belief_revision.json`
- Produces failing tests for the experiment runner and fixture.

- [x] **Step 1: Write the failing tests**

Add tests equivalent to:

```python
import json
from pathlib import Path

import pytest

from bayesprobe.experiment_runner import (
    ExperimentRunConfig,
    run_benchmark_experiment,
)


FIXTURE_PATH = Path("fixtures/benchmarks/toy_belief_revision.json")


def test_run_benchmark_experiment_writes_report(tmp_path: Path):
    report_path = tmp_path / "reports" / "toy-report.json"

    result = run_benchmark_experiment(
        ExperimentRunConfig(
            dataset_path=FIXTURE_PATH,
            report_path=report_path,
        )
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert result.dataset.dataset_name == "toy_belief_revision"
    assert result.report_path == report_path
    assert result.ledger_path is None
    assert result.suite_result.sample_count == 3
    assert result.suite_result.final_accuracy == 1.0
    assert payload["dataset_name"] == "toy_belief_revision"
    assert payload["sample_count"] == 3
    assert [item["sample_id"] for item in payload["results"]] == [
        "toy_active_support",
        "toy_passive_refute",
        "toy_mixed_refute",
    ]
```

Also add:

- `test_run_benchmark_experiment_writes_optional_ledger`
- `test_run_benchmark_experiment_rejects_invalid_config`

- [x] **Step 2: Verify RED**

Run:

```bash
python3 -m pytest tests/test_experiment_runner.py -q
```

Expected: failure because `bayesprobe.experiment_runner` or the fixture does not exist.

### Task 2: Toy Benchmark Fixture

**Files:**
- Create: `fixtures/benchmarks/toy_belief_revision.json`
- Test: `tests/test_experiment_runner.py`

**Interfaces:**
- Produces a JSON object accepted by `load_benchmark_dataset(path)`.

- [x] **Step 1: Add fixture dataset**

Create JSON with:

```json
{
  "dataset_name": "toy_belief_revision",
  "metadata": {
    "version": "0.1",
    "purpose": "Deterministic fixture covering active-only, passive-only, and active-plus-passive BayesProbe benchmark paths."
  },
  "samples": [
    {
      "sample_id": "toy_active_support",
      "question_or_claim": "Does the autonomous active path support H1?",
      "signal_shape": "active_only",
      "gold_best_hypothesis": "H1",
      "gold_update_directions": {"H1": "strengthened"}
    },
    {
      "sample_id": "toy_passive_refute",
      "question_or_claim": "Does the passive benchmark stream refute H1?",
      "signal_shape": "passive_only",
      "gold_best_hypothesis": "H2",
      "gold_update_directions": {"H1": "weakened", "H2": "strengthened"},
      "passive_signals": [
        {
          "signal_id": "S_toy_passive_refute",
          "source_type": "benchmark_stream",
          "source": "fixture",
          "raw_content": "REFUTES: Benchmark passage contradicts H1 and supports H2.",
          "target_hypotheses": ["H1", "H2"]
        }
      ]
    },
    {
      "sample_id": "toy_mixed_refute",
      "question_or_claim": "Can mixed active and passive signals move belief toward H2?",
      "signal_shape": "active_plus_passive",
      "gold_best_hypothesis": "H2",
      "gold_update_directions": {"H2": "strengthened"},
      "passive_signals": [
        {
          "signal_id": "S_toy_mixed_refute",
          "source_type": "benchmark_stream",
          "source": "fixture",
          "raw_content": "REFUTES: Benchmark passage contradicts H1 and supports H2.",
          "target_hypotheses": ["H1", "H2"]
        }
      ]
    }
  ]
}
```

- [x] **Step 2: Run focused test**

Run:

```bash
python3 -m pytest tests/test_experiment_runner.py -q
```

Expected: still fails because `bayesprobe.experiment_runner` is not implemented.

### Task 3: Experiment Runner Implementation

**Files:**
- Create: `bayesprobe/experiment_runner.py`
- Test: `tests/test_experiment_runner.py`

**Interfaces:**
- Produces:
  - `ExperimentRunConfig`
  - `ExperimentRunResult`
  - `run_benchmark_experiment(config: ExperimentRunConfig) -> ExperimentRunResult`

- [x] **Step 1: Implement dataclasses and validation**

Implement:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bayesprobe.benchmark import BenchmarkHarness, BenchmarkSuiteResult
from bayesprobe.benchmark_io import (
    BenchmarkDataset,
    load_benchmark_dataset,
    write_benchmark_report,
)
from bayesprobe.ledger import JsonlLedgerStore


@dataclass(frozen=True)
class ExperimentRunConfig:
    dataset_path: str | Path
    report_path: str | Path
    ledger_path: str | Path | None = None
    max_cycles: int = 1
    max_probes_per_cycle: int = 1

    def __post_init__(self) -> None:
        if self.max_cycles < 1:
            raise ValueError("max_cycles must be at least 1")
        if self.max_probes_per_cycle < 1:
            raise ValueError("max_probes_per_cycle must be at least 1")


@dataclass(frozen=True)
class ExperimentRunResult:
    dataset: BenchmarkDataset
    suite_result: BenchmarkSuiteResult
    report_path: Path
    ledger_path: Path | None = None
```

- [x] **Step 2: Implement orchestration**

Implement:

```python
def run_benchmark_experiment(config: ExperimentRunConfig) -> ExperimentRunResult:
    dataset = load_benchmark_dataset(config.dataset_path)
    ledger_path = Path(config.ledger_path) if config.ledger_path is not None else None
    ledger = JsonlLedgerStore(ledger_path) if ledger_path is not None else None
    harness = BenchmarkHarness(
        ledger=ledger,
        max_cycles=config.max_cycles,
        max_probes_per_cycle=config.max_probes_per_cycle,
    )
    suite_result = harness.run_suite(dataset.samples)
    report_path = Path(config.report_path)
    write_benchmark_report(
        report_path,
        suite_result,
        dataset_name=dataset.dataset_name,
        metadata=dataset.metadata,
    )
    return ExperimentRunResult(
        dataset=dataset,
        suite_result=suite_result,
        report_path=report_path,
        ledger_path=ledger_path,
    )
```

- [x] **Step 3: Run focused tests**

Run:

```bash
python3 -m pytest tests/test_experiment_runner.py tests/test_benchmark_io.py tests/test_benchmark_harness.py -q
```

Expected: all experiment, I/O, and harness tests pass.

### Task 4: Regression Verification

**Files:**
- Test: all pytest files

**Interfaces:**
- Confirms the new runner does not alter existing BayesProbe behavior.

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

- Spec coverage: The plan covers fixture creation, public Python API, optional ledger output, report writing, validation, focused tests, and full regression verification.
- Placeholder scan: No unspecified implementation placeholders remain.
- Type consistency: Public names and signatures match the design spec.
