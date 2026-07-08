# CLI Thin Wrapper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal `bayesprobe run --config <experiment.json>` command that runs benchmark experiments through the existing public SDK.

**Architecture:** Create a focused `bayesprobe/cli.py` module that parses CLI arguments, calls `load_experiment_config`, calls `run_benchmark_experiment`, and prints one summary line. Register the script in `pyproject.toml`; the CLI must not duplicate config loading, experiment orchestration, benchmark scoring, or BayesProbe core logic.

**Tech Stack:** Python 3.11+, argparse, pathlib, subprocess tests, existing BayesProbe public SDK, pytest.

## Global Constraints

- The CLI must stay a thin wrapper over `load_experiment_config` and `run_benchmark_experiment`.
- No CLI overrides for dataset/report/ledger paths.
- No JSON output mode.
- No interactive prompts.
- No subcommands other than `run`.
- No shell completion.
- No changes to `bayesprobe.config`, `bayesprobe.experiment_runner`, benchmark scoring, evidence integration, or core control flow.
- Exit codes must be `0` for success, `1` for expected config/runtime errors, and `2` for usage errors.
- Do not attempt git commits because this workspace is not currently a git repository.

---

## File Structure

- Create `tests/test_cli.py`: behavior tests for CLI success, expected errors, usage errors, and module execution.
- Create `bayesprobe/cli.py`: CLI parser and `main(argv=None) -> int`.
- Modify `pyproject.toml`: add `[project.scripts] bayesprobe = "bayesprobe.cli:main"`.

### Task 1: CLI Tests

**Files:**
- Create: `tests/test_cli.py`

**Interfaces:**
- Consumes planned API:
  - `bayesprobe.cli.main(argv: Sequence[str] | None = None) -> int`
- Consumes fixture:
  - `fixtures/benchmarks/toy_belief_revision.json`

- [x] **Step 1: Write failing tests**

Create tests equivalent to:

```python
import json
import subprocess
import sys
from pathlib import Path

from bayesprobe.cli import main


FIXTURE_PATH = Path("fixtures/benchmarks/toy_belief_revision.json")


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_cli_run_writes_report_ledger_and_prints_summary(tmp_path: Path, capsys):
    config_path = tmp_path / "experiment.json"
    write_json(
        config_path,
        {
            "dataset_path": str(FIXTURE_PATH.resolve()),
            "report_path": "outputs/report.json",
            "ledger_path": "outputs/ledger.jsonl",
        },
    )

    exit_code = main(["run", "--config", str(config_path)])

    captured = capsys.readouterr()
    report_path = tmp_path / "outputs" / "report.json"
    ledger_path = tmp_path / "outputs" / "ledger.jsonl"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert captured.err == ""
    assert "BayesProbe experiment complete" in captured.out
    assert "dataset=toy_belief_revision" in captured.out
    assert "samples=3" in captured.out
    assert "final_accuracy=1.0" in captured.out
    assert f"report={report_path}" in captured.out
    assert f"ledger={ledger_path}" in captured.out
    assert report["sample_count"] == 3
    assert ledger_path.exists()
```

Also add:

- `test_cli_run_returns_one_for_invalid_config`
- `test_cli_run_returns_two_for_missing_required_args`
- `test_cli_module_execution_runs_experiment`

- [x] **Step 2: Verify RED**

Run:

```bash
python3 -m pytest tests/test_cli.py -q
```

Expected: failure because `bayesprobe.cli` does not exist yet.

### Task 2: CLI Implementation

**Files:**
- Create: `bayesprobe/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces:
  - `main(argv: Sequence[str] | None = None) -> int`

- [x] **Step 1: Implement argument parsing and command dispatch**

Implement:

```python
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from bayesprobe import load_experiment_config, run_benchmark_experiment


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as error:
        return int(error.code)
    if args.command == "run":
        return _run_command(args)
    parser.print_help(sys.stderr)
    return 2
```

- [x] **Step 2: Implement run command and summary formatting**

Implement `_run_command(args) -> int`:

```python
def _run_command(args: argparse.Namespace) -> int:
    try:
        config = load_experiment_config(args.config)
        result = run_benchmark_experiment(config)
    except (ValueError, OSError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(_format_summary(result))
    return 0
```

Summary must include dataset, sample count, final accuracy, update-direction accuracy, report path, and ledger path or `None`.

- [x] **Step 3: Run focused CLI tests**

Run:

```bash
python3 -m pytest tests/test_cli.py -q
```

Expected: all CLI tests pass except package script metadata may still fail if tested before Task 3.

### Task 3: Package Script

**Files:**
- Modify: `pyproject.toml`
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces script metadata:
  - `bayesprobe = "bayesprobe.cli:main"`

- [x] **Step 1: Add project script entry**

Update `pyproject.toml`:

```toml
[project.scripts]
bayesprobe = "bayesprobe.cli:main"
```

- [x] **Step 2: Run focused CLI and public API tests**

Run:

```bash
python3 -m pytest tests/test_cli.py tests/test_public_api_and_config.py -q
```

Expected: all CLI and public API tests pass.

### Task 4: Regression Verification

**Files:**
- Test: all pytest files

**Interfaces:**
- Confirms CLI wrapper does not alter existing BayesProbe behavior.

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

- Spec coverage: The plan covers CLI command behavior, module execution, summary output, expected error handling, usage errors, package script metadata, focused tests, and full regression verification.
- Placeholder scan: No unspecified implementation placeholders remain.
- Type consistency: Public names and signatures match the design spec.
