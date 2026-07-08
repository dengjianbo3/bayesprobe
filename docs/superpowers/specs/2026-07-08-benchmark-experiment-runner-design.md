# Benchmark Experiment Runner Design

Date: 2026-07-08
Status: Approved from confirmed next-step design

## Goal

Build a repeatable benchmark experiment pipeline that loads a fixture dataset, runs BayesProbe through the existing benchmark harness, writes a JSON report, and optionally records the BayesProbe ledger.

This slice turns the current benchmark components into an executable experiment loop without changing the BayesProbe agent paradigm, evidence gate, controller flow, or belief-update semantics.

## Scope

The first version covers:

- A checked-in toy benchmark dataset with `active_only`, `passive_only`, and `active_plus_passive` samples.
- A Python API for running a benchmark experiment from dataset path to report path.
- Optional ledger output so experiments can be inspected through existing ledger records.
- Tests that prove the fixture can be loaded, run, scored, and written as a report.

## Non-Goals

- No CLI in this slice.
- No FEVER, PubMedQA, web retrieval, model calls, or dataset downloaders.
- No ReAct/ReWOO/direct-answer baselines.
- No LLM evaluator.
- No statistical significance testing.
- No changes to `BayesProbeCore`, `EvidenceIntegrationGate`, controllers, planner, executor, or benchmark scoring.

## Public API

Create `bayesprobe/experiment_runner.py` with:

- `ExperimentRunConfig`
- `ExperimentRunResult`
- `run_benchmark_experiment(config: ExperimentRunConfig) -> ExperimentRunResult`

`ExperimentRunConfig` fields:

- `dataset_path: str | Path`
- `report_path: str | Path`
- `ledger_path: str | Path | None = None`
- `max_cycles: int = 1`
- `max_probes_per_cycle: int = 1`

`ExperimentRunResult` fields:

- `dataset: BenchmarkDataset`
- `suite_result: BenchmarkSuiteResult`
- `report_path: Path`
- `ledger_path: Path | None = None`

## Data Flow

```text
ExperimentRunConfig
→ load_benchmark_dataset(dataset_path)
→ optional JsonlLedgerStore(ledger_path)
→ BenchmarkHarness(...).run_suite(dataset.samples)
→ write_benchmark_report(report_path, suite_result, dataset_name, metadata)
→ ExperimentRunResult
```

The runner is orchestration only. It must not inspect or mutate hypotheses, evidence events, projections, probes, or belief updates.

## Fixture Dataset

Create `fixtures/benchmarks/toy_belief_revision.json` with:

- `dataset_name`: `toy_belief_revision`
- `metadata.version`: `0.1`
- `metadata.purpose`: a short description of this deterministic fixture
- three samples:
  - `toy_active_support`: active-only sample expecting `H1`
  - `toy_passive_refute`: passive-only sample with benchmark stream evidence expecting `H2`
  - `toy_mixed_refute`: active-plus-passive sample expecting `H2`

The passive signals should use `source_type="benchmark_stream"` and deterministic `REFUTES:` content so the current evidence gate and harness produce stable metrics.

## Report Behavior

The report is exactly the JSON payload produced by `write_benchmark_report`:

- `dataset_name`
- `metadata`
- `sample_count`
- `final_accuracy`
- `update_direction_accuracy`
- `results`

The runner must create parent directories for the report through the existing report writer.

## Ledger Behavior

If `ledger_path` is provided:

- Instantiate `JsonlLedgerStore(ledger_path)`.
- Pass it into `BenchmarkHarness`.
- Return the resolved `ledger_path`.

If `ledger_path` is not provided:

- Use the harness without a user-visible ledger path.
- Return `ledger_path=None`.

## Validation

The runner raises `ValueError` when:

- `max_cycles < 1`
- `max_probes_per_cycle < 1`

Dataset/report parsing and write errors should surface from `benchmark_io` and standard filesystem behavior.

## Test Strategy

Add `tests/test_experiment_runner.py` covering:

- The checked-in toy fixture runs end-to-end and writes a report.
- Optional ledger output records benchmark/core events.
- Invalid config values raise `ValueError`.

Run focused tests first, then the full pytest suite.
