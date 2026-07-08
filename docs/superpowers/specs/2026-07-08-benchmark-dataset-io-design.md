# Benchmark Dataset I/O Design

Date: 2026-07-08
Status: Approved by continuation from benchmark harness MVP

## Goal

Add a small dataset I/O layer so BayesProbe benchmark samples can be loaded from fixture files and suite results can be written as stable machine-readable reports.

This is an engineering bridge between the in-memory `BenchmarkHarness` and future real benchmark datasets. It must make experiments repeatable without expanding the agent core or changing the BayesProbe control loop.

## Scope

The slice covers:

- Loading benchmark samples from `.json` files.
- Loading benchmark samples from `.jsonl` files.
- Supporting JSON object datasets with `dataset_name`, `metadata`, and `samples`.
- Supporting raw JSON sample arrays.
- Writing benchmark suite reports to `.json`.
- Preserving signal-shape values, result metrics, and per-sample counts in reports.

## Non-Goals

- No FEVER, PubMedQA, or custom dataset downloaders.
- No CLI.
- No LLM judge.
- No baseline systems.
- No statistical significance testing.
- No schema migration/versioning system.

## Public API

Create `bayesprobe/benchmark_io.py` with:

- `BenchmarkDataset`
- `load_benchmark_dataset(path: str | Path) -> BenchmarkDataset`
- `write_benchmark_report(path: str | Path, suite_result: BenchmarkSuiteResult, *, dataset_name: str = "", metadata: dict[str, Any] | None = None) -> None`

`BenchmarkDataset.samples` contains existing `BenchmarkSample` objects. This keeps file parsing separate from benchmark execution.

## Input Schema

JSON object form:

```json
{
  "dataset_name": "toy_belief_revision",
  "metadata": {"version": "0.1"},
  "samples": [
    {
      "sample_id": "s1",
      "question_or_claim": "Does the passive signal refute H1?",
      "signal_shape": "passive_only",
      "gold_best_hypothesis": "H2",
      "gold_update_directions": {"H1": "weakened", "H2": "strengthened"},
      "initial_context": "",
      "passive_signals": [
        {
          "signal_id": "S1",
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

Raw JSON array form:

```json
[
  {
    "sample_id": "s1",
    "question_or_claim": "Does active-only execution support H1?",
    "signal_shape": "active_only",
    "gold_best_hypothesis": "H1"
  }
]
```

JSONL form:

```jsonl
{"sample_id":"s1","question_or_claim":"Does active-only execution support H1?","signal_shape":"active_only","gold_best_hypothesis":"H1"}
{"sample_id":"s2","question_or_claim":"Does the passive signal refute H1?","signal_shape":"passive_only","gold_best_hypothesis":"H2","passive_signals":[{"signal_id":"S2","source_type":"benchmark_stream","source":"fixture","raw_content":"REFUTES: Benchmark passage contradicts H1 and supports H2.","target_hypotheses":["H1","H2"]}]}
```

For raw arrays and JSONL, `dataset_name` defaults to the file stem and `metadata` defaults to `{}`.

## Report Schema

Reports are JSON objects:

```json
{
  "dataset_name": "toy_belief_revision",
  "metadata": {"version": "0.1"},
  "sample_count": 1,
  "final_accuracy": 1.0,
  "update_direction_accuracy": 1.0,
  "results": [
    {
      "sample_id": "s1",
      "run_id": "bench_s1",
      "signal_shape": "passive_only",
      "final_best_hypothesis": "H2",
      "gold_best_hypothesis": "H2",
      "final_correct": true,
      "update_direction_accuracy": 1.0,
      "cycle_count": 1,
      "signal_count": 1,
      "active_signal_count": 0,
      "passive_signal_count": 1,
      "evidence_event_count": 1,
      "belief_update_count": 2,
      "projection_kind": "belief_state_projection"
    }
  ]
}
```

## Validation

The loader raises `ValueError` when:

- The file extension is not `.json` or `.jsonl`.
- JSON parsing fails.
- A JSON dataset object does not contain `samples`.
- A sample entry is not an object.
- A signal entry is not an object.
- Required sample or signal fields are missing.
- Existing `BenchmarkSample` or `BenchmarkSignal` validation fails.

## Test Strategy

Add `tests/test_benchmark_io.py` covering:

- JSON object dataset loading.
- Raw JSON array loading.
- JSONL loading.
- Invalid extension, malformed JSON, non-object sample, and missing field errors.
- Report writing from a real `BenchmarkSuiteResult`.
- Loaded samples running through `BenchmarkHarness`.
