# Benchmark Dataset I/O Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add file-based benchmark dataset loading and JSON report writing around the existing in-memory benchmark harness.

**Architecture:** Create a focused `bayesprobe/benchmark_io.py` module that converts JSON/JSONL records into existing `BenchmarkSample` and `BenchmarkSignal` objects, then serializes existing `BenchmarkSuiteResult` objects into report JSON. The module must remain an I/O boundary and must not execute runs, mutate beliefs, create evidence events, or alter the BayesProbe control loop.

**Tech Stack:** Python 3.11+, dataclasses, pathlib, json, existing benchmark dataclasses, pytest.

## Global Constraints

- Preserve the BayesProbe core boundary: this slice must not create `EvidenceEvent`s, belief updates, probes, projections, or hypotheses.
- Use existing `BenchmarkSample` and `BenchmarkSignal` validation instead of duplicating benchmark semantics.
- Support only `.json` and `.jsonl` input files.
- Write only JSON reports.
- No CLI, real dataset downloaders, LLM judge, baseline systems, or statistical tests in this slice.
- Do not attempt git commits because this workspace is not currently a git repository.

---

## File Structure

- Create `tests/test_benchmark_io.py`: behavior tests for dataset loading, validation errors, report writing, and harness compatibility.
- Create `bayesprobe/benchmark_io.py`: public `BenchmarkDataset`, loader, and report writer.

### Task 1: Benchmark Dataset I/O Tests

**Files:**
- Create: `tests/test_benchmark_io.py`

**Interfaces:**
- Consumes: `BenchmarkHarness`, `BenchmarkSample`, `BenchmarkSignalShape`, and the planned `BenchmarkDataset`, `load_benchmark_dataset`, `write_benchmark_report`.
- Produces: failing tests that define JSON/JSONL dataset loading and JSON report writing behavior.

- [x] **Step 1: Write the failing tests**

Create tests with these behaviors:

```python
def test_load_benchmark_dataset_from_json_object(tmp_path: Path):
    path = tmp_path / "toy.json"
    path.write_text(json.dumps({
        "dataset_name": "toy_belief_revision",
        "metadata": {"version": "0.1"},
        "samples": [
            {
                "sample_id": "active_support",
                "question_or_claim": "Does active-only execution support H1?",
                "signal_shape": "active_only",
                "gold_best_hypothesis": "H1",
                "gold_update_directions": {"H1": "strengthened"},
            },
            {
                "sample_id": "passive_refute",
                "question_or_claim": "Does the passive signal refute H1?",
                "signal_shape": "passive_only",
                "gold_best_hypothesis": "H2",
                "gold_update_directions": {"H1": "weakened", "H2": "strengthened"},
                "passive_signals": [
                    {
                        "signal_id": "S_passive",
                        "source_type": "benchmark_stream",
                        "source": "fixture",
                        "raw_content": "REFUTES: Benchmark passage contradicts H1 and supports H2.",
                        "target_hypotheses": ["H1", "H2"],
                    }
                ],
            },
        ],
    }), encoding="utf-8")

    dataset = load_benchmark_dataset(path)

    assert isinstance(dataset, BenchmarkDataset)
    assert dataset.dataset_name == "toy_belief_revision"
    assert dataset.metadata == {"version": "0.1"}
    assert len(dataset.samples) == 2
    assert dataset.samples[0].signal_shape == BenchmarkSignalShape.ACTIVE_ONLY
    assert dataset.samples[1].passive_signals[0].signal_id == "S_passive"
```

Also add tests for raw JSON arrays, JSONL, invalid files, report writing, and running loaded samples through `BenchmarkHarness`.

- [x] **Step 2: Run tests to verify RED**

Run:

```bash
python3 -m pytest tests/test_benchmark_io.py -q
```

Expected result: failure because `bayesprobe.benchmark_io` does not exist yet.

### Task 2: Benchmark Dataset Loader

**Files:**
- Create: `bayesprobe/benchmark_io.py`
- Test: `tests/test_benchmark_io.py`

**Interfaces:**
- Produces:
  - `BenchmarkDataset`
  - `load_benchmark_dataset(path: str | Path) -> BenchmarkDataset`

- [x] **Step 1: Implement dataset dataclass and JSON/JSONL dispatch**

Implement:

```python
@dataclass(frozen=True)
class BenchmarkDataset:
    dataset_name: str
    samples: list[BenchmarkSample]
    metadata: dict[str, Any] = field(default_factory=dict)

def load_benchmark_dataset(path: str | Path) -> BenchmarkDataset:
    dataset_path = Path(path)
    if dataset_path.suffix == ".json":
        return _load_json_dataset(dataset_path)
    if dataset_path.suffix == ".jsonl":
        return _load_jsonl_dataset(dataset_path)
    raise ValueError("benchmark dataset path must end with .json or .jsonl")
```

- [x] **Step 2: Implement sample and signal coercion**

Implement helpers that convert dictionaries into existing benchmark dataclasses:

```python
def _sample_from_mapping(data: Mapping[str, Any]) -> BenchmarkSample:
    passive_signals = [
        _signal_from_mapping(signal)
        for signal in data.get("passive_signals", [])
    ]
    return BenchmarkSample(
        sample_id=data["sample_id"],
        question_or_claim=data["question_or_claim"],
        signal_shape=data.get("signal_shape", BenchmarkSignalShape.ACTIVE_ONLY),
        gold_best_hypothesis=data["gold_best_hypothesis"],
        passive_signals=passive_signals,
        gold_update_directions=dict(data.get("gold_update_directions", {})),
        initial_context=data.get("initial_context", ""),
    )
```

- [x] **Step 3: Run focused loader tests**

Run:

```bash
python3 -m pytest tests/test_benchmark_io.py -q
```

Expected result: loader tests pass; report-writing tests may still fail until Task 3 is complete.

### Task 3: Benchmark Report Writer

**Files:**
- Modify: `bayesprobe/benchmark_io.py`
- Test: `tests/test_benchmark_io.py`

**Interfaces:**
- Produces:
  - `write_benchmark_report(path: str | Path, suite_result: BenchmarkSuiteResult, *, dataset_name: str = "", metadata: dict[str, Any] | None = None) -> None`

- [x] **Step 1: Implement suite result serialization**

Implement JSON output with:

```python
def write_benchmark_report(
    path: str | Path,
    suite_result: BenchmarkSuiteResult,
    *,
    dataset_name: str = "",
    metadata: dict[str, Any] | None = None,
) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_name": dataset_name or report_path.stem,
        "metadata": dict(metadata or {}),
        "sample_count": suite_result.sample_count,
        "final_accuracy": suite_result.final_accuracy,
        "update_direction_accuracy": suite_result.update_direction_accuracy,
        "results": [_sample_result_to_dict(result) for result in suite_result.results],
    }
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
```

- [x] **Step 2: Run focused report tests**

Run:

```bash
python3 -m pytest tests/test_benchmark_io.py tests/test_benchmark_harness.py -q
```

Expected result: all benchmark I/O and harness tests pass.

### Task 4: Regression Verification

**Files:**
- Test: all pytest files

**Interfaces:**
- Confirms: dataset I/O integrates with existing BayesProbe modules without changing core semantics.

- [x] **Step 1: Run full test suite**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
```

Expected result: all tests pass with no failures.

- [x] **Step 2: Remove generated caches**

Run:

```bash
find . -type d \( -name __pycache__ -o -name .pytest_cache \) -prune -exec rm -rf {} +
```

Expected result: no generated cache directories remain.

## Self-Review

- Spec coverage: The plan covers JSON object loading, raw JSON array loading, JSONL loading, validation errors, report writing, and harness compatibility.
- Placeholder scan: No unspecified implementation placeholders remain.
- Type consistency: Public names match the design spec and tests.
