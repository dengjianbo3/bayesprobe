# Benchmark Harness MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic benchmark harness that runs BayesProbe samples across active-only, passive-only, and active-plus-passive cycle shapes and reports basic dual-objective metrics.

**Architecture:** Add a focused `bayesprobe/benchmark.py` module. It coordinates existing initializer, controllers, planner, executor, core integration, and projection helpers while preserving the controller/core boundary: benchmark code may create raw signals and cycles, but it must not create evidence events or update beliefs directly.

**Tech Stack:** Python 3.11+, dataclasses, `enum.StrEnum`, existing Pydantic schemas, pytest.

## Global Constraints

- The harness must not create `EvidenceEvent`s directly.
- The harness must not update posterior beliefs directly.
- The harness must not spawn, reframe, retire, or evolve hypotheses directly.
- Passive benchmark inputs enter as `ExternalSignal` records with `signal_kind=PASSIVE`.
- Active benchmark outputs come from `ProbeExecutor`.
- Mixed samples must use `CycleSignalShape.ACTIVE_PLUS_PASSIVE`.
- No real web search, document retrieval, LLM calls, baseline systems, statistical tests, or human evaluation in this slice.
- Do not attempt git commits because this workspace is not currently a git repository.

---

## File Structure

- Create `tests/test_benchmark_harness.py`: behavior tests for active-only, passive-only, active-plus-passive, aggregation, validation, and ledger smoke.
- Create `bayesprobe/benchmark.py`: public benchmark sample/result dataclasses and harness orchestration logic.

### Task 1: Benchmark Harness Tests

**Files:**
- Create: `tests/test_benchmark_harness.py`

**Interfaces:**
- Consumes: `BenchmarkHarness`, `BenchmarkSample`, `BenchmarkSignal`, `BenchmarkSignalShape`
- Produces: failing tests for benchmark harness behavior before production code exists

- [x] **Step 1: Write failing tests**

Create tests covering:

- `test_benchmark_harness_runs_active_only_sample`
- `test_benchmark_harness_runs_passive_only_sample`
- `test_benchmark_harness_runs_active_plus_passive_sample`
- `test_benchmark_harness_aggregates_suite_metrics`
- `test_benchmark_harness_rejects_invalid_samples`
- `test_benchmark_harness_preserves_ledger_records`

- [x] **Step 2: Verify RED**

Run:

```bash
python3 -m pytest tests/test_benchmark_harness.py -q
```

Expected result: failure because `bayesprobe.benchmark` does not exist yet.

### Task 2: Benchmark Harness Implementation

**Files:**
- Create: `bayesprobe/benchmark.py`

**Interfaces:**
- Consumes:
  - `BayesProbeCore`
  - `BayesProbeInitializer`
  - `AutonomousQuestionRunner`
  - `SynchronizedController`
  - `ProbePlanner`
  - `ProbeExecutor`
  - `build_answer_projection`
- Produces:
  - `BenchmarkSignalShape`
  - `BenchmarkSignal`
  - `BenchmarkSample`
  - `BenchmarkSampleResult`
  - `BenchmarkSuiteResult`
  - `BenchmarkHarness`

- [x] **Step 1: Add public dataclasses and enum**

Implement:

- `BenchmarkSignalShape`
- `BenchmarkSignal`
- `BenchmarkSample`
- `BenchmarkSampleResult`
- `BenchmarkSuiteResult`

Validation must reject empty required fields and passive/mixed samples without passive signals.

- [x] **Step 2: Implement active-only runner path**

`BenchmarkHarness.run_sample(...)` dispatches active-only samples to `AutonomousQuestionRunner.run_question(...)` and scores:

- `final_best_hypothesis`
- `final_correct`
- `update_direction_accuracy`
- `cycle_count`
- `signal_count`
- `evidence_event_count`
- `belief_update_count`
- `projection_kind="answer_projection"`

- [x] **Step 3: Implement passive-only runner path**

Passive-only samples use:

- `BayesProbeInitializer.initialize(...)`
- `SynchronizedController.process_round(...)`
- benchmark-provided passive `ExternalSignal`s

The result must expose `projection_kind="belief_state_projection"` and zero active probes.

- [x] **Step 4: Implement active-plus-passive runner path**

Mixed samples use:

- initializer
- planner
- executor
- benchmark passive signals
- `CycleRecord(signal_shape=ACTIVE_PLUS_PASSIVE)`
- `BayesProbeCore.integrate_cycle(...)`
- `build_answer_projection(...)`

The result must include both active and passive signal counts.

- [x] **Step 5: Implement suite aggregation**

`run_suite(samples)` returns:

- `sample_count`
- `results`
- `final_accuracy`
- `update_direction_accuracy`

- [x] **Step 6: Verify GREEN**

Run:

```bash
python3 -m pytest tests/test_benchmark_harness.py -q
```

Expected result: all benchmark harness tests pass.

### Task 3: Regression Verification

**Files:**
- Test: all pytest files

**Interfaces:**
- Confirms: benchmark harness integrates with existing modules without changing core semantics.

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

- Spec coverage: The plan covers all three cycle shapes, scoring, validation, aggregation, and ledger smoke tests.
- Placeholder scan: No unspecified implementation placeholders remain.
- Type consistency: Public class names and method signatures match the design spec.
