# Synchronized Round Runner MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a first-class fixed-round synchronized runner that executes passive-only, active-only, and active-plus-passive BayesProbe rounds and emits `BeliefStateProjection`s.

**Architecture:** Add `bayesprobe/synchronized_runner.py` as a run-level orchestrator over existing initializer, planner, executor, core integration, and projection helpers. It preserves the controller/core boundary by collecting raw `ExternalSignal`s and delegating evidence construction, likelihood judgment, posterior updates, and hypothesis evolution to `BayesProbeCore.integrate_cycle`.

**Tech Stack:** Python 3.11+, dataclasses, `enum.StrEnum`, existing Pydantic schemas, existing JSONL ledger, pytest.

## Global Constraints

- The runner must not create `EvidenceEvent`s directly.
- The runner must not update posterior beliefs directly.
- The runner must not spawn, split, reframe, reject, retire, or reactivate hypotheses directly.
- Passive signals remain passive; the runner must not wrap them in fake active probes.
- External projections from other agents are passive `ExternalSignal`s, not evidence.
- All round shapes must use `BayesProbeCore.integrate_cycle`.
- No real-time scheduler, networked agent transport, LLM calls, web search, document retrieval, or benchmark scoring changes in this slice.
- Do not attempt git commits because this workspace is not currently a git repository.

---

## File Structure

- Create `tests/test_synchronized_runner.py`: behavior tests for round validation, new-run and existing-run execution, three signal shapes, candidate carry-forward, and ledger records.
- Create `bayesprobe/synchronized_runner.py`: public synchronized runner dataclasses, enum, and orchestration logic.

### Task 1: Synchronized Runner Tests

**Files:**
- Create: `tests/test_synchronized_runner.py`

**Interfaces:**
- Consumes: `SynchronizedRoundRunner`, `SynchronizedRunInput`, `SynchronizedRoundInput`, `SynchronizedRoundShape`, `InitializeRunInput`, `ExternalSignal`
- Produces: failing tests before production code exists

- [x] **Step 1: Write failing tests**

Create tests that cover:

- `test_synchronized_runner_processes_new_run_passive_only_round`
- `test_synchronized_runner_processes_active_only_round`
- `test_synchronized_runner_processes_active_plus_passive_round`
- `test_synchronized_runner_carries_projection_candidates_across_rounds`
- `test_synchronized_runner_accepts_existing_run_state`
- `test_synchronized_runner_rejects_invalid_round_configuration`
- `test_synchronized_runner_writes_projection_ledger_records_without_duplicate_cycles`

- [x] **Step 2: Verify RED**

Run:

```bash
python3 -m pytest tests/test_synchronized_runner.py -q
```

Expected result: failure because `bayesprobe.synchronized_runner` does not exist yet.

### Task 2: Synchronized Runner Implementation

**Files:**
- Create: `bayesprobe/synchronized_runner.py`

**Interfaces:**
- Consumes:
  - `BayesProbeCore`
  - `BayesProbeInitializer`
  - `ProbePlanner`
  - `ProbeExecutor`
  - `DeterministicProbeToolGateway`
  - `build_belief_state_projection`
- Produces:
  - `SynchronizedRoundShape`
  - `SynchronizedRoundInput`
  - `SynchronizedRunInput`
  - `SynchronizedRoundResult`
  - `SynchronizedRunResult`
  - `SynchronizedRoundRunner.run_rounds(input: SynchronizedRunInput) -> SynchronizedRunResult`

- [x] **Step 1: Add public dataclasses and enum**

Implement validation:

- non-empty `round_id`
- `max_probes >= 1`
- passive-only and mixed rounds require passive signals
- active-only rejects passive signals
- passive signal entries must have `SignalKind.PASSIVE`
- run input requires either `initialize_input`, or both `run` and `belief_state`
- existing `run.run_id` must match `belief_state.run_id`
- `rounds` must be non-empty

- [x] **Step 2: Implement passive-only round path**

Create an empty `ProbeSet`, a `CycleRecord(signal_shape=PASSIVE_ONLY)`, call `core.integrate_cycle(...)`, then call `build_belief_state_projection(...)`.

- [x] **Step 3: Implement active-only round path**

Plan probes from the current candidate pool, execute active probes, integrate an `ACTIVE_ONLY` cycle, and emit a belief-state projection.

- [x] **Step 4: Implement active-plus-passive round path**

Plan and execute active probes, combine active and passive signals, integrate an `ACTIVE_PLUS_PASSIVE` cycle, and emit a belief-state projection.

- [x] **Step 5: Implement candidate carry-forward**

After each round, remove selected candidate ids, then put projection-derived change-my-mind candidates before remaining candidates.

- [x] **Step 6: Implement ledger projection write**

If the core has a ledger, append one `belief_state_projection` record per completed round. Do not append custom runner records.

- [x] **Step 7: Verify GREEN**

Run:

```bash
python3 -m pytest tests/test_synchronized_runner.py -q
```

Expected result: synchronized runner tests pass.

### Task 3: Regression Verification

**Files:**
- Test: all pytest files

**Interfaces:**
- Confirms: synchronized runner integrates with existing modules without changing core semantics.

- [x] **Step 1: Run focused integration tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_synchronized_runner.py tests/test_controllers.py tests/test_benchmark_harness.py -q -p no:cacheprovider
```

Expected result: all focused tests pass.

- [x] **Step 2: Run full test suite**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
```

Expected result: all tests pass with no failures.

- [x] **Step 3: Remove generated caches**

Run:

```bash
find . -type d \( -name __pycache__ -o -name .pytest_cache \) -prune -exec rm -rf {} +
```

Expected result: no generated cache directories remain.

## Self-Review

- Spec coverage: The plan covers one-round and multi-round synchronized execution, all three round shapes, validation, candidate carry-forward, and projection ledger writes.
- Placeholder scan: No unspecified implementation placeholders remain.
- Type consistency: Public class names and method signatures match the design spec.
