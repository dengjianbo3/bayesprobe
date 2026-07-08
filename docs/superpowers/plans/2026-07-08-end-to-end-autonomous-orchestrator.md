# End-to-End Autonomous Orchestrator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a user-facing autonomous question runner that coordinates initializer, planner, executor, core integration, projection generation, and explicit stop reasons.

**Architecture:** Add `bayesprobe/question_runner.py` as a run-level orchestrator. It delegates state initialization to `BayesProbeInitializer`, probe selection to `ProbePlanner`, signal production to `ProbeExecutor`, and belief revision to `BayesProbeCore.integrate_cycle`. Extract shared projection helpers so the orchestrator and existing controllers use the same answer/projection semantics without duplicate core integration.

**Tech Stack:** Python 3.11+, dataclasses, `enum.StrEnum`, existing Pydantic schemas, existing JSONL ledger, pytest.

## Global Constraints

- The orchestrator must not create `EvidenceEvent`s directly.
- The orchestrator must not update posterior beliefs directly.
- The orchestrator must not spawn, reframe, retire, or evolve hypotheses directly.
- Tool outputs remain raw `ExternalSignal`s until `BayesProbeCore.integrate_cycle`.
- Existing `AutonomousLoopRunner` remains available for signal-stream and benchmark-style tests.
- No real web search, document retrieval, LLM calls, passive signal injection, synchronized orchestration, or benchmark scoring in this slice.
- Do not attempt git commits because this workspace is not currently a git repository.

---

## File Structure

- Create `tests/test_question_runner.py`: end-to-end behavior tests for one-cycle run, multi-cycle candidate carry-forward, stop reasons, ledger records, and no duplicate integration.
- Create `bayesprobe/question_runner.py`: public autonomous question runner dataclasses and orchestration logic.
- Create `bayesprobe/projections.py`: shared projection helpers used by controllers and question runner.
- Modify `bayesprobe/controllers.py`: replace local projection helper implementations with imports from `bayesprobe.projections`.

### Task 1: Question Runner Tests

**Files:**
- Create: `tests/test_question_runner.py`

**Interfaces:**
- Consumes: `AutonomousQuestionRunner`, `AutonomousQuestionRunConfig`, `AutonomousQuestionStopReason`, `InitializeRunInput`, `BayesProbeCore`, `JsonlLedgerStore`
- Produces: failing tests for the end-to-end autonomous question runner before production code exists

- [x] **Step 1: Write failing tests**

Create tests that cover:

- `test_question_runner_executes_one_end_to_end_cycle`
- `test_question_runner_runs_multiple_cycles_with_candidate_pool_from_projection`
- `test_question_runner_stops_on_confidence_threshold`
- `test_question_runner_stops_on_no_probes_before_empty_cycle`
- `test_question_runner_rejects_invalid_config`
- `test_question_runner_writes_end_to_end_ledger_records`
- `test_question_runner_does_not_duplicate_core_integration`

- [x] **Step 2: Verify RED**

Run:

```bash
python3 -m pytest tests/test_question_runner.py -q
```

Expected result: failure because `bayesprobe.question_runner` does not exist yet.

### Task 2: Shared Projection Helpers

**Files:**
- Create: `bayesprobe/projections.py`
- Modify: `bayesprobe/controllers.py`

**Interfaces:**
- Consumes: `BeliefState`, `CycleResult`, `EvidenceEvent`, projection schemas
- Produces:
  - `build_answer_projection(cycle_id, previous_belief_state, cycle_result) -> AnswerProjection`
  - `build_belief_state_projection(cycle_id, previous_belief_state, cycle_result) -> BeliefStateProjection`

- [x] **Step 1: Extract projection helpers**

Move the existing projection semantics from `controllers.py` into `bayesprobe/projections.py`:

- top hypothesis selection
- posterior summary text
- change-my-mind condition
- answer projection
- belief state projection

- [x] **Step 2: Update controllers**

Update `AutonomousController.run_once(...)` to call `build_answer_projection(...)`.

Update `SynchronizedController.process_round(...)` to call `build_belief_state_projection(...)`.

Existing controller tests must continue to pass.

### Task 3: Question Runner Implementation

**Files:**
- Create: `bayesprobe/question_runner.py`

**Interfaces:**
- Consumes: initializer, planner, executor, core, projection helper
- Produces:
  - `AutonomousQuestionRunConfig`
  - `AutonomousQuestionStopReason`
  - `AutonomousQuestionCycleResult`
  - `AutonomousQuestionRunResult`
  - `AutonomousQuestionRunner.run_question(input: InitializeRunInput) -> AutonomousQuestionRunResult`

- [x] **Step 1: Add public dataclasses and enum**

Implement config validation:

- `max_cycles >= 1`
- `max_probes_per_cycle >= 1`
- `confidence_threshold`, when set, in `[0, 1]`
- `posterior_delta_threshold`, when set, non-negative

- [x] **Step 2: Implement one-cycle orchestration**

`run_question(...)` should:

1. initialize run state
2. allocate next cycle ID through core
3. plan a non-empty probe set
4. execute probes into active signals
5. build `CycleRecord(signal_shape=ACTIVE_ONLY)`
6. call `core.integrate_cycle(...)`
7. call shared `build_answer_projection(...)`
8. append the cycle result
9. return `MAX_CYCLES` when `max_cycles=1`

- [x] **Step 3: Implement multi-cycle candidate carry-forward and stop conditions**

After each cycle:

- extend candidate pool with `answer_projection.change_my_mind_condition.structured_probe_candidates`
- update current belief state
- stop on confidence threshold
- stop on posterior stability
- stop on max cycles
- stop before integration with `NO_PROBES` when planner has no valid probes and `stop_on_no_probes=True`

- [x] **Step 4: Implement ledger projection write**

If the core has a ledger, append one `answer_projection` record per completed cycle.

Do not append orchestrator-only records in this slice.

- [x] **Step 5: Verify GREEN**

Run:

```bash
python3 -m pytest tests/test_question_runner.py tests/test_controllers.py -q
```

Expected result: question runner and controller tests pass.

### Task 4: Regression Verification

**Files:**
- Test: all pytest files

**Interfaces:**
- Confirms: question runner integrates with all existing modules without changing core semantics.

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

- Spec coverage: The plan covers end-to-end orchestration, projection reuse, candidate carry-forward, stop reasons, ledger expectations, no-probe handling, and duplicate integration prevention.
- Placeholder scan: No unspecified implementation placeholders remain.
- Type consistency: Public class names and method signatures match the approved design spec.
