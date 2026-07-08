# Probe Executor / ToolGateway MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic Probe Executor / ToolGateway MVP that turns frozen `ProbeSet`s into raw active `ExternalSignal`s.

**Architecture:** Add `bayesprobe/probe_executor.py` as a focused execution module. It consumes an already-frozen `ProbeSet`, delegates each `ProbeDesign` to a `ProbeToolGateway`, normalizes returned active signals, and returns a `ProbeExecutionResult`. It does not create evidence, update beliefs, evolve hypotheses, generate answers, or close cycles.

**Tech Stack:** Python 3.11+, dataclasses, typing Protocol, existing Pydantic schemas in `bayesprobe.schemas`, existing `JsonlLedgerStore`, pytest.

## Global Constraints

- Executor output is raw `ExternalSignal`, not `EvidenceEvent`.
- Executor must not update posterior beliefs.
- Executor must not evolve hypotheses.
- Executor must not generate answer projections.
- Executor must reject passive gateway signals.
- Executor must preserve probe provenance through `generated_by_probe`.
- Do not implement real web search, real document retrieval, LLM tool use, retries, timeout policy, informative failure signals, autonomous runner integration, or benchmark scoring in this slice.
- Do not attempt git commits because this workspace is not currently a git repository.

---

## File Structure

- Create `tests/test_probe_executor.py`: behavior tests for active signal generation, ordering, empty sets, validation, normalization, ledger boundary, and initializer-planner-executor-core integration.
- Create `bayesprobe/probe_executor.py`: execution context/result dataclasses, gateway protocol, deterministic gateway, executor logic.

### Task 1: Probe Executor Tests

**Files:**
- Create: `tests/test_probe_executor.py`

**Interfaces:**
- Consumes: `ProbeExecutionContext`, `ProbeToolGateway`, `ProbeExecutionResult`, `DeterministicProbeToolGateway`, `ProbeExecutor`, `BayesProbeInitializer`, `ProbePlanner`, `BayesProbeCore`
- Produces: failing tests for executor behavior before production code exists

- [x] **Step 1: Write failing tests**

Create tests that cover:

- `test_executor_turns_probe_set_into_active_signals`
- `test_executor_preserves_probe_and_signal_order`
- `test_executor_returns_empty_result_for_empty_probe_set`
- `test_executor_rejects_probe_set_cycle_mismatch`
- `test_executor_rejects_passive_gateway_signals`
- `test_executor_normalizes_gateway_signals_without_mutating_originals`
- `test_executor_writes_only_execution_and_signal_records_to_ledger`
- `test_planned_probe_set_executes_and_integrates_through_core`

- [x] **Step 2: Verify RED**

Run:

```bash
python3 -m pytest tests/test_probe_executor.py -q
```

Expected result: failure because `bayesprobe.probe_executor` does not exist yet.

### Task 2: Probe Executor Module

**Files:**
- Create: `bayesprobe/probe_executor.py`

**Interfaces:**
- Consumes: `BeliefState`, `ExternalSignal`, `ProbeDesign`, `ProbeSet`, `SignalKind`, `JsonlLedgerStore`
- Produces:
  - `ProbeExecutionContext`
  - `ProbeToolGateway`
  - `ProbeExecutionResult`
  - `DeterministicProbeToolGateway`
  - `ProbeExecutor.execute_probe_set(...) -> ProbeExecutionResult`

- [x] **Step 1: Add public dataclasses and protocol**

Implement:

```python
@dataclass(frozen=True)
class ProbeExecutionContext:
    run_id: str
    cycle_id: str
    belief_state: BeliefState
    metadata: dict[str, Any] = field(default_factory=dict)


class ProbeToolGateway(Protocol):
    def execute_probe(
        self,
        *,
        probe: ProbeDesign,
        context: ProbeExecutionContext,
    ) -> list[ExternalSignal]:
        ...


@dataclass(frozen=True)
class ProbeExecutionResult:
    probe_set: ProbeSet
    signals: list[ExternalSignal]
    executed_probe_ids: list[str]
```

- [x] **Step 2: Implement deterministic gateway**

`DeterministicProbeToolGateway.execute_probe(...)` should return one active signal per probe by default:

- ID: `S_{context.cycle_id}_{probe.id}`
- `signal_kind=SignalKind.ACTIVE`
- `source_type="deterministic_probe_gateway"`
- `source=probe.method`
- `generated_by_probe=probe.id`
- `initial_target_hypotheses=probe.target_hypotheses`
- raw content includes deterministic cue:
  - `SUPPORTS` for `support` or `source_tracing`
  - `REFUTES` for `counterevidence` or `refutation`
  - `ANOMALY` for `anomaly`
  - `NEUTRAL` otherwise

- [x] **Step 3: Implement executor validation and normalization**

`ProbeExecutor.execute_probe_set(...)` should:

1. Validate non-empty `context.run_id` and `context.cycle_id`.
2. Validate `probe_set.cycle_id == context.cycle_id`.
3. Validate every probe is frozen to `probe_set.cycle_id`.
4. Return empty result for empty `ProbeSet` without calling gateway.
5. Execute probes in order.
6. Reject gateway-returned signals unless `signal.signal_kind == SignalKind.ACTIVE`.
7. Normalize returned signal copies to:
   - `cycle_id=context.cycle_id`
   - `generated_by_probe=probe.id`
   - `initial_target_hypotheses=probe.target_hypotheses`
8. Preserve signal ordering.

- [x] **Step 4: Implement ledger writes**

If a ledger is provided, append:

1. one `probe_execution` record with run/cycle/probe_set/executed probe/signal IDs.
2. one `external_signal` record per returned signal.

Do not append evidence, update, evolution, belief state, or projection records.

- [x] **Step 5: Verify GREEN**

Run:

```bash
python3 -m pytest tests/test_probe_executor.py -q
```

Expected result: all probe executor tests pass.

### Task 3: Regression Verification

**Files:**
- Test: all pytest files

**Interfaces:**
- Confirms: executor integrates with initializer, planner, core, schemas, and ledger without changing existing behavior.

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

- Spec coverage: The plan covers active signal generation, ordering, empty sets, validation, normalization, ledger behavior, and initializer-planner-executor-core integration.
- Placeholder scan: No unspecified implementation placeholders remain.
- Type consistency: Public class names and method signatures match the approved design spec.
