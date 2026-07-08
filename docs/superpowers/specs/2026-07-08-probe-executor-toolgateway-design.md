# Probe Executor / ToolGateway MVP Design

## Goal

Build the first deterministic Probe Executor / ToolGateway layer that turns a frozen `ProbeSet` into raw `Active ExternalSignal`s.

This completes the first executable active path:

```text
question -> initializer -> ProbeCandidate pool
         -> planner -> frozen ProbeSet
         -> executor -> Active ExternalSignal
         -> core -> EvidenceEvent / BeliefUpdate / new BeliefState
```

## Design Position

Probe execution is the cycle-local bridge between BayesProbe's active control signal and the external signal stream.

A `ProbeDesign` describes what information should be sought. A `ProbeExecutor` asks a `ProbeToolGateway` to execute those probes and returns raw `ExternalSignal`s. These signals are not evidence yet. They must still pass through the Signal Inbox and Evidence Integration Gate inside `BayesProbeCore`.

The executor must preserve the BayesProbe boundary:

- It may execute selected probes.
- It may create raw active `ExternalSignal`s.
- It may attach provenance linking each signal to the probe that generated it.
- It may write execution/audit records if needed.
- It must not create `EvidenceEvent`s.
- It must not judge reliability, likelihood, or relevance.
- It must not update posteriors.
- It must not evolve hypotheses.
- It must not generate answer projections.

## Non-Goals

This slice will not implement:

- Real web search.
- Real document retrieval.
- Real skill execution.
- LLM-backed tool use.
- Tool retries or timeout policy.
- Informative tool-failure signals.
- Passive signal intake.
- Autonomous runner integration.
- Benchmark scoring.

Those can be added behind the same gateway/executor interface later.

## New Module

Create:

```text
bayesprobe/probe_executor.py
tests/test_probe_executor.py
```

The module should expose:

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

And:

```python
class DeterministicProbeToolGateway:
    def execute_probe(
        self,
        *,
        probe: ProbeDesign,
        context: ProbeExecutionContext,
    ) -> list[ExternalSignal]:
        ...


class ProbeExecutor:
    def __init__(
        self,
        gateway: ProbeToolGateway,
        ledger: JsonlLedgerStore | None = None,
    ) -> None:
        ...

    def execute_probe_set(
        self,
        *,
        probe_set: ProbeSet,
        context: ProbeExecutionContext,
    ) -> ProbeExecutionResult:
        ...
```

## Behavior

### Input Validation

The executor should reject:

- Empty or whitespace-only `context.run_id`.
- Empty or whitespace-only `context.cycle_id`.
- `probe_set.cycle_id != context.cycle_id`.
- Any probe whose `cycle_id != probe_set.cycle_id`.
- Gateway-returned signals whose `signal_kind != SignalKind.ACTIVE`.

Gateway-returned signals must be normalized to the executing cycle:

- `signal.cycle_id == context.cycle_id`
- `signal.generated_by_probe == probe.id`
- `signal.initial_target_hypotheses == probe.target_hypotheses`

The executor should not mutate gateway-returned signal objects in place. It should return copied/normalized signals.

### Empty ProbeSet

If `probe_set.probes == []`, return:

- `signals=[]`
- `executed_probe_ids=[]`

This supports passive-only and explicitly empty synchronized cycles.

### Deterministic Gateway

`DeterministicProbeToolGateway` exists for MVP tests and offline benchmark fixtures.

For each probe, it should return one active `ExternalSignal` by default:

- `id=f"S_{context.cycle_id}_{probe.id}"`
- `cycle_id=context.cycle_id`
- `signal_kind=SignalKind.ACTIVE`
- `source_type="deterministic_probe_gateway"`
- `source=probe.method`
- `generated_by_probe=probe.id`
- `initial_target_hypotheses=probe.target_hypotheses`
- `raw_content` based on `probe.method`, `probe.inquiry_goal`, and target hypotheses

If the probe method contains a deterministic keyword, the raw content can include a corresponding cue for the current deterministic evidence gate:

- `support` or `source_tracing` -> include `SUPPORTS`
- `counterevidence` or `refutation` -> include `REFUTES`
- `anomaly` -> include `ANOMALY`
- otherwise -> include `NEUTRAL`

This is intentionally simple. The real semantics still belong to the Evidence Integration Gate, not the executor.

### Result Ordering

Signals should be returned in probe order. If a gateway returns multiple signals for one probe, preserve gateway order within that probe.

`executed_probe_ids` should include each probe ID once, in execution order.

### Ledger Behavior

If a `JsonlLedgerStore` is provided, append:

1. one `probe_execution` record containing:
   - `run_id`
   - `cycle_id`
   - `probe_set_id`
   - `executed_probe_ids`
   - `signal_ids`
2. one `external_signal` record per returned signal

The executor should not append:

- `evidence_event`
- `belief_update`
- `hypothesis_evolution`
- `belief_state`
- `answer_projection`

### Error Handling

Gateway exceptions should propagate in this slice. Tool failure as an informative active signal is a later design topic because it requires policy decisions about when failure is evidence versus infrastructure noise.

## Integration With Existing Runtime

The expected MVP flow after this slice:

```python
initialization = BayesProbeInitializer().initialize(...)
planning = ProbePlanner().design_probe_set(...)
execution = ProbeExecutor(DeterministicProbeToolGateway()).execute_probe_set(
    probe_set=planning.probe_set,
    context=ProbeExecutionContext(
        run_id=initialization.run.run_id,
        cycle_id=planning.probe_set.cycle_id,
        belief_state=initialization.belief_state,
    ),
)

core_result = BayesProbeCore().integrate_cycle(
    cycle=cycle,
    belief_state=initialization.belief_state,
    probe_set=planning.probe_set,
    signals=execution.signals,
)
```

The executor does not call `BayesProbeCore.integrate_cycle` itself. The controller or runner still owns cycle closure.

## Testing Plan

Add behavior-first tests:

1. `test_executor_turns_probe_set_into_active_signals`
   - Uses a frozen `ProbeSet`.
   - Asserts signals are active, cycle-scoped, generated by probe, and target hypotheses are copied.

2. `test_executor_preserves_probe_and_signal_order`
   - Gateway returns multiple signals for one probe.
   - Asserts order is stable.

3. `test_executor_returns_empty_result_for_empty_probe_set`
   - Empty `ProbeSet`.
   - Asserts no gateway calls and no signals.

4. `test_executor_rejects_probe_set_cycle_mismatch`
   - `probe_set.cycle_id != context.cycle_id`.
   - Raises `ValueError`.

5. `test_executor_rejects_passive_gateway_signals`
   - Custom gateway returns `SignalKind.PASSIVE`.
   - Raises `ValueError`.

6. `test_executor_normalizes_gateway_signals_without_mutating_originals`
   - Gateway returns active signal with placeholder cycle/probe fields.
   - Executor returns normalized copies.
   - Original signal remains unchanged.

7. `test_executor_writes_only_execution_and_signal_records_to_ledger`
   - Uses `JsonlLedgerStore`.
   - Asserts only `probe_execution` and `external_signal` records are written.

8. `test_planned_probe_set_executes_and_integrates_through_core`
   - Full deterministic path:
     - initializer
     - planner
     - executor
     - core integration
   - Asserts `BayesProbeCore` creates evidence events and belief updates from executor signals.

## Acceptance Criteria

- `bayesprobe/probe_executor.py` exposes `ProbeExecutionContext`, `ProbeToolGateway`, `ProbeExecutionResult`, `DeterministicProbeToolGateway`, and `ProbeExecutor`.
- Frozen `ProbeSet`s can be executed into active `ExternalSignal`s.
- Executor preserves probe provenance on each signal.
- Executor supports empty probe sets.
- Executor rejects passive gateway signals.
- Executor normalizes returned signals without mutating originals.
- Executor writes only execution and signal ledger records.
- Existing tests still pass.
- New executor tests pass.

## Known Follow-Ups

After this slice, the next likely layers are:

1. Autonomous end-to-end runner that chains initializer -> planner -> executor -> core.
2. Real ToolGateway adapters for document retrieval, skill execution, and web/search where available.
3. Tool failure policy.
4. Benchmark harness for signal-stream samples.
5. Candidate pool lifecycle updates after core integration.

## Self-Review

- Placeholder scan: No placeholder fields or deferred requirements remain.
- Internal consistency: The executor creates raw signals only; evidence and belief updates remain inside core.
- Scope check: This is one focused implementation slice.
- Ambiguity check: validation, normalization, empty behavior, ledger behavior, and integration boundary are explicit.
