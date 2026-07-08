# Autonomous Self-Loop Runner Design

## Goal

Build the first autonomous self-loop layer on top of the current BayesProbe MVP core. The runner should turn the existing one-cycle `AutonomousController.run_once(...)` into a bounded multi-cycle process that repeatedly collects signals, integrates evidence, updates the belief state, emits answer projections, and stops for an explicit reason.

This is the next tracer bullet toward a usable BayesProbe agent. It is still deterministic and fixture-friendly: no LLM, no network, no real tool execution, and no natural-language question parser in this slice.

## Current Baseline

The current MVP already provides:

- `BayesProbeCore`: the only belief-revision entry point.
- `EvidenceIntegrationGate`: converts normalized `ExternalSignal`s into `EvidenceEvent`s.
- `solve_updates`: deterministic posterior updates.
- `HypothesisEvolutionPolicy`: materializes anomaly-spawned hypotheses.
- `AutonomousController`: executes one active-only cycle and returns an `AnswerProjection`.
- `SynchronizedController`: executes one passive-only round and returns a `BeliefStateProjection`.
- `JsonlLedgerStore`: append-only JSONL ledger, now wired through core and controller projections.

The missing layer is not more theory; it is run orchestration.

## Non-Goals

This slice will not implement:

- Natural-language question to hypotheses conversion.
- Probe planning or real tool execution.
- LLM-backed evidence interpretation.
- Synchronized multi-agent meeting orchestration.
- Benchmark scoring metrics.

Those remain later layers. This slice only creates the autonomous loop interface they can plug into.

## Module Shape

Create a new deep module:

```text
bayesprobe/runners.py
tests/test_autonomous_runner.py
```

The external interface should stay small:

```python
class AutonomousLoopRunner:
    def run(
        self,
        run_id: str,
        initial_belief_state: BeliefState,
        signal_provider: AutonomousSignalProvider,
    ) -> AutonomousRunResult:
        ...
```

The runner owns timing and continuation. It does not own evidence rules, likelihood judgment, posterior math, or hypothesis evolution. Those remain inside `BayesProbeCore`.

## Signal Provider Interface

Introduce a protocol-like interface:

```python
class AutonomousSignalProvider(Protocol):
    def collect_signals(
        self,
        *,
        run_id: str,
        cycle_index: int,
        belief_state: BeliefState,
        previous_answer: AnswerProjection | None,
    ) -> list[ExternalSignal]:
        ...
```

The provider is the seam for future search/tool/skill/benchmark adapters.

For this MVP, tests can use simple in-memory providers:

- A provider that returns one list of signals per cycle.
- A provider that returns no signals to trigger no-signal stop.
- A provider that returns anomaly signals to test hypothesis evolution in a loop.

## Configuration

Add a small config model or dataclass:

```python
@dataclass(frozen=True)
class AutonomousLoopConfig:
    max_cycles: int = 3
    stop_on_no_signals: bool = True
    confidence_threshold: float | None = None
    posterior_delta_threshold: float | None = None
```

Validation rules:

- `max_cycles` must be at least 1.
- `confidence_threshold`, when set, must be between 0 and 1.
- `posterior_delta_threshold`, when set, must be non-negative.

## Stop Reasons

Add an enum:

```python
class AutonomousStopReason(StrEnum):
    MAX_CYCLES = "max_cycles"
    NO_SIGNALS = "no_signals"
    CONFIDENCE_REACHED = "confidence_reached"
    POSTERIOR_STABLE = "posterior_stable"
```

Stop behavior:

- Stop with `NO_SIGNALS` when the provider returns `[]` and `stop_on_no_signals=True`.
- Stop with `CONFIDENCE_REACHED` when the top hypothesis posterior is greater than or equal to `confidence_threshold`.
- Stop with `POSTERIOR_STABLE` when all continuing hypotheses move by no more than `posterior_delta_threshold` between adjacent cycles.
- Stop with `MAX_CYCLES` after completing `max_cycles` cycles when no earlier stop applies.

`NO_SIGNALS` should stop before creating an empty cycle. This keeps the ledger clean: no cycle should be written if no cycle was actually run.

## Run Result

Add a result dataclass:

```python
@dataclass(frozen=True)
class AutonomousRunResult:
    run_id: str
    initial_belief_state: BeliefState
    final_belief_state: BeliefState
    cycle_results: list[ControllerResult]
    final_answer_projection: AnswerProjection | None
    stop_reason: AutonomousStopReason
```

If zero cycles run because the first provider call returns no signals, `cycle_results` should be empty and `final_answer_projection` should be `None`.

## Loop Algorithm

The runner should:

1. Start from `initial_belief_state`.
2. Ask `signal_provider.collect_signals(...)` for the next active signals.
3. If no signals and `stop_on_no_signals=True`, return `NO_SIGNALS`.
4. Call `AutonomousController.run_once(...)`.
5. Append the returned `ControllerResult`.
6. Use the returned `belief_state` as the next cycle's input.
7. Check confidence and posterior stability stop conditions.
8. Continue until a stop condition fires.

The runner should create one `AutonomousController` instance and reuse the same `BayesProbeCore`, preserving core-level cycle ID allocation and ledger wiring.

## Ledger Behavior

The runner itself does not need to append a new run-level record in this slice. Core and controller already append:

- cycle
- external_signal
- probe_set
- evidence_event
- belief_update
- hypothesis_evolution
- belief_state
- answer_projection

Runner tests should assert that multi-cycle runs write records for each executed cycle when the underlying core has a ledger.

## Error Handling

The runner should raise `ValueError` for invalid config values at construction time.

Errors from `BayesProbeCore` or `AutonomousController` should propagate. The runner should not swallow boundary violations such as wrong signal kind, cross-run belief state, or future-cycle belief state.

## Testing Plan

Tests should be behavior-first and use deterministic fixtures.

Required tests:

1. `test_runner_stops_after_max_cycles`
   - Provider returns active signals for more cycles than allowed.
   - Result has exactly `max_cycles` cycle results.
   - Stop reason is `MAX_CYCLES`.

2. `test_runner_stops_before_cycle_when_no_signals`
   - Provider returns `[]` on the first call.
   - Result has no cycle results.
   - Final belief state equals initial belief state.
   - Stop reason is `NO_SIGNALS`.

3. `test_runner_feeds_updated_belief_state_into_next_cycle`
   - Provider returns two rounds of signals.
   - Second provider call sees the first cycle's updated belief state.
   - Cycle IDs are unique and sequential through core allocation.

4. `test_runner_stops_when_confidence_threshold_reached`
   - Provider returns a strong enough deterministic signal.
   - Top hypothesis posterior crosses threshold.
   - Stop reason is `CONFIDENCE_REACHED`.

5. `test_runner_materializes_anomaly_spawned_hypothesis_across_cycles`
   - First cycle receives an anomaly signal.
   - Returned final belief state contains the spawned hypothesis.
   - A later provider call sees that spawned hypothesis in the belief state.

6. `test_runner_writes_ledger_records_for_each_executed_cycle`
   - Use `JsonlLedgerStore(tmp_path / "ledger.jsonl")`.
   - Assert cycle and answer projection records exist for each executed cycle.

7. `test_invalid_runner_config_is_rejected`
   - `max_cycles=0` raises `ValueError`.
   - invalid thresholds raise `ValueError`.

## Acceptance Criteria

The slice is complete when:

- `bayesprobe/runners.py` exposes `AutonomousLoopRunner`, `AutonomousLoopConfig`, `AutonomousStopReason`, `AutonomousRunResult`, and `AutonomousSignalProvider`.
- Existing tests still pass.
- New runner tests pass.
- No production code outside runner/controller/core contracts duplicates evidence, posterior, or evolution logic.
- The runner can execute more than one autonomous cycle using the current deterministic signal fixtures.
- No git commit is attempted unless the workspace is actually a git repository.

## Known Follow-Ups

After this runner exists, the next likely slices are:

1. Question-to-belief initialization.
2. Probe planner and probe executor adapters.
3. Benchmark dataset runner and scoring metrics.
4. Synchronized multi-agent round runner.
