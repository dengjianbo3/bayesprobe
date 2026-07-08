# End-to-End Autonomous Orchestrator Design

## Goal

Build the first user-facing autonomous orchestration layer that can run a BayesProbe question end to end:

```text
problem
-> initialize RunRecord + initial BeliefState + ProbeCandidate pool
-> plan bounded ProbeSet
-> execute probes into Active ExternalSignals
-> integrate signals through BayesProbe Core
-> emit AnswerProjection
-> stop with explicit reason
```

This turns the current collection of working modules into a callable MVP agent path.

## Design Position

The orchestrator is a run-level coordinator. It should make BayesProbe usable from a single problem input while preserving the core paradigm boundaries.

It may:

- Call `BayesProbeInitializer`.
- Call `ProbePlanner`.
- Call `ProbeExecutor`.
- Call `BayesProbeCore.integrate_cycle`.
- Generate an autonomous answer projection by reusing controller projection semantics.
- Apply run-level stop conditions.
- Return a structured run result.

It must not:

- Construct `EvidenceEvent`s directly.
- Score reliability, relevance, or likelihood.
- Update posterior beliefs directly.
- Spawn/reframe/retire hypotheses directly.
- Treat tool outputs as evidence before core integration.
- Convert BayesProbe into a ReAct-style action/observation loop.

## Why Not Modify Existing `AutonomousLoopRunner`

`AutonomousLoopRunner` currently works over a signal-provider seam:

```python
collect_signals(...) -> list[ExternalSignal]
```

That module remains useful for fixture-driven and benchmark-driven signal-stream tests. The new orchestrator should not replace it. Instead, it should add a higher-level path where active signals are produced by BayesProbe's own planner/executor chain.

This keeps two useful entry points:

- `AutonomousLoopRunner`: caller provides signals per cycle.
- `AutonomousQuestionRunner`: caller provides a problem; BayesProbe initializes, plans, executes, integrates, and projects.

## Non-Goals

This slice will not implement:

- Real web search or document retrieval.
- LLM-backed initialization, planning, evidence, or projection.
- Passive signal arrival during active execution.
- Synchronized multi-agent orchestration.
- Benchmark scoring.
- Candidate pool lifecycle updates beyond initial candidates and projection-derived candidates.
- Tool failure as evidence.

## New Module

Create:

```text
bayesprobe/question_runner.py
tests/test_question_runner.py
```

The module should expose:

```python
@dataclass(frozen=True)
class AutonomousQuestionRunConfig:
    max_cycles: int = 3
    stop_on_no_probes: bool = True
    confidence_threshold: float | None = None
    posterior_delta_threshold: float | None = None
    max_probes_per_cycle: int = 2


@dataclass(frozen=True)
class AutonomousQuestionCycleResult:
    cycle: CycleRecord
    probe_set: ProbeSet
    signals: list[ExternalSignal]
    evidence_events: list[EvidenceEvent]
    belief_updates: list[BeliefUpdate]
    hypothesis_evolutions: list[HypothesisEvolution]
    belief_state: BeliefState
    answer_projection: AnswerProjection


class AutonomousQuestionStopReason(StrEnum):
    MAX_CYCLES = "max_cycles"
    NO_PROBES = "no_probes"
    CONFIDENCE_REACHED = "confidence_reached"
    POSTERIOR_STABLE = "posterior_stable"


@dataclass(frozen=True)
class AutonomousQuestionRunResult:
    run: RunRecord
    initial_belief_state: BeliefState
    final_belief_state: BeliefState
    cycle_results: list[AutonomousQuestionCycleResult]
    final_answer_projection: AnswerProjection | None
    stop_reason: AutonomousQuestionStopReason
```

And:

```python
class AutonomousQuestionRunner:
    def __init__(
        self,
        *,
        core: BayesProbeCore,
        initializer: BayesProbeInitializer | None = None,
        planner: ProbePlanner | None = None,
        executor: ProbeExecutor | None = None,
        config: AutonomousQuestionRunConfig | None = None,
    ) -> None:
        ...

    def run_question(self, input: InitializeRunInput) -> AutonomousQuestionRunResult:
        ...
```

## Data Flow

### Start

1. Call initializer with the supplied `InitializeRunInput`.
2. Keep the returned `RunRecord`, initial `BeliefState`, and initial `ProbeCandidate` pool.
3. Use the initial belief state as `current_belief_state`.

### Each Cycle

For cycle `n`:

1. Allocate a cycle ID through `BayesProbeCore.allocate_cycle_id(...)`.
2. Call `ProbePlanner.design_probe_set(...)`.
3. If planner cannot select probes and `stop_on_no_probes=True`, stop with `NO_PROBES`.
4. Call `ProbeExecutor.execute_probe_set(...)`.
5. Build a `CycleRecord` with `signal_shape=CycleSignalShape.ACTIVE_ONLY`.
6. Call `BayesProbeCore.integrate_cycle(...)` with:
   - current belief state
   - planned `ProbeSet`
   - executor-produced active signals
7. Generate an `AnswerProjection` from the integrated cycle result.
8. Append cycle result.
9. Carry the returned `BeliefState` into the next cycle.
10. Add structured probe candidates from the answer projection's change-my-mind condition to the candidate pool for later cycles.
11. Check stop conditions.

## Projection Generation

The orchestrator should not invent a separate answer format. It should reuse the same projection semantics as `AutonomousController`:

- current best hypothesis
- posterior summary
- main uncertainty
- weakest assumption
- main evidence events
- change-my-mind condition
- answer utility notes

For the first slice, it may call a small shared projection helper if one is extracted from `controllers.py`, or it may use `AutonomousController` only if that does not force a second core integration. It must not call `AutonomousController.run_once(...)` after already integrating the cycle, because that would duplicate belief updates.

Preferred implementation: extract projection helper functions from `controllers.py` only if needed and keep behavior covered by existing controller tests.

## Stop Conditions

Use deterministic stop conditions aligned with `AutonomousLoopRunner`:

- `MAX_CYCLES`: after `max_cycles` completed cycles.
- `NO_PROBES`: planner cannot produce a non-empty probe set and `stop_on_no_probes=True`.
- `CONFIDENCE_REACHED`: top hypothesis posterior reaches `confidence_threshold`.
- `POSTERIOR_STABLE`: all continuing hypotheses move by no more than `posterior_delta_threshold`.

Stop condition order after a completed cycle:

1. `CONFIDENCE_REACHED`
2. `POSTERIOR_STABLE`
3. `MAX_CYCLES`

`NO_PROBES` happens before executing an empty active cycle.

## Ledger Behavior

The orchestrator itself does not need a new ledger record in this slice.

Existing modules should write their own records when they have a ledger:

- initializer: `run`, `belief_state`, `probe_candidate`
- planner: `probe_set`
- executor: `probe_execution`, `external_signal`
- core: `cycle`, `external_signal`, `probe_set`, `evidence_event`, `belief_update`, `hypothesis_evolution`, `belief_state`
- projection generation: `answer_projection` if a ledger is available through the orchestrator path

The implementation should avoid duplicate core integration. Duplicate ledger records are acceptable only if they reflect separate modules writing their own audit entries; tests should verify at least one coherent end-to-end ledger sequence.

## Error Handling

- Invalid config values raise `ValueError` at construction.
- Initialization, planner, executor, and core errors propagate.
- No-probe stop is a normal stop only when `stop_on_no_probes=True`.
- If `stop_on_no_probes=False`, planner errors should propagate rather than silently creating fake signals.

## Testing Plan

Add behavior-first tests:

1. `test_question_runner_executes_one_end_to_end_cycle`
   - Uses deterministic initializer/planner/executor/core.
   - Asserts one cycle produces probes, active signals, evidence events, belief updates, and answer projection.

2. `test_question_runner_runs_multiple_cycles_with_candidate_pool_from_projection`
   - `max_cycles=2`.
   - Ensures cycle two has a probe set derived from prior projection candidates.

3. `test_question_runner_stops_on_confidence_threshold`
   - Uses deterministic supportive signals.
   - Stops with `CONFIDENCE_REACHED`.

4. `test_question_runner_stops_on_no_probes_before_empty_cycle`
   - Planner returns or allows empty set.
   - No core cycle is integrated.
   - Stop reason is `NO_PROBES`.

5. `test_question_runner_rejects_invalid_config`
   - `max_cycles=0`
   - `max_probes_per_cycle=0`
   - invalid thresholds

6. `test_question_runner_writes_end_to_end_ledger_records`
   - Uses a shared `JsonlLedgerStore` across initializer/planner/executor/core.
   - Asserts records include run, probe_set, probe_execution, external_signal, evidence_event, belief_update, belief_state, answer_projection.

7. `test_question_runner_does_not_duplicate_core_integration`
   - One cycle should produce exactly one `cycle` record from core and one final `AnswerProjection`.

## Acceptance Criteria

- `bayesprobe/question_runner.py` exposes `AutonomousQuestionRunConfig`, `AutonomousQuestionStopReason`, `AutonomousQuestionCycleResult`, `AutonomousQuestionRunResult`, and `AutonomousQuestionRunner`.
- A caller can run one deterministic autonomous question from `InitializeRunInput` to `AnswerProjection`.
- The runner uses initializer, planner, executor, and core rather than bypassing them.
- The runner does not create evidence or update beliefs directly.
- Multi-cycle runs can carry forward belief state and projection-derived probe candidates.
- Stop reasons are explicit.
- Existing tests still pass.
- New question runner tests pass.

## Known Follow-Ups

After this slice, the next likely layers are:

1. Real ToolGateway adapters.
2. Passive signal injection during autonomous cycles.
3. Synchronized end-to-end orchestrator.
4. Benchmark harness using deterministic datasets.
5. Candidate pool lifecycle management beyond projection-derived candidates.
6. LLM-backed frame building, probe scoring, evidence judging, and projection behind existing interfaces.

## Self-Review

- Placeholder scan: No placeholder fields or deferred requirements remain.
- Internal consistency: The orchestrator coordinates modules but does not own evidence or posterior logic.
- Scope check: This is one focused implementation slice for autonomous end-to-end use.
- Ambiguity check: module boundary, cycle flow, stop conditions, ledger expectations, and projection generation are explicit.
