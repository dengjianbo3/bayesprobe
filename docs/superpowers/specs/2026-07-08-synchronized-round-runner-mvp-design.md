# Synchronized Round Runner MVP Design

Date: 2026-07-08
Status: Proposed design approved in conversation; written spec awaiting review

## Goal

Build a first-class synchronized round runner for BayesProbe fixed-round collaboration. The runner should support human, multi-agent, benchmark, and system-log passive signals while preserving the BayesProbe rule that all raw external information enters as `ExternalSignal` and becomes evidence only through `BayesProbeCore.integrate_cycle`.

This closes the current gap between the low-level `SynchronizedController.process_round(...)` and the benchmark harness's ad hoc mixed-cycle path.

## Scope

The MVP supports one-round execution and bounded N-round execution over an existing or newly initialized BayesProbe run.

Supported synchronized round shapes:

- `passive_only`: process passive signals from humans, agents, logs, benchmarks, or user corrections.
- `active_only`: execute a BayesProbe-selected active probe inside a collaboration round.
- `active_plus_passive`: execute active probes and integrate passive signals under the same cycle boundary.

Each completed round emits a `BeliefStateProjection`.

## Non-Goals

- No real-time meeting scheduler.
- No async protocol or networked multi-agent transport.
- No direct sharing of full internal `BeliefState` as the collaboration protocol.
- No projection decomposition implementation beyond treating external projections as passive signals.
- No LLM calls, web search, document retrieval, or human-in-the-loop UI.
- No benchmark scoring changes in this slice.

## Public API

Create `bayesprobe/synchronized_runner.py` with:

- `SynchronizedRoundShape`
- `SynchronizedRoundInput`
- `SynchronizedRunInput`
- `SynchronizedRoundResult`
- `SynchronizedRunResult`
- `SynchronizedRoundRunner`

### `SynchronizedRoundShape`

Enum values:

- `PASSIVE_ONLY = "passive_only"`
- `ACTIVE_ONLY = "active_only"`
- `ACTIVE_PLUS_PASSIVE = "active_plus_passive"`

### `SynchronizedRoundInput`

Fields:

- `round_id: str`
- `shape: SynchronizedRoundShape`
- `passive_signals: list[ExternalSignal] = []`
- `max_probes: int = 1`
- `metadata: dict[str, Any] = {}`

Validation:

- `round_id` must be non-empty.
- `max_probes >= 1`.
- `passive_only` requires at least one passive signal.
- `active_only` rejects passive signals.
- `active_plus_passive` requires at least one passive signal.
- Every passive signal must have `signal_kind=SignalKind.PASSIVE`.

### `SynchronizedRunInput`

Fields:

- `initialize_input: InitializeRunInput | None`
- `run: RunRecord | None`
- `belief_state: BeliefState | None`
- `probe_candidates: list[ProbeCandidate] = []`
- `rounds: list[SynchronizedRoundInput]`

Validation:

- Either `initialize_input` must be provided, or both `run` and `belief_state` must be provided.
- `rounds` must be non-empty.
- If an existing `run` and `belief_state` are supplied, their `run_id`s must match.

### `SynchronizedRoundResult`

Fields:

- `round_id`
- `cycle`
- `shape`
- `probe_set`
- `signals`
- `active_signal_count`
- `passive_signal_count`
- `belief_state`
- `evidence_events`
- `belief_updates`
- `hypothesis_evolutions`
- `belief_state_projection`
- `selected_probe_candidates`
- `remaining_probe_candidates`

### `SynchronizedRunResult`

Fields:

- `run`
- `initial_belief_state`
- `final_belief_state`
- `round_results`
- `final_belief_state_projection`
- `remaining_probe_candidates`

## Data Flow

### New Run

```text
SynchronizedRunInput.initialize_input
→ BayesProbeInitializer.initialize
→ initial RunRecord + BeliefState + ProbeCandidate pool
→ run synchronized rounds
```

### Existing Run

```text
SynchronizedRunInput.run + SynchronizedRunInput.belief_state
→ caller-provided candidate pool
→ run synchronized rounds
```

## Round Execution

### Passive-Only Round

```text
current BeliefState
→ passive ExternalSignals
→ CycleRecord(signal_shape=PASSIVE_ONLY)
→ empty ProbeSet(may_be_empty=True)
→ BayesProbeCore.integrate_cycle
→ build_belief_state_projection
→ SynchronizedRoundResult
```

This path may reuse `SynchronizedController.process_round(...)` if doing so keeps ledger behavior and result shape clear.

### Active-Only Round

```text
current BeliefState + ProbeCandidate pool
→ ProbePlanner.design_probe_set
→ ProbeExecutor.execute_probe_set
→ active ExternalSignals
→ CycleRecord(signal_shape=ACTIVE_ONLY)
→ BayesProbeCore.integrate_cycle
→ build_belief_state_projection
→ SynchronizedRoundResult
```

### Active-Plus-Passive Round

```text
current BeliefState + ProbeCandidate pool
→ ProbePlanner.design_probe_set
→ ProbeExecutor.execute_probe_set
→ active ExternalSignals + passive ExternalSignals
→ CycleRecord(signal_shape=ACTIVE_PLUS_PASSIVE)
→ BayesProbeCore.integrate_cycle
→ build_belief_state_projection
→ SynchronizedRoundResult
```

## Candidate Pool Policy

The runner starts with initializer-provided candidates for a new run, or caller-provided candidates for an existing run.

After each completed round:

1. Remove selected candidate IDs from the pool.
2. Add `belief_state_projection.change_my_mind_condition.structured_probe_candidates` to the front of the pool.
3. Carry remaining candidates after projection-derived candidates.

This mirrors the autonomous question runner while keeping synchronized collaboration focused on the latest change-my-mind condition.

## Ledger Policy

The runner delegates core records to existing modules:

- initializer writes `run`, initial `belief_state`, and initial `probe_candidate` records.
- planner writes `probe_set` when ledger is present.
- executor writes `probe_execution` and active `external_signal` records.
- core writes `cycle`, normalized `external_signal`, `probe_set`, `evidence_event`, `belief_update`, `hypothesis_evolution`, and updated `belief_state`.

The synchronized runner appends:

- one `belief_state_projection` per completed round.

It should not append custom runner records in this slice.

## Invariants

- The runner must not create `EvidenceEvent`s.
- The runner must not update posterior beliefs.
- The runner must not spawn, split, reframe, reject, retire, or reactivate hypotheses directly.
- Passive signals remain passive; the runner must not wrap them in fake active probes.
- External projections from other agents are passive `ExternalSignal`s, not evidence.
- All round shapes use the same `BayesProbeCore.integrate_cycle` path.

## Error Handling

- Invalid round shape configuration raises `ValueError` before core integration.
- Empty active candidate pool raises `ValueError` for active-only and active-plus-passive rounds unless a later explicit no-probe synchronized mode is added.
- Passive signals with non-passive `signal_kind` raise `ValueError`.
- Existing-run input with mismatched `run_id` raises `ValueError`.

## Tests

Add `tests/test_synchronized_runner.py` covering:

- initializes a new synchronized run and processes one passive-only round.
- processes active-only round with planner and executor.
- processes active-plus-passive round with both signal counts and mixed cycle shape.
- processes multiple rounds while carrying projection-derived candidate pool forward.
- accepts existing run and belief state input.
- rejects invalid round configuration.
- writes one `belief_state_projection` ledger record per completed round and does not duplicate core integration records.

Run focused tests first, then full pytest.

## Fit With Current Roadmap

This implements the missing orchestration layer for Milestone 3 and part of Milestone 4 from `docs/BayesProbe_02_engineering_v0.2_outline.md`:

- fixed-round synchronized loop.
- passive-only cycle.
- active-plus-passive integration.
- projection output for the next collaboration exchange.

It deliberately leaves Projection Decomposition Rule and real multi-agent transport for later slices.
