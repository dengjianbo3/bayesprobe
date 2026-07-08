# Probe Planner Design

## Goal

Build the first deterministic Probe Planner that turns a `ProbeCandidate` pool and current `BeliefState` into a bounded, cycle-frozen `ProbeSet`.

This is the missing bridge between initialization and active signal collection:

```text
problem -> initial BeliefState + ProbeCandidate pool
        -> ProbePlanner freezes ProbeSet
        -> later Probe Executor turns ProbeSet into Active External Signals
        -> BayesProbe Core integrates signals into belief revision
```

## Design Position

Probe planning is BayesProbe's cycle-local active control step. A `ProbeCandidate` is not yet an action, not a tool call, and not evidence. A `ProbeSet` is the bounded set of hypothesis-conditioned inquiries selected for one cycle.

The planner must preserve the Controller/Core boundary:

- It may rank candidate probes.
- It may select and freeze probes for a cycle.
- It may explain why probes were selected or rejected.
- It may write a `probe_set` ledger record.
- It must not execute probes.
- It must not create `ExternalSignal`s.
- It must not interpret evidence.
- It must not update posterior beliefs.
- It must not evolve hypotheses.

## Non-Goals

This slice will not implement:

- Tool or search execution.
- LLM-based probe scoring.
- External signal generation.
- Evidence integration.
- Belief updates.
- Probe candidate generation beyond what already comes from initialization.
- Cross-cycle candidate pool lifecycle management.
- Benchmark scoring.

## New Module

Create:

```text
bayesprobe/probe_planner.py
tests/test_probe_planner.py
```

The module should expose:

```python
@dataclass(frozen=True)
class ProbePlanningConfig:
    max_probes: int = 2
    allow_empty: bool = False
    attack_top_hypothesis_bonus: float = 1.25
    unresolved_uncertainty_bonus: float = 1.1


@dataclass(frozen=True)
class RejectedProbeCandidate:
    candidate: ProbeCandidate
    reason: str
    score: float


@dataclass(frozen=True)
class ProbePlanningResult:
    probe_set: ProbeSet
    selected_candidates: list[ProbeCandidate]
    rejected_candidates: list[RejectedProbeCandidate]
```

And:

```python
class ProbePlanner:
    def __init__(self, ledger: JsonlLedgerStore | None = None) -> None:
        ...

    def design_probe_set(
        self,
        *,
        run_id: str,
        cycle_id: str,
        belief_state: BeliefState,
        candidates: list[ProbeCandidate],
        config: ProbePlanningConfig | None = None,
    ) -> ProbePlanningResult:
        ...
```

## Behavior

### Input Validation

The planner should reject:

- Empty or whitespace-only `run_id`.
- Empty or whitespace-only `cycle_id`.
- `max_probes < 1`.
- `attack_top_hypothesis_bonus <= 0`.
- `unresolved_uncertainty_bonus <= 0`.
- Candidate probes with no target hypotheses.
- Candidate probes that target no hypothesis currently present in `belief_state`.

When `allow_empty=False` and no valid candidates remain, raise `ValueError`.

When `allow_empty=True` and no valid candidates remain, return an empty `ProbeSet` with `may_be_empty=True`.

### Cycle Freezing

Selected probe designs must be frozen to the requested `cycle_id`:

- `probe.cycle_id == cycle_id`
- `probe.id` should be stable and include the requested cycle when practical.
- `candidate.selected_in_cycle == cycle_id` for selected candidates.

The planner should not mutate input candidates in place. It should return copied candidates/probes so caller-owned pools remain unchanged.

### Ranking

The MVP ranking is deterministic and uses existing `ProbeDesign` fields:

```text
score =
  expected_information_gain
  * decision_relevance
  * attack_top_hypothesis_bonus_if_probe_targets_top
  * unresolved_uncertainty_bonus_if_belief_state_has_uncertainty
  / max(cost_estimate, 0.01)
```

Tie-breakers:

1. Higher score first.
2. Lower cost first.
3. Candidate ID ascending.

This keeps tests stable and makes planner behavior inspectable.

### Top Hypothesis Attack Rule

If any valid candidate targets the current top hypothesis, and budget allows at least one probe, at least one selected candidate should target the top hypothesis.

The top hypothesis is the hypothesis with the highest posterior, with ID as deterministic tie-breaker.

This rule is important because BayesProbe should actively seek ways to challenge its current best belief rather than only gathering confirmatory signals.

### ProbeSet Construction

The returned `ProbeSet` should include:

- `probe_set_id=f"ps_{cycle_id}"`
- `cycle_id=cycle_id`
- selected `ProbeDesign`s frozen to `cycle_id`
- `selection_reason` summarizing ranking and selected candidate IDs
- `budget_allocated` with at least:
  - `max_probes`
  - `selected_count`
  - `candidate_count`
- `may_be_empty` from config and selection result

### Rejections

Rejected candidates should include deterministic reasons such as:

- `invalid_no_targets`
- `invalid_unknown_targets`
- `not_selected_budget_limit`

Invalid candidates should appear in `rejected_candidates` when possible, unless validation raises because the planner cannot produce a valid result and `allow_empty=False`.

### Ledger Behavior

If a `JsonlLedgerStore` is provided, append exactly one `probe_set` record after successful planning.

The planner should not append:

- `external_signal`
- `evidence_event`
- `belief_update`
- `hypothesis_evolution`
- `answer_projection`

## Integration With Existing Runtime

The intended first flow after this slice is:

```python
initialization = BayesProbeInitializer().initialize(...)

planning = ProbePlanner().design_probe_set(
    run_id=initialization.run.run_id,
    cycle_id="run_1_cycle_1",
    belief_state=initialization.belief_state,
    candidates=initialization.probe_candidates,
)

core.integrate_cycle(
    cycle=cycle,
    belief_state=initialization.belief_state,
    probe_set=planning.probe_set,
    signals=active_signals_from_later_executor,
)
```

The planner does not call `BayesProbeCore.integrate_cycle` itself. Controllers or future runners decide when a cycle boundary closes.

## Testing Plan

Add behavior-first tests:

1. `test_planner_selects_top_scoring_candidates_and_freezes_cycle`
   - Uses candidates with different expected information gain, relevance, and cost.
   - Asserts selected probe IDs/cycle IDs are frozen to requested cycle.
   - Asserts input candidates are not mutated.

2. `test_planner_prioritizes_probe_that_attacks_top_hypothesis`
   - Current top hypothesis has a valid candidate.
   - Ensures at least one selected probe targets the top hypothesis.

3. `test_planner_rejects_invalid_candidates`
   - Candidate with no targets.
   - Candidate targeting unknown hypothesis.
   - Asserts deterministic rejection reasons.

4. `test_planner_can_return_empty_probe_set_when_allowed`
   - No valid candidates and `allow_empty=True`.
   - Returns empty `ProbeSet` with `may_be_empty=True`.

5. `test_planner_rejects_empty_selection_when_not_allowed`
   - No valid candidates and `allow_empty=False`.
   - Raises `ValueError`.

6. `test_planner_writes_only_probe_set_to_ledger`
   - Uses `JsonlLedgerStore`.
   - Asserts only `probe_set` is written.

7. `test_initializer_probe_candidates_can_be_planned`
   - Uses `BayesProbeInitializer` output.
   - Passes its `probe_candidates` into `ProbePlanner`.
   - Confirms returned `ProbeSet` can be consumed by the existing `BayesProbeCore.integrate_cycle`.

## Acceptance Criteria

- `bayesprobe/probe_planner.py` exposes `ProbePlanningConfig`, `RejectedProbeCandidate`, `ProbePlanningResult`, and `ProbePlanner`.
- Planner selects a bounded `ProbeSet` from `ProbeCandidate`s.
- Selected probes are frozen to the requested cycle.
- Planner does not mutate input candidate pools.
- Planner can return empty sets only when explicitly allowed.
- Planner records rejected candidates with clear reasons.
- Planner writes only `probe_set` ledger records.
- Existing tests still pass.
- New planner tests pass.

## Known Follow-Ups

After this slice, the next likely layers are:

1. Probe Executor / ToolGateway that turns `ProbeSet` into active `ExternalSignal`s.
2. Runner integration so autonomous cycles can call initializer -> planner -> executor -> core.
3. Candidate pool lifecycle management across cycles.
4. Benchmark harness for signal-stream samples.
5. LLM-backed probe scoring behind the same planner interface.

## Self-Review

- Placeholder scan: No placeholder fields or deferred requirements remain.
- Internal consistency: The planner selects probes only; it does not execute or update beliefs.
- Scope check: This is one focused implementation slice.
- Ambiguity check: Validation, ranking, tie-breakers, empty behavior, ledger behavior, and integration boundaries are explicit.
