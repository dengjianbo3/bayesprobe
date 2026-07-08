# Hypothesis Evolution Engine v0.1 Design

Date: 2026-07-08
Status: Approved from core-depth alignment

## Goal

Deepen BayesProbe's core by extracting hypothesis evolution from `BayesProbeCore` into a focused module that can spawn, reframe, and retire hypotheses from signal-grounded evidence pressure.

This slice strengthens the "evolving hypotheses" part of BayesProbe's definition:

```text
signal-grounded belief revision over evolving hypotheses
```

## Scope

Create `bayesprobe/hypothesis_evolution.py` with:

- `HypothesisEvolutionResult`
- `HypothesisEvolutionConfig`
- `HypothesisEvolutionEngine`

The engine consumes:

- current `CycleRecord`
- previous `BeliefState`
- updated hypotheses after belief solving
- current-cycle `EvidenceEvent`s
- current-cycle `BeliefUpdate`s

The engine returns:

- materialized hypotheses
- `HypothesisEvolution` records
- follow-up `ProbeCandidate`s

## Module Interface

```python
engine.evolve(
    *,
    cycle: CycleRecord,
    previous_belief_state: BeliefState,
    updated_hypotheses: list[Hypothesis],
    evidence_events: list[EvidenceEvent],
    belief_updates: list[BeliefUpdate],
) -> HypothesisEvolutionResult
```

`BayesProbeCore` should call this single interface. It should not contain concrete spawn/reframe/retire rules.

## Evolution Rules

### Spawn

Spawn a new hypothesis when an `ANOMALY` evidence event has low likelihood under all active hypotheses.

MVP implementation:

- Trigger on `EvidenceType.ANOMALY`.
- Create one spawned hypothesis per anomaly event.
- Preserve existing ID compatibility: `H_<event.id>_spawned`.
- Add `EvolutionOperation.SPAWN`.
- Include audit fields:
  - `why_existing_hypotheses_failed`
  - `new_hypothesis_prior`
  - `required_next_probe`
  - `trigger_event_type`

### Reframe

Reframe the current top previous hypothesis when strong or broad counterevidence weakens it while the hypothesis still has scope ambiguity.

MVP implementation:

- Trigger when a hypothesis receives a weakening update from `COUNTEREVIDENCE`.
- Require posterior drop of at least `reframe_drop_threshold`.
- Require previous posterior of at least `reframe_min_previous_posterior`.
- Require the previous hypothesis to have non-empty `scope`.
- Create one reframed hypothesis with ID `H_<hypothesis_id>_<cycle_id>_reframed`.
- Preserve the original hypothesis as weakened; do not delete it.
- Add `EvolutionOperation.REFRAME`.
- Include audit fields:
  - `from_statement`
  - `from_scope`
  - `new_scope`
  - `posterior_drop`
  - `required_next_probe`

### Retire

Retire stale hypotheses only after independent counterevidence.

MVP implementation:

- Trigger when a hypothesis posterior is below `retire_posterior_threshold`.
- Require at least `retire_min_independent_counterevents` counterevidence events that target the hypothesis.
- Count only events with `independence >= independent_event_threshold`.
- Do not count low-independence duplicate evidence toward retirement.
- Materialize the retired status on the existing hypothesis.
- Add `EvolutionOperation.RETIRE`.
- Include audit fields:
  - `retired_posterior`
  - `independent_counterevidence_count`
  - `counterevidence_event_ids`

## Probe Candidates

Every spawn or reframe should produce a follow-up `ProbeCandidate`:

- spawned hypothesis: anomaly boundary probe
- reframed hypothesis: scope-disambiguation probe

Retirement does not need a follow-up probe in v0.1.

## Non-Goals

- No merge, split, reject, or reactivate implementation in this slice.
- No LLM model gateway.
- No probabilistic structural learning.
- No changes to Evidence Integration Gate.
- No changes to Belief Solver likelihood math.
- No changes to benchmark scoring.

## Core Integration

`BayesProbeCore.integrate_cycle` should:

1. integrate evidence
2. solve belief updates
3. call `HypothesisEvolutionEngine.evolve(...)`
4. merge evidence-gate probe candidates with evolution probe candidates
5. write evolutions and probe candidates to ledger

Existing anomaly-spawn behavior must remain compatible with current tests.

## Test Strategy

Add `tests/test_hypothesis_evolution.py` covering:

- anomaly evidence spawns a new hypothesis and probe candidate.
- low-independence duplicate counterevidence does not retire a hypothesis.
- independent counterevidence can retire a stale hypothesis.
- counterevidence against a scoped top hypothesis can produce a reframed hypothesis.

Update core tests only as needed to verify that core delegates evolution and includes evolution probe candidates in result/ledger refs.

Run focused evolution/core tests first, then the full pytest suite.
