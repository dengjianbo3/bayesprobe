# Task 3 Report: Coverage-Aware Solver and Frame-Adequacy Policy

## Status

Complete. Task 3 is implemented on `codex/epistemic-kernel-completion` from
reviewed HEAD `3b0dde4`.

## Files

Created:

- `bayesprobe/kernel_config.py`
- `bayesprobe/frame_policy.py`
- `tests/test_frame_policy.py`
- `.superpowers/sdd/task-3-report.md`

Modified:

- `bayesprobe/belief.py`
- `bayesprobe/core.py`
- `bayesprobe/initialization.py`
- `bayesprobe/schemas.py`
- `tests/test_belief.py`
- `tests/test_core_cycles.py`
- `tests/test_initialization.py`

No Task 2 framing/admission semantics, WebUI files, probe files, provider files,
or Task 5 probe replacement behavior were changed.

## Implementation

- Added frozen, fully validated open-coverage, frame-adequacy, expansion, and
  projection policy configuration. Boolean numerics, non-finite values,
  out-of-range probabilities, invalid reserve ordering, and non-positive or
  non-integer limits fail closed.
- Added `CoverageAwareBeliefSolver.solve(...)` and `BeliefSolveResult` as the
  native v0.2 belief-revision interface. Exclusive-open named hypotheses and
  the private unresolved slot normalize together; exhaustive frames normalize
  only named active choices; independent hypotheses retain separate log-odds
  credences.
- Used `effective_update_weight` exactly when present. `None` falls back to the
  v0.1 quality product and explicit `0.0` remains zero. Missing unresolved
  likelihood is neutral during the Task 3 compatibility window.
- Added one-round distribution rounding, minimum unresolved reserve handling,
  exact retirement-mass return before normalization, and `FrameMassUpdate`
  audit records without creating an unresolved `BeliefUpdate`.
- Added `FrameAdequacyPolicy.assess(...)` and all five status transitions.
  Accepted-event filtering, high-verifiability external evidence, distinct
  derivation roots, model-only limits, all-named disconfirmation, and
  unresolved-mass dominance are deterministic.
- Added optional `EvidenceEvent.epistemic_origin` and
  `EvidenceEvent.derivation_root_id` compatibility fields. Missing provenance
  cannot establish external inadequacy; Task 4 can make these fields mandatory.
- Integrated solving, adequacy, legacy evolution, summaries, next-state
  validation, and ledger persistence in core. Native record order is:
  `external_signal`, `evidence_event`, `belief_update`, `frame_mass_update`,
  `frame_adequacy_decision`, `hypothesis_evolution`, `probe_candidate`,
  `belief_state`, with existing cycle/probe-set records retained.
- Kept `solve_updates` as a v0.1-only deprecated migration wrapper. No native
  production caller remains.
- Initialization now marks exhaustive frames adequate and every open frame
  provisional. Summaries include competition, coverage, named active mass,
  unresolved mass, and frame adequacy without treating independent credences
  as a categorical distribution.

## RED Evidence

1. Initial solver/policy RED:
   `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_frame_policy.py tests/test_belief.py -q -p no:cacheprovider`
   produced `2 errors` during collection because
   `CoverageAwareBeliefSolver` did not exist.
2. Core/initialization RED:
   the exact Task 3 focused command produced `35 failed, 90 passed` because
   core still called the legacy wrapper and initialization lacked the new
   status/summary behavior.
3. Native state invariant RED: `1 failed` because inconsistent named plus
   unresolved mass was accepted.
4. Provenance threshold RED: `1 failed` because external origin/root identity
   was not yet represented on evidence.
5. Retirement precision RED: `1 failed` because unresolved mass was rounded
   before an evidence update.
6. Accepted trigger audit RED: `1 failed` because all-named disconfirmation did
   not retain its triggering event id.

## GREEN Evidence

- Final exact focused command:
  `127 passed in 0.32s`.
- Full offline Python suite:
  `971 passed, 10 skipped in 7.86s`.
- Node WebUI stream regression:
  `15 passed, 0 failed`.
- `git diff --check`: clean.

## Self-Review

- Atomicity/order: the complete next `BeliefState` is schema-revalidated before
  any cycle ledger append. Native frame mass and adequacy records precede
  evolution/candidates and the final state. Legacy migrated frames omit only a
  no-op adequacy record to preserve their reviewed ledger layout; actual legacy
  challenge/expansion transitions are recorded.
- Duplicate evidence: prior-cycle and same-cycle duplicate ids remain discarded,
  produce no second belief or frame-mass movement, and persist one canonical
  evidence record.
- Distribution semantics: exclusive-open named plus unresolved mass is one;
  exhaustive MCQ named mass is one; independent credences do not cross-normalize.
  Retirement returns exact pre-rounding mass to unresolved.
- Compatibility: `solve_updates` accepts only v0.1 state, migrates explicitly,
  and delegates. `None` weight uses the legacy quality product; explicit zero
  is not replaced by fallback weight.
- Scope: generic initialization probes were not replaced, and admission,
  framing, provider, probe, and WebUI behavior was not changed.

## Concerns

No blocking concerns. Until Task 4 populates provenance on native evidence,
events without explicit origin/root metadata can challenge and request
expansion but cannot establish externally verified frame inadequacy. Legacy
hypothesis evolution remains bypassed for exclusive-open frames because it
normalizes named hypotheses without the private unresolved slot; bounded open
frame expansion belongs to the later expansion task.
