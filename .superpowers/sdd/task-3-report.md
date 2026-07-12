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

## Review Fix 1

Reviewed base: `0c60121f37a453ebd7a003ba6b1cb9320d0a8b08`.
This section supersedes the earlier exclusive-open retirement concern.

### Changes

- Exclusive-open public cycles now run the existing evolution engine's
  retirement rule against accepted current-cycle evidence. The open path does
  not run anomaly spawning, reframing, or candidate creation.
- Added coverage-aware retirement reconciliation after evolution determines a
  retirement. The retired hypothesis retains its solver posterior and retired
  audit state, leaves `active_hypothesis_ids`, and transfers that posterior to
  unresolved mass in the same returned `BeliefState` without named-only
  renormalization.
- Every retirement transfer creates a stable per-hypothesis `FrameMassUpdate`
  with sequential prior/posterior unresolved values and the final accepted
  retirement-triggering evidence id. Available derivation-root context is
  retained in the reason. A transfer without evolution/evidence audit context
  fails closed.
- Retirement frame-mass records remain after event belief/frame updates and
  before frame adequacy, hypothesis evolution, candidates, and final state.
  The public-cycle regression asserts the complete relevant ledger order.
- Distribution rounding now applies configured minimums on the same decimal
  grid as the final distribution and assigns the residual only to an eligible
  slot. A custom reserve of `0.05001` finishes at or above reserve while named
  plus unresolved mass remains exactly one.
- Inadequacy decisions now report only the unresolved-support events that
  satisfy the qualifying high-verifiability or distinct-root rule. Mixed
  qualifying/non-qualifying coverage is asserted precisely.

### RED Evidence

`PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_frame_policy.py tests/test_belief.py tests/test_core_cycles.py tests/test_initialization.py -q -p no:cacheprovider`
produced `4 failed, 126 passed in 0.36s`. The failures were the unaudited
synthetic retirement transfer, non-grid reserve rounding down to `0.05`, mixed
trigger ids including a non-qualifying event, and production open-cycle
retirement remaining unreachable.

### GREEN Evidence

- Exact Task 3 focused suite: `130 passed in 0.32s`.
- Full offline Python suite: `974 passed, 10 skipped in 7.82s`.
- Node stream regression: `15 passed, 0 failed`.
- `git diff --check`: clean.

### Concerns

No blocking concerns. Open-cycle evolution intentionally reaches only the
existing retirement rule in Task 3; semantic expansion, reframing, anomaly
spawning, and new probe candidates remain deferred to Task 6.

## Review Fix 2

Reviewed base: `b6165fd40866a93827ce3e8af3c961cf7e80b44c`.

### Changes

- A native exclusive-open `FrameState` may now have zero active hypothesis ids
  only when unresolved alternative mass is fully conserved at one. Initial
  `HypothesisFrame` construction still requires one to six hypotheses, and all
  other empty-active-id combinations fail validation.
- `FrameAdequacyPolicy` now treats a fully unresolved frame with no active named
  hypotheses as challenged, requests expansion, and records the accepted
  disconfirming retirement triggers. Already-qualified inadequate or expanding
  rules retain precedence.
- Added public
  `HypothesisEvolutionEngine.retire_stale_hypotheses(...) -> HypothesisEvolutionResult`.
  It returns only updated hypotheses and `RETIRE` audits, always has no probe
  candidates, and performs no spawning, reframing, splitting, or candidate
  creation. Normal evolution reuses the same method, and core no longer calls a
  private retirement helper.
- Retirement eligibility is limited to active and weakened hypotheses. Reframed,
  split, retired, and archived terminal states are ignored, preventing duplicate
  retirement evolutions on later evidence.
- Added a public core-cycle regression that retires every active named
  hypothesis, transfers each posterior into unresolved mass with one audit
  update per hypothesis, returns a schema-valid fully unresolved state, and
  preserves frame-mass, adequacy, evolution, and final-state ledger ordering.
- Added a two-cycle public regression proving fresh counterevidence targeted at
  an already retired hypothesis creates no second retirement evolution, no
  retirement `FrameMassUpdate`, and no additional unresolved transfer.

### RED Evidence

1. Schema lifecycle tests produced `4 failed, 1 passed`: empty active ids were
   rejected unconditionally.
2. Adequacy tests produced `1 failed, 1 passed`: the all-retired frame remained
   provisional while the stronger external inadequacy rule already passed.
3. Retirement-only engine tests produced `5 failed`: the public method did not
   exist.
4. Public core regressions produced `2 failed`: core called the private helper,
   and a second cycle emitted a duplicate `RETIRE` evolution.

### GREEN Evidence

- Exact Task 3 focused suite plus hypothesis-evolution/schema coverage:
  `261 passed in 0.43s`.
- Full offline Python suite: `988 passed, 10 skipped in 7.89s`.
- Node stream regression: `15 passed, 0 failed`.
- `git diff --check`: clean.
- Production private-helper scan: no `_retire_stale_hypotheses` references.

### Concerns

No blocking concerns. This fix exposes and integrates retirement only. Semantic
expansion, anomaly spawning, reframing, and candidate creation remain deferred
to Task 6, and no Task 4/5, provider, probe, or WebUI behavior was added.
