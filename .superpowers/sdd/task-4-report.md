# Task 4 Report: Bind Completed Actions to Signals and Environment Lineage

## Status

Task 4 is implemented in the six owned terminal-bench files. Every completed
gateway action is registered against one frozen Probe plan and one environment
lineage, converted to exactly one `harbor-observation:v3` Signal, and bound back
to its causal action record.

The focused and non-validator nested suites pass. The final full nested suite
has nine expected failures in frozen, out-of-scope validators: eight validate
the removed v2 Signal contract and one still constructs the removed Task 3
`TerminalProbePlan.actions` field.

Commit subject: `feat(terminal-bench): bind terminal signals to causal actions`

## Files

Created:

- `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/causal.py`
- `benchmarks/terminal_bench/tests/test_causal.py`

Modified:

- `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/gateway.py`
- `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/signals.py`
- `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/artifacts.py`
- `benchmarks/terminal_bench/tests/test_gateway.py`

Required report replaced at:

- `.superpowers/sdd/task-4-report.md`

No `actions.py`, `planning.py`, `bayesprobe/`, `environment.py`,
`test_conformance.py`, smoke validator, plan/spec, historical fixture,
generated lock, unrelated test, or pre-existing untracked `reports/` content
was modified.

## Implementation

### Causal registry

- Added strict, frozen `RegisteredPlan` and `CausalActionRecord` models.
- Added canonical compact sorted-JSON serialization with `allow_nan=False` and
  SHA-256 identities.
- `plan_id` hashes run, cycle, full Probe, and full frozen terminal plan.
- `policy_attempt_id` hashes run, cycle, full Probe, and intervention plan.
- `request_fingerprint` hashes the exact bounded executed-request
  representation, including content/patch hashes instead of large bodies.
- `action_id` hashes plan ID, step index, reserved action index, and request
  fingerprint.
- Rejects duplicate plan, policy-attempt, action, step, and Signal identities.
- Rejects observations whose executed request differs from the registered
  step, blank environment states, non-linear state transitions within or
  across plans in one run, and a second mutation in one plan.
- Tracks cumulative intervention generation across plans in one run.
- Uses the post-state as the intervention acknowledgement subject and the
  pre-state as the verification subject.
- Preserves transition predictions and verification targets in each causal
  action record.
- Allows one Signal binding per action and one action binding per Signal, with
  reverse lookup through `record_for_signal()`.

### Gateway execution

- Migrated the gateway from the removed `plan.actions` field to serial
  execution of `plan.steps`.
- Registers the frozen plan before reserving or executing actions.
- Registers only bridge observations returned by completed actions.
- Creates, binds, records, and returns exactly one Signal for each completed
  action.
- Does not create a causal action or Signal for a policy-rejected action or an
  action that was not executed because budget reservation failed.
- Preserves only completed observations in same-run planner history.
- Records policy, contract, provider, and budget decisions without exception
  text or provider-controlled category values.
- Re-raises `TerminalPlanError` and `BudgetExhausted` after recording stable
  artifact categories.
- Keeps all BayesProbe plan, Probe, Signal, and causal metadata out of the
  shared `HarborEnvironmentBridge` and `ActionObservation` model.

### Signal and artifacts

- Upgraded emitted Signals to `harbor-observation:v3` identity inputs.
- Added the exact causal binding block to raw Signal content: action, role,
  plan, policy attempt, request fingerprint, subject state, and verification
  target.
- Binds provenance to the causal subject environment state and references both
  the environment action and causal action artifacts.
- Keeps the model-facing observation independently bounded to 32,768 UTF-8
  bytes.
- Emits write and patch request metadata as byte counts plus SHA-256 hashes;
  large content and patch bodies do not enter Signal content.
- Retains deterministic direct Signal construction for existing fixture
  builders by creating a benchmark-local standalone causal record. Production
  gateway execution always supplies and binds the real registered record.
- Added `causal_actions.jsonl` and `causal_decisions.jsonl` artifact streams.

## TDD Evidence

All pytest commands ran from `benchmarks/terminal_bench`.

### Initial RED

Command:

```text
uv run pytest tests/test_causal.py tests/test_gateway.py -q
```

The first sandboxed invocation exited 2 before test execution because uv could
not open its shared cache. The approved rerun produced the intended RED:

```text
ERROR tests/test_causal.py
ERROR tests/test_gateway.py
ModuleNotFoundError: No module named 'bayesprobe_terminal_bench.causal'
2 errors in 0.11s
```

No production file had been changed when this failure was captured.

### Registry GREEN

After adding the minimal registry:

```text
uv run pytest tests/test_causal.py -q
8 passed in 0.02s
```

### Gateway integration iteration

The first gateway run produced:

```text
7 failed, 9 passed in 0.56s
```

Those failures identified stale test-side expectations: old swallowed errors,
observations not matching their planned requests, and repeated equivalent
Probe/plan identities now correctly rejected as duplicates. Updating the tests
to express Task 4 produced:

```text
uv run pytest tests/test_causal.py tests/test_gateway.py -q
24 passed in 0.49s
```

### Cross-plan lineage RED/GREEN

The added run-lineage regression initially failed because a new plan could
start from an unrelated environment state:

```text
1 failed in 0.06s
Failed: DID NOT RAISE ValueError("non-linear environment state")
```

After tracking the last state by run:

```text
26 passed in 0.49s
```

### Intervention-generation RED/GREEN

Self-review added a cross-plan generation regression. Before the fix:

```text
1 failed in 0.06s
assert 0 == 1
```

After making intervention generation cumulative across the run, final focused
verification was:

```text
uv run pytest tests/test_causal.py tests/test_gateway.py -q
27 passed in 0.48s
```

## Verification

### Focused Task 4 suite

```text
27 passed in 0.48s
```

### Nested suite outside frozen validator files

Command:

```text
uv run pytest -q --ignore=tests/test_benchmark_lock.py --ignore=tests/test_conformance.py
```

Result:

```text
291 passed in 8.14s
```

### Conformance file

Command:

```text
uv run pytest tests/test_conformance.py -q
```

Result:

```text
1 failed, 1 passed in 0.07s
```

The sole failure is
`test_real_runner_closes_terminal_tool_signal_into_evidence_and_cycle`.
Its out-of-scope scripted planner at `tests/test_conformance.py:41` still
constructs `TerminalProbePlan(actions=...)`, so strict Task 3 validation reports
`steps` missing and `actions` forbidden.

### Final full nested suite

Command:

```text
uv run pytest -q --tb=no
```

Result:

```text
9 failed, 349 passed in 8.53s
```

The nine remaining failures are:

1. `test_benchmark_lock.py::test_smoke_classifier_and_cli_exit_codes[1.0-complete-engineering_pass-0]`
2. `test_benchmark_lock.py::test_smoke_classifier_and_cli_exit_codes[0.0-complete-task_failure-0]`
3. `test_benchmark_lock.py::test_smoke_classifier_accepts_public_core_normalized_adapter_signal`
4. `test_benchmark_lock.py::test_smoke_classifier_allows_linked_evidence_without_directional_update[admitted]`
5. `test_benchmark_lock.py::test_smoke_classifier_allows_linked_evidence_without_directional_update[discarded]`
6. `test_benchmark_lock.py::test_smoke_classifier_allows_linked_evidence_without_directional_update[neutral]`
7. `test_benchmark_lock.py::test_smoke_classifier_allows_a_completed_no_signal_no_update_cycle`
8. `test_benchmark_lock.py::test_policy_denied_reserved_action_is_reconciled_without_an_observation`
9. `test_conformance.py::test_real_runner_closes_terminal_tool_signal_into_evidence_and_cycle`

The first eight are all rooted in the out-of-scope smoke validator at
`scripts/validate_smoke_run.py`. It is hard-coded to
`harbor-observation:v2`, exactly one `environment_actions.jsonl` artifact
reference, post-state provenance, the old raw payload, and the old composite
derivation root. Isolated execution confirms the rest of that file passes:

```text
uv run pytest tests/test_benchmark_lock.py -q --tb=no
8 failed, 57 passed in 0.25s
```

Updating that validator or `test_conformance.py` was explicitly outside Task 4
ownership, so neither was changed. The ninth failure is the stale Task 3 plan
constructor described above.

## Checks

- `git diff --check`: clean, no whitespace errors.
- Changed-file scope: only the six owned source/test files plus this required
  report. The pre-existing untracked `reports/` directory was preserved.
- Generated-lock check: no `*.lock` or `uv.lock` changes.
- Stale-contract scan: no `plan.actions` or `harbor-observation:v2` reference in
  the four owned production files.
- Credential-shape scan across all six owned files: no `sk-`, GitHub token,
  Slack token, or AWS access-key-shaped values.
- Large-body tests prove the 1,000,000-byte write and patch sentinels remain in
  the full environment artifact but do not appear in Signal raw content.

## Self-Review

Reviewed the final diff against every Task 4 brief item and the ownership list.
No Critical or Important defect remains in the owned implementation.

One concrete issue was found and fixed during self-review: environment state
continuity was run-wide, but intervention generation initially reset for each
new plan. A RED test demonstrated the reset (`0 != 1`), and the registry now
tracks cumulative intervention generation by run while retaining the separate
one-mutation-per-plan guard.

The only remaining concern is the intentionally frozen old-validator surface:
eight smoke tests reject the new v3 Signal shape, and one conformance fixture
still uses the removed `actions` plan field. Correcting them requires edits to
explicitly out-of-scope files and is therefore deferred.
