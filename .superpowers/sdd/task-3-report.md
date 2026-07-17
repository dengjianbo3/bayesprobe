# Task 3 Report: Make One Terminal Probe a Causally Attributable Plan

## Status

Task 3 is implemented in the four owned benchmark files. The focused Task 3 suite and the required environment regression suite pass. The full nested suite was run once and has ten expected downstream migration failures because out-of-scope gateway and conformance tests still construct the removed `actions` schema.

Commit subject: `feat(terminal-bench): make probe plans causally attributable`

## Scope

Modified:

- `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/actions.py`
- `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/planning.py`
- `benchmarks/terminal_bench/tests/test_actions.py`
- `benchmarks/terminal_bench/tests/test_planning.py`

No environment, budget, gateway, provider-contract, BayesProbe core, historical fixture, generated lock, plan/spec, or unrelated test file was modified. The pre-existing untracked `reports/` directory was preserved.

## Implementation

### Causal plan schema

- Replaced `TerminalProbePlan.actions` with immutable `steps` and removed compatibility with the old field.
- Added strict, frozen `TerminalPlanStep` and `TransitionPrediction` models.
- Added JSON-array-to-tuple normalization for steps and transition predictions.
- Enforced inspect plans as inspect-role, provably read-only steps with no transition predictions.
- Enforced verify plans as verify-role shell commands with non-empty verification targets and no transition predictions.
- Enforced intervene order as optional inspect, exactly one intervention, and one or more trailing verification steps.
- Enforced exactly one classifier-visible mutation, read-only inspection, and shell-only targeted verification.
- Enforced optional transition predictions as exact Probe-target coverage with distinct NFKC/case-folded/whitespace-normalized transition texts.

### Planner contract

- Locked the planner instruction and repair payloads to `terminal_probe_plan:v1`.
- Replaced the single generic repair with an initial attempt plus at most two targeted repairs.
- Charged every attempt through the shared `RunBudget.reserve_model_call()` path.
- Added safe, bounded field diagnostics without validation inputs or exception text.
- Added SHA-256 response-content hashes to attempt telemetry and repair payloads.
- Added redacted invalid-payload shapes rather than forwarding raw invalid content.
- Raised `TerminalPlanError(category="provider_contract_error", attempts=3)` after three invalid responses, with no fallback plan.
- Preserved immediate, stable provider-error handling and SDK retry suppression.
- Updated the instruction to state that writes and patches are interventions, mutation success is acknowledgement rather than verification, verification follows mutation, and transition predictions precede execution.

## TDD Evidence

All commands ran from `benchmarks/terminal_bench`.

### RED

Command:

```text
uv run pytest tests/test_actions.py tests/test_planning.py -q
```

The first sandboxed invocation could not access the shared uv cache and exited 2 before test execution. The approved rerun produced the intended feature RED:

```text
ERROR tests/test_actions.py
ImportError: cannot import name 'TerminalPlanStep' from 'bayesprobe_terminal_bench.actions'
1 error in 0.11s
```

This failed because the new role-aware schema did not exist; production code had not yet been changed.

### Schema GREEN

Command:

```text
uv run pytest tests/test_actions.py -q
```

Result:

```text
46 passed in 0.03s
```

### Integrated repair-loop iteration

The first focused GREEN attempt found three test-side migration mistakes: one no-fallback test accidentally supplied a valid third response, one assertion still referenced `plan.actions`, and one exact telemetry expectation omitted the new safe diagnostic fields.

```text
3 failed, 65 passed in 0.16s
```

The tests were corrected to express the locked Task 3 contract; no production behavior was weakened.

### Focused GREEN

Command:

```text
uv run pytest tests/test_actions.py tests/test_planning.py -q
```

Result:

```text
68 passed in 0.09s
```

### Environment regression

Command:

```text
uv run pytest tests/test_environment.py -q
```

Result:

```text
29 passed in 6.09s
```

### Full nested suite, run once

Command:

```text
uv run pytest -q
```

Result:

```text
10 failed, 336 passed in 8.45s
```

All ten failures are downstream schema-migration failures outside Task 3 ownership:

- `tests/test_conformance.py`: one failure constructing `TerminalProbePlan(actions=...)`.
- `tests/test_gateway.py`: nine failures constructing `TerminalProbePlan(actions=...)`.

The failures consistently report `steps` missing and `actions` forbidden. This is the expected consequence of replacing rather than preserving the old schema. Updating gateway execution to consume `plan.steps` and migrating those tests belongs to later integration work and was not implemented here.

### Final owned-scope verification

Command:

```text
uv run pytest tests/test_actions.py tests/test_planning.py tests/test_environment.py -q
```

Result:

```text
97 passed in 6.18s
```

## Checks

- `git diff --check`: exit 0, no whitespace errors.
- Changed-file check: exactly the four owned source/test files before adding this required report.
- Changed-line secret scan: exit 1 from `rg`, meaning no matches.
- Focused planner telemetry tests confirm provider error text, malformed response accessor text, response IDs, and invalid response bodies are not leaked.

## Self-Review

Reviewed the final diff against every Task 3 brief item. No Critical or Important owned-file issue was found.

Residual concern: the nested suite cannot be fully green until the out-of-scope gateway/conformance consumers migrate from the removed `actions` field to role-aware `steps`. Adding an `actions` compatibility property or accepting the old input would violate the explicit Task 3 schema replacement requirement, so no compatibility shim was added.

## Review Fix Cycle 1

### Review findings addressed

- Fixed a security defect where provider-controlled Pydantic location components were copied verbatim into repair payloads and observer telemetry.
- Changed planner guidance so transition predictions are explicitly optional and their completeness, differentiation, and pre-execution declaration requirements apply only when predictions are provided.

### Root cause and fix

`_safe_field_errors()` previously joined every Pydantic `loc` component with `str(part)`. An unknown extra field name is provider-controlled, so a secret-shaped key could cross the provider boundary through `field_errors` even though raw invalid payload values were redacted.

The minimal fix adds an explicit bounded allow-list of terminal-plan schema field names. Numeric indices are preserved, known field names remain visible, and every other location component becomes the fixed `<field>` placeholder. Error types, set-based deduplication, lexical sorting, and the 32-item cap are unchanged.

### RED evidence

Security regression command:

```text
uv run pytest tests/test_planning.py::test_provider_controlled_field_locations_are_sanitized_everywhere -q
```

Result before production changes:

```text
1 failed in 0.12s
```

The assertion showed the synthetic `sk-...` field name inside persisted `field_errors` in observer telemetry and repair requests.

Instruction regression command:

```text
uv run pytest tests/test_planning.py::test_planner_instruction_states_causal_execution_semantics -q
```

Result before the wording change:

```text
1 failed in 0.08s
```

The old instruction lacked the conditional phrase `Transition predictions are optional; when provided`.

### GREEN evidence

Initial focused fix command:

```text
uv run pytest tests/test_planning.py::test_provider_controlled_field_locations_are_sanitized_everywhere tests/test_planning.py::test_planner_instruction_states_causal_execution_semantics -q
```

Result:

```text
2 passed in 0.02s
```

The security fixture was then nested under `steps[0]` to prove known-name and numeric-index preservation. Its first run exposed the additional legitimate `steps:too_short` Pydantic diagnostic; the expectation was updated to retain both sorted errors. Final nested regression result:

```text
1 passed in 0.04s
```

Focused actions/planning command:

```text
uv run pytest tests/test_actions.py tests/test_planning.py -q
```

Result:

```text
69 passed in 0.08s
```

Environment command:

```text
uv run pytest tests/test_environment.py -q
```

Result:

```text
29 passed in 6.08s
```

Full nested command, run once for this review cycle:

```text
uv run pytest -q
```

Result:

```text
10 failed, 337 passed in 8.52s
```

The ten failures remain exactly the documented downstream schema migration breakpoints: nine in `tests/test_gateway.py` and one in `tests/test_conformance.py`, all caused by out-of-scope construction of the removed `actions` field.

### Checks and self-review

- `git diff --check`: exit 0.
- Changed-file scope before this report append: only `planning.py` and `test_planning.py`.
- Secret candidate scan found only the deliberate synthetic sentinel in the security regression.
- Secret scan excluding that exact test sentinel: exit 1 from `rg`, meaning no candidates.
- Confirmed known schema names and integer indices remain visible as `steps.0`, while the provider-controlled component is `<field>`.
- Confirmed telemetry and both repair payloads contain only sanitized locations and error types; the exception text contains no provider field name.
- Confirmed deduplication, deterministic sorting, and the 32-item cap are structurally unchanged.
- Confirmed inspect/verify instructions no longer imply transition predictions are required.

No Critical or Important issue remains within Task 3 ownership. The only concern is the already documented downstream gateway/conformance migration, which remains intentionally untouched.
