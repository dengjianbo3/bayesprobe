# Task 2 Report: Terminal-Bench Contracts

## Scope

Implemented the Terminal-Bench action, observation, configuration, and shared
budget contracts. Changes are confined to the four task-owned benchmark files
and this report.

## TDD Evidence

### RED

Command:

```sh
uv run --project benchmarks/terminal_bench pytest benchmarks/terminal_bench/tests/test_actions.py benchmarks/terminal_bench/tests/test_config.py -q
```

Output:

```text
ERROR benchmarks/terminal_bench/tests/test_actions.py
ModuleNotFoundError: No module named 'bayesprobe_terminal_bench.actions'
ERROR benchmarks/terminal_bench/tests/test_config.py
ModuleNotFoundError: No module named 'bayesprobe_terminal_bench.config'
2 errors in 0.07s
```

The tests were introduced before either production contract module existed, so
collection failed for the intended missing-feature reason.

### GREEN

Command:

```sh
uv run --project benchmarks/terminal_bench pytest benchmarks/terminal_bench/tests/test_actions.py benchmarks/terminal_bench/tests/test_config.py -q
```

Output:

```text
...................                                                      [100%]
19 passed in 0.06s
```

Broader verification command:

```sh
uv run --project benchmarks/terminal_bench pytest -q
```

Output:

```text
1766 passed, 11 skipped in 14.81s
```

`git diff --check` also completed with no whitespace errors.

## Implementation

- Added frozen, extra-forbidden Pydantic action contracts for shell, direct
  file writes, patches, plans, and executed observations.
- Classified a shell command as non-mutating only when it is a simple command
  in the conservative read-only allowlist. Model-provided
  `mutates_environment` is ignored for this decision, so it cannot reclassify
  a command in either direction.
- Enforced plan modes: inspect actions must be provably read-only, verify plans
  can contain only shell actions, and intervene plans must include a
  potentially mutating action.
- Added `RunBudget` with a shared lock for both hard reservations and reads of
  final action/model-call counters.
- Added frozen benchmark configuration with the specified defaults and bounds.
  `from_sources` merges `os.environ` and `extra_env` with `extra_env` winning,
  returns the API key separately, and has no field that can serialize the key.

## Files Changed

- `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/actions.py`
- `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/config.py`
- `benchmarks/terminal_bench/tests/test_actions.py`
- `benchmarks/terminal_bench/tests/test_config.py`
- `.superpowers/sdd/task-2-report.md`

## Self-Review

- Confirmed action validation uses the allowlist/parser rather than trusting
  the model mutation declaration.
- Confirmed direct write and patch primitives are always potentially mutating.
- Confirmed bounded plan/action/config fields match the task brief.
- Confirmed both counter reservation and counter reads acquire the same lock.
- Confirmed the production modules import no BayesProbe modules, public or
  private. The existing public-reuse policy test passes as part of the broader
  suite.
- Confirmed no files outside the allowed ownership list and this report were
  modified.

## Concerns

None for this task. Signal conversion is intentionally not implemented here:
Task 2 provides `ActionObservation` as the contract that a later gateway/signal
conversion task must accept.
