# Task 2 Report: Adapter-Owned Structured Provider Contract

## Implementation

Added `TerminalContractModelGateway`, a benchmark-local decorator for only
`frame_open_question` and `design_probes`. It copies those requests, attaches a
terminal policy, validates the response with terminal-specific Pydantic models,
and makes at most two targeted repair calls. Unrelated structured requests pass
through unchanged.

The task-frame contract accepts only design/synthesis/open frames with two to
six semantically distinct diagnostic hypotheses, null answer values, and the
five allowed hypothesis types. Extra provider-owned identifiers or belief
fields fail Pydantic's forbidden-extra validation. Explicit
`implementation_policy` and `patch_choice` labels therefore fail without using
statement-wording heuristics.

The Probe contract limits proposals to one through three, checks known target
IDs, requires support and weaken maps keyed exactly by each target, requires an
available terminal capability, and requires a multi-hypothesis discriminator or
frame-coverage proposal for an initial open frame.

Every contract attempt is written through `TrialArtifactStore` to the new
redacted `provider_contract.jsonl` stream. The record contains only stage,
attempt/request metadata, a canonical response SHA-256 when content exists,
top-level required-key presence, validation state, and the bounded Pydantic
location/type diagnostics. Raw response payloads and provider exception text
are not persisted. Repair requests carry only a fully redacted payload shape.

The decorator forwards `adapter_kind`, `model_identity`, `config`, and
`invocation_observer` with the same fallback behavior as `BudgetedModelGateway`.
It delegates every initial and repair request, so composition around the
existing budget decorator charges each physical call; `BudgetExhausted` remains
visible rather than being relabeled as a provider failure.

## RED Evidence

From `benchmarks/terminal_bench`, after adding the new tests and before adding
the contract module:

```text
uv run pytest tests/test_provider_contract.py tests/test_artifacts.py -q

ERROR tests/test_provider_contract.py
ModuleNotFoundError: No module named 'bayesprobe_terminal_bench.provider_contract'
1 error in 0.09s
```

An earlier collection attempt exposed an incorrect non-public test import;
that fixture was corrected before RED was recorded. The recorded RED failure is
therefore the intended missing production module.

Two review tests were then added after the first focused GREEN run. Their RED
run produced two intended failures: malformed Probe request context raised a
`TypeError`, and a shared-budget exhaustion was incorrectly converted into
`ProviderContractError`. The implementation was corrected before rerunning
GREEN.

## GREEN Evidence And Tests

Focused contract and artifact suite:

```text
uv run pytest tests/test_provider_contract.py tests/test_artifacts.py -q
28 passed in 0.06s
```

Specified runner/public-reuse suite:

```text
uv run pytest tests/test_runner_factory.py tests/test_public_reuse.py -q
76 passed in 0.29s
```

Final nested suite, run once from `benchmarks/terminal_bench`:

```text
uv run pytest tests -q
337 passed in 8.22s
```

## Files Changed

- `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/provider_contract.py`
- `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/artifacts.py`
- `benchmarks/terminal_bench/tests/test_provider_contract.py`
- `benchmarks/terminal_bench/tests/test_artifacts.py`
- `.superpowers/sdd/task-2-report.md`

## Self-Review

- Confirmed the new production module imports only the public `bayesprobe`
  root, and the existing public-reuse test passes.
- Confirmed all response diagnostics are derived exclusively from Pydantic
  locations and error types via `safe_field_errors()` and capped at 32 entries.
- Confirmed contract artifacts contain no raw invalid payload, repair payload,
  or provider exception text; response content is represented only by a hash.
- Confirmed terminal-policy injection uses a copied request and that unrelated
  calls retain the original request object unchanged.
- Confirmed valid and invalid repairs have attempt indexes `0`, `1`, and `2`,
  with repair tasks numbered `1` and `2` in their request metadata.
- Confirmed malformed request context fails closed as a contract failure and
  budget exhaustion is preserved before another provider call occurs.
- Ran `git diff --check`; it completed with no whitespace errors. The scoped
  production secret scan returned no matches.

## Concerns

The decorator is intentionally not wired into `build_live_session` in this
task: `runner_factory.py` is outside the explicit ownership boundary. A later
integration task must compose `TerminalContractModelGateway` around the shared
budgeted provider gateway so live framing and Probe design use this contract.

## Review Fix Cycle 1

### Coverage Fix

Added an explicit differential Probe-design test using one partial-map payload
for both paths. The existing public `ModelProbeDesigner` accepts the payload,
whose `support_condition` names only `H1` while the proposal targets `H1` and
`H2`. The same payload is then returned for all three calls made through
`TerminalContractModelGateway`.

The test proves that the terminal contract:

- makes only the initial `design_probes` call and two numbered
  `repair_probe_design` calls;
- raises `ProviderContractError` with `attempts == 3` after exhaustion;
- records exactly three `invalid` attempts; and
- records the safe Pydantic diagnostic
  `proposals.0.support_condition:value_error` for every attempt.

The new focused test passed on its first run, so this cycle is honestly
classified as coverage-only. No production defect was found and
`provider_contract.py` was not modified.

### Exact Test Evidence

New differential test during development:

```text
uv run pytest tests/test_provider_contract.py::test_adapter_rejects_partial_probe_map_that_public_designer_accepts -q
1 passed in 0.04s
```

Focused contract and artifact suites:

```text
uv run pytest tests/test_provider_contract.py tests/test_artifacts.py -q
29 passed in 0.06s
```

Specified runner/public-reuse suites:

```text
uv run pytest tests/test_runner_factory.py tests/test_public_reuse.py -q
76 passed in 0.27s
```

Full nested suite:

```text
uv run pytest tests -q
338 passed in 8.17s
```

The first focused command launch was blocked before pytest started because the
sandbox could not access uv's existing cache. It was rerun with approved cache
access; only the successful pytest execution above is test evidence.

### Self-Review

- Confirmed the exact same partial-map dictionary is accepted by the public
  designer and reused for every terminal-contract response.
- Confirmed the test checks both bounded delegate tasks and persisted invalid
  attempt telemetry, not merely the final exception.
- Confirmed the expected field diagnostic contains only a Pydantic location
  and error type.
- Confirmed no production file, BayesProbe core file, Task 1 fixture, plan,
  specification, generated lock, or pre-existing `reports/` content changed.

### Concerns

None for this review fix. The previously documented later live-session wiring
boundary remains unchanged.
