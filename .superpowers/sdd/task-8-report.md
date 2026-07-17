# Task 8 Report: Reusable Causal Conformance Validator

## Status

`DONE`

Task 8 is implemented and verified. The benchmark now exposes one strict,
offline `validate_trial_trace(Path)` API and both validation scripts delegate
causal trace judgment to it. The 39 previously deferred
`test_benchmark_lock.py` failures were removed by updating only their stale
synthetic artifact helper to the current registry-bound causal and ATIF
contract; experiment-lock assertions and semantics were not weakened.

The independent-review follow-up is also complete. It closes the empty-trace,
decision-routing, provider-identity, exact-ATIF, strict-epistemic-payload, and
precedence-test findings without changing the public API or broadening the
Task 8 architecture.

No live provider, network, Harbor or Docker job, credentials, official reward,
or `.runs` generation was used. No file under `bayesprobe/` was changed, and
the pre-existing untracked `reports/` directory was left untouched.

Commit subjects:

- `feat(terminal-bench): validate causal trace conformance`
- `fix(terminal-bench): harden causal conformance validation`

## Scope

Production and script files:

- `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/conformance.py`
- `benchmarks/terminal_bench/scripts/validate_smoke_run.py`
- `benchmarks/terminal_bench/scripts/validate_paired_gate.py`

Tests and deterministic fixtures:

- `benchmarks/terminal_bench/tests/test_conformance.py`
- `benchmarks/terminal_bench/tests/test_benchmark_lock.py`
- `benchmarks/terminal_bench/tests/fixtures/causal_traces/`

Required report:

- `.superpowers/sdd/task-8-report.md`

No other production or test file was modified.

## Public API

`conformance.py` implements the fixed benchmark-local API from the brief:

- the exact five-member `TraceClassification` enum;
- a strict, frozen, extra-forbidden `ConformanceReport` with the prescribed
  fields;
- `validate_trial_trace(artifact_root: Path) -> ConformanceReport`.

The validator is independent of official reward and parses JSON/JSONL through
structured Pydantic models and duplicate-key-rejecting JSON readers. It
recomputes causal identities and fingerprints rather than trusting copied
artifact fields.

## Validation Semantics

The validator fails closed over the complete Task 3-7 artifact contract:

1. Provider contract attempts must have bounded, contiguous initial/repair
   sequences, correct tasks and terminal valid outcomes.
2. Plans, executed actions, causal actions, Signals, Evidence events, and
   decisions must have exact cardinality and unique identities.
3. Request fingerprints, action IDs, Signal IDs, canonical content hashes,
   environment states, intervention generations, policy attempts, and subject
   state lineage are recomputed and reconciled.
4. Evidence admission/discard records must correspond to their causal guard
   decisions. A conformant guard discard is counted and remains non-failing.
5. Discarded Evidence cannot contribute or cause an Update. Every non-neutral
   contribution and Update must declare exactly one admitted causal route, and
   each Update must have exactly one matching non-neutral contribution.
6. Task-frame and Evidence prompt/schema provenance must retain the active
   `v0.2` identities.
7. Summary counters, configured/default limits, provider token totals, provider
   model and system fingerprint identity, and ATIF final metrics must agree.
8. Harbor's installed `TrajectoryValidator` must accept ATIF-v1.7, and every
   exported action must have exact tool-call/result/action/Signal linkage.
9. All artifact text is scanned with the established secret and evaluator-path
   patterns. Symlinks and unreadable artifact content fail closed.

The deterministic classification table is encoded in one ordered constant and
table-tested:

```text
security/evaluator access -> causal_conformance_error
causal violation          -> causal_conformance_error
provider violation        -> provider_contract_error
budget violation          -> budget_error
adapter violation         -> adapter_error
no violation              -> conformant
```

This preserves the fixed public enum while giving security violations the
planned highest precedence and top-level causal classification.

## Fixtures and Historical Replay

The content-addressed fixture set contains one valid inspect/intervene/verify
trace and four one-defect causal variants:

- request fingerprint mismatch;
- environment-state mismatch;
- discarded Evidence causing an Update;
- an ambiguous multi-Evidence Update route.

`manifest.json` pins SHA-256 for all 45 artifact files. Tests reject digest
drift, absolute source paths, symlinks, secrets, and evaluator paths. The valid
trace includes an observable guard discard and classifies `conformant`; every
broken trace classifies `causal_conformance_error`.

Historical expectations are now evaluated through the reusable API: the two
pre-Probe traces classify `provider_contract_error`, while the old completed
Semaphore/TaskGroup trace classifies `causal_conformance_error`.

## Script Integration

`validate_smoke_run.py` retains its lock, Harbor result, exception, output, and
CLI compatibility. Its former script-local causal parser was removed;
`_complete_trace()` is now a compatibility boolean wrapper around
`validate_trial_trace()`. Provider failure detection also consumes the report
before preserving the existing Harbor exception fallback.

`validate_paired_gate.py` likewise keeps its injectable boolean callback and
existing output surface while its default callback delegates to the reusable
validator. No experiment-lock semantics changed.

## Benchmark-Lock Compatibility

The authorized helper in `test_benchmark_lock.py` now constructs current
`TerminalProbePlan`, `CausalTraceRegistry`, action, Signal, decision, provider
contract, telemetry, summary, and strict ATIF artifacts. Policy-denied cases
retain their original assertions while reconciling reserved action counters
and removing causal artifacts only where no observation exists.

Before modernization, the file reported exactly:

```text
39 failed, 26 passed
```

All 39 failures came from the removed standalone
`signal_from_observation(observation=...)` API. After modernization:

```text
65 passed
```

## TDD Evidence

Tests and fixtures were written before production implementation. The required
initial focused command failed during collection for the intended reason:

```text
ModuleNotFoundError: No module named 'bayesprobe_terminal_bench.conformance'
```

After implementation and wrapper migration, the final focused/wrapper suite
completed with:

```text
102 passed
```

The focused set includes conformance, historical replay, benchmark-lock, smoke
wrapper, and paired-gate coverage.

## Independent Review Follow-Up

The review fixes were implemented test-first in the same worktree and remain
inside Task 8 ownership:

1. Empty artifact directories now fail closed as `adapter_error`. Completed or
   otherwise substantive traces with an incomplete causal envelope are checked
   independently and classify causally, while historical provider failures
   retain provider precedence.
2. Every `causal_decisions.jsonl` row must validate as either a final
   `CausalDecision` or the adapter's explicit diagnostic/action-failure shape.
   Unknown and malformed rows fail closed. Every Signal/Evidence route requires
   exactly one final decision, and every admitted or discarded Evidence event
   must agree with that decision regardless of free-text discard wording.
3. Provider telemetry is required whenever attempts, a substantive trace, or a
   completed ledger trace exists. Successful calls require the
   `system_fingerprint` key; explicit null remains legal, and model/fingerprint
   availability and values must remain consistent across calls.
4. Each ATIF terminal tool call/result is recomputed from its executed
   `ActionObservation`/`CausalActionRecord`. Call ID, function, complete action
   arguments, result source/content, and all result metadata must match exactly.
5. EvidenceEvent, EvidenceContributionDelta, and BeliefUpdate ledger payloads
   are validated before causal reasoning with exact required fields, forbidden
   extras, strict finite numbers, and typed collections. The public root-exported
   `EvidenceContributionDelta` model is reused in strict mode. EvidenceEvent and
   BeliefUpdate use narrow adapter-local strict models because their core models
   are not public root exports. Invalid rows are excluded from downstream
   neutrality and route logic, so a malformed contribution cannot appear
   neutral.
6. Classification precedence tests now activate real security, causal,
   provider, budget, and adapter detectors in representative collisions. They
   no longer rely on injected category labels in `errors.jsonl`.

The strict epistemic mutation matrix was RED before implementation:

```text
uv run pytest tests/test_conformance.py -q \
  -k epistemic_payloads_are_strict_before_causal_reasoning --tb=short
12 failed
```

The malformed string-valued zero contribution was initially accepted as
conformant, directly demonstrating the neutral-fallback defect. After the
adapter-local validators were added, the same matrix produced `12 passed`.
The first complete nested-suite run then exposed an internal-core import in
`test_public_reuse.py`; replacing it with the public root export plus local
models changed that result from `1 failed, 574 passed` to a fully green nested
suite.

## Final Verification

Fresh post-review commands produced:

```text
cd benchmarks/terminal_bench
uv run pytest tests/test_conformance.py tests/test_historical_fixtures.py \
  tests/test_benchmark_lock.py tests/test_paired_gate.py -q --tb=short
133 passed in 1.37s

uv run pytest -q --tb=short
575 passed in 9.84s

cd ../..
uv run pytest tests/test_public_api_and_config.py \
  tests/test_paradigm_conformance.py \
  tests/evaluation/test_paradigm_checkpoint.py -q --tb=short
50 passed in 0.42s

uv run pytest -q --tb=short
1766 passed, 11 skipped in 13.85s
```

Additional checks completed successfully:

- `uv run python -m compileall -q src scripts tests`
- `git diff --check`
- `git diff --name-only -- bayesprobe` returned no paths
- causal fixture secret/evaluator scan returned no matches
- causal fixture symlink scan returned no paths
- manifest integrity and portability tests passed in the nested suite
- the task-generated worktree-root `uv.lock` was removed before staging

## Self-Review

The final diff remains inside the authorized Task 8 ownership boundary. The
validator does not introduce an evidence integrator, posterior updater, task
loop, or core-control-flow change. It does not read official reward. The
scripts contain no second generic validator, active Task 7 ATIF identities are
unchanged, and fixture helper updates preserve the benchmark lock's assertions.
The cumulative follow-up diff imports production schemas only from the
`bayesprobe` package root, and the public-reuse guard passes. The only untracked
path left in the worktree is the pre-existing user-owned `reports/` directory.

No unresolved correctness finding remains from self-review.

## Fix Cycle 2

The second independent re-review identified three remaining fail-closed gaps.
This cycle fixes only those findings:

1. A conformant result now requires a structured trace envelope and at least
   one substantive causal trace, completed cycle, or recognized explicit
   terminal error. Empty, whitespace-only, empty-object, and non-substantive
   one-file traces classify `adapter_error`. Standalone recognized terminal
   errors retain their causal, provider, budget, or adapter classification.
2. `artifact_root` is checked with `is_symlink()` before `is_dir()`. A root
   symlink returns a security violation and is never traversed.
3. Every `errors.jsonl` record must carry a nonempty category from
   `_ERROR_CATEGORY_MAP`, except for the existing special `policy_error`
   category. Missing, empty, and unknown categories add an adapter violation;
   valid policy-denial records continue through their existing correspondence
   validation.

The envelope regression matrix covers all ten recognized artifact families:
ledger, errors, provider contract/telemetry, plans, executed actions, causal
actions, causal decisions, summary, and trajectory. Each family is exercised
as an empty file, whitespace-only file, and syntactically valid empty object.
A separate nonempty task-admission-only ledger probe verifies the semantic
invariant.

### RED Evidence

The tests were added before production changes. The focused mutation command
failed for exactly the reported gaps while all explicit-terminal and
`policy_error` controls passed:

```text
uv run pytest tests/test_conformance.py -q --tb=short \
  -k "isolated_meaningless or nonempty_non_substantive or \
artifact_root_symlink or invalid_errors_category or explicit_terminal_error or \
policy_error_category"
30 failed, 10 passed, 52 deselected in 0.37s
```

### GREEN Evidence

After the narrow validator changes, the same command produced:

```text
40 passed, 52 deselected in 0.13s
```

Fresh full verification produced:

```text
cd benchmarks/terminal_bench
uv run pytest tests/test_conformance.py tests/test_historical_fixtures.py \
  tests/test_benchmark_lock.py tests/test_paired_gate.py -q --tb=short
173 passed in 1.43s

uv run pytest -q --tb=short
615 passed in 9.89s

cd ../..
uv run pytest tests/test_public_api_and_config.py \
  tests/test_paradigm_conformance.py \
  tests/evaluation/test_paradigm_checkpoint.py -q --tb=short
50 passed in 0.43s

uv run pytest -q --tb=short
1766 passed, 11 skipped in 13.88s
```

`uv run python -m compileall -q src scripts tests`, `git diff --check`,
and `git diff -- bayesprobe` also completed successfully. The root `uv.lock`
generated by root verification was removed, and the pre-existing untracked
`reports/` directory remains untouched.

### Fix Cycle 2 Self-Review

The cumulative change modifies only the Task 8 validator, its conformance
tests, and this report. Security scanning still runs before the semantic
envelope early return, classification precedence is unchanged, legitimate
terminal and policy categories remain accepted, and no public-core file or
control flow was modified. No unresolved finding remains.
