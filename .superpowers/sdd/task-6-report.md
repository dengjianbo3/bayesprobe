# Task 6 Report: Compose the Corrected Adapter and Shared Accounting

## Status

`DONE_WITH_CONCERNS`

Task 6 is implemented in the benchmark-local files named by the authoritative
brief. The live BayesProbe arm now composes the public OpenAI gateway, one
shared budget, the terminal contract, and the causal Evidence guard in the
required order. Both Harbor arms use one task deadline, strict provider
identity and token accounting, active v1 runtime locks, bounded semantic
repairs, and classified failure persistence.

The final focused Task 6 suite and relevant root public/paradigm suites pass.
The one required full nested-suite run retained exactly the 39 already-known
Task 8 failures and introduced no new failure. No live Harbor/provider run was
performed; Hard Gate A still requires explicit authorization and a provider
identity canary in the later qualification tasks.

Commit subject:
`feat(terminal-bench): compose causal adapter and shared budgets`

## Scope

Production files:

- `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/config.py`
- `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/deadline.py`
- `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/runner_factory.py`
- `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/react.py`
- `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/agent.py`
- `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/direct_agent.py`

Test files:

- `benchmarks/terminal_bench/tests/test_config.py`
- `benchmarks/terminal_bench/tests/test_runner_factory.py`
- `benchmarks/terminal_bench/tests/test_react.py`
- `benchmarks/terminal_bench/tests/test_agent.py`
- `benchmarks/terminal_bench/tests/test_direct_agent.py`

`tests/test_public_reuse.py` remained unchanged and passed in the focused gate.
This report is `.superpowers/sdd/task-6-report.md`.

No changes were made to `bayesprobe/`, `reports/`, Task 8 benchmark-lock
fixtures, historical fixtures, plans/specifications, or live artifacts. The
pre-existing untracked `reports/` directory was left untouched. No root
`uv.lock` was generated.

## Implementation

### Exact public composition

`build_live_session()` creates exactly one `RunBudget`, `TrialDeadline`,
`TrialArtifactStore`, and `CausalTraceRegistry`, then composes:

```text
OpenAIChatCompletionsModelGateway
  -> BudgetedModelGateway
  -> TerminalContractModelGateway
  -> CausalEvidenceModelGateway
```

The same guarded gateway object is supplied to `ModelTaskFramer`,
`ModelProbeDesigner`, `BayesProbeCore`, `ModelHypothesisExpansionAdapter`, and
`TaskAwareAnswerProjector`. The benchmark-local Harbor gateway proxy injects
the same causal registry into the existing public-reuse path. The unchanged
core is configured with `EvidenceJudgmentRepairPolicy(max_attempts=2)`. No
benchmark-local belief, evidence, or posterior loop was added.

The exact decorator-order test asserts every layer by type and fails if budget
and contract are reversed. It also asserts object identity for every public
model consumer and the Harbor registry.

### Shared calls, tokens, and provider identity

- `RunBudget` has thread-safe action, logical model-call, and provider-token
  counters with a locked Stage 0 default of `160000` provider tokens.
- Every public gateway delegation and every terminal/ReAct initial or repair
  request reserves one logical model call.
- Provider usage is accepted only when `total_tokens` is a non-boolean,
  non-negative integer. Missing, boolean, fractional, string, and negative
  usage fail as `provider_identity_error` without coercion.
- Token overflow is recorded immediately after the response and raises
  `budget_error` before another call or action.
- Public-core responses are observed through the public invocation observer;
  raw response identity captured by the injected client is paired with that
  observation. Terminal and ReAct planners account before returning a valid
  plan or step.
- Model value, fingerprint availability, and fingerprint value are checked
  against the active lock for every provider response.
- Pending observer failures cross the public gateway's intentionally
  suppressing callback boundary. Budget, identity, and artifact persistence
  failures therefore cannot be silently lost.

Provider telemetry contains bounded identity, finish, usage, response ID,
validation fields, and response hashes only. It does not persist raw provider
text.

### Shared deadline

`TrialDeadline` uses the locked official `agent_timeout_seconds`. Each timeout
is recomputed as:

```text
min(configured_timeout, floor(remaining_seconds) - 5)
```

Each operation applies its own configured cap to the shared remaining trial
time. A smaller command cap does not constrain a later provider request. A
non-positive result raises `budget_error` before starting work. The OpenAI
proxy calls `base_client.with_options(timeout=current, max_retries=0)` on every
request.

The same deadline object reaches the public-core provider, terminal planner,
Harbor action bridge, and reactive planner. It is also bound to the shared
`RunBudget`, which rejects an already-expired operation before either a model
call or action reservation can increment. The OpenAI and environment proxies
retain the definitive timeout check immediately before their external
delegates. Deadline expiry uses the typed `DeadlineExhausted` subtype while
retaining the public `budget_error` category, so the reactive controller
rethrows it while preserving normal action-cap exhaustion as a clean stop. The
benchmark-local environment proxy clamps shell action timeouts and temporarily
clamps the existing Harbor bridge's non-shell timeout per invocation,
restoring it afterward. The terminal planner proxy recovers pending
deadline/accounting exceptions after the out-of-scope planner maps transport
failures.

### Reactive baseline and failures

The reactive arm remains a ReAct loop and defines no Belief, Probe, Signal,
Evidence, or posterior types. Its planner uses initial plus at most two
field-directed repairs. Every attempt is separately budgeted and observed.
Repair payloads contain only the original sanitized input, response hash,
field errors, and attempt index.

ReAct plan artifacts omit thought summaries, write contents, and patch bodies;
commands, paths, and completion summaries are bounded and redacted. Stable
provider identity and action-budget categories are preserved.

Both Harbor agents instantiate their artifact store before fallible session
construction, persist a stable category and exception type after later
failures, and rethrow a Harbor-facing exception containing only that category.
The public categories are exactly:

```text
provider_contract_error
provider_transport_error
provider_identity_error
budget_error
adapter_error
causal_conformance_error
policy_error
```

### Active v1 locks

Active runtime loading accepts only `terminal_bench_lock:v1` or
`terminal_bench_paired_gate:v1`, with `terminal_probe_plan:v1`, the locked
provider-token and task-timeout values, provider model identity, explicit
fingerprint availability, and clean runtime Git identity. Paired task image
digests are schema-validated. Active v0.1 identities are rejected.

The v0.1 loader remains only for tests that explicitly exercise historical
lock reading. It is not used by active runtime construction.

## Strict RED/GREEN Evidence

All commands in this section ran from `benchmarks/terminal_bench` unless noted.

### Initial RED matrix

Tests were added before production implementation for exact composition,
shared registry identity, logical calls, aggregate tokens, strict usage and
provider identity, deadline behavior, terminal failure propagation, ReAct
repairs/privacy, classified agent errors, and active v1 locks.

The first composition assertion failed because the core still received
`BudgetedModelGateway` directly instead of the required contract and causal
layers. Focused RED groups then reported:

```text
config/provider-token/deadline: 9 failed
observer/deadline integration: 11 failed
active v1 lock identity: 2 failed
reactive repair/accounting/privacy: 3 failed
BayesProbe agent classification: 3 failed
Direct agent composition/classification: 2 failed
```

The aggregate initial focused run was:

```text
43 failed, 105 passed
```

Two of those failures were the existing nested subprocess tests being unable
to open the sandboxed default uv cache. Setting
`UV_CACHE_DIR=/tmp/bayesprobe-uv-cache` removed that environmental failure;
the remaining failures were the intended Task 6 RED behavior.

### Corrective RED/GREEN cycle

Owned-boundary review added focused regressions before each correction:

```text
test_budgeted_gateway_surfaces_artifact_failure_swallowed_by_delegate
1 failed: DID NOT RAISE RuntimeError

deadline expiry callback + non-shell timeout + fingerprint key + ReAct artifact safety
4 failed

provider identity category + stable action-budget category
2 failed

paired task image-digest validation
1 failed: DID NOT RAISE ValueError

reactive observation-action privacy
1 failed: parsed write content remained in the observation artifact

reactive accounting exception priority
1 failed: telemetry RuntimeError replaced BudgetExhausted

cross-operation deadline cap independence
1 failed: command cap leaked into provider timeout (120 != 360)

pre-reservation deadline guard + both live compositions
6 failed: four missing require_active guards + two missing budget bindings

typed reactive deadline propagation
2 failed: DeadlineExhausted missing
2 failed after subtype addition: controller returned normal success
```

The eight earlier correction sets passed after their narrow implementation
changes: `1 passed`,
`4 passed`, `2 passed`, `1 passed`, `1 passed`, `1 passed`, and `1 passed`,
and `6 passed`, respectively.

The typed deadline correction then passed its two controller/Direct-agent
regressions (`2 passed in 0.04s`). The expanded check including ordinary
max-action exhaustion passed separately (`3 passed in 0.03s`).

### Final focused GREEN

Exact command from the brief, with only the uv cache redirected for the
sandbox:

```text
UV_CACHE_DIR=/tmp/bayesprobe-uv-cache uv run pytest \
  tests/test_config.py \
  tests/test_runner_factory.py \
  tests/test_agent.py \
  tests/test_react.py \
  tests/test_direct_agent.py \
  tests/test_public_reuse.py -q

159 passed in 1.33s
```

Compilation check:

```text
.venv/bin/python -m compileall -q \
  src/bayesprobe_terminal_bench \
  tests/test_config.py tests/test_runner_factory.py tests/test_agent.py \
  tests/test_react.py tests/test_direct_agent.py tests/test_public_reuse.py

passed with no output
```

## Full Nested Suite

The entire nested suite was run once, as required:

```text
UV_CACHE_DIR=/tmp/bayesprobe-uv-cache .venv/bin/pytest -q

39 failed, 440 passed in 9.10s
```

All 39 failures were in `tests/test_benchmark_lock.py`. Every one still failed
at the known Task 8 fixture breakpoint where `_write_smoke_job` calls the
removed standalone `signal_from_observation(observation=...)` interface. The
inventory exactly matched `.superpowers/sdd/task-5-report.md`:

1. `test_smoke_classifier_and_cli_exit_codes`: 4
2. `test_smoke_classifier_rejects_partial_lock`: 1
3. `test_smoke_classifier_accepts_public_core_normalized_adapter_signal`: 1
4. `test_smoke_classifier_allows_linked_evidence_without_directional_update`: 3
5. `test_smoke_classifier_allows_a_completed_no_signal_no_update_cycle`: 1
6. `test_policy_denied_reserved_action_is_reconciled_without_an_observation`: 1
7. `test_all_policy_denied_cycle_needs_no_observation_signal_or_update`: 1
8. `test_smoke_classifier_rejects_orphan_epistemic_rows`: 3
9. `test_smoke_classifier_rejects_missing_or_conflicting_result_identity`: 6
10. `test_smoke_classifier_rejects_stale_complete_lock_identity`: 2
11. `test_smoke_classifier_rejects_well_formed_stale_runtime_lock`: 4
12. `test_smoke_classifier_rejects_false_action_provenance_links`: 12

Count proof: `4 + 1 + 1 + 3 + 1 + 1 + 1 + 3 + 6 + 2 + 4 + 12 = 39`.

No new full-suite failure appeared. The suite was not rerun after narrow
owned-boundary review fixes, honoring the explicit one-run requirement; every
post-run change is covered by the final 159-test focused gate.

## Root and Security Verification

Run from the repository root:

```text
benchmarks/terminal_bench/.venv/bin/pytest \
  tests/test_public_api_and_config.py \
  tests/test_paradigm_conformance.py -q

44 passed in 0.56s
```

Final checks:

- `git diff --check`: passed with no output.
- `git diff -- bayesprobe`: empty.
- Modified paths: only Task 6 owned production/tests plus this report.
- Production credential/evaluator-path scan: no key, bearer, private-key, or
  protected evaluator-path matches.
- Root `uv.lock`: absent.
- `reports/`: still the pre-existing user-owned untracked directory.

## Concerns

1. Exactly 39 Task 8 benchmark-lock fixture failures remain and are unchanged;
   their fixture migration is explicitly outside Task 6 ownership.
2. No live provider, Docker, Harbor verifier, or qualification run was
   performed. Hard Gate A remains closed pending explicit authorization, a
   provider canary, and the later qualification lock workflow.
3. The historical v0.1 reader is intentionally retained for named historical
   tests only; all active runtime entry points require v1.
