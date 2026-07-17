# Task 9 Report: Stage 0 Causal Qualification Lock and Offline Gate

## Status

`DONE - INDEPENDENT REVIEW APPROVED - STOPPED AT HARD GATE A`

Task 9 is implemented and verified entirely offline. The adapter now has a
strict Stage 0 qualification lock, a one-request provider identity artifact
flow for later authorized use, Oracle-derived lock writing, executable
per-task runtime validation, and a fixture-only qualification gate. No live
provider, network, Docker, Harbor, Oracle, canary, or benchmark call was made.
No API key was read or used.

The offline result is deliberately named `offline_gate_passed`. It does not
contain `qualification_passed` and therefore does not imply that live Stage 0
qualification has passed.

Commit subject:

```text
feat(terminal-bench): add causal qualification gate
```

## Scope Reconciliation

The plan's primary Task 9 file list did not include the adapter-owned contract
identity helpers or the active runtime dispatcher. The task brief expressly
authorized minimal changes to `provider_contract.py`, `planning.py`, and
`runner_factory.py` when required for canonical identities and executable lock
handling. Those three narrow changes were necessary and are the only changes
outside the primary file list, apart from this report and the progress ledger.

The plan also assigns three different official task timeouts to one shared
BayesProbe config. The approved executable interpretation is:

1. Keep one qualification config.
2. Invoke it separately for each Task 10 task.
3. Set `BAYESPROBE_BENCH_TASK_TIMEOUT_SECONDS` to that task's locked value:
   `1200`, `900`, or `900`.
4. Resolve the session slug to the locked task at runtime and require the
   configured task timeout to match it exactly.

This uses the existing `TerminalBenchConfig` environment handling and active
runtime-lock dispatcher. It adds no `config.py` or public-core change and no
parallel generic lock subsystem. Legacy `GateTask` locks still accept an
omitted timeout and do not serialize a new null field; the timeout is mandatory
for `CausalQualificationLock`.

Harbor `0.18.0` supports this execution shape through its dataset task filter.
Task 10 must use the shared config with both `--dataset` and
`--include-task-name`, for example:

```text
BAYESPROBE_BENCH_TASK_TIMEOUT_SECONDS=1200 harbor run \
  --config configs/bayesprobe-causal-qualification.yaml \
  --dataset terminal-bench/terminal-bench-2@<locked-dataset-revision> \
  --include-task-name terminal-bench/break-filter-js-from-html
```

The corresponding timeout is `900` for each of
`terminal-bench/cancel-async-tasks` and
`terminal-bench/log-summary-date-ranges`. Three offline `--print-config` runs
confirmed that each command resolves to exactly one target task while retaining
`n_concurrent_trials=1`; no job, container, provider, or network action was
started by those checks.

## Implementation

The lock contract adds strict, frozen `LockedBudgets` and
`CausalQualificationLock` models. It freezes Harbor `0.18.0`, the Terminal-
Bench 2 dataset, task order and refs, image digests, official task timeouts,
Git identities, configured model/base URL/protocol/temperature, seven budgets,
canonical contract identities, and provider-returned identity.

`contract_identity()` and `plan_contract_identity()` hash canonical JSON for:

- `terminal_task_frame:v1` prompt and schema;
- `terminal_probe_design:v1` prompt and schema;
- `terminal_probe_plan:v1` normal prompt, repair prompt, and schema;
- `harbor-observation:v3` schema and causal binding contract.

`capture_provider_identity.py` exposes pure helpers and a later-use CLI. The
helper makes exactly one structured request, retains only configured and
returned identity plus token usage, distinguishes unavailable fingerprint from
a non-null fingerprint, rejects malformed fingerprints, seals content with
SHA-256, names the file by that digest, writes atomically, and rejects secret-
shaped content or tampering. Its CLI was not invoked.

`write_causal_qualification_lock.py` reuses the existing Oracle result reader,
official reward parser, cached image digest resolver, Git identity collector,
and atomic writer. It requires three completed Oracle rewards of `1.0`, exact
frozen refs, cached image digests and task timeouts, clean committed adapter
identity, exact Stage 0 config, current contract hashes, and a valid immutable
provider identity artifact. Its CLI was not invoked.

`validate_causal_qualification.py` reuses Task 8's
`validate_trial_trace()`. Offline mode verifies fixture manifests and hashes,
replays the three preregistered historical classifications, and validates the
sealed conformant synthetic trace. The reusable live-fixture path validates
each locked task independently for task and agent identity, verifier
completion, exception-free completion, ATIF presence and conformance, at least
one complete cycle, provider identity and accounting, and locked dynamic
budgets. Reward is reported but never gates qualification.

Retry is available exactly once for external 429/5xx, network transport,
Docker/Harbor infrastructure, image pull, and verifier infrastructure failures.
Provider-contract, identity, budget, adapter, agent, policy, and causal errors
are never retryable.

## Frozen Stage 0 Identity

Configured model and endpoint:

```text
model: deepseek-v4-flash
base_url: https://api.deepseek.com
provider_protocol: openai_chat_completions
temperature: 0
```

Locked per-task budgets:

```json
{
  "command_timeout_seconds": 120,
  "max_model_calls": 72,
  "max_output_tokens": 8192,
  "max_provider_tokens": 160000,
  "max_total_actions": 24,
  "provider_timeout_seconds": 360,
  "signal_output_bytes": 32768
}
```

The maximum provider-token allowance is `160000` per independently invoked
task and `480000` across all three tasks if every task reaches its cap. This is
a ceiling, not measured or expected spend.

Frozen tasks:

```text
terminal-bench/break-filter-js-from-html
  ref: sha256:59a2641df9bca789642ad4ab3f5790de5ffed6eb4a594ca7846d26422a55c4a8
  agent.timeout_sec: 1200
terminal-bench/cancel-async-tasks
  ref: sha256:7c230a29f27c49c2fff88f4721165f4241e456bd87a94cd525be05ae98c6cbbb
  agent.timeout_sec: 900
terminal-bench/log-summary-date-ranges
  ref: sha256:bd0eb5e8434840a46c623c8d29c71b4a6d0fc5c7bcbf637b6d1aef36b98f5cc5
  agent.timeout_sec: 900
```

Provider-returned model, fingerprint availability/value, immutable artifact
hash, resolved dataset revision, cached image digests, Oracle rewards, and
final Git identities remain intentionally unset until the separately
authorized Task 10 inputs exist.

## TDD Evidence

The first focused run was RED before implementation:

```text
uv run pytest tests/test_qualification.py tests/test_experiment_lock.py -q

ERROR collecting tests/test_qualification.py
ModuleNotFoundError: No module named 'validate_causal_qualification'

ERROR collecting tests/test_experiment_lock.py
ImportError: cannot import name 'CausalQualificationLock'

2 errors in 0.11s
```

After the lock models and identity helpers were added, the lock suite remained
RED with `8 failed, 10 passed`; every failure reached the existing runtime
dispatcher and rejected the new causal schema. This directly demonstrated the
need for the narrow schema branch. The branch then produced `18 passed`.

Self-review regression tests also ran RED before their fixes:

```text
5 failed, 48 passed in 1.01s
```

They exposed legacy null-timeout serialization, malformed fingerprint
acceptance, missing verifier-finish validation, and network transport retry
fallthrough. One fifth failure corrected the expected Task 8 precedence for a
substituted provider model. The final focused suite is `53 passed`.

## Exact Offline CLI JSON

The required fixture-only CLI was run with uv offline mode enforced:

```text
uv run --offline python scripts/validate_causal_qualification.py \
  --historical-fixtures tests/fixtures/historical_traces --offline-only
```

It exited `0` and emitted exactly:

```json
{
  "historical_classification_counts": {
    "causal_conformance_error": 1,
    "provider_contract_error": 2
  },
  "historical_replay_passed": true,
  "historical_traces": [
    {
      "actual_classification": "provider_contract_error",
      "expected_classification": "provider_contract_error",
      "passed": true,
      "task_id": "terminal-bench/break-filter-js-from-html"
    },
    {
      "actual_classification": "causal_conformance_error",
      "expected_classification": "causal_conformance_error",
      "passed": true,
      "task_id": "terminal-bench/cancel-async-tasks"
    },
    {
      "actual_classification": "provider_contract_error",
      "expected_classification": "provider_contract_error",
      "passed": true,
      "task_id": "terminal-bench/log-summary-date-ranges"
    }
  ],
  "offline_gate_passed": true,
  "offline_only": true,
  "schema_version": "terminal_bench_causal_offline_gate:v1",
  "synthetic_classification": "conformant",
  "synthetic_complete_cycles": 1,
  "synthetic_conformant_passed": true,
  "synthetic_fixture": "conformant-inspect-intervene-verify"
}
```

The output contains no `qualification_passed` field.

## Final Verification

Fresh final commands produced:

```text
cd benchmarks/terminal_bench
uv run --offline pytest tests/test_qualification.py \
  tests/test_experiment_lock.py -q
53 passed in 0.94s

uv run --offline pytest -q
660 passed in 10.72s

cd ../..
benchmarks/terminal_bench/.venv/bin/pytest -q
1766 passed, 11 skipped in 12.97s
```

Additional checks:

- `./.venv/bin/python -m compileall -q src scripts tests` passed.
- `git diff --check` passed.
- `git diff --name-only -- bayesprobe` returned no paths.
- No root `uv.lock` was generated.
- The user-owned untracked `reports/` directory remained untouched.

## Self-Review

The final review found and fixed an incorrectly placed planner identity helper
that temporarily made `_planner_instruction()` return `None`. A regression
assertion now proves normal and repair prompt identities differ. The review
also added fail-closed coverage for malformed fingerprints, unfinished
verifiers, provider identity drift, budget excess, transport retry labeling,
and legacy lock serialization.

The final diff remains adapter-local. It introduces no BayesProbe core change,
no generic lock framework, no duplicate conformance validator, no credential
material, and no live artifact. No unresolved correctness finding remains in
the reviewed Task 9 diff.

## Residual Risks and Stop Condition

The remaining risks are intentionally live-only: provider identity may differ
from the configured model, fingerprint availability may vary, cached task
images or dataset resolution may drift, Oracle may not score all three tasks,
or a live task may fail causal qualification. Task 9 cannot resolve any of
those without crossing Hard Gate A.

Task 10 must run the one minimal canary and Oracle only after separate explicit
authorization and an environment-only provider key. The BayesProbe config must
then be invoked once per task with the matching locked timeout. No Tasks 10-14
work was performed here.

## Fix Cycle: Independent Review Corrections

The corrective cycle remains fully offline. It hardens the narrow Stage 0
adapter without changing public `bayesprobe/`, configuration, dependencies,
generated artifacts, or the user-owned `reports/` directory.

### Exact Future Job Shape

The shared BayesProbe qualification config must be run exactly three times,
with one frozen task and one unique Harbor job directory per invocation:

```text
BAYESPROBE_BENCH_TASK_TIMEOUT_SECONDS=1200 harbor run \
  --config configs/bayesprobe-causal-qualification.yaml \
  --job-name bayesprobe-causal-qualification-break-filter-js-from-html \
  --dataset terminal-bench/terminal-bench-2@<locked-dataset-revision> \
  --include-task-name terminal-bench/break-filter-js-from-html

BAYESPROBE_BENCH_TASK_TIMEOUT_SECONDS=900 harbor run \
  --config configs/bayesprobe-causal-qualification.yaml \
  --job-name bayesprobe-causal-qualification-cancel-async-tasks \
  --dataset terminal-bench/terminal-bench-2@<locked-dataset-revision> \
  --include-task-name terminal-bench/cancel-async-tasks

BAYESPROBE_BENCH_TASK_TIMEOUT_SECONDS=900 harbor run \
  --config configs/bayesprobe-causal-qualification.yaml \
  --job-name bayesprobe-causal-qualification-log-summary-date-ranges \
  --dataset terminal-bench/terminal-bench-2@<locked-dataset-revision> \
  --include-task-name terminal-bench/log-summary-date-ranges
```

The live validator intentionally requires the same sealed provider identity
artifact and all three separately generated job directories:

```text
UV_CACHE_DIR=/tmp/bayesprobe-uv-cache uv run --offline python \
  scripts/validate_causal_qualification.py \
  --historical-fixtures tests/fixtures/historical_traces \
  --lock .runs/causal-qualification.lock.json \
  --provider-identity .runs/provider-identity/<sha256>.json \
  --job .runs/harbor/causal-qualification/bayesprobe/bayesprobe-causal-qualification-break-filter-js-from-html \
  --job .runs/harbor/causal-qualification/bayesprobe/bayesprobe-causal-qualification-cancel-async-tasks \
  --job .runs/harbor/causal-qualification/bayesprobe/bayesprobe-causal-qualification-log-summary-date-ranges
```

No command above was run in this corrective cycle.

### Corrective Design

- Lock writing now requires exactly one `config.json` Oracle agent, Oracle on
  every `lock.json` trial, and Oracle in both completed result identity
  locations: `config.agent.name` and `agent_info.name`.
- `CausalQualificationLock` seals the provider artifact digest plus immutable
  fingerprint availability/value. The writer uses only the already validated,
  content-addressed artifact. Live validation reloads that artifact and checks
  its digest, returned model, and fingerprint availability/value against lock.
- Live qualification consumes exactly three `--job` directories. Each must
  contain exactly one result for a unique frozen task; missing, duplicate,
  unknown, and multi-result shapes are rejected before per-task validation.
- Successful BayesProbe summaries carry `runtime_budgets` with exactly the
  seven locked fields plus the uncapped actual `provider_tokens_used`. The
  validator requires all static values and dynamic provider-token evidence to
  equal the lock and telemetry, respectively.

### TDD Evidence

The first corrective RED run was performed before implementation:

```text
UV_CACHE_DIR=/tmp/bayesprobe-uv-cache uv run --offline pytest \
  tests/test_experiment_lock.py tests/test_qualification.py tests/test_agent.py -q

21 failed, 45 passed in 1.25s
```

Failures were the intended missing behavior: extra lock fields,
`validate_causal_qualification_job()` lacking `job_dirs` and
`provider_identity_path`, and no `runtime_budgets` in the agent summary. A
second focused RED check proved that a provider count of `160001` was being
incorrectly clamped to `160000` in the summary.

After the narrow implementation and self-review fixes, the focused GREEN run
was:

```text
UV_CACHE_DIR=/tmp/bayesprobe-uv-cache uv run --offline pytest \
  tests/test_experiment_lock.py tests/test_qualification.py tests/test_agent.py -q

71 passed in 1.66s
```

The focused tests cover Oracle provenance rejection, content-addressed
provider artifact tamper/drift/missing cases, three-job shape rejection,
reward-independent result validation, static and dynamic runtime-budget drift,
and actual provider-token recording. `git diff --check` passed during
self-review, and `git diff --name-only -- bayesprobe` remained empty.

### Provider Artifact Configuration Binding Follow-up

A final controller audit found that live validation bound the immutable
provider artifact digest, returned model, and fingerprint to the lock, but did
not independently compare the artifact's configured model, base URL, protocol,
and temperature with the corresponding locked fields. Two regression cases
first demonstrated the gap by constructing valid, content-addressed artifacts
for a different configured model and a different base URL while updating only
the lock's artifact digest:

```text
UV_CACHE_DIR=/tmp/bayesprobe-uv-cache uv run --offline pytest \
  tests/test_qualification.py::test_live_qualification_binds_provider_artifact_configuration -q

2 failed in 0.24s
```

The live validator now requires all four configured identity fields to match
the lock before accepting the returned identity. The same focused command then
passed both cases, and the complete corrective focus set passed:

```text
2 passed in 0.09s

UV_CACHE_DIR=/tmp/bayesprobe-uv-cache uv run --offline pytest \
  tests/test_qualification.py tests/test_experiment_lock.py tests/test_agent.py -q

73 passed in 1.71s
```

The fixture-only CLI was rerun after this correction and still emitted
`offline_gate_passed=true` with no `qualification_passed` field.

## Independent Review Corrective Cycle

The next independent review approved the four earlier provenance and budget
corrections, then found three additional ways that a future live qualification
could be accepted without a complete audit chain:

1. The validator consumed only trial results and adapter artifacts, without
   binding them to Harbor job configuration and lock metadata.
2. Live reports always assumed zero prior retries, so a failed retry could be
   labeled retryable again.
3. Oracle lock writing accepted an empty `finished_at` string.

Tests first reproduced all three findings, plus the overly broad fallback that
treated any exception type containing `docker` as retryable:

```text
13 failed, 9 passed in 0.85s
```

The correction now requires each current and prior job to carry one matching
Harbor `config.json`, one matching `lock.json`, and one result. Harbor version,
dataset revision, frozen task ref, agent identity, model, per-task timeout,
attempt count, concurrency, and disabled internal retries must agree with the
qualification lock. The successful agent summary records the canonical
SHA-256 of the exact runtime lock, which binds the otherwise indirect image
digest and all remaining frozen fields to the produced trace.

External retries are explicit audit inputs. An initial job uses the fixed
`bayesprobe-causal-qualification-<task-slug>` name. Its sole permitted retry
uses the same name with `-retry-1` and live validation must also receive the
failed original through `--prior-job`. A retry without its prior job, a second
retry name, a non-retryable prior result, or Harbor's own internal retry setting
is rejected. Exception-name fallback now uses a closed allowlist rather than a
substring match.

Inspection of the installed Harbor `0.18.0` source also showed that
`result.json.task_name` is the short task slug, while package identity remains
in `task_id.org`, `task_id.name`, and `task_id.ref`. Synthetic Oracle and live
fixtures were corrected to that real shape; canonical identity is now derived
from `task_id` instead of the display name. The pre-fix real-shape checks were
RED (`2 failed`), then passed after the correction.

Focused GREEN verification after this cycle is:

```text
179 passed in 2.35s
```

The fixture-only CLI still exits zero with the preregistered counts and emits
only `offline_gate_passed=true`.

For the only permitted external retry, retain the two unaffected current jobs,
replace the failed task's current job with its `-retry-1` job, and pass the
failed original separately:

```text
UV_CACHE_DIR=/tmp/bayesprobe-uv-cache uv run --offline python \
  scripts/validate_causal_qualification.py \
  --historical-fixtures tests/fixtures/historical_traces \
  --lock .runs/causal-qualification.lock.json \
  --provider-identity .runs/provider-identity/<sha256>.json \
  --job <unaffected-current-job-1> \
  --job <unaffected-current-job-2> \
  --job <retry-1-job> \
  --prior-job <failed-original-job>
```

No live command was run during this corrective cycle.

### Final Timestamp Correction

The focused re-review found one remaining fail-open case: Oracle completion
accepted any nonempty string. Parameterized tests added malformed and naive
timestamps alongside the empty-string case. Before implementation, the empty
case passed while the two new cases failed (`2 failed, 1 passed`). Lock writing
now parses ISO-8601 and requires timezone information; the same check is
`3 passed`. No other re-review finding was reported.

## Final Controller Verification and Review

The same independent reviewer rechecked the timestamp correction and returned
`APPROVED`. The complete Task 9 implementation range is
`9454008..3317d14`; its five commits are the initial gate, two provenance
hardening cycles, Harbor job/runtime-lock binding, and strict Oracle completion
validation.

Fresh controller verification after the final code commit produced:

```text
focused qualification/lock/agent/runner tests: 181 passed in 2.43s
complete Terminal-Bench adapter suite:          684 passed in 11.70s
complete repository suite:                      1766 passed, 11 skipped in 13.64s
```

`compileall` and `git diff --check` passed. The complete Task 9 range contains
no `bayesprobe/` public-core change and no secret-shaped value. The only
remaining worktree entry is the pre-existing user-owned untracked `reports/`
directory.

The final offline CLI again exited zero with two
`provider_contract_error` traces, one `causal_conformance_error` trace, one
conformant synthetic trace, and `offline_gate_passed=true`. It contains no
`qualification_passed` field. No provider, network, Docker, Harbor, Oracle,
canary, or live benchmark command was run.
