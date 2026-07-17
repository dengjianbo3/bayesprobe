# Task 7 Report: Export Complete ATIF-v1.7 Trajectories

## Status

`DONE_WITH_CONCERNS`

Task 7 is implemented in the four owned benchmark files. Both Harbor agents
now publish exactly `self.logs_dir / "trajectory.json"` for successful runs
and classified post-artifact failures. Publication uses Harbor 0.18.0's real
ATIF models and `TrajectoryValidator`, validates the final redacted payload
before writing, and atomically replaces the destination only after validation
accepts it.

The focused trajectory/agent suite and root public/paradigm suites pass. The
one required full nested-suite run retained exactly the 39 expressly deferred
Task 8 failures in `test_benchmark_lock.py` and introduced no new failure. No
live provider, Harbor job, Docker, network, credential, evaluator, or official
reward call was made.

Commit subject:
`feat(terminal-bench): emit ATIF causal trajectories`

## Scope

Production files:

- `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/trajectory.py`
- `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/agent.py`
- `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/direct_agent.py`

Test file:

- `benchmarks/terminal_bench/tests/test_trajectory.py`

Required report:

- `.superpowers/sdd/task-7-report.md`

No file under `bayesprobe/` was changed. No Task 8 fixture, generated `.runs`
data, lock, plan, historical trace, or user-owned `reports/` content was
changed. The pre-existing untracked `reports/` directory remains untouched.

## Harbor API Inspection

The implementation was based on pinned Harbor 0.18.0 source from the nested
environment, not on a parallel schema. The inspected API provides:

- `harbor.models.trajectories.Trajectory`
- `Agent`, `Step`, `ToolCall`, `Observation`, and `ObservationResult`
- `FinalMetrics`
- `harbor.utils.trajectory_validator.TrajectoryValidator`

The installed `Trajectory` accepts exactly `ATIF-v1.7`, validates sequential
step IDs, and checks every `ObservationResult.source_call_id` against a
same-step `ToolCall.tool_call_id`. Harbor v1.7 also permits deterministic
`source="agent"`, `llm_call_count=0` steps only when LLM metrics and reasoning
content are absent. The exporter uses these contracts directly.

## Implementation

### Validation and atomic publication

`write_atif_trajectory()` builds Harbor model objects, serializes with
`Trajectory.to_json_dict()`, recursively applies benchmark privacy redaction,
and calls the installed `TrajectoryValidator.validate(payload)`. A rejected
payload raises stable `TrajectoryExportError(category="adapter_error")`.

Only after validator acceptance does the exporter create a temporary file in
the Harbor agent log directory, flush and `fsync` it, then use `os.replace()`
to publish `trajectory.json`. Temporary files are cleaned on every exit. A
validation test proves that a forced validator rejection does not replace an
existing destination.

Both agents set `SUPPORTS_ATIF = True` because every supported publication
path passes through this validation-before-replace function. Any trajectory
validation or write failure is persisted as an `adapter_error` at the
`trajectory_export` stage and fails the trial.

### Common ATIF shape

Every trajectory contains:

1. Step 1 as the redacted Harbor user instruction.
2. One deterministic agent step for each executed terminal action.
3. BayesProbe causal and evidence transition steps where applicable.
4. One final system step containing the stop reason and trajectory artifact
   identity.

The root object uses:

```text
schema_version = ATIF-v1.7
trajectory_id = trajectory:<run_id>
experiment_id = terminal_bench_causal:v1
artifact_schema = terminal:v1
```

The final system step never contains official reward. The trajectory does not
read Harbor verifier output or any reward artifact.

### Request-bound terminal actions

Each executed action step has exactly one `ToolCall`, one `Observation`, and
one `ObservationResult`. The result's `source_call_id` equals that step's tool
call ID. The action request is validated through the existing strict terminal
action models and `ActionPolicy` before export, so evaluator-only targets
cannot be published.

BayesProbe action `extra` uses the existing benchmark artifacts' actual:

- `plan_id`
- `policy_attempt_id`
- `probe_id`
- `action_id`
- `signal_id`
- `request_fingerprint`

Signal identity is recovered from the public ledger's artifact reference when
present. For a completed action whose ledger write was interrupted, the
exporter deterministically recomputes the same `S_harbor_...` identity from
the action ID, full-output hash, and frozen `harbor-observation:v3` schema.

The reactive arm has no BayesProbe epistemic loop. Its trajectory-only
lineage uses stable `react-plan`, `react-policy`, `react-step`, `react-action`,
and `react-signal` identities derived from the existing ReAct plan/action
ordering. This records ATIF linkage without creating a benchmark-local
belief, Probe, Evidence, or posterior mechanism.

### BayesProbe transitions and discarded Evidence

The exporter reads the unchanged public core ledger and existing causal
decision artifacts. It emits deterministic transition steps with
`llm_call_count=0` and no `metrics` or `reasoning_content`.

Accepted or public-core-discarded Evidence steps link:

- Probe identity through Signal/action lineage;
- Signal identity;
- Evidence event identity;
- every `belief_update.update_id` whose sensitivity declares that Evidence
  event as a cause.

Causal guard decisions are represented separately, including fail-closed
discard decisions. A discarded judgment with valid action lineage links its
Probe, Signal, and action. A legitimate `unbound_signal` discard with empty
identity fields is preserved without inventing identities.

### Tokens and privacy

`final_metrics.extra.provider_tokens_used` is taken from the one shared
`RunBudget.provider_tokens_used` whenever the session exists. It is never
recomputed in place of the shared budget. The standard ATIF prompt and
completion totals are summed from existing provider telemetry; tests assert
exact `10 + 7 = 17` fixture accounting and exact equality with the shared
provider-token total.

Before Harbor validation, every string in the payload passes through both:

- exact restricted-value replacement for the active API key;
- the benchmark's established sensitive-text/evaluator-path redactor.

Executed actions are also rechecked with `ActionPolicy`. Tests prove secret
values do not survive in the file and protected evaluator paths cause an
`adapter_error` without publication. No hidden reasoning fields are emitted.

### Agent integration

The BayesProbe and direct agents both initialize their existing artifact store
before session construction. On every later classified failure they append
the stable original error record, emit a terminal failure trajectory, and
then raise the stable Harbor-facing category. If trajectory emission fails,
the outward category becomes `adapter_error` as required.

On success, each agent emits and validates the trajectory before updating
Harbor context metadata or writing its arm summary. Tests exercise real
success and classified-failure trajectories for both arms without provider or
environment calls.

## Strict TDD Evidence

All nested commands ran from `benchmarks/terminal_bench`.

### Initial RED

`test_trajectory.py` was created before production implementation. The first
required run failed during collection for the intended missing module:

```text
ModuleNotFoundError: No module named 'bayesprobe_terminal_bench.trajectory'
1 error
```

### Initial GREEN

After the smallest exporter and agent integration were added:

```text
uv run pytest tests/test_trajectory.py -q
8 passed in 0.06s
```

The initial relevant regression run was:

```text
uv run pytest tests/test_agent.py tests/test_direct_agent.py \
  tests/test_trajectory.py -q
27 passed in 1.33s
```

After adding opposite-arm success/failure integration coverage and redacted
reactive plan matching:

```text
29 passed in 1.15s
```

### Post-commit self-review RED/GREEN

Self-review identified a valid fail-closed causal decision whose empty Signal
and action IDs caused export to fail. A regression was added first:

```text
test_unbound_causal_discard_is_recorded_without_inventing_identities
1 failed: TrajectoryExportError: trajectory export failed
```

The exporter was changed to omit unavailable identities while retaining the
discard and reason. Final focused GREEN:

```text
uv run pytest tests/test_trajectory.py tests/test_agent.py \
  tests/test_direct_agent.py -q
30 passed in 1.27s
```

Every generated trajectory assertion in `test_trajectory.py` invokes
Harbor's installed `TrajectoryValidator` against the generated payload or
file.

## Broad Verification

### Required full nested suite, run once

```text
uv run pytest -q
461 passed, 39 failed in 9.42s
```

All 39 failures are in `tests/test_benchmark_lock.py`. They are the exact
deferred Task 8 fixture failures: the stale fixture calls the old
`signal_from_observation(observation=...)` API and does not yet build the
registry-bound causal artifacts required by Tasks 3-5. No Task 7 or other
nested test failed. The broad suite was not rerun after the narrow
self-review fix, honoring the instruction to run it once; the affected
focused suites were rerun and passed.

### Root public and paradigm tests

Run from the repository root:

```text
uv run pytest tests/test_public_api_and_config.py \
  tests/test_paradigm_conformance.py -q -rA
44 passed in 0.27s
```

### Repository hygiene

- `git diff --check`: clean.
- `git diff -- bayesprobe`: empty.
- Staged secret/evaluator scan: no credential pattern, generated evaluator
  path, unfinished marker, or `.runs` path. The sole `/tests/hidden.py`
  occurrence is the intentional negative behavior test.
- Staged file set contains only the four owned implementation/test files and
  this required report.
- User-owned untracked `reports/` remains untouched.

## Concerns

The only remaining failures are the exact 39 Task 8 benchmark-lock fixture
failures authorized by the brief. Task 8 must update those fixtures and add
the reusable causal validator; Task 7 intentionally does not implement that
work.

No Stage 0 or Stage 1 live run was attempted. Qualification still requires
later explicit authorization, provider identity capture, and Task 8 causal
conformance validation.
