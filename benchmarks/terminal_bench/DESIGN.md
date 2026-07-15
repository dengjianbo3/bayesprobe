# BayesProbe Terminal-Bench 2.0 Adapter Design

**Status:** Approved design, pre-implementation

**Date:** 2026-07-14

**Scope:** Terminal-Bench 2.0 integration only

## 1. Decision

The first coding-agent benchmark for BayesProbe will be Terminal-Bench 2.0, run
through the official Harbor harness.

The integration will live in an isolated nested project at
`benchmarks/terminal_bench/`. It will depend on the repository's public
BayesProbe interfaces, but it will not modify `bayesprobe/`, the root
`pyproject.toml`, or the BayesProbe control flow.

BayesProbe remains one complete control system:

```text
Belief State -> Probe -> Signal -> Evidence -> Update
```

Terminal commands are not a second agent paradigm embedded inside BayesProbe.
They are environment actions selected during the Probe phase. Their observed
results become Signals. The Evidence phase interprets those Signals, and only
then may the Belief State change.

The Harbor verifier is the sole authority for task success. BayesProbe's answer
projection may decide that work is ready for verification, but it cannot award
itself success or replace the modified task workspace.

### 1.1 Downstream-consumer contract

The benchmark project is an engineering consumer of the installed BayesProbe
package, not a second implementation of it. Production benchmark source may
import BayesProbe symbols only from the package's top-level public interface:

```python
from bayesprobe import AutonomousQuestionRunner, BayesProbeCore, ProbeExecutor
```

It must instantiate the real `AutonomousQuestionRunner`, `BayesProbeCore`,
evidence path, updater, task framing, and ledger implementations. Benchmark code
may provide adapters for public extension interfaces such as `ProbeToolGateway`,
`ProbeDesigner`, progress observation, and `ModelGateway`, but it may not copy or
replace the autonomous cycle.

Imports from private implementation modules such as `bayesprobe.core` or
`bayesprobe.question_runner` are forbidden in benchmark production source. If a
required Terminal-Bench behavior cannot be expressed through the public
interface, the integration must fail with a documented public-interface gap. It
must not bypass the gap by duplicating kernel behavior in the benchmark project.

## 2. Why Terminal-Bench First

Terminal-Bench is the lower-risk first integration because it exposes a general
terminal environment rather than requiring BayesProbe to generate a repository
patch in a benchmark-specific submission format. Harbor also provides an
official agent interface, container lifecycle, task timeout, artifact layout,
and verifier execution.

SWE-bench remains a future validation target. It is deferred because its local
harness has heavier storage and architecture constraints, and because adapting
an open-ended coding loop to Terminal-Bench first gives a cleaner test of the
BayesProbe method without changing the core.

Local environment implications recorded during design:

- Host architecture is Apple Silicon (`arm64`).
- Docker has approximately 12 CPUs and 25 GB memory assigned.
- Available disk space is approximately 144 GB.
- Harbor is not currently installed.

This is sufficient for a small Terminal-Bench smoke test. It is not treated as
evidence that a full local SWE-bench run is practical.

## 3. Goals

1. Prove that BayesProbe can operate a real, persistent terminal environment and
   reach an official Terminal-Bench verifier.
2. Preserve the BayesProbe epistemic sequence without introducing ReAct,
   ReWOO, or another controller inside the BayesProbe arm.
3. Make terminal observations genuine external Signals rather than model-written
   summaries masquerading as evidence.
4. Produce a complete trace from each Belief State through Probe, Signal,
   Evidence, and Update.
5. Compare BayesProbe with a minimal ReAct shell baseline under the same model,
   executor, action budget, model-call budget, and task timeout.
6. Keep all benchmark-specific code, dependencies, configuration, and generated
   data outside the BayesProbe core package.

## 4. Non-Goals

- Modifying the BayesProbe kernel or public semantics.
- Building a generic coding-agent SDK.
- Running SWE-bench in this milestone.
- Adding WebUI controls for Terminal-Bench.
- Cloud orchestration, distributed workers, or leaderboard submission.
- Multi-agent collaboration.
- Optimizing performance against repeatedly observed smoke-task results.
- Treating three smoke tasks as a representative accuracy measurement.

## 5. Alternatives Considered

### 5.1 External Harbor agent: selected

Implement a Harbor `BaseAgent` in the host process. It calls the existing
`AutonomousQuestionRunner`, while a benchmark adapter exposes Harbor's task
environment as BayesProbe probe actions.

Advantages:

- leaves task containers clean;
- avoids installing BayesProbe and provider credentials into every task image;
- uses Harbor's intended custom-agent boundary;
- isolates benchmark code from the core package.

### 5.2 Install BayesProbe inside every task container: rejected

This would add installation latency, dependency conflicts, image pollution,
credential handling, and task-specific bootstrap behavior. Those variables
would obscure whether failures came from BayesProbe or benchmark setup.

### 5.3 Add shell and coding behavior to the BayesProbe core: rejected

Repository editing and Harbor lifecycle concerns are benchmark capabilities, not
new epistemic primitives. Adding them to the kernel would violate the agreed
scope and make the benchmark reshape the method being evaluated.

## 6. Project Boundary

The implementation milestone will create the following nested project:

```text
benchmarks/terminal_bench/
|-- pyproject.toml
|-- uv.lock
|-- README.md
|-- DESIGN.md
|-- .gitignore
|-- configs/
|   |-- oracle-smoke.yaml
|   |-- bayesprobe-smoke.yaml
|   `-- baseline-smoke.yaml
|-- src/bayesprobe_terminal_bench/
|   |-- agent.py
|   |-- runner_factory.py
|   |-- actions.py
|   |-- planning.py
|   |-- environment.py
|   |-- gateway.py
|   |-- signals.py
|   |-- artifacts.py
|   |-- config.py
|   `-- baseline.py
|-- scripts/
|   |-- write_benchmark_lock.py
|   `-- validate_smoke_run.py
`-- tests/
```

The nested project will own its dependencies and lockfile. It will use a local,
editable dependency on the repository root. The root project will not acquire a
Harbor dependency.

The nested project pins the first implementation to Harbor `0.18.0` and Python
`>=3.12`, matching Harbor's current stable package contract. `environment.py`
contains only the action policy and async/sync environment bridge. There is
deliberately no benchmark-owned `loop.py`.

Generated runs, provider caches, and downloaded benchmark data will be ignored
by the nested `.gitignore`.

## 7. Runtime Architecture

```text
Harbor job
  -> BayesProbeHarborAgent.run(task, environment)
     -> AutonomousQuestionRunner.run_question(task instruction)
        -> Belief State
        -> terminal-aware Probe planner
        -> HarborEnvironmentGateway
           -> shell / write_file / apply_patch
           -> observed stdout, stderr, exit code, timeout, mutation status
        -> ExternalSignal records
        -> existing Evidence interpretation
        -> existing posterior Update
        -> next cycle or existing runner stop reason
  -> Harbor verifier inspects final workspace
  -> official reward and artifacts
```

The adapter will reuse `AutonomousQuestionRunner`; it will not duplicate the
autonomous loop. Benchmark-owned implementations may be supplied through the
runner's existing extension points for probe planning, probe execution, task
framing, and answer projection.

The initial engineering slice directly constructs the existing runner and uses
its existing stop reasons: `NO_PROBES`, `EPISTEMIC_STAGNATION`,
`CONFIDENCE_REACHED`, `POSTERIOR_STABLE`, or `MAX_CYCLES`. The adapter does not
introduce an external completion flag or a second controller. Once
`run_question` returns, Harbor proceeds to the official verifier.

### 7.1 Async bridge

Harbor's agent and environment APIs are asynchronous, while the current
BayesProbe runner and probe executor are synchronous. The bridge will preserve
both interfaces without changing the core:

1. `BayesProbeHarborAgent.run` captures Harbor's running event loop.
2. It executes `AutonomousQuestionRunner.run_question` in a worker thread with
   `asyncio.to_thread`.
3. A synchronous probe action submits the corresponding Harbor environment
   coroutine to the captured loop with `asyncio.run_coroutine_threadsafe`.
4. The worker waits for the result under the configured command timeout.
5. Timeout and cancellation are converted into explicit action results and
   Signals; they are not silently swallowed.

The bridge must never call `asyncio.run` inside Harbor's active event loop.

## 8. Coding-Task Belief Model

Terminal-Bench tasks are open-ended. The adapter must not reduce them to a fixed
yes/no pair or invent a closed answer list.

The Belief State represents competing, revisable claims about:

- the current cause of failure;
- the most promising solution strategy;
- whether an intervention produced the intended environment change;
- whether the task is ready for official verification.

Hypotheses may be introduced, revised, split, or retired through the existing
open-ended hypothesis-admission mechanism. A hypothesis is not accepted merely
because the model generated it repeatedly.

A coding run is complete only when the existing runner returns and Harbor runs
the verifier, or when a hard budget or task timeout is reached. A natural-language
answer alone is not a valid Terminal-Bench deliverable.

## 9. Probe Action Protocol

Each high-level Probe has one of three modes:

- `inspect`: read files, search content, inspect processes, or query environment
  state;
- `intervene`: write files, apply a patch, install allowed dependencies, or
  change configuration;
- `verify`: compile, run tests, exercise a service, or inspect produced output.

These modes classify environment actions inside a selected BayesProbe Probe;
they do not control the autonomous cycle. In particular, the MVP action schema
does not define a `finish` action.

The model produces a structured probe plan:

```json
{
  "mode": "inspect",
  "actions": [
    {
      "type": "shell",
      "command": "pwd",
      "timeout_seconds": 30,
      "mutates_environment": false
    },
    {
      "type": "shell",
      "command": "ls -la",
      "timeout_seconds": 30,
      "mutates_environment": false
    }
  ],
  "expected_observation": "A concrete error linked to one candidate cause"
}
```

The permitted low-level action primitives are:

```text
shell(command, timeout_seconds)
write_file(path, content)
apply_patch(patch)
```

A high-level Probe contains at most three low-level actions. `write_file` and
`apply_patch` are encoded by the adapter rather than by asking the model to build
shell heredocs. This keeps quoting, binary boundaries, and result attribution
deterministic.

The model's `mutates_environment` declaration is advisory, not authoritative.
The adapter permits an `inspect` shell action only when a conservative classifier
can prove that it is one simple read-only command. Shell composition,
redirection, interpreters, package managers, tests, and unknown executables are
treated as potentially mutating. `verify` may execute such shell commands, but
the environment lineage then advances. `write_file` and `apply_patch` are always
mutating interventions.

One schema-repair request is allowed for malformed model output. If repair
fails, the Probe ends with a `plan_error`; the adapter must not infer and execute
an imagined command.

## 10. Signal and Evidence Semantics

Every executed action creates one distinct `ExternalSignal` record containing
at least:

- action and probe identifiers;
- action type and normalized request;
- stdout and stderr;
- exit code, timeout, or transport failure;
- start time, duration, and output truncation metadata;
- pre-action and post-action environment state identifiers;
- `epistemic_origin=TOOL_RESULT`;
- a content hash for the full raw observation.

Successful and failed writes or patches are also Signals. A non-zero exit code,
failed test, compiler error, or service failure is an execution observation, not
a system exception.

The following are not Signals:

- the model's proposed command;
- a chain-of-thought or plan;
- an expected observation;
- a self-assessment that a change probably worked;
- repeated restatements of an earlier result.

The Evidence stage is model-mediated: it evaluates what an observed Signal means
for specific hypotheses. Evidence must cite its source signal identifiers and
environment state. The updater consumes admitted Evidence, not raw model prose.

### 10.1 Environment lineage

Every attempted action classified as potentially mutating advances
`environment_state_id`, including a timeout or non-zero exit, because partial
mutation cannot be excluded. Only shell actions proven read-only retain the
current state identifier. Repeating an observation against an unchanged
environment is correlated evidence and must not be counted as an independent
confirmation merely because it was rerun.

### 10.2 No-signal invariant

If a cycle obtains no admissible external Signal, it may record uncertainty,
planning failure, or exhaustion, but it must not directionally strengthen or
weaken a hypothesis posterior. This is an acceptance invariant for the adapter,
not a benchmark-specific substitute for the BayesProbe updater.

### 10.3 Output limits

The model-facing Signal payload is capped at 32 KB per action. The adapter keeps
the full raw output in the run artifacts and records how the model-facing view
was truncated.

## 11. Security and Isolation

The action policy will deny direct agent access to:

- `/tests` when it contains hidden evaluator material;
- `/solution`;
- `/logs/verifier`;
- the Docker socket;
- Harbor internals, oracle output, or verifier-only artifacts.

The exact paths will be resolved from Harbor's task contract rather than assumed
from a single image. Attempts to access protected paths produce `policy_error`
records and are not executed.

Network access follows the Terminal-Bench task policy. The adapter cannot broaden
a task's network permissions.

Provider credentials are supplied only through environment variables:

```text
BAYESPROBE_BENCH_API_KEY
BAYESPROBE_BENCH_BASE_URL
BAYESPROBE_BENCH_MODEL
```

Configuration and artifacts may record the variable names but never their
values. Tests scan emitted artifacts for secret leakage.

## 12. Fixed MVP Budgets

The initial benchmark adapter uses these preregistered limits:

| Limit | Value |
| --- | ---: |
| Maximum BayesProbe cycles | 3 |
| Maximum high-level probes per cycle | 2 |
| Maximum actions per high-level probe | 3 |
| Maximum terminal actions per trial | 24 |
| Maximum logical model calls per trial | 72 |
| Per-command timeout | 120 seconds |
| Provider request timeout | 360 seconds |
| Model-facing output per action | 32 KB |

The trial timeout remains the official Terminal-Bench value for the task. The
adapter will not extend it to compensate for its own overhead.

A logical model call is one BayesProbe structured request, one terminal-plan
request, or one terminal-plan repair request. Transport retries inside that
request do not create additional logical budget units. Telemetry records every
attempt exposed by the provider Adapter; SDK-internal retries, if any, are not
misreported as separate logical calls.

## 13. Baseline and Fairness

The first scientific control is a minimal benchmark-local `ReActShellAgent`, not
Terminus-2 alone. The control and BayesProbe arms use:

- the same provider, model, base URL, temperature, and per-call token limit;
- the same low-level action executor and action policy;
- the same maximum 24 terminal actions;
- the same maximum 72 model calls;
- the same command and provider timeouts;
- the same official task timeout and verifier.

Only the state and controller differ:

- ReAct keeps a `Thought -> Action -> Observation` history.
- BayesProbe keeps `Belief -> Probe -> Signal -> Evidence -> Update` state.

Token counts, latency, and provider cost are recorded for both arms. The first
comparison is budget-matched by actions, model calls, and task timeout; a
token-normalized comparison is reported separately rather than silently changing
one arm's limits.

Harbor's Oracle agent validates harness health only. Terminus-2 may later serve
as a public reference point, but it is not the sole baseline for the MVP.

Implementation is split into two independently testable plans. The first plan
delivers the BayesProbe engineering vertical slice through the official verifier.
The second plan adds the ReAct control and paired experiment runner only after
the reuse and conformance gates pass. This sequencing does not change the
agreed comparison; it prevents baseline work from hiding integration defects.

## 14. Reproducibility Lock

Before any real BayesProbe or ReAct benchmark run, an Oracle smoke run must
succeed and the adapter must write `benchmark.lock.json` containing:

- exact Harbor version;
- resolved Terminal-Bench dataset version;
- task container image digest;
- BayesProbe repository Git SHA;
- benchmark adapter Git SHA;
- prompt and schema versions;
- all budget and timeout values;
- model and provider configuration excluding secrets;
- selected task identifiers and repetition counts.

No real agent run begins before this lock exists. Task identifiers are frozen
after the Oracle bootstrap and cannot be replaced based on either agent's
outcome.

## 15. Run Artifacts

Each trial produces a benchmark-local view with this shape:

```text
.runs/<run-id>/<trial-id>/
|-- manifest.json
|-- environment_actions.jsonl
|-- provider_telemetry.jsonl
|-- trajectory.json
|-- bayesprobe/
|   |-- ledger.jsonl
|   |-- belief_states.jsonl
|   |-- probes.jsonl
|   |-- signals.jsonl
|   |-- evidence.jsonl
|   `-- summary.json
`-- verifier/
    |-- result.json
    |-- reward.txt
    |-- stdout.txt
    `-- stderr.txt
```

Harbor's job result remains the source of truth for verifier reward. The adapter
adds the BayesProbe epistemic trace and a normalized cross-arm artifact view.
Harbor-compatible trajectory output and BayesProbe's raw trace remain separate;
one must not be reconstructed from the other after the run.

## 16. Error Taxonomy and Retry Policy

Every failed run is classified as one of:

- `infrastructure_error`: Docker, image, dataset, or Harbor transport failure;
- `provider_error`: model endpoint, authentication, timeout, or malformed response;
- `plan_error`: structured probe generation or repair failure;
- `policy_error`: denied action or protected-path access;
- `execution_observation`: command, test, compile, or service failure returned by
  the task environment;
- `epistemic_failure`: observed Signals were interpreted or updated incorrectly;
- `task_failure`: a completed official verifier returned reward zero.

Only an `infrastructure_error` may be automatically retried once, and only when
zero agent actions were executed. A verifier reward of zero is never retried as
infrastructure. All repeated trials are preregistered in the lock file.

## 17. Validation Stages

### 17.1 Stage 0: harness bootstrap

Run the official Oracle on one fixed Terminal-Bench 2.0 task. This validates
Docker, task acquisition, environment startup, and verifier execution. Then
write the reproducibility lock.

The official documentation's canonical larger check can also be run separately:

```bash
harbor run -d terminal-bench/terminal-bench-2 -a oracle -l 5
```

### 17.2 Stage 1A: BayesProbe engineering vertical slice

Run BayesProbe once on the fixed task. It must execute at least one real action,
route the observation through the existing Signal/Evidence/Update path, and
reach the official verifier. This proves the downstream integration, not task
accuracy.

### 17.3 Stage 1B: paired one-task integration smoke

After Stage 1A passes, run BayesProbe and the minimal ReAct baseline once on the
same fixed task under the same locked controls. Both must execute at least one
real action and reach the official verifier.

### 17.4 Stage 2: three-task capability smoke

Run both agents once on the same three preregistered, architecture-neutral tasks:

1. `terminal-bench/break-filter-js-from-html`
2. `terminal-bench/cancel-async-tasks`
3. `terminal-bench/log-summary-date-ranges`

Their package refs are frozen in the benchmark source before the paired run.
The first task remains selected despite its already-observed BayesProbe smoke
reward of zero; it cannot be replaced based on that outcome.

The initial third candidate, `terminal-bench/build-cython-ext`, was rejected at
the Oracle qualification stage before either experimental arm ran. Its official
Oracle verifier failed an upstream `pyknotid` repository test (`1 failed, 17
passed`) despite the solution build completing. The replacement therefore
addresses task/toolchain health and is not based on Direct or BayesProbe reward.
These runs validate process behavior and failure attribution only. They are not
reported as representative benchmark accuracy.

Engineering readiness requires completed verifier runs. Before advancing to a
larger pilot, BayesProbe must additionally receive reward 1 on at least one of
the three tasks. A 0/3 result triggers failure analysis; it does not permit
replacing tasks or repeatedly tuning on them.

### 17.5 Later stages

After the MVP adapter is frozen, use a preregistered 20-30 task pilot. Repeated
`k=5` runs and a full benchmark are later decisions based on runtime, variance,
and cost measured in the pilot.

## 18. Test Strategy

### 18.1 Unit tests

- structured plan schema validation and one-repair limit;
- command and protected-path policy;
- signal truncation with full-output hashing;
- environment-state lineage after mutations;
- Signal and Evidence provenance requirements;
- provider secret redaction and artifact scans;
- budget accounting.

### 18.2 Contract tests

A fake Harbor environment will exercise `inspect`, `intervene`, and `verify`.
It must produce the same normalized Signals as a live environment for equivalent
action results.

### 18.3 Public-reuse tests

- benchmark production source imports BayesProbe only through `from bayesprobe
  import ...`;
- no benchmark production file is named `loop.py`, `core.py`, `evidence.py`, or
  `updater.py`;
- the runner factory returns the real public `AutonomousQuestionRunner` and
  `BayesProbeCore` types;
- the Harbor agent invokes `AutonomousQuestionRunner.run_question` exactly once
  in a worker thread;
- a real root-package runner completes the fake-environment vertical slice.

### 18.4 Concurrency tests

The async bridge must complete without deadlock, propagate cancellation, and
turn command timeout into a recorded Signal. The Harbor event loop must remain
responsive while the synchronous BayesProbe runner operates in its worker
thread.

### 18.5 Paradigm-conformance tests

- planner text cannot be admitted as Signal or Evidence;
- every posterior change traces to admitted Evidence and source Signals;
- an empty-signal cycle cannot directionally change posterior values;
- repeated verification in an unchanged environment is marked correlated;
- the complete `Belief -> Probe -> Signal -> Evidence -> Update` trace is
  present for every completed cycle.

### 18.6 Fairness tests

The BayesProbe and ReAct configurations must resolve to the same executor,
provider controls, action budget, model-call budget, and task timeout.

### 18.7 Live smoke tests

- Oracle receives reward 1 on the fixed bootstrap task;
- the first plan proves BayesProbe reaches the verifier on the fixed integration
  task;
- the follow-up plan proves ReAct reaches the same verifier under the same
  controls;
- the three fixed task identifiers cannot be replaced after outcomes are known.

## 19. Acceptance Criteria

Acceptance is staged so that the BayesProbe integration can be proven before a
second controller is introduced.

### 19.1 BayesProbe engineering vertical slice

The first implementation plan is accepted only when all of the following hold:

1. The benchmark implementation diff contains no path under `bayesprobe/` and
   no root `pyproject.toml` change.
2. An AST import guard proves benchmark production source uses only the
   top-level `bayesprobe` public interface.
3. No benchmark-owned autonomous-loop or kernel implementation file exists.
4. Existing root tests still pass.
5. Nested benchmark unit, contract, public-reuse, concurrency, and conformance
   tests pass.
6. The runner factory returns the installed public `AutonomousQuestionRunner`
   and the Harbor agent calls its `run_question` method exactly once.
7. Harbor can import the BayesProbe custom agent by its configured import path.
8. The official Oracle succeeds on the locked bootstrap task.
9. The real BayesProbe agent reaches the official verifier on the locked task.
10. Every BayesProbe cycle has a complete, provenance-linked epistemic trace.
11. No-signal cycles cannot directionally update posterior values.
12. No API key value appears in configuration, logs, traces, or committed files.
13. A completed Harbor verifier result is required for engineering readiness.

### 19.2 Paired comparison readiness

The follow-up ReAct plan is accepted only when Harbor can import both custom
agents, fairness tests prove identical provider controls and budgets, and both
agents reach the same official verifier on the same locked task. These checks do
not block completion of the first engineering vertical slice.

### 19.3 Larger-pilot gate

Before a 20-30 task pilot begins, both agents must complete the locked three-task
capability smoke and BayesProbe must obtain reward 1 on at least one of those
three tasks. This later gate is not an acceptance condition for the engineering
vertical slice.

## 20. Risks and Controls

| Risk | Control |
| --- | --- |
| Async/sync bridge deadlock | Worker-thread runner, captured loop, focused concurrency tests |
| Model fabricates observations | Only gateway results can create `TOOL_RESULT` Signals |
| Self-reinforcing posterior | Source-linked Evidence and the no-signal invariant |
| Hidden-test leakage | Protected-path policy and artifact review |
| Benchmark overfitting | Freeze task IDs after Oracle; prohibit outcome-based replacement |
| Unfair baseline | Shared executor, provider controls, budgets, timeout, and verifier |
| Secret leakage | Environment-only credentials, redaction, artifact secret scan |
| Local ARM incompatibility | Start with one-task smoke and classify image failures as infrastructure |
| Benchmark code reshapes core | Nested project and acceptance check for an empty core diff |

## 21. References

- [Run Terminal-Bench 2.0](https://www.tbench.ai/docs/run-terminal-bench-2-0)
- [Harbor agents](https://www.harborframework.com/docs/agents)
- [Harbor run evaluations](https://www.harborframework.com/docs/run-jobs/run-evals)
- [Harbor results and artifacts](https://www.harborframework.com/docs/run-jobs/results-and-artifacts)
- [Harbor core concepts](https://www.harborframework.com/docs/core-concepts)
- [Terminal-Bench 2.0 paper](https://arxiv.org/abs/2601.11868)
- [SWE-bench harness reference](https://www.swebench.com/SWE-bench/reference/harness/)
- [SWE-bench evaluation guide](https://www.swebench.com/SWE-bench/guides/evaluation/)
- [SWE-bench Verified](https://www.swebench.com/verified.html)
