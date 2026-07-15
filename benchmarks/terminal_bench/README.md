# Terminal-Bench Engineering Smoke and Paired Gate

This nested project runs one fixed Terminal-Bench 2.0 task through Harbor
0.18.0 and the public BayesProbe adapter. This slice is an engineering test of
the integration path; it is not accuracy evidence or a benchmark accuracy
claim.

After the one-task engineering smoke passes, the paired gate runs a minimal
Direct/ReAct control and BayesProbe on the same three frozen tasks. The gate is
still a capability check, not representative Terminal-Bench accuracy.

## Prerequisites

- Python 3.12 or newer and `uv`
- Harbor 0.18.0 from this nested project's lockfile
- A running Docker daemon with enough capacity for one Terminal-Bench task
- `BAYESPROBE_BENCH_MODEL`, `BAYESPROBE_BENCH_BASE_URL`, and
  `BAYESPROBE_BENCH_API_KEY` for lock creation and the BayesProbe smoke

The Oracle bootstrap itself does not call a model provider and does not require
provider credentials. Dataset or image acquisition may still require network
access when the fixed task is not already cached locally.

## Nested uv Setup

From the repository root:

```bash
uv sync --project benchmarks/terminal_bench --group dev
```

Then enter the nested project for every run command below:

```bash
cd benchmarks/terminal_bench
```

## Oracle Bootstrap

Run the official Oracle once on the fixed task:

```bash
HARBOR_TELEMETRY=off uv run harbor run -c configs/oracle-smoke.yaml
```

The Oracle must reach the official verifier with reward `1.0`. Do not select a
replacement task based on its outcome.

## Lock Creation

Export the provider settings that the later BayesProbe run will use. The key is
passed only to the exact-value leak scan and is never serialized.

```bash
export BAYESPROBE_BENCH_MODEL='<model>'
export BAYESPROBE_BENCH_BASE_URL='<openai-compatible-base-url>'
export BAYESPROBE_BENCH_API_KEY='<provider-key>'
```

Create the reproducibility lock from the latest Oracle job:

```bash
ORACLE_JOB_DIR="$(find .runs/harbor/oracle -mindepth 1 -maxdepth 1 -type d | sort | tail -1)"
uv run python scripts/write_benchmark_lock.py --oracle-job "$ORACLE_JOB_DIR" --output .runs/benchmark.lock.json
```

The writer requires Oracle reward `1.0`, Harbor 0.18.0, a clean committed
adapter tree, and a locally inspectable task image. Ignored `.runs/` content
does not make the adapter dirty. The writer records the current root HEAD,
adapter tree, exact resolved image reference, and inspected image digest, then
writes the lock atomically.

## BayesProbe Smoke

With the same three provider variables still exported, run exactly one real
BayesProbe trial:

```bash
HARBOR_TELEMETRY=off uv run harbor run -c configs/bayesprobe-smoke.yaml
BAYESPROBE_JOB_DIR="$(find .runs/harbor/bayesprobe -mindepth 1 -maxdepth 1 -type d | sort | tail -1)"
uv run python scripts/validate_smoke_run.py --job "$BAYESPROBE_JOB_DIR" --lock .runs/benchmark.lock.json
```

The run must use the lock produced above. Harbor invokes the official verifier
after the public `AutonomousQuestionRunner` returns.

Before classification, the validator rediscovers the exact task image from the
completed Harbor result and package cache, inspects its Docker digest, and
compares the image, root HEAD, and committed adapter tree to the lock. A stale
lock or dirty adapter fails validation even when the lock JSON is well formed.

## Artifact Locations

- `.runs/benchmark.lock.json`: immutable benchmark and runtime identity
- `.runs/harbor/oracle/<job>/`: official Oracle job and trial artifacts
- `.runs/harbor/bayesprobe/<job>/`: official BayesProbe job and trial artifacts
- `.runs/harbor/bayesprobe/<job>/<trial>/agent/bayesprobe/`: BayesProbe ledger,
  terminal actions, provider telemetry, errors, and summary
- `<trial>/verifier/`: official verifier logs and reward files

All `.runs` content is generated local state and must remain untracked.

## Residual Limitation

The current public BayesProbe core aborts an `active_only` cycle with zero
Signals before it can emit a completed integrated cycle. This benchmark slice
does not modify or work around that core behavior. Its validator nevertheless
accepts a future completed no-Signal/no-update cycle as unchanged, while still
rejecting orphan Signals, Evidence, and updates.

## Result Classifications

`validate_smoke_run.py` prints exactly one classification:

| Classification | Exit | Meaning |
| --- | ---: | --- |
| `engineering_pass` | 0 | Verifier completed and the BayesProbe trace is complete. |
| `task_failure` | 0 | Verifier completed with reward `0`; the engineering path still completed. |
| `infrastructure_error` | 1 | Harbor failed before the first agent action. |
| `provider_error` | 1 | The provider failed after agent startup. |
| `conformance_error` | 1 | A completed cycle lacks provenance-linked stages. |

Both zero-exit classifications prove only that the engineering path reached the
official verifier. Neither one establishes representative task accuracy.

## Paired Three-Task Gate

The gate freezes these tasks and package refs before the two experimental arms
run:

1. `terminal-bench/break-filter-js-from-html`
2. `terminal-bench/cancel-async-tasks`
3. `terminal-bench/build-cython-ext`

The first task remains in the gate even though the engineering smoke already
observed a BayesProbe reward of `0`. It must not be replaced after that outcome.

Run the official Oracle on all three tasks:

```bash
HARBOR_TELEMETRY=off uv run harbor run -c configs/oracle-gate.yaml
ORACLE_GATE_JOB=.runs/harbor/gate/oracle/bayesprobe-terminal-bench-oracle-gate
```

After the adapter changes are committed and the worktree is clean, freeze the
resolved tasks, image digests, repository identity, model, and shared budgets:

```bash
export BAYESPROBE_BENCH_MODEL='<model>'
export BAYESPROBE_BENCH_BASE_URL='<openai-compatible-base-url>'
uv run python scripts/write_paired_gate_lock.py \
  --oracle-job "$ORACLE_GATE_JOB" \
  --output .runs/paired-gate.lock.json
```

The Oracle must earn `1.0` on all three tasks or the writer refuses to create
the lock. Lock creation does not require the provider API key.

Export the provider key only for the two real arms, then run them serially:

```bash
export BAYESPROBE_BENCH_API_KEY='<provider-key>'
HARBOR_TELEMETRY=off uv run harbor run -c configs/direct-gate.yaml
HARBOR_TELEMETRY=off uv run harbor run -c configs/bayesprobe-gate.yaml
```

Validate both jobs against the same lock:

```bash
uv run python scripts/validate_paired_gate.py \
  --lock .runs/paired-gate.lock.json \
  --direct-job .runs/harbor/gate/direct/bayesprobe-terminal-bench-direct-gate \
  --bayesprobe-job .runs/harbor/gate/bayesprobe/bayesprobe-terminal-bench-bayesprobe-gate
```

The JSON report contains official per-task rewards, verifier completion,
BayesProbe trace completeness, terminal actions, logical model calls, token
usage, and wall time. The gate passes only when both arms reach all three
official verifiers, all BayesProbe traces conform, and BayesProbe earns reward
`1` on at least one task. A reward of `0` is never retried or reclassified as
infrastructure failure.

Both arms use the same OpenAI-compatible provider controls, low-level terminal
actions, action policy, `24` action budget, `72` logical model-call budget,
`120` second command timeout, `360` second provider timeout, and official task
verifier. Only the controller state differs.

## Secret Handling

Provider credentials belong only in `BAYESPROBE_BENCH_API_KEY`. Do not place a
key in YAML, the lock, command arguments, reports, or Git. The lock writer scans
its canonical serialized output for the resolved key before atomic replacement,
and the adapter redacts that exact value from generated artifacts.
