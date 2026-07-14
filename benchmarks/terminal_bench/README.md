# Terminal-Bench Engineering Smoke

This nested project runs one fixed Terminal-Bench 2.0 task through Harbor
0.18.0 and the public BayesProbe adapter. This slice is an engineering test of
the integration path; it is not accuracy evidence or a benchmark accuracy
claim.

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

The writer requires Oracle reward `1.0`, Harbor 0.18.0, a committed adapter
tree, and a locally inspectable task image. It writes the lock atomically.

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

## Artifact Locations

- `.runs/benchmark.lock.json`: immutable benchmark and runtime identity
- `.runs/harbor/oracle/<job>/`: official Oracle job and trial artifacts
- `.runs/harbor/bayesprobe/<job>/`: official BayesProbe job and trial artifacts
- `.runs/harbor/bayesprobe/<job>/<trial>/agent/bayesprobe/`: BayesProbe ledger,
  terminal actions, provider telemetry, errors, and summary
- `<trial>/verifier/`: official verifier logs and reward files

All `.runs` content is generated local state and must remain untracked.

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

## Secret Handling

Provider credentials belong only in `BAYESPROBE_BENCH_API_KEY`. Do not place a
key in YAML, the lock, command arguments, reports, or Git. The lock writer scans
its canonical serialized output for the resolved key before atomic replacement,
and the adapter redacts that exact value from generated artifacts.
