# Public SDK and Config Layer Design

Date: 2026-07-08
Status: Approved from external integration requirement

## Goal

Make BayesProbe usable by external Python code through a stable public package entrypoint and a JSON configuration loader for benchmark experiments.

The current project can be imported through internal modules, but `bayesprobe.__init__` does not expose a clear supported surface and experiment configuration must be constructed manually in code. This slice creates the first SDK boundary before adding a CLI.

## Scope

The first version covers:

- A public package facade in `bayesprobe/__init__.py`.
- A JSON experiment config loader in `bayesprobe/config.py`.
- Tests proving external code can import supported APIs from `bayesprobe`.
- Tests proving JSON config can drive `run_benchmark_experiment`.

## Non-Goals

- No CLI in this slice.
- No YAML/TOML config support.
- No environment variable interpolation.
- No plugin system or dynamic component registry.
- No changes to BayesProbe core control flow, evidence integration, benchmark scoring, or experiment runner behavior.

## Public API

Expose these names from `bayesprobe`:

- `BenchmarkDataset`
- `BenchmarkHarness`
- `BenchmarkSample`
- `BenchmarkSampleResult`
- `BenchmarkSignal`
- `BenchmarkSignalShape`
- `BenchmarkSuiteResult`
- `ExperimentRunConfig`
- `ExperimentRunResult`
- `load_benchmark_dataset`
- `load_experiment_config`
- `run_benchmark_experiment`
- `write_benchmark_report`

This is the supported integration surface for the current MVP. Internal modules remain importable as Python modules, but external projects should use the package facade where possible.

## Config API

Create `bayesprobe/config.py` with:

- `load_experiment_config(path: str | Path) -> ExperimentRunConfig`
- `experiment_config_from_mapping(data: Mapping[str, Any], *, base_dir: str | Path | None = None) -> ExperimentRunConfig`

JSON config shape:

```json
{
  "dataset_path": "fixtures/benchmarks/toy_belief_revision.json",
  "report_path": "outputs/toy-report.json",
  "ledger_path": "outputs/toy-ledger.jsonl",
  "max_cycles": 1,
  "max_probes_per_cycle": 1
}
```

Required fields:

- `dataset_path`
- `report_path`

Optional fields:

- `ledger_path`
- `max_cycles`
- `max_probes_per_cycle`

## Path Resolution

`load_experiment_config(path)` resolves relative paths inside the JSON file against the config file's parent directory. Absolute paths remain absolute.

`experiment_config_from_mapping(data, base_dir=None)` resolves relative paths against `base_dir` when provided. If `base_dir` is `None`, relative paths remain relative.

This lets external projects keep portable experiment configs next to their datasets and outputs.

## Validation

The config loader raises `ValueError` when:

- The config path does not end with `.json`.
- JSON parsing fails.
- The top-level JSON value is not an object.
- `dataset_path` or `report_path` is missing.
- Any provided path field is not a string.
- `max_cycles` or `max_probes_per_cycle` is not an integer.
- Existing `ExperimentRunConfig` validation rejects numeric values.

## Data Flow

```text
external code
→ from bayesprobe import load_experiment_config, run_benchmark_experiment
→ load_experiment_config("experiment.json")
→ ExperimentRunConfig
→ run_benchmark_experiment(config)
→ ExperimentRunResult + report/ledger files
```

## Test Strategy

Add `tests/test_public_api_and_config.py` covering:

- Importing the public SDK names from `bayesprobe`.
- Loading JSON config with paths resolved relative to the config file.
- Running the loaded config through `run_benchmark_experiment`.
- Rejecting unsupported extensions, malformed JSON, missing required paths, non-string paths, and non-integer numeric fields.

Run focused tests first, then the full pytest suite.
