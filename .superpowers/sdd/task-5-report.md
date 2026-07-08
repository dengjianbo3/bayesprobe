# Task 5 Report: Benchmark And Experiment Configuration Chain

## Scope

Implemented Task 5 exactly on the owned file set:

- `bayesprobe/benchmark.py`
- `bayesprobe/experiment_runner.py`
- `bayesprobe/config.py`
- `tests/test_benchmark_harness.py`
- `tests/test_experiment_runner.py`
- `tests/test_public_api_and_config.py`

No changes were made to evidence-gate internals or core behavior beyond wiring the existing `EvidenceJudgmentRepairPolicy` through the benchmark and experiment configuration chain.

## TDD Log

### RED

Added the exact failing tests/snippets from the task brief:

1. Benchmark harness propagation test:
   - `test_benchmark_harness_passes_judgment_repair_policy_to_created_core`
2. Experiment runner config propagation test:
   - `test_run_benchmark_experiment_uses_judgment_repair_policy_config`
3. JSON config parsing test:
   - `test_experiment_config_from_mapping_parses_judgment_repair_policy`
4. Invalid config cases for `judgment_repair_policy`:
   - non-object policy
   - non-integer `max_attempts`
   - negative `max_attempts`
   - empty `repair_task`

Ran the required RED verification command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_benchmark_harness.py::test_benchmark_harness_passes_judgment_repair_policy_to_created_core tests/test_experiment_runner.py::test_run_benchmark_experiment_uses_judgment_repair_policy_config tests/test_public_api_and_config.py::test_experiment_config_from_mapping_parses_judgment_repair_policy -q -p no:cacheprovider
```

Observed expected failures:

- `BenchmarkHarness.__init__()` did not accept `judgment_repair_policy`
- `ExperimentRunConfig` did not accept `judgment_repair_policy`
- parsed config had no `judgment_repair_policy` attribute

### GREEN

Implemented the exact configuration-chain wiring from the brief:

#### `bayesprobe/benchmark.py`

- Imported `EvidenceJudgmentRepairPolicy`
- Extended `BenchmarkHarness.__init__` with:
  - `judgment_repair_policy: EvidenceJudgmentRepairPolicy | None = None`
- Passed `judgment_repair_policy` through when constructing `BayesProbeCore`

#### `bayesprobe/experiment_runner.py`

- Imported `EvidenceJudgmentRepairPolicy`
- Extended `ExperimentRunConfig` with:
  - `judgment_repair_policy: EvidenceJudgmentRepairPolicy | Mapping[str, Any] | None = None`
- In `run_benchmark_experiment(...)`:
  - normalized via `EvidenceJudgmentRepairPolicy.from_config(...)`
  - passed the resulting policy into `BenchmarkHarness(...)`

#### `bayesprobe/config.py`

- Imported `EvidenceJudgmentRepairPolicy`
- Extended `experiment_config_from_mapping(...)` to populate:
  - `judgment_repair_policy=_optional_judgment_repair_policy(data)`
- Added `_optional_judgment_repair_policy(...)`
  - returns `None` when omitted/null
  - rejects non-object JSON values
  - delegates validation to `EvidenceJudgmentRepairPolicy.from_config(...)`

## Verification

Ran the required GREEN verification command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_benchmark_harness.py tests/test_experiment_runner.py tests/test_public_api_and_config.py -q -p no:cacheprovider
```

Result:

- `38 passed in 0.11s`

This verifies:

- benchmark-created cores receive the repair policy
- experiment-run config accepts mapping policy config
- JSON config parsing produces a validated `EvidenceJudgmentRepairPolicy`
- invalid public config values raise the expected errors

## Commit

Created commit:

- `feat: configure judgment repair policy`

## Notes

- Kept changes scoped to the requested files only.
- Did not alter evidence judgment repair logic, repair attempts, or evidence/core internals.
