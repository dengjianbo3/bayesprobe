# Experiment Artifact Packaging MVP Design

Date: 2026-07-09

## Context

BayesProbe already has a working offline benchmark runner, JSONL ledger support,
schema repair policy, and an OpenAI Responses `ModelGateway` adapter. The next
engineering gap is not another control-flow change; it is reproducible experiment
packaging. Provider-backed benchmarks, prompt version tracking, and later
multi-agent traces all need a stable run artifact directory.

This MVP implements the first Phase 5 capability from `docs/ARCHITECTURE.md`:
`report + ledger + config + prompt versions + dataset snapshot`, with prompt
version fields represented as metadata hooks until the prompt registry is
expanded.

## Goals

- Add an optional `artifact_dir` to experiment configuration.
- Preserve the current BayesProbe execution semantics.
- Write a stable experiment artifact bundle after a benchmark run.
- Capture enough metadata to replay, compare, and audit a run.
- Keep provider secrets out of all artifact files.

## Non-Goals

- No changes to `BayesProbeCore`, evidence integration, posterior updates, or
  probe control flow.
- No provider registry redesign.
- No SQLite persistence adapter.
- No benchmark scoring redesign.
- No live OpenAI calls in the packaging tests.

## Proposed API

`ExperimentRunConfig` gains:

- `artifact_dir: str | Path | None = None`
- `run_name: str | None = None`
- `metadata: Mapping[str, Any] | None = None`

`ExperimentRunResult` gains:

- `artifact_dir: Path | None = None`
- `artifact_manifest_path: Path | None = None`

The JSON config loader resolves `artifact_dir` relative to the config file, just
like `dataset_path`, `report_path`, and `ledger_path`.

## Artifact Directory Semantics

For v0.1, `artifact_dir` is treated as the exact run directory. The runner will
create it if needed.

If `artifact_dir` is not configured, behavior remains unchanged.

If `artifact_dir` is configured:

- the normal configured `report_path` is still written;
- the configured `ledger_path`, if any, is still used;
- a complete artifact bundle is written into `artifact_dir`;
- if no `ledger_path` is configured, the runner uses
  `artifact_dir / "ledger.jsonl"` so the artifact bundle can include a ledger by
  default.

This keeps backward compatibility while making new experiment configs simpler.

## Files Written

The artifact bundle contains:

- `manifest.json`
- `report.json`
- `ledger.jsonl`
- `config_snapshot.json`
- `dataset_snapshot.json`

`report.json` is a copy of the benchmark report object written to
`report_path`. `dataset_snapshot.json` preserves the loaded dataset payload in a
stable JSON representation. `config_snapshot.json` records a sanitized
experiment configuration. `manifest.json` records paths, counts, metadata, and a
sanitized model gateway summary.

## Manifest Shape

`manifest.json` includes:

- `artifact_version`
- `created_at_utc`
- `run_name`
- `dataset_name`
- `sample_count`
- `report_path`
- `ledger_path`
- `config_snapshot_path`
- `dataset_snapshot_path`
- `model_gateway`
- `judgment_repair_policy`
- `metadata`

`model_gateway` includes provider configuration required to understand the run,
such as `kind`, `model`, `api_key_env`, `timeout_seconds`, and
`max_output_tokens`. It never includes an API key value.

## Error Handling

- Invalid `artifact_dir`, `run_name`, or `metadata` values fail during config
  parsing or `ExperimentRunConfig` construction.
- Artifact write failures propagate as normal `OSError` failures through the CLI.
- Existing report and ledger behavior remains unchanged when no artifact
  directory is configured.

## Testing

Tests will be written first.

Coverage:

- config parsing resolves `artifact_dir` and validates `metadata`;
- experiment run writes the full artifact bundle;
- no API-key-shaped value or forbidden secret field is written into artifacts;
- when `ledger_path` is omitted and `artifact_dir` is present, the runner writes
  `artifact_dir / "ledger.jsonl"`;
- CLI summary includes `artifact=...` only when artifacts are enabled;
- existing experiment runner behavior remains green without `artifact_dir`.

## Follow-Up Work

- Prompt registry version records.
- Dataset split and sample filters.
- Provider-backed recorded benchmark fixtures.
- Optional SQLite `LedgerStore`.
- Multi-agent projection and passive-signal collaboration artifacts.
