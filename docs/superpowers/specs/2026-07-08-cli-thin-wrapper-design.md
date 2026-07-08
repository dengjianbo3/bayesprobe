# CLI Thin Wrapper Design

Date: 2026-07-08
Status: Approved as next step after public SDK and config layer

## Goal

Add a minimal command-line entrypoint for running BayesProbe benchmark experiments from JSON config files.

The CLI must stay a thin wrapper over the public SDK:

```text
load_experiment_config(config_path)
→ run_benchmark_experiment(config)
→ print concise summary
```

It must not duplicate experiment configuration, benchmark execution, report writing, or BayesProbe control-flow logic.

## Scope

The first CLI version supports:

- `bayesprobe run --config <experiment.json>`
- `python -m bayesprobe.cli run --config <experiment.json>`
- a package script entry in `pyproject.toml`
- one-line success summary on stdout
- validation/runtime errors on stderr
- deterministic exit codes

## Non-Goals

- No CLI overrides for dataset/report/ledger paths.
- No JSON output mode.
- No interactive prompts.
- No subcommands other than `run`.
- No shell completion.
- No changes to `bayesprobe.config`, `bayesprobe.experiment_runner`, benchmark scoring, evidence integration, or core control flow.

## Public API

Create `bayesprobe/cli.py` with:

- `main(argv: Sequence[str] | None = None) -> int`

The module should also support direct module execution:

```python
if __name__ == "__main__":
    raise SystemExit(main())
```

## Command Behavior

Success:

```bash
bayesprobe run --config experiment.json
```

Prints:

```text
BayesProbe experiment complete: dataset=toy_belief_revision samples=3 final_accuracy=1.0 update_direction_accuracy=1.0 report=/path/to/report.json ledger=/path/to/ledger.jsonl
```

If no ledger path is configured, print `ledger=None`.

## Exit Codes

- `0`: experiment completed and report was written.
- `1`: config loading or experiment execution raised an expected exception.
- `2`: CLI usage error from argument parsing.

Expected exceptions are `ValueError`, `OSError`, and `FileNotFoundError`.

## Packaging

Add to `pyproject.toml`:

```toml
[project.scripts]
bayesprobe = "bayesprobe.cli:main"
```

## Test Strategy

Add `tests/test_cli.py` covering:

- `main(["run", "--config", config_path])` writes report/ledger and prints the summary.
- invalid config file returns `1` and prints an error.
- missing required CLI arguments return `2`.
- `python -m bayesprobe.cli run --config config_path` executes successfully.

Run focused CLI tests first, then the full pytest suite.
