from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from bayesprobe import load_experiment_config, run_benchmark_experiment
from bayesprobe.experiment_runner import ExperimentRunResult


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as error:
        return int(error.code)
    if args.command == "run":
        return _run_command(args)
    if args.command == "eval":
        from bayesprobe.evaluation.cli import run_eval_command

        return run_eval_command(args)
    parser.print_help(sys.stderr)
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bayesprobe")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument(
        "--config",
        required=True,
        help="Path to a BayesProbe experiment JSON config.",
    )
    from bayesprobe.evaluation.cli import add_eval_subparser

    add_eval_subparser(subparsers)
    return parser


def _run_command(args: argparse.Namespace) -> int:
    try:
        config = load_experiment_config(args.config)
        result = run_benchmark_experiment(config)
    except (ValueError, OSError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(_format_summary(result))
    return 0


def _format_summary(result: ExperimentRunResult) -> str:
    suite = result.suite_result
    parts = [
        "BayesProbe experiment complete:",
        f"dataset={result.dataset.dataset_name}",
        f"samples={suite.sample_count}",
        f"final_accuracy={suite.final_accuracy}",
        f"update_direction_accuracy={suite.update_direction_accuracy}",
        f"report={result.report_path}",
        f"ledger={result.ledger_path}",
    ]
    if result.artifact_dir is not None:
        parts.append(f"artifact={result.artifact_dir}")
    return " ".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
