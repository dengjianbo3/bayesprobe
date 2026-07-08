from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bayesprobe.benchmark import BenchmarkHarness, BenchmarkSuiteResult
from bayesprobe.benchmark_io import (
    BenchmarkDataset,
    load_benchmark_dataset,
    write_benchmark_report,
)
from bayesprobe.ledger import JsonlLedgerStore
from bayesprobe.model_gateway import ModelGatewayConfig, build_model_gateway


@dataclass(frozen=True)
class ExperimentRunConfig:
    dataset_path: str | Path
    report_path: str | Path
    ledger_path: str | Path | None = None
    max_cycles: int = 1
    max_probes_per_cycle: int = 1
    model_gateway: ModelGatewayConfig | Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.max_cycles < 1:
            raise ValueError("max_cycles must be at least 1")
        if self.max_probes_per_cycle < 1:
            raise ValueError("max_probes_per_cycle must be at least 1")


@dataclass(frozen=True)
class ExperimentRunResult:
    dataset: BenchmarkDataset
    suite_result: BenchmarkSuiteResult
    report_path: Path
    ledger_path: Path | None = None


def run_benchmark_experiment(config: ExperimentRunConfig) -> ExperimentRunResult:
    dataset = load_benchmark_dataset(config.dataset_path)
    ledger_path = Path(config.ledger_path) if config.ledger_path is not None else None
    ledger = JsonlLedgerStore(ledger_path) if ledger_path is not None else None
    model_gateway = build_model_gateway(config.model_gateway)
    harness = BenchmarkHarness(
        ledger=ledger,
        model_gateway=model_gateway,
        max_cycles=config.max_cycles,
        max_probes_per_cycle=config.max_probes_per_cycle,
    )
    suite_result = harness.run_suite(dataset.samples)
    report_path = Path(config.report_path)
    write_benchmark_report(
        report_path,
        suite_result,
        dataset_name=dataset.dataset_name,
        metadata=dataset.metadata,
    )
    return ExperimentRunResult(
        dataset=dataset,
        suite_result=suite_result,
        report_path=report_path,
        ledger_path=ledger_path,
    )


__all__ = [
    "ExperimentRunConfig",
    "ExperimentRunResult",
    "run_benchmark_experiment",
]
