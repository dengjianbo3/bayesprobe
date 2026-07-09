from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bayesprobe.benchmark import BenchmarkHarness, BenchmarkSuiteResult
from bayesprobe.benchmark_io import (
    BenchmarkDataset,
    load_benchmark_dataset,
    write_benchmark_report,
)
from bayesprobe.experiment_artifacts import write_experiment_artifact_bundle
from bayesprobe.ledger import JsonlLedgerStore
from bayesprobe.model_gateway import (
    EvidenceJudgmentRepairPolicy,
    ModelGatewayConfig,
    build_model_gateway,
)


@dataclass(frozen=True)
class ExperimentRunConfig:
    dataset_path: str | Path
    report_path: str | Path
    ledger_path: str | Path | None = None
    artifact_dir: str | Path | None = None
    run_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    max_cycles: int = 1
    max_probes_per_cycle: int = 1
    model_gateway: ModelGatewayConfig | Mapping[str, Any] | None = None
    judgment_repair_policy: EvidenceJudgmentRepairPolicy | Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.max_cycles < 1:
            raise ValueError("max_cycles must be at least 1")
        if self.max_probes_per_cycle < 1:
            raise ValueError("max_probes_per_cycle must be at least 1")
        if self.artifact_dir is not None and not isinstance(
            self.artifact_dir, (str, Path)
        ):
            raise ValueError("artifact_dir must be a path")
        if self.run_name is not None:
            if not isinstance(self.run_name, str):
                raise ValueError("run_name must be a string")
            if not self.run_name.strip():
                raise ValueError("run_name must not be empty")
            object.__setattr__(self, "run_name", self.run_name.strip())
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be an object")
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class ExperimentRunResult:
    dataset: BenchmarkDataset
    suite_result: BenchmarkSuiteResult
    report_path: Path
    ledger_path: Path | None = None
    artifact_dir: Path | None = None
    artifact_manifest_path: Path | None = None


def run_benchmark_experiment(config: ExperimentRunConfig) -> ExperimentRunResult:
    dataset = load_benchmark_dataset(config.dataset_path)
    artifact_dir = Path(config.artifact_dir) if config.artifact_dir is not None else None
    ledger_path = Path(config.ledger_path) if config.ledger_path is not None else None
    if ledger_path is None and artifact_dir is not None:
        ledger_path = artifact_dir / "ledger.jsonl"
    ledger = JsonlLedgerStore(ledger_path) if ledger_path is not None else None
    model_gateway = build_model_gateway(config.model_gateway)
    judgment_repair_policy = EvidenceJudgmentRepairPolicy.from_config(
        config.judgment_repair_policy
    )
    harness = BenchmarkHarness(
        ledger=ledger,
        model_gateway=model_gateway,
        judgment_repair_policy=judgment_repair_policy,
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
    artifact_manifest_path = None
    if artifact_dir is not None:
        artifact_bundle = write_experiment_artifact_bundle(
            artifact_dir=artifact_dir,
            config=config,
            dataset=dataset,
            report_path=report_path,
            ledger_path=ledger_path,
            sample_count=suite_result.sample_count,
        )
        artifact_manifest_path = artifact_bundle.manifest_path
    return ExperimentRunResult(
        dataset=dataset,
        suite_result=suite_result,
        report_path=report_path,
        ledger_path=ledger_path,
        artifact_dir=artifact_dir,
        artifact_manifest_path=artifact_manifest_path,
    )


__all__ = [
    "ExperimentRunConfig",
    "ExperimentRunResult",
    "run_benchmark_experiment",
]
