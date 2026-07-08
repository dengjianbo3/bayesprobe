from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from bayesprobe.benchmark import (
    BenchmarkSample,
    BenchmarkSampleResult,
    BenchmarkSignal,
    BenchmarkSignalShape,
    BenchmarkSuiteResult,
)


@dataclass(frozen=True)
class BenchmarkDataset:
    dataset_name: str
    samples: list[BenchmarkSample]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.dataset_name.strip():
            raise ValueError("dataset_name must not be empty")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be an object")


def load_benchmark_dataset(path: str | Path) -> BenchmarkDataset:
    dataset_path = Path(path)
    suffix = dataset_path.suffix.lower()
    if suffix == ".json":
        return _load_json_dataset(dataset_path)
    if suffix == ".jsonl":
        return _load_jsonl_dataset(dataset_path)
    raise ValueError("benchmark dataset path must end with .json or .jsonl")


def write_benchmark_report(
    path: str | Path,
    suite_result: BenchmarkSuiteResult,
    *,
    dataset_name: str = "",
    metadata: dict[str, Any] | None = None,
) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_name": dataset_name or report_path.stem,
        "metadata": dict(metadata or {}),
        "sample_count": suite_result.sample_count,
        "final_accuracy": suite_result.final_accuracy,
        "update_direction_accuracy": suite_result.update_direction_accuracy,
        "results": [_sample_result_to_dict(result) for result in suite_result.results],
    }
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_json_dataset(path: Path) -> BenchmarkDataset:
    payload = _load_json(path)
    if isinstance(payload, Mapping):
        if "samples" not in payload:
            raise ValueError("JSON object dataset must include samples")
        samples_payload = payload["samples"]
        if not isinstance(samples_payload, list):
            raise ValueError("JSON object dataset samples must be an array")
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("benchmark dataset metadata must be an object")
        dataset_name = payload.get("dataset_name") or path.stem
        if not isinstance(dataset_name, str):
            raise ValueError("benchmark dataset_name must be a string")
        return BenchmarkDataset(
            dataset_name=dataset_name,
            samples=_samples_from_payload(samples_payload),
            metadata=dict(metadata),
        )
    if isinstance(payload, list):
        return BenchmarkDataset(
            dataset_name=path.stem,
            samples=_samples_from_payload(payload),
        )
    raise ValueError("benchmark JSON dataset must be an object or array")


def _load_jsonl_dataset(path: Path) -> BenchmarkDataset:
    samples = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except JSONDecodeError as error:
            raise ValueError(
                f"could not parse benchmark dataset JSONL line {line_number}"
            ) from error
        samples.append(_sample_from_payload(payload))
    return BenchmarkDataset(dataset_name=path.stem, samples=samples)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except JSONDecodeError as error:
        raise ValueError("could not parse benchmark dataset JSON") from error


def _samples_from_payload(payload: list[Any]) -> list[BenchmarkSample]:
    return [_sample_from_payload(sample_payload) for sample_payload in payload]


def _sample_from_payload(payload: Any) -> BenchmarkSample:
    if not isinstance(payload, Mapping):
        raise ValueError("benchmark sample entry must be an object")
    return _sample_from_mapping(payload)


def _sample_from_mapping(data: Mapping[str, Any]) -> BenchmarkSample:
    try:
        passive_signals = [
            _signal_from_payload(signal_payload)
            for signal_payload in data.get("passive_signals", [])
        ]
        return BenchmarkSample(
            sample_id=data["sample_id"],
            question_or_claim=data["question_or_claim"],
            signal_shape=data.get("signal_shape", BenchmarkSignalShape.ACTIVE_ONLY),
            gold_best_hypothesis=data["gold_best_hypothesis"],
            passive_signals=passive_signals,
            gold_update_directions=dict(data.get("gold_update_directions", {})),
            initial_context=data.get("initial_context", ""),
        )
    except KeyError as error:
        raise ValueError(f"missing required benchmark sample field: {error.args[0]}") from error


def _signal_from_payload(payload: Any) -> BenchmarkSignal:
    if not isinstance(payload, Mapping):
        raise ValueError("benchmark signal entry must be an object")
    try:
        return BenchmarkSignal(
            signal_id=payload["signal_id"],
            source_type=payload["source_type"],
            source=payload["source"],
            raw_content=payload["raw_content"],
            target_hypotheses=list(payload.get("target_hypotheses", [])),
        )
    except KeyError as error:
        raise ValueError(f"missing required benchmark signal field: {error.args[0]}") from error


def _sample_result_to_dict(result: BenchmarkSampleResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["signal_shape"] = result.signal_shape.value
    return payload


__all__ = [
    "BenchmarkDataset",
    "load_benchmark_dataset",
    "write_benchmark_report",
]
