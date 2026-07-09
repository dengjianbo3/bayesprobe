from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bayesprobe.benchmark_io import BenchmarkDataset
from bayesprobe.model_gateway import EvidenceJudgmentRepairPolicy, ModelGatewayConfig


_SECRET_METADATA_KEYS = {"api_key", "openai_api_key", "token", "secret"}


@dataclass(frozen=True)
class ExperimentArtifactBundle:
    artifact_dir: Path
    manifest_path: Path
    report_path: Path
    ledger_path: Path | None
    config_snapshot_path: Path
    dataset_snapshot_path: Path


def write_experiment_artifact_bundle(
    *,
    artifact_dir: str | Path,
    config: Any,
    dataset: BenchmarkDataset,
    report_path: str | Path,
    ledger_path: str | Path | None,
    sample_count: int,
    created_at_utc: datetime | None = None,
) -> ExperimentArtifactBundle:
    target_dir = Path(artifact_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    artifact_report_path = target_dir / "report.json"
    artifact_ledger_path = target_dir / "ledger.jsonl" if ledger_path is not None else None
    config_snapshot_path = target_dir / "config_snapshot.json"
    dataset_snapshot_path = target_dir / "dataset_snapshot.json"
    manifest_path = target_dir / "manifest.json"

    _copy_json_file(Path(report_path), artifact_report_path)
    if ledger_path is not None:
        _copy_text_file(Path(ledger_path), artifact_ledger_path)

    _write_json(config_snapshot_path, _config_snapshot(config, ledger_path=ledger_path))
    _write_json(dataset_snapshot_path, _dataset_snapshot(dataset))
    _write_json(
        manifest_path,
        _manifest_payload(
            config=config,
            dataset=dataset,
            sample_count=sample_count,
            artifact_dir=target_dir,
            report_path=artifact_report_path,
            ledger_path=artifact_ledger_path,
            config_snapshot_path=config_snapshot_path,
            dataset_snapshot_path=dataset_snapshot_path,
            created_at_utc=created_at_utc,
        ),
    )

    return ExperimentArtifactBundle(
        artifact_dir=target_dir,
        manifest_path=manifest_path,
        report_path=artifact_report_path,
        ledger_path=artifact_ledger_path,
        config_snapshot_path=config_snapshot_path,
        dataset_snapshot_path=dataset_snapshot_path,
    )


def _copy_json_file(source: Path, destination: Path) -> None:
    payload = json.loads(source.read_text(encoding="utf-8"))
    _write_json(destination, payload)


def _copy_text_file(source: Path, destination: Path | None) -> None:
    if destination is None:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == destination.resolve():
        if not source.exists():
            destination.touch()
        return
    shutil.copyfile(source, destination)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _dataset_snapshot(dataset: BenchmarkDataset) -> dict[str, Any]:
    samples = []
    for sample in dataset.samples:
        sample_payload = asdict(sample)
        sample_payload["signal_shape"] = sample.signal_shape.value
        samples.append(sample_payload)
    return {
        "dataset_name": dataset.dataset_name,
        "metadata": dict(dataset.metadata),
        "samples": samples,
    }


def _config_snapshot(config: Any, *, ledger_path: str | Path | None) -> dict[str, Any]:
    return {
        "dataset_path": str(Path(config.dataset_path)),
        "report_path": str(Path(config.report_path)),
        "ledger_path": str(Path(ledger_path)) if ledger_path is not None else None,
        "artifact_dir": str(Path(config.artifact_dir)) if config.artifact_dir is not None else None,
        "run_name": config.run_name,
        "metadata": _sanitize_metadata(config.metadata),
        "max_cycles": config.max_cycles,
        "max_probes_per_cycle": config.max_probes_per_cycle,
        "model_gateway": _model_gateway_snapshot(config.model_gateway),
        "judgment_repair_policy": _repair_policy_snapshot(config.judgment_repair_policy),
    }


def _manifest_payload(
    *,
    config: Any,
    dataset: BenchmarkDataset,
    sample_count: int,
    artifact_dir: Path,
    report_path: Path,
    ledger_path: Path | None,
    config_snapshot_path: Path,
    dataset_snapshot_path: Path,
    created_at_utc: datetime | None,
) -> dict[str, Any]:
    created_at = created_at_utc or datetime.now(UTC)
    return {
        "artifact_version": "0.1",
        "created_at_utc": created_at.isoformat().replace("+00:00", "Z"),
        "run_name": config.run_name,
        "artifact_dir": str(artifact_dir),
        "dataset_name": dataset.dataset_name,
        "sample_count": sample_count,
        "report_path": str(report_path),
        "ledger_path": str(ledger_path) if ledger_path is not None else None,
        "config_snapshot_path": str(config_snapshot_path),
        "dataset_snapshot_path": str(dataset_snapshot_path),
        "metadata": _sanitize_metadata(config.metadata),
        "model_gateway": _model_gateway_snapshot(config.model_gateway),
        "judgment_repair_policy": _repair_policy_snapshot(config.judgment_repair_policy),
    }


def _model_gateway_snapshot(config: Any) -> dict[str, Any]:
    if config is None:
        return {"kind": "deterministic"}
    if isinstance(config, ModelGatewayConfig):
        payload = {
            "kind": config.kind,
            "model": config.model,
            "api_key_env": config.api_key_env,
            "timeout_seconds": config.timeout_seconds,
            "max_output_tokens": config.max_output_tokens,
        }
        if config.responses is not None:
            payload["scripted_response_tasks"] = sorted(config.responses)
        return {key: value for key, value in payload.items() if value is not None}
    if isinstance(config, Mapping):
        payload = {
            "kind": str(config.get("kind", "deterministic")),
            "model": config.get("model"),
            "api_key_env": config.get("api_key_env"),
            "timeout_seconds": config.get("timeout_seconds"),
            "max_output_tokens": config.get("max_output_tokens"),
        }
        responses = config.get("responses")
        if isinstance(responses, Mapping):
            payload["scripted_response_tasks"] = sorted(str(task) for task in responses)
        return {key: value for key, value in payload.items() if value is not None}
    return {"kind": type(config).__name__}


def _repair_policy_snapshot(config: Any) -> dict[str, Any]:
    if config is None:
        policy = EvidenceJudgmentRepairPolicy()
    elif isinstance(config, EvidenceJudgmentRepairPolicy):
        policy = config
    elif isinstance(config, Mapping):
        policy = EvidenceJudgmentRepairPolicy.from_config(config)
    else:
        return {"kind": type(config).__name__}
    return {
        "max_attempts": policy.max_attempts,
        "repair_task": policy.repair_task,
    }


def _sanitize_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if _is_secret_metadata_key(key_text):
                continue
            sanitized[key_text] = _sanitize_metadata(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_metadata(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_metadata(item) for item in value]
    return value


def _is_secret_metadata_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(
        normalized == secret_key
        or normalized.endswith(f"_{secret_key}")
        or normalized.endswith(secret_key)
        for secret_key in _SECRET_METADATA_KEYS
    )


__all__ = [
    "ExperimentArtifactBundle",
    "write_experiment_artifact_bundle",
]
