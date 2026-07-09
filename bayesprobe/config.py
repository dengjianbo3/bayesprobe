from __future__ import annotations

import json
from collections.abc import Mapping
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from bayesprobe.experiment_runner import ExperimentRunConfig
from bayesprobe.model_gateway import EvidenceJudgmentRepairPolicy, ModelGatewayConfig


def load_experiment_config(path: str | Path) -> ExperimentRunConfig:
    config_path = Path(path)
    if config_path.suffix.lower() != ".json":
        raise ValueError("experiment config path must end with .json")
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except JSONDecodeError as error:
        raise ValueError("could not parse experiment config JSON") from error
    if not isinstance(payload, Mapping):
        raise ValueError("experiment config must be a JSON object")
    return experiment_config_from_mapping(payload, base_dir=config_path.parent)


def experiment_config_from_mapping(
    data: Mapping[str, Any],
    *,
    base_dir: str | Path | None = None,
) -> ExperimentRunConfig:
    return ExperimentRunConfig(
        dataset_path=_required_path(data, "dataset_path", base_dir=base_dir),
        report_path=_required_path(data, "report_path", base_dir=base_dir),
        ledger_path=_optional_path(data, "ledger_path", base_dir=base_dir),
        max_cycles=_optional_int(data, "max_cycles", default=1),
        max_probes_per_cycle=_optional_int(data, "max_probes_per_cycle", default=1),
        model_gateway=_optional_model_gateway_config(data),
        judgment_repair_policy=_optional_judgment_repair_policy(data),
    )


def _required_path(
    data: Mapping[str, Any],
    field_name: str,
    *,
    base_dir: str | Path | None,
) -> Path:
    if field_name not in data:
        raise ValueError(f"missing required experiment config field: {field_name}")
    value = data[field_name]
    if not isinstance(value, str):
        raise ValueError(f"experiment config field {field_name} must be a string")
    return _resolve_path(value, base_dir=base_dir)


def _optional_path(
    data: Mapping[str, Any],
    field_name: str,
    *,
    base_dir: str | Path | None,
) -> Path | None:
    if field_name not in data or data[field_name] is None:
        return None
    value = data[field_name]
    if not isinstance(value, str):
        raise ValueError(f"experiment config field {field_name} must be a string")
    return _resolve_path(value, base_dir=base_dir)


def _resolve_path(value: str, *, base_dir: str | Path | None) -> Path:
    path = Path(value)
    if path.is_absolute() or base_dir is None:
        return path
    return Path(base_dir) / path


def _optional_int(data: Mapping[str, Any], field_name: str, *, default: int) -> int:
    if field_name not in data:
        return default
    value = data[field_name]
    if type(value) is not int:
        raise ValueError(f"experiment config field {field_name} must be an integer")
    return value


def _optional_model_gateway_config(data: Mapping[str, Any]) -> ModelGatewayConfig | None:
    if "model_gateway" not in data or data["model_gateway"] is None:
        return None
    value = data["model_gateway"]
    if not isinstance(value, Mapping):
        raise ValueError("experiment config field model_gateway must be an object")

    kind = str(value.get("kind", "deterministic"))
    responses = value.get("responses")
    if responses is not None and not isinstance(responses, Mapping):
        raise ValueError("model gateway responses must be an object")
    model = value.get("model")
    if kind == "openai" and model is None:
        raise ValueError("openai model gateway requires model")
    api_key_env = value.get("api_key_env", "OPENAI_API_KEY")
    timeout_seconds = value.get("timeout_seconds", 30.0)
    max_output_tokens = value.get("max_output_tokens")

    return ModelGatewayConfig(
        kind=kind,
        responses=dict(responses) if responses is not None else None,
        model=model,
        api_key_env=api_key_env,
        timeout_seconds=timeout_seconds,
        max_output_tokens=max_output_tokens,
    )


def _optional_judgment_repair_policy(
    data: Mapping[str, Any],
) -> EvidenceJudgmentRepairPolicy | None:
    if "judgment_repair_policy" not in data or data["judgment_repair_policy"] is None:
        return None
    value = data["judgment_repair_policy"]
    if not isinstance(value, Mapping):
        raise ValueError("experiment config field judgment_repair_policy must be an object")
    return EvidenceJudgmentRepairPolicy.from_config(value)


__all__ = [
    "experiment_config_from_mapping",
    "load_experiment_config",
]
