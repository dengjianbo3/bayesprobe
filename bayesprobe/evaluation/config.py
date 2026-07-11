from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bayesprobe.evaluation.hle import HLESelectionConfig
from bayesprobe.evaluation.python_probe import DockerPythonSandboxConfig
from bayesprobe.model_gateway import ModelGatewayConfig, ProviderRequestControls


_PROVIDER_POLICY = {
    "kind": "openai_chat_completions",
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-v4-flash",
    "api_key_env": "DEEPSEEK_API_KEY",
    "temperature": 0,
    "top_p": 1,
    "thinking": "enabled",
    "reasoning_effort": "max",
    "max_output_tokens": 65536,
    "timeout_seconds": 900,
}


@dataclass(frozen=True)
class CapabilityExperimentConfig:
    experiment_name: str
    selection: HLESelectionConfig
    model_gateway: ModelGatewayConfig
    python_sandbox: DockerPythonSandboxConfig
    restricted_root: Path
    report_root: Path
    prompt_registry: dict[str, Any]
    pricing_snapshot: dict[str, Any]
    max_cycles: int = 4
    max_probes_per_cycle: int = 2
    stop_on_no_probes: bool = True
    confidence_threshold: float | None = None
    posterior_delta_threshold: float | None = None
    direct_concurrency: int = 8
    bayesprobe_concurrency: int = 4
    docker_concurrency: int = 4

    @property
    def prompt_registry_sha256(self) -> str:
        return _canonical_sha256(self.prompt_registry)

    @property
    def pricing_snapshot_sha256(self) -> str:
        return _canonical_sha256(self.pricing_snapshot)

    @property
    def config_sha256(self) -> str:
        return _canonical_sha256(self.snapshot())

    def snapshot(self) -> dict[str, Any]:
        controls = self.model_gateway.request_controls
        return {
            "experiment_name": self.experiment_name,
            "dataset": {
                "name": "cais/hle",
                "revision": self.selection.revision,
                "sample_count": self.selection.sample_count,
                "seed": self.selection.seed,
            },
            "provider": {
                "kind": self.model_gateway.kind,
                "base_url": self.model_gateway.base_url,
                "model": self.model_gateway.model,
                "api_key_env": self.model_gateway.api_key_env,
                "temperature": controls.temperature,
                "top_p": controls.top_p,
                "thinking": controls.thinking,
                "reasoning_effort": controls.reasoning_effort,
                "max_output_tokens": self.model_gateway.max_output_tokens,
                "timeout_seconds": self.model_gateway.timeout_seconds,
            },
            "autonomy": {
                "max_cycles": self.max_cycles,
                "max_probes_per_cycle": self.max_probes_per_cycle,
                "stop_on_no_probes": self.stop_on_no_probes,
                "confidence_threshold": self.confidence_threshold,
                "posterior_delta_threshold": self.posterior_delta_threshold,
            },
            "concurrency": {
                "direct": self.direct_concurrency,
                "bayesprobe": self.bayesprobe_concurrency,
                "docker": self.docker_concurrency,
            },
            "python": {
                "image": self.python_sandbox.image,
                "timeout_seconds": self.python_sandbox.timeout_seconds,
                "max_output_bytes": self.python_sandbox.max_output_bytes,
                "pids_limit": self.python_sandbox.pids_limit,
                "memory": self.python_sandbox.memory,
                "cpus": self.python_sandbox.cpus,
                "tmpfs_size": self.python_sandbox.tmpfs_size,
                "user": self.python_sandbox.user,
            },
            "paths": {
                "restricted_root": str(self.restricted_root),
                "report_root": str(self.report_root),
            },
            "prompt_registry": self.prompt_registry,
            "pricing_snapshot": self.pricing_snapshot,
        }


def load_capability_config(path: str | Path) -> CapabilityExperimentConfig:
    config_path = Path(path)
    if config_path.suffix.lower() != ".json":
        raise ValueError("capability config path must end with .json")
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("could not parse capability config JSON") from error
    if not isinstance(payload, Mapping):
        raise ValueError("capability config must be a JSON object")
    return capability_config_from_mapping(payload, base_dir=config_path.parent)


def capability_config_from_mapping(
    payload: Mapping[str, Any],
    *,
    base_dir: str | Path | None = None,
) -> CapabilityExperimentConfig:
    experiment_name = _required_string(payload, "experiment_name")
    dataset = _required_mapping(payload, "dataset")
    selection = HLESelectionConfig(
        revision=_required_string(dataset, "revision"),
        sample_count=dataset.get("sample_count", 100),
        seed=dataset.get("seed", "20260711"),
    )
    if selection.sample_count != 100 or selection.seed != "20260711":
        raise ValueError("capability pilot requires the frozen v0.1 dataset policy")

    supplied_provider = payload.get("provider", {})
    if not isinstance(supplied_provider, Mapping):
        raise ValueError("capability config provider must be an object")
    if "api_key" in supplied_provider:
        raise ValueError("capability config provider api_key is not allowed")
    provider = {**_PROVIDER_POLICY, **dict(supplied_provider)}
    if provider != _PROVIDER_POLICY:
        raise ValueError("capability config must use the frozen v0.1 provider policy")
    model_gateway = ModelGatewayConfig(
        kind=provider["kind"],
        model=provider["model"],
        api_key_env=provider["api_key_env"],
        timeout_seconds=provider["timeout_seconds"],
        max_output_tokens=provider["max_output_tokens"],
        base_url=provider["base_url"],
        request_controls=ProviderRequestControls(
            temperature=provider["temperature"],
            top_p=provider["top_p"],
            thinking=provider["thinking"],
            reasoning_effort=provider["reasoning_effort"],
        ),
    )

    autonomy = dict(payload.get("autonomy", {}))
    expected_autonomy = {
        "max_cycles": 4,
        "max_probes_per_cycle": 2,
        "stop_on_no_probes": True,
        "confidence_threshold": None,
        "posterior_delta_threshold": None,
    }
    autonomy = {**expected_autonomy, **autonomy}
    if autonomy != expected_autonomy:
        raise ValueError("capability config must use the frozen v0.1 autonomy policy")

    concurrency = dict(payload.get("concurrency", {}))
    expected_concurrency = {"direct": 8, "bayesprobe": 4, "docker": 4}
    concurrency = {**expected_concurrency, **concurrency}
    if concurrency != expected_concurrency:
        raise ValueError("capability config must use the frozen v0.1 concurrency policy")

    python_payload = _required_mapping(payload, "python")
    python_sandbox = DockerPythonSandboxConfig(
        image=_required_string(python_payload, "image"),
        timeout_seconds=python_payload.get("timeout_seconds", 30),
        max_output_bytes=python_payload.get("max_output_bytes", 64 * 1024),
        pids_limit=python_payload.get("pids_limit", 64),
        memory=python_payload.get("memory", "1g"),
        cpus=python_payload.get("cpus", 1),
        tmpfs_size=python_payload.get("tmpfs_size", "64m"),
        user=python_payload.get("user", "65532:65532"),
    )
    if python_sandbox != DockerPythonSandboxConfig(image=python_sandbox.image):
        raise ValueError("capability config must use the frozen v0.1 Python policy")

    paths = _required_mapping(payload, "paths")
    restricted_root = _resolve_path(
        _required_string(paths, "restricted_root"),
        base_dir=base_dir,
    )
    report_root = _resolve_path(
        _required_string(paths, "report_root"),
        base_dir=base_dir,
    )
    prompt_registry = dict(_required_mapping(payload, "prompt_registry"))
    pricing_snapshot = dict(_required_mapping(payload, "pricing_snapshot"))
    return CapabilityExperimentConfig(
        experiment_name=experiment_name,
        selection=selection,
        model_gateway=model_gateway,
        python_sandbox=python_sandbox,
        restricted_root=restricted_root,
        report_root=report_root,
        prompt_registry=prompt_registry,
        pricing_snapshot=pricing_snapshot,
        **autonomy,
        direct_concurrency=concurrency["direct"],
        bayesprobe_concurrency=concurrency["bayesprobe"],
        docker_concurrency=concurrency["docker"],
    )


def _required_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"capability config {key} must be an object")
    return value


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"capability config {key} must not be empty")
    return value.strip()


def _resolve_path(value: str, *, base_dir: str | Path | None) -> Path:
    path = Path(value)
    return path if path.is_absolute() or base_dir is None else Path(base_dir) / path


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "CapabilityExperimentConfig",
    "capability_config_from_mapping",
    "load_capability_config",
]
