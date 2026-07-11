import json
from pathlib import Path

import pytest

from bayesprobe.evaluation.config import (
    CapabilityExperimentConfig,
    capability_config_from_mapping,
    load_capability_config,
)
from bayesprobe.model_gateway import ProviderRequestControls


REVISION = "c" * 40


def minimal_mapping():
    return {
        "experiment_name": "BayesProbe HLE Text-MCQ-100 Python-Augmented Capability Pilot v0.1",
        "dataset": {"revision": REVISION},
        "paths": {
            "restricted_root": "artifacts/restricted/hle-pilot-v0.1",
            "report_root": "reports/hle-pilot-v0.1",
        },
        "python": {"image": "bayesprobe-hle-python:v0.1"},
        "prompt_registry": {"version": "v0.1", "prompts": {}},
        "pricing_snapshot": {"as_of": "2026-07-11", "currency": "USD"},
    }


def test_capability_config_freezes_v01_provider_autonomy_and_concurrency():
    config = capability_config_from_mapping(minimal_mapping())

    assert isinstance(config, CapabilityExperimentConfig)
    assert config.selection.revision == REVISION
    assert config.selection.sample_count == 100
    assert config.selection.seed == "20260711"
    assert config.model_gateway.kind == "openai_chat_completions"
    assert config.model_gateway.base_url == "https://api.deepseek.com"
    assert config.model_gateway.model == "deepseek-v4-flash"
    assert config.model_gateway.api_key_env == "DEEPSEEK_API_KEY"
    assert config.model_gateway.timeout_seconds == 900
    assert config.model_gateway.max_output_tokens == 65536
    assert config.model_gateway.request_controls == ProviderRequestControls(
        temperature=0,
        top_p=1,
        thinking="enabled",
        reasoning_effort="max",
    )
    assert config.max_cycles == 4
    assert config.max_probes_per_cycle == 2
    assert config.stop_on_no_probes is True
    assert config.confidence_threshold is None
    assert config.posterior_delta_threshold is None
    assert config.direct_concurrency == 8
    assert config.bayesprobe_concurrency == 4
    assert config.docker_concurrency == 4


def test_capability_config_resolves_paths_relative_to_config_file(tmp_path: Path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "pilot.json"
    config_path.write_text(json.dumps(minimal_mapping()), encoding="utf-8")

    config = load_capability_config(config_path)

    assert config.restricted_root == config_dir / "artifacts/restricted/hle-pilot-v0.1"
    assert config.report_root == config_dir / "reports/hle-pilot-v0.1"


def test_capability_config_rejects_raw_provider_api_key():
    payload = minimal_mapping()
    payload["provider"] = {"api_key": "sk-must-not-persist"}

    with pytest.raises(ValueError, match="api_key is not allowed"):
        capability_config_from_mapping(payload)


def test_capability_config_rejects_changes_to_frozen_v01_policy():
    payload = minimal_mapping()
    payload["provider"] = {"temperature": 0.2}

    with pytest.raises(ValueError, match="frozen v0.1 provider policy"):
        capability_config_from_mapping(payload)


def test_capability_config_hash_is_stable_for_mapping_key_order():
    first = capability_config_from_mapping(minimal_mapping())
    reversed_payload = dict(reversed(list(minimal_mapping().items())))
    second = capability_config_from_mapping(reversed_payload)

    assert first.config_sha256 == second.config_sha256
    assert len(first.prompt_registry_sha256) == 64
    assert len(first.pricing_snapshot_sha256) == 64
