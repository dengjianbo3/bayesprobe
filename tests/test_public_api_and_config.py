import json
from pathlib import Path

import pytest

import bayesprobe
from bayesprobe import (
    BenchmarkDataset,
    BenchmarkHarness,
    BenchmarkSample,
    BenchmarkSampleResult,
    BenchmarkSignal,
    BenchmarkSignalShape,
    BenchmarkSuiteResult,
    DeterministicModelGateway,
    EvidenceJudgment,
    ExperimentRunConfig,
    ExperimentRunResult,
    ModelGateway,
    ModelGatewayConfig,
    ModelGatewayValidationError,
    ScriptedModelGateway,
    StructuredModelRequest,
    build_model_gateway,
    evidence_judgment_from_mapping,
    load_benchmark_dataset,
    load_experiment_config,
    run_benchmark_experiment,
    write_benchmark_report,
)
from bayesprobe.config import experiment_config_from_mapping


FIXTURE_PATH = Path("fixtures/benchmarks/toy_belief_revision.json")


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_public_sdk_exports_supported_names():
    expected_names = {
        "BenchmarkDataset",
        "BenchmarkHarness",
        "BenchmarkSample",
        "BenchmarkSampleResult",
        "BenchmarkSignal",
        "BenchmarkSignalShape",
        "BenchmarkSuiteResult",
        "DeterministicModelGateway",
        "EvidenceJudgment",
        "ExperimentRunConfig",
        "ExperimentRunResult",
        "ModelGateway",
        "ModelGatewayConfig",
        "ModelGatewayValidationError",
        "ScriptedModelGateway",
        "StructuredModelRequest",
        "build_model_gateway",
        "evidence_judgment_from_mapping",
        "load_benchmark_dataset",
        "load_experiment_config",
        "run_benchmark_experiment",
        "write_benchmark_report",
    }

    assert expected_names.issubset(set(bayesprobe.__all__))
    assert BenchmarkDataset is not None
    assert BenchmarkHarness is not None
    assert BenchmarkSample is not None
    assert BenchmarkSampleResult is not None
    assert BenchmarkSignal is not None
    assert BenchmarkSignalShape.ACTIVE_ONLY.value == "active_only"
    assert BenchmarkSuiteResult is not None
    assert DeterministicModelGateway is not None
    assert EvidenceJudgment is not None
    assert ExperimentRunConfig is not None
    assert ExperimentRunResult is not None
    assert ModelGateway is not None
    assert ModelGatewayConfig is not None
    assert ModelGatewayValidationError is not None
    assert ScriptedModelGateway is not None
    assert StructuredModelRequest is not None
    assert build_model_gateway is not None
    assert evidence_judgment_from_mapping is not None
    assert load_benchmark_dataset is not None
    assert load_experiment_config is not None
    assert run_benchmark_experiment is not None
    assert write_benchmark_report is not None


def test_load_experiment_config_resolves_paths_relative_to_config_file(tmp_path: Path):
    config_dir = tmp_path / "experiments"
    config_dir.mkdir()
    config_path = config_dir / "toy-experiment.json"
    write_json(
        config_path,
        {
            "dataset_path": "datasets/toy.json",
            "report_path": "outputs/toy-report.json",
            "ledger_path": "outputs/toy-ledger.jsonl",
            "max_cycles": 1,
            "max_probes_per_cycle": 1,
        },
    )

    config = load_experiment_config(config_path)

    assert config.dataset_path == config_dir / "datasets" / "toy.json"
    assert config.report_path == config_dir / "outputs" / "toy-report.json"
    assert config.ledger_path == config_dir / "outputs" / "toy-ledger.jsonl"
    assert config.max_cycles == 1
    assert config.max_probes_per_cycle == 1


def test_experiment_config_from_mapping_resolves_paths_with_base_dir(tmp_path: Path):
    config = experiment_config_from_mapping(
        {
            "dataset_path": "datasets/toy.json",
            "report_path": "outputs/toy-report.json",
        },
        base_dir=tmp_path,
    )

    assert config.dataset_path == tmp_path / "datasets" / "toy.json"
    assert config.report_path == tmp_path / "outputs" / "toy-report.json"
    assert config.ledger_path is None


def test_experiment_config_from_mapping_keeps_relative_paths_without_base_dir():
    config = experiment_config_from_mapping(
        {
            "dataset_path": "fixtures/benchmarks/toy_belief_revision.json",
            "report_path": "outputs/toy-report.json",
        }
    )

    assert config.dataset_path == Path("fixtures/benchmarks/toy_belief_revision.json")
    assert config.report_path == Path("outputs/toy-report.json")


def test_experiment_config_from_mapping_parses_model_gateway_object(tmp_path: Path):
    config = experiment_config_from_mapping(
        {
            "dataset_path": "datasets/toy.json",
            "report_path": "outputs/toy-report.json",
            "model_gateway": {
                "kind": "scripted",
                "responses": {
                    "judge_evidence": {
                        "evidence_type": "boundary_condition",
                        "likelihoods": {"H1": "weakly_disconfirming"},
                        "interpretation": "JSON configured judgment.",
                    }
                },
            },
        },
        base_dir=tmp_path,
    )

    assert isinstance(config.model_gateway, ModelGatewayConfig)
    assert config.model_gateway.kind == "scripted"
    assert config.model_gateway.responses["judge_evidence"]["evidence_type"] == "boundary_condition"


def test_loaded_config_runs_benchmark_experiment(tmp_path: Path):
    config_path = tmp_path / "toy-experiment.json"
    report_path = tmp_path / "outputs" / "toy-report.json"
    ledger_path = tmp_path / "outputs" / "toy-ledger.jsonl"
    write_json(
        config_path,
        {
            "dataset_path": str(FIXTURE_PATH.resolve()),
            "report_path": "outputs/toy-report.json",
            "ledger_path": "outputs/toy-ledger.jsonl",
        },
    )

    config = load_experiment_config(config_path)
    result = run_benchmark_experiment(config)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert result.report_path == report_path
    assert result.ledger_path == ledger_path
    assert result.suite_result.sample_count == 3
    assert report["dataset_name"] == "toy_belief_revision"
    assert report["final_accuracy"] == 1.0


@pytest.mark.parametrize(
    ("filename", "payload", "expected_message"),
    [
        ("experiment.yaml", "{}", "experiment config path must end with .json"),
        ("malformed.json", "{", "could not parse experiment config JSON"),
        ("array.json", "[]", "experiment config must be a JSON object"),
        (
            "missing_dataset.json",
            json.dumps({"report_path": "report.json"}),
            "missing required experiment config field: dataset_path",
        ),
        (
            "missing_report.json",
            json.dumps({"dataset_path": "dataset.json"}),
            "missing required experiment config field: report_path",
        ),
        (
            "non_string_path.json",
            json.dumps({"dataset_path": 1, "report_path": "report.json"}),
            "experiment config field dataset_path must be a string",
        ),
        (
            "non_integer_cycles.json",
            json.dumps(
                {
                    "dataset_path": "dataset.json",
                    "report_path": "report.json",
                    "max_cycles": "1",
                }
            ),
            "experiment config field max_cycles must be an integer",
        ),
        (
            "non_object_model_gateway.json",
            json.dumps(
                {
                    "dataset_path": "dataset.json",
                    "report_path": "report.json",
                    "model_gateway": [],
                }
            ),
            "experiment config field model_gateway must be an object",
        ),
    ],
)
def test_load_experiment_config_rejects_invalid_config_files(
    tmp_path: Path,
    filename: str,
    payload: str,
    expected_message: str,
):
    config_path = tmp_path / filename
    config_path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match=expected_message):
        load_experiment_config(config_path)
