import json
from pathlib import Path

import pytest

import bayesprobe.experiment_runner as experiment_runner
from bayesprobe.experiment_runner import (
    ExperimentRunConfig,
    run_benchmark_experiment,
)
from bayesprobe.ledger import JsonlLedgerStore
from bayesprobe.openai_gateway import OpenAIModelGatewayConfig


FIXTURE_PATH = Path("fixtures/benchmarks/toy_belief_revision.json")


def test_run_benchmark_experiment_writes_report(tmp_path: Path):
    report_path = tmp_path / "reports" / "toy-report.json"

    result = run_benchmark_experiment(
        ExperimentRunConfig(
            dataset_path=FIXTURE_PATH,
            report_path=report_path,
        )
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert result.dataset.dataset_name == "toy_belief_revision"
    assert result.report_path == report_path
    assert result.ledger_path is None
    assert result.suite_result.sample_count == 3
    assert result.suite_result.final_accuracy == 1.0
    assert result.suite_result.update_direction_accuracy == 1.0
    assert payload["dataset_name"] == "toy_belief_revision"
    assert payload["metadata"]["version"] == "0.1"
    assert payload["sample_count"] == 3
    assert payload["final_accuracy"] == 1.0
    assert [item["sample_id"] for item in payload["results"]] == [
        "toy_active_support",
        "toy_passive_refute",
        "toy_mixed_refute",
    ]
    assert [item["signal_shape"] for item in payload["results"]] == [
        "active_only",
        "passive_only",
        "active_plus_passive",
    ]


def test_run_benchmark_experiment_writes_optional_ledger(tmp_path: Path):
    report_path = tmp_path / "toy-report.json"
    ledger_path = tmp_path / "ledgers" / "toy-ledger.jsonl"

    result = run_benchmark_experiment(
        ExperimentRunConfig(
            dataset_path=FIXTURE_PATH,
            report_path=report_path,
            ledger_path=ledger_path,
        )
    )

    record_types = [record["record_type"] for record in JsonlLedgerStore(ledger_path).read_all()]
    assert result.ledger_path == ledger_path
    assert "run" in record_types
    assert "cycle" in record_types
    assert "external_signal" in record_types
    assert "evidence_event" in record_types
    assert "belief_update" in record_types
    assert "benchmark_sample_result" in record_types


def test_run_benchmark_experiment_uses_model_gateway_config(tmp_path: Path):
    report_path = tmp_path / "toy-report.json"
    ledger_path = tmp_path / "toy-ledger.jsonl"

    result = run_benchmark_experiment(
        ExperimentRunConfig(
            dataset_path=FIXTURE_PATH,
            report_path=report_path,
            ledger_path=ledger_path,
            model_gateway={
                "kind": "scripted",
                "responses": {
                    "judge_evidence": {
                        "evidence_type": "boundary_condition",
                        "likelihoods": {"H1": "weakly_disconfirming", "H2": "neutral"},
                        "interpretation": "Experiment configured scripted judgment.",
                        "quality_overrides": {"reliability": 0.62},
                    }
                },
            },
        )
    )

    evidence_payloads = [
        record["payload"]
        for record in JsonlLedgerStore(ledger_path).read_all("evidence_event")
    ]
    assert result.ledger_path == ledger_path
    assert evidence_payloads[0]["evidence_type"] == "boundary_condition"
    assert evidence_payloads[0]["reliability"] == 0.62


def test_run_benchmark_experiment_uses_judgment_repair_policy_config(tmp_path: Path):
    report_path = tmp_path / "toy-report.json"
    ledger_path = tmp_path / "toy-ledger.jsonl"

    result = run_benchmark_experiment(
        ExperimentRunConfig(
            dataset_path=FIXTURE_PATH,
            report_path=report_path,
            ledger_path=ledger_path,
            model_gateway={
                "kind": "scripted",
                "responses": {
                    "judge_evidence": {
                        "evidence_type": "not_a_type",
                        "likelihoods": {"H1": "neutral", "H2": "neutral"},
                        "interpretation": "Invalid evidence type.",
                    },
                    "repair_evidence_judgment": {
                        "evidence_type": "supporting",
                        "likelihoods": {
                            "H1": "moderately_confirming",
                            "H2": "moderately_disconfirming",
                        },
                        "interpretation": "Experiment repaired judgment.",
                    },
                },
            },
            judgment_repair_policy={"max_attempts": 1},
        )
    )

    evidence_payloads = [
        record["payload"]
        for record in JsonlLedgerStore(ledger_path).read_all("evidence_event")
    ]
    assert result.ledger_path == ledger_path
    assert evidence_payloads[0]["evidence_type"] == "supporting"
    assert evidence_payloads[0]["discard_reason"] is None


def test_run_benchmark_experiment_constructs_openai_gateway_without_network(
    tmp_path: Path,
    monkeypatch,
):
    captured = {}

    class CapturingGateway:
        adapter_kind = "capturing_openai"

        def __init__(self, *, config):
            captured["config"] = config

        def complete_structured(self, request):
            return {
                "evidence_type": "supporting",
                "likelihoods": {
                    "H1": "moderately_confirming",
                    "H2": "moderately_disconfirming",
                },
                "interpretation": "Captured OpenAI fixture.",
                "quality_overrides": {},
            }

    def fake_build_model_gateway(config):
        assert config["kind"] == "openai"
        return CapturingGateway(config=OpenAIModelGatewayConfig(model=config["model"]))

    monkeypatch.setattr(experiment_runner, "build_model_gateway", fake_build_model_gateway)
    report_path = tmp_path / "toy-report.json"

    result = run_benchmark_experiment(
        ExperimentRunConfig(
            dataset_path=FIXTURE_PATH,
            report_path=report_path,
            model_gateway={"kind": "openai", "model": "gpt-5.5"},
        )
    )

    assert captured["config"].model == "gpt-5.5"
    assert result.suite_result.sample_count == 3


@pytest.mark.parametrize(
    "config_kwargs",
    [
        {"max_cycles": 0},
        {"max_probes_per_cycle": 0},
    ],
)
def test_run_benchmark_experiment_rejects_invalid_config(
    tmp_path: Path,
    config_kwargs: dict,
):
    with pytest.raises(ValueError):
        ExperimentRunConfig(
            dataset_path=FIXTURE_PATH,
            report_path=tmp_path / "report.json",
            **config_kwargs,
        )
