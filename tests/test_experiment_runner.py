import json
from pathlib import Path

import pytest

from bayesprobe.benchmark_io import BenchmarkDataset
import bayesprobe.experiment_runner as experiment_runner
from bayesprobe.experiment_artifacts import write_experiment_artifact_bundle
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


def test_run_benchmark_experiment_writes_artifact_bundle(tmp_path: Path):
    report_path = tmp_path / "reports" / "toy-report.json"
    ledger_path = tmp_path / "ledgers" / "toy-ledger.jsonl"
    artifact_dir = tmp_path / "artifacts" / "toy-run"

    result = run_benchmark_experiment(
        ExperimentRunConfig(
            dataset_path=FIXTURE_PATH,
            report_path=report_path,
            ledger_path=ledger_path,
            artifact_dir=artifact_dir,
            run_name="toy-artifact-run",
            metadata={
                "suite": "offline",
                "api_key": "sk-proj-secret-value",
                "apiKey": "camel-secret-value",
                "openaiApiKey": "openai-camel-secret-value",
                "APIKEY": "uppercase-secret-value",
                "OPENAIAPIKEY": "uppercase-openai-secret-value",
                "nested": {
                    "token": "hidden-token-value",
                    "safe": "kept",
                },
            },
            model_gateway={
                "kind": "scripted",
                "api_key": "sk-proj-secret-value",
                "responses": {
                    "judge_evidence": {
                        "evidence_type": "supporting",
                        "likelihoods": {
                            "H1": "moderately_confirming",
                            "H2": "moderately_disconfirming",
                        },
                        "interpretation": "Scripted artifact judgment.",
                    }
                },
            },
            judgment_repair_policy={"max_attempts": 1},
        )
    )

    manifest_path = artifact_dir / "manifest.json"
    config_snapshot_path = artifact_dir / "config_snapshot.json"
    dataset_snapshot_path = artifact_dir / "dataset_snapshot.json"
    artifact_report_path = artifact_dir / "report.json"
    artifact_ledger_path = artifact_dir / "ledger.jsonl"
    model_invocations_path = artifact_dir / "model_invocations.json"

    assert result.artifact_dir == artifact_dir
    assert result.artifact_manifest_path == manifest_path
    assert model_invocations_path.exists()
    assert manifest_path.exists()
    assert config_snapshot_path.exists()
    assert dataset_snapshot_path.exists()
    assert artifact_report_path.exists()
    assert artifact_ledger_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_snapshot = json.loads(config_snapshot_path.read_text(encoding="utf-8"))
    dataset_snapshot = json.loads(dataset_snapshot_path.read_text(encoding="utf-8"))
    artifact_report = json.loads(artifact_report_path.read_text(encoding="utf-8"))
    model_invocations = json.loads(model_invocations_path.read_text(encoding="utf-8"))
    artifact_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            manifest_path,
            config_snapshot_path,
            dataset_snapshot_path,
            artifact_report_path,
            artifact_ledger_path,
            model_invocations_path,
        ]
    )

    assert manifest["artifact_version"] == "0.1"
    assert manifest["run_name"] == "toy-artifact-run"
    assert manifest["dataset_name"] == "toy_belief_revision"
    assert manifest["sample_count"] == 3
    assert manifest["metadata"] == {"suite": "offline", "nested": {"safe": "kept"}}
    assert manifest["model_gateway"]["kind"] == "scripted"
    assert manifest["model_gateway"]["scripted_response_tasks"] == ["judge_evidence"]
    assert "responses" not in manifest["model_gateway"]
    assert config_snapshot["artifact_dir"] == str(artifact_dir)
    assert config_snapshot["ledger_path"] == str(ledger_path)
    assert config_snapshot["metadata"] == {
        "suite": "offline",
        "nested": {"safe": "kept"},
    }
    assert dataset_snapshot["dataset_name"] == "toy_belief_revision"
    assert len(dataset_snapshot["samples"]) == 3
    assert artifact_report["sample_count"] == 3
    assert "sk-proj-secret-value" not in artifact_text
    assert "camel-secret-value" not in artifact_text
    assert "openai-camel-secret-value" not in artifact_text
    assert "uppercase-secret-value" not in artifact_text
    assert "uppercase-openai-secret-value" not in artifact_text
    assert '"api_key"' not in artifact_text
    assert '"apiKey"' not in artifact_text
    assert '"openaiApiKey"' not in artifact_text
    assert '"APIKEY"' not in artifact_text
    assert '"OPENAIAPIKEY"' not in artifact_text
    assert '"token"' not in artifact_text
    assert "hidden-token-value" not in artifact_text
    assert "kept" in artifact_text
    assert manifest["model_invocations_path"] == str(model_invocations_path)
    assert manifest["model_invocation_count"] == 4
    assert manifest["model_invocation_summary"] == [
        {
            "task": "judge_evidence",
            "adapter_kind": "scripted",
            "prompt_id": "evidence_judgment",
            "prompt_version": "v0.1",
            "schema_name": "EvidenceJudgment",
            "schema_version": "v0.1",
            "repair_attempt_index": None,
            "metadata": {},
            "occurrence_count": 4,
        }
    ]
    assert model_invocations == {
        "artifact_version": "0.1",
        "invocation_count": 4,
        "invocations": manifest["model_invocation_summary"],
    }


def test_write_experiment_artifact_bundle_creates_selected_ledger_when_missing(
    tmp_path: Path,
):
    report_path = tmp_path / "reports" / "toy-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("{}", encoding="utf-8")
    artifact_dir = tmp_path / "artifacts" / "toy-run"
    ledger_path = artifact_dir / "ledger.jsonl"

    bundle = write_experiment_artifact_bundle(
        artifact_dir=artifact_dir,
        config=ExperimentRunConfig(
            dataset_path=FIXTURE_PATH,
            report_path=report_path,
            ledger_path=ledger_path,
            artifact_dir=artifact_dir,
            run_name="empty-ledger-run",
        ),
        dataset=BenchmarkDataset(dataset_name="toy_belief_revision", samples=[]),
        report_path=report_path,
        ledger_path=ledger_path,
        sample_count=0,
    )

    assert bundle.ledger_path == ledger_path
    assert ledger_path.exists()
    assert ledger_path.read_text(encoding="utf-8") == ""


def test_run_benchmark_experiment_uses_artifact_ledger_when_ledger_path_is_omitted(
    tmp_path: Path,
):
    report_path = tmp_path / "reports" / "toy-report.json"
    artifact_dir = tmp_path / "artifacts" / "toy-run"
    artifact_ledger_path = artifact_dir / "ledger.jsonl"

    result = run_benchmark_experiment(
        ExperimentRunConfig(
            dataset_path=FIXTURE_PATH,
            report_path=report_path,
            artifact_dir=artifact_dir,
        )
    )

    record_types = [
        record["record_type"] for record in JsonlLedgerStore(artifact_ledger_path).read_all()
    ]
    assert result.ledger_path == artifact_ledger_path
    assert artifact_ledger_path.exists()
    assert "benchmark_sample_result" in record_types


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


def test_experiment_runner_runs_v0_2_dataset_with_recorded_gateway(tmp_path: Path):
    config = ExperimentRunConfig(
        dataset_path=Path("fixtures/benchmarks/bayesprobe_v0_2_methodology.json"),
        report_path=tmp_path / "report.json",
        ledger_path=tmp_path / "ledger.jsonl",
        model_gateway={
            "kind": "recorded",
            "fixture_path": "fixtures/providers/deepseek_chat_evidence_v0_1.json",
        },
        max_cycles=1,
        max_probes_per_cycle=1,
    )

    result = run_benchmark_experiment(config)

    assert result.suite_result.sample_count >= 8
    assert result.report_path.exists()
    assert result.ledger_path is not None
    assert result.ledger_path.exists()
    assert result.suite_result.final_accuracy >= 0.5
    assert result.suite_result.update_direction_accuracy is not None


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
