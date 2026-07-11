import json
from pathlib import Path

from bayesprobe.benchmark_io import BenchmarkDataset
from bayesprobe.experiment_artifacts import write_experiment_artifact_bundle
from bayesprobe.experiment_runner import ExperimentRunConfig


FIXTURE_PATH = Path("fixtures/benchmarks/toy_belief_revision.json")


def write_report(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")


def append_ledger_record(path: Path, record_type: str, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"record_type": record_type, "payload": payload}) + "\n")


def test_model_invocation_artifact_aggregates_duplicate_and_repair_traces(
    tmp_path: Path,
):
    artifact_dir = tmp_path / "artifacts"
    report_path = tmp_path / "report.json"
    ledger_path = tmp_path / "ledger.jsonl"
    write_report(report_path)
    for signal_id in ["S1", "S2"]:
        append_ledger_record(
            ledger_path,
            "evidence_event",
            {
                "id": f"E_{signal_id}",
                "model_trace": {
                    "task": "judge_evidence",
                    "adapter_kind": "scripted",
                    "prompt_id": "evidence_judgment",
                    "prompt_version": "v0.1",
                    "schema_name": "EvidenceJudgment",
                    "schema_version": "v0.1",
                    "metadata": {"safe": "kept", "apiKey": "hidden"},
                },
            },
        )
    append_ledger_record(
        ledger_path,
        "evidence_event",
        {
            "id": "E_repair",
            "model_trace": {
                "task": "repair_evidence_judgment",
                "adapter_kind": "scripted",
                "prompt_id": "evidence_judgment_repair",
                "prompt_version": "v0.1",
                "schema_name": "EvidenceJudgment",
                "schema_version": "v0.1",
                "repair_attempt_index": 1,
                "metadata": {"safe": "repair"},
            },
        },
    )
    append_ledger_record(ledger_path, "evidence_event", {"id": "E_empty", "model_trace": {}})
    append_ledger_record(ledger_path, "belief_update", {"model_trace": {"task": "ignored"}})

    bundle = write_experiment_artifact_bundle(
        artifact_dir=artifact_dir,
        config=ExperimentRunConfig(
            dataset_path=FIXTURE_PATH,
            report_path=report_path,
            ledger_path=ledger_path,
            artifact_dir=artifact_dir,
        ),
        dataset=BenchmarkDataset(dataset_name="toy", samples=[]),
        report_path=report_path,
        ledger_path=ledger_path,
        sample_count=0,
    )

    payload = json.loads(bundle.model_invocations_path.read_text(encoding="utf-8"))
    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    text = bundle.model_invocations_path.read_text(encoding="utf-8")

    assert payload["invocation_count"] == 3
    assert payload["invocations"] == [
        {
            "task": "judge_evidence",
            "adapter_kind": "scripted",
            "prompt_id": "evidence_judgment",
            "prompt_version": "v0.1",
            "schema_name": "EvidenceJudgment",
            "schema_version": "v0.1",
            "repair_attempt_index": None,
            "metadata": {"safe": "kept"},
            "occurrence_count": 2,
        },
        {
            "task": "repair_evidence_judgment",
            "adapter_kind": "scripted",
            "prompt_id": "evidence_judgment_repair",
            "prompt_version": "v0.1",
            "schema_name": "EvidenceJudgment",
            "schema_version": "v0.1",
            "repair_attempt_index": 1,
            "metadata": {"safe": "repair"},
            "occurrence_count": 1,
        },
    ]
    assert manifest["model_invocation_count"] == 3
    assert manifest["model_invocation_summary"] == payload["invocations"]
    assert "hidden" not in text
    assert "apiKey" not in text


def test_model_invocation_artifact_is_empty_for_missing_or_empty_ledger(
    tmp_path: Path,
):
    artifact_dir = tmp_path / "artifacts"
    report_path = tmp_path / "report.json"
    ledger_path = tmp_path / "ledger.jsonl"
    write_report(report_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text("", encoding="utf-8")

    bundle = write_experiment_artifact_bundle(
        artifact_dir=artifact_dir,
        config=ExperimentRunConfig(
            dataset_path=FIXTURE_PATH,
            report_path=report_path,
            ledger_path=ledger_path,
            artifact_dir=artifact_dir,
        ),
        dataset=BenchmarkDataset(dataset_name="toy", samples=[]),
        report_path=report_path,
        ledger_path=ledger_path,
        sample_count=0,
    )

    payload = json.loads(bundle.model_invocations_path.read_text(encoding="utf-8"))
    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    assert bundle.ledger_path == artifact_dir / "ledger.jsonl"
    assert bundle.ledger_path.read_text(encoding="utf-8") == ""
    assert payload == {"artifact_version": "0.1", "invocation_count": 0, "invocations": []}
    assert manifest["model_invocation_count"] == 0
    assert manifest["model_invocation_summary"] == []


def test_artifact_snapshot_includes_openai_base_url_without_raw_api_key(tmp_path: Path):
    from bayesprobe import load_benchmark_dataset

    dataset = load_benchmark_dataset(FIXTURE_PATH)
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps({"results": [], "sample_count": 0}),
        encoding="utf-8",
    )
    config = ExperimentRunConfig(
        dataset_path=FIXTURE_PATH,
        report_path=report_path,
        artifact_dir=tmp_path / "artifact",
        model_gateway={
            "kind": "openai",
            "model": "gpt-5.5",
            "api_key_env": "OPENAI_API_KEY",
            "base_url": "https://provider.example/v1",
        },
        metadata={"api_key": "sk-secret"},
    )

    bundle = write_experiment_artifact_bundle(
        artifact_dir=tmp_path / "artifact",
        config=config,
        dataset=dataset,
        report_path=report_path,
        ledger_path=None,
        sample_count=0,
    )

    manifest_text = bundle.manifest_path.read_text(encoding="utf-8")
    snapshot = json.loads(bundle.config_snapshot_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_text)
    assert snapshot["model_gateway"]["base_url"] == "https://provider.example/v1"
    assert manifest["model_gateway"]["base_url"] == "https://provider.example/v1"
    assert "sk-secret" not in manifest_text
    assert "sk-secret" not in bundle.config_snapshot_path.read_text(encoding="utf-8")


def test_artifact_snapshot_includes_chat_completions_provider_without_raw_api_key(
    tmp_path: Path,
):
    from bayesprobe import load_benchmark_dataset

    dataset = load_benchmark_dataset(FIXTURE_PATH)
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps({"results": [], "sample_count": 0}),
        encoding="utf-8",
    )
    config = ExperimentRunConfig(
        dataset_path=FIXTURE_PATH,
        report_path=report_path,
        artifact_dir=tmp_path / "artifact",
        model_gateway={
            "kind": "openai_chat_completions",
            "model": "provider-model",
            "api_key_env": "PROVIDER_API_KEY",
            "base_url": "https://provider.example/v1",
            "temperature": 0,
            "top_p": 1,
            "thinking": "enabled",
            "reasoning_effort": "max",
        },
        metadata={"api_key": "provider-secret-123"},
    )

    bundle = write_experiment_artifact_bundle(
        artifact_dir=tmp_path / "artifact",
        config=config,
        dataset=dataset,
        report_path=report_path,
        ledger_path=None,
        sample_count=0,
    )

    manifest_text = bundle.manifest_path.read_text(encoding="utf-8")
    snapshot_text = bundle.config_snapshot_path.read_text(encoding="utf-8")
    snapshot = json.loads(snapshot_text)
    manifest = json.loads(manifest_text)
    assert snapshot["model_gateway"] == {
        "kind": "openai_chat_completions",
        "model": "provider-model",
        "api_key_env": "PROVIDER_API_KEY",
        "base_url": "https://provider.example/v1",
        "temperature": 0,
        "top_p": 1,
        "thinking": "enabled",
        "reasoning_effort": "max",
    }
    assert manifest["model_gateway"] == snapshot["model_gateway"]
    assert "provider-secret-123" not in manifest_text
    assert "provider-secret-123" not in snapshot_text


def test_artifact_snapshot_includes_recorded_gateway_fixture_path(tmp_path: Path):
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "dataset_name": "toy",
                "metadata": {},
                "sample_count": 0,
                "final_accuracy": 0.0,
                "update_direction_accuracy": None,
                "results": [],
            }
        ),
        encoding="utf-8",
    )
    config = ExperimentRunConfig(
        dataset_path=FIXTURE_PATH,
        report_path=report_path,
        model_gateway={
            "kind": "recorded",
            "fixture_path": "fixtures/providers/deepseek_chat_evidence_v0_1.json",
        },
    )

    bundle = write_experiment_artifact_bundle(
        artifact_dir=tmp_path / "artifacts",
        config=config,
        dataset=BenchmarkDataset(dataset_name="toy", samples=[]),
        report_path=report_path,
        ledger_path=None,
        sample_count=0,
    )

    config_snapshot = json.loads(bundle.config_snapshot_path.read_text(encoding="utf-8"))
    assert config_snapshot["model_gateway"] == {
        "kind": "recorded",
        "fixture_path": "fixtures/providers/deepseek_chat_evidence_v0_1.json",
    }
