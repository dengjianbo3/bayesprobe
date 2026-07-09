import json
from pathlib import Path

import pytest

from bayesprobe.benchmark import BenchmarkHarness, BenchmarkSignalShape
from bayesprobe.benchmark_io import (
    BenchmarkDataset,
    load_benchmark_dataset,
    write_benchmark_report,
)


def passive_signal_payload(signal_id: str = "S_passive_refute") -> dict:
    return {
        "signal_id": signal_id,
        "source_type": "benchmark_stream",
        "source": "fixture",
        "raw_content": "REFUTES: Benchmark passage contradicts H1 and supports H2.",
        "target_hypotheses": ["H1", "H2"],
    }


def active_sample_payload(sample_id: str = "active_support") -> dict:
    return {
        "sample_id": sample_id,
        "question_or_claim": "Does active-only execution support H1?",
        "signal_shape": "active_only",
        "gold_best_hypothesis": "H1",
        "gold_update_directions": {"H1": "strengthened"},
    }


def passive_sample_payload(sample_id: str = "passive_refute") -> dict:
    return {
        "sample_id": sample_id,
        "question_or_claim": "Does the passive signal refute H1?",
        "signal_shape": "passive_only",
        "gold_best_hypothesis": "H2",
        "gold_update_directions": {"H1": "weakened", "H2": "strengthened"},
        "passive_signals": [passive_signal_payload()],
    }


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_benchmark_dataset_from_json_object(tmp_path: Path):
    path = tmp_path / "toy.json"
    write_json(
        path,
        {
            "dataset_name": "toy_belief_revision",
            "metadata": {"version": "0.1"},
            "samples": [
                active_sample_payload(),
                passive_sample_payload(),
            ],
        },
    )

    dataset = load_benchmark_dataset(path)

    assert isinstance(dataset, BenchmarkDataset)
    assert dataset.dataset_name == "toy_belief_revision"
    assert dataset.metadata == {"version": "0.1"}
    assert len(dataset.samples) == 2
    assert dataset.samples[0].signal_shape == BenchmarkSignalShape.ACTIVE_ONLY
    assert dataset.samples[1].passive_signals[0].signal_id == "S_passive_refute"
    assert dataset.samples[1].gold_update_directions == {
        "H1": "weakened",
        "H2": "strengthened",
    }


def test_load_benchmark_dataset_from_json_list_defaults_name_to_file_stem(tmp_path: Path):
    path = tmp_path / "raw_samples.json"
    write_json(path, [active_sample_payload("raw_active")])

    dataset = load_benchmark_dataset(path)

    assert dataset.dataset_name == "raw_samples"
    assert dataset.metadata == {}
    assert [sample.sample_id for sample in dataset.samples] == ["raw_active"]
    assert dataset.samples[0].signal_shape == BenchmarkSignalShape.ACTIVE_ONLY


def test_load_benchmark_dataset_from_jsonl(tmp_path: Path):
    path = tmp_path / "streamed.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(active_sample_payload("jsonl_active")),
                json.dumps(
                    {
                        **passive_sample_payload("jsonl_mixed"),
                        "signal_shape": "active_plus_passive",
                        "passive_signals": [passive_signal_payload("S_jsonl_mixed")],
                    }
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )

    dataset = load_benchmark_dataset(path)

    assert dataset.dataset_name == "streamed"
    assert dataset.metadata == {}
    assert [sample.sample_id for sample in dataset.samples] == [
        "jsonl_active",
        "jsonl_mixed",
    ]
    assert dataset.samples[1].signal_shape == BenchmarkSignalShape.ACTIVE_PLUS_PASSIVE
    assert dataset.samples[1].passive_signals[0].signal_id == "S_jsonl_mixed"


@pytest.mark.parametrize(
    ("filename", "payload", "expected_message"),
    [
        ("bad.txt", "{}", "must end with .json or .jsonl"),
        ("malformed.json", "{", "could not parse benchmark dataset JSON"),
        ("missing_samples.json", "{}", "JSON object dataset must include samples"),
        (
            "non_object_sample.json",
            json.dumps([None]),
            "benchmark sample entry must be an object",
        ),
        (
            "missing_field.json",
            json.dumps([{"question_or_claim": "claim", "gold_best_hypothesis": "H1"}]),
            "missing required benchmark sample field",
        ),
    ],
)
def test_load_benchmark_dataset_rejects_invalid_files(
    tmp_path: Path,
    filename: str,
    payload: str,
    expected_message: str,
):
    path = tmp_path / filename
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match=expected_message):
        load_benchmark_dataset(path)


def test_write_benchmark_report_round_trips_suite_result(tmp_path: Path):
    dataset_path = tmp_path / "report-suite.json"
    write_json(
        dataset_path,
        [
            active_sample_payload("report_active"),
            passive_sample_payload("report_passive"),
        ],
    )
    dataset = load_benchmark_dataset(dataset_path)
    suite_result = BenchmarkHarness().run_suite(dataset.samples)
    report_path = tmp_path / "reports" / "toy-report.json"

    write_benchmark_report(
        report_path,
        suite_result,
        dataset_name="toy_belief_revision",
        metadata={"version": "0.1"},
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["dataset_name"] == "toy_belief_revision"
    assert payload["metadata"] == {"version": "0.1"}
    assert payload["sample_count"] == 2
    assert payload["final_accuracy"] == 1.0
    assert payload["update_direction_accuracy"] == 1.0
    assert [result["signal_shape"] for result in payload["results"]] == [
        "active_only",
        "passive_only",
    ]
    assert payload["results"][0]["active_signal_count"] == 1
    assert payload["results"][1]["passive_signal_count"] == 1
    assert payload["results"][1]["projection_kind"] == "belief_state_projection"


def test_write_benchmark_report_includes_belief_quality_metrics(tmp_path: Path):
    dataset_path = tmp_path / "quality-report-suite.json"
    write_json(dataset_path, [active_sample_payload("quality_report_active")])
    dataset = load_benchmark_dataset(dataset_path)
    suite_result = BenchmarkHarness().run_suite(dataset.samples)
    report_path = tmp_path / "report.json"

    write_benchmark_report(report_path, suite_result, dataset_name="quality_report")

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    result = payload["results"][0]
    assert result["discarded_evidence_count"] == 0
    assert result["schema_violation_count"] == 0
    assert result["dominant_hypothesis_margin"] > 0
    assert result["belief_revision_efficiency"] == 1.0


def test_loaded_dataset_runs_through_benchmark_harness(tmp_path: Path):
    path = tmp_path / "suite.json"
    write_json(
        path,
        {
            "dataset_name": "loaded_suite",
            "samples": [
                active_sample_payload("loaded_active"),
                passive_sample_payload("loaded_passive"),
            ],
        },
    )

    dataset = load_benchmark_dataset(path)
    suite_result = BenchmarkHarness().run_suite(dataset.samples)

    assert suite_result.sample_count == 2
    assert suite_result.final_accuracy == 1.0
    assert suite_result.update_direction_accuracy == 1.0
