import json
from pathlib import Path

import pytest

from bayesprobe.evaluation.artifacts import CapabilityArtifactStore
from bayesprobe.evaluation.contracts import ArmCaseResult, EvaluationCase
from bayesprobe.evaluation.hle import EvaluationGoldStore
from bayesprobe.evaluation.runner import build_experiment_identity
from bayesprobe.evaluation.scoring import (
    MCQScorer,
    score_and_write_experiment,
)


MANIFEST_HASH = "c" * 64


def identity(manifest_hash=MANIFEST_HASH):
    return build_experiment_identity(
        experiment_name="synthetic scoring pilot",
        code_git_sha="a" * 40,
        dataset_revision_sha="b" * 40,
        selection_manifest_sha256=manifest_hash,
        config_sha256="d" * 64,
        prompt_registry_sha256="e" * 64,
        python_image_digest="sha256:" + "f" * 64,
    )


def cases():
    return [
        EvaluationCase(
            sample_id=f"private_sample_{index}",
            question=f"Private synthetic question {index}? Answer Choices: A. yes B. no",
            choices={"A": "yes", "B": "no"},
        )
        for index in range(6)
    ]


def gold_store(manifest_hash=MANIFEST_HASH):
    return EvaluationGoldStore(
        manifest_sha256=manifest_hash,
        labels={
            f"private_sample_{index}": "A" if index % 2 == 0 else "B"
            for index in range(6)
        },
    )


def completed(sample_id, arm, answer, confidence=0.8, **metrics):
    probabilities = (
        {"A": confidence, "B": 1 - confidence}
        if answer == "A"
        else {"A": 1 - confidence, "B": confidence}
    )
    return ArmCaseResult(
        sample_id=sample_id,
        arm=arm,
        state="completed",
        answer_label=answer,
        probabilities=probabilities,
        answer_summary="Private model summary.",
        process_metrics=metrics,
    )


def failed(sample_id, arm):
    return ArmCaseResult(
        sample_id=sample_id,
        arm=arm,
        state="terminal_failed",
        answer_label=None,
        probabilities=None,
        error_category="provider_timeout",
        process_metrics={"cycles": 1},
    )


def result_fixture():
    gold = gold_store().labels
    results = []
    for index, sample_id in enumerate(gold):
        correct = gold[sample_id]
        wrong = "B" if correct == "A" else "A"
        bayes_answer = correct if index in {0, 1, 2, 3} else wrong
        direct_answer = correct if index in {0, 1, 4} else wrong
        if index == 5:
            results.append(failed(sample_id, "bayesprobe_python"))
        else:
            results.append(
                completed(
                    sample_id,
                    "bayesprobe_python",
                    bayes_answer,
                    cycles=2,
                    probes=3,
                )
            )
        results.append(completed(sample_id, "direct_flash", direct_answer))
    return results


def test_mcq_scorer_keeps_terminal_failures_in_accuracy_denominator():
    report = MCQScorer(bootstrap_resamples=1000).score(
        result_fixture(),
        gold_store(),
        categories={
            **{f"private_sample_{index}": "alpha" for index in range(5)},
            "private_sample_5": "beta",
        },
    )

    assert report.arms["bayesprobe_python"]["total"] == 6
    assert report.arms["bayesprobe_python"]["correct"] == 4
    assert report.arms["bayesprobe_python"]["accuracy"] == pytest.approx(4 / 6)
    assert report.arms["bayesprobe_python"]["terminal_failed"] == 1
    assert report.arms["direct_flash"]["correct"] == 3
    assert report.paired["both_correct"] == 2
    assert report.paired["bayesprobe_only"] == 2
    assert report.paired["direct_only"] == 1
    assert report.paired["both_wrong"] == 1
    assert report.paired["accuracy_difference"] == pytest.approx(1 / 6)
    assert 0 <= report.paired["mcnemar_exact_p_value"] <= 1


def test_mcq_scorer_reports_calibration_coverage_and_process_totals():
    report = MCQScorer(bootstrap_resamples=100).score(
        result_fixture(),
        gold_store(),
    )

    bayes = report.arms["bayesprobe_python"]
    assert bayes["calibration_coverage"] == pytest.approx(5 / 6)
    assert bayes["brier_score"] is not None
    assert bayes["log_loss"] is not None
    assert bayes["ece"] is not None
    assert bayes["mean_entropy"] is not None
    assert bayes["mean_top_two_margin"] is not None
    assert report.process_metrics["bayesprobe_python"]["cycles"] == 11
    assert report.process_metrics["bayesprobe_python"]["probes"] == 15


def test_category_accuracy_is_suppressed_below_five_selected_cases():
    report = MCQScorer(bootstrap_resamples=100).score(
        result_fixture(),
        gold_store(),
        categories={
            **{f"private_sample_{index}": "alpha" for index in range(5)},
            "private_sample_5": "beta",
        },
    )

    assert "alpha" in report.category_metrics
    assert "beta" not in report.category_metrics


def populate_store(store, sample_cases):
    by_sample_arm = {(result.sample_id, result.arm): result for result in result_fixture()}
    for case in sample_cases:
        for arm in ("direct_flash", "bayesprobe_python"):
            store.initialize_case(arm, case.sample_id)
            store.mark_running(arm, case.sample_id)
            store.write_terminal_result(by_sample_arm[(case.sample_id, arm)])


def populate_provider_telemetry(store, sample_cases):
    usage_by_arm = {
        "direct_flash": {
            "input_tokens": 100,
            "cached_input_tokens": 20,
            "reasoning_tokens": 10,
            "output_tokens": 50,
            "total_tokens": 150,
        },
        "bayesprobe_python": {
            "input_tokens": 200,
            "cached_input_tokens": 40,
            "reasoning_tokens": 20,
            "output_tokens": 100,
            "total_tokens": 300,
        },
    }
    latency_by_arm = {"direct_flash": 2.0, "bayesprobe_python": 4.0}
    for case in sample_cases:
        for arm in ("direct_flash", "bayesprobe_python"):
            path = store.paths_for(arm, case.sample_id).provider_invocations_path
            path.write_text(
                json.dumps(
                    {
                        "latency_seconds": latency_by_arm[arm],
                        "outcome": "success",
                        "usage": usage_by_arm[arm],
                    }
                )
                + "\n",
                encoding="utf-8",
            )


def test_score_service_writes_once_and_shareable_outputs_omit_raw_content(tmp_path: Path):
    sample_cases = cases()
    store = CapabilityArtifactStore(
        tmp_path / "restricted",
        identity(),
        secret=b"fixed-test-secret" * 2,
    )
    populate_store(store, sample_cases)

    paths = score_and_write_experiment(
        artifact_store=store,
        cases=sample_cases,
        gold=gold_store(),
        categories={case.sample_id: "alpha" for case in sample_cases},
        report_root=tmp_path / "reports",
        restricted_canaries=["SYNTHETIC-CANARY"],
        provider_secrets=["sk-private-secret"],
        bootstrap_resamples=100,
    )

    assert paths.score_details.exists()
    assert paths.score_marker.exists()
    assert paths.summary_json.exists()
    assert paths.summary_markdown.exists()
    shareable_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            paths.summary_json,
            paths.summary_markdown,
            paths.paired_metrics,
            paths.provenance,
        )
    )
    assert "Private synthetic question" not in shareable_text
    assert "Private model summary" not in shareable_text
    assert "private_sample_" not in shareable_text
    assert "gold_label" not in shareable_text

    with pytest.raises(ValueError, match="already been scored"):
        score_and_write_experiment(
            artifact_store=store,
            cases=sample_cases,
            gold=gold_store(),
            categories={},
            report_root=tmp_path / "reports",
            bootstrap_resamples=100,
        )


def test_score_service_reports_frozen_cost_and_per_correct_efficiency(tmp_path: Path):
    sample_cases = cases()
    store = CapabilityArtifactStore(
        tmp_path / "restricted",
        identity(),
        secret=b"fixed-test-secret" * 2,
    )
    populate_store(store, sample_cases)
    populate_provider_telemetry(store, sample_cases)
    pricing_snapshot = {
        "as_of": "2026-07-11",
        "currency": "USD",
        "rates": {
            "input_uncached_per_million_tokens": 2.0,
            "input_cached_per_million_tokens": 0.5,
            "output_per_million_tokens": 8.0,
        },
        "status": "frozen",
    }

    paths = score_and_write_experiment(
        artifact_store=store,
        cases=sample_cases,
        gold=gold_store(),
        categories={},
        report_root=tmp_path / "reports",
        pricing_snapshot=pricing_snapshot,
        bootstrap_resamples=100,
    )

    summary = json.loads(paths.summary_json.read_text(encoding="utf-8"))
    telemetry = summary["provider_telemetry"]
    assert summary["pricing_snapshot"]["rates"] == pricing_snapshot["rates"]
    assert len(summary["pricing_snapshot"]["sha256"]) == 64
    assert telemetry["direct_flash"]["estimated_cost"] == pytest.approx(0.00342)
    assert telemetry["direct_flash"]["total_tokens_per_correct"] == 300
    assert telemetry["direct_flash"]["latency_seconds_per_correct"] == 4
    assert telemetry["direct_flash"]["estimated_cost_per_correct"] == pytest.approx(
        0.00114
    )
    assert telemetry["bayesprobe_python"]["estimated_cost"] == pytest.approx(
        0.00684
    )
    assert telemetry["bayesprobe_python"]["total_tokens_per_correct"] == 450
    assert telemetry["bayesprobe_python"]["latency_seconds_per_correct"] == 6
    assert telemetry["bayesprobe_python"][
        "estimated_cost_per_correct"
    ] == pytest.approx(0.00171)
    markdown = paths.summary_markdown.read_text(encoding="utf-8")
    assert "## Resource Use" in markdown
    assert "Pricing snapshot" in markdown


def test_score_service_does_not_report_zero_cost_for_missing_success_usage(
    tmp_path: Path,
):
    sample_cases = cases()
    store = CapabilityArtifactStore(
        tmp_path / "restricted",
        identity(),
        secret=b"fixed-test-secret" * 2,
    )
    populate_store(store, sample_cases)
    populate_provider_telemetry(store, sample_cases)
    missing_usage_path = store.paths_for(
        "direct_flash", sample_cases[0].sample_id
    ).provider_invocations_path
    record = json.loads(missing_usage_path.read_text(encoding="utf-8"))
    record["usage"]["output_tokens"] = None
    missing_usage_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    paths = score_and_write_experiment(
        artifact_store=store,
        cases=sample_cases,
        gold=gold_store(),
        categories={},
        report_root=tmp_path / "reports",
        pricing_snapshot={
            "as_of": "2026-07-11",
            "currency": "USD",
            "rates": {
                "input_uncached_per_million_tokens": 2.0,
                "input_cached_per_million_tokens": 0.5,
                "output_per_million_tokens": 8.0,
            },
            "status": "frozen",
        },
        bootstrap_resamples=100,
    )

    telemetry = json.loads(paths.summary_json.read_text(encoding="utf-8"))[
        "provider_telemetry"
    ]["direct_flash"]
    assert telemetry["billable_usage_coverage"] == pytest.approx(5 / 6)
    assert telemetry["estimated_cost"] is None
    assert telemetry["estimated_cost_per_correct"] is None


def test_score_service_rejects_incomplete_or_manifest_mismatch(tmp_path: Path):
    sample_cases = cases()
    store = CapabilityArtifactStore(
        tmp_path / "restricted",
        identity(),
        secret=b"fixed-test-secret" * 2,
    )

    with pytest.raises(ValueError, match="all arm cases are terminal"):
        score_and_write_experiment(
            artifact_store=store,
            cases=sample_cases,
            gold=gold_store(),
            categories={},
            report_root=tmp_path / "reports",
            bootstrap_resamples=100,
        )

    populate_store(store, sample_cases)
    with pytest.raises(ValueError, match="manifest hash"):
        score_and_write_experiment(
            artifact_store=store,
            cases=sample_cases,
            gold=gold_store("0" * 64),
            categories={},
            report_root=tmp_path / "reports",
            bootstrap_resamples=100,
        )
