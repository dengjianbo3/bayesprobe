from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bayesprobe.evaluation.contracts import ArmCaseResult
from bayesprobe.evaluation.hle import EvaluationGoldStore
from bayesprobe.evaluation.paradigm_checkpoint import (
    CHECKPOINT_HASH_PREFIX,
    CHECKPOINT_SAMPLE_COUNT,
    SOURCE_PAIRED_COMPLETED_COUNT,
    build_paradigm_checkpoint_report,
    freeze_paradigm_checkpoint_selection,
)


def _canonical_sha256(payload):
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_source_experiment(root: Path) -> list[str]:
    sample_ids = [f"sample_{index:03d}" for index in range(100)]
    items = [
        {
            "sample_id": sample_id,
            "question": f"Question {index}?",
            "choices": {"A": "left", "B": "right"},
            "category": "synthetic",
            "answer_type": "multipleChoice",
            "dataset_revision": "a" * 40,
            "original_question_sha256": hashlib.sha256(
                f"original:{sample_id}".encode()
            ).hexdigest(),
            "canonical_question_sha256": hashlib.sha256(
                f"canonical:{sample_id}".encode()
            ).hexdigest(),
        }
        for index, sample_id in enumerate(sample_ids)
    ]
    unsigned = {
        "artifact_version": "0.1",
        "dataset_revision": "a" * 40,
        "seed": "20260711",
        "requested_sample_count": 100,
        "eligible_count": 100,
        "rejection_counts": {},
        "category_quotas": {"synthetic": 100},
        "selection_algorithm": "fixture",
        "items": items,
    }
    manifest = {**unsigned, "manifest_sha256": _canonical_sha256(unsigned)}
    root.mkdir(parents=True)
    (root / "selection_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (root / "experiment_identity.json").write_text(
        json.dumps(
            {
                "experiment_id": root.name,
                "selection_manifest_sha256": manifest["manifest_sha256"],
                "code_git_sha": "1" * 40,
            }
        ),
        encoding="utf-8",
    )
    (root / "gold_store.json").write_text(
        json.dumps(
            {
                "artifact_version": "0.1",
                "manifest_sha256": manifest["manifest_sha256"],
                "items": [
                    {"sample_id": sample_id, "gold_label": "A"}
                    for sample_id in sample_ids
                ],
            }
        ),
        encoding="utf-8",
    )

    for arm in ("direct_flash", "bayesprobe_python"):
        for index, sample_id in enumerate(sample_ids[:SOURCE_PAIRED_COMPLETED_COUNT]):
            case_root = root / "arms" / arm / f"case_{index:03d}"
            case_root.mkdir(parents=True)
            result = ArmCaseResult(
                sample_id=sample_id,
                arm=arm,
                state="completed",
                answer_label="A",
                probabilities={"A": 0.75, "B": 0.25},
            )
            (case_root / "result.json").write_text(
                json.dumps(result.__dict__), encoding="utf-8"
            )
        failed_root = root / "arms" / arm / "terminal_failed_fixture"
        failed_root.mkdir(parents=True)
        failed = ArmCaseResult(
            sample_id=sample_ids[SOURCE_PAIRED_COMPLETED_COUNT],
            arm=arm,
            state="terminal_failed",
            answer_label=None,
            probabilities=None,
            error_category="fixture_failure",
        )
        (failed_root / "result.json").write_text(
            json.dumps(failed.__dict__), encoding="utf-8"
        )
    return sample_ids


def test_freeze_selects_exactly_30_from_paired_77_without_reading_gold(tmp_path: Path):
    source = tmp_path / "source"
    sample_ids = _write_source_experiment(source)
    (source / "gold_store.json").write_text("not valid json", encoding="utf-8")
    freeze_path = tmp_path / "restricted" / "selection_freeze.json"

    frozen = freeze_paradigm_checkpoint_selection(source, freeze_path)

    expected = tuple(
        sorted(
            sample_ids[:SOURCE_PAIRED_COMPLETED_COUNT],
            key=lambda sample_id: hashlib.sha256(
                f"{CHECKPOINT_HASH_PREFIX}{sample_id}".encode("utf-8")
            ).hexdigest(),
        )[:CHECKPOINT_SAMPLE_COUNT]
    )
    assert frozen.sample_ids == expected
    assert frozen.source_paired_completed_count == SOURCE_PAIRED_COMPLETED_COUNT
    assert freeze_path.exists()
    payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    assert tuple(payload["sample_ids"]) == expected
    assert payload["selection_rule"] == (
        'sort sha256("paradigm-conformance-v3:" + sample_id), take first 30'
    )
    assert len(payload["source_direct_results_sha256"]) == 64
    assert len(payload["source_experiment_identity_sha256"]) == 64
    assert payload["source_population_policy"] == "completed/completed"


def test_existing_freeze_rejects_changed_reused_direct_result(tmp_path: Path):
    source = tmp_path / "source"
    _write_source_experiment(source)
    freeze_path = tmp_path / "restricted" / "selection_freeze.json"
    frozen = freeze_paradigm_checkpoint_selection(source, freeze_path)
    result_path = next(
        path
        for path in (source / "arms" / "direct_flash").glob("*/result.json")
        if json.loads(path.read_text(encoding="utf-8"))["sample_id"]
        == frozen.sample_ids[0]
    )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["answer_summary"] = "Changed after freezing."
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="existing paradigm checkpoint freeze"):
        freeze_paradigm_checkpoint_selection(source, freeze_path)


def test_prepare_freezes_before_gold_and_reuses_only_direct_results(
    tmp_path: Path,
    monkeypatch,
):
    from bayesprobe.evaluation import paradigm_checkpoint as checkpoint

    source = tmp_path / "source"
    _write_source_experiment(source)
    restricted_root = tmp_path / "restricted"
    report_root = tmp_path / "reports"
    config_path = tmp_path / "pilot.json"
    config_path.write_text(
        json.dumps(
            {
                "experiment_name": "Synthetic checkpoint",
                "dataset": {"revision": "a" * 40},
                "paths": {
                    "restricted_root": str(restricted_root),
                    "report_root": str(report_root),
                },
                "python": {"image": "fixture:v0.1"},
                "prompt_registry": {
                    "version": "v0.1",
                    "prompts": {"fixture": {"version": "v0.1"}},
                },
                "pricing_snapshot": {
                    "status": "frozen",
                    "as_of": "2026-07-14",
                    "currency": "USD",
                    "rates": {
                        "input_uncached_per_million_tokens": 1,
                        "input_cached_per_million_tokens": 1,
                        "output_per_million_tokens": 1,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    preflight_events = []

    def clean_git(path):
        preflight_events.append("git")
        return "c" * 40

    def require_ignored(repository, path):
        preflight_events.append("ignored")

    monkeypatch.setattr(checkpoint, "_clean_git_sha", clean_git)
    monkeypatch.setattr(checkpoint, "_require_ignored_path", require_ignored)
    monkeypatch.setattr(
        checkpoint.DockerPythonSandbox,
        "preflight",
        lambda self: SimpleNamespace(digest="sha256:" + "d" * 64),
    )
    real_load_gold = checkpoint._load_gold_store

    def assert_frozen_before_gold(path):
        assert preflight_events == ["git", "ignored"]
        assert (
            restricted_root / "paradigm-conformance-v3-selection-freeze.json"
        ).exists()
        return real_load_gold(path)

    monkeypatch.setattr(checkpoint, "_load_gold_store", assert_frozen_before_gold)

    message = checkpoint.prepare_paradigm_checkpoint(config_path, source)

    experiment_dirs = [path for path in restricted_root.iterdir() if path.is_dir()]
    assert len(experiment_dirs) == 1
    experiment = experiment_dirs[0]
    assert experiment.name in message
    manifest = json.loads(
        (experiment / "selection_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["requested_sample_count"] == 30
    assert len(manifest["items"]) == 30
    assert len(list(experiment.glob("arms/direct_flash/*/result.json"))) == 30
    assert len(list(experiment.glob("arms/bayesprobe_python/*/result.json"))) == 0
    identity = checkpoint._load_identity(experiment)
    config = checkpoint._load_config_snapshot(experiment)
    checkpoint._validate_checkpoint_artifacts(experiment, identity, config)
    policy_path = experiment / "checkpoint_policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["direct_result_policy"] = "tampered"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(ValueError, match="config or policy"):
        checkpoint._validate_checkpoint_artifacts(experiment, identity, config)
    policy["direct_result_policy"] = "reuse_frozen_source_result"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    store = checkpoint.CapabilityArtifactStore(experiment.parent, identity)
    freeze = json.loads(
        (experiment / "checkpoint_selection_freeze.json").read_text(
            encoding="utf-8"
        )
    )
    checkpoint._validate_reused_direct_results(
        store,
        tuple(freeze["sample_ids"]),
        expected_sha256=freeze["source_direct_results_sha256"],
    )
    first_result = next(experiment.glob("arms/direct_flash/*/result.json"))
    first_result.unlink()
    with pytest.raises(ValueError, match="reused Direct results are incomplete"):
        checkpoint._validate_reused_direct_results(
            store,
            tuple(freeze["sample_ids"]),
            expected_sha256=freeze["source_direct_results_sha256"],
        )


def test_checkpoint_report_compares_cycle_one_final_and_method_invariants():
    labels = {
        f"sample_{index:03d}": "A" if index % 2 == 0 else "B"
        for index in range(CHECKPOINT_SAMPLE_COUNT)
    }
    gold = EvaluationGoldStore(manifest_sha256="b" * 64, labels=labels)
    results = []
    for index, (sample_id, gold_label) in enumerate(labels.items()):
        results.append(
            ArmCaseResult(
                sample_id=sample_id,
                arm="direct_flash",
                state="completed",
                answer_label="A",
                probabilities={"A": 0.75, "B": 0.25},
            )
        )
        final_label = gold_label if index < 10 else "A"
        results.append(
            ArmCaseResult(
                sample_id=sample_id,
                arm="bayesprobe_python",
                state="completed",
                answer_label=final_label,
                probabilities={
                    "A": 0.75 if final_label == "A" else 0.25,
                    "B": 0.75 if final_label == "B" else 0.25,
                },
                process_metrics={
                    "cycles": 4,
                    "cycle_one_answer": "A",
                    "cycle_four_equivalent_answer": final_label,
                    "new_evidence_roots": 1,
                    "revised_evidence_roots": 1,
                    "retracted_evidence_roots": 0,
                    "unchanged_evidence_roots": 2,
                    "falsification_cycles": 1,
                    "epistemic_stagnation": index < 15,
                    "same_root_posterior_drift_violations": 0,
                    "no_change_confidence_increases": 0,
                },
            )
        )

    report = build_paradigm_checkpoint_report(
        results,
        gold,
        order_invariance_verified=True,
    )

    assert report["accuracy"] == {
        "direct_reused": 0.5,
        "bayesprobe_cycle_one": 0.5,
        "bayesprobe_final": 20 / 30,
    }
    assert report["answer_change_matrix"] == {"A->A": 25, "A->B": 5}
    assert report["correctness_transitions"]["wrong_to_correct"] == 5
    assert report["correctness_transitions"]["correct_to_wrong"] == 0
    assert report["root_counts"] == {
        "new": 30,
        "revised": 30,
        "retracted": 0,
        "no_change": 60,
    }
    assert report["falsification_cycle_rate"] == 0.25
    assert report["stagnation_rate"] == 0.5
    assert report["methodology"]["passed"] is True


def test_checkpoint_report_fails_methodology_on_same_root_drift():
    labels = {f"sample_{index:03d}": "A" for index in range(30)}
    gold = EvaluationGoldStore(manifest_sha256="b" * 64, labels=labels)
    results = []
    for index, sample_id in enumerate(labels):
        results.extend(
            [
                ArmCaseResult(
                    sample_id=sample_id,
                    arm="direct_flash",
                    state="completed",
                    answer_label="A",
                    probabilities={"A": 0.75, "B": 0.25},
                ),
                ArmCaseResult(
                    sample_id=sample_id,
                    arm="bayesprobe_python",
                    state="completed",
                    answer_label="A",
                    probabilities={"A": 0.75, "B": 0.25},
                    process_metrics={
                        "cycles": 1,
                        "cycle_one_answer": "A",
                        "cycle_four_equivalent_answer": "A",
                        "new_evidence_roots": 1,
                        "revised_evidence_roots": 0,
                        "retracted_evidence_roots": 0,
                        "unchanged_evidence_roots": 0,
                        "falsification_cycles": 0,
                        "epistemic_stagnation": False,
                        "same_root_posterior_drift_violations": int(index == 0),
                        "no_change_confidence_increases": 0,
                    },
                ),
            ]
        )

    report = build_paradigm_checkpoint_report(
        results,
        gold,
        order_invariance_verified=True,
    )

    assert report["methodology"]["same_root_posterior_drift_violations"] == 1
    assert report["methodology"]["passed"] is False


def test_checkpoint_report_rejects_cycle_four_equivalent_mismatch():
    labels = {f"sample_{index:03d}": "A" for index in range(30)}
    gold = EvaluationGoldStore(manifest_sha256="b" * 64, labels=labels)
    results = []
    for index, sample_id in enumerate(labels):
        results.extend(
            [
                ArmCaseResult(
                    sample_id=sample_id,
                    arm="direct_flash",
                    state="completed",
                    answer_label="A",
                    probabilities={"A": 0.75, "B": 0.25},
                ),
                ArmCaseResult(
                    sample_id=sample_id,
                    arm="bayesprobe_python",
                    state="completed",
                    answer_label="A",
                    probabilities={"A": 0.75, "B": 0.25},
                    process_metrics={
                        "cycles": 1,
                        "cycle_one_answer": "A",
                        "cycle_four_equivalent_answer": "B" if index == 0 else "A",
                        "new_evidence_roots": 1,
                        "revised_evidence_roots": 0,
                        "retracted_evidence_roots": 0,
                        "unchanged_evidence_roots": 0,
                        "falsification_cycles": 0,
                        "epistemic_stagnation": False,
                        "same_root_posterior_drift_violations": 0,
                        "no_change_confidence_increases": 0,
                    },
                ),
            ]
        )

    report = build_paradigm_checkpoint_report(
        results,
        gold,
        order_invariance_verified=True,
    )

    assert report["methodology"]["cycle_four_equivalent_mismatches"] == 1
    assert report["methodology"]["passed"] is False
