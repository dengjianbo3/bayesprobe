from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from bayesprobe.evaluation.arms import BayesProbePythonArm
from bayesprobe.evaluation.artifacts import (
    CapabilityArtifactStore,
    _atomic_private_json,
)
from bayesprobe.evaluation.config import (
    CapabilityExperimentConfig,
    capability_config_from_mapping,
    load_capability_config,
)
from bayesprobe.evaluation.contracts import ArmCaseResult, EvaluationCase
from bayesprobe.evaluation.hle import EvaluationGoldStore
from bayesprobe.evaluation.python_probe import (
    DockerPythonSandbox,
    ResolvedSandboxImage,
)
from bayesprobe.evaluation.runner import (
    CapabilityExperimentRunner,
    ExperimentIdentity,
    build_experiment_identity,
)
from bayesprobe.evaluation.scoring import (
    assert_shareable_payload_safe,
    score_and_write_experiment,
)
from bayesprobe.evidence_roots import EvidenceRootReconciler
from bayesprobe.model_gateway import build_model_gateway
from bayesprobe.schemas import (
    EpistemicOrigin,
    EvidenceEvent,
    EvidenceMemorySnapshot,
    EvidenceType,
    LikelihoodBand,
)


CHECKPOINT_HASH_PREFIX = "paradigm-conformance-v3:"
CHECKPOINT_SAMPLE_COUNT = 30
SOURCE_PAIRED_COMPLETED_COUNT = 77
CHECKPOINT_POLICY_ID = "paradigm_conformance_v3_checkpoint_v0.1"


@dataclass(frozen=True)
class FrozenParadigmCheckpointSelection:
    source_experiment_id: str
    source_experiment_identity_sha256: str
    source_selection_manifest_sha256: str
    source_paired_completed_count: int
    sample_ids: tuple[str, ...]
    selection_sha256: str
    source_direct_results_sha256: str


def freeze_paradigm_checkpoint_selection(
    source_experiment: str | Path,
    freeze_path: str | Path,
) -> FrozenParadigmCheckpointSelection:
    source = Path(source_experiment)
    manifest = _load_verified_manifest(source / "selection_manifest.json")
    source_identity = _load_json_object(
        source / "experiment_identity.json",
        owner="source experiment identity",
    )
    source_experiment_id = source_identity.get("experiment_id")
    if not isinstance(source_experiment_id, str) or not source_experiment_id:
        raise ValueError("source experiment identity has no experiment id")
    if source_identity.get("selection_manifest_sha256") != manifest.get(
        "manifest_sha256"
    ):
        raise ValueError("source experiment identity does not match source manifest")
    source_experiment_identity_sha256 = _canonical_sha256(source_identity)
    items = manifest.get("items")
    if not isinstance(items, list):
        raise ValueError("source selection manifest items must be an array")
    manifest_ids = {
        str(item.get("sample_id"))
        for item in items
        if isinstance(item, Mapping) and isinstance(item.get("sample_id"), str)
    }
    completed = {
        arm: _completed_results(source, arm)
        for arm in ("direct_flash", "bayesprobe_python")
    }
    paired_ids = set(completed["direct_flash"]).intersection(
        completed["bayesprobe_python"]
    )
    if len(paired_ids) != SOURCE_PAIRED_COMPLETED_COUNT:
        raise ValueError(
            "paradigm checkpoint requires exactly 77 paired completed source cases"
        )
    if not paired_ids.issubset(manifest_ids):
        raise ValueError("paired result sample is absent from source manifest")

    selected = tuple(
        sorted(
            paired_ids,
            key=lambda sample_id: hashlib.sha256(
                f"{CHECKPOINT_HASH_PREFIX}{sample_id}".encode("utf-8")
            ).hexdigest(),
        )[:CHECKPOINT_SAMPLE_COUNT]
    )
    selection_sha256 = _canonical_sha256(list(selected))
    source_direct_results_sha256 = _direct_results_sha256(
        completed["direct_flash"],
        selected,
    )
    payload = {
        "artifact_version": "0.1",
        "checkpoint_policy_id": CHECKPOINT_POLICY_ID,
        "source_experiment_id": source_experiment_id,
        "source_experiment_identity_sha256": source_experiment_identity_sha256,
        "source_selection_manifest_sha256": manifest["manifest_sha256"],
        "source_population_policy": "completed/completed",
        "source_paired_completed_count": len(paired_ids),
        "sample_count": len(selected),
        "hash_prefix": CHECKPOINT_HASH_PREFIX,
        "selection_rule": (
            'sort sha256("paradigm-conformance-v3:" + sample_id), take first 30'
        ),
        "sample_ids": list(selected),
        "selection_sha256": selection_sha256,
        "source_direct_results_sha256": source_direct_results_sha256,
    }
    target = Path(freeze_path)
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("existing paradigm checkpoint freeze does not match source")
    else:
        _atomic_private_json(target, payload)
    return FrozenParadigmCheckpointSelection(
        source_experiment_id=source_experiment_id,
        source_experiment_identity_sha256=source_experiment_identity_sha256,
        source_selection_manifest_sha256=str(manifest["manifest_sha256"]),
        source_paired_completed_count=len(paired_ids),
        sample_ids=selected,
        selection_sha256=selection_sha256,
        source_direct_results_sha256=source_direct_results_sha256,
    )


def build_paradigm_checkpoint_report(
    results: Sequence[ArmCaseResult],
    gold: EvaluationGoldStore,
    *,
    order_invariance_verified: bool,
) -> dict[str, Any]:
    if len(gold.labels) != CHECKPOINT_SAMPLE_COUNT:
        raise ValueError("paradigm checkpoint report requires exactly 30 gold labels")
    indexed = {(item.sample_id, item.arm): item for item in results}
    expected = {
        (sample_id, arm)
        for sample_id in gold.labels
        for arm in ("direct_flash", "bayesprobe_python")
    }
    if set(indexed) != expected:
        raise ValueError("checkpoint results must contain both arms for all 30 cases")

    direct_correct = 0
    cycle_one_correct = 0
    final_correct = 0
    change_matrix: Counter[str] = Counter()
    transitions: Counter[str] = Counter()
    root_counts: Counter[str] = Counter()
    total_cycles = 0
    falsification_cycles = 0
    stagnation_count = 0
    drift_violations = 0
    confidence_increases = 0
    cycle_four_equivalent_mismatches = 0
    required_metrics = {
        "cycles",
        "cycle_one_answer",
        "cycle_four_equivalent_answer",
        "new_evidence_roots",
        "revised_evidence_roots",
        "retracted_evidence_roots",
        "unchanged_evidence_roots",
        "falsification_cycles",
        "epistemic_stagnation",
        "same_root_posterior_drift_violations",
        "no_change_confidence_increases",
    }
    falsification_metrics_visible = True

    for sample_id, gold_label in gold.labels.items():
        direct = indexed[(sample_id, "direct_flash")]
        bayesprobe = indexed[(sample_id, "bayesprobe_python")]
        direct_correct += int(
            direct.state == "completed" and direct.answer_label == gold_label
        )
        metrics = bayesprobe.process_metrics
        falsification_metrics_visible = (
            falsification_metrics_visible
            and required_metrics.issubset(metrics)
        )
        cycle_one = metrics.get("cycle_one_answer")
        final = metrics.get("cycle_four_equivalent_answer")
        cycle_four_equivalent_mismatches += int(
            bayesprobe.state == "completed" and final != bayesprobe.answer_label
        )
        cycle_one_is_correct = cycle_one == gold_label
        final_is_correct = (
            bayesprobe.state == "completed"
            and final == bayesprobe.answer_label
            and final == gold_label
        )
        cycle_one_correct += int(cycle_one_is_correct)
        final_correct += int(final_is_correct)
        if isinstance(cycle_one, str) and isinstance(final, str):
            change_matrix[f"{cycle_one}->{final}"] += 1
        if cycle_one_is_correct and final_is_correct:
            transitions["correct_to_correct"] += 1
        elif cycle_one_is_correct:
            transitions["correct_to_wrong"] += 1
        elif final_is_correct:
            transitions["wrong_to_correct"] += 1
        else:
            transitions["wrong_to_wrong"] += 1

        root_counts["new"] += _metric_int(metrics, "new_evidence_roots")
        root_counts["revised"] += _metric_int(metrics, "revised_evidence_roots")
        root_counts["retracted"] += _metric_int(
            metrics, "retracted_evidence_roots"
        )
        root_counts["no_change"] += _metric_int(
            metrics, "unchanged_evidence_roots"
        )
        total_cycles += _metric_int(metrics, "cycles")
        falsification_cycles += _metric_int(metrics, "falsification_cycles")
        stagnation_count += int(metrics.get("epistemic_stagnation") is True)
        drift_violations += _metric_int(
            metrics, "same_root_posterior_drift_violations"
        )
        confidence_increases += _metric_int(
            metrics, "no_change_confidence_increases"
        )

    methodology = {
        "same_root_posterior_drift_violations": drift_violations,
        "no_change_confidence_increases": confidence_increases,
        "cycle_four_equivalent_mismatches": cycle_four_equivalent_mismatches,
        "order_invariant_same_cycle_reconciliation": bool(
            order_invariance_verified
        ),
        "falsification_metrics_visible": falsification_metrics_visible,
    }
    methodology["passed"] = (
        drift_violations == 0
        and confidence_increases == 0
        and cycle_four_equivalent_mismatches == 0
        and bool(order_invariance_verified)
        and falsification_metrics_visible
    )
    total = len(gold.labels)
    return {
        "artifact_version": "0.1",
        "checkpoint_policy_id": CHECKPOINT_POLICY_ID,
        "sample_count": total,
        "accuracy": {
            "direct_reused": direct_correct / total,
            "bayesprobe_cycle_one": cycle_one_correct / total,
            "bayesprobe_final": final_correct / total,
        },
        "answer_change_matrix": dict(sorted(change_matrix.items())),
        "correctness_transitions": {
            key: transitions[key]
            for key in (
                "correct_to_correct",
                "correct_to_wrong",
                "wrong_to_correct",
                "wrong_to_wrong",
            )
        },
        "root_counts": {
            key: root_counts[key]
            for key in ("new", "revised", "retracted", "no_change")
        },
        "falsification_cycle_rate": (
            falsification_cycles / total_cycles if total_cycles else 0.0
        ),
        "stagnation_rate": stagnation_count / total,
        "methodology": methodology,
    }


def prepare_paradigm_checkpoint(
    config_path: str | Path,
    source_experiment: str | Path,
) -> str:
    config = load_capability_config(config_path)
    source = Path(source_experiment)
    repository = _repository_root()
    code_git_sha = _clean_git_sha(repository)
    _require_ignored_path(repository, config.restricted_root)
    freeze_path = config.restricted_root / (
        "paradigm-conformance-v3-selection-freeze.json"
    )
    frozen = freeze_paradigm_checkpoint_selection(source, freeze_path)
    source_manifest = _load_verified_manifest(source / "selection_manifest.json")
    if (
        source_manifest["manifest_sha256"]
        != frozen.source_selection_manifest_sha256
    ):
        raise ValueError("source manifest changed after checkpoint selection freeze")
    source_gold = _load_gold_store(source / "gold_store.json")
    if source_gold.manifest_sha256 != frozen.source_selection_manifest_sha256:
        raise ValueError("source gold store does not match source selection manifest")
    if not set(frozen.sample_ids).issubset(source_gold.labels):
        raise ValueError("source gold store is incomplete for frozen selection")
    selected_set = set(frozen.sample_ids)
    selected_items = [
        item
        for item in source_manifest["items"]
        if item.get("sample_id") in selected_set
    ]
    selected_items.sort(key=lambda item: frozen.sample_ids.index(item["sample_id"]))
    if len(selected_items) != CHECKPOINT_SAMPLE_COUNT:
        raise ValueError("frozen checkpoint items are incomplete")
    subset_manifest = _checkpoint_manifest(source_manifest, selected_items, frozen)
    subset_gold = EvaluationGoldStore(
        manifest_sha256=subset_manifest["manifest_sha256"],
        labels={sample_id: source_gold.labels[sample_id] for sample_id in frozen.sample_ids},
    )

    image = DockerPythonSandbox(config.python_sandbox).preflight()
    checkpoint_policy = _checkpoint_policy(frozen)
    checkpoint_config_sha = _checkpoint_config_sha(config, checkpoint_policy)
    identity = build_experiment_identity(
        experiment_name=f"{config.experiment_name} paradigm conformance v3 checkpoint",
        code_git_sha=code_git_sha,
        dataset_revision_sha=str(subset_manifest["dataset_revision"]),
        selection_manifest_sha256=str(subset_manifest["manifest_sha256"]),
        config_sha256=checkpoint_config_sha,
        prompt_registry_sha256=config.prompt_registry_sha256,
        python_image_digest=image.digest,
    )
    store = CapabilityArtifactStore(config.restricted_root, identity)
    if (store.root / "selection_manifest.json").exists():
        raise ValueError("paradigm checkpoint has already been prepared")
    _atomic_private_json(
        store.root / "gold_store.json",
        {
            "artifact_version": "0.1",
            "manifest_sha256": subset_gold.manifest_sha256,
            "items": [
                {"sample_id": sample_id, "gold_label": gold_label}
                for sample_id, gold_label in subset_gold.labels.items()
            ],
        },
    )
    _atomic_private_json(store.root / "config_snapshot.json", config.snapshot())
    _atomic_private_json(store.root / "checkpoint_selection_freeze.json", asdict(frozen))
    _atomic_private_json(store.root / "checkpoint_policy.json", checkpoint_policy)
    source_direct = _completed_results(source, "direct_flash")
    _seed_reused_direct_results(store, source_direct, frozen.sample_ids)
    _validate_reused_direct_results(
        store,
        frozen.sample_ids,
        expected_sha256=frozen.source_direct_results_sha256,
    )
    # The manifest is the preparation-complete marker and is written last.
    _atomic_private_json(store.root / "selection_manifest.json", subset_manifest)
    return (
        "BayesProbe paradigm checkpoint prepared: "
        f"experiment={identity.experiment_id} samples={CHECKPOINT_SAMPLE_COUNT} "
        f"path={store.root}"
    )


def run_paradigm_checkpoint(experiment_path: str | Path) -> str:
    path = Path(experiment_path)
    identity = _load_identity(path)
    config = _load_config_snapshot(path)
    selection = _load_checkpoint_cases(path)
    image = _validate_checkpoint_runtime(path, identity, config)
    store = CapabilityArtifactStore(path.parent, identity)
    sandbox = DockerPythonSandbox(config.python_sandbox)
    observer = store.provider_observer()
    gateway = build_model_gateway(
        config.model_gateway,
        invocation_observer=observer,
    )
    direct = _ReusedDirectArm()
    bayesprobe = BayesProbePythonArm(
        gateway,
        sandbox,
        image=image,
        invocation_metadata={"experiment_id": identity.experiment_id},
        ledger_factory=lambda case: store.ledger_for("bayesprobe_python", case),
        execution_observer_factory=lambda case: store.python_observer_for(
            "bayesprobe_python", case
        ),
    )
    summary = CapabilityExperimentRunner(
        identity=identity,
        cases=list(selection),
        arms={"direct_flash": direct, "bayesprobe_python": bayesprobe},
        artifact_store=store,
        direct_concurrency=config.direct_concurrency,
        bayesprobe_concurrency=config.bayesprobe_concurrency,
    ).run()
    return (
        "BayesProbe paradigm checkpoint run complete: "
        f"experiment={identity.experiment_id} terminal={summary.terminal_count}/60 "
        f"executed={summary.executed_count}"
    )


class _ReusedDirectArm:
    arm_name = "direct_flash"

    def run_case(self, case: EvaluationCase) -> ArmCaseResult:
        raise RuntimeError("reused Direct checkpoint arm must never be scheduled")


def score_paradigm_checkpoint(experiment_path: str | Path) -> str:
    path = Path(experiment_path)
    identity = _load_identity(path)
    config = _load_config_snapshot(path)
    _validate_checkpoint_artifacts(path, identity, config)
    cases = list(_load_checkpoint_cases(path))
    gold = _load_gold_store(path / "gold_store.json")
    store = CapabilityArtifactStore(path.parent, identity)
    categories = _manifest_categories(path / "selection_manifest.json")
    provider_secret = os.environ.get(config.model_gateway.api_key_env)
    results = [
        ArmCaseResult(**store.load_result(arm, case.sample_id))
        for case in cases
        for arm in ("direct_flash", "bayesprobe_python")
    ]
    report = build_paradigm_checkpoint_report(
        results,
        gold,
        order_invariance_verified=_verify_order_invariance(),
    )
    assert_shareable_payload_safe(
        report,
        restricted_values=[
            value
            for case in cases
            for value in (case.sample_id, case.question, *case.choices.values())
        ],
        canaries=(),
        provider_secrets=[provider_secret] if provider_secret else [],
    )
    report_path = (
        config.report_root / identity.experiment_id / "paradigm_checkpoint.json"
    )
    _atomic_private_json(report_path, report)
    score_and_write_experiment(
        artifact_store=store,
        cases=cases,
        gold=gold,
        categories=categories,
        report_root=config.report_root,
        provider_secrets=[provider_secret] if provider_secret else [],
        pricing_snapshot=config.pricing_snapshot,
    )
    return (
        "BayesProbe paradigm checkpoint scored: "
        f"experiment={identity.experiment_id} report={report_path} "
        f"methodology_passed={report['methodology']['passed']}"
    )


def _completed_results(source: Path, arm: str) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    arm_root = source / "arms" / arm
    if not arm_root.exists():
        return results
    for path in sorted(arm_root.glob("*/result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("state") != "completed":
            continue
        sample_id = payload.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("completed source result has no sample id")
        if sample_id in results:
            raise ValueError("duplicate completed source result")
        results[sample_id] = payload
    return results


def _direct_results_sha256(
    results: Mapping[str, Mapping[str, Any]],
    sample_ids: Sequence[str],
) -> str:
    if not set(sample_ids).issubset(results):
        raise ValueError("reused Direct source results are incomplete")
    normalized = {
        sample_id: asdict(ArmCaseResult(**dict(results[sample_id])))
        for sample_id in sample_ids
    }
    return _canonical_sha256(normalized)


def _seed_reused_direct_results(
    store: CapabilityArtifactStore,
    source_results: Mapping[str, Mapping[str, Any]],
    sample_ids: Sequence[str],
) -> None:
    if not set(sample_ids).issubset(source_results):
        raise ValueError("reused Direct source results are incomplete")
    for sample_id in sample_ids:
        expected = ArmCaseResult(**dict(source_results[sample_id]))
        status = store.status("direct_flash", sample_id)
        if status.get("state") == "completed":
            existing = ArmCaseResult(
                **store.load_result("direct_flash", sample_id)
            )
            if existing != expected:
                raise ValueError("existing reused Direct result changed during prepare")
            continue
        if status.get("state") == "terminal_failed":
            raise ValueError("existing reused Direct result is terminal_failed")
        store.mark_running("direct_flash", sample_id)
        store.write_terminal_result(expected)


def _validate_reused_direct_results(
    store: CapabilityArtifactStore,
    sample_ids: Sequence[str],
    *,
    expected_sha256: str,
) -> None:
    results: dict[str, Mapping[str, Any]] = {}
    for sample_id in sample_ids:
        paths = store.paths_for("direct_flash", sample_id)
        if not paths.status_path.exists() or not paths.result_path.exists():
            raise ValueError("reused Direct results are incomplete")
        status = _load_json_object(paths.status_path, owner="Direct result status")
        if status.get("state") != "completed":
            raise ValueError("reused Direct results are incomplete")
        payload = _load_json_object(paths.result_path, owner="Direct result")
        result = ArmCaseResult(**payload)
        if result.arm != "direct_flash" or result.sample_id != sample_id:
            raise ValueError("reused Direct result identity is invalid")
        results[sample_id] = payload
    if _direct_results_sha256(results, sample_ids) != expected_sha256:
        raise ValueError("reused Direct result digest does not match freeze")


def _checkpoint_policy(
    frozen: FrozenParadigmCheckpointSelection,
) -> dict[str, Any]:
    return {
        "artifact_version": "0.1",
        "checkpoint_policy_id": CHECKPOINT_POLICY_ID,
        "source_experiment_id": frozen.source_experiment_id,
        "source_experiment_identity_sha256": (
            frozen.source_experiment_identity_sha256
        ),
        "source_selection_manifest_sha256": (
            frozen.source_selection_manifest_sha256
        ),
        "source_population_policy": "completed/completed",
        "source_paired_completed_count": frozen.source_paired_completed_count,
        "selection_sha256": frozen.selection_sha256,
        "source_direct_results_sha256": frozen.source_direct_results_sha256,
        "direct_result_policy": "reuse_frozen_source_result",
        "bayesprobe_result_policy": "rerun_current_kernel",
    }


def _checkpoint_config_sha(
    config: CapabilityExperimentConfig,
    policy: Mapping[str, Any],
) -> str:
    return _canonical_sha256(
        {
            "source_config_sha256": config.config_sha256,
            "checkpoint_policy": dict(policy),
        }
    )


def _load_verified_manifest(path: Path) -> dict[str, Any]:
    payload = _load_json_object(path, owner="selection manifest")
    claimed = payload.get("manifest_sha256")
    unsigned = dict(payload)
    unsigned.pop("manifest_sha256", None)
    if claimed != _canonical_sha256(unsigned):
        raise ValueError("selection manifest hash does not match content")
    return payload


def _checkpoint_manifest(
    source_manifest: Mapping[str, Any],
    selected_items: list[Mapping[str, Any]],
    frozen: FrozenParadigmCheckpointSelection,
) -> dict[str, Any]:
    category_counts = Counter(str(item["category"]) for item in selected_items)
    unsigned = {
        "artifact_version": "0.1",
        "dataset_revision": source_manifest["dataset_revision"],
        "seed": CHECKPOINT_HASH_PREFIX.rstrip(":"),
        "requested_sample_count": CHECKPOINT_SAMPLE_COUNT,
        "eligible_count": frozen.source_paired_completed_count,
        "rejection_counts": {},
        "category_quotas": dict(sorted(category_counts.items())),
        "selection_algorithm": "paired_completed_sha256_v3",
        "source_experiment_id": frozen.source_experiment_id,
        "source_experiment_identity_sha256": (
            frozen.source_experiment_identity_sha256
        ),
        "source_selection_manifest_sha256": (
            frozen.source_selection_manifest_sha256
        ),
        "source_population_policy": "completed/completed",
        "items": [dict(item) for item in selected_items],
    }
    return {**unsigned, "manifest_sha256": _canonical_sha256(unsigned)}


def _load_gold_store(path: Path) -> EvaluationGoldStore:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items") if isinstance(payload, Mapping) else None
    if not isinstance(items, list):
        raise ValueError("gold store items must be an array")
    labels = {
        str(item["sample_id"]): str(item["gold_label"])
        for item in items
        if isinstance(item, Mapping)
    }
    if len(labels) != len(items):
        raise ValueError("gold store contains an invalid item")
    return EvaluationGoldStore(
        manifest_sha256=str(payload["manifest_sha256"]),
        labels=labels,
    )


def _load_checkpoint_cases(path: Path) -> tuple[EvaluationCase, ...]:
    manifest = _load_verified_manifest(path / "selection_manifest.json")
    items = manifest.get("items")
    if not isinstance(items, list) or len(items) != CHECKPOINT_SAMPLE_COUNT:
        raise ValueError("checkpoint selection must contain exactly 30 items")
    return tuple(
        EvaluationCase(
            sample_id=str(item["sample_id"]),
            question=str(item["question"]),
            choices=dict(item["choices"]),
        )
        for item in items
    )


def _manifest_categories(path: Path) -> dict[str, str]:
    manifest = _load_verified_manifest(path)
    return {
        str(item["sample_id"]): str(item["category"])
        for item in manifest["items"]
    }


def _load_identity(path: Path) -> ExperimentIdentity:
    payload = json.loads((path / "experiment_identity.json").read_text(encoding="utf-8"))
    return ExperimentIdentity(**payload)


def _load_config_snapshot(path: Path) -> CapabilityExperimentConfig:
    payload = json.loads((path / "config_snapshot.json").read_text(encoding="utf-8"))
    return capability_config_from_mapping(payload)


def _validate_checkpoint_artifacts(
    path: Path,
    identity: ExperimentIdentity,
    config: CapabilityExperimentConfig,
) -> None:
    if path.name != identity.experiment_id:
        raise ValueError("checkpoint directory does not match experiment identity")
    manifest = _load_verified_manifest(path / "selection_manifest.json")
    if manifest["manifest_sha256"] != identity.selection_manifest_sha256:
        raise ValueError("checkpoint manifest does not match experiment identity")
    if manifest.get("dataset_revision") != identity.dataset_revision_sha:
        raise ValueError("checkpoint dataset revision does not match identity")
    if config.prompt_registry_sha256 != identity.prompt_registry_sha256:
        raise ValueError("checkpoint prompt registry does not match identity")

    policy = _load_json_object(
        path / "checkpoint_policy.json",
        owner="checkpoint policy",
    )
    if policy.get("checkpoint_policy_id") != CHECKPOINT_POLICY_ID:
        raise ValueError("checkpoint policy snapshot is invalid")
    if _checkpoint_config_sha(config, policy) != identity.config_sha256:
        raise ValueError("checkpoint config or policy does not match identity")
    freeze = _load_json_object(
        path / "checkpoint_selection_freeze.json",
        owner="checkpoint selection freeze",
    )
    items = manifest.get("items")
    if not isinstance(items, list):
        raise ValueError("checkpoint manifest items must be an array")
    sample_ids = tuple(str(item["sample_id"]) for item in items)
    frozen_ids = freeze.get("sample_ids")
    if not isinstance(frozen_ids, list) or tuple(frozen_ids) != sample_ids:
        raise ValueError("checkpoint sample ids do not match selection freeze")
    if _canonical_sha256(list(sample_ids)) != freeze.get("selection_sha256"):
        raise ValueError("checkpoint selection digest does not match sample ids")
    bound_fields = (
        "source_experiment_id",
        "source_experiment_identity_sha256",
        "source_selection_manifest_sha256",
        "source_paired_completed_count",
        "selection_sha256",
        "source_direct_results_sha256",
    )
    if any(freeze.get(field) != policy.get(field) for field in bound_fields):
        raise ValueError("checkpoint freeze does not match policy snapshot")
    if manifest.get("source_experiment_id") != policy.get("source_experiment_id"):
        raise ValueError("checkpoint manifest does not match source experiment")
    if manifest.get("source_experiment_identity_sha256") != policy.get(
        "source_experiment_identity_sha256"
    ):
        raise ValueError("checkpoint manifest does not match source identity")
    if manifest.get("source_selection_manifest_sha256") != policy.get(
        "source_selection_manifest_sha256"
    ):
        raise ValueError("checkpoint manifest does not match source selection")
    if manifest.get("source_population_policy") != "completed/completed":
        raise ValueError("checkpoint source population policy is invalid")
    store = CapabilityArtifactStore(path.parent, identity)
    _validate_reused_direct_results(
        store,
        sample_ids,
        expected_sha256=str(policy["source_direct_results_sha256"]),
    )


def _validate_checkpoint_runtime(
    path: Path,
    identity: ExperimentIdentity,
    config: CapabilityExperimentConfig,
) -> ResolvedSandboxImage:
    _validate_checkpoint_artifacts(path, identity, config)
    if not os.environ.get(config.model_gateway.api_key_env):
        raise ValueError(
            f"provider API key environment variable {config.model_gateway.api_key_env} is not set"
        )
    if _clean_git_sha(_repository_root()) != identity.code_git_sha:
        raise ValueError("checkpoint Git HEAD does not match prepared identity")
    image = DockerPythonSandbox(config.python_sandbox).preflight()
    if image.digest != identity.python_image_digest:
        raise ValueError("checkpoint Python image digest changed after preparation")
    return image


def _load_json_object(path: Path, *, owner: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{owner} must be an object")
    return payload


def _clean_git_sha(repository: Path) -> str:
    head_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    status_result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    if head_result.returncode != 0 or status_result.returncode != 0:
        raise ValueError("Git checkpoint preflight failed")
    head = head_result.stdout.strip().lower()
    status = status_result.stdout
    if status.strip():
        raise ValueError("Git worktree must be clean before checkpoint preparation or run")
    return head


def _require_ignored_path(repository: Path, path: Path) -> None:
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", str(path)],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError("restricted checkpoint path must be ignored by Git")


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _metric_int(metrics: Mapping[str, Any], name: str) -> int:
    value = metrics.get(name, 0)
    if type(value) not in (int, float) or not math.isfinite(value):
        return 0
    return int(value)


def _verify_order_invariance() -> bool:
    def event(event_id: str, band: LikelihoodBand) -> EvidenceEvent:
        return EvidenceEvent(
            schema_version="v0.2",
            id=event_id,
            derived_from_signal=f"signal_{event_id}",
            epistemic_origin=EpistemicOrigin.MODEL_REASONING,
            derivation_root_id="derivation:checkpoint-order-test",
            contribution_root_id="root:checkpoint-order-test",
            target_hypotheses=["A", "B"],
            evidence_type=EvidenceType.NEUTRAL,
            content=f"Checkpoint order test {event_id}.",
            reliability=1.0,
            independence=1.0,
            relevance=1.0,
            novelty=1.0,
            likelihoods={"A": band, "B": LikelihoodBand.NEUTRAL},
            unresolved_likelihood=None,
            correlation_status="novel",
            effective_update_weight=None,
            discard_reason=None,
        )

    events = [
        event("E1", LikelihoodBand.STRONGLY_CONFIRMING),
        event("E2", LikelihoodBand.WEAKLY_DISCONFIRMING),
    ]
    reconciler = EvidenceRootReconciler()
    forward = reconciler.reconcile_cycle(
        snapshot=EvidenceMemorySnapshot(memory_version=3),
        evidence_events=events,
        falsification_probe_executed=False,
    )
    reverse = reconciler.reconcile_cycle(
        snapshot=EvidenceMemorySnapshot(memory_version=3),
        evidence_events=list(reversed(events)),
        falsification_probe_executed=False,
    )
    return (
        forward.contribution_deltas == reverse.contribution_deltas
        and forward.evidence_events == reverse.evidence_events
    )


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "CHECKPOINT_HASH_PREFIX",
    "CHECKPOINT_SAMPLE_COUNT",
    "SOURCE_PAIRED_COMPLETED_COUNT",
    "FrozenParadigmCheckpointSelection",
    "build_paradigm_checkpoint_report",
    "freeze_paradigm_checkpoint_selection",
    "prepare_paradigm_checkpoint",
    "run_paradigm_checkpoint",
    "score_paradigm_checkpoint",
]
