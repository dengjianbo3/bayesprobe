from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from bayesprobe.evaluation.artifacts import CapabilityArtifactStore, _atomic_private_json
from bayesprobe.evaluation.config import (
    CapabilityExperimentConfig,
    capability_config_from_mapping,
    load_capability_config,
)
from bayesprobe.evaluation.contracts import ArmCaseResult
from bayesprobe.evaluation.search_arms import BayesProbeSearchArm, DirectSearchArm
from bayesprobe.evaluation.runner import (
    CapabilityExperimentRunner,
    ExperimentIdentity,
    build_experiment_identity,
)
from bayesprobe.evaluation.scoring import assert_shareable_payload_safe
from bayesprobe.model_gateway import build_model_gateway
from bayesprobe.tavily_search import TavilySearchClient


_SEARCH_ARMS = ("direct_search", "bayesprobe_search")
_BASELINE_ARMS = ("direct_no_web", "bayesprobe_no_web")
_ARTIFACT_ARMS = ("direct_search", "bayesprobe_search")
_SEARCH_POLICY = {
    "provider": "tavily",
    "topic": "general",
    "search_depth": "advanced",
    "max_search_calls": 2,
    "max_results": 5,
    "chunks_per_source": 3,
    "timeout_seconds": 60,
    "include_answer": False,
    "include_raw_content": False,
}


def prepare_search_matrix(
    config_path: str | Path,
    source_checkpoint: str | Path,
) -> str:
    config = load_capability_config(config_path)
    source = Path(source_checkpoint)
    source_identity = _load_identity(source)
    source_manifest = _load_selection(source / "selection_manifest.json")
    source_gold = _load_gold_labels(source / "gold_store.json")
    if set(source_gold) != {case.sample_id for case in source_manifest["cases"]}:
        raise ValueError("source checkpoint gold store does not match manifest")
    baseline = _baseline_correctness(source, source_gold)
    source_binding = {
        "source_checkpoint_id": source_identity.experiment_id,
        "source_checkpoint_manifest_sha256": source_manifest["manifest_sha256"],
        "source_checkpoint_identity_sha256": _canonical_sha256(
            _read_json_object(source / "experiment_identity.json")
        ),
    }
    identity = build_experiment_identity(
        experiment_name=f"{config.experiment_name} Tavily search matrix",
        code_git_sha=_git_head(),
        dataset_revision_sha=source_manifest["dataset_revision"],
        selection_manifest_sha256=source_manifest["manifest_sha256"],
        config_sha256=_canonical_sha256(
            {
                "capability_config": config.snapshot(),
                "search_policy": _SEARCH_POLICY,
                "source_binding": source_binding,
            }
        ),
        prompt_registry_sha256=config.prompt_registry_sha256,
        python_image_digest=source_identity.python_image_digest,
    )
    store = CapabilityArtifactStore(
        config.restricted_root,
        identity,
        arm_names=_ARTIFACT_ARMS,
    )
    marker = store.root / "selection_manifest.json"
    if marker.exists():
        raise ValueError("search matrix has already been prepared")
    _atomic_private_json(marker, source_manifest["payload"])
    _atomic_private_json(
        store.root / "gold_store.json",
        {
            "artifact_version": "0.1",
            "manifest_sha256": source_manifest["manifest_sha256"],
            "items": [
                {"sample_id": sample_id, "gold_label": label}
                for sample_id, label in source_gold.items()
            ],
        },
    )
    _atomic_private_json(store.root / "config_snapshot.json", config.snapshot())
    _atomic_private_json(store.root / "source_binding.json", source_binding)
    _atomic_private_json(store.root / "search_policy.json", _SEARCH_POLICY)
    _atomic_private_json(store.root / "baseline_correctness.json", baseline)
    return (
        "BayesProbe Tavily search matrix prepared: "
        f"experiment={identity.experiment_id} samples={len(source_gold)} path={store.root}"
    )


def run_search_matrix(experiment_path: str | Path) -> str:
    path = Path(experiment_path)
    identity = _load_identity(path)
    config = _load_config(path)
    selection = _load_selection(path / "selection_manifest.json")
    store = CapabilityArtifactStore(
        path.parent,
        identity,
        arm_names=_ARTIFACT_ARMS,
    )
    model_gateway = build_model_gateway(
        config.model_gateway,
        invocation_observer=store.provider_observer(),
    )
    tavily_client = TavilySearchClient()
    direct = DirectSearchArm(
        model_gateway,
        tavily_client,
        max_search_calls=_SEARCH_POLICY["max_search_calls"],
        invocation_metadata={"experiment_id": identity.experiment_id},
    )
    bayesprobe = BayesProbeSearchArm(
        model_gateway,
        tavily_client,
        max_search_calls=_SEARCH_POLICY["max_search_calls"],
        invocation_metadata={"experiment_id": identity.experiment_id},
        ledger_factory=lambda case: store.ledger_for("bayesprobe_search", case),
    )
    summary = CapabilityExperimentRunner(
        identity=identity,
        cases=selection["cases"],
        arms={"direct_search": direct, "bayesprobe_search": bayesprobe},
        artifact_store=store,
        arm_concurrency={
            "direct_search": config.direct_concurrency,
            "bayesprobe_search": config.bayesprobe_concurrency,
        },
    ).run()
    return (
        "BayesProbe Tavily search matrix run complete: "
        f"experiment={identity.experiment_id} terminal={summary.terminal_count}/"
        f"{summary.task_count} completed={summary.completed_count}"
    )


def score_search_matrix(experiment_path: str | Path) -> str:
    path = Path(experiment_path)
    identity = _load_identity(path)
    config = _load_config(path)
    selection = _load_selection(path / "selection_manifest.json")
    gold = _load_gold_labels(path / "gold_store.json")
    baseline = _read_json_object(path / "baseline_correctness.json")
    source_binding = _read_json_object(path / "source_binding.json")
    search_policy = _read_json_object(path / "search_policy.json")
    store = CapabilityArtifactStore(
        path.parent,
        identity,
        arm_names=_ARTIFACT_ARMS,
    )
    results = [
        ArmCaseResult(**store.load_result(arm, case.sample_id))
        for case in selection["cases"]
        for arm in _ARTIFACT_ARMS
    ]
    report = build_search_matrix_report(
        gold_labels=gold,
        baseline_correctness=baseline,
        search_results=results,
        source_binding=source_binding,
        search_policy=search_policy,
    )
    _assert_shareable_report_safe(
        report,
        cases=selection["cases"],
        provider_secret=os.environ.get(config.model_gateway.api_key_env),
        tavily_secret=os.environ.get("TAVILY_API_KEY"),
    )
    report_path = config.report_root / identity.experiment_id / "search_matrix.json"
    _atomic_private_json(report_path, report)
    _atomic_private_json(
        path / "search_matrix_scoring_complete.json",
        {"report_path": str(report_path)},
    )
    return f"BayesProbe Tavily search matrix scored: experiment={identity.experiment_id} report={report_path}"


def report_search_matrix(experiment_path: str | Path) -> str:
    path = Path(experiment_path)
    identity = _load_identity(path)
    config = _load_config(path)
    selection = _load_selection(path / "selection_manifest.json")
    marker = _read_json_object(path / "search_matrix_scoring_complete.json")
    report_path = Path(str(marker["report_path"]))
    report = _read_json_object(report_path)
    _assert_shareable_report_safe(
        report,
        cases=selection["cases"],
        provider_secret=os.environ.get(config.model_gateway.api_key_env),
        tavily_secret=os.environ.get("TAVILY_API_KEY"),
    )
    return f"BayesProbe Tavily search matrix report verified: experiment={identity.experiment_id} report={report_path}"


def build_search_matrix_report(
    *,
    gold_labels: Mapping[str, str],
    baseline_correctness: Mapping[str, Mapping[str, bool]],
    search_results: Sequence[ArmCaseResult],
    source_binding: Mapping[str, Any],
    search_policy: Mapping[str, Any],
) -> dict[str, Any]:
    sample_ids = set(gold_labels)
    if not sample_ids:
        raise ValueError("search matrix requires at least one gold label")
    if set(baseline_correctness) != sample_ids:
        raise ValueError("search matrix baseline must cover every gold label")
    for sample_id, baseline in baseline_correctness.items():
        if set(baseline) != set(_BASELINE_ARMS):
            raise ValueError(
                f"search matrix baseline arms are invalid for sample {sample_id}"
            )
        if not all(type(value) is bool for value in baseline.values()):
            raise ValueError("search matrix baseline correctness must be boolean")

    indexed = {(result.sample_id, result.arm): result for result in search_results}
    expected = {
        (sample_id, arm)
        for sample_id in sample_ids
        for arm in _SEARCH_ARMS
    }
    if set(indexed) != expected:
        raise ValueError("search matrix results must cover both search arms")

    correct: dict[str, int] = {arm: 0 for arm in (*_BASELINE_ARMS, *_SEARCH_ARMS)}
    direct_transitions: Counter[str] = Counter()
    bayesprobe_transitions: Counter[str] = Counter()
    paired: Counter[str] = Counter()
    coverage: dict[str, Counter[str]] = {
        arm: Counter() for arm in _SEARCH_ARMS
    }
    for sample_id, gold_label in gold_labels.items():
        baseline = baseline_correctness[sample_id]
        for arm, is_correct in baseline.items():
            correct[arm] += int(is_correct)

        search_correctness: dict[str, bool] = {}
        for arm in _SEARCH_ARMS:
            result = indexed[(sample_id, arm)]
            is_correct = result.state == "completed" and result.answer_label == gold_label
            search_correctness[arm] = is_correct
            correct[arm] += int(is_correct)
            _accumulate_search_coverage(coverage[arm], result)
        direct_transitions[
            _transition_key(baseline["direct_no_web"], search_correctness["direct_search"])
        ] += 1
        bayesprobe_transitions[
            _transition_key(
                baseline["bayesprobe_no_web"],
                search_correctness["bayesprobe_search"],
            )
        ] += 1
        paired[
            _transition_key(
                search_correctness["direct_search"],
                search_correctness["bayesprobe_search"],
            )
        ] += 1

    total = len(sample_ids)
    return {
        "artifact_version": "0.1",
        "sample_count": total,
        "source_binding": dict(source_binding),
        "search_policy": dict(search_policy),
        "accuracy": {arm: correct[arm] / total for arm in correct},
        "no_web_to_search_transitions": {
            "direct": _ordered_transitions(direct_transitions),
            "bayesprobe": _ordered_transitions(bayesprobe_transitions),
        },
        "paired_search_comparison": _ordered_transitions(paired),
        "search_coverage": {
            arm: {
                "search_calls": coverage[arm]["search_calls"],
                "search_successes": coverage[arm]["search_successes"],
                "search_failures": coverage[arm]["search_failures"],
                "empty_searches": coverage[arm]["empty_searches"],
                "search_result_count": coverage[arm]["search_result_count"],
                "cases_with_search_success": coverage[arm]["cases_with_search_success"],
            }
            for arm in _SEARCH_ARMS
        },
    }


def _accumulate_search_coverage(metrics: Counter[str], result: ArmCaseResult) -> None:
    process = result.process_metrics
    for key in (
        "search_calls",
        "search_successes",
        "search_failures",
        "empty_searches",
        "search_result_count",
    ):
        value = process.get(key, 0)
        if type(value) is int and value >= 0:
            metrics[key] += value
    if process.get("search_successes", 0) > 0:
        metrics["cases_with_search_success"] += 1


def _transition_key(before: bool, after: bool) -> str:
    return f"{'correct' if before else 'wrong'}_to_{'correct' if after else 'wrong'}"


def _ordered_transitions(values: Counter[str]) -> dict[str, int]:
    return {
        key: values[key]
        for key in (
            "correct_to_correct",
            "correct_to_wrong",
            "wrong_to_correct",
            "wrong_to_wrong",
        )
    }


def _baseline_correctness(
    source: Path,
    gold_labels: Mapping[str, str],
) -> dict[str, dict[str, bool]]:
    direct = _terminal_results(source, "direct_flash")
    bayesprobe = _terminal_results(source, "bayesprobe_python")
    if set(direct) != set(gold_labels) or set(bayesprobe) != set(gold_labels):
        raise ValueError("source checkpoint must have terminal baseline results")
    return {
        sample_id: {
            "direct_no_web": (
                direct[sample_id].get("state") == "completed"
                and direct[sample_id].get("answer_label") == gold_labels[sample_id]
            ),
            "bayesprobe_no_web": (
                bayesprobe[sample_id].get("state") == "completed"
                and bayesprobe[sample_id].get("answer_label")
                == gold_labels[sample_id]
            ),
        }
        for sample_id in gold_labels
    }


def _terminal_results(source: Path, arm: str) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for path in sorted((source / "arms" / arm).glob("*/result.json")):
        payload = _read_json_object(path)
        sample_id = payload.get("sample_id")
        if payload.get("state") not in {"completed", "terminal_failed"}:
            raise ValueError("source checkpoint has non-terminal baseline result")
        if not isinstance(sample_id, str) or not sample_id or sample_id in results:
            raise ValueError("source checkpoint has invalid completed result identity")
        results[sample_id] = payload
    return results


def _load_selection(path: Path) -> dict[str, Any]:
    payload = _read_json_object(path)
    claimed_hash = payload.get("manifest_sha256")
    unsigned = dict(payload)
    unsigned.pop("manifest_sha256", None)
    manifest_sha256 = _canonical_sha256(unsigned)
    if claimed_hash != manifest_sha256:
        raise ValueError("selection manifest hash does not match content")
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("selection manifest items must be an array")
    cases = [
        _case_from_manifest_item(item)
        for item in items
    ]
    if len(cases) != len({case.sample_id for case in cases}):
        raise ValueError("selection manifest sample ids must be unique")
    dataset_revision = payload.get("dataset_revision")
    if not isinstance(dataset_revision, str) or not dataset_revision:
        raise ValueError("selection manifest dataset revision is invalid")
    return {
        "payload": payload,
        "manifest_sha256": manifest_sha256,
        "dataset_revision": dataset_revision,
        "cases": cases,
    }


def _case_from_manifest_item(item: Any):
    from bayesprobe.evaluation.contracts import EvaluationCase

    if not isinstance(item, Mapping):
        raise ValueError("selection manifest item must be an object")
    return EvaluationCase(
        sample_id=item["sample_id"],
        question=item["question"],
        choices=item["choices"],
    )


def _load_gold_labels(path: Path) -> dict[str, str]:
    payload = _read_json_object(path)
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("gold store items must be an array")
    labels: dict[str, str] = {}
    for item in items:
        if not isinstance(item, Mapping) or set(item) != {"sample_id", "gold_label"}:
            raise ValueError("gold store item is invalid")
        sample_id = item["sample_id"]
        label = item["gold_label"]
        if not isinstance(sample_id, str) or not sample_id or not isinstance(label, str):
            raise ValueError("gold store item is invalid")
        if sample_id in labels:
            raise ValueError("gold store sample ids must be unique")
        labels[sample_id] = label
    return labels


def _load_identity(path: Path) -> ExperimentIdentity:
    return ExperimentIdentity(**_read_json_object(path / "experiment_identity.json"))


def _load_config(path: Path) -> CapabilityExperimentConfig:
    return capability_config_from_mapping(
        _read_json_object(path / "config_snapshot.json")
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must be a JSON object")
    return payload


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        check=False,
        text=True,
    )
    value = completed.stdout.strip().lower()
    if completed.returncode != 0 or len(value) != 40:
        raise ValueError("Git HEAD did not resolve to a full commit SHA")
    return value


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _assert_shareable_report_safe(
    report: Mapping[str, Any],
    *,
    cases: Sequence[Any],
    provider_secret: str | None,
    tavily_secret: str | None,
) -> None:
    assert_shareable_payload_safe(
        report,
        restricted_values=[
            value
            for case in cases
            for value in (case.sample_id, case.question, *case.choices.values())
        ],
        canaries=(),
        provider_secrets=[
            value for value in (provider_secret, tavily_secret) if value
        ],
    )


__all__ = [
    "build_search_matrix_report",
    "prepare_search_matrix",
    "report_search_matrix",
    "run_search_matrix",
    "score_search_matrix",
]
