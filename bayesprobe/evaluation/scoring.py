from __future__ import annotations

import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from bayesprobe.evaluation.artifacts import CapabilityArtifactStore
from bayesprobe.evaluation.contracts import ArmCaseResult, EvaluationCase
from bayesprobe.evaluation.hle import EvaluationGoldStore
from bayesprobe.evaluation.statistics import (
    exact_mcnemar_p_value,
    expected_calibration_error,
    multiclass_brier_score,
    multiclass_log_loss,
    paired_bootstrap_interval,
    paired_contingency,
    probability_entropy,
    top_two_margin,
    wilson_interval,
)


_ARMS = ("direct_flash", "bayesprobe_python")
_FORBIDDEN_SHAREABLE_KEYS = {
    "answer_label",
    "answer_summary",
    "choices",
    "code",
    "gold_label",
    "provider_response",
    "python_code",
    "question",
    "raw_content",
    "raw_model_response",
    "sample_id",
    "stderr",
    "stdout",
}


@dataclass(frozen=True)
class EvaluationScoreReport:
    arms: dict[str, dict[str, Any]]
    paired: dict[str, Any]
    category_metrics: dict[str, Any]
    process_metrics: dict[str, dict[str, Any]]
    details: tuple[dict[str, Any], ...]

    def restricted_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScoreArtifactPaths:
    score_details: Path
    score_marker: Path
    report_root: Path
    summary_json: Path
    summary_markdown: Path
    paired_metrics: Path
    provenance: Path


class MCQScorer:
    def __init__(
        self,
        *,
        bootstrap_resamples: int = 10_000,
        bootstrap_seed: str = "20260711",
    ) -> None:
        if type(bootstrap_resamples) is not int or bootstrap_resamples < 1:
            raise ValueError("bootstrap_resamples must be positive")
        self.bootstrap_resamples = bootstrap_resamples
        self.bootstrap_seed = bootstrap_seed

    def score(
        self,
        results: Sequence[ArmCaseResult],
        gold: EvaluationGoldStore,
        *,
        categories: Mapping[str, str] | None = None,
    ) -> EvaluationScoreReport:
        gold_labels = dict(gold.labels)
        if not gold_labels:
            raise ValueError("gold store must not be empty")
        indexed: dict[tuple[str, str], ArmCaseResult] = {}
        for result in results:
            key = (result.sample_id, result.arm)
            if result.arm not in _ARMS:
                raise ValueError(f"unexpected evaluation arm: {result.arm}")
            if key in indexed:
                raise ValueError("duplicate arm result for sample")
            indexed[key] = result
        expected_keys = {
            (sample_id, arm) for sample_id in gold_labels for arm in _ARMS
        }
        if set(indexed) != expected_keys:
            raise ValueError("results must contain exactly both arms for every gold sample")

        correctness: dict[str, dict[str, bool]] = {
            arm: {
                sample_id: _is_correct(indexed[(sample_id, arm)], gold_label)
                for sample_id, gold_label in gold_labels.items()
            }
            for arm in _ARMS
        }
        arm_summaries = {
            arm: _arm_summary(
                [indexed[(sample_id, arm)] for sample_id in gold_labels],
                gold_labels,
                correctness[arm],
            )
            for arm in _ARMS
        }
        bayes_correct = [
            correctness["bayesprobe_python"][sample_id]
            for sample_id in gold_labels
        ]
        direct_correct = [
            correctness["direct_flash"][sample_id] for sample_id in gold_labels
        ]
        contingency = paired_contingency(bayes_correct, direct_correct)
        bootstrap_low, bootstrap_high = paired_bootstrap_interval(
            bayes_correct,
            direct_correct,
            resamples=self.bootstrap_resamples,
            seed=self.bootstrap_seed,
        )
        paired = {
            "both_correct": contingency.both_correct,
            "bayesprobe_only": contingency.bayesprobe_only,
            "direct_only": contingency.direct_only,
            "both_wrong": contingency.both_wrong,
            "accuracy_difference": contingency.accuracy_difference,
            "bootstrap_95_ci": [bootstrap_low, bootstrap_high],
            "bootstrap_resamples": self.bootstrap_resamples,
            "bootstrap_seed": self.bootstrap_seed,
            "mcnemar_exact_p_value": exact_mcnemar_p_value(
                bayesprobe_only=contingency.bayesprobe_only,
                direct_only=contingency.direct_only,
            ),
        }
        category_metrics = _category_metrics(
            categories or {},
            gold_labels=gold_labels,
            correctness=correctness,
        )
        process_metrics = {
            arm: _aggregate_process_metrics(
                [indexed[(sample_id, arm)] for sample_id in gold_labels]
            )
            for arm in _ARMS
        }
        details = tuple(
            {
                "sample_id": sample_id,
                "gold_label": gold_labels[sample_id],
                "category": (categories or {}).get(sample_id),
                "arms": {
                    arm: {
                        **asdict(indexed[(sample_id, arm)]),
                        "correct": correctness[arm][sample_id],
                    }
                    for arm in _ARMS
                },
            }
            for sample_id in gold_labels
        )
        return EvaluationScoreReport(
            arms=arm_summaries,
            paired=paired,
            category_metrics=category_metrics,
            process_metrics=process_metrics,
            details=details,
        )


def score_and_write_experiment(
    *,
    artifact_store: CapabilityArtifactStore,
    cases: list[EvaluationCase],
    gold: EvaluationGoldStore,
    categories: Mapping[str, str],
    report_root: str | Path,
    restricted_canaries: Sequence[str] = (),
    provider_secrets: Sequence[str] = (),
    bootstrap_resamples: int = 10_000,
) -> ScoreArtifactPaths:
    score_marker = artifact_store.root / "scoring_complete.json"
    if score_marker.exists():
        raise ValueError("capability experiment has already been scored")
    if not artifact_store.all_terminal(cases):
        raise ValueError("all arm cases are terminal before scoring")
    if gold.manifest_sha256 != artifact_store.identity.selection_manifest_sha256:
        raise ValueError("gold store manifest hash does not match experiment identity")
    results = [
        ArmCaseResult(**artifact_store.load_result(arm, case.sample_id))
        for case in cases
        for arm in _ARMS
    ]
    report = MCQScorer(bootstrap_resamples=bootstrap_resamples).score(
        results,
        gold,
        categories=categories,
    )
    score_details = artifact_store.root / "score_details.json"
    _write_json(score_details, report.restricted_payload(), private=True)

    target_report_root = Path(report_root) / artifact_store.identity.experiment_id
    target_report_root.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "artifact_version": "0.1",
        "experiment_id": artifact_store.identity.experiment_id,
        "arms": report.arms,
        "paired": report.paired,
        "category_metrics": report.category_metrics,
        "process_metrics": report.process_metrics,
        "provider_telemetry": _provider_telemetry_summary(artifact_store, cases),
        "limitations": [
            "Exploratory capability pilot; not a preregistered causal estimate.",
            "Public HLE text-only multiple-choice subset.",
            "BayesProbe arm includes Python while Direct Flash does not.",
            "BayesProbe posterior values are internal belief mass, not calibrated probability.",
        ],
    }
    paired_payload = {
        "artifact_version": "0.1",
        "experiment_id": artifact_store.identity.experiment_id,
        "aggregate": report.paired,
        "samples": [
            {
                "sample_pseudonym": artifact_store.pseudonym_for(detail["sample_id"]),
                "bayesprobe_correct": detail["arms"]["bayesprobe_python"]["correct"],
                "direct_correct": detail["arms"]["direct_flash"]["correct"],
                "bayesprobe_state": detail["arms"]["bayesprobe_python"]["state"],
                "direct_state": detail["arms"]["direct_flash"]["state"],
            }
            for detail in report.details
        ],
    }
    provenance_payload = {
        "artifact_version": "0.1",
        **asdict(artifact_store.identity),
    }
    restricted_values = [
        value
        for case in cases
        for value in (case.sample_id, case.question, *case.choices.values())
    ]
    for payload in (summary_payload, paired_payload, provenance_payload):
        assert_shareable_payload_safe(
            payload,
            restricted_values=restricted_values,
            canaries=restricted_canaries,
            provider_secrets=provider_secrets,
        )

    summary_json = target_report_root / "summary.json"
    paired_metrics = target_report_root / "paired_metrics.json"
    provenance = target_report_root / "provenance.json"
    summary_markdown = target_report_root / "summary.md"
    _write_json(summary_json, summary_payload, private=False)
    _write_json(paired_metrics, paired_payload, private=False)
    _write_json(provenance, provenance_payload, private=False)
    markdown = _summary_markdown(summary_payload)
    assert_shareable_payload_safe(
        markdown,
        restricted_values=restricted_values,
        canaries=restricted_canaries,
        provider_secrets=provider_secrets,
    )
    _write_text(summary_markdown, markdown)
    _write_json(
        score_marker,
        {
            "artifact_version": "0.1",
            "experiment_id": artifact_store.identity.experiment_id,
            "score_details": str(score_details),
            "report_root": str(target_report_root),
        },
        private=True,
    )
    return ScoreArtifactPaths(
        score_details=score_details,
        score_marker=score_marker,
        report_root=target_report_root,
        summary_json=summary_json,
        summary_markdown=summary_markdown,
        paired_metrics=paired_metrics,
        provenance=provenance,
    )


def assert_shareable_payload_safe(
    payload: Any,
    *,
    restricted_values: Sequence[str],
    canaries: Sequence[str],
    provider_secrets: Sequence[str],
) -> None:
    restricted = {value for value in restricted_values if isinstance(value, str) and value}
    sensitive_substrings = [
        value
        for value in (*canaries, *provider_secrets)
        if isinstance(value, str) and value
    ]

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                key_text = str(key)
                if key_text.lower() in _FORBIDDEN_SHAREABLE_KEYS:
                    raise ValueError(
                        f"shareable artifact leak: forbidden key at {path}.{key_text}"
                    )
                visit(item, f"{path}.{key_text}")
            return
        if isinstance(value, list | tuple):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
            return
        if isinstance(value, str):
            if value in restricted:
                raise ValueError(f"shareable artifact leak: restricted value at {path}")
            for restricted_value in restricted:
                if len(restricted_value) >= 8 and restricted_value in value:
                    raise ValueError(
                        f"shareable artifact leak: restricted content at {path}"
                    )
            if any(secret in value for secret in sensitive_substrings):
                raise ValueError(f"shareable artifact leak: secret content at {path}")

    visit(payload, "$" )


def _is_correct(result: ArmCaseResult, gold_label: str) -> bool:
    return result.state == "completed" and result.answer_label == gold_label


def _arm_summary(
    results: list[ArmCaseResult],
    gold_labels: Mapping[str, str],
    correctness: Mapping[str, bool],
) -> dict[str, Any]:
    total = len(results)
    correct_count = sum(correctness.values())
    interval = wilson_interval(successes=correct_count, total=total)
    calibration_rows = [
        result
        for result in results
        if result.state == "completed" and result.probabilities is not None
    ]
    brier_values = [
        multiclass_brier_score(
            result.probabilities,
            gold_label=gold_labels[result.sample_id],
        )
        for result in calibration_rows
    ]
    log_losses = [
        multiclass_log_loss(
            result.probabilities,
            gold_label=gold_labels[result.sample_id],
        )
        for result in calibration_rows
    ]
    confidence_outcomes = [
        (max(result.probabilities.values()), correctness[result.sample_id])
        for result in calibration_rows
    ]
    correct_confidences = [
        confidence
        for confidence, is_correct in confidence_outcomes
        if is_correct
    ]
    incorrect_confidences = [
        confidence
        for confidence, is_correct in confidence_outcomes
        if not is_correct
    ]
    return {
        "total": total,
        "completed": sum(result.state == "completed" for result in results),
        "terminal_failed": sum(
            result.state == "terminal_failed" for result in results
        ),
        "correct": correct_count,
        "accuracy": correct_count / total,
        "wilson_95_ci": list(interval),
        "calibration_count": len(calibration_rows),
        "calibration_coverage": len(calibration_rows) / total,
        "brier_score": _mean_or_none(brier_values),
        "log_loss": _mean_or_none(log_losses),
        "ece": (
            expected_calibration_error(confidence_outcomes)
            if confidence_outcomes
            else None
        ),
        "mean_top_confidence_correct": _mean_or_none(correct_confidences),
        "mean_top_confidence_incorrect": _mean_or_none(incorrect_confidences),
        "mean_entropy": _mean_or_none(
            [probability_entropy(result.probabilities) for result in calibration_rows]
        ),
        "mean_top_two_margin": _mean_or_none(
            [top_two_margin(result.probabilities) for result in calibration_rows]
        ),
    }


def _category_metrics(
    categories: Mapping[str, str],
    *,
    gold_labels: Mapping[str, str],
    correctness: Mapping[str, Mapping[str, bool]],
) -> dict[str, Any]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for sample_id in gold_labels:
        category = categories.get(sample_id)
        if isinstance(category, str) and category:
            grouped[category].append(sample_id)
    metrics: dict[str, Any] = {}
    for category, sample_ids in sorted(grouped.items()):
        if len(sample_ids) < 5:
            continue
        metrics[category] = {
            arm: {
                "count": len(sample_ids),
                "correct": sum(correctness[arm][sample_id] for sample_id in sample_ids),
                "accuracy": sum(
                    correctness[arm][sample_id] for sample_id in sample_ids
                )
                / len(sample_ids),
            }
            for arm in _ARMS
        }
    return metrics


def _aggregate_process_metrics(results: Sequence[ArmCaseResult]) -> dict[str, Any]:
    numeric: Counter[str] = Counter()
    distributions: dict[str, Counter[str]] = defaultdict(Counter)
    for result in results:
        for key, value in result.process_metrics.items():
            if type(value) in (int, float) and math.isfinite(value):
                numeric[key] += value
            elif isinstance(value, str):
                distributions[key][value] += 1
        if result.error_category is not None:
            distributions["error_category"][result.error_category] += 1
    payload: dict[str, Any] = dict(sorted(numeric.items()))
    payload["distributions"] = {
        key: dict(sorted(counts.items()))
        for key, counts in sorted(distributions.items())
    }
    return payload


def _provider_telemetry_summary(
    store: CapabilityArtifactStore,
    cases: Sequence[EvaluationCase],
) -> dict[str, Any]:
    summary = {
        arm: {
            "attempts": 0,
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "reasoning_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "latency_seconds": 0.0,
            "outcomes": {},
        }
        for arm in _ARMS
    }
    outcome_counts = {arm: Counter() for arm in _ARMS}
    for case in cases:
        for arm in _ARMS:
            path = store.paths_for(arm, case.sample_id).provider_invocations_path
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                summary[arm]["attempts"] += 1
                summary[arm]["latency_seconds"] += float(
                    record.get("latency_seconds", 0)
                )
                usage = record.get("usage", {})
                for key in (
                    "input_tokens",
                    "cached_input_tokens",
                    "reasoning_tokens",
                    "output_tokens",
                    "total_tokens",
                ):
                    value = usage.get(key)
                    if type(value) is int:
                        summary[arm][key] += value
                outcome_counts[arm][str(record.get("outcome", "unknown"))] += 1
    for arm in _ARMS:
        summary[arm]["latency_seconds"] = round(
            summary[arm]["latency_seconds"], 6
        )
        summary[arm]["outcomes"] = dict(sorted(outcome_counts[arm].items()))
    return summary


def _summary_markdown(payload: Mapping[str, Any]) -> str:
    direct = payload["arms"]["direct_flash"]
    bayesprobe = payload["arms"]["bayesprobe_python"]
    paired = payload["paired"]
    lines = [
        "# BayesProbe HLE Capability Pilot Summary",
        "",
        f"Experiment: `{payload['experiment_id']}`",
        "",
        "## Accuracy",
        "",
        f"- BayesProbe Python: {bayesprobe['correct']}/{bayesprobe['total']} ({bayesprobe['accuracy']:.3f})",
        f"- Direct Flash: {direct['correct']}/{direct['total']} ({direct['accuracy']:.3f})",
        f"- Difference: {paired['accuracy_difference']:.3f}",
        "",
        "## Paired Outcomes",
        "",
        f"- Both correct: {paired['both_correct']}",
        f"- BayesProbe only: {paired['bayesprobe_only']}",
        f"- Direct only: {paired['direct_only']}",
        f"- Both wrong: {paired['both_wrong']}",
        "",
        "## Limitations",
        "",
        *(f"- {limitation}" for limitation in payload["limitations"]),
        "",
    ]
    return "\n".join(lines)


def _mean_or_none(values: Sequence[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def _write_json(path: Path, payload: Any, *, private: bool) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _atomic_write(path, text.encode("utf-8"), mode=0o600 if private else 0o644)


def _write_text(path: Path, text: str) -> None:
    _atomic_write(path, text.encode("utf-8"), mode=0o644)


def _atomic_write(path: Path, payload: bytes, *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        os.chmod(path, mode)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary_path.unlink(missing_ok=True)
        raise


__all__ = [
    "EvaluationScoreReport",
    "MCQScorer",
    "ScoreArtifactPaths",
    "assert_shareable_payload_safe",
    "score_and_write_experiment",
]
