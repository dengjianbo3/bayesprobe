from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

from harbor.utils.trajectory_validator import TrajectoryValidator
from pydantic import BaseModel, ConfigDict, ValidationError

from bayesprobe import ExternalSignal, ProbeDesign
from bayesprobe_terminal_bench.actions import ActionObservation, TerminalProbePlan
from bayesprobe_terminal_bench.causal import (
    CausalActionRecord,
    CausalDecision,
    canonical_json,
    canonical_sha256,
    executed_request_from_action,
)
from bayesprobe_terminal_bench.provider_contract import ContractAttempt
from bayesprobe_terminal_bench.planning import (
    _EVALUATOR_PATH_PATTERN,
    _SECRET_VALUE_PATTERNS,
)


class TraceClassification(StrEnum):
    CONFORMANT = "conformant"
    PROVIDER_CONTRACT_ERROR = "provider_contract_error"
    CAUSAL_CONFORMANCE_ERROR = "causal_conformance_error"
    BUDGET_ERROR = "budget_error"
    ADAPTER_ERROR = "adapter_error"


class ConformanceReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    classification: TraceClassification
    complete_cycles: int
    plans: int
    actions: int
    signals: int
    evidence_events: int
    admitted_evidence: int
    discarded_evidence: int
    nonneutral_updates: int
    violations: tuple[str, ...]
    mechanism_metrics: dict[str, float | int]


class _PlanArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    probe_id: str
    cycle_id: str
    plan_id: str
    policy_attempt_id: str
    plan: TerminalProbePlan


class _ViolationBuckets:
    def __init__(self) -> None:
        self.security: set[str] = set()
        self.causal: set[str] = set()
        self.provider: set[str] = set()
        self.budget: set[str] = set()
        self.adapter: set[str] = set()

    def add(self, category: str, message: str) -> None:
        getattr(self, category).add(message)

    def ordered(self) -> tuple[str, ...]:
        result: list[str] = []
        for category in ("security", "causal", "provider", "budget", "adapter"):
            result.extend(
                f"{category}:{message}" for message in sorted(getattr(self, category))
            )
        return tuple(result)


_CLASSIFICATION_PRECEDENCE = (
    ("security", TraceClassification.CAUSAL_CONFORMANCE_ERROR),
    ("causal", TraceClassification.CAUSAL_CONFORMANCE_ERROR),
    ("provider", TraceClassification.PROVIDER_CONTRACT_ERROR),
    ("budget", TraceClassification.BUDGET_ERROR),
    ("adapter", TraceClassification.ADAPTER_ERROR),
)
_SECRET_PATTERNS = _SECRET_VALUE_PATTERNS + (
    re.compile(r"(?<![A-Za-z0-9])tvly-[A-Za-z0-9_-]{12,}", re.IGNORECASE),
)
_EVALUATOR_PATH = re.compile(
    r"(?ix)(?:^|[\s'\"])(?:/root/evaluator|/logs/verifier|/solution|/tests|"
    r"(?:\.\.?/)*(?:logs/verifier|solution|tests)/|/var/run/docker\.sock|"
    r"/run/docker\.sock)(?:[/\s'\"]|$)"
)
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_SIGNAL_SCHEMA_VERSION = "harbor-observation:v3"
_DEFAULT_MAX_ACTIONS = 24
_DEFAULT_MAX_MODEL_CALLS = 72
_DEFAULT_MAX_PROVIDER_TOKENS = 160_000
_ADMIT_REASONS = frozenset(
    {
        "state_scoped_inspection",
        "neutral_mutation_acknowledgement",
        "verified_postcondition",
        "preregistered_causal_transition",
    }
)
_PROMPT_SCHEMA = {
    "frame_open_question": ("open_question_task_framing", "OpenQuestionTaskFrame"),
    "repair_task_frame": ("open_question_task_framing_repair", "OpenQuestionTaskFrame"),
    "design_probes": ("probe_design", "ProbeDesign"),
    "repair_probe_design": ("probe_design_repair", "ProbeDesign"),
    "judge_evidence": ("evidence_judgment", "EvidenceJudgment"),
    "repair_evidence_judgment": ("evidence_judgment_repair", "EvidenceJudgment"),
    "project_answer": ("answer_projection", "AnswerProjection"),
    "repair_answer_projection": ("answer_projection_repair", "AnswerProjection"),
}
_ERROR_CATEGORY_MAP = {
    "causal_conformance_error": "causal",
    "causal_adapter_error": "causal",
    "provider_contract_error": "provider",
    "provider_identity_error": "provider",
    "provider_transport_error": "provider",
    "provider_error": "provider",
    "budget_error": "budget",
    "budget_exhausted": "budget",
    "adapter_error": "adapter",
    "plan_error": "adapter",
}


def validate_trial_trace(artifact_root: Path) -> ConformanceReport:
    """Validate one BayesProbe trial artifact directory without using reward."""
    root = Path(artifact_root)
    violations = _ViolationBuckets()
    if not root.is_dir():
        violations.add("adapter", "artifact root is not a directory")
        return _report(violations=violations)

    trajectory_path = _trajectory_path(root)
    _scan_artifacts(root, trajectory_path=trajectory_path, violations=violations)
    errors = _read_jsonl(root / "errors.jsonl", violations, "adapter", required=False)
    _classify_recorded_errors(errors, violations=violations)
    denied_actions = _policy_denied_action_count(errors, violations=violations)

    ledger = _read_jsonl(
        root / "bayesprobe_ledger.jsonl",
        violations,
        "causal",
        required=False,
    )
    by_type = _ledger_by_type(ledger, violations=violations)
    complete_cycles = _complete_cycles(by_type, violations=violations)

    contract_records = _read_jsonl(
        root / "provider_contract.jsonl",
        violations,
        "provider",
        required=False,
    )
    provider_records = _read_jsonl(
        root / "provider_telemetry.jsonl",
        violations,
        "provider",
        required=False,
    )
    attempts = _validate_provider_contract(
        contract_records,
        provider_records=provider_records,
        violations=violations,
    )
    provider_tokens, provider_calls, provider_model = _validate_provider_telemetry(
        provider_records,
        violations=violations,
    )

    plan_records = _read_jsonl(
        root / "plans.jsonl", violations, "causal", required=False
    )
    action_records = _read_jsonl(
        root / "environment_actions.jsonl",
        violations,
        "causal",
        required=False,
    )
    causal_records = _read_jsonl(
        root / "causal_actions.jsonl",
        violations,
        "causal",
        required=False,
    )
    decision_records = _read_jsonl(
        root / "causal_decisions.jsonl",
        violations,
        "causal",
        required=False,
    )

    substantive_trace = bool(plan_records or action_records)
    if substantive_trace and not attempts:
        violations.add("provider", "provider contract attempts are missing")
    if provider_records and not attempts:
        violations.add("provider", "provider contract attempts are missing")

    plans = _parse_models(plan_records, _PlanArtifact, violations, "causal", "plan")
    actions = _parse_models(
        action_records, ActionObservation, violations, "causal", "executed action"
    )
    causal_actions = _parse_models(
        causal_records, CausalActionRecord, violations, "causal", "causal action"
    )
    decisions = _parse_decisions(decision_records, violations=violations)
    signals = _parse_signals(by_type.get("external_signal", []), violations=violations)

    if substantive_trace:
        _require_current_causal_artifacts(
            plans=plans,
            actions=actions,
            causal_actions=causal_actions,
            signals=signals,
            violations=violations,
        )
    probes = _probes_by_id(by_type, violations=violations)
    action_by_id, signal_by_id = _validate_causal_lineage(
        plans=plans,
        actions=actions,
        causal_actions=causal_actions,
        signals=signals,
        probes=probes,
        violations=violations,
    )
    _validate_decisions(
        decisions=decisions,
        action_by_id=action_by_id,
        signal_by_id=signal_by_id,
        violations=violations,
    )
    evidence, admitted, discarded, nonneutral_updates = _validate_epistemic_lineage(
        by_type=by_type,
        signals=signal_by_id,
        decisions=decisions,
        violations=violations,
    )
    _validate_ledger_prompt_provenance(by_type, violations=violations)

    summary = _read_object(
        root / "summary.json",
        violations,
        "adapter",
        required=substantive_trace,
    )
    _validate_budgets(
        summary=summary,
        reserved_actions=len(actions) + denied_actions,
        provider_calls=provider_calls,
        provider_tokens=provider_tokens,
        violations=violations,
    )
    _validate_atif(
        trajectory_path=trajectory_path,
        actions=actions,
        reserved_actions=len(actions) + denied_actions,
        causal_actions=causal_actions,
        signals=signal_by_id,
        provider_tokens=provider_tokens,
        provider_calls=provider_calls,
        provider_model=provider_model,
        violations=violations,
        required=bool(substantive_trace or complete_cycles),
    )

    action_count = len(actions)
    signal_count = len(signals)
    evidence_count = len(evidence)
    metrics: dict[str, float | int] = {
        "action_signal_ratio": signal_count / action_count if action_count else 0.0,
        "admitted_evidence_rate": admitted / evidence_count if evidence_count else 0.0,
        "discarded_evidence_rate": discarded / evidence_count if evidence_count else 0.0,
        "nonneutral_updates_per_admitted_evidence": (
            nonneutral_updates / admitted if admitted else 0.0
        ),
        "provider_tokens": provider_tokens,
    }
    return _report(
        violations=violations,
        complete_cycles=complete_cycles,
        plans=len(plans),
        actions=action_count,
        signals=signal_count,
        evidence_events=evidence_count,
        admitted_evidence=admitted,
        discarded_evidence=discarded,
        nonneutral_updates=nonneutral_updates,
        mechanism_metrics=metrics,
    )


def _report(
    *,
    violations: _ViolationBuckets,
    complete_cycles: int = 0,
    plans: int = 0,
    actions: int = 0,
    signals: int = 0,
    evidence_events: int = 0,
    admitted_evidence: int = 0,
    discarded_evidence: int = 0,
    nonneutral_updates: int = 0,
    mechanism_metrics: dict[str, float | int] | None = None,
) -> ConformanceReport:
    classification = TraceClassification.CONFORMANT
    for bucket, candidate in _CLASSIFICATION_PRECEDENCE:
        if getattr(violations, bucket):
            classification = candidate
            break
    return ConformanceReport(
        classification=classification,
        complete_cycles=complete_cycles,
        plans=plans,
        actions=actions,
        signals=signals,
        evidence_events=evidence_events,
        admitted_evidence=admitted_evidence,
        discarded_evidence=discarded_evidence,
        nonneutral_updates=nonneutral_updates,
        violations=violations.ordered(),
        mechanism_metrics=mechanism_metrics or {},
    )


def _scan_artifacts(
    root: Path,
    *,
    trajectory_path: Path | None,
    violations: _ViolationBuckets,
) -> None:
    paths = list(root.rglob("*"))
    if trajectory_path is not None and trajectory_path not in paths:
        paths.append(trajectory_path)
    for path in sorted(paths):
        if path.is_symlink():
            violations.add("security", f"symlink artifact is forbidden: {path.name}")
            continue
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            violations.add("adapter", f"artifact is not readable UTF-8: {path.name}")
            continue
        if any(pattern.search(text) for pattern in _SECRET_PATTERNS):
            violations.add("security", f"secret-shaped content found: {path.name}")
        if _EVALUATOR_PATH_PATTERN.search(text) or _EVALUATOR_PATH.search(text):
            violations.add("security", f"evaluator-path content found: {path.name}")


def _trajectory_path(root: Path) -> Path | None:
    local = root / "trajectory.json"
    if local.is_file() or local.is_symlink():
        return local
    sibling = root.parent / "trajectory.json"
    return sibling if sibling.is_file() or sibling.is_symlink() else None


def _read_jsonl(
    path: Path,
    violations: _ViolationBuckets,
    category: str,
    *,
    required: bool,
) -> list[Mapping[str, Any]]:
    if not path.is_file():
        if required:
            violations.add(category, f"missing {path.name}")
        return []
    records: list[Mapping[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        violations.add(category, f"unreadable {path.name}")
        return []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line, object_pairs_hook=_unique_object)
        except (json.JSONDecodeError, ValueError):
            violations.add(category, f"invalid {path.name} line {line_number}")
            continue
        if not isinstance(value, Mapping):
            violations.add(category, f"non-object {path.name} line {line_number}")
            continue
        records.append(value)
    return records


def _read_object(
    path: Path,
    violations: _ViolationBuckets,
    category: str,
    *,
    required: bool,
) -> Mapping[str, Any] | None:
    if not path.is_file():
        if required:
            violations.add(category, f"missing {path.name}")
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        violations.add(category, f"invalid {path.name}")
        return None
    if not isinstance(value, Mapping):
        violations.add(category, f"non-object {path.name}")
        return None
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _ledger_by_type(
    records: Sequence[Mapping[str, Any]],
    *,
    violations: _ViolationBuckets,
) -> dict[str, list[Mapping[str, Any]]]:
    by_type: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        record_type = record.get("record_type")
        payload = record.get("payload")
        if not isinstance(record_type, str) or not isinstance(payload, Mapping):
            violations.add("causal", "ledger record lacks typed object payload")
            continue
        by_type[record_type].append(payload)
    return dict(by_type)


def _complete_cycles(
    by_type: Mapping[str, list[Mapping[str, Any]]],
    *,
    violations: _ViolationBuckets,
) -> int:
    complete = [
        item
        for item in by_type.get("cycle", [])
        if item.get("boundary_status") == "integrated"
        and item.get("completed_at") is not None
    ]
    cycle_ids = [item.get("cycle_id") for item in complete]
    if any(not isinstance(item, str) or not item for item in cycle_ids) or len(
        cycle_ids
    ) != len(set(cycle_ids)):
        violations.add("causal", "completed cycle identities are invalid or duplicated")
    indexes = [item.get("cycle_index") for item in complete]
    if complete and set(indexes) != set(range(1, len(complete) + 1)):
        violations.add("causal", "completed cycle indexes are not contiguous")
    runs = by_type.get("run", [])
    completed_runs = [item for item in runs if item.get("status") == "completed"]
    if complete and len(completed_runs) != 1:
        violations.add("causal", "trace does not have exactly one completed run")
    if complete and len(completed_runs) == 1:
        run_id = completed_runs[0].get("run_id")
        if not isinstance(run_id, str) or any(
            item.get("run_id") != run_id for item in complete
        ):
            violations.add("causal", "completed cycle/run identity mismatch")
    return len(complete)


def _parse_models(
    records: Sequence[Mapping[str, Any]],
    model: type[BaseModel],
    violations: _ViolationBuckets,
    category: str,
    label: str,
) -> list[Any]:
    parsed: list[Any] = []
    for index, record in enumerate(records, start=1):
        try:
            parsed.append(model.model_validate_json(canonical_json(record)))
        except (ValidationError, ValueError, TypeError):
            violations.add(category, f"invalid {label} record {index}")
    return parsed


def _validate_provider_contract(
    records: Sequence[Mapping[str, Any]],
    *,
    provider_records: Sequence[Mapping[str, Any]],
    violations: _ViolationBuckets,
) -> list[ContractAttempt]:
    attempts = _parse_models(
        records, ContractAttempt, violations, "provider", "provider contract attempt"
    )
    by_stage: dict[str, list[ContractAttempt]] = defaultdict(list)
    for attempt in attempts:
        by_stage[attempt.stage].append(attempt)
        if attempt.response_sha256 is not None and not _HEX_64.fullmatch(
            attempt.response_sha256
        ):
            violations.add("provider", "contract response hash is not canonical")
    for stage, expected_tasks in {
        "terminal_task_frame": ("frame_open_question", "repair_task_frame"),
        "terminal_probe_design": ("design_probes", "repair_probe_design"),
    }.items():
        stage_attempts = by_stage.get(stage, [])
        if not stage_attempts:
            if attempts:
                violations.add("provider", f"missing {stage} attempt")
            continue
        indexes = [item.attempt_index for item in stage_attempts]
        if indexes != list(range(len(stage_attempts))) or len(stage_attempts) > 3:
            violations.add("provider", f"invalid {stage} attempt sequence")
        for item in stage_attempts:
            expected_task = expected_tasks[0] if item.attempt_index == 0 else expected_tasks[1]
            if item.request_task != expected_task:
                violations.add("provider", f"invalid {stage} request task")
        if stage_attempts[-1].validation != "valid":
            violations.add("provider", f"{stage} did not finish valid")
        if any(item.validation == "valid" for item in stage_attempts[:-1]):
            violations.add("provider", f"{stage} continued after a valid attempt")
    if provider_records and not attempts:
        violations.add("provider", "provider calls have no bounded contract attempts")
    return attempts


def _validate_provider_telemetry(
    records: Sequence[Mapping[str, Any]],
    *,
    violations: _ViolationBuckets,
) -> tuple[int, int, str | None]:
    tokens = 0
    successful = 0
    identities: set[tuple[object, object]] = set()
    for index, record in enumerate(records, start=1):
        task = record.get("task")
        outcome = record.get("outcome")
        if not isinstance(task, str) or not task:
            violations.add("provider", f"provider call {index} lacks task provenance")
        if outcome != "success":
            violations.add("provider", f"provider call {index} outcome is {outcome!r}")
            continue
        successful += 1
        model = record.get("model")
        fingerprint = record.get("system_fingerprint")
        if not isinstance(model, str) or not model:
            violations.add("provider", f"provider call {index} lacks model identity")
        identities.add((model, fingerprint))
        usage = record.get("usage")
        if not isinstance(usage, Mapping):
            violations.add("provider", f"provider call {index} lacks usage")
            continue
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        total_tokens = usage.get("total_tokens")
        if any(
            type(item) is not int or item < 0
            for item in (input_tokens, output_tokens, total_tokens)
        ):
            violations.add("provider", f"provider call {index} has invalid token usage")
            continue
        if input_tokens + output_tokens != total_tokens:
            violations.add("provider", f"provider call {index} token total disagrees")
        tokens += total_tokens
        if task in _PROMPT_SCHEMA:
            prompt_id, schema_name = _PROMPT_SCHEMA[task]
            if (
                record.get("prompt_id") != prompt_id
                or record.get("prompt_version") != "v0.2"
                or record.get("schema_name") != schema_name
                or record.get("schema_version") != "v0.2"
            ):
                violations.add("provider", f"provider call {index} prompt/schema drift")
        elif task == "terminal_probe_plan":
            if record.get("plan_validation") not in {"valid", "invalid"}:
                violations.add("provider", f"terminal plan call {index} lacks validation")
        else:
            violations.add("provider", f"provider call {index} has unknown task {task!r}")
    if len(identities) > 1:
        violations.add("provider", "provider model or fingerprint identity drift")
    model_identity = next(iter(identities))[0] if len(identities) == 1 else None
    return tokens, successful, model_identity if isinstance(model_identity, str) else None


def _parse_decisions(
    records: Sequence[Mapping[str, Any]],
    *,
    violations: _ViolationBuckets,
) -> list[CausalDecision]:
    typed = [record for record in records if "decision" in record]
    return _parse_models(
        typed, CausalDecision, violations, "causal", "causal decision"
    )


def _parse_signals(
    records: Sequence[Mapping[str, Any]],
    *,
    violations: _ViolationBuckets,
) -> list[ExternalSignal]:
    return _parse_models(records, ExternalSignal, violations, "causal", "Signal")


def _require_current_causal_artifacts(
    *,
    plans: Sequence[_PlanArtifact],
    actions: Sequence[ActionObservation],
    causal_actions: Sequence[CausalActionRecord],
    signals: Sequence[ExternalSignal],
    violations: _ViolationBuckets,
) -> None:
    for name, records in (
        ("plans", plans),
        ("executed actions", actions),
        ("causal actions", causal_actions),
        ("Signals", signals),
    ):
        if not records:
            violations.add("causal", f"current trace has no {name}")
    if len(actions) != len(causal_actions) or len(actions) != len(signals):
        violations.add("causal", "executed action/causal action/Signal cardinality differs")


def _probes_by_id(
    by_type: Mapping[str, list[Mapping[str, Any]]],
    *,
    violations: _ViolationBuckets,
) -> dict[str, ProbeDesign]:
    probes: dict[str, ProbeDesign] = {}
    for probe_set in by_type.get("probe_set", []):
        items = probe_set.get("probes")
        if not isinstance(items, Sequence) or isinstance(items, str | bytes):
            violations.add("causal", "probe set has invalid probes")
            continue
        for item in items:
            try:
                probe = ProbeDesign.model_validate(item)
            except ValidationError:
                violations.add("causal", "probe set has invalid Probe")
                continue
            if probe.id in probes:
                violations.add("causal", f"duplicate Probe ID {probe.id}")
            probes[probe.id] = probe
    return probes


def _validate_causal_lineage(
    *,
    plans: Sequence[_PlanArtifact],
    actions: Sequence[ActionObservation],
    causal_actions: Sequence[CausalActionRecord],
    signals: Sequence[ExternalSignal],
    probes: Mapping[str, ProbeDesign],
    violations: _ViolationBuckets,
) -> tuple[dict[str, CausalActionRecord], dict[str, ExternalSignal]]:
    plan_by_id = _unique_by(plans, "plan_id", violations=violations, label="plan")
    action_by_index = _unique_by(
        actions, "action_index", violations=violations, label="executed action"
    )
    causal_by_id = _unique_by(
        causal_actions, "action_id", violations=violations, label="causal action"
    )
    signal_by_id = _unique_by(signals, "id", violations=violations, label="Signal")
    step_keys: set[tuple[str, int]] = set()
    prior_state_by_run: dict[str, str] = {}
    generation_by_run: dict[str, int] = defaultdict(int)
    bound_signal_ids: set[str] = set()

    for record in sorted(causal_actions, key=lambda item: item.observation.action_index):
        plan = plan_by_id.get(record.plan_id)
        observation = action_by_index.get(record.observation.action_index)
        if plan is None or observation is None:
            violations.add("causal", f"causal action {record.action_id} is orphaned")
            continue
        step_key = (record.plan_id, record.step_index)
        if step_key in step_keys:
            violations.add("causal", f"duplicate completed plan step {step_key}")
        step_keys.add(step_key)
        if record.step_index >= len(plan.plan.steps):
            violations.add("causal", f"causal action {record.action_id} has invalid step")
            continue
        step = plan.plan.steps[record.step_index]
        expected_request = executed_request_from_action(step.action)
        expected_fingerprint = "sha256:" + canonical_sha256(expected_request)
        expected_action_id = "A_" + canonical_sha256(
            {
                "action_index": observation.action_index,
                "plan_id": record.plan_id,
                "request_fingerprint": expected_fingerprint,
                "step_index": record.step_index,
            }
        )
        if record.action_id != expected_action_id:
            violations.add("causal", f"causal action ID mismatch {record.action_id}")
        if record.request_fingerprint != expected_fingerprint:
            violations.add("causal", f"request fingerprint mismatch {record.action_id}")
        if observation != record.observation or observation.action != step.action:
            violations.add("causal", f"executed request mismatch {record.action_id}")
        if (
            record.action_role != step.role
            or record.plan_id != plan.plan_id
            or record.policy_attempt_id != plan.policy_attempt_id
            or record.probe_id != plan.probe_id
            or record.cycle_id != plan.cycle_id
            or record.verification_target != step.verification_target
        ):
            violations.add("causal", f"plan lineage mismatch {record.action_id}")
        expected_pre = prior_state_by_run.get(record.run_id)
        if expected_pre is not None and record.pre_environment_state_id != expected_pre:
            violations.add("causal", f"non-linear environment state {record.action_id}")
        prior_state_by_run[record.run_id] = record.post_environment_state_id
        if record.action_role == "intervene":
            generation_by_run[record.run_id] += 1
        expected_subject = (
            record.pre_environment_state_id
            if record.action_role == "verify"
            else record.post_environment_state_id
        )
        if (
            record.pre_environment_state_id != observation.pre_environment_state_id
            or record.post_environment_state_id != observation.post_environment_state_id
            or record.subject_environment_state_id != expected_subject
            or record.intervention_generation != generation_by_run[record.run_id]
        ):
            violations.add("causal", f"environment lineage mismatch {record.action_id}")
        expected_predictions = {
            item.hypothesis_id: item.expected_transition
            for item in plan.plan.transition_predictions
        }
        if record.transition_predictions != expected_predictions:
            violations.add("causal", f"transition prediction mismatch {record.action_id}")

        matching = [
            signal
            for signal in signals
            if _signal_action_id(signal) == record.action_id
        ]
        if len(matching) != 1:
            violations.add("causal", f"action {record.action_id} has {len(matching)} Signals")
            continue
        signal = matching[0]
        bound_signal_ids.add(signal.id)
        _validate_signal(
            signal=signal,
            record=record,
            plan=plan,
            probe=probes.get(record.probe_id),
            expected_request=expected_request,
            violations=violations,
        )

    if set(signal_by_id) != bound_signal_ids:
        violations.add("causal", "one or more Signals are not uniquely action-bound")
    for plan in plans:
        completed = {index for plan_id, index in step_keys if plan_id == plan.plan_id}
        if completed != set(range(len(plan.plan.steps))):
            violations.add("causal", f"plan {plan.plan_id} is not exactly executed")
        probe = probes.get(plan.probe_id)
        run_ids = {
            item.run_id for item in causal_actions if item.plan_id == plan.plan_id
        }
        if probe is not None and len(run_ids) == 1:
            run_id = next(iter(run_ids))
            expected_plan_id = "PL_" + canonical_sha256(
                {
                    "cycle_id": plan.cycle_id,
                    "probe": probe.model_dump(mode="json"),
                    "run_id": run_id,
                    "plan": plan.plan.model_dump(mode="json"),
                }
            )
            expected_policy_id = "PA_" + canonical_sha256(
                {
                    "cycle_id": plan.cycle_id,
                    "probe": probe.model_dump(mode="json"),
                    "run_id": run_id,
                    "intervention_plan": plan.plan.model_dump(mode="json"),
                }
            )
            if plan.plan_id != expected_plan_id or plan.policy_attempt_id != expected_policy_id:
                violations.add("causal", f"plan identity mismatch {plan.plan_id}")
    return causal_by_id, signal_by_id


def _unique_by(
    items: Sequence[Any],
    field: str,
    *,
    violations: _ViolationBuckets,
    label: str,
) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for item in items:
        key = getattr(item, field)
        if key in result:
            violations.add("causal", f"duplicate {label} ID {key}")
        result[key] = item
    return result


def _signal_action_id(signal: ExternalSignal) -> str | None:
    raw = _json_mapping(signal.raw_content)
    binding = raw.get("causal_binding") if raw is not None else None
    action_id = binding.get("action_id") if isinstance(binding, Mapping) else None
    return action_id if isinstance(action_id, str) else None


def _validate_signal(
    *,
    signal: ExternalSignal,
    record: CausalActionRecord,
    plan: _PlanArtifact,
    probe: ProbeDesign | None,
    expected_request: Mapping[str, Any],
    violations: _ViolationBuckets,
) -> None:
    raw = _json_mapping(signal.raw_content)
    provenance = signal.provenance
    if raw is None or provenance is None:
        violations.add("causal", f"Signal {signal.id} lacks structured provenance")
        return
    binding = raw.get("causal_binding")
    if not isinstance(binding, Mapping):
        violations.add("causal", f"Signal {signal.id} lacks causal binding")
        return
    expected_signal_id = "S_harbor_" + canonical_sha256(
        {
            "action_id": record.action_id,
            "full_output_sha256": record.observation.full_output_sha256,
            "schema_version": _SIGNAL_SCHEMA_VERSION,
        }
    )
    expected_root = "harbor-action:sha256:" + canonical_sha256(
        {
            "action_id": record.action_id,
            "full_output_sha256": record.observation.full_output_sha256,
            "schema_version": _SIGNAL_SCHEMA_VERSION,
        }
    )
    environment_digest = canonical_sha256(
        {
            "run_id": record.run_id,
            "schema_version": _SIGNAL_SCHEMA_VERSION,
            "subject_environment_state_id": record.subject_environment_state_id,
        }
    )
    source = f"harbor-terminal:sha256:{environment_digest}"
    expected_fingerprint = _content_fingerprint(source, signal.raw_content)
    expected_binding = {
        "action_id": record.action_id,
        "action_role": record.action_role,
        "plan_id": record.plan_id,
        "policy_attempt_id": record.policy_attempt_id,
        "request_fingerprint": record.request_fingerprint,
        "subject_environment_state_id": record.subject_environment_state_id,
        "verification_target": record.verification_target,
    }
    expected_raw = {
        "action_index": record.observation.action_index,
        "causal_binding": expected_binding,
        "executed_request": dict(expected_request),
        "observation": record.observation.model_facing_output,
        "post_environment_state_id": record.post_environment_state_id,
        "pre_environment_state_id": record.pre_environment_state_id,
    }
    expected_refs = [
        f"environment_actions.jsonl#{record.observation.action_index}",
        f"causal_actions.jsonl#{record.action_id}",
    ]
    if (
        signal.id != expected_signal_id
        or signal.cycle_id != record.cycle_id
        or signal.generated_by_probe != record.probe_id
        or signal.source_type != "harbor_terminal"
        or signal.source != "harbor:environment"
        or raw != expected_raw
        or provenance.epistemic_origin.value != "tool_result"
        or provenance.source_identity != source
        or provenance.provider_model_or_tool_identity != "harbor:0.18.0"
        or provenance.derivation_root_id != expected_root
        or provenance.correlation_group != f"harbor-env:sha256:{environment_digest}"
        or provenance.canonical_content_fingerprint != expected_fingerprint
        or provenance.environment_state_id != record.subject_environment_state_id
        or list(provenance.artifact_refs) != expected_refs
        or (
            probe is not None
            and tuple(signal.initial_target_hypotheses)
            != tuple(probe.target_hypotheses)
        )
        or plan.probe_id != signal.generated_by_probe
    ):
        violations.add("causal", f"Signal lineage mismatch {signal.id}")


def _json_mapping(value: str) -> Mapping[str, Any] | None:
    try:
        parsed = json.loads(value, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, Mapping) else None


def _content_fingerprint(source_identity: str, raw_content: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", raw_content).split())
    digest = hashlib.sha256(f"{source_identity}\n{normalized}".encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _validate_decisions(
    *,
    decisions: Sequence[CausalDecision],
    action_by_id: Mapping[str, CausalActionRecord],
    signal_by_id: Mapping[str, ExternalSignal],
    violations: _ViolationBuckets,
) -> None:
    for decision in decisions:
        action = action_by_id.get(decision.action_id)
        signal = signal_by_id.get(decision.signal_id)
        if action is None or signal is None or _signal_action_id(signal) != decision.action_id:
            violations.add("causal", f"orphan causal decision for {decision.signal_id}")
            continue
        expected_kind = "admit" if decision.reason_code in _ADMIT_REASONS else "discard"
        if (
            decision.action_role != action.action_role
            or decision.subject_environment_state_id != action.subject_environment_state_id
            or decision.decision != expected_kind
            or not _HEX_64.fullmatch(decision.judgment_response_sha256)
        ):
            violations.add("causal", f"causal decision contradiction {decision.signal_id}")


def _validate_epistemic_lineage(
    *,
    by_type: Mapping[str, list[Mapping[str, Any]]],
    signals: Mapping[str, ExternalSignal],
    decisions: Sequence[CausalDecision],
    violations: _ViolationBuckets,
) -> tuple[dict[str, Mapping[str, Any]], int, int, int]:
    decisions_by_signal: dict[str, list[CausalDecision]] = defaultdict(list)
    for decision in decisions:
        decisions_by_signal[decision.signal_id].append(decision)
    evidence: dict[str, Mapping[str, Any]] = {}
    evidence_by_signal: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in by_type.get("evidence_event", []):
        evidence_id = item.get("id")
        signal_id = item.get("derived_from_signal")
        if not isinstance(evidence_id, str) or not evidence_id or evidence_id in evidence:
            violations.add("causal", "Evidence identities are invalid or duplicated")
            continue
        signal = signals.get(signal_id) if isinstance(signal_id, str) else None
        if signal is None or signal.provenance is None:
            violations.add("causal", f"Evidence {evidence_id} has no Signal")
        else:
            if (
                item.get("epistemic_origin") != "tool_result"
                or item.get("derivation_root_id") != signal.provenance.derivation_root_id
                or item.get("content") != signal.raw_content
            ):
                violations.add("causal", f"Evidence provenance mismatch {evidence_id}")
        evidence[evidence_id] = item
        if isinstance(signal_id, str):
            evidence_by_signal[signal_id].append(item)

    if set(evidence_by_signal) != set(signals) or any(
        len(items) != 1 for items in evidence_by_signal.values()
    ):
        violations.add("causal", "Signal/Evidence cardinality is not exactly one-to-one")

    admitted = 0
    discarded = 0
    final_decision_by_evidence: dict[str, CausalDecision] = {}
    for evidence_id, item in evidence.items():
        signal_id = item.get("derived_from_signal")
        route = decisions_by_signal.get(signal_id, []) if isinstance(signal_id, str) else []
        if not route:
            violations.add("causal", f"Evidence {evidence_id} lacks causal decision")
            continue
        final = route[-1]
        final_decision_by_evidence[evidence_id] = final
        discard_reason = item.get("discard_reason")
        is_discarded = discard_reason is not None
        admitted += int(not is_discarded)
        discarded += int(is_discarded)
        is_causal_discard = (
            isinstance(discard_reason, str)
            and "causal_admissibility:" in discard_reason
        )
        if is_causal_discard and final.decision != "discard":
            violations.add("causal", f"discarded Evidence {evidence_id} lacks guard discard")
        if not is_discarded and final.decision != "admit":
            violations.add("causal", f"admitted Evidence {evidence_id} lacks admitted route")

    contributions: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in by_type.get("evidence_contribution_delta", []):
        root = item.get("contribution_root_id")
        causes = _string_list(item.get("caused_by_event_ids"))
        if not isinstance(root, str) or not root or causes is None:
            violations.add("causal", "contribution lacks declared root or causes")
            continue
        contributions[root].append(item)
        if _contribution_is_nonneutral(item):
            _validate_exact_admitted_route(
                causes=causes,
                evidence=evidence,
                decisions=final_decision_by_evidence,
                label=f"contribution {root}",
                violations=violations,
            )

    nonneutral_updates = 0
    for item in by_type.get("belief_update", []):
        sensitivity = item.get("sensitivity")
        causes = _string_list(
            sensitivity.get("caused_by_event_ids")
            if isinstance(sensitivity, Mapping)
            else None
        )
        root = item.get("evidence_id")
        if causes is None or not isinstance(root, str):
            violations.add("causal", "Update lacks declared evidence causes")
            continue
        if not _update_is_consistent(item):
            violations.add("causal", f"Update direction is inconsistent for {root}")
            continue
        if item.get("direction") == "neutral":
            continue
        nonneutral_updates += 1
        _validate_exact_admitted_route(
            causes=causes,
            evidence=evidence,
            decisions=final_decision_by_evidence,
            label=f"Update {root}",
            violations=violations,
        )
        matching = [
            contribution
            for contribution in contributions.get(root, [])
            if _string_list(contribution.get("caused_by_event_ids")) == causes
            and _contribution_is_nonneutral(contribution)
        ]
        if len(matching) != 1:
            violations.add("causal", f"Update {root} lacks exactly one contribution")

    discarded_ids = {
        evidence_id
        for evidence_id, item in evidence.items()
        if item.get("discard_reason") is not None
    }
    for item in by_type.get("evidence_contribution_delta", []):
        causes = _string_list(item.get("caused_by_event_ids")) or []
        if discarded_ids.intersection(causes):
            violations.add("causal", "discarded Evidence has an accepted contribution")
    for item in by_type.get("belief_update", []):
        sensitivity = item.get("sensitivity")
        causes = _string_list(
            sensitivity.get("caused_by_event_ids")
            if isinstance(sensitivity, Mapping)
            else None
        ) or []
        if discarded_ids.intersection(causes):
            violations.add("causal", "discarded Evidence caused an Update")
    return evidence, admitted, discarded, nonneutral_updates


def _validate_exact_admitted_route(
    *,
    causes: Sequence[str],
    evidence: Mapping[str, Mapping[str, Any]],
    decisions: Mapping[str, CausalDecision],
    label: str,
    violations: _ViolationBuckets,
) -> None:
    if len(causes) != 1 or len(set(causes)) != 1:
        violations.add("causal", f"{label} does not have exactly one cause")
        return
    evidence_id = causes[0]
    event = evidence.get(evidence_id)
    decision = decisions.get(evidence_id)
    if (
        event is None
        or event.get("discard_reason") is not None
        or decision is None
        or decision.decision != "admit"
    ):
        violations.add("causal", f"{label} does not have one admitted causal route")


def _string_list(value: object) -> list[str] | None:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return None
    if any(not isinstance(item, str) or not item for item in value):
        return None
    return list(value)


def _contribution_is_nonneutral(item: Mapping[str, Any]) -> bool:
    delta = item.get("per_hypothesis_delta")
    unresolved = item.get("unresolved_delta")
    return (
        isinstance(delta, Mapping)
        and any(type(value) in (int, float) and value != 0 for value in delta.values())
    ) or (type(unresolved) in (int, float) and unresolved != 0)


def _update_is_consistent(item: Mapping[str, Any]) -> bool:
    prior = item.get("prior")
    posterior = item.get("posterior")
    direction = item.get("direction")
    if type(prior) not in (int, float) or type(posterior) not in (int, float):
        return False
    return (
        direction == "strengthened" and posterior > prior
        or direction == "weakened" and posterior < prior
        or direction == "neutral" and posterior == prior
    )


def _validate_ledger_prompt_provenance(
    by_type: Mapping[str, list[Mapping[str, Any]]],
    *,
    violations: _ViolationBuckets,
) -> None:
    for frame in by_type.get("task_frame", []):
        trace = frame.get("framing_trace")
        frame_task = trace.get("task") if isinstance(trace, Mapping) else None
        if frame_task not in {
            "frame_open_question",
            "repair_task_frame",
        } or not _trace_declaration_matches(trace, task=frame_task):
            violations.add("causal", "task-frame prompt/schema provenance drift")
    for event in by_type.get("evidence_event", []):
        trace = event.get("model_trace")
        if not isinstance(trace, Mapping):
            violations.add("causal", "Evidence lacks model trace")
            continue
        task = trace.get("task")
        if task not in {
            "judge_evidence",
            "repair_evidence_judgment",
        } or not _trace_declaration_matches(trace, task=task):
            violations.add("causal", "Evidence prompt/schema provenance drift")


def _trace_declaration_matches(value: object, *, task: str) -> bool:
    if not isinstance(value, Mapping) or task not in _PROMPT_SCHEMA:
        return False
    prompt_id, schema_name = _PROMPT_SCHEMA[task]
    return (
        value.get("task") == task
        and value.get("prompt_id") == prompt_id
        and value.get("prompt_version") == "v0.2"
        and value.get("schema_name") == schema_name
        and value.get("schema_version") == "v0.2"
    )


def _validate_budgets(
    *,
    summary: Mapping[str, Any] | None,
    reserved_actions: int,
    provider_calls: int,
    provider_tokens: int,
    violations: _ViolationBuckets,
) -> None:
    if summary is None:
        return
    declared = {
        "terminal_actions": reserved_actions,
        "model_calls": provider_calls,
    }
    for field, actual in declared.items():
        value = summary.get(field)
        if type(value) is not int or value < 0 or value != actual:
            violations.add("budget", f"summary {field} counter mismatch")
    if "provider_tokens" in summary and summary.get("provider_tokens") != provider_tokens:
        violations.add("budget", "summary provider token counter mismatch")
    limits = {
        "max_total_actions": _DEFAULT_MAX_ACTIONS,
        "max_model_calls": _DEFAULT_MAX_MODEL_CALLS,
        "max_provider_tokens": _DEFAULT_MAX_PROVIDER_TOKENS,
    }
    for field in tuple(limits):
        if field in summary:
            value = summary[field]
            if type(value) is not int or value <= 0:
                violations.add("budget", f"invalid {field}")
            else:
                limits[field] = value
    if reserved_actions > limits["max_total_actions"]:
        violations.add("budget", "terminal action budget exceeded")
    if provider_calls > limits["max_model_calls"]:
        violations.add("budget", "model-call budget exceeded")
    if provider_tokens > limits["max_provider_tokens"]:
        violations.add("budget", "provider-token budget exceeded")


def _validate_atif(
    *,
    trajectory_path: Path | None,
    actions: Sequence[ActionObservation],
    reserved_actions: int,
    causal_actions: Sequence[CausalActionRecord],
    signals: Mapping[str, ExternalSignal],
    provider_tokens: int,
    provider_calls: int,
    provider_model: str | None,
    violations: _ViolationBuckets,
    required: bool,
) -> None:
    if trajectory_path is None:
        if required:
            violations.add("adapter", "missing trajectory.json")
        return
    payload = _read_object(trajectory_path, violations, "adapter", required=True)
    if payload is None:
        return
    try:
        valid = TrajectoryValidator().validate(dict(payload))
    except Exception:
        valid = False
    if not valid:
        violations.add("adapter", "Harbor ATIF-v1.7 validation failed")
        return
    if payload.get("schema_version") != "ATIF-v1.7":
        violations.add("adapter", "trajectory schema identity drift")
    steps = payload.get("steps")
    if not isinstance(steps, Sequence) or isinstance(steps, str | bytes):
        violations.add("adapter", "trajectory steps are invalid")
        return
    if [step.get("step_id") for step in steps if isinstance(step, Mapping)] != list(
        range(1, len(steps) + 1)
    ):
        violations.add("adapter", "trajectory step IDs are not sequential")
    action_by_id = {item.action_id: item for item in causal_actions}
    signal_id_by_action = {
        action_id: signal.id
        for signal in signals.values()
        if (action_id := _signal_action_id(signal)) is not None
    }
    seen_actions: set[str] = set()
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        extra = step.get("extra")
        if not isinstance(extra, Mapping) or extra.get("kind") != "terminal_action":
            continue
        action_id = extra.get("action_id")
        action = action_by_id.get(action_id) if isinstance(action_id, str) else None
        if action is None or action_id in seen_actions:
            violations.add("adapter", "trajectory has orphan or duplicate action")
            continue
        seen_actions.add(action_id)
        calls = step.get("tool_calls")
        observation = step.get("observation")
        results = observation.get("results") if isinstance(observation, Mapping) else None
        if (
            not isinstance(calls, Sequence)
            or isinstance(calls, str | bytes)
            or len(calls) != 1
            or not isinstance(results, Sequence)
            or isinstance(results, str | bytes)
            or len(results) != 1
            or not isinstance(calls[0], Mapping)
            or not isinstance(results[0], Mapping)
            or results[0].get("source_call_id") != calls[0].get("tool_call_id")
            or extra.get("request_fingerprint") != action.request_fingerprint
            or extra.get("signal_id") != signal_id_by_action.get(action_id)
            or results[0].get("content") != action.observation.model_facing_output
        ):
            violations.add("adapter", f"trajectory linkage mismatch {action_id}")
    if seen_actions != set(action_by_id):
        violations.add("adapter", "trajectory action cardinality mismatch")
    metrics = payload.get("final_metrics")
    extra_metrics = metrics.get("extra") if isinstance(metrics, Mapping) else None
    prompt = metrics.get("total_prompt_tokens") if isinstance(metrics, Mapping) else None
    completion = metrics.get("total_completion_tokens") if isinstance(metrics, Mapping) else None
    if (
        not isinstance(extra_metrics, Mapping)
        or extra_metrics.get("provider_tokens_used") != provider_tokens
        or extra_metrics.get("model_calls_used") != provider_calls
        or extra_metrics.get("terminal_actions_used") != reserved_actions
        or type(prompt) is not int
        or type(completion) is not int
        or prompt + completion != provider_tokens
    ):
        violations.add("budget", "trajectory provider-token or budget totals disagree")
    agent = payload.get("agent")
    if (
        provider_model is not None
        and isinstance(agent, Mapping)
        and agent.get("model_name") != provider_model
    ):
        violations.add("provider", "trajectory provider model identity drift")


def _classify_recorded_errors(
    records: Sequence[Mapping[str, Any]],
    *,
    violations: _ViolationBuckets,
) -> None:
    for record in records:
        category = record.get("category")
        bucket = _ERROR_CATEGORY_MAP.get(category) if isinstance(category, str) else None
        if bucket is not None:
            violations.add(bucket, f"recorded {category}")


def _policy_denied_action_count(
    records: Sequence[Mapping[str, Any]],
    *,
    violations: _ViolationBuckets,
) -> int:
    indexes: set[int] = set()
    for record in records:
        if record.get("category") != "policy_error":
            continue
        action_index = record.get("action_index")
        if (
            type(action_index) is not int
            or action_index < 1
            or action_index in indexes
            or record.get("error_type") != "PolicyViolation"
            or not isinstance(record.get("probe_id"), str)
        ):
            violations.add("causal", "invalid policy-denied action reservation")
            continue
        indexes.add(action_index)
    return len(indexes)


__all__ = ["ConformanceReport", "TraceClassification", "validate_trial_trace"]
