from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from bayesprobe import (
    AutonomousQuestionProgressKind,
    DeterministicModelGateway,
    EpistemicOrigin,
    EvidenceJudgmentRepairPolicy,
    HypothesisRelation,
    HypothesisSeed,
    InitializeRunInput,
    StructuredModelRequest,
    TaskKind,
)
from bayesprobe_terminal_bench.actions import (
    ActionObservation,
    ShellAction,
    TerminalPlanStep,
    TerminalProbePlan,
    TransitionPrediction,
    WriteFileAction,
)
from bayesprobe_terminal_bench.artifacts import TrialArtifactStore
from bayesprobe_terminal_bench.causal import CausalEvidenceModelGateway
from bayesprobe_terminal_bench.config import RunBudget, TerminalBenchConfig
from bayesprobe_terminal_bench.conformance import (
    ConformanceReport,
    TraceClassification,
    validate_trial_trace,
)
from bayesprobe_terminal_bench.gateway import HarborProbeToolGateway
from bayesprobe_terminal_bench.runner_factory import build_runner


CAUSAL_FIXTURES = Path(__file__).parent / "fixtures" / "causal_traces"
HISTORICAL_FIXTURES = Path(__file__).parent / "fixtures" / "historical_traces"


def _copy_conformant_fixture(tmp_path: Path, name: str) -> Path:
    fixture = tmp_path / name
    shutil.copytree(
        CAUSAL_FIXTURES / "conformant-inspect-intervene-verify",
        fixture,
    )
    return fixture


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_jsonl_objects(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


class ScriptedTerminalPlanner:
    def __init__(self, scenario: str = "inspect") -> None:
        self.scenario = scenario
        self.plan_calls = 0

    def plan(
        self,
        *,
        probe: object,
        context: object,
        history: tuple[ActionObservation, ...],
    ) -> TerminalProbePlan:
        self.plan_calls += 1
        assert history == ()
        if self.scenario in {"inspect", "repair_causal_discard"}:
            return TerminalProbePlan(
                mode="inspect",
                steps=(
                    TerminalPlanStep(
                        role="inspect",
                        action=ShellAction(command="ls"),
                    ),
                ),
                expected_observation=(
                    "The observed listing discriminates the hypotheses."
                ),
            )
        predictions = ()
        if self.scenario == "causal_transition":
            predictions = (
                TransitionPrediction(
                    hypothesis_id="H1",
                    expected_transition="The verifier passes if the root cause is H1.",
                ),
                TransitionPrediction(
                    hypothesis_id="H2",
                    expected_transition="The verifier still fails if the effect is H2.",
                ),
            )
        leading_steps = (
            (
                TerminalPlanStep(
                    role="inspect",
                    action=ShellAction(command="ls"),
                ),
            )
            if self.scenario == "inspect_intervene_verify"
            else ()
        )
        return TerminalProbePlan(
            mode="intervene",
            steps=leading_steps + (
                TerminalPlanStep(
                    role="intervene",
                    action=WriteFileAction(
                        path="/workspace/result.txt",
                        content="semaphore implementation",
                    ),
                ),
                TerminalPlanStep(
                    role="verify",
                    action=ShellAction(command="cat /workspace/result.txt"),
                    verification_target="the declared terminal postcondition",
                ),
            ),
            expected_observation="The declared mutation is followed by verification.",
            transition_predictions=predictions,
        )


class SupportingObservationBridge:
    def __init__(self, scenario: str = "inspect") -> None:
        self.scenario = scenario
        self.execute_calls = 0

    def execute(
        self,
        action: ShellAction | WriteFileAction,
        action_index: int,
    ) -> ActionObservation:
        self.execute_calls += 1
        if isinstance(action, WriteFileAction):
            output = "ACKNOWLEDGED: the Semaphore implementation was written."
            before, after = "env:0", "env:1"
        elif (
            self.scenario in {"inspect", "repair_causal_discard"}
            or isinstance(action, ShellAction)
            and action.command == "ls"
        ):
            output = "SUPPORTS H1: ls reports that the expected file is missing."
            before, after = "env:0", "env:0"
        else:
            output = "VERIFIED: the declared postcondition passes."
            before, after = "env:1", "env:1"
        return ActionObservation(
            action_index=action_index,
            action=action,
            stdout=output,
            stderr="",
            return_code=0,
            timed_out=False,
            duration_ms=1,
            pre_environment_state_id=before,
            post_environment_state_id=after,
            full_output_sha256=hashlib.sha256(output.encode("utf-8")).hexdigest(),
            model_facing_output=output,
            output_truncated=False,
        )


class TerminalEvidenceDelegate:
    adapter_kind = "terminal-conformance-delegate"
    model_identity = "terminal-conformance-model"

    def __init__(self, scenario: str) -> None:
        self.scenario = scenario
        self.config = object()
        self.invocation_observer = object()
        self.requests: list[StructuredModelRequest] = []

    def complete_structured(self, request: StructuredModelRequest) -> dict[str, Any]:
        self.requests.append(request)
        if request.task == "frame_open_question":
            return self._frame()
        if request.task == "design_probes":
            return {
                "proposals": [
                    {
                        "purpose": "hypothesis_discrimination",
                        "target_hypotheses": ["H1", "H2"],
                        "inquiry_goal": "Discriminate the two terminal hypotheses.",
                        "expected_observation": "A terminal result favors one target.",
                        "support_condition": {
                            "H1": "The result matches H1.",
                            "H2": "The result matches H2.",
                        },
                        "weaken_condition": {
                            "H1": "The result contradicts H1.",
                            "H2": "The result contradicts H2.",
                        },
                        "reframe_condition": None,
                        "required_capability": "repository_read",
                    }
                ]
            }
        if request.task == "judge_evidence":
            return self._judgment(request)
        if request.task == "repair_evidence_judgment":
            if self.scenario != "repair_causal_discard":
                raise AssertionError(f"unexpected model task: {request.task}")
            return {
                "evidence_type": "supporting",
                "likelihoods": {"H_unbound_target": "strongly_confirming"},
                "unresolved_likelihood": None,
                "frame_fit": "explained_by_named",
                "unexplained_observation": None,
                "interpretation": "A repair response with a mismatched target.",
                "quality_overrides": {},
            }
        if request.task == "project_answer":
            return {
                "answer": "The terminal evidence supports the recorded result.",
                "contract_sections": {
                    "result": "The registered terminal result was assessed.",
                    "verification": "The declared verifier output was used.",
                    "uncertainty": "Only the registered causal route contributed.",
                },
                "main_uncertainty": "The fixture covers one bounded plan.",
                "weakest_assumption": "The synthetic observation is representative.",
                "cited_evidence_ids": [],
            }
        raise AssertionError(f"unexpected model task: {request.task}")

    def _frame(self) -> dict[str, Any]:
        if self.scenario in {"postcondition", "inspect_intervene_verify"}:
            types = ("postcondition", "invariant")
            statements = (
                "The declared postcondition holds after the intervention.",
                "The required invariant holds after the intervention.",
            )
        elif self.scenario == "causal_transition":
            types = ("root_cause", "causal_effect")
            statements = (
                "The first root cause explains the transition.",
                "The second causal effect explains the transition.",
            )
        elif self.scenario == "policy_discard":
            types = ("implementation_policy", "patch_choice")
            statements = (
                "Use an asyncio Semaphore implementation.",
                "Use an unexecuted asyncio TaskGroup implementation.",
            )
        else:
            types = ("root_cause", "current_behavior")
            statements = (
                "The expected file is missing.",
                "The current workspace listing exposes the defect.",
            )
        return {
            "task_kind": "design",
            "answer_relationship": "synthesis",
            "answer_contract": {
                "objective": "Repair and verify the terminal task.",
                "answer_value_type": "structured_text",
                "answer_format": "terminal change with verification",
                "required_sections": ["result", "verification", "uncertainty"],
                "decision_form": "environment_change",
                "permits_synthesis": True,
            },
            "competition": "independent",
            "coverage": "open",
            "hypotheses": [
                {
                    "statement": statement,
                    "type": hypothesis_type,
                    "scope": "The registered terminal environment state.",
                    "falsifiers": [f"The terminal result falsifies {index}."],
                    "predictions": [f"The terminal result predicts {index}."],
                    "answer_value": None,
                }
                for index, (statement, hypothesis_type) in enumerate(
                    zip(statements, types, strict=True),
                    start=1,
                )
            ],
            "coverage_statement": "The frame covers two terminal hypotheses.",
            "coverage_limitation": "Other terminal causes may remain.",
        }

    def _judgment(self, request: StructuredModelRequest) -> dict[str, Any]:
        raw = json.loads(request.input["signal"]["raw_content"])
        role = raw["causal_binding"]["action_role"]
        targets = request.input["target_hypotheses"]
        if self.scenario == "repair_causal_discard":
            return {
                "evidence_type": "not_a_valid_evidence_type",
                "likelihoods": {
                    target: "moderately_confirming" for target in targets
                },
                "unresolved_likelihood": None,
                "frame_fit": "explained_by_named",
                "unexplained_observation": None,
                "interpretation": "The initial semantic judgment requires repair.",
                "quality_overrides": {},
            }
        if role == "intervene" and self.scenario != "policy_discard":
            return {
                "evidence_type": "neutral",
                "likelihoods": {target: "neutral" for target in targets},
                "unresolved_likelihood": None,
                "frame_fit": "underdetermined",
                "unexplained_observation": None,
                "interpretation": "The mutation acknowledgement is semantically neutral.",
                "quality_overrides": {},
            }
        return {
            "evidence_type": "supporting",
            "likelihoods": {
                target: (
                    "strongly_confirming" if target == "H1" else "strongly_disconfirming"
                )
                for target in targets
            },
            "unresolved_likelihood": None,
            "frame_fit": "explained_by_named",
            "unexplained_observation": None,
            "interpretation": (
                "The Semaphore acknowledgement is compared with an unexecuted "
                "TaskGroup policy."
                if self.scenario == "policy_discard"
                else "The delegate semantically judges the terminal observation."
            ),
            "quality_overrides": {},
        }


class NoSignalGateway:
    def __init__(self) -> None:
        self.execute_calls = 0

    def execute_probe(self, *, probe: object, context: object) -> list[object]:
        self.execute_calls += 1
        return []


def _run_input(run_id: str) -> InitializeRunInput:
    return InitializeRunInput(
        run_id=run_id,
        problem="Determine which implementation diagnosis matches the observed test.",
        task_kind=TaskKind.DESIGN,
        hypothesis_relation=HypothesisRelation.EXCLUSIVE_EXHAUSTIVE,
        hypothesis_seeds=[
            HypothesisSeed(
                id="H1",
                statement="The file is missing.",
                prior=0.5,
                predictions=["ls reports missing"],
                falsifiers=["ls reports present"],
            ),
            HypothesisSeed(
                id="H2",
                statement="The file exists but is invalid.",
                prior=0.5,
                predictions=["parser rejects content"],
                falsifiers=["parser accepts content"],
            ),
        ],
    )


def _terminal_run_input(run_id: str) -> InitializeRunInput:
    return InitializeRunInput(
        run_id=run_id,
        problem="Repair and verify the terminal task using causally valid evidence.",
    )


def _guarded_scenario(
    tmp_path: Path,
    scenario: str,
):
    artifacts = TrialArtifactStore(
        tmp_path / scenario / "artifacts",
        restricted_values=(
            "The Semaphore acknowledgement is compared with an unexecuted "
            "TaskGroup policy.",
        ),
    )
    probe_gateway = HarborProbeToolGateway(
        planner=ScriptedTerminalPlanner(scenario),
        bridge=SupportingObservationBridge(scenario),
        artifacts=artifacts,
        budget=RunBudget(max_actions=3, max_model_calls=10),
    )
    delegate = TerminalEvidenceDelegate(scenario)
    model_gateway = CausalEvidenceModelGateway(
        delegate=delegate,
        registry=probe_gateway._causal,
        artifacts=artifacts,
    )
    runner = build_runner(
        model_gateway=model_gateway,
        probe_gateway=probe_gateway,
        ledger_path=tmp_path / scenario / "ledger.jsonl",
        config=TerminalBenchConfig(model="scripted", max_cycles=1),
    )
    if scenario == "repair_causal_discard":
        runner.core._evidence_gate._judgment_repair_policy = (
            EvidenceJudgmentRepairPolicy(max_attempts=2)
        )
    return runner, delegate, artifacts


def _posterior_by_id(belief_state) -> dict[str, float]:
    return {
        hypothesis.id: hypothesis.posterior
        for hypothesis in belief_state.hypotheses
    }


def _causal_decisions(artifacts: TrialArtifactStore) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (artifacts.root / "causal_decisions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]


def test_real_runner_closes_terminal_tool_signal_into_evidence_and_cycle(
    tmp_path: Path,
) -> None:
    runner, _, artifacts = _guarded_scenario(tmp_path, "inspect")

    result = runner.run_question(_terminal_run_input("terminal_conformance"))

    assert len(result.cycle_results) == 1
    cycle = result.cycle_results[0]
    assert cycle.cycle.boundary_status.value == "integrated"
    assert len(cycle.signals) == 1
    signal = cycle.signals[0]
    assert signal.provenance is not None
    assert signal.provenance.epistemic_origin is EpistemicOrigin.TOOL_RESULT
    event = next(
        item for item in cycle.evidence_events if item.derived_from_signal == signal.id
    )
    assert event.epistemic_origin is EpistemicOrigin.TOOL_RESULT
    assert event.discard_reason is None
    assert any(
        event.id in delta.caused_by_event_ids for delta in cycle.contribution_deltas
    )
    assert any(
        event.id in update.sensitivity.get("caused_by_event_ids", [])
        for update in cycle.belief_updates
    )
    assert [decision["reason_code"] for decision in _causal_decisions(artifacts)] == [
        "state_scoped_inspection"
    ]


def test_policy_acknowledgement_discard_leaves_public_core_posterior_unchanged(
    tmp_path: Path,
) -> None:
    runner, delegate, artifacts = _guarded_scenario(tmp_path, "policy_discard")

    result = runner.run_question(_terminal_run_input("terminal_policy_discard"))

    cycle = result.cycle_results[0]
    assert len(cycle.evidence_events) == 2
    assert all(
        event.discard_reason
        == "schema_violation: causal_admissibility:unexecuted_policy_comparison"
        for event in cycle.evidence_events
    )
    assert cycle.contribution_deltas == []
    assert cycle.belief_updates == []
    assert _posterior_by_id(result.final_belief_state) == _posterior_by_id(
        result.initial_belief_state
    )
    decisions = _causal_decisions(artifacts)
    assert len(decisions) == 2
    assert all(
        decision["decision"] == "discard"
        and decision["reason_code"] == "unexecuted_policy_comparison"
        for decision in decisions
    )
    assert [
        request.task
        for request in delegate.requests
        if request.task in {"judge_evidence", "repair_evidence_judgment"}
    ] == ["judge_evidence", "judge_evidence"]
    assert not (artifacts.root / "errors.jsonl").exists()
    assert "TaskGroup policy" not in (
        artifacts.root / "causal_decisions.jsonl"
    ).read_text(encoding="utf-8")


def test_causally_rejected_repairs_never_reach_the_public_core_posterior(
    tmp_path: Path,
) -> None:
    runner, delegate, artifacts = _guarded_scenario(
        tmp_path,
        "repair_causal_discard",
    )

    result = runner.run_question(_terminal_run_input("terminal_repair_discard"))

    cycle = result.cycle_results[0]
    assert len(cycle.evidence_events) == 1
    event = cycle.evidence_events[0]
    assert event.discard_reason == (
        "schema_violation: repair failed after 2 attempt(s): "
        "causal_admissibility:target_mismatch"
    )
    assert event.model_trace["task"] == "repair_evidence_judgment"
    assert event.model_trace["repair_attempt_index"] == 2
    assert cycle.contribution_deltas == []
    assert cycle.belief_updates == []
    assert _posterior_by_id(result.final_belief_state) == _posterior_by_id(
        result.initial_belief_state
    )
    assert [
        request.task
        for request in delegate.requests
        if request.task in {"judge_evidence", "repair_evidence_judgment"}
    ] == [
        "judge_evidence",
        "repair_evidence_judgment",
        "repair_evidence_judgment",
    ]
    assert [
        (decision["decision"], decision["reason_code"])
        for decision in _causal_decisions(artifacts)
    ] == [
        ("admit", "state_scoped_inspection"),
        ("discard", "target_mismatch"),
        ("discard", "target_mismatch"),
    ]


@pytest.mark.parametrize(
    ("scenario", "verification_reason"),
    [
        ("postcondition", "verified_postcondition"),
        ("causal_transition", "preregistered_causal_transition"),
    ],
)
def test_valid_verification_routes_update_through_the_public_core(
    tmp_path: Path,
    scenario: str,
    verification_reason: str,
) -> None:
    runner, _, artifacts = _guarded_scenario(tmp_path, scenario)

    result = runner.run_question(_terminal_run_input(f"terminal_{scenario}"))

    cycle = result.cycle_results[0]
    signals_by_role = {
        json.loads(signal.raw_content)["causal_binding"]["action_role"]: signal
        for signal in cycle.signals
    }
    verify_event = next(
        event
        for event in cycle.evidence_events
        if event.derived_from_signal == signals_by_role["verify"].id
    )
    assert verify_event.discard_reason is None
    assert any(
        verify_event.id in delta.caused_by_event_ids
        for delta in cycle.contribution_deltas
    )
    assert any(
        verify_event.id in update.sensitivity.get("caused_by_event_ids", [])
        for update in cycle.belief_updates
    )
    assert _posterior_by_id(result.final_belief_state) != _posterior_by_id(
        result.initial_belief_state
    )
    assert {
        decision["action_role"]: decision["reason_code"]
        for decision in _causal_decisions(artifacts)
    } == {
        "intervene": "neutral_mutation_acknowledgement",
        "verify": verification_reason,
    }


def test_no_signal_execution_preserves_belief_until_core_rejects_empty_cycle(
    tmp_path: Path,
) -> None:
    gateway = NoSignalGateway()
    progress = []
    runner = build_runner(
        model_gateway=DeterministicModelGateway(),
        probe_gateway=gateway,
        ledger_path=tmp_path / "ledger.jsonl",
        config=TerminalBenchConfig(model="deterministic", max_cycles=1),
        progress_observer=progress.append,
    )

    with pytest.raises(
        ValueError,
        match="active_only cycles require at least one active signal",
    ):
        runner.run_question(_run_input("terminal_no_signal"))

    assert gateway.execute_calls == 1
    initialized = next(
        event
        for event in progress
        if event.kind is AutonomousQuestionProgressKind.INITIALIZATION_COMPLETED
    )
    collected = next(
        event
        for event in progress
        if event.kind is AutonomousQuestionProgressKind.SIGNALS_COLLECTED
    )
    initial = {
        item.id: item.posterior for item in initialized.belief_state.hypotheses
    }
    before_integration = {
        item.id: item.posterior for item in collected.belief_state.hypotheses
    }
    assert collected.signals == ()
    assert before_integration == initial
    assert all(
        event.kind is not AutonomousQuestionProgressKind.CYCLE_INTEGRATED
        for event in progress
    )


def test_conformance_report_is_strict_frozen_and_uses_the_fixed_public_enum() -> None:
    report = validate_trial_trace(
        CAUSAL_FIXTURES / "conformant-inspect-intervene-verify"
    )

    assert report.classification is TraceClassification.CONFORMANT
    assert report.model_config["strict"] is True
    assert report.model_config["frozen"] is True
    assert tuple(item.value for item in TraceClassification) == (
        "conformant",
        "provider_contract_error",
        "causal_conformance_error",
        "budget_error",
        "adapter_error",
    )
    with pytest.raises(ValidationError):
        report.complete_cycles = 0  # type: ignore[misc]
    with pytest.raises(ValidationError):
        ConformanceReport.model_validate(
            {**report.model_dump(), "unexpected": "field"}
        )


def test_conformant_inspect_intervene_verify_fixture_reports_mechanism_counts() -> None:
    report = validate_trial_trace(
        CAUSAL_FIXTURES / "conformant-inspect-intervene-verify"
    )

    expected = ConformanceReport(
        classification=TraceClassification.CONFORMANT,
        complete_cycles=1,
        plans=1,
        actions=3,
        signals=3,
        evidence_events=3,
        admitted_evidence=2,
        discarded_evidence=1,
        nonneutral_updates=2,
        violations=(),
        mechanism_metrics={
            "action_signal_ratio": 1.0,
            "admitted_evidence_rate": 2 / 3,
            "discarded_evidence_rate": 1 / 3,
            "nonneutral_updates_per_admitted_evidence": 1.0,
            "provider_tokens": 60,
        },
    )
    assert report == expected


def test_guard_discard_is_observable_without_becoming_a_causal_error() -> None:
    report = validate_trial_trace(
        CAUSAL_FIXTURES / "conformant-inspect-intervene-verify"
    )

    assert report.discarded_evidence == 1
    assert report.classification is TraceClassification.CONFORMANT
    assert not report.violations


def test_empty_artifact_directory_is_adapter_error(tmp_path: Path) -> None:
    report = validate_trial_trace(tmp_path)

    assert report.classification is TraceClassification.ADAPTER_ERROR
    assert any("envelope" in violation for violation in report.violations)


@pytest.mark.parametrize(
    "filename",
    [
        "bayesprobe_ledger.jsonl",
        "errors.jsonl",
        "provider_contract.jsonl",
        "provider_telemetry.jsonl",
        "plans.jsonl",
        "environment_actions.jsonl",
        "causal_actions.jsonl",
        "causal_decisions.jsonl",
        "summary.json",
        "trajectory.json",
    ],
)
@pytest.mark.parametrize(
    "contents",
    ["", " \n\t", "{}\n"],
    ids=["empty", "whitespace", "empty-object"],
)
def test_isolated_meaningless_envelope_file_is_adapter_error(
    tmp_path: Path,
    filename: str,
    contents: str,
) -> None:
    artifact_root = tmp_path / "isolated-envelope"
    artifact_root.mkdir()
    (artifact_root / filename).write_text(contents, encoding="utf-8")

    report = validate_trial_trace(artifact_root)

    assert report.classification is TraceClassification.ADAPTER_ERROR
    assert report.violations
    assert all(item.startswith("adapter:") for item in report.violations)


def test_nonempty_non_substantive_ledger_is_adapter_error(tmp_path: Path) -> None:
    artifact_root = tmp_path / "non-substantive-ledger"
    artifact_root.mkdir()
    _write_jsonl_objects(
        artifact_root / "bayesprobe_ledger.jsonl",
        [
            {
                "record_type": "task_admission",
                "payload": {"status": "admitted"},
            }
        ],
    )

    report = validate_trial_trace(artifact_root)

    assert report.classification is TraceClassification.ADAPTER_ERROR
    assert any("substantive" in item for item in report.violations)


def test_artifact_root_symlink_is_security_violation_without_following(
    tmp_path: Path,
) -> None:
    target = _copy_conformant_fixture(tmp_path, "symlink-target")
    artifact_root = tmp_path / "artifact-root-link"
    artifact_root.symlink_to(target, target_is_directory=True)

    report = validate_trial_trace(artifact_root)

    assert report.classification is TraceClassification.CAUSAL_CONFORMANCE_ERROR
    assert report.complete_cycles == 0
    assert report.plans == 0
    assert report.violations == ("security:artifact root symlink is forbidden",)


@pytest.mark.parametrize(
    "record",
    [{}, {"category": ""}, {"category": "unknown_error"}],
    ids=["missing", "empty", "unknown"],
)
def test_invalid_errors_category_is_adapter_error(
    tmp_path: Path,
    record: dict[str, str],
) -> None:
    fixture = _copy_conformant_fixture(tmp_path, "invalid-error-category")
    _write_jsonl_objects(fixture / "errors.jsonl", [record])

    report = validate_trial_trace(fixture)

    assert report.classification is TraceClassification.ADAPTER_ERROR
    assert "adapter:invalid errors.jsonl category at record 1" in report.violations


@pytest.mark.parametrize(
    ("category", "expected", "bucket"),
    [
        (
            "causal_conformance_error",
            TraceClassification.CAUSAL_CONFORMANCE_ERROR,
            "causal",
        ),
        (
            "provider_contract_error",
            TraceClassification.PROVIDER_CONTRACT_ERROR,
            "provider",
        ),
        ("budget_error", TraceClassification.BUDGET_ERROR, "budget"),
        ("adapter_error", TraceClassification.ADAPTER_ERROR, "adapter"),
    ],
)
def test_isolated_explicit_terminal_error_keeps_its_classification(
    tmp_path: Path,
    category: str,
    expected: TraceClassification,
    bucket: str,
) -> None:
    artifact_root = tmp_path / category
    artifact_root.mkdir()
    _write_jsonl_objects(
        artifact_root / "errors.jsonl",
        [{"category": category}],
    )

    report = validate_trial_trace(artifact_root)

    assert report.classification is expected
    assert report.violations == (f"{bucket}:recorded {category}",)


def test_policy_error_category_uses_policy_validation_not_adapter_fallback(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "policy-error"
    artifact_root.mkdir()
    _write_jsonl_objects(
        artifact_root / "errors.jsonl",
        [
            {
                "action_index": 1,
                "category": "policy_error",
                "error_type": "PolicyViolation",
                "probe_id": "P1",
            }
        ],
    )

    report = validate_trial_trace(artifact_root)

    assert report.classification is TraceClassification.CAUSAL_CONFORMANCE_ERROR
    assert any("causal diagnostics" in item for item in report.violations)
    assert not any("invalid errors.jsonl category" in item for item in report.violations)


def test_completed_trace_without_plans_or_actions_is_causal_error(
    tmp_path: Path,
) -> None:
    fixture = _copy_conformant_fixture(tmp_path, "completed-without-causal-envelope")
    ledger_path = fixture / "bayesprobe_ledger.jsonl"
    retained_types = {
        "task_admission",
        "task_frame",
        "run",
        "belief_state",
        "cycle",
        "answer_projection",
    }
    _write_jsonl_objects(
        ledger_path,
        [
            record
            for record in _read_jsonl_objects(ledger_path)
            if record["record_type"] in retained_types
        ],
    )
    for name in (
        "plans.jsonl",
        "environment_actions.jsonl",
        "causal_actions.jsonl",
        "causal_decisions.jsonl",
        "trajectory.json",
    ):
        (fixture / name).unlink()
    summary_path = fixture / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["terminal_actions"] = 0
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    report = validate_trial_trace(fixture)

    assert report.classification is TraceClassification.CAUSAL_CONFORMANCE_ERROR
    assert any("plans" in violation for violation in report.violations)
    assert any("executed actions" in violation for violation in report.violations)


@pytest.mark.parametrize(
    "record",
    [
        {"unexpected": "previously ignored"},
        {"category": "policy_error", "stage": "action_policy"},
    ],
)
def test_unknown_or_malformed_causal_decision_record_is_causal_error(
    tmp_path: Path,
    record: dict[str, str],
) -> None:
    fixture = _copy_conformant_fixture(tmp_path, "malformed-causal-decision")
    decisions_path = fixture / "causal_decisions.jsonl"
    records = _read_jsonl_objects(decisions_path)
    records.append(record)
    _write_jsonl_objects(decisions_path, records)

    report = validate_trial_trace(fixture)

    assert report.classification is TraceClassification.CAUSAL_CONFORMANCE_ERROR
    assert any("decision record" in violation for violation in report.violations)


def test_duplicate_final_causal_decision_is_causal_error(tmp_path: Path) -> None:
    fixture = _copy_conformant_fixture(tmp_path, "duplicate-final-decision")
    decisions_path = fixture / "causal_decisions.jsonl"
    records = _read_jsonl_objects(decisions_path)
    records.append(records[0])
    _write_jsonl_objects(decisions_path, records)

    report = validate_trial_trace(fixture)

    assert report.classification is TraceClassification.CAUSAL_CONFORMANCE_ERROR
    assert any("exactly one" in violation for violation in report.violations)


def test_discarded_evidence_requires_guard_discard_independent_of_reason_text(
    tmp_path: Path,
) -> None:
    fixture = _copy_conformant_fixture(tmp_path, "discard-with-admit-decision")
    ledger_path = fixture / "bayesprobe_ledger.jsonl"
    ledger = _read_jsonl_objects(ledger_path)
    discarded = next(
        record["payload"]
        for record in ledger
        if record["record_type"] == "evidence_event"
        and record["payload"]["discard_reason"] is not None
    )
    discarded["discard_reason"] = "guard rejected the judgment"
    _write_jsonl_objects(ledger_path, ledger)

    decisions_path = fixture / "causal_decisions.jsonl"
    decisions = _read_jsonl_objects(decisions_path)
    decision = next(
        record for record in decisions if record["signal_id"] == discarded["derived_from_signal"]
    )
    decision["decision"] = "admit"
    decision["reason_code"] = "state_scoped_inspection"
    _write_jsonl_objects(decisions_path, decisions)

    report = validate_trial_trace(fixture)

    assert report.classification is TraceClassification.CAUSAL_CONFORMANCE_ERROR
    assert any("guard discard" in violation for violation in report.violations)


def test_guard_discard_accepts_arbitrary_public_core_reason_text(tmp_path: Path) -> None:
    fixture = _copy_conformant_fixture(tmp_path, "discard-with-free-text-reason")
    ledger_path = fixture / "bayesprobe_ledger.jsonl"
    ledger = _read_jsonl_objects(ledger_path)
    discarded = next(
        record["payload"]
        for record in ledger
        if record["record_type"] == "evidence_event"
        and record["payload"]["discard_reason"] is not None
    )
    discarded["discard_reason"] = "guard rejected the judgment"
    _write_jsonl_objects(ledger_path, ledger)

    report = validate_trial_trace(fixture)

    assert report.classification is TraceClassification.CONFORMANT
    assert report.discarded_evidence == 1


def test_provider_contract_attempts_require_provider_telemetry(tmp_path: Path) -> None:
    fixture = tmp_path / "attempts-without-telemetry"
    fixture.mkdir()
    shutil.copyfile(
        CAUSAL_FIXTURES
        / "conformant-inspect-intervene-verify"
        / "provider_contract.jsonl",
        fixture / "provider_contract.jsonl",
    )

    report = validate_trial_trace(fixture)

    assert report.classification is TraceClassification.PROVIDER_CONTRACT_ERROR
    assert any("telemetry" in violation for violation in report.violations)


def test_completed_substantive_trace_requires_provider_telemetry(
    tmp_path: Path,
) -> None:
    fixture = _copy_conformant_fixture(tmp_path, "trace-without-telemetry")
    (fixture / "provider_telemetry.jsonl").unlink()

    report = validate_trial_trace(fixture)

    assert report.classification is TraceClassification.PROVIDER_CONTRACT_ERROR
    assert any("telemetry" in violation for violation in report.violations)


def test_successful_provider_calls_require_system_fingerprint_key(
    tmp_path: Path,
) -> None:
    fixture = _copy_conformant_fixture(tmp_path, "missing-fingerprint-keys")
    telemetry_path = fixture / "provider_telemetry.jsonl"
    telemetry = _read_jsonl_objects(telemetry_path)
    for record in telemetry:
        record.pop("system_fingerprint")
    _write_jsonl_objects(telemetry_path, telemetry)

    report = validate_trial_trace(fixture)

    assert report.classification is TraceClassification.PROVIDER_CONTRACT_ERROR
    assert any("fingerprint key" in violation for violation in report.violations)


def test_explicit_null_provider_fingerprint_is_available_and_conformant(
    tmp_path: Path,
) -> None:
    fixture = _copy_conformant_fixture(tmp_path, "explicit-null-fingerprint")
    telemetry_path = fixture / "provider_telemetry.jsonl"
    telemetry = _read_jsonl_objects(telemetry_path)
    for record in telemetry:
        record["system_fingerprint"] = None
    _write_jsonl_objects(telemetry_path, telemetry)

    report = validate_trial_trace(fixture)

    assert report.classification is TraceClassification.CONFORMANT


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("model", "drifted-model"),
        ("system_fingerprint", "drifted-fingerprint"),
        ("system_fingerprint", None),
    ],
)
def test_provider_identity_value_or_availability_drift_is_provider_error(
    tmp_path: Path,
    field: str,
    replacement: str | None,
) -> None:
    fixture = _copy_conformant_fixture(tmp_path, f"provider-drift-{field}")
    telemetry_path = fixture / "provider_telemetry.jsonl"
    telemetry = _read_jsonl_objects(telemetry_path)
    telemetry[0][field] = replacement
    _write_jsonl_objects(telemetry_path, telemetry)

    report = validate_trial_trace(fixture)

    assert report.classification is TraceClassification.PROVIDER_CONTRACT_ERROR
    assert any("identity drift" in violation for violation in report.violations)


@pytest.mark.parametrize(
    "mutation",
    [
        "ls_to_write_file",
        "arguments",
        "call_id",
        "function",
        "result_metadata",
    ],
)
def test_atif_terminal_action_must_exactly_match_executed_action(
    tmp_path: Path,
    mutation: str,
) -> None:
    fixture = _copy_conformant_fixture(tmp_path, f"atif-{mutation}")
    trajectory_path = fixture / "trajectory.json"
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    step = next(
        item
        for item in trajectory["steps"]
        if item.get("extra", {}).get("kind") == "terminal_action"
    )
    call = step["tool_calls"][0]
    result = step["observation"]["results"][0]
    if mutation == "ls_to_write_file":
        call["function_name"] = "terminal.write_file"
        call["arguments"] = {
            "content": "replacement",
            "path": "/workspace/replacement.txt",
            "type": "write_file",
        }
    elif mutation == "arguments":
        call["arguments"]["command"] = "pwd"
    elif mutation == "call_id":
        call["tool_call_id"] = "tool:mutated-action"
        result["source_call_id"] = "tool:mutated-action"
    elif mutation == "function":
        call["function_name"] = "terminal.write_file"
    else:
        result["extra"]["action_index"] = 99
    trajectory_path.write_text(
        json.dumps(trajectory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = validate_trial_trace(fixture)

    assert report.classification is TraceClassification.ADAPTER_ERROR
    assert any("trajectory linkage mismatch" in item for item in report.violations)


@pytest.mark.parametrize(
    ("record_type", "mutation", "label"),
    [
        ("evidence_event", "extra", "EvidenceEvent"),
        ("evidence_event", "missing_required", "EvidenceEvent"),
        ("evidence_event", "numeric_string", "EvidenceEvent"),
        ("evidence_event", "nonfinite", "EvidenceEvent"),
        ("evidence_event", "likelihoods_not_map", "EvidenceEvent"),
        ("evidence_contribution_delta", "extra", "EvidenceContributionDelta"),
        (
            "evidence_contribution_delta",
            "malformed_neutral",
            "EvidenceContributionDelta",
        ),
        (
            "evidence_contribution_delta",
            "nonfinite",
            "EvidenceContributionDelta",
        ),
        ("belief_update", "extra", "BeliefUpdate"),
        ("belief_update", "numeric_string", "BeliefUpdate"),
        ("belief_update", "nonfinite", "BeliefUpdate"),
        ("belief_update", "causes_not_list", "BeliefUpdate"),
    ],
)
def test_epistemic_payloads_are_strict_before_causal_reasoning(
    tmp_path: Path,
    record_type: str,
    mutation: str,
    label: str,
) -> None:
    fixture = _copy_conformant_fixture(
        tmp_path,
        f"strict-{record_type}-{mutation}",
    )
    ledger_path = fixture / "bayesprobe_ledger.jsonl"
    ledger = _read_jsonl_objects(ledger_path)
    payload = next(
        record["payload"]
        for record in ledger
        if record["record_type"] == record_type
    )
    if mutation == "extra":
        payload["unexpected"] = "forbidden"
    elif mutation == "missing_required":
        payload.pop("target_hypotheses")
    elif mutation == "numeric_string":
        field = "reliability" if record_type == "evidence_event" else "prior"
        payload[field] = "0.5"
    elif mutation == "nonfinite":
        field = (
            "reliability"
            if record_type == "evidence_event"
            else "per_hypothesis_delta"
            if record_type == "evidence_contribution_delta"
            else "posterior"
        )
        if field == "per_hypothesis_delta":
            payload[field]["H1"] = float("inf")
        else:
            payload[field] = float("inf")
    elif mutation == "likelihoods_not_map":
        payload["likelihoods"] = ["neutral"]
    elif mutation == "malformed_neutral":
        payload["per_hypothesis_delta"] = {"H1": "0.0", "H2": "0.0"}
        ledger = [
            record for record in ledger if record["record_type"] != "belief_update"
        ]
    else:
        payload["sensitivity"]["caused_by_event_ids"] = "not-a-list"
    _write_jsonl_objects(ledger_path, ledger)

    report = validate_trial_trace(fixture)

    assert report.classification is TraceClassification.CAUSAL_CONFORMANCE_ERROR
    assert any(f"invalid {label} payload" in item for item in report.violations)


def test_historical_traces_replay_to_the_preregistered_classifications() -> None:
    manifest = json.loads(
        (HISTORICAL_FIXTURES / "manifest.json").read_text(encoding="utf-8")
    )

    actual = {
        trace["task_id"]: validate_trial_trace(
            HISTORICAL_FIXTURES / trace["task_id"].split("/", maxsplit=1)[1]
        ).classification.value
        for trace in manifest["traces"]
    }

    assert actual == {
        trace["task_id"]: trace["expected_classification"]
        for trace in manifest["traces"]
    }


@pytest.mark.parametrize(
    "fixture_name",
    [
        "ambiguous-update",
        "discarded-update",
        "environment-state",
        "request-fingerprint",
    ],
)
def test_each_broken_binding_state_or_update_fixture_is_causal_error(
    fixture_name: str,
) -> None:
    report = validate_trial_trace(
        CAUSAL_FIXTURES / "broken-bindings" / fixture_name
    )

    assert report.classification is TraceClassification.CAUSAL_CONFORMANCE_ERROR
    assert report.violations


@pytest.mark.parametrize(
    ("case", "expected", "expected_order"),
    [
        ("conformant", TraceClassification.CONFORMANT, ()),
        ("adapter", TraceClassification.ADAPTER_ERROR, ("adapter",)),
        (
            "budget_over_adapter",
            TraceClassification.BUDGET_ERROR,
            ("budget", "adapter"),
        ),
        (
            "provider_over_budget",
            TraceClassification.PROVIDER_CONTRACT_ERROR,
            ("provider", "budget", "adapter"),
        ),
        (
            "causal_over_provider",
            TraceClassification.CAUSAL_CONFORMANCE_ERROR,
            ("causal", "provider", "budget", "adapter"),
        ),
        (
            "security_over_all",
            TraceClassification.CAUSAL_CONFORMANCE_ERROR,
            ("security", "causal", "provider", "budget", "adapter"),
        ),
    ],
)
def test_classification_precedence_uses_real_detector_collisions(
    tmp_path: Path,
    case: str,
    expected: TraceClassification,
    expected_order: tuple[str, ...],
) -> None:
    fixture = _copy_conformant_fixture(tmp_path, case)
    if case != "conformant":
        (fixture / "trajectory.json").unlink()
    if case in {
        "budget_over_adapter",
        "provider_over_budget",
        "causal_over_provider",
        "security_over_all",
    }:
        summary_path = fixture / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["max_total_actions"] = 1
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
    if case in {"provider_over_budget", "causal_over_provider", "security_over_all"}:
        telemetry_path = fixture / "provider_telemetry.jsonl"
        telemetry = _read_jsonl_objects(telemetry_path)
        for record in telemetry:
            record.pop("system_fingerprint")
        _write_jsonl_objects(telemetry_path, telemetry)
    if case in {"causal_over_provider", "security_over_all"}:
        actions_path = fixture / "causal_actions.jsonl"
        actions = _read_jsonl_objects(actions_path)
        actions[0]["request_fingerprint"] = "sha256:" + "0" * 64
        _write_jsonl_objects(actions_path, actions)
    if case == "security_over_all":
        (fixture / "unsafe.txt").write_text(
            "attempted read of /root/evaluator/secret",
            encoding="utf-8",
        )

    report = validate_trial_trace(fixture)
    actual_order = tuple(
        dict.fromkeys(violation.split(":", maxsplit=1)[0] for violation in report.violations)
    )

    assert report.classification is expected
    assert actual_order == expected_order


def test_causal_fixtures_are_secret_free_evaluator_free_and_portable() -> None:
    manifest = json.loads(
        (CAUSAL_FIXTURES / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == "terminal_causal_trace_fixtures:v1"
    actual_files = {
        path.relative_to(CAUSAL_FIXTURES).as_posix(): (
            "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        )
        for path in CAUSAL_FIXTURES.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    assert manifest["files"] == actual_files

    for path in CAUSAL_FIXTURES.rglob("*"):
        assert not path.is_symlink()
        if not path.is_file():
            continue
        contents = path.read_text(encoding="utf-8")
        assert "/Users/" not in contents
        assert "/root/evaluator" not in contents
        assert "sk-1234567890ab" not in contents
