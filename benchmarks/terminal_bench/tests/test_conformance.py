from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

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
from bayesprobe_terminal_bench.gateway import HarborProbeToolGateway
from bayesprobe_terminal_bench.runner_factory import build_runner


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
        return TerminalProbePlan(
            mode="intervene",
            steps=(
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
        elif self.scenario in {"inspect", "repair_causal_discard"}:
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
        if self.scenario == "postcondition":
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
