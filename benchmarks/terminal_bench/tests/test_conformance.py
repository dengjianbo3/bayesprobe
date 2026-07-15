from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from bayesprobe import (
    AutonomousQuestionProgressKind,
    DeterministicModelGateway,
    EpistemicOrigin,
    HypothesisRelation,
    HypothesisSeed,
    InitializeRunInput,
    TaskKind,
)
from bayesprobe_terminal_bench.actions import (
    ActionObservation,
    ShellAction,
    TerminalProbePlan,
)
from bayesprobe_terminal_bench.artifacts import TrialArtifactStore
from bayesprobe_terminal_bench.config import RunBudget, TerminalBenchConfig
from bayesprobe_terminal_bench.gateway import HarborProbeToolGateway
from bayesprobe_terminal_bench.runner_factory import build_runner


class ScriptedTerminalPlanner:
    def __init__(self) -> None:
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
        return TerminalProbePlan(
            mode="inspect",
            actions=(ShellAction(command="ls"),),
            expected_observation="The observed listing discriminates the hypotheses.",
        )


class SupportingObservationBridge:
    def __init__(self) -> None:
        self.execute_calls = 0

    def execute(self, action: ShellAction, action_index: int) -> ActionObservation:
        self.execute_calls += 1
        output = "SUPPORTS H1: ls reports that the expected file is missing."
        return ActionObservation(
            action_index=action_index,
            action=action,
            stdout=output,
            stderr="",
            return_code=0,
            timed_out=False,
            duration_ms=1,
            pre_environment_state_id="env:0",
            post_environment_state_id="env:0",
            full_output_sha256=hashlib.sha256(output.encode("utf-8")).hexdigest(),
            model_facing_output=output,
            output_truncated=False,
        )


class NoSignalGateway:
    def __init__(self) -> None:
        self.execute_calls = 0

    def execute_probe(self, *, probe: object, context: object) -> list[object]:
        self.execute_calls += 1
        return []


@pytest.fixture
def tool_signal_gateway(tmp_path: Path) -> HarborProbeToolGateway:
    return HarborProbeToolGateway(
        planner=ScriptedTerminalPlanner(),
        bridge=SupportingObservationBridge(),
        artifacts=TrialArtifactStore(
            tmp_path / "artifacts",
            restricted_values=(),
        ),
        budget=RunBudget(max_actions=1, max_model_calls=1),
    )


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


def test_real_runner_closes_terminal_tool_signal_into_evidence_and_cycle(
    tmp_path: Path,
    tool_signal_gateway: HarborProbeToolGateway,
) -> None:
    runner = build_runner(
        model_gateway=DeterministicModelGateway(),
        probe_gateway=tool_signal_gateway,
        ledger_path=tmp_path / "ledger.jsonl",
        config=TerminalBenchConfig(model="deterministic", max_cycles=1),
    )

    result = runner.run_question(_run_input("terminal_conformance"))

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
