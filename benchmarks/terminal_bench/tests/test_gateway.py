from __future__ import annotations

import hashlib
import inspect
import json
from threading import Event, Lock, Thread
from types import SimpleNamespace
import unicodedata
from dataclasses import dataclass
from typing import Any

import pytest

from bayesprobe import EpistemicOrigin, ProbeToolGateway, SignalKind
from bayesprobe_terminal_bench.actions import (
    ActionObservation,
    ApplyPatchAction,
    ShellAction,
    TerminalAction,
    TerminalProbePlan,
    WriteFileAction,
)
from bayesprobe_terminal_bench.config import BudgetExhausted, RunBudget
from bayesprobe_terminal_bench.environment import PolicyViolation
from bayesprobe_terminal_bench.gateway import HarborProbeToolGateway
from bayesprobe_terminal_bench.planning import TerminalPlanError
from bayesprobe_terminal_bench.signals import signal_from_observation


@dataclass
class RecordedArtifacts:
    plans: list[dict[str, Any]]
    observations: list[ActionObservation]
    errors: list[dict[str, Any]]

    def append_plan(self, payload: dict[str, Any]) -> None:
        self.plans.append(payload)

    def append_observation(self, payload: ActionObservation) -> None:
        self.observations.append(payload)

    def append_error(self, payload: dict[str, Any]) -> None:
        self.errors.append(payload)


class ScriptedPlanner:
    def __init__(
        self,
        *,
        plan: TerminalProbePlan | None = None,
        error: BaseException | None = None,
    ) -> None:
        self._plan = plan
        self._error = error
        self.histories: list[tuple[ActionObservation, ...]] = []

    def plan(self, *, probe: object, context: object, history: tuple[ActionObservation, ...]) -> TerminalProbePlan:
        self.histories.append(history)
        if self._error is not None:
            raise self._error
        assert self._plan is not None
        return self._plan


class ScriptedBridge:
    def __init__(self, outcomes: list[ActionObservation | BaseException]) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[tuple[TerminalAction, int]] = []

    def execute(self, action: TerminalAction, action_index: int) -> ActionObservation:
        self.calls.append((action, action_index))
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class CountingBudget:
    def __init__(self, *, outcomes: list[int | BaseException]) -> None:
        self._outcomes = list(outcomes)
        self.calls = 0

    def reserve_action(self) -> int:
        self.calls += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class BlockingPlanner:
    def __init__(self, *, release_first_plan: Event) -> None:
        self._release_first_plan = release_first_plan
        self._lock = Lock()
        self.first_plan_started = Event()
        self.later_plan_started = Event()
        self.histories: list[tuple[ActionObservation, ...]] = []

    def plan(
        self,
        *,
        probe: object,
        context: object,
        history: tuple[ActionObservation, ...],
    ) -> TerminalProbePlan:
        with self._lock:
            invocation_index = len(self.histories)
            self.histories.append(history)
        if invocation_index == 0:
            self.first_plan_started.set()
            assert self._release_first_plan.wait(timeout=2)
        else:
            self.later_plan_started.set()
        return _plan("pwd")


def _plan(*commands: str) -> TerminalProbePlan:
    return TerminalProbePlan(
        mode="inspect",
        actions=tuple(ShellAction(command=command) for command in commands),
        expected_observation="The task workspace state is observed.",
    )


def _intervention_plan(action: TerminalAction) -> TerminalProbePlan:
    return TerminalProbePlan(
        mode="intervene",
        actions=(action,),
        expected_observation="The requested workspace mutation is observed.",
    )


def _observation(
    *,
    action: TerminalAction | None = None,
    action_index: int = 1,
    model_facing_output: str = '{"stdout":"capped result"}',
    full_output_sha256: str = "a" * 64,
    pre_environment_state_id: str = "env:0",
    post_environment_state_id: str = "env:0",
) -> ActionObservation:
    return ActionObservation(
        action_index=action_index,
        action=action or ShellAction(command="pwd"),
        stdout="full stdout that must remain in the environment artifact",
        stderr="full stderr that must remain in the environment artifact",
        return_code=0,
        timed_out=False,
        duration_ms=17,
        pre_environment_state_id=pre_environment_state_id,
        post_environment_state_id=post_environment_state_id,
        full_output_sha256=full_output_sha256,
        model_facing_output=model_facing_output,
        output_truncated=True,
    )


def _canonical_fingerprint(source_identity: str, raw_content: str) -> str:
    canonical_content = " ".join(unicodedata.normalize("NFKC", raw_content).split())
    digest = hashlib.sha256(f"{source_identity}\n{canonical_content}".encode("utf-8"))
    return f"sha256:{digest.hexdigest()}"


def _artifacts() -> RecordedArtifacts:
    return RecordedArtifacts(plans=[], observations=[], errors=[])


def _gateway(
    *,
    planner: ScriptedPlanner,
    bridge: ScriptedBridge,
    artifacts: RecordedArtifacts,
    budget: Any,
) -> HarborProbeToolGateway:
    return HarborProbeToolGateway(
        planner=planner,
        bridge=bridge,
        artifacts=artifacts,
        budget=budget,
    )


def _large_action_text(marker: str) -> str:
    return marker + "x" * (1_000_000 - len(marker))


def test_signal_exposes_exact_executed_shell_request_separately_from_observation(
    probe,
    execution_context,
) -> None:
    observation = _observation(
        action=ShellAction(command="printf tests passed", timeout_seconds=37),
        action_index=4,
        model_facing_output='{"stdout":"only this capped view"}',
        full_output_sha256="b" * 64,
        pre_environment_state_id="env:7",
        post_environment_state_id="env:8",
    )

    signal = signal_from_observation(
        observation=observation,
        probe=probe,
        context=execution_context,
    )
    payload = json.loads(signal.raw_content)

    assert signal.signal_kind is SignalKind.ACTIVE
    assert signal.provenance is not None
    assert signal.provenance.epistemic_origin is EpistemicOrigin.TOOL_RESULT
    assert payload == {
        "action_index": 4,
        "error_category": None,
        "executed_request": {
            "command": "printf tests passed",
            "mutates_environment": True,
            "timeout_seconds": 37,
            "type": "shell",
        },
        "observation": '{"stdout":"only this capped view"}',
        "output_truncated": True,
        "post_environment_state_id": "env:8",
        "pre_environment_state_id": "env:7",
        "return_code": 0,
        "timed_out": False,
    }
    assert "full stdout" not in signal.raw_content
    assert "full stderr" not in signal.raw_content
    assert "printf tests passed" in signal.raw_content
    assert signal.provenance.canonical_content_fingerprint == _canonical_fingerprint(
        signal.provenance.source_identity,
        signal.raw_content,
    )
    assert signal.provenance.derivation_root_id.startswith("harbor-action:sha256:")
    assert signal.provenance.correlation_group.startswith("harbor-env:sha256:")
    assert signal.provenance.environment_state_id == "env:8"
    assert signal.provenance.artifact_refs == ["environment_actions.jsonl#4"]


@pytest.mark.parametrize(
    ("action", "marker", "expected_request"),
    [
        (
            WriteFileAction(
                path="/workspace/model-authored-write.txt",
                content=_large_action_text("WRITE_ACTION_BODY_MUST_NOT_LEAK"),
            ),
            "WRITE_ACTION_BODY_MUST_NOT_LEAK",
            {
                "type": "write_file",
                "path": "/workspace/model-authored-write.txt",
                "content_sha256": hashlib.sha256(
                    _large_action_text("WRITE_ACTION_BODY_MUST_NOT_LEAK").encode("utf-8")
                ).hexdigest(),
                "content_bytes": 1_000_000,
            },
        ),
        (
            ApplyPatchAction(
                patch=_large_action_text("PATCH_ACTION_BODY_MUST_NOT_LEAK"),
            ),
            "PATCH_ACTION_BODY_MUST_NOT_LEAK",
            {
                "type": "apply_patch",
                "strip": 0,
                "patch_sha256": hashlib.sha256(
                    _large_action_text("PATCH_ACTION_BODY_MUST_NOT_LEAK").encode("utf-8")
                ).hexdigest(),
                "patch_bytes": 1_000_000,
            },
        ),
    ],
)
def test_gateway_keeps_large_write_and_patch_inputs_out_of_signal_payloads(
    probe,
    execution_context,
    action: TerminalAction,
    marker: str,
    expected_request: dict[str, object],
) -> None:
    observation = _observation(
        action=action,
        model_facing_output='{"stdout":"mutation completed"}',
        full_output_sha256="c" * 64,
    )
    artifacts = _artifacts()
    gateway = _gateway(
        planner=ScriptedPlanner(plan=_intervention_plan(action)),
        bridge=ScriptedBridge([observation]),
        artifacts=artifacts,
        budget=RunBudget(max_actions=1),
    )

    signals = gateway.execute_probe(probe=probe, context=execution_context)
    payload = json.loads(signals[0].raw_content)

    assert payload["executed_request"] == expected_request
    assert payload["observation"] == '{"stdout":"mutation completed"}'
    assert "action" not in payload
    assert marker not in signals[0].raw_content
    assert len(signals[0].raw_content) < 1_024
    assert artifacts.observations == [observation]
    assert marker in json.dumps(artifacts.observations[0].model_dump(mode="json"))


def test_signal_caps_an_oversized_observation_independently_of_action_artifact(
    probe,
    execution_context,
) -> None:
    observation = _observation(
        action=ShellAction(command="printf tests passed"),
        model_facing_output="z" * 1_000_000,
    )

    signal = signal_from_observation(
        observation=observation,
        probe=probe,
        context=execution_context,
    )
    payload = json.loads(signal.raw_content)

    assert len(payload["observation"].encode("utf-8")) <= 32_768
    assert payload["output_truncated"] is True
    assert len(signal.raw_content.encode("utf-8")) < 66_000


def test_signal_identity_and_roots_are_deterministic_and_distinct_across_lineages(probe, execution_context) -> None:
    first = signal_from_observation(
        observation=_observation(action_index=1),
        probe=probe,
        context=execution_context,
    )
    equivalent = signal_from_observation(
        observation=_observation(action_index=1),
        probe=probe,
        context=execution_context,
    )
    different_action = signal_from_observation(
        observation=_observation(action_index=2),
        probe=probe,
        context=execution_context,
    )
    different_probe = signal_from_observation(
        observation=_observation(action_index=1),
        probe=probe.model_copy(update={"id": "P_cycle_1_verify"}),
        context=execution_context,
    )
    different_cycle = signal_from_observation(
        observation=_observation(action_index=1, post_environment_state_id="env:1"),
        probe=probe.model_copy(update={"cycle_id": "cycle_2"}),
        context=type(execution_context)(
            run_id=execution_context.run_id,
            cycle_id="cycle_2",
            problem=execution_context.problem,
            task_context=execution_context.task_context,
            task_frame={
                "schema_version": "v0.2",
                "task_frame_id": "TF_run_1",
                "admission_decision_id": "TA_run_1",
                "task_kind": "diagnosis",
                "answer_relationship": "open_ended",
                "normalized_question": execution_context.problem,
                "task_context": execution_context.task_context,
                "answer_contract": {
                    "objective": execution_context.problem,
                    "answer_value_type": "structured_text",
                    "answer_format": "plain_text",
                    "required_sections": ["result"],
                    "decision_form": "implementation",
                    "permits_synthesis": True,
                },
                "hypothesis_frame": {
                    "frame_id": "HF_run_1",
                    "competition": "open",
                    "coverage": "open",
                    "rival_sets": {"H_workspace": []},
                    "coverage_statement": "The current hypothesis is incomplete.",
                    "coverage_limitation": "Additional causes may exist.",
                },
                "framing_method": "explicit",
            },
            provider_schema_version=execution_context.provider_schema_version,
            hypotheses=execution_context.hypotheses,
            metadata=dict(execution_context.metadata),
        ),
    )

    assert first.id == equivalent.id
    assert first.provenance == equivalent.provenance
    assert first.id not in {different_action.id, different_probe.id, different_cycle.id}
    assert first.provenance is not None
    assert different_action.provenance is not None
    assert different_probe.provenance is not None
    assert different_cycle.provenance is not None
    assert first.provenance.derivation_root_id not in {
        different_action.provenance.derivation_root_id,
        different_probe.provenance.derivation_root_id,
        different_cycle.provenance.derivation_root_id,
    }
    assert first.provenance.correlation_group == different_action.provenance.correlation_group
    assert first.provenance.correlation_group != different_cycle.provenance.correlation_group


def test_gateway_emits_one_tool_result_signal_per_completed_observation(probe, execution_context) -> None:
    artifacts = _artifacts()
    observations = [_observation(action_index=1), _observation(action_index=2)]
    planner = ScriptedPlanner(plan=_plan("pwd", "ls"))
    gateway = _gateway(
        planner=planner,
        bridge=ScriptedBridge(observations),
        artifacts=artifacts,
        budget=RunBudget(max_actions=2),
    )

    signals = gateway.execute_probe(probe=probe, context=execution_context)

    assert len(signals) == 2
    assert all(
        signal.provenance is not None
        and signal.provenance.epistemic_origin is EpistemicOrigin.TOOL_RESULT
        for signal in signals
    )
    assert [item.action_index for item in artifacts.observations] == [1, 2]
    assert planner.histories == [()]
    assert len(artifacts.plans) == 1
    assert artifacts.errors == []


def test_gateway_records_expected_planner_failure_without_a_signal_or_history(probe, execution_context) -> None:
    artifacts = _artifacts()
    gateway = _gateway(
        planner=ScriptedPlanner(error=TerminalPlanError("planner-secret")),
        bridge=ScriptedBridge([]),
        artifacts=artifacts,
        budget=RunBudget(),
    )

    assert gateway.execute_probe(probe=probe, context=execution_context) == []
    assert artifacts.plans == []
    assert artifacts.observations == []
    assert artifacts.errors == [
        {
            "category": "plan_error",
            "error_type": "TerminalPlanError",
            "probe_id": probe.id,
        }
    ]
    assert "planner-secret" not in json.dumps(artifacts.errors)


def test_gateway_returns_no_signal_when_planning_budget_is_exhausted(probe, execution_context) -> None:
    artifacts = _artifacts()
    gateway = _gateway(
        planner=ScriptedPlanner(error=BudgetExhausted("budget-secret")),
        bridge=ScriptedBridge([]),
        artifacts=artifacts,
        budget=RunBudget(),
    )

    assert gateway.execute_probe(probe=probe, context=execution_context) == []
    assert artifacts.plans == []
    assert artifacts.observations == []
    assert artifacts.errors == [{"category": "budget_exhausted", "probe_id": probe.id}]
    assert "budget-secret" not in json.dumps(artifacts.errors)


def test_gateway_continues_after_policy_rejection_without_fabricating_an_observation(probe, execution_context) -> None:
    artifacts = _artifacts()
    bridge = ScriptedBridge(
        [
            PolicyViolation("policy-secret"),
            _observation(action=ShellAction(command="pwd"), action_index=2),
        ]
    )
    budget = CountingBudget(outcomes=[1, 2])
    gateway = _gateway(
        planner=ScriptedPlanner(plan=_plan("ls", "pwd")),
        bridge=bridge,
        artifacts=artifacts,
        budget=budget,
    )

    signals = gateway.execute_probe(probe=probe, context=execution_context)

    assert len(signals) == 1
    assert [call[1] for call in bridge.calls] == [1, 2]
    assert budget.calls == 2
    assert [item.action_index for item in artifacts.observations] == [2]
    assert artifacts.errors == [
        {
            "action_index": 1,
            "category": "policy_error",
            "error_type": "PolicyViolation",
            "probe_id": probe.id,
        }
    ]
    assert "policy-secret" not in json.dumps(artifacts.errors)


def test_gateway_stops_after_action_budget_exhaustion_without_executing_the_action(probe, execution_context) -> None:
    artifacts = _artifacts()
    bridge = ScriptedBridge([_observation(action_index=1)])
    budget = CountingBudget(outcomes=[BudgetExhausted("budget-secret")])
    gateway = _gateway(
        planner=ScriptedPlanner(plan=_plan("pwd")),
        bridge=bridge,
        artifacts=artifacts,
        budget=budget,
    )

    assert gateway.execute_probe(probe=probe, context=execution_context) == []
    assert budget.calls == 1
    assert bridge.calls == []
    assert artifacts.observations == []
    assert artifacts.errors == [{"category": "budget_exhausted", "probe_id": probe.id}]
    assert "budget-secret" not in json.dumps(artifacts.errors)


def test_gateway_preserves_only_actual_observations_in_later_planner_history(probe, execution_context) -> None:
    artifacts = _artifacts()
    bridge = ScriptedBridge(
        [
            PolicyViolation("rejected"),
            _observation(action_index=2),
            _observation(action_index=3),
            _observation(action_index=4),
        ]
    )
    planner = ScriptedPlanner(plan=_plan("ls", "pwd"))
    gateway = _gateway(
        planner=planner,
        bridge=bridge,
        artifacts=artifacts,
        budget=CountingBudget(outcomes=[1, 2, 3, 4]),
    )

    gateway.execute_probe(probe=probe, context=execution_context)
    gateway.execute_probe(probe=probe, context=execution_context)

    assert planner.histories == [(), (artifacts.observations[0],)]
    assert [item.action_index for item in artifacts.observations] == [2, 3, 4]


def test_gateway_scopes_history_to_each_run_id(probe, execution_context) -> None:
    artifacts = _artifacts()
    planner = ScriptedPlanner(plan=_plan("pwd"))
    first_run_observation = _observation(action_index=1)
    second_run_observation = _observation(action_index=2)
    gateway = _gateway(
        planner=planner,
        bridge=ScriptedBridge(
            [
                first_run_observation,
                second_run_observation,
                _observation(action_index=3),
            ]
        ),
        artifacts=artifacts,
        budget=CountingBudget(outcomes=[1, 2, 3]),
    )
    other_run_context = SimpleNamespace(
        run_id="run_2",
        cycle_id=execution_context.cycle_id,
    )

    gateway.execute_probe(probe=probe, context=execution_context)
    gateway.execute_probe(probe=probe, context=other_run_context)
    gateway.execute_probe(probe=probe, context=execution_context)

    assert planner.histories == [(), (), (first_run_observation,)]


@pytest.mark.parametrize("same_run", [True, False])
def test_gateway_serializes_concurrent_probe_execution(
    probe,
    execution_context,
    same_run: bool,
) -> None:
    release_first_plan = Event()
    planner = BlockingPlanner(release_first_plan=release_first_plan)
    artifacts = _artifacts()
    first_observation = _observation(action_index=1)
    gateway = _gateway(
        planner=planner,
        bridge=ScriptedBridge([first_observation, _observation(action_index=2)]),
        artifacts=artifacts,
        budget=RunBudget(max_actions=2),
    )
    second_context = (
        execution_context
        if same_run
        else SimpleNamespace(
            run_id="run_2",
            cycle_id=execution_context.cycle_id,
        )
    )
    errors: list[BaseException] = []
    second_attempted = Event()

    def execute(context: object, *, is_second: bool) -> None:
        if is_second:
            second_attempted.set()
        try:
            gateway.execute_probe(probe=probe, context=context)
        except BaseException as error:
            errors.append(error)

    first_thread = Thread(target=execute, args=(execution_context,), kwargs={"is_second": False})
    second_thread = Thread(target=execute, args=(second_context,), kwargs={"is_second": True})
    first_thread.start()
    assert planner.first_plan_started.wait(timeout=1)
    second_thread.start()
    assert second_attempted.wait(timeout=1)
    try:
        assert not planner.later_plan_started.wait(timeout=0.2)
    finally:
        release_first_plan.set()
        first_thread.join(timeout=2)
        second_thread.join(timeout=2)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert errors == []
    if same_run:
        assert planner.histories == [(), (first_observation,)]
    else:
        assert planner.histories == [(), ()]


def test_gateway_propagates_unexpected_programmer_errors(probe, execution_context) -> None:
    artifacts = _artifacts()
    gateway = _gateway(
        planner=ScriptedPlanner(error=RuntimeError("programmer-secret")),
        bridge=ScriptedBridge([]),
        artifacts=artifacts,
        budget=RunBudget(),
    )

    with pytest.raises(RuntimeError, match="programmer-secret"):
        gateway.execute_probe(probe=probe, context=execution_context)

    assert artifacts.plans == []
    assert artifacts.observations == []
    assert artifacts.errors == []


def test_gateway_method_matches_the_public_probe_tool_gateway_shape() -> None:
    parameters = tuple(inspect.signature(HarborProbeToolGateway.execute_probe).parameters)

    assert parameters == ("self", "probe", "context")
    public_gateway_type: type[ProbeToolGateway] = HarborProbeToolGateway
    assert public_gateway_type is HarborProbeToolGateway
