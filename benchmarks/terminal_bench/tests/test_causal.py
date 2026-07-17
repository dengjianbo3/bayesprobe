from __future__ import annotations

import hashlib
import json

import pytest

from bayesprobe_terminal_bench.actions import (
    ActionObservation,
    ShellAction,
    TerminalPlanStep,
    TerminalProbePlan,
    TransitionPrediction,
    WriteFileAction,
)
from bayesprobe_terminal_bench.causal import CausalTraceRegistry


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _observation(
    *,
    action: ShellAction | WriteFileAction,
    action_index: int,
    before: str,
    after: str,
) -> ActionObservation:
    return ActionObservation(
        action_index=action_index,
        action=action,
        stdout="done",
        return_code=0,
        duration_ms=1,
        pre_environment_state_id=before,
        post_environment_state_id=after,
        full_output_sha256="a" * 64,
        model_facing_output='{"stdout":"done"}',
    )


def _inspect_plan(action: ShellAction | None = None) -> TerminalProbePlan:
    return TerminalProbePlan(
        mode="inspect",
        steps=(
            TerminalPlanStep(role="inspect", action=action or ShellAction(command="pwd")),
        ),
        expected_observation="The workspace is observed.",
    )


def _intervention_plan() -> TerminalProbePlan:
    return TerminalProbePlan(
        mode="intervene",
        steps=(
            TerminalPlanStep(role="inspect", action=ShellAction(command="pwd")),
            TerminalPlanStep(
                role="intervene",
                action=WriteFileAction(path="/workspace/result.txt", content="done"),
            ),
            TerminalPlanStep(
                role="verify",
                action=ShellAction(command="cat /workspace/result.txt"),
                verification_target="the result file contains done",
            ),
        ),
        expected_observation="The write is acknowledged and then verified.",
        transition_predictions=(
            TransitionPrediction(
                hypothesis_id="H_workspace",
                expected_transition="The workspace defect is repaired.",
            ),
        ),
    )


def test_registry_uses_exact_canonical_plan_policy_request_and_action_identities(
    probe,
    execution_context,
) -> None:
    registry = CausalTraceRegistry()
    plan = _inspect_plan()

    registered = registry.register_plan(
        probe=probe,
        context=execution_context,
        plan=plan,
    )
    observation = _observation(
        action=plan.steps[0].action,
        action_index=4,
        before="env:7",
        after="env:7",
    )
    record = registry.register_action(
        plan=registered,
        step_index=0,
        observation=observation,
    )

    probe_payload = probe.model_dump(mode="json")
    plan_payload = plan.model_dump(mode="json")
    expected_plan_id = "PL_" + _canonical_digest(
        {
            "cycle_id": execution_context.cycle_id,
            "plan": plan_payload,
            "probe": probe_payload,
            "run_id": execution_context.run_id,
        }
    )
    expected_policy_attempt_id = "PA_" + _canonical_digest(
        {
            "cycle_id": execution_context.cycle_id,
            "intervention_plan": plan_payload,
            "probe": probe_payload,
            "run_id": execution_context.run_id,
        }
    )
    expected_request_fingerprint = "sha256:" + _canonical_digest(
        {
            "command": "pwd",
            "mutates_environment": False,
            "timeout_seconds": 120,
            "type": "shell",
        }
    )
    expected_action_id = "A_" + _canonical_digest(
        {
            "action_index": 4,
            "plan_id": expected_plan_id,
            "request_fingerprint": expected_request_fingerprint,
            "step_index": 0,
        }
    )

    assert registered.plan_id == expected_plan_id
    assert registered.policy_attempt_id == expected_policy_attempt_id
    assert record.request_fingerprint == expected_request_fingerprint
    assert record.action_id == expected_action_id


def test_registry_rejects_duplicate_plan_and_action_ids(probe, execution_context) -> None:
    registry = CausalTraceRegistry()
    plan = _inspect_plan()
    registered = registry.register_plan(probe=probe, context=execution_context, plan=plan)
    observation = _observation(
        action=plan.steps[0].action,
        action_index=1,
        before="env:0",
        after="env:0",
    )
    registry.register_action(plan=registered, step_index=0, observation=observation)

    with pytest.raises(ValueError, match="duplicate plan_id"):
        registry.register_plan(probe=probe, context=execution_context, plan=plan)
    with pytest.raises(ValueError, match="duplicate action_id"):
        registry.register_action(plan=registered, step_index=0, observation=observation)


def test_registry_rejects_an_observation_for_a_different_request(
    probe,
    execution_context,
) -> None:
    registry = CausalTraceRegistry()
    plan = _inspect_plan()
    registered = registry.register_plan(probe=probe, context=execution_context, plan=plan)

    with pytest.raises(ValueError, match="executed request does not match"):
        registry.register_action(
            plan=registered,
            step_index=0,
            observation=_observation(
                action=ShellAction(command="ls"),
                action_index=1,
                before="env:0",
                after="env:0",
            ),
        )


def test_registry_tracks_linear_environment_subjects_and_intervention_generation(
    probe,
    execution_context,
) -> None:
    registry = CausalTraceRegistry()
    plan = _intervention_plan()
    registered = registry.register_plan(probe=probe, context=execution_context, plan=plan)
    states = (("env:0", "env:0"), ("env:0", "env:1"), ("env:1", "env:1"))

    records = [
        registry.register_action(
            plan=registered,
            step_index=index,
            observation=_observation(
                action=step.action,
                action_index=index + 1,
                before=states[index][0],
                after=states[index][1],
            ),
        )
        for index, step in enumerate(plan.steps)
    ]

    assert [record.action_role for record in records] == ["inspect", "intervene", "verify"]
    assert [record.subject_environment_state_id for record in records] == [
        "env:0",
        "env:1",
        "env:1",
    ]
    assert [record.intervention_generation for record in records] == [0, 1, 1]
    assert records[1].verification_target is None
    assert records[2].verification_target == "the result file contains done"
    assert records[2].transition_predictions == {
        "H_workspace": "The workspace defect is repaired."
    }


@pytest.mark.parametrize(
    ("before", "after", "message"),
    [
        ("", "env:0", "missing environment state"),
        ("env:0", "", "missing environment state"),
    ],
)
def test_registry_rejects_missing_environment_states(
    probe,
    execution_context,
    before: str,
    after: str,
    message: str,
) -> None:
    registry = CausalTraceRegistry()
    plan = _inspect_plan()
    registered = registry.register_plan(probe=probe, context=execution_context, plan=plan)

    with pytest.raises(ValueError, match=message):
        registry.register_action(
            plan=registered,
            step_index=0,
            observation=_observation(
                action=plan.steps[0].action,
                action_index=1,
                before=before,
                after=after,
            ),
        )


def test_registry_rejects_non_linear_state_and_a_second_mutation(
    probe,
    execution_context,
) -> None:
    plan = TerminalProbePlan(
        mode="verify",
        steps=(
            TerminalPlanStep(
                role="verify",
                action=ShellAction(command="pytest -q"),
                verification_target="the first check passes",
            ),
            TerminalPlanStep(
                role="verify",
                action=ShellAction(command="touch /tmp/again"),
                verification_target="the second check passes",
            ),
        ),
        expected_observation="Both checks complete.",
    )
    registry = CausalTraceRegistry()
    registered = registry.register_plan(probe=probe, context=execution_context, plan=plan)
    registry.register_action(
        plan=registered,
        step_index=0,
        observation=_observation(
            action=plan.steps[0].action,
            action_index=1,
            before="env:0",
            after="env:1",
        ),
    )

    with pytest.raises(ValueError, match="non-linear environment state"):
        registry.register_action(
            plan=registered,
            step_index=1,
            observation=_observation(
                action=plan.steps[1].action,
                action_index=2,
                before="env:9",
                after="env:10",
            ),
        )
    with pytest.raises(ValueError, match="second mutation"):
        registry.register_action(
            plan=registered,
            step_index=1,
            observation=_observation(
                action=plan.steps[1].action,
                action_index=2,
                before="env:1",
                after="env:2",
            ),
        )


def test_registry_rejects_non_linear_state_across_plans_in_the_same_run(
    probe,
    execution_context,
) -> None:
    registry = CausalTraceRegistry()
    first_plan = _inspect_plan()
    first_registered = registry.register_plan(
        probe=probe,
        context=execution_context,
        plan=first_plan,
    )
    registry.register_action(
        plan=first_registered,
        step_index=0,
        observation=_observation(
            action=first_plan.steps[0].action,
            action_index=1,
            before="env:0",
            after="env:0",
        ),
    )
    second_plan = _inspect_plan(ShellAction(command="ls"))
    second_registered = registry.register_plan(
        probe=probe.model_copy(update={"id": "P_cycle_1_followup"}),
        context=execution_context,
        plan=second_plan,
    )

    with pytest.raises(ValueError, match="non-linear environment state"):
        registry.register_action(
            plan=second_registered,
            step_index=0,
            observation=_observation(
                action=second_plan.steps[0].action,
                action_index=2,
                before="env:9",
                after="env:9",
            ),
        )


def test_registry_preserves_intervention_generation_across_plans(
    probe,
    execution_context,
) -> None:
    registry = CausalTraceRegistry()
    intervention_plan = _intervention_plan()
    intervention_registered = registry.register_plan(
        probe=probe,
        context=execution_context,
        plan=intervention_plan,
    )
    states = (("env:0", "env:0"), ("env:0", "env:1"), ("env:1", "env:1"))
    for index, step in enumerate(intervention_plan.steps):
        registry.register_action(
            plan=intervention_registered,
            step_index=index,
            observation=_observation(
                action=step.action,
                action_index=index + 1,
                before=states[index][0],
                after=states[index][1],
            ),
        )

    followup_plan = _inspect_plan()
    followup_registered = registry.register_plan(
        probe=probe.model_copy(update={"id": "P_cycle_1_followup"}),
        context=execution_context,
        plan=followup_plan,
    )
    followup_record = registry.register_action(
        plan=followup_registered,
        step_index=0,
        observation=_observation(
            action=followup_plan.steps[0].action,
            action_index=4,
            before="env:1",
            after="env:1",
        ),
    )

    assert followup_record.intervention_generation == 1


def test_registry_binds_each_action_and_signal_exactly_once(probe, execution_context) -> None:
    registry = CausalTraceRegistry()
    first_plan = _inspect_plan()
    first_registered = registry.register_plan(
        probe=probe,
        context=execution_context,
        plan=first_plan,
    )
    first_record = registry.register_action(
        plan=first_registered,
        step_index=0,
        observation=_observation(
            action=first_plan.steps[0].action,
            action_index=1,
            before="env:0",
            after="env:0",
        ),
    )
    second_probe = probe.model_copy(update={"id": "P_cycle_1_second"})
    second_plan = _inspect_plan(ShellAction(command="ls"))
    second_registered = registry.register_plan(
        probe=second_probe,
        context=execution_context,
        plan=second_plan,
    )
    second_record = registry.register_action(
        plan=second_registered,
        step_index=0,
        observation=_observation(
            action=second_plan.steps[0].action,
            action_index=2,
            before="env:0",
            after="env:0",
        ),
    )

    registry.bind_signal(action_id=first_record.action_id, signal_id="S_one")

    assert registry.record_for_signal("S_one") == first_record
    with pytest.raises(ValueError, match="already has a Signal"):
        registry.bind_signal(action_id=first_record.action_id, signal_id="S_two")
    with pytest.raises(ValueError, match="Signal ID is already bound"):
        registry.bind_signal(action_id=second_record.action_id, signal_id="S_one")
    with pytest.raises(KeyError, match="unknown Signal"):
        registry.record_for_signal("S_missing")
