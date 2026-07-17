from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from bayesprobe import ModelGatewayValidationError, StructuredModelRequest

from bayesprobe_terminal_bench.actions import (
    ActionObservation,
    ShellAction,
    TerminalPlanStep,
    TerminalProbePlan,
    TransitionPrediction,
    WriteFileAction,
)
from bayesprobe_terminal_bench.artifacts import TrialArtifactStore
import bayesprobe_terminal_bench.causal as causal_module
from bayesprobe_terminal_bench.causal import CausalTraceRegistry
from bayesprobe_terminal_bench.signals import signal_from_observation


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


def test_registry_allows_verify_state_changes_without_counting_interventions(
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
    first = registry.register_action(
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
    second = registry.register_action(
        plan=registered,
        step_index=1,
        observation=_observation(
            action=plan.steps[1].action,
            action_index=2,
            before="env:1",
            after="env:2",
        ),
    )

    assert [first.action_role, second.action_role] == ["verify", "verify"]
    assert [first.intervention_generation, second.intervention_generation] == [0, 0]


def test_registry_rejects_a_plan_with_a_second_declared_intervention(
    probe,
    execution_context,
) -> None:
    steps = (
        TerminalPlanStep(
            role="intervene",
            action=WriteFileAction(path="/workspace/one", content="one"),
        ),
        TerminalPlanStep(
            role="intervene",
            action=WriteFileAction(path="/workspace/two", content="two"),
        ),
    )
    plan = TerminalProbePlan.model_construct(
        mode="intervene",
        steps=steps,
        expected_observation="The malformed plan is rejected by the registry.",
        transition_predictions=(),
    )
    registry = CausalTraceRegistry()
    with pytest.raises(ValueError, match="optional inspect, one intervene, then verify"):
        registry.register_plan(probe=probe, context=execution_context, plan=plan)


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

    first_signal = signal_from_observation(
        registry=registry,
        action_id=first_record.action_id,
        probe=probe,
        context=execution_context,
        redact_model_content=lambda value: value,
    )

    assert registry.record_for_signal(first_signal.id) == first_record
    with pytest.raises(ValueError, match="already has a Signal"):
        signal_from_observation(
            registry=registry,
            action_id=first_record.action_id,
            probe=probe,
            context=execution_context,
            redact_model_content=lambda value: value,
        )
    with pytest.raises(ValueError, match="Signal ID is already bound"):
        registry.bind_signal(
            action_id=second_record.action_id,
            signal_builder=lambda _: first_signal,
        )
    with pytest.raises(KeyError, match="unknown Signal"):
        registry.record_for_signal("S_missing")


def test_causal_decision_contract_is_frozen_strict_and_forbids_extra_fields() -> None:
    decision = causal_module.CausalDecision(
        signal_id="S_bound",
        action_id="A_bound",
        action_role="inspect",
        decision="admit",
        reason_code="state_scoped_inspection",
        subject_environment_state_id="env:0",
        judgment_response_sha256="a" * 64,
    )

    assert decision.model_dump(mode="json") == {
        "signal_id": "S_bound",
        "action_id": "A_bound",
        "action_role": "inspect",
        "decision": "admit",
        "reason_code": "state_scoped_inspection",
        "subject_environment_state_id": "env:0",
        "judgment_response_sha256": "a" * 64,
    }
    with pytest.raises(ValidationError, match="frozen"):
        decision.signal_id = "S_changed"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        causal_module.CausalDecision(
            **decision.model_dump(mode="python"),
            raw_judgment={"secret": "must not persist"},
        )


class _RecordingDelegate:
    adapter_kind = "recording-causal-delegate"
    model_identity = "recording-causal-model"

    def __init__(
        self,
        response: dict[str, Any],
        *,
        events: list[str] | None = None,
    ) -> None:
        self.response = response
        self.requests: list[StructuredModelRequest] = []
        self.config = object()
        self.invocation_observer = object()
        self._events = events

    def complete_structured(self, request: StructuredModelRequest) -> dict[str, Any]:
        self.requests.append(request)
        if self._events is not None:
            self._events.append("delegate")
        return self.response


class _RecordingDecisionSink:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.decisions: list[dict[str, Any]] = []

    def append_causal_decision(self, payload: Any) -> None:
        self.events.append("decision")
        if hasattr(payload, "model_dump"):
            payload = payload.model_dump(mode="json")
        self.decisions.append(dict(payload))


def _probe_with_targets(probe, *, probe_id: str, targets: list[str]):
    return probe.model_copy(
        deep=True,
        update={
            "id": probe_id,
            "target_hypotheses": list(targets),
            "support_condition": {
                target: f"The observation supports {target}." for target in targets
            },
            "weaken_condition": {
                target: f"The observation weakens {target}." for target in targets
            },
        },
    )


def _register_plan_signals(
    *,
    registry: CausalTraceRegistry,
    probe,
    execution_context,
    plan: TerminalProbePlan,
    states: tuple[tuple[str, str], ...],
    first_action_index: int = 1,
):
    registered = registry.register_plan(
        probe=probe,
        context=execution_context,
        plan=plan,
    )
    records = []
    signals = []
    for offset, (step, (before, after)) in enumerate(zip(plan.steps, states, strict=True)):
        record = registry.register_action(
            plan=registered,
            step_index=offset,
            observation=_observation(
                action=step.action,
                action_index=first_action_index + offset,
                before=before,
                after=after,
            ),
        )
        records.append(record)
        signals.append(
            signal_from_observation(
                registry=registry,
                action_id=record.action_id,
                probe=probe,
                context=execution_context,
                redact_model_content=lambda value: value,
            )
        )
    return registered, records, signals


def _native_evidence_request(
    *,
    signal,
    record,
    registered,
    hypothesis_types: dict[str, str],
) -> StructuredModelRequest:
    targets = list(hypothesis_types)
    provenance = signal.provenance
    assert provenance is not None
    return StructuredModelRequest(
        task="judge_evidence",
        input={
            "task_context": {
                "problem": "Repair the task workspace.",
                "task_context": "Use the provided workspace only.",
            },
            "hypotheses": [
                {
                    "id": hypothesis_id,
                    "statement": f"Statement for {hypothesis_id}.",
                    "type": hypothesis_type,
                    "scope": "The registered environment state.",
                    "predictions": [f"Prediction for {hypothesis_id}."],
                    "falsifiers": [f"Falsifier for {hypothesis_id}."],
                    "rivals": [],
                }
                for hypothesis_id, hypothesis_type in hypothesis_types.items()
            ],
            "signal": {
                "id": signal.id,
                "cycle_id": signal.cycle_id,
                "signal_kind": signal.signal_kind.value,
                "source_type": signal.source_type,
                "source": signal.source,
                "raw_content": signal.raw_content,
                "generated_by_probe": signal.generated_by_probe,
                "inbox_status": signal.inbox_status.value,
                "initial_target_hypotheses": list(
                    signal.initial_target_hypotheses
                ),
            },
            "provenance": provenance.model_dump(mode="json"),
            "matched_probe": {
                "id": record.probe_id,
                "purpose": "hypothesis_discrimination",
                "target_hypotheses": targets,
                "inquiry_goal": "Discriminate the registered targets.",
                "method": "terminal",
                "expected_observation": "A causally attributable observation.",
            },
            "target_hypotheses": targets,
        },
        prompt_id="evidence_judgment",
        prompt_version="v0.2",
        schema_name="EvidenceJudgment",
        schema_version="v0.2",
        metadata={
            "judgment_route": "native_v0.2",
            "lifecycle_schema_version": "v0.2",
            "frame_competition": "independent",
            "frame_coverage": "open",
            "run_id": record.run_id,
            "cycle_id": record.cycle_id,
            "signal_id": signal.id,
            "belief_context_policy": "blind_no_scores_v1",
            "plan_id": registered.plan_id,
        },
    )


def _judgment(
    targets: list[str],
    *,
    evidence_type: str = "supporting",
    likelihoods: dict[str, str] | None = None,
    frame_fit: str = "explained_by_named",
    interpretation: str = "delegate-private-semantic-judgment",
) -> dict[str, Any]:
    return {
        "evidence_type": evidence_type,
        "likelihoods": likelihoods
        if likelihoods is not None
        else {target: "moderately_confirming" for target in targets},
        "unresolved_likelihood": None,
        "frame_fit": frame_fit,
        "unexplained_observation": None,
        "interpretation": interpretation,
        "quality_overrides": {},
    }


def _intervene_plan(
    *,
    predictions: tuple[TransitionPrediction, ...] = (),
    include_inspect: bool = False,
) -> TerminalProbePlan:
    steps = []
    if include_inspect:
        steps.append(
            TerminalPlanStep(role="inspect", action=ShellAction(command="pwd"))
        )
    steps.extend(
        [
            TerminalPlanStep(
                role="intervene",
                action=WriteFileAction(path="/workspace/result.txt", content="done"),
            ),
            TerminalPlanStep(
                role="verify",
                action=ShellAction(command="cat /workspace/result.txt"),
                verification_target="the result file contains done",
            ),
        ]
    )
    return TerminalProbePlan(
        mode="intervene",
        steps=tuple(steps),
        expected_observation="The mutation is acknowledged and verified.",
        transition_predictions=predictions,
    )


def _current_case(
    *,
    probe,
    execution_context,
    role: str,
    hypothesis_types: dict[str, str],
    predictions: tuple[TransitionPrediction, ...] = (),
):
    registry = CausalTraceRegistry()
    case_probe = _probe_with_targets(
        probe,
        probe_id=f"P_{role}_{'_'.join(hypothesis_types)}",
        targets=list(hypothesis_types),
    )
    if role == "inspect":
        plan = _inspect_plan()
        selected_index = 0
        states = (("env:0", "env:0"),)
    elif role in {"intervene", "verify"}:
        plan = _intervene_plan(predictions=predictions)
        selected_index = 0 if role == "intervene" else 1
        states = (("env:0", "env:1"), ("env:1", "env:1"))
    else:
        raise AssertionError(f"unsupported role fixture: {role}")
    registered, records, signals = _register_plan_signals(
        registry=registry,
        probe=case_probe,
        execution_context=execution_context,
        plan=plan,
        states=states,
    )
    request = _native_evidence_request(
        signal=signals[selected_index],
        record=records[selected_index],
        registered=registered,
        hypothesis_types=hypothesis_types,
    )
    return registry, request, records[selected_index]


def _stale_inspection_case(
    *,
    probe,
    execution_context,
    hypothesis_types: dict[str, str],
):
    registry = CausalTraceRegistry()
    targets = list(hypothesis_types)
    old_probe = _probe_with_targets(
        probe,
        probe_id="P_old_inspection",
        targets=targets,
    )
    old_registered, old_records, old_signals = _register_plan_signals(
        registry=registry,
        probe=old_probe,
        execution_context=execution_context,
        plan=_inspect_plan(),
        states=(("env:0", "env:0"),),
    )
    new_probe = _probe_with_targets(
        probe,
        probe_id="P_new_intervention",
        targets=targets,
    )
    _register_plan_signals(
        registry=registry,
        probe=new_probe,
        execution_context=execution_context,
        plan=_intervene_plan(),
        states=(("env:0", "env:1"), ("env:1", "env:1")),
        first_action_index=2,
    )
    request = _native_evidence_request(
        signal=old_signals[0],
        record=old_records[0],
        registered=old_registered,
        hypothesis_types=hypothesis_types,
    )
    return registry, request, old_records[0]


def _replace_request_input(
    request: StructuredModelRequest,
    request_input: dict[str, Any],
) -> StructuredModelRequest:
    return StructuredModelRequest(
        task=request.task,
        input=request_input,
        prompt_id=request.prompt_id,
        prompt_version=request.prompt_version,
        schema_name=request.schema_name,
        schema_version=request.schema_version,
        metadata=dict(request.metadata),
    )


@pytest.mark.parametrize(
    ("case", "expected_decision", "expected_reason"),
    [
        ("inspection", "admit", "state_scoped_inspection"),
        ("neutral_ack", "admit", "neutral_mutation_acknowledgement"),
        ("postcondition", "admit", "verified_postcondition"),
        ("causal_transition", "admit", "preregistered_causal_transition"),
        ("unbound_over_all", "discard", "unbound_signal"),
        ("target_over_stale_policy", "discard", "target_mismatch"),
        ("stale_over_policy", "discard", "stale_state"),
        ("policy_over_nonneutral", "discard", "unexecuted_policy_comparison"),
        (
            "nonneutral_mutation",
            "discard",
            "nonneutral_mutation_acknowledgement",
        ),
        ("missing_predictions", "discard", "missing_transition_predictions"),
    ],
)
def test_causal_gateway_table_tests_every_reason_code_and_precedence(
    tmp_path: Path,
    probe,
    execution_context,
    case: str,
    expected_decision: str,
    expected_reason: str,
) -> None:
    response: dict[str, Any]
    if case == "inspection":
        registry, request, record = _current_case(
            probe=probe,
            execution_context=execution_context,
            role="inspect",
            hypothesis_types={"H_root": "root_cause"},
        )
        response = _judgment(["H_root"])
    elif case == "neutral_ack":
        registry, request, record = _current_case(
            probe=probe,
            execution_context=execution_context,
            role="intervene",
            hypothesis_types={"H_post": "postcondition"},
        )
        response = _judgment(
            ["H_post"],
            evidence_type="neutral",
            likelihoods={"H_post": "neutral"},
            frame_fit="underdetermined",
        )
    elif case == "postcondition":
        registry, request, record = _current_case(
            probe=probe,
            execution_context=execution_context,
            role="verify",
            hypothesis_types={"H_post": "postcondition"},
        )
        response = _judgment(["H_post"])
    elif case == "causal_transition":
        predictions = (
            TransitionPrediction(
                hypothesis_id="H_root_a",
                expected_transition="The verifier passes after repairing cause A.",
            ),
            TransitionPrediction(
                hypothesis_id="H_root_b",
                expected_transition="The verifier still fails under cause B.",
            ),
        )
        registry, request, record = _current_case(
            probe=probe,
            execution_context=execution_context,
            role="verify",
            hypothesis_types={
                "H_root_a": "root_cause",
                "H_root_b": "causal_effect",
            },
            predictions=predictions,
        )
        response = _judgment(
            ["H_root_a", "H_root_b"],
            likelihoods={
                "H_root_a": "strongly_confirming",
                "H_root_b": "strongly_disconfirming",
            },
        )
    elif case in {"unbound_over_all", "target_over_stale_policy", "stale_over_policy"}:
        registry, request, record = _stale_inspection_case(
            probe=probe,
            execution_context=execution_context,
            hypothesis_types={"H_policy": "implementation_policy"},
        )
        response = _judgment(["H_policy"])
        if case == "unbound_over_all":
            request_input = copy.deepcopy(request.input)
            raw = json.loads(request_input["signal"]["raw_content"])
            raw["causal_binding"]["request_fingerprint"] = "sha256:" + "f" * 64
            request_input["signal"]["raw_content"] = json.dumps(raw, sort_keys=True)
            request = _replace_request_input(request, request_input)
            response = _judgment(
                ["H_policy"],
                likelihoods={"H_wrong": "strongly_confirming"},
            )
        elif case == "target_over_stale_policy":
            response = _judgment(
                ["H_policy"],
                likelihoods={"H_wrong": "strongly_confirming"},
            )
    elif case == "policy_over_nonneutral":
        registry, request, record = _current_case(
            probe=probe,
            execution_context=execution_context,
            role="intervene",
            hypothesis_types={"H_policy": "patch_choice"},
        )
        response = _judgment(["H_policy"])
    elif case == "nonneutral_mutation":
        registry, request, record = _current_case(
            probe=probe,
            execution_context=execution_context,
            role="intervene",
            hypothesis_types={"H_post": "postcondition"},
        )
        response = _judgment(["H_post"])
    elif case == "missing_predictions":
        registry, request, record = _current_case(
            probe=probe,
            execution_context=execution_context,
            role="verify",
            hypothesis_types={"H_root": "root_cause"},
        )
        response = _judgment(["H_root"])
    else:
        raise AssertionError(f"unknown case: {case}")

    delegate = _RecordingDelegate(response)
    artifacts = TrialArtifactStore(
        tmp_path / case,
        restricted_values=("delegate-private-semantic-judgment",),
    )
    gateway = causal_module.CausalEvidenceModelGateway(
        delegate=delegate,
        registry=registry,
        artifacts=artifacts,
    )

    if expected_decision == "discard":
        with pytest.raises(
            ModelGatewayValidationError,
            match=rf"^causal_admissibility:{expected_reason}$",
        ):
            gateway.complete_structured(request)
    else:
        assert gateway.complete_structured(request) is response

    decisions = [
        json.loads(line)
        for line in (artifacts.root / "causal_decisions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(decisions) == 1
    assert decisions[0] == {
        "action_id": record.action_id,
        "action_role": record.action_role,
        "decision": expected_decision,
        "judgment_response_sha256": _canonical_digest(response),
        "reason_code": expected_reason,
        "signal_id": request.input["signal"]["id"],
        "subject_environment_state_id": record.subject_environment_state_id,
    }
    assert "delegate-private-semantic-judgment" not in (
        artifacts.root / "causal_decisions.jsonl"
    ).read_text(encoding="utf-8")
    assert delegate.requests == [request]


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        ("action_id", "unbound_signal"),
        ("plan_id", "unbound_signal"),
        ("executed_request", "unbound_signal"),
        ("signal_targets", "target_mismatch"),
        ("probe_targets", "target_mismatch"),
        ("subject_state", "stale_state"),
        ("provenance_state", "stale_state"),
        ("policy_attempt", "stale_state"),
    ],
)
def test_causal_gateway_validates_exact_request_bound_lineage(
    tmp_path: Path,
    probe,
    execution_context,
    mutation: str,
    expected_reason: str,
) -> None:
    registry, request, _ = _current_case(
        probe=probe,
        execution_context=execution_context,
        role="inspect",
        hypothesis_types={"H_root": "root_cause"},
    )
    request_input = copy.deepcopy(request.input)
    raw = json.loads(request_input["signal"]["raw_content"])
    if mutation == "action_id":
        raw["causal_binding"]["action_id"] = "A_contradiction"
    elif mutation == "plan_id":
        raw["causal_binding"]["plan_id"] = "PL_contradiction"
    elif mutation == "executed_request":
        raw["executed_request"]["command"] = "ls"
    elif mutation == "signal_targets":
        request_input["signal"]["initial_target_hypotheses"] = ["H_other"]
    elif mutation == "probe_targets":
        request_input["matched_probe"]["target_hypotheses"] = ["H_other"]
    elif mutation == "subject_state":
        raw["causal_binding"]["subject_environment_state_id"] = "env:other"
    elif mutation == "provenance_state":
        request_input["provenance"]["environment_state_id"] = "env:other"
    elif mutation == "policy_attempt":
        raw["causal_binding"]["policy_attempt_id"] = "PA_other"
    else:
        raise AssertionError(mutation)
    request_input["signal"]["raw_content"] = json.dumps(raw, sort_keys=True)
    mutated_request = _replace_request_input(request, request_input)
    gateway = causal_module.CausalEvidenceModelGateway(
        delegate=_RecordingDelegate(_judgment(["H_root"])),
        registry=registry,
        artifacts=TrialArtifactStore(tmp_path / mutation, restricted_values=()),
    )

    with pytest.raises(
        ModelGatewayValidationError,
        match=rf"^causal_admissibility:{expected_reason}$",
    ):
        gateway.complete_structured(mutated_request)


def test_same_plan_preintervention_inspection_is_admitted_after_its_mutation(
    tmp_path: Path,
    probe,
    execution_context,
) -> None:
    registry = CausalTraceRegistry()
    case_probe = _probe_with_targets(
        probe,
        probe_id="P_same_plan_preinspection",
        targets=["H_root"],
    )
    registered, records, signals = _register_plan_signals(
        registry=registry,
        probe=case_probe,
        execution_context=execution_context,
        plan=_intervene_plan(include_inspect=True),
        states=(("env:0", "env:0"), ("env:0", "env:1"), ("env:1", "env:1")),
    )
    request = _native_evidence_request(
        signal=signals[0],
        record=records[0],
        registered=registered,
        hypothesis_types={"H_root": "root_cause"},
    )
    gateway = causal_module.CausalEvidenceModelGateway(
        delegate=_RecordingDelegate(_judgment(["H_root"])),
        registry=registry,
        artifacts=TrialArtifactStore(tmp_path / "same-plan", restricted_values=()),
    )

    assert gateway.complete_structured(request)["evidence_type"] == "supporting"
    decision = json.loads(
        (tmp_path / "same-plan" / "causal_decisions.jsonl")
        .read_text(encoding="utf-8")
        .strip()
    )
    assert decision["reason_code"] == "state_scoped_inspection"


@pytest.mark.parametrize("task", ["judge_evidence", "repair_evidence_judgment"])
def test_causal_gateway_delegates_before_discard_without_fabricating_a_judgment(
    probe,
    execution_context,
    task: str,
) -> None:
    registry, judge_request, _ = _current_case(
        probe=probe,
        execution_context=execution_context,
        role="intervene",
        hypothesis_types={"H_post": "postcondition"},
    )
    request = judge_request
    if task == "repair_evidence_judgment":
        request = StructuredModelRequest(
            task=task,
            input={
                "original_request": {
                    "task": judge_request.task,
                    "input": copy.deepcopy(judge_request.input),
                },
                "invalid_payload": {"evidence_type": "invalid"},
                "validation_error": "invalid evidence_type",
                "attempt_index": 1,
            },
            metadata={**judge_request.metadata, "repair_attempt_index": 1},
        )
    events: list[str] = []
    delegate = _RecordingDelegate(_judgment(["H_post"]), events=events)
    artifacts = _RecordingDecisionSink(events)
    gateway = causal_module.CausalEvidenceModelGateway(
        delegate=delegate,
        registry=registry,
        artifacts=artifacts,
    )

    with pytest.raises(
        ModelGatewayValidationError,
        match="^causal_admissibility:nonneutral_mutation_acknowledgement$",
    ):
        gateway.complete_structured(request)

    assert events == ["delegate", "decision"]
    assert delegate.requests == [request]
    assert len(artifacts.decisions) == 1


def test_causal_gateway_preserves_delegate_identity_and_passes_other_tasks_unchanged(
    probe,
    execution_context,
) -> None:
    registry = CausalTraceRegistry()
    response = {"answer": "unchanged delegate response"}
    delegate = _RecordingDelegate(response)
    artifacts = _RecordingDecisionSink([])
    gateway = causal_module.CausalEvidenceModelGateway(
        delegate=delegate,
        registry=registry,
        artifacts=artifacts,
    )
    request = StructuredModelRequest(task="project_answer", input={"run_id": "run_1"})

    assert gateway.adapter_kind is delegate.adapter_kind
    assert gateway.model_identity is delegate.model_identity
    assert gateway.config is delegate.config
    assert gateway.invocation_observer is delegate.invocation_observer
    assert gateway.complete_structured(request) is response
    assert delegate.requests == [request]
    assert artifacts.decisions == []
