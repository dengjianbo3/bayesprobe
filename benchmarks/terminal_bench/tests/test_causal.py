from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from bayesprobe import (
    ModelGatewayValidationError,
    ModelTaskFramer,
    StructuredModelRequest,
    TaskAdmissionDecision,
    TaskAdmissionStatus,
    TaskFramingInput,
    TaskKind,
)
from bayesprobe.evidence_memory import SignalProvenanceNormalizer

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
from bayesprobe_terminal_bench.config import RunBudget
from bayesprobe_terminal_bench.provider_contract import TerminalContractModelGateway
from bayesprobe_terminal_bench.runner_factory import BudgetedModelGateway
import bayesprobe_terminal_bench.signals as signals_module
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


class _ScriptedDelegate:
    adapter_kind = "scripted-causal-delegate"
    model_identity = "scripted-causal-model"

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.requests: list[StructuredModelRequest] = []
        self.config = object()
        self.invocation_observer = object()

    def complete_structured(self, request: StructuredModelRequest) -> dict[str, Any]:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError(f"unexpected model task: {request.task}")
        return self.responses.pop(0)


class _RecordingDecisionSink:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.decisions: list[dict[str, Any]] = []

    def append_causal_decision(self, payload: Any) -> None:
        self.events.append("decision")
        if hasattr(payload, "model_dump"):
            payload = payload.model_dump(mode="json")
        self.decisions.append(dict(payload))


def _terminal_frame(
    hypothesis_types: tuple[str, str],
    *,
    valid: bool = True,
) -> dict[str, Any]:
    answer_contract = {
        "objective": "Diagnose the terminal task from observations.",
        "answer_value_type": "structured_text",
        "answer_format": "structured text with verification",
        "required_sections": ["result", "verification"],
        "decision_form": "environment_change",
        "permits_synthesis": True,
    }
    if not valid:
        answer_contract.pop("objective")
    return {
        "task_kind": "design",
        "answer_relationship": "synthesis",
        "answer_contract": answer_contract,
        "competition": "independent",
        "coverage": "open",
        "hypotheses": [
            {
                "statement": f"Terminal hypothesis {index}.",
                "type": hypothesis_type,
                "scope": "The terminal workspace.",
                "falsifiers": [f"Falsifier {index}."],
                "predictions": [f"Prediction {index}."],
                "answer_value": None,
            }
            for index, hypothesis_type in enumerate(hypothesis_types, start=1)
        ],
        "coverage_statement": "The frame covers the declared terminal hypotheses.",
        "coverage_limitation": "Other terminal causes may remain.",
    }


def _frame_request(run_id: str) -> StructuredModelRequest:
    return StructuredModelRequest(
        task="frame_open_question",
        input={
            "question": "Repair the terminal task.",
            "task_context": "Use the provided workspace and tests.",
        },
        prompt_id="open_question_task_framing",
        prompt_version="v0.2",
        schema_name="OpenQuestionTaskFrame",
        schema_version="v0.2",
        metadata={"run_id": run_id},
    )


def _task_admission() -> TaskAdmissionDecision:
    return TaskAdmissionDecision(
        attempt_id="terminal-task-admission",
        status=TaskAdmissionStatus.ADMITTED,
        epistemic_basis=["The task requests a bounded terminal repair."],
        proposed_task_kind=TaskKind.DESIGN,
        answer_contract_outline={
            "objective": "Diagnose the terminal task from observations.",
            "answer_value_type": "structured_text",
            "decision_form": "environment_change",
            "permits_synthesis": True,
            "required_sections": ["result", "verification"],
        },
        reason="The terminal task is admitted by the benchmark harness.",
    )


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
    signal = SignalProvenanceNormalizer().normalize(
        signal.model_copy(deep=True),
        run_id=record.run_id,
    )
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


def _repair_evidence_request(
    judge_request: StructuredModelRequest,
    *,
    original_declarations: dict[str, Any] | None = None,
    input_attempt_index: int = 1,
    metadata_attempt_index: int = 1,
    prompt_id: str | None = None,
    prompt_version: str | None = None,
    schema_name: str | None = None,
    schema_version: str | None = None,
    metadata_overrides: dict[str, Any] | None = None,
) -> StructuredModelRequest:
    original_request: dict[str, Any] = {
        "task": judge_request.task,
        "input": copy.deepcopy(judge_request.input),
    }
    original_request.update(original_declarations or {})
    return StructuredModelRequest(
        task="repair_evidence_judgment",
        input={
            "original_request": original_request,
            "invalid_payload": {"evidence_type": "invalid"},
            "validation_error": "invalid evidence_type",
            "attempt_index": input_attempt_index,
        },
        prompt_id=prompt_id or "evidence_judgment_repair",
        prompt_version=prompt_version or judge_request.prompt_version,
        schema_name=schema_name or judge_request.schema_name,
        schema_version=schema_version or judge_request.schema_version,
        metadata={
            **judge_request.metadata,
            "repair_attempt_index": metadata_attempt_index,
            **(metadata_overrides or {}),
        },
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
        ("signal_targets", "unbound_signal"),
        ("probe_targets", "target_mismatch"),
        ("subject_state", "unbound_signal"),
        ("provenance_state", "unbound_signal"),
        ("policy_attempt", "unbound_signal"),
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
        request = _repair_evidence_request(judge_request)
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


@pytest.mark.parametrize(
    "contradiction",
    [
        "reverse_binding",
        "duplicate_signal_binding",
        "fingerprint",
        "missing_signal_snapshot",
    ],
)
def test_causal_gateway_propagates_registry_integrity_contradictions(
    probe,
    execution_context,
    contradiction: str,
) -> None:
    registry, request, record = _current_case(
        probe=probe,
        execution_context=execution_context,
        role="inspect",
        hypothesis_types={"H_root": "root_cause"},
    )
    if contradiction == "reverse_binding":
        registry._action_to_signal[record.action_id] = "S_conflicting_binding"
    elif contradiction == "duplicate_signal_binding":
        registry._signal_to_action["S_ambiguous_binding"] = record.action_id
    elif contradiction == "fingerprint":
        registry._actions[record.action_id] = record.model_copy(
            deep=True,
            update={"request_fingerprint": "sha256:" + "f" * 64},
        )
    elif contradiction == "missing_signal_snapshot":
        registry._signals.pop(request.input["signal"]["id"])
    else:
        raise AssertionError(contradiction)
    events: list[str] = []
    delegate = _RecordingDelegate(_judgment(["H_root"]), events=events)
    artifacts = _RecordingDecisionSink(events)
    gateway = causal_module.CausalEvidenceModelGateway(
        delegate=delegate,
        registry=registry,
        artifacts=artifacts,
    )

    with pytest.raises(causal_module.CausalTraceError):
        gateway.complete_structured(request)

    assert events == ["delegate"]
    assert delegate.requests == [request]
    assert artifacts.decisions == []


def test_unknown_signal_remains_an_expected_unbound_discard(
    tmp_path: Path,
    probe,
    execution_context,
) -> None:
    registry, request, _ = _current_case(
        probe=probe,
        execution_context=execution_context,
        role="inspect",
        hypothesis_types={"H_root": "root_cause"},
    )
    request_input = copy.deepcopy(request.input)
    request_input["signal"]["id"] = "S_unknown"
    unknown_request = _replace_request_input(request, request_input)
    unknown_request = StructuredModelRequest(
        task=unknown_request.task,
        input=unknown_request.input,
        prompt_id=unknown_request.prompt_id,
        prompt_version=unknown_request.prompt_version,
        schema_name=unknown_request.schema_name,
        schema_version=unknown_request.schema_version,
        metadata={**unknown_request.metadata, "signal_id": "S_unknown"},
    )
    gateway = causal_module.CausalEvidenceModelGateway(
        delegate=_RecordingDelegate(_judgment(["H_root"])),
        registry=registry,
        artifacts=TrialArtifactStore(tmp_path, restricted_values=()),
    )

    with pytest.raises(
        ModelGatewayValidationError,
        match="^causal_admissibility:unbound_signal$",
    ):
        gateway.complete_structured(unknown_request)


def test_action_record_snapshots_cannot_mutate_later_causal_decisions(
    tmp_path: Path,
    probe,
    execution_context,
) -> None:
    predictions = (
        TransitionPrediction(
            hypothesis_id="H_root_a",
            expected_transition="The verifier passes only for root cause A.",
        ),
        TransitionPrediction(
            hypothesis_id="H_root_b",
            expected_transition="The verifier still fails for root cause B.",
        ),
    )
    registry, request, returned_record = _current_case(
        probe=probe,
        execution_context=execution_context,
        role="verify",
        hypothesis_types={
            "H_root_a": "root_cause",
            "H_root_b": "causal_effect",
        },
        predictions=predictions,
    )
    signal_id = request.input["signal"]["id"]

    returned_record.transition_predictions.clear()
    looked_up = registry.record_for_signal(signal_id)
    looked_up.transition_predictions["H_root_a"] = "caller mutation"
    context = registry._admissibility_context_for_signal(signal_id)
    context.record.transition_predictions.clear()
    context.current_action.transition_predictions.clear()

    response = _judgment(["H_root_a", "H_root_b"])
    gateway = causal_module.CausalEvidenceModelGateway(
        delegate=_RecordingDelegate(response),
        registry=registry,
        artifacts=TrialArtifactStore(tmp_path, restricted_values=()),
    )

    assert gateway.complete_structured(request) is response
    decision = json.loads(
        (tmp_path / "causal_decisions.jsonl").read_text(encoding="utf-8").strip()
    )
    assert decision["reason_code"] == "preregistered_causal_transition"


def test_signal_builder_receives_an_isolated_action_snapshot(
    tmp_path: Path,
    probe,
    execution_context,
) -> None:
    hypothesis_types = {
        "H_root_a": "root_cause",
        "H_root_b": "causal_effect",
    }
    predictions = (
        TransitionPrediction(
            hypothesis_id="H_root_a",
            expected_transition="The verifier passes only for root cause A.",
        ),
        TransitionPrediction(
            hypothesis_id="H_root_b",
            expected_transition="The verifier still fails for root cause B.",
        ),
    )
    registry = CausalTraceRegistry()
    case_probe = _probe_with_targets(
        probe,
        probe_id="P_builder_snapshot",
        targets=list(hypothesis_types),
    )
    plan = _intervene_plan(predictions=predictions)
    registered = registry.register_plan(
        probe=case_probe,
        context=execution_context,
        plan=plan,
    )
    states = (("env:0", "env:1"), ("env:1", "env:1"))
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

    def mutating_builder(builder_record):
        builder_record.transition_predictions.clear()
        return signals_module._build_signal(
            causal_record=builder_record,
            probe=case_probe,
            context=execution_context,
            redact_model_content=lambda value: value,
        )

    signal = registry.bind_signal(
        action_id=records[1].action_id,
        signal_builder=mutating_builder,
    )
    request = _native_evidence_request(
        signal=signal,
        record=records[1],
        registered=registered,
        hypothesis_types=hypothesis_types,
    )
    response = _judgment(list(hypothesis_types))
    gateway = causal_module.CausalEvidenceModelGateway(
        delegate=_RecordingDelegate(response),
        registry=registry,
        artifacts=TrialArtifactStore(tmp_path, restricted_values=()),
    )

    assert gateway.complete_structured(request) is response
    stored = registry.record_for_signal(signal.id)
    assert stored.transition_predictions == {
        prediction.hypothesis_id: prediction.expected_transition
        for prediction in predictions
    }


@pytest.mark.parametrize(
    "malformation",
    [
        "conflicting_input_signal_id",
        "conflicting_metadata_signal_id",
        "signal_cycle_id",
        "signal_kind",
        "signal_source_type",
        "signal_source",
        "signal_raw_observation",
        "signal_generated_by_probe",
        "signal_inbox_status",
        "signal_targets",
        "provenance_epistemic_origin",
        "provenance_source_identity",
        "provenance_provider_identity",
        "provenance_session_id",
        "provenance_parent_signal_ids",
        "provenance_derivation_root_id",
        "provenance_correlation_group",
        "provenance_supplied_correlation_group",
        "provenance_content_fingerprint",
        "provenance_citations",
        "provenance_artifact_refs",
        "provenance_environment_state",
        "prompt_id",
        "prompt_version",
        "schema_name",
        "schema_version",
    ],
)
def test_malformed_judge_request_is_unbound_from_the_registry_signal_snapshot(
    tmp_path: Path,
    probe,
    execution_context,
    malformation: str,
) -> None:
    registry, request, _ = _current_case(
        probe=probe,
        execution_context=execution_context,
        role="inspect",
        hypothesis_types={"H_root": "root_cause"},
    )
    request_input = copy.deepcopy(request.input)
    metadata = dict(request.metadata)
    prompt_id = request.prompt_id
    prompt_version = request.prompt_version
    schema_name = request.schema_name
    schema_version = request.schema_version
    if malformation == "conflicting_input_signal_id":
        request_input["signal_id"] = "S_conflicting_input"
    elif malformation == "conflicting_metadata_signal_id":
        metadata["signal_id"] = "S_conflicting_metadata"
    elif malformation == "signal_cycle_id":
        request_input["signal"]["cycle_id"] = "cycle_conflicting"
    elif malformation == "signal_kind":
        request_input["signal"]["signal_kind"] = "passive"
    elif malformation == "signal_source_type":
        request_input["signal"]["source_type"] = "conflicting_source_type"
    elif malformation == "signal_source":
        request_input["signal"]["source"] = "harbor:conflicting-source"
    elif malformation == "signal_raw_observation":
        raw = json.loads(request_input["signal"]["raw_content"])
        raw["observation"] = "conflicting observation"
        request_input["signal"]["raw_content"] = json.dumps(raw, sort_keys=True)
    elif malformation == "signal_generated_by_probe":
        request_input["signal"]["generated_by_probe"] = "P_conflicting"
    elif malformation == "signal_inbox_status":
        request_input["signal"]["inbox_status"] = "discarded"
    elif malformation == "signal_targets":
        request_input["signal"]["initial_target_hypotheses"] = ["H_other"]
    elif malformation == "provenance_epistemic_origin":
        request_input["provenance"]["epistemic_origin"] = "provider_model"
    elif malformation == "provenance_source_identity":
        request_input["provenance"]["source_identity"] = "conflicting-source"
    elif malformation == "provenance_provider_identity":
        request_input["provenance"]["provider_model_or_tool_identity"] = (
            "conflicting-provider"
        )
    elif malformation == "provenance_session_id":
        request_input["provenance"]["session_id"] = "conflicting-session"
    elif malformation == "provenance_parent_signal_ids":
        request_input["provenance"]["parent_signal_ids"] = ["S_parent"]
    elif malformation == "provenance_derivation_root_id":
        request_input["provenance"]["derivation_root_id"] = "conflicting-root"
    elif malformation == "provenance_correlation_group":
        request_input["provenance"]["correlation_group"] = "conflicting-group"
    elif malformation == "provenance_supplied_correlation_group":
        request_input["provenance"]["supplied_correlation_group"] = "supplied"
    elif malformation == "provenance_content_fingerprint":
        request_input["provenance"]["canonical_content_fingerprint"] = (
            "sha256:" + "f" * 64
        )
    elif malformation == "provenance_citations":
        request_input["provenance"]["citations"] = ["citation"]
    elif malformation == "provenance_artifact_refs":
        request_input["provenance"]["artifact_refs"] = ["conflicting-artifact"]
    elif malformation == "provenance_environment_state":
        request_input["provenance"]["environment_state_id"] = "env:other"
    elif malformation == "prompt_id":
        prompt_id = "conflicting_evidence_prompt"
    elif malformation == "prompt_version":
        prompt_version = "v0.1"
    elif malformation == "schema_name":
        schema_name = "ConflictingEvidenceSchema"
    elif malformation == "schema_version":
        schema_version = "v0.1"
    else:
        raise AssertionError(malformation)
    malformed = StructuredModelRequest(
        task=request.task,
        input=request_input,
        prompt_id=prompt_id,
        prompt_version=prompt_version,
        schema_name=schema_name,
        schema_version=schema_version,
        metadata=metadata,
    )
    delegate = _RecordingDelegate(_judgment(["H_root"]))
    artifacts = _RecordingDecisionSink([])
    gateway = causal_module.CausalEvidenceModelGateway(
        delegate=delegate,
        registry=registry,
        artifacts=artifacts,
    )

    with pytest.raises(
        ModelGatewayValidationError,
        match="^causal_admissibility:unbound_signal$",
    ):
        gateway.complete_structured(malformed)

    assert delegate.requests == [malformed]
    assert [item["reason_code"] for item in artifacts.decisions] == [
        "unbound_signal"
    ]


@pytest.mark.parametrize(
    "malformation",
    [
        "original_metadata",
        "outer_metadata",
        "original_schema",
        "outer_schema",
        "original_prompt",
        "outer_prompt",
        "attempt_index",
    ],
)
def test_conflicting_repair_declarations_are_unbound(
    probe,
    execution_context,
    malformation: str,
) -> None:
    registry, judge_request, _ = _current_case(
        probe=probe,
        execution_context=execution_context,
        role="inspect",
        hypothesis_types={"H_root": "root_cause"},
    )
    kwargs: dict[str, Any] = {}
    if malformation == "original_metadata":
        kwargs["original_declarations"] = {
            "metadata": {
                **judge_request.metadata,
                "signal_id": "S_conflicting_original_metadata",
            }
        }
    elif malformation == "outer_metadata":
        kwargs["metadata_overrides"] = {
            "signal_id": "S_conflicting_outer_metadata"
        }
    elif malformation == "original_schema":
        kwargs["original_declarations"] = {
            "schema_name": judge_request.schema_name,
            "schema_version": "v0.1",
        }
    elif malformation == "outer_schema":
        kwargs["schema_name"] = "ConflictingEvidenceSchema"
    elif malformation == "original_prompt":
        kwargs["original_declarations"] = {
            "prompt_id": "conflicting_original_prompt",
            "prompt_version": judge_request.prompt_version,
        }
    elif malformation == "outer_prompt":
        kwargs["prompt_id"] = "conflicting_repair_prompt"
    elif malformation == "attempt_index":
        kwargs["input_attempt_index"] = 2
        kwargs["metadata_attempt_index"] = 1
    else:
        raise AssertionError(malformation)
    request = _repair_evidence_request(judge_request, **kwargs)
    delegate = _RecordingDelegate(_judgment(["H_root"]))
    artifacts = _RecordingDecisionSink([])
    gateway = causal_module.CausalEvidenceModelGateway(
        delegate=delegate,
        registry=registry,
        artifacts=artifacts,
    )

    with pytest.raises(
        ModelGatewayValidationError,
        match="^causal_admissibility:unbound_signal$",
    ):
        gateway.complete_structured(request)

    assert delegate.requests == [request]
    assert [item["reason_code"] for item in artifacts.decisions] == [
        "unbound_signal"
    ]


def test_likelihood_keys_must_be_strings_before_target_set_comparison(
    probe,
    execution_context,
) -> None:
    registry, request, _ = _current_case(
        probe=probe,
        execution_context=execution_context,
        role="inspect",
        hypothesis_types={"1": "root_cause"},
    )
    response = _judgment(["1"], likelihoods={1: "strongly_confirming"})
    artifacts = _RecordingDecisionSink([])
    gateway = causal_module.CausalEvidenceModelGateway(
        delegate=_RecordingDelegate(response),
        registry=registry,
        artifacts=artifacts,
    )

    with pytest.raises(
        ModelGatewayValidationError,
        match="^causal_admissibility:target_mismatch$",
    ):
        gateway.complete_structured(request)

    assert [item["reason_code"] for item in artifacts.decisions] == [
        "target_mismatch"
    ]


def test_mixed_likelihood_key_types_record_one_stable_target_mismatch_decision(
    probe,
    execution_context,
) -> None:
    registry, request, _ = _current_case(
        probe=probe,
        execution_context=execution_context,
        role="inspect",
        hypothesis_types={"H1": "root_cause"},
    )
    likelihood_orders = (
        {"H1": "strongly_confirming", 1: "neutral"},
        {1: "neutral", "H1": "strongly_confirming"},
    )
    judgment_hashes: list[str] = []

    for likelihoods in likelihood_orders:
        events: list[str] = []
        response = _judgment(["H1"], likelihoods=likelihoods)
        delegate = _RecordingDelegate(response, events=events)
        artifacts = _RecordingDecisionSink(events)
        gateway = causal_module.CausalEvidenceModelGateway(
            delegate=delegate,
            registry=registry,
            artifacts=artifacts,
        )

        with pytest.raises(ModelGatewayValidationError) as raised:
            gateway.complete_structured(request)

        assert str(raised.value) == "causal_admissibility:target_mismatch"
        assert events == ["delegate", "decision"]
        assert delegate.requests == [request]
        assert len(artifacts.decisions) == 1
        decision = artifacts.decisions[0]
        assert decision["decision"] == "discard"
        assert decision["reason_code"] == "target_mismatch"
        assert len(decision["judgment_response_sha256"]) == 64
        judgment_hashes.append(decision["judgment_response_sha256"])

    assert judgment_hashes[0] == judgment_hashes[1]


@pytest.mark.parametrize(
    ("request_initial_type", "expected_reason"),
    [
        ("postcondition", "verified_postcondition"),
        ("current_behavior", "target_mismatch"),
    ],
)
def test_mixed_initial_and_evolved_targets_use_types_per_target(
    tmp_path: Path,
    probe,
    execution_context,
    request_initial_type: str,
    expected_reason: str,
) -> None:
    registry = CausalTraceRegistry()
    registry._register_frame_hypothesis_types(
        run_id=execution_context.run_id,
        hypothesis_types={"H_initial": "postcondition"},
    )
    hypothesis_types = {
        "H_initial": request_initial_type,
        "H_evolved": "invariant",
    }
    case_probe = _probe_with_targets(
        probe,
        probe_id=f"P_mixed_{request_initial_type}",
        targets=list(hypothesis_types),
    )
    registered, records, signals = _register_plan_signals(
        registry=registry,
        probe=case_probe,
        execution_context=execution_context,
        plan=_intervene_plan(),
        states=(("env:0", "env:1"), ("env:1", "env:1")),
    )
    request = _native_evidence_request(
        signal=signals[1],
        record=records[1],
        registered=registered,
        hypothesis_types=hypothesis_types,
    )
    response = _judgment(list(hypothesis_types))
    artifacts = TrialArtifactStore(
        tmp_path / request_initial_type,
        restricted_values=(),
    )
    gateway = causal_module.CausalEvidenceModelGateway(
        delegate=_RecordingDelegate(response),
        registry=registry,
        artifacts=artifacts,
    )

    if expected_reason == "target_mismatch":
        with pytest.raises(
            ModelGatewayValidationError,
            match="^causal_admissibility:target_mismatch$",
        ):
            gateway.complete_structured(request)
    else:
        assert gateway.complete_structured(request) is response

    decision = json.loads(
        (artifacts.root / "causal_decisions.jsonl")
        .read_text(encoding="utf-8")
        .strip()
    )
    assert decision["reason_code"] == expected_reason


def test_mutating_a_bound_signal_return_value_cannot_change_its_registry_snapshot(
    tmp_path: Path,
    probe,
    execution_context,
) -> None:
    registry = CausalTraceRegistry()
    case_probe = _probe_with_targets(
        probe,
        probe_id="P_signal_return_snapshot",
        targets=["H_root"],
    )
    registered, records, signals = _register_plan_signals(
        registry=registry,
        probe=case_probe,
        execution_context=execution_context,
        plan=_inspect_plan(),
        states=(("env:0", "env:0"),),
    )
    request = _native_evidence_request(
        signal=signals[0],
        record=records[0],
        registered=registered,
        hypothesis_types={"H_root": "root_cause"},
    )
    signals[0].source = "caller-mutated-source"
    signals[0].initial_target_hypotheses.clear()
    assert signals[0].provenance is not None
    signals[0].provenance.source_identity = "caller-mutated-provenance"
    response = _judgment(["H_root"])
    gateway = causal_module.CausalEvidenceModelGateway(
        delegate=_RecordingDelegate(response),
        registry=registry,
        artifacts=TrialArtifactStore(tmp_path, restricted_values=()),
    )

    assert gateway.complete_structured(request) is response


def test_json_object_key_order_does_not_break_signal_snapshot_binding(
    tmp_path: Path,
    probe,
    execution_context,
) -> None:
    registry, request, _ = _current_case(
        probe=probe,
        execution_context=execution_context,
        role="inspect",
        hypothesis_types={"H_root": "root_cause"},
    )
    request_input = copy.deepcopy(request.input)
    original_raw_content = request_input["signal"]["raw_content"]
    raw = json.loads(original_raw_content)
    reordered_raw = {key: raw[key] for key in reversed(tuple(raw))}
    request_input["signal"]["raw_content"] = json.dumps(
        reordered_raw,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    assert request_input["signal"]["raw_content"] != original_raw_content
    reordered_request = _replace_request_input(request, request_input)
    response = _judgment(["H_root"])
    gateway = causal_module.CausalEvidenceModelGateway(
        delegate=_RecordingDelegate(response),
        registry=registry,
        artifacts=TrialArtifactStore(tmp_path, restricted_values=()),
    )

    assert gateway.complete_structured(reordered_request) is response


def test_duplicate_json_keys_do_not_bypass_signal_snapshot_binding(
    probe,
    execution_context,
) -> None:
    registry, request, _ = _current_case(
        probe=probe,
        execution_context=execution_context,
        role="inspect",
        hypothesis_types={"H_root": "root_cause"},
    )
    request_input = copy.deepcopy(request.input)
    original_raw_content = request_input["signal"]["raw_content"]
    request_input["signal"]["raw_content"] = (
        '{"action_index":999,' + original_raw_content.removeprefix("{")
    )
    duplicate_key_request = _replace_request_input(request, request_input)
    artifacts = _RecordingDecisionSink([])
    gateway = causal_module.CausalEvidenceModelGateway(
        delegate=_RecordingDelegate(_judgment(["H_root"])),
        registry=registry,
        artifacts=artifacts,
    )

    with pytest.raises(
        ModelGatewayValidationError,
        match="^causal_admissibility:unbound_signal$",
    ):
        gateway.complete_structured(duplicate_key_request)

    assert [decision["reason_code"] for decision in artifacts.decisions] == [
        "unbound_signal"
    ]


def test_likelihood_object_key_order_is_not_semantic(
    tmp_path: Path,
    probe,
    execution_context,
) -> None:
    registry, request, _ = _current_case(
        probe=probe,
        execution_context=execution_context,
        role="inspect",
        hypothesis_types={
            "H_root": "root_cause",
            "H_behavior": "current_behavior",
        },
    )
    response = _judgment(
        ["H_root", "H_behavior"],
        likelihoods={
            "H_behavior": "moderately_disconfirming",
            "H_root": "strongly_confirming",
        },
    )
    gateway = causal_module.CausalEvidenceModelGateway(
        delegate=_RecordingDelegate(response),
        registry=registry,
        artifacts=TrialArtifactStore(tmp_path, restricted_values=()),
    )

    assert gateway.complete_structured(request) is response


def test_concurrent_callers_cannot_mutate_registry_owned_snapshots(
    tmp_path: Path,
    probe,
    execution_context,
) -> None:
    predictions = (
        TransitionPrediction(
            hypothesis_id="H_root_a",
            expected_transition="The verifier passes only for root cause A.",
        ),
        TransitionPrediction(
            hypothesis_id="H_root_b",
            expected_transition="The verifier still fails for root cause B.",
        ),
    )
    registry, request, _ = _current_case(
        probe=probe,
        execution_context=execution_context,
        role="verify",
        hypothesis_types={
            "H_root_a": "root_cause",
            "H_root_b": "causal_effect",
        },
        predictions=predictions,
    )
    signal_id = request.input["signal"]["id"]

    def mutate_snapshot(index: int) -> None:
        record = registry.record_for_signal(signal_id)
        record.transition_predictions.clear()
        record.transition_predictions[f"H_mutated_{index}"] = "caller mutation"
        context = registry._admissibility_context_for_signal(signal_id)
        context.record.transition_predictions.clear()
        context.current_action.transition_predictions.clear()
        context.signal.initial_target_hypotheses.clear()

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(mutate_snapshot, range(64)))

    assert registry.record_for_signal(signal_id).transition_predictions == {
        prediction.hypothesis_id: prediction.expected_transition
        for prediction in predictions
    }
    response = _judgment(["H_root_a", "H_root_b"])
    gateway = causal_module.CausalEvidenceModelGateway(
        delegate=_RecordingDelegate(response),
        registry=registry,
        artifacts=TrialArtifactStore(tmp_path, restricted_values=()),
    )
    assert gateway.complete_structured(request) is response


def test_real_contract_composition_freezes_only_the_final_valid_frame_types(
    tmp_path: Path,
    probe,
    execution_context,
) -> None:
    invalid_frame = _terminal_frame(("root_cause", "causal_effect"), valid=False)
    valid_frame = _terminal_frame(("postcondition", "invariant"))
    response = _judgment(["H1", "H2"])
    provider = _ScriptedDelegate([invalid_frame, valid_frame, response])
    artifacts = TrialArtifactStore(tmp_path, restricted_values=())
    registry = CausalTraceRegistry()
    contracted = TerminalContractModelGateway(
        BudgetedModelGateway(provider, RunBudget(max_model_calls=3)),
        artifacts=artifacts,
    )
    gateway = causal_module.CausalEvidenceModelGateway(
        delegate=contracted,
        registry=registry,
        artifacts=artifacts,
    )

    assert gateway.complete_structured(_frame_request(execution_context.run_id)) is valid_frame

    case_probe = _probe_with_targets(
        probe,
        probe_id="P_final_contract_frame",
        targets=["H1", "H2"],
    )
    registered, records, signals = _register_plan_signals(
        registry=registry,
        probe=case_probe,
        execution_context=execution_context,
        plan=_intervene_plan(),
        states=(("env:0", "env:1"), ("env:1", "env:1")),
    )
    evidence_request = _native_evidence_request(
        signal=signals[1],
        record=records[1],
        registered=registered,
        hypothesis_types={"H1": "postcondition", "H2": "invariant"},
    )

    assert gateway.complete_structured(evidence_request) is response
    assert [request.task for request in provider.requests] == [
        "frame_open_question",
        "repair_task_frame",
        "judge_evidence",
    ]
    decision = json.loads(
        (tmp_path / "causal_decisions.jsonl").read_text(encoding="utf-8").strip()
    )
    assert decision["reason_code"] == "verified_postcondition"


def test_public_task_frame_repair_replaces_only_the_provisional_type_capture(
    tmp_path: Path,
    execution_context,
) -> None:
    invalid_frame = _terminal_frame(("root_cause", "causal_effect"), valid=False)
    repaired_frame = _terminal_frame(("postcondition", "invariant"))
    delegate = _ScriptedDelegate([invalid_frame, repaired_frame])
    registry = CausalTraceRegistry()
    gateway = causal_module.CausalEvidenceModelGateway(
        delegate=delegate,
        registry=registry,
        artifacts=TrialArtifactStore(tmp_path, restricted_values=()),
    )
    framer = ModelTaskFramer(gateway)

    frame = framer.frame(
        TaskFramingInput(
            run_id=execution_context.run_id,
            question="Repair the terminal task.",
            task_context="Use the provided workspace and tests.",
            admission_decision=_task_admission(),
        )
    )

    assert [hypothesis.type for hypothesis in frame.hypothesis_frame.hypotheses] == [
        "postcondition",
        "invariant",
    ]
    assert registry._hypothesis_types_by_run[execution_context.run_id] == {
        "H1": "postcondition",
        "H2": "invariant",
    }
    assert [request.task for request in delegate.requests] == [
        "frame_open_question",
        "repair_task_frame",
    ]
