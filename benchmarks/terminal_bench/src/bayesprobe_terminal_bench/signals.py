from __future__ import annotations

import hashlib
import unicodedata
from typing import Any

from bayesprobe import (
    EpistemicOrigin,
    ExternalSignal,
    ProbeDesign,
    ProbeExecutionBrief,
    SignalKind,
    SignalProvenance,
)

from bayesprobe_terminal_bench.actions import (
    ActionObservation,
    ShellAction,
    TerminalPlanStep,
    TerminalProbePlan,
    action_may_mutate,
)
from bayesprobe_terminal_bench.causal import (
    CausalActionRecord,
    CausalTraceRegistry,
    canonical_json,
    canonical_sha256,
    executed_request_from_action,
)


_HARBOR_TOOL_IDENTITY = "harbor:0.18.0"
_SIGNAL_SCHEMA_VERSION = "harbor-observation:v3"
_MAX_OBSERVATION_BYTES = 32_768


def signal_from_observation(
    *,
    observation: ActionObservation,
    probe: ProbeDesign,
    context: ProbeExecutionBrief,
    causal_record: CausalActionRecord | None = None,
) -> ExternalSignal:
    """Convert one completed Harbor action into one public external signal."""
    if causal_record is None:
        causal_record = _standalone_causal_record(
            observation=observation,
            probe=probe,
            context=context,
        )
    _validate_causal_binding(
        observation=observation,
        probe=probe,
        context=context,
        causal_record=causal_record,
    )
    raw_content = canonical_json(_observation_payload(observation, causal_record))
    environment_digest = canonical_sha256(
        {
            "run_id": context.run_id,
            "schema_version": _SIGNAL_SCHEMA_VERSION,
            "subject_environment_state_id": causal_record.subject_environment_state_id,
        }
    )
    source_identity = f"harbor-terminal:sha256:{environment_digest}"
    derivation_root_id = (
        f"harbor-action:sha256:{canonical_sha256(_root_inputs(causal_record))}"
    )
    signal_id = f"S_harbor_{canonical_sha256(_signal_id_inputs(causal_record))}"
    provenance = SignalProvenance(
        epistemic_origin=EpistemicOrigin.TOOL_RESULT,
        source_identity=source_identity,
        provider_model_or_tool_identity=_HARBOR_TOOL_IDENTITY,
        derivation_root_id=derivation_root_id,
        correlation_group=f"harbor-env:sha256:{environment_digest}",
        canonical_content_fingerprint=_canonical_content_fingerprint(
            source_identity,
            raw_content,
        ),
        artifact_refs=[
            f"environment_actions.jsonl#{observation.action_index}",
            f"causal_actions.jsonl#{causal_record.action_id}",
        ],
        environment_state_id=causal_record.subject_environment_state_id,
    )
    return ExternalSignal(
        id=signal_id,
        cycle_id=context.cycle_id,
        signal_kind=SignalKind.ACTIVE,
        source_type="harbor_terminal",
        source="harbor:environment",
        raw_content=raw_content,
        generated_by_probe=probe.id,
        initial_target_hypotheses=list(probe.target_hypotheses),
        provenance=provenance,
    )


def _observation_payload(
    observation: ActionObservation,
    causal_record: CausalActionRecord,
) -> dict[str, Any]:
    bounded_observation, _ = _bounded_text(
        observation.model_facing_output,
        _MAX_OBSERVATION_BYTES,
    )
    return {
        "action_index": observation.action_index,
        "causal_binding": {
            "action_id": causal_record.action_id,
            "action_role": causal_record.action_role,
            "plan_id": causal_record.plan_id,
            "policy_attempt_id": causal_record.policy_attempt_id,
            "request_fingerprint": causal_record.request_fingerprint,
            "subject_environment_state_id": causal_record.subject_environment_state_id,
            "verification_target": causal_record.verification_target,
        },
        "executed_request": executed_request_from_action(observation.action),
        "observation": bounded_observation,
        "post_environment_state_id": observation.post_environment_state_id,
        "pre_environment_state_id": observation.pre_environment_state_id,
    }


def _root_inputs(causal_record: CausalActionRecord) -> dict[str, Any]:
    return {
        "action_id": causal_record.action_id,
        "full_output_sha256": causal_record.observation.full_output_sha256,
        "schema_version": _SIGNAL_SCHEMA_VERSION,
    }


def _signal_id_inputs(causal_record: CausalActionRecord) -> dict[str, Any]:
    return _root_inputs(causal_record)


def _canonical_content_fingerprint(source_identity: str, raw_content: str) -> str:
    canonical_content = " ".join(unicodedata.normalize("NFKC", raw_content).split())
    digest = hashlib.sha256(
        f"{source_identity}\n{canonical_content}".encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def _validate_causal_binding(
    *,
    observation: ActionObservation,
    probe: ProbeDesign,
    context: ProbeExecutionBrief,
    causal_record: CausalActionRecord,
) -> None:
    if causal_record.observation != observation:
        raise ValueError("causal record does not contain the completed observation")
    if causal_record.probe_id != probe.id:
        raise ValueError("causal record does not match the Probe")
    if (
        causal_record.run_id != context.run_id
        or causal_record.cycle_id != context.cycle_id
    ):
        raise ValueError("causal record does not match the execution context")


def _standalone_causal_record(
    *,
    observation: ActionObservation,
    probe: ProbeDesign,
    context: ProbeExecutionBrief,
) -> CausalActionRecord:
    """Support deterministic fixture construction outside the execution gateway."""
    if not action_may_mutate(observation.action):
        step = TerminalPlanStep(role="inspect", action=observation.action)
        plan = TerminalProbePlan(
            mode="inspect",
            steps=(step,),
            expected_observation="The completed action is observed.",
        )
        step_index = 0
    elif isinstance(observation.action, ShellAction):
        step = TerminalPlanStep(
            role="verify",
            action=observation.action,
            verification_target="the completed action result",
        )
        plan = TerminalProbePlan(
            mode="verify",
            steps=(step,),
            expected_observation="The completed verification is observed.",
        )
        step_index = 0
    else:
        step = TerminalPlanStep(role="intervene", action=observation.action)
        plan = TerminalProbePlan(
            mode="intervene",
            steps=(
                step,
                TerminalPlanStep(
                    role="verify",
                    action=ShellAction(command="pwd"),
                    verification_target="the completed intervention state",
                ),
            ),
            expected_observation="The completed intervention is acknowledged.",
        )
        step_index = 0

    registry = CausalTraceRegistry()
    registered = registry.register_plan(probe=probe, context=context, plan=plan)
    return registry.register_action(
        plan=registered,
        step_index=step_index,
        observation=observation,
    )


def _bounded_text(value: str, limit: int) -> tuple[str, bool]:
    encoded = value.encode("utf-8")
    if len(encoded) <= limit:
        return value, False
    return encoded[:limit].decode("utf-8", errors="ignore"), True
