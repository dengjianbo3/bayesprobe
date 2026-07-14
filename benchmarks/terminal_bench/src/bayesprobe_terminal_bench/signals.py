from __future__ import annotations

import hashlib
import json
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

from bayesprobe_terminal_bench.actions import ActionObservation


_HARBOR_TOOL_IDENTITY = "harbor:0.18.0"
_SIGNAL_SCHEMA_VERSION = "harbor-observation:v1"


def signal_from_observation(
    *,
    observation: ActionObservation,
    probe: ProbeDesign,
    context: ProbeExecutionBrief,
) -> ExternalSignal:
    """Convert one completed Harbor action into one public external signal."""
    raw_content = _canonical_json(_observation_payload(observation))
    environment_digest = _digest(
        {
            "run_id": context.run_id,
            "schema_version": _SIGNAL_SCHEMA_VERSION,
            "post_environment_state_id": observation.post_environment_state_id,
        }
    )
    source_identity = f"harbor-terminal:sha256:{environment_digest}"
    derivation_root_id = (
        f"harbor-action:sha256:{_digest(_root_inputs(observation, probe, context))}"
    )
    signal_id = f"S_harbor_{_digest(_signal_id_inputs(observation, probe, context))}"
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
        artifact_refs=[f"environment_actions.jsonl#{observation.action_index}"],
        environment_state_id=observation.post_environment_state_id,
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


def _observation_payload(observation: ActionObservation) -> dict[str, Any]:
    return {
        "action_index": observation.action_index,
        "action_type": observation.action.type,
        "error_category": observation.error_category,
        "model_facing_output": observation.model_facing_output,
        "output_truncated": observation.output_truncated,
        "post_environment_state_id": observation.post_environment_state_id,
        "pre_environment_state_id": observation.pre_environment_state_id,
        "return_code": observation.return_code,
        "timed_out": observation.timed_out,
    }


def _root_inputs(
    observation: ActionObservation,
    probe: ProbeDesign,
    context: ProbeExecutionBrief,
) -> dict[str, Any]:
    return {
        "action": observation.action.model_dump(mode="json"),
        "action_index": observation.action_index,
        "cycle_id": context.cycle_id,
        "full_output_sha256": observation.full_output_sha256,
        "post_environment_state_id": observation.post_environment_state_id,
        "pre_environment_state_id": observation.pre_environment_state_id,
        "probe_id": probe.id,
        "run_id": context.run_id,
        "schema_version": _SIGNAL_SCHEMA_VERSION,
    }


def _signal_id_inputs(
    observation: ActionObservation,
    probe: ProbeDesign,
    context: ProbeExecutionBrief,
) -> dict[str, Any]:
    return _root_inputs(observation, probe, context)


def _canonical_content_fingerprint(source_identity: str, raw_content: str) -> str:
    canonical_content = " ".join(unicodedata.normalize("NFKC", raw_content).split())
    digest = hashlib.sha256(
        f"{source_identity}\\n{canonical_content}".encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def _digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
