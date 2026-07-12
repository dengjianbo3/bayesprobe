from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import unicodedata
from typing import Literal

from bayesprobe.kernel_config import CorrelationCreditPolicy
from bayesprobe.schemas import (
    EpistemicOrigin,
    EvidenceEvent,
    EvidenceMemorySnapshot,
    ExternalSignal,
    LikelihoodBand,
    SignalProvenance,
    is_forbidden_secret_key_name,
    is_secret_like_value,
)


CorrelationStatus = Literal[
    "novel",
    "duplicate_exact",
    "correlated_restatement",
    "correlated_novel",
]


@dataclass(frozen=True)
class EvidenceMemoryDecision:
    correlation_status: CorrelationStatus
    effective_update_weight: float
    discard_reason: str | None
    remaining_credit: dict[str, float]
    canonical_correlation_group: str


class SignalProvenanceNormalizer:
    """Close raw signals into deterministic, secret-free provenance."""

    def normalize(self, signal: ExternalSignal, *, run_id: str) -> ExternalSignal:
        _reject_secret_signal(signal)
        run_session = _clean_text(run_id)
        supplied = signal.provenance
        origin = supplied.epistemic_origin if supplied else _origin_for(signal)
        source_identity = _clean_text(
            supplied.source_identity
            if supplied
            else f"{signal.source_type}:{signal.source}"
        )
        canonical_content = _clean_text(signal.raw_content)
        fingerprint = _sha256_identity(source_identity, canonical_content)
        provider_identity = (
            supplied.provider_model_or_tool_identity if supplied else None
        )
        if provider_identity is None and origin in {
            EpistemicOrigin.MODEL_REASONING,
            EpistemicOrigin.TOOL_RESULT,
            EpistemicOrigin.DERIVED_SUMMARY,
        }:
            provider_identity = _clean_text(signal.source)
        session_id = supplied.session_id if supplied else None
        if session_id is None and origin == EpistemicOrigin.MODEL_REASONING:
            session_id = run_session
        derivation_root_id = (
            supplied.derivation_root_id
            if supplied
            else f"root:{fingerprint.removeprefix('sha256:')}"
        )
        if origin == EpistemicOrigin.MODEL_REASONING:
            correlation_group = (
                f"model:{provider_identity or source_identity}:{session_id or run_session}"
            )
        elif supplied:
            correlation_group = supplied.correlation_group
        else:
            correlation_group = f"source:{source_identity}"

        provenance = SignalProvenance(
            epistemic_origin=origin,
            source_identity=source_identity,
            provider_model_or_tool_identity=provider_identity,
            session_id=session_id,
            parent_signal_ids=list(supplied.parent_signal_ids) if supplied else [],
            derivation_root_id=derivation_root_id,
            correlation_group=correlation_group,
            canonical_content_fingerprint=fingerprint,
            citations=list(supplied.citations) if supplied else [],
            artifact_refs=list(supplied.artifact_refs) if supplied else [],
            environment_state_id=supplied.environment_state_id if supplied else None,
        )
        _reject_secret_provenance(provenance)
        return signal.model_copy(update={"provenance": provenance})


class EvidenceMemoryManager:
    """Classify signal identity and commit bounded correlation credit."""

    def __init__(self, policy: CorrelationCreditPolicy | None = None) -> None:
        self._policy = policy or CorrelationCreditPolicy()

    def classify(
        self,
        snapshot: EvidenceMemorySnapshot,
        signal: ExternalSignal,
        *,
        likelihoods: dict[str, LikelihoodBand] | None = None,
        unresolved_likelihood: LikelihoodBand | None = None,
        frame_version: int = 1,
        base_effective_weight: float = 0.0,
    ) -> EvidenceMemoryDecision:
        provenance = _required_provenance(signal)
        prior_identities = {
            signal_id: parts
            for signal_id, value in snapshot.source_content_fingerprints.items()
            if (parts := _source_content_identity_parts(value)) is not None
        }
        exact = any(
            prior_identity[0] == provenance.source_identity
            and prior_identity[1] == provenance.canonical_content_fingerprint
            and snapshot.derivation_roots.get(signal_id)
            == provenance.derivation_root_id
            for signal_id, prior_identity in prior_identities.items()
        )
        same_root = provenance.derivation_root_id in snapshot.derivation_roots.values()
        known_parent_roots = [
            snapshot.derivation_roots[parent_id]
            for parent_id in provenance.parent_signal_ids
            if parent_id in snapshot.derivation_roots
        ]
        if any(
            parent_root != provenance.derivation_root_id
            for parent_root in known_parent_roots
        ):
            raise ValueError("derived signals must preserve parent derivation root")
        canonical_group = _canonical_correlation_group(
            snapshot=snapshot,
            provenance=provenance,
            prior_identities=prior_identities,
        )
        group_prefix = f"{canonical_group}|"
        same_group = any(
            key.startswith(group_prefix) for key in snapshot.correlation_credit
        ) or any(
            prior_identity[2] == canonical_group
            for prior_identity in prior_identities.values()
        )
        same_source = any(
            prior_identity[0] == provenance.source_identity
            for prior_identity in prior_identities.values()
        )

        if exact:
            status: CorrelationStatus = "duplicate_exact"
        elif same_root or provenance.parent_signal_ids:
            status = "correlated_restatement"
        elif same_group or same_source:
            status = "correlated_novel"
        else:
            status = "novel"

        cap = self._policy.max_cumulative_effective_weight_per_direction
        credit_keys = _credit_keys(
            canonical_group,
            likelihoods or {},
            unresolved_likelihood=unresolved_likelihood,
            frame_version=frame_version,
        )
        if credit_keys:
            remaining_before = {
                key: max(cap - snapshot.correlation_credit.get(key, 0.0), 0.0)
                for key in credit_keys
            }
        else:
            remaining_before = {
                key: max(cap - used, 0.0)
                for key, used in snapshot.correlation_credit.items()
                if key.startswith(group_prefix)
            }
        if status in {"duplicate_exact", "correlated_restatement"}:
            effective_weight = 0.0
        elif credit_keys:
            effective_weight = min(
                max(float(base_effective_weight), 0.0),
                min(remaining_before.values()),
            )
        else:
            effective_weight = max(float(base_effective_weight), 0.0)
        remaining_after = {
            key: max(remaining - effective_weight, 0.0)
            for key, remaining in remaining_before.items()
        }
        discard_reason = None
        if status == "duplicate_exact":
            discard_reason = "duplicate_exact"
        elif credit_keys and effective_weight == 0.0 and status != "correlated_restatement":
            discard_reason = "correlation_credit_saturated"
        return EvidenceMemoryDecision(
            correlation_status=status,
            effective_update_weight=effective_weight,
            discard_reason=discard_reason,
            remaining_credit=remaining_after,
            canonical_correlation_group=canonical_group,
        )

    def commit(
        self,
        snapshot: EvidenceMemorySnapshot,
        *,
        signal: ExternalSignal,
        event: EvidenceEvent,
        decision: EvidenceMemoryDecision,
    ) -> EvidenceMemorySnapshot:
        if event.id in snapshot.accepted_evidence_ids or any(
            item.partition(":")[0] == event.id
            for item in snapshot.discard_and_schema_history
        ):
            return snapshot

        provenance = _required_provenance(signal)
        accepted_ids = list(snapshot.accepted_evidence_ids)
        discard_history = list(snapshot.discard_and_schema_history)
        if event.discard_reason is None:
            accepted_ids.append(event.id)
        else:
            discard_history.append(f"{event.id}:{event.discard_reason}")

        content_fingerprints = dict(snapshot.content_fingerprints)
        source_content_fingerprints = dict(snapshot.source_content_fingerprints)
        derivation_roots = dict(snapshot.derivation_roots)
        content_fingerprints[signal.id] = provenance.canonical_content_fingerprint
        source_content_fingerprints[signal.id] = _source_content_identity(
            provenance,
            correlation_group=decision.canonical_correlation_group,
        )
        derivation_roots[signal.id] = provenance.derivation_root_id

        cap = self._policy.max_cumulative_effective_weight_per_direction
        correlation_credit = dict(snapshot.correlation_credit)
        if event.discard_reason is None and decision.effective_update_weight > 0:
            for key, remaining in decision.remaining_credit.items():
                correlation_credit[key] = cap - remaining

        counterevidence = {
            hypothesis_id: list(evidence_ids)
            for hypothesis_id, evidence_ids in snapshot.counterevidence_ids_by_hypothesis.items()
        }
        if event.discard_reason is None:
            for hypothesis_id, band in event.likelihoods.items():
                if _direction_for(band) == "disconfirming":
                    counterevidence.setdefault(hypothesis_id, []).append(event.id)

        return EvidenceMemorySnapshot(
            memory_version=snapshot.memory_version,
            accepted_evidence_ids=accepted_ids,
            content_fingerprints=content_fingerprints,
            source_content_fingerprints=source_content_fingerprints,
            derivation_roots=derivation_roots,
            correlation_credit=correlation_credit,
            discovery_evidence_ids=list(snapshot.discovery_evidence_ids),
            counterevidence_ids_by_hypothesis=counterevidence,
            discard_and_schema_history=discard_history,
        )


def _origin_for(signal: ExternalSignal) -> EpistemicOrigin:
    source_type = signal.source_type.casefold()
    if source_type == "model_probe_gateway":
        return EpistemicOrigin.MODEL_REASONING
    if source_type in {"python_sandbox", "tool_result", "deterministic_probe_gateway"}:
        return EpistemicOrigin.TOOL_RESULT
    if source_type in {"retrieved_source", "document_retrieval", "search_result"}:
        return EpistemicOrigin.RETRIEVED_SOURCE
    if source_type == "external_agent_projection":
        return EpistemicOrigin.AGENT_MESSAGE
    if source_type == "human_input":
        return EpistemicOrigin.HUMAN_INPUT
    if source_type == "derived_summary":
        return EpistemicOrigin.DERIVED_SUMMARY
    return EpistemicOrigin.EXTERNAL_OBSERVATION


def _clean_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def _sha256_identity(source_identity: str, content: str) -> str:
    digest = hashlib.sha256(f"{source_identity}\n{content}".encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _reject_secret_signal(signal: ExternalSignal) -> None:
    values = [
        signal.source_type,
        signal.source,
        signal.raw_content,
        signal.generated_by_probe,
        *signal.initial_target_hypotheses,
    ]
    if any(value is not None and is_secret_like_value(value) for value in values):
        raise ValueError("external signal contains secret material")
    if signal.provenance is not None:
        _reject_secret_provenance(signal.provenance)


def _reject_secret_provenance(provenance: SignalProvenance) -> None:
    payload = provenance.model_dump(mode="python")
    for key, value in payload.items():
        if is_forbidden_secret_key_name(key):
            raise ValueError("signal provenance contains secret material")
        values = value if isinstance(value, list) else [value]
        if any(isinstance(item, str) and is_secret_like_value(item) for item in values):
            raise ValueError("signal provenance contains secret material")


def _required_provenance(signal: ExternalSignal) -> SignalProvenance:
    if signal.provenance is None:
        raise ValueError("evidence memory requires normalized signal provenance")
    return signal.provenance


def _source_content_identity(
    provenance: SignalProvenance,
    *,
    correlation_group: str | None = None,
) -> str:
    return json.dumps(
        [
            provenance.source_identity,
            provenance.canonical_content_fingerprint,
            correlation_group or provenance.correlation_group,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _source_content_identity_parts(value: str) -> tuple[str, str, str] | None:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None
    if (
        not isinstance(parsed, list)
        or len(parsed) != 3
        or not all(isinstance(item, str) for item in parsed)
    ):
        return None
    return parsed[0], parsed[1], parsed[2]


def _canonical_correlation_group(
    *,
    snapshot: EvidenceMemorySnapshot,
    provenance: SignalProvenance,
    prior_identities: dict[str, tuple[str, str, str]],
) -> str:
    for signal_id, root in snapshot.derivation_roots.items():
        if root == provenance.derivation_root_id and signal_id in prior_identities:
            return prior_identities[signal_id][2]
    for parent_id in provenance.parent_signal_ids:
        if parent_id in prior_identities:
            return prior_identities[parent_id][2]
    for prior_identity in prior_identities.values():
        if prior_identity[0] == provenance.source_identity:
            return prior_identity[2]
    return provenance.correlation_group


def _direction_for(band: LikelihoodBand) -> Literal["confirming", "disconfirming"] | None:
    if band in {
        LikelihoodBand.WEAKLY_CONFIRMING,
        LikelihoodBand.MODERATELY_CONFIRMING,
        LikelihoodBand.STRONGLY_CONFIRMING,
    }:
        return "confirming"
    if band in {
        LikelihoodBand.WEAKLY_DISCONFIRMING,
        LikelihoodBand.MODERATELY_DISCONFIRMING,
        LikelihoodBand.STRONGLY_DISCONFIRMING,
    }:
        return "disconfirming"
    return None


def _credit_keys(
    correlation_group: str,
    likelihoods: dict[str, LikelihoodBand],
    *,
    unresolved_likelihood: LikelihoodBand | None,
    frame_version: int,
) -> list[str]:
    keys = [
        f"{correlation_group}|{hypothesis_id}|{direction}"
        for hypothesis_id, band in likelihoods.items()
        if (direction := _direction_for(band)) is not None
    ]
    if unresolved_likelihood is not None:
        direction = _direction_for(unresolved_likelihood)
        if direction is not None:
            keys.append(
                f"{correlation_group}|frame:{frame_version}:unresolved|{direction}"
            )
    return keys


__all__ = [
    "EvidenceMemoryDecision",
    "EvidenceMemoryManager",
    "SignalProvenanceNormalizer",
]
