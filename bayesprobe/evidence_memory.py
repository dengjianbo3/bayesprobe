from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import unicodedata
from typing import Any, Literal

from bayesprobe.kernel_config import CorrelationCreditPolicy
from bayesprobe.schemas import (
    EpistemicOrigin,
    EvidenceEvent,
    EvidenceMemorySnapshot,
    EvidenceType,
    ExternalSignal,
    LikelihoodBand,
    SignalProvenance,
    contains_secret_material,
    decode_discard_history_entry,
    encode_discard_history_entry,
    is_forbidden_secret_key_name,
    is_secret_like_value,
)


CorrelationStatus = Literal[
    "novel",
    "duplicate_exact",
    "correlated_restatement",
    "correlated_novel",
]

_CURRENT_MEMORY_VERSION = 2
_SUPPORTED_MEMORY_VERSIONS = frozenset({1, 2})
_MODEL_PROVIDER_FALLBACK_PREFIX = "model_provider_fallback:v1:"
SIGNAL_QUALITY_METRICS = (
    "reliability",
    "independence",
    "relevance",
    "novelty",
    "specificity",
    "verifiability",
)


def cycle_signal_source_content_signature(
    signal: ExternalSignal,
) -> tuple[str, str]:
    return (
        signal.source.strip().lower(),
        " ".join(signal.raw_content.lower().split()),
    )


def observe_cycle_signal_duplicate(
    signal: ExternalSignal,
    seen_signatures: set[tuple[str, str]],
) -> bool:
    signature = cycle_signal_source_content_signature(signal)
    is_duplicate = signature in seen_signatures
    seen_signatures.add(signature)
    return is_duplicate


@dataclass(frozen=True)
class SignalQuality:
    reliability: float
    independence: float
    relevance: float
    novelty: float
    specificity: float
    verifiability: float


class SignalQualityAssessor:
    def assess(
        self,
        *,
        signal: ExternalSignal,
        event_type: EvidenceType,
        is_duplicate: bool = False,
    ) -> SignalQuality:
        if event_type == EvidenceType.SOURCE_CLAIM:
            quality = SignalQuality(
                reliability=0.5,
                independence=0.5,
                relevance=0.7,
                novelty=0.7,
                specificity=0.6,
                verifiability=0.65,
            )
        elif signal.source_type == "model_probe_gateway":
            quality = SignalQuality(
                reliability=0.55,
                independence=0.35,
                relevance=0.85,
                novelty=0.55,
                specificity=0.65,
                verifiability=0.3,
            )
        elif signal.source_type == "python_sandbox":
            quality = SignalQuality(
                reliability=0.75,
                independence=0.35,
                relevance=0.85,
                novelty=0.65,
                specificity=0.9,
                verifiability=0.9,
            )
        elif signal.source_type == "external_agent_projection":
            quality = SignalQuality(
                reliability=0.55,
                independence=0.45,
                relevance=0.75,
                novelty=0.6,
                specificity=0.6,
                verifiability=0.4,
            )
        else:
            quality = SignalQuality(
                reliability=0.8,
                independence=0.8,
                relevance=0.9,
                novelty=0.8,
                specificity=0.7,
                verifiability=0.7,
            )

        if _has_low_reliability_cue(signal.raw_content):
            quality = SignalQuality(
                reliability=min(quality.reliability, 0.35),
                independence=quality.independence,
                relevance=quality.relevance,
                novelty=quality.novelty,
                specificity=quality.specificity,
                verifiability=min(quality.verifiability, 0.4),
            )

        if is_duplicate:
            quality = SignalQuality(
                reliability=quality.reliability,
                independence=0.25,
                relevance=quality.relevance,
                novelty=0.25,
                specificity=quality.specificity,
                verifiability=quality.verifiability,
            )

        provenance = signal.provenance
        if provenance is not None:
            quality = _cap_quality_for_origin(
                quality,
                provenance.epistemic_origin,
            )
        return quality


def _cap_quality_for_origin(
    quality: SignalQuality,
    origin: EpistemicOrigin,
) -> SignalQuality:
    if origin in {
        EpistemicOrigin.MODEL_REASONING,
        EpistemicOrigin.DERIVED_SUMMARY,
    }:
        cap = SignalQuality(
            reliability=0.55,
            independence=0.35,
            relevance=0.85,
            novelty=0.55,
            specificity=0.65,
            verifiability=0.3,
        )
    elif origin == EpistemicOrigin.TOOL_RESULT:
        cap = SignalQuality(
            reliability=0.8,
            independence=0.8,
            relevance=0.9,
            novelty=0.8,
            specificity=0.9,
            verifiability=0.9,
        )
    elif origin == EpistemicOrigin.AGENT_MESSAGE:
        cap = SignalQuality(
            reliability=0.55,
            independence=0.45,
            relevance=0.75,
            novelty=0.6,
            specificity=0.6,
            verifiability=0.4,
        )
    else:
        return quality
    return SignalQuality(
        **{
            metric: min(getattr(quality, metric), getattr(cap, metric))
            for metric in SIGNAL_QUALITY_METRICS
        }
    )


@dataclass(frozen=True)
class EvidenceMemoryDecision:
    correlation_status: CorrelationStatus
    effective_update_weight: float
    discard_reason: str | None
    remaining_credit: dict[str, float]
    canonical_correlation_group: str


def derive_deterministic_computation_root(
    *,
    tool_identity: str,
    computation_inputs: Mapping[str, Any],
) -> str:
    """Hash stable, secret-free computation semantics into one factual root."""
    if not isinstance(tool_identity, str) or not tool_identity.strip():
        raise ValueError("deterministic computation tool identity must not be empty")
    _reject_secret_computation_text(tool_identity)
    canonical_inputs = _canonical_computation_value(computation_inputs)
    payload = json.dumps(
        {
            "computation_inputs": canonical_inputs,
            "tool_identity": tool_identity,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"deterministic-computation:sha256:{digest}"


def canonical_signal_identity_digest(signal: ExternalSignal) -> str:
    provenance = _required_provenance(signal)
    canonical_identity = json.dumps(
        [
            provenance.canonical_content_fingerprint,
            provenance.derivation_root_id,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical_identity.encode("utf-8")).hexdigest()


class SignalProvenanceNormalizer:
    """Close raw signals into deterministic, secret-free provenance."""

    def normalize(self, signal: ExternalSignal, *, run_id: str) -> ExternalSignal:
        _reject_secret_signal(signal)
        supplied = signal.provenance
        supplied_correlation_group = None
        if supplied is not None:
            supplied_correlation_group = (
                supplied.supplied_correlation_group or supplied.correlation_group
            )
        origin = supplied.epistemic_origin if supplied else _origin_for(signal)
        canonical_content = _clean_text(signal.raw_content)
        _reject_secret_hash_text(signal.raw_content)
        _reject_secret_hash_text(canonical_content)
        provider_identity = (
            supplied.provider_model_or_tool_identity if supplied else None
        )
        session_id = supplied.session_id if supplied else None
        if origin == EpistemicOrigin.MODEL_REASONING:
            provider_identity = _model_provider_audit_identity(
                signal,
                provider_identity,
            )
            if session_id is None:
                session_id = run_id
            _reject_secret_hash_text(provider_identity)
            _reject_secret_hash_text(session_id)
            source_identity = _model_source_identity(provider_identity)
            correlation_group = _model_correlation_group(
                provider_identity,
                session_id,
            )
        else:
            raw_source_identity = (
                supplied.source_identity
                if supplied
                else f"{signal.source_type}:{signal.source}"
            )
            source_identity = _clean_text(raw_source_identity)
            _reject_secret_hash_text(raw_source_identity)
            _reject_secret_hash_text(source_identity)
            if provider_identity is None and origin in {
                EpistemicOrigin.TOOL_RESULT,
                EpistemicOrigin.DERIVED_SUMMARY,
            }:
                provider_identity = _clean_text(signal.source)
            if supplied:
                correlation_group = supplied.correlation_group
            else:
                correlation_group = f"source:{source_identity}"

        fingerprint = _sha256_identity(source_identity, canonical_content)
        derivation_root_id = (
            supplied.derivation_root_id
            if supplied
            else f"root:{fingerprint.removeprefix('sha256:')}"
        )

        provenance = SignalProvenance(
            epistemic_origin=origin,
            source_identity=source_identity,
            provider_model_or_tool_identity=provider_identity,
            session_id=session_id,
            parent_signal_ids=list(supplied.parent_signal_ids) if supplied else [],
            derivation_root_id=derivation_root_id,
            correlation_group=correlation_group,
            supplied_correlation_group=supplied_correlation_group,
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

    def validate_signal_lineage(
        self,
        snapshot: EvidenceMemorySnapshot,
        signal: ExternalSignal,
        *,
        canonical_correlation_group: str | None = None,
    ) -> None:
        _require_supported_memory_version(snapshot)
        provenance = _required_provenance(signal)
        prior_source_content = snapshot.source_content_fingerprints.get(signal.id)
        prior_content = snapshot.content_fingerprints.get(signal.id)
        prior_root = snapshot.derivation_roots.get(signal.id)
        identity_present = any(
            value is not None
            for value in (prior_source_content, prior_content, prior_root)
        )
        if identity_present:
            prior_identity = _source_content_identity_parts(prior_source_content)
            if (
                prior_identity is None
                or prior_identity[0] != provenance.source_identity
                or prior_identity[1] != provenance.canonical_content_fingerprint
                or prior_identity[3] != _supplied_correlation_group(provenance)
                or prior_content != provenance.canonical_content_fingerprint
                or prior_root != provenance.derivation_root_id
                or (
                    canonical_correlation_group is not None
                    and prior_identity is not None
                    and prior_identity[2] != canonical_correlation_group
                )
            ):
                raise ValueError("signal id lineage conflict")

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

    def remember_signal_identity(
        self,
        snapshot: EvidenceMemorySnapshot,
        signal: ExternalSignal,
        *,
        canonical_correlation_group: str | None = None,
    ) -> EvidenceMemorySnapshot:
        _require_supported_memory_version(snapshot)
        provenance = _required_provenance(signal)
        prior_identities = {
            signal_id: parts
            for signal_id, value in snapshot.source_content_fingerprints.items()
            if (parts := _source_content_identity_parts(value)) is not None
        }
        canonical_group = canonical_correlation_group or _canonical_correlation_group(
            snapshot=snapshot,
            provenance=provenance,
            prior_identities=prior_identities,
        )
        self.validate_signal_lineage(
            snapshot,
            signal,
            canonical_correlation_group=canonical_group,
        )

        content_fingerprints = dict(snapshot.content_fingerprints)
        source_content_fingerprints = {
            signal_id: _source_content_identity_from_parts(parts)
            for signal_id, value in snapshot.source_content_fingerprints.items()
            if (parts := _source_content_identity_parts(value)) is not None
        }
        derivation_roots = dict(snapshot.derivation_roots)
        content_fingerprints[signal.id] = provenance.canonical_content_fingerprint
        source_content_fingerprints[signal.id] = _source_content_identity(
            provenance,
            correlation_group=canonical_group,
        )
        derivation_roots[signal.id] = provenance.derivation_root_id

        if (
            snapshot.memory_version == _CURRENT_MEMORY_VERSION
            and content_fingerprints == snapshot.content_fingerprints
            and source_content_fingerprints == snapshot.source_content_fingerprints
            and derivation_roots == snapshot.derivation_roots
        ):
            return snapshot

        return EvidenceMemorySnapshot(
            memory_version=_CURRENT_MEMORY_VERSION,
            accepted_evidence_ids=list(snapshot.accepted_evidence_ids),
            content_fingerprints=content_fingerprints,
            source_content_fingerprints=source_content_fingerprints,
            derivation_roots=derivation_roots,
            event_signal_identity_digests=dict(
                snapshot.event_signal_identity_digests
            ),
            correlation_credit=dict(snapshot.correlation_credit),
            discovery_evidence_ids=list(snapshot.discovery_evidence_ids),
            counterevidence_ids_by_hypothesis={
                hypothesis_id: list(evidence_ids)
                for hypothesis_id, evidence_ids in snapshot.counterevidence_ids_by_hypothesis.items()
            },
            discard_and_schema_history=list(snapshot.discard_and_schema_history),
        )

    def validate_event_signal_identity(
        self,
        snapshot: EvidenceMemorySnapshot,
        *,
        event_id: str,
        signal: ExternalSignal,
        require_existing: bool,
    ) -> str:
        _require_supported_memory_version(snapshot)
        digest = canonical_signal_identity_digest(signal)
        existing = snapshot.event_signal_identity_digests.get(event_id)
        if existing is None:
            if require_existing:
                raise ValueError(
                    "evidence event signal identity binding is missing"
                )
            return digest
        if existing != digest:
            raise ValueError("evidence event signal identity conflict")
        return digest

    def validate_transition(
        self,
        prior_snapshot: EvidenceMemorySnapshot,
        next_snapshot: EvidenceMemorySnapshot,
        *,
        evidence_events: list[EvidenceEvent],
        normalized_signals: list[ExternalSignal],
        existing_evidence_ids: list[str],
        frame_version: int,
    ) -> EvidenceMemorySnapshot:
        """Validate one native memory commit as an append-only transaction."""
        try:
            prior = EvidenceMemorySnapshot.model_validate(
                prior_snapshot.model_dump(mode="python")
            )
            candidate = EvidenceMemorySnapshot.model_validate(
                next_snapshot.model_dump(mode="python")
            )
            _require_supported_memory_version(prior)
            _require_supported_memory_version(candidate)
            self._validate_transition(
                prior,
                candidate,
                evidence_events=evidence_events,
                normalized_signals=normalized_signals,
                existing_evidence_ids=existing_evidence_ids,
                frame_version=frame_version,
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            raise ValueError("evidence memory transition is invalid") from None
        return candidate

    def _validate_transition(
        self,
        prior: EvidenceMemorySnapshot,
        candidate: EvidenceMemorySnapshot,
        *,
        evidence_events: list[EvidenceEvent],
        normalized_signals: list[ExternalSignal],
        existing_evidence_ids: list[str],
        frame_version: int,
    ) -> None:
        ordered_events = _unique_transition_events(evidence_events)
        signal_event_groups = _transition_signal_event_groups(
            normalized_signals,
            ordered_events,
        )
        existing_ids = set(existing_evidence_ids)

        for signal, events in signal_event_groups:
            provenance = _required_provenance(signal)
            for event in events:
                if (
                    event.content != signal.raw_content
                    or event.epistemic_origin != provenance.epistemic_origin
                    or event.derivation_root_id != provenance.derivation_root_id
                ):
                    raise ValueError("evidence memory transition is invalid")
                if event.id in existing_ids:
                    self.validate_event_signal_identity(
                        prior,
                        event_id=event.id,
                        signal=signal,
                        require_existing=True,
                    )

        identity_shadow = prior
        for signal in normalized_signals:
            identity_shadow = self.remember_signal_identity(
                identity_shadow,
                signal,
            )
        for signal in normalized_signals:
            self.validate_signal_lineage(identity_shadow, signal)

        expected = prior
        quality_assessor = SignalQualityAssessor()
        seen_signatures: set[tuple[str, str]] = set()
        for signal, events in signal_event_groups:
            is_cycle_duplicate = observe_cycle_signal_duplicate(
                signal,
                seen_signatures,
            )
            classification_snapshot = expected
            for event in events:
                if event.id in existing_ids:
                    expected = self.remember_signal_identity(
                        expected,
                        signal,
                    )
                    continue

                base_weight = (
                    event.reliability
                    * event.independence
                    * event.relevance
                    * event.novelty
                )
                decision = self.classify(
                    classification_snapshot,
                    signal,
                    likelihoods=event.likelihoods,
                    unresolved_likelihood=event.unresolved_likelihood,
                    frame_version=frame_version,
                    base_effective_weight=base_weight,
                )
                quality_cap = quality_assessor.assess(
                    signal=signal,
                    event_type=event.evidence_type,
                    is_duplicate=(
                        is_cycle_duplicate
                        or decision.correlation_status == "duplicate_exact"
                    ),
                )
                if any(
                    getattr(event, metric) > getattr(quality_cap, metric)
                    for metric in SIGNAL_QUALITY_METRICS
                ):
                    raise ValueError("evidence memory transition is invalid")
                if (
                    decision.correlation_status == "correlated_restatement"
                    and (
                        event.independence != 0.0
                        or event.novelty > 0.25
                    )
                ):
                    raise ValueError("evidence memory transition is invalid")
                if (
                    event.correlation_status != decision.correlation_status
                    or event.effective_update_weight is None
                    or not math.isclose(
                        event.effective_update_weight,
                        decision.effective_update_weight,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                    or not _matches_required_discard_semantics(
                        event.discard_reason,
                        decision.discard_reason,
                    )
                ):
                    raise ValueError("evidence memory transition is invalid")
                expected = self.commit(
                    expected,
                    signal=signal,
                    event=event,
                    decision=decision,
                )

        if candidate != expected:
            raise ValueError("evidence memory transition is invalid")

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
        _require_supported_memory_version(snapshot)
        provenance = _required_provenance(signal)
        prior_identities = {
            signal_id: parts
            for signal_id, value in snapshot.source_content_fingerprints.items()
            if (parts := _source_content_identity_parts(value)) is not None
        }
        canonical_group = _canonical_correlation_group(
            snapshot=snapshot,
            provenance=provenance,
            prior_identities=prior_identities,
        )
        self.validate_signal_lineage(
            snapshot,
            signal,
            canonical_correlation_group=canonical_group,
        )
        exact = any(
            prior_identity[0] == provenance.source_identity
            and prior_identity[1] == provenance.canonical_content_fingerprint
            and snapshot.derivation_roots.get(signal_id)
            == provenance.derivation_root_id
            for signal_id, prior_identity in prior_identities.items()
        )
        same_root = provenance.derivation_root_id in snapshot.derivation_roots.values()
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
        elif likelihoods is None and unresolved_likelihood is None:
            remaining_before = {
                key: max(cap - used, 0.0)
                for key, used in snapshot.correlation_credit.items()
                if key.startswith(group_prefix)
            }
        else:
            remaining_before = {}
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
        event_known = _event_id_in_lifecycle(snapshot, event.id)
        signal_identity_digest = self.validate_event_signal_identity(
            snapshot,
            event_id=event.id,
            signal=signal,
            require_existing=event_known,
        )
        snapshot = self.remember_signal_identity(
            snapshot,
            signal,
            canonical_correlation_group=decision.canonical_correlation_group,
        )

        if event_known:
            return snapshot

        accepted_ids = list(snapshot.accepted_evidence_ids)
        discard_history = list(snapshot.discard_and_schema_history)
        if event.discard_reason is None:
            accepted_ids.append(event.id)
        else:
            discard_history.append(
                encode_discard_history_entry(event.id, event.discard_reason)
            )
        event_signal_identity_digests = dict(
            snapshot.event_signal_identity_digests
        )
        event_signal_identity_digests[event.id] = signal_identity_digest

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
            content_fingerprints=dict(snapshot.content_fingerprints),
            source_content_fingerprints=dict(snapshot.source_content_fingerprints),
            derivation_roots=dict(snapshot.derivation_roots),
            event_signal_identity_digests=event_signal_identity_digests,
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


def _has_low_reliability_cue(content: str) -> bool:
    content_lower = content.lower()
    return any(
        cue in content_lower
        for cue in ("rumor", "unverified", "hearsay", "maybe", "unclear")
    )


def _clean_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def _model_provider_audit_identity(
    signal: ExternalSignal,
    supplied_identity: str | None,
) -> str:
    if supplied_identity is not None:
        _reject_secret_hash_text(supplied_identity)
        return supplied_identity
    _reject_secret_hash_text(signal.source_type)
    _reject_secret_hash_text(signal.source)
    encoded_source = json.dumps(
        {
            "source": signal.source,
            "source_type": signal.source_type,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    fallback_identity = f"{_MODEL_PROVIDER_FALLBACK_PREFIX}{encoded_source}"
    _reject_secret_hash_text(fallback_identity)
    return fallback_identity


def _model_source_identity(provider_identity: str) -> str:
    _reject_secret_hash_text(provider_identity)
    digest = _exact_identity_digest({"provider_identity": provider_identity})
    return f"model-source:sha256:{digest}"


def _model_correlation_group(provider_identity: str, session_id: str) -> str:
    _reject_secret_hash_text(provider_identity)
    _reject_secret_hash_text(session_id)
    digest = _exact_identity_digest(
        {
            "provider_identity": provider_identity,
            "session_id": session_id,
        }
    )
    return f"model:sha256:{digest}"


def _exact_identity_digest(components: Mapping[str, str]) -> str:
    encoded = json.dumps(
        components,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _canonical_computation_text(value: str) -> str:
    return _clean_text(value)


def _reject_secret_computation_text(value: str) -> None:
    normalized = _canonical_computation_text(value)
    if is_secret_like_value(value) or is_secret_like_value(normalized):
        raise ValueError("deterministic computation inputs contain secret material")


def _reject_secret_hash_text(value: str) -> None:
    if is_secret_like_value(value) or is_secret_like_value(_clean_text(value)):
        raise ValueError("external signal contains secret material")


def _canonical_computation_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        canonical: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(
                    "deterministic computation inputs require string object keys"
                )
            if not key.strip():
                raise ValueError(
                    "deterministic computation inputs require non-empty object keys"
                )
            normalized_key = _canonical_computation_text(key)
            if (
                is_forbidden_secret_key_name(key)
                or is_forbidden_secret_key_name(normalized_key)
                or is_secret_like_value(key)
                or is_secret_like_value(normalized_key)
            ):
                raise ValueError(
                    "deterministic computation inputs contain secret material"
                )
            canonical[key] = _canonical_computation_value(item)
        return canonical
    if isinstance(value, list | tuple):
        return [_canonical_computation_value(item) for item in value]
    if isinstance(value, str):
        _reject_secret_computation_text(value)
        return value
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                "deterministic computation inputs require finite numbers"
            )
        return value
    raise ValueError("deterministic computation inputs must be canonical JSON values")


def _sha256_identity(source_identity: str, content: str) -> str:
    digest = hashlib.sha256(f"{source_identity}\n{content}".encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _reject_secret_signal(signal: ExternalSignal) -> None:
    if contains_secret_material(signal.model_dump(mode="python")):
        raise ValueError("external signal contains secret material")


def _reject_secret_provenance(provenance: SignalProvenance) -> None:
    if contains_secret_material(provenance.model_dump(mode="python")):
        raise ValueError("signal provenance contains secret material")


def _required_provenance(signal: ExternalSignal) -> SignalProvenance:
    if signal.provenance is None:
        raise ValueError("evidence memory requires normalized signal provenance")
    return signal.provenance


def _require_supported_memory_version(snapshot: EvidenceMemorySnapshot) -> None:
    if (
        type(snapshot.memory_version) is not int
        or snapshot.memory_version not in _SUPPORTED_MEMORY_VERSIONS
    ):
        raise ValueError("unsupported evidence memory version")


def _event_id_in_lifecycle(
    snapshot: EvidenceMemorySnapshot,
    event_id: str,
) -> bool:
    return event_id in snapshot.accepted_evidence_ids or any(
        decode_discard_history_entry(entry)[0] == event_id
        for entry in snapshot.discard_and_schema_history
    )


def _supplied_correlation_group(provenance: SignalProvenance) -> str:
    return provenance.supplied_correlation_group or provenance.correlation_group


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
            _supplied_correlation_group(provenance),
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _source_content_identity_parts(
    value: str,
) -> tuple[str, str, str, str] | None:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None
    if (
        not isinstance(parsed, list)
        or len(parsed) not in {3, 4}
        or not all(isinstance(item, str) for item in parsed)
    ):
        return None
    supplied_group = parsed[3] if len(parsed) == 4 else parsed[2]
    return parsed[0], parsed[1], parsed[2], supplied_group


def _source_content_identity_from_parts(
    parts: tuple[str, str, str, str],
) -> str:
    return json.dumps(parts, ensure_ascii=False, separators=(",", ":"))


def _canonical_correlation_group(
    *,
    snapshot: EvidenceMemorySnapshot,
    provenance: SignalProvenance,
    prior_identities: dict[str, tuple[str, str, str, str]],
) -> str:
    known_groups = {
        prior_identities[signal_id][2]
        for signal_id, root in snapshot.derivation_roots.items()
        if root == provenance.derivation_root_id and signal_id in prior_identities
    }
    known_groups.update(
        prior_identities[parent_id][2]
        for parent_id in provenance.parent_signal_ids
        if parent_id in prior_identities
    )
    known_groups.update(
        prior_identity[2]
        for prior_identity in prior_identities.values()
        if prior_identity[0] == provenance.source_identity
    )
    if len(known_groups) > 1:
        raise ValueError("signal lineage has conflicting canonical correlation groups")
    if known_groups:
        return known_groups.pop()
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


def _unique_transition_events(
    evidence_events: list[EvidenceEvent],
) -> list[EvidenceEvent]:
    events_by_id: dict[str, EvidenceEvent] = {}
    ordered_events: list[EvidenceEvent] = []
    for event in evidence_events:
        prior_event = events_by_id.get(event.id)
        if prior_event is not None:
            if prior_event != event:
                raise ValueError("evidence memory transition is invalid")
            continue
        events_by_id[event.id] = event
        ordered_events.append(event)
    return ordered_events


def _transition_signal_event_groups(
    normalized_signals: list[ExternalSignal],
    ordered_events: list[EvidenceEvent],
) -> list[tuple[ExternalSignal, list[EvidenceEvent]]]:
    groups: list[tuple[ExternalSignal, list[EvidenceEvent]]] = []
    event_index = 0
    for signal in normalized_signals:
        if (
            event_index >= len(ordered_events)
            or ordered_events[event_index].derived_from_signal != signal.id
        ):
            raise ValueError("evidence memory transition is invalid")
        primary = ordered_events[event_index]
        events = [primary]
        event_index += 1
        if (
            event_index < len(ordered_events)
            and ordered_events[event_index].derived_from_signal == signal.id
            and ordered_events[event_index].id == f"{primary.id}_source"
        ):
            events.append(ordered_events[event_index])
            event_index += 1
        groups.append((signal, events))
    if event_index != len(ordered_events):
        raise ValueError("evidence memory transition is invalid")
    return groups


def _matches_required_discard_semantics(
    event_reason: str | None,
    decision_reason: str | None,
) -> bool:
    policy_reasons = {"duplicate_exact", "correlation_credit_saturated"}
    if decision_reason is not None:
        return event_reason == decision_reason
    if event_reason in policy_reasons:
        return False
    return True


__all__ = [
    "EvidenceMemoryDecision",
    "EvidenceMemoryManager",
    "SIGNAL_QUALITY_METRICS",
    "SignalQuality",
    "SignalQualityAssessor",
    "SignalProvenanceNormalizer",
    "cycle_signal_source_content_signature",
    "derive_deterministic_computation_root",
    "observe_cycle_signal_duplicate",
]
