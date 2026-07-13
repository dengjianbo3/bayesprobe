from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math

from bayesprobe.schemas import (
    EpistemicOrigin,
    EpistemicProgress,
    EvidenceContributionDelta,
    EvidenceContributionMode,
    EvidenceEvent,
    EvidenceMemorySnapshot,
    EvidenceRootContribution,
    ExternalSignal,
    LikelihoodBand,
)


LIKELIHOOD_RATIO_BY_BAND: dict[LikelihoodBand, float] = {
    LikelihoodBand.STRONGLY_DISCONFIRMING: 0.1,
    LikelihoodBand.MODERATELY_DISCONFIRMING: 0.3,
    LikelihoodBand.WEAKLY_DISCONFIRMING: 0.7,
    LikelihoodBand.NEUTRAL: 1.0,
    LikelihoodBand.WEAKLY_CONFIRMING: 1.5,
    LikelihoodBand.MODERATELY_CONFIRMING: 3.0,
    LikelihoodBand.STRONGLY_CONFIRMING: 10.0,
}

_VECTOR_ABS_TOLERANCE = 1e-12
_CORRELATION_ROOTED_ORIGINS = frozenset(
    {
        EpistemicOrigin.MODEL_REASONING,
        EpistemicOrigin.RETRIEVED_SOURCE,
        EpistemicOrigin.HUMAN_INPUT,
        EpistemicOrigin.AGENT_MESSAGE,
        EpistemicOrigin.DERIVED_SUMMARY,
    }
)


@dataclass(frozen=True)
class RootReconciliationResult:
    evidence_events: list[EvidenceEvent]
    contribution_deltas: list[EvidenceContributionDelta]
    evidence_memory: EvidenceMemorySnapshot
    epistemic_progress: EpistemicProgress


def resolve_contribution_root_id(
    signal: ExternalSignal,
    parent_contribution_roots: Mapping[str, str] | None = None,
    *,
    canonical_correlation_group: str | None = None,
) -> str:
    provenance = signal.provenance
    if provenance is None:
        raise ValueError(
            "contribution root resolution requires normalized provenance"
        )
    if provenance.parent_signal_ids:
        if parent_contribution_roots is None:
            raise ValueError(
                "child contribution root resolution requires parent contribution roots"
            )
        resolved_parent_roots: list[str] = []
        for parent_id in provenance.parent_signal_ids:
            if parent_id not in parent_contribution_roots:
                raise ValueError(
                    "child contribution root resolution is missing a parent "
                    "contribution root"
                )
            parent_root = parent_contribution_roots[parent_id]
            if (
                not isinstance(parent_root, str)
                or not parent_root
                or parent_root.strip() != parent_root
            ):
                raise ValueError("parent contribution roots must be canonical text")
            resolved_parent_roots.append(parent_root)
        unique_parent_roots = set(resolved_parent_roots)
        if len(unique_parent_roots) != 1:
            raise ValueError(
                "child contribution root resolution requires exactly one parent "
                "contribution root"
            )
        return unique_parent_roots.pop()

    if provenance.epistemic_origin is EpistemicOrigin.DERIVED_SUMMARY:
        raise ValueError("derived summary requires parent signals")
    if provenance.epistemic_origin in _CORRELATION_ROOTED_ORIGINS:
        basis_kind = "correlation_group"
        basis = (
            provenance.correlation_group
            if canonical_correlation_group is None
            else canonical_correlation_group
        )
        if (
            not isinstance(basis, str)
            or not basis
            or basis.strip() != basis
        ):
            raise ValueError("canonical correlation group must be canonical text")
    else:
        basis_kind = "derivation_root_id"
        basis = provenance.derivation_root_id
    encoded = json.dumps(
        {"basis": basis, "basis_kind": basis_kind},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return f"evidence-root:sha256:{digest}"


def _close(left: float, right: float) -> bool:
    return math.isclose(
        left,
        right,
        rel_tol=0.0,
        abs_tol=_VECTOR_ABS_TOLERANCE,
    )


def _vectors_equal(
    previous: EvidenceRootContribution,
    current: EvidenceRootContribution,
) -> bool:
    hypothesis_ids = set(previous.per_hypothesis_log_likelihood).union(
        current.per_hypothesis_log_likelihood
    )
    return all(
        _close(
            previous.per_hypothesis_log_likelihood.get(hypothesis_id, 0.0),
            current.per_hypothesis_log_likelihood.get(hypothesis_id, 0.0),
        )
        for hypothesis_id in hypothesis_ids
    ) and _close(
        previous.unresolved_log_likelihood or 0.0,
        current.unresolved_log_likelihood or 0.0,
    )


def _candidate_contribution(
    root_id: str,
    events: list[EvidenceEvent],
    previous: EvidenceRootContribution | None,
) -> EvidenceRootContribution:
    hypothesis_ids = sorted(
        {
            hypothesis_id
            for event in events
            for hypothesis_id in event.likelihoods
        }
    )
    event_count = len(events)
    per_hypothesis = {
        hypothesis_id: sum(
            _event_quality(event)
            * math.log(
                LIKELIHOOD_RATIO_BY_BAND[
                    event.likelihoods.get(
                        hypothesis_id,
                        LikelihoodBand.NEUTRAL,
                    )
                ]
            )
            for event in events
        )
        / event_count
        for hypothesis_id in hypothesis_ids
    }
    has_unresolved_coordinate = any(
        event.unresolved_likelihood is not None for event in events
    )
    unresolved = (
        sum(
            _event_quality(event)
            * math.log(
                LIKELIHOOD_RATIO_BY_BAND[
                    event.unresolved_likelihood or LikelihoodBand.NEUTRAL
                ]
            )
            for event in events
        )
        / event_count
        if has_unresolved_coordinate
        else None
    )
    active = any(not _close(value, 0.0) for value in per_hypothesis.values())
    active = active or not _close(unresolved or 0.0, 0.0)
    return EvidenceRootContribution(
        contribution_root_id=root_id,
        revision=1 if previous is None else previous.revision + 1,
        assessment_event_ids=[event.id for event in events],
        epistemic_origin=events[0].epistemic_origin,
        per_hypothesis_log_likelihood=per_hypothesis,
        unresolved_log_likelihood=unresolved,
        active=active,
    )


def _event_quality(event: EvidenceEvent) -> float:
    return (
        event.reliability
        * event.independence
        * event.relevance
        * event.novelty
    )


def _mode_for(
    previous: EvidenceRootContribution | None,
    current: EvidenceRootContribution,
) -> EvidenceContributionMode:
    if previous is None:
        return EvidenceContributionMode.NEW_ROOT
    if _vectors_equal(previous, current):
        return EvidenceContributionMode.NO_CHANGE
    if previous.active and not current.active:
        return EvidenceContributionMode.RETRACT_ROOT
    return EvidenceContributionMode.REVISE_ROOT


def _delta_for(
    previous: EvidenceRootContribution | None,
    current: EvidenceRootContribution,
    events: list[EvidenceEvent],
) -> EvidenceContributionDelta:
    previous_values = (
        {} if previous is None else previous.per_hypothesis_log_likelihood
    )
    hypothesis_ids = sorted(
        set(previous_values).union(current.per_hypothesis_log_likelihood)
    )
    unresolved_is_present = (
        current.unresolved_log_likelihood is not None
        or (
            previous is not None
            and previous.unresolved_log_likelihood is not None
        )
    )
    return EvidenceContributionDelta(
        contribution_root_id=current.contribution_root_id,
        mode=_mode_for(previous, current),
        previous_contribution=previous,
        current_contribution=current,
        per_hypothesis_delta={
            hypothesis_id: (
                current.per_hypothesis_log_likelihood.get(hypothesis_id, 0.0)
                - previous_values.get(hypothesis_id, 0.0)
            )
            for hypothesis_id in hypothesis_ids
        },
        unresolved_delta=(
            (current.unresolved_log_likelihood or 0.0)
            - (
                0.0
                if previous is None
                else previous.unresolved_log_likelihood or 0.0
            )
            if unresolved_is_present
            else None
        ),
        caused_by_event_ids=[event.id for event in events],
    )


def _epistemic_progress(
    deltas: list[EvidenceContributionDelta],
    *,
    falsification_probe_executed: bool,
) -> EpistemicProgress:
    counts = {mode: 0 for mode in EvidenceContributionMode}
    maximum_delta = 0.0
    for delta in deltas:
        counts[delta.mode] += 1
        maximum_delta = max(
            maximum_delta,
            *(abs(value) for value in delta.per_hypothesis_delta.values()),
            abs(delta.unresolved_delta or 0.0),
        )
    return EpistemicProgress(
        new_root_count=counts[EvidenceContributionMode.NEW_ROOT],
        revised_root_count=counts[EvidenceContributionMode.REVISE_ROOT],
        retracted_root_count=counts[EvidenceContributionMode.RETRACT_ROOT],
        no_change_count=counts[EvidenceContributionMode.NO_CHANGE],
        max_absolute_contribution_delta=maximum_delta,
        falsification_probe_executed=falsification_probe_executed,
    )


class EvidenceRootReconciler:
    def reconcile_cycle(
        self,
        snapshot: EvidenceMemorySnapshot,
        evidence_events: list[EvidenceEvent],
        falsification_probe_executed: bool,
    ) -> RootReconciliationResult:
        if snapshot.memory_version != 3:
            raise ValueError("evidence root reconciliation requires memory version 3")

        ordered_events = sorted(evidence_events, key=lambda event: event.id)
        events_by_root: dict[str, list[EvidenceEvent]] = defaultdict(list)
        seen_event_ids: set[str] = set()
        for event in ordered_events:
            if event.id in seen_event_ids:
                raise ValueError(
                    "evidence root reconciliation requires unique event ids"
                )
            seen_event_ids.add(event.id)
            if (
                event.schema_version != "v0.2"
                or event.contribution_root_id is None
                or event.epistemic_origin is None
                or event.effective_update_weight is not None
            ):
                raise ValueError(
                    "evidence root reconciliation requires root-bound v0.2 events"
                )
            if event.discard_reason is None:
                events_by_root[event.contribution_root_id].append(event)

        root_contributions = dict(snapshot.root_contributions)
        deltas: list[EvidenceContributionDelta] = []
        for root_id in sorted(events_by_root):
            root_events = events_by_root[root_id]
            previous = root_contributions.get(root_id)
            current = _candidate_contribution(root_id, root_events, previous)
            if previous is None and not current.active:
                continue
            delta = _delta_for(previous, current, root_events)
            root_contributions[root_id] = current
            deltas.append(delta)

        memory_payload = snapshot.model_dump(mode="python")
        memory_payload["root_contributions"] = {
            root_id: root_contributions[root_id]
            for root_id in sorted(root_contributions)
        }
        evidence_memory = EvidenceMemorySnapshot.model_validate(memory_payload)
        return RootReconciliationResult(
            evidence_events=ordered_events,
            contribution_deltas=deltas,
            evidence_memory=evidence_memory,
            epistemic_progress=_epistemic_progress(
                deltas,
                falsification_probe_executed=falsification_probe_executed,
            ),
        )
