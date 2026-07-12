from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from bayesprobe.kernel_config import FrameAdequacyPolicyConfig
from bayesprobe.schemas import (
    EpistemicOrigin,
    EvidenceEvent,
    FrameAdequacyStatus,
    FrameFit,
    FrameState,
    Hypothesis,
    HypothesisCoverage,
    HypothesisStatus,
    LikelihoodBand,
)


_MODERATE_OR_STRONG_CONFIRMING = {
    LikelihoodBand.MODERATELY_CONFIRMING,
    LikelihoodBand.STRONGLY_CONFIRMING,
}
_DISCONFIRMING = {
    LikelihoodBand.WEAKLY_DISCONFIRMING,
    LikelihoodBand.MODERATELY_DISCONFIRMING,
    LikelihoodBand.STRONGLY_DISCONFIRMING,
}
_INACTIVE_STATUSES = {
    HypothesisStatus.RETIRED,
    HypothesisStatus.ARCHIVED,
}


@dataclass(frozen=True)
class FrameAdequacyDecision:
    frame_state: FrameState
    should_expand: bool
    trigger_event_ids: list[str]
    reason: str


class FrameAdequacyPolicy:
    def __init__(
        self,
        *,
        config: FrameAdequacyPolicyConfig | None = None,
    ) -> None:
        self.config = config or FrameAdequacyPolicyConfig()

    def assess(
        self,
        *,
        previous: FrameState,
        events: list[EvidenceEvent],
        hypotheses: list[Hypothesis],
    ) -> FrameAdequacyDecision:
        accepted = [event for event in events if event.discard_reason is None]
        supports_unresolved = [
            event
            for event in accepted
            if event.frame_fit == FrameFit.SUPPORTS_UNRESOLVED
        ]
        trigger_event_ids = _unique_ids(event.id for event in supports_unresolved)

        if previous.coverage == HypothesisCoverage.EXHAUSTIVE:
            return self._decision(
                previous,
                status=FrameAdequacyStatus.ADEQUATE,
                should_expand=False,
                trigger_event_ids=[],
                reason="The declared exhaustive frame remains adequate.",
            )

        if previous.adequacy_status in {
            FrameAdequacyStatus.INADEQUATE,
            FrameAdequacyStatus.EXPANDING,
        }:
            return self._decision(
                previous,
                status=FrameAdequacyStatus.EXPANDING,
                should_expand=True,
                trigger_event_ids=_unique_ids(
                    [*previous.trigger_event_ids, *trigger_event_ids]
                ),
                reason="The inadequate open frame requires hypothesis-space expansion.",
            )

        high_event = next(
            (
                event
                for event in supports_unresolved
                if event.unresolved_likelihood
                == LikelihoodBand.STRONGLY_CONFIRMING
                and event.verifiability
                >= self.config.high_verifiability_threshold
                and _is_non_model_origin(event)
            ),
            None,
        )
        moderate_events = [
            event
            for event in supports_unresolved
            if event.unresolved_likelihood in _MODERATE_OR_STRONG_CONFIRMING
            and event.verifiability
            >= self.config.moderate_verifiability_threshold
            and _derivation_root(event) is not None
        ]
        moderate_roots = {
            root
            for event in moderate_events
            if (root := _derivation_root(event)) is not None
        }
        roots_have_external_support = any(
            _is_non_model_origin(event) for event in moderate_events
        )
        if high_event is not None or (
            len(moderate_roots)
            >= self.config.required_distinct_moderate_roots
            and roots_have_external_support
        ):
            qualifying_trigger_event_ids = (
                [high_event.id]
                if high_event is not None
                else _unique_ids(event.id for event in moderate_events)
            )
            reason = (
                "A strongly confirming, externally verifiable event supports an "
                "unresolved alternative."
                if high_event is not None
                else "Distinct derivation roots support an unresolved alternative."
            )
            return self._decision(
                previous,
                status=FrameAdequacyStatus.INADEQUATE,
                should_expand=True,
                trigger_event_ids=qualifying_trigger_event_ids,
                reason=reason,
            )

        active = [
            hypothesis
            for hypothesis in hypotheses
            if hypothesis.status not in _INACTIVE_STATUSES
        ]
        if not active and not previous.active_hypothesis_ids:
            named_ids = {hypothesis.id for hypothesis in hypotheses}
            retirement_trigger_event_ids = _unique_ids(
                event.id
                for event in accepted
                if any(
                    hypothesis_id in named_ids and band in _DISCONFIRMING
                    for hypothesis_id, band in event.likelihoods.items()
                )
            )
            return self._decision(
                previous,
                status=FrameAdequacyStatus.CHALLENGED,
                should_expand=True,
                trigger_event_ids=(
                    retirement_trigger_event_ids
                    if retirement_trigger_event_ids
                    else list(previous.trigger_event_ids)
                ),
                reason=(
                    "All named hypotheses are retired; unresolved alternatives "
                    "hold all frame mass."
                ),
            )
        unresolved_dominates = (
            previous.unresolved_alternative_mass is not None
            and bool(active)
            and previous.unresolved_alternative_mass
            > max(hypothesis.posterior for hypothesis in active)
        )
        all_named_disconfirmed = _all_named_disconfirmed(accepted, active)
        challenge_trigger_event_ids = trigger_event_ids
        if all_named_disconfirmed and not challenge_trigger_event_ids:
            active_ids = {hypothesis.id for hypothesis in active}
            challenge_trigger_event_ids = _unique_ids(
                event.id
                for event in accepted
                if any(
                    band in _DISCONFIRMING
                    for hypothesis_id, band in event.likelihoods.items()
                    if hypothesis_id in active_ids
                )
            )
        challenged = (
            bool(supports_unresolved)
            or unresolved_dominates
            or all_named_disconfirmed
        )
        if challenged or previous.adequacy_status == FrameAdequacyStatus.CHALLENGED:
            if supports_unresolved:
                reason = "Accepted evidence supports an unresolved alternative."
            elif unresolved_dominates:
                reason = "Unresolved alternative mass exceeds every named candidate."
            elif all_named_disconfirmed:
                reason = "Accepted evidence disconfirms every named candidate."
            else:
                reason = previous.revision_reason or "The open frame remains challenged."
            return self._decision(
                previous,
                status=FrameAdequacyStatus.CHALLENGED,
                should_expand=True,
                trigger_event_ids=(
                    challenge_trigger_event_ids
                    if challenge_trigger_event_ids
                    else list(previous.trigger_event_ids)
                ),
                reason=reason,
            )

        return self._decision(
            previous,
            status=previous.adequacy_status,
            should_expand=False,
            trigger_event_ids=list(previous.trigger_event_ids),
            reason="No accepted event challenges the open frame.",
        )

    @staticmethod
    def _decision(
        previous: FrameState,
        *,
        status: FrameAdequacyStatus,
        should_expand: bool,
        trigger_event_ids: list[str],
        reason: str,
    ) -> FrameAdequacyDecision:
        return FrameAdequacyDecision(
            frame_state=previous.model_copy(
                update={
                    "adequacy_status": status,
                    "revision_reason": reason,
                    "trigger_event_ids": trigger_event_ids,
                }
            ),
            should_expand=should_expand,
            trigger_event_ids=trigger_event_ids,
            reason=reason,
        )


def _all_named_disconfirmed(
    events: list[EvidenceEvent],
    hypotheses: list[Hypothesis],
) -> bool:
    if not hypotheses:
        return False
    return all(
        any(event.likelihoods.get(hypothesis.id) in _DISCONFIRMING for event in events)
        for hypothesis in hypotheses
    )


def _event_origin(event: EvidenceEvent) -> EpistemicOrigin | None:
    return event.epistemic_origin


def _is_non_model_origin(event: EvidenceEvent) -> bool:
    origin = _event_origin(event)
    return origin is not None and origin != EpistemicOrigin.MODEL_REASONING


def _derivation_root(event: EvidenceEvent) -> str | None:
    return event.derivation_root_id


def _unique_ids(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


__all__ = ["FrameAdequacyDecision", "FrameAdequacyPolicy"]
