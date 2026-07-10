from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from bayesprobe.model_gateway import (
    DeterministicModelGateway,
    EvidenceJudgment,
    EvidenceJudgmentRepairPolicy,
    ModelGateway,
    ModelGatewayValidationError,
    ModelInvocationTrace,
    StructuredModelRequest,
    evidence_judgment_from_mapping,
    model_gateway_adapter_kind,
)
from bayesprobe.schemas import (
    BeliefState,
    CycleRecord,
    EvidenceEvent,
    EvidenceType,
    ExternalSignal,
    Hypothesis,
    LikelihoodBand,
    ProbeCandidate,
    ProbeDesign,
    ProbeSet,
)


@dataclass(frozen=True)
class EvidenceIntegrationResult:
    evidence_events: list[EvidenceEvent]
    probe_candidates: list[ProbeCandidate]


@dataclass(frozen=True)
class SignalQuality:
    reliability: float
    independence: float
    relevance: float
    novelty: float
    specificity: float
    verifiability: float


class _EvidenceJudgmentFailure(Exception):
    def __init__(
        self,
        *,
        error: ModelGatewayValidationError,
        model_trace: ModelInvocationTrace,
    ) -> None:
        super().__init__(str(error))
        self.error = error
        self.model_trace = model_trace


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

        return quality


class ProjectionDecomposer:
    _SOURCE_CUES = (
        "because",
        "source",
        "cites",
        "according to",
        "passage",
        "paper",
        "evidence",
    )

    def should_decompose(self, signal: ExternalSignal) -> bool:
        if signal.source_type != "external_agent_projection":
            return False
        content_lower = signal.raw_content.lower()
        return any(cue in content_lower for cue in self._SOURCE_CUES)


class EvidenceIntegrationGate:
    def __init__(
        self,
        *,
        quality_assessor: SignalQualityAssessor | None = None,
        projection_decomposer: ProjectionDecomposer | None = None,
        model_gateway: ModelGateway | None = None,
        judgment_repair_policy: EvidenceJudgmentRepairPolicy | None = None,
    ) -> None:
        self._quality_assessor = quality_assessor or SignalQualityAssessor()
        self._projection_decomposer = projection_decomposer or ProjectionDecomposer()
        self._model_gateway = model_gateway or DeterministicModelGateway()
        self._judgment_repair_policy = judgment_repair_policy or EvidenceJudgmentRepairPolicy()

    def integrate(
        self,
        *,
        cycle: CycleRecord,
        belief_state: BeliefState,
        probe_set: ProbeSet,
        signals: list[ExternalSignal],
    ) -> EvidenceIntegrationResult:
        self._ensure_helpers()
        evidence_events: list[EvidenceEvent] = []
        probe_candidates: list[ProbeCandidate] = []
        seen_signatures: set[tuple[str, str]] = set()

        for index, signal in enumerate(signals, start=1):
            is_duplicate = _is_duplicate_signal(signal=signal, seen_signatures=seen_signatures)
            event_result = self._build_signal_events(
                index=index,
                signal=signal,
                belief_state=belief_state,
                probe_set=probe_set,
                cycle=cycle,
                is_duplicate=is_duplicate,
            )
            evidence_events.extend(event_result.evidence_events)
            probe_candidates.extend(event_result.probe_candidates)

        return EvidenceIntegrationResult(
            evidence_events=evidence_events,
            probe_candidates=probe_candidates,
        )

    def _ensure_helpers(self) -> None:
        if not hasattr(self, "_quality_assessor"):
            self._quality_assessor = SignalQualityAssessor()
        if not hasattr(self, "_projection_decomposer"):
            self._projection_decomposer = ProjectionDecomposer()
        if not hasattr(self, "_model_gateway"):
            self._model_gateway = DeterministicModelGateway()
        if not hasattr(self, "_judgment_repair_policy"):
            self._judgment_repair_policy = EvidenceJudgmentRepairPolicy()

    def _model_trace_for_request(self, request: StructuredModelRequest) -> ModelInvocationTrace:
        return ModelInvocationTrace.from_request(
            request,
            adapter_kind=model_gateway_adapter_kind(self._model_gateway),
        )

    def _build_signal_events(
        self,
        *,
        index: int,
        signal: ExternalSignal,
        belief_state: BeliefState,
        probe_set: ProbeSet,
        cycle: CycleRecord,
        is_duplicate: bool,
    ) -> EvidenceIntegrationResult:
        if signal.source_type == "external_agent_projection":
            sender_event = self._build_projection_sender_event(
                index=index,
                signal=signal,
                belief_state=belief_state,
                probe_set=probe_set,
                cycle=cycle,
                is_duplicate=is_duplicate,
            )
            if not self._projection_decomposer.should_decompose(signal):
                return EvidenceIntegrationResult(
                    evidence_events=[sender_event],
                    probe_candidates=[],
                )
            source_event = self._build_source_claim_event(
                index=index,
                signal=signal,
                belief_state=belief_state,
                probe_set=probe_set,
                cycle=cycle,
                is_duplicate=is_duplicate,
            )
            return EvidenceIntegrationResult(
                evidence_events=[sender_event, source_event],
                probe_candidates=[_verification_probe_candidate(cycle=cycle, event=source_event, signal=signal)],
            )

        return EvidenceIntegrationResult(
            evidence_events=[
                self._build_direct_evidence_event(
                    index=index,
                    signal=signal,
                    belief_state=belief_state,
                    probe_set=probe_set,
                    cycle=cycle,
                    is_duplicate=is_duplicate,
                )
            ],
            probe_candidates=[],
        )

    def _build_direct_evidence_event(
        self,
        *,
        index: int,
        signal: ExternalSignal,
        belief_state: BeliefState,
        probe_set: ProbeSet,
        cycle: CycleRecord,
        is_duplicate: bool,
    ) -> EvidenceEvent:
        hypothesis_ids = self._resolve_target_hypotheses(
            signal=signal,
            belief_state=belief_state,
            probe_set=probe_set,
        )
        request = self._build_judge_evidence_request(
            signal=signal,
            hypothesis_ids=hypothesis_ids,
            cycle=cycle,
            probe_set=probe_set,
        )
        try:
            judgment, model_trace = self._evidence_judgment_with_repair(request=request)
        except _EvidenceJudgmentFailure as failure:
            return self._schema_violation_event(
                index=index,
                signal=signal,
                cycle=cycle,
                hypothesis_ids=hypothesis_ids,
                is_duplicate=is_duplicate,
                error=failure.error,
                model_trace=failure.model_trace,
            )

        return self._event(
            event_id=f"{_scoped_cycle_key(cycle.run_id, cycle.cycle_id)}_E{index}",
            signal=signal,
            hypothesis_ids=hypothesis_ids,
            evidence_type=judgment.evidence_type,
            likelihoods=judgment.likelihoods,
            interpretation=judgment.interpretation,
            is_duplicate=is_duplicate,
            quality_overrides=judgment.quality_overrides,
            model_trace=model_trace,
        )

    def _build_judge_evidence_request(
        self,
        *,
        signal: ExternalSignal,
        hypothesis_ids: list[str],
        cycle: CycleRecord,
        probe_set: ProbeSet,
    ) -> StructuredModelRequest:
        return StructuredModelRequest(
            task="judge_evidence",
            input={
                "signal_id": signal.id,
                "source_type": signal.source_type,
                "source": signal.source,
                "raw_content": signal.raw_content,
                "target_hypotheses": hypothesis_ids,
                "cycle_id": cycle.cycle_id,
                "probe_ids": [probe.id for probe in probe_set.probes],
            },
            prompt_id="evidence_judgment",
            prompt_version="v0.1",
            schema_name="EvidenceJudgment",
            schema_version="v0.1",
        )

    def _evidence_judgment_with_repair(
        self,
        *,
        request: StructuredModelRequest,
    ) -> tuple[EvidenceJudgment, ModelInvocationTrace]:
        model_trace = self._model_trace_for_request(request)
        try:
            payload = self._model_gateway.complete_structured(request)
        except ModelGatewayValidationError as error:
            raise _EvidenceJudgmentFailure(error=error, model_trace=model_trace) from error
        try:
            return evidence_judgment_from_mapping(payload), model_trace
        except ModelGatewayValidationError as error:
            if self._judgment_repair_policy.max_attempts == 0:
                raise _EvidenceJudgmentFailure(error=error, model_trace=model_trace) from error
            return self._repair_evidence_judgment(
                original_request=request,
                invalid_payload=payload,
                validation_error=error,
            )

    def _repair_evidence_judgment(
        self,
        *,
        original_request: StructuredModelRequest,
        invalid_payload: Any,
        validation_error: ModelGatewayValidationError,
    ) -> tuple[EvidenceJudgment, ModelInvocationTrace]:
        latest_invalid_payload = _repair_payload_from(invalid_payload)
        latest_error = validation_error
        latest_trace = self._model_trace_for_request(original_request)
        max_attempts = self._judgment_repair_policy.max_attempts

        for attempt_index in range(1, max_attempts + 1):
            repair_request = StructuredModelRequest(
                task=self._judgment_repair_policy.repair_task,
                input={
                    "original_request": {
                        "task": original_request.task,
                        "input": dict(original_request.input),
                    },
                    "invalid_payload": latest_invalid_payload,
                    "validation_error": str(latest_error),
                    "attempt_index": attempt_index,
                    "allowed_evidence_types": [evidence_type.value for evidence_type in EvidenceType],
                    "allowed_likelihood_bands": [band.value for band in LikelihoodBand],
                    "required_fields": [
                        "evidence_type",
                        "likelihoods",
                        "interpretation",
                    ],
                },
                prompt_id="evidence_judgment_repair",
                prompt_version="v0.1",
                schema_name="EvidenceJudgment",
                schema_version="v0.1",
                metadata={"repair_attempt_index": attempt_index},
            )
            repair_trace = self._model_trace_for_request(repair_request)
            try:
                repair_payload = self._model_gateway.complete_structured(repair_request)
            except ModelGatewayValidationError as error:
                latest_error = error
                latest_trace = repair_trace
                continue
            try:
                return evidence_judgment_from_mapping(repair_payload), repair_trace
            except ModelGatewayValidationError as error:
                latest_invalid_payload = _repair_payload_from(repair_payload)
                latest_error = error
                latest_trace = repair_trace

        failure = ModelGatewayValidationError(
            f"repair failed after {max_attempts} attempt(s): {latest_error}"
        )
        raise _EvidenceJudgmentFailure(error=failure, model_trace=latest_trace) from latest_error

    def _schema_violation_event(
        self,
        *,
        index: int,
        signal: ExternalSignal,
        cycle: CycleRecord,
        hypothesis_ids: list[str],
        is_duplicate: bool,
        error: ModelGatewayValidationError,
        model_trace: ModelInvocationTrace | None = None,
    ) -> EvidenceEvent:
        return self._event(
            event_id=f"{_scoped_cycle_key(cycle.run_id, cycle.cycle_id)}_E{index}",
            signal=signal,
            hypothesis_ids=hypothesis_ids,
            evidence_type=EvidenceType.NEUTRAL,
            likelihoods={hypothesis_id: LikelihoodBand.NEUTRAL for hypothesis_id in hypothesis_ids},
            interpretation="Model gateway judgment failed schema validation.",
            is_duplicate=is_duplicate,
            quality_overrides=_ZERO_QUALITY_OVERRIDES,
            discard_reason=f"schema_violation: {error}",
            model_trace=model_trace,
        )

    def _build_projection_sender_event(
        self,
        *,
        index: int,
        signal: ExternalSignal,
        belief_state: BeliefState,
        probe_set: ProbeSet,
        cycle: CycleRecord,
        is_duplicate: bool,
    ) -> EvidenceEvent:
        hypothesis_ids = self._resolve_target_hypotheses(
            signal=signal,
            belief_state=belief_state,
            probe_set=probe_set,
        )
        endorsed_hypothesis = _endorsed_hypothesis(signal.raw_content, belief_state.hypotheses)
        likelihoods = {hypothesis_id: LikelihoodBand.NEUTRAL for hypothesis_id in hypothesis_ids}
        if endorsed_hypothesis in likelihoods:
            likelihoods[endorsed_hypothesis] = LikelihoodBand.WEAKLY_CONFIRMING

        return self._event(
            event_id=f"{_scoped_cycle_key(cycle.run_id, cycle.cycle_id)}_E{index}",
            signal=signal,
            hypothesis_ids=hypothesis_ids,
            evidence_type=EvidenceType.SENDER_JUDGMENT,
            likelihoods=likelihoods,
            interpretation="External projection treated as sender judgment, not direct source evidence.",
            is_duplicate=is_duplicate,
        )

    def _build_source_claim_event(
        self,
        *,
        index: int,
        signal: ExternalSignal,
        belief_state: BeliefState,
        probe_set: ProbeSet,
        cycle: CycleRecord,
        is_duplicate: bool,
    ) -> EvidenceEvent:
        hypothesis_ids = self._resolve_target_hypotheses(
            signal=signal,
            belief_state=belief_state,
            probe_set=probe_set,
        )
        return self._event(
            event_id=f"{_scoped_cycle_key(cycle.run_id, cycle.cycle_id)}_E{index}_source",
            signal=signal,
            hypothesis_ids=hypothesis_ids,
            evidence_type=EvidenceType.SOURCE_CLAIM,
            likelihoods={hypothesis_id: LikelihoodBand.NEUTRAL for hypothesis_id in hypothesis_ids},
            interpretation="Claimed source separated for direct verification; neutral until verified.",
            is_duplicate=is_duplicate,
        )

    def _event(
        self,
        *,
        event_id: str,
        signal: ExternalSignal,
        hypothesis_ids: list[str],
        evidence_type: EvidenceType,
        likelihoods: dict[str, LikelihoodBand],
        interpretation: str,
        is_duplicate: bool,
        quality_overrides: dict[str, float] | None = None,
        discard_reason: str | None = None,
        model_trace: ModelInvocationTrace | None = None,
    ) -> EvidenceEvent:
        quality = self._quality_assessor.assess(
            signal=signal,
            event_type=evidence_type,
            is_duplicate=is_duplicate,
        )
        if quality_overrides:
            quality = _apply_quality_overrides(quality=quality, overrides=quality_overrides)
        return EvidenceEvent(
            id=event_id,
            derived_from_signal=signal.id,
            target_hypotheses=hypothesis_ids,
            evidence_type=evidence_type,
            content=signal.raw_content,
            reliability=quality.reliability,
            independence=quality.independence,
            relevance=quality.relevance,
            novelty=quality.novelty,
            specificity=quality.specificity,
            verifiability=quality.verifiability,
            likelihoods=likelihoods,
            interpretation=interpretation,
            discard_reason=discard_reason,
            model_trace=model_trace.to_dict() if model_trace is not None else {},
        )

    def _resolve_target_hypotheses(
        self,
        *,
        signal: ExternalSignal,
        belief_state: BeliefState,
        probe_set: ProbeSet,
    ) -> list[str]:
        known_hypothesis_ids = {hypothesis.id for hypothesis in belief_state.hypotheses}
        all_hypotheses = [hypothesis.id for hypothesis in belief_state.hypotheses]

        if signal.initial_target_hypotheses:
            explicit_targets = [
                hypothesis_id
                for hypothesis_id in signal.initial_target_hypotheses
                if hypothesis_id in known_hypothesis_ids
            ]
            return explicit_targets or all_hypotheses

        if signal.generated_by_probe:
            for probe in probe_set.probes:
                if probe.id == signal.generated_by_probe:
                    probe_targets = [
                        hypothesis_id
                        for hypothesis_id in probe.target_hypotheses
                        if hypothesis_id in known_hypothesis_ids
                    ]
                    return probe_targets or all_hypotheses

        return all_hypotheses


def _verification_probe_candidate(
    *,
    cycle: CycleRecord,
    event: EvidenceEvent,
    signal: ExternalSignal,
) -> ProbeCandidate:
    support_condition = {
        hypothesis_id: "The cited source independently supports this hypothesis."
        for hypothesis_id in event.target_hypotheses
    }
    weaken_condition = {
        hypothesis_id: "The cited source is unverifiable, duplicated, or contradicts this hypothesis."
        for hypothesis_id in event.target_hypotheses
    }
    return ProbeCandidate(
        candidate_id=f"pc_{event.id}_verify_source",
        source="passive_signal",
        candidate_probe=ProbeDesign(
            id=f"P_{event.id}_verify_source",
            cycle_id=cycle.cycle_id,
            target_hypotheses=list(event.target_hypotheses),
            inquiry_goal=(
                f"Verify the cited source behind external projection {signal.id} "
                f"from {signal.source}."
            ),
            method="source_tracing",
            support_condition=support_condition,
            weaken_condition=weaken_condition,
            expected_information_gain=0.75,
            decision_relevance=0.85,
            cost_estimate=0.45,
            priority=0.8,
        ),
        priority_features={
            "projection_decomposition": True,
            "source_signal_id": signal.id,
            "source_event_id": event.id,
        },
    )


_QUALITY_METRICS = (
    "reliability",
    "independence",
    "relevance",
    "novelty",
    "specificity",
    "verifiability",
)


_ZERO_QUALITY_OVERRIDES = {
    "reliability": 0.0,
    "independence": 0.0,
    "relevance": 0.0,
    "novelty": 0.0,
    "specificity": 0.0,
    "verifiability": 0.0,
}


def _apply_quality_overrides(
    *,
    quality: SignalQuality,
    overrides: dict[str, float],
) -> SignalQuality:
    values = {metric: getattr(quality, metric) for metric in _QUALITY_METRICS}
    for metric, value in overrides.items():
        if metric in values:
            values[metric] = value
    return SignalQuality(**values)


def _is_duplicate_signal(
    *,
    signal: ExternalSignal,
    seen_signatures: set[tuple[str, str]],
) -> bool:
    signature = (signal.source.strip().lower(), _content_signature(signal.raw_content))
    is_duplicate = signature in seen_signatures
    seen_signatures.add(signature)
    return is_duplicate


def _content_signature(content: str) -> str:
    return " ".join(content.lower().split())


def _has_low_reliability_cue(content: str) -> bool:
    content_lower = content.lower()
    return any(cue in content_lower for cue in ("rumor", "unverified", "hearsay", "maybe", "unclear"))


def _endorsed_hypothesis(content: str, hypotheses: list[Hypothesis]) -> str | None:
    for hypothesis in hypotheses:
        escaped_id = re.escape(hypothesis.id)
        patterns = (
            rf"\bbelieves\s+{escaped_id}\b",
            rf"\bbest hypothesis\s+{escaped_id}\b",
            rf"\bcurrent_best_hypothesis\s*[:=]?\s*{escaped_id}\b",
        )
        if any(re.search(pattern, content, flags=re.IGNORECASE) for pattern in patterns):
            return hypothesis.id
    return None


def _repair_payload_from(payload: Any) -> dict[str, Any]:
    if isinstance(payload, Mapping):
        return dict(payload)
    return {"_raw_payload": payload}


def _scoped_cycle_key(run_id: str, cycle_id: str) -> str:
    if cycle_id.startswith(f"{run_id}_"):
        return cycle_id
    return f"{run_id}_{cycle_id}"


__all__ = [
    "EvidenceIntegrationGate",
    "EvidenceIntegrationResult",
    "ProjectionDecomposer",
    "SignalQualityAssessor",
]
