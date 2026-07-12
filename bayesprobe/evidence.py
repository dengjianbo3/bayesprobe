from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from bayesprobe.evidence_memory import (
    EvidenceMemoryDecision,
    EvidenceMemoryManager,
    SIGNAL_QUALITY_METRICS,
    SignalQuality,
    SignalQualityAssessor,
    SignalProvenanceNormalizer,
    canonical_signal_identity_digest,
    observe_cycle_signal_duplicate,
)
from bayesprobe.lifecycle import BeliefLifecycle, resolve_belief_lifecycle
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
    EvidenceMemorySnapshot,
    EvidenceType,
    ExternalSignal,
    FrameFit,
    Hypothesis,
    LikelihoodBand,
    HypothesisCompetition,
    HypothesisCoverage,
    ProbeCandidate,
    ProbeDesign,
    ProbeSet,
    redact_secret_material,
    validate_canonical_event_binding_id,
)


@dataclass(frozen=True)
class EvidenceIntegrationResult:
    evidence_events: list[EvidenceEvent]
    probe_candidates: list[ProbeCandidate]
    evidence_memory: EvidenceMemorySnapshot | None = None
    normalized_signals: list[ExternalSignal] | None = None


@dataclass(frozen=True)
class _PlannedSignalEvents:
    signal: ExternalSignal
    event_ids: tuple[str, ...]


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
        provenance_normalizer: SignalProvenanceNormalizer | None = None,
        memory_manager: EvidenceMemoryManager | None = None,
    ) -> None:
        self._quality_assessor = quality_assessor or SignalQualityAssessor()
        self._projection_decomposer = projection_decomposer or ProjectionDecomposer()
        self._model_gateway = model_gateway or DeterministicModelGateway()
        self._judgment_repair_policy = judgment_repair_policy or EvidenceJudgmentRepairPolicy()
        self._provenance_normalizer = provenance_normalizer or SignalProvenanceNormalizer()
        self._memory_manager = memory_manager or EvidenceMemoryManager()

    def integrate(
        self,
        *,
        cycle: CycleRecord,
        belief_state: BeliefState,
        probe_set: ProbeSet,
        signals: list[ExternalSignal],
    ) -> EvidenceIntegrationResult:
        self._ensure_helpers()
        lifecycle = resolve_belief_lifecycle(belief_state)
        native_v02 = lifecycle == BeliefLifecycle.NATIVE_V02
        evidence_events: list[EvidenceEvent] = []
        probe_candidates: list[ProbeCandidate] = []
        closed_signals: list[ExternalSignal] = []
        seen_signatures: set[tuple[str, str]] = set()
        working_memory = belief_state.evidence_memory or EvidenceMemorySnapshot()
        prior_evidence_ids = set(
            belief_state.ledger_refs.get("evidence_events", [])
        )

        normalized_signals = [
            self._provenance_normalizer.normalize(
                raw_signal,
                run_id=cycle.run_id,
            )
            for raw_signal in signals
        ]
        planned_signals = self._plan_signal_events(
            cycle=cycle,
            signals=normalized_signals,
            native_v02=native_v02,
        )
        _preflight_migrated_event_set(
            cycle=cycle,
            prior_evidence_ids=prior_evidence_ids,
            planned_signals=planned_signals,
            native_v02=native_v02,
        )
        for planned in planned_signals:
            for event_id in planned.event_ids:
                if event_id in prior_evidence_ids:
                    self._memory_manager.validate_event_signal_identity(
                        working_memory,
                        event_id=event_id,
                        signal=planned.signal,
                        require_existing=True,
                    )

        preflight_memory = working_memory
        for planned in planned_signals:
            signal = planned.signal
            preflight_memory = self._memory_manager.remember_signal_identity(
                preflight_memory,
                signal,
            )

        for planned in planned_signals:
            signal = planned.signal
            self._memory_manager.validate_signal_lineage(
                preflight_memory,
                signal,
            )

        for planned in planned_signals:
            signal = planned.signal
            self._memory_manager.validate_signal_lineage(working_memory, signal)
            is_cycle_duplicate = observe_cycle_signal_duplicate(
                signal,
                seen_signatures,
            )
            closed_signals.append(signal)
            event_id = planned.event_ids[0]
            if event_id in prior_evidence_ids:
                working_memory = self._memory_manager.remember_signal_identity(
                    working_memory,
                    signal,
                )
                evidence_events.append(
                    self._replayed_event(
                        event_id=event_id,
                        signal=signal,
                        belief_state=belief_state,
                        probe_set=probe_set,
                    )
                )
                continue
            prior_decision = self._memory_manager.classify(
                working_memory,
                signal,
                frame_version=(
                    belief_state.frame_state.frame_version
                    if belief_state.frame_state is not None
                    else 1
                ),
            )
            if prior_decision.correlation_status == "duplicate_exact":
                event_result = EvidenceIntegrationResult(
                    evidence_events=[
                        self._duplicate_exact_event(
                            event_id=event_id,
                            signal=signal,
                            belief_state=belief_state,
                            probe_set=probe_set,
                        )
                    ],
                    probe_candidates=[],
                )
            else:
                event_result = self._build_signal_events(
                    event_ids=planned.event_ids,
                    signal=signal,
                    belief_state=belief_state,
                    probe_set=probe_set,
                    cycle=cycle,
                    is_duplicate=is_cycle_duplicate,
                    prior_memory_decision=prior_decision,
                )
            memory_events: list[EvidenceEvent] = []
            classification_snapshot = working_memory
            for event in event_result.evidence_events:
                event, decision = self._apply_memory_decision(
                    event=event,
                    signal=signal,
                    belief_state=belief_state,
                    snapshot=classification_snapshot,
                )
                working_memory = self._memory_manager.commit(
                    working_memory,
                    signal=signal,
                    event=event,
                    decision=decision,
                )
                memory_events.append(event)
            evidence_events.extend(memory_events)
            probe_candidates.extend(event_result.probe_candidates)

        return EvidenceIntegrationResult(
            evidence_events=evidence_events,
            probe_candidates=probe_candidates,
            evidence_memory=working_memory,
            normalized_signals=closed_signals,
        )

    def _plan_signal_events(
        self,
        *,
        cycle: CycleRecord,
        signals: list[ExternalSignal],
        native_v02: bool,
    ) -> list[_PlannedSignalEvents]:
        if signals:
            validate_canonical_event_binding_id(
                _scoped_cycle_key(cycle.run_id, cycle.cycle_id)
            )
        identity_occurrences: dict[str, int] = {}
        planned: list[_PlannedSignalEvents] = []
        for index, signal in enumerate(signals, start=1):
            identity_digest = canonical_signal_identity_digest(signal)
            occurrence = identity_occurrences.get(identity_digest, 0) + 1
            identity_occurrences[identity_digest] = occurrence
            event_id = validate_canonical_event_binding_id(
                _event_id_for_signal(
                    cycle=cycle,
                    signal_identity_digest=identity_digest,
                    index=index,
                    occurrence=occurrence,
                    native_v02=native_v02,
                )
            )
            event_ids = [event_id]
            if (
                signal.source_type == "external_agent_projection"
                and self._projection_decomposer.should_decompose(signal)
            ):
                event_ids.append(
                    validate_canonical_event_binding_id(f"{event_id}_source")
                )
            planned.append(
                _PlannedSignalEvents(
                    signal=signal,
                    event_ids=tuple(event_ids),
                )
            )
        return planned

    def _replayed_event(
        self,
        *,
        event_id: str,
        signal: ExternalSignal,
        belief_state: BeliefState,
        probe_set: ProbeSet,
    ) -> EvidenceEvent:
        hypothesis_ids = self._resolve_target_hypotheses(
            signal=signal,
            belief_state=belief_state,
            probe_set=probe_set,
        )
        unresolved_likelihood = None
        frame_state = belief_state.frame_state
        if frame_state is not None and (
            frame_state.competition == HypothesisCompetition.EXCLUSIVE
            and frame_state.coverage == HypothesisCoverage.OPEN
        ):
            unresolved_likelihood = LikelihoodBand.NEUTRAL
        event = self._event(
            event_id=event_id,
            signal=signal,
            hypothesis_ids=hypothesis_ids,
            evidence_type=EvidenceType.NEUTRAL,
            likelihoods={
                hypothesis_id: LikelihoodBand.NEUTRAL
                for hypothesis_id in hypothesis_ids
            },
            unresolved_likelihood=unresolved_likelihood,
            interpretation="Evidence event identity already exists in the belief ledger.",
            is_duplicate=True,
            discard_reason="duplicate evidence event id",
        )
        provenance = signal.provenance
        return EvidenceEvent.model_validate(
            {
                **event.model_dump(mode="python"),
                "schema_version": (
                    "v0.2" if _is_native_v02_state(belief_state) else "v0.1"
                ),
                "epistemic_origin": provenance.epistemic_origin,
                "derivation_root_id": provenance.derivation_root_id,
                "correlation_status": "duplicate_exact",
                "effective_update_weight": 0.0,
            }
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
        if not hasattr(self, "_provenance_normalizer"):
            self._provenance_normalizer = SignalProvenanceNormalizer()
        if not hasattr(self, "_memory_manager"):
            self._memory_manager = EvidenceMemoryManager()

    def _duplicate_exact_event(
        self,
        *,
        event_id: str,
        signal: ExternalSignal,
        belief_state: BeliefState,
        probe_set: ProbeSet,
    ) -> EvidenceEvent:
        hypothesis_ids = self._resolve_target_hypotheses(
            signal=signal,
            belief_state=belief_state,
            probe_set=probe_set,
        )
        unresolved_likelihood = None
        frame_state = belief_state.frame_state
        if frame_state is not None and (
            frame_state.competition == HypothesisCompetition.EXCLUSIVE
            and frame_state.coverage == HypothesisCoverage.OPEN
        ):
            unresolved_likelihood = LikelihoodBand.NEUTRAL
        return self._event(
            event_id=event_id,
            signal=signal,
            hypothesis_ids=hypothesis_ids,
            evidence_type=EvidenceType.NEUTRAL,
            likelihoods={
                hypothesis_id: LikelihoodBand.NEUTRAL
                for hypothesis_id in hypothesis_ids
            },
            unresolved_likelihood=unresolved_likelihood,
            interpretation="Exact evidence identity already exists in memory.",
            is_duplicate=True,
            discard_reason="duplicate_exact",
        )

    def _apply_memory_decision(
        self,
        *,
        event: EvidenceEvent,
        signal: ExternalSignal,
        belief_state: BeliefState,
        snapshot: EvidenceMemorySnapshot,
    ) -> tuple[EvidenceEvent, EvidenceMemoryDecision]:
        decision = self._memory_manager.classify(
            snapshot,
            signal,
            likelihoods=event.likelihoods,
            unresolved_likelihood=event.unresolved_likelihood,
            frame_version=(
                belief_state.frame_state.frame_version
                if belief_state.frame_state is not None
                else 1
            ),
            base_effective_weight=(
                event.reliability
                * event.independence
                * event.relevance
                * event.novelty
            ),
        )
        independence = event.independence
        novelty = event.novelty
        if decision.correlation_status == "correlated_restatement":
            independence = 0.0
            novelty = min(novelty, 0.25)
        provenance = signal.provenance
        discard_reason = event.discard_reason or decision.discard_reason
        payload = event.model_dump(mode="python")
        payload.update(
            {
                "schema_version": (
                    "v0.2" if _is_native_v02_state(belief_state) else "v0.1"
                ),
                "epistemic_origin": provenance.epistemic_origin,
                "derivation_root_id": provenance.derivation_root_id,
                "correlation_status": decision.correlation_status,
                "effective_update_weight": decision.effective_update_weight,
                "independence": independence,
                "novelty": novelty,
                "discard_reason": discard_reason,
            }
        )
        return EvidenceEvent.model_validate(payload), decision

    def _model_trace_for_request(self, request: StructuredModelRequest) -> ModelInvocationTrace:
        return ModelInvocationTrace.from_request(
            request,
            adapter_kind=model_gateway_adapter_kind(self._model_gateway),
        )

    def _build_signal_events(
        self,
        *,
        event_ids: tuple[str, ...],
        signal: ExternalSignal,
        belief_state: BeliefState,
        probe_set: ProbeSet,
        cycle: CycleRecord,
        is_duplicate: bool,
        prior_memory_decision: EvidenceMemoryDecision,
    ) -> EvidenceIntegrationResult:
        event_id = event_ids[0]
        if signal.source_type == "external_agent_projection":
            sender_event = self._build_projection_sender_event(
                event_id=event_id,
                signal=signal,
                belief_state=belief_state,
                probe_set=probe_set,
                is_duplicate=is_duplicate,
            )
            if len(event_ids) == 1:
                return EvidenceIntegrationResult(
                    evidence_events=[sender_event],
                    probe_candidates=[],
                )
            source_event = self._build_source_claim_event(
                event_id=event_ids[1],
                signal=signal,
                belief_state=belief_state,
                probe_set=probe_set,
                is_duplicate=is_duplicate,
            )
            return EvidenceIntegrationResult(
                evidence_events=[sender_event, source_event],
                probe_candidates=[_verification_probe_candidate(cycle=cycle, event=source_event, signal=signal)],
            )

        return EvidenceIntegrationResult(
            evidence_events=[
                self._build_direct_evidence_event(
                    event_id=event_id,
                    signal=signal,
                    belief_state=belief_state,
                    probe_set=probe_set,
                    cycle=cycle,
                    is_duplicate=is_duplicate,
                    prior_memory_decision=prior_memory_decision,
                )
            ],
            probe_candidates=[],
        )

    def _build_direct_evidence_event(
        self,
        *,
        event_id: str,
        signal: ExternalSignal,
        belief_state: BeliefState,
        probe_set: ProbeSet,
        cycle: CycleRecord,
        is_duplicate: bool,
        prior_memory_decision: EvidenceMemoryDecision,
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
            belief_state=belief_state,
            memory_decision=prior_memory_decision,
        )
        try:
            judgment, model_trace = self._evidence_judgment_with_repair(request=request)
        except _EvidenceJudgmentFailure as failure:
            return self._schema_violation_event(
                event_id=event_id,
                signal=signal,
                hypothesis_ids=hypothesis_ids,
                belief_state=belief_state,
                is_duplicate=is_duplicate,
                error=failure.error,
                model_trace=failure.model_trace,
            )

        return self._event(
            event_id=event_id,
            signal=signal,
            hypothesis_ids=hypothesis_ids,
            evidence_type=judgment.evidence_type,
            likelihoods=judgment.likelihoods,
            unresolved_likelihood=judgment.unresolved_likelihood,
            frame_fit=judgment.frame_fit,
            unexplained_observation=judgment.unexplained_observation,
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
        belief_state: BeliefState,
        memory_decision: EvidenceMemoryDecision,
    ) -> StructuredModelRequest:
        judgment_route = _judgment_route_for_state(belief_state)
        native_v02 = judgment_route == "native_v0.2"
        if native_v02:
            hypotheses_by_id = belief_state.hypotheses_by_id()
            hypotheses = [
                {
                    "id": hypotheses_by_id[hypothesis_id].id,
                    "statement": hypotheses_by_id[hypothesis_id].statement,
                    "type": hypotheses_by_id[hypothesis_id].type,
                    "scope": hypotheses_by_id[hypothesis_id].scope,
                    "posterior": hypotheses_by_id[hypothesis_id].posterior,
                    "predictions": list(
                        hypotheses_by_id[hypothesis_id].predictions
                    ),
                    "falsifiers": list(
                        hypotheses_by_id[hypothesis_id].falsifiers
                    ),
                    "rivals": list(hypotheses_by_id[hypothesis_id].rivals),
                }
                for hypothesis_id in hypothesis_ids
            ]
            frame = {
                "competition": belief_state.frame_state.competition.value,
                "coverage": belief_state.frame_state.coverage.value,
                "frame_version": belief_state.frame_state.frame_version,
            }
            probes = [
                {
                    "id": probe.id,
                    "purpose": probe.probe_type,
                    "target_hypotheses": list(probe.target_hypotheses),
                    "inquiry_goal": probe.inquiry_goal,
                    "support_condition": dict(probe.support_condition),
                    "weaken_condition": dict(probe.weaken_condition),
                    "reframe_condition": (
                        None
                        if probe.reframe_condition is None
                        else dict(probe.reframe_condition)
                    ),
                }
                for probe in probe_set.probes
            ]
            provenance = signal.provenance.model_dump(mode="json")
            memory = {
                "correlation_status": memory_decision.correlation_status,
                "remaining_credit": dict(memory_decision.remaining_credit),
                "accepted_evidence_count": len(
                    belief_state.evidence_memory.accepted_evidence_ids
                ),
            }
        else:
            hypotheses = []
            frame = None
            probes = []
            provenance = None
            memory = None
        request_input = {
            "signal_id": signal.id,
            "source_type": signal.source_type,
            "source": signal.source,
            "raw_content": signal.raw_content,
            "target_hypotheses": hypothesis_ids,
            "cycle_id": cycle.cycle_id,
            "probe_ids": [probe.id for probe in probe_set.probes],
        }
        if native_v02:
            request_input.update(
                {
                    "hypotheses": hypotheses,
                    "frame": frame,
                    "probes": probes,
                    "provenance": provenance,
                    "memory": memory,
                    "allowed_evidence_types": [
                        evidence_type.value for evidence_type in EvidenceType
                    ],
                    "allowed_likelihood_bands": [
                        band.value for band in LikelihoodBand
                    ],
                    "allowed_frame_fits": [frame_fit.value for frame_fit in FrameFit],
                }
            )
        route_metadata = {
            "judgment_route": judgment_route,
            "lifecycle_schema_version": belief_state.schema_version,
            "frame_competition": (
                belief_state.frame_state.competition.value
                if belief_state.frame_state is not None
                else HypothesisCompetition.EXCLUSIVE.value
            ),
            "frame_coverage": (
                belief_state.frame_state.coverage.value
                if belief_state.frame_state is not None
                else HypothesisCoverage.EXHAUSTIVE.value
            ),
        }
        if belief_state.task_frame is not None:
            route_metadata["framing_method"] = (
                belief_state.task_frame.framing_method.value
            )
        if native_v02:
            route_metadata.update(
                {
                    "run_id": cycle.run_id,
                    "cycle_id": cycle.cycle_id,
                    "signal_id": signal.id,
                }
            )
        return StructuredModelRequest(
            task="judge_evidence",
            input=request_input,
            prompt_id="evidence_judgment",
            prompt_version="v0.2" if native_v02 else "v0.1",
            schema_name="EvidenceJudgment",
            schema_version="v0.2" if native_v02 else "v0.1",
            metadata=route_metadata,
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
            return self._validated_evidence_judgment(
                payload=payload,
                original_request=request,
            ), model_trace
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
                    "required_fields": (
                        [
                            "evidence_type",
                            "likelihoods",
                            "unresolved_likelihood",
                            "frame_fit",
                            "unexplained_observation",
                            "interpretation",
                            "quality_overrides",
                        ]
                        if original_request.schema_version == "v0.2"
                        else [
                            "evidence_type",
                            "likelihoods",
                            "interpretation",
                            "quality_overrides",
                        ]
                    ),
                    "allowed_frame_fits": [
                        frame_fit.value for frame_fit in FrameFit
                    ],
                },
                prompt_id="evidence_judgment_repair",
                prompt_version=original_request.prompt_version,
                schema_name="EvidenceJudgment",
                schema_version=original_request.schema_version,
                metadata={
                    **original_request.metadata,
                    "repair_attempt_index": attempt_index,
                },
            )
            repair_trace = self._model_trace_for_request(repair_request)
            try:
                repair_payload = self._model_gateway.complete_structured(repair_request)
            except ModelGatewayValidationError as error:
                latest_error = error
                latest_trace = repair_trace
                continue
            try:
                return self._validated_evidence_judgment(
                    payload=repair_payload,
                    original_request=original_request,
                ), repair_trace
            except ModelGatewayValidationError as error:
                latest_invalid_payload = _repair_payload_from(repair_payload)
                latest_error = error
                latest_trace = repair_trace

        failure = ModelGatewayValidationError(
            f"repair failed after {max_attempts} attempt(s): {latest_error}"
        )
        raise _EvidenceJudgmentFailure(error=failure, model_trace=latest_trace) from latest_error

    def _validated_evidence_judgment(
        self,
        *,
        payload: dict[str, Any],
        original_request: StructuredModelRequest,
    ) -> EvidenceJudgment:
        frame = original_request.input.get("frame")
        if original_request.schema_version == "v0.2":
            if original_request.metadata.get("judgment_route") != "native_v0.2":
                raise ModelGatewayValidationError(
                    "native evidence judgment requires the native v0.2 route"
                )
            judgment = evidence_judgment_from_mapping(
                payload,
                competition=HypothesisCompetition(frame["competition"]),
                coverage=HypothesisCoverage(frame["coverage"]),
            )
        else:
            if (
                original_request.metadata.get("judgment_route")
                != "legacy_v0.1_migration"
            ):
                raise ModelGatewayValidationError(
                    "legacy evidence judgment requires an explicit migration route"
                )
            competition = HypothesisCompetition(
                original_request.metadata["frame_competition"]
            )
            coverage = HypothesisCoverage(
                original_request.metadata["frame_coverage"]
            )
            payload = _migrate_v01_judgment_payload(
                payload,
                competition=competition,
                coverage=coverage,
            )
            judgment = evidence_judgment_from_mapping(
                payload,
                competition=competition,
                coverage=coverage,
            )
        expected_targets = {
            str(hypothesis_id)
            for hypothesis_id in original_request.input.get(
                "target_hypotheses",
                [],
            )
        }
        actual_targets = set(judgment.likelihoods)
        if actual_targets != expected_targets:
            raise ModelGatewayValidationError(
                "evidence judgment likelihood targets must equal "
                f"{sorted(expected_targets)}; got {sorted(actual_targets)}"
            )
        return judgment

    def _schema_violation_event(
        self,
        *,
        event_id: str,
        signal: ExternalSignal,
        hypothesis_ids: list[str],
        belief_state: BeliefState,
        is_duplicate: bool,
        error: ModelGatewayValidationError,
        model_trace: ModelInvocationTrace | None = None,
    ) -> EvidenceEvent:
        return self._event(
            event_id=event_id,
            signal=signal,
            hypothesis_ids=hypothesis_ids,
            evidence_type=EvidenceType.NEUTRAL,
            likelihoods={hypothesis_id: LikelihoodBand.NEUTRAL for hypothesis_id in hypothesis_ids},
            unresolved_likelihood=_exclusive_open_unresolved_likelihood(belief_state),
            frame_fit=FrameFit.UNDERDETERMINED,
            interpretation="Model gateway judgment failed schema validation.",
            is_duplicate=is_duplicate,
            quality_overrides=_ZERO_QUALITY_OVERRIDES,
            discard_reason=f"schema_violation: {error}",
            model_trace=model_trace,
        )

    def _build_projection_sender_event(
        self,
        *,
        event_id: str,
        signal: ExternalSignal,
        belief_state: BeliefState,
        probe_set: ProbeSet,
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
        unresolved_likelihood = _exclusive_open_unresolved_likelihood(belief_state)

        return self._event(
            event_id=event_id,
            signal=signal,
            hypothesis_ids=hypothesis_ids,
            evidence_type=EvidenceType.SENDER_JUDGMENT,
            likelihoods=likelihoods,
            unresolved_likelihood=unresolved_likelihood,
            frame_fit=FrameFit.UNDERDETERMINED,
            interpretation="External projection treated as sender judgment, not direct source evidence.",
            is_duplicate=is_duplicate,
        )

    def _build_source_claim_event(
        self,
        *,
        event_id: str,
        signal: ExternalSignal,
        belief_state: BeliefState,
        probe_set: ProbeSet,
        is_duplicate: bool,
    ) -> EvidenceEvent:
        hypothesis_ids = self._resolve_target_hypotheses(
            signal=signal,
            belief_state=belief_state,
            probe_set=probe_set,
        )
        unresolved_likelihood = _exclusive_open_unresolved_likelihood(belief_state)
        return self._event(
            event_id=event_id,
            signal=signal,
            hypothesis_ids=hypothesis_ids,
            evidence_type=EvidenceType.SOURCE_CLAIM,
            likelihoods={hypothesis_id: LikelihoodBand.NEUTRAL for hypothesis_id in hypothesis_ids},
            unresolved_likelihood=unresolved_likelihood,
            frame_fit=FrameFit.UNDERDETERMINED,
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
        unresolved_likelihood: LikelihoodBand | None = None,
        frame_fit: FrameFit = FrameFit.UNDERDETERMINED,
        unexplained_observation: str | None = None,
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
            effective_overrides = {
                metric: min(value, getattr(quality, metric))
                for metric, value in quality_overrides.items()
            }
            quality = _apply_quality_overrides(
                quality=quality,
                overrides=effective_overrides,
            )
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
            unresolved_likelihood=unresolved_likelihood,
            frame_fit=frame_fit,
            unexplained_observation=unexplained_observation,
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
    values = {
        metric: getattr(quality, metric)
        for metric in SIGNAL_QUALITY_METRICS
    }
    for metric, value in overrides.items():
        if metric in values:
            values[metric] = value
    return SignalQuality(**values)


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
        return dict(redact_secret_material(payload))
    return {"_raw_payload": redact_secret_material(payload)}


def _scoped_cycle_key(run_id: str, cycle_id: str) -> str:
    if cycle_id.startswith(f"{run_id}_"):
        return cycle_id
    return f"{run_id}_{cycle_id}"


def _event_id_for_signal(
    *,
    cycle: CycleRecord,
    signal_identity_digest: str,
    index: int,
    occurrence: int,
    native_v02: bool,
) -> str:
    scoped_cycle = _scoped_cycle_key(cycle.run_id, cycle.cycle_id)
    if not native_v02:
        return f"{scoped_cycle}_E{index}"
    return f"{scoped_cycle}_E_{signal_identity_digest}_{occurrence}"


def _preflight_migrated_event_set(
    *,
    cycle: CycleRecord,
    prior_evidence_ids: set[str],
    planned_signals: list[_PlannedSignalEvents],
    native_v02: bool,
) -> None:
    if native_v02:
        return
    scoped_cycle = _scoped_cycle_key(cycle.run_id, cycle.cycle_id)
    positional_pattern = re.compile(
        rf"{re.escape(scoped_cycle)}_E[1-9][0-9]*(?:_source)?"
    )
    prior_cycle_ids = {
        event_id
        for event_id in prior_evidence_ids
        if positional_pattern.fullmatch(event_id)
    }
    if not prior_cycle_ids:
        return
    planned_event_ids = {
        event_id
        for planned in planned_signals
        for event_id in planned.event_ids
    }
    if planned_event_ids != prior_cycle_ids:
        raise ValueError("evidence event replay set conflict")


def _is_native_v02_state(belief_state: BeliefState) -> bool:
    return (
        resolve_belief_lifecycle(belief_state)
        == BeliefLifecycle.NATIVE_V02
    )


def _exclusive_open_unresolved_likelihood(
    belief_state: BeliefState,
) -> LikelihoodBand | None:
    frame_state = belief_state.frame_state
    if frame_state is None:
        return None
    if (
        frame_state.competition == HypothesisCompetition.EXCLUSIVE
        and frame_state.coverage == HypothesisCoverage.OPEN
    ):
        return LikelihoodBand.NEUTRAL
    return None


def _judgment_route_for_state(belief_state: BeliefState) -> str:
    return resolve_belief_lifecycle(belief_state).value


def _migrate_v01_judgment_payload(
    payload: dict[str, Any],
    *,
    competition: HypothesisCompetition,
    coverage: HypothesisCoverage,
) -> dict[str, Any]:
    """Explicitly complete the reviewed v0.1 provider response shape."""
    if not isinstance(payload, Mapping):
        return payload
    v01_fields = {
        "evidence_type",
        "likelihoods",
        "interpretation",
        "quality_overrides",
    }
    if set(payload) != v01_fields:
        raise ModelGatewayValidationError(
            "legacy evidence judgment requires exactly the reviewed four fields"
        )
    migrated = dict(payload)
    migrated.setdefault("quality_overrides", {})
    exclusive_open = (
        competition == HypothesisCompetition.EXCLUSIVE
        and coverage == HypothesisCoverage.OPEN
    )
    evidence_type = migrated.get("evidence_type")
    likelihoods = migrated.get("likelihoods")
    has_directional_likelihood = isinstance(likelihoods, Mapping) and any(
        likelihood != LikelihoodBand.NEUTRAL.value
        for likelihood in likelihoods.values()
    )
    if evidence_type == EvidenceType.ANOMALY.value and exclusive_open:
        unresolved_likelihood = LikelihoodBand.MODERATELY_CONFIRMING.value
        frame_fit = FrameFit.SUPPORTS_UNRESOLVED.value
        unexplained_observation = "The signal is poorly explained by named hypotheses."
    elif has_directional_likelihood:
        unresolved_likelihood = (
            LikelihoodBand.NEUTRAL.value if exclusive_open else None
        )
        frame_fit = FrameFit.EXPLAINED_BY_NAMED.value
        unexplained_observation = None
    else:
        unresolved_likelihood = (
            LikelihoodBand.NEUTRAL.value if exclusive_open else None
        )
        frame_fit = FrameFit.UNDERDETERMINED.value
        unexplained_observation = None
    migrated.update(
        {
            "unresolved_likelihood": unresolved_likelihood,
            "frame_fit": frame_fit,
            "unexplained_observation": unexplained_observation,
        }
    )
    return migrated


__all__ = [
    "EvidenceIntegrationGate",
    "EvidenceIntegrationResult",
    "ProjectionDecomposer",
    "SignalQualityAssessor",
]
