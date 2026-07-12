from __future__ import annotations

from dataclasses import dataclass, field

from bayesprobe.belief import (
    CoverageAwareBeliefSolver,
    mark_replayed_evidence_events,
    summarize_hypotheses,
)
from bayesprobe.evidence import EvidenceIntegrationGate, EvidenceIntegrationResult
from bayesprobe.evidence_memory import (
    EvidenceMemoryManager,
    SignalProvenanceNormalizer,
)
from bayesprobe.frame_policy import FrameAdequacyDecision, FrameAdequacyPolicy
from bayesprobe.hypothesis_evolution import HypothesisEvolutionEngine
from bayesprobe.inbox import SignalInbox
from bayesprobe.kernel_config import CorrelationCreditPolicy
from bayesprobe.ledger import JsonlLedgerStore
from bayesprobe.lifecycle import BeliefLifecycle, resolve_belief_lifecycle
from bayesprobe.migrations import _carry_v01_migration_receipt
from bayesprobe.model_gateway import EvidenceJudgmentRepairPolicy, ModelGateway
from bayesprobe.schemas import (
    BeliefState,
    BeliefUpdate,
    BoundaryStatus,
    CycleRecord,
    CycleSignalShape,
    EvidenceEvent,
    EvidenceMemorySnapshot,
    ExternalSignal,
    FrameMassUpdate,
    FramingMethod,
    HypothesisEvolution,
    HypothesisCompetition,
    HypothesisCoverage,
    HypothesisStatus,
    ProbeCandidate,
    ProbeSet,
    SignalKind,
    contains_secret_material,
    utc_now,
)
from bayesprobe.task_framing import migrate_legacy_belief_state


@dataclass(frozen=True)
class CycleResult:
    cycle: CycleRecord
    belief_state: BeliefState
    evidence_events: list[EvidenceEvent]
    belief_updates: list[BeliefUpdate]
    frame_mass_updates: list[FrameMassUpdate]
    frame_adequacy_decision: FrameAdequacyDecision
    hypothesis_evolutions: list[HypothesisEvolution]
    probe_candidates: list[ProbeCandidate] = field(default_factory=list)


class BayesProbeCore:
    def __init__(
        self,
        ledger: JsonlLedgerStore | None = None,
        model_gateway: ModelGateway | None = None,
        judgment_repair_policy: EvidenceJudgmentRepairPolicy | None = None,
        correlation_credit_policy: CorrelationCreditPolicy | None = None,
    ) -> None:
        self._ledger = ledger
        self._model_gateway = model_gateway
        self._judgment_repair_policy = judgment_repair_policy
        self._cycle_allocations: dict[str, int] = {}
        configured_credit_policy = (
            correlation_credit_policy or CorrelationCreditPolicy()
        )
        self._correlation_credit_policy = _copy_correlation_credit_policy(
            configured_credit_policy
        )
        self._evidence_memory_manager = EvidenceMemoryManager(
            _copy_correlation_credit_policy(self._correlation_credit_policy)
        )
        self._evidence_gate = self._create_evidence_integration_gate()
        self._belief_solver = self._create_belief_solver()
        self._frame_policy = self._create_frame_adequacy_policy()
        self._evolution_policy = self._create_hypothesis_evolution_policy()

    @property
    def ledger(self) -> JsonlLedgerStore | None:
        return self._ledger

    def allocate_cycle_id(self, base_cycle_id: str) -> str:
        count = self._cycle_allocations.get(base_cycle_id, 0) + 1
        self._cycle_allocations[base_cycle_id] = count
        if count == 1:
            return base_cycle_id
        return f"{base_cycle_id}_r{count}"

    def integrate_cycle(
        self,
        cycle: CycleRecord,
        belief_state: BeliefState,
        probe_set: ProbeSet,
        signals: list[ExternalSignal],
    ) -> CycleResult:
        migrated_belief_state = migrate_legacy_belief_state(belief_state)
        lifecycle = resolve_belief_lifecycle(migrated_belief_state)
        authoritative_belief_state = migrated_belief_state.model_copy(deep=True)
        if resolve_belief_lifecycle(authoritative_belief_state) != lifecycle:
            raise ValueError("belief lifecycle isolation is invalid")
        self._evidence_memory_manager.validate_policy_snapshot(
            authoritative_belief_state.evidence_memory
        )
        authoritative_cycle = cycle.model_copy(deep=True)
        authoritative_probe_set = probe_set.model_copy(deep=True)
        self._validate_cycle_boundary(
            cycle=authoritative_cycle,
            belief_state=authoritative_belief_state,
            probe_set=authoritative_probe_set,
        )
        inbox = self._create_signal_inbox(authoritative_cycle)
        for signal in signals:
            inbox.add(signal.model_copy(deep=True))
        authoritative_closed_signals = _deep_signal_copies(inbox.close())
        authoritative_closed_cycle = authoritative_cycle.model_copy(
            deep=True,
            update={
                "boundary_status": BoundaryStatus.CLOSED,
                "boundary_closed_at": utc_now(),
            }
        )
        self._validate_signal_shape(
            cycle=authoritative_closed_cycle,
            signals=authoritative_closed_signals,
        )
        integration = self._normalize_evidence_integration(
            self._evidence_gate.integrate(
                cycle=authoritative_closed_cycle.model_copy(deep=True),
                belief_state=authoritative_belief_state.model_copy(deep=True),
                probe_set=authoritative_probe_set.model_copy(deep=True),
                signals=_deep_signal_copies(authoritative_closed_signals),
            )
        )
        ledger_signals = _resolve_owned_closed_signals(
            lifecycle=lifecycle,
            inbox_signals=authoritative_closed_signals,
            integration=integration,
            run_id=authoritative_cycle.run_id,
        )
        next_evidence_memory = _resolve_next_evidence_memory(
            lifecycle=lifecycle,
            belief_state=authoritative_belief_state,
            integration=integration,
            normalized_signals=ledger_signals,
            memory_manager=self._evidence_memory_manager,
        )
        evidence_events = mark_replayed_evidence_events(
            authoritative_belief_state,
            integration.evidence_events,
        )
        canonical_evidence_events = _canonical_new_evidence_events(
            authoritative_belief_state.ledger_refs.get("evidence_events", []),
            evidence_events,
        )
        probe_candidates = integration.probe_candidates
        solve_result = self._belief_solver.solve(
            authoritative_belief_state,
            evidence_events,
            run_id=authoritative_cycle.run_id,
            cycle_id=authoritative_cycle.cycle_id,
        )
        open_exclusive_frame = (
            solve_result.frame_state.competition
            == HypothesisCompetition.EXCLUSIVE
            and solve_result.frame_state.coverage == HypothesisCoverage.OPEN
        )
        if open_exclusive_frame:
            accepted_events = [
                event for event in evidence_events if event.discard_reason is None
            ]
            retirement_result = self._evolution_policy.retire_stale_hypotheses(
                cycle=authoritative_closed_cycle,
                hypotheses=solve_result.hypotheses,
                evidence_events=accepted_events,
            )
            solve_result = self._belief_solver.reconcile_retirements(
                solve_result,
                evolved_hypotheses=retirement_result.hypotheses,
                evolutions=retirement_result.evolutions,
                events=accepted_events,
                run_id=authoritative_cycle.run_id,
                cycle_id=authoritative_cycle.cycle_id,
            )
            evolutions = retirement_result.evolutions
        belief_updates = solve_result.belief_updates
        frame_mass_updates = solve_result.frame_mass_updates
        frame_adequacy_decision = self._frame_policy.assess(
            previous=solve_result.frame_state,
            events=evidence_events,
            hypotheses=solve_result.hypotheses,
        )
        record_frame_adequacy_decision = (
            authoritative_belief_state.task_frame.framing_method
            != FramingMethod.LEGACY_MIGRATION
            or frame_adequacy_decision.frame_state.adequacy_status
            != authoritative_belief_state.frame_state.adequacy_status
            or frame_adequacy_decision.should_expand
            or bool(frame_adequacy_decision.trigger_event_ids)
        )
        if open_exclusive_frame:
            evolved_hypotheses = solve_result.hypotheses
        else:
            evolution_result = self._evolution_policy.evolve(
                cycle=authoritative_closed_cycle,
                previous_belief_state=authoritative_belief_state,
                updated_hypotheses=solve_result.hypotheses,
                evidence_events=evidence_events,
                belief_updates=belief_updates,
            )
            evolved_hypotheses = evolution_result.hypotheses
            evolutions = evolution_result.evolutions
            probe_candidates = [
                *probe_candidates,
                *evolution_result.probe_candidates,
            ]
        next_frame_state = frame_adequacy_decision.frame_state.model_copy(
            update={
                "active_hypothesis_ids": [
                    hypothesis.id
                    for hypothesis in evolved_hypotheses
                    if hypothesis.status
                    not in {HypothesisStatus.RETIRED, HypothesisStatus.ARCHIVED}
                ]
            }
        )
        existing_ledger_refs = dict(authoritative_belief_state.ledger_refs)
        merged_ledger_refs = dict(existing_ledger_refs)
        merged_ledger_refs["probe_sets"] = [
            *existing_ledger_refs.get("probe_sets", []),
            authoritative_probe_set.probe_set_id,
        ]
        merged_ledger_refs["evidence_events"] = _append_unique(
            existing_ledger_refs.get("evidence_events", []),
            [event.id for event in canonical_evidence_events],
        )
        merged_ledger_refs["belief_updates"] = [
            *existing_ledger_refs.get("belief_updates", []),
            *(update.update_id for update in belief_updates),
        ]
        merged_ledger_refs["frame_mass_updates"] = [
            *existing_ledger_refs.get("frame_mass_updates", []),
            *(update.update_id for update in frame_mass_updates),
        ]
        merged_ledger_refs["hypothesis_evolutions"] = [
            *existing_ledger_refs.get("hypothesis_evolutions", []),
            *(evolution.evolution_id for evolution in evolutions),
        ]
        merged_ledger_refs["probe_candidates"] = [
            *existing_ledger_refs.get("probe_candidates", []),
            *(candidate.candidate_id for candidate in probe_candidates),
        ]
        posterior_summary, uncertainty_summary = summarize_hypotheses(
            evolved_hypotheses,
            frame_state=next_frame_state,
        )
        updated_state_payload = authoritative_belief_state.model_dump(mode="python")
        updated_state_payload.update(
            {
                "belief_state_id": (
                    f"{authoritative_cycle.run_id}_bs_"
                    f"{authoritative_cycle.cycle_index}"
                ),
                "cycle_id": authoritative_cycle.cycle_id,
                "cycle_index": authoritative_cycle.cycle_index,
                "hypotheses": [
                    hypothesis.model_dump(mode="python")
                    for hypothesis in evolved_hypotheses
                ],
                "frame_state": next_frame_state.model_dump(mode="python"),
                "evidence_memory": next_evidence_memory.model_dump(mode="python"),
                "posterior_summary": posterior_summary,
                "uncertainty_summary": uncertainty_summary,
                "ledger_refs": merged_ledger_refs,
            }
        )
        updated_state = _deep_revalidate_final_belief_state(updated_state_payload)
        updated_state = _carry_v01_migration_receipt(
            authoritative_belief_state,
            updated_state,
        )
        integrated_cycle = authoritative_closed_cycle.model_copy(
            deep=True,
            update={
                "boundary_status": BoundaryStatus.INTEGRATED,
                "completed_at": utc_now(),
            }
        )
        self._append_ledger_records(
            cycle=integrated_cycle,
            signals=ledger_signals,
            probe_set=authoritative_probe_set,
            evidence_events=canonical_evidence_events,
            belief_updates=belief_updates,
            frame_mass_updates=frame_mass_updates,
            frame_adequacy_decision=frame_adequacy_decision,
            record_frame_adequacy_decision=record_frame_adequacy_decision,
            evolutions=evolutions,
            probe_candidates=probe_candidates,
            belief_state=updated_state,
        )
        return CycleResult(
            cycle=integrated_cycle,
            belief_state=updated_state,
            evidence_events=evidence_events,
            belief_updates=belief_updates,
            frame_mass_updates=frame_mass_updates,
            frame_adequacy_decision=frame_adequacy_decision,
            hypothesis_evolutions=evolutions,
            probe_candidates=probe_candidates,
        )

    def _create_signal_inbox(self, cycle: CycleRecord) -> SignalInbox:
        return SignalInbox(cycle_id=cycle.cycle_id)

    def _create_evidence_integration_gate(self) -> EvidenceIntegrationGate:
        return EvidenceIntegrationGate(
            model_gateway=self._model_gateway,
            judgment_repair_policy=self._judgment_repair_policy,
            memory_manager=EvidenceMemoryManager(
                _copy_correlation_credit_policy(
                    self._correlation_credit_policy
                )
            ),
        )

    def _create_belief_solver(self) -> CoverageAwareBeliefSolver:
        return CoverageAwareBeliefSolver()

    def _create_frame_adequacy_policy(self) -> FrameAdequacyPolicy:
        return FrameAdequacyPolicy()

    def _create_hypothesis_evolution_policy(self) -> HypothesisEvolutionEngine:
        return HypothesisEvolutionEngine()

    def _normalize_evidence_integration(
        self,
        integration: EvidenceIntegrationResult | list[EvidenceEvent],
    ) -> EvidenceIntegrationResult:
        if isinstance(integration, EvidenceIntegrationResult):
            return integration
        return EvidenceIntegrationResult(
            evidence_events=list(integration),
            probe_candidates=[],
        )

    def _validate_signal_shape(self, *, cycle: CycleRecord, signals: list[ExternalSignal]) -> None:
        active_count = sum(
            1 for signal in signals if signal.signal_kind == SignalKind.ACTIVE
        )
        passive_count = sum(
            1 for signal in signals if signal.signal_kind == SignalKind.PASSIVE
        )
        if cycle.signal_shape == CycleSignalShape.ACTIVE_ONLY:
            invalid = [signal.id for signal in signals if signal.signal_kind != SignalKind.ACTIVE]
            if invalid:
                raise ValueError("active_only cycles accept only active signals")
            if active_count == 0:
                raise ValueError("active_only cycles require at least one active signal")
            return
        if cycle.signal_shape == CycleSignalShape.PASSIVE_ONLY:
            invalid = [signal.id for signal in signals if signal.signal_kind != SignalKind.PASSIVE]
            if invalid:
                raise ValueError("passive_only cycles accept only passive signals")
            if passive_count == 0:
                raise ValueError("passive_only cycles require at least one passive signal")
            return
        if active_count == 0 or passive_count == 0:
            raise ValueError(
                "active_plus_passive cycles require both active and passive signals"
            )

    def _append_ledger_records(
        self,
        *,
        cycle: CycleRecord,
        signals: list[ExternalSignal],
        probe_set: ProbeSet,
        evidence_events: list[EvidenceEvent],
        belief_updates: list[BeliefUpdate],
        frame_mass_updates: list[FrameMassUpdate],
        frame_adequacy_decision: FrameAdequacyDecision,
        record_frame_adequacy_decision: bool,
        evolutions: list[HypothesisEvolution],
        probe_candidates: list[ProbeCandidate],
        belief_state: BeliefState,
    ) -> None:
        if self._ledger is None:
            return
        self._ledger.append("cycle", cycle)
        for signal in signals:
            self._ledger.append("external_signal", signal)
        self._ledger.append("probe_set", probe_set)
        for event in evidence_events:
            self._ledger.append("evidence_event", event)
        for update in belief_updates:
            self._ledger.append("belief_update", update)
        for update in frame_mass_updates:
            self._ledger.append("frame_mass_update", update)
        if record_frame_adequacy_decision:
            self._ledger.append(
                "frame_adequacy_decision",
                {
                    "cycle_id": cycle.cycle_id,
                    "frame_state": frame_adequacy_decision.frame_state.model_dump(
                        mode="json"
                    ),
                    "should_expand": frame_adequacy_decision.should_expand,
                    "trigger_event_ids": frame_adequacy_decision.trigger_event_ids,
                    "reason": frame_adequacy_decision.reason,
                },
            )
        for evolution in evolutions:
            self._ledger.append("hypothesis_evolution", evolution)
        for candidate in probe_candidates:
            self._ledger.append("probe_candidate", candidate)
        self._ledger.append("belief_state", belief_state)

    def _validate_cycle_boundary(
        self,
        *,
        cycle: CycleRecord,
        belief_state: BeliefState,
        probe_set: ProbeSet,
    ) -> None:
        if cycle.boundary_status != BoundaryStatus.OPEN:
            raise ValueError("cycle must be open before integration")
        if probe_set.cycle_id != cycle.cycle_id:
            raise ValueError("probe set must be frozen to the current cycle")
        if belief_state.run_id != cycle.run_id:
            raise ValueError("belief state must belong to the current run")
        if belief_state.cycle_index > cycle.cycle_index:
            raise ValueError("belief state cannot come from a future cycle")
        for probe in probe_set.probes:
            if probe.cycle_id != cycle.cycle_id:
                raise ValueError("probe design must be frozen to the current cycle")


def _deep_signal_copies(
    signals: list[ExternalSignal],
) -> list[ExternalSignal]:
    return [signal.model_copy(deep=True) for signal in signals]


def _copy_correlation_credit_policy(
    policy: CorrelationCreditPolicy,
) -> CorrelationCreditPolicy:
    return CorrelationCreditPolicy(
        max_cumulative_effective_weight_per_direction=(
            policy.max_cumulative_effective_weight_per_direction
        )
    )


def _scoped_cycle_key(run_id: str, cycle_id: str) -> str:
    if cycle_id.startswith(f"{run_id}_"):
        return cycle_id
    return f"{run_id}_{cycle_id}"


def _deep_revalidate_final_belief_state(payload: dict[str, object]) -> BeliefState:
    """Rebuild the final state so every nested domain validator runs."""
    try:
        return BeliefState.model_validate(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError("final belief state failed recursive validation") from exc


def _append_unique(existing: list[str], additions: list[str]) -> list[str]:
    result = list(existing)
    seen = set(result)
    for item in additions:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def _resolve_next_evidence_memory(
    *,
    lifecycle: BeliefLifecycle,
    belief_state: BeliefState,
    integration: EvidenceIntegrationResult,
    normalized_signals: list[ExternalSignal],
    memory_manager: EvidenceMemoryManager,
) -> EvidenceMemorySnapshot:
    memory = integration.evidence_memory
    if lifecycle == BeliefLifecycle.NATIVE_V02:
        _validate_native_evidence_events(
            integration.evidence_events,
            belief_state=belief_state,
        )
        prior_memory = belief_state.evidence_memory
        if memory is None or prior_memory is None:
            raise ValueError(
                "native evidence integration requires coherent evidence memory"
            )
        try:
            return memory_manager.validate_transition(
                prior_memory,
                memory,
                evidence_events=integration.evidence_events,
                normalized_signals=normalized_signals,
                existing_evidence_ids=belief_state.ledger_refs.get(
                    "evidence_events",
                    [],
                ),
                frame_version=belief_state.frame_state.frame_version,
            )
        except (TypeError, ValueError):
            raise ValueError(
                "native evidence memory transition is invalid"
            ) from None
    if memory is None:
        memory = belief_state.evidence_memory
    if memory is None:
        raise ValueError("belief state requires evidence memory")
    return memory


def _resolve_owned_closed_signals(
    *,
    lifecycle: BeliefLifecycle,
    inbox_signals: list[ExternalSignal],
    integration: EvidenceIntegrationResult,
    run_id: str,
) -> list[ExternalSignal]:
    owned_signals = integration.normalized_signals
    if lifecycle == BeliefLifecycle.NATIVE_V02:
        return _validate_owned_closed_signals(
            inbox_signals,
            owned_signals,
            run_id=run_id,
        )

    # Explicit legacy list-return compatibility may lack a normalized ownership
    # envelope. In either form it may ledger only recursively valid safe values.
    legacy_signals = inbox_signals if owned_signals is None else owned_signals
    validated: list[ExternalSignal] = []
    for signal in legacy_signals:
        try:
            payload = signal.model_dump(mode="python")
        except (AttributeError, TypeError, ValueError):
            raise ValueError("legacy closed signal ownership is invalid") from None
        if contains_secret_material(payload):
            raise ValueError(
                "legacy closed signals contain secret material"
            ) from None
        try:
            validated.append(ExternalSignal.model_validate(payload))
        except (TypeError, ValueError):
            raise ValueError("legacy closed signal ownership is invalid") from None
    return validated


def _validate_owned_closed_signals(
    inbox_signals: list[ExternalSignal],
    owned_signals: list[ExternalSignal] | None,
    *,
    run_id: str,
) -> list[ExternalSignal]:
    try:
        if not isinstance(owned_signals, list) or len(owned_signals) != len(
            inbox_signals
        ):
            raise ValueError
        normalizer = SignalProvenanceNormalizer()
        expected_signals = [
            normalizer.normalize(signal, run_id=run_id)
            for signal in inbox_signals
        ]
        validated_signals: list[ExternalSignal] = []
        for expected, owned in zip(expected_signals, owned_signals, strict=True):
            if not isinstance(owned, ExternalSignal):
                raise ValueError
            payload = owned.model_dump(mode="python")
            if contains_secret_material(payload):
                raise ValueError
            validated = ExternalSignal.model_validate(payload)
            if validated != expected:
                raise ValueError
            validated_signals.append(validated)
        return validated_signals
    except (AttributeError, TypeError, ValueError):
        raise ValueError("native closed signal ownership is invalid") from None


def _validate_native_evidence_events(
    events: list[EvidenceEvent],
    *,
    belief_state: BeliefState,
) -> None:
    frame_state = belief_state.frame_state
    requires_unresolved = (
        frame_state is not None
        and frame_state.competition == HypothesisCompetition.EXCLUSIVE
        and frame_state.coverage == HypothesisCoverage.OPEN
    )
    for event in events:
        if event.schema_version != "v0.2":
            raise ValueError("native evidence event contract is invalid")
        try:
            validated = EvidenceEvent.model_validate(
                event.model_dump(mode="python")
            )
        except (AttributeError, TypeError, ValueError):
            raise ValueError("native evidence event contract is invalid") from None
        if (
            validated.schema_version != "v0.2"
            or validated.effective_update_weight is None
            or (requires_unresolved and validated.unresolved_likelihood is None)
            or (
                not requires_unresolved
                and validated.unresolved_likelihood is not None
            )
        ):
            raise ValueError("native evidence event contract is invalid")


def _canonical_new_evidence_events(
    existing_ids: list[str],
    events: list[EvidenceEvent],
) -> list[EvidenceEvent]:
    seen = set(existing_ids)
    canonical: list[EvidenceEvent] = []
    for event in events:
        if event.id in seen:
            continue
        canonical.append(event)
        seen.add(event.id)
    return canonical
