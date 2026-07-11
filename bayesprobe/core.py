from __future__ import annotations

from dataclasses import dataclass, field

from bayesprobe.belief import (
    mark_replayed_evidence_events,
    solve_updates,
    summarize_hypotheses,
)
from bayesprobe.evidence import EvidenceIntegrationGate, EvidenceIntegrationResult
from bayesprobe.hypothesis_evolution import HypothesisEvolutionEngine
from bayesprobe.inbox import SignalInbox
from bayesprobe.ledger import JsonlLedgerStore
from bayesprobe.model_gateway import EvidenceJudgmentRepairPolicy, ModelGateway
from bayesprobe.schemas import (
    BeliefState,
    BeliefUpdate,
    BoundaryStatus,
    CycleRecord,
    CycleSignalShape,
    EvidenceEvent,
    ExternalSignal,
    HypothesisEvolution,
    ProbeCandidate,
    ProbeSet,
    SignalKind,
    utc_now,
)
from bayesprobe.task_framing import migrate_legacy_belief_state


@dataclass(frozen=True)
class CycleResult:
    cycle: CycleRecord
    belief_state: BeliefState
    evidence_events: list[EvidenceEvent]
    belief_updates: list[BeliefUpdate]
    hypothesis_evolutions: list[HypothesisEvolution]
    probe_candidates: list[ProbeCandidate] = field(default_factory=list)


class BayesProbeCore:
    def __init__(
        self,
        ledger: JsonlLedgerStore | None = None,
        model_gateway: ModelGateway | None = None,
        judgment_repair_policy: EvidenceJudgmentRepairPolicy | None = None,
    ) -> None:
        self._ledger = ledger
        self._model_gateway = model_gateway
        self._judgment_repair_policy = judgment_repair_policy
        self._cycle_allocations: dict[str, int] = {}
        self._evidence_gate = self._create_evidence_integration_gate()
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
        belief_state = migrate_legacy_belief_state(belief_state)
        self._validate_cycle_boundary(cycle=cycle, belief_state=belief_state, probe_set=probe_set)
        inbox = self._create_signal_inbox(cycle)
        for signal in signals:
            inbox.add(signal)
        normalized_signals = inbox.close()
        closed_cycle = cycle.model_copy(
            update={
                "boundary_status": BoundaryStatus.CLOSED,
                "boundary_closed_at": utc_now(),
            }
        )
        self._validate_signal_shape(cycle=closed_cycle, signals=normalized_signals)
        integration = self._normalize_evidence_integration(
            self._evidence_gate.integrate(
                cycle=closed_cycle,
                belief_state=belief_state,
                probe_set=probe_set,
                signals=normalized_signals,
            )
        )
        evidence_events = mark_replayed_evidence_events(
            belief_state,
            integration.evidence_events,
        )
        canonical_evidence_events = _canonical_new_evidence_events(
            belief_state.ledger_refs.get("evidence_events", []),
            evidence_events,
        )
        probe_candidates = integration.probe_candidates
        updated_hypotheses, belief_updates = solve_updates(
            run_id=cycle.run_id,
            cycle_id=cycle.cycle_id,
            belief_state=belief_state,
            events=evidence_events,
        )
        evolution_result = self._evolution_policy.evolve(
            cycle=closed_cycle,
            previous_belief_state=belief_state,
            updated_hypotheses=updated_hypotheses,
            evidence_events=evidence_events,
            belief_updates=belief_updates,
        )
        relation = belief_state.task_frame.hypothesis_frame.relation
        evolved_hypotheses = evolution_result.hypotheses
        evolutions = evolution_result.evolutions
        probe_candidates = [
            *probe_candidates,
            *evolution_result.probe_candidates,
        ]
        existing_ledger_refs = dict(belief_state.ledger_refs)
        merged_ledger_refs = dict(existing_ledger_refs)
        merged_ledger_refs["probe_sets"] = [
            *existing_ledger_refs.get("probe_sets", []),
            probe_set.probe_set_id,
        ]
        merged_ledger_refs["evidence_events"] = _append_unique(
            existing_ledger_refs.get("evidence_events", []),
            [event.id for event in canonical_evidence_events],
        )
        merged_ledger_refs["belief_updates"] = [
            *existing_ledger_refs.get("belief_updates", []),
            *(update.update_id for update in belief_updates),
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
            relation=relation,
        )
        updated_state = belief_state.model_copy(
            update={
                "belief_state_id": f"{cycle.run_id}_bs_{cycle.cycle_index}",
                "cycle_id": cycle.cycle_id,
                "cycle_index": cycle.cycle_index,
                "hypotheses": evolved_hypotheses,
                "posterior_summary": posterior_summary,
                "uncertainty_summary": uncertainty_summary,
                "ledger_refs": merged_ledger_refs,
            }
        )
        integrated_cycle = closed_cycle.model_copy(
            update={
                "boundary_status": BoundaryStatus.INTEGRATED,
                "completed_at": utc_now(),
            }
        )
        self._append_ledger_records(
            cycle=integrated_cycle,
            signals=normalized_signals,
            probe_set=probe_set,
            evidence_events=canonical_evidence_events,
            belief_updates=belief_updates,
            evolutions=evolutions,
            probe_candidates=probe_candidates,
            belief_state=updated_state,
        )
        return CycleResult(
            cycle=integrated_cycle,
            belief_state=updated_state,
            evidence_events=evidence_events,
            belief_updates=belief_updates,
            hypothesis_evolutions=evolutions,
            probe_candidates=probe_candidates,
        )

    def _create_signal_inbox(self, cycle: CycleRecord) -> SignalInbox:
        return SignalInbox(cycle_id=cycle.cycle_id)

    def _create_evidence_integration_gate(self) -> EvidenceIntegrationGate:
        return EvidenceIntegrationGate(
            model_gateway=self._model_gateway,
            judgment_repair_policy=self._judgment_repair_policy,
        )

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


def _scoped_cycle_key(run_id: str, cycle_id: str) -> str:
    if cycle_id.startswith(f"{run_id}_"):
        return cycle_id
    return f"{run_id}_{cycle_id}"


def _append_unique(existing: list[str], additions: list[str]) -> list[str]:
    result = list(existing)
    seen = set(result)
    for item in additions:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


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
