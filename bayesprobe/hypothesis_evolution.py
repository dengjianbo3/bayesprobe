from __future__ import annotations

from dataclasses import dataclass, field

from bayesprobe.belief import normalize_hypotheses
from bayesprobe.schemas import (
    BeliefState,
    BeliefUpdate,
    CycleRecord,
    EvidenceEvent,
    EvidenceType,
    EvolutionOperation,
    Hypothesis,
    HypothesisEvolution,
    HypothesisStatus,
    ProbeCandidate,
    ProbeDesign,
    UpdateDirection,
)


@dataclass(frozen=True)
class HypothesisEvolutionConfig:
    spawn_prior: float = 0.12
    reframe_drop_threshold: float = 0.08
    reframe_min_previous_posterior: float = 0.6
    retire_posterior_threshold: float = 0.2
    retire_min_independent_counterevents: int = 2
    independent_event_threshold: float = 0.5


@dataclass(frozen=True)
class HypothesisEvolutionResult:
    hypotheses: list[Hypothesis]
    evolutions: list[HypothesisEvolution]
    probe_candidates: list[ProbeCandidate] = field(default_factory=list)

    def hypotheses_by_id(self) -> dict[str, Hypothesis]:
        return {hypothesis.id: hypothesis for hypothesis in self.hypotheses}


class HypothesisEvolutionEngine:
    def __init__(self, *, config: HypothesisEvolutionConfig | None = None) -> None:
        self.config = config or HypothesisEvolutionConfig()

    def evolve(
        self,
        *,
        cycle: CycleRecord,
        previous_belief_state: BeliefState,
        updated_hypotheses: list[Hypothesis],
        evidence_events: list[EvidenceEvent],
        belief_updates: list[BeliefUpdate],
    ) -> HypothesisEvolutionResult:
        if previous_belief_state.task_frame is None:
            raise ValueError("belief state requires hypothesis relation metadata")
        relation = previous_belief_state.task_frame.hypothesis_frame.relation
        hypotheses = list(updated_hypotheses)
        evolutions: list[HypothesisEvolution] = []
        probe_candidates: list[ProbeCandidate] = []

        for event in evidence_events:
            if event.evidence_type == EvidenceType.ANOMALY:
                spawn = self._spawn_from_anomaly(
                    cycle=cycle,
                    previous_belief_state=previous_belief_state,
                    event=event,
                )
                if spawn.hypothesis.id not in {hypothesis.id for hypothesis in hypotheses}:
                    hypotheses.append(spawn.hypothesis)
                evolutions.append(spawn.evolution)
                probe_candidates.append(spawn.probe_candidate)

        hypotheses, retire_evolutions = self._retire_stale_hypotheses(
            cycle=cycle,
            hypotheses=hypotheses,
            evidence_events=evidence_events,
        )
        evolutions.extend(retire_evolutions)

        reframe_result = self._reframe_scoped_hypotheses(
            cycle=cycle,
            previous_belief_state=previous_belief_state,
            hypotheses=hypotheses,
            evidence_events=evidence_events,
            belief_updates=belief_updates,
        )
        hypotheses = reframe_result.hypotheses
        evolutions.extend(reframe_result.evolutions)
        probe_candidates.extend(reframe_result.probe_candidates)

        return HypothesisEvolutionResult(
            hypotheses=normalize_hypotheses(hypotheses, relation=relation),
            evolutions=evolutions,
            probe_candidates=probe_candidates,
        )

    def _spawn_from_anomaly(
        self,
        *,
        cycle: CycleRecord,
        previous_belief_state: BeliefState,
        event: EvidenceEvent,
    ) -> _SpawnResult:
        hypothesis_id = f"H_{event.id}_spawned"
        rival_ids = [hypothesis.id for hypothesis in previous_belief_state.hypotheses]
        reason = "Anomaly has low likelihood under all active hypotheses."
        required_next_probe = "probe anomaly boundary condition"
        evolution = HypothesisEvolution(
            evolution_id=f"{event.id}_HE",
            cycle_id=cycle.cycle_id,
            operation=EvolutionOperation.SPAWN,
            from_hypothesis=None,
            to_hypothesis=hypothesis_id,
            triggered_by=[event.id],
            reason=reason,
            audit_fields={
                "why_existing_hypotheses_failed": reason,
                "new_hypothesis_prior": self.config.spawn_prior,
                "required_next_probe": required_next_probe,
                "trigger_event_type": event.evidence_type.value,
            },
        )
        hypothesis = Hypothesis(
            id=hypothesis_id,
            statement=f"Spawned anomaly hypothesis for {', '.join(evolution.triggered_by)}.",
            scope=f"Anomaly follow-up from cycle {cycle.cycle_id}.",
            prior=self.config.spawn_prior,
            posterior=self.config.spawn_prior,
            rivals=rival_ids,
            falsifiers=["A better-targeted probe explains the anomaly under an existing hypothesis."],
            predictions=[required_next_probe],
            created_by="spawned",
            why_existing_hypotheses_failed=reason,
        )
        probe_candidate = ProbeCandidate(
            candidate_id=f"pc_{event.id}_anomaly_followup",
            source="anomaly",
            candidate_probe=ProbeDesign(
                id=f"P_{event.id}_anomaly_followup",
                cycle_id=cycle.cycle_id,
                target_hypotheses=[hypothesis_id],
                inquiry_goal=f"Probe whether anomaly {event.id} supports the spawned hypothesis.",
                method="anomaly_followup",
                support_condition={hypothesis_id: "The anomaly repeats or is independently explained."},
                weaken_condition={hypothesis_id: "The anomaly is explained by an existing hypothesis."},
                expected_information_gain=0.8,
                decision_relevance=0.75,
                cost_estimate=0.45,
                priority=0.8,
            ),
            priority_features={
                "trigger_event_id": event.id,
                "expected_information_gain": 0.8,
                "attacks_top_hypothesis": False,
            },
        )
        return _SpawnResult(
            hypothesis=hypothesis,
            evolution=evolution,
            probe_candidate=probe_candidate,
        )

    def _retire_stale_hypotheses(
        self,
        *,
        cycle: CycleRecord,
        hypotheses: list[Hypothesis],
        evidence_events: list[EvidenceEvent],
    ) -> tuple[list[Hypothesis], list[HypothesisEvolution]]:
        evolutions: list[HypothesisEvolution] = []
        retired_by_id: dict[str, Hypothesis] = {}
        for hypothesis in hypotheses:
            if hypothesis.posterior >= self.config.retire_posterior_threshold:
                continue
            counterevents = [
                event
                for event in evidence_events
                if event.evidence_type == EvidenceType.COUNTEREVIDENCE
                and hypothesis.id in event.target_hypotheses
                and event.independence >= self.config.independent_event_threshold
            ]
            if len(counterevents) < self.config.retire_min_independent_counterevents:
                continue
            retired = hypothesis.model_copy(update={"status": HypothesisStatus.RETIRED})
            retired_by_id[hypothesis.id] = retired
            counterevent_ids = [event.id for event in counterevents]
            evolutions.append(
                HypothesisEvolution(
                    evolution_id=f"{cycle.cycle_id}_{hypothesis.id}_retire_HE",
                    cycle_id=cycle.cycle_id,
                    operation=EvolutionOperation.RETIRE,
                    from_hypothesis=hypothesis.id,
                    to_hypothesis=None,
                    triggered_by=counterevent_ids,
                    reason=(
                        f"{hypothesis.id} posterior fell below retirement threshold "
                        "under independent counterevidence."
                    ),
                    audit_fields={
                        "retired_posterior": hypothesis.posterior,
                        "independent_counterevidence_count": len(counterevents),
                        "counterevidence_event_ids": counterevent_ids,
                    },
                )
            )

        if not retired_by_id:
            return hypotheses, []
        return [
            retired_by_id.get(hypothesis.id, hypothesis)
            for hypothesis in hypotheses
        ], evolutions

    def _reframe_scoped_hypotheses(
        self,
        *,
        cycle: CycleRecord,
        previous_belief_state: BeliefState,
        hypotheses: list[Hypothesis],
        evidence_events: list[EvidenceEvent],
        belief_updates: list[BeliefUpdate],
    ) -> HypothesisEvolutionResult:
        previous_by_id = previous_belief_state.hypotheses_by_id()
        current_by_id = {hypothesis.id: hypothesis for hypothesis in hypotheses}
        counterevent_ids = {
            event.id
            for event in evidence_events
            if event.evidence_type == EvidenceType.COUNTEREVIDENCE
        }
        materialized = list(hypotheses)
        evolutions: list[HypothesisEvolution] = []
        probe_candidates: list[ProbeCandidate] = []

        for update in belief_updates:
            if update.direction != UpdateDirection.WEAKENED:
                continue
            if update.evidence_id not in counterevent_ids:
                continue
            previous = previous_by_id.get(update.hypothesis_id)
            current = current_by_id.get(update.hypothesis_id)
            if previous is None or current is None:
                continue
            if previous.posterior < self.config.reframe_min_previous_posterior:
                continue
            if current.status == HypothesisStatus.RETIRED:
                continue
            posterior_drop = round(update.prior - update.posterior, 4)
            if posterior_drop < self.config.reframe_drop_threshold:
                continue
            if not previous.scope.strip():
                continue
            reframed_id = f"H_{previous.id}_{cycle.cycle_id}_reframed"
            if reframed_id in current_by_id:
                continue

            reframed = Hypothesis(
                id=reframed_id,
                statement=f"Reframed scope of {previous.id}: {previous.statement}",
                scope=f"Narrowed scope after counterevidence: {previous.scope}",
                prior=min(max(current.posterior, 0.05), 0.95),
                posterior=min(max(current.posterior, 0.05), 0.95),
                rivals=[previous.id, *previous.rivals],
                falsifiers=[
                    *previous.falsifiers,
                    "A scope-disambiguation probe fails to distinguish this reframe from the original.",
                ],
                predictions=[
                    *previous.predictions,
                    "Scope-specific evidence should explain the counterevidence pattern.",
                ],
                created_by="reframed",
                why_existing_hypotheses_failed=(
                    f"{previous.id} weakened by counterevidence {update.evidence_id}."
                ),
            )
            materialized.append(reframed)
            current_by_id[reframed.id] = reframed
            required_next_probe = "probe scope boundary and rival explanation"
            evolutions.append(
                HypothesisEvolution(
                    evolution_id=f"{cycle.cycle_id}_{previous.id}_reframe_HE",
                    cycle_id=cycle.cycle_id,
                    operation=EvolutionOperation.REFRAME,
                    from_hypothesis=previous.id,
                    to_hypothesis=reframed.id,
                    triggered_by=[update.evidence_id],
                    reason=f"{previous.id} needs a narrower scope after counterevidence.",
                    audit_fields={
                        "from_statement": previous.statement,
                        "from_scope": previous.scope,
                        "new_scope": reframed.scope,
                        "posterior_drop": posterior_drop,
                        "required_next_probe": required_next_probe,
                    },
                )
            )
            probe_candidates.append(
                ProbeCandidate(
                    candidate_id=f"pc_{cycle.cycle_id}_{previous.id}_reframe_scope",
                    source="uncertainty",
                    candidate_probe=ProbeDesign(
                        id=f"P_{cycle.cycle_id}_{previous.id}_reframe_scope",
                        cycle_id=cycle.cycle_id,
                        target_hypotheses=[previous.id, reframed.id],
                        inquiry_goal=(
                            f"Test whether {reframed.id} better handles counterevidence "
                            f"than {previous.id}."
                        ),
                        method="scope_disambiguation",
                        support_condition={
                            reframed.id: "Scope-specific evidence explains the counterevidence.",
                        },
                        weaken_condition={
                            previous.id: "Counterevidence still applies under the original broad scope.",
                            reframed.id: "The scope change does not explain the counterevidence.",
                        },
                        expected_information_gain=0.7,
                        decision_relevance=0.75,
                        cost_estimate=0.45,
                        priority=0.72,
                    ),
                    priority_features={
                        "posterior_drop": posterior_drop,
                        "trigger_update_id": update.update_id,
                        "attacks_top_hypothesis": True,
                    },
                )
            )

        return HypothesisEvolutionResult(
            hypotheses=materialized,
            evolutions=evolutions,
            probe_candidates=probe_candidates,
        )


@dataclass(frozen=True)
class _SpawnResult:
    hypothesis: Hypothesis
    evolution: HypothesisEvolution
    probe_candidate: ProbeCandidate


__all__ = [
    "HypothesisEvolutionConfig",
    "HypothesisEvolutionEngine",
    "HypothesisEvolutionResult",
]
