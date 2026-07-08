from __future__ import annotations

from dataclasses import dataclass

from bayesprobe.ledger import JsonlLedgerStore
from bayesprobe.schemas import BeliefState, Hypothesis, ProbeCandidate, ProbeSet


@dataclass(frozen=True)
class ProbePlanningConfig:
    max_probes: int = 2
    allow_empty: bool = False
    attack_top_hypothesis_bonus: float = 1.25
    unresolved_uncertainty_bonus: float = 1.1

    def __post_init__(self) -> None:
        if self.max_probes < 1:
            raise ValueError("max_probes must be at least 1")
        if self.attack_top_hypothesis_bonus <= 0:
            raise ValueError("attack_top_hypothesis_bonus must be positive")
        if self.unresolved_uncertainty_bonus <= 0:
            raise ValueError("unresolved_uncertainty_bonus must be positive")


@dataclass(frozen=True)
class RejectedProbeCandidate:
    candidate: ProbeCandidate
    reason: str
    score: float


@dataclass(frozen=True)
class ProbePlanningResult:
    probe_set: ProbeSet
    selected_candidates: list[ProbeCandidate]
    rejected_candidates: list[RejectedProbeCandidate]


@dataclass(frozen=True)
class _ScoredProbeCandidate:
    candidate: ProbeCandidate
    score: float


class ProbePlanner:
    def __init__(self, ledger: JsonlLedgerStore | None = None) -> None:
        self._ledger = ledger

    def design_probe_set(
        self,
        *,
        run_id: str,
        cycle_id: str,
        belief_state: BeliefState,
        candidates: list[ProbeCandidate],
        config: ProbePlanningConfig | None = None,
    ) -> ProbePlanningResult:
        clean_run_id = _clean_required(run_id, "run_id")
        clean_cycle_id = _clean_required(cycle_id, "cycle_id")
        planning_config = config or ProbePlanningConfig()
        top_hypothesis = _top_hypothesis(belief_state)
        valid_candidates, rejected_candidates = _score_valid_candidates(
            belief_state=belief_state,
            candidates=candidates,
            config=planning_config,
            top_hypothesis=top_hypothesis,
        )

        if not valid_candidates:
            if not planning_config.allow_empty:
                raise ValueError("no valid probe candidates available")
            result = self._empty_result(
                cycle_id=clean_cycle_id,
                candidates=candidates,
                rejected_candidates=rejected_candidates,
                config=planning_config,
            )
            self._append_probe_set(result.probe_set)
            return result

        ranked_candidates = sorted(
            valid_candidates,
            key=lambda item: (
                -item.score,
                item.candidate.candidate_probe.cost_estimate,
                item.candidate.candidate_id,
            ),
        )
        selected_scored = ranked_candidates[: planning_config.max_probes]
        selected_scored = _ensure_top_hypothesis_probe(
            selected=selected_scored,
            ranked=ranked_candidates,
            top_hypothesis_id=top_hypothesis.id,
            max_probes=planning_config.max_probes,
        )
        selected_ids = {item.candidate.candidate_id for item in selected_scored}
        selected_candidates = [
            _freeze_candidate(candidate=item.candidate, cycle_id=clean_cycle_id, belief_state=belief_state)
            for item in selected_scored
        ]
        rejected_candidates.extend(
            RejectedProbeCandidate(
                candidate=item.candidate,
                reason="not_selected_budget_limit",
                score=round(item.score, 6),
            )
            for item in ranked_candidates
            if item.candidate.candidate_id not in selected_ids
        )
        probe_set = ProbeSet(
            probe_set_id=f"ps_{clean_cycle_id}",
            cycle_id=clean_cycle_id,
            probes=[candidate.candidate_probe for candidate in selected_candidates],
            selection_reason=_selection_reason(
                run_id=clean_run_id,
                selected_candidates=selected_candidates,
                rejected_candidates=rejected_candidates,
            ),
            budget_allocated={
                "max_probes": planning_config.max_probes,
                "selected_count": len(selected_candidates),
                "candidate_count": len(candidates),
                "valid_candidate_count": len(valid_candidates),
            },
            may_be_empty=False,
        )
        result = ProbePlanningResult(
            probe_set=probe_set,
            selected_candidates=selected_candidates,
            rejected_candidates=rejected_candidates,
        )
        self._append_probe_set(probe_set)
        return result

    def _empty_result(
        self,
        *,
        cycle_id: str,
        candidates: list[ProbeCandidate],
        rejected_candidates: list[RejectedProbeCandidate],
        config: ProbePlanningConfig,
    ) -> ProbePlanningResult:
        probe_set = ProbeSet(
            probe_set_id=f"ps_{cycle_id}",
            cycle_id=cycle_id,
            probes=[],
            selection_reason="No valid probe candidates; empty ProbeSet allowed.",
            budget_allocated={
                "max_probes": config.max_probes,
                "selected_count": 0,
                "candidate_count": len(candidates),
                "valid_candidate_count": 0,
            },
            may_be_empty=True,
        )
        return ProbePlanningResult(
            probe_set=probe_set,
            selected_candidates=[],
            rejected_candidates=rejected_candidates,
        )

    def _append_probe_set(self, probe_set: ProbeSet) -> None:
        if self._ledger is None:
            return
        self._ledger.append("probe_set", probe_set)


def _clean_required(value: str, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must not be empty")
    return cleaned


def _score_valid_candidates(
    *,
    belief_state: BeliefState,
    candidates: list[ProbeCandidate],
    config: ProbePlanningConfig,
    top_hypothesis: Hypothesis,
) -> tuple[list[_ScoredProbeCandidate], list[RejectedProbeCandidate]]:
    known_hypotheses = set(belief_state.hypotheses_by_id())
    valid_candidates: list[_ScoredProbeCandidate] = []
    rejected_candidates: list[RejectedProbeCandidate] = []
    for candidate in candidates:
        targets = candidate.candidate_probe.target_hypotheses
        if not targets:
            rejected_candidates.append(
                RejectedProbeCandidate(candidate=candidate, reason="invalid_no_targets", score=0.0)
            )
            continue
        valid_targets = [target for target in targets if target in known_hypotheses]
        if not valid_targets:
            rejected_candidates.append(
                RejectedProbeCandidate(candidate=candidate, reason="invalid_unknown_targets", score=0.0)
            )
            continue
        valid_candidate = _with_known_targets(candidate=candidate, valid_targets=valid_targets)
        valid_candidates.append(
            _ScoredProbeCandidate(
                candidate=valid_candidate,
                score=_score_candidate(
                    belief_state=belief_state,
                    candidate=valid_candidate,
                    config=config,
                    top_hypothesis_id=top_hypothesis.id,
                ),
            )
        )
    return valid_candidates, rejected_candidates


def _score_candidate(
    *,
    belief_state: BeliefState,
    candidate: ProbeCandidate,
    config: ProbePlanningConfig,
    top_hypothesis_id: str,
) -> float:
    probe = candidate.candidate_probe
    score = probe.expected_information_gain * probe.decision_relevance
    if top_hypothesis_id in probe.target_hypotheses:
        score *= config.attack_top_hypothesis_bonus
    if belief_state.uncertainty_summary.strip():
        score *= config.unresolved_uncertainty_bonus
    score /= max(probe.cost_estimate, 0.01)
    return round(score, 6)


def _ensure_top_hypothesis_probe(
    *,
    selected: list[_ScoredProbeCandidate],
    ranked: list[_ScoredProbeCandidate],
    top_hypothesis_id: str,
    max_probes: int,
) -> list[_ScoredProbeCandidate]:
    if any(_targets_hypothesis(item.candidate, top_hypothesis_id) for item in selected):
        return selected
    top_targeting = [
        item for item in ranked if _targets_hypothesis(item.candidate, top_hypothesis_id)
    ]
    if not top_targeting:
        return selected
    top_item = top_targeting[0]
    remaining = [item for item in selected if item.candidate.candidate_id != top_item.candidate.candidate_id]
    return [top_item, *remaining][:max_probes]


def _targets_hypothesis(candidate: ProbeCandidate, hypothesis_id: str) -> bool:
    return hypothesis_id in candidate.candidate_probe.target_hypotheses


def _freeze_candidate(
    *,
    candidate: ProbeCandidate,
    cycle_id: str,
    belief_state: BeliefState,
) -> ProbeCandidate:
    known_hypotheses = set(belief_state.hypotheses_by_id())
    targets = [
        target
        for target in candidate.candidate_probe.target_hypotheses
        if target in known_hypotheses
    ]
    frozen_probe = candidate.candidate_probe.model_copy(
        update={
            "id": _freeze_probe_id(candidate.candidate_probe.id, cycle_id),
            "cycle_id": cycle_id,
            "target_hypotheses": targets,
        }
    )
    return candidate.model_copy(
        update={
            "candidate_probe": frozen_probe,
            "selected_in_cycle": cycle_id,
        }
    )


def _with_known_targets(*, candidate: ProbeCandidate, valid_targets: list[str]) -> ProbeCandidate:
    if valid_targets == candidate.candidate_probe.target_hypotheses:
        return candidate
    probe = candidate.candidate_probe.model_copy(update={"target_hypotheses": valid_targets})
    return candidate.model_copy(update={"candidate_probe": probe})


def _freeze_probe_id(probe_id: str, cycle_id: str) -> str:
    if cycle_id in probe_id:
        return probe_id
    return f"{probe_id}_{cycle_id}"


def _selection_reason(
    *,
    run_id: str,
    selected_candidates: list[ProbeCandidate],
    rejected_candidates: list[RejectedProbeCandidate],
) -> str:
    selected_ids = ", ".join(candidate.candidate_id for candidate in selected_candidates)
    rejected_count = len(rejected_candidates)
    return (
        f"Selected {selected_ids} for {run_id} using deterministic expected-value ranking; "
        f"rejected_count={rejected_count}."
    )


def _top_hypothesis(belief_state: BeliefState) -> Hypothesis:
    return max(belief_state.hypotheses, key=lambda hypothesis: hypothesis.posterior)


__all__ = [
    "ProbePlanner",
    "ProbePlanningConfig",
    "ProbePlanningResult",
    "RejectedProbeCandidate",
]
