from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bayesprobe.ledger import JsonlLedgerStore
from bayesprobe.schemas import (
    BeliefState,
    Hypothesis,
    ProbeCandidate,
    ProbeDesign,
    RunRecord,
    RunRegime,
    RunStatus,
)


INITIAL_CYCLE_ID = "cycle_0"
INITIALIZATION_METHOD = "deterministic_mvp"


@dataclass(frozen=True)
class HypothesisSeed:
    statement: str
    id: str | None = None
    scope: str | None = None
    prior: float | None = None
    falsifiers: list[str] = field(default_factory=list)
    predictions: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class InitializeRunInput:
    run_id: str
    problem: str
    context: str = ""
    regime: RunRegime = RunRegime.AUTONOMOUS
    hypothesis_seeds: list[HypothesisSeed] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InitializationResult:
    run: RunRecord
    belief_state: BeliefState
    probe_candidates: list[ProbeCandidate]


class BayesProbeInitializer:
    def __init__(self, ledger: JsonlLedgerStore | None = None) -> None:
        self._ledger = ledger

    def initialize(self, input: InitializeRunInput) -> InitializationResult:
        run_id = _clean_required(input.run_id, "run_id")
        problem = _clean_required(input.problem, "problem")
        hypotheses = _build_hypotheses(input=input, problem=problem)
        metadata = {
            **input.metadata,
            "initialization_method": INITIALIZATION_METHOD,
            "context_provided": bool(input.context.strip()),
            "hypothesis_count": len(hypotheses),
            "seeded_hypotheses": bool(input.hypothesis_seeds),
        }
        run = RunRecord(
            run_id=run_id,
            regime=input.regime,
            problem=problem,
            status=RunStatus.RUNNING,
            current_cycle_id=INITIAL_CYCLE_ID,
            metadata=metadata,
        )
        belief_state = BeliefState(
            belief_state_id=f"{run_id}_bs_0",
            run_id=run_id,
            cycle_id=INITIAL_CYCLE_ID,
            cycle_index=0,
            hypotheses=hypotheses,
            posterior_summary={
                "initialization_method": INITIALIZATION_METHOD,
                "hypothesis_count": len(hypotheses),
                "top_hypothesis": _top_hypothesis_id(hypotheses),
                "priors": {hypothesis.id: hypothesis.prior for hypothesis in hypotheses},
            },
            uncertainty_summary=(
                f"Initial rival hypotheses for {problem}; no external signals have been integrated yet."
            ),
        )
        probe_candidates = [
            _probe_candidate(run_id=run_id, problem=problem, hypothesis=hypothesis)
            for hypothesis in hypotheses
        ]
        self._append_ledger(run=run, belief_state=belief_state, probe_candidates=probe_candidates)
        return InitializationResult(
            run=run,
            belief_state=belief_state,
            probe_candidates=probe_candidates,
        )

    def _append_ledger(
        self,
        *,
        run: RunRecord,
        belief_state: BeliefState,
        probe_candidates: list[ProbeCandidate],
    ) -> None:
        if self._ledger is None:
            return
        self._ledger.append("run", run)
        self._ledger.append("belief_state", belief_state)
        for candidate in probe_candidates:
            self._ledger.append("probe_candidate", candidate)


def _clean_required(value: str, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must not be empty")
    return cleaned


def _validate_seed(seed: HypothesisSeed) -> None:
    _clean_required(seed.statement, "hypothesis seed statement")
    if seed.prior is not None and not 0 <= seed.prior <= 1:
        raise ValueError("hypothesis seed prior must be between 0 and 1")


def _default_seeds(problem: str) -> list[HypothesisSeed]:
    return [
        HypothesisSeed(
            id="H1",
            statement=f"The claim or problem direction is supported: {problem}",
            scope=f"Assess conditions under which the claim holds for: {problem}",
            prior=0.5,
            falsifiers=["Reliable counterevidence shows the claim is false or materially misleading."],
            predictions=["Independent supporting signals should align with the claim direction."],
        ),
        HypothesisSeed(
            id="H2",
            statement=f"The claim or problem direction is refuted or materially misleading: {problem}",
            scope=f"Assess conditions under which the claim fails for: {problem}",
            prior=0.5,
            falsifiers=["Reliable supporting evidence shows the claim direction is true enough to proceed."],
            predictions=["Independent counterevidence should challenge the claim direction."],
        ),
    ]


def _build_hypotheses(*, input: InitializeRunInput, problem: str) -> list[Hypothesis]:
    seeds = list(input.hypothesis_seeds) if input.hypothesis_seeds else _default_seeds(problem)
    if len(seeds) < 2:
        raise ValueError("initialization requires at least two rival hypotheses")
    for seed in seeds:
        _validate_seed(seed)

    generated_ids = _hypothesis_ids(seeds)
    default_prior = round(1 / len(seeds), 4)
    hypotheses: list[Hypothesis] = []
    for index, seed in enumerate(seeds):
        hypothesis_id = generated_ids[index]
        prior = default_prior if seed.prior is None else seed.prior
        rivals = [other_id for other_id in generated_ids if other_id != hypothesis_id]
        hypotheses.append(
            Hypothesis(
                id=hypothesis_id,
                statement=seed.statement.strip(),
                scope=seed.scope.strip() if seed.scope and seed.scope.strip() else f"Initial frame for: {problem}",
                prior=prior,
                posterior=prior,
                rivals=rivals,
                falsifiers=seed.falsifiers
                or [f"A reliable signal weakens {hypothesis_id} within the problem frame."],
                predictions=seed.predictions
                or [f"A reliable signal should make {hypothesis_id} more plausible than its rivals."],
                created_by="initial",
            )
        )
    return hypotheses


def _hypothesis_ids(seeds: list[HypothesisSeed]) -> list[str]:
    ids: list[str] = []
    used: set[str] = set()
    for index, seed in enumerate(seeds, start=1):
        preferred_id = seed.id.strip() if seed.id and seed.id.strip() else f"H{index}"
        hypothesis_id = preferred_id
        suffix = 2
        while hypothesis_id in used:
            hypothesis_id = f"{preferred_id}_{suffix}"
            suffix += 1
        ids.append(hypothesis_id)
        used.add(hypothesis_id)
    return ids


def _probe_candidate(*, run_id: str, problem: str, hypothesis: Hypothesis) -> ProbeCandidate:
    probe_id = f"P_{run_id}_{INITIAL_CYCLE_ID}_{hypothesis.id}"
    support_condition = hypothesis.predictions[0] if hypothesis.predictions else "Independent support appears."
    weaken_condition = hypothesis.falsifiers[0] if hypothesis.falsifiers else "Reliable counterevidence appears."
    return ProbeCandidate(
        candidate_id=f"pc_{run_id}_{INITIAL_CYCLE_ID}_{hypothesis.id}",
        source="manual",
        candidate_probe=ProbeDesign(
            id=probe_id,
            cycle_id=INITIAL_CYCLE_ID,
            target_hypotheses=[hypothesis.id],
            inquiry_goal=f"Find a signal that can support or weaken {hypothesis.id} for: {problem}",
            method="source_tracing",
            support_condition={hypothesis.id: support_condition},
            weaken_condition={hypothesis.id: weaken_condition},
        ),
        priority_features={
            "initialization_method": INITIALIZATION_METHOD,
            "target_hypothesis": hypothesis.id,
        },
    )


def _top_hypothesis_id(hypotheses: list[Hypothesis]) -> str:
    return max(hypotheses, key=lambda hypothesis: hypothesis.posterior).id


__all__ = [
    "BayesProbeInitializer",
    "HypothesisSeed",
    "InitializationResult",
    "InitializeRunInput",
]
