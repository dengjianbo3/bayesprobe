from __future__ import annotations

import re
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
_ANSWER_CHOICES_HEADER_RE = re.compile(r"\banswer\s+choices?\s*:\s*", re.IGNORECASE)
_CHOICE_BLOCK_LINE_RE = re.compile(
    r"^\s*([A-Z])[\.\)]\s+(.*?)(?=^\s*[A-Z][\.\)]\s+|\Z)",
    re.MULTILINE | re.DOTALL,
)
_CHOICE_INLINE_RE = re.compile(
    r"(?:^|\s)([A-Z])[\.\)]\s+(.*?)(?=\s+[A-Z][\.\)]\s+|\Z)",
    re.DOTALL,
)


@dataclass(frozen=True)
class HypothesisSeed:
    statement: str
    id: str | None = None
    scope: str | None = None
    prior: float | None = None
    falsifiers: list[str] = field(default_factory=list)
    predictions: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _AnswerChoice:
    label: str
    text: str


@dataclass(frozen=True)
class _AnswerChoiceFrame:
    stem: str
    choices: list[_AnswerChoice]


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
        answer_choice_frame = (
            None if input.hypothesis_seeds else _parse_answer_choice_frame(problem)
        )
        hypotheses = _build_hypotheses(
            input=input,
            problem=problem,
            answer_choice_frame=answer_choice_frame,
        )
        metadata = {
            **input.metadata,
            "initialization_method": INITIALIZATION_METHOD,
            "context_provided": bool(input.context.strip()),
            "hypothesis_count": len(hypotheses),
            "seeded_hypotheses": bool(input.hypothesis_seeds),
            "question_frame": "multiple_choice" if answer_choice_frame else "binary_claim",
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
        probe_candidates = _initial_probe_candidates(
            run_id=run_id,
            problem=problem,
            hypotheses=hypotheses,
            answer_choice_frame=answer_choice_frame,
        )
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


def _default_seeds(
    problem: str,
    *,
    answer_choice_frame: _AnswerChoiceFrame | None = None,
) -> list[HypothesisSeed]:
    if answer_choice_frame is not None:
        return [
            HypothesisSeed(
                id=choice.label,
                statement=f"Answer choice {choice.label} is correct: {choice.text}",
                scope=(
                    f"Assess whether answer choice {choice.label} correctly answers: "
                    f"{answer_choice_frame.stem}"
                ),
                prior=None,
                falsifiers=[
                    f"Another answer choice is better supported than {choice.label}.",
                    f"A counterexample rules out answer choice {choice.label}.",
                ],
                predictions=[
                    f"Reliable reasoning should make answer choice {choice.label} more plausible than its rivals."
                ],
            )
            for choice in answer_choice_frame.choices
        ]
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


def _build_hypotheses(
    *,
    input: InitializeRunInput,
    problem: str,
    answer_choice_frame: _AnswerChoiceFrame | None = None,
) -> list[Hypothesis]:
    seeds = (
        list(input.hypothesis_seeds)
        if input.hypothesis_seeds
        else _default_seeds(problem, answer_choice_frame=answer_choice_frame)
    )
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


def _parse_answer_choice_frame(problem: str) -> _AnswerChoiceFrame | None:
    header = _ANSWER_CHOICES_HEADER_RE.search(problem)
    if header is None:
        return None
    stem = _collapse_whitespace(problem[: header.start()])
    choice_text = problem[header.end():].strip()
    if not stem or not choice_text:
        return None
    choices = _parse_choice_lines(choice_text)
    if len(choices) < 2:
        return None
    return _AnswerChoiceFrame(stem=stem, choices=choices)


def _parse_choice_lines(choice_text: str) -> list[_AnswerChoice]:
    matches = list(_CHOICE_BLOCK_LINE_RE.finditer(choice_text))
    if not matches:
        matches = list(_CHOICE_INLINE_RE.finditer(choice_text))
    choices: list[_AnswerChoice] = []
    used_labels: set[str] = set()
    for match in matches:
        label = match.group(1).strip().upper()
        text = _collapse_whitespace(match.group(2))
        if not text or label in used_labels:
            continue
        choices.append(_AnswerChoice(label=label, text=text))
        used_labels.add(label)
    return choices


def _collapse_whitespace(value: str) -> str:
    return " ".join(value.split())


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


def _initial_probe_candidates(
    *,
    run_id: str,
    problem: str,
    hypotheses: list[Hypothesis],
    answer_choice_frame: _AnswerChoiceFrame | None,
) -> list[ProbeCandidate]:
    candidates: list[ProbeCandidate] = []
    if answer_choice_frame is not None:
        candidates.append(
            _answer_choice_discriminator_candidate(
                run_id=run_id,
                problem=problem,
                hypotheses=hypotheses,
            )
        )
    candidates.extend(
        _probe_candidate(run_id=run_id, problem=problem, hypothesis=hypothesis)
        for hypothesis in hypotheses
    )
    return candidates


def _answer_choice_discriminator_candidate(
    *,
    run_id: str,
    problem: str,
    hypotheses: list[Hypothesis],
) -> ProbeCandidate:
    hypothesis_ids = [hypothesis.id for hypothesis in hypotheses]
    support_condition = {
        hypothesis.id: f"Answer choice {hypothesis.id} is the best supported option."
        for hypothesis in hypotheses
    }
    weaken_condition = {
        hypothesis.id: f"Another answer choice is better supported than {hypothesis.id}."
        for hypothesis in hypotheses
    }
    candidate_summaries = "\n".join(
        f"- {hypothesis.id}: {hypothesis.statement}" for hypothesis in hypotheses
    )
    return ProbeCandidate(
        candidate_id=f"pc_{run_id}_{INITIAL_CYCLE_ID}_answer_choices",
        source="manual",
        candidate_probe=ProbeDesign(
            id=f"P_{run_id}_{INITIAL_CYCLE_ID}_answer_choices",
            cycle_id=INITIAL_CYCLE_ID,
            target_hypotheses=hypothesis_ids,
            inquiry_goal=(
                "Determine which answer choice is best for the problem.\n"
                f"Problem:\n{problem}\n"
                f"Candidate hypotheses:\n{candidate_summaries}"
            ),
            method="answer_choice_discrimination",
            support_condition=support_condition,
            weaken_condition=weaken_condition,
            expected_information_gain=0.95,
            decision_relevance=0.95,
            cost_estimate=0.3,
            priority=0.95,
        ),
        priority_features={
            "initialization_method": INITIALIZATION_METHOD,
            "question_frame": "multiple_choice",
            "probe_role": "answer_choice_discriminator",
            "target_hypotheses": hypothesis_ids,
        },
    )


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
            inquiry_goal=(
                f"Find a signal that can support or weaken {hypothesis.id}.\n"
                f"Hypothesis: {hypothesis.statement}\n"
                f"Problem: {problem}"
            ),
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
