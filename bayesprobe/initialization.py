from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bayesprobe.belief import summarize_hypotheses
from bayesprobe.ledger import JsonlLedgerStore
from bayesprobe.schemas import (
    AnswerChoice,
    BeliefState,
    EvidenceMemorySnapshot,
    FrameAdequacyStatus,
    FrameState,
    FramedHypothesis,
    Hypothesis,
    ProbeCandidate,
    ProbeDesign,
    RunRecord,
    RunRegime,
    RunStatus,
    TaskFrame,
    TaskAdmissionDecision,
    TaskAdmissionStatus,
    TaskKind,
    HypothesisCompetition,
    HypothesisRelation,
    is_secret_like_value,
)
from bayesprobe.task_framing import (
    ExplicitTaskFramer,
    HypothesisSeed,
    TaskFramer,
    TaskFramingError,
    TaskFramingInput,
    parse_legacy_answer_choice_frame,
)
from bayesprobe.task_admission import (
    ExplicitTaskAdmitter,
    TaskAdmitter,
    TaskAdmissionError,
    TaskAdmissionInput,
)


INITIAL_CYCLE_ID = "cycle_0"
INITIALIZATION_METHOD = "task_frame_v0.2"


@dataclass(frozen=True)
class InitializeRunInput:
    run_id: str
    problem: str
    context: str = ""
    task_context: str = ""
    answer_choices: list[AnswerChoice] = field(default_factory=list)
    regime: RunRegime = RunRegime.AUTONOMOUS
    hypothesis_seeds: list[HypothesisSeed] = field(default_factory=list)
    task_kind: TaskKind | None = None
    hypothesis_relation: HypothesisRelation | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InitializationResult:
    run: RunRecord
    task_frame: TaskFrame
    belief_state: BeliefState
    probe_candidates: list[ProbeCandidate]


class BayesProbeInitializer:
    def __init__(
        self,
        ledger: JsonlLedgerStore | None = None,
        task_framer: TaskFramer | None = None,
        task_admitter: TaskAdmitter | None = None,
    ) -> None:
        self._ledger = ledger
        self._task_framer = task_framer or ExplicitTaskFramer()
        self._task_admitter = task_admitter or ExplicitTaskAdmitter()

    def initialize(
        self,
        input: InitializeRunInput,
        admission_decision: TaskAdmissionDecision | None = None,
    ) -> InitializationResult:
        validate_initialize_run_input_security(input)
        run_id = _clean_required(input.run_id, "run_id")
        problem = _clean_required(input.problem, "problem")
        decision = admission_decision or self._assess_admission(
            input,
            run_id=run_id,
            problem=problem,
        )
        if decision.status != TaskAdmissionStatus.ADMITTED:
            raise TaskFramingError("initializer requires an admitted task decision")
        task_frame = self._task_framer.frame(
            TaskFramingInput(
                run_id=run_id,
                question=problem,
                admission_decision=decision,
                task_context=input.task_context,
                answer_choices=list(input.answer_choices),
                hypothesis_seeds=list(input.hypothesis_seeds),
                task_kind=input.task_kind,
                hypothesis_relation=input.hypothesis_relation,
                metadata=dict(input.metadata),
            )
        )
        hypotheses = [
            _hypothesis_from_frame(
                item,
                task_frame.hypothesis_frame.rival_sets[item.id],
            )
            for item in task_frame.hypothesis_frame.hypotheses
        ]
        metadata = {
            **input.metadata,
            "initialization_method": INITIALIZATION_METHOD,
            "context_provided": bool(input.context.strip()),
            "hypothesis_count": len(hypotheses),
            "seeded_hypotheses": bool(input.hypothesis_seeds),
            "question_frame": (
                "multiple_choice"
                if task_frame.task_kind == TaskKind.MULTIPLE_CHOICE
                else "explicit_task_frame"
            ),
            "task_kind": task_frame.task_kind.value,
            "hypothesis_competition": task_frame.hypothesis_frame.competition.value,
            "hypothesis_coverage": task_frame.hypothesis_frame.coverage.value,
            "framing_method": task_frame.framing_method.value,
        }
        run = RunRecord(
            run_id=run_id,
            regime=input.regime,
            problem=problem,
            status=RunStatus.RUNNING,
            current_cycle_id=INITIAL_CYCLE_ID,
            metadata=metadata,
        )
        compatibility_relation = _compatibility_relation(task_frame)
        belief_summary, uncertainty_summary = summarize_hypotheses(
            hypotheses,
            relation=compatibility_relation,
        )
        frame_state = FrameState(
            frame_id=task_frame.hypothesis_frame.frame_id,
            competition=task_frame.hypothesis_frame.competition,
            coverage=task_frame.hypothesis_frame.coverage,
            active_hypothesis_ids=[item.id for item in hypotheses],
            unresolved_alternative_mass=(
                task_frame.hypothesis_frame.unresolved_alternative_mass
            ),
            adequacy_status=FrameAdequacyStatus.PROVISIONAL,
        )
        evidence_memory = EvidenceMemorySnapshot()
        belief_state = BeliefState(
            schema_version="v0.2",
            belief_state_id=f"{run_id}_bs_0",
            run_id=run_id,
            cycle_id=INITIAL_CYCLE_ID,
            cycle_index=0,
            hypotheses=hypotheses,
            task_frame=task_frame,
            frame_state=frame_state,
            evidence_memory=evidence_memory,
            posterior_summary={
                **belief_summary,
                "initialization_method": INITIALIZATION_METHOD,
                "hypothesis_count": len(hypotheses),
                "priors": {hypothesis.id: hypothesis.prior for hypothesis in hypotheses},
            },
            uncertainty_summary=(
                f"{uncertainty_summary} No external signals have been integrated yet."
            ),
        )
        probe_candidates = _initial_probe_candidates(
            run_id=run_id,
            problem=problem,
            hypotheses=hypotheses,
            is_multiple_choice=task_frame.task_kind == TaskKind.MULTIPLE_CHOICE,
        )
        self._append_ledger(
            admission_decision=decision,
            task_frame=task_frame,
            run=run,
            belief_state=belief_state,
            probe_candidates=probe_candidates,
        )
        return InitializationResult(
            run=run,
            task_frame=task_frame,
            belief_state=belief_state,
            probe_candidates=probe_candidates,
        )

    def _assess_admission(
        self,
        input: InitializeRunInput,
        *,
        run_id: str,
        problem: str,
    ) -> TaskAdmissionDecision:
        choices = list(input.answer_choices)
        if not choices:
            parsed = parse_legacy_answer_choice_frame(problem)
            if parsed is not None:
                choices = list(parsed.choices)
        try:
            return self._task_admitter.assess(
                TaskAdmissionInput(
                    attempt_id=f"{run_id}_admission",
                    question=problem,
                    task_context=input.task_context,
                    answer_choices=choices,
                    hypothesis_seeds=list(input.hypothesis_seeds),
                    requested_output_shape=_requested_output_shape(input.metadata),
                    model_metadata={
                        "task_kind": input.task_kind.value
                        if input.task_kind is not None
                        else None
                    },
                )
            )
        except TaskAdmissionError as error:
            raise TaskFramingError(str(error)) from None

    def _append_ledger(
        self,
        *,
        admission_decision: TaskAdmissionDecision,
        task_frame: TaskFrame,
        run: RunRecord,
        belief_state: BeliefState,
        probe_candidates: list[ProbeCandidate],
    ) -> None:
        if self._ledger is None:
            return
        self._ledger.append("task_admission", admission_decision)
        self._ledger.append("task_frame", task_frame)
        self._ledger.append("run", run)
        self._ledger.append("belief_state", belief_state)
        for candidate in probe_candidates:
            self._ledger.append("probe_candidate", candidate)


def validate_compatibility_context_security(context: str) -> None:
    if not isinstance(context, str):
        raise TaskFramingError("compatibility context must be a string")
    if is_secret_like_value(context):
        raise TaskFramingError(
            "compatibility context must not contain secret material"
        )


def validate_initialize_run_input_security(input: InitializeRunInput) -> None:
    validate_compatibility_context_security(input.context)


def _requested_output_shape(metadata: dict[str, Any]) -> str | None:
    value = metadata.get("requested_output_shape")
    return value if isinstance(value, str) and value.strip() else None


def _compatibility_relation(task_frame: TaskFrame) -> HypothesisRelation:
    if task_frame.hypothesis_frame.competition == HypothesisCompetition.INDEPENDENT:
        return HypothesisRelation.INDEPENDENT
    return HypothesisRelation.EXCLUSIVE_EXHAUSTIVE


def _clean_required(value: str, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must not be empty")
    return cleaned


def _hypothesis_from_frame(
    framed: FramedHypothesis,
    rivals: list[str],
) -> Hypothesis:
    return Hypothesis(
        id=framed.id,
        statement=framed.statement,
        scope=framed.scope,
        prior=framed.initial_prior,
        posterior=framed.initial_prior,
        rivals=list(rivals),
        falsifiers=list(framed.falsifiers),
        predictions=list(framed.predictions),
        created_by="initial",
        answer_value=framed.answer_value,
    )


def _initial_probe_candidates(
    *,
    run_id: str,
    problem: str,
    hypotheses: list[Hypothesis],
    is_multiple_choice: bool,
) -> list[ProbeCandidate]:
    candidates: list[ProbeCandidate] = []
    if is_multiple_choice:
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


__all__ = [
    "BayesProbeInitializer",
    "HypothesisSeed",
    "InitializationResult",
    "InitializeRunInput",
    "validate_compatibility_context_security",
    "validate_initialize_run_input_security",
]
