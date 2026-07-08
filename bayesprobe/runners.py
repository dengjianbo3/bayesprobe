from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from bayesprobe.controllers import AutonomousController, ControllerResult
from bayesprobe.core import BayesProbeCore
from bayesprobe.schemas import AnswerProjection, BeliefState, ExternalSignal, Hypothesis


class AutonomousSignalProvider(Protocol):
    def collect_signals(
        self,
        *,
        run_id: str,
        cycle_index: int,
        belief_state: BeliefState,
        previous_answer: AnswerProjection | None,
    ) -> list[ExternalSignal]:
        ...


@dataclass(frozen=True)
class AutonomousLoopConfig:
    max_cycles: int = 3
    stop_on_no_signals: bool = True
    confidence_threshold: float | None = None
    posterior_delta_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.max_cycles < 1:
            raise ValueError("max_cycles must be at least 1")
        if self.confidence_threshold is not None and not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if self.posterior_delta_threshold is not None and self.posterior_delta_threshold < 0:
            raise ValueError("posterior_delta_threshold must be non-negative")


class AutonomousStopReason(StrEnum):
    MAX_CYCLES = "max_cycles"
    NO_SIGNALS = "no_signals"
    CONFIDENCE_REACHED = "confidence_reached"
    POSTERIOR_STABLE = "posterior_stable"


@dataclass(frozen=True)
class AutonomousRunResult:
    run_id: str
    initial_belief_state: BeliefState
    final_belief_state: BeliefState
    cycle_results: list[ControllerResult]
    final_answer_projection: AnswerProjection | None
    stop_reason: AutonomousStopReason


class AutonomousLoopRunner:
    def __init__(
        self,
        core: BayesProbeCore,
        config: AutonomousLoopConfig | None = None,
    ) -> None:
        self.core = core
        self.config = config or AutonomousLoopConfig()
        self._controller = AutonomousController(core=core)

    def run(
        self,
        *,
        run_id: str,
        initial_belief_state: BeliefState,
        signal_provider: AutonomousSignalProvider,
    ) -> AutonomousRunResult:
        current_belief_state = initial_belief_state
        cycle_results: list[ControllerResult] = []
        previous_answer: AnswerProjection | None = None

        for _ in range(self.config.max_cycles):
            previous_belief_state = current_belief_state
            signals = signal_provider.collect_signals(
                run_id=run_id,
                cycle_index=current_belief_state.cycle_index + 1,
                belief_state=current_belief_state,
                previous_answer=previous_answer,
            )
            if not signals and self.config.stop_on_no_signals:
                return AutonomousRunResult(
                    run_id=run_id,
                    initial_belief_state=initial_belief_state,
                    final_belief_state=current_belief_state,
                    cycle_results=list(cycle_results),
                    final_answer_projection=previous_answer,
                    stop_reason=AutonomousStopReason.NO_SIGNALS,
                )

            cycle_result = self._controller.run_once(
                run_id=run_id,
                belief_state=current_belief_state,
                active_signals=signals,
            )
            cycle_results.append(cycle_result)
            current_belief_state = cycle_result.belief_state
            previous_answer = cycle_result.answer_projection

            if self._confidence_reached(current_belief_state):
                return AutonomousRunResult(
                    run_id=run_id,
                    initial_belief_state=initial_belief_state,
                    final_belief_state=current_belief_state,
                    cycle_results=list(cycle_results),
                    final_answer_projection=previous_answer,
                    stop_reason=AutonomousStopReason.CONFIDENCE_REACHED,
                )

            if self._posterior_stable(
                previous=previous_belief_state,
                current=current_belief_state,
            ):
                return AutonomousRunResult(
                    run_id=run_id,
                    initial_belief_state=initial_belief_state,
                    final_belief_state=current_belief_state,
                    cycle_results=list(cycle_results),
                    final_answer_projection=previous_answer,
                    stop_reason=AutonomousStopReason.POSTERIOR_STABLE,
                )

        return AutonomousRunResult(
            run_id=run_id,
            initial_belief_state=initial_belief_state,
            final_belief_state=current_belief_state,
            cycle_results=list(cycle_results),
            final_answer_projection=previous_answer,
            stop_reason=AutonomousStopReason.MAX_CYCLES,
        )

    def _confidence_reached(self, belief_state: BeliefState) -> bool:
        threshold = self.config.confidence_threshold
        if threshold is None:
            return False
        return _top_hypothesis(belief_state).posterior >= threshold

    def _posterior_stable(self, *, previous: BeliefState, current: BeliefState) -> bool:
        threshold = self.config.posterior_delta_threshold
        if threshold is None:
            return False
        return _posterior_delta_is_stable(previous=previous, current=current, threshold=threshold)


def _top_hypothesis(belief_state: BeliefState) -> Hypothesis:
    return max(belief_state.hypotheses, key=lambda hypothesis: hypothesis.posterior)


def _posterior_delta_is_stable(
    *,
    previous: BeliefState,
    current: BeliefState,
    threshold: float,
) -> bool:
    previous_by_id = previous.hypotheses_by_id()
    current_by_id = current.hypotheses_by_id()
    continuing_ids = set(previous_by_id).intersection(current_by_id)
    if not continuing_ids:
        return False
    return all(
        abs(current_by_id[hypothesis_id].posterior - previous_by_id[hypothesis_id].posterior) <= threshold
        for hypothesis_id in continuing_ids
    )


__all__ = [
    "AutonomousLoopConfig",
    "AutonomousLoopRunner",
    "AutonomousRunResult",
    "AutonomousSignalProvider",
    "AutonomousStopReason",
]
