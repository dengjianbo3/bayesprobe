from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from typing import Any

from bayesprobe.core import BayesProbeCore
from bayesprobe.evaluation.arms import (
    _ContextualModelGateway,
    _case_run_id,
    _result_from_payload,
    _run_process_metrics,
)
from bayesprobe.evaluation.contracts import ArmCaseResult, EvaluationCase
from bayesprobe.initialization import BayesProbeInitializer, InitializeRunInput
from bayesprobe.model_gateway import ModelGateway, ModelGatewayValidationError, StructuredModelRequest
from bayesprobe.probe_design import ProbeDesignContext, ProbeDesignResult, ProbeDesigner
from bayesprobe.probe_executor import ProbeExecutor
from bayesprobe.probe_planner import ProbePlanner, ProbePlanningResult
from bayesprobe.question_runner import (
    AutonomousQuestionRunConfig,
    AutonomousQuestionRunResult,
    AutonomousQuestionRunner,
)
from bayesprobe.schemas import (
    AnswerChoice,
    CapabilityDecision,
    CapabilityDescriptor,
    CapabilityKind,
    EpistemicOrigin,
    ProbeCandidate,
    ProbeDesign,
    ProbePurpose,
    ProbeSet,
)
from bayesprobe.tavily_probe import TavilyProbeToolGateway
from bayesprobe.tavily_search import (
    TavilySearchClient,
    TavilySearchRequest,
    TavilySearchResponse,
)


_SEARCH_CAPABILITY = CapabilityDescriptor(
    kind=CapabilityKind.SEARCH,
    available=True,
    cost_class="bounded",
    latency_class="interactive",
    epistemic_origin=EpistemicOrigin.RETRIEVED_SOURCE,
    quality_caps={"verifiability": 0.8, "independence": 0.7},
    executor_adapter_id="tavily_search:v1",
)


class DirectSearchArm:
    arm_name = "direct_search"

    def __init__(
        self,
        model_gateway: ModelGateway,
        tavily_client: TavilySearchClient,
        *,
        max_search_calls: int = 2,
        invocation_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if type(max_search_calls) is not int or max_search_calls < 1:
            raise ValueError("max_search_calls must be a positive integer")
        self._model_gateway = model_gateway
        self._client = tavily_client
        self._max_search_calls = max_search_calls
        self._invocation_metadata = dict(invocation_metadata or {})

    def run_case(self, case: EvaluationCase) -> ArmCaseResult:
        metadata = {
            **self._invocation_metadata,
            "arm": self.arm_name,
            "sample_id": case.sample_id,
        }
        packets: list[dict[str, Any]] = []
        metrics = _SearchMetrics()
        for round_index in range(1, self._max_search_calls + 1):
            query = _plan_direct_query(
                self._model_gateway,
                case=case,
                packets=packets,
                remaining_search_calls=self._max_search_calls - metrics.calls,
                metadata={**metadata, "search_round": round_index},
            )
            if query is None:
                metrics.malformed_queries += 1
                return _search_not_delivered(case, self.arm_name, metrics)
            response = _execute_search(self._client, query=query, metrics=metrics)
            if response is None:
                return _search_not_delivered(case, self.arm_name, metrics)
            packets.append(_packet_from_response(query, response, metrics))

        request = StructuredModelRequest(
            task="answer_multiple_choice_with_search",
            input={
                "question": case.question,
                "choices": dict(case.choices),
                "search_packets": packets,
            },
            prompt_id="direct_multiple_choice_with_search",
            prompt_version="v0.1",
            schema_name="MultipleChoiceAnswer",
            schema_version="v0.1",
            metadata=metadata,
        )
        try:
            payload = self._model_gateway.complete_structured(request)
            result = _result_from_payload(
                case,
                payload,
                arm=self.arm_name,
                model_calls=self._max_search_calls + 1,
                schema_repairs=0,
            )
        except (ModelGatewayValidationError, TypeError, ValueError):
            return ArmCaseResult(
                sample_id=case.sample_id,
                arm=self.arm_name,
                state="terminal_failed",
                answer_label=None,
                probabilities=None,
                error_category="structured_output_invalid",
                process_metrics=metrics.with_model_calls(self._max_search_calls + 1),
            )
        except Exception:
            return ArmCaseResult(
                sample_id=case.sample_id,
                arm=self.arm_name,
                state="terminal_failed",
                answer_label=None,
                probabilities=None,
                error_category="provider_error",
                process_metrics=metrics.with_model_calls(self._max_search_calls + 1),
            )
        return ArmCaseResult(
            sample_id=result.sample_id,
            arm=result.arm,
            state=result.state,
            answer_label=result.answer_label,
            probabilities=result.probabilities,
            answer_summary=result.answer_summary,
            process_metrics={
                **result.process_metrics,
                **metrics.without_model_calls(),
            },
        )


class BayesProbeSearchArm:
    arm_name = "bayesprobe_search"

    def __init__(
        self,
        model_gateway: ModelGateway,
        tavily_client: TavilySearchClient,
        *,
        max_search_calls: int = 2,
        invocation_metadata: Mapping[str, Any] | None = None,
        run_result_observer: Callable[[AutonomousQuestionRunResult], None] | None = None,
    ) -> None:
        if type(max_search_calls) is not int or max_search_calls < 1:
            raise ValueError("max_search_calls must be a positive integer")
        self._model_gateway = model_gateway
        self._client = tavily_client
        self._max_search_calls = max_search_calls
        self._invocation_metadata = dict(invocation_metadata or {})
        self._run_result_observer = run_result_observer
        self.run_config = AutonomousQuestionRunConfig(
            max_cycles=4,
            max_probes_per_cycle=1,
            stop_on_no_probes=True,
            confidence_threshold=None,
            posterior_delta_threshold=None,
        )

    def run_case(self, case: EvaluationCase) -> ArmCaseResult:
        run_id = _case_run_id(self.arm_name, case.sample_id)
        metadata = {
            **self._invocation_metadata,
            "arm": self.arm_name,
            "sample_id": case.sample_id,
            "run_id": run_id,
        }
        contextual_gateway = _ContextualModelGateway(
            self._model_gateway,
            metadata=metadata,
        )
        tavily_gateway = TavilyProbeToolGateway(
            contextual_gateway,
            self._client,
            max_search_calls=self._max_search_calls,
        )
        core = BayesProbeCore(model_gateway=contextual_gateway)
        runner = AutonomousQuestionRunner(
            core=core,
            initializer=BayesProbeInitializer(ledger=core.ledger),
            planner=_SearchBudgetProbePlanner(
                tavily_gateway,
                ledger=core.ledger,
            ),
            executor=ProbeExecutor(gateway=tavily_gateway, ledger=core.ledger),
            config=self.run_config,
            probe_designer=_TavilySearchProbeDesigner(tavily_gateway),
            available_capabilities=(_SEARCH_CAPABILITY,),
        )
        try:
            run_result = runner.run_question(
                InitializeRunInput(
                    run_id=run_id,
                    problem=case.question,
                    answer_choices=[
                        AnswerChoice(label=label, text=text)
                        for label, text in case.choices.items()
                    ],
                    metadata=metadata,
                )
            )
        except Exception:
            return ArmCaseResult(
                sample_id=case.sample_id,
                arm=self.arm_name,
                state="terminal_failed",
                answer_label=None,
                probabilities=None,
                error_category="provider_error",
                process_metrics=tavily_gateway.process_metrics,
            )
        self._observe_run_result(run_result)
        if tavily_gateway.process_metrics["search_successes"] == 0:
            return ArmCaseResult(
                sample_id=case.sample_id,
                arm=self.arm_name,
                state="terminal_failed",
                answer_label=None,
                probabilities=None,
                error_category="search_treatment_not_delivered",
                process_metrics=_run_process_metrics(
                    run_result,
                    python_metrics=tavily_gateway.process_metrics,
                ),
            )
        projection = run_result.final_answer_projection
        final_hypotheses = run_result.final_belief_state.hypotheses_by_id()
        if (
            projection is None
            or projection.current_best_hypothesis not in case.choices
            or set(final_hypotheses) != set(case.choice_labels)
        ):
            return ArmCaseResult(
                sample_id=case.sample_id,
                arm=self.arm_name,
                state="terminal_failed",
                answer_label=None,
                probabilities=None,
                error_category="invalid_final_hypothesis",
                process_metrics=_run_process_metrics(
                    run_result,
                    python_metrics=tavily_gateway.process_metrics,
                ),
            )
        return ArmCaseResult(
            sample_id=case.sample_id,
            arm=self.arm_name,
            state="completed",
            answer_label=projection.current_best_hypothesis,
            probabilities={
                label: final_hypotheses[label].posterior
                for label in case.choice_labels
            },
            answer_summary=projection.answer,
            process_metrics=_run_process_metrics(
                run_result,
                python_metrics=tavily_gateway.process_metrics,
            ),
        )

    def _observe_run_result(self, result: AutonomousQuestionRunResult) -> None:
        if self._run_result_observer is None:
            return
        try:
            self._run_result_observer(result)
        except Exception:
            return


class _TavilySearchProbeDesigner(ProbeDesigner):
    def __init__(self, gateway: TavilyProbeToolGateway) -> None:
        self._gateway = gateway

    def propose(self, context: ProbeDesignContext) -> ProbeDesignResult:
        if self._gateway.search_budget_remaining <= 0:
            return _empty_search_design_result()
        target_ids = [
            hypothesis.id
            for hypothesis in context.belief_state.hypotheses
            if hypothesis.id in set(context.belief_state.frame_state.active_hypothesis_ids)
        ]
        identity = hashlib.sha256(
            f"{context.cycle_id}:{','.join(target_ids)}".encode("utf-8")
        ).hexdigest()[:12]
        probe = ProbeDesign(
            id=f"P_{context.cycle_id}_{identity}",
            cycle_id=context.cycle_id,
            target_hypotheses=target_ids,
            inquiry_goal=(
                "Retrieve discriminative external source material for the active "
                "hypotheses."
            ),
            method="tavily_search",
            purpose=ProbePurpose.SOURCE_VERIFICATION,
            expected_observation=(
                "A source observation that favors or weakens at least one answer choice."
            ),
            required_capability=CapabilityKind.SEARCH,
            support_condition={
                hypothesis_id: "A retrieved source supports its answer claim."
                for hypothesis_id in target_ids
            },
            weaken_condition={
                hypothesis_id: "A retrieved source contradicts its answer claim."
                for hypothesis_id in target_ids
            },
            priority=0.85,
        )
        candidate = ProbeCandidate(
            candidate_id=f"C_{context.cycle_id}_{identity}",
            source="uncertainty",
            candidate_probe=probe,
            priority_features={"server_owned_priority": probe.priority},
        )
        return ProbeDesignResult(
            candidates=[candidate],
            capability_decisions=[
                CapabilityDecision(
                    kind=CapabilityKind.SEARCH,
                    available=True,
                    descriptor=_SEARCH_CAPABILITY,
                    reason="Tavily retrieval is available for source verification",
                )
            ],
        )


def _empty_search_design_result() -> ProbeDesignResult:
    return ProbeDesignResult(
        candidates=[],
        capability_decisions=[
            CapabilityDecision(
                kind=CapabilityKind.SEARCH,
                available=False,
                descriptor=_SEARCH_CAPABILITY,
                reason="Tavily search budget is exhausted",
            )
        ],
    )


class _SearchBudgetProbePlanner(ProbePlanner):
    def __init__(
        self,
        gateway: TavilyProbeToolGateway,
        *,
        ledger: Any = None,
    ) -> None:
        super().__init__(ledger=ledger)
        self._gateway = gateway

    def design_probe_set(self, **kwargs: Any) -> ProbePlanningResult:
        if self._gateway.search_budget_remaining > 0:
            return super().design_probe_set(**kwargs)
        cycle_id = kwargs["cycle_id"]
        return ProbePlanningResult(
            probe_set=ProbeSet(
                probe_set_id=f"ps_{cycle_id}",
                cycle_id=cycle_id,
                probes=[],
                selection_reason="Tavily search budget exhausted.",
                budget_allocated={"max_probes": 0, "selected_count": 0},
                may_be_empty=True,
            ),
            selected_candidates=[],
            rejected_candidates=[],
        )


class _SearchMetrics:
    def __init__(self) -> None:
        self.calls = 0
        self.successes = 0
        self.failures = 0
        self.empty_searches = 0
        self.result_count = 0
        self.unique_urls: set[str] = set()
        self.malformed_queries = 0

    def without_model_calls(self) -> dict[str, int]:
        return {
            "search_calls": self.calls,
            "search_successes": self.successes,
            "search_failures": self.failures,
            "empty_searches": self.empty_searches,
            "search_result_count": self.result_count,
            "unique_urls": len(self.unique_urls),
            "malformed_queries": self.malformed_queries,
            "search_budget_exhausted": 0,
        }

    def with_model_calls(self, model_calls: int) -> dict[str, int]:
        return {"model_calls": model_calls, **self.without_model_calls()}


def _plan_direct_query(
    model_gateway: ModelGateway,
    *,
    case: EvaluationCase,
    packets: list[dict[str, Any]],
    remaining_search_calls: int,
    metadata: Mapping[str, Any],
) -> str | None:
    request = StructuredModelRequest(
        task="plan_web_search",
        input={
            "problem": case.question,
            "choices": dict(case.choices),
            "inquiry_goal": "Find external evidence that discriminates the choices.",
            "prior_search_packets": packets,
            "remaining_search_calls": remaining_search_calls,
        },
        prompt_id="direct_web_search_query",
        prompt_version="v0.1",
        schema_name="WebSearchQuery",
        schema_version="v0.1",
        metadata=dict(metadata),
    )
    try:
        payload = model_gateway.complete_structured(request)
    except Exception:
        return None
    if not isinstance(payload, Mapping) or not isinstance(payload.get("query"), str):
        return None
    query = " ".join(payload["query"].split())
    return query if query and len(query) <= 400 else None


def _execute_search(
    client: TavilySearchClient,
    *,
    query: str,
    metrics: _SearchMetrics,
) -> TavilySearchResponse | None:
    metrics.calls += 1
    try:
        response = client.search(TavilySearchRequest(query=query))
    except Exception:
        metrics.failures += 1
        return None
    if response.outcome != "success":
        metrics.failures += 1
        return None
    if not response.results:
        metrics.successes += 1
        metrics.empty_searches += 1
        return None
    metrics.successes += 1
    return response


def _packet_from_response(
    query: str,
    response: TavilySearchResponse,
    metrics: _SearchMetrics,
) -> dict[str, Any]:
    results = []
    seen_urls: set[str] = set()
    for result in response.results:
        if result.url in seen_urls:
            continue
        seen_urls.add(result.url)
        metrics.result_count += 1
        metrics.unique_urls.add(result.url)
        results.append(
            {
                "url": result.url,
                "title": result.title,
                "content": result.content,
                "score": result.score,
            }
        )
    return {"query": query, "results": results}


def _search_not_delivered(
    case: EvaluationCase,
    arm: str,
    metrics: _SearchMetrics,
) -> ArmCaseResult:
    return ArmCaseResult(
        sample_id=case.sample_id,
        arm=arm,
        state="terminal_failed",
        answer_label=None,
        probabilities=None,
        error_category="search_treatment_not_delivered",
        process_metrics=metrics.with_model_calls(metrics.calls + 1),
    )


__all__ = ["BayesProbeSearchArm", "DirectSearchArm"]
