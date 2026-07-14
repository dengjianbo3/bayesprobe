from __future__ import annotations

import pytest

from bayesprobe.evaluation.contracts import EvaluationCase
from bayesprobe.evaluation.search_arms import BayesProbeSearchArm, DirectSearchArm
from bayesprobe.schemas import EpistemicOrigin, HypothesisStatus
from bayesprobe.tavily_search import (
    TavilySearchResponse,
    TavilySearchResult,
)


class SearchClient:
    def __init__(self, responses: list[TavilySearchResponse]) -> None:
        self.responses = list(responses)
        self.queries: list[str] = []

    def search(self, request):
        self.queries.append(request.query)
        return self.responses.pop(0)


class DirectSearchGateway:
    adapter_kind = "direct-search-test"

    def __init__(self) -> None:
        self.requests = []

    def complete_structured(self, request):
        self.requests.append(request)
        if request.task == "plan_web_search":
            return {"query": f"query {len(self.requests)}"}
        if request.task == "answer_multiple_choice_with_search":
            return {
                "answer_label": "B",
                "choice_probabilities": {"A": 0.1, "B": 0.8, "C": 0.1},
                "answer_summary": "Retrieved sources support B.",
            }
        raise AssertionError(f"unexpected task: {request.task}")


class BayesProbeSearchGateway:
    adapter_kind = "bayesprobe-search-test"

    def __init__(self) -> None:
        self.requests = []

    def complete_structured(self, request):
        self.requests.append(request)
        if request.task == "plan_web_search":
            return {"query": "authoritative source for B"}
        if request.task == "judge_evidence":
            targets = request.input["target_hypotheses"]
            return {
                "evidence_type": "supporting",
                "likelihoods": {
                    target: (
                        "moderately_confirming"
                        if target == "B"
                        else "moderately_disconfirming"
                    )
                    for target in targets
                },
                "unresolved_likelihood": None,
                "frame_fit": "explained_by_named",
                "unexplained_observation": None,
                "interpretation": "The retrieved source favors B.",
                "quality_overrides": {},
            }
        raise AssertionError(f"unexpected task: {request.task}")


class AnomalousBayesProbeSearchGateway(BayesProbeSearchGateway):
    def complete_structured(self, request):
        if request.task != "judge_evidence":
            return super().complete_structured(request)
        self.requests.append(request)
        targets = request.input["target_hypotheses"]
        return {
            "evidence_type": "anomaly",
            "likelihoods": {
                target: "moderately_disconfirming" for target in targets
            },
            "unresolved_likelihood": None,
            "frame_fit": "explained_by_named",
            "unexplained_observation": "The source does not fit any answer choice.",
            "interpretation": "The retrieved source is anomalous for the choices.",
            "quality_overrides": {},
        }


class RetiringBayesProbeSearchGateway(BayesProbeSearchGateway):
    def complete_structured(self, request):
        if request.task != "judge_evidence":
            return super().complete_structured(request)
        self.requests.append(request)
        targets = request.input["target_hypotheses"]
        return {
            "evidence_type": "counterevidence",
            "likelihoods": {
                target: (
                    "strongly_disconfirming"
                    if target == "A"
                    else "strongly_confirming"
                )
                for target in targets
            },
            "unresolved_likelihood": None,
            "frame_fit": "explained_by_named",
            "unexplained_observation": None,
            "interpretation": "Independent retrieved sources refute A.",
            "quality_overrides": {},
        }


def _case() -> EvaluationCase:
    return EvaluationCase(
        sample_id="search_synthetic_1",
        question="Which option is supported?",
        choices={"A": "First", "B": "Second", "C": "Third"},
    )


def _response(url: str) -> TavilySearchResponse:
    return TavilySearchResponse(
        query="test query",
        outcome="success",
        results=(
            TavilySearchResult(
                url=url,
                title="Authoritative source",
                content="A factual observation supporting the second option.",
                score=0.9,
            ),
        ),
    )


def _response_bundle(*urls: str) -> TavilySearchResponse:
    return TavilySearchResponse(
        query="test query",
        outcome="success",
        results=tuple(_response(url).results[0] for url in urls),
    )


def test_direct_second_query_receives_first_packet_and_stays_within_budget():
    model = DirectSearchGateway()
    arm = DirectSearchArm(
        model,
        SearchClient([_response("https://source.test/one"), _response("https://source.test/two")]),
    )

    result = arm.run_case(_case())

    planner_requests = [request for request in model.requests if request.task == "plan_web_search"]
    assert result.state == "completed"
    assert result.process_metrics["search_calls"] == 2
    assert planner_requests[1].input["prior_search_packets"]
    assert model.requests[-1].task == "answer_multiple_choice_with_search"


def test_bayesprobe_search_calls_tavily_only_inside_probe_execution():
    model = BayesProbeSearchGateway()
    run_results = []
    arm = BayesProbeSearchArm(
        model,
        SearchClient([_response("https://source.test/one"), _response("https://source.test/two")]),
        run_result_observer=run_results.append,
    )

    result = arm.run_case(_case())

    observed_origins = {
        signal.provenance.epistemic_origin
        for cycle in run_results[0].cycle_results
        for signal in cycle.signals
        if signal.provenance is not None
    }
    assert result.state == "completed"
    assert result.process_metrics["search_calls"] <= 2
    assert observed_origins == {EpistemicOrigin.RETRIEVED_SOURCE}
    assert all(request.task != "execute_probe" for request in model.requests)


def test_all_operational_search_failures_are_treatment_not_delivered():
    result = DirectSearchArm(
        DirectSearchGateway(),
        SearchClient([TavilySearchResponse(query="test", outcome="provider_error")]),
    ).run_case(_case())

    assert result.state == "terminal_failed"
    assert result.error_category == "search_treatment_not_delivered"


def test_multiple_choice_search_anomaly_cannot_spawn_an_out_of_contract_answer():
    runs = []
    result = BayesProbeSearchArm(
        AnomalousBayesProbeSearchGateway(),
        SearchClient(
            [
                _response("https://source.test/anomaly-one"),
                _response("https://source.test/anomaly-two"),
            ]
        ),
        run_result_observer=runs.append,
    ).run_case(_case())

    assert result.state == "completed"
    assert set(runs[0].final_belief_state.hypotheses_by_id()) == set(
        _case().choice_labels
    )
    assert all(
        evolution.operation.value != "spawn"
        for cycle in runs[0].cycle_results
        for evolution in cycle.hypothesis_evolutions
    )


def test_multiple_choice_search_counterevidence_cannot_retire_a_declared_answer():
    runs = []
    result = BayesProbeSearchArm(
        RetiringBayesProbeSearchGateway(),
        SearchClient(
            [
                _response_bundle(
                    "https://first-source.test/counter-one",
                    "https://second-source.test/counter-one",
                ),
                _response_bundle(
                    "https://third-source.test/counter-two",
                    "https://fourth-source.test/counter-two",
                ),
            ]
        ),
        run_result_observer=runs.append,
    ).run_case(_case())

    final_hypotheses = runs[0].final_belief_state.hypotheses_by_id()
    assert result.state == "completed"
    assert set(final_hypotheses) == set(_case().choice_labels)
    assert all(
        hypothesis.status == HypothesisStatus.ACTIVE
        for hypothesis in final_hypotheses.values()
    )
    assert final_hypotheses["A"].posterior < 0.2
    assert sum(result.probabilities.values()) == pytest.approx(1.0)
    assert not any(
        cycle.hypothesis_evolutions for cycle in runs[0].cycle_results
    )
