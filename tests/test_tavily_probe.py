from __future__ import annotations

from bayesprobe.initialization import BayesProbeInitializer, InitializeRunInput
from bayesprobe.model_gateway import ScriptedModelGateway
from bayesprobe.probe_executor import build_probe_execution_brief
from bayesprobe.schemas import (
    AnswerChoice,
    EpistemicOrigin,
    ProbeDesign,
)
from bayesprobe.tavily_probe import TavilyProbeToolGateway
from bayesprobe.tavily_search import (
    TavilySearchResponse,
    TavilySearchResult,
)


class FakeTavilyClient:
    def __init__(self, response: TavilySearchResponse) -> None:
        self.response = response
        self.queries: list[str] = []

    def search(self, request):
        self.queries.append(request.query)
        return self.response


def _probe() -> ProbeDesign:
    return ProbeDesign(
        id="P_web",
        cycle_id="cycle_web",
        target_hypotheses=["A", "B"],
        inquiry_goal="Find a source that discriminates the two choices.",
        method="web_search",
        support_condition={"A": "An authoritative source supports A."},
        weaken_condition={"A": "An authoritative source contradicts A."},
    )


def _context():
    state = BayesProbeInitializer().initialize(
        InitializeRunInput(
            run_id="run_web",
            problem="Which answer choice is correct?",
            answer_choices=[
                AnswerChoice(label="A", text="First choice"),
                AnswerChoice(label="B", text="Second choice"),
            ],
        )
    ).belief_state
    return build_probe_execution_brief(
        run_id="run_web",
        cycle_id="cycle_web",
        belief_state=state,
        problem="Which answer choice is correct?",
    )


def _query_planner() -> ScriptedModelGateway:
    return ScriptedModelGateway(
        {"plan_web_search": {"query": "authoritative source first versus second"}}
    )


def test_probe_gateway_turns_each_url_into_a_retrieved_source_signal():
    client = FakeTavilyClient(
        TavilySearchResponse(
            query="authoritative source first versus second",
            outcome="success",
            results=(
                TavilySearchResult(
                    url="https://source.test/a",
                    title="Source A",
                    content="A relevant finding.",
                    score=0.9,
                ),
                TavilySearchResult(
                    url="https://source.test/b",
                    title="Source B",
                    content="Another finding.",
                    score=0.8,
                ),
            ),
        )
    )
    gateway = TavilyProbeToolGateway(_query_planner(), client)

    signals = gateway.execute_probe(probe=_probe(), context=_context())

    assert client.queries == ["authoritative source first versus second"]
    assert [signal.source for signal in signals] == [
        "https://source.test/a",
        "https://source.test/b",
    ]
    assert all(
        signal.provenance is not None
        and signal.provenance.epistemic_origin is EpistemicOrigin.RETRIEVED_SOURCE
        for signal in signals
    )
    assert signals[0].provenance.citations == ["https://source.test/a"]
    assert signals[0].generated_by_probe == "P_web"
    assert gateway.process_metrics == {
        "search_calls": 1,
        "search_successes": 1,
        "search_failures": 0,
        "empty_searches": 0,
        "search_result_count": 2,
        "unique_urls": 2,
        "malformed_queries": 0,
        "search_budget_exhausted": 0,
    }


def test_budget_failure_and_search_failure_return_no_reasoning_signal():
    client = FakeTavilyClient(
        TavilySearchResponse(
            query="unused",
            outcome="provider_error",
            error_message="provider unavailable",
        )
    )
    gateway = TavilyProbeToolGateway(_query_planner(), client, max_search_calls=1)

    assert gateway.execute_probe(probe=_probe(), context=_context()) == []
    assert gateway.execute_probe(probe=_probe(), context=_context()) == []

    assert client.queries == ["authoritative source first versus second"]
    assert gateway.process_metrics == {
        "search_calls": 1,
        "search_successes": 0,
        "search_failures": 1,
        "empty_searches": 0,
        "search_result_count": 0,
        "unique_urls": 0,
        "malformed_queries": 0,
        "search_budget_exhausted": 1,
    }


def test_malformed_planner_output_returns_no_signal_without_searching():
    client = FakeTavilyClient(
        TavilySearchResponse(query="unused", outcome="success")
    )
    gateway = TavilyProbeToolGateway(
        ScriptedModelGateway({"plan_web_search": {"query": " "}}),
        client,
    )

    assert gateway.execute_probe(probe=_probe(), context=_context()) == []

    assert client.queries == []
    assert gateway.process_metrics["malformed_queries"] == 1
