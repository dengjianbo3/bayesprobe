from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from bayesprobe.evidence_memory import derive_deterministic_computation_root
from bayesprobe.model_gateway import ModelGateway, StructuredModelRequest
from bayesprobe.probe_executor import ProbeExecutionBrief
from bayesprobe.schemas import (
    EpistemicOrigin,
    ExternalSignal,
    ProbeDesign,
    SignalKind,
    SignalProvenance,
)
from bayesprobe.tavily_search import (
    TavilySearchClient,
    TavilySearchRequest,
    TavilySearchResult,
)


_TAVILY_TOOL_IDENTITY = "tavily_search:v1"


class TavilyProbeToolGateway:
    """Plan one web query, retrieve it, and expose results only as raw signals."""

    def __init__(
        self,
        model_gateway: ModelGateway,
        tavily_client: TavilySearchClient,
        *,
        max_search_calls: int = 2,
    ) -> None:
        if type(max_search_calls) is not int or max_search_calls < 1:
            raise ValueError("max_search_calls must be a positive integer")
        self._model_gateway = model_gateway
        self._client = tavily_client
        self._max_search_calls = max_search_calls
        self._search_calls = 0
        self._search_successes = 0
        self._search_failures = 0
        self._empty_searches = 0
        self._search_result_count = 0
        self._unique_urls: set[str] = set()
        self._malformed_queries = 0
        self._search_budget_exhausted = 0

    @property
    def process_metrics(self) -> dict[str, int]:
        return {
            "search_calls": self._search_calls,
            "search_successes": self._search_successes,
            "search_failures": self._search_failures,
            "empty_searches": self._empty_searches,
            "search_result_count": self._search_result_count,
            "unique_urls": len(self._unique_urls),
            "malformed_queries": self._malformed_queries,
            "search_budget_exhausted": self._search_budget_exhausted,
        }

    def execute_probe(
        self,
        *,
        probe: ProbeDesign,
        context: ProbeExecutionBrief,
    ) -> list[ExternalSignal]:
        if self._search_calls >= self._max_search_calls:
            self._search_budget_exhausted += 1
            return []
        query = self._plan_query(probe=probe, context=context)
        if query is None:
            self._malformed_queries += 1
            return []
        self._search_calls += 1
        try:
            response = self._client.search(TavilySearchRequest(query=query))
        except Exception:
            self._search_failures += 1
            return []
        if response.outcome != "success":
            self._search_failures += 1
            return []
        self._search_successes += 1
        if not response.results:
            self._empty_searches += 1
            return []

        signals: list[ExternalSignal] = []
        response_urls: set[str] = set()
        for index, result in enumerate(response.results, start=1):
            if result.url in response_urls:
                continue
            response_urls.add(result.url)
            self._search_result_count += 1
            self._unique_urls.add(result.url)
            signals.append(
                self._signal_from_result(
                    result=result,
                    index=index,
                    query=query,
                    probe=probe,
                    context=context,
                )
            )
        if not signals:
            self._empty_searches += 1
        return signals

    def _plan_query(
        self,
        *,
        probe: ProbeDesign,
        context: ProbeExecutionBrief,
    ) -> str | None:
        request = StructuredModelRequest(
            task="plan_web_search",
            input={
                "problem": context.problem,
                "task_context": context.task_context,
                "inquiry_goal": probe.inquiry_goal,
                "probe": {
                    "id": probe.id,
                    "method": probe.method,
                    "target_hypotheses": list(probe.target_hypotheses),
                    "support_condition": dict(probe.support_condition),
                    "weaken_condition": dict(probe.weaken_condition),
                },
                "hypotheses": [
                    {
                        "id": hypothesis.id,
                        "statement": hypothesis.statement,
                        "scope": hypothesis.scope,
                        "predictions": list(hypothesis.predictions),
                        "falsifiers": list(hypothesis.falsifiers),
                    }
                    for hypothesis in context.hypotheses
                ],
                "remaining_search_calls": self._max_search_calls - self._search_calls,
            },
            prompt_id="web_search_query",
            prompt_version=context.provider_schema_version,
            schema_name="WebSearchQuery",
            schema_version=context.provider_schema_version,
            metadata={
                "run_id": context.run_id,
                "cycle_id": context.cycle_id,
                "probe_id": probe.id,
            },
        )
        try:
            payload = self._model_gateway.complete_structured(request)
        except Exception:
            return None
        return _query_from_payload(payload)

    def _signal_from_result(
        self,
        *,
        result: TavilySearchResult,
        index: int,
        query: str,
        probe: ProbeDesign,
        context: ProbeExecutionBrief,
    ) -> ExternalSignal:
        source_identity = f"tavily-url:{result.url}"
        derivation_root_id = derive_deterministic_computation_root(
            tool_identity=_TAVILY_TOOL_IDENTITY,
            computation_inputs={
                "query": query,
                "url": result.url,
            },
        )
        signal_digest = hashlib.sha256(
            f"{context.cycle_id}:{probe.id}:{result.url}".encode("utf-8")
        ).hexdigest()[:12]
        return ExternalSignal(
            id=f"S_{context.cycle_id}_{probe.id}_{index}_{signal_digest}",
            cycle_id=context.cycle_id,
            signal_kind=SignalKind.ACTIVE,
            source_type="retrieved_web_source",
            source=result.url,
            raw_content=_raw_source_content(result),
            generated_by_probe=probe.id,
            initial_target_hypotheses=list(probe.target_hypotheses),
            provenance=SignalProvenance(
                epistemic_origin=EpistemicOrigin.RETRIEVED_SOURCE,
                source_identity=source_identity,
                provider_model_or_tool_identity=_TAVILY_TOOL_IDENTITY,
                derivation_root_id=derivation_root_id,
                correlation_group=source_identity,
                canonical_content_fingerprint="pending-normalization",
                citations=[result.url],
            ),
        )


def _query_from_payload(payload: Any) -> str | None:
    if not isinstance(payload, Mapping):
        return None
    query = payload.get("query")
    if not isinstance(query, str):
        return None
    clean_query = " ".join(query.split())
    if not clean_query or len(clean_query) > 400:
        return None
    return clean_query


def _raw_source_content(result: TavilySearchResult) -> str:
    content = result.content.strip()
    if content:
        return f"Title: {result.title}\nURL: {result.url}\n\n{content}"
    return f"Title: {result.title}\nURL: {result.url}"


__all__ = ["TavilyProbeToolGateway"]
