from __future__ import annotations

import os

import pytest

from bayesprobe.tavily_search import TavilySearchClient, TavilySearchRequest


@pytest.mark.skipif(
    os.environ.get("BAYESPROBE_RUN_TAVILY_LIVE") != "1"
    or not os.environ.get("TAVILY_API_KEY"),
    reason="set BAYESPROBE_RUN_TAVILY_LIVE=1 and TAVILY_API_KEY to run Tavily live smoke",
)
def test_tavily_live_search_returns_a_sanitized_result():
    response = TavilySearchClient().search(
        TavilySearchRequest(query="Tavily official documentation")
    )

    assert response.outcome == "success"
    assert response.results
    assert all(result.url.startswith("https://") for result in response.results)
