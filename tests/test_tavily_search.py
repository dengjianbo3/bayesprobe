import json
from dataclasses import asdict

import pytest

from bayesprobe.tavily_search import (
    TavilySearchClient,
    TavilySearchConfig,
    TavilySearchRequest,
)


_SECRET = "tvly-dev-test-secret-value"


def _success_payload():
    return {
        "query": "test query",
        "response_time": 0.12,
        "results": [
            {
                "url": "https://source.test/a#fragment",
                "title": "Source A",
                "content": "  A supported fact.  ",
                "score": 0.9,
            }
        ],
    }


def test_client_posts_frozen_payload_and_never_records_authorization_token():
    captured = {}

    def transport(*, endpoint, headers, payload, timeout_seconds):
        captured.update(
            endpoint=endpoint,
            headers=dict(headers),
            payload=dict(payload),
            timeout_seconds=timeout_seconds,
        )
        return _success_payload()

    client = TavilySearchClient(
        TavilySearchConfig(),
        environ={"TAVILY_API_KEY": _SECRET},
        transport=transport,
    )

    response = client.search(TavilySearchRequest(query="test query"))

    assert response.outcome == "success"
    assert response.results[0].url == "https://source.test/a"
    assert response.results[0].content == "A supported fact."
    assert captured["endpoint"] == "https://api.tavily.com/search"
    assert captured["headers"]["Authorization"] == f"Bearer {_SECRET}"
    assert captured["payload"] == {
        "query": "test query",
        "topic": "general",
        "search_depth": "advanced",
        "chunks_per_source": 3,
        "max_results": 5,
        "include_answer": False,
        "include_raw_content": False,
        "include_images": False,
        "include_image_descriptions": False,
        "include_favicon": False,
        "include_usage": False,
    }
    assert captured["timeout_seconds"] == 60
    assert _SECRET not in json.dumps([asdict(item) for item in client.execution_records()])


@pytest.mark.parametrize("query", ["", "   ", "x" * 401])
def test_client_rejects_blank_and_overlong_queries_before_transport(query):
    called = False

    def transport(**_kwargs):
        nonlocal called
        called = True
        return _success_payload()

    client = TavilySearchClient(
        TavilySearchConfig(),
        environ={"TAVILY_API_KEY": _SECRET},
        transport=transport,
    )

    with pytest.raises(ValueError, match="query"):
        client.search(TavilySearchRequest(query=query))

    assert called is False


@pytest.mark.parametrize(
    ("exception", "expected_outcome"),
    [
        (TimeoutError("token=tvly-dev-test-secret-value"), "timeout"),
        (RuntimeError("status=429 token=tvly-dev-test-secret-value"), "rate_limit"),
    ],
)
def test_client_classifies_transport_failures_without_secret_text(
    exception,
    expected_outcome,
):
    def transport(**_kwargs):
        raise exception

    client = TavilySearchClient(
        TavilySearchConfig(),
        environ={"TAVILY_API_KEY": _SECRET},
        transport=transport,
    )

    response = client.search(TavilySearchRequest(query="test query"))

    assert response.outcome == expected_outcome
    assert response.results == ()
    assert _SECRET not in (response.error_message or "")
    assert _SECRET not in json.dumps([asdict(item) for item in client.execution_records()])


def test_client_marks_malformed_provider_payload_as_invalid_response():
    client = TavilySearchClient(
        TavilySearchConfig(),
        environ={"TAVILY_API_KEY": _SECRET},
        transport=lambda **_kwargs: {"results": "not-a-list"},
    )

    response = client.search(TavilySearchRequest(query="test query"))

    assert response.outcome == "invalid_response"
    assert response.results == ()
