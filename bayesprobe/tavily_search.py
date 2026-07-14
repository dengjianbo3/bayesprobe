from __future__ import annotations

import json
import math
import os
import socket
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


_DEFAULT_ENDPOINT = "https://api.tavily.com/search"
_DEFAULT_API_KEY_ENV = "TAVILY_API_KEY"
_MAX_QUERY_CHARACTERS = 400
_MAX_CONTENT_CHARACTERS = 3_000


class TavilySearchError(RuntimeError):
    pass


@dataclass(frozen=True)
class TavilySearchConfig:
    api_key_env: str = _DEFAULT_API_KEY_ENV
    endpoint: str = _DEFAULT_ENDPOINT
    topic: str = "general"
    search_depth: str = "advanced"
    max_results: int = 5
    chunks_per_source: int = 3
    timeout_seconds: float = 60.0
    max_query_characters: int = _MAX_QUERY_CHARACTERS
    max_content_characters: int = _MAX_CONTENT_CHARACTERS

    def __post_init__(self) -> None:
        _required_text(self.api_key_env, "api_key_env")
        _required_text(self.endpoint, "endpoint")
        if self.topic != "general":
            raise ValueError("Tavily topic must be general")
        if self.search_depth != "advanced":
            raise ValueError("Tavily search_depth must be advanced")
        for name, value, minimum in (
            ("max_results", self.max_results, 1),
            ("chunks_per_source", self.chunks_per_source, 1),
            ("max_query_characters", self.max_query_characters, 1),
            ("max_content_characters", self.max_content_characters, 1),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"Tavily {name} must be an integer at least {minimum}")
        if (
            type(self.timeout_seconds) not in (int, float)
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("Tavily timeout_seconds must be positive and finite")


@dataclass(frozen=True)
class TavilySearchRequest:
    query: str


@dataclass(frozen=True)
class TavilySearchResult:
    url: str
    title: str
    content: str
    score: float | None


@dataclass(frozen=True)
class TavilySearchResponse:
    query: str
    outcome: str
    results: tuple[TavilySearchResult, ...] = ()
    response_time_seconds: float | None = None
    request_id: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class TavilySearchExecutionRecord:
    query: str
    outcome: str
    result_count: int
    response_time_seconds: float | None
    request_id: str | None
    error_message: str | None


TavilyTransport = Callable[..., Mapping[str, Any]]


class TavilySearchClient:
    def __init__(
        self,
        config: TavilySearchConfig | None = None,
        *,
        environ: Mapping[str, str] | None = None,
        transport: TavilyTransport | None = None,
    ) -> None:
        self.config = config or TavilySearchConfig()
        environment = os.environ if environ is None else environ
        api_key = environment.get(self.config.api_key_env)
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError(
                f"Tavily API key environment variable {self.config.api_key_env} is not set"
            )
        self._api_key = api_key.strip()
        self._transport = transport or _default_transport
        self._records: list[TavilySearchExecutionRecord] = []

    def execution_records(self) -> tuple[TavilySearchExecutionRecord, ...]:
        return tuple(self._records)

    def search(self, request: TavilySearchRequest) -> TavilySearchResponse:
        query = _clean_query(request.query, self.config.max_query_characters)
        payload = {
            "query": query,
            "topic": self.config.topic,
            "search_depth": self.config.search_depth,
            "chunks_per_source": self.config.chunks_per_source,
            "max_results": self.config.max_results,
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
            "include_image_descriptions": False,
            "include_favicon": False,
            "include_usage": False,
        }
        try:
            raw_response = self._transport(
                endpoint=self.config.endpoint,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                payload=payload,
                timeout_seconds=self.config.timeout_seconds,
            )
            response = _response_from_payload(query, raw_response, self.config)
        except Exception as error:
            response = TavilySearchResponse(
                query=query,
                outcome=_outcome_for_error(error),
                error_message=_sanitize_error_message(str(error), self._api_key),
            )
        self._records.append(
            TavilySearchExecutionRecord(
                query=response.query,
                outcome=response.outcome,
                result_count=len(response.results),
                response_time_seconds=response.response_time_seconds,
                request_id=response.request_id,
                error_message=response.error_message,
            )
        )
        return response


def _default_transport(
    *,
    endpoint: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    timeout_seconds: float,
) -> Mapping[str, Any]:
    request = Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=dict(headers),
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            raw_content = response.read().decode("utf-8")
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise TavilySearchError(f"HTTP {error.code}: {body}") from error
    except URLError as error:
        raise TavilySearchError(str(error.reason)) from error
    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError as error:
        raise TavilySearchError("Tavily response is not valid JSON") from error
    if not isinstance(parsed, Mapping):
        raise TavilySearchError("Tavily response must be an object")
    return parsed


def _response_from_payload(
    query: str,
    payload: Mapping[str, Any],
    config: TavilySearchConfig,
) -> TavilySearchResponse:
    if not isinstance(payload, Mapping):
        raise TavilySearchError("Tavily response must be an object")
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raise TavilySearchError("Tavily response results must be an array")
    results = tuple(_result_from_payload(item, config) for item in raw_results)
    response_time = _optional_finite_number(payload.get("response_time"))
    request_id = _optional_text(payload.get("request_id"))
    return TavilySearchResponse(
        query=query,
        outcome="success",
        results=results,
        response_time_seconds=response_time,
        request_id=request_id,
    )


def _result_from_payload(
    payload: Any,
    config: TavilySearchConfig,
) -> TavilySearchResult:
    if not isinstance(payload, Mapping):
        raise TavilySearchError("Tavily result must be an object")
    raw_url = payload.get("url")
    if not isinstance(raw_url, str):
        raise TavilySearchError("Tavily result URL must be text")
    url = _canonical_url(raw_url)
    title = _normalized_text(payload.get("title"), fallback=url)
    content = _normalized_text(payload.get("content"), fallback="")
    if len(content) > config.max_content_characters:
        content = content[: config.max_content_characters].rstrip()
    return TavilySearchResult(
        url=url,
        title=title,
        content=content,
        score=_optional_finite_number(payload.get("score")),
    )


def _clean_query(value: Any, maximum: int) -> str:
    if not isinstance(value, str):
        raise ValueError("Tavily query must be text")
    query = " ".join(value.split())
    if not query:
        raise ValueError("Tavily query must not be empty")
    if len(query) > maximum:
        raise ValueError(f"Tavily query must be at most {maximum} characters")
    return query


def _canonical_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise TavilySearchError("Tavily result URL must be an absolute HTTP URL")
    return urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path or "/",
            parsed.query,
            "",
        )
    )


def _normalized_text(value: Any, *, fallback: str) -> str:
    if not isinstance(value, str):
        return fallback
    normalized = " ".join(value.split())
    return normalized or fallback


def _optional_finite_number(value: Any) -> float | None:
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value):
        raise TavilySearchError("Tavily numeric value must be finite")
    return float(value)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TavilySearchError("Tavily text value must be text")
    return value.strip() or None


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Tavily {name} must be non-empty text")
    return value.strip()


def _outcome_for_error(error: Exception) -> str:
    if isinstance(error, (TimeoutError, socket.timeout)):
        return "timeout"
    message = str(error).casefold()
    if "401" in message or "403" in message or "authentication" in message:
        return "authentication"
    if "429" in message or "rate limit" in message:
        return "rate_limit"
    if isinstance(error, TavilySearchError):
        return "invalid_response"
    return "provider_error"


def _sanitize_error_message(value: str, api_key: str) -> str:
    return value.replace(api_key, "<redacted>").strip() or "Tavily request failed"


__all__ = [
    "TavilySearchClient",
    "TavilySearchConfig",
    "TavilySearchError",
    "TavilySearchExecutionRecord",
    "TavilySearchRequest",
    "TavilySearchResponse",
    "TavilySearchResult",
]
