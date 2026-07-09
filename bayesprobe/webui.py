from __future__ import annotations

import argparse
import ipaddress
import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from enum import Enum
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from bayesprobe.core import BayesProbeCore
from bayesprobe.initialization import InitializeRunInput
from bayesprobe.model_gateway import DeterministicModelGateway, ModelGateway
from bayesprobe.openai_gateway import (
    OpenAIModelGatewayConfig,
    OpenAIResponsesModelGateway,
)
from bayesprobe.question_runner import (
    AutonomousQuestionRunConfig,
    AutonomousQuestionRunResult,
    AutonomousQuestionRunner,
)


STATIC_DIR = Path(__file__).with_name("webui_static")
SUPPORTED_PROVIDER_KINDS = {"deterministic", "openai_responses"}
RESERVED_PROVIDER_KINDS = {"openai_chat_completions"}


class WebUIError(Exception):
    status_code = HTTPStatus.BAD_REQUEST
    error_type = "validation_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class UnsupportedProviderError(WebUIError):
    error_type = "unsupported_provider"


class InvalidJSONError(WebUIError):
    error_type = "invalid_json"


class ProviderError(WebUIError):
    status_code = HTTPStatus.BAD_GATEWAY
    error_type = "provider_error"


def handle_autonomous_run_request(
    payload: Mapping[str, Any],
    *,
    client_factory: Callable[..., Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    try:
        request = _parse_autonomous_request(payload)
        provider_kind = _optional_string(
            request["provider"].get("kind"),
            "provider.kind",
            default="deterministic",
        )
        gateway = _build_webui_model_gateway(
            request["provider"], client_factory=client_factory
        )
        core = BayesProbeCore(model_gateway=gateway)
        runner = AutonomousQuestionRunner(
            core=core,
            config=request["runner_config"],
        )
        run_id = _webui_run_id()
        try:
            result = runner.run_question(
                InitializeRunInput(
                    run_id=run_id,
                    problem=request["question"],
                    context=request["context"],
                )
            )
        except Exception as error:
            if provider_kind == "openai_responses":
                raise ProviderError(_generic_provider_error_message()) from error
            raise
        return HTTPStatus.OK, serialize_autonomous_run_result(result)
    except WebUIError as error:
        return int(error.status_code), _error_payload(error.error_type, error.message)
    except Exception:
        return int(HTTPStatus.INTERNAL_SERVER_ERROR), _error_payload(
            "server_error", _generic_server_error_message()
        )


def serialize_autonomous_run_result(
    result: AutonomousQuestionRunResult,
) -> dict[str, Any]:
    return {
        "run_id": result.run.run_id,
        "stop_reason": result.stop_reason.value,
        "final_answer": _dump_domain(result.final_answer_projection),
        "initial_belief_state": _dump_domain(result.initial_belief_state),
        "final_belief_state": _dump_domain(result.final_belief_state),
        "cycles": [
            {
                "cycle_id": cycle.cycle.cycle_id,
                "signal_shape": cycle.cycle.signal_shape.value,
                "cycle": _dump_domain(cycle.cycle),
                "probes": _dump_domain(cycle.probe_set.probes),
                "signals": _dump_domain(cycle.signals),
                "evidence_events": _dump_domain(cycle.evidence_events),
                "belief_updates": _dump_domain(cycle.belief_updates),
                "hypothesis_evolutions": _dump_domain(cycle.hypothesis_evolutions),
                "answer_projection": _dump_domain(cycle.answer_projection),
            }
            for cycle in result.cycle_results
        ],
    }


def create_handler_class() -> type[BaseHTTPRequestHandler]:
    class BayesProbeWebUIHandler(BaseHTTPRequestHandler):
        server_version = "BayesProbeWebUI/0.1"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802
            try:
                if self.path != "/api/runs/autonomous":
                    self._send_json(
                        HTTPStatus.NOT_FOUND,
                        _error_payload("not_found", "not found"),
                    )
                    return
                payload = self._read_json_body()
                status, response = handle_autonomous_run_request(payload)
                self._send_json(status, response)
            except WebUIError as error:
                self._send_json(
                    error.status_code,
                    _error_payload(error.error_type, error.message),
                )
            except Exception:
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    _error_payload("server_error", _generic_server_error_message()),
                )

        def do_GET(self) -> None:  # noqa: N802
            try:
                if self.path in {"/", "/index.html"}:
                    self._send_static("index.html", "text/html; charset=utf-8")
                    return
                if self.path == "/styles.css":
                    self._send_static("styles.css", "text/css; charset=utf-8")
                    return
                if self.path == "/app.js":
                    self._send_static("app.js", "text/javascript; charset=utf-8")
                    return
                self._send_json(HTTPStatus.NOT_FOUND, _error_payload("not_found", "not found"))
            except WebUIError as error:
                self._send_json(
                    error.status_code,
                    _error_payload(error.error_type, error.message),
                )
            except Exception:
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    _error_payload("server_error", _generic_server_error_message()),
                )

        def _read_json_body(self) -> Mapping[str, Any]:
            content_length = self.headers.get("Content-Length")
            if content_length is None:
                raise WebUIError("request body must not be empty")
            try:
                size = int(content_length)
            except ValueError as error:
                raise WebUIError("content-length must be an integer") from error
            raw_body = self.rfile.read(size)
            if not raw_body:
                raise WebUIError("request body must not be empty")
            try:
                payload = json.loads(raw_body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise InvalidJSONError("request body must be valid JSON") from error
            if not isinstance(payload, Mapping):
                raise WebUIError("request payload must be an object")
            return payload

        def _send_static(self, filename: str, content_type: str) -> None:
            path = STATIC_DIR / filename
            if not path.exists():
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    _error_payload("not_found", "static asset not found"),
                )
                return
            try:
                body = path.read_bytes()
            except OSError:
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    _error_payload("server_error", _generic_server_error_message()),
                )
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: int | HTTPStatus, payload: Mapping[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return BayesProbeWebUIHandler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the BayesProbe local WebUI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        host = _require_loopback_host(args.host)
    except ValueError as error:
        parser.error(str(error))
    server = ThreadingHTTPServer((host, args.port), create_handler_class())
    host, port = server.server_address
    print(f"BayesProbe WebUI running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


def _parse_autonomous_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise WebUIError("request payload must be an object")
    question = _required_nonempty_string(payload.get("question"), "question")
    context = _optional_string(payload.get("context"), "context", default="")
    provider = payload.get("provider", {"kind": "deterministic"})
    if provider is None:
        provider = {"kind": "deterministic"}
    if not isinstance(provider, Mapping):
        raise WebUIError("provider must be an object")
    runner_payload = payload.get("runner", {})
    if runner_payload is None:
        runner_payload = {}
    if not isinstance(runner_payload, Mapping):
        raise WebUIError("runner must be an object")
    return {
        "question": question,
        "context": context,
        "provider": dict(provider),
        "runner_config": _runner_config_from_payload(runner_payload),
    }


def _runner_config_from_payload(
    payload: Mapping[str, Any],
) -> AutonomousQuestionRunConfig:
    try:
        return AutonomousQuestionRunConfig(
            max_cycles=_optional_int(payload, "max_cycles", default=3),
            max_probes_per_cycle=_optional_int(payload, "max_probes_per_cycle", default=2),
            stop_on_no_probes=_optional_bool(payload, "stop_on_no_probes", default=True),
            confidence_threshold=_optional_float(payload, "confidence_threshold"),
            posterior_delta_threshold=_optional_float(payload, "posterior_delta_threshold"),
        )
    except ValueError as error:
        raise WebUIError(str(error)) from error


def _build_webui_model_gateway(
    provider: Mapping[str, Any],
    *,
    client_factory: Callable[..., Any] | None,
) -> ModelGateway:
    kind = _optional_string(provider.get("kind"), "provider.kind", default="deterministic")
    if kind in RESERVED_PROVIDER_KINDS:
        raise UnsupportedProviderError(f"provider kind {kind} is not supported in v0.1")
    if kind == "deterministic":
        return DeterministicModelGateway()
    if kind == "openai_responses":
        model = _required_nonempty_string(provider.get("model"), "provider.model")
        api_key = _required_nonempty_string(provider.get("api_key"), "provider.api_key")
        try:
            config = OpenAIModelGatewayConfig(
                model=model,
                base_url=_optional_string(provider.get("base_url"), "provider.base_url"),
                timeout_seconds=_optional_number_from_mapping(
                    provider, "timeout_seconds", default=30.0
                ),
                max_output_tokens=_optional_int_or_none(provider, "max_output_tokens"),
            )
        except ValueError as error:
            raise WebUIError(str(error)) from error
        client = None
        if client_factory is not None:
            try:
                client = client_factory(**_openai_client_kwargs(config, api_key))
            except Exception as error:
                raise ProviderError(_generic_provider_error_message()) from error
        return OpenAIResponsesModelGateway(
            config=config,
            api_key=api_key,
            client=client,
        )
    if kind in SUPPORTED_PROVIDER_KINDS:
        raise UnsupportedProviderError(f"provider kind {kind} is not supported in v0.1")
    raise UnsupportedProviderError(f"unsupported provider kind: {kind}")


def _openai_client_kwargs(
    config: OpenAIModelGatewayConfig, api_key: str
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"api_key": api_key, "timeout": config.timeout_seconds}
    if config.base_url is not None:
        kwargs["base_url"] = config.base_url
    return kwargs


def _dump_domain(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value):
        return _dump_domain(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _dump_domain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_dump_domain(item) for item in value]
    return value


def _required_nonempty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise WebUIError(f"{field_name} must not be empty")
    cleaned = value.strip()
    if not cleaned:
        raise WebUIError(f"{field_name} must not be empty")
    return cleaned


def _optional_string(
    value: Any,
    field_name: str,
    *,
    default: str | None = None,
) -> str | None:
    if value is None:
        return default
    if not isinstance(value, str):
        raise WebUIError(f"{field_name} must be a string")
    cleaned = value.strip()
    if not cleaned:
        if default is not None:
            return default
        raise WebUIError(f"{field_name} must not be empty")
    return cleaned


def _optional_int(
    payload: Mapping[str, Any],
    field_name: str,
    *,
    default: int,
) -> int:
    value = payload.get(field_name, default)
    if type(value) is not int:
        raise WebUIError(f"{field_name} must be an integer")
    return value


def _optional_int_or_none(payload: Mapping[str, Any], field_name: str) -> int | None:
    if field_name not in payload or payload[field_name] is None:
        return None
    value = payload[field_name]
    if type(value) is not int:
        raise WebUIError(f"{field_name} must be an integer")
    if value < 1:
        raise WebUIError(f"{field_name} must be positive")
    return value


def _optional_bool(
    payload: Mapping[str, Any],
    field_name: str,
    *,
    default: bool,
) -> bool:
    value = payload.get(field_name, default)
    if type(value) is not bool:
        raise WebUIError(f"{field_name} must be a boolean")
    return value


def _optional_float(payload: Mapping[str, Any], field_name: str) -> float | None:
    value = payload.get(field_name)
    if value is None:
        return None
    if type(value) not in (int, float):
        raise WebUIError(f"{field_name} must be a number")
    return float(value)


def _optional_number_from_mapping(
    payload: Mapping[str, Any], field_name: str, *, default: float
) -> float:
    if field_name not in payload or payload[field_name] is None:
        return default
    value = payload[field_name]
    if type(value) not in (int, float):
        raise WebUIError(f"{field_name} must be a number")
    if value <= 0:
        raise WebUIError(f"{field_name} must be positive")
    return float(value)


def _webui_run_id() -> str:
    return f"webui_{int(time.time() * 1000)}"


def _require_loopback_host(host: str) -> str:
    if host == "localhost":
        return host
    try:
        parsed = ipaddress.ip_address(host)
    except ValueError as error:
        raise ValueError(
            "--host must be a loopback address (localhost, 127.0.0.1, or ::1)"
        ) from error
    if not parsed.is_loopback:
        raise ValueError(
            "--host must be a loopback address (localhost, 127.0.0.1, or ::1)"
        )
    return host


def _error_payload(error_type: str, message: str) -> dict[str, Any]:
    return {"error": {"type": error_type, "message": _sanitize_error_message(message)}}


def _sanitize_error_message(message: str) -> str:
    return re.sub(r"sk-[A-Za-z0-9_\\-]+", "sk-redacted", message)


def _generic_server_error_message() -> str:
    return "internal server error"


def _generic_provider_error_message() -> str:
    return "provider request failed"


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "create_handler_class",
    "handle_autonomous_run_request",
    "main",
    "serialize_autonomous_run_result",
]
