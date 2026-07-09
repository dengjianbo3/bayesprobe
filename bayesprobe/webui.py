from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
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
        gateway = _build_webui_model_gateway(
            request["provider"], client_factory=client_factory
        )
        core = BayesProbeCore(model_gateway=gateway)
        runner = AutonomousQuestionRunner(
            core=core,
            config=request["runner_config"],
        )
        run_id = _webui_run_id()
        result = runner.run_question(
            InitializeRunInput(
                run_id=run_id,
                problem=request["question"],
                context=request["context"],
            )
        )
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
    class WebUIHandler(BaseHTTPRequestHandler):
        server_version = "BayesProbeWebUI/0.1"

        def do_GET(self) -> None:  # noqa: N802
            try:
                if self.path == "/":
                    self._serve_static_file("index.html")
                    return
                if self.path in {"/styles.css", "/app.js"}:
                    self._serve_static_file(self.path.lstrip("/"))
                    return
                self._write_json(
                    HTTPStatus.NOT_FOUND,
                    _error_payload("not_found", f"unknown path: {self.path}"),
                )
            except WebUIError as error:
                self._write_json(
                    error.status_code,
                    _error_payload(error.error_type, error.message),
                )
            except Exception:
                self._write_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    _error_payload("server_error", _generic_server_error_message()),
                )

        def do_POST(self) -> None:  # noqa: N802
            try:
                if self.path != "/api/runs/autonomous":
                    self._write_json(
                        HTTPStatus.NOT_FOUND,
                        _error_payload("not_found", f"unknown path: {self.path}"),
                    )
                    return
                payload = self._read_json_body()
                status, response = handle_autonomous_run_request(payload)
                self._write_json(status, response)
            except WebUIError as error:
                self._write_json(
                    error.status_code,
                    _error_payload(error.error_type, error.message),
                )
            except Exception:
                self._write_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    _error_payload("server_error", _generic_server_error_message()),
                )

        def log_message(self, format: str, *args: Any) -> None:
            return

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

        def _serve_static_file(self, relative_path: str) -> None:
            file_path = STATIC_DIR / relative_path
            if not file_path.is_file():
                self._write_json(
                    HTTPStatus.NOT_FOUND,
                    _error_payload("not_found", f"unknown path: {self.path}"),
                )
                return
            try:
                body = file_path.read_bytes()
            except OSError:
                self._write_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    _error_payload("server_error", _generic_server_error_message()),
                )
                return
            content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _write_json(self, status: int | HTTPStatus, payload: Mapping[str, Any]) -> None:
            body = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return WebUIHandler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bayesprobe-webui")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as error:
        return int(error.code)

    server = ThreadingHTTPServer((args.host, args.port), create_handler_class())
    print(f"BayesProbe WebUI listening on http://{args.host}:{args.port}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
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
    del client_factory
    kind = _optional_string(provider.get("kind"), "provider.kind", default="deterministic")
    if kind in RESERVED_PROVIDER_KINDS:
        raise UnsupportedProviderError(f"provider kind {kind} is not supported in v0.1")
    if kind == "deterministic":
        return DeterministicModelGateway()
    if kind == "openai_responses":
        raise UnsupportedProviderError("provider kind openai_responses is not wired yet")
    if kind in SUPPORTED_PROVIDER_KINDS:
        raise UnsupportedProviderError(f"provider kind {kind} is not supported in v0.1")
    raise UnsupportedProviderError(f"unsupported provider kind: {kind}")


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


def _webui_run_id() -> str:
    return f"webui_{int(time.time() * 1000)}"


def _error_payload(error_type: str, message: str) -> dict[str, Any]:
    return {"error": {"type": error_type, "message": _sanitize_error_message(message)}}


def _sanitize_error_message(message: str) -> str:
    return re.sub(r"sk-[A-Za-z0-9_\\-]+", "sk-redacted", message)


def _generic_server_error_message() -> str:
    return "internal server error"


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "create_handler_class",
    "handle_autonomous_run_request",
    "main",
    "serialize_autonomous_run_result",
]
