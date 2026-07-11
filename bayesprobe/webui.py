from __future__ import annotations

import argparse
import ipaddress
import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass, replace
from enum import Enum
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel

from bayesprobe.core import BayesProbeCore
from bayesprobe.initialization import (
    BayesProbeInitializer,
    InitializeRunInput,
    validate_compatibility_context_security,
)
from bayesprobe.model_gateway import (
    DeterministicModelGateway,
    ModelGateway,
    ModelGatewayValidationError,
)
from bayesprobe.openai_gateway import (
    OpenAIChatCompletionsModelGateway,
    OpenAIModelGatewayConfig,
    OpenAIResponsesModelGateway,
)
from bayesprobe.probe_executor import ModelBackedProbeToolGateway, ProbeExecutor
from bayesprobe.question_runner import (
    AutonomousQuestionCycleResult,
    AutonomousQuestionProgress,
    AutonomousQuestionProgressKind,
    AutonomousQuestionRunConfig,
    AutonomousQuestionRunResult,
    AutonomousQuestionRunner,
    NeedsReframingResult,
    OutOfScopeResult,
)
from bayesprobe.schemas import AnswerChoice, redact_secret_material
from bayesprobe.task_admission import (
    ExplicitTaskAdmitter,
    ModelTaskAdmitter,
    RoutingTaskAdmitter,
    TaskAdmissionError,
    TaskAdmissionInput,
)
from bayesprobe.task_framing import (
    ExplicitTaskFramer,
    ModelTaskFramer,
    RoutingTaskFramer,
    TaskFramingError,
    TaskFramingInput,
    parse_legacy_answer_choice_frame,
    validate_task_framing_input_security,
)


STATIC_DIR = Path(__file__).with_name("webui_static")
OPENAI_COMPATIBLE_PROVIDER_KINDS = {"openai_responses", "openai_chat_completions"}
SUPPORTED_PROVIDER_KINDS = {"deterministic"} | OPENAI_COMPATIBLE_PROVIDER_KINDS
WEBUI_MIN_PROVIDER_TIMEOUT_SECONDS = 360.0
WEBUI_DEEPSEEK_V4_MIN_OUTPUT_TOKENS = 32768


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


@dataclass(frozen=True)
class _PreparedAutonomousRun:
    runner: AutonomousQuestionRunner
    input: InitializeRunInput
    provider_kind: str


def handle_autonomous_run_request(
    payload: Mapping[str, Any],
    *,
    client_factory: Callable[..., Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    try:
        prepared = _prepare_autonomous_run(payload, client_factory=client_factory)
        try:
            result = prepared.runner.run_question(prepared.input)
        except TaskFramingError as error:
            if prepared.provider_kind in OPENAI_COMPATIBLE_PROVIDER_KINDS:
                raise ProviderError(_provider_error_message(prepared.provider_kind)) from error
            raise WebUIError(str(error)) from error
        except Exception as error:
            if prepared.provider_kind in OPENAI_COMPATIBLE_PROVIDER_KINDS:
                raise ProviderError(
                    _provider_runtime_error_message(
                        error,
                        prepared.provider_kind,
                    )
                ) from error
            raise
        return HTTPStatus.OK, serialize_autonomous_run_result(result)
    except TaskFramingError:
        return int(HTTPStatus.BAD_REQUEST), _task_framing_error_payload()
    except WebUIError as error:
        return int(error.status_code), _error_payload(error.error_type, error.message)
    except Exception:
        return int(HTTPStatus.INTERNAL_SERVER_ERROR), _error_payload(
            "server_error", _generic_server_error_message()
        )


def handle_autonomous_stream_request(
    payload: Mapping[str, Any],
    *,
    event_sink: Callable[[Mapping[str, Any]], None],
    client_factory: Callable[..., Any] | None = None,
) -> tuple[int, dict[str, Any] | None]:
    emitter = _AutonomousProgressEventEmitter(event_sink)
    try:
        prepared = _prepare_autonomous_run(
            payload,
            client_factory=client_factory,
            progress_observer=emitter.emit,
        )
    except TaskFramingError:
        return int(HTTPStatus.BAD_REQUEST), _task_framing_error_payload()
    except WebUIError as error:
        return int(error.status_code), _error_payload(error.error_type, error.message)

    try:
        result = prepared.runner.run_question(prepared.input)
        if isinstance(result, (NeedsReframingResult, OutOfScopeResult)):
            emitter.emit_admission_result(result, run_id=prepared.input.run_id)
    except TaskFramingError as error:
        if prepared.provider_kind in OPENAI_COMPATIBLE_PROVIDER_KINDS:
            emitter.emit_failure("provider_error", _provider_error_message(prepared.provider_kind))
        elif emitter.started:
            emitter.emit_failure("validation_error", str(error))
        else:
            return int(HTTPStatus.BAD_REQUEST), _task_framing_error_payload()
    except Exception as error:
        if prepared.provider_kind in OPENAI_COMPATIBLE_PROVIDER_KINDS:
            emitter.emit_failure(
                "provider_error",
                _provider_runtime_error_message(
                    error,
                    prepared.provider_kind,
                ),
            )
        else:
            emitter.emit_failure("server_error", _generic_server_error_message())
    return int(HTTPStatus.OK), None


def serialize_autonomous_run_result(
    result: AutonomousQuestionRunResult | NeedsReframingResult | OutOfScopeResult,
) -> dict[str, Any]:
    if isinstance(result, (NeedsReframingResult, OutOfScopeResult)):
        return {
            "result_type": result.result_type,
            "admission": redact_secret_material(result.admission),
        }
    return {
        "run_id": result.run.run_id,
        "run": _dump_domain(result.run),
        "stop_reason": result.stop_reason.value,
        "final_answer": _dump_domain(result.final_answer_projection),
        "task_frame": _dump_domain(result.task_frame),
        "initial_belief_state": _dump_domain(result.initial_belief_state),
        "final_belief_state": _dump_domain(result.final_belief_state),
        "cycles": [
            serialize_autonomous_cycle_result(cycle) for cycle in result.cycle_results
        ],
    }


def serialize_autonomous_cycle_result(
    cycle: AutonomousQuestionCycleResult,
) -> dict[str, Any]:
    return {
        "cycle_id": cycle.cycle.cycle_id,
        "signal_shape": cycle.cycle.signal_shape.value,
        "cycle": _dump_domain(cycle.cycle),
        "probes": _dump_domain(cycle.probe_set.probes),
        "signals": _dump_domain(cycle.signals),
        "belief_state": _dump_domain(cycle.belief_state),
        "evidence_events": _dump_domain(cycle.evidence_events),
        "belief_updates": _dump_domain(cycle.belief_updates),
        "hypothesis_evolutions": _dump_domain(cycle.hypothesis_evolutions),
        "answer_projection": _dump_domain(cycle.answer_projection),
    }


class _AutonomousProgressEventEmitter:
    def __init__(self, sink: Callable[[Mapping[str, Any]], None]) -> None:
        self._sink = sink
        self.sequence = 0
        self.run_id: str | None = None
        self.cycle_id: str | None = None
        self.cycle_index: int | None = None

    @property
    def started(self) -> bool:
        return self.sequence > 0

    def emit(self, progress: AutonomousQuestionProgress) -> None:
        self.run_id = progress.run_id
        self.cycle_id = progress.cycle_id
        self.cycle_index = progress.cycle_index
        self.sequence += 1
        self._sink(
            {
                "event": progress.kind.value,
                "sequence": self.sequence,
                "timestamp": _webui_timestamp(),
                "run_id": progress.run_id,
                "cycle_id": progress.cycle_id,
                "cycle_index": progress.cycle_index,
                "data": _serialize_progress_data(progress),
            }
        )

    def emit_failure(self, error_type: str, message: str) -> None:
        self.sequence += 1
        self._sink(
            {
                "event": "run_failed",
                "sequence": self.sequence,
                "timestamp": _webui_timestamp(),
                "run_id": self.run_id,
                "cycle_id": self.cycle_id,
                "cycle_index": self.cycle_index,
                "data": {
                    "error": {
                        "type": error_type,
                        "message": _sanitize_error_message(message),
                    }
                },
            }
        )

    def emit_admission_result(
        self,
        result: NeedsReframingResult | OutOfScopeResult,
        *,
        run_id: str,
    ) -> None:
        self.sequence += 1
        self._sink(
            {
                "event": "task_admission_completed",
                "sequence": self.sequence,
                "timestamp": _webui_timestamp(),
                "run_id": run_id,
                "cycle_id": None,
                "cycle_index": None,
                "data": serialize_autonomous_run_result(result),
            }
        )


def _serialize_progress_data(progress: AutonomousQuestionProgress) -> dict[str, Any]:
    if progress.kind == AutonomousQuestionProgressKind.RUN_STARTED:
        return {}
    if progress.kind == AutonomousQuestionProgressKind.TASK_FRAMING_STARTED:
        return {}
    if progress.kind == AutonomousQuestionProgressKind.TASK_FRAMING_COMPLETED:
        return {"task_frame": _dump_domain(progress.task_frame)}
    if progress.kind == AutonomousQuestionProgressKind.INITIALIZATION_COMPLETED:
        return {
            "run": _dump_domain(progress.run),
            "belief_state": _dump_domain(progress.belief_state),
        }
    if progress.kind == AutonomousQuestionProgressKind.CYCLE_STARTED:
        belief_state = progress.belief_state
        return {
            "belief_summary": {
                "posterior_summary": _dump_domain(belief_state.posterior_summary),
                "uncertainty_summary": belief_state.uncertainty_summary,
            }
            if belief_state is not None
            else {}
        }
    if progress.kind == AutonomousQuestionProgressKind.PROBE_SET_PLANNED:
        return {"probe_set": _dump_domain(progress.probe_set)}
    if progress.kind == AutonomousQuestionProgressKind.PROBE_EXECUTION_STARTED:
        return {
            "probe_count": len(progress.probe_set.probes)
            if progress.probe_set is not None
            else 0
        }
    if progress.kind == AutonomousQuestionProgressKind.SIGNALS_COLLECTED:
        return {"signals": _dump_domain(progress.signals)}
    if progress.kind == AutonomousQuestionProgressKind.EVIDENCE_INTEGRATION_STARTED:
        return {"signal_count": len(progress.signals)}
    if progress.kind == AutonomousQuestionProgressKind.CYCLE_INTEGRATED:
        return (
            serialize_autonomous_cycle_result(progress.cycle_result)
            if progress.cycle_result is not None
            else {}
        )
    if progress.kind == AutonomousQuestionProgressKind.RUN_COMPLETED:
        return (
            serialize_autonomous_run_result(progress.result)
            if progress.result is not None
            else {}
        )
    raise ValueError(f"unsupported progress kind: {progress.kind}")


class _NDJSONEventWriter:
    def __init__(self, handler: BaseHTTPRequestHandler) -> None:
        self._handler = handler
        self.started = False
        self.disconnected = False

    def emit(self, event: Mapping[str, Any]) -> None:
        if self.disconnected:
            return
        try:
            if not self.started:
                self._handler.send_response(HTTPStatus.OK)
                self._handler.send_header(
                    "Content-Type", "application/x-ndjson; charset=utf-8"
                )
                self._handler.send_header("Cache-Control", "no-store")
                self._handler.end_headers()
                self.started = True
            self._handler.wfile.write(
                json.dumps(event, ensure_ascii=False).encode("utf-8") + b"\n"
            )
            self._handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            self.disconnected = True


def create_handler_class(
    *, client_factory: Callable[..., Any] | None = None
) -> type[BaseHTTPRequestHandler]:
    class BayesProbeWebUIHandler(BaseHTTPRequestHandler):
        server_version = "BayesProbeWebUI/0.1"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802
            try:
                if self.path == "/api/runs/autonomous/stream":
                    self._handle_autonomous_stream_post()
                    return
                if self.path != "/api/runs/autonomous":
                    self._send_json(
                        HTTPStatus.NOT_FOUND,
                        _error_payload("not_found", "not found"),
                    )
                    return
                payload = self._read_json_body()
                status, response = handle_autonomous_run_request(
                    payload,
                    client_factory=client_factory,
                )
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

        def _handle_autonomous_stream_post(self) -> None:
            payload = self._read_json_body()
            writer = _NDJSONEventWriter(self)

            status, error = handle_autonomous_stream_request(
                payload,
                event_sink=writer.emit,
                client_factory=client_factory,
            )
            if error is not None and not writer.started:
                self._send_json(status, error)

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
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: int | HTTPStatus, payload: Mapping[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
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


def _prepare_autonomous_run(
    payload: Mapping[str, Any],
    *,
    client_factory: Callable[..., Any] | None,
    progress_observer: Callable[[AutonomousQuestionProgress], None] | None = None,
) -> _PreparedAutonomousRun:
    request = _parse_autonomous_request(payload)
    provider_kind = _optional_string(
        request["provider"].get("kind"),
        "provider.kind",
        default="deterministic",
    )
    assert provider_kind is not None
    input = InitializeRunInput(
        run_id=_webui_run_id(),
        problem=request["question"],
        context=request["context"],
        task_context=request["task_context"],
        answer_choices=request["answer_choices"],
    )
    if provider_kind not in OPENAI_COMPATIBLE_PROVIDER_KINDS:
        _preflight_task_framing(input)
    gateway = _build_webui_model_gateway(
        request["provider"], client_factory=client_factory
    )
    core = BayesProbeCore(model_gateway=gateway)
    executor = None
    if provider_kind in OPENAI_COMPATIBLE_PROVIDER_KINDS:
        executor = ProbeExecutor(
            gateway=ModelBackedProbeToolGateway(gateway),
            ledger=core.ledger,
        )
        task_framer = RoutingTaskFramer(
            explicit_framer=ExplicitTaskFramer(),
            open_framer=ModelTaskFramer(gateway),
        )
        task_admitter = RoutingTaskAdmitter(
            explicit_admitter=ExplicitTaskAdmitter(),
            open_admitter=ModelTaskAdmitter(gateway),
        )
    else:
        task_framer = ExplicitTaskFramer()
        task_admitter = ExplicitTaskAdmitter()
    return _PreparedAutonomousRun(
        runner=AutonomousQuestionRunner(
            core=core,
            initializer=BayesProbeInitializer(
                ledger=core.ledger,
                task_framer=task_framer,
                task_admitter=task_admitter,
            ),
            executor=executor,
            config=request["runner_config"],
            progress_observer=progress_observer,
            task_admitter=task_admitter,
        ),
        input=input,
        provider_kind=provider_kind,
    )


def _preflight_task_framing(input: InitializeRunInput) -> None:
    answer_choices = list(input.answer_choices)
    if not answer_choices:
        parsed = parse_legacy_answer_choice_frame(input.problem)
        if parsed is not None:
            answer_choices = list(parsed.choices)
    try:
        admission = ExplicitTaskAdmitter().assess(
            TaskAdmissionInput(
                attempt_id=f"{input.run_id}_admission",
                question=input.problem,
                task_context=input.task_context,
                answer_choices=answer_choices,
                hypothesis_seeds=list(input.hypothesis_seeds),
            )
        )
    except TaskAdmissionError:
        raise TaskFramingError(
            "unseeded open question requires a model or recorded task framer"
        ) from None
    framing_input = TaskFramingInput(
        run_id=input.run_id,
        question=input.problem,
        admission_decision=admission,
        task_context=input.task_context,
        answer_choices=answer_choices,
        hypothesis_seeds=list(input.hypothesis_seeds),
    )
    if not ExplicitTaskFramer().can_frame(framing_input):
        raise TaskFramingError(
            "unseeded open question requires a model or recorded task framer"
        )


def _task_framing_error_payload() -> dict[str, Any]:
    return _error_payload(
        "validation_error",
        "unseeded open question requires a model or recorded task framer",
    )


def _parse_autonomous_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise WebUIError("request payload must be an object")
    question = _required_nonempty_string(payload.get("question"), "question")
    context = _optional_string(payload.get("context"), "context", default="")
    try:
        validate_compatibility_context_security(context)
    except TaskFramingError as error:
        raise WebUIError(str(error)) from None
    task_context = _optional_string(
        payload.get("task_context"), "task_context", default=""
    )
    answer_choices = _answer_choices_from_payload(payload.get("answer_choices"))
    if answer_choices:
        try:
            admission = ExplicitTaskAdmitter().assess(
                TaskAdmissionInput(
                    attempt_id="webui_request_validation_admission",
                    question=question,
                    task_context=task_context,
                    answer_choices=answer_choices,
                )
            )
            validate_task_framing_input_security(
                TaskFramingInput(
                    run_id="webui_request_validation",
                    question=question,
                    admission_decision=admission,
                    task_context=task_context,
                    answer_choices=answer_choices,
                )
            )
        except TaskAdmissionError:
            raise WebUIError(
                "task framing input must not contain secret material"
            ) from None
        except TaskFramingError as error:
            raise WebUIError(str(error)) from None
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
        "task_context": task_context,
        "answer_choices": answer_choices,
        "provider": dict(provider),
        "runner_config": _runner_config_from_payload(runner_payload),
    }


def _answer_choices_from_payload(value: Any) -> list[AnswerChoice]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise WebUIError("answer_choices must be an array")
    try:
        choices = [AnswerChoice.model_validate(choice) for choice in value]
    except (TypeError, ValueError) as error:
        raise WebUIError(
            "answer_choices must contain non-empty label/text objects"
        ) from error
    if len(choices) < 2:
        raise WebUIError("answer_choices must contain at least two choices")
    if len(choices) > 6:
        raise WebUIError("answer_choices must contain at most six choices")
    labels = [choice.label for choice in choices]
    if len(labels) != len(set(labels)):
        raise WebUIError("answer_choices labels must be unique")
    return choices


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
    if kind == "deterministic":
        return DeterministicModelGateway()
    if kind in OPENAI_COMPATIBLE_PROVIDER_KINDS:
        model = _required_nonempty_string(provider.get("model"), "provider.model")
        api_key = _required_nonempty_string(provider.get("api_key"), "provider.api_key")
        try:
            config = OpenAIModelGatewayConfig(
                model=model,
                base_url=_optional_string(provider.get("base_url"), "provider.base_url"),
                timeout_seconds=_optional_number_from_mapping(
                    provider,
                    "timeout_seconds",
                    default=WEBUI_MIN_PROVIDER_TIMEOUT_SECONDS,
                ),
                max_output_tokens=_optional_int_or_none(provider, "max_output_tokens"),
            )
            if config.timeout_seconds < WEBUI_MIN_PROVIDER_TIMEOUT_SECONDS:
                config = replace(
                    config,
                    timeout_seconds=WEBUI_MIN_PROVIDER_TIMEOUT_SECONDS,
                )
            if _is_official_deepseek_v4_chat(kind, config):
                requested_tokens = config.max_output_tokens or 0
                if requested_tokens < WEBUI_DEEPSEEK_V4_MIN_OUTPUT_TOKENS:
                    config = replace(
                        config,
                        max_output_tokens=WEBUI_DEEPSEEK_V4_MIN_OUTPUT_TOKENS,
                    )
        except ValueError as error:
            raise WebUIError(str(error)) from error
        client = None
        if client_factory is not None:
            try:
                client = client_factory(**_openai_client_kwargs(config, api_key))
            except Exception as error:
                raise ProviderError(_provider_error_message(kind)) from error
        if kind == "openai_chat_completions":
            return OpenAIChatCompletionsModelGateway(
                config=config,
                api_key=api_key,
                client=client,
            )
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


def _webui_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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


def _is_official_deepseek_v4_chat(
    provider_kind: str,
    config: OpenAIModelGatewayConfig,
) -> bool:
    if provider_kind != "openai_chat_completions" or config.base_url is None:
        return False
    try:
        hostname = urlparse(config.base_url).hostname
    except ValueError:
        return False
    return hostname == "api.deepseek.com" and config.model.lower().startswith(
        "deepseek-v4-"
    )


def _provider_runtime_error_message(
    error: Exception,
    provider_kind: str,
) -> str:
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ModelGatewayValidationError) and (
            "exhausted max_tokens before producing structured content" in str(current)
        ):
            return (
                "provider exhausted max output tokens before producing structured "
                "content. Increase max output tokens and retry."
            )
        current = current.__cause__ or current.__context__
    return _provider_error_message(provider_kind)


def _provider_error_message(provider_kind: str) -> str:
    if provider_kind == "openai_responses":
        return (
            "provider request failed for openai_responses. "
            "Use Chat Completions for /chat/completions-compatible providers."
        )
    if provider_kind == "openai_chat_completions":
        return (
            "provider request failed for openai_chat_completions. "
            "Check base URL, model, API key, and max output tokens."
        )
    return _generic_provider_error_message()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "create_handler_class",
    "handle_autonomous_run_request",
    "handle_autonomous_stream_request",
    "main",
    "serialize_autonomous_cycle_result",
    "serialize_autonomous_run_result",
]
