from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from bayesprobe_terminal_bench import __version__
from bayesprobe_terminal_bench.config import TerminalBenchConfig
from bayesprobe_terminal_bench.runner_factory import build_live_session


_REDACTION_MARKER = "[REDACTED]"
_MAX_METADATA_TEXT_LENGTH = 96


def _bounded_text(value: object, *, fallback: str, restricted_value: str) -> str:
    if not isinstance(value, str):
        return fallback
    redacted = value.replace(restricted_value, _REDACTION_MARKER)
    return redacted[:_MAX_METADATA_TEXT_LENGTH] or fallback


def _bounded_count(value: object, *, maximum: int) -> int:
    if type(value) is not int:
        return 0
    return min(max(value, 0), maximum)


def _redact_exception_value(value: object, restricted_value: str) -> object:
    if isinstance(value, str):
        return value.replace(restricted_value, _REDACTION_MARKER)
    if isinstance(value, tuple):
        return tuple(_redact_exception_value(item, restricted_value) for item in value)
    if isinstance(value, list):
        return [_redact_exception_value(item, restricted_value) for item in value]
    if isinstance(value, dict):
        return {
            _redact_exception_value(key, restricted_value): _redact_exception_value(
                item, restricted_value
            )
            for key, item in value.items()
        }
    return value


def _redact_exception(error: BaseException, restricted_value: str) -> None:
    seen: set[int] = set()

    def redact(current: BaseException | None) -> None:
        if current is None or id(current) in seen:
            return
        seen.add(id(current))
        current.args = tuple(
            _redact_exception_value(value, restricted_value) for value in current.args
        )
        notes = getattr(current, "__notes__", None)
        if isinstance(notes, list):
            current.__notes__ = [
                _redact_exception_value(note, restricted_value) for note in notes
            ]
        redact(current.__cause__)
        redact(current.__context__)

    redact(error)


class BayesProbeHarborAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return "bayesprobe"

    def version(self) -> str | None:
        return __version__

    async def setup(self, environment: BaseEnvironment) -> None:
        return None

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        config, api_key = TerminalBenchConfig.from_sources(self.extra_env)
        try:
            session = build_live_session(
                config=config,
                api_key=api_key,
                instruction=instruction,
                environment=environment,
                event_loop=asyncio.get_running_loop(),
                logs_dir=self.logs_dir,
                session_id=self.session_id,
                context_id=str(self.context_id) if self.context_id is not None else None,
            )
            result = await asyncio.to_thread(session.runner.run_question, session.input)
        except Exception as error:
            _redact_exception(error, api_key)
            raise

        metadata: dict[str, Any] = {
            "bayesprobe_run_id": _bounded_text(
                getattr(session.input, "run_id", None),
                fallback="tb_harbor",
                restricted_value=api_key,
            ),
            "bayesprobe_stop_reason": _bounded_text(
                getattr(getattr(result, "stop_reason", None), "value", None),
                fallback="not_admitted",
                restricted_value=api_key,
            ),
            "bayesprobe_cycles": _bounded_count(
                len(getattr(result, "cycle_results", ())),
                maximum=config.max_cycles,
            ),
            "terminal_actions": _bounded_count(
                getattr(session.budget, "actions_used", None),
                maximum=config.max_total_actions,
            ),
            "model_calls": _bounded_count(
                getattr(session.budget, "model_calls_used", None),
                maximum=config.max_model_calls,
            ),
        }
        existing_metadata = context.metadata
        context.metadata = {
            **(dict(existing_metadata) if isinstance(existing_metadata, Mapping) else {}),
            **metadata,
        }
        session.artifacts.write_summary(metadata)
