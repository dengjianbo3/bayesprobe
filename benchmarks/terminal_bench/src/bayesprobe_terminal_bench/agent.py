from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from bayesprobe_terminal_bench import __version__
from bayesprobe_terminal_bench.artifacts import TrialArtifactStore
from bayesprobe_terminal_bench.config import (
    TerminalBenchConfig,
    classify_trial_error,
)
from bayesprobe_terminal_bench.runner_factory import build_live_session
from bayesprobe_terminal_bench.trajectory import write_atif_trajectory


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


def _nonnegative_count(value: object) -> int:
    return value if type(value) is int and value >= 0 else 0


class BayesProbeHarborAgentError(RuntimeError):
    def __init__(self, category: str, *, configuration: bool = False) -> None:
        self.category = category
        prefix = (
            "BayesProbe Harbor agent configuration failed"
            if configuration
            else "BayesProbe Harbor agent failed"
        )
        super().__init__(f"{prefix}: {category}")


class BayesProbeHarborAgent(BaseAgent):
    SUPPORTS_ATIF = True

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
        configuration_error: BayesProbeHarborAgentError | None = None
        try:
            config, api_key = TerminalBenchConfig.from_sources(self.extra_env)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            configuration_error = BayesProbeHarborAgentError(
                classify_trial_error(error),
                configuration=True,
            )
        if configuration_error is not None:
            raise configuration_error from None

        artifacts = TrialArtifactStore(
            Path(self.logs_dir) / "bayesprobe",
            restricted_values=(api_key,),
        )
        session: Any = None
        execution_error: BayesProbeHarborAgentError | None = None
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
                artifacts=artifacts,
            )
            artifacts = session.artifacts
            result = await asyncio.to_thread(session.runner.run_question, session.input)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            category = classify_trial_error(error)
            artifacts.append_error(
                {"category": category, "error_type": type(error).__name__}
            )
            if not self._emit_trajectory(
                artifacts=artifacts,
                session=session,
                instruction=instruction,
                config=config,
                api_key=api_key,
                stop_reason=category,
            ):
                category = "adapter_error"
            execution_error = BayesProbeHarborAgentError(category)
        if execution_error is not None:
            raise execution_error from None

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
            "runtime_budgets": {
                "max_total_actions": config.max_total_actions,
                "max_model_calls": config.max_model_calls,
                "max_provider_tokens": config.max_provider_tokens,
                "max_output_tokens": config.max_output_tokens,
                "command_timeout_seconds": config.command_timeout_seconds,
                "provider_timeout_seconds": config.provider_timeout_seconds,
                "signal_output_bytes": config.signal_output_bytes,
                "provider_tokens_used": _nonnegative_count(
                    getattr(session.budget, "provider_tokens_used", None),
                ),
            },
        }
        if not self._emit_trajectory(
            artifacts=session.artifacts,
            session=session,
            instruction=instruction,
            config=config,
            api_key=api_key,
            stop_reason=metadata["bayesprobe_stop_reason"],
        ):
            raise BayesProbeHarborAgentError("adapter_error") from None
        existing_metadata = context.metadata
        context.metadata = {
            **(dict(existing_metadata) if isinstance(existing_metadata, Mapping) else {}),
            **metadata,
        }
        session.artifacts.write_summary(metadata)

    def _emit_trajectory(
        self,
        *,
        artifacts: object,
        session: object,
        instruction: str,
        config: TerminalBenchConfig,
        api_key: str,
        stop_reason: object,
    ) -> bool:
        session_input = getattr(session, "input", None)
        run_id = _bounded_text(
            getattr(session_input, "run_id", None),
            fallback=str(self.context_id or self.session_id or "tb_harbor"),
            restricted_value=api_key,
        )
        try:
            write_atif_trajectory(
                logs_dir=self.logs_dir,
                artifact_root=getattr(
                    artifacts,
                    "root",
                    Path(self.logs_dir) / "bayesprobe",
                ),
                arm="bayesprobe",
                instruction=instruction,
                run_id=run_id,
                session_id=self.session_id,
                model_name=config.model,
                adapter_version=__version__,
                stop_reason=(
                    stop_reason if isinstance(stop_reason, str) else "adapter_error"
                ),
                budget=getattr(session, "budget", None),
                restricted_values=(api_key,),
            )
        except Exception as error:
            append_error = getattr(artifacts, "append_error", None)
            if callable(append_error):
                append_error(
                    {
                        "category": "adapter_error",
                        "error_type": type(error).__name__,
                        "stage": "trajectory_export",
                    }
                )
            return False
        return True
