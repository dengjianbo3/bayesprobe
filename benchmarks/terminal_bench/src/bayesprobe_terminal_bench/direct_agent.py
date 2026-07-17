from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from openai import OpenAI

from bayesprobe_terminal_bench import __version__
from bayesprobe_terminal_bench.artifacts import TrialArtifactStore
from bayesprobe_terminal_bench.config import (
    RunBudget,
    TerminalBenchConfig,
    classify_trial_error,
)
from bayesprobe_terminal_bench.deadline import (
    DeadlineEnvironmentBridge,
    DeadlineOpenAIClient,
    TrialDeadline,
)
from bayesprobe_terminal_bench.environment import ActionPolicy, HarborEnvironmentBridge
from bayesprobe_terminal_bench.react import (
    OpenAICompatibleReActPlanner,
    ReActController,
)
from bayesprobe_terminal_bench.runner_factory import (
    RepositoryGitIdentity,
    validate_runtime_lock,
)


_MAX_METADATA_TEXT_LENGTH = 96


class DirectHarborAgentError(RuntimeError):
    def __init__(self, category: str, *, configuration: bool = False) -> None:
        self.category = category
        prefix = (
            "Direct Harbor agent configuration failed"
            if configuration
            else "Direct Harbor agent failed"
        )
        super().__init__(f"{prefix}: {category}")


@dataclass(frozen=True)
class DirectSession:
    controller: ReActController
    artifacts: TrialArtifactStore
    budget: RunBudget
    deadline: TrialDeadline


def build_direct_session(
    *,
    config: TerminalBenchConfig,
    api_key: str,
    environment: object,
    event_loop: asyncio.AbstractEventLoop,
    logs_dir: str | Path,
    session_id: str | None = None,
    runtime_git_identity: RepositoryGitIdentity | None = None,
    artifacts: TrialArtifactStore | None = None,
    **_: object,
) -> DirectSession:
    runtime_lock = validate_runtime_lock(
        config.lock_path,
        config,
        arm="direct",
        session_id=session_id,
        runtime_git_identity=runtime_git_identity,
    )
    if config.task_timeout_seconds is None:
        raise ValueError("active runtime requires the official task timeout")
    if not isinstance(runtime_lock, Mapping):
        raise ValueError("active runtime lock must be a mapping")
    expected_model = runtime_lock.get("expected_provider_model")
    expected_fingerprint = runtime_lock.get("expected_system_fingerprint")
    if not isinstance(expected_model, str) or not expected_model:
        raise ValueError("active runtime lock is missing provider identity")
    if expected_fingerprint is not None and not isinstance(expected_fingerprint, str):
        raise ValueError("active runtime lock has invalid provider identity")
    artifacts = artifacts or TrialArtifactStore(
        Path(logs_dir) / "direct",
        restricted_values=(api_key,),
    )
    deadline = TrialDeadline(timeout_seconds=config.task_timeout_seconds)
    budget = RunBudget(
        max_actions=config.max_total_actions,
        max_model_calls=config.max_model_calls,
        max_provider_tokens=config.max_provider_tokens,
        reservation_guard=deadline.require_active,
    )
    base_client = OpenAI(
        api_key=api_key,
        base_url=config.base_url,
        timeout=config.provider_timeout_seconds,
        max_retries=0,
    )
    raw_bridge = HarborEnvironmentBridge(
        loop=event_loop,
        environment=environment,
        policy=ActionPolicy(),
        output_limit_bytes=config.signal_output_bytes,
    )
    bridge = DeadlineEnvironmentBridge(
        delegate=raw_bridge,
        deadline=deadline,
        configured_timeout_seconds=config.command_timeout_seconds,
    )
    client = DeadlineOpenAIClient(
        base_client=base_client,
        deadline=deadline,
        configured_timeout_seconds=config.provider_timeout_seconds,
    )
    planner = OpenAICompatibleReActPlanner(
        config=config,
        budget=budget,
        client=client,
        invocation_observer=artifacts.append_provider_call,
        expected_provider_model=expected_model,
        expected_system_fingerprint=expected_fingerprint,
    )
    return DirectSession(
        controller=ReActController(
            planner=planner,
            bridge=bridge,
            artifacts=artifacts,
            budget=budget,
        ),
        artifacts=artifacts,
        budget=budget,
        deadline=deadline,
    )


def _bounded_text(value: object, *, fallback: str, restricted_value: str) -> str:
    if not isinstance(value, str):
        return fallback
    redacted = value.replace(restricted_value, "[REDACTED]")
    return redacted[:_MAX_METADATA_TEXT_LENGTH] or fallback


def _bounded_count(value: object, *, maximum: int) -> int:
    if type(value) is not int:
        return 0
    return min(max(value, 0), maximum)


class DirectHarborAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return "bayesprobe-direct"

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
        configuration_error: DirectHarborAgentError | None = None
        try:
            config, api_key = TerminalBenchConfig.from_sources(self.extra_env)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            configuration_error = DirectHarborAgentError(
                classify_trial_error(error),
                configuration=True,
            )
        if configuration_error is not None:
            raise configuration_error from None

        artifacts = TrialArtifactStore(
            Path(self.logs_dir) / "direct",
            restricted_values=(api_key,),
        )
        execution_error: DirectHarborAgentError | None = None
        try:
            session = build_direct_session(
                config=config,
                api_key=api_key,
                instruction=instruction,
                environment=environment,
                event_loop=asyncio.get_running_loop(),
                logs_dir=self.logs_dir,
                session_id=self.session_id,
                context_id=(
                    str(self.context_id) if self.context_id is not None else None
                ),
                artifacts=artifacts,
            )
            artifacts = session.artifacts
            result = await asyncio.to_thread(session.controller.run, instruction)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            category = classify_trial_error(error)
            artifacts.append_error(
                {"category": category, "error_type": type(error).__name__}
            )
            execution_error = DirectHarborAgentError(category)
        if execution_error is not None:
            raise execution_error from None

        metadata: dict[str, Any] = {
            "experiment_arm": "direct",
            "stop_reason": _bounded_text(
                getattr(result, "stop_reason", None),
                fallback="unknown",
                restricted_value=api_key,
            ),
            "react_steps": _bounded_count(
                getattr(result, "steps", None),
                maximum=config.max_model_calls,
            ),
            "observations": _bounded_count(
                getattr(result, "observations", None),
                maximum=config.max_total_actions,
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
