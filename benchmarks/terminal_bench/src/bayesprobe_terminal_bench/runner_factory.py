from __future__ import annotations

import asyncio
import json
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bayesprobe import (
    AutonomousQuestionProgressObserver,
    AutonomousQuestionRunConfig,
    AutonomousQuestionRunner,
    BayesProbeCore,
    BayesProbeInitializer,
    CapabilityDescriptor,
    CapabilityKind,
    DeterministicModelGateway,
    EpistemicOrigin,
    ExplicitTaskAdmitter,
    ExplicitTaskFramer,
    HypothesisExpansionService,
    InitializeRunInput,
    JsonlLedgerStore,
    ModelGateway,
    ModelHypothesisExpansionAdapter,
    ModelProbeDesigner,
    ModelTaskAdmitter,
    ModelTaskFramer,
    OpenAIChatCompletionsModelGateway,
    OpenAIModelGatewayConfig,
    ProbeExecutor,
    ProbeToolGateway,
    ProviderRequestControls,
    RoutingTaskAdmitter,
    RoutingTaskFramer,
    StructuredModelRequest,
    TaskAwareAnswerProjector,
)

from bayesprobe_terminal_bench.artifacts import TrialArtifactStore
from bayesprobe_terminal_bench.config import RunBudget, TerminalBenchConfig
from bayesprobe_terminal_bench.environment import ActionPolicy, HarborEnvironmentBridge
from bayesprobe_terminal_bench.gateway import HarborProbeToolGateway
from bayesprobe_terminal_bench.planning import OpenAICompatibleTerminalProbePlanner


_LOCK_FIELD_TYPES: dict[str, type] = {
    "schema_version": str,
    "harbor_version": str,
    "dataset_name": str,
    "dataset_revision": str,
    "task_id": str,
    "task_checksum": str,
    "container_image": str,
    "image_digest": str,
    "root_git_sha": str,
    "adapter_tree_sha": str,
    "n_attempts": int,
    "model": str,
    "provider_protocol": str,
    "api_key_env": str,
    "temperature": int,
    "max_cycles": int,
    "max_probes_per_cycle": int,
    "max_actions_per_probe": int,
    "max_total_actions": int,
    "max_model_calls": int,
    "command_timeout_seconds": int,
    "provider_timeout_seconds": int,
    "max_output_tokens": int,
    "signal_output_bytes": int,
    "terminal_plan_version": str,
}
_LOCK_EXACT_VALUES: dict[str, object] = {
    "schema_version": "terminal_bench_lock:v0.1",
    "harbor_version": "0.18.0",
    "dataset_name": "terminal-bench/terminal-bench-2",
    "task_id": "terminal-bench/break-filter-js-from-html",
    "n_attempts": 1,
    "provider_protocol": "openai_chat_completions",
    "api_key_env": "BAYESPROBE_BENCH_API_KEY",
    "temperature": 0,
    "terminal_plan_version": "terminal_probe_plan:v0.1",
}
_LOCK_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_LOCK_GIT_OBJECT_ID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


@dataclass(frozen=True)
class RepositoryGitIdentity:
    root_git_sha: str
    adapter_tree_sha: str
    adapter_dirty: bool


def collect_repository_git_identity(
    repository_root: Path,
) -> RepositoryGitIdentity:
    root = Path(repository_root).resolve()
    root_git_sha = _git_output(["git", "rev-parse", "HEAD"], cwd=root)
    adapter_tree_sha = _git_output(
        ["git", "rev-parse", "HEAD:benchmarks/terminal_bench"],
        cwd=root,
    )
    adapter_status = _git_output(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            "benchmarks/terminal_bench",
        ],
        cwd=root,
        allow_empty=True,
    )
    return RepositoryGitIdentity(
        root_git_sha=root_git_sha,
        adapter_tree_sha=adapter_tree_sha,
        adapter_dirty=bool(adapter_status),
    )


def _git_output(
    command: list[str],
    *,
    cwd: Path,
    allow_empty: bool = False,
) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        raise ValueError("could not resolve repository Git identity") from None
    output = completed.stdout.strip()
    if not output and not allow_empty:
        raise ValueError("could not resolve repository Git identity")
    return output


def terminal_bench_lock_schema_mismatches(
    payload: Mapping[str, Any],
) -> set[str]:
    mismatches = {
        key
        for key, expected_type in _LOCK_FIELD_TYPES.items()
        if key not in payload or type(payload[key]) is not expected_type
    }
    if "base_url" not in payload or (
        payload.get("base_url") is not None
        and (
            type(payload.get("base_url")) is not str
            or not str(payload["base_url"]).strip()
        )
    ):
        mismatches.add("base_url")
    for key, expected_value in _LOCK_EXACT_VALUES.items():
        if payload.get(key) != expected_value:
            mismatches.add(key)
    for key, expected_type in _LOCK_FIELD_TYPES.items():
        if (
            expected_type is str
            and type(payload.get(key)) is str
            and not str(payload[key]).strip()
        ):
            mismatches.add(key)
    for key in ("dataset_revision", "task_checksum", "image_digest"):
        value = payload.get(key)
        if type(value) is not str or not _LOCK_SHA256.fullmatch(value):
            mismatches.add(key)
    for key in ("root_git_sha", "adapter_tree_sha"):
        value = payload.get(key)
        if type(value) is not str or not _LOCK_GIT_OBJECT_ID.fullmatch(value):
            mismatches.add(key)
    for key, expected_type in _LOCK_FIELD_TYPES.items():
        if (
            expected_type is int
            and key != "temperature"
            and (type(payload.get(key)) is not int or int(payload[key]) <= 0)
        ):
            mismatches.add(key)
    return mismatches


def _terminal_capabilities() -> tuple[CapabilityDescriptor, ...]:
    return (
        CapabilityDescriptor(
            kind=CapabilityKind.REPOSITORY_READ,
            available=True,
            epistemic_origin=EpistemicOrigin.TOOL_RESULT,
            executor_adapter_id="harbor_terminal:v1",
            quality_caps={"verifiability": 0.95, "independence": 0.8},
        ),
        CapabilityDescriptor(
            kind=CapabilityKind.TEST_EXECUTION,
            available=True,
            epistemic_origin=EpistemicOrigin.TOOL_RESULT,
            executor_adapter_id="harbor_terminal:v1",
            quality_caps={"verifiability": 1.0, "independence": 0.9},
        ),
    )


def build_runner(
    *,
    model_gateway: ModelGateway,
    probe_gateway: ProbeToolGateway,
    ledger_path: str | Path,
    config: TerminalBenchConfig,
    progress_observer: AutonomousQuestionProgressObserver | None = None,
) -> AutonomousQuestionRunner:
    ledger = JsonlLedgerStore(Path(ledger_path))
    task_admitter = RoutingTaskAdmitter(
        explicit_admitter=ExplicitTaskAdmitter(),
        open_admitter=ModelTaskAdmitter(model_gateway),
    )
    task_framer = RoutingTaskFramer(
        explicit_framer=ExplicitTaskFramer(),
        open_framer=ModelTaskFramer(model_gateway),
    )
    core = BayesProbeCore(
        ledger=ledger,
        model_gateway=model_gateway,
        hypothesis_expander=HypothesisExpansionService(
            adapter=ModelHypothesisExpansionAdapter(model_gateway)
        ),
    )
    # The public deterministic gateway judges evidence only. Its conformance
    # runs therefore retain the runner's public deterministic projector.
    answer_projector = (
        None
        if type(model_gateway) is DeterministicModelGateway
        else TaskAwareAnswerProjector(model_gateway)
    )
    return AutonomousQuestionRunner(
        core=core,
        initializer=BayesProbeInitializer(
            ledger=ledger,
            task_framer=task_framer,
            task_admitter=task_admitter,
        ),
        executor=ProbeExecutor(gateway=probe_gateway, ledger=ledger),
        config=AutonomousQuestionRunConfig(
            max_cycles=config.max_cycles,
            max_probes_per_cycle=config.max_probes_per_cycle,
            stop_on_no_probes=True,
            confidence_threshold=None,
            posterior_delta_threshold=None,
        ),
        progress_observer=progress_observer,
        task_admitter=task_admitter,
        probe_designer=ModelProbeDesigner(model_gateway),
        available_capabilities=_terminal_capabilities(),
        answer_projector=answer_projector,
    )


class BudgetedModelGateway:
    def __init__(self, delegate: ModelGateway, budget: RunBudget) -> None:
        self._delegate = delegate
        self._budget = budget

    @property
    def adapter_kind(self) -> str:
        value = getattr(self._delegate, "adapter_kind", None)
        return value if isinstance(value, str) and value.strip() else type(self._delegate).__name__

    @property
    def model_identity(self) -> str:
        value = getattr(self._delegate, "model_identity", None)
        return value if isinstance(value, str) and value.strip() else self.adapter_kind

    @property
    def config(self) -> Any:
        return getattr(self._delegate, "config", None)

    @property
    def invocation_observer(self) -> object | None:
        return getattr(self._delegate, "invocation_observer", None)

    def complete_structured(
        self,
        request: StructuredModelRequest,
    ) -> dict[str, Any]:
        self._budget.reserve_model_call()
        return self._delegate.complete_structured(request)


@dataclass(frozen=True)
class LiveSession:
    runner: AutonomousQuestionRunner
    input: InitializeRunInput
    artifacts: TrialArtifactStore
    budget: RunBudget


class ArtifactInvocationObserver:
    def __init__(self, artifacts: TrialArtifactStore) -> None:
        self._artifacts = artifacts

    def observe(self, record: object) -> None:
        try:
            to_dict = getattr(record, "to_dict", None)
            payload = to_dict() if callable(to_dict) else None
            if not isinstance(payload, Mapping):
                return
            self._artifacts.append_provider_call(dict(payload))
        except Exception:
            return


def safe_run_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.-") or "harbor"
    return f"tb_{normalized}"[:96]


def load_and_validate_lock(
    path: Path,
    config: TerminalBenchConfig,
    *,
    runtime_git_identity: RepositoryGitIdentity | None = None,
    repository_root: Path | None = None,
) -> dict[str, object]:
    lock_path = Path(path)
    if not lock_path.is_file():
        raise ValueError("Terminal-Bench lock is required")
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise ValueError("Terminal-Bench lock must contain valid JSON") from None
    if not isinstance(payload, dict):
        raise ValueError("Terminal-Bench lock must be an object")
    expected = {
        "schema_version": "terminal_bench_lock:v0.1",
        "harbor_version": "0.18.0",
        "dataset_name": "terminal-bench/terminal-bench-2",
        "task_id": "terminal-bench/break-filter-js-from-html",
        "model": config.model,
        "base_url": config.base_url,
        "provider_protocol": "openai_chat_completions",
        "api_key_env": config.api_key_env,
        "temperature": 0,
        "max_cycles": config.max_cycles,
        "max_probes_per_cycle": config.max_probes_per_cycle,
        "max_actions_per_probe": config.max_actions_per_probe,
        "max_total_actions": config.max_total_actions,
        "max_model_calls": config.max_model_calls,
        "command_timeout_seconds": config.command_timeout_seconds,
        "provider_timeout_seconds": config.provider_timeout_seconds,
        "max_output_tokens": config.max_output_tokens,
        "signal_output_bytes": config.signal_output_bytes,
        "terminal_plan_version": "terminal_probe_plan:v0.1",
    }
    mismatches = terminal_bench_lock_schema_mismatches(payload) | {
        key
        for key, expected_value in expected.items()
        if key not in payload
        or type(payload[key]) is not type(expected_value)
        or payload[key] != expected_value
    }
    if mismatches:
        raise ValueError(f"Terminal-Bench lock mismatch: {sorted(mismatches)}")
    runtime = runtime_git_identity or collect_repository_git_identity(
        repository_root or Path(__file__).resolve().parents[4]
    )
    runtime_mismatches = {
        key
        for key, actual in {
            "root_git_sha": runtime.root_git_sha,
            "adapter_tree_sha": runtime.adapter_tree_sha,
        }.items()
        if payload.get(key) != actual
    }
    if runtime.adapter_dirty:
        runtime_mismatches.add("dirty_adapter_worktree")
    if runtime_mismatches:
        raise ValueError(
            f"Terminal-Bench lock runtime mismatch: {sorted(runtime_mismatches)}"
        )
    return dict(payload)


def build_live_session(
    *,
    config: TerminalBenchConfig,
    api_key: str,
    instruction: str,
    environment: object,
    event_loop: asyncio.AbstractEventLoop,
    logs_dir: str | Path,
    session_id: str | None,
    context_id: str | None,
    runtime_git_identity: RepositoryGitIdentity | None = None,
) -> LiveSession:
    load_and_validate_lock(
        config.lock_path,
        config,
        runtime_git_identity=runtime_git_identity,
    )
    artifacts = TrialArtifactStore(
        Path(logs_dir) / "bayesprobe",
        restricted_values=(api_key,),
    )
    budget = RunBudget(
        max_actions=config.max_total_actions,
        max_model_calls=config.max_model_calls,
    )
    provider = OpenAIChatCompletionsModelGateway(
        config=OpenAIModelGatewayConfig(
            model=config.model,
            api_key_env=config.api_key_env,
            timeout_seconds=config.provider_timeout_seconds,
            max_output_tokens=config.max_output_tokens,
            base_url=config.base_url,
            request_controls=ProviderRequestControls(temperature=0),
        ),
        api_key=api_key,
        invocation_observer=ArtifactInvocationObserver(artifacts),
    )
    model_gateway = BudgetedModelGateway(provider, budget)
    bridge = HarborEnvironmentBridge(
        loop=event_loop,
        environment=environment,
        policy=ActionPolicy(),
        output_limit_bytes=config.signal_output_bytes,
    )
    planner = OpenAICompatibleTerminalProbePlanner(
        config=config,
        budget=budget,
        api_key=api_key,
        invocation_observer=artifacts.append_provider_call,
    )
    probe_gateway = HarborProbeToolGateway(
        planner=planner,
        bridge=bridge,
        artifacts=artifacts,
        budget=budget,
    )
    run_id = safe_run_id(context_id or session_id or "harbor")
    runner = build_runner(
        model_gateway=model_gateway,
        probe_gateway=probe_gateway,
        ledger_path=artifacts.root / "bayesprobe_ledger.jsonl",
        config=config,
    )
    input = InitializeRunInput(
        run_id=run_id,
        problem=instruction,
        task_context=(
            "Work only in the provided Terminal-Bench task environment. "
            "Use observable repository and test results to diagnose, modify, "
            "and verify the task; Harbor runs the official verifier afterward."
        ),
        metadata={
            "experiment_id": "terminal_bench_engineering_v0.1",
            "arm": "bayesprobe",
            "sample_id": context_id or session_id or run_id,
        },
    )
    return LiveSession(
        runner=runner,
        input=input,
        artifacts=artifacts,
        budget=budget,
    )
