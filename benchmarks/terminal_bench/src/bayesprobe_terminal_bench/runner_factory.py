from __future__ import annotations

import asyncio
import json
import re
import subprocess
from collections.abc import Mapping
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from openai import OpenAI

from bayesprobe import (
    AutonomousQuestionProgressObserver,
    AutonomousQuestionRunConfig,
    AutonomousQuestionRunner,
    AnswerContractOutline,
    AnswerValueType,
    BayesProbeCore,
    BayesProbeInitializer,
    CapabilityDescriptor,
    CapabilityKind,
    DeterministicModelGateway,
    EpistemicOrigin,
    EvidenceJudgmentRepairPolicy,
    ExplicitTaskFramer,
    HypothesisExpansionService,
    InitializeRunInput,
    JsonlLedgerStore,
    ModelGateway,
    ModelHypothesisExpansionAdapter,
    ModelProbeDesigner,
    ModelTaskFramer,
    OpenAIChatCompletionsModelGateway,
    OpenAIModelGatewayConfig,
    ProbeExecutor,
    ProbeToolGateway,
    ProviderRequestControls,
    RecordedTaskAdmitter,
    RoutingTaskFramer,
    StructuredModelRequest,
    TaskAdmissionDecision,
    TaskAdmissionStatus,
    TaskAwareAnswerProjector,
    TaskKind,
)

from bayesprobe_terminal_bench.artifacts import TrialArtifactStore
from bayesprobe_terminal_bench.causal import (
    CausalEvidenceModelGateway,
    CausalTraceRegistry,
)
from bayesprobe_terminal_bench.config import (
    BudgetExhausted,
    ProviderIdentityError,
    RunBudget,
    TerminalBenchConfig,
)
from bayesprobe_terminal_bench.deadline import (
    DeadlineEnvironmentBridge,
    DeadlineOpenAIClient,
    DeadlineTerminalPlanner,
    TrialDeadline,
)
from bayesprobe_terminal_bench.environment import ActionPolicy, HarborEnvironmentBridge
from bayesprobe_terminal_bench.experiment_lock import experiment_lock_sha256
from bayesprobe_terminal_bench.gateway import HarborProbeToolGateway
from bayesprobe_terminal_bench.planning import OpenAICompatibleTerminalProbePlanner
from bayesprobe_terminal_bench.provider_contract import TerminalContractModelGateway


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
_ACTIVE_LOCK_FIELD_TYPES: dict[str, type] = {
    **_LOCK_FIELD_TYPES,
    "max_provider_tokens": int,
    "agent_timeout_seconds": int,
    "expected_provider_model": str,
}
_ACTIVE_LOCK_EXACT_VALUES: dict[str, object] = {
    **{
        key: value
        for key, value in _LOCK_EXACT_VALUES.items()
        if key not in {"schema_version", "terminal_plan_version"}
    },
    "schema_version": "terminal_bench_lock:v1",
    "terminal_plan_version": "terminal_probe_plan:v1",
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
    task_admitter = RecordedTaskAdmitter(
        TaskAdmissionDecision(
            attempt_id="terminal_bench_admission_template",
            status=TaskAdmissionStatus.ADMITTED,
            epistemic_basis=[
                "Harbor supplied a fixed official benchmark task for execution."
            ],
            proposed_task_kind=TaskKind.DESIGN,
            answer_contract_outline=AnswerContractOutline(
                objective="Complete the supplied task in the benchmark environment.",
                answer_value_type=AnswerValueType.STRUCTURED_TEXT,
                decision_form="environment_change",
                permits_synthesis=True,
                required_sections=["result", "verification", "uncertainty"],
            ),
            reason="Official benchmark tasks are admitted by the evaluation harness.",
            model_trace={"source": "terminal_bench_harness:v1"},
        )
    )
    task_framer = RoutingTaskFramer(
        explicit_framer=ExplicitTaskFramer(),
        open_framer=ModelTaskFramer(model_gateway),
    )
    core = BayesProbeCore(
        ledger=ledger,
        model_gateway=model_gateway,
        judgment_repair_policy=EvidenceJudgmentRepairPolicy(max_attempts=2),
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
        try:
            response = self._delegate.complete_structured(request)
        except Exception:
            self._raise_pending_observer_error()
            raise
        self._raise_pending_observer_error()
        return response

    def _raise_pending_observer_error(self) -> None:
        observer = self.invocation_observer
        raise_pending = getattr(observer, "raise_pending", None)
        if callable(raise_pending):
            raise_pending()


@dataclass(frozen=True)
class LiveSession:
    runner: AutonomousQuestionRunner
    input: InitializeRunInput
    artifacts: TrialArtifactStore
    budget: RunBudget
    deadline: TrialDeadline
    runtime_lock_sha256: str


class ArtifactInvocationObserver:
    def __init__(
        self,
        artifacts: TrialArtifactStore,
        *,
        budget: RunBudget,
        expected_model: str,
        expected_system_fingerprint: str | None,
    ) -> None:
        self._artifacts = artifacts
        self._budget = budget
        self._expected_model = expected_model
        self._expected_system_fingerprint = expected_system_fingerprint
        self._captured_identities: deque[tuple[object, object]] = deque()
        self._pending: Exception | None = None
        self._lock = Lock()

    def capture_provider_response(self, response: object) -> None:
        identity = (
            _response_value(response, "model"),
            _response_value(response, "system_fingerprint"),
        )
        with self._lock:
            self._captured_identities.append(identity)

    def observe(self, record: object) -> None:
        to_dict = getattr(record, "to_dict", None)
        payload = to_dict() if callable(to_dict) else None
        if not isinstance(payload, Mapping):
            return
        safe_payload = dict(payload)
        failure: BudgetExhausted | None = None
        captured = self._pop_captured_identity()
        if payload.get("outcome") == "success" or captured is not None:
            try:
                model, fingerprint = (
                    captured
                    if captured is not None
                    else (
                        payload.get("model"),
                        payload.get("system_fingerprint"),
                    )
                )
                self._validate_identity(model=model, fingerprint=fingerprint)
                usage = payload.get("usage")
                self._budget.record_provider_usage(
                    _response_value(usage, "total_tokens")
                )
            except BudgetExhausted as error:
                failure = error
                self.remember_pending(error)
        try:
            self._artifacts.append_provider_call(safe_payload)
        except Exception as error:
            self.remember_pending(error)
            if failure is not None:
                raise failure from error
            raise
        if failure is not None:
            raise failure

    def observe_sdk_response(self, response: object, *, task: str) -> None:
        usage = _response_value(response, "usage")
        payload = {
            "task": task,
            "model": _response_value(response, "model"),
            "system_fingerprint": _response_value(
                response, "system_fingerprint"
            ),
            "response_id": _response_value(response, "id"),
            "finish_reason": _finish_reason(response),
            "usage": {
                "input_tokens": _response_value(usage, "prompt_tokens"),
                "output_tokens": _response_value(usage, "completion_tokens"),
                "total_tokens": _response_value(usage, "total_tokens"),
            },
            "outcome": "success",
        }
        failure: BudgetExhausted | None = None
        try:
            self._validate_identity(
                model=payload["model"],
                fingerprint=payload["system_fingerprint"],
            )
            self._budget.record_provider_usage(payload["usage"]["total_tokens"])
        except BudgetExhausted as error:
            failure = error
            self.remember_pending(error)
        try:
            self._artifacts.append_provider_call(payload)
        except Exception as error:
            self.remember_pending(error)
            if failure is not None:
                raise failure from error
            raise
        if failure is not None:
            raise failure

    def raise_pending(self) -> None:
        with self._lock:
            failure = self._pending
            self._pending = None
        if failure is not None:
            raise failure

    def _pop_captured_identity(self) -> tuple[object, object] | None:
        with self._lock:
            return self._captured_identities.popleft() if self._captured_identities else None

    def remember_pending(self, error: Exception) -> None:
        with self._lock:
            if self._pending is None:
                self._pending = error

    def _validate_identity(self, *, model: object, fingerprint: object) -> None:
        if model != self._expected_model:
            raise ProviderIdentityError("provider model identity drift")
        if fingerprint != self._expected_system_fingerprint:
            raise ProviderIdentityError("provider system fingerprint drift")


def _response_value(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _finish_reason(response: object) -> object:
    choices = _response_value(response, "choices")
    if not isinstance(choices, list | tuple) or not choices:
        return None
    return _response_value(choices[0], "finish_reason")


def safe_run_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.-") or "harbor"
    return f"tb_{normalized}"[:96]


class _SharedRegistryHarborProbeToolGateway(HarborProbeToolGateway):
    _CATEGORY_MAP = {
        "budget_exhausted": "budget_error",
        "causal_adapter_error": "causal_conformance_error",
        "plan_error": "adapter_error",
        "provider_error": "provider_transport_error",
    }

    def __init__(self, *, registry: CausalTraceRegistry, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._causal = registry

    def _record_decision(self, *, category: str, **kwargs: Any) -> None:
        super()._record_decision(
            category=self._CATEGORY_MAP.get(category, category),
            **kwargs,
        )


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


def validate_runtime_lock(
    path: Path,
    config: TerminalBenchConfig,
    *,
    arm: str,
    session_id: str | None,
    runtime_git_identity: RepositoryGitIdentity | None = None,
) -> object:
    payload = _read_runtime_lock(path)
    schema_version = payload.get("schema_version")
    if schema_version == "terminal_bench_lock:v1":
        return _validate_active_single_lock(
            payload,
            config,
            runtime_git_identity=runtime_git_identity,
        )
    if schema_version == "terminal_bench_paired_gate:v1":
        return _validate_active_paired_lock(
            payload,
            config,
            arm=arm,
            session_id=session_id,
            runtime_git_identity=runtime_git_identity,
        )
    if schema_version == "terminal_bench_causal_qualification:v1":
        return _validate_active_causal_qualification_lock(
            payload,
            config,
            arm=arm,
            session_id=session_id,
            runtime_git_identity=runtime_git_identity,
        )
    raise ValueError("active runtime lock mismatch: ['schema_version']")


def _read_runtime_lock(path: Path) -> dict[str, Any]:
    lock_path = Path(path)
    if not lock_path.is_file():
        raise ValueError("Terminal-Bench lock is required")
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise ValueError("Terminal-Bench lock must contain valid JSON") from None
    if not isinstance(payload, dict):
        raise ValueError("Terminal-Bench lock must be an object")
    return payload


def _validate_active_single_lock(
    payload: Mapping[str, Any],
    config: TerminalBenchConfig,
    *,
    runtime_git_identity: RepositoryGitIdentity | None,
) -> dict[str, object]:
    mismatches = _active_common_mismatches(
        payload,
        config,
        field_types=_ACTIVE_LOCK_FIELD_TYPES,
        exact_values=_ACTIVE_LOCK_EXACT_VALUES,
    )
    if mismatches:
        raise ValueError(f"active runtime lock mismatch: {sorted(mismatches)}")
    _validate_runtime_git_identity(payload, runtime_git_identity)
    return dict(payload)


def _validate_active_paired_lock(
    payload: Mapping[str, Any],
    config: TerminalBenchConfig,
    *,
    arm: str,
    session_id: str | None,
    runtime_git_identity: RepositoryGitIdentity | None,
) -> dict[str, object]:
    from bayesprobe_terminal_bench.experiment_lock import (
        FROZEN_GATE_TASK_IDS,
        FROZEN_GATE_TASK_REFS,
        PAIRED_GATE_ARMS,
    )

    field_types = {
        key: value
        for key, value in _ACTIVE_LOCK_FIELD_TYPES.items()
        if key
        not in {
            "task_id",
            "task_checksum",
            "container_image",
            "image_digest",
            "agent_timeout_seconds",
        }
    }
    exact_values = {
        key: value
        for key, value in _ACTIVE_LOCK_EXACT_VALUES.items()
        if key not in {"schema_version", "task_id"}
    }
    exact_values["schema_version"] = "terminal_bench_paired_gate:v1"
    mismatches = _active_common_mismatches(
        payload,
        config,
        field_types=field_types,
        exact_values=exact_values,
        include_agent_timeout=False,
    )
    if payload.get("arms") != PAIRED_GATE_ARMS:
        mismatches.add("arms")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or [
        item.get("task_id") if isinstance(item, Mapping) else None for item in tasks
    ] != list(FROZEN_GATE_TASK_IDS):
        mismatches.add("tasks")
        task_items: list[Mapping[str, Any]] = []
    else:
        task_items = [item for item in tasks if isinstance(item, Mapping)]
        if any(
            item.get("task_ref") != FROZEN_GATE_TASK_REFS[item["task_id"]]
            or type(item.get("image_digest")) is not str
            or not _LOCK_SHA256.fullmatch(item["image_digest"])
            or not isinstance(item.get("agent_timeout_seconds"), int)
            or isinstance(item.get("agent_timeout_seconds"), bool)
            or item["agent_timeout_seconds"] < 1
            for item in task_items
        ):
            mismatches.add("tasks")
    if arm not in PAIRED_GATE_ARMS:
        mismatches.add("arm")
    matched_task = next(
        (
            item
            for item in task_items
            if isinstance(session_id, str)
            and session_id.startswith(
                f"{str(item['task_id']).split('/', 1)[1]}__"
            )
        ),
        None,
    )
    if matched_task is None:
        mismatches.add("session_id")
    elif matched_task.get("agent_timeout_seconds") != config.task_timeout_seconds:
        mismatches.add("agent_timeout_seconds")
    if mismatches:
        raise ValueError(f"active runtime lock mismatch: {sorted(mismatches)}")
    _validate_runtime_git_identity(payload, runtime_git_identity)
    return dict(payload)


def _validate_active_causal_qualification_lock(
    payload: Mapping[str, Any],
    config: TerminalBenchConfig,
    *,
    arm: str,
    session_id: str | None,
    runtime_git_identity: RepositoryGitIdentity | None,
) -> dict[str, object]:
    from bayesprobe_terminal_bench.experiment_lock import CausalQualificationLock

    try:
        lock = CausalQualificationLock.model_validate(payload)
    except Exception:
        raise ValueError(
            "active runtime lock mismatch: ['causal_qualification_lock']"
        ) from None

    mismatches: set[str] = set()
    if arm != "bayesprobe":
        mismatches.add("arm")
    task = next(
        (
            item
            for item in lock.tasks
            if isinstance(session_id, str)
            and session_id.startswith(f"{item.task_id.split('/', 1)[1]}__")
        ),
        None,
    )
    if task is None:
        mismatches.add("session_id")
    elif config.task_timeout_seconds != task.agent_timeout_seconds:
        mismatches.add("agent_timeout_seconds")

    expected = {
        "model": config.model,
        "base_url": config.base_url,
        "max_total_actions": config.max_total_actions,
        "max_model_calls": config.max_model_calls,
        "max_provider_tokens": config.max_provider_tokens,
        "max_output_tokens": config.max_output_tokens,
        "command_timeout_seconds": config.command_timeout_seconds,
        "provider_timeout_seconds": config.provider_timeout_seconds,
        "signal_output_bytes": config.signal_output_bytes,
    }
    actual = {
        "model": lock.model,
        "base_url": lock.base_url,
        **lock.budgets.model_dump(mode="python"),
    }
    mismatches.update(
        key for key, expected_value in expected.items() if actual[key] != expected_value
    )
    if mismatches:
        raise ValueError(f"active runtime lock mismatch: {sorted(mismatches)}")

    result = lock.model_dump(mode="json")
    _validate_runtime_git_identity(result, runtime_git_identity)
    return result


def _active_common_mismatches(
    payload: Mapping[str, Any],
    config: TerminalBenchConfig,
    *,
    field_types: Mapping[str, type],
    exact_values: Mapping[str, object],
    include_agent_timeout: bool = True,
) -> set[str]:
    mismatches = {
        key
        for key, expected_type in field_types.items()
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
    expected: dict[str, object] = {
        **exact_values,
        "model": config.model,
        "base_url": config.base_url,
        "max_cycles": config.max_cycles,
        "max_probes_per_cycle": config.max_probes_per_cycle,
        "max_actions_per_probe": config.max_actions_per_probe,
        "max_total_actions": config.max_total_actions,
        "max_model_calls": config.max_model_calls,
        "max_provider_tokens": config.max_provider_tokens,
        "command_timeout_seconds": config.command_timeout_seconds,
        "provider_timeout_seconds": config.provider_timeout_seconds,
        "max_output_tokens": config.max_output_tokens,
        "signal_output_bytes": config.signal_output_bytes,
    }
    if include_agent_timeout:
        expected["agent_timeout_seconds"] = config.task_timeout_seconds
    mismatches.update(
        key for key, expected_value in expected.items() if payload.get(key) != expected_value
    )
    for key in ("dataset_revision", "task_checksum", "image_digest"):
        if key not in field_types:
            continue
        value = payload.get(key)
        if type(value) is not str or not _LOCK_SHA256.fullmatch(value):
            mismatches.add(key)
    for key in ("root_git_sha", "adapter_tree_sha"):
        value = payload.get(key)
        if type(value) is not str or not _LOCK_GIT_OBJECT_ID.fullmatch(value):
            mismatches.add(key)
    fingerprint = payload.get("expected_system_fingerprint")
    if "expected_system_fingerprint" not in payload or (
        fingerprint is not None
        and (type(fingerprint) is not str or not fingerprint.strip())
    ):
        mismatches.add("expected_system_fingerprint")
    expected_model = payload.get("expected_provider_model")
    if type(expected_model) is not str or not expected_model.strip():
        mismatches.add("expected_provider_model")
    return mismatches


def _validate_runtime_git_identity(
    payload: Mapping[str, Any],
    runtime_git_identity: RepositoryGitIdentity | None,
) -> None:
    runtime = runtime_git_identity or collect_repository_git_identity(
        Path(__file__).resolve().parents[4]
    )
    mismatches = {
        key
        for key, actual in {
            "root_git_sha": runtime.root_git_sha,
            "adapter_tree_sha": runtime.adapter_tree_sha,
        }.items()
        if payload.get(key) != actual
    }
    if runtime.adapter_dirty:
        mismatches.add("dirty_adapter_worktree")
    if mismatches:
        raise ValueError(f"active runtime lock mismatch: {sorted(mismatches)}")


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
    artifacts: TrialArtifactStore | None = None,
) -> LiveSession:
    runtime_lock = validate_runtime_lock(
        config.lock_path,
        config,
        arm="bayesprobe",
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
        Path(logs_dir) / "bayesprobe",
        restricted_values=(api_key,),
    )
    deadline = TrialDeadline(timeout_seconds=config.task_timeout_seconds)
    budget = RunBudget(
        max_actions=config.max_total_actions,
        max_model_calls=config.max_model_calls,
        max_provider_tokens=config.max_provider_tokens,
        reservation_guard=deadline.require_active,
    )
    observer = ArtifactInvocationObserver(
        artifacts,
        budget=budget,
        expected_model=expected_model,
        expected_system_fingerprint=expected_fingerprint,
    )
    base_client = OpenAI(
        api_key=api_key,
        base_url=config.base_url,
        timeout=config.provider_timeout_seconds,
        max_retries=0,
    )
    provider_client = DeadlineOpenAIClient(
        base_client=base_client,
        deadline=deadline,
        configured_timeout_seconds=config.provider_timeout_seconds,
        response_observer=observer.capture_provider_response,
        error_observer=observer.remember_pending,
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
        client=provider_client,
        invocation_observer=observer,
    )
    budgeted = BudgetedModelGateway(provider, budget)
    contracted = TerminalContractModelGateway(
        delegate=budgeted,
        artifacts=artifacts,
    )
    registry = CausalTraceRegistry()
    model_gateway = CausalEvidenceModelGateway(
        delegate=contracted,
        registry=registry,
        artifacts=artifacts,
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
    planner_client = DeadlineOpenAIClient(
        base_client=base_client,
        deadline=deadline,
        configured_timeout_seconds=config.provider_timeout_seconds,
        response_observer=lambda response: observer.observe_sdk_response(
            response,
            task="terminal_probe_plan",
        ),
        error_observer=observer.remember_pending,
    )
    raw_planner = OpenAICompatibleTerminalProbePlanner(
        config=config,
        budget=budget,
        client=planner_client,
        invocation_observer=lambda record: artifacts.append_contract_attempt(
            {"stage": "terminal_probe_plan", **record}
        ),
    )
    planner = DeadlineTerminalPlanner(
        delegate=raw_planner,
        deadline=deadline,
        accounting_observer=observer,
    )
    probe_gateway = _SharedRegistryHarborProbeToolGateway(
        planner=planner,
        bridge=bridge,
        artifacts=artifacts,
        budget=budget,
        registry=registry,
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
            "experiment_id": "terminal_bench_causal:v1",
            "arm": "bayesprobe",
            "sample_id": context_id or session_id or run_id,
        },
    )
    return LiveSession(
        runner=runner,
        input=input,
        artifacts=artifacts,
        budget=budget,
        deadline=deadline,
        runtime_lock_sha256=experiment_lock_sha256(runtime_lock),
    )
