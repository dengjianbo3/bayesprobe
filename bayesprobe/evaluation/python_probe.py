from __future__ import annotations

import hashlib
import math
import re
import subprocess
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable, Literal, Protocol

from bayesprobe.evidence_memory import derive_deterministic_computation_root
from bayesprobe.lifecycle import resolve_belief_lifecycle
from bayesprobe.model_gateway import (
    ModelGateway,
    ModelGatewayValidationError,
    StructuredModelRequest,
    model_gateway_adapter_kind,
    model_gateway_identity,
)
from bayesprobe.probe_executor import ProbeExecutionContext
from bayesprobe.schemas import (
    EpistemicOrigin,
    ExternalSignal,
    ProbeDesign,
    SignalKind,
    SignalProvenance,
)


_IMAGE_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_PLAN_KEYS = frozenset(
    {"mode", "purpose", "target_hypotheses", "expected_observation", "code"}
)
_PYTHON_SANDBOX_TOOL_IDENTITY = "python_sandbox:v1"
_PYTHON_SANDBOX_POLICY_IDENTITY = "python_sandbox_policy:v1"


class SandboxUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class PythonProbePlan:
    mode: str
    purpose: str
    target_hypotheses: tuple[str, ...]
    expected_observation: str
    code: str | None


@dataclass(frozen=True)
class ResolvedSandboxImage:
    requested_reference: str
    digest: str

    def __post_init__(self) -> None:
        _required_text(self.requested_reference, "sandbox image requested_reference")
        if not isinstance(self.digest, str) or not _IMAGE_DIGEST.fullmatch(
            self.digest.strip().lower()
        ):
            raise ValueError("sandbox image digest must be an immutable sha256 digest")
        object.__setattr__(self, "requested_reference", self.requested_reference.strip())
        object.__setattr__(self, "digest", self.digest.strip().lower())


@dataclass(frozen=True)
class DockerPythonSandboxConfig:
    image: str = "bayesprobe-hle-python:v0.1"
    timeout_seconds: float = 30.0
    max_output_bytes: int = 64 * 1024
    pids_limit: int = 64
    memory: str = "1g"
    cpus: float = 1.0
    tmpfs_size: str = "64m"
    user: str = "65532:65532"

    def __post_init__(self) -> None:
        _required_text(self.image, "sandbox image")
        if (
            type(self.timeout_seconds) not in (int, float)
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("sandbox timeout_seconds must be positive")
        for field_name in ("max_output_bytes", "pids_limit"):
            value = getattr(self, field_name)
            if type(value) is not int or value < 1:
                raise ValueError(f"sandbox {field_name} must be positive")
        if type(self.cpus) not in (int, float) or self.cpus <= 0:
            raise ValueError("sandbox cpus must be positive")
        for field_name in ("memory", "tmpfs_size", "user"):
            _required_text(getattr(self, field_name), f"sandbox {field_name}")
        object.__setattr__(self, "image", self.image.strip())


@dataclass(frozen=True)
class PythonExecutionRequest:
    execution_id: str
    run_id: str
    cycle_id: str
    probe_id: str
    code: str
    image: ResolvedSandboxImage
    repair_attempt_index: int = 0

    def __post_init__(self) -> None:
        for field_name in ("execution_id", "run_id", "cycle_id", "probe_id"):
            value = _required_text(
                getattr(self, field_name),
                f"Python execution {field_name}",
            )
            object.__setattr__(self, field_name, value)
        if not isinstance(self.code, str) or not self.code.strip():
            raise ValueError("Python execution code must not be empty")
        if not isinstance(self.image, ResolvedSandboxImage):
            raise ValueError("Python execution image must be resolved")
        if type(self.repair_attempt_index) is not int or self.repair_attempt_index < 0:
            raise ValueError("Python execution repair_attempt_index must be non-negative")


@dataclass(frozen=True)
class PythonExecutionRecord:
    execution_id: str
    run_id: str
    cycle_id: str
    probe_id: str
    code: str
    code_sha256: str
    image_digest: str
    started_at: str
    completed_at: str
    wall_seconds: float
    exit_code: int | None
    stdout: str
    stderr: str
    output_truncated: bool
    timed_out: bool
    policy_violation: bool
    repair_attempt_index: int
    policy_snapshot: dict[str, Any]

    @property
    def success(self) -> bool:
        return (
            self.exit_code == 0
            and not self.timed_out
            and not self.policy_violation
        )


class PythonExecutionObserver(Protocol):
    def observe(self, record: PythonExecutionRecord) -> None:
        ...


def python_probe_plan_from_mapping(
    payload: Any,
    *,
    allowed_hypothesis_ids: set[str],
) -> PythonProbePlan:
    if not isinstance(payload, Mapping):
        raise ValueError("Python probe plan must be an object")
    if set(payload) != _PLAN_KEYS:
        raise ValueError(
            "Python probe plan must contain exactly mode, purpose, "
            "target_hypotheses, expected_observation, and code"
        )
    mode = payload["mode"]
    if mode not in {"python", "reasoning"}:
        raise ValueError("Python probe plan mode must be python or reasoning")
    purpose = _required_text(payload["purpose"], "Python probe plan purpose")
    expected_observation = _required_text(
        payload["expected_observation"],
        "Python probe plan expected_observation",
    )
    raw_targets = payload["target_hypotheses"]
    if not isinstance(raw_targets, list | tuple) or not raw_targets:
        raise ValueError("Python probe plan target_hypotheses must not be empty")
    targets: list[str] = []
    for target in raw_targets:
        normalized = _required_text(target, "Python probe plan target hypothesis")
        if normalized not in allowed_hypothesis_ids:
            raise ValueError(f"Python probe plan has unknown target hypothesis: {normalized}")
        if normalized not in targets:
            targets.append(normalized)
    code = payload["code"]
    if mode == "python":
        if not isinstance(code, str) or not code.strip():
            raise ValueError("Python probe plan python mode requires non-empty code")
    elif code is not None:
        raise ValueError("Python probe plan reasoning mode must not contain code")
    return PythonProbePlan(
        mode=mode,
        purpose=purpose,
        target_hypotheses=tuple(targets),
        expected_observation=expected_observation,
        code=code,
    )


class DockerPythonSandbox:
    def __init__(
        self,
        config: DockerPythonSandboxConfig | None = None,
        *,
        run_command: Callable[..., Any] = subprocess.run,
        popen_factory: Callable[..., Any] = subprocess.Popen,
    ) -> None:
        self.config = config or DockerPythonSandboxConfig()
        self._run_command = run_command
        self._popen_factory = popen_factory

    def preflight(self) -> ResolvedSandboxImage:
        command = [
            "docker",
            "image",
            "inspect",
            self.config.image,
            "--format={{.Id}}",
        ]
        try:
            completed = self._run_command(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
        except FileNotFoundError as error:
            raise SandboxUnavailableError(
                "Docker is unavailable; no host execution fallback is permitted"
            ) from error
        except subprocess.TimeoutExpired as error:
            raise SandboxUnavailableError("Docker image preflight timed out") from error
        if completed.returncode != 0:
            raise SandboxUnavailableError("Docker sandbox image could not be resolved")
        digest = str(completed.stdout).strip().lower()
        if not _IMAGE_DIGEST.fullmatch(digest):
            raise SandboxUnavailableError(
                "Docker sandbox image did not resolve to an immutable sha256 digest"
            )
        return ResolvedSandboxImage(
            requested_reference=self.config.image,
            digest=digest,
        )

    def docker_command(
        self,
        image: ResolvedSandboxImage,
        *,
        container_name: str,
    ) -> list[str]:
        return [
            "docker",
            "run",
            "--rm",
            f"--name={container_name}",
            "--network=none",
            "--read-only",
            f"--user={self.config.user}",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            f"--pids-limit={self.config.pids_limit}",
            f"--memory={self.config.memory}",
            f"--cpus={self.config.cpus:g}",
            f"--tmpfs=/tmp:rw,nosuid,nodev,size={self.config.tmpfs_size}",
            "--env=PYTHONHASHSEED=0",
            "--env=OMP_NUM_THREADS=1",
            "--env=OPENBLAS_NUM_THREADS=1",
            "--env=MKL_NUM_THREADS=1",
            "--env=NUMEXPR_NUM_THREADS=1",
            "--interactive",
            image.digest,
            "python",
            "-s",
            "-",
        ]

    def policy_snapshot(self, *, image_digest: str) -> dict[str, Any]:
        if not isinstance(image_digest, str) or not _IMAGE_DIGEST.fullmatch(
            image_digest
        ):
            raise ValueError("sandbox policy image_digest must be immutable")
        return {
            "runtime": "docker",
            "image_digest": image_digest,
            "user": self.config.user,
            "resources": {
                "cpus": float(self.config.cpus),
                "memory": self.config.memory,
                "pids_limit": self.config.pids_limit,
            },
            "limits": {
                "timeout_seconds": float(self.config.timeout_seconds),
                "max_output_bytes": self.config.max_output_bytes,
            },
            "network": {"mode": "none"},
            "filesystem": {
                "read_only_root": True,
                "host_mounts": [],
                "tmpfs": [
                    {
                        "path": "/tmp",
                        "options": ["rw", "nosuid", "nodev"],
                        "size": self.config.tmpfs_size,
                    }
                ],
            },
            "security": {
                "cap_drop": ["ALL"],
                "no_new_privileges": True,
            },
            "environment": {
                "PYTHONHASHSEED": "0",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            },
            "interpreter": {
                "argv": ["python", "-s", "-"],
                "stdin": "interactive",
            },
        }

    def execute(self, request: PythonExecutionRequest) -> PythonExecutionRecord:
        container_name = _container_name(request.execution_id)
        command = self.docker_command(request.image, container_name=container_name)
        started_at = datetime.now(UTC)
        started_monotonic = time.monotonic()
        try:
            process = self._popen_factory(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as error:
            raise SandboxUnavailableError(
                "Docker is unavailable; no host execution fallback is permitted"
            ) from error

        capture = _BoundedOutputCapture(self.config.max_output_bytes)
        stdout_thread = threading.Thread(
            target=capture.drain,
            args=("stdout", process.stdout),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=capture.drain,
            args=("stderr", process.stderr),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        assert process.stdin is not None
        try:
            process.stdin.write(request.code.encode("utf-8"))
            process.stdin.close()
        except BrokenPipeError:
            pass

        timed_out = False
        try:
            process.wait(timeout=self.config.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            self._remove_container(container_name)
            process.kill()
            process.wait()
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)
        completed_at = datetime.now(UTC)
        stdout = capture.stdout.decode("utf-8", errors="replace")
        stderr = capture.stderr.decode("utf-8", errors="replace")
        return PythonExecutionRecord(
            execution_id=request.execution_id,
            run_id=request.run_id,
            cycle_id=request.cycle_id,
            probe_id=request.probe_id,
            code=request.code,
            code_sha256=hashlib.sha256(request.code.encode("utf-8")).hexdigest(),
            image_digest=request.image.digest,
            started_at=_utc_text(started_at),
            completed_at=_utc_text(completed_at),
            wall_seconds=max(0.0, time.monotonic() - started_monotonic),
            exit_code=process.returncode,
            stdout=stdout,
            stderr=stderr,
            output_truncated=capture.truncated,
            timed_out=timed_out,
            policy_violation=_looks_like_policy_violation(stderr),
            repair_attempt_index=request.repair_attempt_index,
            policy_snapshot=self.policy_snapshot(
                image_digest=request.image.digest,
            ),
        )

    def _remove_container(self, container_name: str) -> None:
        try:
            self._run_command(
                ["docker", "rm", "--force", container_name],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return


class PythonAugmentedProbeToolGateway:
    def __init__(
        self,
        model_gateway: ModelGateway,
        sandbox: DockerPythonSandbox,
        *,
        image: ResolvedSandboxImage | None = None,
        execution_observer: PythonExecutionObserver | None = None,
    ) -> None:
        self._model_gateway = model_gateway
        self._sandbox = sandbox
        self._image = image
        self._execution_observer = execution_observer
        self._process_counts = {
            "python_plans": 0,
            "python_plan_repairs": 0,
            "python_mode_plans": 0,
            "reasoning_plans": 0,
            "python_executions": 0,
            "python_repairs": 0,
            "python_successes": 0,
            "python_timeouts": 0,
            "python_policy_failures": 0,
        }

    @property
    def process_metrics(self) -> dict[str, int]:
        return dict(self._process_counts)

    def execute_probe(
        self,
        *,
        probe: ProbeDesign,
        context: ProbeExecutionContext,
    ) -> list[ExternalSignal]:
        provider_version = resolve_belief_lifecycle(
            context.belief_state
        ).provider_version
        model_identity = model_gateway_identity(self._model_gateway)
        try:
            plan = self._plan_probe(
                probe=probe,
                context=context,
                provider_version=provider_version,
            )
        except Exception:
            return [
                self._failure_signal(
                    probe=probe,
                    context=context,
                    reason="unverified Python probe planning failed",
                )
            ]
        self._process_counts["python_plans"] += 1
        if plan.mode == "reasoning":
            self._process_counts["reasoning_plans"] += 1
            try:
                return [
                    self._reasoning_signal(
                        plan=plan,
                        probe=probe,
                        context=context,
                        provider_version=provider_version,
                        model_identity=model_identity,
                    )
                ]
            except Exception:
                return [
                    self._failure_signal(
                        probe=probe,
                        context=context,
                        reason="unverified reasoning probe failed",
                    )
                ]
        self._process_counts["python_mode_plans"] += 1
        assert plan.code is not None
        image = self._resolved_image()
        first_record = self._execute_code(
            code=plan.code,
            probe=probe,
            context=context,
            image=image,
            repair_attempt_index=0,
        )
        record = first_record
        if _execution_requires_repair(first_record):
            try:
                repaired_code = self._repair_code(
                    plan=plan,
                    record=first_record,
                    probe=probe,
                    context=context,
                    provider_version=provider_version,
                )
            except Exception:
                return [
                    self._failure_signal(
                        probe=probe,
                        context=context,
                        reason="unverified Python code repair failed",
                        source=first_record.image_digest,
                    )
                ]
            if repaired_code is not None:
                record = self._execute_code(
                    code=repaired_code,
                    probe=probe,
                    context=context,
                    image=image,
                    repair_attempt_index=1,
                )
        if not record.success or not record.stdout.strip():
            reason = _execution_failure_reason(record)
            return [
                self._failure_signal(
                    probe=probe,
                    context=context,
                    reason=f"unverified Python sandbox result: {reason}",
                    source=record.image_digest,
                )
            ]
        return [self._execution_signal(plan=plan, record=record, probe=probe, context=context)]

    def _plan_probe(
        self,
        *,
        probe: ProbeDesign,
        context: ProbeExecutionContext,
        provider_version: Literal["v0.1", "v0.2"],
    ) -> PythonProbePlan:
        request_input = _probe_request_input(probe=probe, context=context)
        metadata = _probe_request_metadata(probe=probe, context=context)
        invalid_payload: Any = None
        try:
            invalid_payload = self._model_gateway.complete_structured(
                StructuredModelRequest(
                    task="plan_python_probe",
                    input=request_input,
                    prompt_id="python_probe_plan",
                    prompt_version=provider_version,
                    schema_name="PythonProbePlan",
                    schema_version=provider_version,
                    metadata=metadata,
                )
            )
            return python_probe_plan_from_mapping(
                invalid_payload,
                allowed_hypothesis_ids=set(
                    context.belief_state.hypotheses_by_id()
                ),
            )
        except (ModelGatewayValidationError, TypeError, ValueError) as error:
            validation_error = str(error)
        self._process_counts["python_plan_repairs"] += 1
        repaired_payload = self._model_gateway.complete_structured(
            StructuredModelRequest(
                task="repair_python_probe_plan",
                input={
                    **request_input,
                    "invalid_payload": invalid_payload,
                    "validation_error": validation_error,
                },
                prompt_id="python_probe_plan_repair",
                prompt_version=provider_version,
                schema_name="PythonProbePlan",
                schema_version=provider_version,
                metadata={**metadata, "repair_attempt_index": 1},
            )
        )
        return python_probe_plan_from_mapping(
            repaired_payload,
            allowed_hypothesis_ids=set(context.belief_state.hypotheses_by_id()),
        )

    def _reasoning_signal(
        self,
        *,
        plan: PythonProbePlan,
        probe: ProbeDesign,
        context: ProbeExecutionContext,
        provider_version: Literal["v0.1", "v0.2"],
        model_identity: str,
    ) -> ExternalSignal:
        payload = self._model_gateway.complete_structured(
            StructuredModelRequest(
                task="execute_probe",
                input=_probe_request_input(probe=probe, context=context),
                prompt_id="probe_execution",
                prompt_version=provider_version,
                schema_name="ProbeSignal",
                schema_version=provider_version,
                metadata=_probe_request_metadata(probe=probe, context=context),
            )
        )
        if not isinstance(payload, Mapping):
            raise ModelGatewayValidationError("reasoning probe signal must be an object")
        raw_content = payload.get("raw_content")
        if not isinstance(raw_content, str) or not raw_content.strip():
            raise ModelGatewayValidationError(
                "reasoning probe signal raw_content must not be empty"
            )
        adapter_kind = model_gateway_adapter_kind(self._model_gateway)
        return ExternalSignal(
            id=f"S_{context.cycle_id}_{probe.id}",
            cycle_id=context.cycle_id,
            signal_kind=SignalKind.ACTIVE,
            source_type="model_probe_gateway",
            source=f"model_gateway:{adapter_kind}",
            raw_content=raw_content.strip(),
            generated_by_probe=probe.id,
            initial_target_hypotheses=list(plan.target_hypotheses),
            provenance=SignalProvenance(
                epistemic_origin=EpistemicOrigin.MODEL_REASONING,
                source_identity=f"model_gateway:{model_identity}",
                provider_model_or_tool_identity=model_identity,
                session_id=context.run_id,
                derivation_root_id=(
                    f"model-probe:{context.run_id}:{context.cycle_id}:{probe.id}"
                ),
                correlation_group=f"model:{model_identity}:{context.run_id}",
                canonical_content_fingerprint="pending-normalization",
            ),
        )

    def _execute_code(
        self,
        *,
        code: str,
        probe: ProbeDesign,
        context: ProbeExecutionContext,
        image: ResolvedSandboxImage,
        repair_attempt_index: int,
    ) -> PythonExecutionRecord:
        record = self._sandbox.execute(
            PythonExecutionRequest(
                execution_id=(
                    f"{context.run_id}_{context.cycle_id}_{probe.id}_"
                    f"python_{repair_attempt_index}"
                ),
                run_id=context.run_id,
                cycle_id=context.cycle_id,
                probe_id=probe.id,
                code=code,
                image=image,
                repair_attempt_index=repair_attempt_index,
            )
        )
        self._process_counts["python_executions"] += 1
        if repair_attempt_index > 0:
            self._process_counts["python_repairs"] += 1
        if record.success and record.stdout.strip():
            self._process_counts["python_successes"] += 1
        if record.timed_out:
            self._process_counts["python_timeouts"] += 1
        if record.policy_violation:
            self._process_counts["python_policy_failures"] += 1
        self._observe_execution(record)
        return record

    def _repair_code(
        self,
        *,
        plan: PythonProbePlan,
        record: PythonExecutionRecord,
        probe: ProbeDesign,
        context: ProbeExecutionContext,
        provider_version: Literal["v0.1", "v0.2"],
    ) -> str | None:
        if record.timed_out or record.policy_violation:
            return None
        payload = self._model_gateway.complete_structured(
            StructuredModelRequest(
                task="repair_python_probe_code",
                input={
                    "purpose": plan.purpose,
                    "expected_observation": plan.expected_observation,
                    "original_code": record.code,
                    "execution_error": {
                        "exit_code": record.exit_code,
                        "stdout": record.stdout[-4096:],
                        "stderr": record.stderr[-4096:],
                        "empty_output": not record.stdout.strip(),
                    },
                },
                prompt_id="python_probe_code_repair",
                prompt_version=provider_version,
                schema_name="PythonCodeRepair",
                schema_version=provider_version,
                metadata={
                    **_probe_request_metadata(probe=probe, context=context),
                    "repair_attempt_index": 1,
                },
            )
        )
        if not isinstance(payload, Mapping) or set(payload) != {"code"}:
            return None
        code = payload.get("code")
        return code if isinstance(code, str) and code.strip() else None

    def _resolved_image(self) -> ResolvedSandboxImage:
        if self._image is None:
            self._image = self._sandbox.preflight()
        return self._image

    def _observe_execution(self, record: PythonExecutionRecord) -> None:
        if self._execution_observer is None:
            return
        try:
            self._execution_observer.observe(record)
        except Exception:
            return

    def _execution_signal(
        self,
        *,
        plan: PythonProbePlan,
        record: PythonExecutionRecord,
        probe: ProbeDesign,
        context: ProbeExecutionContext,
    ) -> ExternalSignal:
        policy_snapshot = dict(record.policy_snapshot)
        if policy_snapshot.get("image_digest") != record.image_digest:
            raise ValueError("Python execution policy image does not match record")
        environment_state_id = derive_deterministic_computation_root(
            tool_identity=_PYTHON_SANDBOX_POLICY_IDENTITY,
            computation_inputs=policy_snapshot,
        )
        derivation_root_id = derive_deterministic_computation_root(
            tool_identity=_PYTHON_SANDBOX_TOOL_IDENTITY,
            computation_inputs={
                "code": record.code,
                "plan": {
                    "purpose": plan.purpose,
                    "expected_observation": plan.expected_observation,
                    "target_hypotheses": sorted(plan.target_hypotheses),
                },
                "environment": policy_snapshot,
            },
        )
        raw_content = "\n".join(
            [
                f"purpose: {plan.purpose}",
                f"expected_observation: {plan.expected_observation}",
                f"execution_id: {record.execution_id}",
                f"exit_code: {record.exit_code}",
                f"code:\n{record.code}",
                f"stdout:\n{record.stdout.rstrip()}",
                f"stderr:\n{record.stderr.rstrip()}",
            ]
        )
        return ExternalSignal(
            id=f"S_{context.cycle_id}_{probe.id}",
            cycle_id=context.cycle_id,
            signal_kind=SignalKind.ACTIVE,
            source_type="python_sandbox",
            source=record.image_digest,
            raw_content=raw_content,
            generated_by_probe=probe.id,
            initial_target_hypotheses=list(plan.target_hypotheses),
            provenance=SignalProvenance(
                epistemic_origin=EpistemicOrigin.TOOL_RESULT,
                source_identity=(
                    f"{_PYTHON_SANDBOX_TOOL_IDENTITY}:{record.image_digest}"
                ),
                provider_model_or_tool_identity=_PYTHON_SANDBOX_TOOL_IDENTITY,
                derivation_root_id=derivation_root_id,
                correlation_group=(
                    f"tool:{_PYTHON_SANDBOX_TOOL_IDENTITY}:{record.image_digest}"
                ),
                canonical_content_fingerprint="pending-normalization",
                environment_state_id=environment_state_id,
            ),
        )

    def _failure_signal(
        self,
        *,
        probe: ProbeDesign,
        context: ProbeExecutionContext,
        reason: str,
        source: str = "python_sandbox_unavailable",
    ) -> ExternalSignal:
        return ExternalSignal(
            id=f"S_{context.cycle_id}_{probe.id}",
            cycle_id=context.cycle_id,
            signal_kind=SignalKind.ACTIVE,
            source_type="python_sandbox",
            source=source,
            raw_content=reason,
            generated_by_probe=probe.id,
            initial_target_hypotheses=list(probe.target_hypotheses),
        )


class _BoundedOutputCapture:
    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._captured = 0
        self._values = {"stdout": bytearray(), "stderr": bytearray()}
        self._lock = threading.Lock()
        self.truncated = False

    @property
    def stdout(self) -> bytes:
        return bytes(self._values["stdout"])

    @property
    def stderr(self) -> bytes:
        return bytes(self._values["stderr"])

    def drain(self, name: str, stream: Any) -> None:
        if stream is None:
            return
        while True:
            chunk = stream.read(4096)
            if not chunk:
                return
            with self._lock:
                remaining = self._limit - self._captured
                if remaining > 0:
                    captured_chunk = chunk[:remaining]
                    self._values[name].extend(captured_chunk)
                    self._captured += len(captured_chunk)
                if len(chunk) > max(0, remaining):
                    self.truncated = True


def _probe_request_input(
    *,
    probe: ProbeDesign,
    context: ProbeExecutionContext,
) -> dict[str, Any]:
    problem = context.metadata.get("problem", "")
    initial_context = context.metadata.get("initial_context", "")
    return {
        "problem": problem if isinstance(problem, str) else "",
        "initial_context": initial_context if isinstance(initial_context, str) else "",
        "probe": {
            "id": probe.id,
            "inquiry_goal": probe.inquiry_goal,
            "method": probe.method,
            "target_hypotheses": list(probe.target_hypotheses),
            "support_condition": dict(probe.support_condition),
            "weaken_condition": dict(probe.weaken_condition),
        },
        "hypotheses": [
            {
                "id": hypothesis.id,
                "statement": hypothesis.statement,
                "scope": hypothesis.scope,
                "posterior": hypothesis.posterior,
                "predictions": list(hypothesis.predictions),
                "falsifiers": list(hypothesis.falsifiers),
            }
            for hypothesis in context.belief_state.hypotheses
        ],
    }


def _probe_request_metadata(
    *,
    probe: ProbeDesign,
    context: ProbeExecutionContext,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "run_id": context.run_id,
        "cycle_id": context.cycle_id,
        "probe_id": probe.id,
    }
    for key in ("experiment_id", "arm", "sample_id"):
        value = context.metadata.get(key)
        if isinstance(value, str) and value:
            metadata[key] = value
    return metadata


def _execution_requires_repair(record: PythonExecutionRecord) -> bool:
    if record.timed_out or record.policy_violation:
        return False
    return record.exit_code != 0 or not record.stdout.strip()


def _execution_failure_reason(record: PythonExecutionRecord) -> str:
    if record.timed_out:
        return "execution timed out"
    if record.policy_violation:
        return "sandbox policy violation"
    if record.exit_code != 0:
        return f"execution failed with exit code {record.exit_code}"
    if not record.stdout.strip():
        return "execution produced empty output"
    return "execution did not produce a usable result"


def _container_name(execution_id: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_.-]", "-", execution_id).strip("-.")
    suffix = hashlib.sha256(execution_id.encode("utf-8")).hexdigest()[:10]
    prefix = normalized[:40] or "execution"
    return f"bayesprobe-{prefix}-{suffix}".lower()


def _looks_like_policy_violation(stderr: str) -> bool:
    lowered = stderr.lower()
    return any(
        marker in lowered
        for marker in (
            "operation not permitted",
            "permission denied",
            "network is unreachable",
            "read-only file system",
        )
    )


def _required_text(value: Any, owner: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{owner} must not be empty")
    return value.strip()


def _utc_text(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


__all__ = [
    "DockerPythonSandbox",
    "DockerPythonSandboxConfig",
    "PythonExecutionRecord",
    "PythonExecutionRequest",
    "PythonExecutionObserver",
    "PythonAugmentedProbeToolGateway",
    "PythonProbePlan",
    "ResolvedSandboxImage",
    "SandboxUnavailableError",
    "python_probe_plan_from_mapping",
]
