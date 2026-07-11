from types import SimpleNamespace

import pytest

from bayesprobe.evaluation.python_probe import (
    DockerPythonSandbox,
    DockerPythonSandboxConfig,
    PythonExecutionRecord,
    PythonExecutionRequest,
    PythonProbePlan,
    PythonAugmentedProbeToolGateway,
    ResolvedSandboxImage,
    SandboxUnavailableError,
    python_probe_plan_from_mapping,
)
from bayesprobe.initialization import BayesProbeInitializer, InitializeRunInput
from bayesprobe.evidence import SignalQualityAssessor
from bayesprobe.probe_executor import ProbeExecutionContext
from bayesprobe.schemas import EvidenceType, ExternalSignal, ProbeDesign, SignalKind


IMAGE_DIGEST = "sha256:" + "a" * 64


def test_python_probe_plan_accepts_python_mode_with_code():
    plan = python_probe_plan_from_mapping(
        {
            "mode": "python",
            "purpose": "Compute the discriminating value.",
            "target_hypotheses": ["B", "C"],
            "expected_observation": "The value matches one choice.",
            "code": "print(2 + 2)",
        },
        allowed_hypothesis_ids={"A", "B", "C"},
    )

    assert plan == PythonProbePlan(
        mode="python",
        purpose="Compute the discriminating value.",
        target_hypotheses=("B", "C"),
        expected_observation="The value matches one choice.",
        code="print(2 + 2)",
    )


def test_python_probe_plan_accepts_reasoning_mode_without_code():
    plan = python_probe_plan_from_mapping(
        {
            "mode": "reasoning",
            "purpose": "The question is conceptual.",
            "target_hypotheses": ["A", "B"],
            "expected_observation": "A logical distinction between the choices.",
            "code": None,
        },
        allowed_hypothesis_ids={"A", "B"},
    )

    assert plan.mode == "reasoning"
    assert plan.code is None


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "mode": "python",
                "purpose": "Compute.",
                "target_hypotheses": ["A"],
                "expected_observation": "A value.",
                "code": "",
            },
            "python mode requires non-empty code",
        ),
        (
            {
                "mode": "reasoning",
                "purpose": "Reason.",
                "target_hypotheses": ["A"],
                "expected_observation": "A distinction.",
                "code": "print('not allowed')",
            },
            "reasoning mode must not contain code",
        ),
        (
            {
                "mode": "shell",
                "purpose": "Run shell.",
                "target_hypotheses": ["A"],
                "expected_observation": "Output.",
                "code": "echo unsafe",
            },
            "mode must be python or reasoning",
        ),
        (
            {
                "mode": "python",
                "purpose": "Compute.",
                "target_hypotheses": ["Z"],
                "expected_observation": "A value.",
                "code": "print(1)",
            },
            "unknown target hypothesis",
        ),
    ],
)
def test_python_probe_plan_rejects_invalid_or_unsafe_shapes(payload, message):
    with pytest.raises(ValueError, match=message):
        python_probe_plan_from_mapping(
            payload,
            allowed_hypothesis_ids={"A", "B"},
        )


def test_docker_command_enforces_all_sandbox_controls_without_host_mounts():
    sandbox = DockerPythonSandbox(
        DockerPythonSandboxConfig(image="bayesprobe-hle-python:v0.1")
    )

    command = sandbox.docker_command(
        ResolvedSandboxImage(
            requested_reference="bayesprobe-hle-python:v0.1",
            digest=IMAGE_DIGEST,
        ),
        container_name="bp-execution-1",
    )

    assert command[:3] == ["docker", "run", "--rm"]
    assert "--network=none" in command
    assert "--read-only" in command
    assert "--user=65532:65532" in command
    assert "--cap-drop=ALL" in command
    assert "--security-opt=no-new-privileges" in command
    assert "--pids-limit=64" in command
    assert "--memory=1g" in command
    assert "--cpus=1" in command
    assert "--tmpfs=/tmp:rw,nosuid,nodev,size=64m" in command
    assert "--env=PYTHONHASHSEED=0" in command
    assert "--env=OMP_NUM_THREADS=1" in command
    assert "--env=OPENBLAS_NUM_THREADS=1" in command
    assert "--env=MKL_NUM_THREADS=1" in command
    assert "--env=NUMEXPR_NUM_THREADS=1" in command
    assert "--interactive" in command
    assert command[-4:] == [IMAGE_DIGEST, "python", "-s", "-"]
    assert "--mount" not in command
    assert "-v" not in command
    assert not any(argument.startswith("--volume") for argument in command)


def test_preflight_resolves_immutable_image_id():
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=IMAGE_DIGEST + "\n", stderr="")

    sandbox = DockerPythonSandbox(
        DockerPythonSandboxConfig(image="bayesprobe-hle-python:v0.1"),
        run_command=fake_run,
    )

    resolved = sandbox.preflight()

    assert resolved.digest == IMAGE_DIGEST
    assert calls == [
        [
            "docker",
            "image",
            "inspect",
            "bayesprobe-hle-python:v0.1",
            "--format={{.Id}}",
        ]
    ]


def test_preflight_fails_closed_when_docker_is_unavailable():
    def missing_docker(command, **kwargs):
        raise FileNotFoundError("docker")

    sandbox = DockerPythonSandbox(
        DockerPythonSandboxConfig(image="bayesprobe-hle-python:v0.1"),
        run_command=missing_docker,
    )

    with pytest.raises(SandboxUnavailableError, match="Docker is unavailable"):
        sandbox.preflight()


def test_preflight_rejects_mutable_or_unresolved_image_output():
    def unresolved(command, **kwargs):
        return SimpleNamespace(returncode=0, stdout="bayesprobe:latest\n", stderr="")

    sandbox = DockerPythonSandbox(
        DockerPythonSandboxConfig(image="bayesprobe:latest"),
        run_command=unresolved,
    )

    with pytest.raises(SandboxUnavailableError, match="immutable sha256 digest"):
        sandbox.preflight()


def test_execution_request_and_record_capture_immutable_audit_fields():
    request = PythonExecutionRequest(
        execution_id="exec_1",
        run_id="run_1",
        cycle_id="cycle_1",
        probe_id="probe_1",
        code="print(4)",
        image=ResolvedSandboxImage(
            requested_reference="bayesprobe-hle-python:v0.1",
            digest=IMAGE_DIGEST,
        ),
        repair_attempt_index=0,
    )
    record = PythonExecutionRecord(
        execution_id="exec_1",
        run_id="run_1",
        cycle_id="cycle_1",
        probe_id="probe_1",
        code="print(4)",
        code_sha256="7" * 64,
        image_digest=IMAGE_DIGEST,
        started_at="2026-07-11T00:00:00Z",
        completed_at="2026-07-11T00:00:01Z",
        wall_seconds=1.0,
        exit_code=0,
        stdout="4\n",
        stderr="",
        output_truncated=False,
        timed_out=False,
        policy_violation=False,
        repair_attempt_index=0,
    )

    assert request.code == record.code
    assert request.image.digest == record.image_digest
    assert record.success is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"timeout_seconds": 0}, "timeout_seconds must be positive"),
        ({"max_output_bytes": 0}, "max_output_bytes must be positive"),
        ({"pids_limit": 0}, "pids_limit must be positive"),
    ],
)
def test_sandbox_config_rejects_invalid_resource_limits(kwargs, message):
    with pytest.raises(ValueError, match=message):
        DockerPythonSandboxConfig(**kwargs)


class SequenceModelGateway:
    adapter_kind = "sequence"

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.requests = []

    def complete_structured(self, request):
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeSandbox:
    def __init__(self, records):
        self.records = list(records)
        self.requests = []
        self.image = ResolvedSandboxImage(
            requested_reference="sandbox:v0.1",
            digest=IMAGE_DIGEST,
        )

    def preflight(self):
        return self.image

    def execute(self, request):
        self.requests.append(request)
        return self.records.pop(0)


class RecordingExecutionObserver:
    def __init__(self):
        self.records = []

    def observe(self, record):
        self.records.append(record)


def execution_record(
    *,
    exit_code=0,
    stdout="4\n",
    stderr="",
    timed_out=False,
    policy_violation=False,
    repair_attempt_index=0,
):
    return PythonExecutionRecord(
        execution_id=f"exec_{repair_attempt_index}",
        run_id="run_1",
        cycle_id="cycle_1",
        probe_id="probe_1",
        code="print(4)",
        code_sha256="7" * 64,
        image_digest=IMAGE_DIGEST,
        started_at="2026-07-11T00:00:00Z",
        completed_at="2026-07-11T00:00:01Z",
        wall_seconds=1.0,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        output_truncated=False,
        timed_out=timed_out,
        policy_violation=policy_violation,
        repair_attempt_index=repair_attempt_index,
    )


def probe_context():
    initialized = BayesProbeInitializer().initialize(
        InitializeRunInput(
            run_id="run_1",
            problem="What is 2 + 2? Answer Choices: A. 3 B. 4 C. 5",
        )
    )
    probe = ProbeDesign(
        id="probe_1",
        cycle_id="cycle_1",
        target_hypotheses=["A", "B", "C"],
        inquiry_goal="Compute the exact sum.",
        method="calculation",
    )
    context = ProbeExecutionContext(
        run_id="run_1",
        cycle_id="cycle_1",
        belief_state=initialized.belief_state,
        metadata={
            "problem": initialized.run.problem,
            "initial_context": "Use exact arithmetic.",
            "experiment_id": "experiment_1",
            "arm": "bayesprobe_python",
            "sample_id": "sample_pseudonym",
        },
    )
    return probe, context


def python_plan(code="print(2 + 2)"):
    return {
        "mode": "python",
        "purpose": "Compute the exact sum.",
        "target_hypotheses": ["A", "B", "C"],
        "expected_observation": "The output equals one answer choice.",
        "code": code,
    }


def test_python_augmented_gateway_converts_successful_execution_to_active_signal():
    model = SequenceModelGateway([python_plan()])
    sandbox = FakeSandbox([execution_record()])
    observer = RecordingExecutionObserver()
    gateway = PythonAugmentedProbeToolGateway(
        model,
        sandbox,
        execution_observer=observer,
    )
    probe, context = probe_context()

    signals = gateway.execute_probe(probe=probe, context=context)

    assert len(signals) == 1
    signal = signals[0]
    assert signal.signal_kind == SignalKind.ACTIVE
    assert signal.source_type == "python_sandbox"
    assert signal.source == IMAGE_DIGEST
    assert signal.generated_by_probe == "probe_1"
    assert signal.initial_target_hypotheses == ["A", "B", "C"]
    assert "stdout:\n4" in signal.raw_content
    assert "purpose: Compute the exact sum." in signal.raw_content
    assert len(sandbox.requests) == 1
    assert observer.records == [execution_record()]
    assert model.requests[0].task == "plan_python_probe"
    assert "gold" not in str(model.requests[0].input).lower()


def test_reasoning_mode_uses_model_signal_without_starting_sandbox():
    model = SequenceModelGateway(
        [
            {
                "mode": "reasoning",
                "purpose": "Use a conceptual argument.",
                "target_hypotheses": ["A", "B", "C"],
                "expected_observation": "One option follows logically.",
                "code": None,
            },
            {"raw_content": "A conceptual argument supports answer choice B."},
        ]
    )
    sandbox = FakeSandbox([])
    gateway = PythonAugmentedProbeToolGateway(model, sandbox)
    probe, context = probe_context()

    signal = gateway.execute_probe(probe=probe, context=context)[0]

    assert signal.source_type == "model_probe_gateway"
    assert signal.raw_content == "A conceptual argument supports answer choice B."
    assert sandbox.requests == []
    assert [request.task for request in model.requests] == [
        "plan_python_probe",
        "execute_probe",
    ]


def test_reasoning_mode_provider_failure_becomes_unverified_signal():
    model = SequenceModelGateway(
        [
            {
                "mode": "reasoning",
                "purpose": "Use a conceptual argument.",
                "target_hypotheses": ["A", "B", "C"],
                "expected_observation": "One option follows logically.",
                "code": None,
            },
            RuntimeError("provider unavailable"),
        ]
    )
    gateway = PythonAugmentedProbeToolGateway(model, FakeSandbox([]))
    probe, context = probe_context()

    signal = gateway.execute_probe(probe=probe, context=context)[0]

    assert "unverified" in signal.raw_content.lower()
    assert "reasoning probe failed" in signal.raw_content.lower()


def test_runtime_failure_gets_one_code_repair_and_second_execution():
    model = SequenceModelGateway([python_plan("bad()"), {"code": "print(4)"}])
    sandbox = FakeSandbox(
        [
            execution_record(exit_code=1, stderr="NameError: bad", stdout=""),
            execution_record(repair_attempt_index=1),
        ]
    )
    observer = RecordingExecutionObserver()
    gateway = PythonAugmentedProbeToolGateway(
        model,
        sandbox,
        execution_observer=observer,
    )
    probe, context = probe_context()

    signal = gateway.execute_probe(probe=probe, context=context)[0]

    assert signal.source_type == "python_sandbox"
    assert len(sandbox.requests) == 2
    assert sandbox.requests[1].code == "print(4)"
    assert sandbox.requests[1].repair_attempt_index == 1
    assert [request.task for request in model.requests] == [
        "plan_python_probe",
        "repair_python_probe_code",
    ]
    assert len(observer.records) == 2


def test_code_repair_provider_failure_becomes_unverified_signal():
    model = SequenceModelGateway(
        [python_plan("bad()"), RuntimeError("provider unavailable")]
    )
    sandbox = FakeSandbox(
        [execution_record(exit_code=1, stderr="NameError: bad", stdout="")]
    )
    gateway = PythonAugmentedProbeToolGateway(model, sandbox)
    probe, context = probe_context()

    signal = gateway.execute_probe(probe=probe, context=context)[0]

    assert "unverified" in signal.raw_content.lower()
    assert "repair failed" in signal.raw_content.lower()


def test_timeout_returns_unverified_failure_signal_without_code_repair():
    model = SequenceModelGateway([python_plan()])
    sandbox = FakeSandbox(
        [execution_record(exit_code=-9, stdout="", timed_out=True)]
    )
    gateway = PythonAugmentedProbeToolGateway(model, sandbox)
    probe, context = probe_context()

    signal = gateway.execute_probe(probe=probe, context=context)[0]

    assert "unverified" in signal.raw_content.lower()
    assert "timed out" in signal.raw_content.lower()
    assert len(model.requests) == 1
    assert len(sandbox.requests) == 1


def test_invalid_plan_gets_one_plan_repair():
    invalid = python_plan(code="")
    model = SequenceModelGateway([invalid, python_plan()])
    sandbox = FakeSandbox([execution_record()])
    gateway = PythonAugmentedProbeToolGateway(model, sandbox)
    probe, context = probe_context()

    signal = gateway.execute_probe(probe=probe, context=context)[0]

    assert signal.source_type == "python_sandbox"
    assert [request.task for request in model.requests] == [
        "plan_python_probe",
        "repair_python_probe_plan",
    ]


def test_python_signal_quality_is_verifiable_but_not_independent():
    quality = SignalQualityAssessor().assess(
        signal=ExternalSignal(
            id="S_python",
            cycle_id="cycle_1",
            signal_kind=SignalKind.ACTIVE,
            source_type="python_sandbox",
            source=IMAGE_DIGEST,
            raw_content="stdout: 4",
            generated_by_probe="probe_1",
        ),
        event_type=EvidenceType.SUPPORTING,
    )

    assert quality.verifiability == 0.9
    assert quality.independence == 0.35
    assert quality.reliability == 0.75
