import copy
import json
import unicodedata
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, fields, is_dataclass
from types import SimpleNamespace

import pytest

import bayesprobe.evaluation.python_probe as python_probe_module
from bayesprobe.evidence import EvidenceIntegrationGate
from bayesprobe.evidence_memory import SignalProvenanceNormalizer
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
from bayesprobe.model_gateway import ModelInvocationTrace
from bayesprobe.probe_executor import (
    ProbeExecutionBrief,
    build_probe_execution_brief,
)
from bayesprobe.schemas import (
    BeliefState,
    CycleRecord,
    CycleSignalShape,
    EpistemicOrigin,
    EvidenceType,
    ExternalSignal,
    FramingMethod,
    ProbeDesign,
    ProbeSet,
    SignalKind,
)
from bayesprobe.task_framing import migrate_legacy_belief_state


IMAGE_DIGEST = "sha256:" + "a" * 64
_MIGRATION_MARKERS = (
    "belief_state_v0.1_to_v0.2",
    "task_frame_v0.1_to_v0.2",
)
_NONLEGACY_FRAMING_METHODS = tuple(
    method
    for method in FramingMethod
    if method != FramingMethod.LEGACY_MIGRATION
)
_INVALID_MIGRATION_ENVELOPES = (
    "tag_only",
    "forged_recognized_marker",
    "transferred_receipt",
    "v01_belief_state",
    "v01_task_frame",
    "missing_trace",
    "fake_trace",
    "missing_frame_state",
    "missing_evidence_memory",
    "incoherent_frame_state",
)
_SECRET_MODEL_IDENTITIES = (
    "Authorization: Bearer provider-secret-value-123",
    (
        "\uff21\uff55\uff54\uff48\uff4f\uff52\uff49\uff5a\uff41\uff54"
        "\uff49\uff4f\uff4e\uff1a \uff22\uff45\uff41\uff52\uff45\uff52 "
        "provider-secret-value-123"
    ),
)
_NFKC_SENSITIVE_NAME = "\uff41\uff50\uff49\uff3f\uff4b\uff45\uff59"
_OPAQUE_CODE_PAIRS = (
    pytest.param("    print('value')", "print('value')", id="leading-indentation"),
    pytest.param("print('value')\n", "print('value')", id="trailing-newline"),
    pytest.param(
        "print('first')\nprint('second')",
        "print('first'); print('second')",
        id="line-boundaries",
    ),
    pytest.param("print('\u212a')", "print('K')", id="nfkc-compatible"),
)
_FORBIDDEN_PYTHON_BELIEF_KEYS = {
    "ad_hoc_penalty",
    "applied_ad_hoc_penalty",
    "applied_complexity_penalty",
    "belief_state",
    "complexity_penalty",
    "correlation_credit",
    "current_best_hypothesis",
    "effective_update_weight",
    "evidence_credit",
    "gap",
    "initial_prior",
    "posterior",
    "posterior_summary",
    "prior",
    "rank",
    "ranking",
    "score",
    "scores",
    "top_hypothesis",
    "uncertainty_summary",
    "unresolved_alternative_mass",
}
POLICY_SNAPSHOT = {
    "runtime": "docker",
    "image_digest": IMAGE_DIGEST,
    "user": "65532:65532",
    "resources": {"cpus": 1.0, "memory": "1g", "pids_limit": 64},
    "limits": {"timeout_seconds": 30.0, "max_output_bytes": 64 * 1024},
    "network": {"mode": "none"},
    "filesystem": {
        "read_only_root": True,
        "host_mounts": [],
        "tmpfs": [
            {
                "path": "/tmp",
                "options": ["rw", "nosuid", "nodev"],
                "size": "64m",
            }
        ],
    },
    "security": {"cap_drop": ["ALL"], "no_new_privileges": True},
    "environment": {
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    },
    "interpreter": {"argv": ["python", "-s", "-"], "stdin": "interactive"},
}


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


def test_sandbox_policy_snapshot_captures_every_material_execution_control():
    sandbox = DockerPythonSandbox(
        DockerPythonSandboxConfig(image="bayesprobe-hle-python:v0.1")
    )

    snapshot = sandbox.policy_snapshot(image_digest=IMAGE_DIGEST)

    assert snapshot == POLICY_SNAPSHOT


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
        policy_snapshot=deepcopy(POLICY_SNAPSHOT),
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
        self.preflight_calls = 0
        self.image = ResolvedSandboxImage(
            requested_reference="sandbox:v0.1",
            digest=IMAGE_DIGEST,
        )

    def preflight(self):
        self.preflight_calls += 1
        return self.image

    def execute(self, request):
        self.requests.append(request)
        return self.records.pop(0)


class EchoCodeSandbox(FakeSandbox):
    def __init__(self, *, fail_first: bool = False):
        super().__init__([])
        self.fail_first = fail_first

    def execute(self, request):
        self.requests.append(request)
        failed = self.fail_first and len(self.requests) == 1
        return execution_record(
            execution_id=request.execution_id,
            cycle_id=request.cycle_id,
            probe_id=request.probe_id,
            code=request.code,
            exit_code=1 if failed else 0,
            stdout="" if failed else "value\n",
            stderr="RuntimeError: repair requested" if failed else "",
            repair_attempt_index=request.repair_attempt_index,
        )


class RecordingExecutionObserver:
    def __init__(self):
        self.records = []

    def observe(self, record):
        self.records.append(record)


def execution_record(
    *,
    execution_id=None,
    cycle_id="cycle_1",
    probe_id="probe_1",
    code="print(4)",
    image_digest=IMAGE_DIGEST,
    exit_code=0,
    stdout="4\n",
    stderr="",
    timed_out=False,
    policy_violation=False,
    repair_attempt_index=0,
    policy_snapshot=None,
    copy_policy_snapshot=True,
):
    source_policy_snapshot = policy_snapshot or POLICY_SNAPSHOT
    effective_policy_snapshot = (
        deepcopy(source_policy_snapshot)
        if copy_policy_snapshot
        else source_policy_snapshot
    )
    if policy_snapshot is None:
        effective_policy_snapshot["image_digest"] = image_digest
    return PythonExecutionRecord(
        execution_id=execution_id or f"exec_{repair_attempt_index}",
        run_id="run_1",
        cycle_id=cycle_id,
        probe_id=probe_id,
        code=code,
        code_sha256="7" * 64,
        image_digest=image_digest,
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
        policy_snapshot=effective_policy_snapshot,
    )


def legacy_execution_record():
    return PythonExecutionRecord(
        execution_id="exec_legacy",
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


def native_python_belief_state() -> BeliefState:
    return BayesProbeInitializer().initialize(
        InitializeRunInput(
            run_id="run_1",
            problem="What is 2 + 2? Answer Choices: A. 3 B. 4 C. 5",
        )
    ).belief_state


def probe_context(
    *,
    belief_state: BeliefState | None = None,
    run_id: str = "run_1",
    cycle_id: str = "cycle_1",
) -> tuple[ProbeDesign, ProbeExecutionBrief]:
    state = belief_state or native_python_belief_state()
    probe = ProbeDesign(
        id="probe_1",
        cycle_id=cycle_id,
        target_hypotheses=["A", "B", "C"],
        inquiry_goal="Compute the exact sum.",
        method="calculation",
    )
    context = build_probe_execution_brief(
        run_id=run_id,
        cycle_id=cycle_id,
        belief_state=state,
        problem="What is 2 + 2? Answer Choices: A. 3 B. 4 C. 5",
        task_context="Use exact arithmetic.",
        metadata={
            "experiment_id": "experiment_1",
            "arm": "bayesprobe_python",
            "sample_id": "sample_pseudonym",
        },
    )
    return probe, context


def recursive_keys(value) -> set[str]:
    if is_dataclass(value):
        return recursive_keys(
            {field.name: getattr(value, field.name) for field in fields(value)}
        )
    if hasattr(value, "model_dump"):
        return recursive_keys(value.model_dump(mode="python"))
    if isinstance(value, Mapping):
        return {
            *(str(key).casefold() for key in value),
            *(
                nested_key
                for item in value.values()
                for nested_key in recursive_keys(item)
            ),
        }
    if isinstance(value, list | tuple):
        return {
            nested_key
            for item in value
            for nested_key in recursive_keys(item)
        }
    return set()


def assert_blind_python_requests(requests, *, provider_version: str) -> None:
    assert {request.task for request in requests} == {
        "plan_python_probe",
        "repair_python_probe_plan",
        "repair_python_probe_code",
        "execute_probe",
    }
    for request in requests:
        assert request.prompt_version == provider_version
        assert request.schema_version == provider_version
        assert request.metadata["belief_context_policy"] == "blind_no_scores_v1"
        assert request.metadata["experiment_id"] == "experiment_1"
        assert request.metadata["arm"] == "bayesprobe_python"
        assert request.metadata["sample_id"] == "sample_pseudonym"
        assert _FORBIDDEN_PYTHON_BELIEF_KEYS.isdisjoint(
            recursive_keys(request.input)
        )
        assert _FORBIDDEN_PYTHON_BELIEF_KEYS.isdisjoint(
            recursive_keys(request.metadata)
        )
        trace = ModelInvocationTrace.from_request(
            request,
            adapter_kind="sequence",
        )
        assert trace.metadata["belief_context_policy"] == "blind_no_scores_v1"
        assert _FORBIDDEN_PYTHON_BELIEF_KEYS.isdisjoint(
            recursive_keys(trace.to_dict())
        )


def migrated_python_belief_state(native: BeliefState, marker: str) -> BeliefState:
    payload = native.model_dump(mode="python")
    payload.update(
        {
            "schema_version": "v0.1",
            "frame_state": None,
            "evidence_memory": None,
        }
    )
    if marker == "belief_state_v0.1_to_v0.2":
        payload["task_frame"] = None
    else:
        payload["task_frame"]["schema_version"] = "v0.1"
        payload["task_frame"]["framing_method"] = FramingMethod.EXPLICIT
        payload["task_frame"]["framing_trace"] = {"schema_version": "v0.1"}
    legacy_state = BeliefState.model_validate(payload)

    migrated = migrate_legacy_belief_state(legacy_state)

    assert legacy_state.schema_version == "v0.1"
    assert migrated.task_frame.framing_trace["migration"] == marker
    return migrated


def invalid_python_migration_envelope(
    native: BeliefState,
    kind: str,
) -> BeliefState:
    migrated = migrated_python_belief_state(
        native,
        "belief_state_v0.1_to_v0.2",
    )
    if kind == "tag_only":
        return native.model_copy(
            update={
                "task_frame": native.task_frame.model_copy(
                    update={"framing_method": FramingMethod.LEGACY_MIGRATION}
                )
            }
        )
    if kind == "forged_recognized_marker":
        return native.model_copy(
            update={
                "task_frame": native.task_frame.model_copy(
                    update={
                        "framing_method": FramingMethod.LEGACY_MIGRATION,
                        "framing_trace": {
                            **native.task_frame.framing_trace,
                            "migration": "belief_state_v0.1_to_v0.2",
                        },
                    }
                )
            }
        )
    if kind == "transferred_receipt":
        forged_native = native.model_copy(
            update={
                "task_frame": native.task_frame.model_copy(
                    update={
                        "framing_method": FramingMethod.LEGACY_MIGRATION,
                        "framing_trace": {
                            "migration": "belief_state_v0.1_to_v0.2"
                        },
                    }
                )
            }
        )
        return migrated.model_copy(
            update={
                field_name: getattr(forged_native, field_name)
                for field_name in BeliefState.model_fields
            }
        )
    if kind == "v01_belief_state":
        return migrated.model_copy(update={"schema_version": "v0.1"})
    if kind == "v01_task_frame":
        return migrated.model_copy(
            update={
                "task_frame": migrated.task_frame.model_copy(
                    update={"schema_version": "v0.1"}
                )
            }
        )
    if kind == "missing_trace":
        return migrated.model_copy(
            update={
                "task_frame": migrated.task_frame.model_copy(
                    update={"framing_trace": {}}
                )
            }
        )
    if kind == "fake_trace":
        return migrated.model_copy(
            update={
                "task_frame": migrated.task_frame.model_copy(
                    update={"framing_trace": {"migration": "caller_asserted"}}
                )
            }
        )
    if kind == "missing_frame_state":
        return migrated.model_copy(update={"frame_state": None})
    if kind == "missing_evidence_memory":
        return migrated.model_copy(update={"evidence_memory": None})
    if kind == "incoherent_frame_state":
        return migrated.model_copy(
            update={
                "frame_state": migrated.frame_state.model_copy(
                    update={"frame_id": "mismatched_frame"}
                )
            }
        )
    raise AssertionError(f"unknown invalid migration envelope: {kind}")


def python_plan(code="print(2 + 2)"):
    return {
        "mode": "python",
        "purpose": "Compute the exact sum.",
        "target_hypotheses": ["A", "B", "C"],
        "expected_observation": "The output equals one answer choice.",
        "code": code,
    }


def requests_for_all_python_model_routes(
    *,
    probe: ProbeDesign,
    context: ProbeExecutionBrief,
):
    python_model = SequenceModelGateway(
        [
            python_plan(code=""),
            python_plan(code="bad()"),
            {"code": "print(4)"},
        ]
    )
    PythonAugmentedProbeToolGateway(
        python_model,
        FakeSandbox(
            [
                execution_record(exit_code=1, stderr="NameError: bad", stdout=""),
                execution_record(repair_attempt_index=1),
            ]
        ),
    ).execute_probe(probe=probe, context=context)
    reasoning_model = SequenceModelGateway(
        [
            {
                "mode": "reasoning",
                "purpose": "Use a conceptual argument.",
                "target_hypotheses": ["A", "B", "C"],
                "expected_observation": "One option follows logically.",
                "code": None,
            },
            {"raw_content": "A conceptual argument supports B."},
        ]
    )
    PythonAugmentedProbeToolGateway(
        reasoning_model,
        FakeSandbox([]),
    ).execute_probe(probe=probe, context=context)
    return [*python_model.requests, *reasoning_model.requests]


def test_python_request_input_is_built_only_from_blind_execution_brief():
    probe, context = probe_context()

    request_input = python_probe_module._probe_request_input(
        probe=probe,
        context=context,
    )

    assert request_input["problem"] == context.problem
    assert request_input["initial_context"] == context.task_context
    assert {hypothesis["id"] for hypothesis in request_input["hypotheses"]} == {
        "A",
        "B",
        "C",
    }
    assert _FORBIDDEN_PYTHON_BELIEF_KEYS.isdisjoint(
        recursive_keys(request_input)
    )


def test_native_python_model_routes_are_recursively_blind_to_belief_scores():
    probe, context = probe_context()

    requests = requests_for_all_python_model_routes(
        probe=probe,
        context=context,
    )

    assert_blind_python_requests(requests, provider_version="v0.2")


def _execute_opaque_code(code: str, *, repaired: bool) -> tuple[str, str]:
    probe, context = probe_context()
    model = SequenceModelGateway(
        [
            python_plan("raise RuntimeError('repair')") if repaired else python_plan(code),
            *([{"code": code}] if repaired else []),
        ]
    )
    sandbox = EchoCodeSandbox(fail_first=repaired)

    signal = PythonAugmentedProbeToolGateway(model, sandbox).execute_probe(
        probe=probe,
        context=context,
    )[0]

    assert signal.provenance is not None
    return sandbox.requests[-1].code, signal.provenance.derivation_root_id


@pytest.mark.parametrize(("first_code", "second_code"), _OPAQUE_CODE_PAIRS)
def test_python_plan_path_preserves_byte_exact_code_and_distinct_roots(
    first_code,
    second_code,
):
    first_executed, first_root = _execute_opaque_code(first_code, repaired=False)
    second_executed, second_root = _execute_opaque_code(second_code, repaired=False)

    assert first_executed == first_code
    assert second_executed == second_code
    assert first_root != second_root


@pytest.mark.parametrize(("first_code", "second_code"), _OPAQUE_CODE_PAIRS)
def test_python_repair_path_preserves_byte_exact_code_and_distinct_roots(
    first_code,
    second_code,
):
    first_executed, first_root = _execute_opaque_code(first_code, repaired=True)
    second_executed, second_root = _execute_opaque_code(second_code, repaired=True)

    assert first_executed == first_code
    assert second_executed == second_code
    assert first_root != second_root


@pytest.mark.parametrize(
    "route",
    ["plan", "plan_repair", "reasoning", "code_repair"],
)
@pytest.mark.parametrize("model_identity", _SECRET_MODEL_IDENTITIES)
def test_python_gateway_rejects_secret_identity_before_every_route(
    route,
    model_identity,
):
    if route == "plan":
        outcomes = [python_plan()]
        records = [execution_record()]
    elif route == "plan_repair":
        outcomes = [python_plan(""), python_plan()]
        records = [execution_record()]
    elif route == "reasoning":
        outcomes = [
            {
                "mode": "reasoning",
                "purpose": "Use a conceptual argument.",
                "target_hypotheses": ["A", "B", "C"],
                "expected_observation": "One option follows logically.",
                "code": None,
            },
            {"raw_content": "This must not be requested."},
        ]
        records = []
    else:
        outcomes = [python_plan("bad()"), {"code": "print(4)"}]
        records = [
            execution_record(exit_code=1, stdout="", stderr="NameError"),
            execution_record(repair_attempt_index=1),
        ]
    model = SequenceModelGateway(outcomes)
    model.model_identity = model_identity
    sandbox = FakeSandbox(records)
    gateway = PythonAugmentedProbeToolGateway(model, sandbox)
    state = native_python_belief_state()
    probe, context = probe_context(belief_state=state)
    prior_state = state.model_dump(mode="json")

    with pytest.raises(ValueError, match="model gateway identity") as exc_info:
        gateway.execute_probe(probe=probe, context=context)

    error_text = str(exc_info.value)
    assert model_identity not in error_text
    assert unicodedata.normalize("NFKC", model_identity) not in error_text
    assert model.requests == []
    assert sandbox.preflight_calls == 0
    assert sandbox.requests == []
    assert set(gateway.process_metrics.values()) == {0}
    assert state.model_dump(mode="json") == prior_state


@pytest.mark.parametrize(
    "route",
    ["plan", "plan_repair", "reasoning", "code_repair"],
)
def test_python_gateway_rejects_sensitive_session_before_first_provider_call(
    route,
):
    if route == "plan":
        outcomes = [python_plan()]
        records = [execution_record()]
    elif route == "plan_repair":
        outcomes = [python_plan(""), python_plan()]
        records = [execution_record()]
    elif route == "reasoning":
        outcomes = [
            {
                "mode": "reasoning",
                "purpose": "Use a conceptual argument.",
                "target_hypotheses": ["A", "B", "C"],
                "expected_observation": "One option follows logically.",
                "code": None,
            },
            {"raw_content": "This must not be requested."},
        ]
        records = []
    else:
        outcomes = [python_plan("bad()"), {"code": "print(4)"}]
        records = [
            execution_record(exit_code=1, stdout="", stderr="NameError"),
            execution_record(repair_attempt_index=1),
        ]
    model = SequenceModelGateway(outcomes)
    sandbox = FakeSandbox(records)
    gateway = PythonAugmentedProbeToolGateway(model, sandbox)
    state = native_python_belief_state()
    probe, context = probe_context(
        belief_state=state,
        run_id=_NFKC_SENSITIVE_NAME,
    )
    prior_state = state.model_dump(mode="json")

    with pytest.raises(ValueError, match="model provenance") as exc_info:
        gateway.execute_probe(probe=probe, context=context)

    error_text = str(exc_info.value)
    assert _NFKC_SENSITIVE_NAME not in error_text
    assert "api_key" not in error_text
    assert model.requests == []
    assert sandbox.preflight_calls == 0
    assert sandbox.requests == []
    assert set(gateway.process_metrics.values()) == {0}
    assert state.model_dump(mode="json") == prior_state


def test_python_gateway_rejects_sensitive_adapter_before_first_provider_call():
    model = SequenceModelGateway([python_plan()])
    model.model_identity = "safe-model"
    model.adapter_kind = _NFKC_SENSITIVE_NAME
    sandbox = FakeSandbox([execution_record()])
    gateway = PythonAugmentedProbeToolGateway(model, sandbox)
    state = native_python_belief_state()
    probe, context = probe_context(belief_state=state)
    prior_state = state.model_dump(mode="json")

    with pytest.raises(ValueError, match="model signal source") as exc_info:
        gateway.execute_probe(probe=probe, context=context)

    error_text = str(exc_info.value)
    assert _NFKC_SENSITIVE_NAME not in error_text
    assert "api_key" not in error_text
    assert model.requests == []
    assert sandbox.preflight_calls == 0
    assert sandbox.requests == []
    assert set(gateway.process_metrics.values()) == {0}
    assert state.model_dump(mode="json") == prior_state


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
    assert model.requests[0].prompt_version == "v0.2"
    assert model.requests[0].schema_version == "v0.2"
    assert "gold" not in str(model.requests[0].input).lower()


def test_python_execution_record_legacy_constructor_defaults_policy_snapshot():
    record = legacy_execution_record()

    assert dict(record.policy_snapshot) == {}
    assert record.success is True


def test_legacy_execution_record_cannot_produce_trusted_evidence(
    monkeypatch,
):
    hash_calls = []
    real_derive = python_probe_module.derive_deterministic_computation_root

    def recording_derive(**kwargs):
        hash_calls.append(kwargs)
        return real_derive(**kwargs)

    monkeypatch.setattr(
        python_probe_module,
        "derive_deterministic_computation_root",
        recording_derive,
    )
    gateway = PythonAugmentedProbeToolGateway(
        SequenceModelGateway([python_plan()]),
        FakeSandbox([legacy_execution_record()]),
    )
    probe, context = probe_context()

    signal = gateway.execute_probe(probe=probe, context=context)[0]

    assert signal.provenance is None
    assert "unverified" in signal.raw_content.lower()
    assert hash_calls == []


def _python_signal_for_record(record, *, observer=None):
    gateway = PythonAugmentedProbeToolGateway(
        SequenceModelGateway([python_plan()]),
        FakeSandbox([record]),
        execution_observer=observer,
    )
    probe, context = probe_context()
    return gateway.execute_probe(probe=probe, context=context)[0]


def test_execution_policy_is_immutable_to_observer_mutation():
    mutation_errors = []

    class MutatingObserver:
        def observe(self, record):
            try:
                record.policy_snapshot["network"]["mode"] = "host"
            except TypeError as error:
                mutation_errors.append(error)
            try:
                record.policy_snapshot["resources"]["cpus"] = 99.0
            except TypeError as error:
                mutation_errors.append(error)
            try:
                record.policy_snapshot["interpreter"]["argv"][0] = "pypy"
            except TypeError as error:
                mutation_errors.append(error)

    baseline = _python_signal_for_record(execution_record())
    observed = _python_signal_for_record(
        execution_record(),
        observer=MutatingObserver(),
    )

    assert len(mutation_errors) == 3
    assert observed.provenance is not None
    assert observed.provenance.derivation_root_id == (
        baseline.provenance.derivation_root_id
    )


def test_execution_policy_defensively_copies_original_nested_input():
    original_policy = deepcopy(POLICY_SNAPSHOT)
    record = execution_record(
        policy_snapshot=original_policy,
        copy_policy_snapshot=False,
    )
    original_policy["network"]["mode"] = "host"
    original_policy["resources"]["cpus"] = 99.0
    original_policy["interpreter"]["argv"][0] = "pypy"

    signal = _python_signal_for_record(record)
    baseline = _python_signal_for_record(execution_record())

    assert record.policy_snapshot["network"]["mode"] == "none"
    assert record.policy_snapshot["resources"]["cpus"] == 1.0
    assert record.policy_snapshot["interpreter"]["argv"][0] == "python"
    assert signal.provenance is not None
    assert signal.provenance.derivation_root_id == (
        baseline.provenance.derivation_root_id
    )


def test_shallow_copied_execution_record_keeps_policy_immutable():
    record = execution_record()

    copied = copy.copy(record)

    with pytest.raises(TypeError):
        copied.policy_snapshot["network"]["mode"] = "host"
    assert copied.policy_snapshot["network"]["mode"] == "none"
    assert record.policy_snapshot["network"]["mode"] == "none"


def test_deep_copied_execution_record_keeps_policy_immutable():
    record = execution_record()

    copied = copy.deepcopy(record)

    with pytest.raises(TypeError):
        copied.policy_snapshot["network"]["mode"] = "host"
    assert copied.policy_snapshot["network"]["mode"] == "none"
    assert record.policy_snapshot["network"]["mode"] == "none"


def test_deep_copied_policy_snapshot_is_independent_mutable_serialization():
    record = execution_record()

    serialized = copy.deepcopy(record.policy_snapshot)
    serialized["network"]["mode"] = "host"
    serialized["interpreter"]["argv"][0] = "pypy"

    assert type(serialized) is dict
    assert type(serialized["interpreter"]["argv"]) is list
    assert record.policy_snapshot["network"]["mode"] == "none"
    assert record.policy_snapshot["interpreter"]["argv"][0] == "python"


def test_execution_record_asdict_is_independent_json_compatible_serialization():
    record = execution_record()

    serialized = asdict(record)
    encoded = json.dumps(serialized, sort_keys=True)
    serialized["policy_snapshot"]["resources"]["cpus"] = 99.0
    serialized["policy_snapshot"]["filesystem"]["tmpfs"][0]["options"].append(
        "exec"
    )

    assert type(serialized["policy_snapshot"]) is dict
    assert type(serialized["policy_snapshot"]["filesystem"]["tmpfs"]) is list
    assert '"policy_snapshot"' in encoded
    assert record.policy_snapshot["resources"]["cpus"] == 1.0
    assert record.policy_snapshot["filesystem"]["tmpfs"][0]["options"] == (
        "rw",
        "nosuid",
        "nodev",
    )


def test_copied_execution_records_keep_trusted_deterministic_root():
    record = execution_record()

    signals = [
        _python_signal_for_record(candidate)
        for candidate in (record, copy.copy(record), copy.deepcopy(record))
    ]

    assert all(signal.provenance is not None for signal in signals)
    assert len(
        {
            signal.provenance.derivation_root_id
            for signal in signals
        }
    ) == 1


@pytest.mark.parametrize("copy_record", [copy.copy, copy.deepcopy])
def test_copied_legacy_execution_record_remains_unverified(copy_record):
    copied = copy_record(legacy_execution_record())

    signal = _python_signal_for_record(copied)

    assert signal.provenance is None
    assert "unverified" in signal.raw_content.lower()


def test_repeated_python_computation_reuses_root_and_spends_no_fresh_credit():
    initial_state = native_python_belief_state()
    first_probe, first_context = probe_context(belief_state=initial_state)
    second_probe = first_probe.model_copy(
        update={"id": "probe_2", "cycle_id": "cycle_2"}
    )
    _, second_context = probe_context(
        belief_state=initial_state,
        cycle_id="cycle_2",
    )

    def execute(
        *,
        plan_payload,
        record,
        probe=first_probe,
        context=first_context,
        image_digest=IMAGE_DIGEST,
    ):
        sandbox = FakeSandbox([record])
        sandbox.image = ResolvedSandboxImage(
            requested_reference="sandbox:v0.1",
            digest=image_digest,
        )
        return PythonAugmentedProbeToolGateway(
            SequenceModelGateway([plan_payload]),
            sandbox,
        ).execute_probe(probe=probe, context=context)[0]

    first_signal = execute(
        plan_payload=python_plan(),
        record=execution_record(execution_id="exec_cycle_1"),
    )
    second_signal = execute(
        plan_payload={
            **python_plan(),
            "target_hypotheses": ["C", "B", "A"],
        },
        record=execution_record(
            execution_id="exec_cycle_2",
            cycle_id="cycle_2",
            probe_id="probe_2",
        ),
        probe=second_probe,
        context=second_context,
    )
    changed_code_signal = execute(
        plan_payload=python_plan("print(2 + 3)"),
        record=execution_record(code="print(2 + 3)", stdout="5\n"),
    )
    changed_plan_signal = execute(
        plan_payload={**python_plan(), "purpose": "Compute a different quantity."},
        record=execution_record(),
    )
    changed_image = "sha256:" + "b" * 64
    changed_image_signal = execute(
        plan_payload=python_plan(),
        record=execution_record(image_digest=changed_image),
        image_digest=changed_image,
    )

    assert first_signal.provenance.epistemic_origin == EpistemicOrigin.TOOL_RESULT
    assert first_signal.provenance.environment_state_id.startswith(
        "deterministic-computation:sha256:"
    )
    assert first_signal.provenance.derivation_root_id == (
        second_signal.provenance.derivation_root_id
    )
    assert first_signal.provenance.derivation_root_id != (
        changed_code_signal.provenance.derivation_root_id
    )
    assert first_signal.provenance.derivation_root_id != (
        changed_plan_signal.provenance.derivation_root_id
    )
    assert first_signal.provenance.derivation_root_id != (
        changed_image_signal.provenance.derivation_root_id
    )

    def probe_set(probe, cycle_id):
        return ProbeSet(
            probe_set_id=f"ps_{cycle_id}",
            cycle_id=cycle_id,
            probes=[probe],
            selection_reason="Python provenance regression.",
        )

    gate = EvidenceIntegrationGate()
    first = gate.integrate(
        cycle=CycleRecord(
            cycle_id="cycle_1",
            run_id="run_1",
            cycle_index=1,
            signal_shape=CycleSignalShape.ACTIVE_ONLY,
        ),
        belief_state=initial_state,
        probe_set=probe_set(first_probe, "cycle_1"),
        signals=[first_signal],
    )
    state = initial_state.model_copy(
        update={
            "evidence_memory": first.evidence_memory,
            "ledger_refs": {
                **initial_state.ledger_refs,
                "evidence_events": [event.id for event in first.evidence_events],
            },
        }
    )
    repeated = gate.integrate(
        cycle=CycleRecord(
            cycle_id="cycle_2",
            run_id="run_1",
            cycle_index=2,
            signal_shape=CycleSignalShape.ACTIVE_ONLY,
        ),
        belief_state=state,
        probe_set=probe_set(second_probe, "cycle_2"),
        signals=[second_signal],
    )

    event = repeated.evidence_events[0]
    assert event.correlation_status == "correlated_restatement"
    assert event.independence == 0.0
    assert event.effective_update_weight is None
    assert repeated.evidence_memory.correlation_credit == (
        first.evidence_memory.correlation_credit
    )


def test_python_computation_root_changes_for_every_material_policy_field_only():
    probe, context = probe_context()

    def root_for(policy_snapshot, *, execution_id="exec-policy"):
        image_digest = policy_snapshot["image_digest"]
        sandbox = FakeSandbox(
            [
                execution_record(
                    execution_id=execution_id,
                    image_digest=image_digest,
                    policy_snapshot=policy_snapshot,
                )
            ]
        )
        sandbox.image = ResolvedSandboxImage(
            requested_reference="sandbox:v0.1",
            digest=image_digest,
        )
        gateway = PythonAugmentedProbeToolGateway(
            SequenceModelGateway([python_plan()]),
            sandbox,
        )
        signal = gateway.execute_probe(probe=probe, context=context)[0]
        return signal.provenance.derivation_root_id

    base_root = root_for(POLICY_SNAPSHOT)
    assert base_root == root_for(POLICY_SNAPSHOT, execution_id="exec-policy-other")

    changes = [
        (("runtime",), "another-runtime"),
        (("image_digest",), "sha256:" + "b" * 64),
        (("user",), "1000:1000"),
        (("resources", "cpus"), 2.0),
        (("resources", "memory"), "2g"),
        (("resources", "pids_limit"), 32),
        (("limits", "timeout_seconds"), 15.0),
        (("limits", "max_output_bytes"), 4096),
        (("network", "mode"), "isolated-test-network"),
        (("filesystem", "read_only_root"), False),
        (("filesystem", "host_mounts"), ["/fixture"]),
        (("filesystem", "tmpfs", 0, "path"), "/scratch"),
        (("filesystem", "tmpfs", 0, "options"), ["rw", "nodev"]),
        (("filesystem", "tmpfs", 0, "size"), "32m"),
        (("security", "cap_drop"), ["NET_RAW"]),
        (("security", "no_new_privileges"), False),
        (("environment", "PYTHONHASHSEED"), "1"),
        (("environment", "OMP_NUM_THREADS"), "2"),
        (("environment", "OPENBLAS_NUM_THREADS"), "2"),
        (("environment", "MKL_NUM_THREADS"), "2"),
        (("environment", "NUMEXPR_NUM_THREADS"), "2"),
        (("interpreter", "argv"), ["python", "-I", "-"]),
        (("interpreter", "stdin"), "closed"),
    ]
    for path, changed_value in changes:
        changed_policy = deepcopy(POLICY_SNAPSHOT)
        target = changed_policy
        for component in path[:-1]:
            target = target[component]
        target[path[-1]] = changed_value

        assert root_for(changed_policy) != base_root, path


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
    assert signal.provenance.epistemic_origin == EpistemicOrigin.MODEL_REASONING
    assert signal.raw_content == "A conceptual argument supports answer choice B."
    assert sandbox.requests == []
    assert [request.task for request in model.requests] == [
        "plan_python_probe",
        "execute_probe",
    ]
    assert {request.prompt_version for request in model.requests} == {"v0.2"}
    assert {request.schema_version for request in model.requests} == {"v0.2"}


def test_reasoning_mode_pipe_identity_reuses_preflight_machine_provenance():
    model = SequenceModelGateway(
        [
            {
                "mode": "reasoning",
                "purpose": "Use a conceptual argument.",
                "target_hypotheses": ["A", "B", "C"],
                "expected_observation": "One option follows logically.",
                "code": None,
            },
            {"raw_content": "A pipe-bearing model supports answer choice B."},
        ]
    )
    model.model_identity = "provider|model"
    sandbox = FakeSandbox([])
    gateway = PythonAugmentedProbeToolGateway(model, sandbox)
    probe, context = probe_context()

    signal = gateway.execute_probe(probe=probe, context=context)[0]

    assert signal.provenance.epistemic_origin == EpistemicOrigin.MODEL_REASONING
    assert signal.provenance.provider_model_or_tool_identity == "provider|model"
    assert signal.provenance.source_identity.startswith("model-source:sha256:")
    assert signal.provenance.correlation_group.startswith("model:sha256:")
    assert "provider|model" not in signal.provenance.source_identity
    assert "provider|model" not in signal.provenance.correlation_group
    assert "|" not in signal.provenance.source_identity
    assert "|" not in signal.provenance.correlation_group
    normalized = SignalProvenanceNormalizer().normalize(
        signal,
        run_id=context.run_id,
    )
    assert normalized.provenance.source_identity == signal.provenance.source_identity
    assert normalized.provenance.correlation_group == (
        signal.provenance.correlation_group
    )
    assert sandbox.preflight_calls == 0
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
    assert {request.prompt_version for request in model.requests} == {"v0.2"}
    assert {request.schema_version for request in model.requests} == {"v0.2"}
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
    assert {request.prompt_version for request in model.requests} == {"v0.2"}
    assert {request.schema_version for request in model.requests} == {"v0.2"}


def test_plan_repair_uses_only_fixed_summary_for_raw_payload_and_validation_error():
    secret = "sk-" + "r" * 32
    invalid_payloads = [
        {
            **python_plan(),
            "posterior": 0.99,
            "winner": "B",
            "api_key": secret,
            "raw_provider_value": "raw-provider-value",
        },
        {
            **python_plan(),
            "target_hypotheses": [secret],
        },
    ]

    for invalid_payload in invalid_payloads:
        model = SequenceModelGateway([invalid_payload, python_plan()])
        gateway = PythonAugmentedProbeToolGateway(
            model,
            FakeSandbox([execution_record()]),
        )
        probe, context = probe_context()

        signal = gateway.execute_probe(probe=probe, context=context)[0]

        repair_request = model.requests[1]
        serialized_request = json.dumps(
            {
                "input": repair_request.input,
                "metadata": repair_request.metadata,
            },
            sort_keys=True,
        )
        assert signal.source_type == "python_sandbox"
        assert repair_request.task == "repair_python_probe_plan"
        assert repair_request.input["invalid_payload"] == {
            "status": "schema_invalid"
        }
        assert repair_request.input["validation_error"] == (
            "Python probe plan failed schema validation"
        )
        assert repair_request.metadata["belief_context_policy"] == (
            "blind_no_scores_v1"
        )
        for forbidden_text in (
            "posterior",
            "winner",
            "api_key",
            secret,
            "raw_provider_value",
            "raw-provider-value",
        ):
            assert forbidden_text not in serialized_request


@pytest.mark.parametrize("migration_marker", _MIGRATION_MARKERS)
def test_explicit_migration_uses_v01_for_every_python_model_route(
    migration_marker,
):
    native_state = native_python_belief_state()
    probe, _ = probe_context(belief_state=native_state)
    migrated_state = migrated_python_belief_state(
        native_state,
        migration_marker,
    )
    _, migrated_context = probe_context(
        belief_state=migrated_state,
    )
    requests = requests_for_all_python_model_routes(
        probe=probe,
        context=migrated_context,
    )

    assert migrated_state.task_frame.framing_trace["migration"] == migration_marker
    assert_blind_python_requests(requests, provider_version="v0.1")


@pytest.mark.parametrize("framing_method", _NONLEGACY_FRAMING_METHODS)
def test_python_gateway_rejects_migrated_marker_with_nonlegacy_method(
    framing_method,
):
    native_state = native_python_belief_state()
    state = migrated_python_belief_state(
        native_state,
        "belief_state_v0.1_to_v0.2",
    )
    state = state.model_copy(
        update={
            "task_frame": state.task_frame.model_copy(
                update={"framing_method": framing_method}
            )
        }
    )
    model = SequenceModelGateway([python_plan()])
    sandbox = FakeSandbox([execution_record()])
    gateway = PythonAugmentedProbeToolGateway(model, sandbox)
    prior_state = state.model_dump(mode="json")

    with pytest.raises(ValueError, match="invalid belief lifecycle"):
        probe_context(belief_state=state)

    assert model.requests == []
    assert sandbox.preflight_calls == 0
    assert sandbox.requests == []
    assert set(gateway.process_metrics.values()) == {0}
    assert state.model_dump(mode="json") == prior_state


@pytest.mark.parametrize("invalid_envelope", _INVALID_MIGRATION_ENVELOPES)
def test_invalid_python_migration_envelope_rejects_without_side_effects(
    invalid_envelope,
):
    native_state = native_python_belief_state()
    state = invalid_python_migration_envelope(
        native_state,
        invalid_envelope,
    )
    model = SequenceModelGateway([python_plan()])
    sandbox = FakeSandbox([execution_record()])
    gateway = PythonAugmentedProbeToolGateway(model, sandbox)
    prior_state = state.model_dump(mode="json")

    with pytest.raises(ValueError, match="invalid belief lifecycle"):
        probe_context(belief_state=state)

    assert model.requests == []
    assert sandbox.preflight_calls == 0
    assert sandbox.requests == []
    assert set(gateway.process_metrics.values()) == {0}
    assert state.model_dump(mode="json") == prior_state


def test_unmigrated_v01_python_gateway_rejects_before_model_or_sandbox():
    invalid_state = native_python_belief_state().model_copy(
        update={"schema_version": "v0.1"}
    )
    model = SequenceModelGateway([python_plan()])
    sandbox = FakeSandbox([execution_record()])
    gateway = PythonAugmentedProbeToolGateway(model, sandbox)

    with pytest.raises(ValueError, match="invalid belief lifecycle"):
        probe_context(belief_state=invalid_state)

    assert model.requests == []
    assert sandbox.preflight_calls == 0
    assert sandbox.requests == []
    assert set(gateway.process_metrics.values()) == {0}


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
