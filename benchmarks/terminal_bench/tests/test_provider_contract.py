from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any

import pytest

from bayesprobe import (
    AnswerContract,
    BeliefState,
    CapabilityDescriptor,
    CapabilityKind,
    EpistemicOrigin,
    FramedHypothesis,
    FramingMethod,
    HypothesisCompetition,
    HypothesisFrame,
    ModelProbeDesigner,
    ModelTaskFramer,
    ProbeDesignContext,
    ProbePurpose,
    TaskAdmissionDecision,
    TaskAdmissionStatus,
    TaskFrame,
    TaskFramingInput,
    TaskKind,
    StructuredModelRequest,
)
from bayesprobe_terminal_bench.artifacts import TrialArtifactStore
from bayesprobe_terminal_bench.config import BudgetExhausted, RunBudget
from bayesprobe_terminal_bench.provider_contract import (
    ProviderContractError,
    TerminalContractModelGateway,
)
from bayesprobe_terminal_bench.runner_factory import BudgetedModelGateway


@dataclass
class RecordingGateway:
    responses: list[Any]
    adapter_kind: str = "recording"
    model_identity: str = "recording-model"
    config: object = object()
    invocation_observer: object = object()

    def __post_init__(self) -> None:
        self.requests: list[StructuredModelRequest] = []

    def complete_structured(self, request: StructuredModelRequest) -> dict[str, Any]:
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _frame(*, hypothesis_type: str = "root_cause") -> dict[str, Any]:
    return {
        "task_kind": "design",
        "answer_relationship": "synthesis",
        "answer_contract": {
            "objective": "Diagnose the terminal task from observations.",
            "answer_value_type": "structured_text",
            "answer_format": "structured text with verification",
            "required_sections": ["result", "verification"],
            "decision_form": "environment_change",
            "permits_synthesis": True,
        },
        "competition": "independent",
        "coverage": "open",
        "hypotheses": [
            {
                "statement": "A cancellation path skips task cleanup.",
                "type": hypothesis_type,
                "scope": "The async task runner.",
                "falsifiers": ["Every cancelled task runs its cleanup handler."],
                "predictions": ["A cancellation test leaves a pending task."],
                "answer_value": None,
            },
            {
                "statement": "The concurrency limit is not enforced.",
                "type": hypothesis_type,
                "scope": "The async task runner.",
                "falsifiers": ["A concurrency test never exceeds the limit."],
                "predictions": ["A trace shows more active tasks than permitted."],
                "answer_value": None,
            },
        ],
        "coverage_statement": "The frame covers cleanup and concurrency behavior.",
        "coverage_limitation": "Other runtime interactions remain open alternatives.",
    }


def _frame_request() -> StructuredModelRequest:
    return StructuredModelRequest(
        task="frame_open_question",
        input={"question": "Repair the terminal task.", "task_context": "Use tests."},
        prompt_id="frame",
        prompt_version="v0.2",
        schema_name="OpenQuestionTaskFrame",
        schema_version="v0.2",
        metadata={"run_id": "run"},
    )


def _probe(*, targets: list[str] | None = None) -> dict[str, Any]:
    target_hypotheses = targets or ["H1", "H2"]
    return {
        "proposals": [
            {
                "purpose": "hypothesis_discrimination",
                "target_hypotheses": target_hypotheses,
                "inquiry_goal": "Run tests that distinguish cleanup from concurrency failures.",
                "expected_observation": "The failing test favors one diagnosis.",
                "support_condition": {
                    target: f"The observation supports {target}."
                    for target in target_hypotheses
                },
                "weaken_condition": {
                    target: f"The observation weakens {target}."
                    for target in target_hypotheses
                },
                "reframe_condition": None,
                "required_capability": "test_execution",
            }
        ]
    }


def _probe_request() -> StructuredModelRequest:
    return StructuredModelRequest(
        task="design_probes",
        input={
            "run_id": "run",
            "cycle_id": "cycle_0",
            "task_frame": {"coverage": "open"},
            "hypotheses": [{"id": "H1"}, {"id": "H2"}],
            "available_capabilities": [
                {"kind": "test_execution", "available": True},
                {"kind": "repository_read", "available": True},
            ],
        },
        prompt_id="probe",
        prompt_version="v0.2",
        schema_name="ProbeDesign",
        schema_version="v0.2",
        metadata={"run_id": "run", "cycle_id": "cycle_0"},
    )


def _attempts(path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_terminal_frame_contract_adds_policy_hashes_response_and_forwards_identity(tmp_path) -> None:
    delegate = RecordingGateway([_frame()])
    artifacts = TrialArtifactStore(tmp_path, restricted_values=())
    gateway = TerminalContractModelGateway(delegate, artifacts=artifacts)
    request = _frame_request()

    assert gateway.complete_structured(request) == _frame()
    assert "terminal_policy" not in request.input
    forwarded = delegate.requests[0]
    assert forwarded is not request
    assert forwarded.input["terminal_policy"]["stage"] == "terminal_task_frame"
    assert gateway.adapter_kind == delegate.adapter_kind
    assert gateway.model_identity == delegate.model_identity
    assert gateway.config is delegate.config
    assert gateway.invocation_observer is delegate.invocation_observer

    attempts = _attempts(tmp_path / "provider_contract.jsonl")
    assert attempts == [
        {
            "attempt_index": 0,
            "field_errors": [],
            "request_task": "frame_open_question",
            "required_keys_present": sorted(_frame()),
            "response_sha256": sha256(
                json.dumps(_frame(), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "stage": "terminal_task_frame",
            "validation": "valid",
        }
    ]


@pytest.mark.parametrize("policy", ["implementation_policy", "patch_choice"])
def test_terminal_frame_rejects_explicit_implementation_policies(tmp_path, policy: str) -> None:
    invalid = _frame(hypothesis_type=policy)
    delegate = RecordingGateway([invalid, invalid, invalid])
    gateway = TerminalContractModelGateway(
        delegate,
        artifacts=TrialArtifactStore(tmp_path, restricted_values=()),
    )

    with pytest.raises(ProviderContractError, match="terminal_task_frame") as raised:
        gateway.complete_structured(_frame_request())

    assert raised.value.attempts == 3
    assert [request.task for request in delegate.requests] == [
        "frame_open_question",
        "repair_task_frame",
        "repair_task_frame",
    ]
    attempts = _attempts(tmp_path / "provider_contract.jsonl")
    assert [item["attempt_index"] for item in attempts] == [0, 1, 2]
    assert all(item["validation"] == "invalid" for item in attempts)
    assert all("hypotheses.0.type:literal_error" in item["field_errors"] for item in attempts)


def test_probe_contract_rejects_unknown_targets_and_requires_exact_conditions(tmp_path) -> None:
    invalid = _probe(targets=["H1", "unknown"])
    invalid["proposals"][0]["support_condition"] = {"H1": "only one target"}
    delegate = RecordingGateway([invalid, invalid, invalid])
    gateway = TerminalContractModelGateway(
        delegate,
        artifacts=TrialArtifactStore(tmp_path, restricted_values=()),
    )

    with pytest.raises(ProviderContractError, match="terminal_probe_design"):
        gateway.complete_structured(_probe_request())

    attempts = _attempts(tmp_path / "provider_contract.jsonl")
    assert all(item["validation"] == "invalid" for item in attempts)
    assert all("proposals.0.target_hypotheses:value_error" in item["field_errors"] for item in attempts)


def test_probe_contract_exposes_exact_condition_key_rule_to_repairs(tmp_path) -> None:
    invalid = _probe()
    invalid["proposals"][0]["support_condition"] = {"H1": "partial support"}
    delegate = RecordingGateway([invalid, _probe()])
    gateway = TerminalContractModelGateway(
        delegate,
        artifacts=TrialArtifactStore(tmp_path, restricted_values=()),
    )

    assert gateway.complete_structured(_probe_request()) == _probe()

    expected = {
        "support_condition": {
            "keys": "exactly_target_hypotheses",
            "values": "non_empty_text",
        },
        "weaken_condition": {
            "keys": "exactly_target_hypotheses",
            "values": "non_empty_text",
        },
    }
    assert delegate.requests[0].input["terminal_policy"]["condition_maps"] == expected
    assert delegate.requests[1].input["terminal_policy"]["condition_maps"] == expected


def test_probe_contract_treats_missing_request_targets_as_contract_invalidity(tmp_path) -> None:
    request = _probe_request()
    request = StructuredModelRequest(
        task=request.task,
        input={**request.input, "hypotheses": None},
        prompt_id=request.prompt_id,
        prompt_version=request.prompt_version,
        schema_name=request.schema_name,
        schema_version=request.schema_version,
        metadata=request.metadata,
    )
    invalid = _probe(targets=["H1"])
    delegate = RecordingGateway([invalid, invalid, invalid])
    gateway = TerminalContractModelGateway(
        delegate,
        artifacts=TrialArtifactStore(tmp_path, restricted_values=()),
    )

    with pytest.raises(ProviderContractError, match="terminal_probe_design"):
        gateway.complete_structured(request)

    assert all(item["validation"] == "invalid" for item in _attempts(tmp_path / "provider_contract.jsonl"))


@pytest.mark.parametrize("payload", [{}, {"task_kind": "design"}, ["not", "a", "mapping"]])
def test_contract_records_safe_diagnostics_for_missing_or_non_mapping_payloads(tmp_path, payload: Any) -> None:
    delegate = RecordingGateway([payload, payload, payload])
    gateway = TerminalContractModelGateway(
        delegate,
        artifacts=TrialArtifactStore(tmp_path, restricted_values=()),
    )

    with pytest.raises(ProviderContractError):
        gateway.complete_structured(_frame_request())

    attempts = _attempts(tmp_path / "provider_contract.jsonl")
    if payload == {}:
        assert all(item["validation"] == "empty" for item in attempts)
        assert all(item["response_sha256"] is None for item in attempts)
    else:
        assert all(item["validation"] == "invalid" for item in attempts)
        assert all(item["response_sha256"] is not None for item in attempts)
        assert all(item["field_errors"] for item in attempts)
    assert all("input" not in json.dumps(item) for item in attempts)


@pytest.mark.parametrize("invalid_count", [1, 2])
def test_contract_accepts_first_or_second_repair(tmp_path, invalid_count: int) -> None:
    invalid = {"task_kind": "design"}
    delegate = RecordingGateway([invalid] * invalid_count + [_frame()])
    gateway = TerminalContractModelGateway(
        delegate,
        artifacts=TrialArtifactStore(tmp_path, restricted_values=()),
    )

    assert gateway.complete_structured(_frame_request()) == _frame()
    assert [request.task for request in delegate.requests] == [
        "frame_open_question",
        *("repair_task_frame" for _ in range(invalid_count)),
    ]
    attempts = _attempts(tmp_path / "provider_contract.jsonl")
    assert [item["validation"] for item in attempts] == ["invalid"] * invalid_count + ["valid"]


def test_contract_exhausts_after_three_consecutive_failures(tmp_path) -> None:
    delegate = RecordingGateway([None, {}, {"task_kind": "design"}])
    gateway = TerminalContractModelGateway(
        delegate,
        artifacts=TrialArtifactStore(tmp_path, restricted_values=()),
    )

    with pytest.raises(ProviderContractError, match="after 3 attempts"):
        gateway.complete_structured(_frame_request())

    assert [item["validation"] for item in _attempts(tmp_path / "provider_contract.jsonl")] == [
        "empty",
        "empty",
        "invalid",
    ]


def test_contract_repairs_provider_exception_without_persisting_exception_text(tmp_path) -> None:
    secret = "provider-secret"
    delegate = RecordingGateway([RuntimeError(f"provider failed: {secret}"), _frame()])
    gateway = TerminalContractModelGateway(
        delegate,
        artifacts=TrialArtifactStore(tmp_path, restricted_values=(secret,)),
    )

    assert gateway.complete_structured(_frame_request()) == _frame()
    attempts = _attempts(tmp_path / "provider_contract.jsonl")
    assert [item["validation"] for item in attempts] == ["provider_error", "valid"]
    artifact_text = (tmp_path / "provider_contract.jsonl").read_text(encoding="utf-8")
    assert secret not in artifact_text
    repair_text = json.dumps(delegate.requests[1].input)
    assert secret not in repair_text


def test_shared_budget_exhaustion_is_preserved_before_a_repair_call(tmp_path) -> None:
    delegate = RecordingGateway([{"task_kind": "design"}, _frame()])
    gateway = TerminalContractModelGateway(
        BudgetedModelGateway(delegate, RunBudget(max_model_calls=1)),
        artifacts=TrialArtifactStore(tmp_path, restricted_values=()),
    )

    with pytest.raises(BudgetExhausted, match="model call budget exhausted"):
        gateway.complete_structured(_frame_request())

    assert len(delegate.requests) == 1


def test_unrelated_requests_pass_through_without_contract_telemetry(tmp_path) -> None:
    request = StructuredModelRequest(task="judge_evidence", input={"evidence": "value"})
    delegate = RecordingGateway([{"accepted": True}])
    gateway = TerminalContractModelGateway(
        delegate,
        artifacts=TrialArtifactStore(tmp_path, restricted_values=()),
    )

    assert gateway.complete_structured(request) == {"accepted": True}
    assert delegate.requests == [request]
    assert not (tmp_path / "provider_contract.jsonl").exists()


def _admission() -> TaskAdmissionDecision:
    return TaskAdmissionDecision(
        attempt_id="admission",
        status=TaskAdmissionStatus.ADMITTED,
        epistemic_basis=["The task asks for an environment change."],
        proposed_task_kind=TaskKind.DESIGN,
        answer_contract_outline={
            "objective": "Diagnose the terminal task from observations.",
            "answer_value_type": "structured_text",
            "decision_form": "environment_change",
            "permits_synthesis": True,
            "required_sections": ["result", "verification"],
        },
        reason="The task is an admitted design task.",
    )


def _public_probe_context() -> ProbeDesignContext:
    frame = TaskFrame(
        task_frame_id="frame",
        task_kind=TaskKind.DESIGN,
        answer_relationship="synthesis",
        normalized_question="Repair the terminal task.",
        answer_contract=AnswerContract(
            objective="Diagnose the terminal task from observations.",
            answer_value_type="structured_text",
            answer_format="structured text with verification",
            required_sections=["result", "verification"],
            decision_form="environment_change",
            permits_synthesis=True,
        ),
        hypothesis_frame=HypothesisFrame(
            frame_id="hypotheses",
            competition=HypothesisCompetition.INDEPENDENT,
            coverage="open",
            hypotheses=[
                FramedHypothesis(
                    id="H1",
                    statement="A cancellation path skips task cleanup.",
                    type="claim",
                    scope="The async task runner.",
                    initial_prior=0.5,
                    falsifiers=["Every cancelled task runs cleanup."],
                    predictions=["A cancelled task remains pending."],
                ),
                FramedHypothesis(
                    id="H2",
                    statement="The concurrency limit is not enforced.",
                    type="claim",
                    scope="The async task runner.",
                    initial_prior=0.5,
                    falsifiers=["Concurrency stays within the limit."],
                    predictions=["Too many tasks run at once."],
                ),
            ],
            rival_sets={"H1": [], "H2": []},
            coverage_statement="The frame is open.",
        ),
        framing_method=FramingMethod.MODEL,
    )
    belief = BeliefState(
        belief_state_id="belief",
        run_id="run",
        cycle_id="cycle_1",
        cycle_index=1,
        hypotheses=[
            {
                "id": "H1",
                "statement": "A cancellation path skips task cleanup.",
                "scope": "The async task runner.",
                "prior": 0.5,
                "posterior": 0.5,
                "status": "active",
            },
            {
                "id": "H2",
                "statement": "The concurrency limit is not enforced.",
                "scope": "The async task runner.",
                "prior": 0.5,
                "posterior": 0.5,
                "status": "active",
            },
        ],
    )
    return ProbeDesignContext(
        run_id="run",
        cycle_id="cycle_1",
        task_frame=frame,
        belief_state=belief,
        available_capabilities=(
            CapabilityDescriptor(
                kind=CapabilityKind.TEST_EXECUTION,
                available=True,
                epistemic_origin=EpistemicOrigin.TOOL_RESULT,
                executor_adapter_id="terminal:v1",
            ),
        ),
    )


def test_adapter_rejects_policy_frame_that_public_framer_accepts(tmp_path) -> None:
    policy_frame = _frame(hypothesis_type="implementation_policy")
    public_framer = ModelTaskFramer(RecordingGateway([policy_frame]))
    accepted_frame = public_framer.frame(
        TaskFramingInput(
            run_id="run",
            question="Repair the terminal task.",
            task_context="Use tests.",
            admission_decision=_admission(),
        )
    )
    assert accepted_frame.hypothesis_frame.hypotheses[0].type == "implementation_policy"

    contract = TerminalContractModelGateway(
        RecordingGateway([policy_frame, policy_frame, policy_frame]),
        artifacts=TrialArtifactStore(tmp_path, restricted_values=()),
    )
    with pytest.raises(ProviderContractError):
        contract.complete_structured(_frame_request())


def test_adapter_rejects_partial_probe_map_that_public_designer_accepts(tmp_path) -> None:
    partial_probe = _probe()
    partial_probe["proposals"][0]["support_condition"] = {
        "H1": "partial support"
    }

    accepted_probes = ModelProbeDesigner(RecordingGateway([partial_probe])).propose(
        _public_probe_context()
    )
    assert len(accepted_probes.candidates) == 1

    delegate = RecordingGateway([partial_probe, partial_probe, partial_probe])
    contract = TerminalContractModelGateway(
        delegate,
        artifacts=TrialArtifactStore(tmp_path, restricted_values=()),
    )
    with pytest.raises(ProviderContractError, match="terminal_probe_design") as raised:
        contract.complete_structured(_probe_request())

    assert raised.value.attempts == 3
    assert [request.task for request in delegate.requests] == [
        "design_probes",
        "repair_probe_design",
        "repair_probe_design",
    ]
    attempts = _attempts(tmp_path / "provider_contract.jsonl")
    assert [attempt["validation"] for attempt in attempts] == [
        "invalid",
        "invalid",
        "invalid",
    ]
    assert all(
        "proposals.0.support_condition:value_error" in attempt["field_errors"]
        for attempt in attempts
    )
