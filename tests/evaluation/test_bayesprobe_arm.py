import pytest
from pathlib import Path

from bayesprobe.evaluation.arms import BayesProbePythonArm
from bayesprobe.evaluation.contracts import EvaluationCase
from bayesprobe.ledger import JsonlLedgerStore


class CapabilityGateway:
    adapter_kind = "capability_fixture"

    def __init__(self):
        self.requests = []

    def complete_structured(self, request):
        self.requests.append(request)
        if request.task == "plan_python_probe":
            targets = request.input["probe"]["target_hypotheses"]
            return {
                "mode": "reasoning",
                "purpose": "Use exact arithmetic reasoning.",
                "target_hypotheses": targets,
                "expected_observation": "The reasoning identifies the exact sum.",
                "code": None,
            }
        if request.task == "execute_probe":
            return {
                "raw_content": "SUPPORTS B: Exact arithmetic gives 2 + 2 = 4."
            }
        if request.task == "judge_evidence":
            targets = request.input["target_hypotheses"]
            return {
                "evidence_type": "supporting",
                "likelihoods": {
                    target: (
                        "moderately_confirming"
                        if target == "B"
                        else "moderately_disconfirming"
                    )
                    for target in targets
                },
                "unresolved_likelihood": None,
                "frame_fit": "explained_by_named",
                "unexplained_observation": None,
                "interpretation": "The exact result supports answer B.",
                "quality_overrides": {},
            }
        raise AssertionError(f"unexpected model task: {request.task}")


class ForbiddenSandbox:
    def preflight(self):
        raise AssertionError("reasoning mode must not start Docker")

    def execute(self, request):
        raise AssertionError("reasoning mode must not execute Python")


def make_case():
    return EvaluationCase(
        sample_id="synthetic_1",
        question="What is 2 + 2?\n\nAnswer Choices:\nA. 3\nB. 4\nC. 5",
        choices={"A": "3", "B": "4", "C": "5"},
    )


def test_bayesprobe_arm_runs_provider_backed_belief_revision_end_to_end():
    model = CapabilityGateway()
    run_results = []
    arm = BayesProbePythonArm(
        model,
        ForbiddenSandbox(),
        invocation_metadata={"experiment_id": "experiment_1"},
        run_result_observer=run_results.append,
    )

    result = arm.run_case(make_case())

    assert result.state == "completed"
    assert result.answer_label == "B"
    assert set(result.probabilities) == {"A", "B", "C"}
    assert sum(result.probabilities.values()) == pytest.approx(1.0)
    assert result.probabilities["B"] == max(result.probabilities.values())
    run_result = run_results[0]
    assert [hypothesis.id for hypothesis in run_result.initial_belief_state.hypotheses] == [
        "A",
        "B",
        "C",
    ]
    assert [
        hypothesis.prior for hypothesis in run_result.initial_belief_state.hypotheses
    ] == pytest.approx([1 / 3, 1 / 3, 1 / 3], abs=1e-4)
    assert run_result.final_answer_projection.current_best_hypothesis == "B"
    assert all(
        signal.source_type == "model_probe_gateway"
        for cycle in run_result.cycle_results
        for signal in cycle.signals
    )
    assert any(
        event.discard_reason is None
        for cycle in run_result.cycle_results
        for event in cycle.evidence_events
    )


def test_bayesprobe_arm_never_falls_back_to_deterministic_probe_gateway():
    model = CapabilityGateway()
    arm = BayesProbePythonArm(model, ForbiddenSandbox())

    result = arm.run_case(make_case())

    tasks = [request.task for request in model.requests]
    assert result.state == "completed"
    assert "plan_python_probe" in tasks
    assert "execute_probe" in tasks
    assert "judge_evidence" in tasks
    assert all("deterministic" not in str(request.input).lower() for request in model.requests)


def test_bayesprobe_arm_freezes_autonomous_policy_and_process_metrics():
    arm = BayesProbePythonArm(CapabilityGateway(), ForbiddenSandbox())

    result = arm.run_case(make_case())

    assert arm.run_config.max_cycles == 4
    assert arm.run_config.max_probes_per_cycle == 2
    assert arm.run_config.stop_on_no_probes is True
    assert arm.run_config.confidence_threshold is None
    assert arm.run_config.posterior_delta_threshold is None
    assert result.process_metrics["cycles"] == 4
    assert 1 <= result.process_metrics["probes"] <= 8
    assert result.process_metrics["active_signals"] == result.process_metrics["probes"]
    assert result.process_metrics["stop_reason"] == "max_cycles"
    assert result.process_metrics["python_plans"] == result.process_metrics["probes"]
    assert result.process_metrics["reasoning_plans"] == result.process_metrics["probes"]
    assert result.process_metrics["python_executions"] == 0
    assert isinstance(result.process_metrics["top_answer_reversals"], int)
    assert isinstance(result.process_metrics["final_answer_first_top_cycle"], int)
    assert result.process_metrics["new_evidence_roots"] >= 1
    assert result.process_metrics["revised_evidence_roots"] >= 0
    assert result.process_metrics["retracted_evidence_roots"] >= 0
    assert result.process_metrics["unchanged_evidence_roots"] >= 0
    assert 0 <= result.process_metrics["falsification_cycles"] <= 4
    assert result.process_metrics["max_absolute_contribution_delta"] > 0.0
    assert result.process_metrics["epistemic_stagnation"] is False


def test_bayesprobe_arm_adds_case_context_to_every_provider_request_without_gold():
    model = CapabilityGateway()
    arm = BayesProbePythonArm(
        model,
        ForbiddenSandbox(),
        invocation_metadata={"experiment_id": "experiment_1"},
    )

    arm.run_case(make_case())

    assert model.requests
    for request in model.requests:
        assert request.metadata["experiment_id"] == "experiment_1"
        assert request.metadata["arm"] == "bayesprobe_python"
        assert request.metadata["sample_id"] == "synthetic_1"
        assert "gold" not in str(request.input).lower()


def test_bayesprobe_arm_uses_case_scoped_ledger_factory(tmp_path: Path):
    ledger_path = tmp_path / "case" / "ledger.jsonl"
    arm = BayesProbePythonArm(
        CapabilityGateway(),
        ForbiddenSandbox(),
        ledger_factory=lambda case: JsonlLedgerStore(ledger_path),
    )

    result = arm.run_case(make_case())

    assert result.state == "completed"
    record_types = {
        record["record_type"] for record in JsonlLedgerStore(ledger_path).read_all()
    }
    assert {"run", "belief_state", "probe_execution", "evidence_event"}.issubset(
        record_types
    )
