from pathlib import Path

import pytest

import bayesprobe.benchmark as benchmark_module
from bayesprobe.benchmark import (
    BenchmarkHarness,
    BenchmarkSample,
    BenchmarkSignal,
    BenchmarkSignalShape,
    _update_direction_accuracy,
)
from bayesprobe.ledger import JsonlLedgerStore
from bayesprobe.model_gateway import EvidenceJudgmentRepairPolicy, ScriptedModelGateway
from bayesprobe.probe_executor import build_probe_execution_brief
from bayesprobe.schemas import BeliefUpdate, UpdateDirection
from bayesprobe.task_framing import HypothesisSeed


def passive_refutation_signal(signal_id: str = "S_passive_refute") -> BenchmarkSignal:
    return BenchmarkSignal(
        signal_id=signal_id,
        source_type="benchmark_stream",
        source="fixture",
        raw_content="REFUTES: Benchmark passage contradicts H1 and supports H2.",
        target_hypotheses=["H1", "H2"],
    )


def benchmark_hypothesis_seeds() -> list[HypothesisSeed]:
    return [
        HypothesisSeed(
            id="H1",
            statement="The benchmark's H1 condition holds.",
            prior=0.5,
            scope="Deterministic benchmark fixture.",
            falsifiers=["The benchmark emits a reliable H1 refutation."],
            predictions=["The benchmark emits a reliable H1 support cue."],
        ),
        HypothesisSeed(
            id="H2",
            statement="The benchmark's H2 condition holds instead.",
            prior=0.5,
            scope="Deterministic benchmark fixture.",
            falsifiers=["The benchmark emits a reliable H2 refutation."],
            predictions=["The benchmark emits a reliable H2 support cue."],
        ),
    ]


def test_update_direction_accuracy_scores_net_movement_not_transient_match():
    updates = [
        BeliefUpdate(
            update_id="U1",
            cycle_id="cycle_1",
            evidence_id="E1",
            hypothesis_id="H1",
            prior=0.5,
            posterior=0.6,
            direction=UpdateDirection.STRENGTHENED,
            reason="Transient support.",
        ),
        BeliefUpdate(
            update_id="U2",
            cycle_id="cycle_1",
            evidence_id="E2",
            hypothesis_id="H1",
            prior=0.6,
            posterior=0.4,
            direction=UpdateDirection.WEAKENED,
            reason="Final counterevidence.",
        ),
    ]

    assert _update_direction_accuracy(
        belief_updates=updates,
        gold_update_directions={"H1": "strengthened"},
    ) == 0.0


def test_benchmark_harness_runs_active_only_sample():
    sample = BenchmarkSample(
        sample_id="active_support_1",
        question_or_claim="Does the autonomous active path support H1?",
        hypothesis_seeds=benchmark_hypothesis_seeds(),
        signal_shape=BenchmarkSignalShape.ACTIVE_ONLY,
        gold_best_hypothesis="H1",
        gold_update_directions={"H1": "strengthened"},
    )

    result = BenchmarkHarness().run_sample(sample)

    assert result.sample_id == "active_support_1"
    assert result.signal_shape == BenchmarkSignalShape.ACTIVE_ONLY
    assert result.final_best_hypothesis == "H1"
    assert result.final_correct is True
    assert result.update_direction_accuracy == 1.0
    assert result.projection_kind == "answer_projection"
    assert result.cycle_count == 1
    assert result.active_signal_count == 1
    assert result.passive_signal_count == 0
    assert result.evidence_event_count == 1
    assert result.belief_update_count == 2


def test_benchmark_harness_runs_passive_only_sample():
    sample = BenchmarkSample(
        sample_id="passive_refute_1",
        question_or_claim="Does the passive signal refute H1?",
        hypothesis_seeds=benchmark_hypothesis_seeds(),
        signal_shape=BenchmarkSignalShape.PASSIVE_ONLY,
        gold_best_hypothesis="H2",
        passive_signals=[passive_refutation_signal()],
        gold_update_directions={"H1": "weakened", "H2": "strengthened"},
    )

    result = BenchmarkHarness().run_sample(sample)

    assert result.signal_shape == BenchmarkSignalShape.PASSIVE_ONLY
    assert result.final_best_hypothesis == "H2"
    assert result.final_correct is True
    assert result.update_direction_accuracy == 1.0
    assert result.projection_kind == "belief_state_projection"
    assert result.cycle_count == 1
    assert result.active_signal_count == 0
    assert result.passive_signal_count == 1
    assert result.evidence_event_count == 1
    assert result.belief_update_count == 2


def test_benchmark_harness_runs_active_plus_passive_sample(monkeypatch):
    builder_calls = []

    def recording_builder(**kwargs):
        builder_calls.append(kwargs)
        return build_probe_execution_brief(**kwargs)

    monkeypatch.setattr(
        benchmark_module,
        "build_probe_execution_brief",
        recording_builder,
    )
    sample = BenchmarkSample(
        sample_id="mixed_refute_1",
        question_or_claim="Can a mixed cycle integrate active and passive signals together?",
        initial_context="Use the supplied benchmark observations.",
        hypothesis_seeds=benchmark_hypothesis_seeds(),
        signal_shape=BenchmarkSignalShape.ACTIVE_PLUS_PASSIVE,
        gold_best_hypothesis="H2",
        passive_signals=[passive_refutation_signal()],
        gold_update_directions={"H2": "strengthened"},
    )

    result = BenchmarkHarness().run_sample(sample)

    assert result.signal_shape == BenchmarkSignalShape.ACTIVE_PLUS_PASSIVE
    assert result.final_best_hypothesis == "H2"
    assert result.final_correct is True
    assert result.update_direction_accuracy == 1.0
    assert result.projection_kind == "answer_projection"
    assert result.cycle_count == 1
    assert result.active_signal_count == 1
    assert result.passive_signal_count == 1
    assert result.evidence_event_count == 2
    assert result.belief_update_count == 4
    assert len(builder_calls) == 1
    builder_call = builder_calls[0]
    assert builder_call["run_id"] == result.run_id
    assert builder_call["cycle_id"].startswith(f"{result.run_id}_cycle_1")
    assert set(builder_call["belief_state"].hypotheses_by_id()) == {"H1", "H2"}
    assert builder_call["problem"] == sample.question_or_claim
    assert builder_call["task_context"] == sample.initial_context


def test_benchmark_harness_aggregates_suite_metrics():
    samples = [
        BenchmarkSample(
            sample_id="suite_active",
            question_or_claim="Does active-only aggregate?",
            hypothesis_seeds=benchmark_hypothesis_seeds(),
            signal_shape=BenchmarkSignalShape.ACTIVE_ONLY,
            gold_best_hypothesis="H1",
            gold_update_directions={"H1": "strengthened"},
        ),
        BenchmarkSample(
            sample_id="suite_passive",
            question_or_claim="Does passive-only aggregate?",
            hypothesis_seeds=benchmark_hypothesis_seeds(),
            signal_shape=BenchmarkSignalShape.PASSIVE_ONLY,
            gold_best_hypothesis="H2",
            passive_signals=[passive_refutation_signal("S_suite_passive")],
            gold_update_directions={"H1": "weakened", "H2": "strengthened"},
        ),
    ]

    result = BenchmarkHarness().run_suite(samples)

    assert result.sample_count == 2
    assert [sample_result.sample_id for sample_result in result.results] == [
        "suite_active",
        "suite_passive",
    ]
    assert result.final_accuracy == 1.0
    assert result.update_direction_accuracy == 1.0


@pytest.mark.parametrize(
    "sample_kwargs",
    [
        {"sample_id": "", "question_or_claim": "claim", "gold_best_hypothesis": "H1"},
        {"sample_id": "s", "question_or_claim": "", "gold_best_hypothesis": "H1"},
        {"sample_id": "s", "question_or_claim": "claim", "gold_best_hypothesis": ""},
        {
            "sample_id": "s",
            "question_or_claim": "claim",
            "gold_best_hypothesis": "H1",
            "signal_shape": BenchmarkSignalShape.PASSIVE_ONLY,
        },
        {
            "sample_id": "s",
            "question_or_claim": "claim",
            "gold_best_hypothesis": "H1",
            "signal_shape": BenchmarkSignalShape.ACTIVE_PLUS_PASSIVE,
        },
    ],
)
def test_benchmark_harness_rejects_invalid_samples(sample_kwargs):
    with pytest.raises(ValueError):
        BenchmarkSample(**sample_kwargs)


def test_benchmark_harness_preserves_ledger_records(tmp_path: Path):
    ledger = JsonlLedgerStore(tmp_path / "benchmark-ledger.jsonl")
    harness = BenchmarkHarness(ledger=ledger)
    sample = BenchmarkSample(
        sample_id="ledger_passive",
        question_or_claim="Does benchmark execution preserve the BayesProbe ledger?",
        hypothesis_seeds=benchmark_hypothesis_seeds(),
        signal_shape=BenchmarkSignalShape.PASSIVE_ONLY,
        gold_best_hypothesis="H2",
        passive_signals=[passive_refutation_signal("S_ledger_passive")],
    )

    result = harness.run_sample(sample)

    record_types = [record["record_type"] for record in ledger.read_all()]
    assert result.sample_id == "ledger_passive"
    assert "run" in record_types
    assert "cycle" in record_types
    assert "external_signal" in record_types
    assert "evidence_event" in record_types
    assert "belief_update" in record_types
    assert "epistemic_progress" in record_types
    assert "belief_state_projection" in record_types
    assert "benchmark_sample_result" in record_types


def test_benchmark_harness_passes_model_gateway_to_created_core(tmp_path: Path):
    ledger = JsonlLedgerStore(tmp_path / "gateway-ledger.jsonl")
    gateway = ScriptedModelGateway(
        responses={
            "judge_evidence": {
                "evidence_type": "boundary_condition",
                "likelihoods": {"H1": "weakly_disconfirming", "H2": "neutral"},
                "unresolved_likelihood": None,
                "frame_fit": "explained_by_named",
                "unexplained_observation": None,
                "interpretation": "Harness configured scripted judgment.",
                "quality_overrides": {"reliability": 0.62},
            }
        }
    )
    harness = BenchmarkHarness(ledger=ledger, model_gateway=gateway)
    sample = BenchmarkSample(
        sample_id="gateway_passive",
        question_or_claim="Can benchmark configure model gateway?",
        hypothesis_seeds=benchmark_hypothesis_seeds(),
        signal_shape=BenchmarkSignalShape.PASSIVE_ONLY,
        gold_best_hypothesis="H1",
        passive_signals=[
            BenchmarkSignal(
                signal_id="S_gateway_passive",
                source_type="user_feedback",
                source="user",
                raw_content="No keyword cue.",
                target_hypotheses=["H1", "H2"],
            )
        ],
    )

    harness.run_sample(sample)

    evidence_payloads = [
        record["payload"]
        for record in ledger.read_all("evidence_event")
    ]
    assert evidence_payloads[0]["evidence_type"] == "boundary_condition"
    assert evidence_payloads[0]["reliability"] == 0.62
    assert gateway.requests[0].input["signal"]["id"] == "S_gateway_passive"


def test_benchmark_harness_records_model_trace_in_evidence_ledger(tmp_path: Path):
    ledger = JsonlLedgerStore(tmp_path / "model-trace-ledger.jsonl")
    gateway = ScriptedModelGateway(
        responses={
            "judge_evidence": {
                "evidence_type": "supporting",
                "likelihoods": {
                    "H1": "moderately_confirming",
                    "H2": "moderately_disconfirming",
                },
                "unresolved_likelihood": None,
                "frame_fit": "explained_by_named",
                "unexplained_observation": None,
                "interpretation": "Harness trace judgment.",
                "quality_overrides": {},
            }
        }
    )
    harness = BenchmarkHarness(ledger=ledger, model_gateway=gateway)
    sample = BenchmarkSample(
        sample_id="model_trace_passive",
        question_or_claim="Can benchmark ledger preserve model trace?",
        hypothesis_seeds=benchmark_hypothesis_seeds(),
        signal_shape=BenchmarkSignalShape.PASSIVE_ONLY,
        gold_best_hypothesis="H1",
        passive_signals=[
            BenchmarkSignal(
                signal_id="S_model_trace_passive",
                source_type="user_feedback",
                source="user",
                raw_content="Model trace fixture.",
                target_hypotheses=["H1", "H2"],
            )
        ],
    )

    harness.run_sample(sample)

    evidence_payload = ledger.read_all("evidence_event")[0]["payload"]
    trace = evidence_payload["model_trace"]
    assert trace["task"] == "judge_evidence"
    assert trace["adapter_kind"] == "scripted"
    assert trace["prompt_id"] == "evidence_judgment"
    assert trace["prompt_version"] == "v0.2"
    assert trace["schema_name"] == "EvidenceJudgment"
    assert trace["schema_version"] == "v0.2"
    assert trace["metadata"]["judgment_route"] == "native_v0.2"


def test_benchmark_harness_passes_judgment_repair_policy_to_created_core(tmp_path: Path):
    ledger = JsonlLedgerStore(tmp_path / "repair-ledger.jsonl")
    gateway = ScriptedModelGateway(
        responses={
            "judge_evidence": {
                "evidence_type": "not_a_type",
                "likelihoods": {"H1": "neutral", "H2": "neutral"},
                "unresolved_likelihood": None,
                "frame_fit": "underdetermined",
                "unexplained_observation": None,
                "interpretation": "Invalid evidence type.",
                "quality_overrides": {},
            },
            "repair_evidence_judgment": {
                "evidence_type": "supporting",
                "likelihoods": {
                    "H1": "moderately_confirming",
                    "H2": "moderately_disconfirming",
                },
                "unresolved_likelihood": None,
                "frame_fit": "explained_by_named",
                "unexplained_observation": None,
                "interpretation": "Harness repaired judgment.",
                "quality_overrides": {},
            },
        }
    )
    harness = BenchmarkHarness(
        ledger=ledger,
        model_gateway=gateway,
        judgment_repair_policy=EvidenceJudgmentRepairPolicy(max_attempts=1),
    )
    sample = BenchmarkSample(
        sample_id="repair_passive",
        question_or_claim="Can benchmark configure repair policy?",
        hypothesis_seeds=benchmark_hypothesis_seeds(),
        signal_shape=BenchmarkSignalShape.PASSIVE_ONLY,
        gold_best_hypothesis="H1",
        passive_signals=[
            BenchmarkSignal(
                signal_id="S_repair_passive",
                source_type="user_feedback",
                source="user",
                raw_content="Malformed judgment fixture.",
                target_hypotheses=["H1", "H2"],
            )
        ],
    )

    result = harness.run_sample(sample)

    evidence_payloads = [
        record["payload"]
        for record in ledger.read_all("evidence_event")
    ]
    assert result.belief_update_count == 2
    assert evidence_payloads[0]["evidence_type"] == "supporting"
    assert evidence_payloads[0]["discard_reason"] is None
    assert [request.task for request in gateway.requests] == [
        "judge_evidence",
        "repair_evidence_judgment",
    ]


def test_benchmark_harness_records_schema_violation_without_belief_update(tmp_path: Path):
    ledger = JsonlLedgerStore(tmp_path / "schema-violation-ledger.jsonl")
    gateway = ScriptedModelGateway(
        responses={
            "judge_evidence": {
                "likelihoods": {"H1": "moderately_confirming", "H2": "moderately_disconfirming"},
                "interpretation": "Missing evidence type.",
            }
        }
    )
    harness = BenchmarkHarness(ledger=ledger, model_gateway=gateway)
    sample = BenchmarkSample(
        sample_id="schema_violation_passive",
        question_or_claim="Can benchmark replay schema violations?",
        hypothesis_seeds=benchmark_hypothesis_seeds(),
        signal_shape=BenchmarkSignalShape.PASSIVE_ONLY,
        gold_best_hypothesis="H1",
        passive_signals=[
            BenchmarkSignal(
                signal_id="S_schema_violation_passive",
                source_type="user_feedback",
                source="user",
                raw_content="Malformed judgment fixture.",
                target_hypotheses=["H1", "H2"],
            )
        ],
    )

    result = harness.run_sample(sample)

    evidence_payloads = [
        record["payload"]
        for record in ledger.read_all("evidence_event")
    ]
    assert result.evidence_event_count == 1
    assert result.belief_update_count == 0
    assert evidence_payloads[0]["discard_reason"].startswith("schema_violation:")
    assert evidence_payloads[0]["evidence_type"] == "neutral"
    assert ledger.read_all("belief_update") == []


def test_benchmark_harness_reports_belief_quality_metrics():
    sample = BenchmarkSample(
        sample_id="quality_active",
        question_or_claim="Does active-only quality metric work?",
        hypothesis_seeds=benchmark_hypothesis_seeds(),
        signal_shape=BenchmarkSignalShape.ACTIVE_ONLY,
        gold_best_hypothesis="H1",
        gold_update_directions={"H1": "strengthened"},
    )

    result = BenchmarkHarness().run_sample(sample)

    assert result.discarded_evidence_count == 0
    assert result.schema_violation_count == 0
    assert result.dominant_hypothesis_margin > 0
    assert result.belief_revision_efficiency == pytest.approx(
        result.dominant_hypothesis_margin / 2
    )


def test_benchmark_harness_counts_schema_violations_as_discarded_evidence():
    gateway = ScriptedModelGateway(responses={"judge_evidence": {"likelihoods": {}}})
    sample = BenchmarkSample(
        sample_id="quality_schema_violation",
        question_or_claim="Does schema violation quality metric work?",
        hypothesis_seeds=benchmark_hypothesis_seeds(),
        signal_shape=BenchmarkSignalShape.PASSIVE_ONLY,
        gold_best_hypothesis="H1",
        passive_signals=[
            BenchmarkSignal(
                signal_id="S_quality_schema",
                source_type="benchmark_stream",
                source="fixture",
                raw_content="No valid schema payload.",
                target_hypotheses=["H1", "H2"],
            )
        ],
    )

    result = BenchmarkHarness(model_gateway=gateway).run_sample(sample)

    assert result.discarded_evidence_count == 1
    assert result.schema_violation_count == 1
    assert result.belief_revision_efficiency == 0.0
