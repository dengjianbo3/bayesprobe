from pathlib import Path

import pytest

from bayesprobe.benchmark import (
    BenchmarkHarness,
    BenchmarkSample,
    BenchmarkSignal,
    BenchmarkSignalShape,
)
from bayesprobe.ledger import JsonlLedgerStore
from bayesprobe.model_gateway import ScriptedModelGateway


def passive_refutation_signal(signal_id: str = "S_passive_refute") -> BenchmarkSignal:
    return BenchmarkSignal(
        signal_id=signal_id,
        source_type="benchmark_stream",
        source="fixture",
        raw_content="REFUTES: Benchmark passage contradicts H1 and supports H2.",
        target_hypotheses=["H1", "H2"],
    )


def test_benchmark_harness_runs_active_only_sample():
    sample = BenchmarkSample(
        sample_id="active_support_1",
        question_or_claim="Does the autonomous active path support H1?",
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
    assert result.belief_update_count == 1


def test_benchmark_harness_runs_passive_only_sample():
    sample = BenchmarkSample(
        sample_id="passive_refute_1",
        question_or_claim="Does the passive signal refute H1?",
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


def test_benchmark_harness_runs_active_plus_passive_sample():
    sample = BenchmarkSample(
        sample_id="mixed_refute_1",
        question_or_claim="Can a mixed cycle integrate active and passive signals together?",
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
    assert result.belief_update_count == 3


def test_benchmark_harness_aggregates_suite_metrics():
    samples = [
        BenchmarkSample(
            sample_id="suite_active",
            question_or_claim="Does active-only aggregate?",
            signal_shape=BenchmarkSignalShape.ACTIVE_ONLY,
            gold_best_hypothesis="H1",
            gold_update_directions={"H1": "strengthened"},
        ),
        BenchmarkSample(
            sample_id="suite_passive",
            question_or_claim="Does passive-only aggregate?",
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
    assert "belief_state_projection" in record_types
    assert "benchmark_sample_result" in record_types


def test_benchmark_harness_passes_model_gateway_to_created_core(tmp_path: Path):
    ledger = JsonlLedgerStore(tmp_path / "gateway-ledger.jsonl")
    gateway = ScriptedModelGateway(
        responses={
            "judge_evidence": {
                "evidence_type": "boundary_condition",
                "likelihoods": {"H1": "weakly_disconfirming", "H2": "neutral"},
                "interpretation": "Harness configured scripted judgment.",
                "quality_overrides": {"reliability": 0.62},
            }
        }
    )
    harness = BenchmarkHarness(ledger=ledger, model_gateway=gateway)
    sample = BenchmarkSample(
        sample_id="gateway_passive",
        question_or_claim="Can benchmark configure model gateway?",
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
    assert gateway.requests[0].input["signal_id"] == "S_gateway_passive"


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
