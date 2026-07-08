from pathlib import Path

import pytest

from bayesprobe.core import BayesProbeCore
from bayesprobe.initialization import BayesProbeInitializer, InitializeRunInput
from bayesprobe.ledger import JsonlLedgerStore
from bayesprobe.probe_planner import ProbePlanner, ProbePlanningConfig
from bayesprobe.schemas import (
    BeliefState,
    CycleRecord,
    CycleSignalShape,
    ExternalSignal,
    Hypothesis,
    ProbeCandidate,
    ProbeDesign,
    SignalKind,
)


def make_belief_state(h1_posterior: float = 0.4, h2_posterior: float = 0.6) -> BeliefState:
    return BeliefState(
        belief_state_id="bs_plan_1",
        run_id="run_plan",
        cycle_id="cycle_0",
        hypotheses=[
            Hypothesis(
                id="H1",
                statement="The claim is supported.",
                scope="planning fixture",
                prior=0.5,
                posterior=h1_posterior,
                rivals=["H2"],
                falsifiers=["Reliable counterevidence weakens H1."],
                predictions=["Support should be independently observable."],
            ),
            Hypothesis(
                id="H2",
                statement="The claim is refuted.",
                scope="planning fixture",
                prior=0.5,
                posterior=h2_posterior,
                rivals=["H1"],
                falsifiers=["Reliable support weakens H2."],
                predictions=["Counterevidence should be independently observable."],
            ),
        ],
        uncertainty_summary="The leading hypothesis still needs a direct challenge.",
    )


def make_candidate(
    candidate_id: str,
    target_hypotheses: list[str],
    *,
    expected_information_gain: float = 0.5,
    decision_relevance: float = 0.5,
    cost_estimate: float = 0.5,
) -> ProbeCandidate:
    support_condition = {
        hypothesis_id: f"Independent support appears for {hypothesis_id}."
        for hypothesis_id in target_hypotheses
    }
    weaken_condition = {
        hypothesis_id: f"Independent counterevidence appears for {hypothesis_id}."
        for hypothesis_id in target_hypotheses
    }
    return ProbeCandidate(
        candidate_id=candidate_id,
        source="manual",
        candidate_probe=ProbeDesign(
            id=f"P_{candidate_id}",
            cycle_id="cycle_0",
            target_hypotheses=target_hypotheses,
            inquiry_goal=f"Probe {candidate_id}.",
            method="source_tracing",
            support_condition=support_condition,
            weaken_condition=weaken_condition,
            expected_information_gain=expected_information_gain,
            decision_relevance=decision_relevance,
            cost_estimate=cost_estimate,
        ),
    )


def test_planner_selects_top_scoring_candidates_and_freezes_cycle():
    high = make_candidate(
        "c_high",
        ["H1"],
        expected_information_gain=0.8,
        decision_relevance=0.8,
        cost_estimate=0.2,
    )
    mid = make_candidate(
        "c_mid",
        ["H2"],
        expected_information_gain=0.6,
        decision_relevance=0.6,
        cost_estimate=0.3,
    )
    low = make_candidate(
        "c_low",
        ["H2"],
        expected_information_gain=0.2,
        decision_relevance=0.2,
        cost_estimate=0.2,
    )

    result = ProbePlanner().design_probe_set(
        run_id="run_plan",
        cycle_id="run_plan_cycle_1",
        belief_state=make_belief_state(h1_posterior=0.7, h2_posterior=0.3),
        candidates=[low, high, mid],
        config=ProbePlanningConfig(max_probes=2),
    )

    assert [candidate.candidate_id for candidate in result.selected_candidates] == ["c_high", "c_mid"]
    assert [rejected.candidate.candidate_id for rejected in result.rejected_candidates[0:1]] == ["c_low"]
    assert result.rejected_candidates[0].reason == "not_selected_budget_limit"
    assert result.probe_set.probe_set_id == "ps_run_plan_cycle_1"
    assert result.probe_set.cycle_id == "run_plan_cycle_1"
    assert result.probe_set.budget_allocated["max_probes"] == 2
    assert result.probe_set.budget_allocated["selected_count"] == 2
    assert result.probe_set.budget_allocated["candidate_count"] == 3
    assert "c_high" in result.probe_set.selection_reason
    assert all(probe.cycle_id == "run_plan_cycle_1" for probe in result.probe_set.probes)
    assert all("run_plan_cycle_1" in probe.id for probe in result.probe_set.probes)
    assert all(candidate.selected_in_cycle == "run_plan_cycle_1" for candidate in result.selected_candidates)

    assert high.selected_in_cycle is None
    assert high.candidate_probe.cycle_id == "cycle_0"
    assert low.selected_in_cycle is None
    assert low.candidate_probe.cycle_id == "cycle_0"


def test_planner_prioritizes_probe_that_attacks_top_hypothesis():
    high_non_top = make_candidate(
        "c_high_non_top",
        ["H1"],
        expected_information_gain=0.9,
        decision_relevance=0.9,
        cost_estimate=0.1,
    )
    lower_top = make_candidate(
        "c_lower_top",
        ["H2"],
        expected_information_gain=0.2,
        decision_relevance=0.2,
        cost_estimate=0.5,
    )

    result = ProbePlanner().design_probe_set(
        run_id="run_plan",
        cycle_id="run_plan_cycle_1",
        belief_state=make_belief_state(h1_posterior=0.4, h2_posterior=0.6),
        candidates=[high_non_top, lower_top],
        config=ProbePlanningConfig(max_probes=1),
    )

    assert [candidate.candidate_id for candidate in result.selected_candidates] == ["c_lower_top"]
    assert result.probe_set.probes[0].target_hypotheses == ["H2"]
    assert result.rejected_candidates[0].candidate.candidate_id == "c_high_non_top"
    assert result.rejected_candidates[0].reason == "not_selected_budget_limit"


def test_planner_rejects_invalid_candidates():
    valid = make_candidate("c_valid", ["H1"])
    no_targets = make_candidate("c_no_targets", [])
    unknown = make_candidate("c_unknown", ["HX"])

    result = ProbePlanner().design_probe_set(
        run_id="run_plan",
        cycle_id="run_plan_cycle_1",
        belief_state=make_belief_state(),
        candidates=[no_targets, unknown, valid],
        config=ProbePlanningConfig(max_probes=1),
    )

    rejection_reasons = {
        rejected.candidate.candidate_id: rejected.reason
        for rejected in result.rejected_candidates
    }
    assert [candidate.candidate_id for candidate in result.selected_candidates] == ["c_valid"]
    assert rejection_reasons["c_no_targets"] == "invalid_no_targets"
    assert rejection_reasons["c_unknown"] == "invalid_unknown_targets"


def test_planner_can_return_empty_probe_set_when_allowed():
    result = ProbePlanner().design_probe_set(
        run_id="run_plan",
        cycle_id="run_plan_cycle_1",
        belief_state=make_belief_state(),
        candidates=[],
        config=ProbePlanningConfig(allow_empty=True),
    )

    assert result.selected_candidates == []
    assert result.rejected_candidates == []
    assert result.probe_set.probes == []
    assert result.probe_set.may_be_empty is True
    assert result.probe_set.selection_reason == "No valid probe candidates; empty ProbeSet allowed."


def test_planner_rejects_empty_selection_when_not_allowed():
    with pytest.raises(ValueError):
        ProbePlanner().design_probe_set(
            run_id="run_plan",
            cycle_id="run_plan_cycle_1",
            belief_state=make_belief_state(),
            candidates=[],
        )


@pytest.mark.parametrize(
    "config_kwargs",
    [
        {"max_probes": 0},
        {"attack_top_hypothesis_bonus": 0},
        {"unresolved_uncertainty_bonus": 0},
    ],
)
def test_planner_rejects_invalid_config_values(config_kwargs):
    with pytest.raises(ValueError):
        ProbePlanningConfig(**config_kwargs)


def test_planner_writes_only_probe_set_to_ledger(tmp_path: Path):
    ledger = JsonlLedgerStore(tmp_path / "planner-ledger.jsonl")

    ProbePlanner(ledger=ledger).design_probe_set(
        run_id="run_plan",
        cycle_id="run_plan_cycle_1",
        belief_state=make_belief_state(),
        candidates=[make_candidate("c_valid", ["H2"])],
        config=ProbePlanningConfig(max_probes=1),
    )

    record_types = [record["record_type"] for record in ledger.read_all()]
    assert record_types == ["probe_set"]
    assert "external_signal" not in record_types
    assert "evidence_event" not in record_types
    assert "belief_update" not in record_types
    assert "answer_projection" not in record_types


def test_initializer_probe_candidates_can_be_planned_and_consumed_by_core():
    initialization = BayesProbeInitializer().initialize(
        InitializeRunInput(
            run_id="run_integrated",
            problem="Should the probe planner feed the core cycle cleanly?",
        )
    )
    cycle = CycleRecord(
        cycle_id="run_integrated_cycle_1",
        run_id="run_integrated",
        cycle_index=1,
        signal_shape=CycleSignalShape.ACTIVE_ONLY,
    )
    planning = ProbePlanner().design_probe_set(
        run_id=initialization.run.run_id,
        cycle_id=cycle.cycle_id,
        belief_state=initialization.belief_state,
        candidates=initialization.probe_candidates,
        config=ProbePlanningConfig(max_probes=1),
    )
    selected_probe = planning.probe_set.probes[0]

    result = BayesProbeCore().integrate_cycle(
        cycle=cycle,
        belief_state=initialization.belief_state,
        probe_set=planning.probe_set,
        signals=[
            ExternalSignal(
                id="S_planned_probe",
                cycle_id="pending",
                signal_kind=SignalKind.ACTIVE,
                source_type="benchmark_stream",
                source="fixture",
                raw_content="SUPPORTS: Planned probe returned a supporting signal.",
                generated_by_probe=selected_probe.id,
            )
        ],
    )

    assert result.cycle.cycle_id == "run_integrated_cycle_1"
    assert result.evidence_events[0].target_hypotheses == selected_probe.target_hypotheses
    assert result.belief_updates
