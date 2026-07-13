from pathlib import Path

import pytest

from bayesprobe.core import BayesProbeCore
from bayesprobe.initialization import BayesProbeInitializer, HypothesisSeed, InitializeRunInput
from bayesprobe.ledger import JsonlLedgerStore
from bayesprobe.probe_planner import (
    ProbePlanner,
    ProbePlanningConfig,
    _is_top_falsification,
)
from bayesprobe.schemas import (
    BeliefState,
    CycleRecord,
    CycleSignalShape,
    ExternalSignal,
    Hypothesis,
    ProbeCandidate,
    ProbeDesign,
    ProbePurpose,
    SignalKind,
)


def make_belief_state(
    h1_posterior: float = 0.4,
    h2_posterior: float = 0.6,
    *,
    cycle_index: int = 0,
) -> BeliefState:
    return BeliefState(
        belief_state_id="bs_plan_1",
        run_id="run_plan",
        cycle_id="cycle_0",
        cycle_index=cycle_index,
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


def explicit_test_hypothesis_seeds() -> list[HypothesisSeed]:
    return [
        HypothesisSeed(id="H1", statement="The fixture's H1 condition holds.", prior=0.5, scope="Deterministic test fixture.", falsifiers=["The fixture emits a reliable H1 refutation."], predictions=["The fixture emits a reliable H1 support cue."]),
        HypothesisSeed(id="H2", statement="The fixture's H2 condition holds instead.", prior=0.5, scope="Deterministic test fixture.", falsifiers=["The fixture emits a reliable H2 refutation."], predictions=["The fixture emits a reliable H2 support cue."]),
    ]


def make_candidate(
    candidate_id: str,
    target_hypotheses: list[str],
    *,
    expected_information_gain: float = 0.5,
    decision_relevance: float = 0.5,
    cost_estimate: float = 0.5,
    purpose: ProbePurpose = ProbePurpose.HYPOTHESIS_DISCRIMINATION,
    weaken_condition: dict[str, str] | None = None,
) -> ProbeCandidate:
    support_condition = {
        hypothesis_id: f"Independent support appears for {hypothesis_id}."
        for hypothesis_id in target_hypotheses
    }
    if weaken_condition is None:
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
            purpose=purpose,
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


def test_planner_reserves_real_top_falsifier_after_initial_cycle():
    high_top_targeting_discriminator = make_candidate(
        "c_high_top_targeting_discriminator",
        ["H2"],
        expected_information_gain=1.0,
        decision_relevance=1.0,
        cost_estimate=0.01,
    )
    lower_top_falsifier = make_candidate(
        "c_lower_top_falsifier",
        ["H2"],
        purpose=ProbePurpose.HYPOTHESIS_FALSIFICATION,
        expected_information_gain=0.2,
        decision_relevance=0.2,
        cost_estimate=1.0,
    )

    result = ProbePlanner().design_probe_set(
        run_id="run_plan",
        cycle_id="run_plan_cycle_1",
        belief_state=make_belief_state(
            h1_posterior=0.4,
            h2_posterior=0.6,
            cycle_index=1,
        ),
        candidates=[high_top_targeting_discriminator, lower_top_falsifier],
        config=ProbePlanningConfig(max_probes=1),
    )

    assert [candidate.candidate_id for candidate in result.selected_candidates] == [
        "c_lower_top_falsifier"
    ]
    assert result.probe_set.probes[0].target_hypotheses == ["H2"]
    assert result.rejected_candidates[0].candidate.candidate_id == (
        "c_high_top_targeting_discriminator"
    )
    assert result.rejected_candidates[0].reason == "not_selected_budget_limit"


def test_planner_does_not_force_falsifier_reservation_in_cycle_zero():
    high_top_targeting_discriminator = make_candidate(
        "c_high_top_targeting_discriminator",
        ["H2"],
        expected_information_gain=1.0,
        decision_relevance=1.0,
        cost_estimate=0.01,
    )
    lower_top_falsifier = make_candidate(
        "c_lower_top_falsifier",
        ["H2"],
        purpose=ProbePurpose.HYPOTHESIS_FALSIFICATION,
        expected_information_gain=0.2,
        decision_relevance=0.2,
        cost_estimate=1.0,
    )

    result = ProbePlanner().design_probe_set(
        run_id="run_plan",
        cycle_id="run_plan_cycle_1",
        belief_state=make_belief_state(
            h1_posterior=0.4,
            h2_posterior=0.6,
            cycle_index=0,
        ),
        candidates=[high_top_targeting_discriminator, lower_top_falsifier],
        config=ProbePlanningConfig(max_probes=1),
    )

    assert [candidate.candidate_id for candidate in result.selected_candidates] == [
        "c_high_top_targeting_discriminator"
    ]


@pytest.mark.parametrize(
    ("candidate", "top_hypothesis_id"),
    [
        (
            make_candidate(
                "c_wrong_purpose",
                ["H2"],
                purpose=ProbePurpose.HYPOTHESIS_DISCRIMINATION,
            ),
            "H2",
        ),
        (
            make_candidate(
                "c_wrong_target",
                ["H1"],
                purpose=ProbePurpose.HYPOTHESIS_FALSIFICATION,
            ),
            "H2",
        ),
        (
            make_candidate(
                "c_empty_weaken",
                ["H2"],
                purpose=ProbePurpose.HYPOTHESIS_FALSIFICATION,
                weaken_condition={"H2": "   "},
            ),
            "H2",
        ),
    ],
)
def test_top_falsification_requires_purpose_target_and_nonempty_weaken_condition(
    candidate,
    top_hypothesis_id,
):
    assert _is_top_falsification(candidate, top_hypothesis_id) is False


def test_top_falsification_accepts_explicit_nonempty_top_weaken_condition():
    candidate = make_candidate(
        "c_valid_falsifier",
        ["H2"],
        purpose=ProbePurpose.HYPOTHESIS_FALSIFICATION,
        weaken_condition={"H2": "Observation X would contradict H2."},
    )

    assert _is_top_falsification(candidate, "H2") is True


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


def test_planner_writes_only_planning_diagnostics_to_ledger(tmp_path: Path):
    ledger = JsonlLedgerStore(tmp_path / "planner-ledger.jsonl")

    ProbePlanner(ledger=ledger).design_probe_set(
        run_id="run_plan",
        cycle_id="run_plan_cycle_1",
        belief_state=make_belief_state(),
        candidates=[make_candidate("c_valid", ["H2"])],
        config=ProbePlanningConfig(max_probes=1),
    )

    record_types = [record["record_type"] for record in ledger.read_all()]
    assert record_types == ["probe_planning"]
    payload = ledger.read_all()[0]["payload"]
    assert payload["run_id"] == "run_plan"
    assert payload["cycle_id"] == "run_plan_cycle_1"
    assert payload["selected_candidate_ids"] == ["c_valid"]
    assert "probe_set" not in record_types
    assert "external_signal" not in record_types
    assert "evidence_event" not in record_types
    assert "belief_update" not in record_types
    assert "answer_projection" not in record_types


def test_initializer_probe_candidates_can_be_planned_and_consumed_by_core():
    initialization = BayesProbeInitializer().initialize(
        InitializeRunInput(
            run_id="run_integrated",
            problem="Should the probe planner feed the core cycle cleanly?",
            hypothesis_seeds=explicit_test_hypothesis_seeds(),
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
    assert result.epistemic_progress is not None
