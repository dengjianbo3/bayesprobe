from pathlib import Path

import pytest

from bayesprobe.core import BayesProbeCore
from bayesprobe.initialization import BayesProbeInitializer, HypothesisSeed, InitializeRunInput
from bayesprobe.ledger import JsonlLedgerStore
from bayesprobe.probe_executor import ModelBackedProbeToolGateway, ProbeExecutor
from bayesprobe.schemas import (
    EvidenceContributionMode,
    ExternalSignal,
    RunRegime,
    RunStatus,
    SignalKind,
)
from bayesprobe.synchronized_runner import (
    SynchronizedRoundInput,
    SynchronizedRoundRunner,
    SynchronizedRoundShape,
    SynchronizedRunInput,
)


class SynchronizedSameRootGateway:
    adapter_kind = "synchronized_stagnation_test"
    model_identity = "synchronized-stagnation-model"

    def complete_structured(self, request):
        if request.task == "execute_probe":
            return {
                "raw_content": "MODEL REASONING: The assessment supports H1."
            }
        if request.task == "judge_evidence":
            return {
                "evidence_type": "supporting",
                "likelihoods": {
                    hypothesis_id: (
                        "weakly_confirming"
                        if hypothesis_id == "H1"
                        else "neutral"
                    )
                    for hypothesis_id in request.input["target_hypotheses"]
                },
                "unresolved_likelihood": None,
                "frame_fit": "explained_by_named",
                "unexplained_observation": None,
                "interpretation": "The same model assessment was evaluated.",
                "quality_overrides": {},
            }
        raise AssertionError(f"unexpected task: {request.task}")


class RecordingSynchronizedCore(BayesProbeCore):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.results = []

    def integrate_cycle(self, **kwargs):
        result = super().integrate_cycle(**kwargs)
        self.results.append(result)
        return result


def passive_refutation_signal(signal_id: str = "S_passive_refute") -> ExternalSignal:
    return ExternalSignal(
        id=signal_id,
        cycle_id="pending",
        signal_kind=SignalKind.PASSIVE,
        source_type="benchmark_stream",
        source="passive_fixture",
        raw_content="REFUTES: Benchmark passage contradicts H1 and supports H2.",
        initial_target_hypotheses=["H1", "H2"],
    )


def explicit_test_hypothesis_seeds() -> list[HypothesisSeed]:
    return [
        HypothesisSeed(id="H1", statement="The fixture's H1 condition holds.", prior=0.5, scope="Deterministic test fixture.", falsifiers=["The fixture emits a reliable H1 refutation."], predictions=["The fixture emits a reliable H1 support cue."]),
        HypothesisSeed(id="H2", statement="The fixture's H2 condition holds instead.", prior=0.5, scope="Deterministic test fixture.", falsifiers=["The fixture emits a reliable H2 refutation."], predictions=["The fixture emits a reliable H2 support cue."]),
    ]


def test_synchronized_runner_processes_new_run_passive_only_round():
    runner = SynchronizedRoundRunner(core=BayesProbeCore())

    result = runner.run_rounds(
        SynchronizedRunInput(
            initialize_input=InitializeRunInput(
                run_id="sync_passive_new",
                problem="Can passive synchronized input revise belief state?",
                hypothesis_seeds=explicit_test_hypothesis_seeds(),
            ),
            rounds=[
                SynchronizedRoundInput(
                    round_id="round_1",
                    shape=SynchronizedRoundShape.PASSIVE_ONLY,
                    passive_signals=[passive_refutation_signal()],
                )
            ],
        )
    )

    round_result = result.round_results[0]
    assert result.run.run_id == "sync_passive_new"
    assert result.run.regime == RunRegime.SYNCHRONIZED
    assert result.run.status == RunStatus.COMPLETED
    assert result.run.current_cycle_id == result.final_belief_state.cycle_id
    assert result.initial_belief_state.cycle_id == "cycle_0"
    assert result.final_belief_state == round_result.belief_state
    assert result.final_belief_state_projection == round_result.belief_state_projection
    assert round_result.round_id == "round_1"
    assert round_result.cycle.signal_shape == "passive_only"
    assert round_result.cycle.round_id == "round_1"
    assert round_result.active_signal_count == 0
    assert round_result.passive_signal_count == 1
    assert round_result.signals[0].signal_kind == SignalKind.PASSIVE
    assert round_result.probe_set.probes == []
    assert round_result.belief_state_projection.current_best_hypothesis == "H1"
    assert round_result.evidence_events
    assert round_result.belief_updates == []
    assert round_result.contribution_deltas == []
    assert round_result.epistemic_progress.max_absolute_contribution_delta == 0.0


def test_synchronized_runner_processes_active_only_round():
    runner = SynchronizedRoundRunner(core=BayesProbeCore())

    result = runner.run_rounds(
        SynchronizedRunInput(
            initialize_input=InitializeRunInput(
                run_id="sync_active_new",
                problem="Can synchronized active probing run inside a round?",
                hypothesis_seeds=explicit_test_hypothesis_seeds(),
            ),
            rounds=[
                SynchronizedRoundInput(
                    round_id="round_active",
                    shape=SynchronizedRoundShape.ACTIVE_ONLY,
                    max_probes=1,
                )
            ],
        )
    )

    round_result = result.round_results[0]
    assert round_result.cycle.signal_shape == "active_only"
    assert round_result.cycle.round_id == "round_active"
    assert round_result.probe_set.probes
    assert round_result.active_signal_count == 1
    assert round_result.passive_signal_count == 0
    assert round_result.signals[0].signal_kind == SignalKind.ACTIVE
    assert round_result.signals[0].generated_by_probe == round_result.probe_set.probes[0].id
    assert round_result.belief_state_projection.current_best_hypothesis == "H1"
    assert round_result.selected_probe_candidates
    assert result.remaining_probe_candidates


def test_synchronized_runner_processes_active_plus_passive_round():
    runner = SynchronizedRoundRunner(core=BayesProbeCore())

    result = runner.run_rounds(
        SynchronizedRunInput(
            initialize_input=InitializeRunInput(
                run_id="sync_mixed_new",
                problem="Can synchronized rounds integrate active and passive signals together?",
                hypothesis_seeds=explicit_test_hypothesis_seeds(),
            ),
            rounds=[
                SynchronizedRoundInput(
                    round_id="round_mixed",
                    shape=SynchronizedRoundShape.ACTIVE_PLUS_PASSIVE,
                    passive_signals=[passive_refutation_signal()],
                    max_probes=1,
                )
            ],
        )
    )

    round_result = result.round_results[0]
    assert round_result.cycle.signal_shape == "active_plus_passive"
    assert round_result.cycle.round_id == "round_mixed"
    assert round_result.active_signal_count == 1
    assert round_result.passive_signal_count == 1
    assert [signal.signal_kind for signal in round_result.signals] == [
        SignalKind.ACTIVE,
        SignalKind.PASSIVE,
    ]
    assert round_result.belief_state_projection.current_best_hypothesis == "H1"
    assert len(round_result.evidence_events) == 2
    assert round_result.belief_updates == []
    assert round_result.contribution_deltas == []


def test_synchronized_runner_carries_projection_candidates_across_rounds():
    runner = SynchronizedRoundRunner(core=BayesProbeCore())

    result = runner.run_rounds(
        SynchronizedRunInput(
            initialize_input=InitializeRunInput(
                run_id="sync_candidate_carry",
                problem="Can synchronized rounds carry projection-derived probes forward?",
                hypothesis_seeds=explicit_test_hypothesis_seeds(),
            ),
            rounds=[
                SynchronizedRoundInput(
                    round_id="round_1",
                    shape=SynchronizedRoundShape.ACTIVE_ONLY,
                    max_probes=1,
                ),
                SynchronizedRoundInput(
                    round_id="round_2",
                    shape=SynchronizedRoundShape.ACTIVE_ONLY,
                    max_probes=1,
                ),
            ],
        )
    )

    first_round = result.round_results[0]
    second_round = result.round_results[1]
    first_projection_candidate = (
        first_round.belief_state_projection.change_my_mind_condition.structured_probe_candidates[0]
    )

    assert len(result.round_results) == 2
    assert second_round.probe_set.probes[0].id.startswith(
        first_projection_candidate.candidate_probe.id
    )
    assert second_round.probe_set.probes[0].cycle_id == "sync_candidate_carry_cycle_2"
    assert result.remaining_probe_candidates[0].candidate_id.startswith("pc_sync_candidate_carry_cycle_2")


def test_synchronized_round_exposes_stagnation_without_ending_external_session():
    gateway = SynchronizedSameRootGateway()
    core = RecordingSynchronizedCore(model_gateway=gateway)
    runner = SynchronizedRoundRunner(
        core=core,
        executor=ProbeExecutor(ModelBackedProbeToolGateway(gateway)),
    )

    result = runner.run_rounds(
        SynchronizedRunInput(
            initialize_input=InitializeRunInput(
                run_id="sync_same_root",
                problem="Does repeated synchronized reasoning add information?",
                hypothesis_seeds=explicit_test_hypothesis_seeds(),
            ),
            rounds=[
                SynchronizedRoundInput(
                    round_id="round_1",
                    shape=SynchronizedRoundShape.ACTIVE_ONLY,
                ),
                SynchronizedRoundInput(
                    round_id="round_2",
                    shape=SynchronizedRoundShape.ACTIVE_ONLY,
                ),
            ],
        )
    )

    first, second = result.round_results
    assert len(result.round_results) == 2
    assert first.contribution_deltas[0].mode == EvidenceContributionMode.NEW_ROOT
    assert second.contribution_deltas[0].mode == EvidenceContributionMode.NO_CHANGE
    assert second.epistemic_progress.no_change_count == 1
    assert first.contribution_deltas is core.results[0].contribution_deltas
    assert second.epistemic_progress is core.results[1].epistemic_progress
    assert second.belief_state.run_id == first.belief_state.run_id
    assert result.run.metadata["stop_reason"] == "fixed_rounds_completed"
    assert result.run.metadata["completed_round_count"] == 2


def test_synchronized_runner_accepts_existing_run_state():
    initialization = BayesProbeInitializer().initialize(
        InitializeRunInput(
            run_id="sync_existing",
            problem="Can synchronized runner resume from existing run state?",
            hypothesis_seeds=explicit_test_hypothesis_seeds(),
            regime=RunRegime.SYNCHRONIZED,
        )
    )
    runner = SynchronizedRoundRunner(core=BayesProbeCore())

    result = runner.run_rounds(
        SynchronizedRunInput(
            run=initialization.run,
            belief_state=initialization.belief_state,
            probe_candidates=initialization.probe_candidates,
            rounds=[
                SynchronizedRoundInput(
                    round_id="round_existing",
                    shape=SynchronizedRoundShape.ACTIVE_ONLY,
                    max_probes=1,
                )
            ],
        )
    )

    assert result.run.run_id == initialization.run.run_id
    assert result.run.regime == RunRegime.SYNCHRONIZED
    assert result.run.status == RunStatus.COMPLETED
    assert result.initial_belief_state == initialization.belief_state
    assert result.round_results[0].cycle.run_id == "sync_existing"
    assert result.round_results[0].active_signal_count == 1


@pytest.mark.parametrize(
    "round_kwargs",
    [
        {"round_id": "", "shape": SynchronizedRoundShape.ACTIVE_ONLY},
        {"round_id": "r", "shape": SynchronizedRoundShape.ACTIVE_ONLY, "max_probes": 0},
        {"round_id": "r", "shape": SynchronizedRoundShape.PASSIVE_ONLY},
        {"round_id": "r", "shape": SynchronizedRoundShape.ACTIVE_PLUS_PASSIVE},
        {
            "round_id": "r",
            "shape": SynchronizedRoundShape.ACTIVE_ONLY,
            "passive_signals": [passive_refutation_signal()],
        },
        {
            "round_id": "r",
            "shape": SynchronizedRoundShape.PASSIVE_ONLY,
            "passive_signals": [
                passive_refutation_signal().model_copy(update={"signal_kind": SignalKind.ACTIVE})
            ],
        },
    ],
)
def test_synchronized_runner_rejects_invalid_round_configuration(round_kwargs):
    with pytest.raises(ValueError):
        SynchronizedRoundInput(**round_kwargs)


def test_synchronized_runner_rejects_invalid_run_configuration():
    valid_round = SynchronizedRoundInput(
        round_id="round_1",
        shape=SynchronizedRoundShape.ACTIVE_ONLY,
    )
    initialization = BayesProbeInitializer().initialize(
        InitializeRunInput(
            run_id="sync_invalid",
            problem="Invalid run config fixture.",
            hypothesis_seeds=explicit_test_hypothesis_seeds(),
        )
    )

    with pytest.raises(ValueError):
        SynchronizedRunInput(rounds=[])
    with pytest.raises(ValueError):
        SynchronizedRunInput(run=initialization.run, rounds=[valid_round])
    with pytest.raises(ValueError):
        SynchronizedRunInput(
            run=initialization.run,
            belief_state=initialization.belief_state.model_copy(update={"run_id": "other"}),
            rounds=[valid_round],
        )


def test_synchronized_runner_writes_projection_ledger_records_without_duplicate_cycles(tmp_path: Path):
    ledger = JsonlLedgerStore(tmp_path / "sync-runner-ledger.jsonl")
    runner = SynchronizedRoundRunner(core=BayesProbeCore(ledger=ledger))

    runner.run_rounds(
        SynchronizedRunInput(
            initialize_input=InitializeRunInput(
                run_id="sync_ledger",
                problem="Does synchronized runner write coherent ledger records?",
                hypothesis_seeds=explicit_test_hypothesis_seeds(),
            ),
            rounds=[
                SynchronizedRoundInput(
                    round_id="round_passive",
                    shape=SynchronizedRoundShape.PASSIVE_ONLY,
                    passive_signals=[passive_refutation_signal("S_ledger_passive")],
                ),
                SynchronizedRoundInput(
                    round_id="round_active",
                    shape=SynchronizedRoundShape.ACTIVE_ONLY,
                    max_probes=1,
                ),
            ],
        )
    )

    record_types = [record["record_type"] for record in ledger.read_all()]
    assert record_types.count("cycle") == 2
    assert record_types.count("belief_state_projection") == 2
    assert "probe_execution" in record_types
    assert "external_signal" in record_types
    assert "evidence_event" in record_types
    assert "epistemic_progress" in record_types
