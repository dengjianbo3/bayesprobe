from __future__ import annotations

import math

import pytest

from bayesprobe.evidence_memory import (
    SignalProvenanceNormalizer,
    derive_deterministic_computation_root,
)
from bayesprobe.evidence_roots import (
    LIKELIHOOD_RATIO_BY_BAND,
    EvidenceRootReconciler,
    resolve_contribution_root_id,
)
from bayesprobe.schemas import (
    EpistemicOrigin,
    EvidenceContributionMode,
    EvidenceEvent,
    EvidenceMemorySnapshot,
    EvidenceRootContribution,
    EvidenceType,
    ExternalSignal,
    LikelihoodBand,
    SignalKind,
    SignalProvenance,
)


def event(
    event_id: str,
    *,
    root: str = "eroot:model",
    likelihoods: dict[str, LikelihoodBand] | None = None,
    h1: LikelihoodBand = LikelihoodBand.NEUTRAL,
    unresolved: LikelihoodBand | None = None,
    origin: EpistemicOrigin = EpistemicOrigin.MODEL_REASONING,
    discard_reason: str | None = None,
) -> EvidenceEvent:
    assessed_likelihoods = likelihoods or {
        "H1": h1,
        "H2": LikelihoodBand.NEUTRAL,
    }
    return EvidenceEvent(
        schema_version="v0.2",
        id=event_id,
        derived_from_signal=f"S_{event_id}",
        epistemic_origin=origin,
        derivation_root_id=f"derivation:{root}",
        contribution_root_id=root,
        target_hypotheses=list(assessed_likelihoods),
        evidence_type=EvidenceType.NEUTRAL,
        content=f"Assessment for {event_id}.",
        reliability=1.0,
        independence=1.0,
        relevance=1.0,
        novelty=1.0,
        likelihoods=assessed_likelihoods,
        unresolved_likelihood=unresolved,
        correlation_status="novel",
        effective_update_weight=None,
        discard_reason=discard_reason,
    )


def reconcile_one(event_id: str, band: LikelihoodBand):
    return EvidenceRootReconciler().reconcile_cycle(
        snapshot=EvidenceMemorySnapshot(memory_version=3),
        evidence_events=[event(event_id, h1=band)],
        falsification_probe_executed=False,
    )


def signal_with_provenance(
    signal_id: str,
    *,
    origin: EpistemicOrigin,
    correlation_group: str,
    derivation_root_id: str,
    cycle_id: str = "cycle_1",
    parent_signal_ids: list[str] | None = None,
) -> ExternalSignal:
    return ExternalSignal(
        id=signal_id,
        cycle_id=cycle_id,
        signal_kind=SignalKind.ACTIVE,
        source_type=origin.value,
        source="fixture",
        raw_content=f"Content for {signal_id}.",
        provenance=SignalProvenance(
            epistemic_origin=origin,
            source_identity=f"source:{signal_id}",
            parent_signal_ids=parent_signal_ids or [],
            derivation_root_id=derivation_root_id,
            correlation_group=correlation_group,
            canonical_content_fingerprint=f"sha256:{'a' * 64}",
        ),
    )


def test_likelihood_ratio_table_matches_the_existing_belief_solver():
    assert LIKELIHOOD_RATIO_BY_BAND == {
        LikelihoodBand.STRONGLY_DISCONFIRMING: 0.1,
        LikelihoodBand.MODERATELY_DISCONFIRMING: 0.3,
        LikelihoodBand.WEAKLY_DISCONFIRMING: 0.7,
        LikelihoodBand.NEUTRAL: 1.0,
        LikelihoodBand.WEAKLY_CONFIRMING: 1.5,
        LikelihoodBand.MODERATELY_CONFIRMING: 3.0,
        LikelihoodBand.STRONGLY_CONFIRMING: 10.0,
    }


def test_same_root_repetition_replaces_instead_of_accumulating():
    reconciler = EvidenceRootReconciler()
    first = reconciler.reconcile_cycle(
        snapshot=EvidenceMemorySnapshot(memory_version=3),
        evidence_events=[
            event("E1", h1=LikelihoodBand.MODERATELY_CONFIRMING)
        ],
        falsification_probe_executed=False,
    )
    second = reconciler.reconcile_cycle(
        snapshot=first.evidence_memory,
        evidence_events=[
            event("E2", h1=LikelihoodBand.MODERATELY_CONFIRMING)
        ],
        falsification_probe_executed=False,
    )

    assert first.contribution_deltas[0].mode == EvidenceContributionMode.NEW_ROOT
    assert second.contribution_deltas[0].mode == EvidenceContributionMode.NO_CHANGE
    assert second.contribution_deltas[0].per_hypothesis_delta == {
        "H1": 0.0,
        "H2": 0.0,
    }


def test_same_cycle_same_root_events_are_meaned_and_order_independent():
    events = [
        event("E1", h1=LikelihoodBand.STRONGLY_CONFIRMING),
        event("E2", h1=LikelihoodBand.WEAKLY_DISCONFIRMING),
    ]
    forward = EvidenceRootReconciler().reconcile_cycle(
        snapshot=EvidenceMemorySnapshot(memory_version=3),
        evidence_events=events,
        falsification_probe_executed=False,
    )
    reverse = EvidenceRootReconciler().reconcile_cycle(
        snapshot=EvidenceMemorySnapshot(memory_version=3),
        evidence_events=list(reversed(events)),
        falsification_probe_executed=False,
    )

    assert forward.contribution_deltas == reverse.contribution_deltas
    assert forward.evidence_events == reverse.evidence_events
    assert len(forward.contribution_deltas) == 1
    assert forward.contribution_deltas[0].current_contribution.assessment_event_ids == [
        "E1",
        "E2",
    ]
    assert forward.contribution_deltas[0].per_hypothesis_delta["H1"] == pytest.approx(
        (math.log(10.0) + math.log(0.7)) / 2.0
    )


def test_same_root_counterassessment_can_reverse_prior_contribution():
    first = reconcile_one("E1", LikelihoodBand.STRONGLY_CONFIRMING)
    second = EvidenceRootReconciler().reconcile_cycle(
        snapshot=first.evidence_memory,
        evidence_events=[
            event("E2", h1=LikelihoodBand.STRONGLY_DISCONFIRMING)
        ],
        falsification_probe_executed=True,
    )

    delta = second.contribution_deltas[0]
    assert delta.mode == EvidenceContributionMode.REVISE_ROOT
    assert delta.per_hypothesis_delta["H1"] < 0
    assert second.epistemic_progress.falsification_probe_executed is True


def test_independent_roots_create_two_independent_contributions():
    result = EvidenceRootReconciler().reconcile_cycle(
        snapshot=EvidenceMemorySnapshot(memory_version=3),
        evidence_events=[
            event(
                "E1",
                root="eroot:source-a",
                h1=LikelihoodBand.MODERATELY_CONFIRMING,
            ),
            event(
                "E2",
                root="eroot:source-b",
                h1=LikelihoodBand.MODERATELY_CONFIRMING,
            ),
        ],
        falsification_probe_executed=False,
    )

    assert [item.mode for item in result.contribution_deltas] == [
        EvidenceContributionMode.NEW_ROOT,
        EvidenceContributionMode.NEW_ROOT,
    ]
    assert [item.contribution_root_id for item in result.contribution_deltas] == [
        "eroot:source-a",
        "eroot:source-b",
    ]
    assert result.epistemic_progress.new_root_count == 2


def test_all_neutral_reassessment_retracts_an_active_root():
    first = reconcile_one("E1", LikelihoodBand.MODERATELY_CONFIRMING)
    second = EvidenceRootReconciler().reconcile_cycle(
        snapshot=first.evidence_memory,
        evidence_events=[event("E2")],
        falsification_probe_executed=False,
    )

    delta = second.contribution_deltas[0]
    assert delta.mode == EvidenceContributionMode.RETRACT_ROOT
    assert delta.current_contribution.active is False
    assert delta.current_contribution.revision == 2
    assert delta.per_hypothesis_delta["H1"] == pytest.approx(-math.log(3.0))
    assert second.epistemic_progress.retracted_root_count == 1


def test_same_root_can_move_support_between_hypotheses():
    reconciler = EvidenceRootReconciler()
    first = reconciler.reconcile_cycle(
        snapshot=EvidenceMemorySnapshot(memory_version=3),
        evidence_events=[
            event(
                "E1",
                likelihoods={"B": LikelihoodBand.STRONGLY_CONFIRMING},
            )
        ],
        falsification_probe_executed=False,
    )
    second = reconciler.reconcile_cycle(
        snapshot=first.evidence_memory,
        evidence_events=[
            event(
                "E2",
                likelihoods={"C": LikelihoodBand.MODERATELY_CONFIRMING},
            )
        ],
        falsification_probe_executed=False,
    )

    delta = second.contribution_deltas[0]
    assert delta.mode == EvidenceContributionMode.REVISE_ROOT
    assert delta.current_contribution.per_hypothesis_log_likelihood == {
        "C": math.log(3.0)
    }
    assert delta.per_hypothesis_delta == {
        "B": -math.log(10.0),
        "C": math.log(3.0),
    }


def test_discarded_events_contribute_zero_and_are_preserved_in_output():
    snapshot = EvidenceMemorySnapshot(memory_version=3)
    discarded = event(
        "E_discarded",
        h1=LikelihoodBand.STRONGLY_CONFIRMING,
        discard_reason="schema_violation:invalid judgment",
    )

    result = EvidenceRootReconciler().reconcile_cycle(
        snapshot=snapshot,
        evidence_events=[discarded],
        falsification_probe_executed=False,
    )

    assert result.evidence_events == [discarded]
    assert result.contribution_deltas == []
    assert result.evidence_memory == snapshot
    assert result.epistemic_progress.max_absolute_contribution_delta == 0.0


def test_reconciliation_preserves_snapshot_and_event_input_immutability():
    prior = reconcile_one("E1", LikelihoodBand.WEAKLY_CONFIRMING)
    snapshot_dump = prior.evidence_memory.model_dump(mode="python")
    events = [event("E3"), event("E2", root="eroot:a")]
    event_dumps = [item.model_dump(mode="python") for item in events]

    result = EvidenceRootReconciler().reconcile_cycle(
        snapshot=prior.evidence_memory,
        evidence_events=events,
        falsification_probe_executed=False,
    )

    assert prior.evidence_memory.model_dump(mode="python") == snapshot_dump
    assert [item.model_dump(mode="python") for item in events] == event_dumps
    assert [item.id for item in result.evidence_events] == ["E2", "E3"]


def test_memory_root_order_is_canonical_independent_of_snapshot_map_order():
    root_a = EvidenceRootContribution(
        contribution_root_id="eroot:a",
        revision=1,
        assessment_event_ids=["E1"],
        epistemic_origin=EpistemicOrigin.MODEL_REASONING,
        per_hypothesis_log_likelihood={"H1": math.log(1.5)},
    )
    root_z = root_a.model_copy(
        update={"contribution_root_id": "eroot:z", "assessment_event_ids": ["E2"]}
    )
    forward_snapshot = EvidenceMemorySnapshot(
        memory_version=3,
        root_contributions={"eroot:a": root_a, "eroot:z": root_z},
    )
    reverse_snapshot = EvidenceMemorySnapshot(
        memory_version=3,
        root_contributions={"eroot:z": root_z, "eroot:a": root_a},
    )

    forward = EvidenceRootReconciler().reconcile_cycle(
        snapshot=forward_snapshot,
        evidence_events=[],
        falsification_probe_executed=False,
    )
    reverse = EvidenceRootReconciler().reconcile_cycle(
        snapshot=reverse_snapshot,
        evidence_events=[],
        falsification_probe_executed=False,
    )

    assert list(forward.evidence_memory.root_contributions) == [
        "eroot:a",
        "eroot:z",
    ]
    assert forward.evidence_memory.model_dump_json() == (
        reverse.evidence_memory.model_dump_json()
    )


def test_reconciler_requires_memory_v3_and_root_bound_v02_events():
    reconciler = EvidenceRootReconciler()

    with pytest.raises(ValueError, match="memory version 3"):
        reconciler.reconcile_cycle(
            snapshot=EvidenceMemorySnapshot(memory_version=2),
            evidence_events=[],
            falsification_probe_executed=False,
        )

    unbound = event("E1").model_copy(update={"contribution_root_id": None})
    with pytest.raises(ValueError, match="root-bound v0.2"):
        reconciler.reconcile_cycle(
            snapshot=EvidenceMemorySnapshot(memory_version=3),
            evidence_events=[unbound],
            falsification_probe_executed=False,
        )


def test_root_resolution_requires_normalized_provenance():
    signal = ExternalSignal(
        id="S1",
        cycle_id="cycle_1",
        signal_kind=SignalKind.ACTIVE,
        source_type="model_probe_gateway",
        source="fixture-model",
        raw_content="An assessment.",
    )

    with pytest.raises(ValueError, match="normalized provenance"):
        resolve_contribution_root_id(signal)


def test_model_signals_from_same_provider_session_share_root_across_cycles():
    normalizer = SignalProvenanceNormalizer()
    first = normalizer.normalize(
        ExternalSignal(
            id="S1",
            cycle_id="cycle_1",
            signal_kind=SignalKind.ACTIVE,
            source_type="model_probe_gateway",
            source="fixture-model",
            raw_content="First assessment.",
        ),
        run_id="run_same_session",
    )
    second = normalizer.normalize(
        ExternalSignal(
            id="S2",
            cycle_id="cycle_2",
            signal_kind=SignalKind.ACTIVE,
            source_type="model_probe_gateway",
            source="fixture-model",
            raw_content="Second assessment.",
        ),
        run_id="run_same_session",
    )

    assert resolve_contribution_root_id(first) == resolve_contribution_root_id(second)


def test_correlation_root_uses_explicit_canonical_correlation_group():
    signal = signal_with_provenance(
        "S_raw_group",
        origin=EpistemicOrigin.RETRIEVED_SOURCE,
        correlation_group="group:raw-declaration",
        derivation_root_id="derivation:unused",
    )
    canonical_signal = signal.model_copy(
        update={
            "provenance": signal.provenance.model_copy(
                update={"correlation_group": "group:canonical"}
            )
        }
    )

    assert resolve_contribution_root_id(
        signal,
        canonical_correlation_group="group:canonical",
    ) == resolve_contribution_root_id(canonical_signal)


@pytest.mark.parametrize(
    ("parent_origin", "child_origin"),
    [
        pytest.param(
            EpistemicOrigin.MODEL_REASONING,
            EpistemicOrigin.DERIVED_SUMMARY,
            id="model-to-derived-summary",
        ),
        pytest.param(
            EpistemicOrigin.RETRIEVED_SOURCE,
            EpistemicOrigin.TOOL_RESULT,
            id="retrieved-source-to-tool",
        ),
        pytest.param(
            EpistemicOrigin.TOOL_RESULT,
            EpistemicOrigin.EXTERNAL_OBSERVATION,
            id="tool-to-external-observation",
        ),
        pytest.param(
            EpistemicOrigin.EXTERNAL_OBSERVATION,
            EpistemicOrigin.MODEL_REASONING,
            id="external-observation-to-model",
        ),
    ],
)
def test_child_inherits_parent_root_unchanged_across_origin_changes(
    parent_origin: EpistemicOrigin,
    child_origin: EpistemicOrigin,
):
    parent = signal_with_provenance(
        "S_parent",
        origin=parent_origin,
        correlation_group="group:parent",
        derivation_root_id="derivation:parent",
    )
    parent_root = resolve_contribution_root_id(parent)
    child = signal_with_provenance(
        "S_child",
        origin=child_origin,
        correlation_group="group:child-must-not-be-used",
        derivation_root_id="derivation:child-must-not-be-used",
        parent_signal_ids=[parent.id],
    )

    assert resolve_contribution_root_id(
        child,
        parent_contribution_roots={parent.id: parent_root},
    ) == parent_root


@pytest.mark.parametrize(
    ("first_origin", "second_origin", "correlation_group", "derivation_root"),
    [
        pytest.param(
            EpistemicOrigin.MODEL_REASONING,
            EpistemicOrigin.RETRIEVED_SOURCE,
            "group:shared-canonical-basis",
            "unused:correlation-rooted",
            id="correlation-basis",
        ),
        pytest.param(
            EpistemicOrigin.TOOL_RESULT,
            EpistemicOrigin.EXTERNAL_OBSERVATION,
            "unused:derivation-rooted",
            "derivation:shared-canonical-basis",
            id="derivation-basis",
        ),
    ],
)
def test_root_signal_basis_is_not_split_by_origin_label(
    first_origin: EpistemicOrigin,
    second_origin: EpistemicOrigin,
    correlation_group: str,
    derivation_root: str,
):
    first = signal_with_provenance(
        "S_root_1",
        origin=first_origin,
        correlation_group=correlation_group,
        derivation_root_id=derivation_root,
    )
    second = signal_with_provenance(
        "S_root_2",
        origin=second_origin,
        correlation_group=correlation_group,
        derivation_root_id=derivation_root,
    )

    assert resolve_contribution_root_id(first) == resolve_contribution_root_id(second)


def test_parentless_derived_summary_fails_closed():
    summary = signal_with_provenance(
        "S_summary",
        origin=EpistemicOrigin.DERIVED_SUMMARY,
        correlation_group="group:must-not-mint-root",
        derivation_root_id="derivation:must-not-mint-root",
    )

    with pytest.raises(ValueError, match="derived summary requires parent signals"):
        resolve_contribution_root_id(summary)


def test_identical_basis_text_across_root_policies_resolves_distinctly():
    shared_basis = "canonical:shared-basis-text"
    correlation_rooted = signal_with_provenance(
        "S_correlation",
        origin=EpistemicOrigin.MODEL_REASONING,
        correlation_group=shared_basis,
        derivation_root_id="unused:correlation-policy",
    )
    derivation_rooted = signal_with_provenance(
        "S_derivation",
        origin=EpistemicOrigin.TOOL_RESULT,
        correlation_group="unused:derivation-policy",
        derivation_root_id=shared_basis,
    )

    assert resolve_contribution_root_id(correlation_rooted) != (
        resolve_contribution_root_id(derivation_rooted)
    )


def test_child_without_parent_root_mapping_fails_closed():
    child = signal_with_provenance(
        "S_child",
        origin=EpistemicOrigin.DERIVED_SUMMARY,
        correlation_group="group:child",
        derivation_root_id="derivation:child",
        parent_signal_ids=["S_parent"],
    )

    with pytest.raises(ValueError, match="requires parent contribution roots"):
        resolve_contribution_root_id(child)


def test_child_with_missing_parent_root_fails_closed():
    child = signal_with_provenance(
        "S_child",
        origin=EpistemicOrigin.DERIVED_SUMMARY,
        correlation_group="group:child",
        derivation_root_id="derivation:child",
        parent_signal_ids=["S_parent"],
    )

    with pytest.raises(ValueError, match="missing a parent contribution root"):
        resolve_contribution_root_id(child, parent_contribution_roots={})


def test_child_with_multiple_distinct_parent_roots_fails_closed():
    child = signal_with_provenance(
        "S_child",
        origin=EpistemicOrigin.TOOL_RESULT,
        correlation_group="group:child",
        derivation_root_id="derivation:child",
        parent_signal_ids=["S_parent_1", "S_parent_2"],
    )

    with pytest.raises(ValueError, match="exactly one parent contribution root"):
        resolve_contribution_root_id(
            child,
            parent_contribution_roots={
                "S_parent_1": "evidence-root:parent-1",
                "S_parent_2": "evidence-root:parent-2",
            },
        )


def test_deterministic_tool_inputs_resolve_to_independent_roots():
    first_computation = derive_deterministic_computation_root(
        tool_identity="python:sum",
        computation_inputs={"values": [1, 2]},
    )
    second_computation = derive_deterministic_computation_root(
        tool_identity="python:sum",
        computation_inputs={"values": [1, 3]},
    )
    first = signal_with_provenance(
        "S_tool_1",
        origin=EpistemicOrigin.TOOL_RESULT,
        correlation_group="tool:shared",
        derivation_root_id=first_computation,
    )
    second = signal_with_provenance(
        "S_tool_2",
        origin=EpistemicOrigin.TOOL_RESULT,
        correlation_group="tool:shared",
        derivation_root_id=second_computation,
    )

    assert resolve_contribution_root_id(first) != resolve_contribution_root_id(second)
