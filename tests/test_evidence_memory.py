import json
import re
from dataclasses import replace

import pytest

from bayesprobe.evidence import EvidenceIntegrationGate
from bayesprobe.evidence_memory import (
    EvidenceMemoryManager,
    SignalProvenanceNormalizer,
    derive_deterministic_computation_root,
)
from bayesprobe.initialization import BayesProbeInitializer, InitializeRunInput
from bayesprobe.kernel_config import CorrelationCreditPolicy
from bayesprobe.schemas import (
    AnswerChoice,
    CycleRecord,
    CycleSignalShape,
    EpistemicOrigin,
    EvidenceEvent,
    EvidenceMemorySnapshot,
    EvidenceType,
    ExternalSignal,
    FramingMethod,
    LikelihoodBand,
    ProbeSet,
    SignalKind,
    SignalProvenance,
)


class CountingGateway:
    adapter_kind = "counting"

    def __init__(self) -> None:
        self.requests = []

    def complete_structured(self, request):
        self.requests.append(request)
        return {
            "evidence_type": "supporting",
            "likelihoods": {
                "A": "moderately_confirming",
                "B": "moderately_disconfirming",
            },
            "unresolved_likelihood": None,
            "frame_fit": "explained_by_named",
            "unexplained_observation": None,
            "interpretation": "The source favors A over B.",
            "quality_overrides": {},
        }


def _state():
    return BayesProbeInitializer().initialize(
        InitializeRunInput(
            run_id="run_memory",
            problem="Which option is supported?",
            answer_choices=[
                AnswerChoice(label="A", text="Option A"),
                AnswerChoice(label="B", text="Option B"),
            ],
        )
    ).belief_state


def _cycle(index: int) -> CycleRecord:
    return CycleRecord(
        cycle_id=f"cycle_{index}",
        run_id="run_memory",
        cycle_index=index,
        signal_shape=CycleSignalShape.ACTIVE_ONLY,
    )


def _probe_set(index: int) -> ProbeSet:
    return ProbeSet(
        probe_set_id=f"ps_{index}",
        cycle_id=f"cycle_{index}",
        probes=[],
        selection_reason="Evidence-memory fixture.",
        may_be_empty=True,
    )


def _signal(
    signal_id: str,
    content: str,
    *,
    root: str = "root-source-1",
) -> ExternalSignal:
    return ExternalSignal(
        id=signal_id,
        cycle_id="pending",
        signal_kind=SignalKind.ACTIVE,
        source_type="retrieved_source",
        source="source.example/report",
        raw_content=content,
        initial_target_hypotheses=["A", "B"],
        provenance=SignalProvenance(
            epistemic_origin=EpistemicOrigin.RETRIEVED_SOURCE,
            source_identity="source.example/report",
            derivation_root_id=root,
            correlation_group="source.example/report",
            canonical_content_fingerprint="to-be-normalized",
            citations=["source.example/report#finding"],
        ),
    )


def test_deterministic_computation_root_canonicalizes_object_key_order_only():
    first = derive_deterministic_computation_root(
        tool_identity="deterministic probe gateway",
        computation_inputs={
            "inquiry_goal": "Cafe\u0301   comparison\nresult",
            "conditions": {"B": "weakens", "A": "supports"},
            "targets": ["A", "B"],
        },
    )
    equivalent = derive_deterministic_computation_root(
        tool_identity="deterministic probe gateway",
        computation_inputs={
            "targets": ["A", "B"],
            "conditions": {"A": "supports", "B": "weakens"},
            "inquiry_goal": "Cafe\u0301   comparison\nresult",
        },
    )
    changed = derive_deterministic_computation_root(
        tool_identity="deterministic probe gateway",
        computation_inputs={
            "targets": ["A", "B"],
            "conditions": {"A": "supports", "B": "weakens"},
            "inquiry_goal": "A materially different comparison",
        },
    )

    assert first == equivalent
    assert first != changed
    assert re.fullmatch(
        r"deterministic-computation:sha256:[0-9a-f]{64}",
        first,
    )


@pytest.mark.parametrize(
    ("first_value", "second_value"),
    [
        ("if True:\n    print(1)", "if True:\n  print(1)"),
        ("print(1)\nprint(2)", "print(1) print(2)"),
        ("print('a b')", "print('a  b')"),
        ("value = '\u2163'", "value = 'IV'"),
        ("Cafe\u0301", "Caf\u00e9"),
    ],
)
def test_deterministic_computation_root_preserves_exact_unicode_string_values(
    first_value,
    second_value,
):
    first = derive_deterministic_computation_root(
        tool_identity="safe-tool",
        computation_inputs={"code": first_value},
    )
    second = derive_deterministic_computation_root(
        tool_identity="safe-tool",
        computation_inputs={"code": second_value},
    )

    assert first != second


@pytest.mark.parametrize(
    "computation_inputs",
    [
        {"nested": {"api_key": "ordinary-looking-value"}},
        {"code": "token=provider-secret-value-123"},
    ],
)
def test_deterministic_computation_root_rejects_secrets_before_hashing(
    computation_inputs,
):
    with pytest.raises(ValueError) as captured:
        derive_deterministic_computation_root(
            tool_identity="safe-tool",
            computation_inputs=computation_inputs,
        )

    assert "secret" in str(captured.value).lower()
    assert "provider-secret-value-123" not in str(captured.value)


def test_normalization_is_deterministic_and_model_session_groups_are_stable():
    normalizer = SignalProvenanceNormalizer()
    first = ExternalSignal(
        id="S_model_1",
        cycle_id="pending",
        signal_kind=SignalKind.ACTIVE,
        source_type="model_probe_gateway",
        source="model_gateway:scripted",
        raw_content="Cafe\u0301   result\n supports H1.",
    )
    equivalent = first.model_copy(
        update={"id": "S_model_2", "raw_content": "Caf\u00e9 result supports H1."}
    )
    different = first.model_copy(
        update={"id": "S_model_3", "raw_content": "A different conclusion."}
    )

    normalized = normalizer.normalize(first, run_id="run_1")
    normalized_equivalent = normalizer.normalize(equivalent, run_id="run_1")
    normalized_different = normalizer.normalize(different, run_id="run_1")
    another_session = normalizer.normalize(different, run_id="run_2")

    assert normalized.provenance.canonical_content_fingerprint == (
        normalized_equivalent.provenance.canonical_content_fingerprint
    )
    assert normalized.provenance.correlation_group == (
        normalized_different.provenance.correlation_group
    )
    assert normalized.provenance.correlation_group != (
        another_session.provenance.correlation_group
    )
    assert normalized.provenance.derivation_root_id != (
        normalized_different.provenance.derivation_root_id
    )


def test_derived_summary_preserves_supplied_derivation_root():
    normalizer = SignalProvenanceNormalizer()
    summary = ExternalSignal(
        id="S_summary",
        cycle_id="pending",
        signal_kind=SignalKind.PASSIVE,
        source_type="derived_summary",
        source="summary-worker",
        raw_content="A compact restatement.",
        provenance=SignalProvenance(
            epistemic_origin=EpistemicOrigin.DERIVED_SUMMARY,
            source_identity="summary-worker",
            provider_model_or_tool_identity="summary-model",
            session_id="run_1",
            parent_signal_ids=["S_parent"],
            derivation_root_id="root-parent",
            correlation_group="group-parent",
            canonical_content_fingerprint="replace-me",
        ),
    )

    normalized = normalizer.normalize(summary, run_id="run_1")

    assert normalized.provenance.derivation_root_id == "root-parent"
    assert normalized.provenance.parent_signal_ids == ["S_parent"]


def test_supplied_model_group_cannot_override_stable_provider_session_group():
    normalizer = SignalProvenanceNormalizer()
    base = ExternalSignal(
        id="S_supplied_model_1",
        cycle_id="pending",
        signal_kind=SignalKind.ACTIVE,
        source_type="custom_model_adapter",
        source="provider/model-a",
        raw_content="First model conclusion.",
        provenance=SignalProvenance(
            epistemic_origin=EpistemicOrigin.MODEL_REASONING,
            source_identity="provider/model-a",
            provider_model_or_tool_identity="provider/model-a",
            session_id="session-1",
            derivation_root_id="root-model-1",
            correlation_group="caller-group-1",
            canonical_content_fingerprint="replace-me",
        ),
    )
    second = base.model_copy(
        update={
            "id": "S_supplied_model_2",
            "raw_content": "Second model conclusion.",
            "provenance": base.provenance.model_copy(
                update={
                    "derivation_root_id": "root-model-2",
                    "correlation_group": "caller-group-2",
                }
            ),
        }
    )

    first_normalized = normalizer.normalize(base, run_id="run_memory")
    second_normalized = normalizer.normalize(second, run_id="run_memory")

    assert first_normalized.provenance.correlation_group == (
        second_normalized.provenance.correlation_group
    )
    assert first_normalized.provenance.correlation_group.startswith("model:")


def test_normalization_rejects_secret_material_without_echoing_it():
    signal = ExternalSignal(
        id="S_secret",
        cycle_id="pending",
        signal_kind=SignalKind.PASSIVE,
        source_type="human_input",
        source="human",
        raw_content="Authorization: Bearer provider-secret-value-123",
    )

    with pytest.raises(ValueError) as captured:
        SignalProvenanceNormalizer().normalize(signal, run_id="run_1")

    assert "secret" in str(captured.value).lower()
    assert "provider-secret-value-123" not in str(captured.value)


def test_normalization_rejects_secret_revealed_by_unicode_canonicalization():
    signal = ExternalSignal(
        id="S_unicode_secret",
        cycle_id="pending",
        signal_kind=SignalKind.PASSIVE,
        source_type="human_input",
        source="human",
        raw_content=(
            "\uff21\uff55\uff54\uff48\uff4f\uff52\uff49\uff5a\uff41\uff54\uff49\uff4f\uff4e\uff1a "
            "\uff22\uff45\uff41\uff52\uff45\uff52 provider-secret-value-123"
        ),
    )

    with pytest.raises(ValueError, match="secret"):
        SignalProvenanceNormalizer().normalize(signal, run_id="run_1")


def test_exact_cross_cycle_repeat_produces_no_update_or_provider_call():
    gateway = CountingGateway()
    gate = EvidenceIntegrationGate(
        model_gateway=gateway,
        provenance_normalizer=SignalProvenanceNormalizer(),
        memory_manager=EvidenceMemoryManager(),
    )
    state = _state()
    first = gate.integrate(
        cycle=_cycle(1),
        belief_state=state,
        probe_set=_probe_set(1),
        signals=[_signal("S_first", "The audited value supports A.")],
    )
    state = state.model_copy(update={"evidence_memory": first.evidence_memory})

    repeated = gate.integrate(
        cycle=_cycle(2),
        belief_state=state,
        probe_set=_probe_set(2),
        signals=[_signal("S_repeat", "The audited value supports A.")],
    )

    event = repeated.evidence_events[0]
    assert event.discard_reason == "duplicate_exact"
    assert event.effective_update_weight == 0.0
    assert len(gateway.requests) == 1


def test_native_event_identity_survives_reordered_and_inserted_signals():
    gate = EvidenceIntegrationGate(model_gateway=CountingGateway())
    state = _state()
    cycle = _cycle(1)
    first_signal = _signal("S_first", "The first audited observation.", root="root-1")
    second_signal = _signal("S_second", "The second audited observation.", root="root-2")
    inserted = _signal("S_inserted", "An inserted audited observation.", root="root-3")

    original = gate.integrate(
        cycle=cycle,
        belief_state=state,
        probe_set=_probe_set(1),
        signals=[first_signal, second_signal],
    )
    reordered = gate.integrate(
        cycle=cycle,
        belief_state=state,
        probe_set=_probe_set(1),
        signals=[inserted, second_signal, first_signal],
    )

    original_ids = {
        event.derived_from_signal: event.id for event in original.evidence_events
    }
    reordered_ids = {
        event.derived_from_signal: event.id for event in reordered.evidence_events
    }
    assert reordered_ids["S_first"] == original_ids["S_first"]
    assert reordered_ids["S_second"] == original_ids["S_second"]


def test_native_event_identity_is_unique_for_duplicate_signals_in_one_batch():
    gateway = CountingGateway()
    duplicate = _signal("S_duplicate_1", "The same audited observation.")

    result = EvidenceIntegrationGate(model_gateway=gateway).integrate(
        cycle=_cycle(1),
        belief_state=_state(),
        probe_set=_probe_set(1),
        signals=[duplicate, duplicate.model_copy(update={"id": "S_duplicate_2"})],
    )

    assert len({event.id for event in result.evidence_events}) == 2
    assert result.evidence_events[0].id.endswith("_1")
    assert result.evidence_events[1].id.endswith("_2")
    assert len(gateway.requests) == 1


def test_native_event_identity_distinguishes_same_content_with_different_roots():
    gate = EvidenceIntegrationGate(model_gateway=CountingGateway())
    state = _state()
    cycle = _cycle(1)
    first = _signal("S_same_content_root_1", "Shared wording.", root="root-1")
    second = _signal("S_same_content_root_2", "Shared wording.", root="root-2")

    original = gate.integrate(
        cycle=cycle,
        belief_state=state,
        probe_set=_probe_set(1),
        signals=[first, second],
    )
    reordered = gate.integrate(
        cycle=cycle,
        belief_state=state,
        probe_set=_probe_set(1),
        signals=[second, first],
    )

    original_ids = {
        event.derived_from_signal: event.id for event in original.evidence_events
    }
    reordered_ids = {
        event.derived_from_signal: event.id for event in reordered.evidence_events
    }
    assert reordered_ids == original_ids


def test_same_root_restatement_has_zero_independence():
    gateway = CountingGateway()
    gate = EvidenceIntegrationGate(
        model_gateway=gateway,
        provenance_normalizer=SignalProvenanceNormalizer(),
        memory_manager=EvidenceMemoryManager(),
    )
    state = _state()
    first = gate.integrate(
        cycle=_cycle(1),
        belief_state=state,
        probe_set=_probe_set(1),
        signals=[_signal("S_first", "The audited value supports A.")],
    )
    state = state.model_copy(update={"evidence_memory": first.evidence_memory})

    restated = gate.integrate(
        cycle=_cycle(2),
        belief_state=state,
        probe_set=_probe_set(2),
        signals=[
            _signal(
                "S_restatement",
                "In other words, the audit points toward option A.",
            )
        ],
    )

    event = restated.evidence_events[0]
    assert event.correlation_status == "correlated_restatement"
    assert event.independence == 0.0
    assert event.effective_update_weight == 0.0
    assert len(gateway.requests) == 2


@pytest.mark.parametrize("value", [True, 0, -1, float("inf"), float("nan"), "1"])
def test_correlation_credit_policy_rejects_invalid_caps(value):
    with pytest.raises(ValueError, match="correlation credit cap"):
        CorrelationCreditPolicy(value)


def test_credit_keys_include_direction_subject_and_internal_unresolved_subject():
    manager = EvidenceMemoryManager(
        CorrelationCreditPolicy(max_cumulative_effective_weight_per_direction=1.0)
    )
    signal = SignalProvenanceNormalizer().normalize(
        _signal("S_credit_1", "First independent measurement.", root="root-1"),
        run_id="run_memory",
    )
    decision = manager.classify(
        EvidenceMemorySnapshot(),
        signal,
        likelihoods={
            "A": LikelihoodBand.MODERATELY_CONFIRMING,
            "B": LikelihoodBand.MODERATELY_DISCONFIRMING,
        },
        unresolved_likelihood=LikelihoodBand.WEAKLY_CONFIRMING,
        frame_version=3,
        base_effective_weight=0.6,
    )
    group = signal.provenance.correlation_group

    assert decision.remaining_credit == {
        f"{group}|A|confirming": 0.4,
        f"{group}|B|disconfirming": 0.4,
        f"{group}|frame:3:unresolved|confirming": 0.4,
    }
    assert "H_other" not in repr(decision.remaining_credit)


def test_accepted_neutral_event_preserves_existing_directional_credit():
    manager = EvidenceMemoryManager()
    signal = SignalProvenanceNormalizer().normalize(
        _signal("S_neutral_credit", "A neutral observation.", root="root-neutral"),
        run_id="run_memory",
    )
    group = signal.provenance.correlation_group
    original_credit = {
        f"{group}|A|confirming": 0.2,
        f"{group}|B|disconfirming": 0.35,
        f"{group}|frame:3:unresolved|confirming": 0.1,
        "other-group|A|confirming": 0.4,
    }
    snapshot = EvidenceMemorySnapshot(correlation_credit=original_credit)
    decision = manager.classify(
        snapshot,
        signal,
        likelihoods={"A": LikelihoodBand.NEUTRAL, "B": LikelihoodBand.NEUTRAL},
        unresolved_likelihood=LikelihoodBand.NEUTRAL,
        frame_version=3,
        base_effective_weight=0.4,
    )
    event = EvidenceEvent(
        id="E_neutral_credit",
        derived_from_signal=signal.id,
        target_hypotheses=["A", "B"],
        evidence_type=EvidenceType.NEUTRAL,
        content=signal.raw_content,
        likelihoods={"A": LikelihoodBand.NEUTRAL, "B": LikelihoodBand.NEUTRAL},
        unresolved_likelihood=LikelihoodBand.NEUTRAL,
        effective_update_weight=decision.effective_update_weight,
    )

    committed = manager.commit(
        snapshot,
        signal=signal,
        event=event,
        decision=decision,
    )

    assert decision.remaining_credit == {}
    assert committed.accepted_evidence_ids == [event.id]
    assert committed.correlation_credit == original_credit


def test_correlation_credit_saturation_stays_visible_with_zero_weight():
    manager = EvidenceMemoryManager(
        CorrelationCreditPolicy(max_cumulative_effective_weight_per_direction=0.5)
    )
    normalizer = SignalProvenanceNormalizer()
    memory = EvidenceMemorySnapshot()

    for index in (1, 2):
        signal = normalizer.normalize(
            _signal(
                f"S_credit_{index}",
                f"Independent wording {index}.",
                root=f"root-{index}",
            ),
            run_id="run_memory",
        )
        decision = manager.classify(
            memory,
            signal,
            likelihoods={"A": LikelihoodBand.MODERATELY_CONFIRMING},
            base_effective_weight=0.5,
        )
        event = EvidenceEvent(
            id=f"E_credit_{index}",
            derived_from_signal=signal.id,
            epistemic_origin=signal.provenance.epistemic_origin,
            derivation_root_id=signal.provenance.derivation_root_id,
            target_hypotheses=["A"],
            evidence_type=EvidenceType.SUPPORTING,
            content=signal.raw_content,
            likelihoods={"A": LikelihoodBand.MODERATELY_CONFIRMING},
            correlation_status=decision.correlation_status,
            effective_update_weight=decision.effective_update_weight,
            discard_reason=decision.discard_reason,
        )
        memory = manager.commit(memory, signal=signal, event=event, decision=decision)

    assert decision.correlation_status == "correlated_novel"
    assert decision.effective_update_weight == 0.0
    assert decision.discard_reason == "correlation_credit_saturated"
    assert memory.discard_and_schema_history == [
        '["E_credit_2","correlation_credit_saturated"]'
    ]


def test_discard_history_uses_exact_event_id_with_colons_for_idempotency():
    manager = EvidenceMemoryManager()
    signal = SignalProvenanceNormalizer().normalize(
        _signal("S_discard_colons", "A malformed provider result."),
        run_id="run_memory",
    )
    decision = manager.classify(EvidenceMemorySnapshot(), signal)
    event = EvidenceEvent(
        id="run:cycle:event:1",
        derived_from_signal=signal.id,
        target_hypotheses=["A"],
        evidence_type=EvidenceType.NEUTRAL,
        content=signal.raw_content,
        likelihoods={"A": LikelihoodBand.NEUTRAL},
        discard_reason="schema_violation:invalid judgment",
    )

    committed = manager.commit(
        EvidenceMemorySnapshot(),
        signal=signal,
        event=event,
        decision=decision,
    )
    recommitted = manager.commit(
        committed,
        signal=signal,
        event=event,
        decision=decision,
    )

    assert committed.discard_and_schema_history == [
        '["run:cycle:event:1","schema_violation:invalid judgment"]'
    ]
    assert recommitted == committed


def _committed_signal_identity():
    manager = EvidenceMemoryManager()
    signal = SignalProvenanceNormalizer().normalize(
        _signal("S_reused", "Stable signal content.", root="root-stable"),
        run_id="run_memory",
    )
    decision = manager.classify(
        EvidenceMemorySnapshot(),
        signal,
        likelihoods={"A": LikelihoodBand.MODERATELY_CONFIRMING},
        base_effective_weight=0.4,
    )
    event = EvidenceEvent(
        id="E_reused_1",
        derived_from_signal=signal.id,
        target_hypotheses=["A"],
        evidence_type=EvidenceType.SUPPORTING,
        content=signal.raw_content,
        likelihoods={"A": LikelihoodBand.MODERATELY_CONFIRMING},
    )
    snapshot = manager.commit(
        EvidenceMemorySnapshot(),
        signal=signal,
        event=event,
        decision=decision,
    )
    return manager, snapshot, signal, decision


@pytest.mark.parametrize("conflict", ["source", "content", "root", "group"])
def test_signal_id_lineage_conflict_fails_before_commit(conflict):
    manager, snapshot, signal, decision = _committed_signal_identity()
    provenance_updates = {}
    conflicting_decision = decision
    if conflict == "source":
        provenance_updates["source_identity"] = "source.example/other"
    elif conflict == "content":
        provenance_updates["canonical_content_fingerprint"] = "sha256:" + "b" * 64
    elif conflict == "root":
        provenance_updates["derivation_root_id"] = "root-conflict"
    else:
        conflicting_decision = replace(
            decision,
            canonical_correlation_group="group-conflict",
        )
    conflicting_signal = signal.model_copy(
        update={
            "provenance": signal.provenance.model_copy(update=provenance_updates),
        }
    )
    event = EvidenceEvent(
        id=f"E_reused_conflict_{conflict}",
        derived_from_signal=signal.id,
        target_hypotheses=["A"],
        evidence_type=EvidenceType.SUPPORTING,
        content=signal.raw_content,
        likelihoods={"A": LikelihoodBand.MODERATELY_CONFIRMING},
    )

    with pytest.raises(ValueError, match="signal id lineage conflict"):
        manager.commit(
            snapshot,
            signal=conflicting_signal,
            event=event,
            decision=conflicting_decision,
        )

    assert snapshot.accepted_evidence_ids == ["E_reused_1"]
    assert len(snapshot.source_content_fingerprints) == 1
    assert len(snapshot.correlation_credit) == 1


def test_identical_signal_id_reuse_is_idempotent():
    manager, snapshot, signal, decision = _committed_signal_identity()
    event = EvidenceEvent(
        id="E_reused_1",
        derived_from_signal=signal.id,
        target_hypotheses=["A"],
        evidence_type=EvidenceType.SUPPORTING,
        content=signal.raw_content,
        likelihoods={"A": LikelihoodBand.MODERATELY_CONFIRMING},
    )

    recommitted = manager.commit(
        snapshot,
        signal=signal,
        event=event,
        decision=decision,
    )

    assert recommitted == snapshot


def test_neutral_event_still_correlates_later_model_reasoning_from_same_session():
    manager = EvidenceMemoryManager()
    normalizer = SignalProvenanceNormalizer()
    first = normalizer.normalize(
        ExternalSignal(
            id="S_model_neutral_1",
            cycle_id="pending",
            signal_kind=SignalKind.ACTIVE,
            source_type="model_probe_gateway",
            source="model_gateway:scripted",
            raw_content="The first model observation is neutral.",
        ),
        run_id="run_memory",
    )
    first_decision = manager.classify(
        EvidenceMemorySnapshot(),
        first,
        likelihoods={"A": LikelihoodBand.NEUTRAL},
        base_effective_weight=0.4,
    )
    first_event = EvidenceEvent(
        id="E_model_neutral_1",
        derived_from_signal=first.id,
        target_hypotheses=["A"],
        evidence_type=EvidenceType.NEUTRAL,
        content=first.raw_content,
        likelihoods={"A": LikelihoodBand.NEUTRAL},
    )
    memory = manager.commit(
        EvidenceMemorySnapshot(),
        signal=first,
        event=first_event,
        decision=first_decision,
    )
    second = normalizer.normalize(
        first.model_copy(
            update={
                "id": "S_model_neutral_2",
                "raw_content": "A later, distinct model observation.",
                "provenance": None,
            }
        ),
        run_id="run_memory",
    )

    decision = manager.classify(memory, second)

    assert decision.correlation_status == "correlated_novel"


def test_same_source_cannot_become_independent_by_changing_declared_group():
    manager = EvidenceMemoryManager()
    normalizer = SignalProvenanceNormalizer()
    first = normalizer.normalize(
        _signal("S_source_1", "First source observation.", root="root-1"),
        run_id="run_memory",
    )
    first_decision = manager.classify(EvidenceMemorySnapshot(), first)
    first_event = EvidenceEvent(
        id="E_source_1",
        derived_from_signal=first.id,
        target_hypotheses=["A"],
        evidence_type=EvidenceType.NEUTRAL,
        content=first.raw_content,
        likelihoods={"A": LikelihoodBand.NEUTRAL},
    )
    memory = manager.commit(
        EvidenceMemorySnapshot(),
        signal=first,
        event=first_event,
        decision=first_decision,
    )
    changed_group = _signal(
        "S_source_2",
        "Second source observation.",
        root="root-2",
    )
    changed_group = changed_group.model_copy(
        update={
            "provenance": changed_group.provenance.model_copy(
                update={"correlation_group": "declared-as-new"}
            )
        }
    )
    second = normalizer.normalize(changed_group, run_id="run_memory")

    decision = manager.classify(memory, second)

    assert decision.correlation_status == "correlated_novel"


def test_same_source_changing_declared_groups_shares_cumulative_credit():
    manager = EvidenceMemoryManager(
        CorrelationCreditPolicy(max_cumulative_effective_weight_per_direction=0.5)
    )
    normalizer = SignalProvenanceNormalizer()
    memory = EvidenceMemorySnapshot()
    decisions = []

    for index, declared_group in enumerate(
        ["canonical-source-group", "fresh-group-2", "fresh-group-3"],
        start=1,
    ):
        raw_signal = _signal(
            f"S_changing_group_{index}",
            f"Distinct source observation {index}.",
            root=f"root-{index}",
        )
        raw_signal = raw_signal.model_copy(
            update={
                "provenance": raw_signal.provenance.model_copy(
                    update={"correlation_group": declared_group}
                )
            }
        )
        signal = normalizer.normalize(raw_signal, run_id="run_memory")
        decision = manager.classify(
            memory,
            signal,
            likelihoods={"A": LikelihoodBand.MODERATELY_CONFIRMING},
            base_effective_weight=0.3,
        )
        event = EvidenceEvent(
            id=f"E_changing_group_{index}",
            derived_from_signal=signal.id,
            target_hypotheses=["A"],
            evidence_type=EvidenceType.SUPPORTING,
            content=signal.raw_content,
            likelihoods={"A": LikelihoodBand.MODERATELY_CONFIRMING},
            correlation_status=decision.correlation_status,
            effective_update_weight=decision.effective_update_weight,
            discard_reason=decision.discard_reason,
        )
        memory = manager.commit(memory, signal=signal, event=event, decision=decision)
        decisions.append(decision)

    assert [decision.effective_update_weight for decision in decisions] == [
        0.3,
        0.2,
        0.0,
    ]
    assert decisions[-1].discard_reason == "correlation_credit_saturated"
    assert memory.correlation_credit == {
        "canonical-source-group|A|confirming": 0.5
    }
    assert {
        json.loads(identity)[2]
        for identity in memory.source_content_fingerprints.values()
    } == {"canonical-source-group"}


def test_supplied_group_replay_is_idempotent_while_credit_stays_canonical():
    manager = EvidenceMemoryManager()
    normalizer = SignalProvenanceNormalizer()
    memory = EvidenceMemorySnapshot()

    signals = []
    events = []
    for index, supplied_group in enumerate(
        ["canonical-source-group", "caller-supplied-group"],
        start=1,
    ):
        raw_signal = _signal(
            f"S_group_identity_{index}",
            f"Distinct group identity observation {index}.",
            root=f"root-group-identity-{index}",
        )
        raw_signal = raw_signal.model_copy(
            update={
                "provenance": raw_signal.provenance.model_copy(
                    update={"correlation_group": supplied_group}
                )
            }
        )
        signal = normalizer.normalize(raw_signal, run_id="run_memory")
        decision = manager.classify(
            memory,
            signal,
            likelihoods={"A": LikelihoodBand.MODERATELY_CONFIRMING},
            base_effective_weight=0.25,
        )
        event = EvidenceEvent(
            id=f"E_group_identity_{index}",
            derived_from_signal=signal.id,
            target_hypotheses=["A"],
            evidence_type=EvidenceType.SUPPORTING,
            content=signal.raw_content,
            likelihoods={"A": LikelihoodBand.MODERATELY_CONFIRMING},
            correlation_status=decision.correlation_status,
            effective_update_weight=decision.effective_update_weight,
        )
        memory = manager.commit(memory, signal=signal, event=event, decision=decision)
        signals.append(signal)
        events.append(event)

    second_identity = json.loads(
        memory.source_content_fingerprints[signals[1].id]
    )
    recommitted = manager.commit(
        memory,
        signal=signals[1],
        event=events[1],
        decision=manager.classify(memory, signals[1]),
    )

    assert memory.memory_version == 2
    assert second_identity[2:] == [
        "canonical-source-group",
        "caller-supplied-group",
    ]
    assert memory.correlation_credit == {
        "canonical-source-group|A|confirming": 0.5
    }
    assert recommitted == memory


@pytest.mark.parametrize(
    ("source_identities", "derivation_roots", "expected_message"),
    [
        (
            {"S1": "shared-source", "S2": "shared-source"},
            {"S1": "root-1", "S2": "root-2"},
            "source identity has conflicting canonical correlation groups",
        ),
        (
            {"S1": "source-1", "S2": "source-2"},
            {"S1": "shared-root", "S2": "shared-root"},
            "derivation root has conflicting canonical correlation groups",
        ),
    ],
)
def test_snapshot_rejects_conflicting_canonical_lineage_groups(
    source_identities,
    derivation_roots,
    expected_message,
):
    fingerprints = {
        "S1": "sha256:" + "a" * 64,
        "S2": "sha256:" + "b" * 64,
    }
    identities = {
        signal_id: json.dumps(
            [source_identities[signal_id], fingerprint, f"group-{index}"],
            separators=(",", ":"),
        )
        for index, (signal_id, fingerprint) in enumerate(
            fingerprints.items(),
            start=1,
        )
    }

    with pytest.raises(ValueError, match=expected_message):
        EvidenceMemorySnapshot(
            content_fingerprints=fingerprints,
            source_content_fingerprints=identities,
            derivation_roots=derivation_roots,
        )


def test_canonical_group_is_map_order_independent_and_stays_saturated():
    manager = EvidenceMemoryManager(
        CorrelationCreditPolicy(max_cumulative_effective_weight_per_direction=0.5)
    )
    fingerprints = {
        "S_unrelated": "sha256:" + "a" * 64,
        "S_canonical": "sha256:" + "b" * 64,
    }
    identities = {
        "S_unrelated": '["other-source","sha256:' + "a" * 64 + '","other-group"]',
        "S_canonical": '["source.example/report","sha256:'
        + "b" * 64
        + '","canonical-source-group"]',
    }
    roots = {"S_unrelated": "other-root", "S_canonical": "root-canonical"}
    snapshot = EvidenceMemorySnapshot(
        content_fingerprints=fingerprints,
        source_content_fingerprints=identities,
        derivation_roots=roots,
        correlation_credit={"canonical-source-group|A|confirming": 0.5},
    )
    reversed_snapshot = EvidenceMemorySnapshot(
        content_fingerprints=dict(reversed(fingerprints.items())),
        source_content_fingerprints=dict(reversed(identities.items())),
        derivation_roots=dict(reversed(roots.items())),
        correlation_credit={"canonical-source-group|A|confirming": 0.5},
    )
    raw_signal = _signal(
        "S_reordered",
        "A new observation from the canonical source.",
        root="root-new",
    )
    raw_signal = raw_signal.model_copy(
        update={
            "provenance": raw_signal.provenance.model_copy(
                update={"correlation_group": "caller-supplied-fresh-group"}
            )
        }
    )
    signal = SignalProvenanceNormalizer().normalize(raw_signal, run_id="run_memory")

    decisions = [
        manager.classify(
            candidate,
            signal,
            likelihoods={"A": LikelihoodBand.MODERATELY_CONFIRMING},
            base_effective_weight=0.3,
        )
        for candidate in (snapshot, reversed_snapshot)
    ]

    assert [item.canonical_correlation_group for item in decisions] == [
        "canonical-source-group",
        "canonical-source-group",
    ]
    assert [item.effective_update_weight for item in decisions] == [0.0, 0.0]
    assert [item.discard_reason for item in decisions] == [
        "correlation_credit_saturated",
        "correlation_credit_saturated",
    ]


def test_unknown_parent_is_ledger_visible_but_receives_zero_independent_credit():
    manager = EvidenceMemoryManager()
    signal = SignalProvenanceNormalizer().normalize(
        ExternalSignal(
            id="S_unknown_parent",
            cycle_id="pending",
            signal_kind=SignalKind.PASSIVE,
            source_type="derived_summary",
            source="summary-worker",
            raw_content="A summary whose parent is not in local memory.",
            provenance=SignalProvenance(
                epistemic_origin=EpistemicOrigin.DERIVED_SUMMARY,
                source_identity="summary-worker",
                parent_signal_ids=["S_not_locally_known"],
                derivation_root_id="root-declared-by-summary",
                correlation_group="summary-group",
                canonical_content_fingerprint="replace-me",
            ),
        ),
        run_id="run_memory",
    )

    decision = manager.classify(
        EvidenceMemorySnapshot(),
        signal,
        likelihoods={"A": LikelihoodBand.MODERATELY_CONFIRMING},
        base_effective_weight=0.4,
    )
    event = EvidenceEvent(
        id="E_unknown_parent",
        derived_from_signal=signal.id,
        target_hypotheses=["A"],
        evidence_type=EvidenceType.SUPPORTING,
        content=signal.raw_content,
        likelihoods={"A": LikelihoodBand.MODERATELY_CONFIRMING},
        correlation_status=decision.correlation_status,
        effective_update_weight=decision.effective_update_weight,
    )
    memory = manager.commit(
        EvidenceMemorySnapshot(),
        signal=signal,
        event=event,
        decision=decision,
    )

    assert decision.correlation_status == "correlated_restatement"
    assert decision.effective_update_weight == 0.0
    assert memory.accepted_evidence_ids == ["E_unknown_parent"]
    assert memory.correlation_credit == {}


def test_known_derived_parent_cannot_change_derivation_root():
    manager = EvidenceMemoryManager()
    normalizer = SignalProvenanceNormalizer()
    parent = normalizer.normalize(
        _signal("S_parent", "Parent observation.", root="root-parent"),
        run_id="run_memory",
    )
    parent_decision = manager.classify(EvidenceMemorySnapshot(), parent)
    parent_event = EvidenceEvent(
        id="E_parent",
        derived_from_signal=parent.id,
        target_hypotheses=["A"],
        evidence_type=EvidenceType.NEUTRAL,
        content=parent.raw_content,
        likelihoods={"A": LikelihoodBand.NEUTRAL},
    )
    memory = manager.commit(
        EvidenceMemorySnapshot(),
        signal=parent,
        event=parent_event,
        decision=parent_decision,
    )
    derived = ExternalSignal(
        id="S_bad_summary",
        cycle_id="pending",
        signal_kind=SignalKind.PASSIVE,
        source_type="derived_summary",
        source="summary-worker",
        raw_content="A derived restatement.",
        provenance=SignalProvenance(
            epistemic_origin=EpistemicOrigin.DERIVED_SUMMARY,
            source_identity="summary-worker",
            parent_signal_ids=["S_parent"],
            derivation_root_id="root-changed",
            correlation_group="summary-group",
            canonical_content_fingerprint="replace-me",
        ),
    )
    derived = normalizer.normalize(derived, run_id="run_memory")

    with pytest.raises(ValueError, match="preserve parent derivation root"):
        manager.classify(memory, derived)


def test_prejudgment_classification_reports_existing_group_credit():
    manager = EvidenceMemoryManager()
    normalizer = SignalProvenanceNormalizer()
    first = normalizer.normalize(
        _signal("S_precredit_1", "First credited observation.", root="root-1"),
        run_id="run_memory",
    )
    first_decision = manager.classify(
        EvidenceMemorySnapshot(),
        first,
        likelihoods={"A": LikelihoodBand.MODERATELY_CONFIRMING},
        base_effective_weight=0.4,
    )
    first_event = EvidenceEvent(
        id="E_precredit_1",
        derived_from_signal=first.id,
        target_hypotheses=["A"],
        evidence_type=EvidenceType.SUPPORTING,
        content=first.raw_content,
        likelihoods={"A": LikelihoodBand.MODERATELY_CONFIRMING},
    )
    memory = manager.commit(
        EvidenceMemorySnapshot(),
        signal=first,
        event=first_event,
        decision=first_decision,
    )
    second = normalizer.normalize(
        _signal("S_precredit_2", "Second credited observation.", root="root-2"),
        run_id="run_memory",
    )

    decision = manager.classify(memory, second)

    assert decision.remaining_credit == {
        f"{second.provenance.correlation_group}|A|confirming": 0.6
    }


def test_epistemic_origin_caps_quality_even_when_source_type_looks_external():
    gateway = CountingGateway()
    signal = ExternalSignal(
        id="S_spoofed_model",
        cycle_id="pending",
        signal_kind=SignalKind.ACTIVE,
        source_type="benchmark_stream",
        source="provider/model-a",
        raw_content="A model-generated conclusion favors A.",
        initial_target_hypotheses=["A", "B"],
        provenance=SignalProvenance(
            epistemic_origin=EpistemicOrigin.MODEL_REASONING,
            source_identity="provider/model-a",
            provider_model_or_tool_identity="provider/model-a",
            session_id="session-1",
            derivation_root_id="root-spoofed",
            correlation_group="caller-group",
            canonical_content_fingerprint="replace-me",
        ),
    )

    result = EvidenceIntegrationGate(model_gateway=gateway).integrate(
        cycle=_cycle(1),
        belief_state=_state(),
        probe_set=_probe_set(1),
        signals=[signal],
    )

    event = result.evidence_events[0]
    assert event.reliability == 0.55
    assert event.independence == 0.35
    assert event.novelty == 0.55
    assert event.verifiability == 0.3


def test_model_origin_caps_provider_labeled_source_claim_and_overrides():
    class SourceClaimGateway(CountingGateway):
        def complete_structured(self, request):
            payload = super().complete_structured(request)
            payload["evidence_type"] = "source_claim"
            payload["quality_overrides"] = {
                "reliability": 1.0,
                "independence": 1.0,
                "novelty": 1.0,
                "verifiability": 1.0,
            }
            return payload

    result = EvidenceIntegrationGate(model_gateway=SourceClaimGateway()).integrate(
        cycle=_cycle(1),
        belief_state=_state(),
        probe_set=_probe_set(1),
        signals=[
            ExternalSignal(
                id="S_model_source_claim",
                cycle_id="pending",
                signal_kind=SignalKind.ACTIVE,
                source_type="benchmark_stream",
                source="provider/model-a",
                raw_content="A model labels its own output as a source claim.",
                initial_target_hypotheses=["A", "B"],
                provenance=SignalProvenance(
                    epistemic_origin=EpistemicOrigin.MODEL_REASONING,
                    source_identity="provider/model-a",
                    provider_model_or_tool_identity="provider/model-a",
                    session_id="session-1",
                    derivation_root_id="root-model-source-claim",
                    correlation_group="caller-group",
                    canonical_content_fingerprint="replace-me",
                ),
            )
        ],
    )

    event = result.evidence_events[0]
    assert event.evidence_type == EvidenceType.SOURCE_CLAIM
    assert event.reliability == 0.5
    assert event.independence == 0.35
    assert event.novelty == 0.55
    assert event.verifiability == 0.3


def test_native_judgment_request_contains_full_semantics_provenance_and_memory():
    gateway = CountingGateway()
    gate = EvidenceIntegrationGate(model_gateway=gateway)

    result = gate.integrate(
        cycle=_cycle(1),
        belief_state=_state(),
        probe_set=_probe_set(1),
        signals=[_signal("S_context", "The audited value supports A.")],
    )

    request = gateway.requests[0]
    hypothesis = request.input["hypotheses"][0]
    assert request.prompt_version == "v0.2"
    assert request.schema_version == "v0.2"
    assert hypothesis.keys() >= {
        "id",
        "statement",
        "type",
        "scope",
        "posterior",
        "predictions",
        "falsifiers",
        "rivals",
    }
    assert request.input["frame"] == {
        "competition": "exclusive",
        "coverage": "exhaustive",
        "frame_version": 1,
    }
    assert request.input["provenance"]["derivation_root_id"] == "root-source-1"
    assert request.input["memory"]["correlation_status"] == "novel"
    assert request.input["probes"] == []
    assert result.evidence_events[0].schema_version == "v0.2"
    assert result.evidence_events[0].frame_fit.value == "explained_by_named"


def test_native_v02_route_rejects_exact_legacy_four_field_judgment():
    class LegacyFourFieldGateway(CountingGateway):
        def complete_structured(self, request):
            self.requests.append(request)
            return {
                "evidence_type": "supporting",
                "likelihoods": {
                    "A": "moderately_confirming",
                    "B": "moderately_disconfirming",
                },
                "interpretation": "Legacy-shaped response on a native request.",
                "quality_overrides": {},
            }

    gateway = LegacyFourFieldGateway()

    result = EvidenceIntegrationGate(model_gateway=gateway).integrate(
        cycle=_cycle(1),
        belief_state=_state(),
        probe_set=_probe_set(1),
        signals=[_signal("S_native_legacy_shape", "A native signal.")],
    )

    event = result.evidence_events[0]
    assert event.discard_reason.startswith("schema_violation:")
    assert "missing field" in event.discard_reason
    assert gateway.requests[0].schema_version == "v0.2"
    assert gateway.requests[0].metadata["judgment_route"] == "native_v0.2"
    assert event.model_trace["metadata"]["judgment_route"] == "native_v0.2"


def test_explicit_migration_route_completes_exact_legacy_shape_auditably():
    class LegacyFourFieldGateway(CountingGateway):
        def complete_structured(self, request):
            self.requests.append(request)
            return {
                "evidence_type": "supporting",
                "likelihoods": {
                    "A": "moderately_confirming",
                    "B": "moderately_disconfirming",
                },
                "interpretation": "Reviewed legacy response shape.",
                "quality_overrides": {},
            }

    gateway = LegacyFourFieldGateway()
    state = _state()
    state = state.model_copy(
        update={
            "task_frame": state.task_frame.model_copy(
                update={"framing_method": FramingMethod.LEGACY_MIGRATION}
            )
        }
    )

    result = EvidenceIntegrationGate(model_gateway=gateway).integrate(
        cycle=_cycle(1),
        belief_state=state,
        probe_set=_probe_set(1),
        signals=[_signal("S_explicit_legacy", "A reviewed migrated signal.")],
    )

    event = result.evidence_events[0]
    assert event.discard_reason is None
    assert gateway.requests[0].schema_version == "v0.1"
    assert gateway.requests[0].metadata["judgment_route"] == "legacy_v0.1_migration"
    assert event.model_trace["metadata"]["judgment_route"] == (
        "legacy_v0.1_migration"
    )


def test_v02_evidence_event_requires_native_provenance_and_memory_fields():
    with pytest.raises(ValueError, match="v0.2 evidence event requires"):
        EvidenceEvent(
            schema_version="v0.2",
            id="E_invalid_native",
            derived_from_signal="S_invalid_native",
            target_hypotheses=["A"],
            evidence_type=EvidenceType.NEUTRAL,
            content="Missing native identity fields.",
            likelihoods={"A": LikelihoodBand.NEUTRAL},
        )


def test_v02_evidence_event_rejects_incoherent_frame_fit():
    with pytest.raises(ValueError, match="supports_unresolved"):
        EvidenceEvent(
            schema_version="v0.2",
            id="E_bad_frame_fit",
            derived_from_signal="S_bad_frame_fit",
            epistemic_origin=EpistemicOrigin.TOOL_RESULT,
            derivation_root_id="root-bad",
            target_hypotheses=["A"],
            evidence_type=EvidenceType.ANOMALY,
            content="Named candidates miss this result.",
            likelihoods={"A": LikelihoodBand.MODERATELY_DISCONFIRMING},
            unresolved_likelihood=LikelihoodBand.NEUTRAL,
            frame_fit="supports_unresolved",
            correlation_status="novel",
            effective_update_weight=0.4,
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"content_fingerprints": {"S1": "sk-provider-secret-123"}},
        {"source_content_fingerprints": {"api_key": "sha256:abc"}},
        {"correlation_credit": {"group|H_other|confirming": 0.2}},
        {"correlation_credit": {"group|A|confirming": float("nan")}},
    ],
)
def test_evidence_memory_snapshot_rejects_secret_or_invalid_native_data(payload):
    with pytest.raises(ValueError):
        EvidenceMemorySnapshot(**payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"content_fingerprints": {"S1": "sha256:" + "a" * 64}},
        {
            "content_fingerprints": {"S1": "sha256:" + "a" * 64},
            "source_content_fingerprints": {
                "S2": '["source","sha256:' + "a" * 64 + '","group"]'
            },
            "derivation_roots": {"S1": "root-1"},
        },
    ],
)
def test_evidence_memory_snapshot_requires_coherent_identity_map_keys(payload):
    with pytest.raises(ValueError, match="identity map keys"):
        EvidenceMemorySnapshot(**payload)


@pytest.mark.parametrize(
    "identity",
    [
        "not-json",
        '{"source":"source","fingerprint":"sha256:abc","group":"group","extra":1}',
        '["source","sha256:abc","group","extra"]',
        '["source","sha256:abc","group"]',
        ' ["source","sha256:' + "a" * 64 + '","group"] ',
    ],
)
def test_evidence_memory_snapshot_requires_exact_canonical_source_identity(identity):
    fingerprint = "sha256:" + "a" * 64
    with pytest.raises(ValueError, match="source_content_fingerprints"):
        EvidenceMemorySnapshot(
            content_fingerprints={"S1": fingerprint},
            source_content_fingerprints={"S1": identity},
            derivation_roots={"S1": "root-1"},
        )


@pytest.mark.parametrize(
    "key",
    [
        "group|A|confirming|extra",
        "group||confirming",
        "|A|confirming",
        "group|A|neutral",
        "group|frame:0:unresolved|confirming",
        "group|frame:one:unresolved|confirming",
        "group|frame:1:unresolved|disconfirming|extra",
        "group|H_other|confirming",
    ],
)
def test_evidence_memory_snapshot_rejects_malformed_credit_key_grammar(key):
    with pytest.raises(ValueError, match="correlation credit"):
        EvidenceMemorySnapshot(correlation_credit={key: 0.2})


def test_native_belief_state_rejects_credit_for_unknown_hypothesis_subject():
    state = _state()
    payload = state.model_dump(mode="python")
    payload["evidence_memory"]["correlation_credit"] = {
        "group|UNKNOWN|confirming": 0.2
    }

    with pytest.raises(ValueError, match="unknown hypothesis"):
        type(state).model_validate(payload)
