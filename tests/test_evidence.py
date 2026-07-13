import pytest

from bayesprobe.evidence import EvidenceIntegrationGate, EvidenceIntegrationResult
from bayesprobe.evidence_memory import EvidenceMemoryManager
from bayesprobe.evidence_roots import EvidenceRootReconciler
from bayesprobe.initialization import BayesProbeInitializer, InitializeRunInput
from bayesprobe.schemas import (
    AnswerChoice,
    BoundaryStatus,
    CycleRecord,
    CycleSignalShape,
    EpistemicOrigin,
    EvidenceContributionMode,
    ExternalSignal,
    ProbeDesign,
    ProbePurpose,
    ProbeSet,
    SignalInboxStatus,
    SignalKind,
    SignalProvenance,
)


class RecordingEvidenceGateway:
    adapter_kind = "recording-evidence"

    def __init__(self, bands=None):
        self.requests = []
        self._bands = list(bands or ["moderately_confirming"])

    def complete_structured(self, request):
        self.requests.append(request)
        band = self._bands[min(len(self.requests) - 1, len(self._bands) - 1)]
        return {
            "evidence_type": "supporting",
            "likelihoods": {
                "A": band,
                "B": "moderately_disconfirming",
            },
            "unresolved_likelihood": None,
            "frame_fit": "explained_by_named",
            "unexplained_observation": None,
            "interpretation": "The signal discriminates between A and B.",
            "quality_overrides": {},
        }


def _state(run_id="run_native_evidence"):
    return BayesProbeInitializer().initialize(
        InitializeRunInput(
            run_id=run_id,
            problem="Which option is supported?",
            task_context="Use the audited observations.",
            answer_choices=[
                AnswerChoice(label="A", text="Option A"),
                AnswerChoice(label="B", text="Option B"),
            ],
        )
    ).belief_state


def _cycle(index):
    return CycleRecord(
        cycle_id=f"cycle_{index}",
        run_id="run_native_evidence",
        cycle_index=index,
        signal_shape=CycleSignalShape.ACTIVE_ONLY,
        boundary_status=BoundaryStatus.CLOSED,
    )


def _probe_set(index, *, purpose=ProbePurpose.HYPOTHESIS_FALSIFICATION):
    probe = ProbeDesign(
        id=f"P{index}",
        cycle_id=f"cycle_{index}",
        target_hypotheses=["A", "B"],
        inquiry_goal="Try to falsify option A.",
        method="model_reasoning",
        purpose=purpose,
        expected_observation="An observation that conflicts with option A.",
        support_condition={"A": "Designer-authored support condition."},
        weaken_condition={"A": "Designer-authored weaken condition."},
        reframe_condition={"A": "Designer-authored reframe condition."},
    )
    return ProbeSet(
        probe_set_id=f"ps_{index}",
        cycle_id=f"cycle_{index}",
        probes=[probe],
        selection_reason="Frozen evidence test probe set.",
    )


def _model_signal(signal_id, content, *, probe_id="P1", kind=SignalKind.ACTIVE,
                  inbox_status=SignalInboxStatus.ACCEPTED):
    return ExternalSignal(
        id=signal_id,
        cycle_id="pending",
        signal_kind=kind,
        source_type="model_probe_gateway",
        source="model_gateway:recording-evidence",
        raw_content=content,
        generated_by_probe=probe_id,
        inbox_status=inbox_status,
        initial_target_hypotheses=["A", "B"],
    )


def _recursive_keys(value):
    if isinstance(value, dict):
        keys = set(value)
        for item in value.values():
            keys.update(_recursive_keys(item))
        return keys
    if isinstance(value, (list, tuple)):
        keys = set()
        for item in value:
            keys.update(_recursive_keys(item))
        return keys
    return set()


def _state_after(state, result):
    return state.model_copy(
        update={
            "evidence_memory": result.evidence_memory,
            "ledger_refs": {
                **state.ledger_refs,
                "evidence_events": [
                    *state.ledger_refs.get("evidence_events", []),
                    *(event.id for event in result.evidence_events),
                ],
            },
        }
    )


def test_evidence_judge_request_is_blind_to_belief_and_credit():
    gateway = RecordingEvidenceGateway()
    result = EvidenceIntegrationGate(model_gateway=gateway).integrate(
        cycle=_cycle(1),
        belief_state=_state(),
        probe_set=_probe_set(1),
        signals=[_model_signal("S_blind", "An audited model observation.")],
    )

    request = gateway.requests[-1]
    forbidden = {
        "prior",
        "posterior",
        "current_best_hypothesis",
        "correlation_credit",
        "remaining_credit",
        "support_condition",
        "weaken_condition",
        "reframe_condition",
    }
    assert forbidden.isdisjoint(_recursive_keys(request.input))
    assert request.metadata["belief_context_policy"] == "blind_no_scores_v1"
    assert result.evidence_events[0].model_trace["metadata"][
        "belief_context_policy"
    ] == "blind_no_scores_v1"


def test_native_judge_request_uses_explicit_blind_allowlist():
    gateway = RecordingEvidenceGateway()
    EvidenceIntegrationGate(model_gateway=gateway).integrate(
        cycle=_cycle(1),
        belief_state=_state(),
        probe_set=_probe_set(1),
        signals=[_model_signal("S_allowlist", "An allowlisted observation.")],
    )

    request_input = gateway.requests[-1].input
    assert set(request_input) == {
        "task_context",
        "hypotheses",
        "signal",
        "provenance",
        "matched_probe",
        "target_hypotheses",
    }
    assert set(request_input["task_context"]) == {"problem", "task_context"}
    assert all(
        set(hypothesis)
        == {
            "id",
            "statement",
            "type",
            "scope",
            "predictions",
            "falsifiers",
            "rivals",
        }
        for hypothesis in request_input["hypotheses"]
    )
    assert set(request_input["signal"]) == {
        "id",
        "cycle_id",
        "signal_kind",
        "source_type",
        "source",
        "raw_content",
        "generated_by_probe",
        "inbox_status",
        "initial_target_hypotheses",
    }
    assert set(request_input["provenance"]) == {
        "epistemic_origin",
        "source_identity",
        "provider_model_or_tool_identity",
        "session_id",
        "parent_signal_ids",
        "derivation_root_id",
        "correlation_group",
        "supplied_correlation_group",
        "canonical_content_fingerprint",
        "citations",
        "artifact_refs",
        "environment_state_id",
    }
    assert set(request_input["matched_probe"]) == {
        "id",
        "purpose",
        "target_hypotheses",
        "inquiry_goal",
        "method",
        "expected_observation",
    }


def test_integration_result_has_independent_default_v3_outputs():
    first = EvidenceIntegrationResult(evidence_events=[], probe_candidates=[])
    second = EvidenceIntegrationResult(evidence_events=[], probe_candidates=[])

    assert first.contribution_deltas == []
    assert second.contribution_deltas == []
    assert first.contribution_deltas is not second.contribution_deltas
    assert first.epistemic_progress.model_dump() == {
        "new_root_count": 0,
        "revised_root_count": 0,
        "retracted_root_count": 0,
        "no_change_count": 0,
        "max_absolute_contribution_delta": 0.0,
        "falsification_probe_executed": False,
    }


def test_two_model_outputs_from_one_run_revise_one_persisted_root():
    gateway = RecordingEvidenceGateway(
        ["moderately_confirming", "strongly_confirming"]
    )
    gate = EvidenceIntegrationGate(model_gateway=gateway)
    state = _state()
    first = gate.integrate(
        cycle=_cycle(1),
        belief_state=state,
        probe_set=_probe_set(1),
        signals=[_model_signal("S_model_1", "First model observation.")],
    )
    second = gate.integrate(
        cycle=_cycle(2),
        belief_state=_state_after(state, first),
        probe_set=_probe_set(2),
        signals=[
            _model_signal(
                "S_model_2",
                "Second model observation.",
                probe_id="P2",
            )
        ],
    )

    assert len(first.evidence_memory.root_contributions) == 1
    assert len(second.evidence_memory.root_contributions) == 1
    assert first.contribution_deltas[0].mode is EvidenceContributionMode.NEW_ROOT
    assert second.contribution_deltas[0].mode in {
        EvidenceContributionMode.REVISE_ROOT,
        EvidenceContributionMode.NO_CHANGE,
    }
    assert all(event.effective_update_weight is None for event in first.evidence_events)
    assert all(event.effective_update_weight is None for event in second.evidence_events)


def test_native_v3_never_calls_legacy_credit_allocation(monkeypatch):
    def fail_credit_allocation(*args, **kwargs):
        raise AssertionError("native v3 called legacy credit allocation")

    monkeypatch.setattr(
        EvidenceMemoryManager,
        "_commit_correlation_credit",
        fail_credit_allocation,
    )

    result = EvidenceIntegrationGate(
        model_gateway=RecordingEvidenceGateway()
    ).integrate(
        cycle=_cycle(1),
        belief_state=_state(),
        probe_set=_probe_set(1),
        signals=[_model_signal("S_no_credit", "A native observation.")],
    )

    assert result.evidence_memory.correlation_credit == {}
    assert result.evidence_events[0].effective_update_weight is None


def _lineage_signal(
    signal_id,
    *,
    origin,
    source,
    root,
    group,
    parent_ids=None,
):
    return ExternalSignal(
        id=signal_id,
        cycle_id="pending",
        signal_kind=SignalKind.ACTIVE,
        source_type=(
            "derived_summary"
            if origin is EpistemicOrigin.DERIVED_SUMMARY
            else "retrieved_source"
        ),
        source=source,
        raw_content=f"Observation from {signal_id}.",
        initial_target_hypotheses=["A", "B"],
        provenance=SignalProvenance(
            epistemic_origin=origin,
            source_identity=source,
            parent_signal_ids=list(parent_ids or []),
            derivation_root_id=root,
            correlation_group=group,
            canonical_content_fingerprint="normalized-by-gate",
        ),
    )


def test_child_before_parent_inherits_parent_contribution_root():
    parent = _lineage_signal(
        "S_parent",
        origin=EpistemicOrigin.RETRIEVED_SOURCE,
        source="source-parent",
        root="shared-derivation-root",
        group="parent-correlation-group",
    )
    child = _lineage_signal(
        "S_child",
        origin=EpistemicOrigin.DERIVED_SUMMARY,
        source="summary-worker",
        root="shared-derivation-root",
        group="child-correlation-group",
        parent_ids=[parent.id],
    )

    gateway = RecordingEvidenceGateway()
    result = EvidenceIntegrationGate(model_gateway=gateway).integrate(
        cycle=_cycle(1),
        belief_state=_state(),
        probe_set=_probe_set(1),
        signals=[child, parent],
    )

    roots = {
        event.derived_from_signal: event.contribution_root_id
        for event in result.evidence_events
    }
    assert roots[child.id] == roots[parent.id]
    assert result.evidence_memory.signal_contribution_roots == roots
    assert len(result.contribution_deltas) == 1
    assert [request.input["signal"]["id"] for request in gateway.requests] == [
        parent.id,
        child.id,
    ]
    assert [signal.id for signal in result.normalized_signals] == [
        child.id,
        parent.id,
    ]


def test_native_cycle_reconciles_exactly_once(monkeypatch):
    calls = []
    original = EvidenceRootReconciler.reconcile_cycle

    def record_reconciliation(self, snapshot, evidence_events, falsification_probe_executed):
        calls.append([event.id for event in evidence_events])
        return original(
            self,
            snapshot,
            evidence_events,
            falsification_probe_executed,
        )

    monkeypatch.setattr(
        EvidenceRootReconciler,
        "reconcile_cycle",
        record_reconciliation,
    )

    result = EvidenceIntegrationGate(
        model_gateway=RecordingEvidenceGateway()
    ).integrate(
        cycle=_cycle(1),
        belief_state=_state(),
        probe_set=_probe_set(1),
        signals=[
            _model_signal("S_reconcile_1", "First reconciled observation."),
            _model_signal("S_reconcile_2", "Second reconciled observation."),
        ],
    )

    assert calls == [[event.id for event in result.evidence_events]]


def _assert_native_root_bound(events):
    assert events
    assert all(event.schema_version == "v0.2" for event in events)
    assert all(event.contribution_root_id is not None for event in events)
    assert all(event.effective_update_weight is None for event in events)


def test_native_replay_duplicate_and_projection_events_are_root_bound():
    gate = EvidenceIntegrationGate(model_gateway=RecordingEvidenceGateway())
    state = _state()
    cycle = _cycle(1)
    probe_set = _probe_set(1)
    original_signal = _model_signal("S_original", "A repeated observation.")
    first = gate.integrate(
        cycle=cycle,
        belief_state=state,
        probe_set=probe_set,
        signals=[original_signal],
    )
    replay = gate.integrate(
        cycle=cycle,
        belief_state=_state_after(state, first),
        probe_set=probe_set,
        signals=[original_signal],
    )
    duplicate = gate.integrate(
        cycle=_cycle(2),
        belief_state=_state_after(state, first),
        probe_set=_probe_set(2),
        signals=[
            _model_signal(
                "S_duplicate",
                "A repeated observation.",
                probe_id="P2",
            )
        ],
    )
    projection = gate.integrate(
        cycle=_cycle(2),
        belief_state=_state_after(state, first),
        probe_set=_probe_set(2),
        signals=[
            ExternalSignal(
                id="S_projection",
                cycle_id="pending",
                signal_kind=SignalKind.PASSIVE,
                source_type="external_agent_projection",
                source="agent-a",
                raw_content="Agent A cites source X as evidence for option A.",
                inbox_status=SignalInboxStatus.ACCEPTED,
                initial_target_hypotheses=["A", "B"],
            )
        ],
    )

    _assert_native_root_bound(replay.evidence_events)
    _assert_native_root_bound(duplicate.evidence_events)
    _assert_native_root_bound(projection.evidence_events)
    assert len(projection.evidence_events) == 2
    assert len(
        {event.contribution_root_id for event in projection.evidence_events}
    ) == 1


@pytest.mark.parametrize("case", ["missing", "cycle", "parentless", "conflict"])
def test_native_root_resolution_fails_closed_before_judgment(case):
    parent_a = _lineage_signal(
        "S_parent_a",
        origin=EpistemicOrigin.RETRIEVED_SOURCE,
        source="source-a",
        root="shared-root",
        group="group-a",
    )
    parent_b = _lineage_signal(
        "S_parent_b",
        origin=EpistemicOrigin.RETRIEVED_SOURCE,
        source="source-b",
        root="different-root",
        group="group-b",
    )
    if case == "missing":
        signals = [
            _lineage_signal(
                "S_missing_child",
                origin=EpistemicOrigin.DERIVED_SUMMARY,
                source="summary-worker",
                root="shared-root",
                group="summary-group",
                parent_ids=["S_absent"],
            )
        ]
    elif case == "cycle":
        signals = [
            _lineage_signal(
                "S_cycle_a",
                origin=EpistemicOrigin.DERIVED_SUMMARY,
                source="summary-a",
                root="shared-root",
                group="summary-a",
                parent_ids=["S_cycle_b"],
            ),
            _lineage_signal(
                "S_cycle_b",
                origin=EpistemicOrigin.DERIVED_SUMMARY,
                source="summary-b",
                root="shared-root",
                group="summary-b",
                parent_ids=["S_cycle_a"],
            ),
        ]
    elif case == "parentless":
        signals = [
            _lineage_signal(
                "S_parentless_summary",
                origin=EpistemicOrigin.DERIVED_SUMMARY,
                source="summary-worker",
                root="shared-root",
                group="summary-group",
            )
        ]
    else:
        signals = [
            _lineage_signal(
                "S_conflicting_child",
                origin=EpistemicOrigin.DERIVED_SUMMARY,
                source="summary-worker",
                root="shared-root",
                group="summary-group",
                parent_ids=[parent_a.id, parent_b.id],
            ),
            parent_b,
            parent_a,
        ]
    gateway = RecordingEvidenceGateway()

    with pytest.raises(ValueError):
        EvidenceIntegrationGate(model_gateway=gateway).integrate(
            cycle=_cycle(1),
            belief_state=_state(),
            probe_set=_probe_set(1),
            signals=signals,
        )

    assert gateway.requests == []


@pytest.mark.parametrize(
    ("kind", "inbox_status", "purpose", "expected"),
    [
        (
            SignalKind.ACTIVE,
            SignalInboxStatus.ACCEPTED,
            ProbePurpose.HYPOTHESIS_FALSIFICATION,
            True,
        ),
        (
            SignalKind.PASSIVE,
            SignalInboxStatus.ACCEPTED,
            ProbePurpose.HYPOTHESIS_FALSIFICATION,
            False,
        ),
        (
            SignalKind.ACTIVE,
            SignalInboxStatus.DEFERRED,
            ProbePurpose.HYPOTHESIS_FALSIFICATION,
            False,
        ),
        (
            SignalKind.ACTIVE,
            SignalInboxStatus.ACCEPTED,
            ProbePurpose.HYPOTHESIS_DISCRIMINATION,
            False,
        ),
    ],
)
def test_falsification_progress_comes_only_from_accepted_active_frozen_probe(
    kind,
    inbox_status,
    purpose,
    expected,
):
    result = EvidenceIntegrationGate(
        model_gateway=RecordingEvidenceGateway()
    ).integrate(
        cycle=_cycle(1),
        belief_state=_state(),
        probe_set=_probe_set(1, purpose=purpose),
        signals=[
            _model_signal(
                "S_falsification_progress",
                "A probe-generated observation.",
                kind=kind,
                inbox_status=inbox_status,
            )
        ],
    )

    assert result.epistemic_progress.falsification_probe_executed is expected
