import json
import math
import re
from dataclasses import replace

import pytest

import bayesprobe.evidence as evidence_module
import bayesprobe.evidence_memory as evidence_memory
from bayesprobe.evidence import EvidenceIntegrationGate
from bayesprobe.evidence_memory import (
    EvidenceMemoryManager,
    SignalProvenanceNormalizer,
    derive_deterministic_computation_root,
    resolve_signal_contribution_roots,
)
from bayesprobe.evidence_roots import (
    EvidenceRootReconciler,
    resolve_contribution_root_id,
)
from bayesprobe.initialization import BayesProbeInitializer, InitializeRunInput
from bayesprobe.kernel_config import CorrelationCreditPolicy
from bayesprobe.schemas import (
    AnswerChoice,
    BeliefState,
    CycleRecord,
    CycleSignalShape,
    EpistemicOrigin,
    EvidenceEvent,
    EvidenceMemorySnapshot,
    EvidenceRootContribution,
    EvidenceType,
    ExternalSignal,
    FramingMethod,
    FrameFit,
    LikelihoodBand,
    ProbeSet,
    SignalKind,
    SignalProvenance,
)
from bayesprobe.task_framing import migrate_legacy_belief_state


_MIGRATION_MARKERS = (
    "belief_state_v0.1_to_v0.2",
    "task_frame_v0.1_to_v0.2",
)
_NONLEGACY_FRAMING_METHODS = tuple(
    method
    for method in FramingMethod
    if method != FramingMethod.LEGACY_MIGRATION
)
_INVALID_MIGRATION_ENVELOPES = (
    "tag_only",
    "forged_recognized_marker",
    "transferred_receipt",
    "v01_belief_state",
    "v01_task_frame",
    "missing_trace",
    "fake_trace",
    "missing_frame_state",
    "missing_evidence_memory",
    "incoherent_frame_state",
)
_NFKC_SECRET_VALUE = (
    "\uff21\uff55\uff54\uff48\uff4f\uff52\uff49\uff5a\uff41\uff54"
    "\uff49\uff4f\uff4e\uff1a \uff22\uff45\uff41\uff52\uff45\uff52 "
    "provider-secret-value-123"
)
_NFKC_SENSITIVE_NAME = "\uff41\uff50\uff49\uff3f\uff4b\uff45\uff59"


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


class LegacyCountingGateway(CountingGateway):
    def complete_structured(self, request):
        self.requests.append(request)
        return {
            "evidence_type": "supporting",
            "likelihoods": {
                "A": "moderately_confirming",
                "B": "moderately_disconfirming",
            },
            "interpretation": "The source favors A over B.",
            "quality_overrides": {},
        }


class RecordingProvenanceNormalizer(SignalProvenanceNormalizer):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def normalize(self, signal, *, run_id):
        self.calls.append(signal.id)
        return super().normalize(signal, run_id=run_id)


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


def _migrated_state(marker: str) -> BeliefState:
    payload = _state().model_dump(mode="python")
    payload.update(
        {
            "schema_version": "v0.1",
            "frame_state": None,
            "evidence_memory": None,
        }
    )
    if marker == "belief_state_v0.1_to_v0.2":
        payload["task_frame"] = None
    else:
        payload["task_frame"]["schema_version"] = "v0.1"
        payload["task_frame"]["framing_method"] = FramingMethod.EXPLICIT
        payload["task_frame"]["framing_trace"] = {"schema_version": "v0.1"}
    legacy_state = BeliefState.model_validate(payload)

    migrated = migrate_legacy_belief_state(legacy_state)

    assert legacy_state.schema_version == "v0.1"
    assert migrated.task_frame.framing_trace["migration"] == marker
    return migrated


def _invalid_migration_envelope(kind: str) -> BeliefState:
    native = _state()
    migrated = _migrated_state("belief_state_v0.1_to_v0.2")
    if kind == "tag_only":
        return native.model_copy(
            update={
                "task_frame": native.task_frame.model_copy(
                    update={"framing_method": FramingMethod.LEGACY_MIGRATION}
                )
            }
        )
    if kind == "forged_recognized_marker":
        return native.model_copy(
            update={
                "task_frame": native.task_frame.model_copy(
                    update={
                        "framing_method": FramingMethod.LEGACY_MIGRATION,
                        "framing_trace": {
                            **native.task_frame.framing_trace,
                            "migration": "belief_state_v0.1_to_v0.2",
                        },
                    }
                )
            }
        )
    if kind == "transferred_receipt":
        forged_native = native.model_copy(
            update={
                "task_frame": native.task_frame.model_copy(
                    update={
                        "framing_method": FramingMethod.LEGACY_MIGRATION,
                        "framing_trace": {
                            "migration": "belief_state_v0.1_to_v0.2"
                        },
                    }
                )
            }
        )
        return migrated.model_copy(
            update={
                field_name: getattr(forged_native, field_name)
                for field_name in BeliefState.model_fields
            }
        )
    if kind == "v01_belief_state":
        return migrated.model_copy(update={"schema_version": "v0.1"})
    if kind == "v01_task_frame":
        return migrated.model_copy(
            update={
                "task_frame": migrated.task_frame.model_copy(
                    update={"schema_version": "v0.1"}
                )
            }
        )
    if kind == "missing_trace":
        return migrated.model_copy(
            update={
                "task_frame": migrated.task_frame.model_copy(
                    update={"framing_trace": {}}
                )
            }
        )
    if kind == "fake_trace":
        return migrated.model_copy(
            update={
                "task_frame": migrated.task_frame.model_copy(
                    update={"framing_trace": {"migration": "caller_asserted"}}
                )
            }
        )
    if kind == "missing_frame_state":
        return migrated.model_copy(update={"frame_state": None})
    if kind == "missing_evidence_memory":
        return migrated.model_copy(update={"evidence_memory": None})
    if kind == "incoherent_frame_state":
        return migrated.model_copy(
            update={
                "frame_state": migrated.frame_state.model_copy(
                    update={"frame_id": "mismatched_frame"}
                )
            }
        )
    raise AssertionError(f"unknown invalid migration envelope: {kind}")


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


def _derived_signal(
    signal_id: str,
    content: str,
    *,
    parent_id: str,
    root: str,
) -> ExternalSignal:
    return ExternalSignal(
        id=signal_id,
        cycle_id="pending",
        signal_kind=SignalKind.ACTIVE,
        source_type="derived_summary",
        source="summary-worker",
        raw_content=content,
        initial_target_hypotheses=["A", "B"],
        provenance=SignalProvenance(
            epistemic_origin=EpistemicOrigin.DERIVED_SUMMARY,
            source_identity="summary-worker",
            parent_signal_ids=[parent_id],
            derivation_root_id=root,
            correlation_group="summary-worker",
            canonical_content_fingerprint="to-be-normalized",
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


def test_model_origin_without_provider_identity_uses_exact_safe_fallback():
    normalizer = SignalProvenanceNormalizer()
    first = ExternalSignal(
        id="S_model_fallback_1",
        cycle_id="pending",
        signal_kind=SignalKind.ACTIVE,
        source_type="model_probe_gateway",
        source="fallback  provider",
        raw_content="One fallback model observation.",
    )
    equivalent = first.model_copy(update={"id": "S_model_fallback_2"})
    whitespace_distinct = first.model_copy(
        update={
            "id": "S_model_fallback_3",
            "source": "fallback provider",
        }
    )

    normalized = normalizer.normalize(first, run_id="run_fallback")
    normalized_equivalent = normalizer.normalize(
        equivalent,
        run_id="run_fallback",
    )
    normalized_distinct = normalizer.normalize(
        whitespace_distinct,
        run_id="run_fallback",
    )

    assert normalized.provenance.provider_model_or_tool_identity == (
        "model_provider_fallback:v1:"
        '{"source":"fallback  provider",'
        '"source_type":"model_probe_gateway"}'
    )
    assert normalized.provenance == normalized_equivalent.provenance
    assert normalized.provenance.provider_model_or_tool_identity != (
        normalized_distinct.provenance.provider_model_or_tool_identity
    )
    assert normalized.provenance.source_identity != (
        normalized_distinct.provenance.source_identity
    )
    assert normalized.provenance.correlation_group != (
        normalized_distinct.provenance.correlation_group
    )
    assert normalized.provenance.canonical_content_fingerprint != (
        normalized_distinct.provenance.canonical_content_fingerprint
    )


def test_shared_model_provenance_keys_are_exact_pipe_safe_and_normalizer_owned():
    provider_identity = "provider|model  K"
    session_id = "session|one"

    keys = evidence_memory.derive_model_provenance_keys(
        provider_identity=provider_identity,
        session_id=session_id,
    )

    assert isinstance(keys, evidence_memory.ModelProvenanceKeys)
    assert re.fullmatch(r"model-source:sha256:[0-9a-f]{64}", keys.source_identity)
    assert re.fullmatch(r"model:sha256:[0-9a-f]{64}", keys.correlation_group)
    assert "|" not in keys.source_identity
    assert "|" not in keys.correlation_group
    assert provider_identity not in keys.source_identity
    assert provider_identity not in keys.correlation_group
    supplied = ExternalSignal(
        id="S_shared_model_keys",
        cycle_id="pending",
        signal_kind=SignalKind.ACTIVE,
        source_type="custom_model_adapter",
        source="model-provider",
        raw_content="One shared-key model observation.",
        provenance=SignalProvenance(
            epistemic_origin=EpistemicOrigin.MODEL_REASONING,
            source_identity="temporary-source",
            provider_model_or_tool_identity=provider_identity,
            session_id=session_id,
            derivation_root_id="root-shared-model-keys",
            correlation_group="temporary-group",
            canonical_content_fingerprint="replace-me",
        ),
    )

    normalized = SignalProvenanceNormalizer().normalize(
        supplied,
        run_id="unused-run",
    )

    assert normalized.provenance.provider_model_or_tool_identity == provider_identity
    assert normalized.provenance.source_identity == keys.source_identity
    assert normalized.provenance.correlation_group == keys.correlation_group


def test_shared_model_provenance_keys_reject_nfkc_sensitive_session_before_hash(
    monkeypatch,
):
    digest_calls = []
    original_sha256 = evidence_memory.hashlib.sha256

    def recording_sha256(value=b""):
        digest_calls.append(value)
        return original_sha256(value)

    monkeypatch.setattr(evidence_memory.hashlib, "sha256", recording_sha256)

    with pytest.raises(ValueError, match="model provenance") as exc_info:
        evidence_memory.derive_model_provenance_keys(
            provider_identity="provider|model",
            session_id=_NFKC_SENSITIVE_NAME,
        )

    assert _NFKC_SENSITIVE_NAME not in str(exc_info.value)
    assert "api_key" not in str(exc_info.value)
    assert digest_calls == []


def test_shared_model_gateway_signal_source_preserves_safe_exact_adapter():
    assert evidence_memory.derive_model_gateway_signal_source(
        "custom|adapter"
    ) == "model_gateway:custom|adapter"


@pytest.mark.parametrize(
    "adapter_kind",
    [" custom-adapter", "custom-adapter\nnext", _NFKC_SENSITIVE_NAME],
)
def test_shared_model_gateway_signal_source_rejects_invalid_adapter(
    adapter_kind,
):
    with pytest.raises(ValueError, match="model signal source") as exc_info:
        evidence_memory.derive_model_gateway_signal_source(adapter_kind)

    assert adapter_kind not in str(exc_info.value)
    assert "api_key" not in str(exc_info.value)


def test_model_correlation_group_uses_exact_session_identity():
    normalizer = SignalProvenanceNormalizer()
    signal = ExternalSignal(
        id="S_exact_session",
        cycle_id="pending",
        signal_kind=SignalKind.ACTIVE,
        source_type="custom_model_adapter",
        source="provider/model-a",
        raw_content="One exact-session model observation.",
        provenance=SignalProvenance(
            epistemic_origin=EpistemicOrigin.MODEL_REASONING,
            source_identity="caller-model-source",
            provider_model_or_tool_identity="provider/model-a",
            derivation_root_id="root-exact-session",
            correlation_group="caller-model-group",
            canonical_content_fingerprint="replace-me",
        ),
    )

    spaced = normalizer.normalize(signal, run_id="session  one")
    collapsed = normalizer.normalize(signal, run_id="session one")

    assert spaced.provenance.provider_model_or_tool_identity == "provider/model-a"
    assert collapsed.provenance.provider_model_or_tool_identity == "provider/model-a"
    assert spaced.provenance.session_id == "session  one"
    assert collapsed.provenance.session_id == "session one"
    assert spaced.provenance.source_identity == collapsed.provenance.source_identity
    assert spaced.provenance.canonical_content_fingerprint == (
        collapsed.provenance.canonical_content_fingerprint
    )
    assert spaced.provenance.correlation_group != (
        collapsed.provenance.correlation_group
    )


def test_non_model_source_normalization_keeps_nfkc_whitespace_semantics():
    normalizer = SignalProvenanceNormalizer()
    first = ExternalSignal(
        id="S_human_source_1",
        cycle_id="pending",
        signal_kind=SignalKind.PASSIVE,
        source_type="human_input",
        source="source K  alpha",
        raw_content="One human observation.",
    )
    equivalent = first.model_copy(
        update={
            "id": "S_human_source_2",
            "source": "source \u212a alpha",
        }
    )

    normalized = normalizer.normalize(first, run_id="run_human")
    normalized_equivalent = normalizer.normalize(
        equivalent,
        run_id="run_human",
    )

    assert normalized.provenance.source_identity == (
        normalized_equivalent.provenance.source_identity
    )
    assert normalized.provenance.correlation_group == (
        normalized_equivalent.provenance.correlation_group
    )
    assert normalized.provenance.canonical_content_fingerprint == (
        normalized_equivalent.provenance.canonical_content_fingerprint
    )


def test_model_session_secret_is_rejected_before_any_identity_digest(monkeypatch):
    signal = ExternalSignal(
        id="S_model_secret_session",
        cycle_id="pending",
        signal_kind=SignalKind.ACTIVE,
        source_type="model_probe_gateway",
        source="model_gateway:scripted",
        raw_content="One safe model observation.",
    )
    digest_calls = []
    original_sha256 = evidence_memory.hashlib.sha256

    def recording_sha256(value=b""):
        digest_calls.append(value)
        return original_sha256(value)

    monkeypatch.setattr(evidence_memory.hashlib, "sha256", recording_sha256)

    with pytest.raises(ValueError, match="secret") as exc_info:
        SignalProvenanceNormalizer().normalize(
            signal,
            run_id=_NFKC_SECRET_VALUE,
        )

    assert _NFKC_SECRET_VALUE not in str(exc_info.value)
    assert digest_calls == []


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
    assert first_normalized.provenance.supplied_correlation_group == "caller-group-1"
    assert second_normalized.provenance.supplied_correlation_group == "caller-group-2"

    manager = EvidenceMemoryManager()
    memory = EvidenceMemorySnapshot()
    first_event = None
    for index, signal in enumerate((first_normalized, second_normalized), start=1):
        decision = manager.classify(
            memory,
            signal,
            likelihoods={"A": LikelihoodBand.MODERATELY_CONFIRMING},
            base_effective_weight=0.25,
        )
        event = EvidenceEvent(
            id=f"E_supplied_model_{index}",
            derived_from_signal=signal.id,
            target_hypotheses=["A"],
            evidence_type=EvidenceType.SUPPORTING,
            content=signal.raw_content,
            likelihoods={"A": LikelihoodBand.MODERATELY_CONFIRMING},
            correlation_status=decision.correlation_status,
            effective_update_weight=decision.effective_update_weight,
        )
        memory = manager.commit(memory, signal=signal, event=event, decision=decision)
        if first_event is None:
            first_event = event

    canonical_group = first_normalized.provenance.correlation_group
    first_identity = json.loads(
        memory.source_content_fingerprints[first_normalized.id]
    )
    second_identity = json.loads(
        memory.source_content_fingerprints[second_normalized.id]
    )
    replayed = manager.commit(
        memory,
        signal=first_normalized,
        event=first_event,
        decision=manager.classify(memory, first_normalized),
    )
    changed = base.model_copy(
        update={
            "provenance": base.provenance.model_copy(
                update={"correlation_group": "caller-group-changed"}
            )
        }
    )
    changed_normalized = normalizer.normalize(changed, run_id="run_memory")

    assert first_identity[2:] == [canonical_group, "caller-group-1"]
    assert second_identity[2:] == [canonical_group, "caller-group-2"]
    assert memory.correlation_credit == {f"{canonical_group}|A|confirming": 0.5}
    assert replayed == memory
    with pytest.raises(ValueError, match="signal id lineage conflict"):
        manager.validate_signal_lineage(memory, changed_normalized)


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
    state = state.model_copy(
        update={
            "evidence_memory": first.evidence_memory,
            "ledger_refs": {
                **state.ledger_refs,
                "evidence_events": [
                    *state.ledger_refs.get("evidence_events", []),
                    *(event.id for event in first.evidence_events),
                ],
            },
        }
    )

    repeated = gate.integrate(
        cycle=_cycle(2),
        belief_state=state,
        probe_set=_probe_set(2),
        signals=[_signal("S_repeat", "The audited value supports A.")],
    )

    event = repeated.evidence_events[0]
    assert event.discard_reason == "duplicate_exact"
    assert event.effective_update_weight is None
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


def test_native_event_id_and_binding_share_canonical_signal_identity_digest():
    signal = _signal(
        "S_native_binding",
        "A native event identity binding.",
        root="root-native-binding",
    )
    normalized = SignalProvenanceNormalizer().normalize(
        signal,
        run_id="run_memory",
    )

    result = EvidenceIntegrationGate(model_gateway=CountingGateway()).integrate(
        cycle=_cycle(1),
        belief_state=_state(),
        probe_set=_probe_set(1),
        signals=[signal],
    )

    event = result.evidence_events[0]
    digest = evidence_memory.canonical_signal_identity_digest(normalized)
    assert event.id.endswith(f"_E_{digest}_1")
    assert result.evidence_memory.event_signal_identity_digests == {
        event.id: digest
    }


def test_memory_transition_validator_accepts_production_and_identity_only_replay():
    state = _state()
    gate = EvidenceIntegrationGate(model_gateway=CountingGateway())
    first_signal = _signal(
        "S_transition_first",
        "A stable transition observation.",
        root="root-transition",
    )
    first = gate.integrate(
        cycle=_cycle(1),
        belief_state=state,
        probe_set=_probe_set(1),
        signals=[first_signal],
    )
    manager = EvidenceMemoryManager()

    validated_first = manager.validate_transition(
        state.evidence_memory,
        first.evidence_memory,
        evidence_events=first.evidence_events,
        normalized_signals=first.normalized_signals,
        existing_evidence_ids=state.ledger_refs.get("evidence_events", []),
        frame_version=state.frame_state.frame_version,
    )

    replay_state = state.model_copy(
        update={
            "evidence_memory": validated_first,
            "ledger_refs": {
                **state.ledger_refs,
                "evidence_events": [
                    event.id for event in first.evidence_events
                ],
            },
        }
    )
    replay = gate.integrate(
        cycle=_cycle(1),
        belief_state=replay_state,
        probe_set=_probe_set(1),
        signals=[
            first_signal.model_copy(update={"id": "S_transition_replay"})
        ],
    )

    validated_replay = manager.validate_transition(
        validated_first,
        replay.evidence_memory,
        evidence_events=replay.evidence_events,
        normalized_signals=replay.normalized_signals,
        existing_evidence_ids=replay_state.ledger_refs["evidence_events"],
        frame_version=replay_state.frame_state.frame_version,
    )

    assert validated_replay == replay.evidence_memory
    assert set(validated_replay.content_fingerprints) == {
        "S_transition_first",
        "S_transition_replay",
    }
    assert validated_replay.accepted_evidence_ids == (
        validated_first.accepted_evidence_ids
    )
    assert validated_replay.correlation_credit == validated_first.correlation_credit


def test_legacy_memory_transition_rejects_replay_only_credit_replacement():
    state = _migrated_state("belief_state_v0.1_to_v0.2")
    first = EvidenceIntegrationGate(
        model_gateway=LegacyCountingGateway()
    ).integrate(
        cycle=_cycle(1),
        belief_state=state,
        probe_set=_probe_set(1),
        signals=[
            _signal(
                "S_transition_credit",
                "A directional transition observation.",
                root="root-transition-credit",
            )
        ],
    )
    assert first.evidence_memory.correlation_credit
    replaced = first.evidence_memory.model_copy(
        update={"correlation_credit": {}}
    )

    with pytest.raises(ValueError, match="evidence memory transition"):
        EvidenceMemoryManager().validate_transition(
            first.evidence_memory,
            replaced,
            evidence_events=first.evidence_events,
            normalized_signals=first.normalized_signals,
            existing_evidence_ids=[
                event.id for event in first.evidence_events
            ],
            frame_version=state.frame_state.frame_version,
        )


def test_existing_binding_preflight_precedes_identity_or_classification(
    monkeypatch,
):
    state = _state()
    first = EvidenceIntegrationGate(model_gateway=CountingGateway()).integrate(
        cycle=_cycle(1),
        belief_state=state,
        probe_set=_probe_set(1),
        signals=[
            _signal(
                "S_binding_preflight_prior",
                "The original bound observation.",
                root="root-binding-preflight",
            )
        ],
    )
    prior_event = first.evidence_events[0]
    changed_signal = SignalProvenanceNormalizer().normalize(
        _signal(
            "S_binding_preflight_changed",
            "A changed observation must not reuse the event.",
            root="root-binding-preflight-changed",
        ),
        run_id=state.run_id,
    )
    replay_event = EvidenceEvent.model_validate(
        {
            **prior_event.model_dump(mode="python"),
            "derived_from_signal": changed_signal.id,
            "epistemic_origin": changed_signal.provenance.epistemic_origin,
            "derivation_root_id": changed_signal.provenance.derivation_root_id,
            "evidence_type": EvidenceType.NEUTRAL,
            "likelihoods": {
                "A": LikelihoodBand.NEUTRAL,
                "B": LikelihoodBand.NEUTRAL,
            },
            "correlation_status": "duplicate_exact",
            "effective_update_weight": None,
            "discard_reason": "duplicate evidence event id",
        }
    )
    manager = EvidenceMemoryManager()
    candidate = manager.remember_signal_identity(
        first.evidence_memory,
        changed_signal,
    )
    calls = []
    original_remember = EvidenceMemoryManager.remember_signal_identity
    original_classify = EvidenceMemoryManager.classify

    def recording_remember(self, *args, **kwargs):
        calls.append("remember")
        return original_remember(self, *args, **kwargs)

    def recording_classify(self, *args, **kwargs):
        calls.append("classify")
        return original_classify(self, *args, **kwargs)

    monkeypatch.setattr(
        EvidenceMemoryManager,
        "remember_signal_identity",
        recording_remember,
    )
    monkeypatch.setattr(
        EvidenceMemoryManager,
        "classify",
        recording_classify,
    )

    with pytest.raises(ValueError, match="evidence memory transition"):
        manager.validate_transition(
            first.evidence_memory,
            candidate,
            evidence_events=[replay_event],
            normalized_signals=[changed_signal],
            existing_evidence_ids=[prior_event.id],
            frame_version=state.frame_state.frame_version,
        )

    assert calls == []


def test_memory_transition_validator_accepts_projection_two_event_reconstruction():
    state = _state()
    result = EvidenceIntegrationGate().integrate(
        cycle=_cycle(1),
        belief_state=state,
        probe_set=_probe_set(1),
        signals=[
            ExternalSignal(
                id="S_projection_transition",
                cycle_id="pending",
                signal_kind=SignalKind.PASSIVE,
                source_type="external_agent_projection",
                source="agent-a",
                raw_content=(
                    "Agent A cites source X as evidence while favoring option A."
                ),
                initial_target_hypotheses=["A", "B"],
            )
        ],
    )

    validated = EvidenceMemoryManager().validate_transition(
        state.evidence_memory,
        result.evidence_memory,
        evidence_events=result.evidence_events,
        normalized_signals=result.normalized_signals,
        existing_evidence_ids=[],
        frame_version=state.frame_state.frame_version,
    )

    assert len(result.evidence_events) == 2
    sender_event, source_event = result.evidence_events
    assert sender_event.correlation_status == source_event.correlation_status
    assert set(source_event.likelihoods.values()) == {LikelihoodBand.NEUTRAL}
    assert validated == result.evidence_memory


def test_memory_transition_reconstructs_projection_credit_cumulatively():
    manager = EvidenceMemoryManager(
        CorrelationCreditPolicy(
            max_cumulative_effective_weight_per_direction=0.15
        )
    )
    prior = EvidenceMemorySnapshot()
    signal = SignalProvenanceNormalizer().normalize(
        ExternalSignal(
            id="S_directional_projection_transition",
            cycle_id="pending",
            signal_kind=SignalKind.PASSIVE,
            source_type="external_agent_projection",
            source="agent-a",
            raw_content="Agent A cites source X and both claims support A.",
            initial_target_hypotheses=["A"],
        ),
        run_id="run_memory",
    )
    likelihoods = {"A": LikelihoodBand.MODERATELY_CONFIRMING}
    event_specs = [
        (
            "E_directional_projection_transition",
            EvidenceType.SENDER_JUDGMENT,
            {
                "reliability": 0.55,
                "independence": 0.45,
                "relevance": 0.75,
                "novelty": 0.6,
                "specificity": 0.6,
                "verifiability": 0.4,
            },
        ),
        (
            "E_directional_projection_transition_source",
            EvidenceType.SOURCE_CLAIM,
            {
                "reliability": 0.5,
                "independence": 0.45,
                "relevance": 0.7,
                "novelty": 0.6,
                "specificity": 0.6,
                "verifiability": 0.4,
            },
        ),
    ]
    identity_snapshot = prior
    working = prior
    events = []

    for event_id, event_type, quality in event_specs:
        decision = manager.classify(
            identity_snapshot,
            signal,
            credit_snapshot=working,
            likelihoods=likelihoods,
            unresolved_likelihood=LikelihoodBand.NEUTRAL,
            base_effective_weight=(
                quality["reliability"]
                * quality["independence"]
                * quality["relevance"]
                * quality["novelty"]
            ),
        )
        event = _directional_memory_event(
            signal=signal,
            event_id=event_id,
            decision=decision,
            likelihoods=likelihoods,
            event_type=event_type,
            quality=quality,
        )
        working = manager.commit(
            working,
            signal=signal,
            event=event,
            decision=decision,
        )
        events.append(event)

    validated = manager.validate_transition(
        prior,
        working,
        evidence_events=events,
        normalized_signals=[signal],
        existing_evidence_ids=[],
        frame_version=1,
    )

    assert [event.correlation_status for event in events] == ["novel", "novel"]
    assert [event.effective_update_weight for event in events] == pytest.approx(
        [0.111375, 0.038625]
    )
    assert validated.correlation_credit == {
        f"{signal.provenance.correlation_group}|A|confirming": pytest.approx(
            0.15
        )
    }


def test_memory_transition_rejects_each_projection_event_content_rewrite():
    state = _state()
    result = EvidenceIntegrationGate().integrate(
        cycle=_cycle(1),
        belief_state=state,
        probe_set=_probe_set(1),
        signals=[
            ExternalSignal(
                id="S_projection_content_binding",
                cycle_id="pending",
                signal_kind=SignalKind.PASSIVE,
                source_type="external_agent_projection",
                source="agent-a",
                raw_content=(
                    "Agent A cites source X as evidence while favoring option A."
                ),
                initial_target_hypotheses=["A", "B"],
            )
        ],
    )
    manager = EvidenceMemoryManager()
    assert len(result.evidence_events) == 2

    for event_index in range(2):
        rewritten_events = list(result.evidence_events)
        rewritten_events[event_index] = rewritten_events[event_index].model_copy(
            update={"content": f"{rewritten_events[event_index].content} "}
        )

        with pytest.raises(ValueError, match="evidence memory transition"):
            manager.validate_transition(
                state.evidence_memory,
                result.evidence_memory,
                evidence_events=rewritten_events,
                normalized_signals=result.normalized_signals,
                existing_evidence_ids=[],
                frame_version=state.frame_state.frame_version,
            )


def _same_signature_different_lineage_signals():
    first = _signal(
        "S_signature_first",
        "The same source repeats this audited observation.",
        root="root-signature-first",
    )
    second = first.model_copy(
        update={
            "id": "S_signature_second",
            "source": "  SOURCE.EXAMPLE/REPORT  ",
            "raw_content": (
                "  THE SAME SOURCE\nREPEATS THIS AUDITED OBSERVATION.  "
            ),
            "provenance": first.provenance.model_copy(
                update={
                    "derivation_root_id": "root-signature-second",
                    "correlation_group": "caller-supplied-second-group",
                }
            ),
        }
    )
    return first, second


def test_same_batch_source_content_duplicate_uses_shared_quality_cap():
    state = _state()
    first, second = _same_signature_different_lineage_signals()
    result = EvidenceIntegrationGate(model_gateway=CountingGateway()).integrate(
        cycle=_cycle(1),
        belief_state=state,
        probe_set=_probe_set(1),
        signals=[first, second],
    )

    first_event, second_event = result.evidence_events
    assert first_event.correlation_status == "novel"
    assert first_event.independence == 0.8
    assert first_event.novelty == 0.8
    assert first_event.effective_update_weight is None
    assert second_event.correlation_status == "correlated_novel"
    assert second_event.independence == 0.25
    assert second_event.novelty == 0.25
    assert second_event.effective_update_weight is None
    assert EvidenceMemoryManager().validate_transition(
        state.evidence_memory,
        result.evidence_memory,
        evidence_events=result.evidence_events,
        normalized_signals=result.normalized_signals,
        existing_evidence_ids=[],
        frame_version=state.frame_state.frame_version,
    ) == result.evidence_memory


def test_distinct_cycle_signatures_keep_standard_quality():
    first = _signal(
        "S_signature_distinct_first",
        "The first audited observation.",
        root="root-signature-distinct-first",
    )
    second = _signal(
        "S_signature_distinct_second",
        "A materially distinct audited observation.",
        root="root-signature-distinct-second",
    )

    result = EvidenceIntegrationGate(model_gateway=CountingGateway()).integrate(
        cycle=_cycle(1),
        belief_state=_state(),
        probe_set=_probe_set(1),
        signals=[first, second],
    )

    assert [event.independence for event in result.evidence_events] == [0.8, 0.8]
    assert [event.novelty for event in result.evidence_events] == [0.8, 0.8]
    assert all(
        event.effective_update_weight is None
        for event in result.evidence_events
    )


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


def test_batch_preflight_normalizes_once_and_stops_before_provider():
    class RecordingNormalizer(SignalProvenanceNormalizer):
        def __init__(self):
            self.calls = []

        def normalize(self, signal, *, run_id):
            self.calls.append((signal.id, signal.raw_content))
            return super().normalize(signal, run_id=run_id)

    normalizer = RecordingNormalizer()
    gateway = CountingGateway()
    first = _signal("S_same_batch", "First same-batch observation.", root="root-1")
    conflicting = _signal(
        "S_same_batch",
        "Changed same-batch observation.",
        root="root-2",
    )

    with pytest.raises(ValueError, match="signal id lineage conflict"):
        EvidenceIntegrationGate(
            model_gateway=gateway,
            provenance_normalizer=normalizer,
        ).integrate(
            cycle=_cycle(1),
            belief_state=_state(),
            probe_set=_probe_set(1),
            signals=[first, conflicting],
        )

    assert normalizer.calls == [
        (first.id, first.raw_content),
        (conflicting.id, conflicting.raw_content),
    ]
    assert gateway.requests == []


@pytest.mark.parametrize(
    "location",
    ["id", "cycle_id", "generated_by_probe", "initial_target_hypotheses"],
)
def test_recursive_signal_secret_validation_precedes_identity_hash(
    monkeypatch,
    location,
):
    signal = _signal("S_prehash_secret", "An ordinary observation.")
    update = {
        location: (
            [_NFKC_SECRET_VALUE]
            if location == "initial_target_hypotheses"
            else _NFKC_SECRET_VALUE
        )
    }
    signal = signal.model_copy(update=update)
    hash_calls = []
    original_hash = evidence_memory._sha256_identity

    def recording_hash(source_identity, content):
        hash_calls.append((source_identity, content))
        return original_hash(source_identity, content)

    monkeypatch.setattr(evidence_memory, "_sha256_identity", recording_hash)

    with pytest.raises(ValueError, match="secret") as exc_info:
        SignalProvenanceNormalizer().normalize(signal, run_id="run_memory")

    assert _NFKC_SECRET_VALUE not in str(exc_info.value)
    assert hash_calls == []


def test_projection_secondary_event_id_is_validated_during_batch_planning(
    monkeypatch,
):
    validated_ids = []

    def validate_event_id(event_id):
        validated_ids.append(event_id)
        if event_id.endswith("_source"):
            raise ValueError("canonical event binding id is invalid")
        return event_id

    monkeypatch.setattr(
        evidence_module,
        "validate_canonical_event_binding_id",
        validate_event_id,
        raising=False,
    )
    gateway = CountingGateway()
    state = _state()
    prior_state = state.model_dump(mode="json")

    with pytest.raises(ValueError, match="canonical event binding id"):
        EvidenceIntegrationGate(model_gateway=gateway).integrate(
            cycle=_cycle(1),
            belief_state=state,
            probe_set=_probe_set(1),
            signals=[
                ExternalSignal(
                    id="S_projection_event_id",
                    cycle_id="pending",
                    signal_kind=SignalKind.PASSIVE,
                    source_type="external_agent_projection",
                    source="agent-a",
                    raw_content="Agent A cites source X while favoring option A.",
                    initial_target_hypotheses=["A", "B"],
                )
            ],
        )

    assert validated_ids[-1].endswith("_source")
    assert gateway.requests == []
    assert state.model_dump(mode="json") == prior_state


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
    state = state.model_copy(
        update={
            "evidence_memory": first.evidence_memory,
            "ledger_refs": {
                **state.ledger_refs,
                "evidence_events": [
                    *state.ledger_refs.get("evidence_events", []),
                    *(event.id for event in first.evidence_events),
                ],
            },
        }
    )

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
    assert event.effective_update_weight is None
    assert len(gateway.requests) == 2


@pytest.mark.parametrize("value", [True, 0, -1, float("inf"), float("nan"), "1"])
def test_correlation_credit_policy_rejects_invalid_caps(value):
    with pytest.raises(ValueError, match="correlation credit cap"):
        CorrelationCreditPolicy(value)


@pytest.mark.parametrize("cap", [1.0, 0.2], ids=["default", "custom"])
def test_manager_policy_snapshot_accepts_credit_at_exact_cap(cap):
    manager = EvidenceMemoryManager(
        CorrelationCreditPolicy(
            max_cumulative_effective_weight_per_direction=cap
        )
    )
    snapshot = EvidenceMemorySnapshot(
        correlation_credit={"unrelated-policy-group|A|confirming": cap}
    )

    assert manager.validate_policy_snapshot(snapshot) is snapshot
    assert manager.validate_transition(
        snapshot,
        snapshot,
        evidence_events=[],
        normalized_signals=[],
        existing_evidence_ids=[],
        frame_version=1,
    ) == snapshot


@pytest.mark.parametrize(
    "snapshot",
    [
        EvidenceMemorySnapshot.model_construct(memory_version=True),
        EvidenceMemorySnapshot.model_construct(
            correlation_credit={
                "bypass-group|A|confirming": True,
            }
        ),
    ],
    ids=["boolean-version", "boolean-credit"],
)
def test_manager_policy_snapshot_rejects_model_construct_scalar_bypass(snapshot):
    with pytest.raises(ValueError):
        EvidenceMemoryManager().validate_policy_snapshot(snapshot)


@pytest.mark.parametrize(
    "snapshot",
    [
        EvidenceMemorySnapshot.model_construct(memory_version=True),
        EvidenceMemorySnapshot.model_construct(
            correlation_credit={
                "bypass-group|A|confirming": True,
            }
        ),
    ],
    ids=["boolean-version", "boolean-credit"],
)
def test_transition_recursively_rejects_model_construct_scalar_bypass(snapshot):
    with pytest.raises(ValueError, match="evidence memory transition is invalid"):
        EvidenceMemoryManager().validate_transition(
            snapshot,
            snapshot,
            evidence_events=[],
            normalized_signals=[],
            existing_evidence_ids=[],
            frame_version=1,
        )


@pytest.mark.parametrize("cap", [1.0, 0.2], ids=["default", "custom"])
def test_manager_policy_snapshot_rejects_smallest_representable_overcap(cap):
    manager = EvidenceMemoryManager(
        CorrelationCreditPolicy(
            max_cumulative_effective_weight_per_direction=cap
        )
    )
    credit_key = "unrelated-policy-group|A|confirming"
    overcap = math.nextafter(cap, math.inf)
    snapshot = EvidenceMemorySnapshot(
        correlation_credit={credit_key: overcap}
    )

    with pytest.raises(ValueError, match="correlation credit policy") as exc_info:
        manager.validate_policy_snapshot(snapshot)

    error_text = str(exc_info.value)
    assert credit_key not in error_text
    assert repr(overcap) not in error_text


@pytest.mark.parametrize(
    "operation",
    ["remember", "classify-identity", "classify-credit", "commit"],
)
def test_manager_entry_points_reject_unrelated_overcap_credit(operation):
    manager = EvidenceMemoryManager(
        CorrelationCreditPolicy(
            max_cumulative_effective_weight_per_direction=0.2
        )
    )
    signal = SignalProvenanceNormalizer().normalize(
        _signal(
            "S_policy_entry_point",
            "A policy-entry signal supports option A.",
            root="root-policy-entry-point",
        ),
        run_id="run_memory",
    )
    valid = EvidenceMemorySnapshot()
    invalid = EvidenceMemorySnapshot(
        correlation_credit={
            "unrelated-policy-group|B|disconfirming": 0.3
        }
    )
    likelihoods = {"A": LikelihoodBand.MODERATELY_CONFIRMING}
    decision = manager.classify(
        valid,
        signal,
        likelihoods=likelihoods,
        unresolved_likelihood=LikelihoodBand.NEUTRAL,
        base_effective_weight=0.1,
    )
    event = _directional_memory_event(
        signal=signal,
        event_id="E_policy_entry_point",
        decision=decision,
        likelihoods=likelihoods,
    )

    with pytest.raises(ValueError, match="correlation credit policy"):
        if operation == "remember":
            manager.remember_signal_identity(invalid, signal)
        elif operation == "classify-identity":
            manager.classify(invalid, signal)
        elif operation == "classify-credit":
            manager.classify(valid, signal, credit_snapshot=invalid)
        else:
            manager.commit(
                invalid,
                signal=signal,
                event=event,
                decision=decision,
            )


def test_transition_rejects_unchanged_overcap_prior_without_reconstruction():
    manager = EvidenceMemoryManager(
        CorrelationCreditPolicy(
            max_cumulative_effective_weight_per_direction=0.2
        )
    )
    invalid = EvidenceMemorySnapshot(
        correlation_credit={"unrelated-policy-group|A|confirming": 0.3}
    )

    with pytest.raises(ValueError, match="evidence memory transition is invalid"):
        manager.validate_transition(
            invalid,
            invalid,
            evidence_events=[],
            normalized_signals=[],
            existing_evidence_ids=[],
            frame_version=1,
        )


def test_transition_rejects_overcap_candidate_before_reconstruction(monkeypatch):
    manager = EvidenceMemoryManager(
        CorrelationCreditPolicy(
            max_cumulative_effective_weight_per_direction=0.2
        )
    )
    prior = EvidenceMemorySnapshot()
    candidate = EvidenceMemorySnapshot(
        correlation_credit={"unrelated-policy-group|A|confirming": 0.3}
    )
    reconstruction_calls = []
    original_validate_transition = manager._validate_transition

    def recording_validate_transition(*args, **kwargs):
        reconstruction_calls.append(True)
        return original_validate_transition(*args, **kwargs)

    monkeypatch.setattr(
        manager,
        "_validate_transition",
        recording_validate_transition,
    )

    with pytest.raises(ValueError, match="evidence memory transition is invalid"):
        manager.validate_transition(
            prior,
            candidate,
            evidence_events=[],
            normalized_signals=[],
            existing_evidence_ids=[],
            frame_version=1,
        )

    assert reconstruction_calls == []


def test_commit_rejects_one_ulp_above_cap_without_clamping():
    manager = EvidenceMemoryManager(
        CorrelationCreditPolicy(
            max_cumulative_effective_weight_per_direction=0.2
        )
    )
    signal = SignalProvenanceNormalizer().normalize(
        _signal(
            "S_one_ulp_commit",
            "A submitted event claims one ULP too much weight.",
            root="root-one-ulp-commit",
        ),
        run_id="run_memory",
    )
    snapshot = EvidenceMemorySnapshot()
    canonical = manager.classify(snapshot, signal)
    credit_key = (
        f"{canonical.canonical_correlation_group}|A|confirming"
    )
    inflated_weight = math.nextafter(0.2, math.inf)
    decision = evidence_memory.EvidenceMemoryDecision(
        correlation_status="novel",
        effective_update_weight=inflated_weight,
        discard_reason=None,
        remaining_credit={credit_key: 0.0},
        canonical_correlation_group=canonical.canonical_correlation_group,
    )
    event = _directional_memory_event(
        signal=signal,
        event_id="E_one_ulp_commit",
        decision=decision,
        likelihoods={"A": LikelihoodBand.MODERATELY_CONFIRMING},
    )

    with pytest.raises(
        ValueError,
        match="directional correlation credit decision is inconsistent",
    ):
        manager.commit(
            snapshot,
            signal=signal,
            event=event,
            decision=decision,
        )

    assert snapshot.correlation_credit == {}
    assert snapshot.accepted_evidence_ids == []


def test_commit_rejects_one_ulp_remaining_credit_mismatch():
    manager = EvidenceMemoryManager(
        CorrelationCreditPolicy(
            max_cumulative_effective_weight_per_direction=0.2
        )
    )
    signal = SignalProvenanceNormalizer().normalize(
        _signal(
            "S_one_ulp_remaining",
            "A submitted decision misstates its remaining credit.",
            root="root-one-ulp-remaining",
        ),
        run_id="run_memory",
    )
    snapshot = EvidenceMemorySnapshot()
    likelihoods = {"A": LikelihoodBand.MODERATELY_CONFIRMING}
    decision = manager.classify(
        snapshot,
        signal,
        likelihoods=likelihoods,
        unresolved_likelihood=LikelihoodBand.NEUTRAL,
        base_effective_weight=0.2,
    )
    credit_key = next(iter(decision.remaining_credit))
    forged_decision = replace(
        decision,
        remaining_credit={credit_key: math.nextafter(0.0, math.inf)},
    )

    with pytest.raises(
        ValueError,
        match="directional correlation credit decision is inconsistent",
    ):
        manager.commit(
            snapshot,
            signal=signal,
            event=_directional_memory_event(
                signal=signal,
                event_id="E_one_ulp_remaining",
                decision=forged_decision,
                likelihoods=likelihoods,
            ),
            decision=forged_decision,
        )


def test_classification_rounds_credit_increment_down_before_commit():
    cap = 0.15
    prior_used = 0.015
    manager = EvidenceMemoryManager(
        CorrelationCreditPolicy(
            max_cumulative_effective_weight_per_direction=cap
        )
    )
    signal = SignalProvenanceNormalizer().normalize(
        _signal(
            "S_safe_credit_increment",
            "A production decision consumes the representable remainder.",
            root="root-safe-credit-increment",
        ),
        run_id="run_memory",
    )
    credit_key = f"{signal.provenance.correlation_group}|A|confirming"
    snapshot = EvidenceMemorySnapshot(
        correlation_credit={credit_key: prior_used}
    )
    decision = manager.classify(
        snapshot,
        signal,
        likelihoods={"A": LikelihoodBand.MODERATELY_CONFIRMING},
        unresolved_likelihood=LikelihoodBand.NEUTRAL,
        base_effective_weight=1.0,
    )
    raw_subtraction = cap - prior_used
    expected_weight = math.nextafter(raw_subtraction, 0.0)

    assert prior_used + raw_subtraction > cap
    assert decision.effective_update_weight == expected_weight
    assert prior_used + decision.effective_update_weight <= cap
    event = _directional_memory_event(
        signal=signal,
        event_id="E_safe_credit_increment",
        decision=decision,
        likelihoods={"A": LikelihoodBand.MODERATELY_CONFIRMING},
    )
    committed = manager.commit(
        snapshot,
        signal=signal,
        event=event,
        decision=decision,
    )
    persisted_used = committed.correlation_credit[credit_key]

    assert persisted_used == prior_used + decision.effective_update_weight
    assert decision.remaining_credit[credit_key] == cap - persisted_used


def test_transition_rejects_one_ulp_native_event_weight_mismatch():
    manager = EvidenceMemoryManager(
        CorrelationCreditPolicy(
            max_cumulative_effective_weight_per_direction=0.2
        )
    )
    prior = EvidenceMemorySnapshot()
    signal = SignalProvenanceNormalizer().normalize(
        _signal(
            "S_one_ulp_transition",
            "A native event claims one ULP above its decision.",
            root="root-one-ulp-transition",
        ),
        run_id="run_memory",
    )
    likelihoods = {"A": LikelihoodBand.MODERATELY_CONFIRMING}
    quality = {
        "reliability": 0.8,
        "independence": 0.8,
        "relevance": 0.9,
        "novelty": 0.8,
        "specificity": 0.7,
        "verifiability": 0.7,
    }
    decision = manager.classify(
        prior,
        signal,
        likelihoods=likelihoods,
        unresolved_likelihood=LikelihoodBand.NEUTRAL,
        base_effective_weight=(
            quality["reliability"]
            * quality["independence"]
            * quality["relevance"]
            * quality["novelty"]
        ),
    )
    event = _directional_memory_event(
        signal=signal,
        event_id="E_one_ulp_transition",
        decision=decision,
        likelihoods=likelihoods,
        quality=quality,
    )
    candidate = manager.commit(
        prior,
        signal=signal,
        event=event,
        decision=decision,
    )
    assert manager.validate_transition(
        prior,
        candidate,
        evidence_events=[event],
        normalized_signals=[signal],
        existing_evidence_ids=[],
        frame_version=1,
    ) == candidate
    inflated_event = event.model_copy(
        update={
            "effective_update_weight": math.nextafter(
                event.effective_update_weight,
                math.inf,
            )
        }
    )

    with pytest.raises(ValueError, match="evidence memory transition is invalid"):
        manager.validate_transition(
            prior,
            candidate,
            evidence_events=[inflated_event],
            normalized_signals=[signal],
            existing_evidence_ids=[],
            frame_version=1,
        )


@pytest.mark.parametrize("marker", _MIGRATION_MARKERS)
def test_explicit_migration_empty_memory_respects_custom_policy(marker):
    manager = EvidenceMemoryManager(
        CorrelationCreditPolicy(
            max_cumulative_effective_weight_per_direction=0.2
        )
    )

    result = EvidenceIntegrationGate(
        model_gateway=CountingGateway(),
        memory_manager=manager,
    ).integrate(
        cycle=_cycle(1),
        belief_state=_migrated_state(marker),
        probe_set=_probe_set(1),
        signals=[
            _signal(
                f"S_policy_migration_{marker}",
                "A migrated observation supports option A.",
                root=f"root-policy-migration-{marker}",
            )
        ],
    )

    assert result.evidence_events[0].schema_version == "v0.1"
    assert all(
        credit <= 0.2
        for credit in result.evidence_memory.correlation_credit.values()
    )


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


def _directional_memory_event(
    *,
    signal: ExternalSignal,
    event_id: str,
    decision: evidence_memory.EvidenceMemoryDecision,
    likelihoods: dict[str, LikelihoodBand],
    event_type: EvidenceType = EvidenceType.SUPPORTING,
    quality: dict[str, float] | None = None,
) -> EvidenceEvent:
    return EvidenceEvent(
        schema_version="v0.2",
        id=event_id,
        derived_from_signal=signal.id,
        epistemic_origin=signal.provenance.epistemic_origin,
        derivation_root_id=signal.provenance.derivation_root_id,
        target_hypotheses=list(likelihoods),
        evidence_type=event_type,
        content=signal.raw_content,
        likelihoods=likelihoods,
        unresolved_likelihood=LikelihoodBand.NEUTRAL,
        frame_fit=FrameFit.UNDERDETERMINED,
        interpretation="Directional memory-credit regression.",
        correlation_status=decision.correlation_status,
        effective_update_weight=decision.effective_update_weight,
        discard_reason=decision.discard_reason,
        **(quality or {}),
    )


@pytest.mark.parametrize(
    ("cap", "prior_used", "base_weights", "expected_weights"),
    [
        pytest.param(1.0, 0.0, [0.75, 0.75], [0.75, 0.25], id="default"),
        pytest.param(
            0.2,
            0.05,
            [0.1, 0.1, 0.1],
            [0.1, 0.05, 0.0],
            id="custom-with-prior",
        ),
    ],
)
def test_same_signal_events_freeze_identity_but_advance_directional_credit(
    cap,
    prior_used,
    base_weights,
    expected_weights,
):
    manager = EvidenceMemoryManager(
        CorrelationCreditPolicy(
            max_cumulative_effective_weight_per_direction=cap
        )
    )
    signal = SignalProvenanceNormalizer().normalize(
        _signal(
            "S_multi_event_credit",
            "One projection yields multiple directional events.",
            root="root-multi-event-credit",
        ),
        run_id="run_memory",
    )
    credit_key = f"{signal.provenance.correlation_group}|A|confirming"
    prior = EvidenceMemorySnapshot(
        correlation_credit=({credit_key: prior_used} if prior_used else {})
    )
    identity_snapshot = prior
    working = prior
    decisions = []

    for index, base_weight in enumerate(base_weights, start=1):
        decision = manager.classify(
            identity_snapshot,
            signal,
            credit_snapshot=working,
            likelihoods={"A": LikelihoodBand.MODERATELY_CONFIRMING},
            unresolved_likelihood=LikelihoodBand.NEUTRAL,
            base_effective_weight=base_weight,
        )
        event = _directional_memory_event(
            signal=signal,
            event_id=(
                "E_multi_event_credit"
                if index == 1
                else f"E_multi_event_credit_source_{index}"
            ),
            decision=decision,
            likelihoods={"A": LikelihoodBand.MODERATELY_CONFIRMING},
        )
        working = manager.commit(
            working,
            signal=signal,
            event=event,
            decision=decision,
        )
        decisions.append(decision)

    expected_status = "correlated_novel" if prior_used else "novel"
    assert [decision.correlation_status for decision in decisions] == [
        expected_status
    ] * len(decisions)
    assert [decision.effective_update_weight for decision in decisions] == (
        pytest.approx(expected_weights)
    )
    assert working.correlation_credit == {credit_key: pytest.approx(cap)}
    assert sum(
        decision.effective_update_weight
        for decision in decisions
        if decision.discard_reason is None
    ) == pytest.approx(cap - prior_used)
    if expected_weights[-1] == 0.0:
        assert decisions[-1].discard_reason == "correlation_credit_saturated"


def test_same_signal_multi_hypothesis_credit_uses_shared_current_minimum():
    manager = EvidenceMemoryManager()
    signal = SignalProvenanceNormalizer().normalize(
        _signal(
            "S_multi_subject_credit",
            "One projection bears on two hypotheses.",
            root="root-multi-subject-credit",
        ),
        run_id="run_memory",
    )
    group = signal.provenance.correlation_group
    keys = {
        "A": f"{group}|A|confirming",
        "B": f"{group}|B|confirming",
    }
    prior = EvidenceMemorySnapshot(
        correlation_credit={keys["A"]: 0.2, keys["B"]: 0.8}
    )
    likelihoods = {
        "A": LikelihoodBand.MODERATELY_CONFIRMING,
        "B": LikelihoodBand.MODERATELY_CONFIRMING,
    }
    working = prior
    decisions = []

    for index in (1, 2):
        decision = manager.classify(
            prior,
            signal,
            credit_snapshot=working,
            likelihoods=likelihoods,
            unresolved_likelihood=LikelihoodBand.NEUTRAL,
            base_effective_weight=0.4,
        )
        working = manager.commit(
            working,
            signal=signal,
            event=_directional_memory_event(
                signal=signal,
                event_id=f"E_multi_subject_credit_{index}",
                decision=decision,
                likelihoods=likelihoods,
            ),
            decision=decision,
        )
        decisions.append(decision)

    assert [decision.effective_update_weight for decision in decisions] == (
        pytest.approx([0.2, 0.0])
    )
    assert decisions[1].discard_reason == "correlation_credit_saturated"
    assert working.correlation_credit == {
        keys["A"]: pytest.approx(0.4),
        keys["B"]: pytest.approx(1.0),
    }


def test_same_signal_opposite_directions_use_independent_credit_keys():
    manager = EvidenceMemoryManager()
    signal = SignalProvenanceNormalizer().normalize(
        _signal(
            "S_opposite_direction_credit",
            "One projection contains claims in opposite directions.",
            root="root-opposite-direction-credit",
        ),
        run_id="run_memory",
    )
    identity_snapshot = EvidenceMemorySnapshot()
    working = identity_snapshot
    decisions = []

    for index, band in enumerate(
        (
            LikelihoodBand.MODERATELY_CONFIRMING,
            LikelihoodBand.MODERATELY_DISCONFIRMING,
        ),
        start=1,
    ):
        likelihoods = {"A": band}
        decision = manager.classify(
            identity_snapshot,
            signal,
            credit_snapshot=working,
            likelihoods=likelihoods,
            unresolved_likelihood=LikelihoodBand.NEUTRAL,
            base_effective_weight=0.7,
        )
        working = manager.commit(
            working,
            signal=signal,
            event=_directional_memory_event(
                signal=signal,
                event_id=f"E_opposite_direction_credit_{index}",
                decision=decision,
                likelihoods=likelihoods,
            ),
            decision=decision,
        )
        decisions.append(decision)

    group = signal.provenance.correlation_group
    assert [decision.correlation_status for decision in decisions] == [
        "novel",
        "novel",
    ]
    assert [decision.effective_update_weight for decision in decisions] == (
        pytest.approx([0.7, 0.7])
    )
    assert working.correlation_credit == {
        f"{group}|A|confirming": pytest.approx(0.7),
        f"{group}|A|disconfirming": pytest.approx(0.7),
    }


def test_direct_commit_rejects_stale_directional_credit_decision():
    manager = EvidenceMemoryManager()
    signal = SignalProvenanceNormalizer().normalize(
        _signal(
            "S_stale_multi_event_credit",
            "One projection emits two supporting events.",
            root="root-stale-multi-event-credit",
        ),
        run_id="run_memory",
    )
    initial = EvidenceMemorySnapshot()
    likelihoods = {"A": LikelihoodBand.MODERATELY_CONFIRMING}
    first_decision = manager.classify(
        initial,
        signal,
        likelihoods=likelihoods,
        unresolved_likelihood=LikelihoodBand.NEUTRAL,
        base_effective_weight=0.75,
    )
    stale_second_decision = manager.classify(
        initial,
        signal,
        likelihoods=likelihoods,
        unresolved_likelihood=LikelihoodBand.NEUTRAL,
        base_effective_weight=0.25,
    )
    first = manager.commit(
        initial,
        signal=signal,
        event=_directional_memory_event(
            signal=signal,
            event_id="E_stale_multi_event_credit",
            decision=first_decision,
            likelihoods=likelihoods,
        ),
        decision=first_decision,
    )
    prior = first.model_dump(mode="json")

    with pytest.raises(ValueError, match="directional correlation credit"):
        manager.commit(
            first,
            signal=signal,
            event=_directional_memory_event(
                signal=signal,
                event_id="E_stale_multi_event_credit_source",
                decision=stale_second_decision,
                likelihoods=likelihoods,
            ),
            decision=stale_second_decision,
        )

    assert first.model_dump(mode="json") == prior


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
    assert committed.event_signal_identity_digests == {
        event.id: evidence_memory.canonical_signal_identity_digest(signal)
    }


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
    assert committed.event_signal_identity_digests == {
        event.id: evidence_memory.canonical_signal_identity_digest(signal)
    }
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


def test_direct_commit_rejects_known_event_with_different_signal_binding():
    manager, snapshot, _, _ = _committed_signal_identity()
    conflicting_signal = SignalProvenanceNormalizer().normalize(
        _signal(
            "S_rebound",
            "Different content must not rebind E_reused_1.",
            root="root-rebound",
        ),
        run_id="run_memory",
    )
    decision = manager.classify(
        snapshot,
        conflicting_signal,
        likelihoods={"A": LikelihoodBand.MODERATELY_CONFIRMING},
        base_effective_weight=0.4,
    )
    event = EvidenceEvent(
        id="E_reused_1",
        derived_from_signal=conflicting_signal.id,
        target_hypotheses=["A"],
        evidence_type=EvidenceType.SUPPORTING,
        content=conflicting_signal.raw_content,
        likelihoods={"A": LikelihoodBand.MODERATELY_CONFIRMING},
    )
    prior = snapshot.model_dump(mode="json")

    with pytest.raises(ValueError, match="event signal identity conflict"):
        manager.commit(
            snapshot,
            signal=conflicting_signal,
            event=event,
            decision=decision,
        )

    assert snapshot.model_dump(mode="json") == prior
    assert conflicting_signal.id not in snapshot.content_fingerprints


def test_direct_commit_rejects_known_event_without_historical_binding():
    manager, snapshot, signal, decision = _committed_signal_identity()
    historical = EvidenceMemorySnapshot(
        memory_version=snapshot.memory_version,
        accepted_evidence_ids=list(snapshot.accepted_evidence_ids),
        content_fingerprints=dict(snapshot.content_fingerprints),
        source_content_fingerprints=dict(snapshot.source_content_fingerprints),
        derivation_roots=dict(snapshot.derivation_roots),
        correlation_credit=dict(snapshot.correlation_credit),
        discovery_evidence_ids=list(snapshot.discovery_evidence_ids),
        counterevidence_ids_by_hypothesis={
            key: list(value)
            for key, value in snapshot.counterevidence_ids_by_hypothesis.items()
        },
        discard_and_schema_history=list(snapshot.discard_and_schema_history),
    )
    event = EvidenceEvent(
        id="E_reused_1",
        derived_from_signal=signal.id,
        target_hypotheses=["A"],
        evidence_type=EvidenceType.SUPPORTING,
        content=signal.raw_content,
        likelihoods={"A": LikelihoodBand.MODERATELY_CONFIRMING},
    )

    with pytest.raises(ValueError, match="event signal identity binding is missing"):
        manager.commit(
            historical,
            signal=signal,
            event=event,
            decision=decision,
        )


def test_identity_only_write_preserves_event_signal_bindings():
    manager, snapshot, signal, _ = _committed_signal_identity()
    replay_signal = signal.model_copy(update={"id": "S_reused_alias"})

    remembered = manager.remember_signal_identity(snapshot, replay_signal)

    assert remembered.event_signal_identity_digests == (
        snapshot.event_signal_identity_digests
    )


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


def test_native_unknown_external_parent_fails_closed_before_judgment():
    gateway = CountingGateway()
    signal = _derived_signal(
        "S_unknown_external_parent",
        "A summary whose external parent is not in local memory.",
        parent_id="S_external_parent",
        root="root-declared-by-summary",
    )

    with pytest.raises(ValueError, match="missing a parent signal"):
        EvidenceIntegrationGate(model_gateway=gateway).integrate(
            cycle=_cycle(1),
            belief_state=_state(),
            probe_set=_probe_set(1),
            signals=[signal],
        )

    assert gateway.requests == []


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


@pytest.mark.parametrize("operation", ["remember", "classify"])
def test_direct_memory_operations_reject_known_parent_root_mismatch(operation):
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
        if operation == "remember":
            manager.remember_signal_identity(memory, derived)
        else:
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


def test_native_judgment_request_contains_blind_semantics_and_provenance():
    gateway = CountingGateway()
    gate = EvidenceIntegrationGate(model_gateway=gateway)
    state = _state()

    assert state.task_frame.framing_trace == {"source": "answer_choices"}
    assert "migration" not in state.task_frame.framing_trace

    result = gate.integrate(
        cycle=_cycle(1),
        belief_state=state,
        probe_set=_probe_set(1),
        signals=[_signal("S_context", "The audited value supports A.")],
    )

    request = gateway.requests[0]
    hypothesis = request.input["hypotheses"][0]
    assert request.prompt_version == "v0.2"
    assert request.schema_version == "v0.2"
    assert set(hypothesis) == {
        "id",
        "statement",
        "type",
        "scope",
        "predictions",
        "falsifiers",
        "rivals",
    }
    assert request.input["task_context"] == {
        "problem": "Which option is supported?",
        "task_context": "",
    }
    assert request.input["provenance"]["derivation_root_id"] == "root-source-1"
    assert request.input["matched_probe"] is None
    assert "memory" not in request.input
    assert request.metadata["belief_context_policy"] == "blind_no_scores_v1"
    assert result.evidence_events[0].schema_version == "v0.2"
    assert result.evidence_events[0].effective_update_weight is None
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


@pytest.mark.parametrize("migration_marker", _MIGRATION_MARKERS)
def test_explicit_migration_route_completes_exact_legacy_shape_auditably(
    migration_marker,
):
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
    state = _migrated_state(migration_marker)

    result = EvidenceIntegrationGate(model_gateway=gateway).integrate(
        cycle=_cycle(1),
        belief_state=state,
        probe_set=_probe_set(1),
        signals=[_signal("S_explicit_legacy", "A reviewed migrated signal.")],
    )

    event = result.evidence_events[0]
    assert event.discard_reason is None
    assert state.task_frame.framing_trace["migration"] == migration_marker
    assert gateway.requests[0].schema_version == "v0.1"
    assert gateway.requests[0].metadata["judgment_route"] == "legacy_v0.1_migration"
    assert event.model_trace["metadata"]["judgment_route"] == (
        "legacy_v0.1_migration"
    )


@pytest.mark.parametrize("framing_method", _NONLEGACY_FRAMING_METHODS)
def test_migrated_marker_with_nonlegacy_method_rejects_before_evidence_side_effects(
    framing_method,
):
    state = _migrated_state("belief_state_v0.1_to_v0.2")
    state = state.model_copy(
        update={
            "task_frame": state.task_frame.model_copy(
                update={"framing_method": framing_method}
            )
        }
    )
    gateway = CountingGateway()
    normalizer = RecordingProvenanceNormalizer()
    prior_state = state.model_dump(mode="json")

    with pytest.raises(ValueError, match="invalid belief lifecycle"):
        EvidenceIntegrationGate(
            model_gateway=gateway,
            provenance_normalizer=normalizer,
        ).integrate(
            cycle=_cycle(1),
            belief_state=state,
            probe_set=_probe_set(1),
            signals=[_signal("S_migration_method_conflict", "Must not be judged.")],
        )

    assert gateway.requests == []
    assert normalizer.calls == []
    assert state.model_dump(mode="json") == prior_state


@pytest.mark.parametrize(
    "marker",
    [
        pytest.param("belief_state_v0.1_to_v0.2", id="recognized"),
        pytest.param("caller_asserted", id="fake"),
        pytest.param("", id="empty"),
        pytest.param(7, id="non_string"),
    ],
)
def test_native_migration_trace_key_rejects_before_evidence_side_effects(marker):
    state = _state()
    state = state.model_copy(
        update={
            "task_frame": state.task_frame.model_copy(
                update={
                    "framing_trace": {
                        **state.task_frame.framing_trace,
                        "migration": marker,
                    }
                }
            )
        }
    )
    gateway = CountingGateway()
    normalizer = RecordingProvenanceNormalizer()
    prior_state = state.model_dump(mode="json")

    with pytest.raises(ValueError, match="invalid belief lifecycle"):
        EvidenceIntegrationGate(
            model_gateway=gateway,
            provenance_normalizer=normalizer,
        ).integrate(
            cycle=_cycle(1),
            belief_state=state,
            probe_set=_probe_set(1),
            signals=[_signal("S_native_migration_trace", "Must not be judged.")],
        )

    assert gateway.requests == []
    assert normalizer.calls == []
    assert state.model_dump(mode="json") == prior_state


@pytest.mark.parametrize("invalid_envelope", _INVALID_MIGRATION_ENVELOPES)
def test_invalid_migration_envelope_rejects_before_provider_or_memory(
    invalid_envelope,
):
    gateway = CountingGateway()
    state = _invalid_migration_envelope(invalid_envelope)
    prior_state = state.model_dump(mode="json")

    with pytest.raises(ValueError, match="invalid belief lifecycle"):
        EvidenceIntegrationGate(model_gateway=gateway).integrate(
            cycle=_cycle(1),
            belief_state=state,
            probe_set=_probe_set(1),
            signals=[_signal("S_invalid_migration", "Must not be judged.")],
        )

    assert gateway.requests == []
    assert state.model_dump(mode="json") == prior_state


def test_unmigrated_v01_direct_gate_rejects_before_provider_or_memory():
    gateway = CountingGateway()
    state = _state().model_copy(update={"schema_version": "v0.1"})
    prior_memory = state.evidence_memory.model_dump(mode="json")

    with pytest.raises(ValueError, match="invalid belief lifecycle"):
        EvidenceIntegrationGate(model_gateway=gateway).integrate(
            cycle=_cycle(1),
            belief_state=state,
            probe_set=_probe_set(1),
            signals=[_signal("S_invalid_lifecycle", "Must not be judged.")],
        )

    assert gateway.requests == []
    assert state.evidence_memory.model_dump(mode="json") == prior_memory


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


def test_identity_write_rejects_unsupported_memory_version():
    snapshot = EvidenceMemorySnapshot.model_construct(memory_version=4)
    signal = SignalProvenanceNormalizer().normalize(
        _signal("S_unsupported_memory", "Unsupported memory version."),
        run_id="run_memory",
    )

    with pytest.raises(ValueError, match="unsupported evidence memory version"):
        EvidenceMemoryManager().remember_signal_identity(snapshot, signal)


def test_v3_commit_identity_persists_root_binding_without_credit():
    manager = EvidenceMemoryManager()
    signal = SignalProvenanceNormalizer().normalize(
        _signal("S_native_identity", "A native root-bound observation."),
        run_id="run_memory",
    )
    root_id = resolve_contribution_root_id(signal)
    event = EvidenceEvent(
        schema_version="v0.2",
        id="E_native_identity",
        derived_from_signal=signal.id,
        epistemic_origin=signal.provenance.epistemic_origin,
        derivation_root_id=signal.provenance.derivation_root_id,
        contribution_root_id=root_id,
        target_hypotheses=["A"],
        evidence_type=EvidenceType.SUPPORTING,
        content=signal.raw_content,
        likelihoods={"A": LikelihoodBand.MODERATELY_CONFIRMING},
        correlation_status="novel",
        effective_update_weight=None,
    )

    committed = manager.commit_identity(
        EvidenceMemorySnapshot(memory_version=3),
        signal=signal,
        event=event,
    )

    assert committed.memory_version == 3
    assert committed.accepted_evidence_ids == [event.id]
    assert committed.signal_contribution_roots == {signal.id: root_id}
    assert committed.correlation_credit == {}


def test_v3_classification_never_allocates_correlation_credit():
    signal = SignalProvenanceNormalizer().normalize(
        _signal("S_native_classification", "A native classified observation."),
        run_id="run_memory",
    )

    decision = EvidenceMemoryManager().classify(
        EvidenceMemorySnapshot(memory_version=3),
        signal,
        likelihoods={"A": LikelihoodBand.STRONGLY_CONFIRMING},
        base_effective_weight=1.0,
    )

    assert decision.correlation_status == "novel"
    assert decision.effective_update_weight == 0.0
    assert decision.remaining_credit == {}
    assert decision.discard_reason is None


def _v3_root_bound_event(
    signal: ExternalSignal,
    *,
    event_id: str,
    root_id: str,
    correlation_status: str,
    likelihood: LikelihoodBand = LikelihoodBand.MODERATELY_CONFIRMING,
) -> EvidenceEvent:
    return EvidenceEvent(
        schema_version="v0.2",
        id=event_id,
        derived_from_signal=signal.id,
        epistemic_origin=signal.provenance.epistemic_origin,
        derivation_root_id=signal.provenance.derivation_root_id,
        contribution_root_id=root_id,
        target_hypotheses=["A"],
        evidence_type=(
            EvidenceType.COUNTEREVIDENCE
            if likelihood
            in {
                LikelihoodBand.WEAKLY_DISCONFIRMING,
                LikelihoodBand.MODERATELY_DISCONFIRMING,
                LikelihoodBand.STRONGLY_DISCONFIRMING,
            }
            else EvidenceType.SUPPORTING
        ),
        content=signal.raw_content,
        reliability=0.8,
        independence=0.8,
        relevance=0.9,
        novelty=0.8,
        specificity=0.7,
        verifiability=0.7,
        likelihoods={"A": likelihood},
        correlation_status=correlation_status,
        effective_update_weight=None,
    )


def _normalized_same_source_signal(
    signal_id: str,
    content: str,
    *,
    correlation_group: str,
    derivation_root_id: str,
) -> ExternalSignal:
    raw = _signal(signal_id, content, root=derivation_root_id)
    raw = raw.model_copy(
        update={
            "provenance": raw.provenance.model_copy(
                update={"correlation_group": correlation_group}
            )
        }
    )
    return SignalProvenanceNormalizer().normalize(raw, run_id="run_memory")


def test_v3_same_cycle_same_source_uses_one_canonical_root_in_either_order():
    first = _normalized_same_source_signal(
        "S_canonical_a",
        "The first independently worded observation.",
        correlation_group="group:canonical-first",
        derivation_root_id="root:canonical-first",
    )
    second = _normalized_same_source_signal(
        "S_canonical_b",
        "The later independently worded observation.",
        correlation_group="group:raw-second",
        derivation_root_id="root:raw-second",
    )
    prior = EvidenceMemorySnapshot(memory_version=3)

    forward = resolve_signal_contribution_roots(prior, [first, second])
    reverse = resolve_signal_contribution_roots(prior, [second, first])

    expected_root = resolve_contribution_root_id(
        first,
        canonical_correlation_group="group:canonical-first",
    )
    assert forward == reverse
    assert forward.ordered_signal_ids == (first.id, second.id)
    assert forward.canonical_correlation_groups == {
        first.id: "group:canonical-first",
        second.id: "group:canonical-first",
    }
    assert set(forward.signal_contribution_roots.values()) == {expected_root}

    manager = EvidenceMemoryManager()
    memory = prior
    events = []
    for signal_id in forward.ordered_signal_ids:
        signal = {first.id: first, second.id: second}[signal_id]
        decision = manager.classify(memory, signal)
        assert decision.canonical_correlation_group == (
            forward.canonical_correlation_groups[signal_id]
        )
        assert decision.effective_update_weight == 0.0
        assert decision.remaining_credit == {}
        event = _v3_root_bound_event(
            signal,
            event_id=f"E_{signal_id}",
            root_id=forward.signal_contribution_roots[signal_id],
            correlation_status=decision.correlation_status,
        )
        memory = manager.commit_identity(memory, signal=signal, event=event)
        events.append(event)

    reconciled = EvidenceRootReconciler().reconcile_cycle(
        memory,
        events,
        falsification_probe_executed=False,
    )

    assert set(reconciled.evidence_memory.signal_contribution_roots.values()) == {
        expected_root
    }
    assert list(reconciled.evidence_memory.root_contributions) == [expected_root]
    assert reconciled.evidence_memory.correlation_credit == {}


def test_v3_transition_reconstructs_nonzero_root_create_and_revision():
    manager = EvidenceMemoryManager()
    prior = EvidenceMemorySnapshot(memory_version=3)
    first = _normalized_same_source_signal(
        "S_transition_active_first",
        "The first active transition observation.",
        correlation_group="group:persisted-canonical",
        derivation_root_id="root:transition-first",
    )
    first_resolution = resolve_signal_contribution_roots(prior, [first])
    root_id = first_resolution.signal_contribution_roots[first.id]
    first_decision = manager.classify(prior, first)
    first_event = _v3_root_bound_event(
        first,
        event_id="E_transition_active_first",
        root_id=root_id,
        correlation_status=first_decision.correlation_status,
    )
    first_identity = manager.commit_identity(
        prior,
        signal=first,
        event=first_event,
    )
    quality = 0.8 * 0.8 * 0.9 * 0.8
    first_contribution = EvidenceRootContribution(
        contribution_root_id=root_id,
        revision=1,
        assessment_event_ids=[first_event.id],
        epistemic_origin=first.provenance.epistemic_origin,
        per_hypothesis_log_likelihood={"A": quality * math.log(3.0)},
        active=True,
    )
    first_payload = first_identity.model_dump(mode="python")
    first_payload["root_contributions"] = {root_id: first_contribution}
    first_candidate = EvidenceMemorySnapshot.model_validate(first_payload)

    assert manager.validate_transition(
        prior,
        first_candidate,
        evidence_events=[first_event],
        normalized_signals=[first],
        existing_evidence_ids=[],
        frame_version=1,
    ) == first_candidate
    assert first_candidate.root_contributions[root_id].active is True
    assert first_candidate.root_contributions[
        root_id
    ].per_hypothesis_log_likelihood["A"] != 0.0

    second = _normalized_same_source_signal(
        "S_transition_active_second",
        "The revised active transition observation.",
        correlation_group="group:later-raw-declaration",
        derivation_root_id="root:transition-second",
    )
    second_resolution = resolve_signal_contribution_roots(
        first_candidate,
        [second],
    )
    assert second_resolution.signal_contribution_roots[second.id] == root_id
    assert second_resolution.canonical_correlation_groups[second.id] == (
        "group:persisted-canonical"
    )
    second_decision = manager.classify(first_candidate, second)
    second_event = _v3_root_bound_event(
        second,
        event_id="E_transition_active_second",
        root_id=root_id,
        correlation_status=second_decision.correlation_status,
        likelihood=LikelihoodBand.WEAKLY_DISCONFIRMING,
    )
    second_identity = manager.commit_identity(
        first_candidate,
        signal=second,
        event=second_event,
    )
    revised_contribution = EvidenceRootContribution(
        contribution_root_id=root_id,
        revision=2,
        assessment_event_ids=[second_event.id],
        epistemic_origin=second.provenance.epistemic_origin,
        per_hypothesis_log_likelihood={"A": quality * math.log(0.7)},
        active=True,
    )
    revised_payload = second_identity.model_dump(mode="python")
    revised_payload["root_contributions"] = {root_id: revised_contribution}
    revised_candidate = EvidenceMemorySnapshot.model_validate(revised_payload)

    assert manager.validate_transition(
        first_candidate,
        revised_candidate,
        evidence_events=[second_event],
        normalized_signals=[second],
        existing_evidence_ids=[first_event.id],
        frame_version=1,
    ) == revised_candidate

    forged_payload = revised_candidate.model_dump(mode="python")
    forged_payload["root_contributions"][root_id][
        "per_hypothesis_log_likelihood"
    ]["A"] += 0.1
    forged_candidate = EvidenceMemorySnapshot.model_validate(forged_payload)

    with pytest.raises(ValueError, match="evidence memory transition is invalid"):
        manager.validate_transition(
            first_candidate,
            forged_candidate,
            evidence_events=[second_event],
            normalized_signals=[second],
            existing_evidence_ids=[first_event.id],
            frame_version=1,
        )


def test_v3_commit_identity_rejects_changed_root_for_repeated_signal_id():
    manager = EvidenceMemoryManager()
    signal = SignalProvenanceNormalizer().normalize(
        _signal("S_repeated_root", "A repeated root-bound observation."),
        run_id="run_memory",
    )

    root_id = resolve_contribution_root_id(signal)

    def event(event_id, contribution_root_id):
        return EvidenceEvent(
            schema_version="v0.2",
            id=event_id,
            derived_from_signal=signal.id,
            epistemic_origin=signal.provenance.epistemic_origin,
            derivation_root_id=signal.provenance.derivation_root_id,
            contribution_root_id=contribution_root_id,
            target_hypotheses=["A"],
            evidence_type=EvidenceType.NEUTRAL,
            content=signal.raw_content,
            likelihoods={"A": LikelihoodBand.NEUTRAL},
            correlation_status="novel",
            effective_update_weight=None,
        )

    committed = manager.commit_identity(
        EvidenceMemorySnapshot(memory_version=3),
        signal=signal,
        event=event("E_repeated_root_1", root_id),
    )

    with pytest.raises(ValueError, match="signal contribution root conflict"):
        manager.commit_identity(
            committed,
            signal=signal,
            event=event(
                "E_repeated_root_2",
                "evidence-root:sha256:" + "b" * 64,
            ),
        )


def test_v3_commit_identity_rejects_unresolved_root_binding():
    signal = SignalProvenanceNormalizer().normalize(
        _signal("S_wrong_root", "An observation with the wrong root binding."),
        run_id="run_memory",
    )
    event = EvidenceEvent(
        schema_version="v0.2",
        id="E_wrong_root",
        derived_from_signal=signal.id,
        epistemic_origin=signal.provenance.epistemic_origin,
        derivation_root_id=signal.provenance.derivation_root_id,
        contribution_root_id="evidence-root:sha256:" + "f" * 64,
        target_hypotheses=["A"],
        evidence_type=EvidenceType.NEUTRAL,
        content=signal.raw_content,
        likelihoods={"A": LikelihoodBand.NEUTRAL},
        correlation_status="novel",
        effective_update_weight=None,
    )

    with pytest.raises(ValueError, match="signal contribution root conflict"):
        EvidenceMemoryManager().commit_identity(
            EvidenceMemorySnapshot(memory_version=3),
            signal=signal,
            event=event,
        )


def test_v3_commit_identity_rejects_child_whose_parent_is_not_bound():
    child = SignalProvenanceNormalizer().normalize(
        _derived_signal(
            "S_unbound_parent_child",
            "A summary whose parent root is unavailable.",
            parent_id="S_unbound_parent",
            root="root-unbound-parent",
        ),
        run_id="run_memory",
    )
    event = EvidenceEvent(
        schema_version="v0.2",
        id="E_unbound_parent_child",
        derived_from_signal=child.id,
        epistemic_origin=child.provenance.epistemic_origin,
        derivation_root_id=child.provenance.derivation_root_id,
        contribution_root_id="evidence-root:sha256:" + "e" * 64,
        target_hypotheses=["A"],
        evidence_type=EvidenceType.NEUTRAL,
        content=child.raw_content,
        likelihoods={"A": LikelihoodBand.NEUTRAL},
        correlation_status="correlated_restatement",
        effective_update_weight=None,
    )

    with pytest.raises(ValueError, match="missing a parent contribution root"):
        EvidenceMemoryManager().commit_identity(
            EvidenceMemorySnapshot(memory_version=3),
            signal=child,
            event=event,
        )


def test_v3_root_resolution_is_input_order_independent_for_same_cycle_parent():
    normalizer = SignalProvenanceNormalizer()
    parent = normalizer.normalize(
        _signal("S_root_parent", "A root observation.", root="root-shared"),
        run_id="run_memory",
    )
    child = normalizer.normalize(
        _derived_signal(
            "S_root_child",
            "A summary of the root observation.",
            parent_id=parent.id,
            root="root-shared",
        ),
        run_id="run_memory",
    )
    snapshot = EvidenceMemorySnapshot(memory_version=3)

    forward = resolve_signal_contribution_roots(snapshot, [parent, child])
    reversed_result = resolve_signal_contribution_roots(snapshot, [child, parent])

    expected_root = resolve_contribution_root_id(parent)
    assert forward == reversed_result
    assert forward.ordered_signal_ids == (parent.id, child.id)
    assert forward.signal_contribution_roots == {
        parent.id: expected_root,
        child.id: expected_root,
    }


def test_v3_transition_reconstruction_is_input_order_independent_for_parent_child():
    manager = EvidenceMemoryManager()
    normalizer = SignalProvenanceNormalizer()
    parent = normalizer.normalize(
        _signal(
            "S_transition_parent",
            "A transition root observation.",
            root="root-transition-shared",
        ),
        run_id="run_memory",
    )
    child = normalizer.normalize(
        _derived_signal(
            "S_transition_child",
            "A transition summary.",
            parent_id=parent.id,
            root="root-transition-shared",
        ),
        run_id="run_memory",
    )
    signals_by_id = {signal.id: signal for signal in (parent, child)}
    prior = EvidenceMemorySnapshot(memory_version=3)
    resolution = resolve_signal_contribution_roots(prior, [child, parent])
    expected = prior
    events_by_signal_id = {}
    for signal_id in resolution.ordered_signal_ids:
        signal = signals_by_id[signal_id]
        decision = manager.classify(expected, signal)
        event = EvidenceEvent(
            schema_version="v0.2",
            id=f"E_{signal_id}",
            derived_from_signal=signal.id,
            epistemic_origin=signal.provenance.epistemic_origin,
            derivation_root_id=signal.provenance.derivation_root_id,
            contribution_root_id=(
                resolution.signal_contribution_roots[signal.id]
            ),
            target_hypotheses=["A"],
            evidence_type=EvidenceType.NEUTRAL,
            content=signal.raw_content,
            reliability=0.0,
            independence=0.0,
            relevance=0.0,
            novelty=0.0,
            specificity=0.0,
            verifiability=0.0,
            likelihoods={"A": LikelihoodBand.NEUTRAL},
            correlation_status=decision.correlation_status,
            effective_update_weight=None,
        )
        events_by_signal_id[signal.id] = event
        expected = manager.commit_identity(
            expected,
            signal=signal,
            event=event,
        )
    expected = EvidenceRootReconciler().reconcile_cycle(
        expected,
        list(events_by_signal_id.values()),
        falsification_probe_executed=False,
    ).evidence_memory

    for signals in ([parent, child], [child, parent]):
        events = [events_by_signal_id[signal.id] for signal in signals]
        assert manager.validate_transition(
            prior,
            expected,
            evidence_events=events,
            normalized_signals=signals,
            existing_evidence_ids=[],
            frame_version=1,
        ) == expected


def test_v3_root_resolution_rejects_missing_parent():
    child = SignalProvenanceNormalizer().normalize(
        _derived_signal(
            "S_missing_parent_child",
            "A summary without its parent.",
            parent_id="S_missing_parent",
            root="root-missing-parent",
        ),
        run_id="run_memory",
    )

    with pytest.raises(ValueError, match="missing a parent signal"):
        resolve_signal_contribution_roots(
            EvidenceMemorySnapshot(memory_version=3),
            [child],
        )


def test_v3_root_resolution_rejects_parent_cycle():
    normalizer = SignalProvenanceNormalizer()
    first = normalizer.normalize(
        _derived_signal(
            "S_cycle_first",
            "The first cyclic summary.",
            parent_id="S_cycle_second",
            root="root-cycle",
        ),
        run_id="run_memory",
    )
    second = normalizer.normalize(
        _derived_signal(
            "S_cycle_second",
            "The second cyclic summary.",
            parent_id="S_cycle_first",
            root="root-cycle",
        ),
        run_id="run_memory",
    )

    with pytest.raises(ValueError, match="contains a cycle"):
        resolve_signal_contribution_roots(
            EvidenceMemorySnapshot(memory_version=3),
            [first, second],
        )


def test_v3_root_resolution_rejects_cycle_between_previously_bound_signal_ids():
    normalizer = SignalProvenanceNormalizer()
    first = normalizer.normalize(
        _signal("S_bound_cycle_first", "First bound signal.", root="root-cycle"),
        run_id="run_memory",
    )
    second = normalizer.normalize(
        _signal("S_bound_cycle_second", "Second bound signal.", root="root-cycle"),
        run_id="run_memory",
    )
    memory = EvidenceMemorySnapshot(memory_version=3)
    manager = EvidenceMemoryManager()
    for signal in (first, second):
        memory = manager.remember_signal_identity(memory, signal)
    payload = memory.model_dump(mode="python")
    root_id = resolve_contribution_root_id(first)
    payload["signal_contribution_roots"] = {
        first.id: root_id,
        second.id: root_id,
    }
    memory = EvidenceMemorySnapshot.model_validate(payload)
    first_cycle = normalizer.normalize(
        first.model_copy(
            update={
                "provenance": first.provenance.model_copy(
                    update={"parent_signal_ids": [second.id]}
                )
            }
        ),
        run_id="run_memory",
    )
    second_cycle = normalizer.normalize(
        second.model_copy(
            update={
                "provenance": second.provenance.model_copy(
                    update={"parent_signal_ids": [first.id]}
                )
            }
        ),
        run_id="run_memory",
    )

    with pytest.raises(ValueError, match="contains a cycle"):
        resolve_signal_contribution_roots(
            memory,
            [first_cycle, second_cycle],
        )


@pytest.mark.parametrize("reverse", [False, True])
def test_v3_root_resolution_rejects_duplicate_signal_id_definitions(reverse):
    normalizer = SignalProvenanceNormalizer()
    root_signal = normalizer.normalize(
        _signal(
            "S_duplicate_definition",
            "One signal id must have one root definition.",
        ),
        run_id="run_memory",
    )
    self_parent = normalizer.normalize(
        root_signal.model_copy(
            update={
                "provenance": root_signal.provenance.model_copy(
                    update={"parent_signal_ids": [root_signal.id]}
                )
            }
        ),
        run_id="run_memory",
    )
    signals = [root_signal, self_parent]
    if reverse:
        signals.reverse()

    with pytest.raises(ValueError, match="signal id lineage conflict"):
        resolve_signal_contribution_roots(
            EvidenceMemorySnapshot(memory_version=3),
            signals,
        )


def test_v3_root_resolution_rejects_conflicting_parent_roots():
    normalizer = SignalProvenanceNormalizer()
    first = normalizer.normalize(
        _signal("S_parent_first", "First root.", root="root-first").model_copy(
            update={
                "provenance": _signal(
                    "S_parent_first",
                    "First root.",
                    root="root-first",
                ).provenance.model_copy(
                    update={"epistemic_origin": EpistemicOrigin.EXTERNAL_OBSERVATION}
                )
            }
        ),
        run_id="run_memory",
    )
    second_raw = _signal("S_parent_second", "Second root.", root="root-second")
    second = normalizer.normalize(
        second_raw.model_copy(
            update={
                "provenance": second_raw.provenance.model_copy(
                    update={"epistemic_origin": EpistemicOrigin.EXTERNAL_OBSERVATION}
                )
            }
        ),
        run_id="run_memory",
    )
    child_raw = _derived_signal(
        "S_conflicting_child",
        "A summary joining incompatible roots.",
        parent_id=first.id,
        root="root-first",
    )
    child = normalizer.normalize(
        child_raw.model_copy(
            update={
                "provenance": child_raw.provenance.model_copy(
                    update={"parent_signal_ids": [first.id, second.id]}
                )
            }
        ),
        run_id="run_memory",
    )

    with pytest.raises(ValueError, match="exactly one parent contribution root"):
        resolve_signal_contribution_roots(
            EvidenceMemorySnapshot(memory_version=3),
            [child, second, first],
        )


def test_v3_root_resolution_rejects_parentless_derived_summary():
    raw = _derived_signal(
        "S_parentless_summary",
        "A parentless derived summary.",
        parent_id="S_removed_parent",
        root="root-parentless",
    )
    parentless = SignalProvenanceNormalizer().normalize(
        raw.model_copy(
            update={
                "provenance": raw.provenance.model_copy(
                    update={"parent_signal_ids": []}
                )
            }
        ),
        run_id="run_memory",
    )

    with pytest.raises(ValueError, match="derived summary requires parent signals"):
        resolve_signal_contribution_roots(
            EvidenceMemorySnapshot(memory_version=3),
            [parentless],
        )


def test_v1_identity_write_upgrades_all_identities_to_v2():
    fingerprint = "sha256:" + "a" * 64
    snapshot = EvidenceMemorySnapshot(
        memory_version=1,
        content_fingerprints={"S_legacy": fingerprint},
        source_content_fingerprints={
            "S_legacy": '["source.example/report","'
            + fingerprint
            + '","source.example/report"]'
        },
        derivation_roots={"S_legacy": "root-legacy"},
    )
    signal = SignalProvenanceNormalizer().normalize(
        _signal("S_upgrade", "Identity upgrade observation.", root="root-upgrade"),
        run_id="run_memory",
    )

    upgraded = EvidenceMemoryManager().remember_signal_identity(snapshot, signal)

    assert upgraded.memory_version == 2
    assert all(
        len(json.loads(identity)) == 4
        for identity in upgraded.source_content_fingerprints.values()
    )


@pytest.mark.parametrize("memory_version", [0, 4, 999])
def test_native_belief_state_rejects_unsupported_memory_version(memory_version):
    state = _state()
    payload = state.model_dump(mode="python")
    payload["evidence_memory"]["memory_version"] = memory_version

    with pytest.raises(ValueError, match="memory_version"):
        type(state).model_validate(payload)


def test_native_belief_state_rejects_correlation_credit():
    state = _state()
    payload = state.model_dump(mode="python")
    payload["evidence_memory"]["correlation_credit"] = {
        "group|UNKNOWN|confirming": 0.2
    }

    with pytest.raises(ValueError, match="memory v3 does not use correlation credit"):
        type(state).model_validate(payload)
