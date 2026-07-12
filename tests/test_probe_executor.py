import json
from pathlib import Path
import unicodedata

import pytest

from bayesprobe.core import BayesProbeCore
from bayesprobe.evidence import EvidenceIntegrationGate
from bayesprobe.evidence_memory import (
    EvidenceMemoryManager,
    SignalProvenanceNormalizer,
)
from bayesprobe.initialization import BayesProbeInitializer, HypothesisSeed, InitializeRunInput
from bayesprobe.ledger import JsonlLedgerStore
from bayesprobe.model_gateway import ScriptedModelGateway
from bayesprobe.openai_gateway import (
    OpenAIChatCompletionsModelGateway,
    OpenAIModelGatewayConfig,
    OpenAIResponsesModelGateway,
)
from bayesprobe.probe_executor import (
    DeterministicProbeToolGateway,
    ModelBackedProbeToolGateway,
    ProbeExecutionContext,
    ProbeExecutor,
)
from bayesprobe.recorded_gateway import RecordedModelGateway
from bayesprobe.probe_planner import ProbePlanner, ProbePlanningConfig
from bayesprobe.schemas import (
    AnswerChoice,
    BeliefState,
    CycleRecord,
    CycleSignalShape,
    EpistemicOrigin,
    EvidenceEvent,
    EvidenceMemorySnapshot,
    EvidenceType,
    ExternalSignal,
    FramingMethod,
    Hypothesis,
    LikelihoodBand,
    ProbeDesign,
    ProbeSet,
    SignalKind,
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
    "bare_v01",
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
_SECRET_MODEL_IDENTITIES = (
    "Authorization: Bearer provider-secret-value-123",
    (
        "\uff21\uff55\uff54\uff48\uff4f\uff52\uff49\uff5a\uff41\uff54"
        "\uff49\uff4f\uff4e\uff1a \uff22\uff45\uff41\uff52\uff45\uff52 "
        "provider-secret-value-123"
    ),
)
_NFKC_SENSITIVE_NAME = "\uff41\uff50\uff49\uff3f\uff4b\uff45\uff59"
_OPENAI_MODEL_IDENTITY_PREFIX = "openai_model_identity:v1:"


def parse_openai_model_identity(identity: str) -> dict[str, str]:
    assert identity.startswith(_OPENAI_MODEL_IDENTITY_PREFIX)
    encoded = identity.removeprefix(_OPENAI_MODEL_IDENTITY_PREFIX)
    payload = json.loads(encoded)
    assert list(payload) == ["adapter_kind", "model", "provider_origin"]
    assert encoded == json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return payload


def assert_sha256_identity(value: str, *, prefix: str) -> None:
    assert value.startswith(prefix)
    digest = value.removeprefix(prefix)
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")


class RecordingGateway:
    def __init__(self, signals_by_probe_id: dict[str, list[ExternalSignal]] | None = None):
        self.calls: list[str] = []
        self.signals_by_probe_id = signals_by_probe_id or {}

    def execute_probe(self, *, probe: ProbeDesign, context: ProbeExecutionContext) -> list[ExternalSignal]:
        self.calls.append(probe.id)
        return self.signals_by_probe_id.get(
            probe.id,
            [
                ExternalSignal(
                    id=f"S_gateway_{probe.id}",
                    cycle_id=context.cycle_id,
                    signal_kind=SignalKind.ACTIVE,
                    source_type="recording_gateway",
                    source=probe.method,
                    raw_content=f"SUPPORTS: gateway result for {probe.id}.",
                )
            ],
        )


def explicit_test_hypothesis_seeds() -> list[HypothesisSeed]:
    return [
        HypothesisSeed(id="H1", statement="The fixture's H1 condition holds.", prior=0.5, scope="Deterministic test fixture.", falsifiers=["The fixture emits a reliable H1 refutation."], predictions=["The fixture emits a reliable H1 support cue."]),
        HypothesisSeed(id="H2", statement="The fixture's H2 condition holds instead.", prior=0.5, scope="Deterministic test fixture.", falsifiers=["The fixture emits a reliable H2 refutation."], predictions=["The fixture emits a reliable H2 support cue."]),
    ]


class PassiveGateway:
    def execute_probe(self, *, probe: ProbeDesign, context: ProbeExecutionContext) -> list[ExternalSignal]:
        return [
            ExternalSignal(
                id="S_passive_bad",
                cycle_id=context.cycle_id,
                signal_kind=SignalKind.PASSIVE,
                source_type="bad_gateway",
                source=probe.method,
                raw_content="This should not be accepted as active execution output.",
            )
        ]


def make_belief_state() -> BeliefState:
    return BeliefState(
        belief_state_id="bs_exec",
        run_id="run_exec",
        cycle_id="cycle_0",
        hypotheses=[
            Hypothesis(
                id="H1",
                statement="The claim is supported.",
                scope="execution fixture",
                prior=0.5,
                posterior=0.5,
                rivals=["H2"],
                falsifiers=["Reliable counterevidence weakens H1."],
                predictions=["Support should be independently observable."],
            ),
            Hypothesis(
                id="H2",
                statement="The claim is refuted.",
                scope="execution fixture",
                prior=0.5,
                posterior=0.5,
                rivals=["H1"],
                falsifiers=["Reliable support weakens H2."],
                predictions=["Counterevidence should be independently observable."],
            ),
        ],
    )


def make_native_belief_state() -> BeliefState:
    return BayesProbeInitializer().initialize(
        InitializeRunInput(
            run_id="run_exec",
            problem="Which answer choice is correct?",
            hypothesis_seeds=explicit_test_hypothesis_seeds(),
        )
    ).belief_state


def make_migrated_belief_state(marker: str) -> BeliefState:
    payload = make_native_belief_state().model_dump(mode="python")
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


def make_invalid_migration_envelope(kind: str) -> BeliefState:
    native = make_native_belief_state()
    migrated = make_migrated_belief_state("belief_state_v0.1_to_v0.2")
    if kind == "bare_v01":
        return make_belief_state()
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


def make_probe(
    probe_id: str,
    target_hypotheses: list[str],
    *,
    cycle_id: str = "run_exec_cycle_1",
    method: str = "source_tracing",
) -> ProbeDesign:
    return ProbeDesign(
        id=probe_id,
        cycle_id=cycle_id,
        target_hypotheses=target_hypotheses,
        inquiry_goal=f"Probe {probe_id}.",
        method=method,
        support_condition={hypothesis_id: "Independent support appears." for hypothesis_id in target_hypotheses},
        weaken_condition={hypothesis_id: "Independent counterevidence appears." for hypothesis_id in target_hypotheses},
    )


def make_probe_set(
    probes: list[ProbeDesign],
    *,
    cycle_id: str = "run_exec_cycle_1",
    may_be_empty: bool = False,
) -> ProbeSet:
    return ProbeSet(
        probe_set_id=f"ps_{cycle_id}",
        cycle_id=cycle_id,
        probes=probes,
        selection_reason="fixture probe set",
        may_be_empty=may_be_empty,
    )


def make_context(cycle_id: str = "run_exec_cycle_1") -> ProbeExecutionContext:
    return ProbeExecutionContext(
        run_id="run_exec",
        cycle_id=cycle_id,
        belief_state=make_belief_state(),
    )


def test_executor_turns_probe_set_into_active_signals():
    probe_set = make_probe_set(
        [
            make_probe("P1", ["H1"]),
            make_probe("P2", ["H2"], method="counterevidence_scan"),
        ]
    )

    result = ProbeExecutor(DeterministicProbeToolGateway()).execute_probe_set(
        probe_set=probe_set,
        context=make_context(),
    )

    assert result.probe_set == probe_set
    assert result.executed_probe_ids == ["P1", "P2"]
    assert [signal.generated_by_probe for signal in result.signals] == ["P1", "P2"]
    assert [signal.initial_target_hypotheses for signal in result.signals] == [["H1"], ["H2"]]
    assert all(signal.signal_kind == SignalKind.ACTIVE for signal in result.signals)
    assert all(signal.cycle_id == "run_exec_cycle_1" for signal in result.signals)
    assert "SUPPORTS" in result.signals[0].raw_content
    assert "REFUTES" in result.signals[1].raw_content


def test_repeated_deterministic_probe_reuses_root_and_spends_no_fresh_credit():
    state = make_native_belief_state()
    first_probe = ProbeDesign(
        id="P_semantics_cycle_1",
        cycle_id="run_exec_cycle_1",
        target_hypotheses=["H2", "H1"],
        inquiry_goal="Compare the same audited observation.",
        method="source_tracing",
        probe_type="discriminative_test",
        support_condition={"H1": "Supports H1.", "H2": "Supports H2."},
        weaken_condition={"H1": "Weakens H1.", "H2": "Weakens H2."},
        reframe_condition={"unresolved": "Reframe if neither is explained."},
    )
    second_probe = first_probe.model_copy(
        update={
            "id": "P_semantics_cycle_2",
            "cycle_id": "run_exec_cycle_2",
            "target_hypotheses": ["H1", "H2"],
        }
    )

    def execute(probe: ProbeDesign, cycle_id: str):
        return ProbeExecutor(DeterministicProbeToolGateway()).execute_probe_set(
            probe_set=make_probe_set([probe], cycle_id=cycle_id),
            context=ProbeExecutionContext(
                run_id="run_exec",
                cycle_id=cycle_id,
                belief_state=state,
            ),
        ).signals[0]

    first_signal = execute(first_probe, "run_exec_cycle_1")
    second_signal = execute(second_probe, "run_exec_cycle_2")
    changed_signal = execute(
        second_probe.model_copy(
            update={"inquiry_goal": "Compute a materially different observation."}
        ),
        "run_exec_cycle_2",
    )

    assert first_signal.provenance.epistemic_origin == EpistemicOrigin.TOOL_RESULT
    assert first_signal.provenance.derivation_root_id == (
        second_signal.provenance.derivation_root_id
    )
    assert first_signal.provenance.derivation_root_id != (
        changed_signal.provenance.derivation_root_id
    )

    gate = EvidenceIntegrationGate()
    first = gate.integrate(
        cycle=CycleRecord(
            cycle_id="run_exec_cycle_1",
            run_id="run_exec",
            cycle_index=1,
            signal_shape=CycleSignalShape.ACTIVE_ONLY,
        ),
        belief_state=state,
        probe_set=make_probe_set([first_probe], cycle_id="run_exec_cycle_1"),
        signals=[first_signal],
    )
    state = state.model_copy(
        update={
            "evidence_memory": first.evidence_memory,
            "ledger_refs": {
                **state.ledger_refs,
                "evidence_events": [event.id for event in first.evidence_events],
            },
        }
    )
    repeated = gate.integrate(
        cycle=CycleRecord(
            cycle_id="run_exec_cycle_2",
            run_id="run_exec",
            cycle_index=2,
            signal_shape=CycleSignalShape.ACTIVE_ONLY,
        ),
        belief_state=state,
        probe_set=make_probe_set([second_probe], cycle_id="run_exec_cycle_2"),
        signals=[second_signal],
    )

    event = repeated.evidence_events[0]
    assert event.correlation_status == "correlated_restatement"
    assert event.independence == 0.0
    assert event.effective_update_weight == 0.0
    assert repeated.evidence_memory.correlation_credit == (
        first.evidence_memory.correlation_credit
    )


def test_model_backed_probe_gateway_turns_model_result_into_active_signal():
    model_gateway = ScriptedModelGateway(
        responses={
            "execute_probe": {
                "raw_content": (
                    "A direct comparison supports H1 and rules out H2."
                )
            }
        }
    )
    probe = make_probe(
        "P_choice",
        ["H1", "H2"],
        method="answer_choice_discrimination",
    )
    initialized = BayesProbeInitializer().initialize(
        InitializeRunInput(
            run_id="run_exec",
            problem="Which answer choice is correct?",
            hypothesis_seeds=explicit_test_hypothesis_seeds(),
        )
    )
    context = ProbeExecutionContext(
        run_id="run_exec",
        cycle_id="run_exec_cycle_1",
        belief_state=initialized.belief_state,
        metadata={
            "problem": "Which answer choice is correct?",
            "task_context": "Use graph-chain irreducibility and aperiodicity.",
            "initial_context": "SUPPORTS: This initial signal must remain outside model execution.",
        },
    )

    result = ProbeExecutor(
        ModelBackedProbeToolGateway(model_gateway)
    ).execute_probe_set(
        probe_set=make_probe_set([probe]),
        context=context,
    )

    signal = result.signals[0]
    request = model_gateway.requests[0]
    assert request.task == "execute_probe"
    assert request.prompt_id == "probe_execution"
    assert request.schema_name == "ProbeSignal"
    assert request.prompt_version == "v0.2"
    assert request.schema_version == "v0.2"
    assert request.input["problem"] == "Which answer choice is correct?"
    assert request.input["task_context"] == (
        "Use graph-chain irreducibility and aperiodicity."
    )
    assert "initial_context" not in request.input
    assert "SUPPORTS: This initial signal" not in json.dumps(request.input)
    assert request.input["probe"]["target_hypotheses"] == ["H1", "H2"]
    assert request.input["hypotheses"][0]["statement"] == (
        "The fixture's H1 condition holds."
    )
    assert signal.signal_kind == SignalKind.ACTIVE
    assert signal.source_type == "model_probe_gateway"
    assert signal.source == "model_gateway:scripted"
    assert signal.raw_content.startswith("A direct comparison")
    assert signal.provenance.provider_model_or_tool_identity == "scripted"
    assert signal.provenance.session_id == "run_exec"


@pytest.mark.parametrize("model_identity", _SECRET_MODEL_IDENTITIES)
def test_model_backed_probe_rejects_secret_identity_before_provider_call(
    model_identity,
):
    model_gateway = ScriptedModelGateway(
        responses={
            "execute_probe": {"raw_content": "This must not be requested."}
        }
    )
    model_gateway.model_identity = model_identity
    state = make_native_belief_state()

    with pytest.raises(ValueError, match="model gateway identity") as exc_info:
        ModelBackedProbeToolGateway(model_gateway).execute_probe(
            probe=make_probe("P_secret_identity", ["H1", "H2"]),
            context=ProbeExecutionContext(
                run_id="run_exec",
                cycle_id="run_exec_cycle_1",
                belief_state=state,
            ),
        )

    error_text = str(exc_info.value)
    assert model_identity not in error_text
    assert unicodedata.normalize("NFKC", model_identity) not in error_text
    assert model_gateway.requests == []


def test_model_backed_probe_rejects_sensitive_session_before_provider_call():
    model_gateway = ScriptedModelGateway(
        responses={
            "execute_probe": {"raw_content": "This must not be requested."}
        }
    )
    state = make_native_belief_state()
    prior_state = state.model_dump(mode="json")

    with pytest.raises(ValueError, match="model provenance") as exc_info:
        ModelBackedProbeToolGateway(model_gateway).execute_probe(
            probe=make_probe("P_sensitive_session", ["H1", "H2"]),
            context=ProbeExecutionContext(
                run_id=_NFKC_SENSITIVE_NAME,
                cycle_id="run_exec_cycle_1",
                belief_state=state,
            ),
        )

    error_text = str(exc_info.value)
    assert _NFKC_SENSITIVE_NAME not in error_text
    assert "api_key" not in error_text
    assert model_gateway.requests == []
    assert state.model_dump(mode="json") == prior_state


@pytest.mark.parametrize("invalid_envelope", _INVALID_MIGRATION_ENVELOPES)
def test_model_backed_probe_gateway_rejects_invalid_migration_envelope(
    invalid_envelope,
):
    model_gateway = ScriptedModelGateway(
        responses={"execute_probe": {"raw_content": "Must not execute."}}
    )
    gateway = ModelBackedProbeToolGateway(model_gateway)
    state = make_invalid_migration_envelope(invalid_envelope)
    prior_state = state.model_dump(mode="json")

    with pytest.raises(ValueError, match="invalid belief lifecycle"):
        gateway.execute_probe(
            probe=make_probe("P_invalid_migration", ["H1", "H2"]),
            context=ProbeExecutionContext(
                run_id="run_exec",
                cycle_id="run_exec_cycle_1",
                belief_state=state,
            ),
        )

    assert model_gateway.requests == []
    assert state.model_dump(mode="json") == prior_state


@pytest.mark.parametrize("migration_marker", _MIGRATION_MARKERS)
def test_model_backed_probe_gateway_uses_v01_only_for_explicit_migration(
    migration_marker,
):
    model_gateway = ScriptedModelGateway(
        responses={"execute_probe": {"raw_content": "Legacy migrated execution."}}
    )
    migrated = make_migrated_belief_state(migration_marker)

    ModelBackedProbeToolGateway(model_gateway).execute_probe(
        probe=make_probe("P_migrated", ["H1", "H2"]),
        context=ProbeExecutionContext(
            run_id="run_exec",
            cycle_id="run_exec_cycle_1",
            belief_state=migrated,
        ),
    )

    assert migrated.task_frame.framing_method.value == "legacy_migration"
    assert migrated.task_frame.framing_trace["migration"] == migration_marker
    assert model_gateway.requests[0].prompt_version == "v0.1"
    assert model_gateway.requests[0].schema_version == "v0.1"


@pytest.mark.parametrize("framing_method", _NONLEGACY_FRAMING_METHODS)
def test_model_backed_probe_rejects_migrated_marker_with_nonlegacy_method(
    framing_method,
):
    state = make_migrated_belief_state("belief_state_v0.1_to_v0.2")
    state = state.model_copy(
        update={
            "task_frame": state.task_frame.model_copy(
                update={"framing_method": framing_method}
            )
        }
    )
    model_gateway = ScriptedModelGateway(
        responses={"execute_probe": {"raw_content": "Must not execute."}}
    )
    prior_state = state.model_dump(mode="json")

    with pytest.raises(ValueError, match="invalid belief lifecycle"):
        ModelBackedProbeToolGateway(model_gateway).execute_probe(
            probe=make_probe("P_migration_method_conflict", ["H1", "H2"]),
            context=ProbeExecutionContext(
                run_id="run_exec",
                cycle_id="run_exec_cycle_1",
                belief_state=state,
            ),
        )

    assert model_gateway.requests == []
    assert state.model_dump(mode="json") == prior_state


def test_model_backed_probe_gateway_uses_task_frame_context_when_metadata_is_empty():
    model_gateway = ScriptedModelGateway(
        responses={"execute_probe": {"raw_content": "A controlled comparison is required."}}
    )
    initialized = BayesProbeInitializer().initialize(
        InitializeRunInput(
            run_id="run_task_context_fallback",
            problem="Which answer choice is correct?",
            task_context="Use the supplied theorem definitions.",
            answer_choices=[
                AnswerChoice(label="A", text="First result"),
                AnswerChoice(label="B", text="Second result"),
            ],
        )
    )
    probe = make_probe("P_choice", ["A", "B"], method="answer_choice_discrimination")

    result = ProbeExecutor(ModelBackedProbeToolGateway(model_gateway)).execute_probe_set(
        probe_set=make_probe_set([probe]),
        context=ProbeExecutionContext(
            run_id="run_task_context_fallback",
            cycle_id="run_exec_cycle_1",
            belief_state=initialized.belief_state,
            metadata={"problem": "Which answer choice is correct?"},
        ),
    )

    assert model_gateway.requests[0].input["task_context"] == (
        "Use the supplied theorem definitions."
    )
    assert model_gateway.requests[0].prompt_version == "v0.2"
    assert model_gateway.requests[0].schema_version == "v0.2"
    assert result.signals[0].provenance.provider_model_or_tool_identity == "scripted"


@pytest.mark.parametrize(
    ("gateway_type", "adapter_kind"),
    [
        (OpenAIResponsesModelGateway, "openai"),
        (
            OpenAIChatCompletionsModelGateway,
            "openai_chat_completions",
        ),
    ],
)
def test_model_probe_provenance_uses_injective_openai_component_identity(
    gateway_type,
    adapter_kind,
):
    class StubOpenAICompatibleGateway(gateway_type):
        def complete_structured(self, request):
            return {"raw_content": "A model-backed probe observation."}

    probe = make_probe("P_model_identity", ["H1", "H2"])
    probe_set = make_probe_set([probe])
    initialized = BayesProbeInitializer().initialize(
        InitializeRunInput(
            run_id="run_exec",
            problem="Which model-backed claim is supported?",
            hypothesis_seeds=explicit_test_hypothesis_seeds(),
        )
    )
    context = ProbeExecutionContext(
        run_id="run_exec",
        cycle_id="run_exec_cycle_1",
        belief_state=initialized.belief_state,
    )

    def execute(model: str, base_url: str):
        gateway = StubOpenAICompatibleGateway(
            config=OpenAIModelGatewayConfig(model=model, base_url=base_url)
        )
        return ProbeExecutor(
            ModelBackedProbeToolGateway(gateway)
        ).execute_probe_set(
            probe_set=probe_set,
            context=context,
        ).signals[0]

    first = execute("model-a", "https://provider.example:8443/v1")
    same_provider_model = execute(
        "model-a",
        (
            "HTTPS://user:ignored@PROVIDER.EXAMPLE:8443/other"
            "?ignored=value#fragment"
        ),
    )
    boundary_distinct = execute(
        "8443:model-a",
        "https://provider.example/v1",
    )
    different_provider = execute(
        "model-a",
        "https://other.example:8443/v1",
    )
    normalizer = SignalProvenanceNormalizer()
    first = normalizer.normalize(first, run_id=context.run_id)
    same_provider_model = normalizer.normalize(
        same_provider_model,
        run_id=context.run_id,
    )
    boundary_distinct = normalizer.normalize(
        boundary_distinct,
        run_id=context.run_id,
    )
    different_provider = normalizer.normalize(
        different_provider,
        run_id=context.run_id,
    )

    assert first.source == different_provider.source == (
        f"model_gateway:{adapter_kind}"
    )
    first_identity = first.provenance.provider_model_or_tool_identity
    assert parse_openai_model_identity(first_identity) == {
        "adapter_kind": adapter_kind,
        "model": "model-a",
        "provider_origin": "https://provider.example:8443",
    }
    assert_sha256_identity(
        first.provenance.source_identity,
        prefix="model-source:sha256:",
    )
    assert first.provenance.source_identity == (
        same_provider_model.provenance.source_identity
    )
    assert first.provenance.source_identity != (
        boundary_distinct.provenance.source_identity
    )
    assert first_identity == (
        same_provider_model.provenance.provider_model_or_tool_identity
    )
    assert first_identity != (
        boundary_distinct.provenance.provider_model_or_tool_identity
    )
    assert first.provenance.correlation_group == (
        same_provider_model.provenance.correlation_group
    )
    assert first.provenance.correlation_group != (
        boundary_distinct.provenance.correlation_group
    )
    assert first.provenance.correlation_group != (
        different_provider.provenance.correlation_group
    )


@pytest.mark.parametrize(
    ("gateway_type", "adapter_kind"),
    [
        (OpenAIResponsesModelGateway, "openai"),
        (
            OpenAIChatCompletionsModelGateway,
            "openai_chat_completions",
        ),
    ],
)
def test_pipe_bearing_openai_model_uses_preflight_machine_provenance(
    gateway_type,
    adapter_kind,
):
    class StubOpenAICompatibleGateway(gateway_type):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.requests = []

        def complete_structured(self, request):
            self.requests.append(request)
            return {"raw_content": "One pipe-bearing model observation."}

    initialized = BayesProbeInitializer().initialize(
        InitializeRunInput(
            run_id="run_pipe_model",
            problem="Which pipe-bearing model observation is supported?",
            hypothesis_seeds=explicit_test_hypothesis_seeds(),
        )
    )
    gateway = StubOpenAICompatibleGateway(
        config=OpenAIModelGatewayConfig(
            model="provider|model",
            base_url="https://provider.example/v1",
        )
    )
    signal = ModelBackedProbeToolGateway(gateway).execute_probe(
        probe=make_probe("P_pipe_model", ["H1", "H2"]),
        context=ProbeExecutionContext(
            run_id="run_pipe_model",
            cycle_id="run_pipe_model_cycle_1",
            belief_state=initialized.belief_state,
        ),
    )[0]

    audit_identity = signal.provenance.provider_model_or_tool_identity
    assert len(gateway.requests) == 1
    assert parse_openai_model_identity(audit_identity) == {
        "adapter_kind": adapter_kind,
        "model": "provider|model",
        "provider_origin": "https://provider.example",
    }
    assert_sha256_identity(
        signal.provenance.source_identity,
        prefix="model-source:sha256:",
    )
    assert_sha256_identity(
        signal.provenance.correlation_group,
        prefix="model:sha256:",
    )
    assert audit_identity not in signal.provenance.source_identity
    assert audit_identity not in signal.provenance.correlation_group
    assert "|" not in signal.provenance.source_identity
    assert "|" not in signal.provenance.correlation_group

    normalized = SignalProvenanceNormalizer().normalize(
        signal,
        run_id="run_pipe_model",
    )

    assert normalized.provenance.provider_model_or_tool_identity == audit_identity
    assert normalized.provenance.source_identity == signal.provenance.source_identity
    assert normalized.provenance.correlation_group == (
        signal.provenance.correlation_group
    )


@pytest.mark.parametrize(
    ("first_model", "distinct_model"),
    [
        ("K", "\u212a"),
        ("model  alpha", "model alpha"),
    ],
)
@pytest.mark.parametrize(
    ("gateway_type", "adapter_kind"),
    [
        (OpenAIResponsesModelGateway, "openai"),
        (
            OpenAIChatCompletionsModelGateway,
            "openai_chat_completions",
        ),
    ],
)
def test_exact_openai_model_identity_survives_normalization_and_memory(
    gateway_type,
    adapter_kind,
    first_model,
    distinct_model,
):
    class StubOpenAICompatibleGateway(gateway_type):
        def complete_structured(self, request):
            return {"raw_content": "One exact model observation."}

    initialized = BayesProbeInitializer().initialize(
        InitializeRunInput(
            run_id="run_exact_model_identity",
            problem="Which model observation is independently sourced?",
            hypothesis_seeds=explicit_test_hypothesis_seeds(),
        )
    )

    def execute(model: str, base_url: str, index: int) -> ExternalSignal:
        cycle_id = f"run_exact_model_identity_cycle_{index}"
        probe = make_probe(
            f"P_exact_model_{index}",
            ["H1", "H2"],
            cycle_id=cycle_id,
        )
        signal = ProbeExecutor(
            ModelBackedProbeToolGateway(
                StubOpenAICompatibleGateway(
                    config=OpenAIModelGatewayConfig(
                        model=model,
                        base_url=base_url,
                    )
                )
            )
        ).execute_probe_set(
            probe_set=make_probe_set([probe], cycle_id=cycle_id),
            context=ProbeExecutionContext(
                run_id="run_exact_model_identity",
                cycle_id=cycle_id,
                belief_state=initialized.belief_state,
            ),
        ).signals[0]
        return SignalProvenanceNormalizer().normalize(
            signal,
            run_id="run_exact_model_identity",
        )

    first = execute(
        first_model,
        "https://provider.example/v1",
        1,
    )
    distinct = execute(
        distinct_model,
        "https://provider.example/v1",
        2,
    )
    equivalent = execute(
        first_model,
        (
            "HTTPS://user:ignored@PROVIDER.EXAMPLE:443/other"
            "?ignored=value#fragment"
        ),
        3,
    )

    first_provider = first.provenance.provider_model_or_tool_identity
    distinct_provider = distinct.provenance.provider_model_or_tool_identity
    equivalent_provider = equivalent.provenance.provider_model_or_tool_identity
    assert parse_openai_model_identity(first_provider) == {
        "adapter_kind": adapter_kind,
        "model": first_model,
        "provider_origin": "https://provider.example",
    }
    assert parse_openai_model_identity(distinct_provider) == {
        "adapter_kind": adapter_kind,
        "model": distinct_model,
        "provider_origin": "https://provider.example",
    }
    assert first_provider == equivalent_provider
    assert first_provider != distinct_provider
    assert first.provenance.source_identity == equivalent.provenance.source_identity
    assert first.provenance.source_identity != distinct.provenance.source_identity
    assert first.provenance.correlation_group == equivalent.provenance.correlation_group
    assert first.provenance.correlation_group != distinct.provenance.correlation_group
    assert first.provenance.canonical_content_fingerprint == (
        equivalent.provenance.canonical_content_fingerprint
    )
    assert first.provenance.canonical_content_fingerprint != (
        distinct.provenance.canonical_content_fingerprint
    )
    for normalized in (first, distinct, equivalent):
        assert_sha256_identity(
            normalized.provenance.source_identity,
            prefix="model-source:sha256:",
        )
        assert_sha256_identity(
            normalized.provenance.correlation_group,
            prefix="model:sha256:",
        )
        assert "|" not in normalized.provenance.correlation_group

    manager = EvidenceMemoryManager()

    def remember(
        memory: EvidenceMemorySnapshot,
        signal: ExternalSignal,
        index: int,
    ):
        decision = manager.classify(
            memory,
            signal,
            likelihoods={"H1": LikelihoodBand.MODERATELY_CONFIRMING},
            base_effective_weight=0.1,
        )
        event = EvidenceEvent(
            id=f"E_exact_model_{index}",
            derived_from_signal=signal.id,
            target_hypotheses=["H1"],
            evidence_type=EvidenceType.SUPPORTING,
            content=signal.raw_content,
            likelihoods={"H1": LikelihoodBand.MODERATELY_CONFIRMING},
            correlation_status=decision.correlation_status,
            effective_update_weight=decision.effective_update_weight,
        )
        return decision, manager.commit(
            memory,
            signal=signal,
            event=event,
            decision=decision,
        )

    first_decision, memory = remember(EvidenceMemorySnapshot(), first, 1)
    distinct_decision, memory = remember(memory, distinct, 2)
    equivalent_decision, memory = remember(memory, equivalent, 3)

    assert first_decision.correlation_status == "novel"
    assert distinct_decision.correlation_status == "novel"
    assert equivalent_decision.correlation_status == "correlated_novel"
    first_memory_identity = json.loads(
        memory.source_content_fingerprints[first.id]
    )
    distinct_memory_identity = json.loads(
        memory.source_content_fingerprints[distinct.id]
    )
    equivalent_memory_identity = json.loads(
        memory.source_content_fingerprints[equivalent.id]
    )
    assert first_memory_identity[:3] == equivalent_memory_identity[:3]
    assert first_memory_identity[0] != distinct_memory_identity[0]
    assert first_memory_identity[1] != distinct_memory_identity[1]
    assert first_memory_identity[2] != distinct_memory_identity[2]
    assert {
        key.partition("|")[0] for key in memory.correlation_credit
    } == {
        first.provenance.correlation_group,
        distinct.provenance.correlation_group,
    }


def test_recorded_probe_provenance_distinguishes_fixture_and_model_identity():
    initialized = BayesProbeInitializer().initialize(
        InitializeRunInput(
            run_id="run_recorded_identity",
            problem="Which fixture-backed claim is supported?",
            hypothesis_seeds=explicit_test_hypothesis_seeds(),
        )
    )
    context = ProbeExecutionContext(
        run_id="run_recorded_identity",
        cycle_id="run_exec_cycle_1",
        belief_state=initialized.belief_state,
        metadata={"problem": "Which fixture-backed claim is supported?"},
    )
    probe = make_probe("P_recorded", ["H1", "H2"])
    response = [
        {
            "match": {"task": "execute_probe"},
            "response": {"raw_content": "A recorded observation supports H1."},
        }
    ]

    def execute(fixture_name, model):
        gateway = RecordedModelGateway(
            fixture_name=fixture_name,
            responses=response,
            metadata={"provider_kind": "recorded-provider", "model": model},
        )
        return ProbeExecutor(ModelBackedProbeToolGateway(gateway)).execute_probe_set(
            probe_set=make_probe_set([probe]),
            context=context,
        ).signals[0]

    first = execute("fixture-a", "model-a")
    same = execute("fixture-a", "model-a")
    different_fixture = execute("fixture-b", "model-a")
    different_model = execute("fixture-a", "model-b")

    assert first.provenance.correlation_group == same.provenance.correlation_group
    assert first.provenance.correlation_group != (
        different_fixture.provenance.correlation_group
    )
    assert first.provenance.correlation_group != (
        different_model.provenance.correlation_group
    )


def test_executor_preserves_probe_and_signal_order():
    p1_s1 = ExternalSignal(
        id="S_P1_1",
        cycle_id="placeholder",
        signal_kind=SignalKind.ACTIVE,
        source_type="fixture_gateway",
        source="fixture",
        raw_content="SUPPORTS: first P1 signal.",
    )
    p1_s2 = ExternalSignal(
        id="S_P1_2",
        cycle_id="placeholder",
        signal_kind=SignalKind.ACTIVE,
        source_type="fixture_gateway",
        source="fixture",
        raw_content="SUPPORTS: second P1 signal.",
    )
    p2_s1 = ExternalSignal(
        id="S_P2_1",
        cycle_id="placeholder",
        signal_kind=SignalKind.ACTIVE,
        source_type="fixture_gateway",
        source="fixture",
        raw_content="SUPPORTS: first P2 signal.",
    )
    gateway = RecordingGateway({"P1": [p1_s1, p1_s2], "P2": [p2_s1]})
    probe_set = make_probe_set([make_probe("P1", ["H1"]), make_probe("P2", ["H2"])])

    result = ProbeExecutor(gateway).execute_probe_set(
        probe_set=probe_set,
        context=make_context(),
    )

    assert gateway.calls == ["P1", "P2"]
    assert result.executed_probe_ids == ["P1", "P2"]
    assert [signal.id for signal in result.signals] == ["S_P1_1", "S_P1_2", "S_P2_1"]
    assert [signal.generated_by_probe for signal in result.signals] == ["P1", "P1", "P2"]


def test_executor_returns_empty_result_for_empty_probe_set():
    gateway = RecordingGateway()
    probe_set = make_probe_set([], may_be_empty=True)

    result = ProbeExecutor(gateway).execute_probe_set(
        probe_set=probe_set,
        context=make_context(),
    )

    assert gateway.calls == []
    assert result.signals == []
    assert result.executed_probe_ids == []
    assert result.probe_set == probe_set


def test_executor_rejects_probe_set_cycle_mismatch():
    probe_set = make_probe_set([make_probe("P1", ["H1"])], cycle_id="run_exec_cycle_1")

    with pytest.raises(ValueError):
        ProbeExecutor(RecordingGateway()).execute_probe_set(
            probe_set=probe_set,
            context=make_context(cycle_id="run_exec_cycle_2"),
        )


def test_executor_rejects_passive_gateway_signals():
    probe_set = make_probe_set([make_probe("P1", ["H1"])])

    with pytest.raises(ValueError):
        ProbeExecutor(PassiveGateway()).execute_probe_set(
            probe_set=probe_set,
            context=make_context(),
        )


def test_executor_normalizes_gateway_signals_without_mutating_originals():
    original_signal = ExternalSignal(
        id="S_original",
        cycle_id="placeholder",
        signal_kind=SignalKind.ACTIVE,
        source_type="fixture_gateway",
        source="fixture",
        raw_content="SUPPORTS: raw gateway signal.",
        generated_by_probe=None,
        initial_target_hypotheses=["stale"],
    )
    probe_set = make_probe_set([make_probe("P1", ["H1"])])
    gateway = RecordingGateway({"P1": [original_signal]})

    result = ProbeExecutor(gateway).execute_probe_set(
        probe_set=probe_set,
        context=make_context(),
    )

    normalized = result.signals[0]
    assert normalized is not original_signal
    assert normalized.id == "S_original"
    assert normalized.cycle_id == "run_exec_cycle_1"
    assert normalized.generated_by_probe == "P1"
    assert normalized.initial_target_hypotheses == ["H1"]
    assert original_signal.cycle_id == "placeholder"
    assert original_signal.generated_by_probe is None
    assert original_signal.initial_target_hypotheses == ["stale"]


def test_executor_writes_only_execution_diagnostics_to_ledger(tmp_path: Path):
    ledger = JsonlLedgerStore(tmp_path / "executor-ledger.jsonl")
    probe_set = make_probe_set([make_probe("P1", ["H1"])])

    ProbeExecutor(DeterministicProbeToolGateway(), ledger=ledger).execute_probe_set(
        probe_set=probe_set,
        context=make_context(),
    )

    record_types = [record["record_type"] for record in ledger.read_all()]
    assert record_types == ["probe_execution"]
    assert "external_signal" not in record_types
    assert "evidence_event" not in record_types
    assert "belief_update" not in record_types
    assert "hypothesis_evolution" not in record_types
    assert "answer_projection" not in record_types


def test_planned_probe_set_executes_and_integrates_through_core():
    initialization = BayesProbeInitializer().initialize(
        InitializeRunInput(
            run_id="run_full_active_path",
            problem="Can the active path produce signals for the core?",
            hypothesis_seeds=explicit_test_hypothesis_seeds(),
        )
    )
    cycle = CycleRecord(
        cycle_id="run_full_active_path_cycle_1",
        run_id="run_full_active_path",
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
    execution = ProbeExecutor(DeterministicProbeToolGateway()).execute_probe_set(
        probe_set=planning.probe_set,
        context=ProbeExecutionContext(
            run_id=initialization.run.run_id,
            cycle_id=cycle.cycle_id,
            belief_state=initialization.belief_state,
        ),
    )

    result = BayesProbeCore().integrate_cycle(
        cycle=cycle,
        belief_state=initialization.belief_state,
        probe_set=planning.probe_set,
        signals=execution.signals,
    )

    assert execution.signals
    assert result.evidence_events
    assert result.belief_updates
    assert result.belief_state.cycle_id == cycle.cycle_id
