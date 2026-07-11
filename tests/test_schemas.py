import pytest

import bayesprobe.schemas as schemas
import bayesprobe

from bayesprobe.schemas import (
    AnswerContractOutline,
    AnswerRelationship,
    AnswerValueType,
    BeliefState,
    AnswerContract,
    ChangeMyMindCondition,
    CycleRecord,
    CycleSignalShape,
    EpistemicOrigin,
    EvidenceMemorySnapshot,
    EvidenceEvent,
    EvidenceType,
    ExternalSignal,
    FrameAdequacyStatus,
    FrameState,
    FramedHypothesis,
    FramingMethod,
    HypothesisCompetition,
    HypothesisCoverage,
    Hypothesis,
    HypothesisFrame,
    HypothesisRelation,
    HypothesisStatus,
    LikelihoodBand,
    ProbeCandidate,
    ProbeDesign,
    ProbeSet,
    RunRecord,
    RunRegime,
    SignalKind,
    SignalProvenance,
    TaskFrame,
    TaskAdmissionDecision,
    TaskAdmissionStatus,
    TaskKind,
    is_forbidden_secret_key_name,
    is_secret_like_value,
)


def make_v02_task_frame(
    *,
    task_kind: TaskKind = TaskKind.EXPLANATION,
    competition: HypothesisCompetition = HypothesisCompetition.EXCLUSIVE,
    coverage: HypothesisCoverage = HypothesisCoverage.OPEN,
    priors: list[float] | None = None,
    unresolved: float | None = 0.5,
) -> TaskFrame:
    prior_values = priors or [0.25, 0.25]
    hypotheses = [
        FramedHypothesis(
            id=f"H{index}",
            statement=f"Candidate {index} explains the observation.",
            type="candidate",
            scope="The stated task.",
            initial_prior=prior,
            falsifiers=[f"Candidate {index} is contradicted."],
            predictions=[f"Candidate {index} predicts the observation."],
            answer_value=f"candidate-{index}",
        )
        for index, prior in enumerate(prior_values, start=1)
    ]
    ids = [hypothesis.id for hypothesis in hypotheses]
    rivals = (
        {item: [other for other in ids if other != item] for item in ids}
        if competition == HypothesisCompetition.EXCLUSIVE
        else {item: [] for item in ids}
    )
    return TaskFrame(
        schema_version="v0.2",
        task_frame_id="v02_task_frame",
        admission_decision_id="admission_1",
        task_kind=task_kind,
        answer_relationship=AnswerRelationship.SELECTION,
        normalized_question="Which candidate answers the question?",
        answer_contract=AnswerContract(
            objective="Select the best supported candidate.",
            answer_value_type=AnswerValueType.SHORT_TEXT,
            answer_format="A short candidate value.",
            required_sections=["answer", "uncertainty"],
            decision_form="candidate_selection",
            permits_synthesis=False,
        ),
        hypothesis_frame=HypothesisFrame(
            frame_id="v02_hypothesis_frame",
            competition=competition,
            coverage=coverage,
            hypotheses=hypotheses,
            rival_sets=rivals,
            coverage_statement="The named candidates are provisional.",
            unresolved_alternative_mass=unresolved,
        ),
        framing_method=FramingMethod.EXPLICIT,
    )


def test_exact_answer_frame_is_exclusive_open_with_unresolved_mass():
    frame = make_v02_task_frame(
        task_kind=TaskKind.EXACT_ANSWER,
        competition=HypothesisCompetition.EXCLUSIVE,
        coverage=HypothesisCoverage.OPEN,
        priors=[0.25, 0.25],
        unresolved=0.50,
    )

    assert frame.answer_relationship == AnswerRelationship.SELECTION
    assert frame.hypothesis_frame.coverage == HypothesisCoverage.OPEN
    assert frame.hypothesis_frame.unresolved_alternative_mass == 0.50
    assert "relation" not in frame.hypothesis_frame.model_dump()


def test_exact_answer_frame_accepts_one_initial_candidate():
    frame = make_v02_task_frame(
        task_kind=TaskKind.EXACT_ANSWER,
        priors=[0.5],
        unresolved=0.5,
    )

    assert [item.id for item in frame.hypothesis_frame.hypotheses] == ["H1"]


@pytest.mark.parametrize(
    ("competition", "coverage", "unresolved"),
    [
        (HypothesisCompetition.INDEPENDENT, HypothesisCoverage.OPEN, None),
        (HypothesisCompetition.EXCLUSIVE, HypothesisCoverage.EXHAUSTIVE, 0.0),
    ],
)
def test_exact_answer_frame_rejects_non_exclusive_open_shape(
    competition,
    coverage,
    unresolved,
):
    with pytest.raises(
        ValueError,
        match="exact-answer tasks require an exclusive-open frame",
    ):
        make_v02_task_frame(
            task_kind=TaskKind.EXACT_ANSWER,
            competition=competition,
            coverage=coverage,
            priors=[0.5, 0.5],
            unresolved=unresolved,
        )


def test_non_exact_v02_frame_rejects_one_initial_candidate():
    with pytest.raises(ValueError, match="new task frames require at least two hypotheses"):
        make_v02_task_frame(priors=[0.5], unresolved=0.5)


def test_independent_frame_rejects_shared_unresolved_mass():
    with pytest.raises(
        ValueError,
        match="independent frames do not use shared unresolved mass",
    ):
        make_v02_task_frame(
            competition=HypothesisCompetition.INDEPENDENT,
            coverage=HypothesisCoverage.OPEN,
            priors=[0.5, 0.5],
            unresolved=0.2,
        )


def test_independent_exhaustive_frame_is_valid_without_shared_mass():
    frame = make_v02_task_frame(
        competition=HypothesisCompetition.INDEPENDENT,
        coverage=HypothesisCoverage.EXHAUSTIVE,
        priors=[0.5, 0.5],
        unresolved=None,
    )

    assert frame.hypothesis_frame.coverage == HypothesisCoverage.EXHAUSTIVE


def test_exhaustive_frame_rejects_positive_unresolved_mass():
    with pytest.raises(
        ValueError,
        match="unresolved mass is legal only for exclusive-open frames",
    ):
        FrameState(
            frame_id="frame_1",
            competition=HypothesisCompetition.EXCLUSIVE,
            coverage=HypothesisCoverage.EXHAUSTIVE,
            active_hypothesis_ids=["H1", "H2"],
            unresolved_alternative_mass=0.1,
            adequacy_status=FrameAdequacyStatus.ADEQUATE,
        )


@pytest.mark.parametrize(
    "native_fields",
    [
        {
            "competition": HypothesisCompetition.EXCLUSIVE,
            "coverage": HypothesisCoverage.EXHAUSTIVE,
        },
        {
            "competition": HypothesisCompetition.INDEPENDENT,
            "coverage": HypothesisCoverage.OPEN,
        },
    ],
    ids=["consistent", "contradictory"],
)
def test_hypothesis_frame_rejects_mixed_legacy_and_native_relation_input(
    native_fields,
):
    frame = make_v02_task_frame(
        competition=HypothesisCompetition.EXCLUSIVE,
        coverage=HypothesisCoverage.EXHAUSTIVE,
        priors=[0.5, 0.5],
        unresolved=0.0,
    ).hypothesis_frame

    with pytest.raises(
        ValueError,
        match="legacy relation cannot be combined with competition or coverage",
    ):
        HypothesisFrame.model_validate(
            {
                **frame.model_dump(mode="python"),
                "relation": HypothesisRelation.EXCLUSIVE_EXHAUSTIVE,
                **native_fields,
            }
        )


def test_v02_belief_state_requires_frame_state_and_evidence_memory():
    with pytest.raises(ValueError, match="v0.2 belief state requires frame_state"):
        BeliefState(
            schema_version="v0.2",
            belief_state_id="bs",
            run_id="run",
            cycle_id="cycle_0",
            hypotheses=[],
            task_frame=make_v02_task_frame(),
        )


def test_v02_task_frame_rejects_compatibility_defaulted_answer_contract():
    frame = make_v02_task_frame()
    legacy_contract = AnswerContract(
        objective="Select the best supported candidate.",
        required_sections=["answer", "uncertainty"],
        decision_form="candidate_selection",
        permits_synthesis=False,
    )

    with pytest.raises(
        ValueError,
        match="v0.2 answer contract requires answer_value_type",
    ):
        TaskFrame.model_validate(
            {
                **frame.model_dump(mode="python", exclude={"answer_contract"}),
                "answer_contract": legacy_contract,
            }
        )


@pytest.mark.parametrize(
    "decision, message",
    [
        (
            {
                "attempt_id": "attempt_1",
                "status": TaskAdmissionStatus.ADMITTED,
                "epistemic_basis": ["The task requests a bounded answer."],
                "reason": "The task is admissible.",
            },
            "admitted decisions require proposed_task_kind",
        ),
        (
            {
                "attempt_id": "attempt_1",
                "status": TaskAdmissionStatus.NEEDS_REFRAMING,
                "epistemic_basis": ["The answer target is ambiguous."],
                "reason": "Clarification is required.",
            },
            "needs_reframing decisions require clarification_questions",
        ),
        (
            {
                "attempt_id": "attempt_1",
                "status": TaskAdmissionStatus.OUT_OF_SCOPE,
                "epistemic_basis": ["The request is not epistemically assessable."],
                "proposed_task_kind": TaskKind.EXPLANATION,
                "reason": "No supported task kind applies.",
            },
            "out_of_scope decisions must not propose a task kind",
        ),
    ],
)
def test_task_admission_decision_enforces_status_contract(decision, message):
    with pytest.raises(ValueError, match=message):
        TaskAdmissionDecision(**decision)


def test_task_admission_decision_accepts_complete_admission():
    decision = TaskAdmissionDecision(
        attempt_id="attempt_1",
        status=TaskAdmissionStatus.ADMITTED,
        epistemic_basis=["The task requests a bounded answer."],
        proposed_task_kind=TaskKind.EXACT_ANSWER,
        answer_contract_outline=AnswerContractOutline(
            objective="Return the requested scalar.",
            answer_value_type=AnswerValueType.NUMBER,
            decision_form="exact_answer",
            permits_synthesis=False,
            required_sections=["answer"],
        ),
        reason="The task is admissible.",
    )

    assert decision.status == TaskAdmissionStatus.ADMITTED


def test_task_admission_decision_rejects_nested_dynamic_credential_trace():
    credential_field = "api" + "_key"
    credential_value = "sk-" + "runtimecredentialvalue"

    with pytest.raises(ValueError, match="secret"):
        TaskAdmissionDecision(
            attempt_id="attempt_1",
            status=TaskAdmissionStatus.ADMITTED,
            epistemic_basis=["The task requests a bounded answer."],
            proposed_task_kind=TaskKind.EXACT_ANSWER,
            answer_contract_outline=AnswerContractOutline(
                objective="Return the requested scalar.",
                answer_value_type=AnswerValueType.NUMBER,
                decision_form="exact_answer",
                permits_synthesis=False,
                required_sections=["answer"],
            ),
            reason="The task is admissible.",
            model_trace={"nested": [{credential_field: credential_value}]},
        )


@pytest.mark.parametrize(
    "payload",
    [
        {
            "attempt_id": "attempt_admitted_basis",
            "status": "admitted",
            "epistemic_basis": ["Credential: provider-value-123"],
            "proposed_task_kind": "exact_answer",
            "answer_contract_outline": {
                "objective": "Return the supported number.",
                "answer_value_type": "number",
                "decision_form": "single_value",
                "permits_synthesis": False,
                "required_sections": ["answer"],
            },
            "clarification_questions": [],
            "reason": "The task has a bounded answer.",
        },
        {
            "attempt_id": "attempt_admitted_contract",
            "status": "admitted",
            "epistemic_basis": ["The task has a bounded answer."],
            "proposed_task_kind": "exact_answer",
            "answer_contract_outline": {
                "objective": "Return sk-abcdefghijklmnop.",
                "answer_value_type": "number",
                "decision_form": "single_value",
                "permits_synthesis": False,
                "required_sections": ["answer"],
            },
            "clarification_questions": [],
            "reason": "The task has a bounded answer.",
        },
        {
            "attempt_id": "attempt_reframe",
            "status": "needs_reframing",
            "epistemic_basis": ["The requested objective is underspecified."],
            "proposed_task_kind": None,
            "answer_contract_outline": None,
            "clarification_questions": ["Authorization: Bearer abcdefghijklmnop1"],
            "reason": "A clarification is required.",
        },
        {
            "attempt_id": "attempt_scope",
            "status": "out_of_scope",
            "epistemic_basis": ["The request is outside available capabilities."],
            "proposed_task_kind": None,
            "answer_contract_outline": None,
            "clarification_questions": [],
            "reason": "password=provider-value-123",
        },
    ],
    ids=["admitted_basis", "admitted_contract", "reframing", "out_of_scope"],
)
def test_task_admission_decision_rejects_secret_material_in_semantic_fields(payload):
    with pytest.raises(ValueError, match="secret"):
        TaskAdmissionDecision.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("accepted_evidence_ids", ["E1", " e1 "]),
        ("discovery_evidence_ids", ["D1", "d1"]),
        ("discard_and_schema_history", ["accepted", " ACCEPTED "]),
        ("accepted_evidence_ids", [" "]),
    ],
)
def test_evidence_memory_rejects_invalid_identity_lists(field, value):
    with pytest.raises(ValueError, match=field):
        EvidenceMemorySnapshot(**{field: value})


@pytest.mark.parametrize("value", [["E1", " e1 "], [""]])
def test_evidence_memory_rejects_invalid_nested_counterevidence_ids(value):
    with pytest.raises(ValueError, match="counterevidence_ids_by_hypothesis"):
        EvidenceMemorySnapshot(counterevidence_ids_by_hypothesis={"H1": value})


def _signal_provenance(**overrides) -> SignalProvenance:
    return SignalProvenance(
        epistemic_origin=EpistemicOrigin.RETRIEVED_SOURCE,
        source_identity="source-1",
        derivation_root_id="root-1",
        correlation_group="group-1",
        canonical_content_fingerprint="sha256:abc",
        **overrides,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("parent_signal_ids", ["S1", " s1 "]),
        ("citations", ["citation-1", " CITATION-1 "]),
        ("artifact_refs", ["artifact-1", "artifact-1"]),
        ("parent_signal_ids", [""]),
    ],
)
def test_signal_provenance_rejects_invalid_identity_lists(field, value):
    with pytest.raises(ValueError, match=field):
        _signal_provenance(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("falsifiers", ["Contradicted by source A.", " contradicted BY source a. "]),
        ("predictions", ["Prediction A.", "PREDICTION A."]),
        ("falsifiers", [" "]),
    ],
)
def test_framed_hypothesis_rejects_invalid_claim_lists(field, value):
    payload = {
        "id": "H1",
        "statement": "Candidate one explains the observation.",
        "type": "candidate",
        "scope": "The stated task.",
        "initial_prior": 0.5,
        "falsifiers": ["Contradictory evidence appears."],
        "predictions": ["The observation is reproduced."],
        field: value,
    }

    with pytest.raises(ValueError, match=field):
        FramedHypothesis(**payload)


def test_list_contracts_strip_values_without_reordering():
    memory = EvidenceMemorySnapshot(accepted_evidence_ids=[" E2 ", "E1"])
    provenance = _signal_provenance(citations=[" citation-2 ", "citation-1"])
    hypothesis = FramedHypothesis(
        id="H1",
        statement="Candidate one explains the observation.",
        type="candidate",
        scope="The stated task.",
        initial_prior=0.5,
        falsifiers=[" Second falsifier. ", "First falsifier."],
        predictions=[" Second prediction. ", "First prediction."],
    )

    assert memory.accepted_evidence_ids == ["E2", "E1"]
    assert provenance.citations == ["citation-2", "citation-1"]
    assert hypothesis.falsifiers == ["Second falsifier.", "First falsifier."]
    assert hypothesis.predictions == ["Second prediction.", "First prediction."]


def test_v02_domain_contracts_are_publicly_exported():
    names = {
        "TaskAdmissionStatus",
        "AnswerRelationship",
        "AnswerValueType",
        "HypothesisCompetition",
        "HypothesisCoverage",
        "FrameAdequacyStatus",
        "FrameFit",
        "EpistemicOrigin",
        "ProbePurpose",
        "CapabilityKind",
        "ProjectionMode",
        "AnswerContractOutline",
        "TaskAdmissionDecision",
        "FrameState",
        "SignalProvenance",
        "EvidenceMemorySnapshot",
        "FrameMassUpdate",
        "CapabilityDescriptor",
        "CapabilityDecision",
        "migrate_task_frame_v0_1",
        "migrate_belief_state_v0_1",
    }

    assert names.issubset(set(bayesprobe.__all__))


def _open_task_frame() -> TaskFrame:
    return TaskFrame(
        task_frame_id="run_frame_task_frame",
        task_kind=TaskKind.CLAIM_VERIFICATION,
        normalized_question="How should the model-scale claim be tested?",
        task_context="Evaluate on a frozen real-task distribution.",
        answer_contract=AnswerContract(
            objective="Design a discriminating validation protocol.",
            required_sections=["hypotheses", "controls", "decision_rule"],
            decision_form="experimental_protocol",
            permits_synthesis=True,
        ),
        hypothesis_frame=HypothesisFrame(
            frame_id="run_frame_hypothesis_frame",
            relation=HypothesisRelation.INDEPENDENT,
            hypotheses=[
                FramedHypothesis(
                    id="H1",
                    statement="Scale has an independent positive effect.",
                    type="causal_claim",
                    scope="Matched agent and compute conditions.",
                    initial_prior=0.5,
                    falsifiers=["The controlled effect is negligible."],
                    predictions=["Performance rises under matched controls."],
                ),
                FramedHypothesis(
                    id="H2",
                    statement="The apparent effect is caused by confounding.",
                    type="confounding_explanation",
                    scope="Unmatched published comparisons.",
                    initial_prior=0.5,
                    falsifiers=["The effect survives all matched controls."],
                    predictions=["The effect shrinks after matching resources."],
                ),
            ],
            rival_sets={"H1": [], "H2": []},
            coverage_statement="Tests the causal claim and its main confounder.",
            coverage_limitation="Other task-specific interactions may exist.",
        ),
        framing_method=FramingMethod.MODEL,
        framing_trace={"task": "frame_open_question", "schema_version": "v0.1"},
    )


def test_task_frame_accepts_independent_open_hypotheses():
    frame = _open_task_frame()
    assert frame.hypothesis_frame.relation == HypothesisRelation.INDEPENDENT
    assert [item.id for item in frame.hypothesis_frame.hypotheses] == ["H1", "H2"]


def test_v02_belief_state_rejects_nested_v01_task_frame():
    legacy_frame = _open_task_frame()

    with pytest.raises(
        ValueError,
        match="v0.2 belief state requires a v0.2 task_frame",
    ):
        BeliefState(
            schema_version="v0.2",
            belief_state_id="bs_1",
            run_id="run_1",
            cycle_id="cycle_1",
            hypotheses=[],
            task_frame=legacy_frame,
            frame_state=FrameState(
                frame_id=legacy_frame.hypothesis_frame.frame_id,
                competition=legacy_frame.hypothesis_frame.competition,
                coverage=legacy_frame.hypothesis_frame.coverage,
                active_hypothesis_ids=[
                    item.id for item in legacy_frame.hypothesis_frame.hypotheses
                ],
                adequacy_status=FrameAdequacyStatus.PROVISIONAL,
            ),
            evidence_memory=EvidenceMemorySnapshot(),
        )


@pytest.mark.parametrize(
    "mutator, message",
    [
        (lambda frame: frame.model_copy(update={"answer_contract": frame.answer_contract.model_copy(update={"required_sections": []})}), "required_sections"),
        (lambda frame: frame.model_copy(update={"hypothesis_frame": frame.hypothesis_frame.model_copy(update={"hypotheses": [frame.hypothesis_frame.hypotheses[0], frame.hypothesis_frame.hypotheses[1].model_copy(update={"id": "H1"})]})}), "ids must be unique"),
        (lambda frame: frame.model_copy(update={"hypothesis_frame": frame.hypothesis_frame.model_copy(update={"rival_sets": {"H1": ["missing"], "H2": []}})}), "unknown rival"),
        (lambda frame: frame.model_copy(update={"framing_trace": {"api_key": "forbidden"}}), "secret"),
    ],
)
def test_task_frame_rejects_invalid_contract(mutator, message):
    with pytest.raises(ValueError, match=message):
        TaskFrame.model_validate(mutator(_open_task_frame()).model_dump())


def test_task_frame_rejects_tuple_secret_material():
    frame = _open_task_frame().model_copy(
        update={"framing_trace": {"nested": ("not-json-compatible",)}}
    )

    with pytest.raises(ValueError, match="JSON-compatible"):
        TaskFrame.model_validate(frame.model_dump())


def test_task_frame_rejects_nested_list_secret_material():
    frame = _open_task_frame().model_copy(
        update={"framing_trace": {"nested": [{"deeper": ["sk-123456789012"]}]}}
    )

    with pytest.raises(ValueError, match="secret"):
        TaskFrame.model_validate(frame.model_dump())


def test_task_frame_rejects_secret_mapping_key():
    frame = _open_task_frame().model_copy(
        update={"framing_trace": {"sk-123456789012": "metadata"}}
    )

    with pytest.raises(ValueError, match="secret"):
        TaskFrame.model_validate(frame.model_dump())


@pytest.mark.parametrize(
    "key",
    ["api_key", "Api-Key", "AUTHORIZATION", "access_token", "secret.value"],
)
def test_forbidden_secret_key_name_normalizes_common_variants(key):
    assert is_forbidden_secret_key_name(key)


@pytest.mark.parametrize(
    "key",
    [
        "provider_api_key",
        "refresh_token",
        "client_secret",
        "db_password",
        "access_key_id",
        "proxyAuthorization",
    ],
)
def test_forbidden_secret_key_name_detects_real_affixed_fields(key):
    assert is_forbidden_secret_key_name(key)


@pytest.mark.parametrize(
    "key",
    [
        "tokenization",
        "token_count",
        "secretary",
        "password_policy",
        "credential_score",
        "cookie_policy",
    ],
)
def test_forbidden_secret_key_name_allows_benign_semantic_fields(key):
    assert not is_forbidden_secret_key_name(key)


def test_secret_predicates_allow_ordinary_tokenization_prose():
    assert not is_secret_like_value("Tokenization is a useful concept.")


@pytest.mark.parametrize(
    "value",
    [
        "sk-abcdefghijklmnop",
        "password = correct-horse-battery-staple",
        "credential: provider-value-123",
        "Authorization: Bearer abcdefghijklmnop",
        "-----BEGIN PRIVATE KEY-----",
        "access_key='AKIAEXAMPLEVALUE'",
    ],
)
def test_secret_value_predicate_detects_credential_text_forms(value):
    assert is_secret_like_value(value)


@pytest.mark.parametrize(
    "value",
    [
        "ghp_" + "a" * 36,
        "gho_" + "b" * 36,
        "ghu_" + "c" * 36,
        "ghs_" + "d" * 36,
        "ghr_" + "e" * 36,
        "github_pat_" + "A1" * 20,
        (
            "eyJhbGciOiJIUzI1NiJ9."
            "eyJzdWIiOiIxMjM0NTY3ODkwIn0."
            "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        ),
        "AKIAIOSFODNN7EXAMPLE",
        "xox" + "b-123456789012-1234567890123-abcdefghijklmnopqrstuvwx",
        "Bearer abcdefghijklmnopqrstuvwx",
    ],
)
def test_secret_value_predicate_detects_common_generic_credentials(value):
    assert is_secret_like_value(value)


@pytest.mark.parametrize(
    "value",
    [
        "Tokenization is a useful concept.",
        "Compare password policies without including a password value.",
        "Bearer authentication should use short-lived credentials.",
        "Private-key cryptography has different trust assumptions.",
        "The source discusses access key rotation practices.",
        "Bearer authentication",
        "Bearer authorization scheme",
        "This source discusses bearer tokens without including one.",
    ],
)
def test_secret_value_predicate_preserves_ordinary_source_text(value):
    assert not is_secret_like_value(value)


def test_shared_redaction_preserves_benign_keys_and_removes_affixed_secret_keys():
    redact_secret_material = getattr(schemas, "redact_secret_material")
    payload = {
        "provider_api_key": "hidden-provider-value",
        "refresh_token": "hidden-refresh-value",
        "client_secret": "hidden-client-value",
        "db_password": "hidden-password-value",
        "access_key_id": "AKIAIOSFODNN7EXAMPLE",
        "tokenization": "kept",
        "token_count": 12,
        "secretary": "kept",
        "password_policy": "kept",
        "credential_score": 0.8,
        "cookie_policy": "kept",
    }

    assert redact_secret_material(payload) == {
        "tokenization": "kept",
        "token_count": 12,
        "secretary": "kept",
        "password_policy": "kept",
        "credential_score": 0.8,
        "cookie_policy": "kept",
    }


def test_shared_redaction_removes_forbidden_fields_and_redacts_secret_strings():
    redact_secret_material = getattr(schemas, "redact_secret_material")
    payload = {
        "private_key": "first-private-value",
        "password": "second-password-value",
        "credential": "third-credential-value",
        "access_key": "fourth-access-value",
        "nested": {
            "authorization_text": "Authorization: Bearer abcdefghijklmnop",
            "ordinary_source": "A source compares password policies.",
        },
    }

    sanitized = redact_secret_material(payload)
    serialized = repr(sanitized)

    for forbidden in (
        "private_key",
        "credential",
        "access_key",
        "first-private-value",
        "second-password-value",
        "third-credential-value",
        "fourth-access-value",
        "abcdefghijklmnop",
    ):
        assert forbidden not in serialized
    assert sanitized == {
        "nested": {
            "ordinary_source": "A source compares password policies.",
        }
    }


@pytest.mark.parametrize(
    "mutator",
    [
        lambda frame, secret: frame.model_copy(
            update={"normalized_question": f"Question contains {secret}"}
        ),
        lambda frame, secret: frame.model_copy(
            update={"task_context": f"Context contains {secret}"}
        ),
        lambda frame, secret: frame.model_copy(
            update={
                "answer_contract": frame.answer_contract.model_copy(
                    update={"objective": f"Objective contains {secret}"}
                )
            }
        ),
        lambda frame, secret: frame.model_copy(
            update={
                "hypothesis_frame": frame.hypothesis_frame.model_copy(
                    update={
                        "hypotheses": [
                            frame.hypothesis_frame.hypotheses[0].model_copy(
                                update={"statement": secret}
                            ),
                            frame.hypothesis_frame.hypotheses[1],
                        ]
                    }
                )
            }
        ),
        lambda frame, secret: frame.model_copy(
            update={
                "hypothesis_frame": frame.hypothesis_frame.model_copy(
                    update={"coverage_statement": secret}
                )
            }
        ),
    ],
)
def test_task_frame_rejects_secret_material_in_semantic_fields(mutator):
    secret = "sk-abcdefghijklmnop"

    with pytest.raises(ValueError, match="secret"):
        TaskFrame.model_validate(mutator(_open_task_frame(), secret).model_dump())


@pytest.mark.parametrize(
    "secret",
    [
        "ghp_" + "a" * 36,
        (
            "eyJhbGciOiJIUzI1NiJ9."
            "eyJzdWIiOiIxMjM0NTY3ODkwIn0."
            "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        ),
        "Bearer abcdefghijklmnopqrstuvwx",
    ],
)
def test_task_frame_rejects_common_generic_credentials(secret):
    frame = _open_task_frame().model_copy(
        update={"normalized_question": f"Question contains {secret}"}
    )

    with pytest.raises(ValueError, match="secret"):
        TaskFrame.model_validate(frame.model_dump())


def test_minimal_run_cycle_and_belief_state_round_trip():
    run = RunRecord(run_id="run_1", regime=RunRegime.AUTONOMOUS, problem="Decide X")
    cycle = CycleRecord(
        cycle_id="cycle_1",
        run_id=run.run_id,
        cycle_index=1,
        signal_shape=CycleSignalShape.ACTIVE_ONLY,
    )
    hypothesis = Hypothesis(
        id="H1",
        statement="X is true",
        scope="sample scope",
        prior=0.5,
        posterior=0.5,
        rivals=["H2"],
        falsifiers=["A strong counterexample would weaken H1."],
        predictions=["Evidence A is likely if H1 is true."],
    )
    belief_state = BeliefState(
        belief_state_id="bs_1",
        run_id=run.run_id,
        cycle_id=cycle.cycle_id,
        hypotheses=[hypothesis],
    )

    loaded = BeliefState.model_validate_json(belief_state.model_dump_json())

    assert loaded.hypotheses[0].id == "H1"
    assert loaded.hypotheses[0].status == HypothesisStatus.ACTIVE


def test_probe_set_can_be_empty_for_passive_only_cycle():
    probe_set = ProbeSet(
        probe_set_id="ps_1",
        cycle_id="cycle_1",
        probes=[],
        selection_reason="Passive-only synchronized cycle.",
        may_be_empty=True,
    )

    assert probe_set.probes == []
    assert probe_set.may_be_empty is True


def test_external_signal_kinds_and_change_my_mind_candidates():
    candidate = ProbeCandidate(
        candidate_id="pc_1",
        source="change_my_mind",
        candidate_probe=ProbeDesign(
            id="P1",
            cycle_id="cycle_2",
            target_hypotheses=["H1"],
            inquiry_goal="Check if source A is independent.",
            method="source_tracing",
            support_condition={"H1": "Source A is independent."},
            weaken_condition={"H1": "Source A shares origin with source B."},
        ),
    )
    condition = ChangeMyMindCondition(
        human_readable_condition="I would lower H1 if source A is not independent.",
        structured_probe_candidates=[candidate],
    )
    signal = ExternalSignal(
        id="S1",
        cycle_id="cycle_1",
        signal_kind=SignalKind.PASSIVE,
        source_type="external_agent_projection",
        source="agent_a",
        raw_content="Agent A believes H1 because source A supports it.",
    )

    assert condition.structured_probe_candidates[0].candidate_probe.method == "source_tracing"
    assert signal.signal_kind == SignalKind.PASSIVE


def test_evidence_event_model_trace_defaults_to_empty_dict():
    event = EvidenceEvent(
        id="E1",
        derived_from_signal="S1",
        target_hypotheses=["H1"],
        evidence_type=EvidenceType.SUPPORTING,
        content="SUPPORTS: evidence.",
        likelihoods={"H1": LikelihoodBand.MODERATELY_CONFIRMING},
    )

    assert event.model_trace == {}


def test_evidence_event_model_trace_round_trips_through_json():
    event = EvidenceEvent(
        id="E1",
        derived_from_signal="S1",
        target_hypotheses=["H1"],
        evidence_type=EvidenceType.SUPPORTING,
        content="SUPPORTS: evidence.",
        likelihoods={"H1": LikelihoodBand.MODERATELY_CONFIRMING},
        model_trace={
            "task": "judge_evidence",
            "adapter_kind": "scripted",
            "prompt_id": "evidence_judgment",
            "prompt_version": "v0.1",
            "schema_name": "EvidenceJudgment",
            "schema_version": "v0.1",
            "metadata": {},
        },
    )

    loaded = EvidenceEvent.model_validate_json(event.model_dump_json())

    assert loaded.model_trace == {
        "task": "judge_evidence",
        "adapter_kind": "scripted",
        "prompt_id": "evidence_judgment",
        "prompt_version": "v0.1",
        "schema_name": "EvidenceJudgment",
        "schema_version": "v0.1",
        "metadata": {},
    }
