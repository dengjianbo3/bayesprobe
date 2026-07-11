import pytest

from bayesprobe.model_gateway import ScriptedModelGateway, StructuredModelRequest
from bayesprobe.schemas import AnswerChoice, HypothesisRelation, TaskKind
from bayesprobe.task_framing import (
    ExplicitTaskFramer,
    HypothesisSeed,
    ModelTaskFramer,
    RecordedTaskFramer,
    RoutingTaskFramer,
    TaskFramingError,
    TaskFramingInput,
    task_frame_from_mapping,
)


class QueueModelGateway:
    adapter_kind = "queue_test"

    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.requests: list[StructuredModelRequest] = []

    def complete_structured(self, request: StructuredModelRequest) -> dict:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError(f"unexpected model task: {request.task}")
        return self.responses.pop(0)


VALID_OPEN_FRAME = {
    "task_kind": "claim_verification",
    "answer_contract": {
        "objective": "Design a discriminating validation protocol.",
        "required_sections": [
            "hypotheses",
            "experimental_design",
            "controls",
            "metrics",
            "decision_rule",
            "limitations",
        ],
        "decision_form": "experimental_protocol",
        "permits_synthesis": True,
    },
    "hypothesis_relation": "independent",
    "hypotheses": [
        {
            "statement": "Scale has an independent positive effect under matched conditions.",
            "type": "causal_claim",
            "scope": "Matched task, scaffold, and inference budget.",
            "falsifiers": ["The controlled effect is negligible or negative."],
            "predictions": ["Performance increases across matched sizes."],
        },
        {
            "statement": "The apparent scale effect is materially confounded.",
            "type": "confounding_explanation",
            "scope": "Comparisons with unmatched resources.",
            "falsifiers": ["The effect survives matched controls."],
            "predictions": ["The effect shrinks after matching resources."],
        },
    ],
    "coverage_statement": "Covers the target effect and the primary confounder.",
    "coverage_limitation": "Conditional task interactions remain possible.",
}


def test_explicit_framer_uses_structured_choices_without_text_parsing():
    frame = ExplicitTaskFramer().frame(
        TaskFramingInput(
            run_id="run_choices",
            question="Which result follows?",
            answer_choices=[
                AnswerChoice(label="A", text="First result"),
                AnswerChoice(label="B", text="Second result"),
            ],
        )
    )

    assert frame.task_kind == TaskKind.MULTIPLE_CHOICE
    assert frame.hypothesis_frame.relation == HypothesisRelation.EXCLUSIVE_EXHAUSTIVE
    assert [item.id for item in frame.hypothesis_frame.hypotheses] == ["A", "B"]


def test_explicit_framer_parses_english_legacy_choices():
    frame = ExplicitTaskFramer().frame(
        TaskFramingInput(
            run_id="run_en_choices",
            question="Which result follows?\nAnswer Choices:\nA. First result\nB. Second result",
        )
    )

    assert frame.task_kind == TaskKind.MULTIPLE_CHOICE
    assert frame.normalized_question == "Which result follows?"


def test_explicit_framer_parses_chinese_legacy_choices():
    frame = ExplicitTaskFramer().frame(
        TaskFramingInput(
            run_id="run_cn_choices",
            question="哪一项正确？\n答案选项：\nA. 第一项\nB. 第二项",
        )
    )

    assert frame.task_kind == TaskKind.MULTIPLE_CHOICE


def test_explicit_framer_uses_explicit_seeds():
    frame = ExplicitTaskFramer().frame(
        TaskFramingInput(
            run_id="run_seeds",
            question="Which explanation fits?",
            hypothesis_seeds=[
                HypothesisSeed(statement="The first explanation fits.", prior=0.5),
                HypothesisSeed(statement="The second explanation fits.", prior=0.5),
            ],
        )
    )

    assert frame.task_kind == TaskKind.DECISION
    assert [item.initial_prior for item in frame.hypothesis_frame.hypotheses] == [0.5, 0.5]


def test_explicit_framer_rejects_unseeded_open_question():
    with pytest.raises(TaskFramingError, match="requires a model or recorded task framer"):
        ExplicitTaskFramer().frame(
            TaskFramingInput(
                run_id="run_open",
                question="这个命题应该如何验证？",
            )
        )


def test_explicit_framer_can_frame_without_materializing(monkeypatch):
    framer = ExplicitTaskFramer()
    materializations = 0
    original_frame = ExplicitTaskFramer.frame

    def count_materializations(self, input):
        nonlocal materializations
        materializations += 1
        return original_frame(self, input)

    monkeypatch.setattr(ExplicitTaskFramer, "frame", count_materializations)

    assert framer.can_frame(
        TaskFramingInput(
            run_id="run_choices",
            question="Which result follows?",
            answer_choices=[
                AnswerChoice(label="A", text="First result"),
                AnswerChoice(label="B", text="Second result"),
            ],
        )
    )
    assert framer.can_frame(
        TaskFramingInput(
            run_id="run_legacy",
            question="Which result follows? Answer Choices: A. First result B. Second result",
        )
    )
    assert not framer.can_frame(
        TaskFramingInput(run_id="run_open", question="How should this claim be tested?")
    )
    assert materializations == 0


@pytest.mark.parametrize(
    "input",
    [
        TaskFramingInput(
            run_id="run_one_choice",
            question="Which result follows?",
            answer_choices=[AnswerChoice(label="A", text="First result")],
        ),
        TaskFramingInput(
            run_id="run_one_seed",
            question="Which explanation fits?",
            hypothesis_seeds=[HypothesisSeed(statement="The only explanation.")],
        ),
        TaskFramingInput(
            run_id="run_conflict",
            question="Which explanation fits?",
            answer_choices=[
                AnswerChoice(label="A", text="First result"),
                AnswerChoice(label="B", text="Second result"),
            ],
            hypothesis_seeds=[
                HypothesisSeed(statement="The first explanation.", prior=0.5),
                HypothesisSeed(statement="The second explanation.", prior=0.5),
            ],
        ),
        TaskFramingInput(
            run_id="run_partial_priors",
            question="Which explanation fits?",
            hypothesis_seeds=[
                HypothesisSeed(statement="The first explanation.", prior=0.5),
                HypothesisSeed(statement="The second explanation."),
            ],
        ),
        TaskFramingInput(
            run_id="run_invalid_priors",
            question="Which explanation fits?",
            hypothesis_seeds=[
                HypothesisSeed(statement="The first explanation.", prior=0.7),
                HypothesisSeed(statement="The second explanation.", prior=0.7),
            ],
        ),
    ],
)
def test_explicit_framer_capability_rejects_invalid_explicit_inputs(input):
    framer = ExplicitTaskFramer()

    assert not framer.can_frame(input)
    with pytest.raises(TaskFramingError):
        framer.frame(input)


@pytest.mark.parametrize(
    "seed",
    [
        HypothesisSeed(statement="First explanation.", prior="0.5"),
        HypothesisSeed(statement="First explanation.", prior=float("nan")),
        HypothesisSeed(statement="First explanation.", prior=float("inf")),
        HypothesisSeed(statement="First explanation.", id=1),
        HypothesisSeed(statement="First explanation.", scope=object()),
        HypothesisSeed(statement="First explanation.", falsifiers=["Valid", 3]),
        HypothesisSeed(statement="First explanation.", predictions="not a list"),
    ],
)
def test_explicit_framer_rejects_malformed_seed_values(seed):
    companion = HypothesisSeed(
        statement="Second explanation.",
        prior=seed.prior if seed.prior is not None else None,
    )
    input = TaskFramingInput(
        run_id="run_malformed_seed",
        question="Which explanation fits?",
        hypothesis_seeds=[seed, companion],
    )
    framer = ExplicitTaskFramer()

    assert not framer.can_frame(input)
    with pytest.raises(TaskFramingError):
        framer.frame(input)


def test_explicit_framer_defaults_valid_independent_seed_credences():
    frame = ExplicitTaskFramer().frame(
        TaskFramingInput(
            run_id="run_independent",
            question="Which conditions apply?",
            task_context="Keep the two conditions separate.",
            hypothesis_relation=HypothesisRelation.INDEPENDENT,
            hypothesis_seeds=[
                HypothesisSeed(statement="The first condition applies."),
                HypothesisSeed(statement="The second condition applies."),
            ],
        )
    )

    assert frame.task_context == "Keep the two conditions separate."
    assert [item.initial_prior for item in frame.hypothesis_frame.hypotheses] == [0.5, 0.5]
    assert frame.hypothesis_frame.rival_sets == {"H1": [], "H2": []}


def test_model_task_framer_calls_gateway_before_returning_frame():
    gateway = ScriptedModelGateway({"frame_open_question": VALID_OPEN_FRAME})

    frame = ModelTaskFramer(gateway).frame(
        TaskFramingInput(
            run_id="run_model",
            question="这个命题应该如何验证？",
            task_context="Use matched conditions.",
        )
    )

    assert [request.task for request in gateway.requests] == ["frame_open_question"]
    assert gateway.requests[0].input == {
        "question": "这个命题应该如何验证？",
        "task_context": "Use matched conditions.",
        "supported_task_kinds": [
            "claim_verification",
            "explanation",
            "diagnosis",
            "design",
            "decision",
        ],
        "supported_relations": ["exclusive_exhaustive", "independent"],
        "hypothesis_count": {"minimum": 2, "maximum": 6},
    }
    assert [item.id for item in frame.hypothesis_frame.hypotheses] == ["H1", "H2"]
    assert [item.initial_prior for item in frame.hypothesis_frame.hypotheses] == [0.5, 0.5]
    assert frame.hypothesis_frame.rival_sets == {"H1": [], "H2": []}
    assert frame.framing_method.value == "model"


def test_model_task_framer_repairs_once_then_accepts_without_retaining_secrets():
    gateway = QueueModelGateway(
        [
            {
                "task_kind": "claim_verification",
                "hypotheses": [],
                "api_key": "sk-abcdefghijklmnop",
            },
            VALID_OPEN_FRAME,
        ]
    )

    frame = ModelTaskFramer(gateway).frame(
        TaskFramingInput(run_id="run_repair", question="How should this be tested?")
    )

    assert [request.task for request in gateway.requests] == [
        "frame_open_question",
        "repair_task_frame",
    ]
    assert gateway.requests[1].metadata["repair_attempt_index"] == 1
    assert gateway.requests[1].input["invalid_payload"]["api_key"] == "[REDACTED]"
    assert "sk-abcdefghijklmnop" not in repr(gateway.requests[1])
    assert frame.framing_trace["repair_attempt_index"] == 1


def test_model_task_framer_fails_after_one_invalid_repair():
    gateway = QueueModelGateway(
        [
            {"task_kind": "claim_verification", "hypotheses": []},
            {"task_kind": "claim_verification", "hypotheses": []},
        ]
    )

    with pytest.raises(TaskFramingError, match="invalid after 1 repair attempt"):
        ModelTaskFramer(gateway).frame(
            TaskFramingInput(run_id="run_bad_repair", question="How should this be tested?")
        )


def test_task_frame_from_mapping_rejects_provider_owned_beliefs():
    payload = {
        **VALID_OPEN_FRAME,
        "hypotheses": [
            {**VALID_OPEN_FRAME["hypotheses"][0], "id": "provider_h1"},
            VALID_OPEN_FRAME["hypotheses"][1],
        ],
    }

    with pytest.raises(TaskFramingError, match="cannot assign ids or beliefs"):
        task_frame_from_mapping(
            payload,
            run_id="run_strict",
            question="How should this be tested?",
            task_context="",
            method="model",
            trace={},
        )


def test_recorded_task_framer_returns_a_run_specific_deep_copy():
    source_frame = ExplicitTaskFramer().frame(
        TaskFramingInput(
            run_id="fixture",
            question="Which explanation fits?",
            hypothesis_seeds=[
                HypothesisSeed(statement="The first explanation."),
                HypothesisSeed(statement="The second explanation."),
            ],
        )
    )

    frame = RecordedTaskFramer(source_frame).frame(
        TaskFramingInput(run_id="replay", question="How should this be tested?")
    )

    assert frame.task_frame_id == "replay_task_frame"
    assert frame.hypothesis_frame.frame_id == "replay_hypothesis_frame"
    assert frame.normalized_question == "How should this be tested?"
    assert frame.framing_method.value == "recorded"
    assert source_frame.framing_method.value == "explicit"


def test_routing_task_framer_keeps_explicit_mcq_off_the_model_gateway():
    gateway = QueueModelGateway([])

    frame = RoutingTaskFramer(
        explicit_framer=ExplicitTaskFramer(),
        open_framer=ModelTaskFramer(gateway),
    ).frame(
        TaskFramingInput(
            run_id="run_routed_mcq",
            question="Which result follows? Answer Choices: A. First B. Second",
        )
    )

    assert frame.task_kind == TaskKind.MULTIPLE_CHOICE
    assert gateway.requests == []
