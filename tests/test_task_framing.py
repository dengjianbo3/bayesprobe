import pytest

from bayesprobe.schemas import AnswerChoice, HypothesisRelation, TaskKind
from bayesprobe.task_framing import (
    ExplicitTaskFramer,
    HypothesisSeed,
    TaskFramingError,
    TaskFramingInput,
)


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
