import builtins
from pathlib import Path

import pytest

from bayesprobe.evaluation.hle import HLEDatasetAdapter, HLESelectionConfig


REVISION = "a" * 40


def make_row(
    sample_id: str,
    *,
    category: str = "mathematics",
    question: str | None = None,
    answer: str = "B",
    answer_type: str = "multipleChoice",
    image=None,
    **extra,
):
    return {
        "id": sample_id,
        "category": category,
        "question": question
        or "What is 1 + 1?\nA. 1\nB. 2\nC. 3",
        "answer": answer,
        "answer_type": answer_type,
        "image": image,
        "rationale": extra.pop("rationale", "must never be copied"),
        "canary": extra.pop("canary", "BENCHMARK-CANARY-MUST-NOT-COPY"),
        **extra,
    }


def test_selection_config_requires_full_immutable_revision_sha():
    with pytest.raises(ValueError, match="full 40-character commit SHA"):
        HLESelectionConfig(revision="main")


def test_adapter_canonicalizes_choices_and_maps_label_gold():
    prepared = HLEDatasetAdapter().prepare_rows(
        [make_row("synthetic_1")],
        HLESelectionConfig(revision=REVISION, sample_count=1),
    )

    case = prepared.runtime_cases[0]
    entry = prepared.manifest_entries[0]
    assert case.question == (
        "What is 1 + 1?\n\nAnswer Choices:\nA. 1\nB. 2\nC. 3"
    )
    assert case.choices == {"A": "1", "B": "2", "C": "3"}
    assert prepared.gold_store.labels == {"synthetic_1": "B"}
    assert entry.original_question_sha256 != entry.canonical_question_sha256
    assert entry.category == "mathematics"


def test_adapter_maps_unique_exact_choice_text_gold():
    prepared = HLEDatasetAdapter().prepare_rows(
        [make_row("synthetic_1", answer="2")],
        HLESelectionConfig(revision=REVISION, sample_count=1),
    )

    assert prepared.gold_store.labels["synthetic_1"] == "B"


def test_adapter_counts_each_eligibility_rejection_reason():
    rows = [
        make_row("eligible"),
        make_row("wrong_type", answer_type="shortAnswer"),
        make_row("has_image", image="image-bytes"),
        make_row("missing_id") | {"id": ""},
        make_row("missing_question") | {"question": ""},
        make_row("missing_answer") | {"answer": ""},
        make_row("missing_category") | {"category": ""},
        make_row("one_choice", question="Choose.\nA. Only one"),
        make_row("bad_gold", answer="Z"),
        make_row(
            "ambiguous_gold",
            question="Choose.\nA. same\nB. same\nC. other",
            answer="same",
        ),
    ]

    prepared = HLEDatasetAdapter().prepare_rows(
        rows,
        HLESelectionConfig(revision=REVISION, sample_count=1),
    )

    assert prepared.rejection_counts == {
        "ambiguous_gold": 1,
        "answer_type": 1,
        "gold_not_mappable": 1,
        "image_present": 1,
        "missing_answer": 1,
        "missing_category": 1,
        "missing_id": 1,
        "missing_question": 1,
        "unparseable_choices": 1,
    }


def test_adapter_refuses_to_reduce_requested_sample_count():
    with pytest.raises(ValueError, match="fewer than 2 eligible rows"):
        HLEDatasetAdapter().prepare_rows(
            [make_row("only_one")],
            HLESelectionConfig(revision=REVISION, sample_count=2),
        )


def test_selection_uses_proportional_largest_remainder_quotas():
    rows = [
        make_row(f"math_{index}", category="math") for index in range(5)
    ] + [
        make_row(f"history_{index}", category="history") for index in range(3)
    ] + [
        make_row(f"law_{index}", category="law") for index in range(2)
    ]

    prepared = HLEDatasetAdapter().prepare_rows(
        rows,
        HLESelectionConfig(revision=REVISION, sample_count=6),
    )

    assert prepared.category_quotas == {"history": 2, "law": 1, "math": 3}
    assert len(prepared.runtime_cases) == 6


def test_largest_remainder_ties_break_by_category_name():
    rows = [
        make_row("alpha_1", category="alpha"),
        make_row("beta_1", category="beta"),
        make_row("gamma_1", category="gamma"),
    ]

    prepared = HLEDatasetAdapter().prepare_rows(
        rows,
        HLESelectionConfig(revision=REVISION, sample_count=2),
    )

    assert prepared.category_quotas == {"alpha": 1, "beta": 1, "gamma": 0}


def test_seeded_selection_and_manifest_hash_are_order_independent():
    rows = [make_row(f"sample_{index}") for index in range(8)]
    config = HLESelectionConfig(
        revision=REVISION,
        sample_count=4,
        seed="20260711",
    )

    forward = HLEDatasetAdapter().prepare_rows(rows, config)
    reverse = HLEDatasetAdapter().prepare_rows(list(reversed(rows)), config)

    assert [case.sample_id for case in forward.runtime_cases] == [
        case.sample_id for case in reverse.runtime_cases
    ]
    assert forward.manifest_sha256 == reverse.manifest_sha256


def test_canonical_question_round_trips_through_bayesprobe_initializer():
    prepared = HLEDatasetAdapter().prepare_rows(
        [
            make_row(
                "synthetic_1",
                question=(
                    "Which statement follows? Answer Choices: "
                    "A) First statement B) Second statement C) Third statement"
                ),
                answer="C",
            )
        ],
        HLESelectionConfig(revision=REVISION, sample_count=1),
    )

    assert prepared.runtime_cases[0].choice_labels == ("A", "B", "C")


def test_gated_dataset_dependency_is_loaded_lazily(monkeypatch, tmp_path: Path):
    real_import = builtins.__import__

    def rejecting_import(name, *args, **kwargs):
        if name == "datasets":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", rejecting_import)

    with pytest.raises(RuntimeError, match=r"Install bayesprobe\[hle\]"):
        HLEDatasetAdapter().prepare(HLESelectionConfig(revision=REVISION))
