from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Any

from bayesprobe.evaluation.contracts import EvaluationCase
from bayesprobe.initialization import BayesProbeInitializer, InitializeRunInput


_FULL_COMMIT_SHA = re.compile(r"^[0-9a-fA-F]{40}$")
_ANSWER_CHOICES_HEADER = re.compile(r"\banswer\s+choices?\s*:\s*", re.IGNORECASE)
_CHOICE_BLOCK_LINE = re.compile(
    r"^\s*([A-Z])[\.\)]\s+(.*?)(?=^\s*[A-Z][\.\)]\s+|\Z)",
    re.MULTILINE | re.DOTALL,
)
_CHOICE_INLINE = re.compile(
    r"(?:^|\s)([A-Z])[\.\)]\s+(.*?)(?=\s+[A-Z][\.\)]\s+|\Z)",
    re.DOTALL,
)
_GOLD_LABEL = re.compile(r"^\(?([A-Z])\)?[\.\)]?$", re.IGNORECASE)


@dataclass(frozen=True)
class HLESelectionConfig:
    revision: str
    sample_count: int = 100
    seed: str = "20260711"

    def __post_init__(self) -> None:
        if not isinstance(self.revision, str) or not _FULL_COMMIT_SHA.fullmatch(
            self.revision.strip()
        ):
            raise ValueError("HLE revision must be a full 40-character commit SHA")
        if type(self.sample_count) is not int or self.sample_count < 1:
            raise ValueError("HLE sample_count must be a positive integer")
        if not isinstance(self.seed, str) or not self.seed.strip():
            raise ValueError("HLE selection seed must not be empty")
        object.__setattr__(self, "revision", self.revision.strip().lower())
        object.__setattr__(self, "seed", self.seed.strip())


@dataclass(frozen=True)
class HLEManifestEntry:
    sample_id: str
    question: str
    choices: dict[str, str]
    category: str
    answer_type: str
    dataset_revision: str
    original_question_sha256: str
    canonical_question_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_runtime_case(self) -> EvaluationCase:
        return EvaluationCase(
            sample_id=self.sample_id,
            question=self.question,
            choices=self.choices,
        )


@dataclass(frozen=True)
class EvaluationGoldStore:
    manifest_sha256: str
    labels: dict[str, str]


@dataclass(frozen=True)
class PreparedEvaluationSet:
    dataset_revision: str
    seed: str
    requested_sample_count: int
    eligible_count: int
    rejection_counts: dict[str, int]
    category_quotas: dict[str, int]
    manifest_entries: tuple[HLEManifestEntry, ...]
    manifest_sha256: str
    gold_store: EvaluationGoldStore

    @property
    def runtime_cases(self) -> tuple[EvaluationCase, ...]:
        return tuple(entry.to_runtime_case() for entry in self.manifest_entries)

    def selection_manifest_payload(self) -> dict[str, Any]:
        return {
            "artifact_version": "0.1",
            "dataset_revision": self.dataset_revision,
            "seed": self.seed,
            "requested_sample_count": self.requested_sample_count,
            "eligible_count": self.eligible_count,
            "rejection_counts": dict(self.rejection_counts),
            "category_quotas": dict(self.category_quotas),
            "selection_algorithm": "proportional_largest_remainder_sha256_v0.1",
            "manifest_sha256": self.manifest_sha256,
            "items": [entry.to_dict() for entry in self.manifest_entries],
        }


@dataclass(frozen=True)
class _EligibleRow:
    entry: HLEManifestEntry
    gold_label: str


class HLEDatasetAdapter:
    dataset_name = "cais/hle"

    def prepare(self, config: HLESelectionConfig) -> PreparedEvaluationSet:
        try:
            from datasets import load_dataset
        except ImportError as error:
            raise RuntimeError(
                "HLE evaluation requires the optional datasets dependency. "
                "Install bayesprobe[hle]."
            ) from error
        rows = load_dataset(
            self.dataset_name,
            split="test",
            revision=config.revision,
        )
        return self.prepare_rows(rows, config)

    def prepare_rows(
        self,
        rows: Iterable[Mapping[str, Any]],
        config: HLESelectionConfig,
    ) -> PreparedEvaluationSet:
        eligible: list[_EligibleRow] = []
        rejection_counts: Counter[str] = Counter()
        seen_ids: set[str] = set()
        for raw_row in rows:
            parsed, rejection = _eligible_row(raw_row, config=config)
            if rejection is not None:
                rejection_counts[rejection] += 1
                continue
            assert parsed is not None
            if parsed.entry.sample_id in seen_ids:
                rejection_counts["duplicate_id"] += 1
                continue
            seen_ids.add(parsed.entry.sample_id)
            eligible.append(parsed)

        eligible.sort(key=lambda item: item.entry.sample_id)
        if len(eligible) < config.sample_count:
            raise ValueError(
                f"HLE eligible pool has fewer than {config.sample_count} eligible rows"
            )
        quotas = _category_quotas(eligible, sample_count=config.sample_count)
        selected = _select_rows(eligible, quotas=quotas, seed=config.seed)
        manifest_entries = tuple(item.entry for item in selected)
        manifest_payload = _unsigned_manifest_payload(
            dataset_revision=config.revision,
            seed=config.seed,
            requested_sample_count=config.sample_count,
            eligible_count=len(eligible),
            rejection_counts=dict(sorted(rejection_counts.items())),
            category_quotas=quotas,
            entries=manifest_entries,
        )
        manifest_sha256 = _canonical_sha256(manifest_payload)
        gold_store = EvaluationGoldStore(
            manifest_sha256=manifest_sha256,
            labels={item.entry.sample_id: item.gold_label for item in selected},
        )
        return PreparedEvaluationSet(
            dataset_revision=config.revision,
            seed=config.seed,
            requested_sample_count=config.sample_count,
            eligible_count=len(eligible),
            rejection_counts=dict(sorted(rejection_counts.items())),
            category_quotas=quotas,
            manifest_entries=manifest_entries,
            manifest_sha256=manifest_sha256,
            gold_store=gold_store,
        )


def _eligible_row(
    row: Mapping[str, Any],
    *,
    config: HLESelectionConfig,
) -> tuple[_EligibleRow | None, str | None]:
    if not isinstance(row, Mapping):
        return None, "invalid_row"
    if row.get("answer_type") != "multipleChoice":
        return None, "answer_type"
    if not _empty_image(row.get("image")):
        return None, "image_present"
    required_values: dict[str, str] = {}
    for field_name in ("id", "question", "answer", "category"):
        value = row.get(field_name)
        if not isinstance(value, str) or not value.strip():
            return None, f"missing_{field_name}"
        required_values[field_name] = value.strip()

    parsed_question = _parse_question(required_values["question"])
    if parsed_question is None:
        return None, "unparseable_choices"
    canonical_question, choices = parsed_question
    gold_label, gold_error = _canonical_gold_label(
        required_values["answer"],
        choices,
    )
    if gold_error is not None:
        return None, gold_error
    assert gold_label is not None
    if not _initializer_matches(canonical_question, choices):
        return None, "initializer_mismatch"

    entry = HLEManifestEntry(
        sample_id=required_values["id"],
        question=canonical_question,
        choices=choices,
        category=required_values["category"],
        answer_type="multipleChoice",
        dataset_revision=config.revision,
        original_question_sha256=_text_sha256(required_values["question"]),
        canonical_question_sha256=_text_sha256(canonical_question),
    )
    return _EligibleRow(entry=entry, gold_label=gold_label), None


def _parse_question(question: str) -> tuple[str, dict[str, str]] | None:
    header = _ANSWER_CHOICES_HEADER.search(question)
    if header is not None:
        stem = question[: header.start()]
        choice_body = question[header.end() :]
    else:
        block_matches = list(_CHOICE_BLOCK_LINE.finditer(question))
        if len(block_matches) >= 2:
            first_choice = block_matches[0].start()
            stem = question[:first_choice]
            choice_body = question[first_choice:]
        else:
            inline_matches = list(_CHOICE_INLINE.finditer(question))
            if len(inline_matches) < 2:
                return None
            first_choice = inline_matches[0].start(1)
            stem = question[:first_choice]
            choice_body = question[first_choice:]
    normalized_stem = _collapse_whitespace(stem)
    if not normalized_stem:
        return None
    choices = _parse_choices(choice_body)
    if len(choices) < 2:
        return None
    canonical = (
        f"{normalized_stem}\n\nAnswer Choices:\n"
        + "\n".join(f"{label}. {text}" for label, text in choices.items())
    )
    return canonical, choices


def _parse_choices(choice_body: str) -> dict[str, str]:
    matches = list(_CHOICE_BLOCK_LINE.finditer(choice_body))
    if len(matches) < 2:
        matches = list(_CHOICE_INLINE.finditer(choice_body))
    choices: dict[str, str] = {}
    for match in matches:
        label = match.group(1).strip().upper()
        text = _collapse_whitespace(match.group(2))
        if not text or label in choices:
            return {}
        choices[label] = text
    return choices


def _canonical_gold_label(
    answer: str,
    choices: Mapping[str, str],
) -> tuple[str | None, str | None]:
    label_match = _GOLD_LABEL.fullmatch(answer.strip())
    if label_match is not None:
        label = label_match.group(1).upper()
        return (label, None) if label in choices else (None, "gold_not_mappable")
    normalized_answer = _collapse_whitespace(answer).casefold()
    matches = [
        label
        for label, text in choices.items()
        if _collapse_whitespace(text).casefold() == normalized_answer
    ]
    if len(matches) == 1:
        return matches[0], None
    if len(matches) > 1:
        return None, "ambiguous_gold"
    return None, "gold_not_mappable"


def _initializer_matches(question: str, choices: Mapping[str, str]) -> bool:
    try:
        initialized = BayesProbeInitializer().initialize(
            InitializeRunInput(run_id="hle_adapter_validation", problem=question)
        )
    except ValueError:
        return False
    hypotheses = initialized.belief_state.hypotheses
    if tuple(hypothesis.id for hypothesis in hypotheses) != tuple(choices):
        return False
    return all(
        _collapse_whitespace(choices[hypothesis.id])
        in _collapse_whitespace(hypothesis.statement)
        for hypothesis in hypotheses
    )


def _category_quotas(
    eligible: Sequence[_EligibleRow], *, sample_count: int
) -> dict[str, int]:
    counts = Counter(item.entry.category for item in eligible)
    exact = {
        category: Fraction(count * sample_count, len(eligible))
        for category, count in counts.items()
    }
    quotas = {category: int(value) for category, value in exact.items()}
    remaining = sample_count - sum(quotas.values())
    ranked_categories = sorted(
        exact,
        key=lambda category: (-(exact[category] - quotas[category]), category),
    )
    for category in ranked_categories[:remaining]:
        quotas[category] += 1
    return dict(sorted(quotas.items()))


def _select_rows(
    eligible: Sequence[_EligibleRow],
    *,
    quotas: Mapping[str, int],
    seed: str,
) -> list[_EligibleRow]:
    grouped: dict[str, list[_EligibleRow]] = defaultdict(list)
    for item in eligible:
        grouped[item.entry.category].append(item)
    selected: list[_EligibleRow] = []
    for category in sorted(grouped):
        ranked = sorted(
            grouped[category],
            key=lambda item: (_seeded_hash(seed, item.entry.sample_id), item.entry.sample_id),
        )
        selected.extend(ranked[: quotas[category]])
    return sorted(
        selected,
        key=lambda item: (_seeded_hash(seed, item.entry.sample_id), item.entry.sample_id),
    )


def _unsigned_manifest_payload(
    *,
    dataset_revision: str,
    seed: str,
    requested_sample_count: int,
    eligible_count: int,
    rejection_counts: Mapping[str, int],
    category_quotas: Mapping[str, int],
    entries: Sequence[HLEManifestEntry],
) -> dict[str, Any]:
    return {
        "artifact_version": "0.1",
        "dataset_revision": dataset_revision,
        "seed": seed,
        "requested_sample_count": requested_sample_count,
        "eligible_count": eligible_count,
        "rejection_counts": dict(rejection_counts),
        "category_quotas": dict(category_quotas),
        "selection_algorithm": "proportional_largest_remainder_sha256_v0.1",
        "items": [entry.to_dict() for entry in entries],
    }


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _seeded_hash(seed: str, sample_id: str) -> str:
    return _text_sha256(f"{seed}:{sample_id}")


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _collapse_whitespace(value: str) -> str:
    return " ".join(value.split())


def _empty_image(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    if isinstance(value, (list, dict)):
        return not value
    return False


__all__ = [
    "EvaluationGoldStore",
    "HLEDatasetAdapter",
    "HLEManifestEntry",
    "HLESelectionConfig",
    "PreparedEvaluationSet",
]
