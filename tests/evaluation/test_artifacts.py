import json
import stat
from pathlib import Path

from bayesprobe.evaluation.artifacts import write_prepared_evaluation_set
from bayesprobe.evaluation.hle import HLEDatasetAdapter, HLESelectionConfig


REVISION = "b" * 40
CANARY = "SYNTHETIC-CANARY-DO-NOT-COPY"


def make_row(sample_id: str):
    return {
        "id": sample_id,
        "category": "synthetic-math",
        "question": "What is 2 + 2?\nA. 3\nB. 4\nC. 5",
        "answer": "B",
        "answer_type": "multipleChoice",
        "image": None,
        "rationale": "Private rationale text.",
        "canary": CANARY,
    }


def test_prepared_set_writes_separate_runtime_manifest_and_gold_store(tmp_path: Path):
    prepared = HLEDatasetAdapter().prepare_rows(
        [make_row("synthetic_1"), make_row("synthetic_2")],
        HLESelectionConfig(revision=REVISION, sample_count=2),
    )

    paths = write_prepared_evaluation_set(tmp_path / "restricted", prepared)

    manifest = json.loads(paths.selection_manifest.read_text(encoding="utf-8"))
    gold = json.loads(paths.gold_store.read_text(encoding="utf-8"))
    assert {item["sample_id"] for item in manifest["items"]} == {
        "synthetic_1",
        "synthetic_2",
    }
    assert all("gold_label" not in item for item in manifest["items"])
    assert all(set(item) == {"gold_label", "sample_id"} for item in gold["items"])
    assert gold["manifest_sha256"] == prepared.manifest_sha256
    assert "Private rationale text." not in paths.selection_manifest.read_text(
        encoding="utf-8"
    )
    assert CANARY not in paths.selection_manifest.read_text(encoding="utf-8")
    assert CANARY not in paths.gold_store.read_text(encoding="utf-8")


def test_restricted_artifact_permissions_are_private(tmp_path: Path):
    prepared = HLEDatasetAdapter().prepare_rows(
        [make_row("synthetic_1")],
        HLESelectionConfig(revision=REVISION, sample_count=1),
    )

    paths = write_prepared_evaluation_set(tmp_path / "restricted", prepared)

    assert stat.S_IMODE(paths.root.stat().st_mode) == 0o700
    assert stat.S_IMODE(paths.selection_manifest.stat().st_mode) == 0o600
    assert stat.S_IMODE(paths.gold_store.stat().st_mode) == 0o600
