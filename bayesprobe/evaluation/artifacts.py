from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bayesprobe.evaluation.hle import PreparedEvaluationSet


@dataclass(frozen=True)
class PreparedEvaluationPaths:
    root: Path
    selection_manifest: Path
    gold_store: Path


def write_prepared_evaluation_set(
    root: str | Path,
    prepared: PreparedEvaluationSet,
) -> PreparedEvaluationPaths:
    restricted_root = Path(root)
    restricted_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(restricted_root, 0o700)
    selection_manifest = restricted_root / "selection_manifest.json"
    gold_store = restricted_root / "gold_store.json"
    _atomic_private_json(selection_manifest, prepared.selection_manifest_payload())
    _atomic_private_json(
        gold_store,
        {
            "artifact_version": "0.1",
            "manifest_sha256": prepared.manifest_sha256,
            "items": [
                {"sample_id": sample_id, "gold_label": gold_label}
                for sample_id, gold_label in prepared.gold_store.labels.items()
            ],
        },
    )
    return PreparedEvaluationPaths(
        root=restricted_root,
        selection_manifest=selection_manifest,
        gold_store=gold_store,
    )


def _atomic_private_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        os.chmod(path, 0o600)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary_path.unlink(missing_ok=True)
        raise


__all__ = ["PreparedEvaluationPaths", "write_prepared_evaluation_set"]
