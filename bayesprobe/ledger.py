from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from bayesprobe.schemas import utc_now


class JsonlLedgerStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record_type: str, record: BaseModel | dict[str, Any]) -> None:
        payload = record.model_dump(mode="json") if isinstance(record, BaseModel) else record
        envelope = {
            "record_type": record_type,
            "recorded_at": utc_now().isoformat(),
            "payload": payload,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(envelope, ensure_ascii=False, sort_keys=True) + "\n")

    def read_all(self, record_type: str | None = None) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                envelope = json.loads(line)
                if record_type is None or envelope["record_type"] == record_type:
                    records.append(envelope)
        return records
