from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from threading import Lock
from typing import Any

from pydantic import BaseModel


class TrialArtifactStore:
    def __init__(self, root: Path, *, restricted_values: tuple[str, ...]) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._restricted_values = tuple(value for value in restricted_values if value)
        self._lock = Lock()

    def append_plan(self, payload: Any) -> None:
        self._append("plans.jsonl", payload)

    def append_observation(self, payload: Any) -> None:
        self._append("environment_actions.jsonl", payload)

    def append_provider_call(self, payload: Any) -> None:
        self._append("provider_telemetry.jsonl", payload)

    def append_error(self, payload: Any) -> None:
        self._append("errors.jsonl", payload)

    def write_summary(self, payload: Mapping[str, Any]) -> None:
        safe = self._redact(payload)
        (self.root / "summary.json").write_text(
            json.dumps(safe, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    def _append(self, filename: str, payload: Any) -> None:
        line = json.dumps(self._redact(payload), ensure_ascii=False, sort_keys=True) + "\n"
        with self._lock:
            with (self.root / filename).open("a", encoding="utf-8") as handle:
                handle.write(line)

    def _redact(self, value: Any) -> Any:
        if isinstance(value, BaseModel):
            return self._redact(value.model_dump(mode="json"))
        if isinstance(value, str):
            for restricted in self._restricted_values:
                value = value.replace(restricted, "[REDACTED]")
            return value
        if isinstance(value, Mapping):
            return {self._redact(str(key)): self._redact(item) for key, item in value.items()}
        if isinstance(value, Sequence):
            return [self._redact(item) for item in value]
        return value
