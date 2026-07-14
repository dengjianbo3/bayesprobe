from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from threading import Lock
from typing import Any, ClassVar

from pydantic import BaseModel


class TrialArtifactStore:
    _REDACTION_MARKER = "[REDACTED]"
    _root_locks: ClassVar[dict[Path, Lock]] = {}
    _root_locks_lock: ClassVar[Lock] = Lock()

    def __init__(self, root: Path, *, restricted_values: tuple[str, ...]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.root = self.root.resolve()
        self._restricted_values = tuple(
            sorted({value for value in restricted_values if value}, key=lambda value: (-len(value), value))
        )
        self._replacement = (
            ""
            if any(value in self._REDACTION_MARKER for value in self._restricted_values)
            else self._REDACTION_MARKER
        )
        with self._root_locks_lock:
            self._lock = self._root_locks.setdefault(self.root, Lock())

    def append_plan(self, payload: Any) -> None:
        self._append("plans.jsonl", payload)

    def append_observation(self, payload: Any) -> None:
        self._append("environment_actions.jsonl", payload)

    def append_provider_call(self, payload: Any) -> None:
        self._append("provider_telemetry.jsonl", payload)

    def append_error(self, payload: Any) -> None:
        self._append("errors.jsonl", payload)

    def write_summary(self, payload: Mapping[str, Any]) -> None:
        with self._lock:
            contents = json.dumps(
                self._redact(payload), ensure_ascii=False, sort_keys=True, indent=2
            ) + "\n"
            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=self.root,
                    prefix=".summary-",
                    suffix=".tmp",
                    delete=False,
                ) as handle:
                    temporary_path = Path(handle.name)
                    handle.write(contents)
                os.replace(temporary_path, self.root / "summary.json")
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)

    def _append(self, filename: str, payload: Any) -> None:
        with self._lock:
            line = json.dumps(self._redact(payload), ensure_ascii=False, sort_keys=True) + "\n"
            with (self.root / filename).open("a", encoding="utf-8") as handle:
                handle.write(line)

    def _redact(self, value: Any) -> Any:
        if isinstance(value, BaseModel):
            return self._redact(value.model_dump(mode="json"))
        if isinstance(value, str):
            return self._redact_string(value)
        if isinstance(value, Mapping):
            redacted: dict[str, Any] = {}
            for key, item in value.items():
                safe_key = self._redact_string(str(key))
                if safe_key in redacted:
                    raise ValueError("redacted mapping keys collide")
                redacted[safe_key] = self._redact(item)
            return redacted
        if isinstance(value, Sequence):
            return [self._redact(item) for item in value]
        return value

    def _redact_string(self, value: str) -> str:
        while True:
            redacted = value
            for restricted in self._restricted_values:
                redacted = redacted.replace(restricted, self._replacement)
            if redacted == value:
                return redacted
            value = redacted
