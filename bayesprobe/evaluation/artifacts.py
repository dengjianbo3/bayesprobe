from __future__ import annotations

import json
import hmac
import os
import secrets
import tempfile
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from bayesprobe.evaluation.contracts import ArmCaseResult, EvaluationCase
from bayesprobe.evaluation.hle import PreparedEvaluationSet
from bayesprobe.evaluation.python_probe import (
    PythonExecutionObserver,
    PythonExecutionRecord,
)
from bayesprobe.ledger import JsonlLedgerStore
from bayesprobe.provider_telemetry import (
    JsonlProviderInvocationObserver,
    ProviderInvocationObserver,
    ProviderInvocationRecord,
)


@dataclass(frozen=True)
class PreparedEvaluationPaths:
    root: Path
    selection_manifest: Path
    gold_store: Path


@dataclass(frozen=True)
class CapabilityCasePaths:
    root: Path
    status_path: Path
    result_path: Path
    ledger_path: Path
    provider_invocations_path: Path
    python_executions_path: Path


class CapabilityArtifactStore:
    arm_names = ("direct_flash", "bayesprobe_python")

    def __init__(
        self,
        restricted_root: str | Path,
        identity: Any,
        *,
        secret: bytes | None = None,
    ) -> None:
        self.identity = identity
        self.root = Path(restricted_root) / identity.experiment_id
        _private_directory(self.root)
        self._lock = threading.RLock()
        self._secret_path = self.root / "experiment_secret.bin"
        self._secret = self._load_or_create_secret(secret)
        _atomic_private_json(
            self.root / "experiment_identity.json",
            asdict(identity),
        )

    def pseudonym_for(self, sample_id: str) -> str:
        return hmac.new(
            self._secret,
            sample_id.encode("utf-8"),
            "sha256",
        ).hexdigest()

    def paths_for(self, arm: str, sample_id: str) -> CapabilityCasePaths:
        self._validate_arm(arm)
        case_root = self.root / "arms" / arm / self.pseudonym_for(sample_id)
        return CapabilityCasePaths(
            root=case_root,
            status_path=case_root / "status.json",
            result_path=case_root / "result.json",
            ledger_path=case_root / "ledger.jsonl",
            provider_invocations_path=case_root / "provider_invocations.jsonl",
            python_executions_path=case_root / "python_executions.jsonl",
        )

    def initialize_case(self, arm: str, sample_id: str) -> CapabilityCasePaths:
        paths = self.paths_for(arm, sample_id)
        with self._lock:
            _private_directory(paths.root)
            if not paths.status_path.exists():
                _atomic_private_json(
                    paths.status_path,
                    {
                        "artifact_version": "0.1",
                        "experiment_id": self.identity.experiment_id,
                        "arm": arm,
                        "sample_pseudonym": self.pseudonym_for(sample_id),
                        "state": "pending",
                        "attempt_count": 0,
                        "started_at": None,
                        "completed_at": None,
                    },
                )
            for audit_path in (
                paths.ledger_path,
                paths.provider_invocations_path,
                paths.python_executions_path,
            ):
                if not audit_path.exists():
                    _atomic_private_bytes(audit_path, b"")
        return paths

    def status(self, arm: str, sample_id: str) -> dict[str, Any]:
        paths = self.initialize_case(arm, sample_id)
        payload = json.loads(paths.status_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("capability case status must be an object")
        return payload

    def should_run(
        self,
        arm: str,
        sample_id: str,
        *,
        now: datetime | None = None,
        stale_after: timedelta = timedelta(hours=24),
    ) -> bool:
        status = self.status(arm, sample_id)
        if status["state"] == "pending":
            return True
        if status["state"] in {"completed", "terminal_failed"}:
            return False
        if status["state"] != "running":
            raise ValueError(f"invalid capability case state: {status['state']}")
        started_at = status.get("started_at")
        if not isinstance(started_at, str):
            return True
        started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        current = now or datetime.now(UTC)
        return current - started >= stale_after

    def mark_running(self, arm: str, sample_id: str) -> None:
        with self._lock:
            status = self.status(arm, sample_id)
            if status["state"] in {"completed", "terminal_failed"}:
                raise ValueError("terminal case is immutable")
            status.update(
                {
                    "state": "running",
                    "attempt_count": int(status.get("attempt_count", 0)) + 1,
                    "started_at": _utc_now_text(),
                    "completed_at": None,
                }
            )
            _atomic_private_json(self.paths_for(arm, sample_id).status_path, status)

    def write_terminal_result(self, result: ArmCaseResult) -> None:
        with self._lock:
            paths = self.initialize_case(result.arm, result.sample_id)
            status = self.status(result.arm, result.sample_id)
            if status["state"] in {"completed", "terminal_failed"}:
                raise ValueError("terminal case is immutable")
            if status["state"] != "running":
                raise ValueError("capability case must be running before terminal write")
            _atomic_private_json(paths.result_path, asdict(result))
            status.update(
                {
                    "state": result.state,
                    "completed_at": _utc_now_text(),
                }
            )
            _atomic_private_json(paths.status_path, status)

    def load_result(self, arm: str, sample_id: str) -> dict[str, Any]:
        path = self.paths_for(arm, sample_id).result_path
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("capability arm result must be an object")
        return payload

    def all_terminal(self, cases: list[EvaluationCase]) -> bool:
        return all(
            self.status(arm, case.sample_id)["state"]
            in {"completed", "terminal_failed"}
            for case in cases
            for arm in self.arm_names
        )

    def ledger_for(self, arm: str, case: EvaluationCase) -> JsonlLedgerStore:
        paths = self.initialize_case(arm, case.sample_id)
        return JsonlLedgerStore(paths.ledger_path)

    def provider_observer(self) -> ProviderInvocationObserver:
        return _CapabilityProviderObserver(self)

    def python_observer_for(
        self,
        arm: str,
        case: EvaluationCase,
    ) -> PythonExecutionObserver:
        paths = self.initialize_case(arm, case.sample_id)
        return _CapabilityPythonObserver(paths.python_executions_path)

    def _write_status_for_test(
        self,
        arm: str,
        sample_id: str,
        payload: dict[str, Any],
    ) -> None:
        _atomic_private_json(self.paths_for(arm, sample_id).status_path, payload)

    def _load_or_create_secret(self, supplied: bytes | None) -> bytes:
        if self._secret_path.exists():
            stored = self._secret_path.read_bytes()
            if supplied is not None and not hmac.compare_digest(stored, supplied):
                raise ValueError("capability artifact secret does not match existing store")
            return stored
        value = supplied if supplied is not None else secrets.token_bytes(32)
        if not isinstance(value, bytes) or len(value) < 32:
            raise ValueError("capability artifact secret must contain at least 32 bytes")
        _atomic_private_bytes(self._secret_path, value)
        return value

    @classmethod
    def _validate_arm(cls, arm: str) -> None:
        if arm not in cls.arm_names:
            raise ValueError(f"unsupported capability arm: {arm}")


class _CapabilityProviderObserver:
    def __init__(self, store: CapabilityArtifactStore) -> None:
        self._store = store
        self._observers: dict[tuple[str, str], JsonlProviderInvocationObserver] = {}
        self._lock = threading.Lock()

    def observe(self, record: ProviderInvocationRecord) -> None:
        arm = record.context.arm
        sample_id = record.context.sample_id
        if arm is None or sample_id is None:
            raise ValueError("capability provider telemetry requires arm and sample context")
        key = (arm, sample_id)
        with self._lock:
            observer = self._observers.get(key)
            if observer is None:
                paths = self._store.initialize_case(arm, sample_id)
                observer = JsonlProviderInvocationObserver(
                    paths.provider_invocations_path
                )
                self._observers[key] = observer
        observer.observe(record)


class _CapabilityPythonObserver:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()

    def observe(self, record: PythonExecutionRecord) -> None:
        line = (
            json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8")
        with self._lock:
            descriptor = os.open(
                self._path,
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                0o600,
            )
            try:
                os.write(descriptor, line)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)


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


def _atomic_private_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        os.chmod(path, 0o600)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary_path.unlink(missing_ok=True)
        raise


def _private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path, 0o700)


def _utc_now_text() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "CapabilityArtifactStore",
    "CapabilityCasePaths",
    "PreparedEvaluationPaths",
    "write_prepared_evaluation_set",
]
