from __future__ import annotations

from enum import StrEnum
from typing import Literal

from bayesprobe.migrations import (
    RECOGNIZED_V01_TO_V02_MIGRATION_MARKERS,
    _has_v01_migration_receipt,
)
from bayesprobe.schemas import BeliefState, FramingMethod


class BeliefLifecycle(StrEnum):
    NATIVE_V02 = "native_v0.2"
    LEGACY_V01_MIGRATION = "legacy_v0.1_migration"

    @property
    def provider_version(self) -> Literal["v0.1", "v0.2"]:
        if self is BeliefLifecycle.NATIVE_V02:
            return "v0.2"
        return "v0.1"


def _invalid_lifecycle_error() -> ValueError:
    return ValueError(
        "invalid belief lifecycle: requires native v0.2 or explicit legacy migration"
    )


def _validated_runtime_envelope(belief_state: BeliefState) -> BeliefState:
    task_frame = belief_state.task_frame
    if (
        belief_state.schema_version != "v0.2"
        or task_frame is None
        or task_frame.schema_version != "v0.2"
        or belief_state.frame_state is None
        or belief_state.evidence_memory is None
    ):
        raise _invalid_lifecycle_error()
    try:
        return BeliefState.model_validate(
            belief_state.model_dump(mode="python")
        )
    except (TypeError, ValueError):
        raise _invalid_lifecycle_error() from None


def resolve_belief_lifecycle(belief_state: BeliefState) -> BeliefLifecycle:
    validated = _validated_runtime_envelope(belief_state)
    task_frame = validated.task_frame
    if task_frame is None:
        raise _invalid_lifecycle_error()
    has_migration_marker = "migration" in task_frame.framing_trace
    marker = task_frame.framing_trace.get("migration")
    if task_frame.framing_method == FramingMethod.LEGACY_MIGRATION:
        if (
            not isinstance(marker, str)
            or marker not in RECOGNIZED_V01_TO_V02_MIGRATION_MARKERS
            or not _has_v01_migration_receipt(belief_state)
        ):
            raise _invalid_lifecycle_error()
        return BeliefLifecycle.LEGACY_V01_MIGRATION
    if has_migration_marker:
        raise _invalid_lifecycle_error()
    return BeliefLifecycle.NATIVE_V02
