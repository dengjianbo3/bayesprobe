from __future__ import annotations

from enum import StrEnum
from typing import Literal

from bayesprobe.schemas import BeliefState, FramingMethod


class BeliefLifecycle(StrEnum):
    NATIVE_V02 = "native_v0.2"
    LEGACY_V01_MIGRATION = "legacy_v0.1_migration"

    @property
    def provider_version(self) -> Literal["v0.1", "v0.2"]:
        if self is BeliefLifecycle.NATIVE_V02:
            return "v0.2"
        return "v0.1"


def resolve_belief_lifecycle(belief_state: BeliefState) -> BeliefLifecycle:
    task_frame = belief_state.task_frame
    if (
        task_frame is not None
        and task_frame.framing_method == FramingMethod.LEGACY_MIGRATION
    ):
        return BeliefLifecycle.LEGACY_V01_MIGRATION
    if (
        belief_state.schema_version == "v0.2"
        and task_frame is not None
        and belief_state.frame_state is not None
        and belief_state.evidence_memory is not None
    ):
        return BeliefLifecycle.NATIVE_V02
    raise ValueError(
        "invalid belief lifecycle: requires native v0.2 or explicit legacy migration"
    )
