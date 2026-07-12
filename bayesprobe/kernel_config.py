from __future__ import annotations

from dataclasses import dataclass
import math


def _validate_probability(name: str, value: float) -> None:
    if type(value) not in {int, float} or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between zero and one")


def _validate_positive_integer(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class OpenCoveragePolicy:
    initial_unresolved_mass: float = 0.50
    minimum_unresolved_reserve: float = 0.05

    def __post_init__(self) -> None:
        _validate_probability(
            "initial_unresolved_mass",
            self.initial_unresolved_mass,
        )
        _validate_probability(
            "minimum_unresolved_reserve",
            self.minimum_unresolved_reserve,
        )
        if self.minimum_unresolved_reserve > self.initial_unresolved_mass:
            raise ValueError(
                "minimum_unresolved_reserve cannot exceed initial_unresolved_mass"
            )


@dataclass(frozen=True)
class FrameAdequacyPolicyConfig:
    high_verifiability_threshold: float = 0.75
    moderate_verifiability_threshold: float = 0.50
    required_distinct_moderate_roots: int = 2

    def __post_init__(self) -> None:
        _validate_probability(
            "high_verifiability_threshold",
            self.high_verifiability_threshold,
        )
        _validate_probability(
            "moderate_verifiability_threshold",
            self.moderate_verifiability_threshold,
        )
        if (
            self.moderate_verifiability_threshold
            > self.high_verifiability_threshold
        ):
            raise ValueError(
                "moderate_verifiability_threshold cannot exceed "
                "high_verifiability_threshold"
            )
        _validate_positive_integer(
            "required_distinct_moderate_roots",
            self.required_distinct_moderate_roots,
        )


@dataclass(frozen=True)
class ExpansionPolicy:
    max_frame_revisions: int = 3
    max_active_hypotheses: int = 8
    max_repair_attempts: int = 1

    def __post_init__(self) -> None:
        _validate_positive_integer("max_frame_revisions", self.max_frame_revisions)
        _validate_positive_integer("max_active_hypotheses", self.max_active_hypotheses)
        _validate_positive_integer("max_repair_attempts", self.max_repair_attempts)


@dataclass(frozen=True)
class ProjectionPolicy:
    exact_top_threshold: float = 0.60
    exact_margin_threshold: float = 0.15
    exact_max_unresolved_mass: float = 0.20
    max_repair_attempts: int = 1

    def __post_init__(self) -> None:
        _validate_probability("exact_top_threshold", self.exact_top_threshold)
        _validate_probability("exact_margin_threshold", self.exact_margin_threshold)
        _validate_probability(
            "exact_max_unresolved_mass",
            self.exact_max_unresolved_mass,
        )
        _validate_positive_integer("max_repair_attempts", self.max_repair_attempts)


__all__ = [
    "ExpansionPolicy",
    "FrameAdequacyPolicyConfig",
    "OpenCoveragePolicy",
    "ProjectionPolicy",
]
