from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist


@dataclass(frozen=True)
class PairedContingency:
    both_correct: int
    bayesprobe_only: int
    direct_only: int
    both_wrong: int

    @property
    def total(self) -> int:
        return (
            self.both_correct
            + self.bayesprobe_only
            + self.direct_only
            + self.both_wrong
        )

    @property
    def accuracy_difference(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.bayesprobe_only - self.direct_only) / self.total


def wilson_interval(
    *,
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    if type(successes) is not int or type(total) is not int:
        raise ValueError("Wilson counts must be integers")
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("Wilson counts must satisfy 0 <= successes <= total")
    if not 0 < confidence < 1:
        raise ValueError("Wilson confidence must be between zero and one")
    if total == 0:
        return 0.0, 1.0
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def paired_contingency(
    bayesprobe_correct: Sequence[bool],
    direct_correct: Sequence[bool],
) -> PairedContingency:
    if len(bayesprobe_correct) != len(direct_correct):
        raise ValueError("paired correctness sequences must have the same length")
    both_correct = bayesprobe_only = direct_only = both_wrong = 0
    for bayesprobe_value, direct_value in zip(
        bayesprobe_correct,
        direct_correct,
        strict=True,
    ):
        if bool(bayesprobe_value) and bool(direct_value):
            both_correct += 1
        elif bool(bayesprobe_value):
            bayesprobe_only += 1
        elif bool(direct_value):
            direct_only += 1
        else:
            both_wrong += 1
    return PairedContingency(
        both_correct=both_correct,
        bayesprobe_only=bayesprobe_only,
        direct_only=direct_only,
        both_wrong=both_wrong,
    )


def exact_mcnemar_p_value(*, bayesprobe_only: int, direct_only: int) -> float:
    for value in (bayesprobe_only, direct_only):
        if type(value) is not int or value < 0:
            raise ValueError("McNemar discordant counts must be non-negative integers")
    discordant = bayesprobe_only + direct_only
    if discordant == 0:
        return 1.0
    lower_tail = sum(
        math.comb(discordant, index)
        for index in range(min(bayesprobe_only, direct_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2 * lower_tail)


def paired_bootstrap_interval(
    bayesprobe_correct: Sequence[bool],
    direct_correct: Sequence[bool],
    *,
    resamples: int = 10_000,
    seed: str = "20260711",
    confidence: float = 0.95,
) -> tuple[float, float]:
    if len(bayesprobe_correct) != len(direct_correct):
        raise ValueError("paired correctness sequences must have the same length")
    if not bayesprobe_correct:
        return 0.0, 0.0
    if type(resamples) is not int or resamples < 1:
        raise ValueError("bootstrap resamples must be positive")
    if not 0 < confidence < 1:
        raise ValueError("bootstrap confidence must be between zero and one")
    seed_value = int.from_bytes(hashlib.sha256(seed.encode("utf-8")).digest()[:8])
    generator = random.Random(seed_value)
    size = len(bayesprobe_correct)
    differences: list[float] = []
    for _ in range(resamples):
        difference_sum = 0
        for _ in range(size):
            index = generator.randrange(size)
            difference_sum += int(bool(bayesprobe_correct[index])) - int(
                bool(direct_correct[index])
            )
        differences.append(difference_sum / size)
    differences.sort()
    alpha = (1 - confidence) / 2
    return _quantile(differences, alpha), _quantile(differences, 1 - alpha)


def multiclass_brier_score(
    probabilities: Mapping[str, float],
    *,
    gold_label: str,
) -> float:
    _validate_gold_distribution(probabilities, gold_label)
    return math.fsum(
        (float(probability) - (1.0 if label == gold_label else 0.0)) ** 2
        for label, probability in probabilities.items()
    )


def multiclass_log_loss(
    probabilities: Mapping[str, float],
    *,
    gold_label: str,
    floor: float = 1e-6,
) -> float:
    _validate_gold_distribution(probabilities, gold_label)
    if not 0 < floor <= 1:
        raise ValueError("log-loss floor must be in the interval (0, 1]")
    return -math.log(max(float(probabilities[gold_label]), floor))


def expected_calibration_error(
    outcomes: Sequence[tuple[float, bool]],
    *,
    max_bins: int = 10,
) -> float:
    if not outcomes:
        return 0.0
    if type(max_bins) is not int or max_bins < 1:
        raise ValueError("ECE max_bins must be positive")
    normalized: list[tuple[float, bool]] = []
    for confidence, correct in outcomes:
        if (
            type(confidence) not in (int, float)
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            raise ValueError("ECE confidence values must be finite and in [0, 1]")
        normalized.append((float(confidence), bool(correct)))
    normalized.sort(key=lambda item: item[0])
    bin_count = min(max_bins, len(normalized))
    quotient, remainder = divmod(len(normalized), bin_count)
    offset = 0
    weighted_error = 0.0
    for bin_index in range(bin_count):
        bin_size = quotient + (1 if bin_index < remainder else 0)
        current_bin = normalized[offset : offset + bin_size]
        offset += bin_size
        mean_confidence = math.fsum(item[0] for item in current_bin) / bin_size
        accuracy = sum(int(item[1]) for item in current_bin) / bin_size
        weighted_error += bin_size / len(normalized) * abs(accuracy - mean_confidence)
    return weighted_error


def probability_entropy(probabilities: Mapping[str, float]) -> float:
    _validate_probability_values(probabilities)
    return -math.fsum(
        float(value) * math.log(float(value))
        for value in probabilities.values()
        if value > 0
    )


def top_two_margin(probabilities: Mapping[str, float]) -> float:
    _validate_probability_values(probabilities)
    if len(probabilities) < 2:
        raise ValueError("top-two margin requires at least two probabilities")
    top_values = sorted((float(value) for value in probabilities.values()), reverse=True)
    return top_values[0] - top_values[1]


def _quantile(values: Sequence[float], probability: float) -> float:
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def _validate_gold_distribution(
    probabilities: Mapping[str, float], gold_label: str
) -> None:
    _validate_probability_values(probabilities)
    if gold_label not in probabilities:
        raise ValueError("gold label must be present in probabilities")


def _validate_probability_values(probabilities: Mapping[str, float]) -> None:
    if not isinstance(probabilities, Mapping) or not probabilities:
        raise ValueError("probabilities must be a non-empty object")
    for value in probabilities.values():
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError("probabilities must be finite and in [0, 1]")


__all__ = [
    "PairedContingency",
    "exact_mcnemar_p_value",
    "expected_calibration_error",
    "multiclass_brier_score",
    "multiclass_log_loss",
    "paired_bootstrap_interval",
    "paired_contingency",
    "probability_entropy",
    "top_two_margin",
    "wilson_interval",
]
