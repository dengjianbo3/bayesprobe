import math

import pytest

from bayesprobe.evaluation.statistics import (
    exact_mcnemar_p_value,
    expected_calibration_error,
    multiclass_brier_score,
    multiclass_log_loss,
    paired_bootstrap_interval,
    paired_contingency,
    probability_entropy,
    top_two_margin,
    wilson_interval,
)


def test_wilson_interval_matches_known_50_of_100_value():
    low, high = wilson_interval(successes=50, total=100)

    assert low == pytest.approx(0.40383153, abs=1e-7)
    assert high == pytest.approx(0.59616847, abs=1e-7)


def test_wilson_interval_handles_boundary_counts():
    assert wilson_interval(successes=0, total=0) == (0.0, 1.0)
    low, high = wilson_interval(successes=0, total=10)
    assert low == 0.0
    assert 0 < high < 0.5


def test_paired_contingency_counts_all_four_outcomes():
    counts = paired_contingency(
        bayesprobe_correct=[True, True, False, False, True],
        direct_correct=[True, False, True, False, True],
    )

    assert counts.both_correct == 2
    assert counts.bayesprobe_only == 1
    assert counts.direct_only == 1
    assert counts.both_wrong == 1
    assert counts.accuracy_difference == 0.0


def test_paired_contingency_rejects_different_lengths():
    with pytest.raises(ValueError, match="same length"):
        paired_contingency([True], [True, False])


def test_exact_mcnemar_uses_two_sided_binomial_probability():
    assert exact_mcnemar_p_value(bayesprobe_only=3, direct_only=1) == pytest.approx(
        0.625
    )
    assert exact_mcnemar_p_value(bayesprobe_only=0, direct_only=0) == 1.0


def test_paired_bootstrap_is_seeded_and_bounds_observed_differences():
    bayesprobe = [True, True, True, False, False, True]
    direct = [True, False, False, True, False, True]

    first = paired_bootstrap_interval(
        bayesprobe,
        direct,
        resamples=1000,
        seed="20260711",
    )
    second = paired_bootstrap_interval(
        bayesprobe,
        direct,
        resamples=1000,
        seed="20260711",
    )

    assert first == second
    assert -1 <= first[0] <= first[1] <= 1
    assert first[0] <= (4 / 6 - 3 / 6) <= first[1]


def test_multiclass_brier_and_log_loss_match_known_distribution():
    probabilities = {"A": 0.1, "B": 0.7, "C": 0.2}

    assert multiclass_brier_score(probabilities, gold_label="B") == pytest.approx(
        0.14
    )
    assert multiclass_log_loss(probabilities, gold_label="B") == pytest.approx(
        -math.log(0.7)
    )


def test_log_loss_clips_zero_gold_probability():
    assert multiclass_log_loss(
        {"A": 1.0, "B": 0.0},
        gold_label="B",
    ) == pytest.approx(-math.log(1e-6))


def test_expected_calibration_error_uses_equal_frequency_bins():
    outcomes = [(0.9, True), (0.8, True), (0.6, False), (0.5, False)]

    assert expected_calibration_error(outcomes, max_bins=2) == pytest.approx(0.35)


def test_entropy_and_top_two_margin_match_known_distribution():
    probabilities = {"A": 0.1, "B": 0.7, "C": 0.2}

    assert probability_entropy(probabilities) == pytest.approx(
        -(0.1 * math.log(0.1) + 0.7 * math.log(0.7) + 0.2 * math.log(0.2))
    )
    assert top_two_margin(probabilities) == pytest.approx(0.5)
