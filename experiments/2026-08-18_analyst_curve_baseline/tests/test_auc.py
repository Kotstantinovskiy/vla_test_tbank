import math

import pytest

from analyst_curve_baseline.build import BUDGETS, mean_curve, trapezoid_auc


def test_budgets_cover_assignment_grid():
    assert BUDGETS == [0, 1, 2, 3, 5, 10, 25]


def test_constant_curve_auc_is_the_constant():
    points = {k: 0.5 for k in BUDGETS}
    auc = trapezoid_auc(points)
    assert auc["auc_raw"] == pytest.approx(12.5)
    assert auc["auc_normalized"] == pytest.approx(0.5)
    assert auc["auc_log2_normalized"] == pytest.approx(0.5)


def test_step_curve_auc_matches_hand_computation():
    # 0 at k=0, then 1.0 from k=1 onward.
    points = {k: (0.0 if k == 0 else 1.0) for k in BUDGETS}
    auc = trapezoid_auc(points)
    # trapezoid on [0,1] contributes 0.5, the rest 24 full units.
    assert auc["auc_raw"] == pytest.approx(24.5)
    assert auc["auc_normalized"] == pytest.approx(24.5 / 25)
    # log2 spacing: first segment (0->1) has width 1 of total log2(26).
    expected_log = (0.5 * 1 + (math.log2(26) - 1)) / math.log2(26)
    assert auc["auc_log2_normalized"] == pytest.approx(expected_log)


def test_log_auc_rewards_early_gains_more_than_late_gains():
    early = {0: 0.0, 1: 0.8, 2: 0.8, 3: 0.8, 5: 0.8, 10: 0.8, 25: 0.8}
    late = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 5: 0.0, 10: 0.8, 25: 0.8}
    assert (
        trapezoid_auc(early)["auc_log2_normalized"]
        > trapezoid_auc(late)["auc_log2_normalized"] + 0.3
    )


def test_mean_curve_averages_tasks():
    rates = {0: {k: 1.0 for k in BUDGETS}, 1: {k: 0.0 for k in BUDGETS}}
    curve = mean_curve(rates, [0, 1])
    assert all(v == pytest.approx(0.5) for v in curve.values())
