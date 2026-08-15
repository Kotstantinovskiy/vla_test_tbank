import pytest

from vla_cost_curve.aggregate import wilson_interval


def test_wilson_interval_contains_observed_rate():
    low, high = wilson_interval(7, 10)
    assert low < 0.7 < high


def test_wilson_all_success_is_bounded():
    low, high = wilson_interval(20, 20)
    assert 0 < low < 1
    assert high == pytest.approx(1.0)

