import pytest

from vla_cost_curve.selection import select_first_k


def test_selects_first_matching_global_episode_ids():
    tasks = [["other"], ["target"], ["target", "alias"], ["other"], ["target"]]
    assert select_first_k(tasks, "target", 2) == [1, 2]


def test_fails_if_budget_is_not_available():
    with pytest.raises(ValueError, match="only 1 episodes"):
        select_first_k([["target"]], "target", 2)

