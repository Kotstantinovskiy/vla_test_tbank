import pytest

from vla_cost_curve.selection import build_manifest, select_first_k


def test_selects_first_matching_global_episode_ids():
    tasks = [["other"], ["target"], ["target", "alias"], ["other"], ["target"]]
    assert select_first_k(tasks, "target", 2) == [1, 2]


def test_fails_if_budget_is_not_available():
    with pytest.raises(ValueError, match="only 1 episodes"):
        select_first_k([["target"]], "target", 2)


def test_manifest_uses_explicit_suite_environment_mapping():
    tasks = [
        ["open the middle drawer of the cabinet"],
        ["put the wine bottle on the rack"],
        ["open the top drawer and put the bowl inside"],
    ]
    manifest = build_manifest(tasks, budgets=(1,))
    assert [manifest["tasks"][str(i)]["env_task_id"] for i in range(3)] == [0, 9, 3]
