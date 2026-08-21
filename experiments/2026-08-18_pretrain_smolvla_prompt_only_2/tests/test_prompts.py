import pytest

from pretrain_smolvla_prompt_only_2.constants import (
    NONSENSE_PROMPT,
    TARGET_ENV_TASK_IDS,
    TARGET_INSTRUCTIONS,
)
from pretrain_smolvla_prompt_only_2.evaluate import (
    assert_environment_instruction,
    prompt_for,
)


def test_all_ten_goal_tasks_covered():
    assert sorted(TARGET_INSTRUCTIONS) == list(range(10))
    assert TARGET_ENV_TASK_IDS == {task_id: task_id for task_id in range(10)}
    assert len(set(TARGET_INSTRUCTIONS.values())) == 10


def test_prompt_conditions_are_explicit():
    assert prompt_for("true", 0) == TARGET_INSTRUCTIONS[0]
    assert prompt_for("wrong", 0) == TARGET_INSTRUCTIONS[1]
    assert prompt_for("wrong", 9) == TARGET_INSTRUCTIONS[0]
    assert prompt_for("nonsense", 0) == NONSENSE_PROMPT


def test_wrong_prompt_never_matches_true():
    for task_id in TARGET_INSTRUCTIONS:
        assert prompt_for("wrong", task_id) != TARGET_INSTRUCTIONS[task_id]


def test_unknown_condition_is_rejected():
    with pytest.raises(ValueError):
        prompt_for("typo", 0)


def test_environment_mapping_and_runtime_assertion():
    class FakeEnv:
        num_envs = 2

        @staticmethod
        def call(name):
            assert name == "task_description"
            return (TARGET_INSTRUCTIONS[1], TARGET_INSTRUCTIONS[1])

    assert assert_environment_instruction(FakeEnv(), 1, 1) == TARGET_INSTRUCTIONS[1]

    with pytest.raises(RuntimeError, match="mapping mismatch"):
        assert_environment_instruction(FakeEnv(), 2, 2)
