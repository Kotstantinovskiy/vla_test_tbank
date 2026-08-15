import pytest

from smolvla_prompt_only.constants import (
    NONSENSE_PROMPT,
    TARGET_ENV_TASK_IDS,
    TARGET_INSTRUCTIONS,
)
from smolvla_prompt_only.evaluate import assert_environment_instruction, prompt_for


def test_prompt_conditions_are_explicit():
    assert prompt_for("true", 0) == TARGET_INSTRUCTIONS[0]
    assert prompt_for("wrong", 0) == TARGET_INSTRUCTIONS[1]
    assert prompt_for("nonsense", 0) == NONSENSE_PROMPT


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

    assert TARGET_ENV_TASK_IDS == {0: 0, 1: 9, 2: 3}
    assert assert_environment_instruction(FakeEnv(), 1, 9) == TARGET_INSTRUCTIONS[1]

    with pytest.raises(RuntimeError, match="mapping mismatch"):
        assert_environment_instruction(FakeEnv(), 2, 3)
