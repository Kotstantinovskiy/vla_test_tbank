import pytest

from smolvla_action_steps.constants import TARGET_ENV_TASK_IDS, TARGET_INSTRUCTIONS
from smolvla_action_steps.evaluate import assert_environment_instruction, prompt_for


def test_prompt_conditions_are_deterministic() -> None:
    assert prompt_for("true", 0) == "open the middle drawer of the cabinet"
    assert prompt_for("wrong", 0) == "put the wine bottle on the rack"
    assert "dax" in prompt_for("nonsense", 2)
    with pytest.raises(ValueError):
        prompt_for("unknown", 0)


def test_environment_mapping_is_checked_before_rollout() -> None:
    class FakeEnv:
        num_envs = 1

        @staticmethod
        def call(name):
            assert name == "task_description"
            return (TARGET_INSTRUCTIONS[2],)

    assert TARGET_ENV_TASK_IDS == {0: 0, 1: 9, 2: 3}
    assert assert_environment_instruction(FakeEnv(), 2, 3) == TARGET_INSTRUCTIONS[2]
    with pytest.raises(RuntimeError, match="actual"):
        assert_environment_instruction(FakeEnv(), 1, 9)
