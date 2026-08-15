import pytest

from smolvla_prompt_only.constants import NONSENSE_PROMPT, TARGET_INSTRUCTIONS
from smolvla_prompt_only.evaluate import prompt_for


def test_prompt_conditions_are_explicit():
    assert prompt_for("true", 0) == TARGET_INSTRUCTIONS[0]
    assert prompt_for("wrong", 0) == TARGET_INSTRUCTIONS[1]
    assert prompt_for("nonsense", 0) == NONSENSE_PROMPT


def test_unknown_condition_is_rejected():
    with pytest.raises(ValueError):
        prompt_for("typo", 0)
