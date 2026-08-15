import pytest

from smolvla_action_steps.evaluate import prompt_for


def test_prompt_conditions_are_deterministic() -> None:
    assert prompt_for("true", 0) == "open the middle drawer of the cabinet"
    assert prompt_for("wrong", 0) == "put the wine bottle on the rack"
    assert "dax" in prompt_for("nonsense", 2)
    with pytest.raises(ValueError):
        prompt_for("unknown", 0)
