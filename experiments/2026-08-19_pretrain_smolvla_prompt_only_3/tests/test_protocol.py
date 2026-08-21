from __future__ import annotations

import pytest

from pretrain_smolvla_prompt_only_3.aggregate import wilson_interval
from pretrain_smolvla_prompt_only_3.constants import (
    EVAL_BATCH_SIZE,
    MASTER_SEED,
    NONSENSE_PROMPT,
    N_EVAL_EPISODES,
    PROMPT_CONDITIONS,
    TARGET_INSTRUCTIONS,
    noise_seed,
    prompt_for,
)
from pretrain_smolvla_prompt_only_3.evaluate import all_labels, parse_label


def test_batch_size_is_one_by_protocol():
    # Per-episode noise reseeding is only well-defined for batch=1: batched
    # sampling would interleave a single stream across sub-envs.
    assert EVAL_BATCH_SIZE == 1


def test_noise_seeds_unique_and_deterministic():
    seeds = [noise_seed(episode) for episode in range(N_EVAL_EPISODES)]
    assert len(set(seeds)) == N_EVAL_EPISODES
    assert seeds[0] == MASTER_SEED
    assert seeds == [noise_seed(episode) for episode in range(N_EVAL_EPISODES)]


def test_labels_roundtrip():
    labels = all_labels()
    assert len(labels) == len(PROMPT_CONDITIONS) * len(TARGET_INSTRUCTIONS)
    assert len(labels) == len(set(labels))
    for label in labels:
        condition, task_id = parse_label(label)
        assert condition in PROMPT_CONDITIONS
        assert task_id in TARGET_INSTRUCTIONS
    with pytest.raises(ValueError):
        parse_label("true__task_99")
    with pytest.raises(ValueError):
        parse_label("bogus__task_1")


def test_prompt_for_conditions():
    assert prompt_for("true", 7) == "turn on the stove"
    assert prompt_for("wrong", 9) == TARGET_INSTRUCTIONS[0]
    assert prompt_for("nonsense", 3) == NONSENSE_PROMPT
    with pytest.raises(ValueError):
        prompt_for("absent", 0)


def test_wilson():
    low, high = wilson_interval(1, 20)
    assert low == pytest.approx(0.0089, abs=1e-3)
    assert high == pytest.approx(0.2359, abs=1e-3)
