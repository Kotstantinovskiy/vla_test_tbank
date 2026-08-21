from __future__ import annotations

"""LiberoEnv subclass: prompted-task predicate success + behavioral metrics.

Same contract as the v2 seen-scene probe (prompted goal_state redefines
`info["is_success"]` and terminates the episode; env's own predicate kept as
`env_task_success`), extended with per-episode behavioral summaries needed at
the goal-scene floor, where binary success may not discriminate conditions:

- ``min_eef_target_dist``: closest approach of the end effector to the env's
  task target object over the episode (the target is fixed PER ENV, so the
  metric is comparable across prompt conditions of one env);
- ``final_target_displacement`` / ``max_target_displacement``: how far the
  target object moved from its reset position.
"""

from typing import Any

import numpy as np
from lerobot.envs.libero import LiberoEnv


class PredicateLiberoEnv(LiberoEnv):
    def __init__(
        self,
        *args: Any,
        prompted_goal_states: list[list[str]] | None = None,
        expect_env_equivalent: bool = False,
        behavior_target: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._prompted_goal_states = (
            [list(state) for state in prompted_goal_states]
            if prompted_goal_states
            else None
        )
        self._expect_env_equivalent = bool(expect_env_equivalent)
        self._behavior_target = behavior_target
        self._records: list[dict[str, Any]] = []
        self._current: dict[str, Any] | None = None
        self._target_reset_pos: np.ndarray | None = None

    # -- helpers ---------------------------------------------------------

    def _problem(self):
        assert self._env is not None
        return self._env.env

    def _prompted_satisfied(self) -> bool:
        problem = self._problem()
        return all(
            bool(problem._eval_predicate(list(state)))
            for state in self._prompted_goal_states
        )

    def _target_pos(self) -> np.ndarray:
        state = self._problem().object_states_dict[self._behavior_target]
        return np.asarray(state.get_geom_state()["pos"], dtype=float).copy()

    def _eef_pos(self) -> np.ndarray | None:
        controller = self._env.robots[0].controller
        if controller.ee_pos is None:
            return None
        return np.asarray(controller.ee_pos, dtype=float)

    # -- gym API ---------------------------------------------------------

    def reset(self, seed=None, **kwargs):
        observation, info = super().reset(seed=seed, **kwargs)
        object_states = self._problem().object_states_dict
        if self._prompted_goal_states is not None:
            missing = sorted(
                {
                    arg
                    for state in self._prompted_goal_states
                    for arg in state[1:]
                    if arg not in object_states
                }
            )
            if missing:
                raise RuntimeError(
                    f"Prompted predicate args missing from scene "
                    f"object_states_dict: {missing}"
                )
        if self._behavior_target is not None:
            if self._behavior_target not in object_states:
                raise RuntimeError(
                    f"Behavior target {self._behavior_target!r} missing from scene"
                )
            self._target_reset_pos = self._target_pos()
        self._current = {
            "seed": int(seed) if seed is not None else None,
            "prompted_success": False,
            "prompted_first_step": None,
            "env_task_success": False,
            "env_first_step": None,
            "steps": 0,
            "consistency_violations": 0,
            "min_eef_target_dist": None,
            "final_target_displacement": None,
            "max_target_displacement": None,
        }
        self._records.append(self._current)
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        record = self._current
        if record is None:
            raise RuntimeError("step() called before reset()")
        record["steps"] += 1
        env_success = bool(info["is_success"])
        if env_success and not record["env_task_success"]:
            record["env_task_success"] = True
            record["env_first_step"] = record["steps"]
        info["env_task_success"] = env_success
        if self._prompted_goal_states is not None:
            prompted_now = self._prompted_satisfied()
            if prompted_now and not record["prompted_success"]:
                record["prompted_success"] = True
                record["prompted_first_step"] = record["steps"]
            if self._expect_env_equivalent and prompted_now != env_success:
                record["consistency_violations"] += 1
            info["is_success"] = bool(record["prompted_success"])
            terminated = bool(terminated or record["prompted_success"])
        if self._behavior_target is not None:
            target_pos = self._target_pos()
            displacement = float(np.linalg.norm(target_pos - self._target_reset_pos))
            record["final_target_displacement"] = displacement
            if (
                record["max_target_displacement"] is None
                or displacement > record["max_target_displacement"]
            ):
                record["max_target_displacement"] = displacement
            eef = self._eef_pos()
            if eef is not None:
                dist = float(np.linalg.norm(eef - target_pos))
                if (
                    record["min_eef_target_dist"] is None
                    or dist < record["min_eef_target_dist"]
                ):
                    record["min_eef_target_dist"] = dist
        return observation, reward, terminated, truncated, info

    def get_predicate_records(self) -> list[dict[str, Any]]:
        return self._records
