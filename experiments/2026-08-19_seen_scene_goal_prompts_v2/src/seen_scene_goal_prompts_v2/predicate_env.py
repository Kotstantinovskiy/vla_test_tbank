from __future__ import annotations

"""LiberoEnv subclass that scores the PROMPTED task's goal predicate.

The env's own predicate keeps driving `env_task_success`; the prompted
predicate (a parsed BDDL goal_state, e.g. [["turnon", "flat_stove_1"]])
redefines `info["is_success"]` and additionally terminates the episode, so
`eval_policy` aggregates prompted-task success without modification.

Runs inside AsyncVectorEnv worker subprocesses; per-episode records are
fetched afterwards with `vector_env.call("get_predicate_records")` and joined
to eval_policy's per_episode list by episode seed.
"""

from typing import Any

from lerobot.envs.libero import LiberoEnv


class PredicateLiberoEnv(LiberoEnv):
    def __init__(
        self,
        *args: Any,
        prompted_goal_states: list[list[str]] | None = None,
        expect_env_equivalent: bool = False,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._prompted_goal_states = (
            [list(state) for state in prompted_goal_states]
            if prompted_goal_states
            else None
        )
        self._expect_env_equivalent = bool(expect_env_equivalent)
        self._records: list[dict[str, Any]] = []
        self._current: dict[str, Any] | None = None

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

    # -- gym API ---------------------------------------------------------

    def reset(self, seed=None, **kwargs):
        observation, info = super().reset(seed=seed, **kwargs)
        if self._prompted_goal_states is not None:
            object_states = self._problem().object_states_dict
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
        self._current = {
            "seed": int(seed) if seed is not None else None,
            "prompted_success": False,
            "prompted_first_step": None,
            "env_task_success": False,
            "env_first_step": None,
            "steps": 0,
            "consistency_violations": 0,
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
        return observation, reward, terminated, truncated, info

    def get_predicate_records(self) -> list[dict[str, Any]]:
        return self._records
