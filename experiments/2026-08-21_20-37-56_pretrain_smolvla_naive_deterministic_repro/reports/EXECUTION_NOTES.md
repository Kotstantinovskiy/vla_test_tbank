# Execution notes

The first production determinism gate exposed an evaluation-harness error
before fan-out. Although Gym and policy-noise seeds were tied to the logical
episode, LeRobot's LIBERO wrapper selected the initial-state bank entry from a
mutable `init_state_id` counter. Reversing episode order therefore changed the
initial state assigned to each logical episode. The invalid comparison produced
18/20 versus 19/20 successes with mismatches at episodes 2, 3, and 6.

The failed JSON, videos, and log are preserved under
`results/recovery/init_state_order_failure/`. No training was repeated. The
harness was repaired to set and record `init_state_id = episode_index` before
every single-episode rollout. Re-evaluating the unchanged checkpoint produced
18/20 in both orders with zero per-episode mismatches; the passing record is
`artifacts/production_smoke.json`.

The first fan-out was mistakenly prepared for all ten `libero_goal` tasks.
After the user corrected the scope to the three assignment tasks, the
orchestrator was stopped. Desired task-1/task-2 k=5 training was left running
to completion; out-of-scope task-3 training was interrupted before any checkpoint was
saved. The old 30-point status, task-3 log, and launcher records are preserved
under `results/recovery/all_10_task_launch_20260821/`. The corrected plan has
9 independent adaptations and 180 main rollout videos.
