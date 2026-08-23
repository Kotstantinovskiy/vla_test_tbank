# Execution notes

The production task-0/k=1 determinism gate completed before fan-out and passed
with 3/20 successes in both episode orders and zero per-episode mismatches.

The initial deferred launcher targeted an all-ten-task naive predecessor. It
was stopped before low-k fan-out began when the scope was corrected to the
assignment's task IDs 0–2. Its PID and log are preserved under
`results/recovery/all_10_task_launch_20260821/`. The corrected plan has 9
independent adaptations and 180 main rollout videos.
