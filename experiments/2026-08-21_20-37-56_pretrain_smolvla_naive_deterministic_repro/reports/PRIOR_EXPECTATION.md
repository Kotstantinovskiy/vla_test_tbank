# Prior expectation — recorded before preparation and rollout

Scope correction, recorded 2026-08-21 after the task-0 production gate but
before inspecting new task-1/task-2 rollouts: the full run is restricted to
the assignment's task IDs 0–2. The original ten-task expectation below is
preserved rather than rewritten. The archived task-0–2 reference was
0.90/0.95/0.90 at k=5/10/25; the deterministic reproduction is expected to
remain within roughly ±0.10 at each budget.

Recorded 2026-08-21 after choosing training seed 1000 and before inspecting any
new target rollout.

The archived experiment reported mean success across ten tasks of 0.83, 0.875,
and 0.85 at k=5, 10, and 25. With the same checkpoint, demonstrations, recipe,
and training seed, this reproduction should remain near those values; a
practical predeclared tolerance is ±0.10 at each budget. Strict monotonicity is
not expected, and the prior k=25 dip may persist.

The new rollout estimator is deliberately different: batch size one and
per-episode SmolVLA noise seeds replace the archived process-history-dependent
noise stream. Individual tasks can therefore move by several successes out of
20, and exact equality with the archived curve is not predicted.

The forward/reverse episode-order determinism check must match exactly. If it
fails, fan-out must stop. If a budget mean differs from the archived mean by
more than 0.15, inspect the checkpoint, selected demonstrations, training log,
and evaluation harness before assigning a scientific interpretation.
