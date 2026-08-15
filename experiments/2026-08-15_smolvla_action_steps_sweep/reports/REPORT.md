# Action-step sweep report

This report is generated from the completed rollout JSON files. Logical task IDs
0/1/2 map to `libero_goal` environment IDs 0/9/3. The corrected Task 1 baseline
is frozen; every new point uses the same checkpoint weights and changes only
`n_action_steps` at inference.

## Mean success

| demos | n=1 | n=5 | n=10 | n=25 | n=50 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 5 | 0.700 | 0.750 | 0.867 | 0.733 | 0.783 |
| 10 | 0.667 | 0.750 | 0.867 | 0.850 | 0.783 |
| 25 | 0.783 | 0.800 | 0.783 | 0.817 | 0.867 |

## Best horizons by task and budget

| task | demos | best n_action_steps | success | delta vs paired n=50 |
|---:|---:|---|---:|---:|
| 0 | 0 | 1, 5, 10, 25, 50 | 0.000 | +0.000 |
| 0 | 5 | 10 | 1.000 | +0.400 |
| 0 | 10 | 25 | 0.850 | +0.100 |
| 0 | 25 | 50 | 0.900 | +0.000 |
| 1 | 0 | 1, 5, 10, 25, 50 | 0.000 | +0.000 |
| 1 | 5 | 50 | 0.900 | +0.000 |
| 1 | 10 | 5, 50 | 0.850 | +0.000 |
| 1 | 25 | 1, 50 | 0.900 | +0.000 |
| 2 | 0 | 1, 5, 10, 25, 50 | 0.000 | +0.000 |
| 2 | 5 | 5, 25 | 0.900 | +0.050 |
| 2 | 10 | 10 | 1.000 | +0.250 |
| 2 | 25 | 5, 25 | 0.850 | +0.050 |

## Interpretation

The prediction of the largest gain on task 1 was not supported. The best shorter-horizon point for each task, with delta versus its paired n=50 anchor, is: task 0: k=5, n=10, 1.00 (+0.40); task 1: k=25, n=1, 0.90 (+0.00); task 2: k=10, n=10, 1.00 (+0.25). The effect is heterogeneous across both tasks and demonstration budgets.

Across the three adapted budgets, the best single fixed horizon is n=10 with mean success 0.839, versus 0.811 for paired n=50. Its task-average deltas are task 0 +0.117, task 1 -0.083, task 2 +0.050. A single global horizon therefore hides opposing task-specific effects.

The largest paired-anchor mismatch is at task 0, k=5: n=50 rerun 0.60 versus frozen baseline 0.85. The new sweep resets the policy RNG immediately before every horizon to pair flow-matching samples; the historical evaluator did not use that exact RNG protocol. Consequently, delta versus paired n=50 is the primary inference estimate, while delta versus frozen baseline is reported separately for protocol transparency.

## Language-control gate

All true-prompt zero-shot points stayed at the success floor; the locked protocol therefore skips additional language controls.
