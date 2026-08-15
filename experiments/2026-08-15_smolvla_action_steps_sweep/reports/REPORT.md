# Action-step sweep report

This report is generated from the completed rollout JSON files. The original
Task 1 baseline is frozen; every new point uses the same checkpoint weights and
changes only `n_action_steps` at inference.

## Mean success

| demos | n=1 | n=5 | n=10 | n=25 | n=50 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 5 | 0.250 | 0.233 | 0.333 | 0.217 | 0.200 |
| 10 | 0.117 | 0.200 | 0.267 | 0.283 | 0.250 |
| 25 | 0.250 | 0.250 | 0.267 | 0.250 | 0.300 |

## Best horizons by task and budget

| task | demos | best n_action_steps | success | delta vs paired n=50 |
|---:|---:|---|---:|---:|
| 0 | 0 | 1, 5, 10, 25, 50 | 0.000 | +0.000 |
| 0 | 5 | 10 | 1.000 | +0.400 |
| 0 | 10 | 25 | 0.850 | +0.100 |
| 0 | 25 | 50 | 0.900 | +0.000 |
| 1 | 0 | 1, 5, 10, 25, 50 | 0.000 | +0.000 |
| 1 | 5 | 1, 5, 10, 25, 50 | 0.000 | +0.000 |
| 1 | 10 | 1, 5, 10, 25, 50 | 0.000 | +0.000 |
| 1 | 25 | 1, 5, 10, 25, 50 | 0.000 | +0.000 |
| 2 | 0 | 1, 5, 10, 25, 50 | 0.000 | +0.000 |
| 2 | 5 | 1, 5, 10, 25, 50 | 0.000 | +0.000 |
| 2 | 10 | 1, 5, 10, 25, 50 | 0.000 | +0.000 |
| 2 | 25 | 1, 5, 10, 25, 50 | 0.000 | +0.000 |

## Interpretation

The prediction of the largest gain on task 1 was not supported. Tasks 1 and 2 remained at zero for every budget and horizon; only the already learned drawer task 0 responded to the inference change. This suggests that tasks 1 and 2 fail before open-loop compounding error becomes the main bottleneck.

For task 0, k=5 peaks at n=10 (1.00, +0.40 versus paired n=50), k=10 peaks at n=25 (0.85, +0.10), while k=25 is best at n=50 (0.90). The effect is therefore not uniform across k, and n=1 is never uniquely best.

Across the three adapted budgets, the best single fixed horizon is n=10 with mean success 0.289, versus 0.250 for paired n=50. This aggregate gain is driven entirely by task 0 and should not be presented as recovery of cross-task generalization.

The paired n=50 rerun differs from the frozen historical result at task 0, k=5 (0.60 versus 0.85). The new sweep resets the policy RNG immediately before every horizon to pair flow-matching samples; the historical evaluator did not use that exact RNG protocol. Consequently, delta versus paired n=50 is the primary inference estimate, while delta versus frozen baseline is reported separately for protocol transparency.

## Language-control gate

All true-prompt zero-shot points stayed at the success floor; the locked protocol therefore skips additional language controls.
