# Few-shot cost curve on the official-data pretrain

Every task/budget adaptation starts from the frozen pretrain checkpoint
independently (expert-only: vision encoder and VLM frozen; action expert,
state/action projections trainable). Demos are the official
demo_0..demo_{k-1} of each task from the in-repo libero_goal conversion.

Normalization disclosure: k>0 points run under target-dataset statistics
(LeRobot swaps normalizer stats at fine-tune time), while the k=0
reference ran under pretraining statistics; both come from the same
conversion pipeline.

| task | instruction | k=5 | k=10 | k=25 |
|---:|---|---:|---:|---:|
| 0 | `open the middle drawer of the cabinet` | 3/20 (0.15) | 7/20 (0.35) | 9/20 (0.45) |
| 1 | `put the bowl on the stove` | 16/20 (0.80) | 18/20 (0.90) | 19/20 (0.95) |
| 2 | `put the wine bottle on top of the cabinet` | 15/20 (0.75) | 18/20 (0.90) | 15/20 (0.75) |
| 3 | `open the top drawer and put the bowl inside` | 3/20 (0.15) | 9/20 (0.45) | 8/20 (0.40) |
| 4 | `put the bowl on top of the cabinet` | 15/20 (0.75) | 16/20 (0.80) | 16/20 (0.80) |
| 5 | `push the plate to the front of the stove` | 14/20 (0.70) | 14/20 (0.70) | 14/20 (0.70) |
| 6 | `put the cream cheese in the bowl` | 4/20 (0.20) | 7/20 (0.35) | 17/20 (0.85) |
| 7 | `turn on the stove` | 17/20 (0.85) | 20/20 (1.00) | 20/20 (1.00) |
| 8 | `put the bowl on the plate` | 9/20 (0.45) | 15/20 (0.75) | 19/20 (0.95) |
| 9 | `put the wine bottle on the rack` | 14/20 (0.70) | 17/20 (0.85) | 19/20 (0.95) |

## Cost curve

k=0 (prompt-only reference): 0.005 over 10 tasks.

| mean | k=5 | k=10 | k=25 |
|---|---:|---:|---:|
| all 10 tasks | 0.550 | 0.705 | 0.780 |
| tasks 0-2 (assignment) | 0.567 | 0.717 | 0.717 |
