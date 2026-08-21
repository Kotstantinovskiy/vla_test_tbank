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
| 0 | `open the middle drawer of the cabinet` | 18/20 (0.90) | 19/20 (0.95) | 20/20 (1.00) |
| 1 | `put the bowl on the stove` | 20/20 (1.00) | 20/20 (1.00) | 18/20 (0.90) |
| 2 | `put the wine bottle on top of the cabinet` | 19/20 (0.95) | 18/20 (0.90) | 16/20 (0.80) |
| 3 | `open the top drawer and put the bowl inside` | 11/20 (0.55) | 12/20 (0.60) | 15/20 (0.75) |
| 4 | `put the bowl on top of the cabinet` | 18/20 (0.90) | 18/20 (0.90) | 14/20 (0.70) |
| 5 | `push the plate to the front of the stove` | 12/20 (0.60) | 17/20 (0.85) | 18/20 (0.90) |
| 6 | `put the cream cheese in the bowl` | 10/20 (0.50) | 11/20 (0.55) | 16/20 (0.80) |
| 7 | `turn on the stove` | 20/20 (1.00) | 20/20 (1.00) | 20/20 (1.00) |
| 8 | `put the bowl on the plate` | 19/20 (0.95) | 20/20 (1.00) | 18/20 (0.90) |
| 9 | `put the wine bottle on the rack` | 19/20 (0.95) | 20/20 (1.00) | 15/20 (0.75) |

## Cost curve

k=0 (prompt-only reference): 0.005 over 10 tasks.

| mean | k=5 | k=10 | k=25 |
|---|---:|---:|---:|
| all 10 tasks | 0.830 | 0.875 | 0.850 |
| tasks 0-2 (assignment) | 0.950 | 0.950 | 0.900 |
