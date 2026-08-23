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
| 0 | `open the middle drawer of the cabinet` | 18/20 (0.90) | 18/20 (0.90) | 19/20 (0.95) |
| 1 | `put the bowl on the stove` | 19/20 (0.95) | 20/20 (1.00) | 20/20 (1.00) |
| 2 | `put the wine bottle on top of the cabinet` | 16/20 (0.80) | 19/20 (0.95) | 16/20 (0.80) |

## Cost curve

k=0 (prompt-only reference): 0.000 over tasks 0-2.

| mean | k=5 | k=10 | k=25 |
|---|---:|---:|---:|
| tasks 0-2 (assignment) | 0.883 | 0.950 | 0.917 |
