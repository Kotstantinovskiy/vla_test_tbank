# Full-fine-tune few-shot cost curve on the official-data pretrain

Every task/budget adaptation starts from the frozen pretrain checkpoint
independently and fine-tunes the whole policy (VLM text tower, vision
encoder, connector, action expert, projections; only LeRobot's
unused-by-design guard tensors stay frozen). Demos are the official
demo_0..demo_{k-1} of each task from the in-repo libero_goal conversion.
Each adapted checkpoint is evaluated at inference n_action_steps=50
(trained default) and 25 on identical per-episode seeds/init states.

Normalization disclosure: k>0 points run under target-dataset statistics
(LeRobot swaps normalizer stats at fine-tune time), while the k=0
reference ran under pretraining statistics; both come from the same
conversion pipeline.

## Inference n_action_steps = 50

| task | instruction | k=1 | k=2 | k=3 |
|---:|---|---:|---:|---:|
| 0 | `open the middle drawer of the cabinet` | 2/20 (0.10) | 20/20 (1.00) | 11/20 (0.55) |
| 1 | `put the bowl on the stove` | 19/20 (0.95) | 20/20 (1.00) | 20/20 (1.00) |
| 2 | `put the wine bottle on top of the cabinet` | 14/20 (0.70) | 17/20 (0.85) | 15/20 (0.75) |

## Inference n_action_steps = 25

| task | instruction | k=1 | k=2 | k=3 |
|---:|---|---:|---:|---:|
| 0 | `open the middle drawer of the cabinet` | 0/20 (0.00) | 8/20 (0.40) | 15/20 (0.75) |
| 1 | `put the bowl on the stove` | 19/20 (0.95) | 20/20 (1.00) | 19/20 (0.95) |
| 2 | `put the wine bottle on top of the cabinet` | 13/20 (0.65) | 17/20 (0.85) | 14/20 (0.70) |

## Cost curve

k=0 (prompt-only reference): 0.000 over tasks 0-2.

| mean tasks 0-2 | k=1 | k=2 | k=3 |
|---|---:|---:|---:|
| n=50 | 0.583 | 0.950 | 0.767 |
| n=25 | 0.533 | 0.750 | 0.800 |
