# Image-augmented full-fine-tune cost curve on the official-data pretrain

Identical to the full-FT experiment except that lerobot's default
dataset image transforms are enabled during training (photometric
jitter: brightness/contrast/saturation/hue/sharpness, plus RandomAffine
±5°/5% translate; up to 3 sampled per frame). Evaluation applies no
augmentation. Demos are the official demo_0..demo_{k-1} of each task
from the in-repo libero_goal conversion.
Each adapted checkpoint is evaluated at inference n_action_steps=50
(trained default) and 25 on identical per-episode seeds/init states.

Normalization disclosure: k>0 points run under target-dataset statistics
(LeRobot swaps normalizer stats at fine-tune time), while the k=0
reference ran under pretraining statistics; both come from the same
conversion pipeline.

## Inference n_action_steps = 50

| task | instruction | k=1 | k=2 | k=3 |
|---:|---|---:|---:|---:|
| 0 | `open the middle drawer of the cabinet` | 3/20 (0.15) | 17/20 (0.85) | 12/20 (0.60) |
| 1 | `put the bowl on the stove` | 19/20 (0.95) | 20/20 (1.00) | 20/20 (1.00) |
| 2 | `put the wine bottle on top of the cabinet` | 13/20 (0.65) | 17/20 (0.85) | 15/20 (0.75) |

## Inference n_action_steps = 25

| task | instruction | k=1 | k=2 | k=3 |
|---:|---|---:|---:|---:|
| 0 | `open the middle drawer of the cabinet` | 0/20 (0.00) | 14/20 (0.70) | 14/20 (0.70) |
| 1 | `put the bowl on the stove` | 20/20 (1.00) | 18/20 (0.90) | 20/20 (1.00) |
| 2 | `put the wine bottle on top of the cabinet` | 14/20 (0.70) | 15/20 (0.75) | 15/20 (0.75) |

## Cost curve

k=0 (prompt-only reference): 0.000 over tasks 0-2.

| mean tasks 0-2 | k=1 | k=2 | k=3 |
|---|---:|---:|---:|
| n=50 | 0.583 | 0.900 | 0.783 |
| n=25 | 0.567 | 0.783 | 0.817 |
## No-augmentation reference (full-FT experiment, same seeds)

| mean tasks 0-2 | k=1 | k=2 | k=3 |
|---|---:|---:|---:|
| n=50 (no aug) | 0.583 | 0.950 | 0.767 |
| n=25 (no aug) | 0.533 | 0.750 | 0.800 |
