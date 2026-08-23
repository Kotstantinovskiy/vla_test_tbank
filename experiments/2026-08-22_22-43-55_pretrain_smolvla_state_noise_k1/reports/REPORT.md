# Proprioception-noise augmentation at k=1 (full fine-tune)

Each (task, alpha) adaptation is an independent full fine-tune of the
pinned pretrain on the task's official demo_0, with additive zero-mean
Gaussian noise `alpha * eps` applied to the normalized proprioceptive
state inside policy.forward during training only (STATE is normalized
MEAN_STD, so alpha equals sigma_i = alpha * Std(s_i) in raw units).
Actions and images are untouched; evaluation (n_action_steps=50, 20
episodes, seed bank 1000..1019) applies no noise. alpha=0.00 is the
in-experiment control (strict no-op, full-FT k=1 recipe).

| task | instruction | α=0.00 | α=0.01 | α=0.03 | α=0.05 | full-FT ref |
|---:|---|---:|---:|---:|---:|---:|
| 0 | `open the middle drawer of the cabinet` | 3/20 (0.15) | 2/20 (0.10) | 4/20 (0.20) | 4/20 (0.20) | 2/20 |
| 1 | `put the bowl on the stove` | 19/20 (0.95) | 17/20 (0.85) | 17/20 (0.85) | 18/20 (0.90) | 19/20 |
| 2 | `put the wine bottle on top of the cabinet` | 15/20 (0.75) | 16/20 (0.80) | 19/20 (0.95) | 19/20 (0.95) | 14/20 |

## Mean over tasks 0-2

| | α=0.00 | α=0.01 | α=0.03 | α=0.05 |
|---|---:|---:|---:|---:|
| mean success | 0.617 | 0.583 | 0.667 | 0.683 |

Full-FT k=1 reference mean (external): 0.583.
