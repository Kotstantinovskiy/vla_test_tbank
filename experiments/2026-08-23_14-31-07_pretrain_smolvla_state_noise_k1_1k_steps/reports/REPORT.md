# High-alpha proprioception noise at k=1, 1000 training steps

Each (task, alpha) adaptation is an independent full fine-tune of the
pinned pretrain on the task's official demo_0, with additive zero-mean
Gaussian noise `alpha * eps` applied to the normalized proprioceptive
state inside policy.forward during training only (STATE is normalized
MEAN_STD, so alpha equals sigma_i = alpha * Std(s_i) in raw units).
Training budget is HALVED against the first sweep: 1000 steps with the
proportionally compressed auto-scaled LR schedule; the alpha=0.00 arm
at 1000 steps is therefore the only valid in-experiment baseline.
Actions and images are untouched; evaluation (n_action_steps=50, 20
episodes, seed bank 1000..1019) applies no noise. alpha=0.00 is the
in-experiment control (strict no-op, full-FT k=1 recipe).

| task | instruction | α=0.00 | α=0.08 | α=0.10 | α=0.20 | 2k-step α=0 ref |
|---:|---|---:|---:|---:|---:|---:|
| 0 | `open the middle drawer of the cabinet` | 3/20 (0.15) | 6/20 (0.30) | 6/20 (0.30) | 6/20 (0.30) | 3/20 |
| 1 | `put the bowl on the stove` | 19/20 (0.95) | 18/20 (0.90) | 18/20 (0.90) | 18/20 (0.90) | 19/20 |
| 2 | `put the wine bottle on top of the cabinet` | 14/20 (0.70) | 16/20 (0.80) | 18/20 (0.90) | 16/20 (0.80) | 15/20 |

## Mean over tasks 0-2

| | α=0.00 | α=0.08 | α=0.10 | α=0.20 |
|---|---:|---:|---:|---:|
| mean success | 0.600 | 0.667 | 0.700 | 0.667 |

2000-step alpha=0 reference mean (external): 0.617.
