# Cost curve from PLAIN smolvla_base (no libero_90 pretrain)

Ablation of the in-domain pretrain: every task/budget adaptation starts
from schema-adapted lerobot/smolvla_base (community SO-100 pretraining,
never saw LIBERO/Franka) with the byte-identical recipe of the
pretrained cost curve (expert-only, 2000 steps, batch 32, official
demo_0..demo_{k-1}, same eval). The gap to the frozen pretrained-curve
reference is the value of the libero_90 pretrain in demonstrations.

No k=0 point by design: LIBERO state/action projections are initialized
only at fine-tune time (an untrained eval would act through random
projections).

| task | instruction | k=1 | k=2 | k=3 | k=5 | k=10 | k=25 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | `open the middle drawer of the cabinet` | 0/20 (0.00) | 9/20 (0.45) | 15/20 (0.75) | 17/20 (0.85) | 18/20 (0.90) | 17/20 (0.85) |
| 1 | `put the bowl on the stove` | 17/20 (0.85) | 17/20 (0.85) | 19/20 (0.95) | 19/20 (0.95) | 19/20 (0.95) | 19/20 (0.95) |
| 2 | `put the wine bottle on top of the cabinet` | 16/20 (0.80) | 17/20 (0.85) | 17/20 (0.85) | 17/20 (0.85) | 19/20 (0.95) | 17/20 (0.85) |
| 3 | `open the top drawer and put the bowl inside` | 10/20 (0.50) | 4/20 (0.20) | 4/20 (0.20) | 7/20 (0.35) | 16/20 (0.80) | 10/20 (0.50) |
| 4 | `put the bowl on top of the cabinet` | 12/20 (0.60) | 18/20 (0.90) | 16/20 (0.80) | 18/20 (0.90) | 18/20 (0.90) | 16/20 (0.80) |
| 5 | `push the plate to the front of the stove` | 14/20 (0.70) | 15/20 (0.75) | 16/20 (0.80) | 11/20 (0.55) | 13/20 (0.65) | 17/20 (0.85) |
| 6 | `put the cream cheese in the bowl` | 1/20 (0.05) | 5/20 (0.25) | 11/20 (0.55) | 12/20 (0.60) | 12/20 (0.60) | 11/20 (0.55) |
| 7 | `turn on the stove` | 16/20 (0.80) | 20/20 (1.00) | 19/20 (0.95) | 20/20 (1.00) | 20/20 (1.00) | 20/20 (1.00) |
| 8 | `put the bowl on the plate` | 6/20 (0.30) | 17/20 (0.85) | 16/20 (0.80) | 16/20 (0.80) | 19/20 (0.95) | 17/20 (0.85) |
| 9 | `put the wine bottle on the rack` | 5/20 (0.25) | 10/20 (0.50) | 5/20 (0.25) | 12/20 (0.60) | 19/20 (0.95) | 14/20 (0.70) |

## Cost curve

Frozen pretrained-curve reference (same recipe from the libero_90 pretrain): mean-10 k=1: 0.550, k=2: 0.705, k=3: 0.780, k=5: 0.830, k=10: 0.875, k=25: 0.850.

| mean | k=1 | k=2 | k=3 | k=5 | k=10 | k=25 |
|---|---:|---:|---:|---:|---:|---:|
| all 10 tasks | 0.485 | 0.660 | 0.690 | 0.745 | 0.865 | 0.790 |
| tasks 0-2 (assignment) | 0.550 | 0.717 | 0.850 | 0.883 | 0.933 | 0.883 |
