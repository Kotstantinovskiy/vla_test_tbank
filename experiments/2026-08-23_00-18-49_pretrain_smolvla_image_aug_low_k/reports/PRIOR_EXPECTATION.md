# Prior expectation — recorded before preparation and any rollout

Recorded 2026-08-23, before preparation, training, or any rollout of this
experiment.

Reference (full-FT, no augmentation, same seeds/demos/evaluation, mean over
tasks 0–2):

| | k=1 | k=2 | k=3 |
|---|---:|---:|---:|
| n=50 | 0.583 | 0.950 | 0.767 |
| n=25 | 0.533 | 0.750 | 0.800 |

Per-task n=50 successes/20: task 0 — 2/20/11, task 1 — 19/20/20,
task 2 — 14/17/15 (k=1/2/3).

Predictions for enabling lerobot's default image transforms during training:

1. Small overall effect: every budget mean (n=50) within ±0.12 of the
   full-FT reference. LIBERO evaluates in the same renderer and scenes as
   the demos, so the photometric part fights a distribution shift that does
   not exist at eval time; the informative part is the RandomAffine, which
   adds spatial diversity relevant to the 20 varied init states.
2. Direction by budget: at k=1 augmentation helps or is neutral (visual
   overfitting to a single episode is strongest there): mean(k=1, n=50) ≥
   the reference 0.583 − 0.05. At k=3 it is neutral-to-slightly-negative
   (augmentation makes fitting harder within the fixed 2000 steps).
3. The unstable drawer points move the most again: task 0 / k=2 (n=50) drops
   from the reference 20/20 (that peak is fragile), while task 0 / k=1 and
   k=3 change by at most ±4 successes. Tasks 1–2 stay within ±2 successes
   per point.
4. Training loss at step 2000 is higher than the full-FT counterpart for
   every point (augmented inputs are harder to fit); sanity expectation.
5. The n=25 vs n=50 pattern stays task-0-driven as in the sibling
   experiments.
6. The determinism gate (task 0 / k=1, forward vs reverse, both variants)
   passes exactly — evaluation is untouched by training-time augmentation.

## Amendment (2026-08-23, still before any run): budget-dependent steps

Before preparation or any rollout, the protocol was changed to train
1000/1500/2000 steps at k=1/2/3 (the reference trained 2000 everywhere), so
for k=1 and k=2 this experiment now differs from the full-FT reference in
both augmentation and optimization length; only k=3 isolates augmentation.
Amended predictions:

- A7. The k=3 points (the clean augmentation comparison) stay within ±0.12
  of the reference means (n=50: 0.767, n=25: 0.800), per prediction 1.
- A8. Halving steps at k=1 does not collapse performance: mean(k=1, n=50)
  stays within ±0.15 of the reference 0.583. Rationale: k=1 loss plateaued
  well before step 1000 in the full-FT gate run.
- A9. Prediction 3's expectation that the fragile task 0 / k=2 20/20 peak is
  not reproduced is now even stronger (fewer steps + augmentation).

Falsification handling: if a budget mean differs from the reference by more
than 0.15, inspect training logs and a sample of rollout videos before
interpreting; augmentation bugs (e.g. transforms leaking into evaluation or
applied to non-image keys) must be ruled out first. Single-training-seed
caveat applies.
