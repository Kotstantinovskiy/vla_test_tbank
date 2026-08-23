# Prior expectation — recorded before preparation and any rollout

Recorded 2026-08-23, before preparation, training, or any rollout of this
experiment.

References (same seeds/demos/evaluation, mean over tasks 0–2, n=50 unless
noted):

| method | k=1 | k=2 | k=3 | k=5 | k=10 | k=25 |
|---|---:|---:|---:|---:|---:|---:|
| expert-only (repro / naive repro) | 0.600 | 0.683 | 0.667 | 0.883 | 0.950 | 0.917 |
| full-FT (2000 steps) | 0.583 | 0.950 | 0.767 | — | — | — |
| image-aug full-FT (1000/1500/2000) | 0.583 | 0.900 | 0.783 | — | — | — |
| state-noise α=0.10 @1000 steps (k=1) | 0.700 | — | — | — | — | — |

Best-known per budget: k=1 0.700, k=2 0.950, k=3 0.783 (n=50) / 0.817
(image-aug n=25), k=5/10/25 expert-only naive 0.883/0.950/0.917.

Predictions for the bundle (full FT + image aug + state noise α=0.10 +
budget-dependent steps):

1. k=1 (n=50): mean ≥ 0.65 — the state-noise gain survives the addition of
   image augmentation (both fight overfitting through different channels);
   an interaction wiping it out (mean < 0.60) would falsify additivity.
2. k=2 (n=50): mean in [0.80, 0.95]. Risk factors: noise untested at k=2
   and 1500 steps; the full-FT 0.950 peak was drawer-driven and fragile.
3. k=3 (n=50): mean in [0.70, 0.85], i.e. at or above full-FT 0.767.
4. k=5/10/25 (n=50): the full-FT-based bundle is at or above the
   expert-only naive baseline at k=5 (≥ 0.88) and within ±0.05 of it at
   k=10/25 (the naive baseline is already near ceiling; full FT's advantage
   shrinks as data grows). A drop below 0.85 at k=25 would repeat the
   documented k=25 dip of the archived curves.
5. n=35 lands between n=50 and n=25 at k≤2 and is within ±0.03 of the
   better of the two at k≥3; n=25 remains worst at k=1 and roughly ties
   n=50 by k≥3 (as in image-aug).
6. Monotonicity: the bundle mean (n=50) is non-decreasing in k except for
   at most one inversion of ≤ 0.05 (the k=2→k=3 drawer dip may persist).
7. The determinism gate passes exactly at all three variants.
8. Wall-clock: with 2+2 slots per GPU the whole pipeline (18 trainings, 57
   evaluation runs) completes in under 6 hours.

Falsification handling: if the bundle is below the best single-ingredient
reference by more than 0.10 at any budget, inspect training logs (loss
scale under the combined augmentations), the parameter audit, and rollout
videos before interpreting; per-point differences of ±2 successes are
within noise for 20 episodes. Single-training-seed caveat applies — the
assignment's two-seed requirement remains outstanding and is the natural
next run (bundle + baseline at seed 2).
