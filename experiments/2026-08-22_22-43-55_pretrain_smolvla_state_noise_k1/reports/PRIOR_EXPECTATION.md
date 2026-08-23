# Prior expectation — recorded before preparation and any rollout

Recorded 2026-08-22, before preparation, training, or any rollout of this
experiment.

Reference (full-FT, k=1, n=50, same seeds/demos/evaluation):
task 0 (drawer) 2/20, task 1 (bowl→stove) 19/20, task 2 (wine→cabinet)
14/20, mean 0.583.

Predictions:

1. The alpha=0.00 control lands within ±2 successes per task of the full-FT
   k=1 reference (same recipe re-run; deviations measure GPU training
   nondeterminism only).
2. Small noise helps or is neutral at k=1: the best alpha in {0.01, 0.03}
   beats alpha=0.00 on the 3-task mean by at least +0.05. Mechanism: with a
   single demo the policy overfits exact proprioception; noise widens the
   state neighborhood mapped to the demonstrated behavior.
3. The effect is largest on task 0 (drawer, 2/20 baseline, most headroom and
   the most contact-precision-sensitive): its best-alpha success is at least
   double the alpha=0.00 arm. Tasks 1–2 move by at most ±3 successes.
4. alpha=0.05 is past the sweet spot: at or below alpha=0.03's mean (too
   much state corruption at batch-effective scale; the model may learn to
   ignore proprioception and lean on vision, which is not necessarily bad
   but should not beat the moderate setting).
5. Training loss at step 2000 increases monotonically with alpha (noisy
   inputs are harder to fit); this is a sanity expectation, not a success
   metric.
6. The determinism gate (task 0 / alpha 0.00, forward vs reverse, n=50)
   passes exactly.

Falsification handling: if alpha=0.00 deviates from the full-FT reference by
more than 4 successes on any task, suspect the wrapper is not a no-op (RNG
stream contamination) and audit the implementation before interpreting the
sweep. Single-training-seed caveat applies; per-point differences of ±2
successes are within noise for 20 episodes.
