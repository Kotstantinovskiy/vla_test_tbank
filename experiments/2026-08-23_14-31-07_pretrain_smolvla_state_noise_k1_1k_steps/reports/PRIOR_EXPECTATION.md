# Prior expectation — recorded before preparation and any rollout

Recorded 2026-08-23, before preparation, training, or any rollout of this
experiment. Adaptive-continuation disclosure: the alpha grid {0.08, 0.10,
0.20} was chosen after observing the 2000-step sweep
(`2026-08-22_22-43-55_pretrain_smolvla_state_noise_k1`: means
0.617/0.583/0.667/0.683 at alpha 0.00/0.01/0.03/0.05, task 2 carrying the
gain 15→19/20, drawer flat at 2–4/20), and additionally the training budget
is halved to 1000 steps. No arm of THIS experiment has run.

Predictions:

1. Steps effect at alpha=0.00: 1000 steps lose little against 2000 — the
   1k control mean lands within [0.52, 0.65] (2k control: 0.617), with task 1
   ≥ 16/20. Basis: training loss plateaued near 0.03 well before step 1000
   in every k=1 run observed so far.
2. Noise still helps at 1000 steps: the best noisy arm beats the 1k
   alpha=0.00 control by at least +0.05 on the mean.
3. The dose-response bends: alpha=0.20 is NOT the best arm; the best arm is
   0.08 or 0.10. At 20% of per-dimension std the state token is unreliable
   enough that the policy must partly ignore proprioception, and at k=1 that
   costs more than it protects. Concretely mean(0.20) ≤ mean(best of
   0.08/0.10).
4. Task pattern repeats: task 2 (wine→cabinet) carries the gain (≥ 17/20 at
   the best alpha); task 0 (drawer) stays ≤ 6/20 at every arm — its k=1
   failure is not proprioception-bound.
5. The determinism gate (task 0 / alpha 0.00, forward vs reverse, n=50)
   passes exactly.
6. Training loss at step 1000 increases monotonically with alpha.

Falsification handling: if the 1k alpha=0.00 control collapses (mean below
0.45), the steps budget dominates everything and noise conclusions from this
experiment are secondary; report the steps effect as the headline instead.
Single-training-seed caveat applies; ±2 successes per point are within noise.
