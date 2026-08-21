# Prior-informed expectations and decision rules (base cost curve)

Recorded 2026-08-19 before running any training or rollout of this
experiment.

Priors: the pretrained curve (frozen reference, single seed): mean-10 =
0.55/0.705/0.78/0.83/0.875/0.85 at k=1/2/3/5/10/25; tasks 0-2 =
0.567/0.717/0.717/0.95/0.95/0.90. Plain smolvla_base was trained on real
SO-100 community data — LIBERO's simulated Franka is out of domain for the
frozen VLM/vision tower, and only the ~100M expert + projections adapt.

## Predictions

1. **The base curve is far below the pretrained curve at every k.**
   mean-10 guesses: k=1: 0.00-0.05; k=2: 0.00-0.08; k=3: 0.02-0.12;
   k=5: 0.05-0.25; k=10: 0.15-0.40; k=25: 0.30-0.55.
2. **The gap narrows with k but does not close by k=25** (frozen
   out-of-domain vision should cap expert-only adaptation): predicted
   mean-10 deficit at k=25 >= 0.25.
3. **Demo-equivalence headline**: pretrained k=1 (0.55) is predicted to be
   unreachable by base at ANY k <= 25, i.e. the libero_90 pretrain is worth
   more than 25 demos per task in the low-budget regime.
4. Per-task shape: "push the plate" (task 5, quasi-static shove) is the most
   likely early riser for base; long-horizon drawer+bowl tasks (0, 3, 6)
   stay near zero through k <= 5.

## Decision rules (fixed before results)

- R1bc: base mean-10 at k=25 < pretrained mean-10 at k=1 -> headline claim
  "the in-domain pretrain is worth > 25 demos/task" is licensed.
- R2bc: base curve within 0.1 of the pretrained curve at k >= 10 -> the
  pretrain's value is mostly a low-k effect; report it as demo-equivalence
  at low k only, and flag expert-only capacity (not data) as the binding
  factor for the pretrained curve too.
- R3bc: base curve ~0 everywhere (mean-10 < 0.05 even at k=25) -> expert-only
  from an out-of-domain base cannot adapt at all; any base-vs-pretrained
  comparison must say "under THIS recipe"; a small unfreezing probe becomes
  the natural follow-up before claiming demo-equivalence numbers.
- R4bc: any training/eval harness failure (missing checkpoints, failed
  audits) is reported as such, never as a low score.
