# Predictions (recorded 2026-08-17, before conversion of libero_90 completed and before any training)

Prior-informed: the uncorrected pretrain, the mirrorfix preparation, the
prompt-only runs, and the provenance audit all exist.

1. **Conversion integrity.** Round-trip verification will show bit-exact
   actions/state and frame MAE at pure codec-noise level (crf-18 AV1 at
   128x128: expected MAE ~1, well under the h264-chain's ~2-11), for both
   suites.
2. **Training dynamics.** Loss curve comparable to the two previous pretrains
   (final ≈ 0.3-0.4 at 30k steps). Native 128x128 frames decode faster than
   224x224 h264, so data loading should not bottleneck 4 GPUs at 16
   workers/rank.
3. **Positive control (seen libero_90 task 0, frozen checkpoint, 20
   episodes, 128x128 rendering).** ≥ 0.5, plausibly ≥ 0.8. Stronger prior
   than for mirrorfix: this checkpoint's train/eval gap is codec noise only —
   same renderer, same resolution, same orientation, same state recipe. If
   this stays at the floor, the remaining suspects are the eval harness
   itself or the state/action conventions, not the visual domain.
4. **Zero-shot on held-out `libero_goal` tasks 0-2.** Still likely at/near
   the floor: held-out means held-out, and the orientation diagnostic showed
   input convention was not the binding constraint for the old checkpoint.
   Weakly higher chance of nonzero successes than before, since for the first
   time literally every input convention matches between training and
   evaluation.
5. **Downstream adaptation (k=5/10/25 from the official target conversion).**
   Comparable to or slightly above the naive baseline (0.90/0.92/0.97),
   with two protocol improvements that matter more than the mean: "first k"
   now selects official demo_0..demo_{k-1}, and seen/target statistics come
   from one pipeline, shrinking the normalizer-swap confound.

Falsifier for the conversion: any bit-exactness failure or frame MAE above
the threshold in `artifacts/conversion_verification_*.json` blocks training.
