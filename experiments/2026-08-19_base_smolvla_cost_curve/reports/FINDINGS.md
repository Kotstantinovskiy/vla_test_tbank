# Findings and verdict against pre-registered rules (base cost curve)

Written 2026-08-20 after all 60 train+eval jobs completed (no failures;
post-hoc trainable-parameter audit clean: 99.9M expert/projections trainable,
350.2M vision/VLM frozen, forbidden set empty). Predictions:
[PRIOR_EXPECTATION.md](PRIOR_EXPECTATION.md); tables: [REPORT.md](REPORT.md).

## Headline: the libero_90 pretrain is worth almost nothing once k >= 1

| mean-10 | k=1 | k=2 | k=3 | k=5 | k=10 | k=25 |
|---|---:|---:|---:|---:|---:|---:|
| plain smolvla_base | 0.485 | 0.660 | 0.690 | 0.745 | 0.865 | 0.790 |
| libero_90 pretrain (frozen ref) | 0.550 | 0.705 | 0.780 | 0.830 | 0.875 | 0.850 |
| gap | −0.065 | −0.045 | −0.090 | −0.085 | −0.010 | −0.060 |

Tasks 0–2: base 0.550/0.717/0.850/0.883/0.933/0.883 — at k=3 base even
EXCEEDS the reference (0.850 vs 0.717; single seed, within noise).

- **Predictions 1–3 refuted decisively.** Predicted base k=1 was 0.00–0.05;
  measured 0.485. The embodiment/domain gap (real SO-100 -> simulated
  Franka) did not prevent expert-only adaptation even from ONE demo.
- **R1bc ("pretrain worth > 25 demos") — does not fire**, spectacularly:
  base at k=1 already exceeds what the rule asked base to reach by k=25.
- **R2bc fires beyond its own premise**: base is within 0.1 of the pretrained
  curve at EVERY budget, not just k >= 10. Under this recipe the in-domain
  pretrain's measurable value at k >= 1 is ~0.05–0.09 success, concentrated
  at low k, and its only unambiguous contribution is the k=0 point (0.005
  zero-shot needs SOME in-domain training; base has no defined k=0 at all).
- The k=25 dip reproduces on base too (0.865 -> 0.790), mirroring the
  reference curve's non-monotonicity — consistent with the fixed-2000-steps
  undertraining hypothesis rather than anything pretrain-specific.

## Interpretation

The heavy lifting is done by smolvla_base's generic VLM/vision features plus
retraining the ~100M action expert on target demos; 90 tasks of in-domain
LIBERO pretraining add little that one target demonstration does not.
This coheres with the diagnostic probes: language is an exact-string
selector re-keyed by fine-tuning anyway, and scene-side execution adapts
from target frames — both of which k >= 1 supplies directly.

## Implications for the take-home narrative

1. The cost curve's steepness is a property of smolvla_base + the recipe,
   not of our pretrain; reports must not attribute low-k performance to the
   in-domain pretrain.
2. Task-2 methods built on seen-data co-training (retrieval/relabeling) must
   now beat a stronger null: "just fine-tune base on the k demos". Their
   value proposition narrows to k=0 and to the residual ~0.05–0.09 low-k gap.
3. Positive framing for Task 5: 1 demo + 2000 expert-only steps ≈ 0.5
   success on held-out LIBERO-Goal from a generic base — few-shot
   adaptation, not pretraining, is the binding resource.

## Limitations

- Single training seed on both curves (assignment's two-seed requirement
  outstanding); eval noise not per-episode-pinned (shared harness with the
  reference by design; Wilson CIs cover it).
- Expert-only recipe with 2000 fixed steps; conclusions are "under this
  recipe" (R3bc's unfreezing follow-up is moot given the high scores, but a
  steps sweep remains relevant to the k=25 dip).
- Co-tenancy scheduling (3 jobs/GPU) declared in protocol; per-job commands
  identical to the reference.
