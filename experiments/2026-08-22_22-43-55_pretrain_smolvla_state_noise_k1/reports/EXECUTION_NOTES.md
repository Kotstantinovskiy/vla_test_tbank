# Execution notes

## 2026-08-23: Trackio finalization repair

The first full run completed all 12 science stages (trainings, evaluations,
aggregation) but crashed in the Trackio finalization step: the copied
reporting code defaulted to `results/summary/cost_curve.png` while this
experiment's aggregator writes `noise_curve.png`. The path was fixed and the
idempotent orchestrator rerun; all trainings/evaluations were detected as
complete and reused, only aggregation + Trackio re-executed. No scientific
stage ran twice.

## 2026-08-23: grid extension attempted and reverted

An in-place extension of the alpha grid to {0.08, 0.10, 0.20} was prepared
(constants, tests, predictions addendum, regenerated 21-point plan) and then
reverted on user instruction before any new arm trained or evaluated. The
extension moved to the separate experiment
`pretrain_smolvla_state_noise_k1_1k_steps`; this experiment stays a frozen
snapshot of the original 12-point sweep. The evaluation plan was regenerated
back to 12 points.
