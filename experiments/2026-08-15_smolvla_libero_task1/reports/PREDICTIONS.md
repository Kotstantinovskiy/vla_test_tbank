# A-priori predictions for Task 1

Status: **LOCKED BEFORE ANY ROLLOUT OR TARGET-TASK FINE-TUNING**  
Lock date: 2026-08-15 (Europe/Moscow)  
Git commit: recorded by `scripts/lock_predictions.sh`.

## Experimental contract

- Seen checkpoint: `crislmfroes/smolvla-libero-90`, immutable Hub revision
  `418f9d0e5b48585bcee1e1a7d47e302629af78da`.
- Provenance: fine-tuned from `lerobot/smolvla_base` for 30,000 steps on
  4,500 LIBERO-90 episodes; the checkpoint card does not list any
  `libero_goal` data.
- Held-out tasks: `libero_goal`, environment task IDs 0, 1, and 2.
- Demonstrations: the first 5, 10, or 25 episodes for each task in dataset
  order. No filtering by success, length, or visual inspection is allowed.
- Zero-shot evaluation: 20 episodes per task and condition, seeds derived
  deterministically from master seed 1000.
- Language controls use identical reset seeds: (a) the instruction of the next
  target task, cyclically, and (b) a fixed nonsense instruction.
- Adaptation: separate single-task full-parameter SmolVLA fine-tunes, always
  starting from the exact seen checkpoint. Optimizer settings are inherited
  from the checkpoint; no augmentation, PEFT, replay, regularization, or
  checkpoint selection by rollout success.
- Training budget: 2,000 optimizer steps for every `(task, k)` pair, batch size
  32, seed 1000. This deliberately holds optimization compute fixed so the
  horizontal axis measures demonstration count rather than compute.
- Adapted policies: 20 rollout episodes per `(task, k)`, with the same master
  seed 1000. The final checkpoint is used; no best-of-run selection.

If a software incompatibility makes one item impossible, the deviation must be
written to `reports/DEVIATIONS.md` before the affected result is inspected.

## Numerical prediction

| Task ID | True k=0 | Wrong language | Nonsense | FT k=5 | FT k=10 | FT k=25 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.15 | 0.05 | 0.00 | 0.35 | 0.55 | 0.75 |
| 1 | 0.10 | 0.05 | 0.00 | 0.30 | 0.50 | 0.70 |
| 2 | 0.05 | 0.00 | 0.00 | 0.20 | 0.40 | 0.65 |
| **Mean** | **0.10** | **0.03** | **0.00** | **0.28** | **0.48** | **0.70** |

## Shape and mechanisms

I predict a monotone but sublinear cost curve. Five demonstrations should
teach the new object/goal association but cover too few reset states, so
closed-loop covariate shift will still dominate. The largest gain should occur
between 5 and 10 demonstrations, when the policy sees enough state diversity
to recover from small execution errors. The 10-to-25 gain should be smaller as
residual failures become long-horizon control failures rather than missing task
semantics.

The true-instruction condition should beat both controls. If it does not, the
main alternative explanation is that the seen policy relies on visual scene
priors and memorized motor programs, with language acting as a weak or ignored
feature. The paired-seed controls are intended to distinguish this from reset
difficulty.

At k=5 I expect visible overfitting: lower action-prediction loss without a
commensurate success increase. Task-specific fine-tuning should outperform a
joint three-task adapter at equal per-task steps in the very-low-data regime,
but joint adaptation is expected to catch up or surpass it by k=25 through
shared regularization. The latter comparison belongs to Task 2 and is not part
of this locked baseline.
