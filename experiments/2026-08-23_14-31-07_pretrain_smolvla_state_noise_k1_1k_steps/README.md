# High-alpha state noise at k=1, 1000 training steps

Continuation of
[`2026-08-22_22-43-55_pretrain_smolvla_state_noise_k1`](../2026-08-22_22-43-55_pretrain_smolvla_state_noise_k1/)
(means 0.617/0.583/0.667/0.683 at α 0.00–0.05, 2000 steps — still rising at
the grid edge) with two deliberate changes:

1. **Higher noise**: α ∈ {0.08, 0.10, 0.20} (declared adaptive extension),
   plus the mandatory **α=0.00 control re-trained at this step budget**.
2. **Half the training budget**: 1000 optimizer steps instead of 2000
   (lerobot auto-scales the LR schedule, so the schedule compresses too —
   which is exactly why the 1k α=0 control exists; 2000-step numbers are
   context, not a baseline).

Everything else is byte-identical to the first sweep (code copied verbatim):
full fine-tune, seed 1000, k=1 (official demo_0), Gaussian noise `α·ε` on
the normalized state in training only from a dedicated RNG stream (seed
91000), evaluation at n_action_steps=50, 20 episodes, seed banks 1000..1019,
all videos kept, forward/reverse determinism gate on task 0 / α=0.

12 jobs = 3 tasks × 4 alphas. Predictions frozen in
`reports/PRIOR_EXPECTATION.md` before any run; the adaptive grid choice is
disclosed there and in the protocol. Large checkpoints:
`/var/tmp/vla_outputs/state_noise_k1_1k_20260823_143107`.

Single-training-seed experiment (seed 1000); k=1 only.

## Status

Launched 2026-08-23 14:38, **completed 2026-08-23 17:49 local** — 12/12
trainings and evaluations, 0 failures; determinism gate 3/20 == 3/20, zero
per-episode mismatches. Means over tasks 0-2 at alpha 0.00/0.08/0.10/0.20:
0.600 / 0.667 / 0.700 / 0.667. See `reports/REPORT.md`.

## Launch sequence

```bash
cd experiments/2026-08-23_14-31-07_pretrain_smolvla_state_noise_k1_1k_steps

source scripts/common_env.sh && pytest -q   # 0. tests
scripts/prepare.sh                          # 1. preflight artifacts
scripts/audit.sh                            # 2. full-FT parameter audit
scripts/smoke_dataset.sh                    # 3. k=1 selection smoke
scripts/smoke_env.sh                        # 4. real-env smoke
scripts/production_smoke.sh <gpu>           # 5. train task0/alpha0 (1000 steps) + gate
scripts/run_all.sh                          # 6. fan-out + aggregation + Trackio
scripts/status.sh                           # progress
```
