# Proprioception-noise augmentation at k=1

Does training-time Gaussian noise on the robot state help one-demo
adaptation? Motivated by proprioception-shift work ("Adapt Your Body", 2025;
AugInsert): at k=1 the policy sees one trajectory of arm states and may bind
actions to millimeter-exact proprioception.

12 independent full fine-tunes (recipe identical to
`2026-08-22_18-10-40_pretrain_smolvla_full_ft_low_k`): assignment tasks 0–2 ×
α ∈ {0.00, 0.01, 0.03, 0.05}, k=1 (official demo_0 only). During training,
`observation.state` is perturbed **after** the MEAN_STD normalizer inside
`policy.forward`: `s̃ = s + α·ε`, `ε ~ N(0, I)` — equivalent to
σᵢ = α·Std(sᵢ) per dimension in raw units. Ground-truth actions and images
are untouched; evaluation applies no noise. Noise comes from a dedicated
`torch.Generator` (seed 91000), so the main RNG stream is untouched and the
α=0.00 arm re-runs the full-FT k=1 recipe exactly (in-experiment control;
full-FT reference: 2/20, 19/20, 14/20, mean 0.583).

Implementation: in-process wrapper around lerobot's `train()` with
`make_policy` patched to install the forward wrapper; the training loop
itself is unmodified. Evaluation: n_action_steps=50 only, 20 episodes,
batch 1, env/noise seeds `1000 + episode_index`, init state = episode index,
horizon 300, all videos retained. Gate: task 0 / α=0.00 forward vs reverse
episode order must match exactly.

Predictions frozen in `reports/PRIOR_EXPECTATION.md` before any run. Large
checkpoints go to `/var/tmp/vla_outputs/state_noise_k1_20260822_224355`.

Single-training-seed experiment (seed 1000); k=1 only.

## Status

Launched 2026-08-22 23:25, **completed 2026-08-23 11:16** (a Trackio
finalization crash from a stale `cost_curve.png` default was fixed and only
the finalization stage rerun; science stages ran once). 12/12 trainings and
evaluations, 0 failures; determinism gate 3/20 == 3/20 with zero per-episode
mismatches. Means over tasks 0-2 at alpha 0.00/0.01/0.03/0.05:
0.617 / 0.583 / 0.667 / 0.683. See `reports/REPORT.md`.

## Launch sequence

```bash
cd experiments/2026-08-22_22-43-55_pretrain_smolvla_state_noise_k1

source scripts/common_env.sh && pytest -q   # 0. tests
scripts/prepare.sh                          # 1. preflight artifacts
scripts/audit.sh                            # 2. full-FT parameter audit
scripts/smoke_dataset.sh                    # 3. k=1 selection smoke
scripts/smoke_env.sh                        # 4. real-env smoke
scripts/production_smoke.sh <gpu>           # 5. train task0/alpha0 + determinism gate
scripts/run_all.sh                          # 6. fan-out + aggregation + Trackio
scripts/status.sh                           # progress
```
