# Image-augmentation full-FT low-k cost curve

Does standard training-time image augmentation help low-demo adaptation?
The two augmentation families used across VLA / visuomotor imitation are
photometric distortion (color jitter — RT-1, robomimic, lerobot recipes) and
small geometric shifts (random crop/shift/affine — DrQ(-v2), robomimic,
Diffusion Policy). lerobot 0.6.1's default dataset `image_transforms` bundle
contains exactly these two families:
ColorJitter brightness/contrast (0.8–1.2), saturation (0.5–1.5), hue
(±0.05), SharpnessJitter (0.5–1.5), and RandomAffine (±5°, translate ≤5%),
with up to 3 transforms sampled per frame. This experiment enables that
bundle verbatim (`--dataset.image_transforms.enable=true`, both cameras) — a
protocol test asserts the installed defaults match the documented values.

Everything else follows the full-FT experiment
(`2026-08-22_18-10-40_pretrain_smolvla_full_ft_low_k`, the paired
no-augmentation reference): 9 adaptations (tasks 0–2 × k=1/2/3) from the
pinned pretrain, full fine-tune, seed 1000, batch 32, fp32. One additional
protocol change: training steps are budget-dependent — **1000 / 1500 / 2000
at k=1/2/3** (the reference trained 2000 everywhere), so for k=1/2 the
comparison confounds augmentation with optimization length; only k=3
isolates augmentation (disclosed in `configs/protocol.yaml`);
evaluation without augmentation at `n_action_steps=50` and `25` (18 points),
20 episodes, env/noise seeds `1000 + episode_index`, init state = episode
index, all videos retained; determinism gate on task 0 / k=1 at both
variants before fan-out.

Predictions frozen in `reports/PRIOR_EXPECTATION.md` before any run. Large
checkpoints go to `/var/tmp/vla_outputs/image_aug_low_k_20260823_001849`.

Single-training-seed experiment (seed 1000).

## Status

Launched 2026-08-23 ~14:44, **completed 2026-08-23** — 9/9 trainings
(1000/1500/2000 steps at k=1/2/3), 18/18 evaluations, 0 failures.
Determinism gate passed exactly at both variants (n=50 3/20==3/20,
n=25 0/20==0/20). Mean cost curve over tasks 0-2 (full-FT no-aug reference
in parentheses): n=50 0.583 (0.583) / 0.900 (0.950) / 0.783 (0.767); n=25
0.567 (0.533) / 0.783 (0.750) / 0.817 (0.800) at k=1/2/3. See
`reports/REPORT.md` and `results/summary/`.

## Launch sequence

```bash
cd experiments/2026-08-23_00-18-49_pretrain_smolvla_image_aug_low_k

source scripts/common_env.sh && pytest -q   # 0. tests (incl. transform-defaults invariant)
scripts/prepare.sh                          # 1. preflight artifacts
scripts/audit.sh                            # 2. full-FT parameter audit
scripts/smoke_dataset.sh                    # 3. dataset selection smoke
scripts/smoke_env.sh                        # 4. real-env smoke
scripts/production_smoke.sh <gpu>           # 5. gate: train task0/k1 + fwd/rev eval at n=50,25
scripts/run_all.sh                          # 6. fan-out + aggregation + Trackio
scripts/status.sh                           # progress
```
