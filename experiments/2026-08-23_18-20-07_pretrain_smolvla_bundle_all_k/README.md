# Bundle cost curve, k=1..25

Compiles the ingredients that helped in the single-ingredient experiments
into one recipe and measures the full cost curve on the three assignment
tasks:

- **full fine-tune** (largest single win: k=2 0.683 → 0.950 over expert-only);
- **state noise α=0.10** on the normalized proprioception, training only
  (best k=1 known: 0.600 → 0.700 at 1000 steps); applied uniformly at every
  budget — an extrapolation beyond k=1 that the bundle deliberately tests;
- **lerobot default image transforms** (photometric + RandomAffine;
  robustness to re-planning, neutral-to-positive means);
- **budget-dependent steps**: 1000 / 1500 / 2000 at k=1 / 2 / ≥3.

18 training jobs (tasks 0–2 × k ∈ {1,2,3,5,10,25}) from the pinned pretrain,
seed 1000; evaluation is clean (no augmentation, no noise) at
`n_action_steps` ∈ {50, 35, 25} — 54 points, 20 episodes each, env/noise
seeds `1000 + episode_index`, init state = episode index, horizon 300, all
videos retained.

## Scheduling (idle-minimizing)

Slot-based orchestrator instead of one-worker-per-GPU: each of GPUs 1–3
runs **2 concurrent trainings + 2 concurrent evaluations** (~30 GB per
training, ~12 GB per evaluation on 144 GB cards). Evaluations are dispatched
the moment their checkpoint finishes and overlap the remaining trainings;
drained training workers convert to evaluation workers; shorter jobs are
scheduled first. The determinism gate runs its three action-steps variants
concurrently on separate GPUs. Override with `VLA_GPU_IDS`,
`VLA_TRAIN_SLOTS`, `VLA_EVAL_SLOTS`.

Predictions frozen in `reports/PRIOR_EXPECTATION.md` before any run. Large
checkpoints go to `/var/tmp/vla_outputs/bundle_all_k_20260823_182007`
(~35 GB expected). Single-training-seed experiment (seed 1000).

## Status

Created 2026-08-23 18:20; **not launched yet**.

## Launch sequence

```bash
cd experiments/2026-08-23_18-20-07_pretrain_smolvla_bundle_all_k

source scripts/common_env.sh && pytest -q   # 0. tests
scripts/prepare.sh                          # 1. preflight artifacts
scripts/audit.sh                            # 2. full-FT parameter audit
scripts/smoke_dataset.sh                    # 3. 18 dataset selections
scripts/smoke_env.sh                        # 4. real-env smoke
scripts/production_smoke.sh 1 1 2 3         # 5. gate: train task0/k1 on GPU1,
                                            #    then fwd/rev at n=50/35/25 in
                                            #    parallel on GPUs 1/2/3
scripts/run_all.sh                          # 6. slot-based fan-out + aggregation
scripts/status.sh                           # progress
```
