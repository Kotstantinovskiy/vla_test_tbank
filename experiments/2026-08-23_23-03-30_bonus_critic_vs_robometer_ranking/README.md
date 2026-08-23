# Bonus B: own progress critic vs Robometer ranking

This experiment ranks already evaluated SmolVLA policies without using rollout
reward as model input. It compares the progress signal learned in
`2026-08-23_20-49-13_bonus_qwen35_progress_critic` with the pinned published
`aliangdw/Robometer-4B-LIBERO` checkpoint.

## Fixed protocol

- Target tasks: LIBERO-Goal logical IDs 0, 1, and 2.
- Candidates per task: the canonical seen pretrain checkpoint and final bundle
  checkpoints at `k={1,2,3,5,10,25}`.
- Evaluation data: all 20 saved rollout videos per candidate, 420 videos total.
- Action schedule: **only `n_action_steps=50`**. No result from 25 or 35 enters
  this experiment.
- Input to both critics: the exact task instruction plus four RGB frames at
  rounded `linspace(0, final_frame, 4)` indices.
- Candidate score: mean endpoint progress over its 20 rollouts.
- Ground truth ranking: environment success rate over those same 20 rollouts.
- Primary metrics: per-task and macro Spearman rho and Kendall tau. Secondary
  metrics: top-set hit and regret, pooled correlations, and episode bootstrap
  confidence intervals.

The blind manifest contains no reward, outcome, or success fields. The
aggregator is gated by a SHA-256 seal written only after both critics produced
exactly 420 predictions. The experiment does not train or optimize a policy.

The own critic checkpoint is fixed before ranking as step 2000: it is the best
among checkpoints actually saved every 200 steps. (An unsaved step-1300
validation point was slightly better and cannot be selected reproducibly.)

## Run

```bash
source scripts/common_env.sh
scripts/prepare.sh
scripts/launch.sh
```

Status is written atomically to `results/status.json`; the detached log is
`results/logs/run.log`.
