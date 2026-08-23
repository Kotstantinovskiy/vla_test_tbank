# Full fine-tune low-k cost curve

Full-fine-tune counterpart of
`2026-08-21_20-37-56_pretrain_smolvla_low_k_deterministic_repro`: the same
nine adaptations (assignment task IDs 0–2 × k=1/2/3) from the same pinned
official-data pretrain, with a single protocol change — the whole policy is
trainable (`train_expert_only=false`, `freeze_vision_encoder=false`,
`train_state_proj=true`) instead of the action expert only. LeRobot's SmolVLA
keeps four unused-by-design guard tensor groups frozen even in a full
fine-tune (vlm `lm_head`, final `text_model.norm`, the last retained VLM text
layer, the expert `lm_head`); the parameter audit asserts that exactly this
set and nothing else stays frozen.

Everything else is held fixed to keep the comparison paired with the
expert-only repro: training seed 1000, 2000 steps, batch 32, fp32, SmolVLA
preset optimizer (AdamW lr 1e-4); evaluation with batch size one, env seed and
flow-noise seed `1000 + episode_index`, LIBERO `init_state_id =
episode_index`, horizon 300, 20 episodes per point, all videos retained.

Each of the 9 adapted checkpoints is evaluated at two inference-time
action-step settings — `n_action_steps=50` (trained default) and `25`
(re-predict twice per chunk) — on identical per-episode seeds/init states,
giving 18 evaluation points (9 training jobs). The override is applied at
load time only; training always uses 50. Before fan-out, task 0 / k=1 is
evaluated in forward and reverse episode order at both variants; per-episode
outcomes and rewards must match exactly.

Predictions are frozen in `reports/PRIOR_EXPECTATION.md` before preparation
and rollouts. Large checkpoints go to
`/var/tmp/vla_outputs/full_ft_low_k_20260822_181040`; reviewable manifests,
results, reports, logs, and artifact links stay here.

This is a single-training-seed experiment (seed 1000) and does not by itself
satisfy the assignment's two-seed requirement.

## Status

2026-08-22 18:21: preflight completed (prepare, full-FT parameter audit,
dataset and environment smokes all passed). The production/determinism gate
was started (task 0 / k=1 training) and **stopped on request at ~step 30 of
2000**; no checkpoint was written and the output root is empty, so a relaunch
restarts the gate from scratch. No orchestrator fan-out ran and no rollout
was evaluated.

2026-08-22 (later): before any rollout was evaluated, the evaluation protocol
was extended to cover inference `n_action_steps` ∈ {50, 25} per checkpoint
(18 evaluation points); the extension is recorded in the predictions addendum
and `configs/protocol.yaml`.

2026-08-22 ~22:00 (relaunch): **completed** — 9/9 trainings, 18/18
evaluations, 0 failures. Determinism gate passed exactly at both variants
(n=50 2/20==2/20, n=25 0/20==0/20, zero per-episode mismatches).
Mean cost curve over tasks 0-2: n=50 0.583/0.950/0.767 and n=25
0.533/0.750/0.800 at k=1/2/3 (expert-only repro reference:
0.600/0.683/0.667). See `reports/REPORT.md`, `results/summary/`, and
`reports/EXECUTION_NOTES.md` (mid-run TRAIN_WORKERS 8→32 for later jobs).

## Launch sequence

```bash
cd experiments/2026-08-22_18-10-40_pretrain_smolvla_full_ft_low_k

# 0. Focused tests (no GPU).
source scripts/common_env.sh && pytest -q

# 1. Preflight artifacts: base/backbone manifests, episode manifest,
#    evaluation plan, LIBERO config, artifact symlinks.
scripts/prepare.sh

# 2. Trainable-parameter audit (full-FT flags; writes
#    artifacts/trainable_parameters.json, must leave only guard tensors frozen).
scripts/audit.sh

# 3. Dataset selection smoke: instantiate all 9 task/k datasets, verify
#    loaded episode indices.
scripts/smoke_dataset.sh

# 4. Real-environment smoke: create each target env, assert instruction, reset.
scripts/smoke_env.sh

# 5. Production + determinism gate: train task 0/k=1, evaluate forward and
#    reverse episode order (writes artifacts/production_smoke.json).
scripts/production_smoke.sh <gpu>

# 6. Full fan-out over the remaining points, then aggregation and Trackio.
scripts/run_all.sh

# Progress at any time:
scripts/status.sh
```
