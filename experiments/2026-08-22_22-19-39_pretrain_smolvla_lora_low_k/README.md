# LoRA-on-VLM low-k cost curve

Third rung of the adaptation ladder on the same protocol: **expert-only**
(`2026-08-21_20-37-56_pretrain_smolvla_low_k_deterministic_repro`) →
**LoRA (this experiment)** → **full-FT**
(`2026-08-22_18-10-40_pretrain_smolvla_full_ft_low_k`). The same nine
adaptations (assignment task IDs 0–2 × k=1/2/3) start from the same pinned
official-data pretrain; the VLM is adapted only through LoRA adapters
(r=16, α=32 on every used VLM linear: text layers 0–14, vision encoder,
connector), while the action expert and projections train fully — exactly
the expert-only baseline's trainable set — via PEFT `modules_to_save`.
Training uses lerobot 0.6.1's native PEFT integration (peft 0.20.0).

After each training job the adapter is merged into the base weights and
saved as a plain SmolVLA checkpoint, so the evaluation pipeline is byte-for-
byte the one used by the sibling experiments: each merged checkpoint is
evaluated at inference `n_action_steps=50` and `25` (18 points), batch size
one, env/noise seeds `1000 + episode_index`, LIBERO `init_state_id =
episode_index`, horizon 300, 20 episodes per point, all videos retained.
Before fan-out, task 0 / k=1 runs forward and reverse episode order at both
variants and must match exactly.

Predictions are frozen in `reports/PRIOR_EXPECTATION.md` before preparation
and rollouts. Large checkpoints go to
`/var/tmp/vla_outputs/lora_low_k_20260822_221939`; reviewable manifests,
results, reports, logs, and artifact links stay here.

Single-training-seed experiment (seed 1000); the assignment's two-seed
requirement remains outstanding.

## Status

Created 2026-08-22 22:19; **not launched yet**.

## Launch sequence

```bash
cd experiments/2026-08-22_22-19-39_pretrain_smolvla_lora_low_k

# 0. Focused tests (no GPU).
source scripts/common_env.sh && pytest -q

# 1. Preflight artifacts: base/backbone manifests, episode manifest,
#    evaluation plan, LIBERO config, artifact symlinks.
scripts/prepare.sh

# 2. LoRA trainable-parameter audit (wraps the base policy with the frozen
#    PEFT config; writes artifacts/trainable_parameters.json).
scripts/audit.sh

# 3. Dataset selection smoke.
scripts/smoke_dataset.sh

# 4. Real-environment smoke.
scripts/smoke_env.sh

# 5. Production + determinism gate: train task 0/k=1 (with merge), evaluate
#    forward and reverse order at n=50 and n=25.
scripts/production_smoke.sh <gpu>

# 6. Full fan-out, aggregation, Trackio.
scripts/run_all.sh

# Progress:
scripts/status.sh
```
