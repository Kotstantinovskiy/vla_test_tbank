# Deterministic low-k few-shot reproduction

Reproduction of `2026-08-18_pretrain_smolvla_few_shot_tune_low_k` for k=1/2/3
with training seed 1000 on assignment task IDs 0–2. Each of the 9 adaptations
starts independently from
the pinned official-data pretrain.
An experiment-local runtime view symlinks its unchanged weights and rewrites
only the VLM/tokenizer paths to the pinned offline backbone.

The material protocol change is evaluation determinism: batch size is one,
both the environment seed and SmolVLA flow-sampling seed are
`1000 + episode_index`, and LIBERO `init_state_id` is pinned to
`episode_index`. Before fan-out, task 0 / k=1 is evaluated in forward and
reverse episode order; per-episode outcomes and rewards must match exactly.

Predictions were frozen in `reports/PRIOR_EXPECTATION.md` before preparation or
rollouts. Large checkpoints live in
`/var/tmp/vla_outputs/low_k_deterministic_repro_20260821_203756`; reviewable
manifests, results, reports, logs, and artifact links remain here.

The experiment remains a single-training-seed curve and therefore does not by
itself satisfy the assignment's two-seed requirement.
