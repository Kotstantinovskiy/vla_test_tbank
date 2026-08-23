# Execution notes

## 2026-08-22: first launch aborted

The first launch was stopped on user request at ~training step 30 of the
production gate; no checkpoint was written and no rollout was evaluated. The
aborted launcher log is `results/logs/launcher_aborted_20260822_1821.log`.

## 2026-08-22: evaluation extended to n_action_steps ∈ {50, 25}

Before any rollout of this experiment was evaluated, the protocol was
extended to evaluate every adapted checkpoint at inference n_action_steps=50
and 25 (18 evaluation points). Predictions for the new variant were recorded
in the addendum of `reports/PRIOR_EXPECTATION.md` before relaunch.

## 2026-08-22: determinism gate

Passed at both variants: task 0 / k=1 forward vs reverse episode order,
n=50 2/20 == 2/20 and n=25 0/20 == 0/20, zero per-episode mismatches.

## 2026-08-22: TRAIN_WORKERS raised 8 -> 32 mid-run

Training was dataloader-bound (log_freq metrics: data_s ~1.2 s vs
updt_s ~0.23 s per step; 256 CPUs mostly idle), making each 2000-step job
~45 min instead of ~15-20 min. `TRAIN_WORKERS` was raised from 8 to 32 after
the gate job and the first fan-out wave (task_1_k_1, task_2_k_1, task_0_k_2)
had already started; those four jobs trained with num_workers=8, later jobs
with 32.

This is an infrastructure throughput knob, not a protocol change: PyTorch
DataLoader batch order is defined by the seeded sampler and is independent of
worker count, so trained weights are unaffected in expectation and the
training seed, steps, batch size, precision, and evaluation protocol are
unchanged. Each job's exact value is recorded in its own
`train_config.json` inside the checkpoint tree.
