# Bonus B: Qwen3.5-4B progress critic

This self-contained experiment implements stages 1–6 of the Bonus B preparation only. It
is designed to train a video progress signal on canonical LIBERO-90 expert videos and reserves complete
task instructions for validation. It intentionally contains no checkpoint ranking, no
Robometer inference, and no policy optimization.

## Protocol

- Base model: `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Input: the exact LIBERO instruction and four chronological top-camera frames sampled from
  a trajectory prefix.
- Target: absolute endpoint progress, discretized into 32 uniform bins.
- Head: one linear 32-bin classifier on the final prompt hidden state.
- Adaptation: LoRA r=32, alpha=64 on the language backbone; frozen vision encoder.
- No preference head, success head, failed trajectories, rewind augmentation, ranking, or
  Robometer comparison.
- Data: canonical rot180 official LIBERO-90 only. The three assignment LIBERO-Goal tasks are
  asserted absent.
- Split: deterministic by complete natural-language task instruction, seed 1000, 10% held out.
- Gradient checkpointing: explicitly disabled. An OOM must first be addressed by reducing
  per-device batch size.

The model is 4B, so this is scientifically an own-4B critic rather than the assignment's
literal “small critic”. That limitation must be disclosed in the final bonus report.

## Implemented stages 1–6

1. Pin the Qwen3.5-4B base revision and the progress-only protocol.
2. Validate the canonical LIBERO-90 conversion and build the episode manifest.
3. Make a deterministic task-disjoint train/validation split with seed 1000.
4. Decode four chronological prefix frames and assign the normalized endpoint-progress bin.
5. Implement LoRA training, a frozen-vision audit, fixed validation, loss plotting,
   Trackio logging, and adapter/head checkpoint saving.
6. Provide GPU-0 data-smoke and 50-step benchmark entrypoints, including full-run runtime
   projection without starting the 2,000-step job.

Checkpoint ranking, Robometer evaluation, and policy optimization are outside this experiment.

## Commands

```bash
cd /vla_test/experiments/2026-08-23_20-49-13_bonus_qwen35_progress_critic
source scripts/common_env.sh

# Validate the canonical dataset and write the task-level split manifest.
scripts/prepare.sh

# Decode real train/validation clips and write a reviewable contact sheet.
scripts/smoke_data.sh

# Focused tests.
pytest -q

# GPU-0 engineering benchmark: 50 optimizer steps and validation at 0/25/50.
# The LR scheduler retains the declared 2,000-step horizon.
scripts/benchmark_50.sh
```

The benchmark writes:

- `results/benchmark_50/metrics.jsonl`;
- `results/benchmark_50/loss_curve.png` with train and validation loss;
- `results/benchmark_50/summary.json` with speed, peak VRAM, and 2,000-step ETA;
- `artifacts/trainable_parameters.json` with the LoRA/freeze audit.

The full run completed on physical GPU 0 on 2026-08-23 with the pinned protocol:

```bash
scripts/launch_full.sh
```

It runs 2,000 optimizer steps, evaluates the fixed validation subset every 100 steps, and
saves adapter/head checkpoints every 200 steps. Runtime state, PID, metrics, and logs are in
`results/status.json`, `results/train_full.pid`, `results/training/metrics.jsonl`, and
`results/logs/train_full.log` respectively.

Training completed all 2,000 steps in 48.25 minutes. Validation cross-entropy improved from
5.087 at step 0 to a best 2.196 at step 1,300 and finished at 2.216. The final checkpoint is
`results/training/checkpoints/002000`; see `reports/TRAINING.md`.

All generated data-smoke and 50-step benchmark artifacts were removed on 2026-08-23 at the
user's request. Running the corresponding commands recreates them from the pinned protocol.
