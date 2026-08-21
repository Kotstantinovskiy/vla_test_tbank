# Project instructions

## Purpose

This repository studies low-demo adaptation of SmolVLA on LIBERO. The primary
scientific output is a cost curve: task success as a function of the number of
demonstrations. Reproducibility, protocol honesty, and separation of training
and evaluation effects are more important than producing a favorable number.
Read `PROBLEM.md` (assignment version 2026-08-15: target tasks are named by
instruction text, and a "ceiling" clause requires probing k=1-2 budgets when
the naive baseline saturates by k=5) and the root `README.md` for the
repository layout before changing an experiment.

## Repository model

- Treat the repository as a journal of nearly self-contained experiments.
- Put every new experimental idea in
  `experiments/YYYY-MM-DD_HH-MM-SS_short_name/`, using the local 24-hour
  creation timestamp (`date +%Y-%m-%d_%H-%M-%S`).
- Keep that experiment's code, scripts, configuration, tests, reports, raw
  results, summaries, media, and artifact links inside its own directory.
- Do not import mutable code from another experiment. Copy the required version
  into the new experiment so old runs remain reproducible.
- Shared root scripts are allowed only for stable infrastructure with the same
  contract across experiments, currently the combined Trackio dashboard and
  the revision-pinned official LIBERO HDF5 downloader.
- Do not rewrite a completed experiment to test a new idea. Create a new
  timestamped experiment. A repair of an implementation error may stay in the
  original experiment, but document the error and preserve the scientific
  protocol.

Expected experiment layout:

```text
experiments/YYYY-MM-DD_HH-MM-SS_name/
├── README.md
├── pyproject.toml
├── configs/
├── scripts/
├── src/
├── tests/
├── results/{raw,summary,media,logs}/
├── artifacts/
└── reports/
```

## Environment and commands

- Use the root uv workspace and Python 3.12 environment. Do not create a second
  ad-hoc environment for an experiment.
- Prefer `uv sync --frozen` and commands from the root `.venv`. If `uv` is not
  on `PATH`, locate the existing binary rather than installing another copy.
- From an experiment, source `scripts/common_env.sh` before manual commands so
  dataset, Hugging Face, LIBERO, MuJoCo, output, and Trackio paths are consistent.
- When adding an experiment package, add it to the root `project.dependencies`
  and `tool.uv.sources`, then update `uv.lock` intentionally.
- Pin model and dataset revisions in the experiment protocol and recorded
  artifacts. Avoid network access during training after preparation completes.
- The `uv` binary lives at `/var/tmp/vla_tools/uv`; it is not on `PATH`.

## Canonical data and checkpoints

- Canonical datasets are the in-repo conversions of the official LIBERO HDF5
  (`yifengzhu-hf/LIBERO-datasets@f13aa24`), stored at
  `/var/tmp/vla_libero_official_rot180/{libero_90,libero_goal}` with a
  `conversion_manifest.json` in each root. Contract: frames are
  rot180(official) == the LeRobot eval convention, native 128x128, AV1 crf-18,
  fps=20; actions are bit-exact official float32 with the gripper channel in
  {-1,+1} (the environment's convention); the 8-dim state is official
  ee_pos + ee_ori + gripper_states; episode order is the official order, so
  the first k episodes of a task are official demo_0..demo_{k-1}. Round-trip
  verification artifacts live in the converting experiment
  (`2026-08-17_smolvla_pretrain_libero`). Do not reintroduce third-party
  conversions; the assignment's named dataset (`nvidia/LIBERO_LeRobot_v3`) is
  less complete (73 task rows, one task missing, ~3900 demos, {0,1} gripper),
  and our deviation from it must be declared in reports.
- The canonical seen checkpoint is
  `/var/tmp/vla_outputs/seen_libero90_official_20260817/checkpoints/030000/pretrained_model`
  (official-data pretrain; seen positive control 20/20). Record its
  `model.safetensors` SHA-256 in every experiment that consumes it.
- The crislmfroes lineage (mirrored pretrain, its baselines and prompt-only
  runs) was deleted from the worktree on 2026-08-18; its headline numbers
  survive only in README registry tombstones and git history. Do not resurrect
  those artifacts silently.

## Experimental integrity

- Record predictions before inspecting target rollout results.
- Select demonstrations deterministically from the conversion manifest: the
  first `k` episodes of a task, asserting that official demo indices are
  contiguous from zero (first k == official demo_0..demo_{k-1}). Never
  cherry-pick successful demos. After building a training dataset with
  `--dataset.episodes`, verify which episodes actually loaded (some
  lerobot/datasets versions silently load the wrong ones; lerobot 0.6.1 as
  pinned is verified correct).
- Save the exact episode manifest, task mapping, language instruction, seed,
  model checkpoint, dataset revision, trainable-parameter audit, and commands.
- Keep independent task/budget adaptations independent: each must start from the
  declared base checkpoint unless the protocol explicitly states otherwise.
- Do not silently change a frozen baseline, number of steps, seeds, rollout
  horizon, action scheduling, or evaluation episodes after observing results.
- Distinguish training completion from evaluation completion. A saved checkpoint
  is not a successful job if rollout evaluation or artifact validation failed.
- Never fabricate missing metrics. Surface failed and incomplete conditions in
  status files and reports.
- For parameter-efficient runs, audit names and counts before training. Assert
  that every intended frozen vision/VLM parameter is frozen and every intended
  action expert/projection group is trainable.
- Validate environment task IDs against their natural-language task descriptions
  before rollouts. Dataset task tables are indexed by instruction text and
  repeated instructions are merged across scenes; never map tasks by row
  number.
- A frozen checkpoint's floor result on held-out tasks is interpretable only
  next to a passing positive control (the same frozen checkpoint succeeding on
  a seen task with production eval settings). Zero-shot experiments must also
  carry language controls (true / wrong-task / nonsense prompts).
- Disclose the fine-tune normalizer swap in every cost-curve experiment:
  LeRobot replaces checkpoint normalization statistics with target-dataset
  statistics, so k=0 and k>0 points run under different normalizers.
- The assignment requires at least two training seeds for cost-curve claims;
  single-seed runs must be labeled as such in protocol and reports.

## Artifacts and storage

- Write lightweight, reviewable outputs under the experiment directory.
- Large datasets and checkpoints may live under `/var/tmp`, but expose them
  through clearly named links inside the experiment's `artifacts/` directory.
- Never commit caches, full datasets, generated checkpoint trees, or local
  Trackio databases. Respect `.gitignore` and do not remove user artifacts.
- Store machine-readable JSON/CSV summaries in addition to plots.
- For rollout media, preserve outcome, task, budget, episode index, and seed in
  a manifest. Verify that every path referenced by a completed result exists.
- Video policy: record all evaluation episodes to disk (every success must be
  reviewable); Trackio receives only representative media (episode 0 per task,
  or the first success and first failure per budget).
- When an experiment is deleted on request, keep a tombstone row in the root
  README registry with its headline numbers and deletion date, and prefer
  moving irreplaceable outputs to a dated `/var/tmp/*recovery*` directory over
  outright deletion; note explicitly when raw data was never committed and is
  therefore gone.

## Trackio

- Each experiment owns its Trackio database and media under
  `artifacts/trackio/` and uses a unique project name.
- Log training loss, gradient norm, learning rate, timing/throughput, evaluation
  success, metric tables, plots, reports, and the media required by that
  experiment's protocol.
- Resume live runs instead of creating duplicate runs. LeRobot abbreviates
  displayed steps such as `1K`; recover exact steps from the known log frequency
  or another collision-free source.
- The combined root dashboard is built with `scripts/index_trackio.sh` and
  started with `scripts/show_trackio.sh`. Do not point experiment logging
  directly at the root snapshot directory.
- Ensure browser media is copied into Trackio storage; do not rely on symlinks
  that resolve outside the served dashboard directory.
- `scripts/index_trackio.sh` adds projects to the `.trackio-dashboard`
  snapshot but never removes them; after deleting or renaming an experiment,
  clean its stale `*.db` and `media/` entries out of the snapshot and restart
  the dashboard, otherwise phantom projects appear in the sidebar.

## Long-running jobs

- Before launching, resuming, or stopping work, inspect `results/status.json`,
  relevant logs, process IDs, GPU utilization, and checkpoint completeness.
- Do not start a duplicate orchestrator or training job. Do not kill or replace
  a running job merely because GPU utilization is momentarily zero during data
  loading or checkpointing.
- Preserve completed checkpoints when evaluation fails. Fix the evaluation issue
  and rerun only the missing stage when possible.
- Detached jobs must write separate logs and an atomic machine-readable status.
  Make reruns idempotent: completed stages should be detected and skipped.
- When reporting progress, provide exact completed/failed/pending counts, active
  steps, and the actual blocker. Do not describe evaluation failures as training
  failures.

## LIBERO evaluation setup

- Every experiment must use its own explicit `LIBERO_CONFIG_PATH`, normally
  `artifacts/libero_config`, and preparation must write `config.yaml`; creating
  only the directory is not sufficient.
- Generate the config noninteractively from the active uv environment before
  importing `libero.libero`. Detached jobs have no stdin, so LIBERO's first-run
  prompt will fail with `EOFError` and must never be relied upon.
- Validate that `benchmark_root`, `bddl_files`, and `init_states` exist. Treat
  package-local assets separately because current LIBERO wheels may resolve
  them through the LIBERO cache.
- Evaluation entrypoints must idempotently preflight or restore the config before
  environment creation, even if preparation was already run.
- Before fanning out evaluation, create one real target-suite environment,
  assert its natural-language task description, call `reset`, and close it. A
  model-loading smoke test alone does not validate LIBERO setup.
- Then run one complete evaluation point with the production episode count and
  media settings. Verify its JSON, per-episode outcomes, and every expected video
  before launching the remaining evaluations.
- If evaluation setup fails after training, preserve the completed checkpoints
  and retry evaluation only. Never repeat optimizer steps to repair an
  environment-configuration error.

## Implementation and verification

- Preserve unrelated work in the dirty worktree. Inspect before editing and use
  focused patches.
- Use `rg`/`rg --files` for discovery and `apply_patch` for source edits.
- Avoid destructive commands. Move questionable incomplete outputs to a clearly
  named recovery location instead of deleting them.
- Add tests for protocol invariants, selection logic, task mappings, checkpoint
  completeness, metric aggregation, and media manifests.
- Before a full GPU run, perform proportionate checks: compile/import, focused
  tests, parameter audit, dataset/schema validation, and a short real smoke run.
- After changes, run focused tests and `git diff --check`. Do not run every
  expensive evaluation merely to validate a small reporting-only edit.
- Keep documentation and commands accurate. Report deviations, repairs, and
  unresolved limitations explicitly.
