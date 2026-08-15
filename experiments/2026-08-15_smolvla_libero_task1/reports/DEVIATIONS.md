# Protocol deviations

This file is append-only after the prediction lock. Record a deviation before
looking at the result produced under the changed protocol.

## 2026-08-15 — pin last usable pre-rewrite target-data revision

Before target training, preparation against the then-current dataset revision
`86958911c0f959db2bbbdb107eb3e17c5f9c798e` failed. Its episode metadata maps
episode 399 to `data/chunk-000/file-025.parquet`, while that shard physically
contains only episodes 61–63. The same mismatch occurs throughout the target
range. Revision `9176d427966503c81ac9f8f96502e50861a15ee7` maps episode 399 to
file 149 and has the same 377-file inventory as the physical data shards. Its
remaining per-episode boundary defects are handled separately below.

All target-data operations are therefore pinned to revision `9176d427...`, the
latest usable revision before the metadata rewrite. Repository,
episodes, ordering, task instructions, and all optimization/evaluation settings
are unchanged. No target training or success rollout had run when this deviation
was recorded.

## 2026-08-15 — repair selected episode frame bounds for LeRobot 0.6.1

The mandatory one-step training smoke test failed before reading a batch or
performing an optimizer update. In revision `9176d427...`, the columns
`dataset_from_index` and `dataset_to_index` are offsets local to each parquet
shard; LeRobot 0.6.1's `EpisodeAwareSampler` treats them as global frame
indices. For example, episode 399 has shard-local bounds `[621, 757)`, while
its stored and physically verified global `index` range is `[104020, 104156)`.

The same smoke also exposed shifted `data/file_index` boundaries: for example,
episode 387 is reported in shard 147, while the physical episode column places
it in shard 148. Preparation therefore downloads the bounded candidate shard
interval 145–193, derives the physical file mapping from `episode_index`, and
replaces the file mapping plus the two frame-bound fields only for the 75
preregistered target episodes. It checks that every selected episode occurs in
exactly one shard and that every corrected range has the declared length,
retains the upstream metadata as `file-000.parquet.upstream`, and writes a
row-level JSON audit to `meta/FRAME_BOUNDS_REPAIR.json`. The verified selection
contains 12,696 frames; the preliminary 12,058 count silently omitted episode
387 because of its bad shard pointer.

The frame-bound problem was recorded and repaired before any target optimizer
step. The shifted shard pointer was then exposed by the first parallel 5-demo
launch: task 2 stopped before producing a batch or optimizer step, while tasks
0 and 1 had started. None of the corrected shard pointers belong to the five
episodes used by those two runs, and both loaded frame counts were verified
against all five selected episodes, so those valid runs were retained. The
failed task-2 directory is preserved under `artifacts/checkpoints/failed/` and its clean rerun
uses the repaired metadata. Demonstrations, order, frames, actions, images,
statistics, and checkpoints are otherwise unchanged.

## 2026-08-15 — correct logical-task to LIBERO environment mapping

The first evaluation implementation treated experiment logical task IDs 0/1/2
as if they were suite-local `libero_goal` IDs. Only logical task 0 happened to
match. The intended instructions map to environment IDs 0/9/3: middle drawer,
wine bottle on rack, and top drawer plus bowl. Consequently, the previous task
1/2 rollouts used correct policy prompts but measured the success predicates of
unrelated environments; those success numbers and videos were invalid.

Demonstration selection and training are unaffected because episodes were
selected by exact instruction text. All zero-shot conditions and adapted task
1/2 evaluations were rerun in place with the original weights, seeds, episode
count, and inference settings. Evaluators now record both logical and
environment IDs and assert the environment's real `task_description` before
overriding the policy prompt. The superseded files remain recoverable from git
history and from the local correction backup under `/var/tmp`.
