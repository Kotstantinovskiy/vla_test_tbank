# Protocol deviations

This file is append-only after the prediction lock. Record a deviation before
looking at the result produced under the changed protocol.

## 2026-08-15 — pin last internally consistent target-data revision

Before target training, preparation against the then-current dataset revision
`86958911c0f959db2bbbdb107eb3e17c5f9c798e` failed. Its episode metadata maps
episode 399 to `data/chunk-000/file-025.parquet`, while that shard physically
contains only episodes 61–63. The same mismatch occurs throughout the target
range. Revision `9176d427966503c81ac9f8f96502e50861a15ee7` maps episode 399 to
file 149 and has 377 metadata file IDs matching the 377 physical data shards.

All target-data operations are therefore pinned to revision `9176d427...`, the
latest internally consistent revision before the metadata rewrite. Repository,
episodes, ordering, task instructions, and all optimization/evaluation settings
are unchanged. No target training or success rollout had run when this deviation
was recorded.
