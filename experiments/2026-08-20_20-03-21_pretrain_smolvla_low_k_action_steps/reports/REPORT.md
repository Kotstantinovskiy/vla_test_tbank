# Low-k action-step sweep on the official-data pretrain

All action-step conditions for a task/budget use the same adapted
checkpoint and the same per-episode environment/noise/init-state bank.
Only binary success is aggregated; all rollout videos remain on disk.

| task set | k | n=1 | n=10 | n=25 | prior n=50* |
|---|---:|---:|---:|---:|---:|
| tasks 0–2 | 1 | 0.333 | 0.567 | 0.567 | 0.567 |
| tasks 0–2 | 2 | 0.467 | 0.800 | 0.767 | 0.717 |
| tasks 0–2 | 3 | 0.650 | 0.717 | 0.733 | 0.717 |

*The previous n=50 result is descriptive only: it used batch=4
and one process RNG stream, whereas this experiment uses batch=1
and explicit per-episode policy-noise seeds. A claimed improvement
must be confirmed with a paired n=50 rerun and a second training seed.
