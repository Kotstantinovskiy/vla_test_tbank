# Bonus B ranking result

Only `n_action_steps=50` is included. Both critics scored the same four uniformly sampled frames from each of 420 rollout videos before environment labels were joined. The 420 predictions per critic are protected by the hashes in `artifacts/scoring_complete.json`.

## Primary result

| Critic | Macro Spearman | 95% episode-bootstrap CI | Macro Kendall | Top-set accuracy | Mean top regret |
|---|---:|---:|---:|---:|---:|
| own Qwen3.5 critic | 0.294 | [0.072, 0.602] | 0.245 | 0.333 | 0.200 |
| Robometer-4B-LIBERO | 0.300 | [0.129, 0.590] | 0.284 | 0.667 | 0.333 |

Robometer wins the preregistered macro-Spearman point estimate by only 0.006 (0.300 versus 0.294), but this is **not a convincing difference**: the bootstrap 95% interval for own minus Robometer is [-0.379, 0.328], and the bootstrap probability that the own critic is higher is 0.481. The honest conclusion on three tasks is therefore no statistically supported winner.

## Task dependence

| Task | Own Spearman | Robometer Spearman | Own top pick (true success) | Robometer top pick (true success) |
|---:|---:|---:|---|---|
| 0 | -0.071 | 0.893 | bundle_k_1 (0.35) | bundle_k_5 (0.90) |
| 1 | 0.808 | -0.611 | bundle_k_25 (0.95) | pretrain (0.00) |
| 2 | 0.145 | 0.618 | bundle_k_10 (0.95) | bundle_k_10 (0.95) |

Robometer is excellent on drawer and wine-bottle ranking but reverses much of the bowl ranking and selects the zero-success pretrain there. The own critic is strong on bowl, nearly uninformative on wine bottle, and reversed on drawer. Accordingly, Robometer gets the true top set on 2/3 tasks but has worse mean top regret (0.333 versus 0.200) because its bowl miss costs a full success point.

The pooled cross-task correlation is secondary because the two reward models have task-dependent score calibration: it favors the own critic (Spearman 0.491) over Robometer (-0.187), while the preregistered within-task macro metric is essentially tied.

## Limitations

- Only three target tasks and seven candidates per task are ranked; 20 rollout episodes make the true success estimates discrete and noisy.
- Robometer-4B-LIBERO was trained on LIBERO suites including Goal, while the own critic was trained only on expert LIBERO-90 video. This is the requested ready-foundation comparison, not a held-out-domain comparison.
- Bundle rollouts use per-episode deterministic batch-1 noise. Pretrain rollouts use the same seeds and init-state IDs but were generated in batch 4, so episode indices are not a strictly paired policy-noise comparison.
- This experiment evaluates reward-free ranking only. It does not optimize either policy against the learned signal and therefore does not answer the reward-hacking part of Bonus B.

Machine-readable details are in `results/summary/metrics.json`, `candidate_scores.csv`, and `episode_scores.csv`.
