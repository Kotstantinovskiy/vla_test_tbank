# Prior expectation — locked before rollouts

Removing one article is expected to hurt less than a full semantic paraphrase,
but the exact-string sensitivity seen in prior probes makes a measurable
negative delta plausible. The transformation is uniform and all ten tasks are
reported regardless of exact-baseline strength.

## Quantitative predictions and decision rules (added 2026-08-20, still before any rollout)

Priors: v2 paraphrases that DROPPED a task-defining descriptor collapsed
(-0.45..-0.85, McNemar p<0.05 on 4/4). Deleting one article removes no
descriptor and no relation — the minimal possible string edit.

Predictions:
1. exact anchors: >= 0.5 on >= 8/10 tasks (all are seen tasks).
2. article_drop mean delta in [-0.35, -0.05]; smaller than the v2
   descriptor-dropping deltas on the shared envs.

Decision rules:
- R1ad (token-level brittleness): mean delta <= -0.4 or McNemar p<0.05 on
  >= 5/10 pairs -> even a contentless single-token edit breaks retrieval;
  the selector keys on near-exact token sequences; relabeling must include
  trivial surface variants (articles, function words).
- R2ad (robust to function words): mean delta >= -0.1 and McNemar n.s. on
  >= 8/10 -> the v1/v2 collapse came from content tokens, not string
  identity; relabeling can focus on content-word variants.
- R3ad (intermediate): report per-pair deltas; no single-lever claim.
- R4ad (weak anchor): any exact anchor < 7/20 stays in the tables but its
  pair is excluded from rule counting (pre-registered interpretation rule,
  not data filtering).
