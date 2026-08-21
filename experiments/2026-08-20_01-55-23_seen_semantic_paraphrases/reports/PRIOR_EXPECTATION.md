# Prior expectation — locked before rollouts

Earlier less-controlled goal-style rewrites reduced four seen tasks from
11–17/20 to 0–2/20. The new paraphrases preserve every task-defining object
descriptor and relation, so a smaller but still negative average delta is
expected. Results will be reported for all ten preregistered tasks, including
anchors whose exact baseline is weak; no post-result task filtering.

## Quantitative predictions and decision rules (added 2026-08-20, still before any rollout)

Priors: v2's goal-style rewrites dropped descriptors and collapsed
(-0.45..-0.85 on 4/4 pairs). These 10 paraphrases preserve every
task-defining descriptor and relation.

Predictions:
1. exact anchors: >= 0.5 on >= 8/10 tasks.
2. paraphrase mean delta in [-0.5, -0.15]; largest drops where the verb or
   token order changes most ("switch the stove on", "shut the cabinet's top
   drawer"), smallest for single-verb swaps ("put" -> "place").

Decision rules:
- R1sp (descriptor hypothesis wins): mean delta >= -0.15 and McNemar n.s.
  on >= 8/10 -> v1/v2 collapse was about LOSING task-defining tokens;
  descriptor-preserving rewording is safe; relabeling should prioritize
  descriptor variants over verb variants.
- R2sp (any-rewording brittleness): mean delta <= -0.4 or McNemar p<0.05 on
  >= 5/10 -> the selector keys on the whole token sequence; relabeling needs
  broad paraphrase coverage including verb/structure variants.
- R3sp (intermediate): correlate per-pair drop with edit type (verb swap /
  reorder / possessive restructure) and report; no single-lever claim.
- R4sp (weak anchor): any exact anchor < 7/20 -> its pair stays in tables
  but is excluded from rule counting (pre-registered interpretation rule).
