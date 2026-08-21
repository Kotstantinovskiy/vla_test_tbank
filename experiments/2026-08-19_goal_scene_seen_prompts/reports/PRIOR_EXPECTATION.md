# Prior-informed expectations and pre-registered decision rules (probe 3)

Recorded 2026-08-19 before running any rollout of this experiment.

Known priors: k=0 true prompts in the goal scene = 1/200 (prompt_only, exact
replication across two runs; the single success is task 4 episode 4). In SEEN
scenes (v2): trained strings execute at native rate even under cross prompts;
truthful paraphrases collapse to 0-2/20; a related-but-novel string triggers
the scene's trained skill (15/20).

## Predictions

1. **true block replicates prompt_only per episode** (same checkpoint, seed
   bank, batch size): envs 3/8/9/7 -> 0/20; env 4 -> 1/20 with the success at
   episode 4. Any deviation = harness difference to investigate before
   interpretation (rule R4p3).
2. **seen_twin — the decisive cells**: predicted ≈ true (each within +-2 of
   its true twin; no McNemar-significant gain). Rationale: the anchor already
   shows a verbatim trained string failing in the goal scene (0/20), so the
   scene side is binding and prompt wording should not rescue k=0.
   Lower-confidence alternative (would be a big result): seen_twin > true by
   >= 0.15 on >= 2 envs -> the string side matters even at k=0.
3. **Behavior separates engagement from idling**: under true and seen_twin
   the model approaches the correct object (median min eef->target distance
   well below nonsense); under nonsense it stays far/idle. Quantitative
   guess: median min dist <= 0.15 m for true/seen_twin vs >= 0.25 m for
   nonsense on the same envs.
4. **seen_cross**: prompted-skill success ~0 (novel scene blocks execution
   regardless of string); behavioral pull toward the PROMPTED object rather
   than the env's object would still indicate selector engagement.

## Decision rules (fixed before results)

- R1p3 (**string side contributes at k=0**): seen_twin - true >= +0.15 with
  McNemar p < 0.05 on >= 2 envs -> prompt relabeling helps even zero-shot;
  add inference-time prompt relabeling as a free baseline improvement and
  predict k=0 gains for the relabeling method.
- R2p3 (**scene-side execution failure, selector engaged**): success ~0
  everywhere but true/seen_twin approach distances are significantly smaller
  than nonsense (paired within env) -> the selector engages the right object
  and execution breaks mid-skill; retrieval co-training (visual variety) is
  the complementary lever to relabeling, and low-k gains should come from
  vision-side adaptation more than language-side.
- R3p3 (**selector scene-gated**): seen_twin behavior ≈ nonsense (no
  differential approach) -> in a novel scene the language selector does not
  even engage; relabeling alone cannot help k=0 (only k>=1); scene-side
  co-training becomes the primary Task-2 ingredient.
- R4p3 (**harness check**): true block failing to replicate prompt_only
  per-episode outcomes, or any consistency violation -> fix measurement
  before interpreting.
