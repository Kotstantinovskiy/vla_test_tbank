# Prior expectation — locked before new rollouts

The three verbatim goal strings are expected to have low prompted-predicate
success in the compatible seen hosts. Goal 0 already had 0/20 prompted success
in a related predecessor experiment, while the host skill still fired. The
same fallback-to-host-skill pattern is plausible for goals 1 and 2, but their
semantic object mappings make them new measurements rather than replications.

Decision rule: report prompted success and native host success side by side;
do not infer instruction following from motion or from native host success.

## Quantitative predictions and decision rules (added 2026-08-20, still before any rollout)

Priors: v2's goal-0 point in the SAME SCENE10 host: prompted 0/20, host-skill
fallback 15/20. Hosts for goals 1/2 are new measurements (semantic object
mapping: goal "bowl" -> white_bowl_1, goal "cabinet" -> white_cabinet_1).

Predictions:
1. seen controls: >= 0.7 on all three hosts (seen tasks).
2. goal prompted success: goal 0 <= 2/20 (near-replication of v2 under the
   new noise protocol); goals 1/2 <= 3/20 each (exact-string selector: novel
   strings retrieve no skill).
3. host-skill fallback under goal prompts (env metric): >= 0.4 for goal 0
   (v2 saw 0.75); uncertain for goals 1/2 (string-similarity to the host
   instruction is lower).

Decision rules:
- R1gh (exact-string model extended): all goal prompted <= 2/20 with seen
  controls >= 0.7 -> novel goal strings retrieve nothing even with
  compatible objects present; relabeling remains a training-time lever only.
- R2gh (novel string CAN drive a skill): any goal prompted >= 6/20 ->
  refutes the strict exact-string selector; language generalization is
  partially present zero-shot; reweigh relabeling upward for k=0 claims.
- R3gh (graded fallback generalizes): host env-task success under goal
  prompts >= 0.5 on >= 2 hosts -> v2's related-string fallback is a general
  phenomenon; low-k failure videos must not be read as instruction
  understanding.
- R4gh (unhealthy host): any seen control < 0.5 -> that host's rows are
  reported but excluded from rule counting.
