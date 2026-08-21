# Predictions (recorded 2026-08-18, before any training or evaluation run)

Prior-informed: the crislmfroes-lineage naive baseline
(`2026-08-16_self_smolvla_naive_learn_baseline`, same recipe: expert-only,
2000 steps, batch 32, 20 eval episodes) reached per-task means of ~0.86 over
all 10 goal tasks and 0.900/0.917/0.967 on tasks 0-2 at k=5/10/25, despite
its train/eval mirror mismatch and normalizer swap across differing
conversion pipelines.

For this run on the official-data pretrain (all conventions matched,
seen-control 20/20, demos = official demo_0..demo_{k-1}):

1. **Mean over 10 tasks**: k=5 >= 0.80, k=10 >= 0.85, k=25 >= 0.88, i.e. at
   or slightly above the old baseline; monotone in k on average, with
   plausible per-task non-monotonicity at 20-episode resolution (+-0.1).
2. **Tasks 0-2 (assignment)**: comparable to or above 0.90/0.92/0.97.
3. **Hardest tasks**: expected laggards are 3 (two-stage drawer+bowl) and 5
   ("push", verb unseen in pretraining); prediction: they still exceed 0.5
   at k=25.
4. **Where the matched conventions could show**: if the mirror/domain gap was
   consuming adaptation capacity, k=5 should benefit most (fewer
   demonstrations needed to overwrite the mismatch); a k=5 mean clearly above
   0.90 over 10 tasks would support that.
5. The k=0 -> k=5 jump (0.005 -> ~0.8) remains partially attributable to the
   normalizer swap; this run cannot separate that effect (disclosed in
   protocol.yaml), only a stats-pinned ablation could.

Falsifiers/surprises: any task at 0/20 for all budgets (would suggest a
per-task pipeline defect rather than difficulty); mean at k=25 below the old
baseline's 0.86 (would suggest the official conversion hurt adaptation).
