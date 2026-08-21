# Findings (written 2026-08-20 after 20/20 points, determinism smoke PASSED)

Mean delta -0.15; McNemar p<0.05 on 2/10 pairs. Weak-anchor rule R4ad
excludes the alphabet-soup pair (exact 2/20), leaving 2/9 significant.

- **Verdict: R3ad (intermediate).** Deleting the first article is harmless
  for 7/10 tasks (deltas 0..-0.1) but collapses two: "put (the) black bowl
  on top of the cabinet" 19/20 -> 8/20 and "open (the) microwave"
  19/20 -> 8/20 (both p=0.001).
- Function-word sensitivity is TASK-specific, not universal: the selector is
  not keyed to the exact token sequence globally, but some tasks sit near a
  brittle boundary. Notably bowl-on-cabinet is the most string-fragile task
  across every probe (v2 goal-phrase -0.85, here -0.55, paraphrase -0.95).
- For relabeling: article/function-word variants are cheap and mostly safe
  to add, but they alone do not explain the v2 collapse (content tokens do).
