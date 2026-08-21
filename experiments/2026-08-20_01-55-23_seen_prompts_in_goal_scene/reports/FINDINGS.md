# Findings (written 2026-08-20 after 11/11 points, determinism smoke PASSED)

Pooled seen-prompt success **18/180** — R1sg's floor (<=5/180) is REFUTED;
**R2sg fires: scene transfer is skill-dependent**, correcting probe 3's
"the novel scene blocks everything":

- "open the top drawer of the cabinet": **10/20** in the goal scene.
- "close the top drawer ... and put the black bowl on top of it": **7/20**
  (note: the close-conjunct is trivially true at reset — drawers start
  closed; the reset guard checks the full conjunction, so the effective
  predicate here is bowl-on-top-of-cabinet. Read it as that skill
  transferring at 7/20).
- the other seven prompts: 0-1/20; controls clean (true_goal 0/20 = the
  prompt_only_3 floor; nonsense 0/20).

Caveats before strong claims: (a) top-vs-middle confound — the host's own
instruction ("open the MIDDLE drawer") scored 0/20 while "open the TOP
drawer" scored 10/20; whether the model opens the top drawer under OTHER
prompts too (a scene default rather than instruction following) needs the
stored videos / auxiliary predicates; (b) single init bank (goal task 0).
Contrast with probe 3: the same bowl-on-cabinet string scored 0/20 there
under a different init bank and stream noise — transfer is also
init/noise-sensitive.

Implication: zero-shot in the goal scene is NOT uniformly scene-blocked;
drawer/bowl primitives do fire from verbatim seen strings under the right
inits. Retrieval co-training has more to work with than probe 3 suggested.
