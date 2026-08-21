# Prior expectation — locked before rollouts

Five related seen strings already failed to unlock the novel goal scene in a
predecessor probe. The nine-point deterministic extension is therefore
expected to remain near floor. A non-zero result under a verbatim trained
string is still meaningful because success is evaluated with that string's
own BDDL predicate rather than the host task's predicate.

## Quantitative predictions and decision rules (added 2026-08-20, still before any rollout)

Priors: probe 3 (goal_scene_seen_prompts): 4 seen twins + 2 seen_cross in the
goal scene all 0/20 by prompted predicate; true anchor "turn on the stove"
20/20 seen -> 0/20 goal. This experiment widens to 9 verbatim seen strings in
ONE goal env (goal task 0) with their own predicates, shared inits and
noise-paired episodes.

Predictions:
1. true_goal control (host's own instruction "open the middle drawer"):
   0-2/20 (k=0 floor; prompt_only_3 gave 0/20 for this task).
2. seen_prompt points: 0-2/20 each; pooled <= 5/180. Most likely nonzero
   candidate: "turn on the stove" (the stove IS present in the goal scene
   and the skill is the simplest), still predicted <= 2/20 given probe 3's
   anchor result of 0/20.
3. nonsense control: 0/20 prompted-undefined; env metric 0-2/20.

Decision rules:
- R1sg (scene-gating confirmed at scale): pooled seen_prompt success
  <= 5/180 -> the novel scene blocks skill execution regardless of string;
  closes the question with n=9 instead of probe 3's n=6.
- R2sg (some skills DO transfer): any single seen_prompt >= 5/20 -> scene
  transfer is skill-dependent; identify what distinguishes the transferring
  skill (object overlap, motion type) before Task-2 design freezes.
- R3sg (control failure): true_goal >= 5/20 would contradict prompt_only_3's
  floor under the new noise bank at this scale — investigate the harness
  before interpreting anything else.
