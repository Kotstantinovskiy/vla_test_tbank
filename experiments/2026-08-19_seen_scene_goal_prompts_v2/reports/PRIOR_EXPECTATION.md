# Prior-informed expectations and pre-registered decision rules (v2)

Recorded 2026-08-19 before running any rollout of this experiment. v1
(2026-08-18_seen_scene_goal_prompts) results ARE known and inform these
priors: trained 11-20/20; paraphrases 0-2/20; cross/absent/nonsense 0/20 on
the env metric; videos suggested the cross prompts get EXECUTED. v2 turns
that video read into a number (prompted-task predicate) and adds the full
goal-prompt slice.

## Predictions

1. **Replication**: identical seeds/init-state machinery -> trained points
   reproduce v1 exactly (17, 12, 11, 14, 20, 16 of 20); paraphrase points
   reproduce v1 on the env metric (0, 2, 2, 0) and stay within +-1 of it on
   the prompted metric (predicates identical or near-identical); nonsense
   0/20. consistency_violations = 0 everywhere it is asserted.
2. **cross, prompted metric — the decisive cells**:
   - prompt "turn on the stove" in the frying-pan env: >= 0.7 (v1 videos
     showed the burner turning red).
   - prompt "put the frying pan on the stove" in the stove env: 0.3-0.8
     (grasp+place is harder than knob-turning; v1 video showed the pan
     moved).
3. **goal slice**: goals 3/4/8/9 alias the paraphrase points (0-2/20); goal 7
   aliases trained turn-on-stove (20/20); goal 1/2/5/6 skipped (no evaluable
   scene). The one NEW point — goal 0 "open the middle drawer of the
   cabinet", a string never trained anywhere, in the KITCHEN_SCENE10
   close-top-drawer env — predicted <= 3/20 under the exact-string-selector
   model from v1.
4. **trained baseline of the new SCENE10 env** ("close the top drawer of the
   cabinet"): >= 0.7 (a seen task).

## Decision rules (fixed before results)

- R1v2 (**instruction following confirmed**): cross prompted success >= 0.5
  in at least one direction AND >= 0.3 in both -> language causally selects
  among trained skills; v1's R2 verdict stands with quantitative backing;
  Task-2 co-training must preserve the selector (mixed-task batches), and
  hindsight relabeling attacks the string brittleness.
- R2v2 (**measurement or interpretation problem**): cross prompted success
  < 0.3 in both directions -> the v1 video read was over-optimistic or the
  predicate/termination machinery is wrong; audit videos vs predicate logs
  before any Task-2 claim.
- R3v2 (**exact-string model too strong**): goal-0 point >= 0.3 -> a novel
  string CAN drive a coherent skill; downgrade pure string-brittleness,
  reweigh toward scene/embodiment factors and retrieval.
- R4v2 (**measurement bug**): any consistency violation, or any trained point
  deviating from v1 -> fix the harness before interpreting anything.
