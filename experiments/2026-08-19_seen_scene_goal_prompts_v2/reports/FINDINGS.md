# Findings and verdict against pre-registered rules (v2)

Written 2026-08-19 after the 16-point run completed (320/320 episodes, no
failures, 0 predicate-consistency violations, and per-episode env-metric
replication of v1 on all 14 shared points). Tables: [REPORT.md](REPORT.md);
rules: [PRIOR_EXPECTATION.md](PRIOR_EXPECTATION.md).

## Headline: instruction following is near-ceiling for trained strings

The v1 video read is now a number. Success below = the PROMPTED task's
predicate:

| env | prompt | prompted succ | env-task succ |
|---|---|---:|---:|
| stove env ("turn on the stove") | "put the frying pan on the stove" | **17/20** | 0/20 |
| pan env ("put the frying pan on the stove") | "turn on the stove" | **20/20** | 0/20 |

The prompted skill executes at (or above) its native trained rate: pan
placement 17/20 under cross vs 16/20 in its own env; stove-on 20/20 in both.
Median first-success step: 79 (stove-on), 173 (pan placement). **R1v2 fires
at maximum strength** — language causally and cleanly selects among trained
skills; the v1 "0/20 cross collapse" was purely the metric, not the policy.

## The novel-string point (goal 0) — R3v2 does not fire, with a twist

"open the middle drawer of the cabinet" (never trained anywhere) in the
SCENE10 env (trained: "close the top drawer of the cabinet"):

- prompted predicate (middle drawer open): **0/20** — the exact-string
  selector model stands; a novel string retrieves no new skill.
- env-task predicate (top drawer closed): **15/20** — under the novel-but-
  related prompt the model largely fell back to the scene's trained skill
  (its trained rate there: 20/20).

Contrast with nonsense ("dax florp"): env-task 0/20 in both probe scenes.
So the fallback hierarchy is graded: trained string -> trained skill;
related-but-novel string -> scene's trained skill, degraded; nonsense ->
nothing. Note the tension with the paraphrase block, where near-synonyms of
the trained string did NOT trigger the trained skill (0-2/20): dropping a
word from the trained instruction breaks execution, while a structurally
similar sentence about the same fixture ("...the X drawer of the cabinet")
keeps the scene skill running. Hypotheses (not adjudicated here): token-level
prefix/structure overlap matters more than synonymy; or the drawer-close
skill is partly affordance-driven (drawer starts open). The failure-funnel
probe can separate these.

## Paraphrase and nonsense under the redefined metric

Unchanged from v1 (predicates coincide): 0/20, 2/20, 2/20 and 1/20 (the
drawer paraphrase scored 1 episode on the goal-3 predicate while the env's
stricter open+in conjunction stayed 0). Nonsense 0/20. The surface-form
brittleness verdict (v1 R2) stands.

## Rules

- **R1v2 — FIRES** (cross prompted 0.85 and 1.00, both >= 0.3, one >= 0.5).
- **R2v2 — does not fire** (no measurement problem; predictions met).
- **R3v2 — does not fire** (goal-0 prompted 0/20 < 0.3).
- **R4v2 — clean** (0 consistency violations; exact v1 replication).

## Implications for Task 2

1. The action expert and the language selector are both healthy; the ONLY
   broken link is string-to-skill generalization. This is the strongest
   possible justification for **hindsight instruction relabeling /
   augmentation**: teach the selector that many strings map to one skill.
2. Co-training must preserve the working selector: mixed-task batches with
   the original strings kept alongside augmented ones.
3. The goal-0 fallback (15/20 scene skill under a related novel string) warns
   that in the goal scene, novel instructions may trigger the *nearest seen
   skill* rather than nothing — relevant when reading k=0/low-k failures.
4. Retrieval caveat from the slice: goal tasks 1/2/5/6 have no seen scene
   even containing their predicate objects together (e.g. bowl+stove never
   co-occur in libero_90) — retrieval for those tasks can only supply partial
   skills (approach/grasp/place primitives), not end-to-end demonstrations.

## Limitations

- Env-task success after a prompted-success termination is unobservable;
  immaterial here (v1 env metric was 0/20 on cross, replicated).
- One novel-string point only (goal 0); the graded-fallback claim rests on
  n=1 env plus two nonsense controls.
- Single eval seed bank; frozen checkpoint (no training seeds involved).
