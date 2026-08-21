# Findings and verdict against pre-registered rules (probe 3)

Written 2026-08-19 after the 13-point run completed (260/260 episodes, no
failures, 0 predicate-consistency violations). Tables: [REPORT.md](REPORT.md);
rules: [PRIOR_EXPECTATION.md](PRIOR_EXPECTATION.md); the harness discovery
made during the smoke (policy sampling noise is not pinned by episode seeds)
is documented separately in [HARNESS_NOTE.md](HARNESS_NOTE.md).

## Headline: the SCENE is the binding constraint at k=0

Verbatim trained strings do NOT rescue zero-shot in the goal scene:

| env (goal task) | true prompt | seen-twin (trained string) |
|---|---:|---:|
| turn on the stove (anchor: true IS trained) | 0/20 | — |
| open top drawer + bowl inside | 0/20 | 0/20 |
| put the bowl on top of the cabinet | **5/20** | **0/20** |
| put the bowl on the plate | 0/20 | 0/20 |
| put the wine bottle on the rack | 0/20 | 0/20 |
| seen_cross (both directions) | — | 0/20 |

- **R1p3 (string side helps at k=0) — REFUTED decisively.** seen_twin never
  beats true; on task 4 the direction is *reversed* (5/20 -> 0/20 on paired
  inits, McNemar p = 0.0625, n.s.). Inference-time prompt relabeling will NOT
  unlock zero-shot; language-side methods act only through training (k >= 1).
- The anchor nails it alone: "turn on the stove" is literally a trained
  string that executes 20/20 in its seen scene (v2) and 0/20 here. Same
  string, same skill, new scene -> failure. Scene appearance gates skill
  execution.

## Behavioral metrics: engagement without completion, plus a confound

- Under true prompts on the bowl tasks the model **engages**: median closest
  eef->bowl distance 0.05-0.07 m, bowl displaced >5 cm in 10-11/20 episodes
  (videos: arm descends into the object cluster, manipulates around the bowl,
  never completes the place). Failure is mid-funnel (grasp/place), not
  "no attempt".
- **The nonsense control did not idle in the goal scene** — in the drawer env
  it approached to 0.057 m and moved the bowl in 14/20 episodes (videos show
  the same reach-into-cluster motion). In seen scenes (v1/v2) nonsense
  produced idling; in the novel cluttered scene the model has a
  reach-toward-cluster default. Consequently R2p3's "engagement is
  language-specific" premise holds only on the stove env (true 0.213 m vs
  nonsense 0.317 m); on the drawer env the distance metric cannot separate
  language-driven from default engagement.
- Verdict between R2p3/R3p3: neither fires cleanly. The honest summary is
  stronger and simpler: **in the novel scene the prompt string has little
  measurable effect at all** — success ~0 for every string (trained, goal,
  nonsense), and object-directed motion appears under every string as well.
  Scene dominates both selection and execution at k=0.

## Secondary observations

1. true task 4 = 5/20 (this point IS the smoke run; see HARNESS_NOTE — under
   prompt_only's process layout the same condition gave 1/20; k=0 point
   estimates carry unpinned sampling variance).
2. true > seen_twin on task 4 (5-0) inverts the seen-scene pattern (where the
   trained string always wins). Hypothesis for later: the policy conditions
   on a JOINT scene-language embedding, not a pure string key; a goal-styled
   string may sit closer to the goal scene's visual context. n=1 env, noise
   caveat applies — recorded as a hypothesis, not a claim.
3. Together with v2: the selector story is now two-sided — in seen scenes
   language cleanly selects skills (17-20/20 cross); in a novel scene no
   string reaches any skill. Zero-shot failure is scene-side; string
   brittleness (v1/v2 paraphrase collapse) matters for how fast language
   generalizes DURING adaptation, not for k=0.

## Implications for Task 2

1. **k=0 cannot be fixed from the language side** — drop any plan to claim
   zero-shot gains from relabeling; its predicted value is at k >= 1
   (learning "many strings -> one skill" during fine-tuning).
2. **Scene-side variety is the complementary ingredient**: retrieval
   co-training should prioritize *visual* diversity of retrieved demos;
   behavioral evidence (mid-funnel failure) suggests grasp/place execution in
   the new visual context is what breaks — exactly what a failure-funnel
   probe can quantify per stage (next diagnostic).
3. The reach-into-cluster default warns that low-k failure videos in the goal
   scene are not evidence of instruction understanding by themselves.

## Limitations

- Sampling-noise caveat for all point estimates (HARNESS_NOTE); paired
  McNemar rests on shared env init states as pre-registered.
- Behavior target is one object per env; drawer-task funnel is misordered
  for this metric (the correct first subgoal is the drawer, not the bowl).
- Single novel scene (libero_goal is one kitchen); scene-dominance is shown
  for this scene, not scenes in general.
