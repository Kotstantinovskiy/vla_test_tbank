# Harness note: policy sampling noise is NOT pinned by episode seeds

Written 2026-08-19 after the smoke point and BEFORE launching the remaining
points. Pre-registered prediction 1 (per-episode replication of prompt_only's
task-4 outcomes) FAILED, and rule R4p3 required diagnosing before
interpretation. Diagnosis:

## Evidence

- Smoke `true__put_the_bowl_on_top_of_the_cabinet` (goal task 4, fresh
  process): 5/20, successes at episodes 0/3/11/14/18.
- prompt_only_2 task 4 (same checkpoint, same episode seeds 1000-1019, same
  init states, same batch size): 1/20, success at episode 4 only — itself an
  EXACT replication of prompt_only across two runs.
- lerobot's eval seeds torch once per process (`set_seed`); episode seeds are
  passed only to `env.reset`. SmolVLA's flow-matching action sampling draws
  from the GLOBAL torch RNG stream.
- prompt_only(_2) evaluated tasks 0..9 sequentially in ONE process, so its
  RNG stream position at task 4 reflected ~80 prior episodes; the smoke ran
  task 4 first. Same env randomness, different action noise -> different
  trajectories.

## Implications

1. Rollouts are deterministic only for an identical process layout (that is
   what prompt_only vs prompt_only_2 and v1-vs-v2 replications established —
   all replicated point runs had identical per-process evaluation order).
2. Point estimates such as the cost curve's k=0 = 0.005 carry unpinned
   policy-sampling variance: goal task 4 alone is 1/20 under one noise stream
   and 5/20 under another (Wilson CIs overlap; both consistent with a true
   rate around 0.1). Success-rate CIs remain the honest uncertainty
   statement; per-episode "exact determinism" claims must be qualified by
   process layout.
3. Within THIS experiment every point runs as its own fresh process
   (orchestrator spawns one process per label), so all points share the RNG
   stream position at episode 0 and the paired analysis rests, as
   pre-registered, on shared env seeds/init states; action-noise alignment
   across conditions degrades after episode 0 and is NOT claimed.
4. Prediction 1 of PRIOR_EXPECTATION is therefore VOID as a harness check
   (its premise — that episode seeds pin the whole rollout — was wrong);
   the predicate/consistency machinery remains validated by v2 (0 violations,
   exact replication under identical layout). Predictions 2-4 and rules
   R1p3-R3p3 stand unchanged.

Follow-up recorded for the backlog: quantify eval-noise variance explicitly
(rerun one condition under several RNG streams) and fold it into the cost
curve's error bars alongside the required second training seed.
