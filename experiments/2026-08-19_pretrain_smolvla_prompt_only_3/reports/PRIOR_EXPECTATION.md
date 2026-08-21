# Prior-informed expectations and decision rules (prompt-only v3)

Recorded 2026-08-19 before running any rollout of this experiment.

Priors: _2 (stream noise, one layout) gave true 1/200 (only task 4 episode 4),
wrong 0/200, nonsense 0/200. A fresh-process probe-3 smoke of task 4 true
gave 5/20 under a different noise stream. Same env seeds and init states in
all cases.

## Predictions

1. **Determinism smoke passes**: true__task_4 evaluated in two different
   process layouts produces IDENTICAL per-episode outcomes (this is the whole
   point of the protocol change). Failure = the reseed hook does not actually
   pin the sampling noise -> fix before running anything else.
2. **true**: pooled 2-8/200. Task 4 is the only task expected clearly above
   zero (1-6/20, consistent with rate ~0.1-0.25 seen across streams); tasks
   0-3 and 5-9 each 0-1/20. Note _3's noise bank differs from both previous
   streams, so per-episode outcomes need not match either.
3. **wrong**: 0-2/200 (floor; wrong instructions gave 0/200 in _2).
4. **nonsense**: 0-2/200 (floor in _2; the goal-scene "reach into cluster"
   default seen in probe 3 does not produce predicate successes).
5. Batch-size change (4 -> 1) does not change success rates beyond noise
   (env dynamics are per-episode; only numeric batching of the policy
   forward differs).

## Decision rules

- R1po3: determinism smoke fails -> stop, fix seeding, rerun smoke; nothing
  else is interpretable.
- R2po3: true pooled > 15/200 or any single task >= 8/20 -> _2's k=0 was a
  substantially unlucky stream; flag the cost curve's k=0 reference for
  revision (with both numbers reported side by side).
- R3po3: results within prediction 2-4 ranges -> _3 becomes the canonical
  k=0 = its pooled true rate; curve reports cite _2 and _3 with the noise
  note; no re-labeling of prior conclusions needed (all were CI-compatible).
