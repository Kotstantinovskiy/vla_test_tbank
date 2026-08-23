# Prior expectation — recorded before preparation and any rollout

Recorded 2026-08-22, before preparation, training, or any target rollout of
this experiment. Training seed is fixed at 1000, matching the deterministic
low-k reproduction.

Reference (expert-only, same seed/demos/evaluation protocol,
`2026-08-21_20-37-56_pretrain_smolvla_low_k_deterministic_repro`):

| task | k=1 | k=2 | k=3 |
|---:|---:|---:|---:|
| 0 (drawer) | 3/20 (0.15) | 6/20 (0.30) | 7/20 (0.35) |
| 1 (bowl→stove) | 17/20 (0.85) | 17/20 (0.85) | 17/20 (0.85) |
| 2 (wine→cabinet) | 16/20 (0.80) | 18/20 (0.90) | 16/20 (0.80) |
| mean 0–2 | 0.600 | 0.683 | 0.667 |

Predictions for the full fine-tune (whole policy trainable, same recipe
otherwise):

1. The curve stays within ±0.15 of the expert-only means at each budget; I do
   not expect a dramatic shift in either direction from 2000 steps on 1–3
   demos.
2. Direction: at k=1 full FT is at or slightly below expert-only (more
   capacity fitting one demonstration raises overfitting/forgetting risk, and
   the frozen VLM is what the expert-only baseline relied on); at k=2–3 full
   FT is at or slightly above expert-only. Concretely: mean(k=1) within
   [0.45, 0.65], mean(k=3) within [0.62, 0.82].
3. Task 0 (drawer) remains the weakest task at every budget and shows the
   largest relative movement, because its expert-only scores leave the most
   headroom in both directions.
4. Training loss at step 2000 is lower than the expert-only counterpart for
   every task/budget (strictly more capacity on the same tiny dataset); this
   is a sanity expectation, not a success claim.
5. The forward/reverse determinism gate on task 0 / k=1 passes exactly, as it
   did for the expert-only repro; the gate is protocol-level and does not
   depend on the trainable set.

## Addendum: inference n_action_steps ∈ {50, 25} (recorded 2026-08-22, before any full-FT rollout)

Added before any rollout of this experiment was evaluated (the first launch
was aborted at training step 30 of the production gate; no evaluation ran).
Every adapted checkpoint will be evaluated at inference n_action_steps=50 and
25 on identical per-episode seeds/init states. Reference on the expert-only
checkpoints (`2026-08-20_20-03-21_pretrain_smolvla_low_k_action_steps`,
n=25 vs the n=50 repro): 0.567/0.767/0.733 vs 0.600/0.683/0.667 at k=1/2/3 —
re-planning twice per chunk helped at k=2–3 and slightly hurt at k=1.

Predictions 6–7:

6. The same qualitative pattern holds for full-FT checkpoints: n=25 within
   ±0.10 of n=50 at every budget, with n=25 ≥ n=50 more often at k=2–3 than
   at k=1.
7. Points 1–3 above are stated for the n=50 (trained-default) curve; the
   determinism gate must pass exactly at both variants.

Falsification handling: if a budget mean differs from the expert-only mean by
more than 0.15, first inspect training logs (divergence/NaN, loss scale),
the parameter audit, and a sample of rollout videos before assigning a
scientific interpretation. This experiment is single-training-seed (1000) and
must be labeled as such; it does not by itself satisfy the assignment's
two-seed requirement.
