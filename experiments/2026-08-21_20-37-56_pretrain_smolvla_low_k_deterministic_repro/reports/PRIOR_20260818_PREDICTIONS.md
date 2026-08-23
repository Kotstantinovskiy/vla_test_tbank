# Predictions (recorded 2026-08-18, before any training or evaluation run)

Prior-informed: the sibling baseline
(`2026-08-18_pretrain_smolvla_naive_baseline_few_shot_tune`, byte-identical
recipe) measured, at the time of writing, k=5 mean 0.830 and k=10 mean 0.875
over all ten tasks (0.95 on tasks 0-2 at both budgets), with laggards on
tasks 3, 5, 6. This experiment adds k=1/2/3 because the baseline saturates by
k=5 on the assignment tasks (amendment's "ceiling" clause).

1. **Monotone rise with visible spread** — unlike k=5..25, the low-k region
   should finally separate regimes: predicted means over all ten tasks
   k=1: 0.25-0.55, k=2: 0.45-0.70, k=3: 0.60-0.78 (vs 0.83 at k=5).
2. **Tasks 0-2 (assignment)**: higher than the 10-task mean at every k
   (their sibling values were near-ceiling already at k=5); prediction
   k=1 >= 0.4, k=3 >= 0.75 on their mean.
3. **Single-demo brittleness**: k=1 trains ~490 epochs on one trajectory;
   success will depend on how far eval init states sit from the single
   demonstrated start. Expect high per-task variance: some tasks >= 0.7
   (short reach, e.g. 7 "turn on the stove"), some <= 0.2 (multi-stage 3,
   push-verb 5, cream-cheese 6).
4. **Ordering violations are plausible per task** (k=1 > k=2 on some task at
   20-episode resolution), but not on the 10-task mean.

Purpose: locate the knee of the cost curve. If even k=1 lands >= 0.5 on
tasks 0-2, the interesting method question shifts from "more demos" to
"cheaper training at equal quality", per the amendment.
