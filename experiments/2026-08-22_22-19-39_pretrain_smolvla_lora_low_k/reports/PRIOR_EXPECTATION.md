# Prior expectation — recorded before preparation and any rollout

Recorded 2026-08-22, before preparation, training, or any target rollout of
this experiment. Training seed 1000; evaluation protocol identical to the
expert-only deterministic repro and the full-FT experiment, so all three
methods are compared pairwise on the same episodes.

References (same seeds/demos/evaluation, mean over tasks 0–2):

| method | k=1 | k=2 | k=3 |
|---|---:|---:|---:|
| expert-only (n=50) | 0.600 | 0.683 | 0.667 |
| full-FT (n=50) | 0.583 | 0.950 | 0.767 |
| full-FT (n=25) | 0.533 | 0.750 | 0.800 |

The full-FT gain over expert-only was concentrated at k=2 and almost entirely
on task 0 (drawer: 6/20 → 20/20 at n=50).

Predictions for LoRA-on-VLM (r=16, α=32) + fully trained expert:

1. The LoRA curve lies between expert-only and full-FT at every budget
   (n=50): the expert path is identical to the baseline, so LoRA cannot be
   much worse than expert-only; rank-16 updates recover only part of the
   full-FT VLM shift.
2. Concretely at n=50: mean(k=1) in [0.50, 0.65]; mean(k=2) in [0.70, 0.95],
   i.e. at least half of the full-FT k=2 gain over expert-only comes through
   the low-rank VLM update; mean(k=3) in [0.65, 0.80].
3. The k=2 drawer effect partially reproduces: task 0 / k=2 (n=50) lands
   strictly above the expert-only 6/20 but below the full-FT 20/20.
4. n=25 vs n=50 behaves like in the other experiments: differences
   concentrated on task 0, tasks 1–2 within ±2 successes per point.
5. The determinism gate passes exactly at both variants (protocol-level
   property, independent of the adaptation method).
6. Trainable parameters: ~100M full-training copies (expert + projections,
   as in the baseline) plus roughly 5–10M LoRA parameters; the audit reports
   the exact split before fan-out.

Falsification handling: if LoRA lands below expert-only by more than 0.10 at
any budget mean, first inspect the merge step (adapter actually applied,
correct normalizer statistics copied), the audit, and training loss curves
before interpreting scientifically. Single-training-seed caveat applies as in
the sibling experiments.
