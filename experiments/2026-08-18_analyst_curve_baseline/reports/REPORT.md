# Combined naive-baseline cost curve and AUC

Single frozen recipe (expert-only, 2000 steps, seed 1000, official
demo_0..demo_{k-1}) measured across seven budgets by three experiments;
this experiment only aggregates their published summaries.

| series | k=0 | k=1 | k=2 | k=3 | k=5 | k=10 | k=25 | AUC raw | nAUC | nAUC(log2) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mean all 10 | 0.005 | 0.550 | 0.705 | 0.780 | 0.830 | 0.875 | 0.850 | 20.46 | 0.818 | 0.689 |
| mean tasks 0-2 | 0.000 | 0.567 | 0.717 | 0.717 | 0.950 | 0.950 | 0.900 | 21.93 | 0.877 | 0.728 |

AUC is the trapezoid under success(k) for k in [0, 25]. `nAUC` divides
by the range (25): the linear-interpolation average success over the
whole budget range. `nAUC(log2)` spaces nodes by log2(1+k), so the
cheap-demo regime the assignment targets dominates the score; it is
the recommended headline scalar for comparing adaptation methods
(a method that lifts k=1..3 moves it far more than one that lifts k=25).

Per-task curves and AUCs: `results/summary/combined_curve.csv`.
Caveats inherited from the sources: single training seed; k=0 ran under
pretraining normalization statistics while k>0 used target statistics.
