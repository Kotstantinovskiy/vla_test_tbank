# Seen-scene x goal-prompts v2: prompted-predicate success

Frozen official-data pretrain in SEEN libero_90 scenes; only the
prompt varies. v2 primary success = the PROMPTED task's goal
predicate, evaluated per step on the running scene (episodes also
terminate on prompted success). `env succ` is the env task's own
predicate — the v1-comparable secondary metric. Same seeds and init
states within an env -> paired exact McNemar.

| env instruction | block | prompt | prompted succ | 95% CI | env succ |
|---|---|---|---:|---|---:|
| `put the black bowl on top of the cabinet` | trained | `put the black bowl on top of the cabinet` | 17/20 | [0.64, 0.95] | 17/20 |
| `put the black bowl on top of the cabinet` | paraphrase | `put the bowl on top of the cabinet` | 0/20 | [0.00, 0.16] | 0/20 |
| `put the black bowl on the plate` | trained | `put the black bowl on the plate` | 12/20 | [0.39, 0.78] | 12/20 |
| `put the black bowl on the plate` | paraphrase | `put the bowl on the plate` | 2/20 | [0.03, 0.30] | 2/20 |
| `put the wine bottle on the wine rack` | trained | `put the wine bottle on the wine rack` | 11/20 | [0.34, 0.74] | 11/20 |
| `put the wine bottle on the wine rack` | paraphrase | `put the wine bottle on the rack` | 2/20 | [0.03, 0.30] | 2/20 |
| `open the top drawer of the cabinet and put the bowl in it` | trained | `open the top drawer of the cabinet and put the bowl in it` | 14/20 | [0.48, 0.85] | 14/20 |
| `open the top drawer of the cabinet and put the bowl in it` | paraphrase | `open the top drawer and put the bowl inside` | 1/20 | [0.01, 0.24] | 0/20 |
| `turn on the stove` | trained | `turn on the stove` | 20/20 | [0.84, 1.00] | 20/20 |
| `put the frying pan on the stove` | trained | `put the frying pan on the stove` | 16/20 | [0.58, 0.92] | 16/20 |
| `turn on the stove` | cross | `put the frying pan on the stove` | 17/20 | [0.64, 0.95] | 0/20 |
| `put the frying pan on the stove` | cross | `turn on the stove` | 20/20 | [0.84, 1.00] | 0/20 |
| `close the top drawer of the cabinet` | trained | `close the top drawer of the cabinet` | 20/20 | [0.84, 1.00] | 20/20 |
| `close the top drawer of the cabinet` | goal | `open the middle drawer of the cabinet` | 0/20 | [0.00, 0.16] | 15/20 |
| `turn on the stove` | nonsense | `perform the dax florp twice` | 0/20 (env) | [0.00, 0.16] | 0/20 |
| `put the black bowl on top of the cabinet` | nonsense | `perform the dax florp twice` | 0/20 (env) | [0.00, 0.16] | 0/20 |

## Goal-prompt slice (all 10 libero_goal instructions)

| goal id | prompt | status | host env | relationship | prompted succ |
|---:|---|---|---|---|---:|
| 0 | `open the middle drawer of the cabinet` | point (`goal__open_the_middle_drawer_of_the_cabinet`) | `close the top drawer of the cabinet` | novel_string | 0/20 |
| 1 | `put the bowl on the stove` | skipped | — | — | — |
| 2 | `put the wine bottle on top of the cabinet` | skipped | — | — | — |
| 3 | `open the top drawer and put the bowl inside` | alias (`paraphrase__open_the_top_drawer_of_the_cabinet_and_p`) | `open the top drawer of the cabinet and put the bowl in it` | paraphrase_of_trained | 1/20 |
| 4 | `put the bowl on top of the cabinet` | alias (`paraphrase__put_the_black_bowl_on_top_of_the_cabinet`) | `put the black bowl on top of the cabinet` | paraphrase_of_trained | 0/20 |
| 5 | `push the plate to the front of the stove` | skipped | — | — | — |
| 6 | `put the cream cheese in the bowl` | skipped | — | — | — |
| 7 | `turn on the stove` | alias (`trained__turn_on_the_stove`) | `turn on the stove` | verbatim_trained | 20/20 |
| 8 | `put the bowl on the plate` | alias (`paraphrase__put_the_black_bowl_on_the_plate`) | `put the black bowl on the plate` | paraphrase_of_trained | 2/20 |
| 9 | `put the wine bottle on the rack` | alias (`paraphrase__put_the_wine_bottle_on_the_wine_rack`) | `put the wine bottle on the wine rack` | paraphrase_of_trained | 2/20 |

## Paired vs trained prompt (same env, same init states; primary metric)

| env | block | delta | discordant (trained-only / condition-only) | McNemar p |
|---|---|---:|---|---:|
| `put the black bowl on top of the cabinet` | paraphrase | -0.85 | 17 / 0 | 0.000 |
| `put the black bowl on the plate` | paraphrase | -0.50 | 11 / 1 | 0.006 |
| `put the wine bottle on the wine rack` | paraphrase | -0.45 | 10 / 1 | 0.012 |
| `open the top drawer of the cabinet and put the bowl in it` | paraphrase | -0.65 | 13 / 0 | 0.000 |
| `turn on the stove` | cross | -0.15 | 3 / 0 | 0.250 |
| `put the frying pan on the stove` | cross | +0.20 | 0 / 4 | 0.125 |
| `close the top drawer of the cabinet` | goal | -1.00 | 20 / 0 | 0.000 |
| `turn on the stove` | nonsense | -1.00 | 20 / 0 | 0.000 |
| `put the black bowl on top of the cabinet` | nonsense | -0.85 | 17 / 0 | 0.000 |
