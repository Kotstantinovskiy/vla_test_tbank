# Goal-scene x seen-prompts (probe 3): trained strings in a novel scene

Frozen official-data pretrain in the GOAL scene (libero_goal); prompts
are verbatim libero_90-trained strings vs the goal instructions.
Primary success = the PROMPTED task's predicate; behavioral metrics
discriminate conditions at the floor.

| env instruction | block | prompt | prompted succ | 95% CI | med min dist (m) | moved>5cm |
|---|---|---|---:|---|---:|---:|
| `turn on the stove` | true | `turn on the stove` | 0/20 | [0.00, 0.16] | 0.213 | 0/20 |
| `open the top drawer and put the bowl inside` | true | `open the top drawer and put the bowl inside` | 0/20 | [0.00, 0.16] | 0.109 | 0/20 |
| `open the top drawer and put the bowl inside` | seen_twin | `open the top drawer of the cabinet and put the bowl in it` | 0/20 | [0.00, 0.16] | 0.094 | 0/20 |
| `put the bowl on top of the cabinet` | true | `put the bowl on top of the cabinet` | 5/20 | [0.11, 0.47] | 0.066 | 11/20 |
| `put the bowl on top of the cabinet` | seen_twin | `put the black bowl on top of the cabinet` | 0/20 | [0.00, 0.16] | 0.096 | 3/20 |
| `put the bowl on the plate` | true | `put the bowl on the plate` | 0/20 | [0.00, 0.16] | 0.052 | 10/20 |
| `put the bowl on the plate` | seen_twin | `put the black bowl on the plate` | 0/20 | [0.00, 0.16] | 0.091 | 0/20 |
| `put the wine bottle on the rack` | true | `put the wine bottle on the rack` | 0/20 | [0.00, 0.16] | 0.118 | 0/20 |
| `put the wine bottle on the rack` | seen_twin | `put the wine bottle on the wine rack` | 0/20 | [0.00, 0.16] | 0.141 | 0/20 |
| `put the bowl on the plate` | seen_cross | `put the black bowl on top of the cabinet` | 0/20 | [0.00, 0.16] | 0.095 | 1/20 |
| `put the bowl on top of the cabinet` | seen_cross | `put the black bowl on the plate` | 0/20 | [0.00, 0.16] | 0.087 | 0/20 |
| `turn on the stove` | nonsense | `perform the dax florp twice` | 0/20 (env) | [0.00, 0.16] | 0.317 | 0/20 |
| `open the top drawer and put the bowl inside` | nonsense | `perform the dax florp twice` | 0/20 (env) | [0.00, 0.16] | 0.057 | 14/20 |

## Paired vs the true goal prompt (same env, same init states)

| env | block | prompt | true → condition | Δ succ | Δ med min dist | McNemar p |
|---|---|---|---|---:|---:|---:|
| `open the top drawer and put the bowl ins` | seen_twin | `open the top drawer of the cabinet and p` | 0/20 → 0/20 | +0.00 | -0.015 | 1 |
| `put the bowl on top of the cabinet` | seen_twin | `put the black bowl on top of the cabinet` | 5/20 → 0/20 | -0.25 | +0.030 | 0.0625 |
| `put the bowl on the plate` | seen_twin | `put the black bowl on the plate` | 0/20 → 0/20 | +0.00 | +0.038 | 1 |
| `put the wine bottle on the rack` | seen_twin | `put the wine bottle on the wine rack` | 0/20 → 0/20 | +0.00 | +0.023 | 1 |
| `put the bowl on the plate` | seen_cross | `put the black bowl on top of the cabinet` | 0/20 → 0/20 | +0.00 | +0.043 | 1 |
| `put the bowl on top of the cabinet` | seen_cross | `put the black bowl on the plate` | 5/20 → 0/20 | -0.25 | +0.021 | 0.0625 |
| `turn on the stove` | nonsense | `perform the dax florp twice` | 0/20 → 0/20 | +0.00 | +0.104 | 1 |
| `open the top drawer and put the bowl ins` | nonsense | `perform the dax florp twice` | 0/20 → 0/20 | +0.00 | -0.052 | 1 |
